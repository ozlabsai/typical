"""Hash-bound R047-M1 retention manifest, evaluation receipts, and promotion gate.

The legacy evaluators produce useful metrics but do not all record the adapter
that they loaded.  This module closes that provenance gap in three stages:

1. ``build_manifest`` freezes every evaluation input, evaluator implementation,
   model inventory, threshold, and causal preassignment before training.
2. ``run_evaluation`` executes the exact frozen evaluator command and binds its
   report (and prediction evidence) to the loaded reference/candidate tree.
3. ``finalize`` revalidates all bytes and requires reports for the reference,
   control, and treatment.  Only the treatment can advance to R047-M2.

No function fabricates or repairs evaluator results; malformed or incomplete
evidence raises ``ValueError``.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import compare_r047_causal as causal_comparator
from scripts import eval_r047_executor as causal_evaluator
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, verify_inventory
from scripts.preflight_r043_assets import file_hash, tree_hash


MANIFEST_SCHEMA = "r047-retention-manifest-v1"
EVALUATION_RECEIPT_SCHEMA = "r047-retention-evaluation-receipt-v1"
FINAL_RECEIPT_SCHEMA = "r047-retention-final-v1"
ROLES = ("reference", "control", "treatment")
SUITES = ("retention", "action", "renderer", "mmhe", "pair")
RETENTION_SUITES = ("wf", "wh", "replay")
ACTION_MODES = ("raw", "calibrated")
RENDERER_CELLS = ("bare_no", "bare_yes", "canonical_no", "canonical_yes")
MM_SUITES = ("he", "en", "real", "mismatch", "gold_dropped")
PAIR_METRICS = ("accuracy", "exact_action_accuracy", "twin_both_correct")

INPUT_KEYS = {
    "retention_wf", "retention_wh", "retention_replay",
    "action_eval", "action_calibration", "renderer",
    "mm_he", "mm_en", "mm_real", "mm_shuffled", "mm_noimage",
    "mm_mismatch", "mm_gold_dropped", "mm_root", "pair",
}
CODE_KEYS = {
    "retention_evaluator", "action_evaluator", "calibration_evaluator",
    "renderer_evaluator", "mmhe_evaluator", "pair_evaluator",
    "shared_scorer", "action_policy", "receipt_wrapper",
}
CODE_BASENAMES = {
    "retention_evaluator": "eval_gemma_decision_lora.py",
    "action_evaluator": "score_action_panel.py",
    "calibration_evaluator": "fit_internal_calibration.py",
    "renderer_evaluator": "probe_action_renderer.py",
    "mmhe_evaluator": "eval_gemma_mm_he.py",
    "pair_evaluator": "eval_r043_pairs.py",
    "shared_scorer": "train_gemma_decision_lora.py",
    "action_policy": "build_action_panel.py",
    "receipt_wrapper": "finalize_r047_retention.py",
}
SUITE_INPUTS = {
    "retention": ("retention_wf", "retention_wh", "retention_replay"),
    "action": ("action_eval", "action_calibration"),
    "renderer": ("renderer",),
    "mmhe": ("mm_he", "mm_en", "mm_real", "mm_shuffled", "mm_noimage",
             "mm_mismatch", "mm_gold_dropped", "mm_root"),
    "pair": ("pair",),
}
SUITE_CODE = {
    "retention": ("retention_evaluator", "shared_scorer"),
    "action": ("action_evaluator", "calibration_evaluator", "shared_scorer",
               "action_policy"),
    "renderer": ("renderer_evaluator", "action_evaluator", "calibration_evaluator",
                 "shared_scorer", "action_policy"),
    "mmhe": ("mmhe_evaluator", "shared_scorer"),
    "pair": ("pair_evaluator", "action_evaluator", "calibration_evaluator",
             "shared_scorer", "action_policy"),
}
EVIDENCE_KEYS = {
    "retention": {"report"},
    "action": {"report", "predictions", "calibration_predictions"},
    "renderer": {"report", "predictions"},
    "mmhe": {"report"},
    "pair": {"report", "predictions"},
}
PARAMETERS = {
    "retention": {"samples": 128, "max_length": 8192, "seed": 42,
                  "adapter_scale": 1.0},
    "action": {"rows": 256, "max_length": 8192, "calibrated": True},
    "renderer": {"per_label": 65, "max_length": 8192},
    "mmhe": {"rows_per_suite": 128, "max_length": 8192,
             "base_comparison": True},
    "pair": {"pairs": 256, "presentations": 512, "max_length": 8192},
}
THRESHOLDS = {
    "retention_max_accuracy_drop": 0.02,
    "action_max_accuracy_drop": 1 / 256,
    "renderer_max_accuracy_drop": 1 / 65,
    "mmhe_max_accuracy_drop": 1 / 128,
    "pair_max_accuracy_drop": 0.01,
    "treatment_requires_mmhe_absolute_promotion": True,
    "treatment_requires_causal_primary_gate": True,
}
FROZEN_CONTROL_TRAIN_SHA256 = (
    "179b6497fb6050dc5974b2545b1fed198fc0c00b200481e6e29c5bf1f8e27694")
FROZEN_TREATMENT_TRAIN_SHA256 = (
    "10366cc7656a89331ff85fd88e304637f5487f82b50f7ccd69be30436357a586")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _load_object(payload, label):
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return path.read_bytes()


def _artifact(path, label):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    if path.is_file():
        if path.stat().st_size <= 0:
            raise ValueError(f"{label} must be nonempty")
        return {"kind": "file", "sha256": file_hash(path), "files": 1}
    if path.is_dir():
        if any(item.is_symlink() for item in path.rglob("*")):
            raise ValueError(f"{label} tree must not contain symbolic links")
        digest, files = tree_hash(path)
        if files <= 0:
            raise ValueError(f"{label} tree must be nonempty")
        return {"kind": "tree", "sha256": digest, "files": files}
    raise ValueError(f"{label} does not exist")


def _validate_artifact_record(value, label):
    if (not isinstance(value, dict) or set(value) != {"kind", "sha256", "files"}
            or value.get("kind") not in {"file", "tree"}
            or not _valid_sha(value.get("sha256"))
            or not isinstance(value.get("files"), int)
            or isinstance(value.get("files"), bool) or value["files"] <= 0
            or (value["kind"] == "file" and value["files"] != 1)):
        raise ValueError(f"invalid {label} artifact binding")


def _validate_manifest(value):
    expected = {
        "schema_version", "model_revision", "model_inventory_sha256",
        "model_tree_sha256", "model_tree_files", "reference_provenance_sha256",
        "causal_assignment_sha256", "causal_recipe_sha256", "inputs",
        "evaluator_code", "parameters", "thresholds", "required_roles",
        "required_suites",
    }
    if set(value) != expected or value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("retention manifest schema differs from the frozen contract")
    if value.get("model_revision") != FROZEN_BASE_REVISION:
        raise ValueError("retention manifest model revision is not frozen")
    for field in ("model_inventory_sha256", "model_tree_sha256",
                  "reference_provenance_sha256", "causal_assignment_sha256",
                  "causal_recipe_sha256"):
        if not _valid_sha(value.get(field)):
            raise ValueError(f"retention manifest {field} is invalid")
    if (not isinstance(value.get("model_tree_files"), int)
            or isinstance(value.get("model_tree_files"), bool)
            or value["model_tree_files"] <= 0):
        raise ValueError("retention manifest model tree count is invalid")
    if set(value.get("inputs", {})) != INPUT_KEYS:
        raise ValueError("retention manifest input set is incomplete")
    for key, binding in value["inputs"].items():
        _validate_artifact_record(binding, f"input {key}")
    if (set(value.get("evaluator_code", {})) != CODE_KEYS
            or any(not _valid_sha(item) for item in value["evaluator_code"].values())):
        raise ValueError("retention manifest evaluator-code set is incomplete")
    if value.get("parameters") != PARAMETERS or value.get("thresholds") != THRESHOLDS:
        raise ValueError("retention manifest parameters or thresholds changed")
    if value.get("required_roles") != list(ROLES) or value.get("required_suites") != list(SUITES):
        raise ValueError("retention manifest coverage contract changed")
    return value


def _validate_assignment(payload, manifest):
    assignment = _load_object(payload, "causal assignment")
    if _sha(payload) != manifest["causal_assignment_sha256"]:
        raise ValueError("causal assignment hash differs from retention manifest")
    if (set(assignment) != causal_comparator.ASSIGNMENT_FIELDS
            or assignment.get("schema_version") != "r047-causal-slot-assignment-v1"
            or assignment.get("control_slot") != "slot_b"
            or assignment.get("treatment_slot") != "slot_a"
            or assignment.get("control_train_sha256") != FROZEN_CONTROL_TRAIN_SHA256
            or assignment.get("treatment_train_sha256") != FROZEN_TREATMENT_TRAIN_SHA256
            or assignment.get("model_revision") != manifest["model_revision"]
            or assignment.get("model_inventory_sha256")
            != manifest["model_inventory_sha256"]
            or assignment.get("reference_provenance_sha256")
            != manifest["reference_provenance_sha256"]
            or assignment.get("recipe_sha256") != manifest["causal_recipe_sha256"]):
        raise ValueError("causal assignment differs from retention manifest")
    return assignment


def _validate_live_artifacts(manifest, input_paths, code_paths):
    if set(input_paths) != INPUT_KEYS or set(code_paths) != CODE_KEYS:
        raise ValueError("live retention inputs/evaluator code are incomplete")
    for key in INPUT_KEYS:
        if _artifact(input_paths[key], f"input {key}") != manifest["inputs"][key]:
            raise ValueError(f"retention input {key} differs from prelaunch manifest")
    for key in CODE_KEYS:
        if Path(code_paths[key]).name != CODE_BASENAMES[key]:
            raise ValueError(f"evaluator code {key} has the wrong entrypoint name")
        if file_hash(code_paths[key]) != manifest["evaluator_code"][key]:
            raise ValueError(f"evaluator code {key} differs from prelaunch manifest")
    if Path(code_paths["receipt_wrapper"]).resolve() != Path(__file__).resolve():
        raise ValueError("receipt wrapper is not this frozen implementation")


def build_manifest(*, model_inventory_path, model_path, reference_provenance_path,
                   causal_assignment_path, causal_recipe_path, input_paths, code_paths):
    """Return the complete prelaunch retention manifest without writing results."""
    inventory, inventory_sha = verify_inventory(
        model_inventory_path, model_path, FROZEN_BASE_REVISION)
    reference_payload = _read(reference_provenance_path, "reference provenance")
    reference = causal_evaluator.validate_reference_provenance(reference_payload)
    assignment_payload = _read(causal_assignment_path, "causal assignment")
    recipe_payload = _read(causal_recipe_path, "causal recipe")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "model_revision": FROZEN_BASE_REVISION,
        "model_inventory_sha256": inventory_sha,
        "model_tree_sha256": inventory["tree_sha256"],
        "model_tree_files": inventory["files"],
        "reference_provenance_sha256": _sha(reference_payload),
        "causal_assignment_sha256": _sha(assignment_payload),
        "causal_recipe_sha256": _sha(recipe_payload),
        "inputs": {key: _artifact(input_paths[key], f"input {key}")
                   for key in sorted(INPUT_KEYS)},
        "evaluator_code": {key: file_hash(code_paths[key]) for key in sorted(CODE_KEYS)},
        "parameters": PARAMETERS,
        "thresholds": THRESHOLDS,
        "required_roles": list(ROLES),
        "required_suites": list(SUITES),
    }
    _validate_manifest(manifest)
    _validate_live_artifacts(manifest, input_paths, code_paths)
    assignment = _validate_assignment(assignment_payload, manifest)
    if (reference["model_revision"] != manifest["model_revision"]
            or reference["model_inventory_sha256"] != inventory_sha
            or reference["adapter_weights_sha256"]
            != assignment["reference_adapter_sha256"]):
        raise ValueError("reference provenance differs from model/assignment")
    return manifest


def _validate_model(manifest, model_inventory_payload, model_path):
    if _sha(model_inventory_payload) != manifest["model_inventory_sha256"]:
        raise ValueError("model inventory hash differs from retention manifest")
    inventory = causal_evaluator.validate_causal_model_inventory(
        model_path, model_inventory_payload, manifest["model_revision"])
    if (inventory["tree_sha256"] != manifest["model_tree_sha256"]
            or inventory["files"] != manifest["model_tree_files"]):
        raise ValueError("loaded model tree differs from retention manifest")
    return inventory


def _adapter_binding(role, adapter_path, provenance_payload, assignment, manifest,
                     source_checkpoint_provenance_payload=None):
    if role == "reference":
        if _sha(provenance_payload) != manifest["reference_provenance_sha256"]:
            raise ValueError("reference provenance hash differs from retention manifest")
        values = causal_evaluator.validate_loaded_reference_adapter(
            adapter_path, provenance_payload, source_checkpoint_provenance_payload,
            manifest["model_revision"], manifest["model_inventory_sha256"])
        provenance = causal_evaluator.validate_reference_provenance(provenance_payload)
        return {
            "provenance_sha256": _sha(provenance_payload),
            "adapter_sha256": values[1], "adapter_config_sha256": values[2],
            "adapter_tree_sha256": values[3], "adapter_tree_files": values[4],
            "slot": None, "provenance": provenance,
        }
    receipt = causal_evaluator.validate_causal_run_receipt(provenance_payload)
    expected_slot = assignment[f"{role}_slot"]
    expected_train = assignment[f"{role}_train_sha256"]
    if (receipt["slot"] != expected_slot or receipt["train_sha256"] != expected_train
            or receipt["recipe_sha256"] != manifest["causal_recipe_sha256"]
            or receipt["model_revision"] != manifest["model_revision"]
            or receipt["model_inventory_sha256"] != manifest["model_inventory_sha256"]
            or receipt["checkpoint_step"] != 400):
        raise ValueError(f"{role} run receipt differs from causal assignment")
    values = causal_evaluator.validate_loaded_candidate_adapter(
        adapter_path, provenance_payload, manifest["model_revision"],
        manifest["model_inventory_sha256"])
    return {
        "provenance_sha256": _sha(provenance_payload),
        "adapter_sha256": values[1], "adapter_config_sha256": values[2],
        "adapter_tree_sha256": values[3], "adapter_tree_files": values[4],
        "slot": expected_slot, "provenance": receipt,
    }


def _validate_mm_inputs(report, manifest):
    inputs = report.get("inputs")
    expected_mm = {
        name: manifest["inputs"][f"mm_{name}"]["sha256"]
        for name in ("real", "shuffled", "noimage", "mismatch", "gold_dropped")
    }
    if (not isinstance(inputs, dict)
            or inputs.get("he_left_sha256") != manifest["inputs"]["mm_he"]["sha256"]
            or inputs.get("he_right_sha256") != manifest["inputs"]["mm_en"]["sha256"]
            or inputs.get("mm") != expected_mm):
        raise ValueError("MM/HE report input hashes differ from retention manifest")


def _validate_report_input_binding(suite, report, manifest):
    if suite == "renderer" and report.get("input_sha256") != manifest["inputs"]["renderer"]["sha256"]:
        raise ValueError("renderer report input hash differs from retention manifest")
    if suite == "pair":
        if (report.get("input_sha256") != manifest["inputs"]["pair"]["sha256"]
                or report.get("model_revision") != manifest["model_revision"]):
            raise ValueError("pair report input/model binding differs from retention manifest")
    if suite == "mmhe":
        _validate_mm_inputs(report, manifest)


def _execution_plan_sha256(role, suite, manifest, binding):
    plan = {
        "schema_version": "r047-retention-execution-plan-v1",
        "role": role, "suite": suite, "slot": binding["slot"],
        "model_revision": manifest["model_revision"],
        "model_inventory_sha256": manifest["model_inventory_sha256"],
        "adapter_provenance_sha256": binding["provenance_sha256"],
        "adapter_tree_sha256": binding["adapter_tree_sha256"],
        "inputs": {key: manifest["inputs"][key] for key in SUITE_INPUTS[suite]},
        "code": {key: manifest["evaluator_code"][key] for key in SUITE_CODE[suite]},
        "wrapper_sha256": manifest["evaluator_code"]["receipt_wrapper"],
        "parameters": PARAMETERS[suite],
    }
    return _sha(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())


def _seal_completed_evaluation(*, manifest_payload, manifest_sha256, assignment_payload,
                               role, suite, report_path, evidence_paths, adapter_path,
                               provenance_payload, model_inventory_payload, model_path,
                               input_paths, code_paths,
                               source_checkpoint_provenance_payload=None):
    """Seal outputs created by ``run_evaluation``; never a substitute for execution."""
    if _sha(manifest_payload) != manifest_sha256:
        raise ValueError("retention manifest hash mismatch")
    manifest = _validate_manifest(_load_object(manifest_payload, "retention manifest"))
    if role not in ROLES or suite not in SUITES:
        raise ValueError("invalid retention evaluation role or suite")
    if _sha(manifest_payload) != manifest_sha256:
        raise ValueError("retention manifest hash mismatch")
    assignment = _validate_assignment(assignment_payload, manifest)
    _validate_live_artifacts(manifest, input_paths, code_paths)
    _validate_model(manifest, model_inventory_payload, model_path)
    binding = _adapter_binding(
        role, adapter_path, provenance_payload, assignment, manifest,
        source_checkpoint_provenance_payload)
    if set(evidence_paths) != EVIDENCE_KEYS[suite] or Path(report_path) != Path(evidence_paths["report"]):
        raise ValueError(f"{suite} evaluation evidence set is incomplete")
    evidence = {key: file_hash(path) for key, path in sorted(evidence_paths.items())}
    report_payload = _read(report_path, f"{suite} report")
    report = _load_object(report_payload, f"{suite} report")
    _validate_report_input_binding(suite, report, manifest)
    return {
        "schema_version": EVALUATION_RECEIPT_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "role": role,
        "slot": binding["slot"],
        "suite": suite,
        "model_revision": manifest["model_revision"],
        "model_inventory_sha256": manifest["model_inventory_sha256"],
        "model_tree_sha256": manifest["model_tree_sha256"],
        "model_tree_files": manifest["model_tree_files"],
        "adapter_provenance_sha256": binding["provenance_sha256"],
        "adapter_sha256": binding["adapter_sha256"],
        "adapter_config_sha256": binding["adapter_config_sha256"],
        "adapter_tree_sha256": binding["adapter_tree_sha256"],
        "adapter_tree_files": binding["adapter_tree_files"],
        "input_sha256": {key: manifest["inputs"][key]["sha256"]
                         for key in SUITE_INPUTS[suite]},
        "evaluator_code_sha256": {key: manifest["evaluator_code"][key]
                                  for key in SUITE_CODE[suite]},
        "parameters": PARAMETERS[suite],
        "execution_plan_sha256": _execution_plan_sha256(role, suite, manifest, binding),
        "evidence_sha256": evidence,
    }


def _run_checked(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _execute_suite(suite, out, adapter_path, model_path, input_paths, code_paths,
                   manifest, run):
    """Execute one suite into an unpublished scratch directory."""
    cwd = Path(code_paths["receipt_wrapper"]).resolve().parent.parent
    python = sys.executable
    common_adapter = ["--adapter", str(Path(adapter_path).resolve())]
    if suite == "retention":
        command = [python, str(Path(code_paths["retention_evaluator"]).resolve()),
                   "--model", str(Path(model_path).resolve()), *common_adapter,
                   "--adapter-scale", "1.0", "--samples", "128", "--max-length", "8192",
                   "--seed", "42"]
        for name in RETENTION_SUITES:
            command += ["--val", f"{name}={Path(input_paths[f'retention_{name}']).resolve()}"]
        result = run(command, cwd)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("retention evaluator did not emit one JSON report") from error
        report_path = _write_json_new(out / "report.json", report)
        return report_path, {"report": report_path}
    if suite == "action":
        calibration_out = out / "calibration"
        calibration_command = [
            python, str(Path(code_paths["action_evaluator"]).resolve()),
            "--model", str(Path(model_path).resolve()), *common_adapter,
            "--data", str(Path(input_paths["action_calibration"]).resolve()),
            "--out", str(calibration_out), "--max-length", "8192"]
        run(calibration_command, cwd)
        evaluation_out = out / "evaluation"
        command = [python, str(Path(code_paths["action_evaluator"]).resolve()),
                   "--model", str(Path(model_path).resolve()), *common_adapter,
                   "--data", str(Path(input_paths["action_eval"]).resolve()),
                   "--out", str(evaluation_out), "--limit", "256", "--max-length", "8192",
                   "--calibration-data", str(Path(input_paths["action_calibration"]).resolve()),
                   "--calibration-predictions", str(calibration_out / "predictions.jsonl")]
        run(command, cwd)
        report_path = evaluation_out / "report.json"
        return report_path, {
            "report": report_path, "predictions": evaluation_out / "predictions.jsonl",
            "calibration_predictions": calibration_out / "predictions.jsonl"}
    if suite == "renderer":
        evaluation_out = out / "evaluation"
        command = [python, str(Path(code_paths["renderer_evaluator"]).resolve()),
                   "--model", str(Path(model_path).resolve()), *common_adapter,
                   "--data", str(Path(input_paths["renderer"]).resolve()),
                   "--out", str(evaluation_out), "--per-label", "65", "--max-length", "8192"]
        run(command, cwd)
        report_path = evaluation_out / "report.json"
        return report_path, {"report": report_path,
                             "predictions": evaluation_out / "predictions.jsonl"}
    if suite == "mmhe":
        report_path = out / "report.json"
        command = [python, str(Path(code_paths["mmhe_evaluator"]).resolve()),
                   "--model", str(Path(model_path).resolve()), *common_adapter,
                   "--he-left", str(Path(input_paths["mm_he"]).resolve()),
                   "--he-right", str(Path(input_paths["mm_en"]).resolve()),
                   "--mm-dir", str(Path(input_paths["mm_root"]).resolve()),
                   "--out", str(report_path), "--max-length", "8192"]
        run(command, cwd)
        return report_path, {"report": report_path}
    evaluation_out = out / "evaluation"
    command = [python, str(Path(code_paths["pair_evaluator"]).resolve()),
               "--model", str(Path(model_path).resolve()),
               "--model-revision", manifest["model_revision"], *common_adapter,
               "--data", str(Path(input_paths["pair"]).resolve()),
               "--out", str(evaluation_out), "--max-length", "8192"]
    run(command, cwd)
    report_path = evaluation_out / "report.json"
    return report_path, {"report": report_path,
                         "predictions": evaluation_out / "predictions.jsonl"}


def run_evaluation(*, manifest_payload, manifest_sha256, assignment_payload, role, suite,
                   out, adapter_path, provenance_payload, model_inventory_payload,
                   model_path, input_paths, code_paths,
                   source_checkpoint_provenance_payload=None, _run=_run_checked):
    """Execute one exact frozen evaluator command and atomically make its receipt.

    The receipt is emitted only after the subprocess exits successfully and its
    expected evidence files exist.  ``_run`` is a test seam; production callers
    use the checked subprocess runner above.
    """
    if role not in ROLES or suite not in SUITES:
        raise ValueError("invalid retention evaluation role or suite")
    final_out = Path(out)
    if final_out.exists():
        raise ValueError("refusing to overwrite retention evaluation output")
    manifest = _validate_manifest(_load_object(manifest_payload, "retention manifest"))
    assignment = _validate_assignment(assignment_payload, manifest)
    _validate_live_artifacts(manifest, input_paths, code_paths)
    _validate_model(manifest, model_inventory_payload, model_path)
    _adapter_binding(role, adapter_path, provenance_payload, assignment, manifest,
                     source_checkpoint_provenance_payload)
    mm_names = {"real": "scienceqa_test.jsonl",
                "shuffled": "scienceqa_test_shuffled.jsonl",
                "noimage": "scienceqa_test_noimage.jsonl",
                "mismatch": "scienceqa_test_mismatch.jsonl",
                "gold_dropped": "scienceqa_test_gold_dropped.jsonl"}
    for name, filename in mm_names.items():
        if Path(input_paths[f"mm_{name}"]).resolve() != (Path(input_paths["mm_root"]) / filename).resolve():
            raise ValueError(f"MM input {name} is not the frozen file inside mm_root")

    final_out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{final_out.name}.tmp-", dir=final_out.parent))
    try:
        report_path, evidence = _execute_suite(
            suite, scratch, adapter_path, model_path, input_paths, code_paths,
            manifest, _run)
        receipt = _seal_completed_evaluation(
            manifest_payload=manifest_payload, manifest_sha256=manifest_sha256,
            assignment_payload=assignment_payload, role=role, suite=suite,
            report_path=report_path, evidence_paths=evidence, adapter_path=adapter_path,
            provenance_payload=provenance_payload,
            model_inventory_payload=model_inventory_payload, model_path=model_path,
            input_paths=input_paths, code_paths=code_paths,
            source_checkpoint_provenance_payload=source_checkpoint_provenance_payload)
        receipt_path = _write_json_new(scratch / "receipt.json", receipt)
        relative_report = report_path.relative_to(scratch)
        relative_receipt = receipt_path.relative_to(scratch)
        relative_evidence = {key: path.relative_to(scratch)
                             for key, path in evidence.items()}
        scratch.replace(final_out)
        return {"report": str(final_out / relative_report),
                "receipt": str(final_out / relative_receipt),
                "evidence": {key: str(final_out / path)
                             for key, path in relative_evidence.items()}}
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def _write_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _number(value, label):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError(f"{label} must be a finite probability")
    return float(value)


def _parse_retention(report, label):
    result = {}
    for suite in RETENTION_SUITES:
        row = report.get(suite)
        if (not isinstance(row, dict) or row.get("n") != 128
                or row.get("skipped_overlength") != 0):
            raise ValueError(f"{label}.{suite} must score 128 rows without skips")
        result[suite] = _number(row.get("accuracy"), f"{label}.{suite}.accuracy")
    return result


def _parse_action(report, label):
    result = {}
    for mode in ACTION_MODES:
        row = report.get(mode)
        errors = row.get("error_categories") if isinstance(row, dict) else None
        if (not isinstance(errors, dict) or row.get("n") != 256
                or errors.get("invalid") != 0):
            raise ValueError(f"{label}.{mode} must score 256 rows with zero invalid")
        result[mode] = _number(row.get("exact_action_accuracy"),
                               f"{label}.{mode}.exact_action_accuracy")
    return result


def _parse_renderer(report, label):
    if report.get("per_label") != 65:
        raise ValueError(f"{label} renderer must use 65 cases per label")
    result = {}
    for cell in RENDERER_CELLS:
        row = report.get(cell)
        if not isinstance(row, dict) or row.get("n") != 130:
            raise ValueError(f"{label}.{cell} must score 130 presentations")
        result[cell] = _number(report.get(f"{cell}_both_orders"),
                               f"{label}.{cell}_both_orders")
    return result


def _parse_mmhe(report, label):
    for endpoint in ("base", "candidate"):
        values = report.get(endpoint)
        if not isinstance(values, dict):
            raise ValueError(f"{label}.{endpoint} is missing")
        for suite in ("he", "en", "real", "shuffled", "noimage", "mismatch",
                      "gold_dropped"):
            row = values.get(suite)
            if not isinstance(row, dict) or row.get("n") != 128:
                raise ValueError(f"{label}.{endpoint}.{suite} must score 128 rows")
    comparison = report.get("comparison")
    gates = comparison.get("gates") if isinstance(comparison, dict) else None
    if (not isinstance(gates, dict) or not gates
            or any(type(value) is not bool for value in gates.values())
            or type(comparison.get("promotion_pass")) is not bool
            or comparison["promotion_pass"] != all(gates.values())):
        raise ValueError(f"{label} MM/HE absolute promotion is inconsistent")
    metrics = {suite: _number(report["candidate"][suite].get("accuracy"),
                              f"{label}.candidate.{suite}.accuracy")
               for suite in MM_SUITES}
    return metrics, comparison["promotion_pass"]


def _parse_pair(report, label):
    if report.get("pairs") != 256 or report.get("presentations") != 512:
        raise ValueError(f"{label} pair report must contain 256 pairs/512 presentations")
    return {metric: _number(report.get(metric), f"{label}.{metric}")
            for metric in PAIR_METRICS}


PARSERS = {"retention": _parse_retention, "action": _parse_action,
           "renderer": _parse_renderer, "mmhe": _parse_mmhe, "pair": _parse_pair}


def _validate_receipt(receipt, *, role, suite, manifest_sha256, manifest, binding,
                      evidence_paths):
    expected_fields = {
        "schema_version", "manifest_sha256", "role", "slot", "suite",
        "model_revision", "model_inventory_sha256", "model_tree_sha256",
        "model_tree_files", "adapter_provenance_sha256", "adapter_sha256",
        "adapter_config_sha256", "adapter_tree_sha256", "adapter_tree_files",
        "input_sha256", "evaluator_code_sha256", "parameters",
        "execution_plan_sha256", "evidence_sha256",
    }
    if set(receipt) != expected_fields or receipt.get("schema_version") != EVALUATION_RECEIPT_SCHEMA:
        raise ValueError(f"{role}/{suite} evaluation receipt schema is invalid")
    expected = {
        "manifest_sha256": manifest_sha256, "role": role, "slot": binding["slot"],
        "suite": suite, "model_revision": manifest["model_revision"],
        "model_inventory_sha256": manifest["model_inventory_sha256"],
        "model_tree_sha256": manifest["model_tree_sha256"],
        "model_tree_files": manifest["model_tree_files"],
        "adapter_provenance_sha256": binding["provenance_sha256"],
        "adapter_sha256": binding["adapter_sha256"],
        "adapter_config_sha256": binding["adapter_config_sha256"],
        "adapter_tree_sha256": binding["adapter_tree_sha256"],
        "adapter_tree_files": binding["adapter_tree_files"],
        "input_sha256": {key: manifest["inputs"][key]["sha256"]
                         for key in SUITE_INPUTS[suite]},
        "evaluator_code_sha256": {key: manifest["evaluator_code"][key]
                                  for key in SUITE_CODE[suite]},
        "parameters": PARAMETERS[suite],
        "execution_plan_sha256": _execution_plan_sha256(role, suite, manifest, binding),
        "evidence_sha256": {key: file_hash(path)
                            for key, path in sorted(evidence_paths.items())},
    }
    failures = [key for key, value in expected.items() if receipt.get(key) != value]
    if failures:
        raise ValueError(f"{role}/{suite} evaluation receipt mismatch: {failures}")


def _validate_causal_comparison(payload, assignment, assignment_sha256, bindings):
    report = _load_object(payload, "causal comparison")
    provenance = report.get("provenance")
    preassignment = report.get("preassignment")
    receipt_hashes = provenance.get("run_receipt_sha256") if isinstance(provenance, dict) else None
    expected_preassignment = {
        "assignment_sha256": assignment_sha256,
        "control_slot": assignment["control_slot"],
        "treatment_slot": assignment["treatment_slot"],
        "pre_frozen_slot_assignment": True,
        "evaluator_sha256": assignment["evaluator_sha256"],
        "comparator_sha256": assignment["comparator_sha256"],
        "bootstrap_seed": assignment["bootstrap_seed"],
        "bootstrap_replicates": assignment["bootstrap_replicates"],
    }
    if (report.get("schema_version") != "r047-causal-comparison-v2"
            or not isinstance(report.get("co_primary_gate"), dict)
            or type(report["co_primary_gate"].get("passed")) is not bool
            or preassignment != expected_preassignment
            or not isinstance(provenance, dict)
            or provenance.get("assignment") != assignment
            or not isinstance(receipt_hashes, dict)):
        raise ValueError("causal comparison provenance is invalid")
    for role in ("control", "treatment"):
        slot = assignment[f"{role}_slot"]
        if receipt_hashes.get(slot) != bindings[role]["provenance_sha256"]:
            raise ValueError(f"causal comparison used a different {role} run receipt")
    if provenance.get("reference_provenance_sha256") != bindings["reference"]["provenance_sha256"]:
        raise ValueError("causal comparison used different reference provenance")
    return report


def finalize(*, manifest_payload, manifest_sha256, assignment_payload, recipe_payload,
             model_inventory_payload, model_path, input_paths, code_paths,
             causal_comparison_payload, role_bundles,
             source_checkpoint_provenance_payload):
    """Revalidate all postrun evidence and return the sole R047-M2 promotion receipt."""
    if _sha(manifest_payload) != manifest_sha256:
        raise ValueError("retention manifest hash mismatch")
    manifest = _validate_manifest(_load_object(manifest_payload, "retention manifest"))
    assignment = _validate_assignment(assignment_payload, manifest)
    if _sha(recipe_payload) != manifest["causal_recipe_sha256"]:
        raise ValueError("causal recipe hash differs from retention manifest")
    _validate_model(manifest, model_inventory_payload, model_path)
    _validate_live_artifacts(manifest, input_paths, code_paths)
    if set(role_bundles) != set(ROLES):
        raise ValueError("retention finalization requires reference/control/treatment")

    bindings, parsed, report_hashes, receipt_hashes = {}, {}, {}, {}
    for role in ROLES:
        bundle = role_bundles[role]
        if set(bundle) != {"adapter", "provenance", "evaluations"}:
            raise ValueError(f"{role} bundle schema is invalid")
        provenance_payload = _read(bundle["provenance"], f"{role} provenance")
        bindings[role] = _adapter_binding(
            role, bundle["adapter"], provenance_payload, assignment, manifest,
            source_checkpoint_provenance_payload if role == "reference" else None)
        if set(bundle["evaluations"]) != set(SUITES):
            raise ValueError(f"{role} evaluation coverage is incomplete")
        parsed[role], report_hashes[role], receipt_hashes[role] = {}, {}, {}
        for suite in SUITES:
            evaluation = bundle["evaluations"][suite]
            if set(evaluation) != {"report", "receipt", "evidence"}:
                raise ValueError(f"{role}/{suite} evaluation bundle schema is invalid")
            if (set(evaluation["evidence"]) != EVIDENCE_KEYS[suite]
                    or Path(evaluation["report"]) != Path(evaluation["evidence"]["report"])):
                raise ValueError(f"{role}/{suite} evaluation evidence is incomplete")
            report_payload = _read(evaluation["report"], f"{role}/{suite} report")
            report = _load_object(report_payload, f"{role}/{suite} report")
            _validate_report_input_binding(suite, report, manifest)
            receipt_payload = _read(evaluation["receipt"], f"{role}/{suite} receipt")
            receipt = _load_object(receipt_payload, f"{role}/{suite} receipt")
            _validate_receipt(
                receipt, role=role, suite=suite, manifest_sha256=manifest_sha256,
                manifest=manifest, binding=bindings[role],
                evidence_paths=evaluation["evidence"])
            parsed[role][suite] = PARSERS[suite](report, f"{role}.{suite}")
            report_hashes[role][suite] = _sha(report_payload)
            receipt_hashes[role][suite] = _sha(receipt_payload)

    causal = _validate_causal_comparison(
        causal_comparison_payload, assignment, manifest["causal_assignment_sha256"], bindings)
    parent = parsed["reference"]
    arm_results = {}
    for role in ("control", "treatment"):
        values = parsed[role]
        gates = {}
        for suite in RETENTION_SUITES:
            gates[f"{suite}_drop_at_most_.02"] = (
                values["retention"][suite] >= parent["retention"][suite] - 0.02)
        for mode in ACTION_MODES:
            gates[f"{mode}_action_drop_at_most_1_of_256"] = (
                values["action"][mode] >= parent["action"][mode] - 1 / 256)
        for cell in RENDERER_CELLS:
            gates[f"{cell}_drop_at_most_1_of_65"] = (
                values["renderer"][cell] >= parent["renderer"][cell] - 1 / 65)
        for suite in MM_SUITES:
            gates[f"{suite}_drop_at_most_1_of_128"] = (
                values["mmhe"][0][suite] >= parent["mmhe"][0][suite] - 1 / 128)
        for metric in PAIR_METRICS:
            gates[f"pair_{metric}_drop_at_most_.01"] = (
                values["pair"][metric] >= parent["pair"][metric] - 0.01)
        gates["absolute_mmhe_promotion"] = values["mmhe"][1]
        arm_results[role] = {"gates": gates, "preservation_pass": all(gates.values())}

    causal_sha = _sha(causal_comparison_payload)
    treatment_advance = (causal["co_primary_gate"]["passed"]
                         and arm_results["treatment"]["preservation_pass"])
    return {
        "schema_version": FINAL_RECEIPT_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "causal_assignment_sha256": manifest["causal_assignment_sha256"],
        "causal_comparison_sha256": causal_sha,
        "causal_primary_pass": causal["co_primary_gate"]["passed"],
        "model_inventory_sha256": manifest["model_inventory_sha256"],
        "adapter_provenance_sha256": {role: bindings[role]["provenance_sha256"]
                                      for role in ROLES},
        "evaluation_report_sha256": report_hashes,
        "evaluation_receipt_sha256": receipt_hashes,
        "metrics": parsed,
        "arms": arm_results,
        "all_required_reports_present": True,
        "treatment_advance_to_m2": treatment_advance,
    }


def _pairs(values, label):
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{label} values must use NAME=PATH")
        key, path = item.split("=", 1)
        if not key or not path or key in result:
            raise ValueError(f"invalid or duplicate {label} key")
        result[key] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common_artifacts(command):
        command.add_argument("--model-inventory", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--input", action="append", default=[])
        command.add_argument("--code", action="append", default=[])

    prepare = subparsers.add_parser("prepare")
    common_artifacts(prepare)
    prepare.add_argument("--reference-provenance", required=True)
    prepare.add_argument("--causal-assignment", required=True)
    prepare.add_argument("--causal-recipe", required=True)
    prepare.add_argument("--out", required=True)

    run = subparsers.add_parser("run")
    common_artifacts(run)
    run.add_argument("--manifest", required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--causal-assignment", required=True)
    run.add_argument("--role", choices=ROLES, required=True)
    run.add_argument("--suite", choices=SUITES, required=True)
    run.add_argument("--adapter", required=True)
    run.add_argument("--provenance", required=True)
    run.add_argument("--reference-source-provenance")
    run.add_argument("--out", required=True)

    finish = subparsers.add_parser("finalize")
    common_artifacts(finish)
    finish.add_argument("--manifest", required=True)
    finish.add_argument("--manifest-sha256", required=True)
    finish.add_argument("--causal-assignment", required=True)
    finish.add_argument("--causal-recipe", required=True)
    finish.add_argument("--causal-comparison", required=True)
    finish.add_argument("--reference-source-provenance", required=True)
    finish.add_argument("--bundle-spec", required=True)
    finish.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        output = Path(args.out)
        if output.exists():
            raise ValueError("refusing to overwrite frozen retention manifest")
        manifest = build_manifest(
            model_inventory_path=args.model_inventory, model_path=args.model,
            reference_provenance_path=args.reference_provenance,
            causal_assignment_path=args.causal_assignment,
            causal_recipe_path=args.causal_recipe, input_paths=_pairs(args.input, "input"),
            code_paths=_pairs(args.code, "code"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"manifest": str(output), "sha256": file_hash(output)},
                         indent=2, sort_keys=True))
        return 0
    input_paths, code_paths = _pairs(args.input, "input"), _pairs(args.code, "code")
    manifest_payload = _read(args.manifest, "retention manifest")
    assignment_payload = _read(args.causal_assignment, "causal assignment")
    inventory_payload = _read(args.model_inventory, "model inventory")
    if args.command == "run":
        if args.role == "reference" and not args.reference_source_provenance:
            raise ValueError("reference evaluation requires source-checkpoint provenance")
        result = run_evaluation(
            manifest_payload=manifest_payload, manifest_sha256=args.manifest_sha256,
            assignment_payload=assignment_payload, role=args.role, suite=args.suite,
            out=args.out, adapter_path=args.adapter,
            provenance_payload=_read(args.provenance, f"{args.role} provenance"),
            model_inventory_payload=inventory_payload, model_path=args.model,
            input_paths=input_paths, code_paths=code_paths,
            source_checkpoint_provenance_payload=(
                _read(args.reference_source_provenance, "reference source provenance")
                if args.reference_source_provenance else None))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "finalize":
        bundle_spec = _load_object(_read(args.bundle_spec, "bundle spec"), "bundle spec")
        result = finalize(
            manifest_payload=manifest_payload, manifest_sha256=args.manifest_sha256,
            assignment_payload=assignment_payload,
            recipe_payload=_read(args.causal_recipe, "causal recipe"),
            model_inventory_payload=inventory_payload, model_path=args.model,
            input_paths=input_paths, code_paths=code_paths,
            causal_comparison_payload=_read(args.causal_comparison, "causal comparison"),
            role_bundles=bundle_spec,
            source_checkpoint_provenance_payload=_read(
                args.reference_source_provenance, "reference source provenance"))
        output = Path(args.out)
        if output.exists():
            raise ValueError("refusing to overwrite final retention receipt")
        _write_json_new(output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["treatment_advance_to_m2"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
