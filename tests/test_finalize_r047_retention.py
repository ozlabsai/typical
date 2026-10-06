import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import scripts.finalize_r047_retention as gate
from scripts import compare_r047_causal as causal_comparator
from scripts import eval_r047_executor as causal_evaluator
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, build_inventory
from scripts.preflight_r043_assets import file_hash, tree_hash


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value)
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _adapter(root, payload):
    root = Path(root)
    weights = _write(root / "adapter_model.safetensors", payload)
    config = _write(root / "adapter_config.json", {"rank": 32})
    digest, files = tree_hash(root)
    return {
        "path": root, "weights": file_hash(weights), "config": file_hash(config),
        "tree": digest, "files": files,
    }


def _retention(value):
    return {suite: {"n": 128, "accuracy": value, "skipped_overlength": 0}
            for suite in gate.RETENTION_SUITES}


def _action(value):
    return {mode: {"n": 256, "exact_action_accuracy": value,
                   "error_categories": {"correct": 256, "invalid": 0}}
            for mode in gate.ACTION_MODES}


def _renderer(value, input_sha):
    result = {"per_label": 65, "input_sha256": input_sha}
    for cell in gate.RENDERER_CELLS:
        result[cell] = {"n": 130, "accuracy": value}
        result[f"{cell}_both_orders"] = value
    return result


def _mmhe(value, inputs, absolute=True):
    summaries = {suite: {"n": 128, "accuracy": value}
                 for suite in ("he", "en", "real", "shuffled", "noimage",
                               "mismatch", "gold_dropped")}
    return {
        "inputs": {
            "he_left_sha256": inputs["mm_he"]["sha256"],
            "he_right_sha256": inputs["mm_en"]["sha256"],
            "mm": {name: inputs[f"mm_{name}"]["sha256"]
                   for name in ("real", "shuffled", "noimage", "mismatch",
                                "gold_dropped")},
        },
        "base": copy.deepcopy(summaries), "candidate": copy.deepcopy(summaries),
        "comparison": {"gates": {"absolute": absolute},
                       "promotion_pass": absolute},
    }


def _pair(value, input_sha):
    return {"pairs": 256, "presentations": 512, "accuracy": value,
            "exact_action_accuracy": value, "twin_both_correct": value,
            "input_sha256": input_sha, "model_revision": FROZEN_BASE_REVISION}


def _fixture(tmp_path):
    model = (tmp_path / "models--google--gemma-4-12B-it" / "snapshots"
             / FROZEN_BASE_REVISION)
    _write(model / "config.json", {})
    _write(model / "model.safetensors", b"model")
    inventory_path = _write(tmp_path / "model-inventory.json", build_inventory(model))
    inventory_payload = inventory_path.read_bytes()
    inventory_sha = _sha(inventory_payload)

    source_payload = b'{"source":"r044-r"}\n'
    source_path = _write(tmp_path / "source-provenance.json", source_payload)
    adapters = {
        "reference": _adapter(tmp_path / "reference", b"reference"),
        "control": _adapter(tmp_path / "control", b"control"),
        "treatment": _adapter(tmp_path / "treatment", b"treatment"),
    }
    reference = {
        "schema_version": "r047-reference-provenance-v1",
        "adapter_weights_sha256": adapters["reference"]["weights"],
        "adapter_config_sha256": adapters["reference"]["config"],
        "adapter_tree_sha256": adapters["reference"]["tree"],
        "adapter_tree_files": adapters["reference"]["files"],
        "model_revision": FROZEN_BASE_REVISION,
        "model_inventory_sha256": inventory_sha,
        "source_checkpoint_provenance_sha256": _sha(source_payload),
    }
    reference_path = _write(tmp_path / "reference-provenance.json", reference)
    recipe_path = _write(tmp_path / "recipe.json", {"schema_version": "test-recipe"})
    recipe_sha = file_hash(recipe_path)

    assignment = {field: _sha(field.encode())
                  for field in causal_comparator.ASSIGNMENT_FIELDS}
    assignment.update({
        "schema_version": "r047-causal-slot-assignment-v1",
        "control_slot": "slot_b", "treatment_slot": "slot_a",
        "control_train_sha256": gate.FROZEN_CONTROL_TRAIN_SHA256,
        "treatment_train_sha256": gate.FROZEN_TREATMENT_TRAIN_SHA256,
        "reference_provenance_sha256": file_hash(reference_path),
        "reference_adapter_sha256": adapters["reference"]["weights"],
        "parent_adapter_sha256": adapters["reference"]["weights"],
        "model_revision": FROZEN_BASE_REVISION,
        "model_inventory_sha256": inventory_sha,
        "recipe_sha256": recipe_sha,
        "bootstrap_seed": causal_comparator.FROZEN_BOOTSTRAP_SEED,
        "bootstrap_replicates": causal_comparator.MIN_BOOTSTRAP_REPLICATES,
    })
    assignment_path = _write(tmp_path / "assignment.json", assignment)
    assignment_payload = assignment_path.read_bytes()

    input_paths = {}
    mm_keys = {"mm_real": "scienceqa_test.jsonl",
               "mm_shuffled": "scienceqa_test_shuffled.jsonl",
               "mm_noimage": "scienceqa_test_noimage.jsonl",
               "mm_mismatch": "scienceqa_test_mismatch.jsonl",
               "mm_gold_dropped": "scienceqa_test_gold_dropped.jsonl"}
    for key in sorted(gate.INPUT_KEYS - {"mm_root", *mm_keys}):
        input_paths[key] = _write(tmp_path / "inputs" / f"{key}.jsonl", f"{key}\n")
    mm_root = tmp_path / "inputs" / "mm-root"
    _write(mm_root / "image.png", b"pixels")
    input_paths["mm_root"] = mm_root
    for key, filename in mm_keys.items():
        input_paths[key] = _write(mm_root / filename, f"{key}\n")
    code_paths = {
        key: (_write(tmp_path / "code" / gate.CODE_BASENAMES[key], f"# {key}\n")
              if key != "receipt_wrapper" else Path(gate.__file__))
        for key in gate.CODE_KEYS}
    manifest = gate.build_manifest(
        model_inventory_path=inventory_path, model_path=model,
        reference_provenance_path=reference_path,
        causal_assignment_path=assignment_path, causal_recipe_path=recipe_path,
        input_paths=input_paths, code_paths=code_paths)
    manifest_path = _write(tmp_path / "retention-manifest.json", manifest)
    manifest_payload = manifest_path.read_bytes()
    manifest_sha = file_hash(manifest_path)

    provenances = {"reference": reference_path}
    for role in ("control", "treatment"):
        slot = assignment[f"{role}_slot"]
        receipt = {
            "schema_version": "r047-causal-run-receipt-v1", "slot": slot,
            "train_sha256": assignment[f"{role}_train_sha256"],
            "recipe_sha256": recipe_sha,
            "parent_adapter_sha256": adapters["reference"]["weights"],
            "model_revision": FROZEN_BASE_REVISION,
            "adapter_sha256": adapters[role]["weights"],
            "config_sha256": _sha(f"{role}-config".encode()),
            "metrics_sha256": _sha(f"{role}-metrics".encode()),
            "log_sha256": _sha(f"{role}-log".encode()),
            "model_inventory_sha256": inventory_sha,
            "checkpoint_payload_sha256": adapters[role]["tree"],
            "checkpoint_step": 400, "status": "complete",
        }
        assert set(receipt) == causal_evaluator.RUN_RECEIPT_FIELDS
        provenances[role] = _write(tmp_path / f"{role}-run-receipt.json", receipt)

    bundles = {}
    for role in gate.ROLES:
        reports = {
            "retention": _retention(.9),
            "action": _action(.9),
            "renderer": _renderer(.9, manifest["inputs"]["renderer"]["sha256"]),
            "mmhe": _mmhe(.9, manifest["inputs"]),
            "pair": _pair(.9, manifest["inputs"]["pair"]["sha256"]),
        }
        evaluations = {}
        for suite, report in reports.items():
            root = tmp_path / "evaluations" / role / suite
            report_path = _write(root / "report.json", report)
            evidence = {"report": report_path}
            if "predictions" in gate.EVIDENCE_KEYS[suite]:
                evidence["predictions"] = _write(root / "predictions.jsonl", "{}\n")
            if "calibration_predictions" in gate.EVIDENCE_KEYS[suite]:
                evidence["calibration_predictions"] = _write(
                    root / "calibration-predictions.jsonl", "{}\n")
            receipt = gate._seal_completed_evaluation(
                manifest_payload=manifest_payload, manifest_sha256=manifest_sha,
                assignment_payload=assignment_payload, role=role, suite=suite,
                report_path=report_path, evidence_paths=evidence,
                adapter_path=adapters[role]["path"],
                provenance_payload=provenances[role].read_bytes(),
                model_inventory_payload=inventory_payload, model_path=model,
                input_paths=input_paths, code_paths=code_paths,
                source_checkpoint_provenance_payload=(
                    source_payload if role == "reference" else None))
            receipt_path = _write(root / "receipt.json", receipt)
            evaluations[suite] = {"report": report_path, "receipt": receipt_path,
                                  "evidence": evidence}
        bundles[role] = {"adapter": adapters[role]["path"],
                         "provenance": provenances[role],
                         "evaluations": evaluations}

    causal = {
        "schema_version": "r047-causal-comparison-v2",
        "preassignment": {
            "assignment_sha256": _sha(assignment_payload),
            "control_slot": assignment["control_slot"],
            "treatment_slot": assignment["treatment_slot"],
            "pre_frozen_slot_assignment": True,
            "evaluator_sha256": assignment["evaluator_sha256"],
            "comparator_sha256": assignment["comparator_sha256"],
            "bootstrap_seed": assignment["bootstrap_seed"],
            "bootstrap_replicates": assignment["bootstrap_replicates"],
        },
        "provenance": {
            "assignment": assignment,
            "run_receipt_sha256": {
                assignment["control_slot"]: file_hash(provenances["control"]),
                assignment["treatment_slot"]: file_hash(provenances["treatment"]),
            },
            "reference_provenance_sha256": file_hash(provenances["reference"]),
        },
        "co_primary_gate": {"passed": True},
    }
    values = {
        "manifest_payload": manifest_payload, "manifest_sha256": manifest_sha,
        "assignment_payload": assignment_payload, "recipe_payload": recipe_path.read_bytes(),
        "model_inventory_payload": inventory_payload, "model_path": model,
        "input_paths": input_paths, "code_paths": code_paths,
        "causal_comparison_payload": (json.dumps(causal, sort_keys=True) + "\n").encode(),
        "role_bundles": bundles,
        "source_checkpoint_provenance_payload": source_payload,
    }
    return values


def test_pass_requires_and_binds_all_three_models_and_all_suites(tmp_path):
    values = _fixture(tmp_path)
    result = gate.finalize(**values)
    assert result["causal_primary_pass"] is True
    assert result["all_required_reports_present"] is True
    assert result["arms"]["control"]["preservation_pass"] is True
    assert result["arms"]["treatment"]["preservation_pass"] is True
    assert result["treatment_advance_to_m2"] is True
    assert set(result["evaluation_report_sha256"]) == set(gate.ROLES)


def test_completed_treatment_regression_fails_without_fabricating_result(tmp_path):
    values = _fixture(tmp_path)
    evaluation = values["role_bundles"]["treatment"]["evaluations"]["retention"]
    report = json.loads(Path(evaluation["report"]).read_text())
    report["wf"]["accuracy"] = .87
    _write(evaluation["report"], report)
    receipt = gate._seal_completed_evaluation(
        manifest_payload=values["manifest_payload"],
        manifest_sha256=values["manifest_sha256"],
        assignment_payload=values["assignment_payload"], role="treatment",
        suite="retention", report_path=evaluation["report"],
        evidence_paths=evaluation["evidence"],
        adapter_path=values["role_bundles"]["treatment"]["adapter"],
        provenance_payload=Path(values["role_bundles"]["treatment"]["provenance"]).read_bytes(),
        model_inventory_payload=values["model_inventory_payload"],
        model_path=values["model_path"], input_paths=values["input_paths"],
        code_paths=values["code_paths"])
    _write(evaluation["receipt"], receipt)
    result = gate.finalize(**values)
    assert result["arms"]["treatment"]["gates"]["wf_drop_at_most_.02"] is False
    assert result["treatment_advance_to_m2"] is False


def test_rejects_report_tamper_after_evaluation_receipt(tmp_path):
    values = _fixture(tmp_path)
    report = values["role_bundles"]["treatment"]["evaluations"]["pair"]["report"]
    Path(report).write_text(Path(report).read_text().replace('"accuracy": 0.9',
                                                              '"accuracy": 0.8', 1))
    with pytest.raises(ValueError, match="evaluation receipt mismatch"):
        gate.finalize(**values)


def test_rejects_missing_suite_and_wrong_arm_receipt(tmp_path):
    values = _fixture(tmp_path)
    del values["role_bundles"]["control"]["evaluations"]["renderer"]
    with pytest.raises(ValueError, match="coverage is incomplete"):
        gate.finalize(**values)

    values = _fixture(tmp_path / "wrong-arm")
    values["role_bundles"]["treatment"]["provenance"] = (
        values["role_bundles"]["control"]["provenance"])
    with pytest.raises(ValueError, match="treatment run receipt differs"):
        gate.finalize(**values)


def test_rejects_wrong_adapter_report_receipt_swap(tmp_path):
    values = _fixture(tmp_path)
    control = values["role_bundles"]["control"]["evaluations"]["pair"]
    treatment = values["role_bundles"]["treatment"]["evaluations"]["pair"]
    control["report"] = treatment["report"]
    control["receipt"] = treatment["receipt"]
    control["evidence"] = treatment["evidence"]
    with pytest.raises(ValueError, match="control/pair evaluation receipt mismatch"):
        gate.finalize(**values)


def test_wrapper_executes_frozen_evaluator_before_receipt(tmp_path):
    values = _fixture(tmp_path)
    bundle = values["role_bundles"]["treatment"]
    calls = []

    class Completed:
        stdout = json.dumps(_retention(.9))

    def fake_run(command, cwd):
        calls.append((command, cwd))
        return Completed()

    output = gate.run_evaluation(
        manifest_payload=values["manifest_payload"],
        manifest_sha256=values["manifest_sha256"],
        assignment_payload=values["assignment_payload"], role="treatment",
        suite="retention", out=tmp_path / "wrapper-output",
        adapter_path=bundle["adapter"],
        provenance_payload=Path(bundle["provenance"]).read_bytes(),
        model_inventory_payload=values["model_inventory_payload"],
        model_path=values["model_path"], input_paths=values["input_paths"],
        code_paths=values["code_paths"], _run=fake_run)
    assert len(calls) == 1
    command = calls[0][0]
    assert command[1] == str(Path(values["code_paths"]["retention_evaluator"]).resolve())
    assert command[command.index("--adapter") + 1] == str(Path(bundle["adapter"]).resolve())
    receipt = json.loads(Path(output["receipt"]).read_text())
    assert receipt["adapter_tree_sha256"] == tree_hash(bundle["adapter"])[0]
    assert gate._valid_sha(receipt["execution_plan_sha256"])


def test_wrapper_failure_is_unpublished_and_same_path_can_rerun(tmp_path):
    values = _fixture(tmp_path)
    bundle = values["role_bundles"]["treatment"]
    output_path = tmp_path / "atomic-output"

    def fail(command, cwd):
        raise subprocess.CalledProcessError(1, command)

    arguments = {
        "manifest_payload": values["manifest_payload"],
        "manifest_sha256": values["manifest_sha256"],
        "assignment_payload": values["assignment_payload"], "role": "treatment",
        "suite": "retention", "out": output_path, "adapter_path": bundle["adapter"],
        "provenance_payload": Path(bundle["provenance"]).read_bytes(),
        "model_inventory_payload": values["model_inventory_payload"],
        "model_path": values["model_path"], "input_paths": values["input_paths"],
        "code_paths": values["code_paths"],
    }
    with pytest.raises(subprocess.CalledProcessError):
        gate.run_evaluation(**arguments, _run=fail)
    assert not output_path.exists()
    assert not list(tmp_path.glob(".atomic-output.tmp-*"))

    class Completed:
        stdout = json.dumps(_retention(.9))

    result = gate.run_evaluation(**arguments, _run=lambda command, cwd: Completed())
    assert Path(result["receipt"]).is_file()
    assert output_path.is_dir()


def test_rejects_rehashed_complete_control_treatment_assignment_swap(tmp_path):
    values = _fixture(tmp_path)
    manifest = json.loads(values["manifest_payload"])
    assignment = json.loads(values["assignment_payload"])
    assignment.update({
        "control_slot": "slot_a", "treatment_slot": "slot_b",
        "control_train_sha256": gate.FROZEN_TREATMENT_TRAIN_SHA256,
        "treatment_train_sha256": gate.FROZEN_CONTROL_TRAIN_SHA256,
    })
    payload = (json.dumps(assignment, indent=2, sort_keys=True) + "\n").encode()
    manifest["causal_assignment_sha256"] = _sha(payload)
    with pytest.raises(ValueError, match="causal assignment differs"):
        gate._validate_assignment(payload, manifest)


def test_rejects_tampered_evaluator_and_nonpassing_causal_gate(tmp_path):
    values = _fixture(tmp_path)
    Path(values["code_paths"]["pair_evaluator"]).write_text("# changed\n")
    with pytest.raises(ValueError, match="evaluator code pair_evaluator differs"):
        gate.finalize(**values)

    values = _fixture(tmp_path / "causal")
    causal = json.loads(values["causal_comparison_payload"])
    causal["co_primary_gate"]["passed"] = False
    values["causal_comparison_payload"] = json.dumps(causal).encode()
    result = gate.finalize(**values)
    assert result["causal_primary_pass"] is False
    assert result["arms"]["treatment"]["preservation_pass"] is True
    assert result["treatment_advance_to_m2"] is False
