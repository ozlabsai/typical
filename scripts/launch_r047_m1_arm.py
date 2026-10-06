"""Trusted, chronological, fail-closed launcher for one R047-M1 arm."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import finalize_r047_m1 as finalizer
from scripts import precheck_r047_causal as precheck
from scripts.atomic_publish import publish_directory_noreplace
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, verify_inventory
from scripts.preflight_r043_assets import file_hash


INTENT_SCHEMA = precheck.INTENT_SCHEMA
COMPLETION_SCHEMA = precheck.COMPLETION_SCHEMA
REFERENCE_KEYS = {
    "assignment", "reference_predictions", "reference_report",
    "reference_provenance", "recipe", "preparation_receipt", "panel",
    "panel_manifest", "panel_preflight", "inventory_manifest", "arm_preflight",
    "reference_receipt",
}
CONTROL_KEYS = {
    "slot_b_predictions", "slot_b_report", "slot_b_run_receipt", "control_receipt",
    "slot_b_launch_intent", "slot_b_launch_completion",
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _read(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0 or path.is_symlink():
        raise ValueError(f"{label} must be a nonempty regular file")
    return path.read_bytes()


def _object(payload, label):
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_payload(value))
    return path


def _reference_inputs(paths, *, require_frozen_panel, require_frozen_assignment):
    if set(paths) != REFERENCE_KEYS:
        raise ValueError("reference precheck paths are incomplete")
    payloads = {key: _read(path, key.replace("_", " ")) for key, path in paths.items()}
    return {
        "assignment_payload": payloads["assignment"],
        "assignment_sha256": _sha(payloads["assignment"]),
        "reference_predictions": payloads["reference_predictions"],
        "reference_report": payloads["reference_report"],
        "reference_report_sha256": _sha(payloads["reference_report"]),
        "reference_provenance_payload": payloads["reference_provenance"],
        "recipe_payload": payloads["recipe"],
        "preparation_receipt_payload": payloads["preparation_receipt"],
        "preparation_receipt_sha256": _sha(payloads["preparation_receipt"]),
        "panel_payload": payloads["panel"],
        "panel_manifest_payload": payloads["panel_manifest"],
        "panel_manifest_sha256": _sha(payloads["panel_manifest"]),
        "panel_preflight_payload": payloads["panel_preflight"],
        "inventory_manifest_payload": payloads["inventory_manifest"],
        "arm_preflight_payload": payloads["arm_preflight"],
        "_require_frozen_panel": require_frozen_panel,
        "_require_frozen_assignment": require_frozen_assignment,
    }


def _exact_receipt(generated, path, label):
    actual = _read(path, f"immutable {label} receipt")
    expected = precheck._receipt_payload(generated)
    if actual != expected:
        raise ValueError(f"immutable {label} receipt differs from fresh precheck")
    return _sha(actual)


def _rerun_prerequisites(slot, reference_paths, control_paths, *,
                         require_frozen_panel, require_frozen_assignment):
    reference_inputs = _reference_inputs(
        reference_paths, require_frozen_panel=require_frozen_panel,
        require_frozen_assignment=require_frozen_assignment)
    reference = precheck.precheck_reference(**reference_inputs)
    reference_sha = _exact_receipt(
        reference, reference_paths["reference_receipt"], "reference")
    if slot == "slot_b":
        if control_paths not in (None, {}):
            raise ValueError("control artifacts are forbidden before slot_b training")
        return {"reference_receipt_sha256": reference_sha}, reference_inputs
    if not isinstance(control_paths, dict) or set(control_paths) != CONTROL_KEYS:
        raise ValueError("slot_a requires complete immutable control evidence")
    control_payloads = {key: _read(path, key.replace("_", " "))
                        for key, path in control_paths.items()}
    control = precheck.precheck_control(
        **reference_inputs,
        reference_receipt_payload=_read(
            reference_paths["reference_receipt"], "immutable reference receipt"),
        reference_receipt_sha256=reference_sha,
        slot_b_predictions=control_payloads["slot_b_predictions"],
        slot_b_report=control_payloads["slot_b_report"],
        slot_b_report_sha256=_sha(control_payloads["slot_b_report"]),
        slot_b_run_receipt=control_payloads["slot_b_run_receipt"],
        slot_b_run_receipt_sha256=_sha(control_payloads["slot_b_run_receipt"]),
        slot_b_launch_intent_payload=control_payloads["slot_b_launch_intent"],
        slot_b_launch_intent_sha256=_sha(control_payloads["slot_b_launch_intent"]),
        slot_b_launch_completion_payload=control_payloads["slot_b_launch_completion"],
        slot_b_launch_completion_sha256=_sha(control_payloads["slot_b_launch_completion"]),
    )
    control_sha = _exact_receipt(control, control_paths["control_receipt"], "control")
    return {"reference_receipt_sha256": reference_sha,
            "control_receipt_sha256": control_sha}, reference_inputs


def _validate_launch_inputs(*, slot, train_path, recipe_payload, assignment_payload,
                            trainer_path, model_inventory_path, model_path, parent_adapter,
                            arm_manifest_path, image_root, source_checkpoint_provenance_path,
                            evaluator_path, comparator_path, precheck_path,
                            require_frozen_assignment):
    if slot not in {"slot_a", "slot_b"}:
        raise ValueError("R047 launch slot must be slot_b/control or slot_a/treatment")
    assignment = _object(assignment_payload, "assignment")
    recipe = _object(recipe_payload, "recipe")
    role = "control" if slot == "slot_b" else "treatment"
    if (assignment.get(f"{role}_slot") != slot
            or _sha(recipe_payload) != assignment.get("recipe_sha256")):
        raise ValueError("requested slot does not match the frozen role/recipe")
    if require_frozen_assignment:
        precheck._validate_frozen_recipe(recipe, assignment)
    elif (recipe.get("schema_version") != finalizer.RECIPE_SCHEMA_VERSION
          or recipe.get("trainer_config") != finalizer.FROZEN_TRAINING_INVARIANTS):
        raise ValueError("recipe differs from the complete R047 trainer contract")
    train_sha = file_hash(train_path)
    if (train_sha != assignment.get(f"{role}_train_sha256")
            or train_sha != recipe.get("arm_sha256", {}).get(role)):
        raise ValueError("requested slot/train payload differs from the frozen assignment")
    if file_hash(trainer_path) != recipe.get("trainer_sha256"):
        raise ValueError("trainer implementation differs from the frozen recipe")
    for name, path in (("evaluator", evaluator_path), ("comparator", comparator_path),
                       ("precheck", precheck_path)):
        if file_hash(path) != assignment.get(f"{name}_sha256"):
            raise ValueError(f"{name} implementation differs from frozen assignment")
    _, inventory_sha = verify_inventory(
        model_inventory_path, model_path, FROZEN_BASE_REVISION)
    if (inventory_sha != recipe.get("model_inventory_sha256")
            or inventory_sha != assignment.get("model_inventory_sha256")):
        raise ValueError("model inventory differs from the frozen recipe")
    parent = finalizer._validate_parent(parent_adapter)
    if any(recipe.get(key) != value for key, value in parent.items()):
        raise ValueError("parent adapter differs from the frozen recipe")
    checked_train_sha, image_binding = finalizer._validate_train(
        train_path, arm_manifest_path, image_root)
    if checked_train_sha != train_sha or any(recipe.get(key) != value
                                             for key, value in image_binding.items()):
        raise ValueError("training/image payload differs from the frozen recipe")
    source_payload = _read(source_checkpoint_provenance_path,
                           "source checkpoint provenance")
    source_sha = _sha(source_payload)
    return role, assignment, recipe, train_sha, inventory_sha, parent, source_sha


def _trainer_command(trainer_path, model_path, parent_adapter, train_path, run_out, config):
    command = [sys.executable, str(Path(trainer_path).resolve()),
               "--model", str(Path(model_path).resolve()),
               "--init-adapter", str(Path(parent_adapter).resolve().parent),
               "--train", str(Path(train_path).resolve()),
               "--out", str(Path(run_out).resolve())]
    flags = {
        "steps": "steps", "microbatch": "microbatch", "grad_accum": "grad-accum",
        "max_length": "max-length", "lr": "lr", "min_lr": "min-lr",
        "weight_decay": "weight-decay", "rank": "rank", "alpha": "alpha",
        "save_every": "save-every", "save_steps": "save-steps",
        "eval_every": "eval-every", "eval_samples": "eval-samples", "seed": "seed",
        "pair_consistency_weight": "pair-consistency-weight",
        "pair_consistency_min_gold_probability": "pair-consistency-min-gold-probability",
        "pair_consistency_clip": "pair-consistency-clip",
        "pair_consistency_beta": "pair-consistency-beta",
        "pair_flip_weight": "pair-flip-weight", "pair_flip_margin": "pair-flip-margin",
        "pair_flip_start_step": "pair-flip-start-step",
    }
    for key, flag in flags.items():
        command += [f"--{flag}", str(config[key])]
    if config["project_consistency_against_guard"]:
        command.append("--project-consistency-against-guard")
    if config["stage_counterfactual_pairs"]:
        command.append("--stage-counterfactual-pairs")
    return command


def _run_checked(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def launch(*, slot, out, train_path, arm_manifest_path, image_root, model_inventory_path,
           model_path, parent_adapter, trainer_path, evaluator_path, comparator_path,
           precheck_path, source_checkpoint_provenance_path, reference_paths,
           control_paths=None, require_frozen_panel=True, require_frozen_assignment=True,
           _run=_run_checked):
    """Re-run mandatory prechecks, execute the trainer, and atomically publish evidence."""
    final_out = Path(out)
    if final_out.exists():
        raise ValueError("refusing to overwrite R047 training output")
    prerequisites, reference_inputs = _rerun_prerequisites(
        slot, reference_paths, control_paths,
        require_frozen_panel=require_frozen_panel,
        require_frozen_assignment=require_frozen_assignment)
    assignment_payload = reference_inputs["assignment_payload"]
    recipe_payload = reference_inputs["recipe_payload"]
    preparation = _object(reference_inputs["preparation_receipt_payload"],
                          "preparation receipt")
    launcher_sha = file_hash(Path(__file__))
    if launcher_sha != preparation.get("launcher_sha256"):
        raise ValueError("running launcher differs from frozen preparation receipt")
    role, assignment, recipe, train_sha, inventory_sha, parent, source_sha = (
        _validate_launch_inputs(
            slot=slot, train_path=train_path, recipe_payload=recipe_payload,
            assignment_payload=assignment_payload, trainer_path=trainer_path,
            model_inventory_path=model_inventory_path, model_path=model_path,
            parent_adapter=parent_adapter, arm_manifest_path=arm_manifest_path,
            image_root=image_root,
            source_checkpoint_provenance_path=source_checkpoint_provenance_path,
            evaluator_path=evaluator_path, comparator_path=comparator_path,
            precheck_path=precheck_path,
            require_frozen_assignment=require_frozen_assignment))
    reference_provenance = _object(
        reference_inputs["reference_provenance_payload"], "reference provenance")
    if source_sha != reference_provenance.get("source_checkpoint_provenance_sha256"):
        raise ValueError("source checkpoint provenance differs from frozen reference")
    plan = precheck.launch_command_plan(
        slot=slot, role=role, assignment_sha=_sha(assignment_payload),
        assignment=assignment, recipe=recipe,
        train_sha=train_sha, inventory_sha=inventory_sha, parent=parent,
        source_sha=source_sha, prerequisites=prerequisites)
    plan_sha = _sha(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())

    final_out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{final_out.name}.tmp-", dir=final_out.parent))
    try:
        intent = {
            "schema_version": INTENT_SCHEMA, "status": "invoking_trainer",
            "slot": slot, "role": role, "command_plan_sha256": plan_sha,
            "assignment_sha256": _sha(assignment_payload),
            "recipe_sha256": _sha(recipe_payload), "train_sha256": train_sha,
            "model_inventory_sha256": inventory_sha,
            "parent_adapter_sha256": parent["parent_adapter_sha256"],
            "parent_adapter_tree_sha256": parent["parent_adapter_tree_sha256"],
            "source_checkpoint_provenance_sha256": source_sha,
            "preparation_receipt_sha256": reference_inputs["preparation_receipt_sha256"],
            "launcher_sha256": launcher_sha,
            "prerequisite_receipt_sha256": prerequisites,
        }
        intent_path = _write_new(scratch / "launch-intent.json", intent)
        run_out = scratch / "run"
        command = _trainer_command(
            trainer_path, model_path, parent_adapter, train_path, run_out,
            recipe["trainer_config"])
        result = _run(command, Path(trainer_path).resolve().parent.parent)
        log_path = scratch / "train.log"
        log_path.write_text((getattr(result, "stdout", "") or "")
                            + (getattr(result, "stderr", "") or ""))
        if log_path.stat().st_size <= 0:
            raise ValueError("trainer completed without a nonempty execution log")
        run_receipt = finalizer.finalize(
            slot=slot, train_path=train_path, arm_manifest_path=arm_manifest_path,
            recipe_path=reference_paths["recipe"], config_path=run_out / "config.json",
            metrics_path=run_out / "metrics.jsonl", log_path=log_path,
            checkpoint_path=run_out / "step-000400", parent_adapter=parent_adapter,
            image_root=image_root, model_inventory_path=model_inventory_path,
            trainer_path=trainer_path, panel_path=reference_paths["panel"],
            panel_manifest_path=reference_paths["panel_manifest"],
            panel_preflight_path=reference_paths["panel_preflight"],
            inventory_manifest_path=reference_paths["inventory_manifest"],
            evaluator_path=evaluator_path, comparator_path=comparator_path,
            precheck_path=precheck_path, arm_preflight_path=reference_paths["arm_preflight"],
            base_revision=FROZEN_BASE_REVISION, model_path=model_path)
        run_receipt_path = _write_new(scratch / "run-receipt.json", run_receipt)
        completion = {
            "schema_version": COMPLETION_SCHEMA, "status": "trainer_completed",
            "slot": slot, "role": role, "command_plan_sha256": plan_sha,
            "launch_intent_sha256": file_hash(intent_path),
            "run_receipt_sha256": file_hash(run_receipt_path),
            "prerequisite_receipt_sha256": prerequisites,
        }
        _write_new(scratch / "launch-completion.json", completion)
        publish_directory_noreplace(scratch, final_out)
        return {"out": str(final_out), "intent": str(final_out / "launch-intent.json"),
                "completion": str(final_out / "launch-completion.json"),
                "run_receipt": str(final_out / "run-receipt.json")}
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=("slot_b", "slot_a"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--arm-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--parent-adapter", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--precheck", required=True)
    parser.add_argument("--reference-source-provenance", required=True)
    parser.add_argument("--reference-spec", required=True)
    parser.add_argument("--control-spec")
    args = parser.parse_args(argv)
    reference_paths = _object(_read(args.reference_spec, "reference spec"), "reference spec")
    control_paths = (_object(_read(args.control_spec, "control spec"), "control spec")
                     if args.control_spec else None)
    result = launch(
        slot=args.slot, out=args.out, train_path=args.train,
        arm_manifest_path=args.arm_manifest, image_root=args.image_root,
        model_inventory_path=args.model_inventory, model_path=args.model,
        parent_adapter=args.parent_adapter, trainer_path=args.trainer,
        evaluator_path=args.evaluator, comparator_path=args.comparator,
        precheck_path=args.precheck,
        source_checkpoint_provenance_path=args.reference_source_provenance,
        reference_paths=reference_paths, control_paths=control_paths)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
