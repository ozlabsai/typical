"""Freeze the R047-M1 inputs before either blinded training slot starts.

Requires a verified model inventory and writes only recipe/provenance/preassignment
inputs.  Candidate-output hashes deliberately do not exist at this stage.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts import compare_r047_causal as comparator
from scripts import eval_r047_executor as evaluator
from scripts import finalize_r047_m1 as finalizer
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, verify_inventory
from scripts.preflight_r043_assets import file_hash


FROZEN_R044_CHECKPOINT_PROVENANCE_SHA256 = (
    "d9968e9d57341c765ea5b5fef5d83f76022585e9dccd2177d34802bb7e495cb9")
RECIPE_NAME = "recipe.json"
REFERENCE_NAME = "r047-reference-provenance-v1.json"
ASSIGNMENT_NAME = "slot-assignment.json"
PREPARATION_NAME = "r047-preparation-receipt-v1.json"


def _write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _nonempty_hash(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return file_hash(path)


def _reference_provenance(parent_binding, model_inventory_sha256,
                          source_checkpoint_provenance):
    source_hash = _nonempty_hash(source_checkpoint_provenance, "R044 checkpoint provenance")
    if source_hash != FROZEN_R044_CHECKPOINT_PROVENANCE_SHA256:
        raise ValueError("R044 checkpoint provenance differs from the frozen source receipt")
    return {
        "schema_version": "r047-reference-provenance-v1",
        "adapter_weights_sha256": parent_binding["parent_adapter_sha256"],
        "adapter_config_sha256": parent_binding["parent_adapter_config_sha256"],
        "adapter_tree_sha256": parent_binding["parent_adapter_tree_sha256"],
        "adapter_tree_files": parent_binding["parent_adapter_tree_files"],
        "model_revision": FROZEN_BASE_REVISION,
        "model_inventory_sha256": model_inventory_sha256,
        "source_checkpoint_provenance_sha256": source_hash,
    }


def _assignment(*, control_sha256, treatment_sha256, recipe_sha256,
                reference_provenance_sha256, parent_adapter_sha256,
                model_inventory_sha256, evaluation_binding):
    return {
        "schema_version": "r047-causal-slot-assignment-v1",
        "control_slot": "slot_b",
        "treatment_slot": "slot_a",
        "control_train_sha256": control_sha256,
        "treatment_train_sha256": treatment_sha256,
        "reference_provenance_sha256": reference_provenance_sha256,
        "reference_adapter_sha256": parent_adapter_sha256,
        "parent_adapter_sha256": parent_adapter_sha256,
        "model_revision": FROZEN_BASE_REVISION,
        "model_inventory_sha256": model_inventory_sha256,
        "recipe_sha256": recipe_sha256,
        **evaluation_binding,
        "bootstrap_seed": comparator.FROZEN_BOOTSTRAP_SEED,
        "bootstrap_replicates": comparator.MIN_BOOTSTRAP_REPLICATES,
    }


def _validate_assignment(value):
    if (set(value) != comparator.ASSIGNMENT_FIELDS
            or value["schema_version"] != "r047-causal-slot-assignment-v1"
            or value["control_slot"] != "slot_b"
            or value["treatment_slot"] != "slot_a"
            or value["control_train_sha256"] != finalizer.KNOWN_ARM_SHA256["control"]
            or value["treatment_train_sha256"] != finalizer.KNOWN_ARM_SHA256["treatment"]
            or value["bootstrap_seed"] != comparator.FROZEN_BOOTSTRAP_SEED
            or value["bootstrap_replicates"] != comparator.MIN_BOOTSTRAP_REPLICATES):
        raise ValueError("slot assignment differs from the frozen comparator contract")
    hashes = comparator.ASSIGNMENT_FIELDS - {
        "schema_version", "control_slot", "treatment_slot", "model_revision",
        "bootstrap_seed", "bootstrap_replicates",
    }
    if any(not comparator._valid_sha(value[field]) for field in hashes):
        raise ValueError("slot assignment contains an invalid SHA-256")


def prepare(*, out, model_inventory_path, model_path, parent_adapter,
            arm_manifest_path, control_train_path, treatment_train_path, image_root,
            trainer_path, panel_path, panel_manifest_path, panel_preflight_path,
            inventory_manifest_path, evaluator_path, comparator_path, precheck_path,
            arm_preflight_path, source_checkpoint_provenance):
    """Validate frozen inputs, then write the three pre-training artifacts once."""
    out = Path(out)
    if out.exists():
        raise ValueError("R047-M1 preparation artifacts already exist")
    out.parent.mkdir(parents=True, exist_ok=True)

    _, model_inventory_sha256 = verify_inventory(
        model_inventory_path, model_path, FROZEN_BASE_REVISION)
    parent_binding = finalizer._validate_parent(parent_adapter)
    control_sha256, control_images = finalizer._validate_train(
        control_train_path, arm_manifest_path, image_root)
    treatment_sha256, treatment_images = finalizer._validate_train(
        treatment_train_path, arm_manifest_path, image_root)
    if control_images != treatment_images:
        raise ValueError("frozen R047 arms disagree about the image tree")
    evaluation_binding = finalizer._evaluation_bindings(
        panel_path, panel_manifest_path, panel_preflight_path, inventory_manifest_path,
        evaluator_path, comparator_path, precheck_path)
    arm_preflight_sha256 = finalizer._validate_arm_preflight(arm_preflight_path)
    if (evaluation_binding["panel_preflight_sha256"]
            != comparator.FROZEN_PANEL_PREFLIGHT_SHA256
            or evaluation_binding["inventory_manifest_sha256"]
            != comparator.FROZEN_INVENTORY_MANIFEST_SHA256):
        raise ValueError("R047 panel provenance differs from the frozen comparator contract")

    recipe = {
        "schema_version": finalizer.RECIPE_SCHEMA_VERSION,
        "base_revision": FROZEN_BASE_REVISION,
        **parent_binding,
        "model_inventory_sha256": model_inventory_sha256,
        "arm_manifest_sha256": finalizer.FROZEN_ARM_MANIFEST_SHA256,
        "arm_sha256": finalizer.KNOWN_ARM_SHA256,
        **control_images,
        "trainer_sha256": _nonempty_hash(trainer_path, "trainer implementation"),
        "arm_preflight_sha256": arm_preflight_sha256,
        **evaluation_binding,
        "bootstrap_replicates": comparator.MIN_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": comparator.FROZEN_BOOTSTRAP_SEED,
        "trainer_config": dict(finalizer.FROZEN_TRAINING_INVARIANTS),
    }
    reference = _reference_provenance(
        parent_binding, model_inventory_sha256, source_checkpoint_provenance)
    evaluator.validate_reference_provenance(json.dumps(reference).encode())

    with tempfile.TemporaryDirectory(prefix=f".{out.name}.tmp-", dir=out.parent) as temporary:
        staging = Path(temporary)
        recipe_path = staging / RECIPE_NAME
        reference_path = staging / REFERENCE_NAME
        assignment_path = staging / ASSIGNMENT_NAME
        preparation_path = staging / PREPARATION_NAME
        _write_new(recipe_path, recipe)
        finalizer._validate_recipe(recipe_path, model_inventory_sha256, trainer_path,
                                   parent_binding, control_images, evaluation_binding,
                                   arm_preflight_sha256)
        _write_new(reference_path, reference)
        assignment = _assignment(
            control_sha256=control_sha256, treatment_sha256=treatment_sha256,
            recipe_sha256=file_hash(recipe_path),
            reference_provenance_sha256=file_hash(reference_path),
            parent_adapter_sha256=parent_binding["parent_adapter_sha256"],
            model_inventory_sha256=model_inventory_sha256,
            evaluation_binding={**evaluation_binding,
                                "arm_preflight_sha256": arm_preflight_sha256})
        _validate_assignment(assignment)
        _write_new(assignment_path, assignment)
        preparation = {
            "schema_version": "r047-m1-preparation-receipt-v1",
            "status": "complete",
            "recipe_sha256": file_hash(recipe_path),
            "reference_provenance_sha256": file_hash(reference_path),
            "assignment_sha256": file_hash(assignment_path),
            "model_revision": FROZEN_BASE_REVISION,
            "model_inventory_sha256": model_inventory_sha256,
            "parent_adapter_sha256": parent_binding["parent_adapter_sha256"],
            "control_train_sha256": control_sha256,
            "treatment_train_sha256": treatment_sha256,
            "source_checkpoint_provenance_sha256": reference[
                "source_checkpoint_provenance_sha256"],
            "panel_sha256": evaluation_binding["panel_sha256"],
            "arm_preflight_sha256": arm_preflight_sha256,
            "trainer_sha256": recipe["trainer_sha256"],
            "evaluator_sha256": evaluation_binding["evaluator_sha256"],
            "comparator_sha256": evaluation_binding["comparator_sha256"],
            "precheck_sha256": evaluation_binding["precheck_sha256"],
            "preparer_sha256": file_hash(Path(__file__)),
            "finalizer_sha256": file_hash(Path(finalizer.__file__)),
        }
        _write_new(preparation_path, preparation)
        staging.replace(out)
    recipe_path = out / RECIPE_NAME
    reference_path = out / REFERENCE_NAME
    assignment_path = out / ASSIGNMENT_NAME
    preparation_path = out / PREPARATION_NAME
    return {
        "recipe": str(recipe_path), "recipe_sha256": file_hash(recipe_path),
        "reference_provenance": str(reference_path),
        "reference_provenance_sha256": file_hash(reference_path),
        "assignment": str(assignment_path), "assignment_sha256": file_hash(assignment_path),
        "preparation_receipt": str(preparation_path),
        "preparation_receipt_sha256": file_hash(preparation_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--parent-adapter", required=True)
    parser.add_argument("--arm-manifest", required=True)
    parser.add_argument("--control-train", required=True)
    parser.add_argument("--treatment-train", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--panel-preflight", required=True)
    parser.add_argument("--inventory-manifest", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--precheck", required=True)
    parser.add_argument("--arm-preflight", required=True)
    parser.add_argument("--reference-source-provenance", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        out=args.out, model_inventory_path=args.model_inventory, model_path=args.model,
        parent_adapter=args.parent_adapter, arm_manifest_path=args.arm_manifest,
        control_train_path=args.control_train, treatment_train_path=args.treatment_train,
        image_root=args.image_root, trainer_path=args.trainer, panel_path=args.panel,
        panel_manifest_path=args.panel_manifest, panel_preflight_path=args.panel_preflight,
        inventory_manifest_path=args.inventory_manifest, evaluator_path=args.evaluator,
        comparator_path=args.comparator, precheck_path=args.precheck,
        arm_preflight_path=args.arm_preflight,
        source_checkpoint_provenance=args.reference_source_provenance),
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
