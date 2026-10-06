"""Fail-closed reference-first and control-ceiling prechecks for R047-M1."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from scripts import compare_r047_causal as comparator
from scripts import eval_r047_executor as evaluator


REFERENCE_SCHEMA = "r047-causal-reference-precheck-v2"
CONTROL_SCHEMA = "r047-causal-control-precheck-v2"
INTENT_SCHEMA = "r047-m1-launch-intent-v2"
COMPLETION_SCHEMA = "r047-m1-launch-completion-v2"
CONTROL_CEILING = .95
PREPARATION_FIELDS = {
    "schema_version", "status", "recipe_sha256", "reference_provenance_sha256",
    "assignment_sha256", "model_revision", "model_inventory_sha256",
    "parent_adapter_sha256", "control_train_sha256", "treatment_train_sha256",
    "source_checkpoint_provenance_sha256", "panel_sha256",
    "arm_preflight_sha256", "trainer_sha256", "evaluator_sha256",
    "comparator_sha256", "precheck_sha256", "preparer_sha256",
    "finalizer_sha256", "scalar_scorer_sha256", "launcher_sha256",
    "atomic_publisher_sha256",
    "superseded_preparation_receipt_sha256", "protocol_amendment_sha256",
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _receipt_payload(receipt):
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _with_binding(receipt):
    receipt["binding_sha256"] = _sha(evaluator._canonical_json(receipt).encode())
    return receipt


def launch_command_plan(*, slot, role, assignment_sha, assignment, recipe,
                        train_sha, inventory_sha, parent, source_sha, prerequisites):
    """Describe the frozen invocation without local paths or launcher imports."""
    return {
        "schema_version": "r047-m1-trainer-command-plan-v2",
        "slot": slot, "role": role,
        "assignment_sha256": assignment_sha,
        "recipe_sha256": assignment["recipe_sha256"],
        "train_sha256": train_sha, "model_revision": assignment["model_revision"],
        "model_inventory_sha256": inventory_sha,
        "parent_adapter_sha256": parent["parent_adapter_sha256"],
        "parent_adapter_tree_sha256": parent["parent_adapter_tree_sha256"],
        "source_checkpoint_provenance_sha256": source_sha,
        "trainer_sha256": recipe["trainer_sha256"],
        "trainer_config": recipe["trainer_config"],
        "validation": [], "prerequisites": prerequisites,
    }


def _validate_frozen_recipe(recipe, assignment):
    """Recompute the complete production recipe contract without trusting its hash."""
    from scripts import finalize_r047_m1 as finalizer
    from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION

    direct = {
        "schema_version": finalizer.RECIPE_SCHEMA_VERSION,
        "base_revision": FROZEN_BASE_REVISION,
        "parent_adapter_sha256": finalizer.FROZEN_PARENT_SHA256,
        "parent_adapter_tree_sha256": finalizer.FROZEN_PARENT_TREE_SHA256,
        "parent_adapter_tree_files": finalizer.FROZEN_PARENT_TREE_FILES,
        "parent_adapter_config_sha256": finalizer.FROZEN_PARENT_CONFIG_SHA256,
        "model_inventory_sha256": assignment["model_inventory_sha256"],
        "arm_manifest_sha256": finalizer.FROZEN_ARM_MANIFEST_SHA256,
        "arm_sha256": finalizer.KNOWN_ARM_SHA256,
        "image_tree_sha256": finalizer.FROZEN_IMAGE_TREE_SHA256,
        "image_tree_files": finalizer.FROZEN_IMAGE_TREE_FILES,
        "trainer_sha256": _sha(
            Path(__file__).with_name("train_gemma_decision_lora.py").read_bytes()),
        "arm_preflight_sha256": assignment["arm_preflight_sha256"],
        "panel_sha256": assignment["panel_sha256"],
        "panel_manifest_sha256": assignment["panel_manifest_sha256"],
        "panel_preflight_sha256": assignment["panel_preflight_sha256"],
        "inventory_manifest_sha256": assignment["inventory_manifest_sha256"],
        "evaluator_sha256": assignment["evaluator_sha256"],
        "comparator_sha256": assignment["comparator_sha256"],
        "precheck_sha256": assignment["precheck_sha256"],
        "bootstrap_replicates": comparator.MIN_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": comparator.FROZEN_BOOTSTRAP_SEED,
    }
    config = recipe.get("trainer_config") if isinstance(recipe, dict) else None
    if (not isinstance(recipe, dict) or set(recipe) != finalizer.RECIPE_FIELDS
            or any(recipe.get(key) != value for key, value in direct.items())
            or not isinstance(config, dict)
            or set(config) != finalizer.TRAINER_CONFIG_KEYS
            or any(not finalizer._same(config.get(key), value)
                   for key, value in finalizer.FROZEN_TRAINING_INVARIANTS.items())
            or not finalizer._finite(recipe)):
        raise ValueError("recipe differs from the complete frozen R047-M1 contract")


def _assignment_context(*, assignment_payload, assignment_sha256, panel_payload,
                        panel_manifest_payload, panel_manifest_sha256,
                        panel_preflight_payload, inventory_manifest_payload,
                        arm_preflight_payload, reference_provenance_payload,
                        recipe_payload, preparation_receipt_payload,
                        preparation_receipt_sha256,
                        _require_frozen_panel=True, _require_frozen_assignment=True):
    if (not comparator._valid_sha(assignment_sha256)
            or _sha(assignment_payload) != assignment_sha256):
        raise ValueError("frozen slot-assignment hash mismatch")
    try:
        assignment = json.loads(assignment_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid frozen slot assignment") from error
    if not isinstance(assignment, dict) or set(assignment) != comparator.ASSIGNMENT_FIELDS:
        raise ValueError("slot assignment schema differs from the frozen contract")
    if (assignment.get("schema_version") != "r047-causal-slot-assignment-v2"
            or assignment.get("control_slot") != "slot_b"
            or assignment.get("treatment_slot") != "slot_a"):
        raise ValueError("invalid pre-frozen control/treatment slot assignment")
    hash_fields = comparator.ASSIGNMENT_FIELDS - {
        "schema_version", "control_slot", "treatment_slot", "model_revision",
        "bootstrap_seed", "bootstrap_replicates"}
    if any(not comparator._valid_sha(assignment.get(field)) for field in hash_fields):
        raise ValueError("slot assignment contains an invalid SHA-256")
    if not isinstance(assignment.get("model_revision"), str) or not assignment["model_revision"]:
        raise ValueError("slot assignment model revision is invalid")
    frozen_reference = None
    if _require_frozen_assignment:
        from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION
        from scripts.finalize_r047_m1 import (
            FROZEN_PARENT_CONFIG_SHA256,
            FROZEN_PARENT_SHA256,
            FROZEN_PARENT_TREE_FILES,
            FROZEN_PARENT_TREE_SHA256,
            KNOWN_ARM_SHA256,
        )
        from scripts.prepare_r047_m1 import FROZEN_R044_CHECKPOINT_PROVENANCE_SHA256

        if (assignment["control_train_sha256"] != KNOWN_ARM_SHA256["control"]
                or assignment["treatment_train_sha256"] != KNOWN_ARM_SHA256["treatment"]):
            raise ValueError("slot assignment train hashes differ from the frozen arms")
        if (assignment["parent_adapter_sha256"] != FROZEN_PARENT_SHA256
                or assignment["reference_adapter_sha256"] != FROZEN_PARENT_SHA256
                or assignment["model_revision"] != FROZEN_BASE_REVISION):
            raise ValueError("slot assignment differs from the frozen reference parent")
        frozen_reference = {
            "adapter_config_sha256": FROZEN_PARENT_CONFIG_SHA256,
            "adapter_tree_sha256": FROZEN_PARENT_TREE_SHA256,
            "adapter_tree_files": FROZEN_PARENT_TREE_FILES,
            "source_checkpoint_provenance_sha256":
                FROZEN_R044_CHECKPOINT_PROVENANCE_SHA256,
        }

    comparator_sha256 = _sha(Path(comparator.__file__).read_bytes())
    evaluator_sha256 = _sha(Path(evaluator.__file__).read_bytes())
    precheck_sha256 = _sha(Path(__file__).read_bytes())
    launcher_sha256 = _sha(Path(__file__).with_name("launch_r047_m1_arm.py").read_bytes())
    scalar_scorer_sha256 = _sha(
        Path(__file__).with_name("singleton_option_scorer.py").read_bytes())
    atomic_publisher_sha256 = _sha(
        Path(__file__).with_name("atomic_publish.py").read_bytes())
    if (assignment["bootstrap_seed"] != comparator.FROZEN_BOOTSTRAP_SEED
            or assignment["bootstrap_replicates"] != comparator.MIN_BOOTSTRAP_REPLICATES):
        raise ValueError("slot assignment bootstrap differs from the frozen contract")
    if assignment["comparator_sha256"] != comparator_sha256:
        raise ValueError("running comparator differs from frozen preassignment")
    if assignment["evaluator_sha256"] != evaluator_sha256:
        raise ValueError("running evaluator differs from frozen preassignment")
    if ("precheck_sha256" in assignment
            and assignment["precheck_sha256"] != precheck_sha256):
        raise ValueError("running precheck differs from frozen preassignment")
    if assignment["launcher_sha256"] != launcher_sha256:
        raise ValueError("running launcher differs from frozen preassignment")
    if assignment["scalar_scorer_sha256"] != scalar_scorer_sha256:
        raise ValueError("running scalar scorer differs from frozen preassignment")
    if assignment["atomic_publisher_sha256"] != atomic_publisher_sha256:
        raise ValueError("running atomic publisher differs from frozen preassignment")
    if _require_frozen_panel and (
            assignment["panel_preflight_sha256"]
            != comparator.FROZEN_PANEL_PREFLIGHT_SHA256
            or assignment["inventory_manifest_sha256"]
            != comparator.FROZEN_INVENTORY_MANIFEST_SHA256
            or assignment["arm_preflight_sha256"]
            != comparator.FROZEN_ARM_PREFLIGHT_SHA256):
        raise ValueError("panel provenance differs from frozen preassignment")
    if assignment["reference_adapter_sha256"] != assignment["parent_adapter_sha256"]:
        raise ValueError("frozen reference adapter must be the exact parent adapter")

    if (not comparator._valid_sha(panel_manifest_sha256)
            or _sha(panel_manifest_payload) != panel_manifest_sha256
            or assignment["panel_manifest_sha256"] != panel_manifest_sha256):
        raise ValueError("frozen panel-manifest hash mismatch")
    if _sha(panel_payload) != assignment["panel_sha256"]:
        raise ValueError("raw panel hash differs from frozen slot assignment")
    if (_sha(panel_preflight_payload) != assignment["panel_preflight_sha256"]
            or _sha(inventory_manifest_payload) != assignment["inventory_manifest_sha256"]
            or _sha(arm_preflight_payload) != assignment["arm_preflight_sha256"]):
        raise ValueError("panel preflight/inventory hash differs from frozen assignment")
    comparator.validate_causal_supporting_provenance(
        panel_payload, panel_manifest_payload, panel_preflight_payload,
        inventory_manifest_payload)

    if _sha(reference_provenance_payload) != assignment["reference_provenance_sha256"]:
        raise ValueError("reference provenance hash differs from frozen assignment")
    reference_provenance = evaluator.validate_reference_provenance(
        reference_provenance_payload)
    if (reference_provenance["adapter_weights_sha256"]
            != assignment["reference_adapter_sha256"]
            or reference_provenance["model_revision"] != assignment["model_revision"]
            or reference_provenance["model_inventory_sha256"]
            != assignment["model_inventory_sha256"]):
        raise ValueError("reference provenance differs from frozen model/adapter assignment")

    if _sha(recipe_payload) != assignment["recipe_sha256"]:
        raise ValueError("recipe hash differs from frozen assignment")
    try:
        recipe = json.loads(recipe_payload)
        preparation = json.loads(preparation_receipt_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid recipe or preparation receipt JSON") from error
    if _require_frozen_assignment:
        _validate_frozen_recipe(recipe, assignment)
    elif (not isinstance(recipe, dict)
          or recipe.get("schema_version") != "r047-m1-recipe-v1"):
        raise ValueError("recipe schema differs from the frozen R047-M1 contract")
    if (not comparator._valid_sha(preparation_receipt_sha256)
            or _sha(preparation_receipt_payload) != preparation_receipt_sha256):
        raise ValueError("preparation receipt hash mismatch")
    from scripts.prepare_r047_m1 import _protocol_amendment_bindings

    preparation_lineage = _protocol_amendment_bindings()
    if any(assignment[field] != value for field, value in preparation_lineage.items()):
        raise ValueError(
            "slot assignment/preparation receipt protocol lineage differs from current authority")
    expected_preparation = {
        "schema_version": "r047-m1-preparation-receipt-v2",
        "status": "complete",
        "recipe_sha256": assignment["recipe_sha256"],
        "reference_provenance_sha256": assignment["reference_provenance_sha256"],
        "assignment_sha256": assignment_sha256,
        "model_revision": assignment["model_revision"],
        "model_inventory_sha256": assignment["model_inventory_sha256"],
        "parent_adapter_sha256": assignment["parent_adapter_sha256"],
        "control_train_sha256": assignment["control_train_sha256"],
        "treatment_train_sha256": assignment["treatment_train_sha256"],
        "source_checkpoint_provenance_sha256": reference_provenance[
            "source_checkpoint_provenance_sha256"],
        "panel_sha256": assignment["panel_sha256"],
        "arm_preflight_sha256": assignment["arm_preflight_sha256"],
        "trainer_sha256": recipe.get("trainer_sha256"),
        "evaluator_sha256": evaluator_sha256,
        "comparator_sha256": comparator_sha256,
        "precheck_sha256": precheck_sha256,
        "preparer_sha256": _sha(
            Path(__file__).with_name("prepare_r047_m1.py").read_bytes()),
        "finalizer_sha256": _sha(
            Path(__file__).with_name("finalize_r047_m1.py").read_bytes()),
        "scalar_scorer_sha256": assignment["scalar_scorer_sha256"],
        "launcher_sha256": assignment["launcher_sha256"],
        "atomic_publisher_sha256": assignment["atomic_publisher_sha256"],
        "superseded_preparation_receipt_sha256": assignment[
            "superseded_preparation_receipt_sha256"],
        "protocol_amendment_sha256": assignment["protocol_amendment_sha256"],
    }
    if (not isinstance(preparation, dict)
            or set(preparation) != PREPARATION_FIELDS
            or preparation != expected_preparation):
        raise ValueError("preparation receipt differs from frozen launch inputs")
    if (frozen_reference is not None
            and any(reference_provenance[field] != expected
                    for field, expected in frozen_reference.items())):
        raise ValueError("reference provenance differs from the frozen reference parent")

    panel_rows, panel_manifest = comparator.validate_causal_panel_payload(
        panel_payload, panel_manifest_payload, require_frozen=_require_frozen_panel)
    parity_rows = evaluator.causal_parity_subset(panel_rows)
    common = {
        "model_revision": assignment["model_revision"],
        "model_inventory_sha256": assignment["model_inventory_sha256"],
        "panel_preflight_sha256": assignment["panel_preflight_sha256"],
        "inventory_manifest_sha256": assignment["inventory_manifest_sha256"],
        "evaluator_sha256": assignment["evaluator_sha256"],
        "scalar_scorer_sha256": assignment["scalar_scorer_sha256"],
        "parity_rows": len(parity_rows),
        "parity_subset_ids_sha256": _sha(
            ("\n".join(row["id"] for row in parity_rows) + "\n").encode()),
        "rows": panel_manifest["outputs"]["primary"]["rows"],
        "clusters": panel_manifest["clusters"]["causal_primary"],
    }
    return {
        "assignment": assignment,
        "assignment_sha256": assignment_sha256,
        "comparator_sha256": comparator_sha256,
        "evaluator_sha256": evaluator_sha256,
        "precheck_sha256": precheck_sha256,
        "preparation_receipt_sha256": preparation_receipt_sha256,
        "launcher_sha256": assignment["launcher_sha256"],
        "scalar_scorer_sha256": assignment["scalar_scorer_sha256"],
        "atomic_publisher_sha256": assignment["atomic_publisher_sha256"],
        "preparation_lineage": preparation_lineage,
        "recipe": recipe,
        "panel_rows": panel_rows,
        "panel_manifest_sha256": panel_manifest_sha256,
        "reference_provenance": reference_provenance,
        "common": common,
    }


def _reference_precheck(*, reference_predictions, reference_report,
                        reference_report_sha256, **common_inputs):
    context = _assignment_context(**common_inputs)
    assignment = context["assignment"]
    records = comparator._read_payload(reference_predictions)
    comparator._bind_predictions_to_panel(records, context["panel_rows"])
    comparator.validate_panel_coverage(records)
    provenance = context["reference_provenance"]
    expectation = {
        **context["common"], "provenance_role": "reference",
        "adapter_sha256": assignment["reference_adapter_sha256"],
        "reference_provenance_sha256": assignment["reference_provenance_sha256"],
        "adapter_config_sha256": provenance["adapter_config_sha256"],
        "adapter_tree_sha256": provenance["adapter_tree_sha256"],
        "adapter_tree_files": provenance["adapter_tree_files"],
        "source_checkpoint_provenance_sha256": provenance[
            "source_checkpoint_provenance_sha256"],
    }
    report = comparator._load_evaluator_report(
        reference_report, reference_report_sha256, "reference", reference_predictions,
        assignment["panel_sha256"], context["panel_manifest_sha256"], expectation)
    metrics = {f"K{arity}": comparator.causal_metrics(records, arity)
               for arity in (2, 3)}
    receipt = _with_binding({
        "schema_version": REFERENCE_SCHEMA,
        "mode": "reference",
        "status": "pass",
        "assignment": {
            "sha256": context["assignment_sha256"],
            "control_slot": assignment["control_slot"],
            "treatment_slot": assignment["treatment_slot"],
        },
        "artifacts": {
            "assignment_sha256": context["assignment_sha256"],
            "reference_report_sha256": reference_report_sha256,
            "reference_predictions_sha256": _sha(reference_predictions),
            "reference_provenance_sha256": assignment["reference_provenance_sha256"],
            "reference_adapter_sha256": assignment["reference_adapter_sha256"],
            "reference_adapter_config_sha256": provenance["adapter_config_sha256"],
            "reference_adapter_tree_sha256": provenance["adapter_tree_sha256"],
            "source_checkpoint_provenance_sha256": provenance[
                "source_checkpoint_provenance_sha256"],
            "model_revision": assignment["model_revision"],
            "model_inventory_sha256": assignment["model_inventory_sha256"],
            "model_tree_sha256": report["model_tree_sha256"],
            "panel_sha256": assignment["panel_sha256"],
            "panel_manifest_sha256": assignment["panel_manifest_sha256"],
            "panel_preflight_sha256": assignment["panel_preflight_sha256"],
            "inventory_manifest_sha256": assignment["inventory_manifest_sha256"],
            "arm_preflight_sha256": assignment["arm_preflight_sha256"],
            "control_train_sha256": assignment["control_train_sha256"],
            "treatment_train_sha256": assignment["treatment_train_sha256"],
            "parent_adapter_sha256": assignment["parent_adapter_sha256"],
            "recipe_sha256": assignment["recipe_sha256"],
            "preparation_receipt_sha256": context[
                "preparation_receipt_sha256"],
            "launcher_sha256": context["launcher_sha256"],
            "scalar_scorer_sha256": context["scalar_scorer_sha256"],
            "atomic_publisher_sha256": context["atomic_publisher_sha256"],
            **context["preparation_lineage"],
            "evaluator_sha256": context["evaluator_sha256"],
            "comparator_sha256": context["comparator_sha256"],
            "precheck_sha256": context["precheck_sha256"],
        },
        "metrics": metrics,
    })
    return receipt, context, records, report


def precheck_reference(*, reference_predictions, reference_report,
                       reference_report_sha256, assignment_payload, assignment_sha256,
                       panel_payload, panel_manifest_payload, panel_manifest_sha256,
                       panel_preflight_payload, inventory_manifest_payload,
                       arm_preflight_payload, reference_provenance_payload,
                       recipe_payload, preparation_receipt_payload,
                       preparation_receipt_sha256,
                       _require_frozen_panel=True, _require_frozen_assignment=True):
    """Validate and bind the frozen reference before any candidate comparison."""
    return _reference_precheck(
        reference_predictions=reference_predictions,
        reference_report=reference_report,
        reference_report_sha256=reference_report_sha256,
        assignment_payload=assignment_payload, assignment_sha256=assignment_sha256,
        panel_payload=panel_payload, panel_manifest_payload=panel_manifest_payload,
        panel_manifest_sha256=panel_manifest_sha256,
        panel_preflight_payload=panel_preflight_payload,
        inventory_manifest_payload=inventory_manifest_payload,
        arm_preflight_payload=arm_preflight_payload,
        reference_provenance_payload=reference_provenance_payload,
        recipe_payload=recipe_payload,
        preparation_receipt_payload=preparation_receipt_payload,
        preparation_receipt_sha256=preparation_receipt_sha256,
        _require_frozen_panel=_require_frozen_panel,
        _require_frozen_assignment=_require_frozen_assignment)[0]


def validate_launch_chain(*, context, slot, role, prerequisites, run_receipt_sha256,
                          launch_intent_payload, launch_intent_sha256,
                          launch_completion_payload, launch_completion_sha256):
    """Validate one role-agnostic launch chain against frozen preparation authority."""
    assignment = context["assignment"]
    provenance = context["reference_provenance"]
    if (role not in {"control", "treatment"}
            or assignment.get(f"{role}_slot") != slot
            or not isinstance(prerequisites, dict) or not prerequisites
            or any(not comparator._valid_sha(value) for value in prerequisites.values())):
        raise ValueError(f"{slot} launch role/prerequisites differ from frozen chronology")
    train_sha256 = assignment[f"{role}_train_sha256"]
    plan = launch_command_plan(
        slot=slot, role=role, assignment_sha=context["assignment_sha256"],
        assignment=assignment, recipe=context["recipe"],
        train_sha=train_sha256,
        inventory_sha=assignment["model_inventory_sha256"],
        parent={"parent_adapter_sha256": assignment["parent_adapter_sha256"],
                "parent_adapter_tree_sha256": provenance["adapter_tree_sha256"]},
        source_sha=provenance["source_checkpoint_provenance_sha256"],
        prerequisites=prerequisites)
    plan_sha = _sha(evaluator._canonical_json(plan).encode())
    expected_intent = {
        "schema_version": INTENT_SCHEMA, "status": "invoking_trainer",
        "slot": slot, "role": role, "command_plan_sha256": plan_sha,
        "assignment_sha256": context["assignment_sha256"],
        "recipe_sha256": assignment["recipe_sha256"],
        "train_sha256": train_sha256,
        "model_inventory_sha256": assignment["model_inventory_sha256"],
        "parent_adapter_sha256": assignment["parent_adapter_sha256"],
        "parent_adapter_tree_sha256": provenance["adapter_tree_sha256"],
        "source_checkpoint_provenance_sha256": provenance[
            "source_checkpoint_provenance_sha256"],
        "preparation_receipt_sha256": context["preparation_receipt_sha256"],
        "launcher_sha256": context["launcher_sha256"],
        "prerequisite_receipt_sha256": prerequisites,
    }
    expected_completion = {
        "schema_version": COMPLETION_SCHEMA, "status": "trainer_completed",
        "slot": slot, "role": role, "command_plan_sha256": plan_sha,
        "launch_intent_sha256": launch_intent_sha256,
        "run_receipt_sha256": run_receipt_sha256,
        "prerequisite_receipt_sha256": prerequisites,
    }
    for label, payload, digest, expected in (
            ("intent", launch_intent_payload, launch_intent_sha256,
             expected_intent),
            ("completion", launch_completion_payload,
             launch_completion_sha256, expected_completion)):
        if not comparator._valid_sha(digest) or _sha(payload) != digest:
            raise ValueError(f"{slot} launch {label} hash mismatch")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {slot} launch {label} JSON") from error
        if not isinstance(value, dict) or value != expected:
            raise ValueError(f"{slot} launch {label} differs from frozen launch chain")
    return {"launch_intent_sha256": launch_intent_sha256,
            "launch_completion_sha256": launch_completion_sha256}


def precheck_control(*, reference_receipt_payload, reference_receipt_sha256,
                     slot_b_predictions, slot_b_report, slot_b_report_sha256,
                     slot_b_run_receipt, slot_b_run_receipt_sha256,
                     slot_b_launch_intent_payload, slot_b_launch_intent_sha256,
                     slot_b_launch_completion_payload, slot_b_launch_completion_sha256,
                     **reference_inputs):
    """Revalidate the reference, then reject an uninformative slot-B control ceiling."""
    reference, context, reference_records, reference_report = _reference_precheck(
        **reference_inputs)
    expected_reference_payload = _receipt_payload(reference)
    if (not comparator._valid_sha(reference_receipt_sha256)
            or _sha(reference_receipt_payload) != reference_receipt_sha256
            or reference_receipt_payload != expected_reference_payload):
        raise ValueError("reference receipt is not the exact revalidated immutable receipt")

    assignment = context["assignment"]
    receipt_expectation = {
        "train_sha256": assignment["control_train_sha256"],
        "recipe_sha256": assignment["recipe_sha256"],
        "parent_adapter_sha256": assignment["parent_adapter_sha256"],
        "model_revision": assignment["model_revision"],
        "model_inventory_sha256": assignment["model_inventory_sha256"],
    }
    run_receipt = comparator._load_run_receipt(
        slot_b_run_receipt, slot_b_run_receipt_sha256, "slot_b", receipt_expectation)
    validate_launch_chain(
        context=context, slot="slot_b", role="control",
        prerequisites={"reference_receipt_sha256": reference_receipt_sha256},
        run_receipt_sha256=slot_b_run_receipt_sha256,
        launch_intent_payload=slot_b_launch_intent_payload,
        launch_intent_sha256=slot_b_launch_intent_sha256,
        launch_completion_payload=slot_b_launch_completion_payload,
        launch_completion_sha256=slot_b_launch_completion_sha256)
    records = comparator._read_payload(slot_b_predictions)
    comparator._bind_predictions_to_panel(records, context["panel_rows"])
    comparator.validate_panel_coverage(records)
    comparator._assert_matching_structure(records, reference_records)
    report_expectation = {
        **context["common"], "provenance_role": "candidate",
        **receipt_expectation,
        "adapter_sha256": run_receipt["adapter_sha256"],
        "run_receipt_sha256": slot_b_run_receipt_sha256,
        "checkpoint_payload_sha256": run_receipt["checkpoint_payload_sha256"],
        "adapter_tree_sha256": run_receipt["checkpoint_payload_sha256"],
    }
    report = comparator._load_evaluator_report(
        slot_b_report, slot_b_report_sha256, "slot_b", slot_b_predictions,
        assignment["panel_sha256"], context["panel_manifest_sha256"],
        report_expectation)
    if ((report["model_tree_sha256"], report["model_tree_files"])
            != (reference_report["model_tree_sha256"],
                reference_report["model_tree_files"])):
        raise ValueError("slot_b and reference evaluator reports used different model trees")

    metrics = {f"K{arity}": comparator.causal_metrics(records, arity)
               for arity in (2, 3)}
    ceilings = {name: value["clauses"]["macro_balanced_accuracy"]
                for name, value in metrics.items()}
    exceeded = [name for name, value in ceilings.items() if value > CONTROL_CEILING]
    if exceeded:
        details = ", ".join(f"{name}={ceilings[name]:.6f}" for name in exceeded)
        raise ValueError(
            f"uninformative control ceiling above {CONTROL_CEILING:.2f}: {details}")
    return _with_binding({
        "schema_version": CONTROL_SCHEMA,
        "mode": "control",
        "status": "pass",
        "assignment": reference["assignment"],
        "artifacts": {
            "assignment_sha256": context["assignment_sha256"],
            "reference_receipt_sha256": reference_receipt_sha256,
            "slot_b_run_receipt_sha256": slot_b_run_receipt_sha256,
            "slot_b_launch_intent_sha256": slot_b_launch_intent_sha256,
            "slot_b_launch_completion_sha256": slot_b_launch_completion_sha256,
            "slot_b_report_sha256": slot_b_report_sha256,
            "slot_b_predictions_sha256": _sha(slot_b_predictions),
            "slot_b_adapter_sha256": run_receipt["adapter_sha256"],
            "slot_b_checkpoint_payload_sha256": run_receipt[
                "checkpoint_payload_sha256"],
            "control_train_sha256": assignment["control_train_sha256"],
            "model_revision": assignment["model_revision"],
            "model_inventory_sha256": assignment["model_inventory_sha256"],
            "model_tree_sha256": report["model_tree_sha256"],
            "panel_sha256": assignment["panel_sha256"],
            "panel_manifest_sha256": assignment["panel_manifest_sha256"],
            "panel_preflight_sha256": assignment["panel_preflight_sha256"],
            "inventory_manifest_sha256": assignment["inventory_manifest_sha256"],
            "arm_preflight_sha256": assignment["arm_preflight_sha256"],
            "recipe_sha256": assignment["recipe_sha256"],
            "preparation_receipt_sha256": context["preparation_receipt_sha256"],
            "launcher_sha256": context["launcher_sha256"],
            "scalar_scorer_sha256": context["scalar_scorer_sha256"],
            "atomic_publisher_sha256": context["atomic_publisher_sha256"],
            **context["preparation_lineage"],
            "evaluator_sha256": context["evaluator_sha256"],
            "comparator_sha256": context["comparator_sha256"],
            "precheck_sha256": context["precheck_sha256"],
        },
        "metrics": metrics,
        "control_ceiling": {
            "threshold": CONTROL_CEILING,
            "operator": "<=",
            "clause_macro_balanced_accuracy": ceilings,
        },
        "informative": True,
    })


def _write_new(path, receipt):
    """Atomically publish one immutable sibling-temp receipt without overwrite."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite immutable receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_receipt_payload(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"refusing to overwrite immutable receipt: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def write_reference_receipt(path, **inputs):
    return _write_new(path, precheck_reference(**inputs))


def write_control_receipt(path, **inputs):
    return _write_new(path, precheck_control(**inputs))


def _add_reference_inputs(parser):
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--reference-report-sha256", required=True)
    parser.add_argument("--reference-provenance", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--preparation-receipt", required=True)
    parser.add_argument("--preparation-receipt-sha256", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--panel-manifest-sha256", required=True)
    parser.add_argument("--panel-preflight", required=True)
    parser.add_argument("--inventory-manifest", required=True)
    parser.add_argument("--arm-preflight", required=True)


def _reference_cli_inputs(args):
    return {
        "assignment_payload": Path(args.assignment).read_bytes(),
        "assignment_sha256": args.assignment_sha256,
        "reference_predictions": Path(args.reference_predictions).read_bytes(),
        "reference_report": Path(args.reference_report).read_bytes(),
        "reference_report_sha256": args.reference_report_sha256,
        "reference_provenance_payload": Path(args.reference_provenance).read_bytes(),
        "recipe_payload": Path(args.recipe).read_bytes(),
        "preparation_receipt_payload": Path(args.preparation_receipt).read_bytes(),
        "preparation_receipt_sha256": args.preparation_receipt_sha256,
        "panel_payload": Path(args.panel).read_bytes(),
        "panel_manifest_payload": Path(args.panel_manifest).read_bytes(),
        "panel_manifest_sha256": args.panel_manifest_sha256,
        "panel_preflight_payload": Path(args.panel_preflight).read_bytes(),
        "inventory_manifest_payload": Path(args.inventory_manifest).read_bytes(),
        "arm_preflight_payload": Path(args.arm_preflight).read_bytes(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    reference = modes.add_parser("reference")
    _add_reference_inputs(reference)
    reference.add_argument("--out", required=True)
    control = modes.add_parser("control")
    _add_reference_inputs(control)
    control.add_argument("--reference-receipt", required=True)
    control.add_argument("--reference-receipt-sha256", required=True)
    control.add_argument("--slot-b-predictions", required=True)
    control.add_argument("--slot-b-report", required=True)
    control.add_argument("--slot-b-report-sha256", required=True)
    control.add_argument("--slot-b-run-receipt", required=True)
    control.add_argument("--slot-b-run-receipt-sha256", required=True)
    control.add_argument("--slot-b-launch-intent", required=True)
    control.add_argument("--slot-b-launch-intent-sha256", required=True)
    control.add_argument("--slot-b-launch-completion", required=True)
    control.add_argument("--slot-b-launch-completion-sha256", required=True)
    control.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    inputs = _reference_cli_inputs(args)
    if args.mode == "reference":
        receipt = write_reference_receipt(args.out, **inputs)
    else:
        receipt = write_control_receipt(
            args.out, **inputs,
            reference_receipt_payload=Path(args.reference_receipt).read_bytes(),
            reference_receipt_sha256=args.reference_receipt_sha256,
            slot_b_predictions=Path(args.slot_b_predictions).read_bytes(),
            slot_b_report=Path(args.slot_b_report).read_bytes(),
            slot_b_report_sha256=args.slot_b_report_sha256,
            slot_b_run_receipt=Path(args.slot_b_run_receipt).read_bytes(),
            slot_b_run_receipt_sha256=args.slot_b_run_receipt_sha256,
            slot_b_launch_intent_payload=Path(args.slot_b_launch_intent).read_bytes(),
            slot_b_launch_intent_sha256=args.slot_b_launch_intent_sha256,
            slot_b_launch_completion_payload=Path(args.slot_b_launch_completion).read_bytes(),
            slot_b_launch_completion_sha256=args.slot_b_launch_completion_sha256)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
