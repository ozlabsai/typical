"""Fail-closed reference-first and control-ceiling prechecks for R047-M1."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from scripts import compare_r047_causal as comparator
from scripts import eval_r047_executor as evaluator


REFERENCE_SCHEMA = "r047-causal-reference-precheck-v1"
CONTROL_SCHEMA = "r047-causal-control-precheck-v1"
CONTROL_CEILING = .95


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _receipt_payload(receipt):
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _with_binding(receipt):
    receipt["binding_sha256"] = _sha(evaluator._canonical_json(receipt).encode())
    return receipt


def _assignment_context(*, assignment_payload, assignment_sha256, panel_payload,
                        panel_manifest_payload, panel_manifest_sha256,
                        panel_preflight_payload, inventory_manifest_payload,
                        arm_preflight_payload, reference_provenance_payload,
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
    if (assignment.get("schema_version") != "r047-causal-slot-assignment-v1"
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
        _require_frozen_panel=_require_frozen_panel,
        _require_frozen_assignment=_require_frozen_assignment)[0]


def precheck_control(*, reference_receipt_payload, reference_receipt_sha256,
                     slot_b_predictions, slot_b_report, slot_b_report_sha256,
                     slot_b_run_receipt, slot_b_run_receipt_sha256, **reference_inputs):
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
            slot_b_run_receipt_sha256=args.slot_b_run_receipt_sha256)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
