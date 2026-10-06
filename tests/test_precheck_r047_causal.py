import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_compare_r047_causal as fixtures  # noqa: E402

from scripts import compare_r047_causal as comparator  # noqa: E402
from scripts import precheck_r047_causal as precheck  # noqa: E402
from scripts.eval_r047_executor import _canonical_json  # noqa: E402
from scripts.precheck_r047_causal import (  # noqa: E402
    precheck_control,
    precheck_reference,
    write_control_receipt,
    write_reference_receipt,
)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _preparation_payload(artifacts, recipe_payload=fixtures.RECIPE_PAYLOAD):
    assignment = json.loads(artifacts["assignment"])
    provenance = json.loads(artifacts["reference_provenance"])
    recipe = json.loads(recipe_payload)
    value = {
        "schema_version": "r047-m1-preparation-receipt-v1",
        "status": "complete",
        "recipe_sha256": assignment["recipe_sha256"],
        "reference_provenance_sha256": assignment["reference_provenance_sha256"],
        "assignment_sha256": _sha(artifacts["assignment"]),
        "model_revision": assignment["model_revision"],
        "model_inventory_sha256": assignment["model_inventory_sha256"],
        "parent_adapter_sha256": assignment["parent_adapter_sha256"],
        "control_train_sha256": assignment["control_train_sha256"],
        "treatment_train_sha256": assignment["treatment_train_sha256"],
        "source_checkpoint_provenance_sha256": provenance[
            "source_checkpoint_provenance_sha256"],
        "panel_sha256": assignment["panel_sha256"],
        "arm_preflight_sha256": assignment["arm_preflight_sha256"],
        "trainer_sha256": recipe["trainer_sha256"],
        "evaluator_sha256": assignment["evaluator_sha256"],
        "comparator_sha256": assignment["comparator_sha256"],
        "precheck_sha256": assignment["precheck_sha256"],
        "preparer_sha256": _sha(
            Path(precheck.__file__).with_name("prepare_r047_m1.py").read_bytes()),
        "finalizer_sha256": _sha(
            Path(precheck.__file__).with_name("finalize_r047_m1.py").read_bytes()),
    }
    return _payload(value)


def _reference_args(artifacts, recipe_payload=fixtures.RECIPE_PAYLOAD):
    preparation = _preparation_payload(artifacts, recipe_payload)
    return {
        "assignment_payload": artifacts["assignment"],
        "assignment_sha256": _sha(artifacts["assignment"]),
        "reference_predictions": artifacts["predictions"]["reference"],
        "reference_report": artifacts["reports"]["reference"],
        "reference_report_sha256": _sha(artifacts["reports"]["reference"]),
        "reference_provenance_payload": artifacts["reference_provenance"],
        "recipe_payload": recipe_payload,
        "preparation_receipt_payload": preparation,
        "preparation_receipt_sha256": _sha(preparation),
        "panel_payload": artifacts["panel"],
        "panel_manifest_payload": artifacts["manifest"],
        "panel_manifest_sha256": _sha(artifacts["manifest"]),
        "panel_preflight_payload": artifacts["preflight"],
        "inventory_manifest_payload": artifacts["inventory"],
        "arm_preflight_payload": artifacts["arm_preflight"],
        "_require_frozen_panel": False,
        "_require_frozen_assignment": False,
    }


def _control_args(artifacts, reference_receipt):
    return {
        **_reference_args(artifacts),
        "reference_receipt_payload": _payload(reference_receipt),
        "reference_receipt_sha256": _sha(_payload(reference_receipt)),
        "slot_b_predictions": artifacts["predictions"]["slot_b"],
        "slot_b_report": artifacts["reports"]["slot_b"],
        "slot_b_report_sha256": _sha(artifacts["reports"]["slot_b"]),
        "slot_b_run_receipt": artifacts["receipts"]["slot_b"],
        "slot_b_run_receipt_sha256": _sha(artifacts["receipts"]["slot_b"]),
    }


def _frozen_recipe(assignment):
    from scripts import finalize_r047_m1 as finalizer
    from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION

    return {
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
            Path(precheck.__file__).with_name(
                "train_gemma_decision_lora.py").read_bytes()),
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
        "trainer_config": dict(finalizer.FROZEN_TRAINING_INVARIANTS),
    }


def _reseal(record):
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = _sha(_canonical_json(unsigned).encode())


def test_reference_then_control_happy_path_binds_metrics_and_provenance():
    artifacts = fixtures._artifacts()
    reference = precheck_reference(**_reference_args(artifacts))
    control = precheck_control(**_control_args(artifacts, reference))

    assert reference["assignment"] == {
        "sha256": _sha(artifacts["assignment"]),
        "control_slot": "slot_b",
        "treatment_slot": "slot_a",
    }
    assert reference["metrics"]["K2"]["arity"] == 2
    assert reference["metrics"]["K3"]["arity"] == 3
    assert reference["artifacts"]["reference_predictions_sha256"] == _sha(
        artifacts["predictions"]["reference"])
    assert control["informative"] is True
    assert control["metrics"]["K2"]["clauses"]["macro_balanced_accuracy"] == .5
    assert control["metrics"]["K3"]["clauses"]["macro_balanced_accuracy"] == .5
    assert control["artifacts"]["reference_receipt_sha256"] == _sha(_payload(reference))


def test_reference_rejects_tampering_and_fully_rehashed_role_train_swap():
    artifacts = fixtures._artifacts()
    with pytest.raises(ValueError, match="panel hash"):
        precheck_reference(**{
            **_reference_args(artifacts),
            "panel_payload": artifacts["panel"] + b"\n",
        })

    with pytest.raises(ValueError, match="preparation receipt"):
        precheck_reference(**{
            **_reference_args(artifacts),
            "preparation_receipt_payload": b"{}\n",
            "preparation_receipt_sha256": _sha(b"{}\n"),
        })

    assignment = json.loads(artifacts["assignment"])
    assignment["control_slot"], assignment["treatment_slot"] = (
        assignment["treatment_slot"], assignment["control_slot"])
    assignment["control_train_sha256"], assignment["treatment_train_sha256"] = (
        assignment["treatment_train_sha256"], assignment["control_train_sha256"])
    swapped = (json.dumps(assignment, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="control/treatment slot assignment"):
        precheck_reference(**{
            **_reference_args(artifacts),
            "assignment_payload": swapped,
            "assignment_sha256": _sha(swapped),
        })


def test_production_assignment_pins_frozen_parent_and_source(monkeypatch):
    from scripts import build_r047_model_inventory as inventory
    from scripts import finalize_r047_m1 as finalizer
    from scripts import prepare_r047_m1 as prepare

    artifacts = fixtures._artifacts()
    assignment = json.loads(artifacts["assignment"])
    provenance = json.loads(artifacts["reference_provenance"])
    monkeypatch.setattr(finalizer, "KNOWN_ARM_SHA256", {
        "control": assignment["control_train_sha256"],
        "treatment": assignment["treatment_train_sha256"],
    })
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_SHA256",
                        assignment["parent_adapter_sha256"])
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_CONFIG_SHA256",
                        provenance["adapter_config_sha256"])
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_TREE_SHA256",
                        provenance["adapter_tree_sha256"])
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_TREE_FILES",
                        provenance["adapter_tree_files"])
    monkeypatch.setattr(inventory, "FROZEN_BASE_REVISION",
                        assignment["model_revision"])
    monkeypatch.setattr(prepare, "FROZEN_R044_CHECKPOINT_PROVENANCE_SHA256",
                        provenance["source_checkpoint_provenance_sha256"])
    recipe = _frozen_recipe(assignment)
    recipe_payload = _payload(recipe)
    assignment["recipe_sha256"] = _sha(recipe_payload)
    assignment_payload = _payload(assignment)
    production_artifacts = {**artifacts, "assignment": assignment_payload}
    args = {**_reference_args(production_artifacts, recipe_payload),
            "_require_frozen_assignment": True}

    assert precheck_reference(**args)["status"] == "pass"
    forged = copy.deepcopy(assignment)
    forged["parent_adapter_sha256"] = "f" * 64
    forged["reference_adapter_sha256"] = "f" * 64
    forged_payload = _payload(forged)
    with pytest.raises(ValueError, match="frozen reference parent"):
        precheck_reference(**{
            **args,
            "assignment_payload": forged_payload,
            "assignment_sha256": _sha(forged_payload),
        })

    mutated_recipe = {**recipe, "unexpected": True}
    mutated_recipe_payload = _payload(mutated_recipe)
    mutated_assignment = copy.deepcopy(assignment)
    mutated_assignment["recipe_sha256"] = _sha(mutated_recipe_payload)
    mutated_assignment_payload = _payload(mutated_assignment)
    mutated_artifacts = {**artifacts, "assignment": mutated_assignment_payload}
    with pytest.raises(ValueError, match="complete frozen R047-M1 contract"):
        precheck_reference(**{
            **_reference_args(mutated_artifacts, mutated_recipe_payload),
            "_require_frozen_assignment": True,
        })


def test_assignment_precheck_source_hash_is_bound():
    artifacts = fixtures._artifacts()
    assignment = json.loads(artifacts["assignment"])
    assert assignment["precheck_sha256"] == _sha(Path(precheck.__file__).read_bytes())
    receipt = precheck_reference(**_reference_args(artifacts))
    assert receipt["artifacts"]["precheck_sha256"] == assignment["precheck_sha256"]

    assignment["precheck_sha256"] = "f" * 64
    payload = _payload(assignment)
    with pytest.raises(ValueError, match="running precheck"):
        precheck_reference(**{
            **_reference_args(artifacts),
            "assignment_payload": payload,
            "assignment_sha256": _sha(payload),
        })


@pytest.mark.parametrize("kind", ["model", "report", "prediction"])
def test_reference_rejects_wrong_model_report_or_prediction_provenance(kind):
    artifacts = fixtures._artifacts()
    args = _reference_args(artifacts)
    if kind == "model":
        report = json.loads(artifacts["reports"]["reference"])
        report["model_revision"] = "wrong-revision"
        args["reference_report"] = (json.dumps(report, sort_keys=True) + "\n").encode()
        args["reference_report_sha256"] = _sha(args["reference_report"])
    elif kind == "report":
        args["reference_report"] += b"\n"
    else:
        report = json.loads(artifacts["reports"]["reference"])
        report["predictions_sha256"] = "0" * 64
        args["reference_report"] = (json.dumps(report, sort_keys=True) + "\n").encode()
        args["reference_report_sha256"] = _sha(args["reference_report"])
    with pytest.raises(ValueError, match=(
            "provenance/count mismatch" if kind != "report" else "evaluator-report hash")):
        precheck_reference(**args)


def test_control_revalidates_reference_receipt_and_sealed_predictions():
    artifacts = fixtures._artifacts()
    reference = precheck_reference(**_reference_args(artifacts))
    changed = copy.deepcopy(reference)
    changed["assignment"]["control_slot"] = "slot_a"
    with pytest.raises(ValueError, match="reference receipt"):
        precheck_control(**{
            **_control_args(artifacts, reference),
            "reference_receipt_payload": _payload(changed),
            "reference_receipt_sha256": _sha(_payload(changed)),
        })

    records = [json.loads(line) for line in artifacts["predictions"]["slot_b"].splitlines()]
    records[0]["probabilities"] = {"no": .2, "yes": .8}
    _reseal(records[0])
    predictions = b"".join(
        (json.dumps(item, sort_keys=True) + "\n").encode() for item in records)
    with pytest.raises(ValueError, match="provenance/count mismatch"):
        precheck_control(**{
            **_control_args(artifacts, reference),
            "slot_b_predictions": predictions,
        })


def test_control_ceiling_aborts_without_partial_output(tmp_path):
    base = fixtures._artifacts()
    perfect = fixtures._arm(base["rows"], True)
    artifacts = fixtures._artifacts(control=perfect)
    reference = precheck_reference(**_reference_args(artifacts))
    out = tmp_path / "control-receipt.json"

    with pytest.raises(ValueError, match="uninformative control ceiling"):
        write_control_receipt(out, **_control_args(artifacts, reference))
    assert not out.exists()
    assert not list(tmp_path.glob(f".{out.name}.tmp-*"))


def test_atomic_receipt_output_refuses_overwrite(tmp_path):
    artifacts = fixtures._artifacts()
    reference_path = tmp_path / "reference-receipt.json"
    reference = write_reference_receipt(
        reference_path, **_reference_args(artifacts))
    original = reference_path.read_bytes()

    with pytest.raises(ValueError, match="overwrite"):
        write_reference_receipt(reference_path, **_reference_args(artifacts))
    assert reference_path.read_bytes() == original

    control_path = tmp_path / "control-receipt.json"
    control = write_control_receipt(
        control_path, **_control_args(artifacts, reference))
    assert json.loads(control_path.read_text()) == control
    assert not list(tmp_path.glob(".*.tmp-*"))
