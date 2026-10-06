import copy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.compare_r047_causal as comparator_module
import scripts.eval_r047_executor as evaluator_module
from scripts.build_r047_panel import _cluster_rows, _jsonl
from scripts.compare_r047_causal import (
    MIN_BOOTSTRAP_REPLICATES, _gate, causal_metrics, compare_payloads,
    validate_panel_coverage,
)
from scripts.eval_r047_executor import (
    _canonical_json, causal_coverage_report, causal_parity_subset, seal_causal_predictions,
)


SHA = {name: hashlib.sha256(name.encode()).hexdigest() for name in (
    "control-train", "treatment-train", "control-adapter",
    "treatment-adapter", "parent-adapter", "recipe", "base-inventory",
    "config", "metrics", "log", "reference-provenance", "arm-preflight")}
SHA["control-train"] = comparator_module.FROZEN_CONTROL_TRAIN_SHA256
SHA["treatment-train"] = comparator_module.FROZEN_TREATMENT_TRAIN_SHA256
RECIPE_PAYLOAD = (json.dumps({
    "schema_version": "r047-m1-recipe-v1",
    "trainer_sha256": hashlib.sha256(b"trainer").hexdigest(),
}, sort_keys=True) + "\n").encode()
SHA["recipe"] = hashlib.sha256(RECIPE_PAYLOAD).hexdigest()
MODEL_REVISION = "revision-707f0a3"


def _rows():
    return (_cluster_rows("unseen_k2", ("decimal", "ledger"), 0, 0, 4701)
            + _cluster_rows("k3", ("decimal", "ledger", "policy"), 0, 0, 4701))


def _arm(rows, treatment):
    predictions = []
    for row in rows:
        gold = row["targets"]["decision"]["label"]
        role = row["meta"]["prediction_role"]
        yes = ((.9 if gold == "yes" else .1) if treatment
               else (.6 if role == "clause" else .4))
        predictions.append({"id": row["id"],
                            "probabilities": {"no": 1 - yes, "yes": yes}})
    return seal_causal_predictions(rows, predictions)


def _payload(records):
    return "".join(json.dumps(item, sort_keys=True) + "\n" for item in records).encode()


def _manifest(panel, rows):
    value = {
        "experiment": "r047_primary_causal_panel", "version": 1,
        "contract": {"frozen_production": False},
        "counts_per_composition": {"unseen_k2": 1, "k3": 1, "seen_k2": 1},
        "clusters": {"unseen_k2": 1, "k3": 1, "seen_k2_near_transfer": 0,
                     "causal_primary": 2},
        "compositions": {"unseen_k2": [["decimal", "ledger"]],
                         "k3": [["decimal", "ledger", "policy"]],
                         "seen_k2_near_transfer": []},
        "outputs": {"primary": {"file": "primary.jsonl", "rows": len(rows),
                                "sha256": hashlib.sha256(panel).hexdigest()}},
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _assignment(panel, manifest, preflight, inventory, reference_provenance):
    return {
        "schema_version": "r047-causal-slot-assignment-v1",
        "control_slot": "slot_b", "treatment_slot": "slot_a",
        "control_train_sha256": SHA["control-train"],
        "treatment_train_sha256": SHA["treatment-train"],
        "reference_provenance_sha256": hashlib.sha256(reference_provenance).hexdigest(),
        "reference_adapter_sha256": SHA["parent-adapter"],
        "parent_adapter_sha256": SHA["parent-adapter"],
        "model_revision": MODEL_REVISION, "recipe_sha256": SHA["recipe"],
        "model_inventory_sha256": SHA["base-inventory"],
        "panel_sha256": hashlib.sha256(panel).hexdigest(),
        "panel_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "panel_preflight_sha256": hashlib.sha256(preflight).hexdigest(),
        "inventory_manifest_sha256": hashlib.sha256(inventory).hexdigest(),
        "arm_preflight_sha256": SHA["arm-preflight"],
        "evaluator_sha256": hashlib.sha256(
            Path(evaluator_module.__file__).read_bytes()).hexdigest(),
        "comparator_sha256": hashlib.sha256(
            Path(comparator_module.__file__).read_bytes()).hexdigest(),
        "precheck_sha256": hashlib.sha256(
            Path(comparator_module.__file__).with_name(
                "precheck_r047_causal.py").read_bytes()).hexdigest(),
        "bootstrap_seed": 470047, "bootstrap_replicates": 20_000,
    }


def _supporting_provenance(panel, manifest, rows):
    inventory = (json.dumps({
        "version": 1,
        "training": [{"path": "train.jsonl", "sha256": SHA["control-train"]}],
        "benchmarks": [{"path": "benchmark.jsonl", "sha256": SHA["metrics"]}],
    }, sort_keys=True) + "\n").encode()
    preflight = (json.dumps({
        "experiment": "r047_primary_panel_preflight", "status": "pass",
        "panel_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "inventory_manifest_sha256": hashlib.sha256(inventory).hexdigest(),
        "outputs": {"primary": {"sha256": hashlib.sha256(panel).hexdigest(),
                                  "rows": len(rows)}},
        "causal": {"primary": {"rows": len(rows)}},
    }, sort_keys=True) + "\n").encode()
    return preflight, inventory


def _reference_provenance():
    value = {
        "schema_version": "r047-reference-provenance-v1",
        "adapter_weights_sha256": SHA["parent-adapter"],
        "adapter_config_sha256": SHA["config"],
        "adapter_tree_sha256": hashlib.sha256(b"reference-tree").hexdigest(),
        "adapter_tree_files": 3, "model_revision": MODEL_REVISION,
        "model_inventory_sha256": SHA["base-inventory"],
        "source_checkpoint_provenance_sha256": SHA["reference-provenance"],
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _receipt(slot, train, adapter):
    value = {"schema_version": "r047-causal-run-receipt-v1", "slot": slot,
             "train_sha256": train, "recipe_sha256": SHA["recipe"],
             "parent_adapter_sha256": SHA["parent-adapter"],
             "model_revision": MODEL_REVISION, "adapter_sha256": adapter,
             "model_inventory_sha256": SHA["base-inventory"],
             "checkpoint_payload_sha256": hashlib.sha256(
                 f"checkpoint-{slot}".encode()).hexdigest(),
             "config_sha256": SHA["config"], "metrics_sha256": SHA["metrics"],
             "log_sha256": SHA["log"], "checkpoint_step": 400, "status": "complete"}
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _report(predictions, panel, manifest, preflight, inventory, rows,
            reference_provenance, receipt=None):
    run = json.loads(receipt) if receipt is not None else None
    records = [json.loads(line) for line in predictions.decode().splitlines()]
    report = causal_coverage_report(records)
    parity = causal_parity_subset(rows)
    report.update({
        "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
        "input_sha256": hashlib.sha256(panel).hexdigest(),
        "panel_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "panel_preflight_sha256": hashlib.sha256(preflight).hexdigest(),
        "inventory_manifest_sha256": hashlib.sha256(inventory).hexdigest(),
        "evaluator_sha256": hashlib.sha256(
            Path(evaluator_module.__file__).read_bytes()).hexdigest(),
        "adapter_sha256": (run["adapter_sha256"] if run else SHA["parent-adapter"]),
        "model_revision": MODEL_REVISION, "panel_rows": len(rows),
        "model_inventory_sha256": SHA["base-inventory"],
        "model_tree_sha256": hashlib.sha256(b"model-tree").hexdigest(),
        "model_tree_files": 17,
        "adapter_tree_sha256": (run["checkpoint_payload_sha256"] if run else
                                json.loads(reference_provenance)["adapter_tree_sha256"]),
        "adapter_tree_files": (4 if run else
                               json.loads(reference_provenance)["adapter_tree_files"]),
        "batch_scalar_parity": {
            "status": "pass", "rows": len(parity),
            "subset_ids_sha256": hashlib.sha256(
                ("\n".join(row["id"] for row in parity) + "\n").encode()).hexdigest(),
            "exact_discrete_outputs": True, "max_abs_probability_delta": 0.0,
            "tolerance": 1e-5,
            "selection": "first_frozen_row_per_arity_language_order_role_family_primitive"},
    })
    if run:
        report.update({
            "provenance_role": "candidate", "train_sha256": run["train_sha256"],
            "recipe_sha256": SHA["recipe"],
            "parent_adapter_sha256": SHA["parent-adapter"],
            "run_receipt_sha256": hashlib.sha256(receipt).hexdigest(),
            "checkpoint_payload_sha256": run["checkpoint_payload_sha256"],
        })
    else:
        report.update({
            "provenance_role": "reference",
            "reference_provenance_sha256": hashlib.sha256(
                reference_provenance).hexdigest(),
            "adapter_config_sha256": SHA["config"],
            "source_checkpoint_provenance_sha256": SHA["reference-provenance"],
        })
    return (json.dumps(report, sort_keys=True) + "\n").encode()


def _artifacts(control=None, treatment=None):
    rows = _rows()
    control = _arm(rows, False) if control is None else control
    treatment = _arm(rows, True) if treatment is None else treatment
    panel, predictions = _jsonl(rows), {
        "slot_a": _payload(treatment), "slot_b": _payload(control),
        "reference": _payload(control)}
    manifest = _manifest(panel, rows)
    preflight, inventory = _supporting_provenance(panel, manifest, rows)
    reference_provenance = _reference_provenance()
    receipts = {
        "slot_a": _receipt("slot_a", SHA["treatment-train"], SHA["treatment-adapter"]),
        "slot_b": _receipt("slot_b", SHA["control-train"], SHA["control-adapter"])}
    reports = {
        "slot_a": _report(predictions["slot_a"], panel, manifest, preflight,
                          inventory, rows, reference_provenance, receipts["slot_a"]),
        "slot_b": _report(predictions["slot_b"], panel, manifest, preflight,
                          inventory, rows, reference_provenance, receipts["slot_b"]),
        "reference": _report(predictions["reference"], panel, manifest, preflight,
                             inventory, rows, reference_provenance),
    }
    assignment = (json.dumps(_assignment(
        panel, manifest, preflight, inventory, reference_provenance),
        sort_keys=True) + "\n").encode()
    return {"rows": rows, "panel": panel, "manifest": manifest,
            "preflight": preflight, "inventory": inventory,
            "arm_preflight": b"arm-preflight",
            "reference_provenance": reference_provenance,
            "predictions": predictions, "reports": reports, "receipts": receipts,
            "assignment": assignment}


def _compare(artifacts=None, replicates=MIN_BOOTSTRAP_REPLICATES,
             require_frozen=False, **overrides):
    value = _artifacts() if artifacts is None else artifacts
    args = {
        "slot_a_predictions": value["predictions"]["slot_a"],
        "slot_b_predictions": value["predictions"]["slot_b"],
        "reference_predictions": value["predictions"]["reference"],
        "slot_a_report": value["reports"]["slot_a"],
        "slot_b_report": value["reports"]["slot_b"],
        "reference_report": value["reports"]["reference"],
        "panel_payload": value["panel"], "panel_manifest_payload": value["manifest"],
        "panel_preflight_payload": value["preflight"],
        "inventory_manifest_payload": value["inventory"],
        "arm_preflight_payload": value["arm_preflight"],
        "reference_provenance_payload": value["reference_provenance"],
        "assignment_payload": value["assignment"],
        "slot_a_run_receipt": value["receipts"]["slot_a"],
        "slot_b_run_receipt": value["receipts"]["slot_b"],
        "slot_a_report_sha256": hashlib.sha256(value["reports"]["slot_a"]).hexdigest(),
        "slot_b_report_sha256": hashlib.sha256(value["reports"]["slot_b"]).hexdigest(),
        "reference_report_sha256": hashlib.sha256(
            value["reports"]["reference"]).hexdigest(),
        "slot_a_run_receipt_sha256": hashlib.sha256(
            value["receipts"]["slot_a"]).hexdigest(),
        "slot_b_run_receipt_sha256": hashlib.sha256(
            value["receipts"]["slot_b"]).hexdigest(),
        "panel_manifest_sha256": hashlib.sha256(value["manifest"]).hexdigest(),
        "assignment_sha256": hashlib.sha256(value["assignment"]).hexdigest(),
        "replicates": replicates, "seed": 470047,
        "_require_frozen_panel": require_frozen}
    args.update(overrides)
    return compare_payloads(**args)


def _reseal(record):
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()


def test_comparator_recomputes_separate_k2_k3_gates_and_slices():
    result = _compare()
    assert result["co_primary_gate"]["passed"]
    assert result["preassignment"]["pre_frozen_slot_assignment"] is True
    for name in ("K2", "K3"):
        stratum = result["strata"][name]
        assert stratum["gate"]["deltas"]["clause_macro_ba"] == .5
        assert stratum["gate"]["deltas"]["final_accuracy"] >= .05
        assert stratum["paired_bootstrap"]["clause_macro_ba"]["replicates"] == 20_000
        assert stratum["treatment"]["direction"]["primitive"]["decimal"]["rate"] == 1
        assert stratum["treatment"]["direction"]["primitive"]["ledger"]["rate"] == 1
        assert stratum["treatment"]["direction"]["hamming1_pairs"]["pair_both_correct"] == 1


def test_comparator_never_pools_k2_and_k3():
    base = _artifacts()
    control = [json.loads(line) for line in base["predictions"]["slot_b"].decode().splitlines()]
    treatment = [json.loads(line) for line in base["predictions"]["slot_a"].decode().splitlines()]
    by_id = {item["id"]: item for item in control}
    for item in treatment:
        if item["arity"] == 2:
            item["probabilities"] = copy.deepcopy(by_id[item["id"]]["probabilities"])
            _reseal(item)
    result = _compare(_artifacts(control, treatment))
    assert result["co_primary_gate"]["strata"] == {"K2": False, "K3": True}


def test_comparator_rejects_one_order_only_gain():
    base = _artifacts()
    control = [json.loads(line) for line in base["predictions"]["slot_b"].decode().splitlines()]
    treatment = [json.loads(line) for line in base["predictions"]["slot_a"].decode().splitlines()]
    by_id = {item["id"]: item for item in control}
    for item in treatment:
        if item["option_order"] == "no_yes":
            item["probabilities"] = copy.deepcopy(by_id[item["id"]]["probabilities"])
            _reseal(item)
    result = _compare(_artifacts(control, treatment))
    assert all(any("mechanism gain is not positive for order no_yes" in failure
                   for failure in result["strata"][name]["gate"]["failures"])
               for name in ("K2", "K3"))


def test_gate_rejects_cluster_hamming1_and_direction_cell_regression():
    records = _artifacts()["predictions"]
    control = causal_metrics([json.loads(line) for line in records["slot_b"].decode().splitlines()], 2)
    treatment = causal_metrics([json.loads(line) for line in records["slot_a"].decode().splitlines()], 2)
    treatment["complete_semantic_clusters"].update(correct=-1, accuracy=-1)
    treatment["direction"]["hamming1_pairs"].update(both_correct=-1, pair_both_correct=-1)
    treatment["direction"]["primitive_language_order"][
        "decimal|en|no_yes"]["rate"] = 0
    control["direction"]["primitive_language_order"][
        "decimal|en|no_yes"]["rate"] = 1
    gate = _gate(control, treatment, control,
                 {"clause_macro_ba": {"delta_lcb_95": .1}})
    assert "complete_semantic_clusters correctness regressed" in gate["failures"]
    assert "Hamming-1 pair-both correctness regressed" in gate["failures"]
    assert any("direction primitive_language_order cell" in item for item in gate["failures"])


def test_gate_rejects_either_clause_label_recall_regression():
    records = _artifacts()["predictions"]
    control = causal_metrics(
        [json.loads(line) for line in records["slot_b"].decode().splitlines()], 2)
    treatment = causal_metrics(
        [json.loads(line) for line in records["slot_a"].decode().splitlines()], 2)
    treatment["clauses"]["tpr"] = control["clauses"]["tpr"] - .01
    treatment["clauses"]["tnr"] = control["clauses"]["tnr"] - .01
    gate = _gate(control, treatment, control,
                 {name: {"delta_lcb_95": .1} for name in (
                     "clause_macro_ba", "vector_accuracy", "coherence",
                     "final_accuracy")})
    assert "clause yes_tpr recall regressed" in gate["failures"]
    assert "clause no_tnr recall regressed" in gate["failures"]
    assert "equal prior weight" in gate["label_prior_contract"]["primary"]


def test_secondary_ceiling_rule_is_fixed_noninferiority_and_primary_ceiling_fails():
    records = _artifacts()["predictions"]
    perfect = causal_metrics(
        [json.loads(line) for line in records["slot_a"].decode().splitlines()], 2)
    bootstrap = {name: {"delta_lcb_95": 0.0} for name in (
        "clause_macro_ba", "vector_accuracy", "coherence", "final_accuracy")}
    gate = _gate(perfect, perfect, perfect, bootstrap)
    assert all(item["branch"] == "ceiling_noninferiority" and item["passed"]
               for item in gate["secondary_gate_contract"].values())
    assert "uninformative_primary_ceiling: harder panel required" in gate["failures"]


def test_metrics_report_confusions_order_stability_and_direction_cells():
    metrics = causal_metrics(_arm(_rows(), True), 2)
    assert metrics["clauses"]["tpr"] == metrics["clauses"]["tnr"] == 1
    assert metrics["option_invariance"]["yes_probability_abs_delta_max"] == 0
    assert metrics["direction"]["primitive_language_order"]


def test_partial_or_substitute_panel_fails():
    with pytest.raises(ValueError, match="panel provenance differs from frozen preassignment"):
        _compare(require_frozen=True)
    value = _artifacts()
    changed = copy.deepcopy(value["rows"])
    changed[0]["state"] += " tampered"
    panel, manifest = _jsonl(changed), _manifest(_jsonl(changed), changed)
    preflight, inventory = _supporting_provenance(panel, manifest, changed)
    reference_provenance = value["reference_provenance"]
    assignment = (json.dumps(_assignment(
        panel, manifest, preflight, inventory, reference_provenance),
        sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="sealed row hash differs"):
        _compare(value, panel_payload=panel, panel_manifest_payload=manifest,
                 panel_preflight_payload=preflight,
                 inventory_manifest_payload=inventory,
                 panel_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                 assignment_payload=assignment,
                 assignment_sha256=hashlib.sha256(assignment).hexdigest())


@pytest.mark.parametrize("target,match", [
    ("assignment_sha256", "slot-assignment hash mismatch"),
    ("slot_a_report_sha256", "slot_a evaluator-report hash mismatch"),
    ("panel_manifest_sha256", "panel-manifest hash mismatch"),
    ("slot_a_run_receipt_sha256", "slot_a run-receipt hash mismatch")])
def test_external_hash_tampering_fails(target, match):
    with pytest.raises(ValueError, match=match):
        _compare(**{target: "0" * 64})


@pytest.mark.parametrize("payload,match", [
    ("panel_preflight_payload", "preflight/inventory hash"),
    ("inventory_manifest_payload", "preflight/inventory hash"),
    ("arm_preflight_payload", "preflight/inventory hash"),
    ("reference_provenance_payload", "reference provenance hash"),
])
def test_supporting_provenance_payload_tampering_fails(payload, match):
    with pytest.raises(ValueError, match=match):
        _compare(**{payload: b"{}\n"})


@pytest.mark.parametrize("field,value,match", [
    ("comparator_sha256", "0" * 64, "running comparator differs"),
    ("evaluator_sha256", "0" * 64, "running evaluator differs"),
    ("precheck_sha256", "0" * 64, "running precheck differs"),
    ("bootstrap_seed", 17, "bootstrap differs"),
    ("bootstrap_replicates", 20001, "bootstrap differs"),
])
def test_frozen_code_and_bootstrap_assignment_tampering_fails(field, value, match):
    artifacts = _artifacts()
    assignment = json.loads(artifacts["assignment"])
    assignment[field] = value
    payload = (json.dumps(assignment, sort_keys=True) + "\n").encode()
    kwargs = {"assignment_payload": payload,
              "assignment_sha256": hashlib.sha256(payload).hexdigest()}
    if field == "bootstrap_seed":
        kwargs["seed"] = value
    if field == "bootstrap_replicates":
        kwargs["replicates"] = value
    with pytest.raises(ValueError, match=match):
        _compare(artifacts, **kwargs)


def test_reference_provenance_schema_tampering_fails_even_when_rehashed():
    artifacts = _artifacts()
    provenance = json.loads(artifacts["reference_provenance"])
    provenance["unexpected"] = True
    payload = (json.dumps(provenance, sort_keys=True) + "\n").encode()
    assignment = json.loads(artifacts["assignment"])
    assignment["reference_provenance_sha256"] = hashlib.sha256(payload).hexdigest()
    assignment_payload = (json.dumps(assignment, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="reference provenance schema"):
        _compare(artifacts, reference_provenance_payload=payload,
                 assignment_payload=assignment_payload,
                 assignment_sha256=hashlib.sha256(assignment_payload).hexdigest())


@pytest.mark.parametrize("field", [
    "predictions_sha256", "input_sha256", "panel_manifest_sha256", "adapter_sha256",
    "panel_preflight_sha256", "inventory_manifest_sha256", "evaluator_sha256",
    "adapter_tree_sha256", "checkpoint_payload_sha256", "model_revision",
    "train_sha256", "recipe_sha256", "parent_adapter_sha256", "records"])
def test_evaluator_report_provenance_tampering_fails(field):
    value = _artifacts()
    report = json.loads(value["reports"]["slot_a"])
    report[field] = 0 if field == "records" else "tampered"
    payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="provenance/count mismatch"):
        _compare(value, slot_a_report=payload,
                 slot_a_report_sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("field", [
    "adapter_sha256", "model_revision", "model_inventory_sha256",
    "panel_preflight_sha256", "inventory_manifest_sha256", "evaluator_sha256",
    "reference_provenance_sha256", "adapter_config_sha256", "adapter_tree_sha256",
    "source_checkpoint_provenance_sha256", "provenance_role"])
def test_reference_report_provenance_tampering_fails(field):
    value = _artifacts()
    report = json.loads(value["reports"]["reference"])
    report[field] = "tampered"
    payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="provenance/count mismatch"):
        _compare(value, reference_report=payload,
                 reference_report_sha256=hashlib.sha256(payload).hexdigest())


def test_evaluator_adapter_tree_file_count_tampering_fails():
    value = _artifacts()
    report = json.loads(value["reports"]["slot_a"])
    report["adapter_tree_files"] = 0
    payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="adapter-tree receipt is invalid"):
        _compare(value, slot_a_report=payload,
                 slot_a_report_sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("field,value", [
    ("status", "fail"), ("exact_discrete_outputs", False), ("tolerance", 1e-4),
    ("max_abs_probability_delta", 1.1e-5), ("rows", 0),
    ("subset_ids_sha256", "0" * 64)])
def test_evaluator_parity_contract_tampering_fails(field, value):
    artifacts = _artifacts()
    report = json.loads(artifacts["reports"]["slot_a"])
    report["batch_scalar_parity"][field] = value
    payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="parity contract failed|parity subset differs"):
        _compare(artifacts, slot_a_report=payload,
                 slot_a_report_sha256=hashlib.sha256(payload).hexdigest())


def test_evaluator_parity_subset_must_match_across_all_models():
    artifacts = _artifacts()
    report = json.loads(artifacts["reports"]["slot_a"])
    report["batch_scalar_parity"]["subset_ids_sha256"] = hashlib.sha256(b"other").hexdigest()
    payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="parity subset differs from frozen panel"):
        _compare(artifacts, slot_a_report=payload,
                 slot_a_report_sha256=hashlib.sha256(payload).hexdigest())


def test_assignment_slot_and_training_provenance_tampering_fails():
    value = _artifacts()
    assignment = json.loads(value["assignment"])
    assignment["control_slot"] = "slot_a"
    payload = (json.dumps(assignment, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="control/treatment slot assignment"):
        _compare(value, assignment_payload=payload,
                 assignment_sha256=hashlib.sha256(payload).hexdigest())
    assignment = json.loads(value["assignment"])
    assignment["control_train_sha256"] = "f" * 64
    payload = (json.dumps(assignment, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="control/treatment slot assignment"):
        _compare(value, assignment_payload=payload,
                 assignment_sha256=hashlib.sha256(payload).hexdigest())

    assignment = json.loads(value["assignment"])
    assignment["control_train_sha256"], assignment["treatment_train_sha256"] = (
        assignment["treatment_train_sha256"], assignment["control_train_sha256"])
    payload = (json.dumps(assignment, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="control/treatment slot assignment"):
        _compare(value, assignment_payload=payload,
                 assignment_sha256=hashlib.sha256(payload).hexdigest())


def test_complete_role_and_train_hash_swap_fails_even_when_rehashed():
    value = _artifacts()
    assignment = json.loads(value["assignment"])
    assignment["control_slot"], assignment["treatment_slot"] = (
        assignment["treatment_slot"], assignment["control_slot"])
    assignment["control_train_sha256"], assignment["treatment_train_sha256"] = (
        assignment["treatment_train_sha256"], assignment["control_train_sha256"])
    payload = (json.dumps(assignment, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="control/treatment slot assignment"):
        _compare(value, assignment_payload=payload,
                 assignment_sha256=hashlib.sha256(payload).hexdigest())


def test_prediction_tampering_coverage_and_bootstrap_fail_closed():
    records = _arm(_rows(), False)
    records[0]["language"] = "he"
    with pytest.raises(ValueError, match="seal mismatch"):
        validate_panel_coverage(records)
    records = _arm(_rows(), False)
    records.pop()
    with pytest.raises(ValueError, match="incomplete truth-cube|lacks one final"):
        validate_panel_coverage(records)
    with pytest.raises(ValueError, match="bootstrap differs from frozen preassignment"):
        _compare(replicates=19_999)
