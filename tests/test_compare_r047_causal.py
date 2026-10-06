import copy
import hashlib
import json

import pytest

from scripts.compare_r047_causal import (
    MIN_BOOTSTRAP_REPLICATES,
    _gate,
    causal_metrics,
    compare_payloads,
    validate_panel_coverage,
)
from scripts.eval_r047_executor import CAUSAL_SCHEMA_VERSION, _canonical_json


def _sealed(record):
    result = {"schema_version": CAUSAL_SCHEMA_VERSION,
              "row_sha256": hashlib.sha256(record["id"].encode()).hexdigest(), **record}
    result["record_sha256"] = hashlib.sha256(_canonical_json(result).encode()).hexdigest()
    return result


def _panel(treatment):
    result = []
    configurations = (("k2", ["decimal", "ledger"]),
                      ("k3", ["decimal", "ledger", "policy"]))
    for cluster, composition in configurations:
        k = len(composition)
        for mask in range(2 ** k):
            pattern = [bool(mask & (1 << index)) for index in range(k)]
            for language in ("en", "he"):
                for order in ("no_yes", "yes_no"):
                    view = f"{cluster}-{mask}-{language}-{order}"
                    common = {
                        "semantic_cluster_id": cluster, "semantic_view_id": view,
                        "scenario_id": f"scenario-{cluster}", "composition": composition,
                        "arity": k, "language": language, "option_order": order,
                        "truth_hamming_weight": sum(pattern), "truth_pattern": pattern,
                        "seen_edge_count": sum(pattern),
                    }
                    final_gold = "yes" if all(pattern) else "no"
                    if treatment:
                        final_yes = .9 if final_gold == "yes" else .1
                    else:
                        final_yes = .4
                    result.append(_sealed({
                        "id": f"{view}-final", "gold": final_gold,
                        "probabilities": {"no": 1 - final_yes, "yes": final_yes},
                        **common, "prompt_family": "final-v1", "prediction_role": "final",
                        "primitive": None, "decisive_role": "final",
                    }))
                    for index, primitive in enumerate(composition):
                        gold = "yes" if pattern[index] else "no"
                        if treatment:
                            yes = .9 if gold == "yes" else .1
                        else:
                            yes = .6
                        decisive = all(value for position, value in enumerate(pattern)
                                       if position != index)
                        result.append(_sealed({
                            "id": f"{view}-{primitive}", "gold": gold,
                            "probabilities": {"no": 1 - yes, "yes": yes},
                            **common, "prompt_family": "clause-v1", "prediction_role": "clause",
                            "primitive": primitive,
                            "decisive_role": "decisive" if decisive else "non_decisive",
                        }))
    return result


def _payload(records):
    return "".join(json.dumps(item, sort_keys=True) + "\n" for item in records).encode()


def _comparison(control=None, treatment=None, replicates=MIN_BOOTSTRAP_REPLICATES):
    control = _panel(False) if control is None else control
    treatment = _panel(True) if treatment is None else treatment
    first, second = _payload(treatment), _payload(control)
    first_hash, second_hash = hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest()
    assignment = json.dumps({"control_sha256": second_hash,
                             "treatment_sha256": first_hash}, sort_keys=True).encode()
    return compare_payloads(
        first, second, assignment, hashlib.sha256(assignment).hexdigest(),
        first_hash, second_hash, replicates=replicates, seed=17,
        reference_payload=_payload(control),
        reference_sha256=hashlib.sha256(_payload(control)).hexdigest())


def test_causal_comparator_recomputes_separate_k2_k3_gates_and_slices():
    result = _comparison()
    assert result["co_primary_gate"] == {
        "passed": True, "strata": {"K2": True, "K3": True},
        "rule": "intersection-union; K2 and K3 must each pass"}
    for name in ("K2", "K3"):
        stratum = result["strata"][name]
        assert stratum["gate"]["deltas"]["clause_macro_ba"] == .5
        assert stratum["gate"]["deltas"]["final_ba"] == .5
        assert stratum["paired_bootstrap"]["clause_macro_ba"]["replicates"] == 20_000
        assert stratum["treatment"]["direction"]["decimal"]["rate"] == 1
        assert stratum["treatment"]["direction"]["ledger"]["rate"] == 1
        assert stratum["treatment"]["option_invariance"]["semantic_agreement"] == 1
        assert stratum["treatment"]["clauses"]["primitive_language"]
        assert stratum["treatment"]["clauses"]["truth_hamming_weight"]
        assert stratum["treatment"]["clauses"]["seen_edge_count"]
        assert stratum["treatment"]["complete_semantic_views"]["accuracy"] == 1
        assert stratum["treatment"]["complete_semantic_clusters"]["accuracy"] == 1
        assert all(value > 0 for value in
                   stratum["gate"]["dimension_deltas"]["clauses"]["option_order"].values())
        assert set(stratum["gate"]["treatment_x_order_interaction"]) == {"clauses", "final"}
    assert result["blinding"]["revealed_only_after_both_hashes_validated"] is True


def test_causal_comparator_never_lets_k3_gain_rescue_failed_k2():
    control, treatment = _panel(False), _panel(True)
    control_by_id = {item["id"]: item for item in control}
    for item in treatment:
        if item["arity"] == 2:
            item["probabilities"] = copy.deepcopy(control_by_id[item["id"]]["probabilities"])
            unsigned = {key: value for key, value in item.items() if key != "record_sha256"}
            item["record_sha256"] = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()
    result = _comparison(control, treatment)
    assert result["co_primary_gate"]["strata"] == {"K2": False, "K3": True}
    assert result["co_primary_gate"]["passed"] is False


def test_causal_comparator_rejects_one_order_only_gain():
    control, treatment = _panel(False), _panel(True)
    control_by_id = {item["id"]: item for item in control}
    for item in treatment:
        if item["option_order"] == "no_yes":
            item["probabilities"] = copy.deepcopy(control_by_id[item["id"]]["probabilities"])
            unsigned = {key: value for key, value in item.items() if key != "record_sha256"}
            item["record_sha256"] = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()
    result = _comparison(control, treatment)
    for name in ("K2", "K3"):
        failures = result["strata"][name]["gate"]["failures"]
        assert any("mechanism gain is not positive for order no_yes" in item
                   for item in failures)
        assert result["strata"][name]["gate"]["passed"] is False


def test_gate_rejects_complete_cluster_regression():
    control = causal_metrics(_panel(False), 2)
    treatment = causal_metrics(_panel(True), 2)
    treatment["complete_semantic_clusters"]["correct"] = -1
    treatment["complete_semantic_clusters"]["accuracy"] = -1
    gate = _gate(control, treatment, control,
                 {"clause_macro_ba": {"delta_lcb_95": .1}})
    assert not gate["passed"]
    assert "complete_semantic_clusters correctness regressed" in gate["failures"]


def test_causal_metrics_report_confusions_and_order_probability_stability():
    metrics = causal_metrics(_panel(True), 2)
    assert metrics["clauses"]["tp"] > 0 and metrics["clauses"]["tn"] > 0
    assert metrics["clauses"]["tpr"] == metrics["clauses"]["tnr"] == 1
    assert metrics["option_invariance"]["yes_probability_abs_delta_max"] == 0
    assert metrics["option_invariance"]["jsd_max_nats"] == 0


def test_causal_comparator_fails_closed_on_tamper_coverage_hash_and_bootstrap():
    tampered = _panel(False)
    tampered[0]["language"] = "he"
    with pytest.raises(ValueError, match="seal mismatch"):
        validate_panel_coverage(tampered)

    incomplete = _panel(False)
    incomplete.pop()
    with pytest.raises(ValueError, match="incomplete truth-cube|lacks one final"):
        validate_panel_coverage(incomplete)

    with pytest.raises(ValueError, match="at least 20000"):
        _comparison(replicates=19_999)

    control, treatment = _panel(False), _panel(True)
    first, second = _payload(control), _payload(treatment)
    assignment = json.dumps({"control_sha256": hashlib.sha256(first).hexdigest(),
                             "treatment_sha256": hashlib.sha256(second).hexdigest()}).encode()
    with pytest.raises(ValueError, match="file hash mismatch"):
        compare_payloads(first, second, assignment, hashlib.sha256(assignment).hexdigest(),
                         "0" * 64, hashlib.sha256(second).hexdigest())


def test_final_gate_requires_hash_frozen_structurally_identical_reference():
    control, treatment = _panel(False), _panel(True)
    first, second = _payload(control), _payload(treatment)
    hashes = hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest()
    assignment = json.dumps({"control_sha256": hashes[0],
                             "treatment_sha256": hashes[1]}).encode()
    assignment_hash = hashlib.sha256(assignment).hexdigest()
    with pytest.raises(ValueError, match="requires frozen reference"):
        compare_payloads(first, second, assignment, assignment_hash, *hashes)
    with pytest.raises(ValueError, match="reference prediction hash mismatch"):
        compare_payloads(first, second, assignment, assignment_hash, *hashes,
                         reference_payload=first, reference_sha256="0" * 64)

    changed = copy.deepcopy(control)
    changed[0]["row_sha256"] = "f" * 64
    unsigned = {key: value for key, value in changed[0].items() if key != "record_sha256"}
    changed[0]["record_sha256"] = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()
    reference = _payload(changed)
    with pytest.raises(ValueError, match="semantic metadata differs"):
        compare_payloads(first, second, assignment, assignment_hash, *hashes,
                         reference_payload=reference,
                         reference_sha256=hashlib.sha256(reference).hexdigest())


def test_causal_comparator_rejects_semantic_metadata_drift_between_arms():
    control, treatment = _panel(False), _panel(True)
    changed = copy.deepcopy(treatment)
    changed[0]["scenario_id"] = "different"
    unsigned = {key: value for key, value in changed[0].items() if key != "record_sha256"}
    changed[0]["record_sha256"] = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()
    first, second = _payload(control), _payload(changed)
    hashes = hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest()
    assignment = json.dumps({"control_sha256": hashes[0],
                             "treatment_sha256": hashes[1]}).encode()
    with pytest.raises(ValueError, match="semantic metadata differs"):
        compare_payloads(first, second, assignment, hashlib.sha256(assignment).hexdigest(),
                         *hashes)
