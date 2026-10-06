import copy
import math

import pytest

from scripts.exact_policy_executor import execute_policy
from scripts.score_exact_policy_panel import _risk_coverage, classify, summarize


def _action(name):
    return {"name": name, "payload": {"code": name, "rank": 1}}


def _row(row_id="row-0", *, language="en", order="no_yes", side=0):
    policy = {
        "fields": ["risk", "evidence"],
        "mode": "collect",
        "rules": [
            {"when": {"not": {"field": "risk"}}, "action": _action("review"),
             "primary_rank": 0},
            {"when": {"field": "evidence"}, "action": _action("retain"),
             "primary_rank": None},
        ],
        "fallback": None,
    }
    trace = {"risk": True, "evidence": False}
    outcome = execute_policy(policy, trace).as_dict()
    labels = ["no", "yes"] if order == "no_yes" else ["yes", "no"]
    questions = {}
    targets = {}
    for index, field in enumerate(policy["fields"]):
        qid = f"q{index:02d}"
        questions[qid] = {
            "type": "noul",
            "instructions": f"is {field} true?",
            "criteria": {label: label.upper() for label in labels},
        }
        targets[qid] = {"label": "yes" if trace[field] else "no"}
    return {
        "id": row_id,
        "state": "record",
        "questions": questions,
        "targets": targets,
        "meta": {
            "panel": "exact_policy_v1",
            "split": "eval",
            "semantic_group_id": "cluster-0",
            "counterfactual_side": side,
            "language": language,
            "option_order": order,
            "policy_mode": "collect",
            "policy_topology": "collect-independent",
            "topology_slice": "heldout",
            "renderer_family": "dispatch",
            "source_family": "support",
            "field_order": policy["fields"],
            "qid_to_field": {"q00": "risk", "q01": "evidence"},
            "decision_threshold": 0.5,
            "action_trace": trace,
            "policy": policy,
            "gold_outcome": outcome,
        },
    }


def _prediction(row, *, risk=0.9, evidence=0.1):
    return {
        "id": row["id"],
        "probabilities": {
            "q00": {"no": 1 - risk, "yes": risk},
            "q01": {"no": 1 - evidence, "yes": evidence},
        },
    }


def test_classify_supports_empty_action_set_and_null_primary():
    row = _row()
    result = classify(row, _prediction(row)["probabilities"])
    assert result["error_category"] == "correct"
    assert result["predicted_outcome"].primary is None
    assert result["predicted_outcome"].actions == ()
    assert result["gold_outcome_probability"] == pytest.approx(0.81)
    assert result["complete_outcome_nll"] == pytest.approx(-math.log(0.81))
    assert result["categorical_brier"] == pytest.approx(0.0524)
    assert result["top_outcome_confidence"] == pytest.approx(0.81)
    assert result["top_outcome_correct"] is True


def test_classify_executes_multi_action_set_with_null_primary():
    row = _row()
    row["meta"]["action_trace"] = {"risk": True, "evidence": True}
    row["targets"]["q01"] = {"label": "yes"}
    row["meta"]["gold_outcome"] = execute_policy(
        row["meta"]["policy"], row["meta"]["action_trace"]
    ).as_dict()
    result = classify(row, _prediction(row, evidence=0.9)["probabilities"])
    assert result["error_category"] == "correct"
    assert result["predicted_outcome"].primary is None
    assert [action.name for action in result["predicted_outcome"].actions] == ["retain"]


def test_option_order_does_not_change_semantics():
    left, right = _row("left", order="no_yes"), _row("right", order="yes_no")
    assert classify(left, _prediction(left)["probabilities"])["predicted_outcome"] == classify(
        right, _prediction(right)["probabilities"]
    )["predicted_outcome"]


def test_summarize_separates_rendered_endpoints_and_clusters():
    rows = [
        _row("a", language="en", order="no_yes", side=0),
        _row("b", language="he", order="yes_no", side=0),
        _row("c", language="en", order="no_yes", side=1),
        _row("d", language="he", order="yes_no", side=1),
    ]
    predictions = [_prediction(row) for row in rows]
    report = summarize(rows, predictions)
    assert report["rendered"]["exact_outcome_accuracy"] == 1.0
    assert report["semantic_endpoints"] == {"n": 2, "all_views_exact_accuracy": 1.0}
    assert report["independent_clusters"] == {"n": 1, "all_views_exact_accuracy": 1.0}
    assert set(report["slices"]["language"]) == {"en", "he"}
    calibration = report["complete_outcome_calibration"]
    assert calibration["factorization_assumption"] == "independent_bernoulli_semantic_fields"
    assert calibration["valid_only"]["n"] == 4
    assert calibration["headline_fail_closed"]["categorical_brier"] == pytest.approx(
        0.0524
    )
    assert calibration["headline_fail_closed"]["aurc"] == 0.0
    endpoint = calibration["semantic_endpoint_robustness_fail_closed"]
    cluster = calibration["independent_cluster_robustness_fail_closed"]
    assert endpoint["n"] == 2
    assert endpoint["all_views_top_outcome_accuracy"] == 1.0
    assert "aurc" not in endpoint
    assert "fixed_bin_ece" not in endpoint
    assert "top_outcome_confidence" not in endpoint
    assert cluster["n"] == 1


def test_summarize_counts_invalid_predictions_without_hiding_data_errors():
    row = _row()
    bad_prediction = _prediction(row)
    bad_prediction["probabilities"]["q00"] = {"no": 0.2, "yes": 0.2}
    report = summarize([row], [bad_prediction])
    assert report["rendered"]["coverage"] == 0.0
    assert report["rendered"]["invalid"] == 1
    assert report["rendered"]["exact_outcome_accuracy"] == 0.0
    assert report["rendered"]["field_accuracy"] == 0.0
    calibration = report["complete_outcome_calibration"]
    assert calibration["valid_only"]["n"] == 0
    assert calibration["headline_fail_closed"]["complete_outcome_nll"] == pytest.approx(
        -math.log(1e-12)
    )
    assert calibration["headline_fail_closed"]["categorical_brier"] == 2.0
    assert calibration["headline_fail_closed"]["aurc"] == 1.0

    broken_row = copy.deepcopy(row)
    broken_row["meta"]["gold_outcome"] = {
        "primary": _action("impossible"), "actions": [_action("impossible")]
    }
    with pytest.raises(ValueError, match="gold outcome"):
        summarize([broken_row], [])


def test_duplicate_prediction_ids_fail_closed():
    row = _row()
    prediction = _prediction(row)
    with pytest.raises(ValueError, match="prediction ids"):
        summarize([row], [prediction, prediction])


def test_outcome_calibration_marginalizes_many_traces_into_same_outcome():
    row = _row()
    row["meta"]["policy"] = {
        "fields": ["risk", "evidence"],
        "mode": "priority",
        "rules": [
            {"when": {"field": "risk"}, "action": _action("retain")},
            {"when": {"field": "evidence"}, "action": _action("review")},
        ],
        "fallback": _action("fallback"),
    }
    row["meta"]["policy_mode"] = "priority"
    row["meta"]["gold_outcome"] = execute_policy(
        row["meta"]["policy"], row["meta"]["action_trace"]
    ).as_dict()
    # Priority maps both evidence values to retain when risk=true, so the gold
    # outcome probability is 0.6 rather than a single trace's 0.6 * 0.8.
    result = classify(row, _prediction(row, risk=0.6, evidence=0.2)["probabilities"])
    assert result["gold_outcome_probability"] == pytest.approx(0.6)
    assert result["top_outcome_confidence"] == pytest.approx(0.6)
    assert result["complete_outcome_nll"] == pytest.approx(-math.log(0.6))


def test_outcome_calibration_handles_deterministic_field_probabilities():
    row = _row()
    result = classify(row, _prediction(row, risk=1.0, evidence=0.0)["probabilities"])
    assert result["gold_outcome_probability"] == 1.0
    assert result["complete_outcome_nll"] == 0.0
    assert result["categorical_brier"] == 0.0
    assert result["top_outcome_confidence"] == 1.0


def test_fail_closed_calibration_does_not_reward_partial_coverage():
    rows = [_row("valid"), _row("missing")]
    report = summarize(rows, [_prediction(rows[0])])
    calibration = report["complete_outcome_calibration"]
    assert calibration["valid_only"]["n"] == 1
    assert calibration["headline_fail_closed"]["n"] == 2
    assert calibration["headline_fail_closed"]["categorical_brier"] == pytest.approx(
        (0.0524 + 2.0) / 2
    )
    assert calibration["headline_fail_closed"]["top_outcome_accuracy"] == 0.5
    # The invalid row is maximally confident and therefore appears first in
    # selective prediction, making the AURC worse than final error alone.
    assert calibration["headline_fail_closed"]["aurc"] == pytest.approx(0.75)
    cluster = calibration["independent_cluster_robustness_fail_closed"]
    assert cluster["n"] == 1
    assert cluster["all_views_top_outcome_accuracy"] == 0.0
    assert cluster["mean_view_categorical_brier"] == pytest.approx(
        (0.0524 + 2.0) / 2
    )


def test_risk_coverage_is_tie_invariant():
    rows = [_row("correct"), _row("wrong")]
    predictions = [
        _prediction(rows[0], risk=0.9, evidence=0.1),
        _prediction(rows[1], risk=0.1, evidence=0.9),
    ]
    first = summarize(rows, predictions)["complete_outcome_calibration"][
        "headline_fail_closed"
    ]
    second = summarize(list(reversed(rows)), list(reversed(predictions)))[
        "complete_outcome_calibration"
    ]["headline_fail_closed"]
    assert first["aurc"] == second["aurc"]
    assert first["risk_coverage"] == second["risk_coverage"]


def test_tied_risk_coverage_uses_whole_tie_groups():
    entries = [
        {"top_outcome_confidence": 0.9, "top_outcome_correct": True},
        {"top_outcome_confidence": 0.9, "top_outcome_correct": False},
        {"top_outcome_confidence": 0.5, "top_outcome_correct": True},
    ]
    aurc, curve = _risk_coverage(entries)
    assert aurc == pytest.approx(4 / 9)
    assert curve[0] == {
        "target_coverage": 0.1,
        "coverage": pytest.approx(2 / 3),
        "risk": 0.5,
        "minimum_confidence": 0.9,
    }


def test_calibration_contract_requires_half_threshold_and_strict_probabilities():
    row = _row()
    row["meta"]["decision_threshold"] = 0.6
    with pytest.raises(ValueError, match="requires decision threshold 0.5"):
        classify(row, _prediction(row)["probabilities"])

    row["meta"]["decision_threshold"] = 0.5
    probabilities = _prediction(row)["probabilities"]
    probabilities["q00"] = {"no": 0.1000000005, "yes": 0.9}
    result = classify(row, probabilities)
    assert result["gold_outcome_probability"] == pytest.approx(
        (0.9 / 1.0000000005) * 0.9
    )

    probabilities = _prediction(row)["probabilities"]
    probabilities["q00"] = {"no": 0.100000002, "yes": 0.9}
    with pytest.raises(ValueError, match="do not sum to one"):
        classify(row, probabilities)
