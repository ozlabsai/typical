import copy

import pytest

from scripts.exact_policy_executor import execute_policy
from scripts.score_exact_policy_panel import classify, summarize


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


def test_summarize_counts_invalid_predictions_without_hiding_data_errors():
    row = _row()
    bad_prediction = _prediction(row)
    bad_prediction["probabilities"]["q00"] = {"no": 0.2, "yes": 0.2}
    report = summarize([row], [bad_prediction])
    assert report["rendered"]["coverage"] == 0.0
    assert report["rendered"]["invalid"] == 1
    assert report["rendered"]["exact_outcome_accuracy"] == 0.0
    assert report["rendered"]["field_accuracy"] == 0.0

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
