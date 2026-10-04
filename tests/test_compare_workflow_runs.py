import json

import pytest

from scripts.compare_workflow_runs import compare


def write_run(path, rows, observations):
    path.mkdir()
    (path / "scores.json").write_text(json.dumps({"scores": rows}))
    (path / "results.json").write_text(json.dumps({"runs": [{"cases": [
        {"observations": observations},
    ]}]}))


def row(run_id, case, exact, primary):
    return {"run_id": run_id, "label_set_id": "labels-consensus", "case_id": case,
            "policy_id": "p", "metrics": {"exact_actions": exact, "primary_action": primary}}


def test_case_transitions_gaps_and_recorded_threshold_margins(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, [row("a", "1", True, True), row("a", "2", False, True)], [
        {"probabilities": {"true": .7, "false": .3}, "threshold": .6},
        {"probabilities": {"true": .2, "false": .8}, "threshold": None},
    ])
    write_run(b, [row("b", "1", False, False), row("b", "2", True, True)], [
        {"probabilities": {"true": .9, "false": .1}},
    ])

    result = compare(a, b)
    assert result["shared_case_policy_pairs"] == 2
    assert result["exact_action_transitions"] == {
        "correct_to_wrong": 1, "wrong_to_correct": 1}
    assert result["run_a_primary_exact_gap"]["primary_only_correct"] == 1
    assert result["run_a_primary_exact_gap"]["exact_only_correct"] == 0
    assert result["run_a_probabilities"]["probability_observations"] == 2
    assert result["run_a_probabilities"]["explicit_threshold_margins"] == pytest.approx([.1])
    assert result["run_b_probabilities"] == {
        "probability_observations": 1, "explicit_threshold_margins": []}
