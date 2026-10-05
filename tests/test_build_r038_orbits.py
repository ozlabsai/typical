from collections import Counter

from scripts.build_r038_orbits import _by_source, add_replay, build
from scripts.compare_r038_orbits import compare
from scripts.eval_r038_orbits import audit


def test_r038_arms_are_matched_complete_and_source_disjoint():
    complete, partial, evaluation = build(train_sources=8, eval_sources=8, seed=42)
    assert len(complete) == len(partial) == 256
    assert len(evaluation) == 256
    assert not ({row["meta"]["source_id"] for row in complete} &
                {row["meta"]["source_id"] for row in evaluation})
    assert all(len(rows) == 32 for rows in _by_source(complete).values())
    assert all(len(rows) == 32 and len({(row["meta"]["world"], row["meta"]["polarity"],
                                          row["meta"]["wording"], row["meta"]["criteria_style"],
                                          row["meta"]["yes_first"]) for row in rows}) == 16
               for rows in _by_source(partial).values())
    for name in ("world", "polarity", "wording", "criteria_style", "yes_first"):
        assert Counter(row["meta"][name] for row in complete) == Counter(
            row["meta"][name] for row in partial)
    assert Counter(row["targets"]["decision"]["label"] for row in complete) == Counter(
        row["targets"]["decision"]["label"] for row in partial)


def test_r038_keeps_option_order_and_counterfactual_worlds():
    complete, _, _ = build(train_sources=8, eval_sources=8, seed=42)
    by_source = _by_source(complete)
    for rows in by_source.values():
        states = {row["meta"]["world"]: row["state"] for row in rows}
        assert states[0] != states[1]
        assert any(list(row["questions"]["decision"]["criteria"])[0] == "yes" for row in rows)
        assert any(list(row["questions"]["decision"]["criteria"])[0] == "no" for row in rows)


def test_r038_replay_prefix_is_identical_between_arms():
    complete, partial, _ = build(train_sources=8, eval_sources=8, seed=42)
    replay = [{"id": "retained-a"}, {"id": "retained-b"}]
    full, half = add_replay(replay, complete, partial)
    assert full[:2] == half[:2] == replay
    assert len(full) == len(half) == 258


def test_r038_audit_recovers_perfect_twin_and_order_metrics():
    _, _, rows = build(train_sources=8, eval_sources=8, seed=42)
    predictions = [{"id": row["id"], "probabilities": {
        "yes": float(row["targets"]["decision"]["label"] == "yes"),
        "no": float(row["targets"]["decision"]["label"] == "no"),
    }} for row in rows]
    report = audit(rows, predictions)
    assert report["accuracy"] == report["counterfactual_twins_both_correct"] == 1.0
    assert report["nll"] == report["brier"] == 0.0
    assert report["both_orders_correct"] == 1.0
    assert report["mean_semantic_order_delta"] == 0.0


def test_r038_comparison_uses_source_paired_bootstrap():
    _, _, rows = build(train_sources=8, eval_sources=8, seed=42)
    perfect = [{"id": row["id"], "probabilities": {
        "yes": float(row["targets"]["decision"]["label"] == "yes"),
        "no": float(row["targets"]["decision"]["label"] == "no"),
    }} for row in rows]
    wrong = [{"id": row["id"], "probabilities": {"yes": 1 - item["probabilities"]["yes"],
                                                      "no": 1 - item["probabilities"]["no"]}}
             for row, item in zip(rows, perfect)]
    report = compare(rows, perfect, wrong, samples=100, seed=42)
    assert report["accuracy_delta"] == report["twins_delta"] == 1.0
    assert report["nll_ratio"] == report["brier_ratio"] == 0.0
    assert all(report["gates"].values())
