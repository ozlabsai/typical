import pytest

from scripts.eval_s2_crossed import aggregate_predictions, parse_adapter, select_subset


def prediction(example_id, source_id, group, order, gold, predicted, p_yes, correct, nll, brier):
    return {
        "example_id": example_id, "source_id": source_id, "source_group": group,
        "question_style": "interrogative", "criteria_style": "explicit",
        "polarity": "positive", "context_view": "compact", "placement": "middle",
        "option_order": order, "gold": gold, "predicted": predicted, "p_yes": p_yes,
        "correct": correct, "nll": nll, "brier": brier,
    }


def test_aggregates_cells_recall_groups_order_robustness_and_failure_ids():
    rows = [
        prediction("x1-ny", "src1", "g1", ("no", "yes"), "yes", "yes", .8, True, .2, .04),
        prediction("x1-yn", "src1", "g1", ("yes", "no"), "yes", "no", .3, False, 1.2, .09),
        prediction("x2-ny", "src2", "g2", ("no", "yes"), "no", "no", .1, True, .1, .01),
        prediction("x2-yn", "src2", "g2", ("yes", "no"), "no", "no", .15, True, .15, .0225),
    ]
    result = aggregate_predictions(rows)

    assert result["predictions"] == rows
    assert result["overall"] == {"n": 4, "accuracy": .75, "nll": .4125, "brier": .040625}
    assert result["per_condition_recall"] == {
        "no": {"n": 2, "recall": 1.0}, "yes": {"n": 2, "recall": .5}}
    assert result["per_group_all_correct"] == {
        "n_groups": 2, "all_correct_groups": 1, "all_correct_rate": .5}
    assert result["order_robustness"] == {
        "n_pairs": 2, "both_correct_rate": .5, "mean_abs_aligned_p_yes_delta": .275}
    assert result["failures"] == [{
        "example_id": "x1-yn", "source_group": "g1", "source_id": "src1",
        "gold": "yes", "predicted": "no", "p_yes": .3,
    }]
    assert all("state" not in failure for failure in result["failures"])
    assert len(result["cells"]) == 2


def test_rejects_incomplete_order_pairs_and_empty_inputs():
    one = prediction("x", "src", "g", ("no", "yes"), "yes", "yes", .7, True, .3, .09)
    with pytest.raises(ValueError, match="order pair incomplete"):
        aggregate_predictions([one])
    with pytest.raises(ValueError, match="empty"):
        aggregate_predictions([])


def test_adapter_parser_accepts_explicit_base_convention():
    assert parse_adapter("base=base") == ("base", "base")
    assert parse_adapter("s1=/models/adapter") == ("s1", "/models/adapter")
    with pytest.raises(Exception):
        parse_adapter("not-a-pair")


def test_compact_and_full_subsets_preserve_both_order_pairs():
    examples = []
    for view in ("compact", "full"):
        for order in (("no", "yes"), ("yes", "no")):
            examples.append({"source_id": "src", "source_group": "group",
                             "question_style": "interrogative", "criteria_style": "explicit",
                             "polarity": "negative", "context_view": view, "placement": "middle",
                             "option_order": order})
    assert len(select_subset(examples, "all")) == 4
    assert len(select_subset(examples, "compact")) == 2
    assert len(select_subset(examples, "full")) == 2
    with pytest.raises(ValueError, match="subset"):
        select_subset(examples, "training")
