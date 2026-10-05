import copy

import pytest

from scripts.compare_r043_pilot import compare
from scripts.eval_r043_pairs import audit
from tests.test_eval_r043_pairs import panel, predictions


REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


def _report(rows, values):
    report = audit(rows, values)
    report.update({"input_sha256": "eval-hash", "model_revision": REVISION})
    return report


def test_r043_promotes_only_a_two_point_b_win_over_a_and_c():
    rows = panel()
    failed = [row["meta"]["source_id"] for row in rows[:8]]
    arm_predictions = {
        "base": predictions(rows),
        "a": predictions(rows, failed),
        "b": predictions(rows),
        "c": predictions(rows),
        "d": predictions(rows),
    }
    reports = {arm: _report(rows, values) for arm, values in arm_predictions.items()}
    result = compare(rows, arm_predictions, reports, input_sha256="eval-hash",
                     model_revision=REVISION, samples=300)
    assert result["deltas"]["b_minus_a"]["exact_action_accuracy"]["delta"] == pytest.approx(8 / 256)
    assert result["deltas"]["difference_in_differences"]["twin_both_correct"]["delta"] == pytest.approx(8 / 256)
    assert result["deltas"]["b_minus_a"]["twin_both_correct"]["lower_95"] > 0
    assert result["promotion_pass"]


def test_r043_rejects_an_effect_explained_by_the_shuffled_control():
    rows = panel()
    failed = [row["meta"]["source_id"] for row in rows[:8]]
    arm_predictions = {"base": predictions(rows), "a": predictions(rows, failed),
                       "b": predictions(rows), "c": predictions(rows),
                       "d": predictions(rows, failed)}
    reports = {arm: _report(rows, values) for arm, values in arm_predictions.items()}
    result = compare(rows, arm_predictions, reports, input_sha256="eval-hash",
                     model_revision=REVISION, samples=300)
    assert result["deltas"]["b_minus_a"]["exact_action_accuracy"]["delta"] > .02
    assert result["deltas"]["difference_in_differences"]["exact_action_accuracy"]["delta"] == 0
    assert not result["gates"]["difference_in_differences_exact_action_accuracy_at_least_.02"]
    assert not result["promotion_pass"]


def test_r043_comparison_rejects_report_prediction_disagreement():
    rows = panel()
    arm_predictions = {arm: predictions(rows) for arm in ("base", "a", "b", "c", "d")}
    reports = {arm: _report(rows, values) for arm, values in arm_predictions.items()}
    reports["b"] = copy.deepcopy(reports["b"])
    reports["b"]["presentations"] = 511
    with pytest.raises(ValueError, match="does not match predictions"):
        compare(rows, arm_predictions, reports, input_sha256="eval-hash",
                model_revision=REVISION, samples=100)


def test_r043_gates_polarity_order_and_family_regressions():
    rows = panel()
    affected = [row["meta"]["source_id"] for row in rows
                if row["paired_views"][0]["meta"]["polarity"] == "positive"
                and row["paired_views"][0]["meta"]["yes_first"]][:4]
    arm_predictions = {arm: predictions(rows, affected if arm == "b" else ())
                       for arm in ("base", "a", "b", "c", "d")}
    reports = {arm: _report(rows, values) for arm, values in arm_predictions.items()}
    result = compare(rows, arm_predictions, reports, input_sha256="eval-hash",
                     model_revision=REVISION, samples=100)
    assert not result["gates"]["b_polarity_accuracy_delta_at_most_.01"]
    assert not result["gates"]["b_option_order_accuracy_delta_at_most_.01"]
    assert not result["gates"]["b_positive_accuracy_not_over_.02_below_a"]
    assert not result["gates"]["b_invoice_exact_action_accuracy_not_over_.02_below_a"]


def test_r043_gates_nll_and_brier_against_both_controls():
    rows = panel()
    arm_predictions = {arm: predictions(rows) for arm in ("base", "a", "b", "c", "d")}
    for item in arm_predictions["b"]:
        yes_is_predicted = item["probabilities"]["yes"] > .5
        item["probabilities"] = {"yes": .6 if yes_is_predicted else .4,
                                 "no": .4 if yes_is_predicted else .6}
    reports = {arm: _report(rows, values) for arm, values in arm_predictions.items()}
    result = compare(rows, arm_predictions, reports, input_sha256="eval-hash",
                     model_revision=REVISION, samples=100)
    for control in ("a", "c"):
        assert not result["gates"][f"b_nll_not_over_1pct_worse_than_{control}"]
        assert not result["gates"][f"b_brier_not_over_1pct_worse_than_{control}"]
