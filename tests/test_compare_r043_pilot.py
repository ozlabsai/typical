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
        "c": predictions(rows, failed),
    }
    reports = {arm: _report(rows, values) for arm, values in arm_predictions.items()}
    result = compare(rows, arm_predictions, reports, input_sha256="eval-hash",
                     model_revision=REVISION, samples=300)
    assert result["deltas"]["b_minus_a"]["exact_action_accuracy"]["delta"] == pytest.approx(8 / 256)
    assert result["deltas"]["b_minus_c"]["twin_both_correct"]["delta"] == pytest.approx(8 / 256)
    assert result["promotion_pass"]


def test_r043_comparison_rejects_report_prediction_disagreement():
    rows = panel()
    arm_predictions = {arm: predictions(rows) for arm in ("base", "a", "b", "c")}
    reports = {arm: _report(rows, values) for arm, values in arm_predictions.items()}
    reports["b"] = copy.deepcopy(reports["b"])
    reports["b"]["presentations"] = 511
    with pytest.raises(ValueError, match="does not match predictions"):
        compare(rows, arm_predictions, reports, input_sha256="eval-hash",
                model_revision=REVISION, samples=100)
