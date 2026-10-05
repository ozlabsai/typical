import pytest

from scripts.compare_renderer_probe import LOCKED_R037_SHA256, compare


def _records(bare_negative_correct):
    records = []
    for renderer in ("bare", "canonical"):
        for label in ("no", "yes"):
            for index in range(65):
                correct = renderer != "bare" or label != "no" or index < bare_negative_correct
                for yes_first in (False, True):
                    records.append({"id": f"{renderer}-{label}-{index}", "renderer": renderer,
                                    "label": label, "yes_first": yes_first, "correct": correct})
    return records


def _report(records):
    report = {"input_sha256": LOCKED_R037_SHA256, "per_label": 65,
              "bare_mean_order_delta": 0.0, "canonical_mean_order_delta": 0.0}
    for renderer in ("bare", "canonical"):
        for label in ("no", "yes"):
            rows = [row for row in records if row["renderer"] == renderer and row["label"] == label]
            report[f"{renderer}_{label}"] = {"n": len(rows)}
            report[f"{renderer}_{label}_both_orders"] = sum(
                all(row["correct"] for row in rows if row["id"] == f"{renderer}-{label}-{index}")
                for index in range(65)) / 65
    return report


def test_r040_renderer_probe_requires_complete_panel_and_meaningful_paired_win():
    control, treatment = _records(0), _records(59)
    result = compare(_report(treatment), _report(control), treatment, control)
    assert result["bare_no_paired_delta"] == 59 / 65
    assert all(result["gates"].values())


def test_r040_renderer_probe_rejects_a_shortcut_small_panel():
    control, treatment = _records(0), _records(59)
    report = _report(treatment)
    report["per_label"] = 1
    with pytest.raises(ValueError, match="locked"):
        compare(report, _report(control), treatment, control)
