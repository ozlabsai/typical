from scripts.compare_renderer_probe import compare


def test_r040_renderer_probe_requires_repair_and_matched_control_win():
    control = {
        "bare_no_both_orders": 0.0, "bare_yes_both_orders": 1.0, "bare_mean_order_delta": 0.0,
        "canonical_no_both_orders": 1.0, "canonical_yes_both_orders": 1.0, "canonical_mean_order_delta": 0.0,
    }
    treatment = {**control, "bare_no_both_orders": 59 / 65}
    report = compare(treatment, control)
    assert report["bare_no_both_orders_delta"] == 59 / 65
    assert all(report["gates"].values())
