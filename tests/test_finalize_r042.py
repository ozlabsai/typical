import pytest

from scripts.finalize_r042 import finalize


def _retention(value):
    return {suite: {"accuracy": value} for suite in ("wf", "wh", "replay")}


def _action(value):
    return {mode: {"n": 256, "exact_action_accuracy": value} for mode in ("raw", "calibrated")}


def _args():
    return ({"promotion_pass": True}, {"gates": {"locked_probe": True}}, _retention(.90),
            _retention(.88), _retention(.88), {"comparison": {"promotion_pass": True}},
            {"comparison": {"promotion_pass": True}}, _action(.90), _action(.90), _action(.90))


def test_r042_final_gate_requires_all_retention_and_action_conditions():
    assert finalize(*_args())["promotion_pass"]

    args = list(_args())
    args[4] = _retention(.87)  # prose loses more than two points on every generic screen.
    result = finalize(*args)
    assert not result["promotion_pass"]
    assert not result["gates"]["prose_wf_accuracy_loses_at_most_.02"]

    args = list(_args())
    args[8] = _action(.89)  # Card loses more than 2/256 versus both base and prose.
    result = finalize(*args)
    assert not result["gates"]["card_raw_action_loses_at_most_2_of_256_vs_base"]
    assert not result["gates"]["card_calibrated_action_not_over_2_of_256_below_prose"]


def test_r042_final_gate_rejects_non_256_action_reports():
    args = list(_args())
    args[7]["raw"]["n"] = 128
    with pytest.raises(ValueError, match="must be 256"):
        finalize(*args)
