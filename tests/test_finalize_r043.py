import copy

import pytest

from scripts.finalize_r043 import finalize


REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


def _retention(value):
    return {suite: {"n": 128, "accuracy": value, "skipped_overlength": 0}
            for suite in ("wf", "wh", "replay")}


def _action(value):
    return {mode: {"n": 256, "exact_action_accuracy": value,
                   "error_categories": {"invalid": 0}}
            for mode in ("raw", "calibrated")}


def _mmhe():
    suites = {suite: {"n": 128} for suite in
              ("he", "en", "real", "shuffled", "noimage", "mismatch", "gold_dropped")}
    return {"base": copy.deepcopy(suites), "candidate": copy.deepcopy(suites),
            "comparison": {"gates": {"locked_policy": True}, "promotion_pass": True}}


def _inputs():
    comparison = {"model_revision": REVISION, "gates": {"pilot": True}, "promotion_pass": True}
    retentions = {"base": _retention(.90), "a": _retention(.90),
                  "b": _retention(.89), "c": _retention(.90)}
    actions = {arm: _action(.90) for arm in ("base", "a", "b", "c")}
    return comparison, retentions, actions, _mmhe()


def test_r043_terminal_gate_combines_causal_and_preservation_checks():
    assert finalize(*_inputs(), model_revision=REVISION)["promotion_pass"]
    comparison, retentions, actions, mmhe = _inputs()
    retentions["b"] = _retention(.87)
    result = finalize(comparison, retentions, actions, mmhe, model_revision=REVISION)
    assert not result["gates"]["b_wf_loses_at_most_.02_vs_base"]
    assert not result["promotion_pass"]


def test_r043_terminal_gate_fails_on_missing_rows():
    comparison, retentions, actions, mmhe = _inputs()
    retentions["a"]["wf"]["skipped_overlength"] = 1
    with pytest.raises(ValueError, match="without skipping"):
        finalize(comparison, retentions, actions, mmhe, model_revision=REVISION)
