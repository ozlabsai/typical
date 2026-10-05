import copy

from scripts.audit_r043_training import ARMS, COMMON_CONFIG, TREATMENTS, audit


def _config(arm):
    return {**COMMON_CONFIG, "train": [f"/run/arm-{arm}.jsonl"],
            "val": ["wf=/run/val-wf.jsonl", "wh=/run/val-wh.jsonl",
                    "replay=/run/val-replay.jsonl"],
            "pair_flip_weight": TREATMENTS[arm]}


def _metrics(arm):
    rows = [{"step": 0}]
    for step in range(1, 351):
        paired = 8 if step >= 201 else 0
        rows.append({"step": step, "staged_single_slots": 8 - paired,
                     "staged_counterfactual_slots": paired,
                     "staged_singles_seen": min(step, 200) * 8,
                     "staged_counterfactual_pairs_seen": max(0, step - 200) * 8,
                     "pair_flip_seen": paired,
                     "pair_flip_weight_configured": TREATMENTS[arm],
                     "pair_flip_weight": TREATMENTS[arm] if step >= 201 else 0.0})
    return rows


def test_r043_training_audit_proves_exact_stage_and_treatment_exposure():
    result = audit({arm: _config(arm) for arm in ARMS},
                   {arm: _metrics(arm) for arm in ARMS})
    assert result["promotion_pass"]
    assert result["arms"]["b"] == {"singles": 1600, "pair_slots": 1200,
                                      "paired_views": 2400, "configured_weight": .25}


def test_r043_training_audit_rejects_early_pairs_and_control_treatment():
    configs = {arm: _config(arm) for arm in ARMS}
    metrics = {arm: _metrics(arm) for arm in ARMS}
    metrics["b"][200] = copy.deepcopy(metrics["b"][200])
    metrics["b"][200]["staged_counterfactual_slots"] = 8
    configs["d"]["pair_flip_weight"] = .25
    result = audit(configs, metrics)
    assert not result["gates"]["b_staged_exposure_exact"]
    assert not result["gates"]["d_config_exact"]
    assert not result["gates"]["only_b_and_c_receive_treatment"]
    assert not result["promotion_pass"]
