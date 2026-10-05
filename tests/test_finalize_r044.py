import copy
import json

import pytest

from scripts.finalize_r044 import finalize, main


PARENT = "/frozen/parent"


def _manifest():
    return {"experiment": "r044_mm_replay", "logical_rows": {"arm-a": 800, "arm-b": 800},
            "leakage_checks": {"selected_base_image_hashes_unique": True,
                               "selected_source_ids_unique": True,
                               "protected_image_overlap": 0, "protected_decision_overlap": 0,
                               "arms_equal_without_visual_image": True}}


def _config(arm):
    return {"steps": 100, "microbatch": 1, "grad_accum": 8, "lr": 5e-6,
            "min_lr": 5e-7, "pair_flip_weight": .25, "pair_flip_margin": 1.0,
            "pair_flip_start_step": 0, "stage_counterfactual_pairs": False,
            "init_adapter": PARENT, "train": [f"/run/arm-{arm}.jsonl"]}


def _metrics():
    return [{"step": 0}] + [{"step": step, "pair_flip_weight": .25,
                              "pair_flip_seen": int(step <= 100)}
                             for step in range(1, 101)]


def _retention(value=.9):
    return {suite: {"n": 128, "accuracy": value, "skipped_overlength": 0}
            for suite in ("wf", "wh", "replay")}


def _action(value=.9):
    return {mode: {"n": 256, "exact_action_accuracy": value,
                   "error_categories": {"invalid": 0}}
            for mode in ("raw", "calibrated")}


def _renderer(value=.9):
    result = {"per_label": 65}
    for cell in ("bare_no", "bare_yes", "canonical_no", "canonical_yes"):
        result[cell] = {"n": 130}
        result[f"{cell}_both_orders"] = value
    return result


def _mmhe(value=.9, absolute=True):
    suites = {suite: {"n": 128, "accuracy": value} for suite in
              ("he", "en", "real", "shuffled", "noimage", "mismatch", "gold_dropped")}
    return {"base": copy.deepcopy(suites), "candidate": copy.deepcopy(suites),
            "comparison": {"gates": {"absolute": absolute}, "promotion_pass": absolute}}


def _pairs(value=.9):
    return {"pairs": 256, "presentations": 512, "accuracy": value,
            "exact_action_accuracy": value, "twin_both_correct": value}


def _inputs():
    return (_manifest(), {arm: _config(arm) for arm in ("a", "b")},
            {arm: _metrics() for arm in ("a", "b")},
            {arm: _retention() for arm in ("parent", "a", "b")},
            {arm: _action() for arm in ("parent", "a", "b")},
            {arm: _renderer() for arm in ("parent", "a", "b")},
            {arm: _mmhe() for arm in ("parent", "a", "b")})


def test_r044_promotes_b_and_keeps_causal_claim_separate():
    args = _inputs()
    args[-1]["a"] = _mmhe(absolute=False)
    args[-1]["a"]["candidate"]["real"]["accuracy"] = 110 / 128
    args[-1]["a"]["candidate"]["gold_dropped"]["accuracy"] = 111 / 128
    args[-1]["b"]["candidate"]["real"]["accuracy"] = 112 / 128
    args[-1]["b"]["candidate"]["gold_dropped"]["accuracy"] = 113 / 128
    args[-1]["parent"]["candidate"]["real"]["accuracy"] = 112 / 128
    args[-1]["parent"]["candidate"]["gold_dropped"]["accuracy"] = 113 / 128
    result = finalize(*args, parent_adapter=PARENT,
                      pairs={arm: _pairs() for arm in ("parent", "a", "b")})
    assert result["promotion_pass"]
    assert result["causal_image_replay"]["claim_pass"]


def test_r044_rejects_relative_preservation_without_absolute_mmhe_pass():
    args = _inputs()
    args[-1]["b"] = _mmhe(absolute=False)
    result = finalize(*args, parent_adapter=PARENT,
                      pairs={arm: _pairs() for arm in ("parent", "a", "b")})
    assert not result["gates"]["b_absolute_mmhe_promotion_pass"]
    assert not result["promotion_pass"]


def test_r044_enforces_one_error_and_exact_exposure():
    args = _inputs()
    args[-1]["b"]["candidate"]["he"]["accuracy"] = .9 - 2 / 128
    result = finalize(*args, parent_adapter=PARENT,
                      pairs={arm: _pairs() for arm in ("parent", "a", "b")})
    assert not result["gates"]["b_he_loses_at_most_1_of_128_vs_parent"]
    args = _inputs()
    args[2]["a"][-1]["pair_flip_seen"] = 0
    with pytest.raises(ValueError, match="each of 100 pairs exactly once"):
        finalize(*args, parent_adapter=PARENT,
                 pairs={arm: _pairs() for arm in ("parent", "a", "b")})


def test_r044_cli_returns_two_for_completed_nonpromotion(tmp_path):
    manifest, configs, metrics, retentions, actions, renderers, mmhe = _inputs()
    actions["b"] = _action(.89)
    paths = {}
    for name, value in {"manifest": manifest,
                        **{f"{arm}-config": value for arm, value in configs.items()},
                        **{f"{arm}-retention": value for arm, value in retentions.items()},
                        **{f"{arm}-action": value for arm, value in actions.items()},
                        **{f"{arm}-renderer": value for arm, value in renderers.items()},
                        **{f"{arm}-mmhe": value for arm, value in mmhe.items()}}.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    for arm, rows in metrics.items():
        paths[f"{arm}-metrics"] = tmp_path / f"{arm}-metrics.jsonl"
        paths[f"{arm}-metrics"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    for arm in ("parent", "a", "b"):
        paths[f"{arm}-pairs"] = tmp_path / f"{arm}-pairs.json"
        paths[f"{arm}-pairs"].write_text(json.dumps(_pairs()))
    argv = ["--parent-adapter", PARENT, "--out", str(tmp_path / "out.json")]
    for name, path in paths.items():
        argv.extend((f"--{name}", str(path)))
    assert main(argv) == 2
    assert not json.loads((tmp_path / "out.json").read_text())["promotion_pass"]
