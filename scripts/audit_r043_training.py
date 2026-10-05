"""Audit R043 configs and every staged-training exposure before promotion."""

import argparse
import json
import math
from pathlib import Path


ARMS = ("a", "b", "c", "d")
TREATMENTS = {"a": 0.0, "b": .25, "c": .25, "d": 0.0}
COMMON_CONFIG = {
    "steps": 350, "microbatch": 1, "grad_accum": 8, "max_length": 8192,
    "lr": 2e-5, "min_lr": 2e-6, "weight_decay": .01, "rank": 32, "alpha": 64,
    "save_every": 350, "eval_every": 350, "eval_samples": 128, "seed": 43,
    "pair_flip_margin": 1.0, "pair_flip_start_step": 201,
    "stage_counterfactual_pairs": True, "pair_consistency_weight": 0.0,
    "pair_consistency_min_gold_probability": .8, "pair_consistency_clip": 4.0,
    "pair_consistency_beta": 1.0,
    "project_consistency_against_guard": False,
}


def _load(path):
    return json.loads(Path(path).read_text())


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _same(left, right):
    if type(right) in (int, float) and type(left) in (int, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def audit(configs, metrics):
    if set(configs) != set(ARMS) or set(metrics) != set(ARMS):
        raise ValueError("training audit requires exactly A/B/C/D")
    gates, summary = {}, {}
    for arm in ARMS:
        config = configs[arm]
        rows = metrics[arm]
        if not isinstance(config, dict) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{arm}: invalid config or metrics schema")
        recipe = all(key in config and _same(config[key], value)
                     for key, value in COMMON_CONFIG.items())
        train = config.get("train")
        recipe &= (isinstance(train, list) and len(train) == 1 and
                   isinstance(train[0], str) and
                   _same(config.get("pair_flip_weight"), TREATMENTS[arm]) and
                   Path(train[0]).name == f"arm-{arm}.jsonl")
        val_items = config.get("val", [])
        val_names = {name: Path(path).name for name, path in
                     (item.split("=", 1) for item in val_items
                      if isinstance(item, str) and "=" in item)}
        recipe &= (len(val_items) == 3 and val_names == {
            "wf": "val-wf.jsonl", "wh": "val-wh.jsonl", "replay": "val-replay.jsonl"})
        gates[f"{arm}_config_exact"] = recipe

        if len(rows) != 351 or [row.get("step") for row in rows] != list(range(351)):
            raise ValueError(f"{arm}: metrics must contain baseline plus every step 1..350")
        exposure_ok = weight_ok = True
        for row in rows[1:]:
            step = row["step"]
            paired = 8 if step >= 201 else 0
            singles = 8 - paired
            exposure_ok &= (
                row.get("staged_single_slots") == singles and
                row.get("staged_counterfactual_slots") == paired and
                row.get("staged_singles_seen") == min(step, 200) * 8 and
                row.get("staged_counterfactual_pairs_seen") == max(0, step - 200) * 8 and
                row.get("pair_flip_seen") == paired)
            configured = TREATMENTS[arm]
            effective = configured if step >= 201 else 0.0
            weight_ok &= (_same(row.get("pair_flip_weight_configured"), configured) and
                          _same(row.get("pair_flip_weight"), effective))
        gates[f"{arm}_staged_exposure_exact"] = exposure_ok
        gates[f"{arm}_treatment_phase_exact"] = weight_ok
        final = rows[-1]
        summary[arm] = {
            "singles": final.get("staged_singles_seen"),
            "pair_slots": final.get("staged_counterfactual_pairs_seen"),
            "paired_views": 2 * final.get("staged_counterfactual_pairs_seen", -1),
            "configured_weight": config.get("pair_flip_weight"),
        }
    gates["only_b_and_c_receive_treatment"] = all(
        _same(configs[arm].get("pair_flip_weight"), TREATMENTS[arm]) for arm in ARMS)
    return {"arms": summary, "gates": gates, "promotion_pass": all(gates.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}-config", required=True)
        parser.add_argument(f"--{arm}-metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(
        {arm: _load(getattr(args, f"{arm}_config")) for arm in ARMS},
        {arm: _read(getattr(args, f"{arm}_metrics")) for arm in ARMS})
    output = Path(args.out)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["promotion_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
