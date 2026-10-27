"""Distillation data: the multi-lens planner's call at every decision point of many randomised raids.

    uv run --no-sync python airdef/dataset.py [--out .context/typically/jobs/airdef_t] [--train 30000]

Each raid runs the planner (planner.py) with 15% exploration so off-plan states appear too, under a randomised
night: magazine posture, stock levels, priority sites, damage weights, allied assets, laser weather, Pk scale.
Every decision point becomes a row: the verbalised state (engine.verbalise, byte-identical to the live demo), the
options as candidates, and a soft target = softmax of the planner's expected costs (near-ties stay soft, so the
model's confidence means something). Train / val / eval are split by raid seed; eval is 100% held-out raids.
"""
import argparse
import json
import math
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from engine import Q, label, run_headless
from sim import DEFAULT_CONFIG, EFFECTORS, SCENARIOS, SITES

SPLITS = {"train": range(1000, 1200), "val": range(2000, 2030), "eval": range(3000, 3060)}


def random_night(rng):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update(reserve=rng.choice([0, 1, 1, 2]), allied=rng.random() < 0.8, pk_scale=round(rng.uniform(0.85, 1.05), 2),
               iron_beam_weather="cloudy" if rng.random() < 0.2 else "clear",
               priority_sites=rng.sample([s[0] for s in SITES], rng.randint(1, 5)),
               priority_multiplier=rng.choice([2.0, 3.0, 5.0]), forecast_s=rng.choice([120, 300, 900, 900, 1800]))
    cfg["tier_value"] = {k: v * rng.uniform(0.4, 2.5) for k, v in cfg["tier_value"].items()}
    scale = {e: rng.choice([0.2, 0.4, 0.7, 1.0, 1.0, 1.3]) for e in ("arrow3", "arrow2", "thaad", "dsling", "ironDome", "sm3", "sm6")}
    return cfg, scale


def raid(job):
    """One randomised raid -> its decision points, keeping every fire / fire-two / let-fall decision and a sample of
    the waits (a waiting threat is re-checked every few seconds, so waits would otherwise be ~90% of all points)."""
    scenario, seed, explore = job
    rng = random.Random(seed * 7919 + hash(scenario) % 1000)
    cfg, scale = random_night(rng)
    SCENARIOS[scenario]["stock_scale"] = {**SCENARIOS[scenario].get("stock_scale", {}), **scale}   # per-process copy
    rec = []
    run_headless(scenario, cfg=cfg, seed=seed, policies=("planner",), explore=explore, record=rec)
    act = lambda r: min(r["opts"], key=lambda o: o[1])[0].split(":")[0]
    keep = [r for r in rec if act(r) != "wait"]
    waits = [r for r in rec if act(r) == "wait"]
    return keep + rng.sample(waits, min(len(waits), 2 * len(keep) + 20))


def to_row(r, rng):
    opts = list(r["opts"])
    rng.shuffle(opts)
    cmin = min(c for _, c in opts)
    tau = 0.15 * max(cmin, 0) + 0.02   # $M: options within ~tau of the best keep real probability mass
    w = [math.exp(-(c - cmin) / tau) for _, c in opts]
    z = sum(w)
    target = [x / z for x in w]
    best = max(range(len(opts)), key=lambda i: target[i])
    cls = r["state"].split(", a ")[1].split(" from")[0]
    return {"state": r["state"], "query": Q, "candidates": [label(k) for k, _ in opts], "target": target, "p_null": 0.0,
            "task": "airdef_engage", "label": best,
            "meta": {"qtype": "choice", "family": "airdef_engage", "scenario": r["scenario"], "cls": cls,
                     "best": opts[best][0], "margin": sorted(w, reverse=True)[1] / z if len(w) > 1 else 0.0}}


def collect(seeds, explore, procs=10):
    jobs = [(sc, s, explore) for s in seeds for sc in SCENARIOS]
    with Pool(procs) as p:
        recs = [r for out in p.imap_unordered(raid, jobs) for r in out]
    seen, out = set(), []
    for r in recs:   # one row per distinct (state, options): the memo of the live demo works the same way
        k = r["state"] + str(sorted(k for k, _ in r["opts"]))
        if k not in seen and len(r["opts"]) > 1:
            seen.add(k)
            out.append(r)
    return out


def cell(r):
    a = r["meta"]["best"].split(":")[0]
    return ("fire" if a.startswith("fire") else a, r["meta"]["cls"])


def balance(rows, n, rng, weight=None):
    """Cap each (action, threat class) cell: firing, waiting and letting fall all get real weight, for every class.
    weight: threat class -> cap multiplier (e.g. {"ballistic missile": 2.5} to lean on a weak spot)."""
    weight = weight or {}
    rng.shuffle(rows)
    cells = Counter(cell(r) for r in rows)
    w = lambda c: weight.get(c[1], 1.0)
    cap = max(1, n // max(len(cells), 1))
    while sum(min(k, cap * w(c)) for c, k in cells.items()) < min(n, len(rows)) and cap < n:
        cap += max(1, cap // 10)
    used, out = Counter(), []
    for r in rows:
        c = cell(r)
        if used[c] < cap * w(c) and len(out) < n:
            used[c] += 1
            out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / ".context/typically/jobs/airdef_t"))
    ap.add_argument("--train", type=int, default=30000)
    ap.add_argument("--ballistic_weight", type=float, default=1.0, help="cap multiplier for ballistic-missile cells")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    data = {}
    for split, seeds in SPLITS.items():
        recs = collect(seeds, explore=0.15 if split == "train" else 0.0)
        data[split] = [to_row(r, rng) for r in recs]
        print(f"{split}: {len(recs)} distinct decision points from {len(seeds) * len(SCENARIOS)} raids")
    with (out / "pool_train.jsonl").open("w") as f:   # the full unbalanced pool: reweight later without regenerating
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data["train"])
    train = balance(data["train"], a.train, rng, {"ballistic missile": a.ballistic_weight})
    val = balance(data["val"], 1500, rng)
    rng.shuffle(data["eval"])
    files = {"train": train, "val": val}
    for cls in ("ballistic missile", "rocket", "drone", "cruise missile"):   # natural mix of held-out raids, per class
        files[f"eval/airdef_{cls.replace(' ', '_')}"] = [r for r in data["eval"] if r["meta"]["cls"] == cls][:500]
    files["eval/airdef_fire_moments"] = [r for r in data["eval"] if r["meta"]["best"].startswith("fire")][:500]
    for name, rows in files.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        best = Counter(r["meta"]["best"].split(":")[0] for r in rows)
        print(f"{name:30s} {len(rows):6d}  best action {dict(best)}  soft (margin>0.2) {sum(r['meta']['margin'] > 0.2 for r in rows) / max(len(rows), 1):.0%}")
