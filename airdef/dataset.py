"""Distillation data for both decision levels (DECISIONS.md), from the planner playing many randomised raids.

    uv run --no-sync python airdef/dataset.py [--out .context/typically/jobs/airdef_v3] [--global_rows 24000] [--local_rows 12000]

GLOBAL rows: "Which interceptor should take this threat?" over the operative options (Arrow 3, Iron Dome, Fighter jets,
Wait for cheaper, Wait, Let it fall, ...); soft target = softmax of the planner's expected costs, so near-ties stay
soft. LOCAL rows: "How many interceptors should this unit fire at the threat now?" as an ordinal Score 0-3 (the trainer
applies ordinal smoothing to qtype "score"). Raids get a randomised night (posture, stock, priorities, damage weights,
allies, weather, Pk scale, early warning) and 15% exploration. Train / val / eval are split by raid seed; eval also gets
the curated situation drill (situations.py) when it exists.
"""
import argparse
import json
import math
import random
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from engine import LEVELS, Q_BATTERY, run_headless
from sim import DEFAULT_CONFIG, SCENARIOS, SITES, in_israel

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"train": range(1000, 1120), "val": range(2000, 2020), "eval": range(3000, 3040)}
BATTERY_ROWS = 6000
LEVEL_BOOST = 8    # extra level-2 raids per seed, each with its own route: the floor-vs-border trade-off is rare in raids
NORTH = ["haifa_bay", "haifa", "tiberias", "nahariya", "kiryat_shmona", "ramat_david"]
CLS = re.compile(r", a (ballistic missile|rocket|cruise missile|drone) from")


def random_night(rng):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update(reserve=rng.choice([0, 1, 1, 2]), allied=rng.random() < 0.8, pk_scale=round(rng.uniform(0.85, 1.05), 2),
               iron_beam_weather="cloudy" if rng.random() < 0.2 else "clear",
               priority_sites=rng.sample([s[0] for s in SITES], rng.randint(1, 5)),
               priority_multiplier=rng.choice([2.0, 3.0, 5.0]), forecast_s=rng.choice([120, 300, 900, 900, 1800]),
               # the policy: an ammo floor, how hard to avoid going below it, how much stopping it before the border matters
               ammo_floor={"ironDome": rng.choice([0, 8, 12, 12, 20])}, floor_price=rng.choice([0.3, 1.0, 1.0, 5.0]),
               border_price=rng.choice([0.0, 5.0, 5.0, 25.0]))
    cfg["tier_value"] = {k: v * rng.uniform(0.4, 2.5) for k, v in cfg["tier_value"].items()}
    scale = {e: rng.choice([0.2, 0.4, 0.7, 1.0, 1.0, 1.3]) for e in ("arrow3", "arrow2", "thaad", "dsling", "ironDome", "sm3", "sm6")}
    return cfg, scale


def raid(job):
    """One randomised raid -> its decision points at both levels. Waits and holds recur every few seconds while a threat
    is pending, so they are subsampled against the decisions that commit something."""
    scenario, seed, explore = job
    rng = random.Random(seed * 7919 + hash(scenario) % 1000)
    cfg, scale = random_night(rng)
    SCENARIOS[scenario]["stock_scale"] = {**SCENARIOS[scenario].get("stock_scale", {}), **scale}   # per-process copy
    sc = SCENARIOS[scenario]
    if sc.get("level") == 2:   # a new route each time: from just across the northern border to a northern site, seen late or early
        while True:
            lon = rng.uniform(35.2, 35.9)
            o = (rng.uniform(33.35 if lon > 35.5 else 33.15, 33.7), lon)
            if not in_israel(*o):
                break
        sc["threats"] = [{**sc["threats"][0], "origin": o, "target": rng.choice(NORTH), "classify_s": rng.choice([5, 10, 20, 40, 60])}]
    if "batteries" in sc:   # a level: random stocks around the floor and random standing commitments, so every case varies
        sc["batteries"] = {b: (None if n is None else rng.choice([8, 11, 12, 13, 16, 25, 40])) for b, n in sc["batteries"].items()}
        sc["plans"] = {b: (rng.choice([1, 2, 3]), 600) for b in sc["batteries"] if rng.random() < 0.3}
    rec = []
    run_headless(scenario, cfg=cfg, seed=seed, policies=("planner",), explore=explore, record=rec)
    for r in rec:
        r["scenario"] = scenario
    g = [r for r in rec if r["level"] == "global"]
    loc = [r for r in rec if r["level"] == "local"]
    bat = [r for r in rec if r["level"] == "battery"]
    gk = [r for r in g if r["best"] not in ("wait", "hold")]
    gw = [r for r in g if r["best"] in ("wait", "hold")]
    lk = [r for r in loc if r["best"] > 0]
    lh = [r for r in loc if r["best"] == 0]
    return gk + rng.sample(gw, min(len(gw), 2 * len(gk) + 20)) + lk + rng.sample(lh, min(len(lh), len(lk) + 10)) + bat


def to_row(r, rng):
    cls = (CLS.search(r["state"]) or [None, "unknown"])[1]
    if r["level"] == "local":
        n = int(r["best"])
        return {"state": r["state"], "query": r["q"], "candidates": LEVELS, "target": [float(i == n) for i in range(len(LEVELS))],
                "p_null": 0.0, "task": "airdef_local", "label": n,
                "meta": {"qtype": "score", "family": "airdef_local", "scenario": r["scenario"], "cls": cls, "best": f"fire{n}", "margin": 0.0}}
    if r["level"] == "battery":   # which battery: soft target over the planner's per-battery costs, like the global call
        opts = list(r["opts"])
        rng.shuffle(opts)
        cmin = min(c for _, c in opts)
        tau = 0.15 * max(cmin, 0) + 0.02
        w = [math.exp(-(c - cmin) / tau) for _, c in opts]
        z = sum(w)
        return {"state": r["state"], "query": r["q"], "candidates": [lab for lab, _ in opts], "target": [x / z for x in w], "p_null": 0.0,
                "task": "airdef_battery", "label": max(range(len(opts)), key=lambda i: w[i]),
                "meta": {"qtype": "choice", "family": "airdef_battery", "scenario": r["scenario"], "cls": cls, "best": r["best"],
                         "margin": sorted(w, reverse=True)[1] / z}}
    opts = list(r["opts"])
    rng.shuffle(opts)
    cmin = min(c for _, c in opts)
    tau = 0.15 * max(cmin, 0) + 0.02   # $M: options within ~tau of the best keep real probability mass
    w = [math.exp(-(c - cmin) / tau) for _, c in opts]
    z = sum(w)
    target = [x / z for x in w]
    best = max(range(len(opts)), key=lambda i: target[i])
    return {"state": r["state"], "query": r["q"], "candidates": [lab for lab, _ in opts], "target": target, "p_null": 0.0,
            "task": "airdef_global", "label": best,
            "meta": {"qtype": "choice", "family": "airdef_global", "scenario": r["scenario"], "cls": cls, "best": r["best"],
                     "margin": sorted(w, reverse=True)[1] / z if len(w) > 1 else 0.0}}


def collect(seeds, explore, procs=10):
    jobs = [(sc, s, explore) for s in seeds for sc in SCENARIOS]
    jobs += [(f"level2_{c}", s * 100 + k, explore) for s in seeds for c in "abcd" for k in range(LEVEL_BOOST // 4)]
    with Pool(procs) as p:
        recs = [r for out in p.imap_unordered(raid, jobs) for r in out]
    seen, out = set(), []
    for r in recs:   # one row per distinct (question, state, options): the live demo's memo works the same way
        k = (r["q"], r["state"], str(sorted(lab for lab, _ in r.get("opts", []))))
        if k not in seen and (r["level"] == "local" or len(r["opts"]) > 1):
            seen.add(k)
            out.append(r)
    return out


def cell(r):
    b = r["meta"]["best"]
    if r["meta"]["family"] == "airdef_local":
        return ("local", b, r["meta"]["cls"])
    if r["meta"]["family"] == "airdef_battery":   # the levels' battery choices (floor, plans, border) get half the quota
        return ("battery", r["meta"]["scenario"].startswith("level"))
    a = b.split(":")[0]
    if r["meta"]["scenario"].startswith("level"):   # the levels' own quota: their trade-offs never drown in raid drones
        return ("level", a)
    return ("global", a, r["meta"]["cls"])


def balance(rows, n, rng):
    """Cap each (level, action, threat class) cell so rare-but-important calls (fire counts, aircraft, let it fall)
    get real weight next to the frequent ones."""
    rng.shuffle(rows)
    cells = Counter(cell(r) for r in rows)
    cap = max(1, n // max(len(cells), 1))
    while sum(min(k, cap) for k in cells.values()) < min(n, len(rows)) and cap < n:
        cap += max(1, cap // 10)
    used, out = Counter(), []
    for r in rows:
        c = cell(r)
        if used[c] < cap and len(out) < n:
            used[c] += 1
            out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".context/typically/jobs/airdef_v5"))
    ap.add_argument("--global_rows", type=int, default=24000)
    ap.add_argument("--local_rows", type=int, default=12000)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    data = {}
    for split, seeds in SPLITS.items():
        recs = collect(seeds, explore=0.15 if split == "train" else 0.0)
        data[split] = [to_row(r, rng) for r in recs]
        print(f"{split}: {len(recs)} distinct decision points from {len(seeds) * len(SCENARIOS)} raids", flush=True)
    with (out / "pool_train.jsonl").open("w") as f:   # the full unbalanced pool: reweight later without regenerating
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data["train"])
    G = lambda rows: [r for r in rows if r["meta"]["family"] == "airdef_global"]
    Lo = lambda rows: [r for r in rows if r["meta"]["family"] == "airdef_local"]
    Ba = lambda rows: [r for r in rows if r["meta"]["family"] == "airdef_battery"]
    train = (balance(G(data["train"]), a.global_rows, rng) + balance(Lo(data["train"]), a.local_rows, rng)
             + balance(Ba(data["train"]), BATTERY_ROWS, rng))
    val = balance(G(data["val"]), 1000, rng) + balance(Lo(data["val"]), 500, rng) + balance(Ba(data["val"]), 300, rng)
    rng.shuffle(train)
    ev = data["eval"]
    rng.shuffle(ev)
    files = {"train": train, "val": val,
             "eval/airdef_global": G(ev)[:600],
             "eval/airdef_global_commits": [r for r in G(ev) if r["meta"]["best"] not in ("wait", "hold")][:500],
             "eval/airdef_local": Lo(ev)[:600],
             "eval/airdef_battery": Ba(ev)[:600]}
    drill = ROOT / ".context/airdef/situations.jsonl"
    if drill.exists():
        files["eval/airdef_situations"] = [json.loads(x) for x in drill.read_text().splitlines()]
    for name, rows in files.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        best = Counter(r["meta"]["best"].split(":")[0] for r in rows)
        print(f"{name:32s} {len(rows):6d}  {dict(best.most_common(8))}", flush=True)
