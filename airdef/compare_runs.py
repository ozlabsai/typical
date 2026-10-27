"""Compare two Typical checkpoints on whole raids from a remote_harness run, seed by seed.

    uv run --no-sync python airdef/compare_runs.py .context/airdef/remote/<run id>/out [more out dirs ...] [--a v3] [--b v4]

Both checkpoints play the same raid on each seed, so the paired difference b - a per seed cancels most of the
raid-to-raid variance. Reported per scenario: mean total cost and hits for doctrine, planner, a and b, the mean paired
difference in total cost with its standard error, and on how many seeds b beat a.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="+")
ap.add_argument("--a", default="v3")
ap.add_argument("--b", default="v4")
a = ap.parse_args()

runs = defaultdict(dict)   # (scenario, seed) -> {who: ledger row}
for f in (f for d in a.out for f in Path(d).glob("*.json")):
    name = f.stem.split("__")[0]
    for r in json.loads(f.read_text()):
        L = r["ledger"]
        runs[(r["scenario"], r["seed"])][name] = L["typical"]
        runs[(r["scenario"], r["seed"])].update(doctrine=L["doctrine"], planner=L["planner"])   # same raid for both models

print(f"{'scenario':20s} {'n':>2s}  {'doctrine':>14s} {'planner':>14s} {a.a:>14s} {a.b:>14s}   {a.b}-{a.a} total (se)   {a.b} better")
for sc in sorted({k[0] for k in runs}):
    keys = sorted(k for k in runs if k[0] == sc and a.a in runs[k] and a.b in runs[k])
    if not keys:
        continue
    col = lambda who: f"{st.mean(runs[k][who]['total'] for k in keys):8,.0f} · {st.mean(runs[k][who]['leaks'] for k in keys):4.1f}"
    d = [runs[k][a.b]["total"] - runs[k][a.a]["total"] for k in keys]
    se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
    print(f"{sc:20s} {len(keys):2d}  {col('doctrine'):>14s} {col('planner'):>14s} {col(a.a):>14s} {col(a.b):>14s}   "
          f"{st.mean(d):+9,.0f} ({se:,.0f})   {sum(x < 0 for x in d)}/{len(d)}")
