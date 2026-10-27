"""Gate: whole raids with Typical in the loop vs the planner (its teacher) and fire-at-everything.

    uv run --no-sync python airdef/harness.py [scenario ...] [--model path/to/best.pt] [--seeds 1 2]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
from engine import run_headless   # noqa: E402
from sim import SCENARIOS, money   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
ap.add_argument("--model", default="OzLabs/typical-small")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
a = ap.parse_args()

import scorer   # noqa: E402
m = scorer.load(a.model)
grand = {}
for sc in a.scenarios:
    for seed in a.seeds:
        t0 = time.time()
        w = run_headless(sc, model=m, seed=seed)
        print(f"\n{sc} seed {seed} ({len(w.threats)} threats, {time.time() - t0:.0f}s)")
        for n, x in w.ledger().items():
            g = grand.setdefault(n, [0.0, 0.0, 0])
            g[0] += x["total"]; g[1] += x["spent"]; g[2] += x["leaks"]
            extra = (f"  agrees w/ planner {x['agree'] / max(x['decisions'], 1):.0%}, escalated {x['escalated']}, "
                     f"{x['ms'] / max(x['calls'], 1):.0f} ms/call") if n == "typical" else ""
            print(f"  {n:8s} total {x['total']:9.0f}  spent {money(x['spent']):>9s}  hits {x['leaks']:3d}  kills {x['kills']:3d}{extra}")
print("\nall runs:", {n: dict(total=round(g[0]), spent=money(g[1]), hits=g[2]) for n, g in grand.items()})
