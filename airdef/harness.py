"""Gate: run scenarios headless with Typical in the loop and compare the three policies.

    uv run --no-sync python airdef/harness.py [scenario ...] [--model OzLabs/typical-small] [--seed 7]
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
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--show", type=int, default=0, help="print N Typical decisions that broke doctrine")
a = ap.parse_args()

from typical import Typical   # noqa: E402
m = Typical.from_pretrained(a.model, device="auto", max_states=64)
for sc in a.scenarios:
    t0 = time.time()
    w = run_headless(sc, model=m, seed=a.seed)
    L = w.ledger()
    print(f"\n{sc}  ({len(w.threats)} threats, attacker spent {money(w.attacker_cost)}, {time.time() - t0:.0f}s wall)")
    for n, x in L.items():
        print(f"  {n:9s} spent {money(x['spent']):>9s}  leaks {x['leaks']:3d}  damage {x['damage']:7.0f}  "
              f"let-fall {x['fell_open']:3d}  kills {x['kills']:3d}  total {x['total']:8.0f}")
    t = L["typical"]
    d = max(t["decisions"], 1)
    print(f"  typical: {t['decisions']} decisions ({t['calls']} model calls), follows doctrine {t['follow'] / d:.0%}, "
          f"agrees with optimizer {t['agree'] / d:.0%}, escalated {t['escalated']}, {t['ms'] / max(t['calls'], 1):.0f} ms/call")
    shown = 0
    for tid, s in w.policies["typical"].st.items():
        i = s["info"] or {}
        if shown < a.show and i.get("dist") and i["action"] != i["rule_action"]:
            shown += 1
            print("\n" + i["state"])
            for k, p in sorted(i["dist"].items(), key=lambda kv: -kv[1]):
                print(f"    {p:.2f} {k}{'  <- doctrine' if k == i['rule_action'] else ''}")
            print(f"    {i['p_null']:.2f} (abstain)")
