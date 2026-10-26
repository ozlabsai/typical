"""Score a Typical checkpoint on the airdef eval files the way the live demo uses it.

    uv run --no-sync python airdef/eval_model.py [--model OzLabs/typical-small | path/to/best.pt] [--limit 400]

Per file: accuracy (argmax == doctrine), ∅ on doctrine-silent rows (p_null > 0.5), median top probability,
and how many decisions the demo's default bars (top < 0.42 or ∅ > 0.9) would send to the operator.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
from typical import Typical   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="OzLabs/typical-small")
ap.add_argument("--data", default=str(ROOT / ".context/typically/jobs/airdef/eval"))
ap.add_argument("--limit", type=int, default=400)
ap.add_argument("--out", default="")
a = ap.parse_args()

m = Typical.from_pretrained(a.model, device="auto", max_states=4)
report = {}
for f in sorted(Path(a.data).glob("*.jsonl")):
    rows = [json.loads(x) for x in f.read_text().splitlines()][: a.limit]
    hit = n = sil_ok = sil = esc = 0
    tops, t0 = [], time.time()
    for r in rows:
        d = m.choice(r["state"], r["query"], r["candidates"])
        pn = d.pop("p_null")
        top = max(d, key=d.get)
        esc += d[top] < 0.42 or pn > 0.9
        if r["label"] < 0:
            sil += 1
            sil_ok += pn > 0.5
            continue
        n += 1
        hit += top == r["candidates"][r["label"]]
        tops.append(d[top])
    report[f.stem] = dict(acc=hit / max(n, 1), n=n, silent_abstain=(sil_ok / sil) if sil else None, n_silent=sil,
                          median_top=statistics.median(tops) if tops else None, escalate=esc / len(rows))
    x = report[f.stem]
    silent = f"∅-on-silent {x['silent_abstain']:.0%} (n={sil})" if sil else ""
    print(f"{f.stem:28s} acc {x['acc']:.1%} (n={n})  median top {x['median_top']:.2f}  escalate {x['escalate']:.0%}  "
          f"{silent}  [{time.time() - t0:.0f}s]", flush=True)
if a.out:
    Path(a.out).write_text(json.dumps(report, indent=1))
