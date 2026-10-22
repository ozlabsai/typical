#!/usr/bin/env python3
"""R056 step 5: per-question temperatures fitted on the dev split's teacher probabilities (never benchmark data).

For every WorkflowEvals question id with >= --min-n dev rows, T minimises the cross-entropy between the teacher
distribution and softmax(log p_student / T) over a log grid; ids with fewer rows keep T=1. Only question ids are
keyed (no type-level fallback), so JevBench/Hebrew questions are served untouched.
  python3 scripts/r056_calibrate.py devsel/step-N.json temperatures.json
R062: with a dev .jsonl as last argument, temperatures are keyed `type:<choice|noul|score>` instead (the server's
per-type fallback), for question ids that never repeat (Hebrew/general dev).
  python3 scripts/r056_calibrate.py devsel/step-N.json temperatures.json dev_he.jsonl
"""
import json
import math
import sys
from collections import defaultdict

GRID = [round(math.exp(math.log(0.25) + i * (math.log(6) - math.log(0.25)) / 120), 4) for i in range(121)]


def ce(rows, t):
    total = 0.0
    for r in rows:
        z = [math.log(max(p, 1e-12)) / t for p in r["student"]]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        total -= sum(q * (x - lse) for q, x in zip(r["teacher"], z))
    return total / len(rows)


def main(src, dst, min_n=30, dev=None):
    rows = json.load(open(src))["rows"]
    types = {f"{r['id']}/{q}": f"type:{v['type']}" for r in map(json.loads, open(dev)) for q, v in r["questions"].items()} if dev else {}
    by = defaultdict(list)
    for r in rows:
        by[types.get(r["id"]) or r["id"].rsplit("/", 1)[1]].append(r)
    temps, report = {}, {}
    for qid, rs in sorted(by.items()):
        if len(rs) < min_n:
            continue
        best = min(GRID, key=lambda t: ce(rs, t))
        temps[qid] = best
        report[qid] = {"n": len(rs), "T": best, "ce_T1": round(ce(rs, 1.0), 4), "ce_T": round(ce(rs, best), 4)}
    json.dump(temps, open(dst, "w"), indent=1, sort_keys=True)
    allrows = [r for rs in by.values() for r in rs]
    before = ce(allrows, 1.0)
    after = sum(ce(rs, temps.get(q, 1.0)) * len(rs) for q, rs in by.items()) / len(allrows)
    print(json.dumps({"questions_fitted": len(temps), "dev_ce_T1": round(before, 4), "dev_ce_fitted": round(after, 4),
                      "per_question": report}, indent=1))


if __name__ == "__main__":
    assert ce([{"student": [0.9, 0.1], "teacher": [0.9, 0.1]}], 1.0) < ce([{"student": [0.9, 0.1], "teacher": [0.9, 0.1]}], 2.0)
    extra = sys.argv[3:]
    main(sys.argv[1], sys.argv[2], *(int(x) for x in extra if not x.endswith(".jsonl")),
         **({"dev": extra[-1]} if extra and extra[-1].endswith(".jsonl") else {}))
