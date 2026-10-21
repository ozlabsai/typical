#!/usr/bin/env python3
"""R055 readout check (Cygnet letter readout vs native readout), selection on the Hebrew dev split only.

  dev   <d1 dev.jsonl> <out.jsonl>          HE+MX rows of the D1 dev split as jevbench tasks
  fit   <dev tasks> <T=1 run dir>...        Cygnet calibration temperature by dev NLL (grid and rule of
                                            cygnet-recipe calibration/fit.py), one T per run dir
  score <tasks.jsonl> <run dir>...          accuracy / NLL / Brier / ECE of each run on a task file
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hejev_score import load_results, load_tasks, split_metrics  # noqa: E402

GRID = [round(0.5 + 0.1 * i, 2) for i in range(46)]  # as cygnet-recipe calibration/fit.py
NOUL = {"yes": "yes", "no": "no", "true": "yes", "false": "no"}


def to_task(r):
    (q,) = r["questions"].values()
    gold = r["targets"]["decision"]["label"]
    if q["type"] == "noul":  # jevbench noul: criteria keyed true/false, labels no/yes
        crit = {("true" if NOUL[k] == "yes" else "false"): v for k, v in q["criteria"].items()}
        labels, expected = ["no", "yes"], NOUL[str(gold).lower()]
    elif q["type"] == "score":
        crit, labels, expected = q["criteria"], [str(i) for i in range(len(q["criteria"]))], int(gold)
    else:
        crit, labels, expected = q["criteria"], list(q["criteria"]), gold
    assert str(expected) in labels, r["id"]
    return {"id": r["id"], "family": r["meta"].get("family") or r["meta"]["d1_source"], "state": r["state"],
            "question": {"type": q["type"], "instructions": q["instructions"], "criteria": crit},
            "labels": labels, "expected": expected, "split": "public", "group": r["meta"]["twin_id"],
            "provenance": {"lang": r["meta"]["lang"], "source": "D1 dev (training distribution, not a test set)"}}


def temper(p, t):
    z = {k: max(v, 1e-12) ** (1 / t) for k, v in p.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def nll(tasks, rows, t):
    pairs = [(tasks[i], r["probs"]) for i, r in rows.items() if i in tasks and r.get("valid") and r.get("probs")]
    return sum(-math.log(max(temper(p, t)[str(task["expected"])], 1e-12)) for task, p in pairs) / len(pairs), len(pairs)


def main(cmd, *args):
    if cmd == "dev":
        rows = [json.loads(l) for l in open(args[0])]
        out = [to_task(r) for r in rows if r["meta"]["lang"] in ("he", "mx")]
        Path(args[1]).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in out))
        print(f"{len(out)} dev tasks")
    elif cmd == "fit":
        tasks = load_tasks([args[0]])
        for d in args[1:]:
            rows = load_results([d])
            best = min(GRID, key=lambda t: nll(tasks, rows, t)[0])
            (n1, n), (nb, _) = nll(tasks, rows, 1.0), nll(tasks, rows, best)
            print(json.dumps({"run": d, "T": best, "n": n, "nll_T1": round(n1, 4), "nll_T": round(nb, 4)}))
    elif cmd == "score":
        tasks = load_tasks([args[0]])
        for d in args[1:]:
            rows = load_results([d])
            m = split_metrics(tasks, rows)
            m["nll"] = nll(tasks, rows, 1.0)[0]
            print(json.dumps({"run": d, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}))


if __name__ == "__main__":
    main(*sys.argv[1:])
