#!/usr/bin/env python3
"""zoom-ai/contested-fast-decisions (EVAL ONLY, never train on it) against a TypeSafe-compatible /v1/systemone server.

README protocol: one `choice` question per single-label decision; one yes/no `noul` question per label for
multi-label decisions (label predicted iff noul > .5); only the task name and candidate labels, no descriptions.
Exact set match vs `true_label`; accuracy over all 818 decisions, single-label and multi-label.
One request per text carries all of its questions (as WorkflowEvals' `questions all`).
  python scripts/contested_eval.py --data train.jsonl --endpoint http://127.0.0.1:8090 --model <id> --out x.jsonl
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def questions(row):
    qs = {}
    for i, d in enumerate(row["output"]["classifications"]):
        if d["multi_label"]:
            for j, label in enumerate(d["labels"]):
                qs[f"d{i}_l{j}"] = {"type": "noul", "instructions": f"{d['task']}: {label}"}
        else:
            qs[f"d{i}"] = {"type": "choice", "instructions": d["task"], "criteria": {l: None for l in d["labels"]}}
    return qs


def predict(row, answers):
    out = []
    for i, d in enumerate(row["output"]["classifications"]):
        if d["multi_label"]:
            pred = [l for j, l in enumerate(d["labels"]) if answers[f"d{i}_l{j}"]["noul"] > .5]
        else:
            pred = [answers[f"d{i}"]["choice"]]
        out.append({"id": row["id"], "task": d["task"], "multi_label": d["multi_label"], "pred": pred,
                    "gold": d["true_label"], "correct": set(pred) == set(d["true_label"])})
    return out


def ask(endpoint, model, row, timeout):
    body = json.dumps({"state": row["input"], "model": model, "questions": questions(row)}).encode()
    req = urllib.request.Request(f"{endpoint}/v1/systemone", body, {"Content-Type": "application/json",
                                                                    "Authorization": "Bearer local"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return predict(row, json.load(r)["answers"])
        except Exception as e:  # noqa: BLE001 -- retried, then counted wrong
            err = repr(e)
    return [{"id": row["id"], "task": d["task"], "multi_label": d["multi_label"], "pred": None,
             "gold": d["true_label"], "correct": False, "error": err} for d in row["output"]["classifications"]]


def summary(rows):
    acc = lambda rs: sum(r["correct"] for r in rs) / len(rs) if rs else None  # noqa: E731
    return {"accuracy": acc(rows), "single_label": acc([r for r in rows if not r["multi_label"]]),
            "multi_label": acc([r for r in rows if r["multi_label"]]), "n": len(rows),
            "errors": sum("error" in r for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    data = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    with ThreadPoolExecutor(a.workers) as pool:
        rows = [r for rs in pool.map(lambda row: ask(a.endpoint, a.model, row, a.timeout), data) for r in rs]
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = summary(rows)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps({"model": a.model, **s}, indent=1) + "\n")
    print(json.dumps(s))


if __name__ == "__main__":
    main()
