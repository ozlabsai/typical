#!/usr/bin/env python3
"""Score Hebrew JevBench v0 runs of the stock jevbench runner (`python -m jevbench.cli run`).

Per split: n, coverage, accuracy, valid rate, Brier, ECE (jevbench.metrics conventions; invalid = wrong,
excluded from Brier/ECE). Paired on twins (he-<id> / mx-<id> vs <id>): HE-EN accuracy gap, prediction
agreement, mean Jensen-Shannon divergence (log2, in [0,1]) between EN and HE probability vectors.

Results args are result .jsonl files or run directories (every *.jsonl with `task_id` rows is read; on duplicate
task ids an ok row wins, so resumed/sharded runs merge).
  uv run python scripts/hejev_score.py --en .context/g0-cygnet/jevbench.jsonl --he <he run> --mx <mx run>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JB = ROOT / ".context" / "jevbench-v154"
OUT = ROOT / ".context" / "hebrew-bench" / "hejev-v0"
sys.path.insert(0, str(JB))
from jevbench.metrics import brier_score, ece_top_label  # noqa: E402

TASKS = {"en": [JB / "datasets" / "public" / f"{t}.jsonl" for t in ("original", "easy", "hard")],
         "he": [OUT / "translated.jsonl"], "mx": [OUT / "mixed.jsonl"], "native": [OUT / "native.jsonl"]}
TWINS = {"he": "he-", "mx": "mx-"}


def load_tasks(paths) -> dict:
    return {t["id"]: t for p in paths for t in map(json.loads, Path(p).open())}


def load_results(paths) -> dict:
    rows = {}
    for p in map(Path, paths):
        for f in sorted(p.glob("*.jsonl")) if p.is_dir() else [p]:
            for line in f.open():
                r = json.loads(line)
                if "task_id" not in r:
                    break  # tasks / ledger / manifest files inside a run dir
                if not (rows.get(r["task_id"], {}).get("ok") and not r.get("ok")):
                    rows[r["task_id"]] = r
    return rows


def jsd(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}
    kl = lambda a: sum(a[k] * math.log2(a[k] / m[k]) for k in keys if a.get(k, 0) > 0)  # noqa: E731
    return max(0.0, (kl(p) + kl(q)) / 2)


def split_metrics(tasks: dict, rows: dict) -> dict:
    hit = [(tasks[i], rows[i]) for i in tasks if i in rows]
    scored = [(t, r) for t, r in hit if r.get("valid") and r.get("probs")]
    briers = [brier_score(r["probs"], str(t["expected"]), t["labels"]) for t, r in scored]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"n_tasks": len(tasks), "n_results": len(hit), "coverage": len(hit) / len(tasks) if tasks else None,
            "accuracy": mean([bool(r.get("correct")) for _, r in hit]),
            "valid_rate": mean([bool(r.get("valid")) for _, r in hit]),
            "brier": mean(briers),
            "ece": ece_top_label([(max(r["probs"].values()), bool(r.get("correct"))) for _, r in scored])["ece"]
            if scored else None}


def paired(prefix: str, en: dict, other: dict) -> dict:
    pairs = [(en[i[len(prefix):]], r) for i, r in other.items() if i.startswith(prefix) and i[len(prefix):] in en]
    both = [(a, b) for a, b in pairs if a.get("valid") and b.get("valid")]
    probs = [(a, b) for a, b in both if a.get("probs") and b.get("probs")]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    acc_en, acc_x = mean([bool(a.get("correct")) for a, _ in pairs]), mean([bool(b.get("correct")) for _, b in pairs])
    return {"n_pairs": len(pairs), "acc_en": acc_en, "acc_other": acc_x,
            "acc_gap_other_minus_en": None if acc_en is None else acc_x - acc_en,
            "flips_en_right_other_wrong": sum(bool(a.get("correct")) and not b.get("correct") for a, b in pairs),
            "flips_en_wrong_other_right": sum(not a.get("correct") and bool(b.get("correct")) for a, b in pairs),
            "prediction_agreement": mean([a["predicted"] == b["predicted"] for a, b in both]),
            "mean_jsd": mean([jsd(a["probs"], b["probs"]) for a, b in probs]), "n_jsd": len(probs)}


def score(results: dict, tasks: dict | None = None) -> dict:
    """results: split -> {task_id: row}; tasks: split -> {task_id: task} (defaults to the shipped files)."""
    tasks = tasks or {s: load_tasks(TASKS[s]) for s in results}
    out = {"splits": {s: split_metrics(tasks[s], rows) for s, rows in results.items()}}
    if "en" in results:
        tier = lambda i: i.split("-", 1)[0]  # noqa: E731  original / easy / hard
        out["paired"] = {f"{s}_vs_en": {"all": paired(p, results["en"], results[s]),
                                        **{t: paired(p, results["en"], {i: r for i, r in results[s].items()
                                                                        if tier(i[len(p):]) == t})
                                           for t in ("original", "easy", "hard")}}
                         for s, p in TWINS.items() if s in results}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in TASKS:
        ap.add_argument(f"--{s}", nargs="+", default=[], help=f"results for the {s} split")
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args(argv)
    results = {s: load_results(getattr(args, s)) for s in TASKS if getattr(args, s)}
    report = score(results)
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    for s, m in report["splits"].items():
        print(f"{s:7s} " + "  ".join(f"{k}={fmt(v)}" for k, v in m.items()))
    for name, by_tier in report.get("paired", {}).items():
        for t, m in by_tier.items():
            print(f"{name} [{t}] " + "  ".join(f"{k}={fmt(v)}" for k, v in m.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
