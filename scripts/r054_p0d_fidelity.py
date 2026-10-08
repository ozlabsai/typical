"""R054 P0d: does our Claude-CLI teacher reproduce the PUBLISHED claude-fable-5-1 labels?

Samples ~150 WorkflowEvals benchmark questions (equal per workflow, equal per question kind within a
workflow, deterministic hash order, only rows whose published anthropic label is fable-5-1, not the
opus fallback), asks claude-fable-5-1 --effort high one question per call through system_one_adapter's
structured probability path (same system prompt as the gold run; schema via `claude --json-schema`),
and compares to the published labels. Protocol-fidelity diagnostic only: never train on these items.

uv run --project .context/research_sources/WorkflowEvals python scripts/r054_p0d_fidelity.py
"""
import argparse
import glob
import hashlib
import json
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / ".context" / "research_sources" / "WorkflowEvals")]
import r053_teacher_label as t  # noqa: E402

HF = {"security_incidents": "security-incidents", "agent_trace_observability": "agent-trace-observability",
      "invoice_processing": "invoice-processing", "customer_service": "customer-service"}
OUT = ROOT / ".context" / "r054-phase0" / "p0d"


def sample(n: int) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    per_wf = n // len(HF) + 1
    for wf, slug in HF.items():
        f = glob.glob(str(Path.home() / f".cache/huggingface/hub/datasets--typesafe--evalsafe-{slug}/snapshots/*/data/questions.parquet"))[0]
        cells = defaultdict(list)
        for r in pq.read_table(f).to_pylist():
            a = r["anthropic"]
            if a and a["status"] == "answered" and a["model"]["name"] == "claude-fable-5-1":
                cells[r["kind"]].append({**r, "workflow": wf})
        for i, kind in enumerate(sorted(cells)):
            cells[kind].sort(key=lambda r: hashlib.sha256(r["question_instance_id"].encode()).hexdigest())
            take = per_wf // len(cells) + (i < per_wf % len(cells))
            rows += cells[kind][:take]
    return rows[:n]


def published(row) -> dict[str, float]:
    return {p["option"]: p["probability"] for p in row["anthropic"]["probabilities"]}


def compare(ours: dict, pub: dict, kind: str) -> dict:
    if kind == "noul":
        return {"dp": abs(ours["true"] - pub["true"]), "flip": (ours["true"] > .5) != (pub["true"] > .5)}
    return {"dp": max(abs(ours[k] - pub.get(k, 0.0)) for k in ours), "flip": max(ours, key=ours.get) != max(pub, key=pub.get)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-usd", type=float, default=40.0)
    args = ap.parse_args()
    from system_one_adapter import SystemOneAdapterClient

    t.LEDGER["max_usd"] = args.max_usd
    t.CACHE = OUT / "cache"
    client = SystemOneAdapterClient(structured_outputs=True, llm_answer_mode="probabilities",
                                    normalize_probabilities=True, n_retry_malformed_structure=1)
    rows = sample(args.n)

    def one(row):
        q = json.loads(row["question_json"])
        try:
            probs = t.ask(client, "cli", "claude-fable-5-1", json.loads(row["state_json"]), {row["question_id"]: q})
            return row, probs[row["question_id"]], None
        except Exception as exc:
            return row, None, f"{type(exc).__name__}: {exc}"[:300]

    out, errors = [], []
    with ThreadPoolExecutor(args.workers) as pool:
        for row, ours, err in pool.map(one, rows):
            if err:
                errors.append({"qid": row["question_instance_id"], "error": err})
                continue
            pub = published(row)
            out.append({"workflow": row["workflow"], "kind": row["kind"], "question_instance_id": row["question_instance_id"],
                        "ours": ours, "published": pub, **compare(ours, pub, row["kind"])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    stats = t.STATS["claude-fable-5-1"]
    by = defaultdict(list)
    for r in out:
        by[(r["workflow"], r["kind"])].append(r)
        by[("all", r["kind"])].append(r)
        by[("all", "all")].append(r)
    noul = [r["dp"] for r in out if r["kind"] == "noul"]
    summary = {
        "n_answered": len(out), "n_errors": len(errors), "errors": errors,
        "noul_mean_abs_dp": st.mean(noul) if noul else None,
        "mean_abs_dp_all": st.mean(r["dp"] for r in out),
        "argmax_flip_rate": st.mean(r["flip"] for r in out),
        "cells": {f"{w}/{k}": {"n": len(v), "mean_dp": round(st.mean(r["dp"] for r in v), 4),
                               "flip": round(st.mean(r["flip"] for r in v), 4)} for (w, k), v in sorted(by.items())},
        "reasoning_tokens_per_call": stats["reasoning_tokens"] / max(stats["calls"], 1),
        "output_tokens_per_call": stats["output_tokens"] / max(stats["calls"], 1),
        "notional_usd": round(stats["usd"], 2), "new_calls": t.LEDGER["calls"], "cached_calls": t.LEDGER["cached"],
        "gate": "mean noul |dp| <= .06 and flips <= 5%",
    }
    summary["gate_pass"] = bool(noul) and summary["noul_mean_abs_dp"] <= .06 and summary["argmax_flip_rate"] <= .05
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cells", "errors")}, indent=1))


if __name__ == "__main__":
    main()
