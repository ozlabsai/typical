#!/usr/bin/env python3
"""R056: Hebrew twins of generated workflow node requests (R055 D1 approach, at node level).

For ~25% of generated cases (sha256(case_id) % 4 == 0; security, agent_trace, customer; invoice packets are
too long), every node request {state, questions} gets a Hebrew twin: state leaves, question instructions and
option texts are translated by claude-haiku-5-5 (R055 D1 translator prompt), option IDs / digits are checked
deterministically (one fix retry). Twins inherit the English node's teacher targets at mix time
(meta.twin_id = English request_id). Every CLI response is cached by sha256 under .context/r056/twins/cache.

  uv run --project .context/research_sources/WorkflowEvals python scripts/r056_translate_twins.py --workers 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hejev_build as hj  # noqa: E402  (walker, digit/ID QA, translator prompt, mirror schema)
import r053_teacher_label as r053  # noqa: E402  (cached `claude -p` transport + $ ledger)
import r055_translate_pool as tp  # noqa: E402  (D1 translator system prompt)

ROOT = hj.ROOT
GEN = ROOT / ".context/r056/gen"
OUT = ROOT / ".context/r056/twins"
r053.CACHE = OUT / "cache"
TRANSLATOR = "claude-haiku-5-5"
WORKFLOWS = ("security_incidents", "agent_trace_observability", "customer_service")


def twin_case(case_id: str) -> bool:
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 4 == 0


def rewrite(node: dict, fn):
    """hj.rewrite for a multi-question node: state leaves (multi-word only), each question's instructions and
    option texts; never option IDs, dict keys or question ids."""
    n = 0

    def leaf(path, s):
        nonlocal n
        if not re.search(r"[A-Za-z]{2,}", s) or (path.startswith("state") and not re.search(r"\s", s.strip())):
            return s
        key = f"{n:03d}_" + re.sub(r"[^A-Za-z0-9]+", "_", path)[:40].strip("_")
        n += 1
        return fn(key, s)

    state = hj.walk(node["state"], leaf, "state")
    questions = {qid: {**q, "instructions": hj.walk(q.get("instructions"), leaf, f"instructions_{qid}"),
                       "criteria": hj.walk(q.get("criteria"), leaf, f"criteria_{qid}")}
                 for qid, q in node["questions"].items()}
    return state, questions


def labels(node: dict) -> list[str]:
    return sorted({k for q in node["questions"].values() if isinstance(q.get("criteria"), dict) for k in q["criteria"]})


def translate_node(node: dict) -> dict:
    src = {}
    rewrite(node, lambda k, s: src.setdefault(k, s))
    prev = None
    for attempt in range(2):
        user = (f"Workflow node {node['node']} of case {node['case_id']}.\n"
                f"Option IDs (machine labels, keep verbatim wherever they appear): {json.dumps(labels(node))}\n"
                "Translate every value of this JSON object into Hebrew; return exactly the same keys.\n\n"
                + json.dumps(src, ensure_ascii=False, indent=1))
        if prev:
            user += ("\n\nA checker rejected a previous Hebrew translation of this item:\n"
                     + "\n".join(f"- {i}" for i in prev[1]) + "\n\nPrevious translation:\n"
                     + json.dumps(prev[0], ensure_ascii=False, indent=1)
                     + "\n\nReturn a corrected full translation that fixes every issue.")
        raw = r053.claude(TRANSLATOR, tp.TRANSLATE_SYSTEM, user, effort="low", schema=hj.mirror_schema(src))
        he = raw.get("structured_output") or {}
        issues = hj.det_issues(src, he, labels(node)) + [f"{k}: missing" for k in src if not he.get(k)]
        if not issues:
            state, questions = rewrite(node, lambda k, s: he[k])
            return {**node, "request_id": f"{node['request_id']}-he", "state": state, "questions": questions,
                    "twin_id": node["request_id"], "lang": "he", "translator": TRANSLATOR, "attempt": attempt}
        prev = (he, issues)
    return {"request_id": node["request_id"], "status": "det_fail", "issues": issues[:5]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-usd", type=float, default=60.0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    r053.LEDGER.update(usd=hj.spent(), max_usd=args.max_usd)
    nodes = [n for d in sorted(ROOT.glob(".context/r056/gen*/")) for wf in WORKFLOWS if (d / wf / "nodes.jsonl").exists()
             for n in map(json.loads, (d / wf / "nodes.jsonl").open()) if twin_case(n["case_id"])]
    for n in nodes:
        n["workflow"] = next(wf for wf in WORKFLOWS if f"-{wf.split('_')[0]}-" in n["case_id"])

    def job(node):
        try:
            return translate_node(node)
        except Exception as exc:  # budget / transport: recorded, node simply has no twin
            return {"request_id": node["request_id"], "status": f"error:{type(exc).__name__}", "issues": [str(exc)[:200]]}

    stats = Counter()
    with ThreadPoolExecutor(args.workers) as ex, (OUT / "twins.jsonl.tmp").open("w") as sink:
        for i, res in enumerate(ex.map(job, nodes), 1):
            stats[res.get("status", "ok")] += 1
            if "twin_id" in res:
                sink.write(json.dumps(res, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == len(nodes):
                print(f"{i}/{len(nodes)} {dict(stats)} spent ${r053.LEDGER['usd']:.2f}", flush=True)
    (OUT / "twins.jsonl.tmp").replace(OUT / "twins.jsonl")
    (OUT / "summary.json").write_text(json.dumps({"nodes": len(nodes), "status": dict(stats),
                                                  "notional_usd": round(hj.spent(), 2)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
