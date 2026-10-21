#!/usr/bin/env python3
"""R056 data: label inputs for the thinking teachers, then the training mix.

  labelin  generated nodes (.context/r056/gen/<wf>/nodes.jsonl) -> .context/r056/label_in/{gemma,qwen}.jsonl
           gemma: agent_trace, customer, invoice (security never uses Gemma targets);
           qwen:  security first, then agent_trace (coordinator: Qwen capacity to security, then trace).
  mix      see cmd_mix.
  python3 scripts/r056_build.py labelin
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R6 = ROOT / ".context/r056"
GEN = R6 / "gen"
WFS = ("security_incidents", "agent_trace_observability", "customer_service", "invoice_processing")


def jsonl(path):
    return [json.loads(line) for line in Path(path).open()] if Path(path).exists() else []


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {"rows": len(rows), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def is_dev(case_id: str) -> bool:
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 10 == 0


def nodes(wf):
    return [{"id": n["request_id"], "state": n["state"], "questions": n["questions"], "case_id": n["case_id"]}
            for n in jsonl(GEN / wf / "nodes.jsonl")]


def cmd_labelin(args):
    sec, trace, cust, inv = (nodes(wf) for wf in WFS)
    out = {"gemma": write(R6 / "label_in/gemma.jsonl", trace + cust + inv),
           "qwen": write(R6 / "label_in/qwen.jsonl", sec + trace)}
    q = lambda rows: sum(len(r["questions"]) for r in rows)  # noqa: E731
    print(json.dumps({**out, "questions": {"security": q(sec), "agent_trace": q(trace), "customer": q(cust),
                                           "invoice": q(inv)}}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("labelin", "mix"))
    args = ap.parse_args(argv)
    {"labelin": cmd_labelin, "mix": lambda a: cmd_mix(a)}[args.cmd](args)


if __name__ == "__main__":
    main()
