#!/usr/bin/env python3
"""R053 stage 3: teacher-labeled node questions -> singleton trainer rows for
scripts/train_gemma_decision_lora.py, split 80/20 by case_id, plus a manifest with pilot stats.

State and option texts are rendered exactly as scripts/serve_typesafe_gemma_lora.py would render the
same node (asserted per row), so training matches serving.

uv run python scripts/r053_build_train_rows.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gemma_decision_runtime import (  # noqa: E402
    _blank,
    _structured,
    render_decision_prompt,
    render_typesafe_decision_prompt,
)

DATA = ROOT / ".context" / "r053-data"
WORKFLOWS = ("security_incidents", "customer_service", "agent_trace_observability", "invoice_processing")
MID_BAND = (0.15, 0.85)


def server_items(question: dict) -> list[tuple[str, object]]:
    """(label, raw criterion) in the server's order (serve_typesafe_gemma_lora._items)."""
    criteria = question.get("criteria")
    if question["type"] == "noul":
        criteria = criteria or {}
        labels = [*criteria, *(k for k in ("true", "false") if k not in criteria)]
        return [(label, criteria.get(label)) for label in labels]
    if question["type"] == "choice":
        return list(criteria.items())
    return [(str(i), value) for i, value in enumerate(criteria)]


def option_text(kind: str, label: str, value) -> str:
    """Option text exactly as render_typesafe_decision_prompt writes it."""
    if kind == "choice":
        return label if _blank(value) else f"{label}: {_structured(value)}"
    if kind == "score":
        return f"Level {label}" if _blank(value) else _structured(value)
    semantic = {"true": "Yes", "false": "No"}[label]
    return semantic if _blank(value) else f"{semantic}: {_structured(value)}"


def split_of(case_id: str) -> str:
    return "heldout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 5 == 0 else "train"


def train_row(wf: str, label: dict, state) -> dict:
    question, qid = label["question"], label["qid"]
    raw = server_items(question)
    texts = [(lab, option_text(question["type"], lab, value)) for lab, value in raw]
    rendered_state = json.dumps(state, ensure_ascii=False, indent=1)
    instructions = question.get("instructions") or ""
    if render_decision_prompt(rendered_state, instructions, texts) != render_typesafe_decision_prompt(
            state, question.get("instructions"), raw, question_type=question["type"]):
        raise ValueError(f"{label['request_id']}/{qid}: training render differs from serving render")
    soft = [label["consensus"][lab] for lab, _ in raw]
    criteria = [t for _, t in texts] if question["type"] == "score" else dict(texts)
    return {
        "id": f"r053/{wf}/{label['case_id']}/{label['node']}/{label['request_id']}/{qid}",
        "state": rendered_state,
        "questions": {qid: {"type": question["type"], "instructions": instructions, "criteria": criteria}},
        "targets": {qid: {"label": raw[soft.index(max(soft))][0]}},
        "soft_targets": {qid: soft},
        "meta": {"source": "r053", "workflow": wf, "node": label["node"], "case_id": label["case_id"],
                 "request_id": label["request_id"], "split": split_of(label["case_id"]),
                 "teachers": {t: [p[lab] for lab, _ in raw] for t, p in label["teachers"].items()}},
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stats(labels: list[dict], one: list[dict]) -> dict:
    noul = [r for r in labels if r["question"]["type"] == "noul"]
    multi = [r for r in labels if r["question"]["type"] != "noul"]
    teachers = sorted({t for r in labels for t in r["teachers"]})
    out = {"questions": len(labels), "noul": len(noul), "choice_or_score": len(multi)}
    if len(teachers) == 2 and noul:
        a, b = teachers
        out["noul_teacher_mean_abs_diff"] = sum(abs(r["teachers"][a]["true"] - r["teachers"][b]["true"])
                                                for r in noul) / len(noul)
        out["consensus_noul_mid_band_frac"] = sum(MID_BAND[0] <= r["consensus"]["true"] <= MID_BAND[1]
                                                  for r in noul) / len(noul)
        for t in teachers:
            out[f"mid_band_frac[{t}]"] = sum(MID_BAND[0] <= r["teachers"][t]["true"] <= MID_BAND[1]
                                             for r in noul) / len(noul)
    if len(teachers) == 2 and multi:
        top = [{t: max(p, key=p.get) for t, p in r["teachers"].items()} for r in multi]
        out["choice_score_argmax_agreement"] = sum(len(set(t.values())) == 1 for t in top) / len(top)
    if one:  # does grouping a node's questions into one request shift probabilities?
        by_key = {(r["request_id"], r["qid"]): r for r in labels}
        pairs = [(r, by_key[(r["request_id"], r["qid"])]) for r in one if (r["request_id"], r["qid"]) in by_key]
        out["one_vs_all"] = {"n": len(pairs), **{
            f"mean_abs_p_one_minus_p_all[{t}]": sum(abs(o["teachers"][t]["true"] - a["teachers"][t]["true"])
                                                    for o, a in pairs) / max(len(pairs), 1)
            for t in teachers}}
    return out


def main() -> int:
    rows, labels_all, one_all, per_wf = [], [], [], {}
    for wf in WORKFLOWS:
        labels = read_jsonl(DATA / wf / "labels.jsonl")
        if not labels:
            continue
        nodes = {n["request_id"]: n for n in read_jsonl(DATA / wf / "nodes.jsonl")}
        wf_rows = [train_row(wf, label, nodes[label["request_id"]]["state"]) for label in labels]
        rows += wf_rows
        labels_all += labels
        one = read_jsonl(DATA / wf / "labels_one.jsonl")
        one_all += one
        per_wf[wf] = {"cases": len({r["meta"]["case_id"] for r in wf_rows}),
                      "nodes": len({r["meta"]["request_id"] for r in wf_rows}),
                      "rows": dict(Counter(r["meta"]["split"] for r in wf_rows)), **stats(labels, one)}
    manifest = {"workflows": per_wf, "overall": stats(labels_all, one_all), "files": {}}
    for split in ("train", "heldout"):
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["meta"]["split"] == split)
        (DATA / f"{split}.jsonl").write_text(body)
        manifest["files"][f"{split}.jsonl"] = {"rows": body.count("\n"),
                                              "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
