"""Stream Jev-shaped HF datasets into our row schema (PLAN6 phase 6 "typed workflow" mix).

uv run scripts/jev_hf_datasets.py                 # full build -> data_wf_hf/{train,eval}/*.jsonl
uv run scripts/jev_hf_datasets.py --limit 20      # smoke: 20 rows per file

Row: {state, query, candidates, target, p_null, task, label, meta{family, qtype, ...}}; train rows also
carry meta.fam_bucket="W" and meta.family=<source>; query is
instructions + criteria rendered by pcdm_jev.decider.query_text (same renderer the JevBench adapter
uses). Every file is leak-checked (normalized exact-match) against the 231 JevBench public states.

train/  cua_s1_forms.jsonl           cua-ai/cua-s1-forms train, capped (--cap_train) with skip <= 35%, MIT
        jev4b_distill.jsonl          MagaBitmex/jev-4b-distill-data data3/train, PROGRAMMATIC gold (teacher dist kept in meta.teacher)
        systemone_lite.jsonl         dwidlee/systemone-lite-general train, capped (--cap_lite), letter aliases removed
val/    cua_s1_forms.jsonl           its validation split, capped (--cap_val)
eval/   cua_s1_forms_test.jsonl      form-signature-disjoint synthetic test
        jev4b_adversarial.jsonl      jev-4b data_hard/eval (120 states: injection, distractors, filler)
        systemone_lite_hard.jsonl    systemone-lite test_hard (layout / paraphrase / option-subset shift)
        mind2web_choice.jsonl        AndeyTait/JevForge-Mind2Web test (website-disjoint): choice over elements + noul, CC-BY-4.0
        cua_s1_forms_demo.jsonl      196 real rows (3 real forms x 3 real PDFs)
        typed_decisions_test.jsonl   LocalLLaMA/typed-decisions test (400 cases x 5 q), soft gold
        typed_decisions_train.jsonl  its train split (1200 x 5) -- EVAL-ONLY until whole workflows are held out
        jevlogs_triage.jsonl         reachjalil/jevlogs: Noul, gold = Loghub anomaly label (NOT Jev's route)
        pagerduty_trigger.jsonl      reachjalil/jev-luna-pagerduty-trigger: Choice K=3 + Noul page_now
        tree_choice_cap.jsonl        reachjalil/jev-tree-choice-cap: Choice K=320 flat + region/service/mode steps
"""
import argparse
import collections
import glob
import json
import random
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pcdm"))
from data import norm_text  # noqa: E402
from pcdm_jev.decider import query_text  # noqa: E402

JEV_PUBLIC = "/tmp/jevbench/datasets/public/*.jsonl"
YES_NO = ["no", "yes"]


def jsonl(repo, path):
    with open(hf_hub_download(repo, path, repo_type="dataset")) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parquet(repo, path):
    import pandas as pd
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset")).to_dict("records")


def row(task, state, question, candidates, label=None, probs=None, meta=None):
    """probs: optional soft distribution aligned with candidates (label defaults to its argmax)."""
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    if probs is None:
        target = [1.0 if j == label else 0.0 for j in range(len(candidates))]
    else:
        s = sum(probs) or 1.0
        target = [p / s for p in probs]
        label = max(range(len(candidates)), key=lambda j: target[j]) if label is None else label
    return {"state": state, "question": question, "query": query_text(question),
            "candidates": list(candidates), "target": target,
            "p_null": 0.0, "task": task, "label": label, "meta": {"qtype": question["type"], **(meta or {})}}


# ---------------------------------------------------------------- sources
FORM_Q = {"type": "choice",
          "instructions": "The agent is filling a form from a document. Which action should it take on the focused element?",
          "criteria": {"fill <entity>: <value>": "Type this document entity's value into the element (label must match the element's meaning).",
                       "check": "Tick this checkbox or radio button.", "click": "Click this button or link.",
                       "skip": "Leave this element untouched (nothing in the document belongs here)."}}


def cua_s1_forms(split, cap, seed, max_skip=0.35):
    rows = [row("cua_s1_forms", r["context"], FORM_Q, r["options"], r["label"],
                meta={"family": "cua_s1_forms", "split": split, **r.get("meta", {})})
            for r in jsonl("cua-ai/cua-s1-forms", f"{split}.jsonl")]
    if cap and len(rows) > cap:  # 52% of the source is `skip`; cap it at max_skip of the sample
        rng = random.Random(seed)
        skip = [r for r in rows if r["meta"]["action"] == "skip"]
        rest = [r for r in rows if r["meta"]["action"] != "skip"]
        n_skip = min(len(skip), int(cap * max_skip))
        rows = rng.sample(skip, n_skip) + rng.sample(rest, min(len(rest), cap - n_skip))
        rng.shuffle(rows)
    return rows


SCORE_WORDS = {"low": 0, "medium": 1, "high": 2}  # jev-4b writes score gold as words; criteria is the ordered level list


def jev4b(path):
    """MagaBitmex/jev-4b-distill-data: programmatic `gold` only; the Jev teacher distribution (from the
    *_teacher twin) is stored in meta.teacher and never used as target. K<2 extraction questions dropped."""
    teacher = {r["state_id"]: r["teacher"] for r in jsonl("MagaBitmex/jev-4b-distill-data", path.replace(".jsonl", "_teacher.jsonl"))}
    out = []
    for r in jsonl("MagaBitmex/jev-4b-distill-data", path):
        for qid, q in r["questions"].items():
            crit, g = q["criteria"], r["gold"][qid]
            if q["type"] == "noul":
                cands, label = YES_NO, YES_NO.index(g)
            elif q["type"] == "score":
                cands, label = [str(i) for i in range(len(crit))], SCORE_WORDS[g]
            else:
                cands, label = list(crit), list(crit).index(g)
            if len(cands) < 2:
                continue
            t = teacher.get(r["state_id"], {}).get(qid, {})
            out.append(row("jev4b", r["state"], q, cands, label,
                           meta={"family": "jev4b_distill", "qid": qid, "kind": r["kind"], "state_id": r["state_id"],
                                 "teacher": t.get("probabilities") or ({"yes": t["noul"]} if "noul" in t else None)}))
    return out


def systemone_lite(split, cap, seed):
    """dwidlee/systemone-lite-general: letter aliases -> real option keys; qtype from meta.schema_hint."""
    out = []
    for r in parquet("dwidlee/systemone-lite-general", f"data/{split}-00000-of-00001.parquet"):
        m = json.loads(r["meta"])
        qtype = m.get("schema_hint", "choice")
        crit = dict(v.split(": ", 1) for v in r["criteria_values"])
        keys = sorted(crit, key=int) if qtype == "score" else (YES_NO if qtype == "noul" else list(crit))
        instr = re.sub(r"\s*—?\s*[^.?!—]*\bletter\b[^.?!]*[.?!]?", "", r["instructions"]).strip() or r["instructions"]
        q = {"type": qtype, "instructions": instr,
             "criteria": [crit[k] for k in keys] if qtype == "score" else
             ({"true": crit["yes"], "false": crit["no"]} if qtype == "noul" else crit)}
        out.append(row("systemone_lite", r["state"], q, keys, keys.index(r["label_key"]),
                       meta={"family": "systemone_lite", "task": r["task"], "gym": m["gym"], "hard": m.get("hard", False), "split": split}))
    if cap and len(out) > cap:
        out = random.Random(seed).sample(out, cap)
    return out


def mind2web():
    out = []
    for r in jsonl("AndeyTait/JevForge-Mind2Web", "data/test.jsonl"):
        req, tg = r["request"], r["targets"]
        for qid, q in req["questions"].items():
            if q["type"] == "noul":
                cands, probs = YES_NO, [tg[qid]["false"], tg[qid]["true"]]
            else:  # candidates "e3: <desc>" -- element ids alone carry no meaning, descriptions alone repeat (many `div`)
                keys = list(q["criteria"])
                cands, probs = [f"{k}: {q['criteria'][k]}" for k in keys], [tg[qid][k] for k in keys]
            out.append(row("mind2web", req["state"], q, cands, probs=probs,
                           meta={"family": "mind2web", "qid": qid, "website": r["source_group"], "id": r["id"],
                                 "n_positive": len(r["meta"]["positive_ids"])}))
    return out


def typed_decisions(split):
    out = []
    for r in parquet("LocalLLaMA/typed-decisions", f"all/{split}-00000-of-00001.parquet"):
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        for qid, q in qs.items():
            crit = q.get("criteria")
            # JevBench label conventions: noul -> no/yes, score -> "0".."n", choice -> criteria keys
            keys = [str(i) for i in range(len(crit))] if q["type"] == "score" else (
                ["false", "true"] if q["type"] == "noul" else list(crit))
            cands = YES_NO if q["type"] == "noul" else keys
            g = gold[qid]["probabilities"]
            out.append(row("typed_decisions", r["state"], q, cands, probs=[g[k] for k in keys],
                           meta={"family": r["workflow"], "qid": qid, "case": r["id"], "split": split,
                                 "confidence": gold[qid].get("confidence")}))
    return out


LOG_Q = {"type": "noul",
         "instructions": "This log line belongs to an anomalous block or alert and is worth routing to an expensive analysis model.",
         "criteria": {"true": "The line comes from a failing block / alert (Loghub anomaly label).",
                      "false": "Routine line; nothing here needs analysis."}}


def jevlogs():
    out = []
    for f in ["inputs_hdfs", "inputs_bgl", "inputs_adversarial"]:
        for r in jsonl("reachjalil/jevlogs-log-triage-benchmark", f"data/{f}.jsonl"):
            state = f"{r['severityText']} {r['body']}"
            out.append(row("jevlogs_triage", state, LOG_Q, YES_NO, int(r["label"] == "anomaly"),
                           meta={"family": f"log_{r['dataset']}", "label_granularity": "block" if r["dataset"] == "hdfs" else "line",
                                 "injection": r.get("injection", False), "id": r["id"]}))
    return out


PAGE_Q = {"type": "choice", "instructions": "Under PagerDuty alerting principles, what should the on-call system do with this log line?",
          "criteria": {"ignore": "No human action needed; keep it in the log stream.",
                       "ticket": "A human should look during business hours; open a ticket, do not page.",
                       "page": "A human must act now; page the on-call engineer."}}
PAGE_NOW_Q = {"type": "noul", "instructions": "A human must act on this log line right now (page the on-call engineer).",
              "criteria": {"true": "Paid work, data, or availability is at risk and needs immediate intervention.",
                           "false": "Informational, self-healed, or can wait for a ticket."}}


def pagerduty():
    out = []
    for r in jsonl("reachjalil/jev-luna-pagerduty-trigger", "data/stream.jsonl"):
        state = f"[{r['service']}] {r['severityText']}: {r['body']}"
        meta = {"family": r["family"], "trap": r["trap"], "trap_kind": r.get("trap_kind"), "id": r["id"]}
        cands = list(PAGE_Q["criteria"])
        out.append(row("pagerduty_action", state, PAGE_Q, cands, cands.index(r["gold_action"]), meta=meta))
        out.append(row("pagerduty_page_now", state, PAGE_NOW_Q, YES_NO, int(r["gold_page"]), meta=meta))
    return out


def tree_choice_cap():
    tax = json.load(open(hf_hub_download("reachjalil/jev-tree-choice-cap", "data/taxonomy.json", repo_type="dataset")))
    leaves, regions, services, modes = [], {}, {}, {}
    for reg in tax["children"]:
        regions[reg["id"].removeprefix("reg_")] = reg["label"]
        for svc in reg["children"]:
            services[svc["id"].split("_svc_")[1]] = svc["label"]
            for leaf in svc["children"]:
                modes[leaf["value"]["mode"]] = leaf["label"]
                leaves.append((leaf["id"], f"{reg['label']} / {svc['label']} / {leaf['label']}"))
    leaf_ids = [i for i, _ in leaves]
    flat_q = {"type": "choice", "instructions": "Which incident-catalog entry (region / service / failure mode) does this ticket describe?"}
    step_qs = {"region": ({"type": "choice", "instructions": "Which region does this ticket concern?"}, regions),
               "service": ({"type": "choice", "instructions": "Which service does this ticket concern?"}, services),
               "mode": ({"type": "choice", "instructions": "Which failure mode does this ticket describe?"}, modes)}
    out = []
    for r in jsonl("reachjalil/jev-tree-choice-cap", "data/tickets.jsonl"):
        meta = {"family": "incident_catalog", "gold_index": r["gold_index"], "tail": r["gold_index"] >= 255,
                "trap": r["trap"], "trap_kind": r.get("trap_kind"), "id": r["id"]}
        out.append(row("tree_choice_flat320", r["body"], flat_q, [d for _, d in leaves], leaf_ids.index(r["gold_id"]), meta=meta))
        for step, (q, vocab) in step_qs.items():
            keys = list(vocab)
            out.append(row(f"tree_choice_{step}", r["body"], q, [vocab[k] for k in keys], keys.index(r[f"gold_{step}"]),
                           meta=dict(meta, step=step)))
    return out


# ---------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data_wf_hf"))
    ap.add_argument("--limit", type=int, default=None, help="rows per file (smoke test)")
    ap.add_argument("--cap_train", type=int, default=60_000)
    ap.add_argument("--cap_val", type=int, default=3_000)
    ap.add_argument("--cap_lite", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jev_states = {norm_text(s if isinstance(s, str) else json.dumps(s, ensure_ascii=False))
                  for f in glob.glob(JEV_PUBLIC) for s in (json.loads(l)["state"] for l in open(f))}
    builds = {
        "train/cua_s1_forms.jsonl": lambda: cua_s1_forms("train", args.cap_train, args.seed),
        "train/jev4b_distill.jsonl": lambda: jev4b("data3/train.jsonl"),
        "train/systemone_lite.jsonl": lambda: systemone_lite("train", args.cap_lite, args.seed),
        "val/cua_s1_forms.jsonl": lambda: cua_s1_forms("validation", args.cap_val, args.seed),
        "eval/cua_s1_forms_test.jsonl": lambda: cua_s1_forms("test", None, args.seed),
        "eval/cua_s1_forms_demo.jsonl": lambda: cua_s1_forms("demo", None, args.seed),
        "eval/typed_decisions_test.jsonl": lambda: typed_decisions("test"),
        "eval/typed_decisions_train.jsonl": lambda: typed_decisions("train"),
        "eval/jevlogs_triage.jsonl": jevlogs,
        "eval/pagerduty_trigger.jsonl": pagerduty,
        "eval/tree_choice_cap.jsonl": tree_choice_cap,
        "eval/jev4b_adversarial.jsonl": lambda: jev4b("data_hard/eval.jsonl"),
        "eval/systemone_lite_hard.jsonl": lambda: systemone_lite("test_hard", None, args.seed),
        "eval/mind2web_choice.jsonl": mind2web,
    }
    report = {}
    for rel, build in builds.items():
        rows = build()[: args.limit] if args.limit else build()
        if rel.startswith("train/"):
            for r in rows:
                r["meta"]["fam_bucket"] = "W"
        leaks = sum(norm_text(r["state"]) in jev_states for r in rows)
        assert all(0 <= r["label"] < len(r["candidates"]) and len(r["target"]) == len(r["candidates"]) for r in rows), rel
        path = Path(args.out) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ks = [len(r["candidates"]) for r in rows]
        qt = collections.Counter(r["meta"]["qtype"] for r in rows)
        report[rel] = {"rows": len(rows), "K": [min(ks), max(ks)], "qtypes": dict(qt), "jevbench_leaks": leaks}
        print(f"{rel:36s} rows={len(rows):7d} K={min(ks)}-{max(ks)} {dict(qt)} leaks={leaks}", flush=True)
    (Path(args.out) / "manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
