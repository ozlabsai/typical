#!/usr/bin/env python3
"""R058: three idea heads on frozen features of the R056 student (docs/research/R058_IDEA_HEADS.md).

  rows   <r056 dir> <out_dir> [max_train]     train/dev question rows + dev case files from R056's generated data
  cache  <rows.jsonl> <out_dir> --model M --model-revision R --adapter A [--temperatures T]
                                              [GPU] answer-slot state (lm_head input) + option logits per question
  train  <train_cache> <dev_cache> <out_dir>  4 heads on the frozen features; dev answers of every arm -> pred.jsonl
  replay <cases_dir> <pred.jsonl> <out_dir>   [WorkflowEvals venv] replay each workflow program on every arm's answers
                                              vs program(teacher target) on held-out synthetic cases; metrics + gates
  serve  <heads.pt> <head> --model-id ID ...  [GPU] student + one head behind the TypeSafe /v1/systemone API
  diag   <security results.json>              [WorkflowEvals venv] NON-SHIPPABLE fixed [.15,.75] projection replay

Rows (input of `cache`): {case_id, request_id, workflow, node, qid, lang, question (wire), state, teachers, target}.
uv run --project .context/research_sources/WorkflowEvals python scripts/r058_heads.py replay ...
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WE = Path(os.environ.get("WORKFLOWEVALS", ROOT / ".context" / "research_sources" / "WorkflowEvals"))
WORKFLOWS = ("security_incidents", "agent_trace_observability", "invoice_processing", "customer_service")
MIN_CASES = 10
BAND = (0.15, 0.75)  # security is_true_positive middle cell; also the reported "middle band" for every binary


def compile_slices(we: Path = WE) -> dict[str, tuple[float, float]]:
    """Burden slices read from the workflow code: nouls whose two thresholds bracket a middle action.

    Only security's triage gate has one (`p < P_CLOSE` closes, `p > P_ACT` acts, else middle); every other noul in
    the four programs is a single floor (two-way), so it has no neutral cell and is left alone."""
    src = (we / "evals" / "security_incidents" / "workflow.py").read_text()
    lo, hi = (float(re.search(rf"^{k} = ([\d.]+)$", src, re.M)[1]) for k in ("P_CLOSE", "P_ACT"))
    assert "p = a.is_true_positive.noul" in src and lo < hi
    return {"is_true_positive": (lo, hi)}


# ---------------------------------------------------------------- replay (WorkflowEvals venv; no torch)

def wire(qtype: str, probs: dict[str, float]) -> dict:
    """Probabilities -> the answer the harness consumes (argmax/expectation/confidence via the R053 temper at T=1)."""
    from scripts.r053_calibration_headroom import temper
    if qtype == "noul":
        return {"type": "noul", "noul": probs["true"]}
    return temper({"type": qtype, "probabilities": probs}, 1.0)


def replay_case(adapter, case_id: str, lookup: dict):
    """Run one case's program on a {(node, state_hash, qid): answer} lookup; None if a needed answer is missing."""
    from scripts.r053_calibration_headroom import ReplaySession
    try:
        return adapter.execute(case_id, ReplaySession(lookup, [p.policy_id for p in adapter.bundle.policies], None))
    except KeyError:  # ponytail: the program took a branch with no cached answer (unseen node)
        return None


def auprc(scores: list[float], labels: list[int]) -> float | None:
    """Average precision (step-wise), the usual AUPRC estimate."""
    pos = sum(labels)
    if not pos:
        return None
    hits, total = 0, 0.0
    for k, (_, y) in enumerate(sorted(zip(scores, labels), key=lambda t: -t[0]), 1):
        if y:
            hits += 1
            total += hits / k
    return total / pos


def replay_main(cases_dir: Path, pred_path: Path, out: Path) -> dict:
    sys.path[:0] = [str(WE), str(ROOT)]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from importlib import import_module

    from core.contract import fingerprint
    from core.scoring import compare
    from scripts.r053_calibration_headroom import HEADLINE

    preds = [json.loads(line) for line in pred_path.read_text().splitlines()]
    arms = list(preds[0]["arms"])
    nodes = {}
    for wf in WORKFLOWS:
        path = cases_dir / wf / "nodes.jsonl"
        for line in (path.read_text().splitlines() if path.exists() else []):
            n = json.loads(line)
            nodes[n["request_id"]] = (n["node"], fingerprint(n["state"]))
    lookups = defaultdict(lambda: defaultdict(dict))  # arm -> case_id -> key -> wire
    for r in preds:
        if r["request_id"] not in nodes:  # Hebrew twins: scored for argmax drift only, never replayed
            continue
        key = (*nodes[r["request_id"]], r["qid"])
        for arm, probs in r["arms"].items():
            lookups[arm][r["case_id"]][key] = wire(r["type"], probs)
        if r.get("target"):
            lookups["teacher"][r["case_id"]][key] = wire(r["type"], r["target"])
            lookups["teacher_corr"][r["case_id"]][key] = wire(r["type"], r.get("target_corr") or r["target"])

    def score(wf: str, ref_name: str) -> dict:
        path = cases_dir / wf / "cases.jsonl"
        cases = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        cases = [c for c in cases if c["case_id"] in lookups["teacher"]]
        hits, n, ref_missing, arm_missing = defaultdict(int), 0, 0, defaultdict(int)
        for c in cases:
            cid = c["case_id"]
            adapter = import_module(f"evals.{wf}.eval_adapter").Adapter(
                WE / "evals" / wf, SimpleNamespace(inputs={cid: c["input"]}, metadata={cid: c["metadata"]}))
            metric = next(m for m in adapter.bundle.metrics if m.metric_id == HEADLINE[wf])
            ref = replay_case(adapter, cid, lookups[ref_name][cid])
            if ref is None:
                ref_missing += 1  # teacher labels do not cover the reference path: no reference for this case
                continue
            got = {arm: replay_case(adapter, cid, lookups[arm][cid]) for arm in arms}
            for e in ref.decisions:
                if e.status != "ok":
                    continue
                cmp = {}
                for arm, result in got.items():
                    a = next((d for d in result.decisions if d.policy_id == e.policy_id), None) if result else None
                    cmp[arm] = False if a is None or a.status != "ok" else compare(metric, a, e)
                if any(v is None for v in cmp.values()):
                    continue  # no primary action on one side: metric undefined, as in core.scoring
                n += 1
                for arm, v in cmp.items():
                    hits[arm] += bool(v)
                    arm_missing[arm] += got[arm] is None
        return {"cases": len(cases), "decisions": n, "ref_missing_cases": ref_missing,
                "agreement": {arm: hits[arm] / n if n else None for arm in arms},
                "arm_unseen_branch_decisions": dict(arm_missing)}

    res = {"arms": arms, "workflows": {wf: score(wf, "teacher") for wf in WORKFLOWS}, "headline_metric": HEADLINE}
    # preregistered: the mean is over workflows with >= MIN_CASES replayable dev cases (invoice has ~1)
    scored = [wf for wf in WORKFLOWS if res["workflows"][wf]["cases"] - res["workflows"][wf]["ref_missing_cases"] >= MIN_CASES]
    res["mean_agreement"] = {arm: sum(res["workflows"][wf]["agreement"][arm] for wf in scored) / len(scored)
                             for arm in arms}
    res["mean_over"] = scored
    res["binary"] = binary_metrics(preds, arms)
    log = json.loads((pred_path.parent / "train_log.json").read_text())
    taus = log.get("tau_variants") or {"": log["tau"]}
    res["gates"] = {k: v for sfx in taus for k, v in gates(res, preds, taus, sfx).items()}
    if any("target_corr" in r for r in preds):  # R059: security also scored against the gold-style corrected targets
        sec = [r for r in preds if r["workflow"] == "security_incidents" and r["type"] == "noul" and r.get("lang") == "en"
               and r.get("target_corr")]
        res["security_vs_corrected"] = {
            "replay": score("security_incidents", "teacher_corr"),
            "noul_mae": {arm: sum(abs(r["arms"][arm]["true"] - r["target_corr"]["true"]) for r in sec) / len(sec)
                         for arm in arms},
            "is_true_positive_mae": {arm: (lambda rows: sum(abs(r["arms"][arm]["true"] - r["target_corr"]["true"])
                                                             for r in rows) / max(len(rows), 1))(
                [r for r in sec if r["qid"] == "is_true_positive"]) for arm in arms},
            "n_noul": len(sec)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def gates(res: dict, preds: list[dict], taus: dict, sfx: str = "") -> dict:
    """Each idea's own go/kill rule (idea files §4), applied to held-out synthetic replay. sfx "+corr" = R059 variant."""
    mean = res["mean_agreement"]
    sec = {a: res["workflows"]["security_incidents"]["agreement"][a] for a in res["arms"]}
    mae = {a: res["binary"]["all_noul"][a]["mae"] for a in res["arms"]}
    base, spread, gate, cert = (n + sfx for n in ("base0", "a_spread", "b_gate", "c_cert"))
    he = [r for r in preds if r.get("lang") == "he"]
    argmax = lambda p: max(p, key=p.get)  # noqa: E731
    he_flips = sum(argmax(r["arms"][spread]) != argmax(r["arms"][base]) for r in he) / len(he) if he else None
    a = {"mae_cut_vs_base0": mae[base] - mae[spread], "agreement_gain_vs_base0": mean[spread] - mean[base],
         "he_argmax_flip_rate_vs_base0": he_flips, "he_rows": len(he)}
    a["pass"] = a["mae_cut_vs_base0"] >= 0.025 and a["agreement_gain_vs_base0"] >= 0.020 and (he_flips or 0) <= 2 / 128
    tau = taus[sfx]
    rows = [r for r in preds if r.get("js_student_target") is not None and r.get("lang") == "en"]
    blends = [k for k in mean if k.startswith(f"blend{sfx}_g")]
    best_const = max(["student", *blends], key=mean.get)
    b = {"tau": tau, "gate_auprc": auprc([r["extra"][f"{gate}.gate"] for r in rows],
                                         [int(r["js_student_target"] >= tau) for r in rows]),
         "positive_rate": sum(r["js_student_target"] >= tau for r in rows) / len(rows),
         "gain_vs_student": mean[gate] - mean["student"], "gain_vs_best_constant": mean[gate] - mean[best_const],
         "best_constant_arm": best_const, "security_vs_student": sec[gate] - sec["student"]}
    b["pass"] = (b["gate_auprc"] or 0) >= 0.50 and b["gain_vs_best_constant"] >= 0.020 and b["security_vs_student"] >= 0
    c = {"security": sec[cert], "mean": mean[cert], "security_base0": sec[base], "mean_base0": mean[base]}
    # econ_law_pl.md's numbers are official-WorkflowEvals thresholds; on synthetic replay (a different scale) the
    # head must also not lose to its own uncertified distribution (base0), fixed before any real score.
    c["pass"] = sec[cert] >= 0.50 and mean[cert] >= 0.57 and mean[cert] >= mean[base]
    return {spread: a, gate: b, cert: c}


def binary_metrics(preds: list[dict], arms: list[str]) -> dict:
    """MAE of P(true) to the teacher target and middle-band rates, all binaries and the security slice."""
    out = {}
    for name, keep in (("all_noul", lambda r: True), ("is_true_positive", lambda r: r["qid"] == "is_true_positive")):
        rows = [r for r in preds if r["type"] == "noul" and r.get("target") and r.get("lang") == "en" and keep(r)]
        if not rows:
            continue
        mid = lambda p: BAND[0] <= p <= BAND[1]  # noqa: E731
        out[name] = {"n": len(rows), "teacher_middle_band": sum(mid(r["target"]["true"]) for r in rows) / len(rows),
                     **{arm: {"mae": sum(abs(r["arms"][arm]["true"] - r["target"]["true"]) for r in rows) / len(rows),
                              "middle_band": sum(mid(r["arms"][arm]["true"]) for r in rows) / len(rows)}
                        for arm in arms}}
    return out


def diag_main(results: Path) -> dict:
    """NON-SHIPPABLE DIAGNOSTIC: an official security run replayed with is_true_positive clamped to the code's
    neutral cell (econ_law_pl.md's fixed projection; no certificate, so it hard-codes the evaluator's thresholds)."""
    sys.path[:0] = [str(WE), str(ROOT)]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from core.workflow import load
    from scripts.r053_calibration_headroom import replay, run_answers
    lo, hi = compile_slices()["is_true_positive"]
    wf, lookup = load("security_incidents"), run_answers(results)
    lookup = {c: {k: a | {"legend": {}} if a["type"] == "score" else a for k, a in case.items()} for c, case in lookup.items()}
    clamped = {c: {k: ({**a, "noul": min(hi, max(lo, a["noul"]))} if k[2] == "is_true_positive" else a)
                   for k, a in case.items()} for c, case in lookup.items()}
    raw, clamp = replay(wf, lookup, None), replay(wf, clamped, None)
    return {"label": "NON-SHIPPABLE DIAGNOSTIC: fixed [.15,.75] projection of is_true_positive",
            "exact_actions_raw": raw["exact_actions"], "exact_actions_clamped": clamp["exact_actions"],
            "case_errors": [raw["case_errors"], clamp["case_errors"]]}


# ---------------------------------------------------------------- rows (stdlib)

def r056_target(wf: str, teachers: dict) -> dict | None:
    """R056's training target (r056_build.teacher_target): security = Qwen only; else mean(Gemma, Qwen);
    Gemma alone is smoothed off exact 0/1 (0.96 p + 0.04 / K); Qwen alone as is."""
    g, q = teachers.get("gemma"), teachers.get("qwen")
    if wf == "security_incidents":
        return q
    if g and q:
        return {k: (g[k] + q[k]) / 2 for k in g}
    if g and any(x in (0.0, 1.0) for x in g.values()):
        return {k: 0.96 * x + 0.04 / len(g) for k, x in g.items()}
    return g or q


def probs_of(question: dict, answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    keys = list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(question["criteria"]))]
    z = sum(answer["probabilities"].get(k, 0.0) for k in keys) or 1.0
    return {k: answer["probabilities"].get(k, 0.0) / z for k in keys}


def rows_main(r056: Path, out: Path, max_train: int) -> None:
    """rows_{train,dev}.jsonl + dev_cases/<wf>/{cases,nodes}.jsonl from every R056 generator dir.
    dev = R056's held-out split (sha256(case_id) % 10 == 0), every question of every node (labeled or not, so
    head-driven branches have answers) + Hebrew twins of dev nodes; train = labeled non-dev questions, <= max_train."""
    import hashlib
    is_dev = lambda c: int(hashlib.sha256(c.encode()).hexdigest(), 16) % 10 == 0  # noqa: E731  (r056_build.is_dev)
    labels = defaultdict(dict)
    for path in sorted((r056 / "labels").glob("labels_*.jsonl")):
        teacher = "qwen" if "qwen" in path.name else "gemma"
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "answer" in r:
                labels[(r["id"], r["qid"])][teacher] = r["answer"]
    nodes, cases = {}, defaultdict(dict)
    for gen in sorted(r056.glob("gen*")):
        for wf in WORKFLOWS:
            for name, sink in (("nodes.jsonl", nodes), ("cases.jsonl", None)):
                path = gen / wf / name
                for line in (path.read_text().splitlines() if path.exists() else []):
                    r = json.loads(line)
                    if sink is None:
                        cases[wf][r["case_id"]] = r
                    else:
                        nodes[r["request_id"]] = r | {"workflow": wf, "lang": "en"}
    twins = r056 / "twins" / "twins.jsonl"
    for line in (twins.read_text().splitlines() if twins.exists() else []):
        t = json.loads(line)
        if is_dev(t["case_id"]) and t["twin_id"] in nodes:
            nodes[t["request_id"]] = t  # twins inherit the English node's teacher labels (R056 mix rule)
    # R056's mix decides which nodes are clean (13-gram decontamination vs every benchmark) and in which split
    mix = {split: {json.loads(line)["id"].split(":")[1] for line in (r056 / "mix" / f).read_text().splitlines()
                   if json.loads(line)["meta"].get("lang") == "en" and ":" in json.loads(line)["id"]}
           for split, f in (("train", "train.jsonl"), ("dev", "dev_wf.jsonl"))}
    dirty = {n["case_id"] for rid, n in nodes.items() if n["lang"] == "en" and is_dev(n["case_id"]) and rid not in mix["dev"]
             and any((rid, q) in labels for q in n["questions"])}  # a labeled dev node the mix dropped
    nodes = {rid: n for rid, n in nodes.items() if n["case_id"] not in dirty
             and (is_dev(n["case_id"]) or n.get("twin_id", rid) in mix["train"])}
    train, dev = [], []
    for rid, n in nodes.items():
        src = n.get("twin_id", rid)
        for qid, q in n["questions"].items():
            teachers = {k: probs_of(nodes[src]["questions"][qid], a) for k, a in labels.get((src, qid), {}).items()}
            row = {"case_id": n["case_id"], "request_id": rid, "workflow": n["workflow"], "node": n["node"], "qid": qid,
                   "lang": n["lang"], "question": q, "state": n["state"], "teachers": teachers,
                   "target": r056_target(n["workflow"], teachers)}
            if is_dev(n["case_id"]):
                dev.append(row)
            elif row["target"] and n["lang"] == "en":
                train.append(row)
    train.sort(key=lambda r: hashlib.sha256(f"{r['request_id']}/{r['qid']}".encode()).hexdigest())
    train = train[:max_train]
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("rows_train.jsonl", train), ("rows_dev.jsonl", dev)):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for wf in WORKFLOWS:
        d = out / "dev_cases" / wf
        d.mkdir(parents=True, exist_ok=True)
        dev_ids = {c for c in cases[wf] if is_dev(c) and c not in dirty}
        (d / "cases.jsonl").write_text("".join(json.dumps(cases[wf][c], ensure_ascii=False) + "\n" for c in sorted(dev_ids)))
        (d / "nodes.jsonl").write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in nodes.values()
                                               if n["case_id"] in dev_ids and n["lang"] == "en"))
    summary = {"dirty_dev_cases_dropped": len(dirty), "train": len(train), "dev": len(dev), "dev_labeled": sum(bool(r["target"]) for r in dev),
               "dev_he": sum(r["lang"] == "he" for r in dev),
               "dev_cases": {wf: sum(is_dev(c) and c not in dirty for c in cases[wf]) for wf in WORKFLOWS},
               "train_by_wf": {wf: sum(r["workflow"] == wf for r in train) for wf in WORKFLOWS}}
    (out / "rows_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- features (GPU; repo venv)

K = 26  # option letters
TYPES = {"noul": 0, "choice": 1, "score": 2}


def temperature_of(temps: dict, qid: str, qtype: str) -> float:
    """R056 serving temperature, looked up exactly as serve_typesafe_gemma_lora does."""
    return temps.get(qid, temps.get(f"type:{qtype}", 1.0))


class Student:
    """The R056 student through the serving runtime; also returns the answer-slot state (input of lm_head)."""

    def __init__(self, model, revision, adapter, adapter_revision=None, max_context=16384):
        from scripts.gemma_decision_runtime import GemmaDecisionRuntime
        self.rt = GemmaDecisionRuntime(model, revision, adapter, adapter_revision, max_context, "bfloat16").load()
        self.grab = {}
        self.rt._model.get_output_embeddings().register_forward_pre_hook(
            lambda _m, args: self.grab.__setitem__("h", args[0][0, -1].float()))

    def __call__(self, state, question: dict):
        """-> (h [d], option logits [n], labels, input tokens), the exact singleton serving forward."""
        import torch

        from scripts.gemma_decision_runtime import render_typesafe_decision_prompt
        from scripts.r053_build_train_rows import server_items
        items = server_items(question)
        ids, tokens = self.rt._input_ids(render_typesafe_decision_prompt(
            state, question.get("instructions"), items, question_type=question["type"]))
        with self.rt._lock, torch.inference_mode():
            logits = self.rt._model(input_ids=ids.to(self.rt.device), use_cache=False, logits_to_keep=1).logits[0, -1]
            logits = logits.float()
            s = torch.stack([torch.logsumexp(logits[g.to(logits.device)], 0) for g in self.rt._groups[:len(items)]])
            h = self.grab["h"].cpu()  # inside the lock: the server answers requests from several threads
        return h, s.cpu(), [k for k, _ in items], tokens


def cache_main(rows_path: Path, out: Path, args) -> None:
    """Shards of 500 rows (resumable): feats-NNNNN.pt {h fp16 [n,d], s fp32 [n,K] -inf padded} + meta-NNNNN.jsonl."""
    import torch

    from scripts.r053_build_train_rows import server_items
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    out.mkdir(parents=True, exist_ok=True)
    student = Student(args.model, args.model_revision, args.adapter)
    r = rows[0]  # the cached logits must be the served ones
    _, s0, labels0, _ = student(r["state"], r["question"])
    served = student.rt.score_options(r["state"], r["question"].get("instructions"),
                                      server_items(r["question"]), question_type=r["question"]["type"]).probabilities
    assert torch.allclose(torch.softmax(s0, 0).double(), torch.tensor([served[k] for k in labels0]), atol=1e-5), served
    shard = 500
    for start in range(0, len(rows), shard):
        name = f"{start // shard:05d}"
        if (out / f"meta-{name}.jsonl").exists():
            continue
        hs, ss, meta = [], [], []
        for r in rows[start:start + shard]:
            try:
                h, s, labels, tokens = student(r["state"], r["question"])
            except ValueError as exc:  # Overlength: no answer (the served model would 422 too)
                print(f"skip {r['request_id']}/{r['qid']}: {exc}", flush=True)
                continue
            hs.append(h.half())
            ss.append(torch.cat([s, s.new_full((K - len(s),), float("-inf"))]))
            meta.append({k: v for k, v in r.items() if k not in ("state", "question")}
                        | {"type": r["question"]["type"], "labels": labels, "tokens": tokens})
        torch.save({"h": torch.stack(hs), "s": torch.stack(ss)}, out / f"feats-{name}.pt")
        (out / f"meta-{name}.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta))
        print(f"{min(start + shard, len(rows))}/{len(rows)}", flush=True)
    (out / "DONE").write_text("ok\n")


def load_cache(d: Path):
    import torch
    names = sorted(p.name[5:-6] for p in d.glob("meta-*.jsonl"))
    feats = [torch.load(d / f"feats-{n}.pt") for n in names]
    meta = [json.loads(line) for n in names for line in (d / f"meta-{n}.jsonl").read_text().splitlines()]
    return torch.cat([f["h"] for f in feats]).float(), torch.cat([f["s"] for f in feats]), meta


# ---------------------------------------------------------------- heads (CPU is enough)
# Every hyperparameter below is fixed before any dev score is computed (R058 preregistration, see the doc).
HP = {
    "opt": {"lr": 1e-3, "weight_decay": 1e-2, "batch": 256, "epochs": 30, "seed": 58, "internal_val_mod": 10},
    "spread": {"lambda": 1.0, "rho": 0.01, "gh_points": 5},             # forecasting.md §2
    "gate": {"lambda": 1.0, "rho": 1.0, "beta": 0.1, "tau_pct": 70, "band_kappa": 20.0},  # neuro_bio.md §2
    "cert": {"lambda": 1.0, "pass": 0.5},                                # econ_law_pl.md §2
    "blends": [0.25, 0.5, 0.75, 1.0],                                    # constant-g comparators for the gate
}


def batch_of(h, s, meta, qvocab, slices, temps=None):
    """Tensors for the heads. s is the *served* student logit (R056 temperature applied), as every arm starts there."""
    import torch
    n = len(meta)
    temps = temps or {}
    t = torch.tensor([temperature_of(temps, m["qid"], m["type"]) for m in meta])
    s = s / t[:, None]
    mask = torch.isfinite(s)
    tpos = torch.tensor([m["labels"].index("true") if m["type"] == "noul" else 0 for m in meta])
    fpos = torch.tensor([m["labels"].index("false") if m["type"] == "noul" else 0 for m in meta])

    def dist(key):
        out, has = torch.zeros(n, K), torch.zeros(n, dtype=torch.bool)
        for i, m in enumerate(meta):
            p = m.get(key) if key == "target" else (m.get("teachers") or {}).get(key)
            if p:
                out[i, :len(m["labels"])] = torch.tensor([p[k] for k in m["labels"]])
                has[i] = True
        return out, has

    m_, has_m = dist("target")
    names = sorted({k for m in meta for k in (m.get("teachers") or {})})
    teachers = [dist(k) for k in names]
    # 1e-6 inside the cell: the program's tests are strict, and float round trips must not cross an endpoint
    lo_hi = torch.tensor([slices.get(m["qid"], (0.0, 1.0)) for m in meta], dtype=torch.float64) + torch.tensor([1e-6, -1e-6])
    return {"h": h, "s": s.masked_fill(~mask, float("-inf")), "mask": mask, "n": n,
            "typ": torch.tensor([TYPES[m["type"]] for m in meta]), "noul": torch.tensor([m["type"] == "noul" for m in meta]),
            "qid": torch.tensor([qvocab.get(m["qid"], 0) for m in meta]), "tpos": tpos, "fpos": fpos,
            "m": m_, "has_m": has_m, "teachers": teachers,
            "slice": torch.tensor([m["qid"] in slices for m in meta]), "lo": lo_hi[:, 0].float(), "hi": lo_hi[:, 1].float()}


def take(b, idx):
    return {k: (v[idx] if hasattr(v, "shape") and v.shape and v.shape[0] == b["n"] else
                [(a[idx], c[idx]) for a, c in v] if k == "teachers" else v) for k, v in b.items()} | {"n": len(idx)}


def build_heads(d: int, nq: int):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    def zero(layer):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def kl(m, logp, mask):  # KL(m || p) per row, 0 log 0 = 0
        return (torch.xlogy(m, m) - m * logp.masked_fill(~mask, 0)).sum(-1)

    def ptrue(p, b):
        return p.gather(1, b["tpos"][:, None])[:, 0]

    def set_binary(p, b, q):  # write P(true)=q into the noul rows of a [N,K] distribution
        p = p.clone()
        r = b["noul"].nonzero()[:, 0]
        p[r, b["tpos"][r]] = q[r]
        p[r, b["fpos"][r]] = 1 - q[r]
        return p

    def teacher_extremes(b):
        ps = [(ptrue(t, b), has) for t, has in b["teachers"]]
        lo = torch.stack([torch.where(has, p, torch.full_like(p, 2.0)) for p, has in ps]).min(0).values
        hi = torch.stack([torch.where(has, p, torch.full_like(p, -1.0)) for p, has in ps]).max(0).values
        return lo, hi  # min/max over the teachers present (one teacher: itself)

    class Base(nn.Module):  # (0) linear head on the state, residual to the served student logits; KL to the target
        def __init__(self):
            super().__init__()
            self.w = zero(nn.Linear(d, K))

        def logits(self, b):
            return b["s"] + self.w(b["h"])

        def predict(self, b):
            return {"p": torch.softmax(self.logits(b), -1)}

        def loss(self, b):
            return kl(b["m"], torch.log_softmax(self.logits(b), -1), b["mask"]).mean()

    class Spread(Base):  # (a) criterion spread on binaries: p = E_{C~N(c,s^2)} sigma(z - C), 5-pt Gauss-Hermite
        def __init__(self):
            super().__init__()
            self.z, self.c, self.sd = zero(nn.Linear(d, 1)), zero(nn.Linear(d, 1)), zero(nn.Linear(d, 1))
            self.ec, self.es = nn.Embedding(nq, 1), nn.Embedding(nq, 1)
            nn.init.zeros_(self.ec.weight)
            nn.init.zeros_(self.es.weight)
            self.sd.bias.data.fill_(-2.0)  # softplus(-2) = .13: start near the plain student
            u, w = np.polynomial.hermite.hermgauss(HP["spread"]["gh_points"])
            self.register_buffer("u", torch.tensor(u, dtype=torch.float32) * math.sqrt(2))
            self.register_buffer("gw", torch.tensor(w, dtype=torch.float32) / math.sqrt(math.pi))

        def binary(self, b):
            s = b["s"].masked_fill(~b["mask"], 0)
            z = s.gather(1, b["tpos"][:, None])[:, 0] - s.gather(1, b["fpos"][:, None])[:, 0] + self.z(b["h"])[:, 0]
            c = self.ec(b["qid"])[:, 0] + self.c(b["h"])[:, 0]
            sd = F.softplus(self.es(b["qid"])[:, 0] + self.sd(b["h"])[:, 0])
            q = torch.sigmoid(z[:, None] - c[:, None] - sd[:, None] * self.u)
            p = (q * self.gw).sum(1)
            return p, (q * q * self.gw).sum(1) - p * p, sd

        def predict(self, b):
            p, v, sd = self.binary(b)
            return {"p": set_binary(super().predict(b)["p"], b, p), "spread": sd}

        def loss(self, b):
            hp, nb = HP["spread"], b["noul"]
            p, v, sd = self.binary(b)
            (t1, h1), (t2, h2) = b["teachers"][:2] if len(b["teachers"]) >= 2 else (b["teachers"][0],) * 2
            d2 = (ptrue(t1, b) - ptrue(t2, b)) ** 2 / 2
            both = h1 & h2 & (len(b["teachers"]) >= 2)
            binary = (p - ptrue(b["m"], b)) ** 2 + hp["lambda"] * both * (v - d2) ** 2 + hp["rho"] * sd ** 2
            multi = kl(b["m"], torch.log_softmax(self.logits(b), -1), b["mask"])
            return torch.where(nb, binary, multi).mean()

    class Gate(Base):  # (b) replay gate: p = softmax((1-g) l0 + g l1), l0 = served student, l1 = l0 + W1 h
        def __init__(self, tau=1.0):
            super().__init__()
            self.gh, self.gf, self.gt = zero(nn.Linear(d, 1)), zero(nn.Linear(2, 1)), nn.Embedding(3, 1)
            nn.init.zeros_(self.gt.weight)
            self.register_buffer("tau", torch.tensor(float(tau)))

        def gate(self, b):
            p0 = torch.softmax(b["s"], -1)
            ent = -torch.xlogy(p0, p0).sum(-1)
            top = p0.topk(2, -1).values
            return torch.sigmoid(self.gh(b["h"])[:, 0] + self.gf(torch.stack([ent, top[:, 0] - top[:, 1]], -1))[:, 0]
                                 + self.gt(b["typ"])[:, 0])

        def logits_g(self, b, g):
            return b["s"] + g[:, None] * self.w(b["h"])

        def predict(self, b, g=None):
            g = self.gate(b) if g is None else torch.full((b["n"],), float(g))
            return {"p": torch.softmax(self.logits_g(b, g), -1), "gate": g}

        def loss(self, b):
            hp = HP["gate"]
            g = self.gate(b)
            logp = torch.log_softmax(self.logits_g(b, g), -1)
            logp0 = torch.log_softmax(b["s"], -1)
            tag = (js(b["m"], torch.softmax(b["s"], -1), b["mask"]) / self.tau).clamp(max=1)
            p = ptrue(logp.exp(), b)
            k, mt = hp["band_kappa"], ptrue(b["m"], b)
            band = sum(torch.where(mt > th, F.softplus(k * (th - p)), F.softplus(k * (p - th))) / k
                       for th in (b["lo"], b["hi"])) * b["slice"]
            return (kl(b["m"], logp, b["mask"]) + hp["lambda"] * F.binary_cross_entropy(g, tag, reduction="none")
                    + hp["rho"] * band + hp["beta"] * (1 - g) * kl(logp0.exp(), logp, b["mask"])).mean()

    class Cert(Base):  # (c) burden certificates on the compiled slice; uncertified crossings -> neutral cell
        def __init__(self):
            super().__init__()
            self.cert = nn.Linear(d, 2)  # [crossing above hi, crossing below lo]
            nn.init.zeros_(self.cert.weight)

        def predict(self, b):
            p = super().predict(b)["p"]
            c = torch.sigmoid(self.cert(b["h"]))
            q = ptrue(p, b)
            up_ok, down_ok = c[:, 0] > HP["cert"]["pass"], c[:, 1] > HP["cert"]["pass"]
            q2 = torch.where((q > b["hi"]) & ~up_ok, b["hi"], torch.where((q < b["lo"]) & ~down_ok, b["lo"], q))
            q2 = torch.where(b["slice"], q2, q)
            return {"p": set_binary(p, b, q2), "cert_up": c[:, 0], "cert_down": c[:, 1]}

        def loss(self, b):
            lo_t, hi_t = teacher_extremes(b)
            y = torch.stack([lo_t > b["hi"], hi_t < b["lo"]], -1).float()  # unanimity: min > hi / max < lo
            any_t = torch.stack([has for _, has in b["teachers"]]).any(0)
            bce = F.binary_cross_entropy_with_logits(self.cert(b["h"]), y, reduction="none").sum(-1) * b["slice"] * any_t
            return super().loss(b) + HP["cert"]["lambda"] * bce.mean()

    def js(p, q, mask):
        mid = (p + q) / 2
        return (kl(p, mid.clamp_min(1e-12).log(), mask) + kl(q, mid.clamp_min(1e-12).log(), mask)) / 2

    return {"base0": Base, "a_spread": Spread, "b_gate": Gate, "c_cert": Cert}, js


def fit(model, b, train_idx, val_idx):
    """AdamW minibatches; keep the epoch with the best internal-val loss (internal val = train cases only)."""
    import copy

    import torch
    hp = HP["opt"]
    torch.manual_seed(hp["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    best, best_state, log = float("inf"), None, []
    for epoch in range(hp["epochs"]):
        model.train()
        for chunk in torch.randperm(len(train_idx)).split(hp["batch"]):
            loss = model.loss(take(b, train_idx[chunk]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(model.loss(take(b, val_idx)))
        log.append(val)
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"best_epoch": log.index(best), "val_loss": log}


def load_corrected(path: Path | None) -> dict:
    """R059 gold-style security targets: (request_id, qid) -> corrected P(true)."""
    if not path:
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {(r.get("request_id") or r["id"], r["qid"]): float(r.get("p_corrected", r.get("p"))) for r in rows}


def with_corrected(meta: list[dict], corr: dict) -> list[dict]:
    """Security nouls retargeted to the corrected P(true) (target and the lone security teacher alike)."""
    out = []
    for m in meta:
        p = corr.get((m.get("twin_id") or m["request_id"].removesuffix("-he"), m["qid"]))
        if m["workflow"] == "security_incidents" and m["type"] == "noul" and p is not None and m.get("target"):
            t = {"true": p, "false": 1 - p}
            m = m | {"target": t, "teachers": {"qwen": t}}
        out.append(m)
    return out


def train_main(train_dir: Path, dev_dir: Path, out: Path, temps_path: Path | None, corr_path: Path | None = None) -> None:
    import hashlib

    import torch
    temps = json.loads(temps_path.read_text()) if temps_path else {}
    slices = compile_slices()
    h_tr, s_tr, m_tr = load_cache(train_dir)
    keep = [i for i, m in enumerate(m_tr) if m.get("target")]
    h_tr, s_tr, m_tr = h_tr[keep], s_tr[keep], [m_tr[i] for i in keep]
    h_dev, s_dev, m_dev = load_cache(dev_dir)
    assert not {m["case_id"] for m in m_tr} & {m["case_id"] for m in m_dev}, "train/dev case overlap"
    mu, sd = h_tr.mean(0), h_tr.std(0).clamp_min(1e-3)
    norm = lambda h: ((h - mu) / sd).clamp(-10, 10)  # noqa: E731  z-score (Gemma has rogue dims)
    qvocab = {q: i + 1 for i, q in enumerate(sorted({m["qid"] for m in m_tr}))}  # 0 = unseen qid
    btr = batch_of(norm(h_tr), s_tr, m_tr, qvocab, slices, temps)
    bdev = batch_of(norm(h_dev), s_dev, m_dev, qvocab, slices, temps)
    in_val = torch.tensor([int(hashlib.sha256(f"r058-val/{m['case_id']}".encode()).hexdigest(), 16) % HP["opt"]["internal_val_mod"] == 0
                           for m in m_tr])
    tr_idx, va_idx = (~in_val).nonzero()[:, 0], in_val.nonzero()[:, 0]
    assert len(va_idx) and len(tr_idx), "empty internal split"  # salted: R056's dev rule is also sha256 % 10
    heads, js = build_heads(h_tr.shape[1], len(qvocab) + 1)
    with torch.no_grad():
        tau = float(torch.quantile(js(btr["m"], torch.softmax(btr["s"], -1), btr["mask"]), HP["gate"]["tau_pct"] / 100))
    models, logs = {}, {}
    variants = {"": (btr, tau)}
    corr = load_corrected(corr_path)
    if corr:  # R059 extra: every head once more against the corrected security targets
        bc = batch_of(norm(h_tr), s_tr, with_corrected(m_tr, corr), qvocab, slices, temps)
        with torch.no_grad():
            tau_c = float(torch.quantile(js(bc["m"], torch.softmax(bc["s"], -1), bc["mask"]), HP["gate"]["tau_pct"] / 100))
        variants["+corr"] = (bc, tau_c)
    for sfx, (bt, tau_v) in variants.items():
        for name, cls in heads.items():
            models[name + sfx] = cls(tau_v) if name == "b_gate" else cls()
            logs[name + sfx] = fit(models[name + sfx], bt, tr_idx, va_idx)
            print(name + sfx, logs[name + sfx]["best_epoch"], f"{min(logs[name + sfx]['val_loss']):.4f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"hp": HP, "tau": tau, "mu": mu, "sd": sd, "qvocab": qvocab, "slices": slices, "temps": temps,
                "d": h_tr.shape[1], "state": {k: m.state_dict() for k, m in models.items()}}, out / "heads.pt")

    with torch.no_grad():
        arms = {"student": torch.softmax(bdev["s"], -1)}
        extra = {}
        for name, m in models.items():
            pred = m.predict(bdev)
            arms[name] = pred["p"]
            extra |= {f"{name}.{k}": v for k, v in pred.items() if k != "p"}
        for sfx in variants:
            for g in HP["blends"]:
                arms[f"blend{sfx}_g{g}"] = models["b_gate" + sfx].predict(bdev, g)["p"]
        # NON-SHIPPABLE DIAGNOSTIC: the student's security answer clamped to the program's neutral cell, no certificate.
        q = arms["student"].gather(1, bdev["tpos"][:, None])[:, 0]
        clamp = torch.where(bdev["slice"], torch.minimum(torch.maximum(q, bdev["lo"]), bdev["hi"]), q)
        set_rows = bdev["slice"].nonzero()[:, 0]
        arms["diag_clamp"] = arms["student"].clone()
        arms["diag_clamp"][set_rows, bdev["tpos"][set_rows]] = clamp[set_rows]
        arms["diag_clamp"][set_rows, bdev["fpos"][set_rows]] = 1 - clamp[set_rows]
        js_dev = js(bdev["m"], torch.softmax(bdev["s"], -1), bdev["mask"])
    m_corr = with_corrected(m_dev, corr)
    with (out / "pred.jsonl").open("w") as f:
        for i, m in enumerate(m_dev):
            n = len(m["labels"])
            f.write(json.dumps({**{k: m[k] for k in ("case_id", "request_id", "workflow", "node", "qid", "type", "lang")},
                                "target": m.get("target"),
                                **({"target_corr": m_corr[i]["target"]} if corr else {}),
                                "arms": {a: dict(zip(m["labels"], p[i, :n].double().div(p[i, :n].sum()).tolist()))
                                         for a, p in arms.items()},
                                "extra": {k: float(v[i]) for k, v in extra.items()},
                                "js_student_target": float(js_dev[i]) if m.get("target") else None}) + "\n")
    (out / "train_log.json").write_text(json.dumps({"tau": tau, "n_train": len(tr_idx), "n_internal_val": len(va_idx),
                                                    "n_dev": len(m_dev), "corrected": str(corr_path) if corr else None,
                                                    "tau_variants": {k: v[1] for k, v in variants.items()}, "fits": logs, "hp": HP, "slices": slices},
                                                   indent=1) + "\n")


def serve_main(a) -> None:
    """Student + one head (or `student` alone) as a TypeSafe-compatible /v1/systemone endpoint (questions all)."""
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException

    from scripts.gemma_decision_runtime import OptionScores, Overlength
    from scripts.serve_typesafe_gemma_lora import SystemOneRequest, _answer
    from scripts.singleton_option_scorer import probabilities_from_option_scores
    ck = torch.load(a.heads)
    heads, _ = build_heads(ck["d"], len(ck["qvocab"]) + 1)
    head = None
    if a.head != "student":
        head = heads[a.head](ck["tau"]) if a.head == "b_gate" else heads[a.head]()
        head.load_state_dict(ck["state"][a.head])
        head.eval()
    student = Student(a.model, a.model_revision, a.adapter)
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": a.model_id, "description": f"R056 student + R058 {a.head}", "release_date": "2026-10-21"}]}

    def system_one(req: SystemOneRequest):
        if req.model != a.model_id:
            raise HTTPException(404, "model not found")
        answers, tokens = {}, 0
        for qid, q in req.questions.items():
            try:
                h, s, labels, n = student(req.state, q.model_dump())
            except Overlength as exc:
                raise HTTPException(422, str(exc)) from exc
            meta = [{"qid": qid, "type": q.type, "labels": labels}]
            pad = torch.cat([s, s.new_full((K - len(s),), float("-inf"))])[None]
            b = batch_of(((h[None] - ck["mu"]) / ck["sd"]).clamp(-10, 10), pad, meta, ck["qvocab"], ck["slices"], ck["temps"])
            with torch.no_grad():
                p = head.predict(b)["p"] if head else torch.softmax(b["s"], -1)
            logp = p[0, :len(labels)].double().clamp_min(1e-300).log().tolist()
            answers[qid] = _answer(q, OptionScores(probabilities_from_option_scores(labels, logp), n))
            tokens += n
        return {"model": a.model_id, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}

    system_one.__annotations__["req"] = SystemOneRequest  # this module uses postponed annotations
    app.post("/v1/systemone")(system_one)
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    import argparse
    cmd, *argv = sys.argv[1:] or ["help"]
    if cmd == "replay":
        print(json.dumps(replay_main(Path(argv[0]), Path(argv[1]), Path(argv[2])), indent=1))
    elif cmd == "rows":
        rows_main(Path(argv[0]), Path(argv[1]), int(argv[2]) if len(argv) > 2 else 20000)
    elif cmd == "diag":
        print(json.dumps(diag_main(Path(argv[0])), indent=1))
    elif cmd == "cache":
        ap = argparse.ArgumentParser()
        ap.add_argument("rows", type=Path)
        ap.add_argument("out", type=Path)
        for k in ("--model", "--model-revision", "--adapter"):
            ap.add_argument(k, required=True)
        a = ap.parse_args(argv)
        cache_main(a.rows, a.out, a)
    elif cmd == "train":
        ap = argparse.ArgumentParser()
        for k in ("train", "dev", "out"):
            ap.add_argument(k, type=Path)
        ap.add_argument("--temperatures", type=Path)
        ap.add_argument("--corrected", type=Path, help="R059 security_targets_corrected.jsonl (optional variant)")
        a = ap.parse_args(argv)
        train_main(a.train, a.dev, a.out, a.temperatures, a.corrected)
    elif cmd == "serve":
        ap = argparse.ArgumentParser()
        ap.add_argument("heads", type=Path)
        ap.add_argument("head", choices=("student", "base0", "a_spread", "b_gate", "c_cert"))
        for k in ("--model", "--model-revision", "--adapter", "--model-id"):
            ap.add_argument(k, required=True)
        ap.add_argument("--port", type=int, default=8090)
        serve_main(ap.parse_args(argv))
    else:
        raise SystemExit(__doc__)
