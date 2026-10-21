#!/usr/bin/env python3
"""R059: gold-style label anchor for R056 security noul targets (Qwen3.8-27B-thinking only).

  sample  400 R056 synthetic security noul questions from the clean (13-gram decontaminated) train split of
          R056's mix, stratified by Qwen probability bin (equal per bin, random within a bin = the node mix as
          in the data), ordered round-robin over bins so any budget-truncated prefix stays stratified.
  label   --teacher astra|fable: one question per call through the gold run's structured probability path
          (system_one_adapter structured_outputs=True, strict json schema). astra = openai/gpt-6-astra over
          OpenRouter (reasoning high), fable = claude-fable-5-1 via `claude -p --effort high` (the R054 P0d path).
  fit     Qwen vs gold-style (= mean of the teachers present), isotonic vs sigma(a*logit(p)+b) under 5-fold CV,
          then apply the chosen correction to every R056 security Qwen noul target.
  push    .context/r059 (minus the raw-response cache) -> private HF dataset guychuk/typical-r056-data r059/.

Stats over the anchor are population-weighted (bin population share / bin sample share).
uv run --project .context/research_sources/WorkflowEvals --with scikit-learn --with huggingface_hub \
    python scripts/r059_anchor_labels.py <cmd>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / ".context" / "research_sources" / "WorkflowEvals")]
import r053_teacher_label as t  # noqa: E402

OUT = ROOT / ".context" / "r059"
REPO = "guychuk/typical-r056-data"
SEC_NODES = ("gen/security_incidents/nodes.jsonl", "gen_gpt/security_incidents/nodes.jsonl")  # security gens: Haiku, GPT
TEACHERS = {"astra": ("openrouter", "openai/gpt-6-astra"), "fable": ("cli", "claude-fable-5-1")}
BINS = ("[0,.05)", "[.05,.15)", "[.15,.5)", "[.5,.75]", "(.75,.9)", "[.9,1]")
LO, HI = .15, .75  # middle band
EPS = .01  # Qwen's probability grid; logit clip


def pbin(p: float) -> int:
    return sum(p >= e for e in (.05, .15, .5)) + (p > .75) + (p >= .9)


def band(p: float) -> int:
    return -1 if p < LO else (1 if p > HI else 0)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def hf(path: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, path, repo_type="dataset"))


def security_rows() -> tuple[dict, list[dict]]:
    """(security nodes by request_id, every R056 Qwen noul label row on them; last row per (id, qid) wins)."""
    nodes = {n["request_id"]: n for p in SEC_NODES for n in jsonl(hf(p))}
    rows = {}
    for r in jsonl(hf("labels/labels_qwen.jsonl")):
        if r["id"] in nodes and (r.get("answer") or {}).get("type") == "noul":
            rows[(r["id"], r["qid"])] = r
    return nodes, list(rows.values())


def cmd_sample(args):
    nodes, rows = security_rows()
    clean = set()  # R056's mix decides which nodes are clean and in which split (as R058 does)
    for line in hf("mix/train.jsonl").open():
        m = json.loads(line)
        if m["meta"].get("lang") == "en" and m["meta"].get("workflow") == "security_incidents":
            clean.add(m["id"].split(":")[1])
    pool = defaultdict(list)
    for r in rows:
        if r["id"] in clean:
            pool[pbin(r["answer"]["noul"])].append(r)
    rng = random.Random(59)
    picked = []
    for b in range(len(BINS)):
        cands = sorted(pool[b], key=lambda r: (r["id"], r["qid"]))
        rng.shuffle(cands)
        picked.append(cands[: args.n // len(BINS) + (b < args.n % len(BINS))])
    anchor = []
    for i in range(max(map(len, picked))):  # round-robin over bins
        for b, rs in enumerate(picked):
            if i < len(rs):
                r, n = rs[i], nodes[rs[i]["id"]]
                anchor.append({"id": r["id"], "qid": r["qid"], "case_id": n["case_id"], "node": n["node"],
                               "p_qwen": r["answer"]["noul"], "bin": b, "state": n["state"],
                               "question": n["questions"][r["qid"]]})
    write(OUT / "anchor_questions.jsonl", anchor)
    summary = {"pool_clean_train": {BINS[b]: len(pool[b]) for b in range(len(BINS))},
               "all_security_noul_qwen": len(rows), "clean_train_nodes": len(clean),
               "anchor": {BINS[b]: len(picked[b]) for b in range(len(BINS))},
               "anchor_qids": Counter(a["qid"] for a in anchor), "anchor_nodes": Counter(a["node"] for a in anchor)}
    (OUT / "sample_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def cmd_label(args):
    from system_one_adapter import SystemOneAdapterClient

    backend, model = TEACHERS[args.teacher]
    model = args.model or model
    t.CACHE, t.LEDGER["max_usd"] = OUT / "cache", args.max_usd
    client = SystemOneAdapterClient(structured_outputs=True, llm_answer_mode="probabilities",
                                    normalize_probabilities=True, n_retry_malformed_structure=1)
    anchor = jsonl(OUT / "anchor_questions.jsonl")[: args.limit]

    def one(a):
        try:
            return a, t.ask(client, backend, model, a["state"], {a["qid"]: a["question"]})[a["qid"]]["true"], None
        except Exception as exc:  # budget stops are instant; keep the rest
            return a, None, f"{type(exc).__name__}: {exc}"[:300]

    out, errors = [], Counter()
    with ThreadPoolExecutor(args.workers) as pool:
        for a, p, err in pool.map(one, anchor):
            if err:
                errors[err[:120]] += 1
            else:
                out.append({"id": a["id"], "qid": a["qid"], "teacher": model, "p": p})
    write(OUT / f"labels_{args.teacher}.jsonl", out)
    s = t.STATS[model]
    summary = {"teacher": model, "effort": "high", "labeled": len(out), "errors": dict(errors),
               "usd": round(s["usd"], 4), "usd_per_question": round(s["usd"] / max(len(out), 1), 5),
               "calls": s["calls"], "new_calls": t.LEDGER["calls"], "cached_calls": t.LEDGER["cached"],
               **{f"{k}_per_call": round(s[k] / max(s["calls"], 1), 1)
                  for k in ("input_tokens", "output_tokens", "reasoning_tokens", "wall_s")}}
    (OUT / f"labels_{args.teacher}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# --------------------------------------------------------------------------- fit


def logit(p):
    import numpy as np
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class LogitLinear:
    """sigma(a*logit(p)+b) fit on weighted MAE, the selection metric. ponytail: a soft-target BCE fit regresses
    to the conditional mean, which pushed 50% of targets into the band and lost to raw Qwen (CV MAE .150 vs .141)."""

    def fit(self, x, y, w):
        import numpy as np
        from scipy.optimize import minimize

        z = logit(x)
        self.a, self.b = minimize(lambda ab: w @ np.abs(1 / (1 + np.exp(-(ab[0] * z + ab[1]))) - y),
                                  [1.0, 0.0], method="Nelder-Mead").x
        return self

    def predict(self, x):
        import numpy as np
        return 1 / (1 + np.exp(-(self.a * logit(x) + self.b)))


def isotonic():
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")


MODELS = {"logit_linear": LogitLinear, "isotonic": isotonic}


def metrics(pred, gold, w) -> dict:
    """Population-weighted agreement of a prediction (Qwen or corrected) with gold-style."""
    import numpy as np

    w = w / w.sum()
    bp, bg = np.vectorize(band)(pred), np.vectorize(band)(gold)
    return {"mae": round(float(w @ np.abs(pred - gold)), 4),
            "mid_rate_pred": round(float(w @ (bp == 0)), 4), "mid_rate_gold": round(float(w @ (bg == 0)), 4),
            "mid_agree": round(float(w @ ((bp == 0) == (bg == 0))), 4),
            "side_0.5_flip": round(float(w @ ((pred > .5) != (gold > .5))), 4),
            "band_flip": round(float(w @ (bp != bg)), 4)}


def cv(name, x, y, w, strata, k=5) -> tuple:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    pred = np.zeros_like(y)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=59).split(x, strata):
        pred[te] = MODELS[name]().fit(x[tr], y[tr], w[tr]).predict(x[te])
    return pred, metrics(pred, y, w)


def cmd_fit(args):
    import numpy as np

    anchor = jsonl(OUT / "anchor_questions.jsonl")
    labels = defaultdict(dict)
    for name in TEACHERS:
        path = OUT / f"labels_{name}.jsonl"
        for r in jsonl(path) if path.exists() else []:
            labels[(r["id"], r["qid"])][name] = r["p"]
    rows = [{**a, "teachers": labels[(a["id"], a["qid"])]} for a in anchor if labels[(a["id"], a["qid"])]]
    for r in rows:
        r["p_gold_style"] = sum(r["teachers"].values()) / len(r["teachers"])
    pop = json.loads((OUT / "sample_summary.json").read_text())["pool_clean_train"]
    nb = Counter(r["bin"] for r in rows)
    x = np.array([r["p_qwen"] for r in rows])
    y = np.array([r["p_gold_style"] for r in rows])
    w = np.array([pop[BINS[r["bin"]]] / nb[r["bin"]] for r in rows])
    strata = np.array([r["bin"] for r in rows])
    both = [r for r in rows if len(r["teachers"]) == 2]
    res = {"n_anchor": len(rows), "n_both_teachers": len(both),
           "teacher_coverage": Counter("+".join(sorted(r["teachers"])) for r in rows),
           "anchor_per_bin": {BINS[b]: nb[b] for b in sorted(nb)},
           "qwen_vs_gold_style": metrics(x, y, w)}
    for name in TEACHERS:
        sub = [i for i, r in enumerate(rows) if name in r["teachers"]]
        tp = np.array([rows[i]["teachers"][name] for i in sub])
        res[f"qwen_vs_{name}"] = metrics(x[sub], tp, w[sub])
    if both:
        ib = [i for i, r in enumerate(rows) if len(r["teachers"]) == 2]
        res["astra_vs_fable"] = metrics(np.array([rows[i]["teachers"]["astra"] for i in ib]),
                                        np.array([rows[i]["teachers"]["fable"] for i in ib]), w[ib])
    res["cv"] = {"raw_qwen": res["qwen_vs_gold_style"], **{name: cv(name, x, y, w, strata)[1] for name in MODELS}}
    chosen = "isotonic" if res["cv"]["isotonic"]["mae"] < res["cv"]["logit_linear"]["mae"] - .01 else "logit_linear"
    res["chosen"] = chosen

    # per question id: is_true_positive vs the other nouls (same model form, fit per group)
    tp_mask = np.array([r["qid"] == "is_true_positive" for r in rows])
    glob_pred = cv(chosen, x, y, w, strata)[0]
    group_pred = np.zeros_like(y)
    res["per_qid"] = {}
    for gname, m in (("is_true_positive", tp_mask), ("other_nouls", ~tp_mask)):
        if m.sum() < 15:
            continue
        group_pred[m] = cv(chosen, x[m], y[m], w[m], strata[m])[0]  # sklearn warns on thin bins; fine
        g = {"n": int(m.sum()), "qwen": metrics(x[m], y[m], w[m]),
             "global_correction_cv": metrics(glob_pred[m], y[m], w[m]),
             "group_correction_cv": metrics(group_pred[m], y[m], w[m])}
        if chosen == "logit_linear":
            f = LogitLinear().fit(x[m], y[m], w[m])
            g["group_params"] = {"a": round(f.a, 4), "b": round(f.b, 4)}
        res["per_qid"][gname] = g
    per_group = len(res["per_qid"]) == 2 and metrics(group_pred, y, w)["mae"] < res["cv"][chosen]["mae"] - .01
    res["per_qid_applied"] = per_group

    # apply to ALL R056 security Qwen noul targets
    fits = {None: MODELS[chosen]().fit(x, y, w)}
    if per_group:
        fits = {g: MODELS[chosen]().fit(x[m], y[m], w[m]) for g, m in (("tp", tp_mask), ("other", ~tp_mask))}
    if chosen == "logit_linear":
        res["params"] = {g or "all": {"a": round(f.a, 4), "b": round(f.b, 4)} for g, f in fits.items()}
    else:
        f = fits[None] if not per_group else None
        res["isotonic_knots"] = {"x": [round(v, 4) for v in f.X_thresholds_], "y": [round(v, 4) for v in f.y_thresholds_]} if f else None
    _, all_rows = security_rows()
    px = np.array([r["answer"]["noul"] for r in all_rows])
    if per_group:
        istp = np.array([r["qid"] == "is_true_positive" for r in all_rows])
        pc = np.where(istp, fits["tp"].predict(px), fits["other"].predict(px))
    else:
        pc = fits[None].predict(px)
    out = [{**r, "p_corrected": round(float(p), 4)} for r, p in zip(all_rows, pc)]
    write(OUT / "security_targets_corrected.jsonl", out)
    mid = lambda v: round(float(np.mean((v >= LO) & (v <= HI))), 4)  # noqa: E731
    res["applied"] = {"n": len(out), "mid_rate_qwen": mid(px), "mid_rate_corrected": mid(pc),
                      "exact01_qwen": round(float(np.mean((px == 0) | (px == 1))), 4),
                      "mean_qwen": round(float(px.mean()), 4), "mean_corrected": round(float(pc.mean()), 4)}
    write(OUT / "anchor_labels.jsonl", [{k: v for k, v in r.items() if k not in ("state", "question")} | {
        "p_cv_corrected": round(float(p), 4)} for r, p in zip(rows, glob_pred)])
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def cmd_push(args):
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(OUT), path_in_repo="r059",
                          ignore_patterns=["cache/*", "*.log"], commit_message="R059: gold-style security anchor + corrected targets")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=400)
    s = sub.add_parser("label")
    s.add_argument("--teacher", choices=tuple(TEACHERS), required=True)
    s.add_argument("--model", help="override, e.g. openai/gpt-6-astra:batch")
    s.add_argument("--max-usd", type=float, required=True, help="stop starting calls past this (in-flight calls finish)")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int)
    sub.add_parser("fit")
    sub.add_parser("push")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"sample": cmd_sample, "label": cmd_label, "fit": cmd_fit, "push": cmd_push}[args.cmd](args)


if __name__ == "__main__":
    main()
