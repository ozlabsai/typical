#!/usr/bin/env python3
"""R060-A: how far are R056's agent_trace teacher labels (mean of Gemma-think and Qwen-think) from the gold labeler's
GPT side (openai/gpt-6-astra, which dominates agent_trace gold)? Reuses R059's labeling path and fit machinery.

  sample  pilot: R056 agent_trace questions from the clean EN train split of R056's mix that have both teacher
          labels, stratified by question id (2x weight on attribution, first_bad_step, claims_supported,
          within_grant and the score fields), weighted round-robin so a budget-truncated prefix stays stratified.
          full: every R056 agent_trace question (all generators, both splits) + every security choice/score question.
  label   --set pilot|full: astra, one question per call, strict json schema, reasoning high. Responses are cached
          (.context/r060a/cache), so any rerun resumes for free. --dry-run spends nothing: uncached calls are
          refused by a $0 budget, counted, and priced at the measured $/question.
  fit     teacher mean vs astra: argmax flips, noul MAE, choice/score TVD per question id; per-qid monotone
          correction under 5-fold CV (noul: R059's logit-linear vs isotonic; choice/score: temperature, which
          cannot move an argmax, and temperature + per-class bias where the labels are fixed).
  push    .context/r060a (minus cache) -> guychuk/typical-r056-data r060a/.

uv run --project .context/research_sources/WorkflowEvals --with scikit-learn --with huggingface_hub \
    python scripts/r060a_trace_gap.py <cmd>
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import r059_anchor_labels as r059  # puts scripts/ and WorkflowEvals on sys.path
from r059_anchor_labels import ROOT, hf, jsonl, write

t = r059.t
OUT = ROOT / ".context" / "r060a"
MODEL = "openai/gpt-6-astra"
TRACE = "agent_trace_observability"
TRACE_NODES = tuple(f"{g}/{TRACE}/nodes.jsonl" for g in ("gen", "gen_gpt", "gen_gemma", "gen_qwen"))
PRIORITY = {"attribution", "first_bad_step", "claims_supported", "within_grant",
            "expressed_satisfaction", "request_specificity"}


def labels_of(q: dict, answer: dict | None) -> dict | None:
    """An R056 teacher answer -> {label: p} in the keys wire_probs uses (noul true/false, score "0".., choice keys)."""
    if not answer:
        return None
    if q["type"] == "noul":
        return {"true": float(answer["noul"]), "false": 1 - float(answer["noul"])}
    keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    p = {k: max(0.0, float(answer.get("probabilities", {}).get(k, 0.0))) for k in keys}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()} if s > 0 else None


def nodes_of(paths, wf) -> dict:
    return {n["request_id"]: {**n, "workflow": wf} for p in paths for n in jsonl(hf(p))}


def cmd_sample(args):
    trace = nodes_of(TRACE_NODES, TRACE)
    sec = nodes_of(r059.SEC_NODES, "security_incidents")
    split = {}
    for name in ("train", "dev_wf"):
        for line in hf(f"mix/{name}.jsonl").open():
            m = json.loads(line)
            if m["meta"].get("lang") == "en" and m["meta"].get("workflow"):
                split[m["id"].split(":")[1]] = name
    teach = {name: {(r["id"], r["qid"]): r.get("answer") for r in jsonl(hf(f"labels/labels_{name}.jsonl"))}
             for name in ("gemma", "qwen")}

    def row(n, qid):
        q = n["questions"][qid]
        r = {"id": n["request_id"], "qid": qid, "type": q["type"], "workflow": n["workflow"], "case_id": n["case_id"],
             "node": n["node"], "split": split.get(n["request_id"], "unused"), "state": n["state"], "question": q}
        for name in teach:
            r[f"p_{name}"] = labels_of(q, teach[name].get((n["request_id"], qid)))
        return r

    full = [row(n, qid) for n in trace.values() for qid in n["questions"]]
    full += [row(n, qid) for n in sec.values() for qid, q in n["questions"].items() if q["type"] != "noul"]
    full.sort(key=lambda r: (r["workflow"], r["id"], r["qid"]))
    write(OUT / "full_questions.jsonl", full)

    pool = defaultdict(list)
    for r in full:
        if r["workflow"] == TRACE and r["split"] == "train" and r["p_gemma"] and r["p_qwen"]:
            pool[r["qid"]].append(r)
    rng = random.Random(60)
    for qid in sorted(pool):
        rng.shuffle(pool[qid])
    pilot, i = [], 0
    while len(pilot) < args.n and i < max(map(len, pool.values())):  # weighted round-robin: 2/priority qid, 1/other
        for qid in sorted(pool):
            for r in pool[qid][i * (1 + (qid in PRIORITY)): (i + 1) * (1 + (qid in PRIORITY))]:
                pilot.append({k: v for k, v in r.items() if k != "split"})
        i += 1
    pilot = pilot[: args.n]
    for r in pilot:
        r["p_mean"] = {k: (r["p_gemma"][k] + r["p_qwen"][k]) / 2 for k in r["p_gemma"]}
    write(OUT / "pilot_questions.jsonl", pilot)
    summary = {"pool_clean_train_by_qid": {q: len(v) for q, v in sorted(pool.items())},
               "pilot_by_qid": Counter(r["qid"] for r in pilot),
               "full": {"n": len(full), "by_workflow_type": Counter(f"{r['workflow']}/{r['type']}" for r in full),
                        "by_workflow_split": Counter(f"{r['workflow']}/{r['split']}" for r in full)}}
    (OUT / "sample_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def cmd_label(args):
    from system_one_adapter import SystemOneAdapterClient

    t.CACHE, t.LEDGER["max_usd"] = OUT / "cache", 0.0 if args.dry_run else args.max_usd
    client = SystemOneAdapterClient(structured_outputs=True, llm_answer_mode="probabilities",
                                    normalize_probabilities=True, n_retry_malformed_structure=1)
    qs = [r for r in jsonl(OUT / f"{args.set}_questions.jsonl") if args.workflow in (None, r["workflow"])][: args.limit]

    def one(r):
        try:
            return r, t.ask(client, "openrouter", MODEL, r["state"], {r["qid"]: r["question"]})[r["qid"]], None
        except Exception as exc:  # budget stops are instant; keep the rest
            return r, None, f"{type(exc).__name__}: {exc}"[:300]

    out, errors, missing = [], Counter(), Counter()
    with ThreadPoolExecutor(args.workers) as pool:
        for r, p, err in pool.map(one, qs):
            if err:
                errors[err[:120]] += 1
                missing[r["workflow"]] += err.startswith("BudgetExceeded")
            else:
                out.append({"id": r["id"], "qid": r["qid"], "workflow": r["workflow"], "type": r["type"],
                            "teacher": MODEL, "p": p})
    s = t.STATS[MODEL]
    summary = {"set": args.set, "workflow": args.workflow, "questions": len(qs), "labeled": len(out),
               "errors": dict(errors), "usd": round(s["usd"], 4), "new_usd": round(t.LEDGER["usd"], 4),
               "usd_per_question": round(s["usd"] / max(len(out), 1), 5), "new_calls": t.LEDGER["calls"],
               "cached_calls": t.LEDGER["cached"],
               **{f"{k}_per_call": round(s[k] / max(s["calls"], 1), 1)
                  for k in ("input_tokens", "output_tokens", "reasoning_tokens", "wall_s")}}
    if args.dry_run:  # price uncached questions at the measured $/question of earlier runs covering that workflow
        rate = {}
        for f in OUT.glob("labels_astra_*_summary.json"):
            m = json.loads(f.read_text())
            if m["labeled"] >= 20:
                rate[m["workflow"] or TRACE] = m["usd_per_question"]
        summary["dry_run"] = {"uncached_by_workflow": dict(missing), "usd_per_question": rate,
                              "est_usd": round(sum(n * rate.get(wf, float("nan")) for wf, n in missing.items()), 2)}
    else:
        name = f"{args.set}{'_' + args.workflow if args.workflow else ''}"
        write(OUT / f"labels_astra_{name}.jsonl", out)
        (OUT / f"labels_astra_{name}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# --------------------------------------------------------------------------- fit


def temper(p, a, b=0.0):
    """softmax(a * log(clip(p, eps)) + b): the multiclass sigma(a*logit(p)+b). b = 0 keeps the argmax (pure
    temperature); a per-class bias b (fixed-label qids only) can move it, like R059's intercept."""
    import numpy as np
    z = a * np.log(np.clip(p, r059.EPS, 1)) + b
    e = np.exp(z - z.max())
    return e / e.sum()


def tvd(p, q) -> float:  # for a noul this is |p(true) - q(true)|, i.e. noul MAE
    return float(.5 * abs(p - q).sum())


def top(v, typ):
    return v[0] > .5 if typ == "noul" else v.argmax()


def cv_temper(P, Q, bias: bool, k=5):
    """5-fold CV temperature (+ per-class bias) for one question id, fit on mean TVD. Rows may differ in width
    (first_bad_step), so the bias is only for fixed-label qids."""
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.model_selection import KFold

    pred = [None] * len(P)
    for tr, te in KFold(min(k, len(P)), shuffle=True, random_state=60).split(P):
        x0 = np.r_[1.0, np.zeros(len(P[0]) if bias else 0)]
        f = minimize(lambda x: np.mean([tvd(temper(P[i], x[0], x[1:] if bias else 0.0), Q[i]) for i in tr]), x0,
                     method="Nelder-Mead", options={"maxiter": 4000}).x
        for i in te:
            pred[i] = temper(P[i], f[0], f[1:] if bias else 0.0)
    return pred


def cmd_fit(args):
    import numpy as np

    rows = {(r["id"], r["qid"]): r for r in jsonl(OUT / "pilot_questions.jsonl")}
    astra = {(r["id"], r["qid"]): r["p"] for r in jsonl(OUT / "labels_astra_pilot.jsonl")}
    rows = [{**rows[k], "p_astra": astra[k]} for k in rows if k in astra]
    pop = json.loads((OUT / "sample_summary.json").read_text())["pool_clean_train_by_qid"]
    nq = Counter(r["qid"] for r in rows)
    res = {"n": len(rows), "by_qid": {}}
    flips, gaps, ws = [], [], []

    def vec(r, name):
        return np.array([r[f"p_{name}"][k] for k in r["p_astra"]])

    for qid in sorted(nq):
        rs = [r for r in rows if r["qid"] == qid]
        typ = rs[0]["type"]
        A = [vec(r, "astra") for r in rs]
        g = {"type": typ, "n": len(rs)}
        for name in ("mean", "gemma", "qwen"):
            P = [vec(r, name) for r in rs]
            flip = np.array([top(p, typ) != top(a, typ) for p, a in zip(P, A)])
            gap = np.array([tvd(p, a) for p, a in zip(P, A)])
            g[name] = {"flip": round(flip.mean(), 3), "noul_mae" if typ == "noul" else "tvd": round(gap.mean(), 3)}
            if name == "mean":
                flips.append(flip.mean()), gaps.append(gap.mean()), ws.append(pop[qid])
                if typ == "score":
                    ev = lambda v: v @ np.arange(len(v))  # noqa: E731
                    g[name]["mean_minus_astra_expected_score"] = round(np.mean([ev(p) - ev(a) for p, a in zip(P, A)]), 3)
                g[name]["astra_mid_rate" if typ == "noul" else "astra_max_p"] = round(float(np.mean(
                    [r059.LO <= a[0] <= r059.HI for a in A] if typ == "noul" else [a.max() for a in A])), 3)
                g[name]["teacher_mid_rate" if typ == "noul" else "teacher_max_p"] = round(float(np.mean(
                    [r059.LO <= p[0] <= r059.HI for p in P] if typ == "noul" else [p.max() for p in P])), 3)
        if typ == "noul":  # R059 machinery on p(true): identity vs logit-linear vs isotonic, 5-fold CV
            px, py, w = np.array([vec(r, "mean")[0] for r in rs]), np.array([a[0] for a in A]), np.ones(len(rs))
            strata = np.array([min(r059.pbin(p), 4) // 2 for p in px])  # 3 coarse bins: thin qids
            for m in r059.MODELS:
                pred = r059.cv(m, px, py, w, strata)[0]
                g[f"cv_{m}"] = {"flip": round(float(np.mean((pred > .5) != (py > .5))), 3),
                                "noul_mae": round(float(np.abs(pred - py).mean()), 3)}
        else:
            for name, bias in (("cv_temperature", False), ("cv_temperature_bias", True)):
                if bias and len({len(a) for a in A}) > 1:
                    continue
                pred = cv_temper([vec(r, "mean") for r in rs], A, bias)
                g[name] = {"flip": round(float(np.mean([p.argmax() != a.argmax() for p, a in zip(pred, A)])), 3),
                           "tvd": round(float(np.mean([tvd(p, a) for p, a in zip(pred, A)])), 3)}
        res["by_qid"][qid] = g
    w = np.array(ws) / sum(ws)
    res["population_weighted"] = {"flip": round(float(w @ flips), 3), "tvd (= noul MAE)": round(float(w @ gaps), 3)}
    best = [min((v for k, v in g.items() if k.startswith("cv_") or k == "mean"),  # ponytail: picked on CV score
                key=lambda v: v.get("tvd", v.get("noul_mae"))) for g in res["by_qid"].values()]
    res["population_weighted_best_cv_correction"] = {
        "flip": round(float(w @ [b["flip"] for b in best]), 3),
        "tvd (= noul MAE)": round(float(w @ [b.get("tvd", b.get("noul_mae")) for b in best]), 3)}
    res["rank_by_flip"] = sorted(res["by_qid"], key=lambda q: -res["by_qid"][q]["mean"]["flip"])
    write(OUT / "pilot_labels.jsonl", [{k: v for k, v in r.items() if k not in ("state", "question")} for r in rows])
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def cmd_push(args):
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=r059.REPO, repo_type="dataset", folder_path=str(OUT), path_in_repo="r060a",
                          ignore_patterns=["cache/*", "*.log", "full_questions.jsonl"],
                          commit_message="R060-A: astra pilot labels on R056 agent_trace questions")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample").add_argument("--n", type=int, default=900)
    s = sub.add_parser("label")
    s.add_argument("--set", choices=("pilot", "full"), required=True)
    s.add_argument("--workflow", choices=(TRACE, "security_incidents"))
    s.add_argument("--max-usd", type=float, default=0.0, help="stop starting calls past this (in-flight calls finish)")
    s.add_argument("--dry-run", action="store_true", help="$0 budget: count uncached questions and price them")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int)
    sub.add_parser("fit")
    sub.add_parser("push")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"sample": cmd_sample, "label": cmd_label, "fit": cmd_fit, "push": cmd_push}[args.cmd](args)


if __name__ == "__main__":
    main()
