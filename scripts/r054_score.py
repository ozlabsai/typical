"""R054 Phase 0 scoreboard: WorkflowEvals' own consensus scores + per-question gaps + JevBench/231.

Reads .context/r054-phase0/<run>/<workflow>/{scores.json,results.json} and <run>/jevbench/*.jsonl.
Per-question stats reuse the leaderboard_questions logic: noul MAE / noul side flip / choice+score argmax
flip vs labels-consensus, over questions the run answered that the consensus also answered.

uv run --project .context/research_sources/WorkflowEvals python scripts/r054_score.py [run ...]
"""
import json
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / ".context"
P0 = CTX / "r054-phase0"
sys.path[:0] = [str(ROOT / "scripts"), str(CTX / "research_sources" / "WorkflowEvals")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from r053_calibration_headroom import HEADLINE, teacher_answers  # noqa: E402

WF = list(HEADLINE)
REFERENCE = {  # published / saved WF consensus numbers (sec exact, trace primary, inv exact, cust exact)
    "frozen Cygnet": (.333, .523, .531, .698), "R044": (.338, .550, .518, .708), "Jev 1.13": (.617, .716, .618, .760)}
SAVED = {"frozen Cygnet": CTX / "g0-cygnet", "R044": CTX / "r044-workflows"}
N = {"security_incidents": 240, "agent_trace_observability": 111, "invoice_processing": 150, "customer_service": 204}


def wf_scores(run_dir: Path) -> dict:
    out = {}
    for wf in WF:
        f = run_dir / wf / "scores.json"
        if f.exists():
            s = json.loads(f.read_text())
            out[wf] = next(x["metrics"] for x in s["series"] if x["label_set_id"] == "labels-consensus")
    return out


def execution(run_dir: Path) -> dict:
    out = {}
    for wf in WF:
        f = run_dir / wf / "results.json"
        if not f.exists():
            continue
        cases = json.loads(f.read_text())["runs"][0]["cases"]
        ex = [c["execution"] for c in cases]
        out[wf] = {"cases": len(cases), "failed": sum(e["status"] != "ok" for e in ex),
                   "wall_s_mean": st.mean(e["wall_time_s"] or 0 for e in ex),
                   "calls": sum(e["calls"] or 0 for e in ex),
                   "input_tokens": sum(e["input_tokens"] or 0 for e in ex),
                   "output_tokens": sum(e["output_tokens"] or 0 for e in ex)}
    return out


def run_answers(results_path: Path) -> dict[str, dict]:
    """r053's lookup, skipping failed calls (no response), which count as unanswered here."""
    run = json.loads(results_path.read_text())["runs"][0]
    return {c["case_id"]: {(call["node_id"], call["state_hash"], q): a for call in c["calls"] if call.get("response")
                           for q, a in call["response"]["answers"].items()} for c in run["cases"]}


def per_question(run_dir: Path, consensus: dict) -> dict:
    out = {}
    for wf in WF:
        f = run_dir / wf / "results.json"
        if not f.exists():
            continue
        nm, nf, cf = [], [], []
        for cid, qs in run_answers(f).items():
            for k, a in qs.items():
                c = consensus[wf].get(cid, {}).get(k)
                if not c:
                    continue
                if a["type"] == "noul":
                    nm.append(abs(a["noul"] - c["noul"]))
                    nf.append((a["noul"] > .5) != (c["noul"] > .5))
                else:
                    p = {str(x): y for x, y in a["probabilities"].items()}
                    cp = {str(x): y for x, y in c["probabilities"].items()}
                    cf.append(max(p, key=p.get) != max(cp, key=cp.get))
        out[wf] = {"noul_mae": st.mean(nm) if nm else None, "noul_flip": st.mean(nf) if nf else None,
                   "choice_flip": st.mean(cf) if cf else None, "n": len(nm) + len(cf)}
    return out


def jev231(run_dir: Path):
    rows = [json.loads(line) for f in sorted((run_dir / "jevbench").glob("*.jsonl")) if f.name.startswith(("easy", "original", "hard"))
            for line in f.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    lat = sorted(r["latency_s"] or 0 for r in rows)
    return {"correct": sum(bool(r["correct"]) for r in rows), "n": len(rows), "failed": sum(not r["ok"] for r in rows),
            "latency_p50": lat[len(lat) // 2], "latency_p95": lat[int(len(lat) * .95)],
            "output_tokens": sum((r.get("usage") or {}).get("output_tokens") or 0 for r in rows)}


def subset_scores(runs: dict, case_ids: dict) -> dict:
    """Rescore saved runs + published Jev on the same case subset (paired comparison for a partial run)."""
    from core.io import load_results
    from core.scoring import build_score_data
    from plot import published_runs

    def score(bundle, ids, keep=lambda r: True):
        kept = [r.model_copy(update={"cases": [c for c in r.cases if c.case_id in ids]}) for r in bundle.runs if keep(r)]
        sd = build_score_data(bundle.model_copy(update={"runs": kept}), label_ids=["labels-consensus"], model_selection="all")
        return sd.series[0].metrics

    out = {name: {} for name in [*runs, "Jev 1.13"]}
    for wf, ids in case_ids.items():
        for name, d in runs.items():
            if (d / wf / "results.json").exists():
                out[name][wf] = score(load_results(d / wf / "results.json"), ids)
        out["Jev 1.13"][wf] = score(published_runs(wf), ids, lambda r: "jev-1.13.0" in r.run_id)
    return out


def headline(s: dict) -> tuple:
    vals = tuple(s[wf][m] if wf in s else None for wf, m in HEADLINE.items())
    return vals, (sum(vals) / 4 if None not in vals else None)


def main():
    runs = sys.argv[1:] or sorted(d.name for d in P0.iterdir() if d.is_dir() and (d / "DONE").exists())
    consensus = {wf: teacher_answers(wf, "consensus") for wf in WF}
    report = {}
    for name in runs:
        d = P0 / name
        s = wf_scores(d)
        vals, mean = headline(s)
        report[name] = {"headline": dict(zip(WF, vals)), "mean": mean,
                        "invoice_primary": s.get("invoice_processing", {}).get("primary_action"),
                        "per_question": per_question(d, consensus), "execution": execution(d), "jev231": jev231(d)}
    for name, d in SAVED.items():
        report[f"ref:{name}"] = {"per_question": per_question(d, consensus)}
    partial = [n for n in runs if any(e["cases"] < n_ for e, n_ in
               ((report[n]["execution"].get(w, {"cases": 0}), N[w]) for w in WF))]
    for name in partial:  # paired same-case table for budget-limited subset runs
        ids = {w: {c["case_id"] for c in json.loads((P0 / name / w / "results.json").read_text())["runs"][0]["cases"]}
               for w in WF if (P0 / name / w / "results.json").exists()}
        others = {n: P0 / n for n in runs} | SAVED
        sub = subset_scores(others, ids)
        report[name]["same_case_subset"] = {"n_cases": {w: len(v) for w, v in ids.items()},
                                            **{k: {"headline": dict(zip(WF, (h := headline(v))[0])), "mean": h[1]}
                                               for k, v in sub.items()}}
    (P0 / "scoreboard.json").write_text(json.dumps(report, indent=1) + "\n")

    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    print("| run | security exact | trace primary | invoice exact | invoice primary | customer exact | mean | Jev/231 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, (a, b, c, e) in REFERENCE.items():
        print(f"| {name} (ref) | {a:.3f} | {b:.3f} | {c:.3f} | - | {e:.3f} | {(a + b + c + e) / 4:.3f} | - |")
    for name in runs:
        r = report[name]
        j = r["jev231"]
        print(f"| {name} | " + " | ".join(fmt(r["headline"][w]) for w in WF[:3]) + f" | {fmt(r['invoice_primary'])} | "
              f"{fmt(r['headline'][WF[3]])} | **{fmt(r['mean'])}** | {j['correct'] if j else '-'} |")
    for name in partial:
        sub = report[name]["same_case_subset"]
        print(f"\nsame-case subset of {name} (cases {sub['n_cases']}):")
        for k, v in sub.items():
            if k != "n_cases":
                print(f"| {k} | " + " | ".join(fmt(v["headline"][w]) for w in WF) + f" | **{fmt(v['mean'])}** |")
    print("\nper-question vs consensus (noulMAE / noulFlip / choiceFlip)")
    for name, r in report.items():
        print(f"{name:28s}" + "".join(
            f"{'%s/%s/%s' % tuple(fmt(r['per_question'][w][k]) for k in ('noul_mae', 'noul_flip', 'choice_flip')) if w in r['per_question'] else '-':>22s}"
            for w in WF))


if __name__ == "__main__":
    main()
