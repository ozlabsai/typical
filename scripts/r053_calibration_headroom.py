"""R053 ORACLE HEADROOM DIAGNOSTIC: temperature-scale saved per-question answers,
replay the WorkflowEvals programs offline and score against labels-consensus.

T is picked on the evaluation labels themselves, so the "best T" rows are an
upper bound on what calibration alone could buy, NOT a model-selection procedure.

Usage: uv run --project .context/research_sources/WorkflowEvals python scripts/r053_calibration_headroom.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / ".context"
WE = CTX / "research_sources" / "WorkflowEvals"
RUNS = {"R044": CTX / "r044-workflows", "Cygnet": CTX / "g0-cygnet", "Winnow": CTX / "g0-winnow"}
# Headline metric per workflow (invoice also reports primary_action).
HEADLINE = {
    "security_incidents": "exact_actions",
    "agent_trace_observability": "primary_action",
    "invoice_processing": "exact_actions",
    "customer_service": "exact_actions",
}
GLOBAL_T = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
TYPE_T = [0.5, 1, 1.5, 2, 3]
EPS = 1e-6
LABEL = "ORACLE HEADROOM DIAGNOSTIC (T chosen on eval labels; not a model-selection procedure)"


def temper(answer: dict, t: float) -> dict:
    """Temperature-scale one wire answer {"type", ...} the way the harness consumes it."""
    if answer["type"] == "noul":
        p = min(max(answer["noul"], EPS), 1 - EPS)
        return {"type": "noul", "noul": 1 / (1 + math.exp(-math.log(p / (1 - p)) / t))}
    raw = {k: v ** (1 / t) for k, v in answer["probabilities"].items()}
    z = sum(raw.values())
    probs = {k: v / z for k, v in raw.items()}
    k, top = len(probs), max(probs.values())
    out = {"type": answer["type"], "probabilities": probs,
           "confidence": (top - 1 / k) / (1 - 1 / k) if k > 1 else 1.0}
    if answer["type"] == "choice":
        out["choice"] = max(probs, key=probs.get)
    else:
        out["score"] = sum(int(s) * v for s, v in probs.items())
        out["legend"] = {}
    return out


def run_answers(results_path: Path) -> dict[str, dict]:
    """case_id -> {(node, state_hash, qid): wire answer} from a saved run's calls."""
    run = json.loads(results_path.read_text())["runs"][0]
    return {
        c["case_id"]: {(call["node_id"], call["state_hash"], q): a
                       for call in c["calls"] for q, a in call["response"]["answers"].items()}
        for c in run["cases"]
    }


def teacher_answers(workflow: str, role: str) -> dict[str, dict]:
    """Same lookup built from the HF questions.parquet for one labeler (openai/anthropic/consensus)."""
    import pyarrow.parquet as pq
    from core.contract import fingerprint
    from core.dataset import DATASETS
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASETS[workflow], "data/questions.parquet", repo_type="dataset")
    out: dict[str, dict] = {}
    for row in pq.read_table(path).to_pylist():
        a = row[role]
        if a is None or a["status"] != "answered":
            continue
        probs = {p["option"]: p["probability"] for p in a["probabilities"]}
        if row["kind"] == "noul":
            wire = {"type": "noul", "noul": probs["true"]}
        elif row["kind"] == "choice":
            # answer_json is null on ties; the harness choice is the (first) argmax then.
            choice = json.loads(a["answer_json"] or "null") or max(probs, key=probs.get)
            wire = {"type": "choice", "choice": choice,
                    "confidence": a["confidence"], "probabilities": probs}
        else:
            wire = {"type": "score", "score": a["expected_score"], "confidence": a["confidence"],
                    "probabilities": probs, "legend": {}}
        key = (row["node_id"], fingerprint(json.loads(row["state_json"])), row["question_id"])
        out.setdefault(row["case_id"], {})[key] = wire
    return out


class ReplaySession:
    """Stands in for core.session.Session: answers come from a lookup instead of a model."""

    def __init__(self, answers: dict, policy_ids: list[str], temps: dict | None):
        from types import SimpleNamespace
        self.answers, self.policy_ids, self.temps = answers, policy_ids, temps
        self.node = "semantic"
        self.config = SimpleNamespace(model="replay", thinking=None)

    def system_one(self, _model, state, questions):
        from core.contract import fingerprint
        from core.question_types import SystemOneResponse
        h = fingerprint(state)
        answers = {q: self.answers[(self.node, h, q)] for q in questions}  # KeyError = unseen branch
        if self.temps:
            answers = {q: temper(a, self.temps[a["type"]]) for q, a in answers.items()}
        blob = {"model": "replay", "usage": {"input_tokens": 0, "output_tokens": 0}, "answers": answers}
        return SystemOneResponse.model_validate_json(json.dumps(blob))


def replay(workflow, lookup: dict[str, dict], temps: dict | None) -> dict:
    """Run the official workflow program on stored answers; score with core.scoring."""
    from core.contract import CaseResult, Decision, ModelIdentity, Run
    from core.scoring import build_score_data

    bundle = workflow.bundle
    policy_ids = [p.policy_id for p in bundle.policies]
    cases, errors = [], 0
    for case in bundle.cases:
        try:
            result = workflow.execute(case.case_id, ReplaySession(lookup.get(case.case_id, {}), policy_ids, temps))
        except Exception as exc:  # ponytail: unseen branch/missing answer counts as wrong, like the runner
            errors += 1
            result = CaseResult(case_id=case.case_id, decisions=[
                Decision.failed(p.policy_id, p.revision, repr(exc)) for p in bundle.policies])
        result.case_hash, result.input_status = case.content_hash, "verified"
        cases.append(result)
    scored = bundle.model_copy(update={"runs": [Run(run_id="replay", model=ModelIdentity(name="replay"), cases=cases)]})
    series = build_score_data(scored, label_ids=["labels-consensus"], model_selection="all").series[0]
    return {**series.metrics, "decisions": series.decisions, "case_errors": errors}


def mean(scores: dict) -> float:
    return sum(scores[w][m] for w, m in HEADLINE.items()) / len(HEADLINE)


def fmt(scores: dict) -> str:
    cells = [f"{scores[w][m]:.4f}" for w, m in HEADLINE.items()]
    cells.insert(3, f"{scores['invoice_processing']['primary_action']:.4f}")
    errs = sum(scores[w]["case_errors"] for w in HEADLINE)
    return " | ".join(cells + [f"**{mean(scores):.4f}**", str(errs)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CTX / "r053-calibration-headroom")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(WE))
    from core.workflow import load

    workflows = {w: load(w) for w in HEADLINE}
    started = time.time()
    official = {name: {w: json.loads((d / w / "scores.json").read_text()) for w in HEADLINE} for name, d in RUNS.items()}
    for name, by_wf in official.items():
        for w, s in by_wf.items():
            assert s["dataset_revision"] == workflows[w].bundle.dataset_revision, (name, w)
    official = {name: {w: next(x["metrics"] for x in s["series"] if x["label_set_id"] == "labels-consensus")
                       for w, s in by_wf.items()} for name, by_wf in official.items()}

    out = {"label": LABEL, "headline_metric": HEADLINE, "mean": "equal-workflow mean of headline metrics",
           "official_t1": official, "global": {}, "best_global": {}}
    for name, d in RUNS.items():
        lookups = {w: run_answers(d / w / "results.json") for w in HEADLINE}
        sweep = {t: {w: replay(workflows[w], lookups[w], dict.fromkeys(("noul", "choice", "score"), t))
                     for w in HEADLINE} for t in GLOBAL_T}
        out["global"][name] = {str(t): {**s, "mean": mean(s)} for t, s in sweep.items()}
        best = max(GLOBAL_T, key=lambda t: mean(sweep[t]))
        out["best_global"][name] = {"T": best, "mean": mean(sweep[best]), "t1_mean": mean(sweep[1])}
        print(f"{name}: T=1 {mean(sweep[1]):.4f}  best T={best} {mean(sweep[best]):.4f}  ({time.time() - started:.0f}s)")
        if name == "R044":
            # Memoize on the temperatures of the question types each workflow actually asks.
            kinds = {w: sorted({a["type"] for case in lookups[w].values() for a in case.values()}) for w in HEADLINE}
            cache, per_type = {}, []
            for tn, tc, ts in itertools.product(TYPE_T, repeat=3):
                temps = {"noul": tn, "choice": tc, "score": ts}
                s = {}
                for w in HEADLINE:
                    key = (w, *(temps[k] for k in kinds[w]))
                    if key not in cache:
                        cache[key] = replay(workflows[w], lookups[w], temps)
                    s[w] = cache[key]
                per_type.append({"temps": temps, **s, "mean": mean(s)})
            per_type.sort(key=lambda r: -r["mean"])
            out["per_type_R044"] = per_type
            print(f"R044 per-type best {per_type[0]['temps']} {per_type[0]['mean']:.4f}  ({time.time() - started:.0f}s)")

    out["teacher_ceiling"] = {
        role: {**(s := {w: replay(workflows[w], teacher_answers(w, role), None) for w in HEADLINE}), "mean": mean(s)}
        for role in ("openai", "anthropic", "consensus")
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    head = "| row | security exact | agent_trace primary | invoice exact | invoice primary | customer exact | mean | case errors |\n|---|---|---|---|---|---|---|---|\n"
    rows = [f"| R044 T={t} | {fmt(out['global']['R044'][str(t)])} |" for t in GLOBAL_T]
    for name in ("Cygnet", "Winnow"):
        best = out["best_global"][name]["T"]
        rows += [f"| {name} T=1 | {fmt(out['global'][name]['1'])} |",
                 f"| {name} best T={best} | {fmt(out['global'][name][str(best)])} |"]
    top = out["per_type_R044"][0]
    rows.append(f"| R044 per-type best {top['temps']} | {fmt(top)} |")
    rows += [f"| teacher {r} only | {fmt(out['teacher_ceiling'][r])} |" for r in ("openai", "anthropic", "consensus")]
    (args.out / "README.md").write_text(
        f"# R053 calibration headroom\n\n**{LABEL}.**\n\n"
        "Saved per-question answers are temperature-scaled (noul: sigmoid(logit(p)/T); choice/score: p^(1/T) "
        "renormalized, argmax/expectation and confidence=(pmax-1/K)/(1-1/K) recomputed), the official "
        "WorkflowEvals programs are replayed offline and scored with core.scoring against labels-consensus. "
        "Answers missing for a branch the transformed run takes count as wrong (`case errors`; per-workflow "
        "counts in results.json), so T!=1 cells with errors are lower bounds (mostly customer_service at T>1, "
        "security at T<1). invoice_processing is T-invariant: its thresholds are all 0.5/argmax. "
        "Teacher rows replay one labeler's published answers (consensus row = self-replay sanity check).\n\n"
        f"Mean = equal-workflow mean of {HEADLINE}.\n\n" + head + "\n".join(rows) + "\n\n"
        "Generated by `scripts/r053_calibration_headroom.py`; full numbers in `results.json`.\n")
    print(head + "\n".join(rows))


if __name__ == "__main__":
    main()
