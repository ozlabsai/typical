"""R061 BENCHMARK ROUTING DIAGNOSTIC: fast (R056 single pass) by default, deliberate (thinking) on the
top-k% most uncertain R056 questions. Replays the official WorkflowEvals programs offline on saved
per-question answers (r053 replay) and scores against labels-consensus.

k is swept ON THE BENCHMARK, so no row here is a model-selection result; a real routing threshold
must be fixed on synthetic dev data (R061 step 2).

Usage: uv run --project .context/research_sources/WorkflowEvals python scripts/r061_route_offline.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r053_calibration_headroom as r053  # noqa: E402  (replay, temper, HEADLINE, mean)

CTX, HEADLINE = r053.CTX, r053.HEADLINE
FAST = CTX / "r056" / "pod" / "wf"
SLOW = {"gemma": CTX / "r054-phase0" / "gemma4-12b-think", "qwen": CTX / "r054-phase0" / "qwen38-27b-think"}
KS = (0, 5, 10, 20, 30, 50, 100)  # fixed before looking at any result
MODES = ("replace", "average")
DECODE_1 = 75.0  # ponytail: assumed single-stream Gemma-12B decode tok/s on one H100 (not measured here)
DECODE_B = 2000.0  # batched tok/s, P0a vLLM log at ~100 running requests
LABEL = "BENCHMARK DIAGNOSTIC (k swept on WorkflowEvals labels; not a routing-selection procedure)"


class Seen(dict):
    """Answer lookup that records which keys the replayed program actually asked."""

    def __init__(self, *a):
        super().__init__(*a)
        self.asked = set()

    def __getitem__(self, key):
        self.asked.add(key)
        return super().__getitem__(key)


def load(path: Path):
    """case_id -> {(node, state_hash, qid): wire answer}, plus (case, key) -> (output_tokens, wall_s)."""
    answers, cost = {}, {}
    for case in json.loads(path.read_text())["runs"][0]["cases"]:
        for call in case["calls"]:
            for q, a in ((call.get("response") or {}).get("answers") or {}).items():
                key = (call["node_id"], call["state_hash"], q)
                if a["type"] == "score":
                    a.setdefault("legend", {})  # saved runs drop it; SystemOneResponse requires it
                answers.setdefault(case["case_id"], {})[key] = a
                cost[case["case_id"], key] = (call.get("output_tokens") or 0, call.get("wall_time_s") or 0.0)
    return answers, cost


def probs(a: dict) -> list[float]:
    return [a["noul"], 1 - a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def entropy(a: dict, _q: str) -> float:
    p = probs(a)
    return -sum(x * math.log(x) for x in p if x > 0) / math.log(len(p)) if len(p) > 1 else 0.0


def margin(a: dict, _q: str) -> float:
    p = sorted(probs(a), reverse=True) + [0.0]
    return 1 - (p[0] - p[1])


def threshold(a: dict, q: str) -> float:
    """noul: closeness to the program gate; choice/score: margin."""
    if a["type"] != "noul":
        return margin(a, q)
    # ponytail: security is_true_positive bands + a generic 0.5 gate; per-policy floors (.6/.75/.85) not encoded
    gates = (0.15, 0.60, 0.75) if q == "is_true_positive" else (0.5,)
    return 1 - 2 * min(abs(a["noul"] - t) for t in gates)


SIGNALS = {"entropy": entropy, "margin": margin, "threshold": threshold}


def average(f: dict, s: dict) -> dict:
    if f["type"] == "noul":
        return {"type": "noul", "noul": (f["noul"] + s["noul"]) / 2}
    keys = f["probabilities"].keys() | s["probabilities"].keys()
    p = {k: (f["probabilities"].get(k, 0) + s["probabilities"].get(k, 0)) / 2 for k in keys}
    return r053.temper({"type": f["type"], "probabilities": p}, 1)  # rebuilds choice/score/confidence


def route(fast: dict, slow: dict, signal, k: int, mode: str):
    """Routed lookup + the routed (case, key) set. Routed keys with no thinking answer are dropped,
    so the program raises on them: unseen branch = wrong."""
    pool = sorted(((signal(a, key[2]), c, key) for c, d in fast.items() for key, a in d.items()), reverse=True)
    top = {(c, key) for _, c, key in pool[: round(len(pool) * k / 100)]}
    if k == 100 and mode == "replace":  # pure thinking run: its own branches are all answerable
        return {c: Seen(d) for c, d in slow.items()}, {(c, key) for c, d in slow.items() for key in d}, 0
    out, missing = {}, 0
    for c, d in fast.items():
        m = out[c] = Seen()
        for key, a in d.items():
            if (c, key) in top:
                t = slow.get(c, {}).get(key)
                if t is None:
                    missing += 1
                    continue
                a = t if mode == "replace" else average(a, t)
            m[key] = a
    return out, top, missing


def evaluate(w, wf, fast, slow, slow_cost, signal, k, mode, fills):
    """fills: (R056-alone, thinking-alone) case_id -> CaseResult, substituted for cases that hit an unseen
    branch to bracket the unseen-as-wrong lower bound with two counterfactuals."""
    lookup, top, missing = route(fast, slow, signal, k, mode)
    cases, errs = r053.execute_cases(wf, lookup, None)
    s = r053.score_cases(wf, cases)
    alt = [r053.score_cases(wf, [f[c.case_id] if c.case_id in errs else c for c in cases])[HEADLINE[w]] if errs
           else s[HEADLINE[w]] for f in fills]
    asked = {(c, key) for c, d in lookup.items() for key in d.asked}
    routed = asked & top
    tok = [slow_cost[x][0] for x in routed if x in slow_cost]
    return {"score": s[HEADLINE[w]], "invoice_primary": s.get("primary_action"),
            "case_errors": len(errs), "routed_missing_thinking": missing,
            "score_unseen_as_fast": alt[0], "score_unseen_as_think": alt[1],
            "asked": len(asked), "routed": len(routed), "routed_tokens": sum(tok),
            "routed_contended_wall_s": sum(slow_cost[x][1] for x in routed if x in slow_cost)}


def sweep(workflows, fast, slow, slow_cost):
    fills = {w: tuple({c.case_id: c for c in r053.execute_cases(workflows[w], x[w], None)[0]} for x in (fast, slow))
             for w in HEADLINE}
    rows = {}
    for sig in SIGNALS:
        for mode in MODES:
            for k in KS:
                if k == 0 and (sig, mode) != ("entropy", "replace"):
                    continue  # identical to the k=0 row
                per = {w: evaluate(w, workflows[w], fast[w], slow[w], slow_cost[w], SIGNALS[sig], k, mode, fills[w])
                       for w in HEADLINE}
                rows[f"{sig}/{mode}/{k}"] = {**per, **{m: sum(per[w][f] for w in HEADLINE) / len(HEADLINE) for m, f in (
                    ("mean", "score"), ("mean_unseen_as_fast", "score_unseen_as_fast"),
                    ("mean_unseen_as_think", "score_unseen_as_think"))}}
                print(f"  {sig:9s} {mode:7s} k={k:3d}  mean {rows[f'{sig}/{mode}/{k}']['mean']:.4f}", flush=True)
    return rows


def table(rows: dict, ncases: dict) -> str:
    head = ("| signal / mode / k | security | trace | invoice | invoice prim. | customer | **mean** | routed % of asked "
            "| think tok / case | case errors (routed w/o think ans.) | mean, errored cases as R056-alone / think-alone |"
            "\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    out = []
    for name, r in rows.items():
        asked = sum(r[w]["asked"] for w in HEADLINE)
        routed = sum(r[w]["routed"] for w in HEADLINE)
        tok = sum(r[w]["routed_tokens"] for w in HEADLINE) / sum(ncases.values())
        errs = sum(r[w]["case_errors"] for w in HEADLINE)
        miss = sum(r[w]["routed_missing_thinking"] for w in HEADLINE)
        cells = [f"{r[w]['score']:.3f}" for w in HEADLINE]
        cells.insert(3, f"{r['invoice_processing']['invoice_primary']:.3f}")
        out.append(f"| {name} | " + " | ".join(cells) + f" | **{r['mean']:.4f}** | {100 * routed / max(asked, 1):.1f} "
                   f"| {tok:.0f} | {errs} ({miss}) | {r['mean_unseen_as_fast']:.4f} / {r['mean_unseen_as_think']:.4f} |")
    return head + "\n".join(out) + "\n"


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(r053.WE))
    from core.workflow import load as load_wf

    workflows = {w: load_wf(w) for w in HEADLINE}
    fast, slow, slow_cost = {}, {"gemma": {}, "qwen": {}}, {"gemma": {}, "qwen": {}}
    for w in HEADLINE:
        fast[w], _ = load(FAST / w / "results.json")
        for t in SLOW:
            slow[t][w], slow_cost[t][w] = load(SLOW[t] / w / "results.json")
        cov = sum(key in slow["gemma"][w].get(c, {}) for c, d in fast[w].items() for key in d)
        print(f"{w}: R056 keys {sum(map(len, fast[w].values()))}, with a Gemma-think answer {cov}", flush=True)

    out = {"label": LABEL, "headline_metric": HEADLINE, "ks": KS, "decode_tok_s_single": DECODE_1,
           "decode_tok_s_batched": DECODE_B}
    print("full set, R056 -> Gemma-think")
    out["gemma_full"] = sweep(workflows, fast, slow["gemma"], slow_cost["gemma"])
    ncases_full = {w: len(workflows[w].bundle.cases) for w in HEADLINE}

    # Qwen-think 235-case subset: restrict the bundles, the R056 pool and both thinking lookups to its cases.
    sub_ids = {w: {c["case_id"] for c in json.loads((SLOW["qwen"] / w / "results.json").read_text())["runs"][0]["cases"]}
               for w in HEADLINE}
    sub_wf = {}
    for w, wf in workflows.items():
        bundle = wf.bundle.model_copy(update={"cases": [c for c in wf.bundle.cases if c.case_id in sub_ids[w]]})
        sub_wf[w] = SimpleNamespace(bundle=bundle, execute=wf.execute)
    pick = lambda d: {w: {c: v for c, v in d[w].items() if c in sub_ids[w]} for w in HEADLINE}  # noqa: E731
    ncases_sub = {w: len(sub_wf[w].bundle.cases) for w in HEADLINE}
    print(f"subset cases {ncases_sub}")
    for t in ("qwen", "gemma"):
        print(f"subset, R056 -> {t}-think")
        out[f"{t}_subset"] = sweep(sub_wf, pick(fast), pick(slow[t]), slow_cost[t])

    dest = CTX / "r061-route-offline"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(out, indent=1, default=list) + "\n")
    md = (f"# R061 offline routing\n\n**{LABEL}.**\n\n"
          f"## Full WorkflowEvals ({sum(ncases_full.values())} cases), R056 -> Gemma-4-12B think\n\n"
          + table(out["gemma_full"], ncases_full)
          + f"\n## Qwen subset ({sum(ncases_sub.values())} cases), R056 -> Qwen3.8-27B think\n\n"
          + table(out["qwen_subset"], ncases_sub)
          + "\n## Same subset, R056 -> Gemma think\n\n" + table(out["gemma_subset"], ncases_sub)
          + "\nGenerated by `scripts/r061_route_offline.py`.\n")
    (dest / "README.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
