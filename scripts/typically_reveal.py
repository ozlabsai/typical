"""Base vs a tuned model on held-out cases. `reveal(rows, run_models)` is the scoring core, shared by this CLI
(-> site/data/typically_northwind.json, needs the API server on --url), site/server.py /api/typically/results, and the GPU pod
(`--job <slug> <base>`, run by scripts/typically_spike_pod.sh after eval: writes runs/co_<slug>/reveal.json, which
typically_job pulls back so the server never loads either model for it). `score` is the one forward pass both server and pod use.
Stdlib only at import; `score` / `--job` import torch + inference/typical.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = "data_co_a/eval/a_oneliner.jsonl"
COMPANY = "Northwind Freight"
sys.path.insert(0, str(ROOT / "inference"))
TOP = lambda v: {"answer": v["argmax"], "p": round(v["probs"][v["argmax"]], 2)}


def reveal(rows, run_models, company="", tuned=""):
    """rows: eval rows (one-liner questions per held-out case). run_models(state, rows) -> [standard_results, yours_results],
    each a list parallel to rows of {"probs", "p_null", "argmax"}. Decisions are keyed by task suffix."""
    cases = defaultdict(list)
    for r in rows:
        cases[r["state"]].append(r)
    sides = ("standard", "yours")
    hits, decisions, disagreements = {s: defaultdict(list) for s in sides}, {}, []
    # ponytail: accuracy = top answer among the offered options, same as the page shows (eval_wf's acc_k);
    # a p_null >= .5 "none of these fit" is counted separately instead of as a wrong answer
    abstained, fixed, broken = dict.fromkeys(sides, 0), 0, 0
    for state, group in cases.items():
        for r, *res in zip(group, *run_models(state, group)):
            key, gold = re.sub(r"^(co_[a-z0-9]+|import)_", "", r["task"]), r["candidates"][r["label"]]
            decisions[key] = {"key": key, "question": r["query"], "type": r["meta"]["qtype"], "labels": r["candidates"]}
            v = dict(zip(sides, res))
            ok = {s: v[s]["argmax"] == gold for s in sides}
            for s in sides:
                hits[s][key].append(ok[s])
                abstained[s] += v[s]["p_null"] >= 0.5
            fixed += ok["yours"] and not ok["standard"]
            broken += ok["standard"] and not ok["yours"]
            if v["standard"]["argmax"] != v["yours"]["argmax"]:
                disagreements.append({"case": state, "key": key, "question": r["query"], "decided": gold,
                                      **{s: TOP(v[s]) for s in sides}})
    acc = lambda xs: sum(xs) / len(xs)
    flat = lambda h: [x for xs in h.values() for x in xs]
    disagreements.sort(key=lambda d: (d["yours"]["answer"] != d["decided"], len(d["case"])))  # yours right first, shortest read best
    return {"company": company, "tuned": tuned, "n_cases": len(cases), "n_answers": len(flat(hits["standard"])),
            "score": {s: acc(flat(hits[s])) for s in sides},
            "decisions": [{**d, **{s: acc(hits[s][k]) for s in sides}} for k, d in decisions.items()],
            "fixed": fixed, "broken": broken, "abstained": abstained, "disagreements": disagreements[:30],
            "measured": f"{len(cases)} {company or 'held-out'} cases neither model saw while learning"}


def company(slug: str) -> str:
    return slug.replace("_", " ").title()


def queries(rows) -> list[tuple]:
    """Eval rows -> (type, question, labels) queries, as /compare's decisions."""
    return [(r["meta"]["qtype"], r["query"], r["candidates"]) for r in rows]


def score(m, state: str, qs: list[tuple]) -> list[dict]:
    """One model (a typical.Typical, already activate()d), one state, all (type, question, labels) queries in a single KV encode
    -> [{"probs", "p_null", "argmax"(, "expected")}]. ValueError on a query with < 2 labels. The caller serializes the device."""
    from typical.core import to_labels
    from typical.native import native_kv_decide
    qs = [(typ, q, ["no", "yes"] if typ == "noul" else labels) for typ, q, labels in qs]
    for typ, question, labels in qs:
        if len(labels) < 2:
            raise ValueError(f"'{typ}' query {question!r} needs >=2 labels")
    # the device->host copy in to_labels must stay inside the caller's lock too, or it races the next forward
    raws = native_kv_decide(m.head, m.model, state, [(q, l) for _, q, l in qs], max_state=m.max_state, max_suffix=2048)
    results = []
    for (typ, _, labels), raw in zip(qs, raws):
        probs, p_null = to_labels(raw, labels)
        entry = {"probs": probs, "p_null": p_null, "argmax": max(probs, key=probs.get)}
        if typ == "score":
            entry["expected"] = sum(i * probs[lv] for i, lv in enumerate(labels))
        results.append(entry)
    return results


def job(slug: str, base: str) -> None:
    """On the pod (cwd = the repo, after a job's training + eval): runs/base vs runs/co_<slug> on its held-out cases -> reveal.json,
    the same dict site/server.py caches."""
    from typical import Typical
    backbones = {}   # one shared backbone, LoRA swapped by activate(), as the server's get_model
    models = [Typical.from_pretrained(f"runs/{r}/best.pt", backbones=backbones) for r in ("base", f"co_{slug}")]

    def run_models(state, group):
        return [m.activate() or score(m, state, queries(group)) for m in models]
    rows = [json.loads(line) for line in open(f"data_co_{slug}/eval/import_oneliner.jsonl")]
    res = {**reveal(rows, run_models, company(slug), f"local:co_{slug}"), "base": f"typical-{base}"}
    Path(f"runs/co_{slug}/reveal.json").write_text(json.dumps(res))
    print(f"reveal: {res['n_cases']} cases, standard {res['score']['standard']:.3f}  yours {res['score']['yours']:.3f}")


def compare(url, state, rows, models):
    body = {"state": state, "models": models,
            "decisions": [{"type": r["meta"]["qtype"], "question": r["query"], "labels": r["candidates"]} for r in rows]}
    req = urllib.request.Request(url + "/api/typically/compare", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        return json.load(f)["models"]


def main():
    if sys.argv[1:2] == ["--job"]:
        return job(*sys.argv[2:4])
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", default="local:co_f")
    ap.add_argument("--url", default="http://localhost:8787")
    ap.add_argument("--out", default=str(ROOT / "site/data/typically_northwind.json"))
    a = ap.parse_args()
    models, t0, n = ["typical-small", a.tuned], time.time(), [0]

    def run_models(state, rows):
        out = compare(a.url, state, rows, models)
        n[0] += 1
        if n[0] % 20 == 0:
            print(f"{n[0]} cases, {time.time() - t0:.0f}s", file=sys.stderr)
        return [out[m]["results"] for m in models]

    result = reveal([json.loads(l) for l in (ROOT / EVAL).open()], run_models, COMPANY, a.tuned)
    result["source"] = f"scripts/typically_reveal.py; base = OzLabs/typical-small, yours = {a.tuned}, eval file {EVAL}"
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"runtime {time.time() - t0:.0f}s\nscore standard {result['score']['standard']:.3f}  yours {result['score']['yours']:.3f}")
    for d in result["decisions"]:
        print(f"  {d['key']:9s} standard {d['standard']:.3f}  yours {d['yours']:.3f}")
    print(f"fixed {result['fixed']}  broken {result['broken']}  abstained {result['abstained']}")


if __name__ == "__main__":
    main()
