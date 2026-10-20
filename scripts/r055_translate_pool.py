#!/usr/bin/env python3
"""R055 D1: Hebrew translated twins of our own English decision training rows -> .context/he-train/d1/

  translate  sample ~5k license-clean, eval-free rows (stratified by source and question type), translate
             state/instructions/option texts with claude-haiku-5-5; one fix retry on a deterministic
             digit/option-ID failure. Every CLI response is cached by sha256 (cache/), so reruns are free.
  build      deterministic QA -> 60-item claude-opus-5-5 spot-check -> 13-gram contamination drop against
             Hebrew JevBench v0 and public JevBench -> EN/HE/MX twin rows -> 95/5 split by twin_id.
             Safe to run on a partial cache (it uses whatever has been translated).

  uv run --project .context/research_sources/WorkflowEvals python scripts/r055_translate_pool.py translate
  uv run --project .context/research_sources/WorkflowEvals python scripts/r055_translate_pool.py build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hejev_build as hj  # noqa: E402  (translatable-leaf walker, digit/ID QA, translator prompt)
import r053_teacher_label as r053  # noqa: E402  (cached `claude -p` transport + $ ledger)

ROOT = hj.ROOT
OUT = ROOT / ".context" / "he-train" / "d1"
r053.CACHE = OUT / "cache"  # overrides hejev's cache; hj.spent() reads r053.CACHE, so it follows
TRANSLATOR, JUDGE = "claude-haiku-5-5", "claude-opus-5-5"
TRANSLATE_SYSTEM = hj.TRANSLATE_SYSTEM + """
Do not use slash gender forms (e.g. עובד/ת, אישר/ה); use the single natural generic form an Israeli writer would."""
MAX_STATE_TOKENS = 6000  # ponytail: chars/4 estimate, no tokenizer load
CONTAM = [*sorted((ROOT / ".context/hebrew-bench/hejev-v0").glob("*.jsonl")),
          *sorted((ROOT / ".context/jevbench-v154/datasets/public").glob("*.jsonl")),
          *sorted((ROOT / ".context/hebrew-bench/external").glob("*.jsonl"))]  # R055 external Hebrew panel

# Source quotas. Excluded on purpose: wf routing/categorical/quality (MASSIVE, SNIPS, DBpedia, Yelp, ... third-
# party text; MASSIVE-he is a guardrail test), u UNLI/AmbiEnt (academic-only / no license), kb sciq (CC BY-NC),
# logiqa (unclear), mmlu_aux (adjacent to Global-MMLU-he), r044 replay (tasksource mix incl. typed-decisions),
# r044 hebrew (MASSIVE) and visual (ScienceQA, NC), gemma-data val-public (JevBench).
QUOTA = {"wh": 1600, "wf": 1500, "u": 700, "kb": 600, "r044_prose": 200, "r044_counterfactual": 100,
         "s2_policy": 300}
LICENSE = {"wh": "repo-owned synthetic (decisionmix_v2)", "wf": "repo-owned synthetic (workflow_corpus)",
           "u": "repo-owned synthetic (decisionmix_v2_u)", "r044_prose": "repo-owned programmatic (r042)",
           "r044_counterfactual": "repo-owned programmatic (r043)",
           "s2_policy": "repo-owned programmatic (s2 policy counterfactual from data_wf)",
           "kb:arc_easy": "CC-BY-SA-4.0", "kb:arc_challenge": "CC-BY-SA-4.0", "kb:openbookqa": "Apache-2.0",
           "kb:commonsense_qa": "MIT", "kb:qasc": "CC-BY-4.0", "kb:aqua_rat": "Apache-2.0", "kb:medmcqa": "MIT"}
WF_FAMILIES = {"policy_permit", "action_select", "enum_extract", "support", "fact", "adequacy", "severity",
               "relevance", "completeness"}
U_TASKS = {"u_ordinal_confusion", "u_conflicting_sources", "u_partial_evidence", "u_noisy_sensor"}

# --------------------------------------------------------------------------- pool


def from_legacy(r: dict, rid: str, qtype: str) -> dict | None:
    """state/query/candidates/target -> trainer schema with one question `decision`."""
    if r.get("p_null") or r.get("image") or not isinstance(r.get("state"), str):
        return None
    cands, lines = r["candidates"], r["query"].split("\n")
    parts = re.split(r"\s{2,}", lines[-1].strip()) if len(lines) > 1 else []
    opts = dict(p.split(": ", 1) for p in parts if ": " in p)
    if len(opts) == len(parts) and set(cands) <= set(opts):  # "k1: desc  k2: desc", listed in any order
        instructions, keys, descs = "\n".join(lines[:-1]), cands, [opts[c] for c in cands]
    elif len(opts) == len(parts) == len(cands) and not set(cands) & set(opts):
        # renamed keys (yes/no -> true/false, 0/1 -> low/medium): same order as the query lists them
        instructions, keys, descs = "\n".join(lines[:-1]), cands, list(opts.values())
    elif opts:  # option list that does not match the candidates: ambiguous mapping, skip
        return None
    else:  # plain multiple choice: letter keys, candidate texts become option texts
        instructions, keys, descs = r["query"], [chr(65 + i) for i in range(len(cands))], cands
    if len(keys) > 26:
        return None
    label = keys[r["label"]]
    if qtype == "score":  # trainer score = list criteria, labels are indices
        criteria, label = [f"{k}: {d}" if k != d else d for k, d in zip(keys, descs)], str(r["label"])
    else:
        criteria = dict(zip(keys, descs))
    return {"id": rid, "state": r["state"],
            "questions": {"decision": {"type": qtype, "instructions": instructions, "criteria": criteria}},
            "targets": {"decision": {"label": label}}, "soft_targets": {"decision": list(r["target"])},
            "meta": {k: v for k, v in (r.get("meta") or {}).items() if isinstance(v, (str, int, float, bool))}}


def candidates() -> dict[str, list[dict]]:
    pool = defaultdict(list)
    for name, path in (("wf", "data_wf/train.jsonl"), ("wh", "data_wh/train.jsonl"), ("u", "data_u/train.jsonl"),
                       ("kb", "data_kb/train.jsonl")):
        for i, line in enumerate((ROOT / path).open()):
            r = json.loads(line)
            m = r.get("meta") or {}
            if name == "wf" and (m.get("family") not in WF_FAMILIES or m.get("source")):
                continue
            if name == "u" and r["task"] not in U_TASKS:
                continue
            if name == "kb" and f"kb:{r['task']}" not in LICENSE:
                continue
            qtype = "choice" if name == "kb" else (m.get("qtype") or m.get("decision_type"))
            row = from_legacy(r, f"d1-{name}-{i:06d}", qtype)
            if row:
                row["meta"].update(d1_source=name, d1_stratum=f"{qtype}/{r['task'] if name == 'kb' else m.get('family') or r['task']}",
                                   license=LICENSE.get(name) or LICENSE[f"kb:{r['task']}"])
                pool[name].append(row)
    for line in (ROOT / ".context/r044-data/arm-b.jsonl").open():
        r = json.loads(line)
        if r.get("pair_mode") == "counterfactual":
            name, r = "r044_counterfactual", r["paired_views"][0]
        elif (r.get("meta") or {}).get("source") == "r042_programmatic":
            name = "r044_prose"
        else:
            continue
        pool[name].append(trainer_row(r, name, r["meta"].get("family") or r["meta"].get("predicate", "")))
    for line in (ROOT / ".context/s2-matched-arms/arm-a.jsonl").open():
        r = json.loads(line)
        if (r.get("meta") or {}).get("source") == "programmatic_policy_counterfactual":
            pool["s2_policy"].append(trainer_row(r, "s2_policy", r["meta"]["context_view"]))
    return pool


def trainer_row(r: dict, name: str, sub: str) -> dict:
    (qid, q), = r["questions"].items()
    meta = {k: v for k, v in r["meta"].items() if isinstance(v, (str, int, float, bool))}
    meta.update(d1_source=name, d1_stratum=f"{q['type']}/{sub.split(':')[0]}", license=LICENSE[name])
    row = {"id": f"d1-{name}-{r['id']}", "state": r["state"], "questions": {"decision": q},
           "targets": {"decision": r["targets"][qid]}, "meta": meta}
    if r.get("soft_targets"):
        row["soft_targets"] = {"decision": r["soft_targets"][qid]}
    return row


def sample(seed: int = 55) -> list[dict]:
    """Per source: round-robin over strata (qtype/family) in seeded order; the global order interleaves
    sources by rank so a partial run is still stratified."""
    rank = lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()  # noqa: E731
    out = []
    for name, rows in sorted(candidates().items()):
        rows = [r for r in rows if len(json.dumps(r["state"], ensure_ascii=False)) / 4 <= MAX_STATE_TOKENS]
        strata = defaultdict(list)
        for r in sorted(rows, key=lambda r: rank(r["id"])):
            strata[r["meta"]["d1_stratum"]].append(r)
        picked, queues = [], [strata[k] for k in sorted(strata)]
        while len(picked) < QUOTA[name] and any(queues):
            for q in queues:
                if q and len(picked) < QUOTA[name]:
                    picked.append(q.pop(0))
        out += [(i / len(picked), r) for i, r in enumerate(picked)]
    return [r for _, r in sorted(out, key=lambda t: (t[0], rank(t[1]["id"])))]


# --------------------------------------------------------------------------- translate


def as_item(row: dict) -> dict:
    return {"state": row["state"], "question": row["questions"]["decision"]}


def labels(row: dict) -> list[str]:
    c = row["questions"]["decision"]["criteria"]
    return [str(i) for i in range(len(c))] if isinstance(c, list) else list(c)


def translate_row(row: dict) -> dict:
    src = hj.segments(as_item(row))
    prev = None
    for attempt in range(2):
        user = (f"Item {row['id']} (question type {row['questions']['decision']['type']}).\n"
                f"Option IDs (machine labels, keep verbatim wherever they appear): {json.dumps(labels(row))}\n"
                "Translate every value of this JSON object into Hebrew; return exactly the same keys.\n\n"
                + json.dumps(src, ensure_ascii=False, indent=1))
        if prev:
            user += ("\n\nA checker rejected a previous Hebrew translation of this item:\n"
                     + "\n".join(f"- {i}" for i in prev[1]) + "\n\nPrevious translation:\n"
                     + json.dumps(prev[0], ensure_ascii=False, indent=1)
                     + "\n\nReturn a corrected full translation that fixes every issue.")
        raw = r053.claude(TRANSLATOR, TRANSLATE_SYSTEM, user, effort="low", schema=hj.mirror_schema(src))
        he = raw.get("structured_output") or {}
        issues = hj.det_issues(src, he, labels(row)) + [f"{k}: missing" for k in src if not he.get(k)]
        if not issues:
            return {"id": row["id"], "status": "ok", "attempt": attempt, "he": he}
        prev = (he, issues)
    return {"id": row["id"], "status": "det_fail", "issues": issues[:5]}


def cmd_translate(args):
    rows = sample()[: args.limit or None]
    stop = datetime.fromisoformat(args.stop_utc).timestamp() if args.stop_utc else float("inf")
    done = Counter()

    def job(row):
        if time.time() > stop:
            return {"id": row["id"], "status": "skipped_deadline"}
        try:
            return translate_row(row)
        except r053.BudgetExceeded as exc:
            return {"id": row["id"], "status": "budget", "error": str(exc)}
        except Exception as exc:
            return {"id": row["id"], "status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}

    with ThreadPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(job, rows), 1):
            done[res["status"]] += 1
            if i % 100 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} {dict(done)} spent ${r053.LEDGER['usd']:.2f}", flush=True)


# --------------------------------------------------------------------------- build

JUDGE_SYSTEM = """You are a bilingual Hebrew-English QA reviewer for translated AI decision training items.
You get the English source fields of one item, their Hebrew translation, the option IDs and the gold answer.
Flag only material problems: (1) meaning drift that could change the correct answer (an added, dropped or
altered fact, condition, negation, quantifier, number or time relation); (2) answer leakage: the Hebrew makes the
gold answer more or less obvious than the English does. Do not flag style or kept English terms/names.
Return ok=true when there is no material issue; otherwise ok=false with one short issue per entry."""

NGRAM = 13


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def grams(text: str) -> set[int]:
    w = words(text)
    return {hash(" ".join(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def leaves(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in leaves(v)]
    if isinstance(x, list):
        return [s for v in x for s in leaves(v)]
    return []


def row_text(row: dict) -> str:
    return "\n".join(leaves(row["state"]) + leaves(row["questions"]))


def make_twins(row: dict, he: dict) -> list[dict]:
    state_he, q_he = hj.rewrite(as_item(row), lambda k, s: he[k])  # same leaf order/keys as hj.segments
    tid = row["id"]
    out = []
    for lang, state, q in (("en", row["state"], row["questions"]["decision"]), ("he", state_he, q_he),
                           ("mx", state_he, row["questions"]["decision"])):
        out.append({**row, "id": f"{tid}-{lang}", "state": state, "questions": {"decision": q},
                    "meta": {**row["meta"], "twin_id": tid, "lang": lang, "translator": TRANSLATOR}})
    return out


def cmd_build(args):
    rows = sample()
    r053.LEDGER["max_usd"] = -1.0  # cache-only replay: any uncached translator call raises BudgetExceeded
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(cached_translation, rows))
    r053.LEDGER["max_usd"] = args.max_usd
    qa = Counter(r["status"] for r in results)
    ok = [(row, res["he"]) for row, res in zip(rows, results) if res["status"] == "ok"]

    # spot-check: 60 random accepted items, opus judge
    rnd = random.Random(55)
    spot = rnd.sample(ok, min(60, len(ok)))

    def judge(pair):
        row, he = pair
        payload = {"option_ids": labels(row), "gold_answer": row["targets"]["decision"]["label"],
                   "english_source": hj.segments(as_item(row)), "hebrew_translation": he}
        raw = r053.claude(JUDGE, JUDGE_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=1),
                          effort="medium", schema=hj.JUDGE_SCHEMA)
        return {"id": row["id"], **(raw.get("structured_output") or {"ok": None, "issues": ["no output"]})}

    with ThreadPoolExecutor(8) as ex:
        verdicts = list(ex.map(judge, spot))
    flagged = [v for v in verdicts if v.get("ok") is False]

    # contamination
    ref = set()
    for path in CONTAM:
        for line in path.open():
            ref |= grams("\n".join(leaves(json.loads(line))))
    rows_out, contam, mnli = [], 0, 0
    for row, he in ok:
        if row["id"] in {v["id"] for v in flagged}:  # judge-flagged spot-check items are dropped too
            continue
        if row["meta"].get("family") == "support":  # wf support = SNLI+MultiNLI premises; HebNLI test is
            mnli += 1                                 # translated MultiNLI, so drop rather than risk overlap
            continue
        twins = make_twins(row, he)
        if any(grams(row_text(t)) & ref for t in twins[:2]):
            contam += 1
            continue
        rows_out += twins

    dev_ids = {r["meta"]["twin_id"] for r in rows_out
               if int(hashlib.sha256(r["meta"]["twin_id"].encode()).hexdigest(), 16) % 100 < 5}
    split = {"train": [r for r in rows_out if r["meta"]["twin_id"] not in dev_ids],
             "dev": [r for r in rows_out if r["meta"]["twin_id"] in dev_ids]}
    files = {}
    for name, rs in split.items():
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
        files[name] = {"rows": len(rs), "twins": len(rs) // 3, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "spot_check.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in verdicts))
    by = lambda key: dict(sorted(Counter(key(r) for r in rows_out).items()))  # noqa: E731
    manifest = {
        "task": "R055 D1 translated twins (EN/HE/MX); train-only, never eval",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "translator": TRANSLATOR, "judge": JUDGE, "sampled": len(rows), "quota": QUOTA,
        "translation_status": dict(qa), "contamination_dropped_twins": contam,
        "dropped_wf_support_mnli_risk_twins": mnli,
        "contamination_refs": [str(p.relative_to(ROOT)) for p in CONTAM], "ngram": NGRAM,
        "spot_check": {"n": len(verdicts), "flagged": len(flagged),
                       "flag_rate": round(len(flagged) / max(1, len(verdicts)), 3)},
        "files": files, "by_source": by(lambda r: r["meta"]["d1_source"]),
        "by_type": by(lambda r: r["questions"]["decision"]["type"]), "by_lang": by(lambda r: r["meta"]["lang"]),
        "by_source_type": by(lambda r: f"{r['meta']['d1_source']}/{r['questions']['decision']['type']}"),
        "licenses": LICENSE, "notional_cli_cost_usd": round(hj.spent(), 2), "max_usd": args.max_usd,
        "exclusions": "wf routing/categorical/quality (third-party text incl. MASSIVE), u UNLI/AmbiEnt, kb "
                      "sciq/logiqa/mmlu_aux, r044 replay/hebrew/visual, anything JevBench/WorkflowEvals/"
                      "typed-decisions derived; states > 6k tokens (chars/4).",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("translation_status", "contamination_dropped_twins", "dropped_wf_support_mnli_risk_twins", "spot_check",
                                               "files", "by_source", "by_type", "notional_cli_cost_usd")}, indent=1))


def cached_translation(row: dict) -> dict:
    """translate_row against the cache only; any uncached call -> `missing`."""
    try:
        return translate_row(row)
    except r053.BudgetExceeded:
        return {"id": row["id"], "status": "missing"}
    except Exception as exc:
        return {"id": row["id"], "status": "error", "error": str(exc)[:200]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("translate", "build"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stop-utc", default="", help="ISO time; translate stops starting new items after it")
    ap.add_argument("--max-usd", type=float, default=120.0, help="hard cap on notional CLI cost, all runs")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    r053.LEDGER.update(usd=hj.spent(), max_usd=args.max_usd)  # cap counts every cached response
    {"translate": cmd_translate, "build": cmd_build}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
