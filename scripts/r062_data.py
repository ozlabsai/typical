#!/usr/bin/env python3
"""R062 data: Hebrew question pool for Qwen3.8-27B-think distillation, then the training mix.

  translate  ~12k English tasksource typed decisions (guychuk/typical-r060-data tasksource/train.jsonl; choice/noul/
             score incl. procedural, no wili, <= 4k tokens) -> natural Israeli Hebrew with claude-haiku-5-5 (CLI,
             sha256 cache in .context/r062/cache, concurrency 16, notional cap $40), round-robin over task families
             so a partial run is diverse. Deterministic QA (digits, option IDs, option keys, target in options) with
             one fix retry. Accepted rows -> .context/r062/ts_twins.jsonl (EN/HE/MX twins, shared meta.twin_id).
  labelin    every EN/HE/MX question of the pool (wire format: noul true/false) -> .context/r062/label_in.jsonl,
             HE/MX first. Decontaminated (13-gram + MinHash) before labelling.
  mix        Qwen labels -> soft targets (exact 0/1 smoothed to [.02,.98]) in serving-identical trainer format;
             ~60% HE/MX, ~20% EN twins, ~15% EN anchors (KL to base), ~5% images; 5% dev by twin_id/case.

  uv run --project .context/research_sources/WorkflowEvals --with datasketch python scripts/r062_data.py <cmd>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / ".context/r062"
try:  # translate/labelin only (laptop); `mix` runs on the pod with the stdlib alone
    import hejev_build as hj  # noqa: E402  (leaf walker, digit/ID QA, schema)
    import r053_teacher_label as r053  # noqa: E402  (cached `claude -p` + $ ledger)
    from r055_translate_pool import TRANSLATE_SYSTEM  # noqa: E402  (hejev prompt + no slash-gender forms)
    r053.CACHE = R2 / "cache"
except ImportError:
    hj = r053 = TRANSLATE_SYSTEM = None
TS = ROOT / ".context/r060/tasksource"
TRANSLATOR = "claude-haiku-5-5"
N_TS, MAX_TOK, SEED = 12_000, 4000, 62
_LOCK = threading.Lock()


def jsonl(path, tolerant=False):
    out = []
    if Path(path).exists():
        for line in Path(path).open():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if not tolerant:
                    raise
    return out


def rank(s: str) -> str:
    return hashlib.sha256(f"{SEED}:{s}".encode()).hexdigest()


def keys_of(q: dict) -> list[str]:
    c = q.get("criteria")
    return list(c) if isinstance(c, dict) else [str(i) for i in range(len(c))]


def to_wire(q: dict) -> dict:
    """Trainer/legacy question -> TypeSafe wire question (what jevbench/serving send): noul keys true/false;
    a noul option text that only restates yes/no is dropped (serving renders the bare Yes/No)."""
    q = {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
    if q["type"] == "noul":
        c = q["criteria"] or {}
        c = {"true": c.get("yes"), "false": c.get("no")} if "yes" in c else c
        q["criteria"] = {k: ("" if str(v or "").strip().lower().rstrip(".") in ("yes", "no", "true", "false", "כן", "לא") else v)
                         for k, v in c.items()}
    return q


# --------------------------------------------------------------------------- translate

def ts_pool() -> list[dict]:
    """English tasksource rows (no wili, <= MAX_TOK by chars/4), round-robin over task families (meta.source)."""
    fams = defaultdict(list)
    for r in jsonl(TS / "train.jsonl"):
        m = r["meta"]
        if m.get("lang") != "en" or "wili" in m["source"].lower():
            continue
        if len(json.dumps([r["state"], r["questions"]], ensure_ascii=False)) / 4 > MAX_TOK:
            continue
        r = {**r, "questions": {k: to_wire(q) for k, q in r["questions"].items()}}
        fams[m["source"]].append(r)
    queues = [sorted(v, key=lambda r: rank(r["id"])) for _, v in sorted(fams.items(), key=lambda kv: rank(kv[0]))]
    out = []
    while len(out) < N_TS and any(queues):
        for q in queues:
            if q and len(out) < N_TS:
                out.append(q.pop(0))
    return out


def rewrite_row(row: dict, fn):
    """hj.rewrite generalised to multi-question rows: state, then each question's instructions and option texts
    (never keys or labels), leaves visited in a fixed order as fn(key, text)."""
    n = 0

    def leaf(path, s):
        nonlocal n
        if not re.search(r"[A-Za-z]{2,}", s) or (path.startswith("state") and not re.search(r"\s", s.strip())):
            return s
        key = f"{n:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", path)[:40].strip("_")
        n += 1
        return fn(key, s)

    state = hj.walk(row["state"], leaf, "state")
    qs = {qid: {**q, "instructions": hj.walk(q.get("instructions") or "", leaf, f"q{i}.instructions"),
                "criteria": hj.walk(q["criteria"], leaf, f"q{i}.criteria")}
          for i, (qid, q) in enumerate(row["questions"].items())}
    return state, qs


def segments(row: dict) -> dict:
    segs = {}
    rewrite_row(row, lambda k, s: segs.setdefault(k, s))
    return segs


def det_issues(row: dict, src: dict, he: dict) -> list[str]:
    labels = sorted({k for q in row["questions"].values() for k in keys_of(q)})
    issues = hj.det_issues(src, he, labels)  # digits + snake_case option IDs verbatim
    issues += [f"{k}: missing" for k in src if not he.get(k)]
    issues += [f"{k}: instructions not translated" for k in src  # hj's own check keys on "NN_instructions"
               if "_instructions" in k and he.get(k) and not re.search(r"[א-ת]", he[k])]
    if not issues:  # option-key QA (R055 bug class): keys survive, every target is an option
        st, qs = rewrite_row(row, lambda k, s: he[k])
        for qid, q in qs.items():
            if keys_of(q) != keys_of(row["questions"][qid]):
                issues.append(f"{qid}: option keys changed")
            if str(row["targets"][qid]["label"]) not in keys_of(q):
                issues.append(f"{qid}: target not an option")
    return issues


def translate_row(row: dict) -> dict:
    src, prev = segments(row), None
    labels = sorted({k for q in row["questions"].values() for k in keys_of(q)})
    for attempt in range(2):
        user = (f"Item {row['id']} (question types {sorted({q['type'] for q in row['questions'].values()})}).\n"
                f"Option IDs (machine labels, keep verbatim wherever they appear): {json.dumps(labels, ensure_ascii=False)}\n"
                "Translate every value of this JSON object into Hebrew; return exactly the same keys.\n\n"
                + json.dumps(src, ensure_ascii=False, indent=1))
        if prev:
            user += ("\n\nA checker rejected a previous Hebrew translation of this item:\n"
                     + "\n".join(f"- {i}" for i in prev[1]) + "\n\nPrevious translation:\n"
                     + json.dumps(prev[0], ensure_ascii=False, indent=1)
                     + "\n\nReturn a corrected full translation that fixes every issue.")
        raw = r053.claude(TRANSLATOR, TRANSLATE_SYSTEM, user, effort="low", schema=hj.mirror_schema(src))
        he = raw.get("structured_output") or {}
        he = {k: v.replace("\\n", "\n") if isinstance(v, str) and "\\n" not in src.get(k, "") else v
              for k, v in he.items()}  # Haiku sometimes writes newlines as a literal backslash-n
        issues = det_issues(row, src, he)
        if not issues:
            return {"id": row["id"], "status": "ok", "attempt": attempt, "he": he}
        prev = (he, issues)
    return {"id": row["id"], "status": "det_fail", "issues": issues[:5]}


def make_twins(row: dict, he: dict) -> list[dict]:
    state_he, qs_he = rewrite_row(row, lambda k, s: he[k])
    tid = f"r062ts-{row['id']}"
    meta = {k: v for k, v in row["meta"].items() if isinstance(v, (str, int, float, bool))}
    meta.pop("anchor", None)
    out = []
    for lang, state, qs in (("en", row["state"], row["questions"]), ("he", state_he, qs_he),
                            ("mx", state_he, row["questions"])):
        out.append({"id": f"{tid}-{lang}", "state": state, "questions": qs, "targets": row["targets"],
                    "meta": {**meta, "twin_id": tid, "lang": lang, "translator": TRANSLATOR, "r062_source": "ts"}})
    return out


def cmd_translate(args):
    rows = ts_pool()
    print(f"pool {len(rows)} rows from {len({r['meta']['source'] for r in rows})} task families", flush=True)
    out_path = R2 / "ts_twins.jsonl"
    done = {r["meta"]["twin_id"] for r in jsonl(out_path, tolerant=True)}
    status, sink = Counter(), out_path.open("a")

    def job(row):
        if f"r062ts-{row['id']}" in done:
            return "done"
        try:
            res = translate_row(row)
        except r053.BudgetExceeded:
            return "budget"
        except Exception as exc:  # noqa: BLE001  counted; the row is retried on the next run (cache keeps the rest)
            print(f"error {row['id']}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            return "error"
        if res["status"] == "ok":
            with _LOCK:
                sink.write("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in make_twins(row, res["he"])))
                sink.flush()
        return res["status"]

    with ThreadPoolExecutor(args.workers) as ex:
        for i, s in enumerate(ex.map(job, rows), 1):
            status[s] += 1
            if i % 200 == 0 or i == len(rows):
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(rows)} {dict(status)} spent ${r053.LEDGER['usd']:.2f}",
                      flush=True)


# --------------------------------------------------------------------------- pool -> teacher label input

D1 = ROOT / ".context/he-train/d1"  # R055 D1 twins after the neg-label fix (sha 4f27e1ac = HF translated-twins/)
HT = R2 / "src/hebrew-train"  # guychuk/typical-hebrew-train native/ + open/
CFD = ROOT / ".context/r060/src/cfd/data/train.jsonl"  # zoom-ai/contested-fast-decisions (eval-only)
OPEN_PER_SOURCE = 1000  # MASSIVE / HeQ / sentiment: 3k so they don't dominate
LANG_ORDER = {"he": 0, "mx": 1, "en": 2}


def group_of(r: dict) -> str:
    return r["meta"].get("twin_id") or r["meta"].get("case_id") or r["id"]


def is_dev(group: str) -> bool:
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % 20 == 0  # 5% by twin_id / case


def wire_row(r: dict, **meta) -> dict:
    """Gold labels (D1/open targets, native meta.intended) ride along for the teacher-vs-gold agreement stat only."""
    gold = {q: str(t["label"]) for q, t in (r.get("targets") or {}).items()}
    gold.update({q: str(v) for q, v in ((r.get("meta") or {}).get("intended") or {}).items()})
    return {"id": r["id"], "state": r["state"], "questions": {k: to_wire(q) for k, q in r["questions"].items()},
            "targets": {q: {"label": {"yes": "true", "no": "false"}.get(g, g)} for q, g in gold.items()},
            "meta": {**{k: v for k, v in (r.get("meta") or {}).items() if isinstance(v, (str, int, float, bool))}, **meta}}


def pool() -> list[dict]:
    rows = [wire_row(r, r062_source="d1") for split in ("train", "dev") for r in jsonl(D1 / f"{split}.jsonl")]
    rows += jsonl(R2 / "ts_twins.jsonl", tolerant=True)  # wire already (to_wire before translation)
    rows += [wire_row(r, lang="he", r062_source="native", case_id=r["id"])
             for r in jsonl(HT / "native/native_unlabeled.jsonl")]
    by = defaultdict(list)
    for r in jsonl(HT / "open/open_train.jsonl"):
        by[r["meta"]["source"]].append(r)
    rows += [wire_row(r, lang="he", r062_source="open") for _, rs in sorted(by.items())
             for r in sorted(rs, key=lambda r: rank(r["id"]))[:OPEN_PER_SOURCE]]
    return rows


def decon():
    """13-gram + MinHash (char 5-gram Jaccard > .5) index over public JevBench (all of datasets/), WorkflowEvals
    cases, hejev-v0, the external Hebrew panel and contested-fast-decisions (R060 slice rules)."""
    import r055_hebrew_open as ho
    import r060_tasksource_slice as tsl
    items = tsl.test_items()
    items["contested"] = [ho.texts_of_task(r) for r in jsonl(CFD)]
    grams, (lsh, sigs) = tsl.gram_index(items), tsl.minhash_index(items)
    return lambda r: tsl.contaminated(tsl.row_texts(r), grams, lsh, sigs)


def clean(rows: list[dict], hit) -> tuple[list[dict], Counter]:
    bad = {}
    for r in rows:
        why = hit(r)
        if why:
            bad[group_of(r)] = why.split(":")[0] + ":" + why.split(":")[1].split("#")[0]
    return [r for r in rows if group_of(r) not in bad], Counter(bad.values())


def cmd_labelin(args):
    """All pool questions, decontaminated, in labelling order (whole twin groups, HE, MX, EN, groups by hash).
    Rewritten in full on every call (streaming: ts_twins.jsonl grows); the labeller skips answered (id, qid)."""
    rows, dropped = clean(pool(), decon())
    rows.sort(key=lambda r: (rank(group_of(r)), LANG_ORDER.get(r["meta"].get("lang"), 3)))
    tmp = R2 / "label_in.jsonl.tmp"
    tmp.write_text("".join(json.dumps({"id": r["id"], "state": r["state"], "questions": r["questions"]},
                                      ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(R2 / "label_in.jsonl")
    (R2 / "pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    # R055 already labelled native/ with this teacher and protocol (Qwen3.8-27B-FP8 think, system_one, vLLM 0.31):
    # reuse those answers where the wire question is unchanged, so the pod skips them.
    import r055_build_mix as bm
    nat = {r["id"]: r for r in jsonl(HT / "native/native_unlabeled.jsonl")}
    seed = [a for a in jsonl(HT / "native/native_labels.jsonl") if "answer" in a and a["id"] in nat
            and bm.wire_question(nat[a["id"]]["questions"][a["qid"]]) == to_wire(nat[a["id"]]["questions"][a["qid"]])]
    (R2 / "labels").mkdir(exist_ok=True)
    (R2 / "labels/seed_r055_native.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in seed))
    q = Counter()
    for r in rows:
        q[f"{r['meta']['r062_source']}/{r['meta'].get('lang')}"] += len(r["questions"])
    m = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": len(rows),
         "questions": sum(q.values()), "by_source_lang": dict(sorted(q.items())),
         "decon_dropped_groups": dict(dropped), "native_labels_reused": len(seed)}
    (R2 / "pool_manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(m, indent=1))


# --------------------------------------------------------------------------- mix

def smooth(t: list[float]) -> list[float]:
    """Exact 0/1 teacher distributions -> [.02, .98] (R056 rule: p <- .96 p + .04 / K)."""
    return [0.96 * x + 0.04 / len(t) for x in t] if any(x in (0.0, 1.0) for x in t) else t


def labelled(rows: list[dict], answers: dict, stats: Counter) -> list[dict]:
    """Wire rows + Qwen answers -> trainer rows whose prompts equal the serving prompt (r056_build.trainer_question);
    soft target = Qwen's distribution (smoothed), label = its argmax."""
    from r056_build import answer_probs, trainer_question, wire_items
    out = []
    for r in rows:
        qs, soft, tg = {}, {}, {}
        for qid, q in r["questions"].items():
            a = answers.get((r["id"], qid))
            try:
                t = answer_probs(q, a) if a else None
            except (KeyError, TypeError, ValueError):
                t = None
            if t is None:
                stats["unlabelled"] += 1
                continue
            stats["smoothed"] += any(x in (0.0, 1.0) for x in t)
            t = smooth(t)
            keys = [k for k, _ in wire_items(q)]
            qs[qid], soft[qid] = trainer_question(q), t
            tg[qid] = {"label": keys[max(range(len(t)), key=t.__getitem__)]}
            gold = (r.get("targets") or {}).get(qid, {}).get("label")
            if gold is not None:
                stats["gold_n"] += 1
                stats["gold_agree"] += str(gold) == tg[qid]["label"]
        if qs:
            out.append({"id": r["id"], "state": r["state"], "questions": qs, "targets": tg, "soft_targets": soft,
                        "meta": {**r["meta"], "teacher": "Qwen/Qwen3.8-27B-FP8 thinking"}})
    return out


def pres(rows: list[dict]) -> int:
    return sum(len(r["questions"]) for r in rows)


def cmd_prep(args):
    """Label-free parts of the mix, decontaminated here (laptop): EN tasksource anchors (never a translated row or a
    sibling of one; KL-to-base on the pod) and the R060/R056 ScienceQA image replay -> static/."""
    from r056_build import trainer_question, write
    T, hit = args.total, decon()
    pool_ts = ts_pool()
    skip = {r["id"] for r in pool_ts} | {r["meta"].get("group_id") for r in pool_ts}
    anchors, na = [], 0
    for r in sorted(jsonl(TS / "train.jsonl"), key=lambda r: rank(r["id"])):
        if na >= 0.15 * T:
            break
        m = r["meta"]
        if m.get("lang") != "en" or r["id"] in skip or m.get("group_id") in skip or "wili" in m["source"].lower():
            continue
        if hit(r):
            continue
        qs = {k: trainer_question(to_wire(q)) for k, q in r["questions"].items()}
        anchors.append({**r, "questions": qs, "meta": {**m, "anchor": True, "lang": "en", "r062_source": "anchor"}})
        na += len(qs)
    img = [{**r, "meta": {**r["meta"], "r062_source": "img"}}
           for r in jsonl(ROOT / ".context/r060/mix/img.jsonl")][: int(0.05 * T)]
    img, dropped = clean(img, hit)
    assert not any(hit(r) for r in anchors + img), "contamination left"
    out = R2 / "static"
    out.mkdir(exist_ok=True)
    print(json.dumps({"anchors.jsonl": write(out / "anchors.jsonl", anchors), "img.jsonl": write(out / "img.jsonl", img),
                      "anchor_questions": na, "img_dropped": dict(dropped)}))


def cmd_mix(args):
    """Runs on the pod (stdlib only): <root>/pool.jsonl (decontaminated by labelin) + <root>/labels/*.jsonl +
    <root>/static/ -> <root>/mix/."""
    from r056_build import write
    T, root = args.total, Path(args.root)
    answers = {}
    for f in sorted(root.glob("labels/*.jsonl")):
        for a in jsonl(f, tolerant=True):
            if "answer" in a:
                answers[(a["id"], a["qid"])] = a["answer"]
    stats = Counter()
    rows = labelled(jsonl(root / "pool.jsonl"), answers, stats)
    dev = [r for r in rows if is_dev(group_of(r)) and r["meta"].get("lang") in ("he", "mx")]
    dev = sorted(dev, key=lambda r: rank(r["id"]))
    dev_he, n = [], 0
    for r in dev:
        if n >= args.dev_questions:
            break
        dev_he.append(r)
        n += len(r["questions"])
    train = [r for r in rows if not is_dev(group_of(r))]
    he_only = [r for r in train if r["meta"]["r062_source"] in ("native", "open")]
    groups = defaultdict(dict)
    for r in train:
        if r["meta"]["r062_source"] in ("ts", "d1"):
            groups[r["meta"]["twin_id"]][r["meta"]["lang"]] = r
    order = {s: sorted((g for g, v in groups.items() if next(iter(v.values()))["meta"]["r062_source"] == s), key=rank)
             for s in ("ts", "d1")}
    seq = []  # ts : d1 = 2 : 1 (tasksource general decisions are closest in kind to Hebrew JevBench)
    while order["ts"] or order["d1"]:
        for s in ("ts", "ts", "d1"):
            if order[s]:
                seq.append(order[s].pop(0))
    picked, hemx, en = list(he_only), pres(he_only), 0
    for g in seq:
        if hemx >= 0.60 * T:
            break
        hm = [groups[g][l] for l in ("he", "mx") if l in groups[g]]
        if not hm:
            continue
        picked += hm
        hemx += pres(hm)
        if "en" in groups[g] and en < 0.20 * T:
            picked.append(groups[g]["en"])
            en += len(groups[g]["en"]["questions"])
    anchors, img = jsonl(root / "static/anchors.jsonl"), jsonl(root / "static/img.jsonl")
    random.Random(SEED).shuffle(picked)
    out = root / "mix"
    out.mkdir(exist_ok=True)
    everything = picked + anchors + img
    twin_keys = {(r["meta"]["twin_id"], q) for r in everything if r["meta"].get("twin_id") for q in r["questions"]}
    twin_views = sum(len(r["questions"]) for r in everything if r["meta"].get("twin_id"))
    count = lambda rs, key: dict(sorted(Counter(k for r in rs for k in [key(r)] * len(r["questions"])).items()))  # noqa: E731
    manifest = {
        "task": "R062 mix: Qwen3.8-27B-think soft labels on Hebrew/mixed/English twins + native + open; EN anchors "
                "(KL to base); ScienceQA image replay",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {"train_text.jsonl": write(out / "train_text.jsonl", picked),
                  "anchors.jsonl": write(out / "anchors.jsonl", anchors), "img.jsonl": write(out / "img.jsonl", img),
                  "dev_he.jsonl": write(out / "dev_he.jsonl", dev_he)},
        "presentations": pres(everything), "slots_with_twins": pres(everything) - twin_views + len(twin_keys),
        "share": {k: round(v / pres(everything), 3) for k, v in count(
            everything, lambda r: {"he": "he_mx", "mx": "he_mx"}.get(r["meta"].get("lang"), "en")
            if r["meta"]["r062_source"] not in ("anchor", "img") else r["meta"]["r062_source"]).items()},
        "train_by_source_lang": count(everything, lambda r: f"{r['meta']['r062_source']}/{r['meta'].get('lang')}"),
        "dev_he_by_source_lang": count(dev_he, lambda r: f"{r['meta']['r062_source']}/{r['meta'].get('lang')}"),
        "labels": {"answers": len(answers), **stats,
                   "teacher_gold_agreement": round(stats["gold_agree"] / max(1, stats["gold_n"]), 4)},
        "contamination": "pool decontaminated by `labelin`, anchors/images by `prep` (13-gram + MinHash char-5gram "
                         "Jaccard > .5 vs public JevBench (all datasets/), WorkflowEvals cases, hejev-v0, external "
                         "Hebrew panel, contested-fast-decisions); see pool_manifest.json",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("translate", "labelin", "prep", "mix"))
    ap.add_argument("--root", default=str(R2), help="mix: dir with pool.jsonl, labels/, static/")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--total", type=int, default=35_000, help="mix: target presentations")
    ap.add_argument("--dev-questions", type=int, default=1500)
    ap.add_argument("--max-usd", type=float, default=40.0, help="hard cap on notional CLI cost, all runs")
    args = ap.parse_args(argv)
    if args.cmd == "translate":
        r053.LEDGER.update(usd=hj.spent(), max_usd=args.max_usd)  # hj.spent() reads r053.CACHE (= R2/cache)
    {"translate": cmd_translate, "labelin": cmd_labelin, "prep": cmd_prep, "mix": cmd_mix}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
