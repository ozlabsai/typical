#!/usr/bin/env python3
"""R055 GPU half: Qwen labels for the native Hebrew cases, then the training mix (.context/r055/mix/).

  label  (pod, stdlib only) POST every native question, one per request, to r054_systemone_server.py `llm`
         shims (Qwen3.8-27B-FP8 thinking); append raw answers to --out; resumable; stops at --deadline.
         python3 scripts/r055_build_mix.py label --data native_unlabeled.jsonl --out native_labels.jsonl \
             --endpoint http://127.0.0.1:8010 --endpoint http://127.0.0.1:8011 --deadline 2026-10-21T00:30:00Z
  build  (local) D1 twins + D2 open (cap 4k) + Qwen-agreeing native + EN replay + 200 image rows -> train.jsonl,
         dev_he.jsonl / dev_en.jsonl (D1 dev + 10% native by case), images/, manifest.json; 13-gram assert vs
         hejev-v0 + external panel.
         uv run --project .context/research_sources/WorkflowEvals --with pandas python scripts/r055_build_mix.py build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D1 = ROOT / ".context/he-train/d1"
D2 = ROOT / ".context/he-train/d2"
R044 = ROOT / ".context/r044-data"
OUT = ROOT / ".context/r055/mix"
SEED = 55
PRESENTATIONS = 20_000
SLOTS_PER_STEP = 8  # --microbatch 1 --grad-accum 8


def jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {"rows": len(rows), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def bucket(key: str, mod: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % mod


# --------------------------------------------------------------------------- label

def wire_question(q: dict) -> dict:
    """Trainer schema -> TypeSafe wire: noul criteria are true/false, not yes/no."""
    if q["type"] == "noul" and "yes" in (q.get("criteria") or {}):  # already-wire rows (R056 nodes) pass through
        return {**q, "criteria": {"true": q["criteria"]["yes"], "false": q["criteria"]["no"]}}
    return q


def soft_target(q: dict, answer: dict) -> list[float]:
    """Answer -> probabilities in trainer option order (noul: no, yes; score: index)."""
    if q["type"] == "noul":
        p = float(answer["noul"])
        return [1 - p if k == "no" else p for k in q["criteria"]]
    keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    probs = answer["probabilities"]
    return [float(probs.get(k, 0.0)) for k in keys]


def cmd_label(args):
    rows = jsonl(args.data)
    done = set()
    if Path(args.out).exists():
        done = {(r["id"], r["qid"]) for r in jsonl(args.out) if "answer" in r}
    jobs = [(r, qid) for r in rows for qid in r["questions"] if (r["id"], qid) not in done]
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00")).timestamp()
    print(f"{len(jobs)} questions to label ({len(done)} done)", flush=True)
    sink = open(args.out, "a")

    def one(job_index):
        row, qid = jobs[job_index]
        if time.time() > deadline:
            return "skipped"
        endpoint = args.endpoint[job_index % len(args.endpoint)]
        body = json.dumps({"state": row["state"], "questions": {qid: wire_question(row["questions"][qid])}}).encode()
        for attempt in range(4):  # ponytail: ThreadingHTTPServer backlog is 5; bursts reset, so retry
            try:
                req = urllib.request.Request(f"{endpoint}/v1/systemone", body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    answer = json.loads(resp.read())["answers"][qid]
                record = {"id": row["id"], "qid": qid, "answer": answer}
                break
            except Exception as exc:  # recorded, never hidden; unlabeled questions are simply not kept
                record = {"id": row["id"], "qid": qid, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                time.sleep(2 + 3 * attempt)
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        return "ok" if "answer" in record else "error"

    with ThreadPoolExecutor(args.workers) as pool:
        stats = Counter(pool.map(one, range(len(jobs))))
    print(json.dumps(stats), flush=True)


# --------------------------------------------------------------------------- build

def label_native(rows, labels):
    """Keep a question only where Qwen's argmax == meta.intended; soft target = Qwen's distribution."""
    answers = {(r["id"], r["qid"]): r["answer"] for r in labels if "answer" in r}
    kept, stats = [], Counter()
    for row in rows:
        questions, targets, soft = {}, {}, {}
        for qid, q in row["questions"].items():
            answer = answers.get((row["id"], qid))
            if answer is None:
                stats["unlabeled"] += 1
                continue
            probs = soft_target(q, answer)
            keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else [str(i) for i in range(len(q["criteria"]))]
            intended = row["meta"]["intended"][qid]
            stats["labeled"] += 1
            if sum(probs) <= 0 or keys[max(range(len(keys)), key=probs.__getitem__)] != intended:
                stats["disagree"] += 1
                continue
            stats["agree"] += 1
            questions[qid], targets[qid], soft[qid] = q, {"label": intended}, probs
        if questions:
            meta = {k: v for k, v in row["meta"].items() if k not in ("grounding", "needs_label", "rationale")}
            kept.append({"id": row["id"], "state": row["state"], "questions": questions, "targets": targets,
                         "soft_targets": soft, "meta": {**meta, "labeler": "Qwen/Qwen3.8-27B-FP8 thinking"}})
    return kept, stats


def round_robin(groups: dict, n: int, rng: random.Random) -> list:
    queues = [rng.sample(v, len(v)) for _, v in sorted(groups.items())]
    out = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def cmd_build(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    import r055_hebrew_open as ho  # 13-gram index rules shared with D2 decon
    import r055_translate_pool as tp  # D1 source pools

    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    d1_train, d1_dev = jsonl(D1 / "train.jsonl"), jsonl(D1 / "dev.jsonl")
    d1_ids = {r["meta"]["twin_id"] for r in d1_train + d1_dev}

    opened = defaultdict(list)
    for r in jsonl(D2 / "open_train.jsonl"):
        opened[r["meta"]["source"]].append(r)
    open_rows = round_robin(opened, args.open_cap, rng)

    native, native_stats = [], {}
    if (D2 / "native_unlabeled.jsonl").exists() and Path(args.labels).exists():
        native, native_stats = label_native(jsonl(D2 / "native_unlabeled.jsonl"), jsonl(args.labels))
    native_dev = [r for r in native if bucket(r["id"], 10) == 0]
    native_train = [r for r in native if bucket(r["id"], 10) != 0]

    arm_b = jsonl(R044 / "arm-b.jsonl")
    images = [r for r in arm_b if r.get("image")]
    r044_text = [r for r in arm_b if not r.get("image") and not r.get("paired_views")
                 and f"d1-r044_prose-{r['id']}" not in d1_ids]
    for r in images + r044_text:
        r["meta"] = {**r.get("meta", {}), "lang": r.get("meta", {}).get("lang", "en"),
                     "r055_source": f"r044:{r['meta'].get('source')}"}

    for r in d1_train + d1_dev:
        r["meta"]["r055_source"] = f"d1:{r['meta']['d1_source']}"
    for r in open_rows:
        r["meta"]["r055_source"] = f"open:{r['meta']['source']}"
    for r in native:
        r["meta"]["r055_source"] = r["meta"]["source"]

    # EN replay sized so HE-or-MX is ~half of all presentations (floor --en-replay-min, cap --en-replay-max).
    fixed = d1_train + open_rows + native_train + images + r044_text
    he = sum(r["meta"].get("lang") in ("he", "mx") for r in fixed)
    n_pool = min(args.en_replay_max, max(args.en_replay_min - len(r044_text), he - (len(fixed) - he)))
    pool = defaultdict(list)
    for name, rows in tp.candidates().items():
        for r in rows:
            if r["id"] not in d1_ids and len(json.dumps(r["state"], ensure_ascii=False)) / 4 <= tp.MAX_STATE_TOKENS:
                r["meta"].update(lang="en", r055_source=f"pool:{name}")
                pool[name].append(r)
    en_pool = round_robin(pool, max(0, n_pool), rng)

    # 13-gram decontamination against every test file (hejev-v0 + external panel), then a hard assert.
    tests = sorted(ho.HEJEV.glob("*.jsonl")) + sorted(ho.EXT.glob("*.jsonl"))
    freq = Counter()
    for p in tests:
        for r in jsonl(p):
            freq.update(set().union(*(ho.grams(t) for t in ho.texts_of_task(r))))
    index = {g for g, c in freq.items() if c < 20}  # same boilerplate rule as D2 decon
    hit = lambda r: any(ho.grams(t) & index  # noqa: E731  (states may be JSON objects: every string leaf)
                        for t in ho.texts_of_task({"state": r["state"], "questions": r["questions"]}))
    train = d1_train + open_rows + native_train + en_pool + r044_text + images
    # Rows whose trainer encoding exceeds --max-length (found on the pod with the Gemma tokenizer), with their twins.
    drop = set(Path(args.drop_ids).read_text().split()) if args.drop_ids else set()
    drop_twins = {r["meta"].get("twin_id") for r in train + d1_dev if r["id"] in drop} - {None}
    keep = lambda r: r["id"] not in drop and r["meta"].get("twin_id") not in drop_twins  # noqa: E731
    train, d1_dev, native_dev = [r for r in train if keep(r)], [r for r in d1_dev if keep(r)], [r for r in native_dev if keep(r)]
    dropped = Counter(r["meta"]["r055_source"] for r in train if hit(r))
    bad_twins = {r["meta"]["twin_id"] for r in train if r["meta"].get("twin_id") and hit(r)}
    train = [r for r in train if not hit(r) and r["meta"].get("twin_id") not in bad_twins]
    assert not any(hit(r) for r in train), "13-gram contamination left in the mix"
    rng.shuffle(train)

    dev_he = [r for r in d1_dev if r["meta"]["lang"] in ("he", "mx")] + native_dev
    dev_en = [r for r in d1_dev if r["meta"]["lang"] == "en"]
    for rows in (dev_he, dev_en):
        rng.shuffle(rows)  # the trainer evaluates the first --eval-samples rows

    if (OUT / "images").exists():
        shutil.rmtree(OUT / "images")
    shutil.copytree(R044 / "images", OUT / "images")
    files = {"train.jsonl": write(OUT / "train.jsonl", train), "dev_he.jsonl": write(OUT / "dev_he.jsonl", dev_he),
             "dev_en.jsonl": write(OUT / "dev_en.jsonl", dev_en)}

    pres = sum(len(r["questions"]) for r in train)  # one presentation per question view
    twin_keys = {(r["meta"]["twin_id"], q) for r in train if r["meta"].get("twin_id") for q in r["questions"]}
    twin_views = sum(len(r["questions"]) for r in train if r["meta"].get("twin_id"))
    slots_b = pres - twin_views + len(twin_keys)
    views_per_slot_b = pres / slots_b
    by = lambda rows, key: dict(sorted(Counter(key(r) for r in rows).items()))  # noqa: E731
    he_views = sum(len(r["questions"]) for r in train if r["meta"].get("lang") in ("he", "mx"))
    manifest = {
        "task": "R055 Hebrew training mix (train-only); dev = D1 dev + 10% native by case",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
        "inputs": {str(p).replace(f"{ROOT}/", ""): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [D1 / "train.jsonl", D1 / "dev.jsonl", D2 / "open_train.jsonl", R044 / "arm-b.jsonl",
                    *[q for q in (D2 / "native_unlabeled.jsonl", Path(args.labels)) if q.exists()]]},
        "files": files,
        "native_labeling": {**native_stats, "agreement_rate": round(native_stats.get("agree", 0) /
                                                                     max(1, native_stats.get("labeled", 0)), 4),
                            "kept_cases": len(native), "train_cases": len(native_train), "dev_cases": len(native_dev)},
        "train_by_lang": by(train, lambda r: r["meta"].get("lang")),
        "train_by_source": by(train, lambda r: r["meta"]["r055_source"]),
        "dev_he_by_source": by(dev_he, lambda r: r["meta"]["r055_source"]),
        "presentations_per_epoch": pres, "he_or_mx_fraction": round(he_views / pres, 4),
        "en_replay": {"r044_text": len(r044_text), "pool": len(en_pool)},
        "contamination": {"test_files": [str(p.relative_to(ROOT)) for p in tests], "index_13grams": len(index),
                          "dropped_rows_by_source": dict(dropped), "dropped_twin_ids": len(bad_twins),
                          "final_assert": "passed"},
        "overlong_dropped": {"ids": sorted(drop), "twin_ids": sorted(drop_twins), "max_length": 8192},
        "arm_b_slots": {"slots": slots_b, "views_per_slot": round(views_per_slot_b, 4)},
        "steps_for_20k_presentations": {"A": round(PRESENTATIONS / SLOTS_PER_STEP),
                                        "B": round(PRESENTATIONS / SLOTS_PER_STEP / views_per_slot_b)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("files", "native_labeling", "train_by_lang", "he_or_mx_fraction",
                                               "en_replay", "contamination", "arm_b_slots",
                                               "steps_for_20k_presentations")}, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--data", required=True)
    lab.add_argument("--out", required=True)
    lab.add_argument("--endpoint", action="append", required=True)
    lab.add_argument("--deadline", required=True, help="UTC ISO time; later questions are skipped")
    lab.add_argument("--workers", type=int, default=192)
    lab.add_argument("--timeout", type=float, default=3600, help="per-request seconds (a stalled shim must not absorb every worker)")
    bld = sub.add_parser("build")
    bld.add_argument("--labels", default=str(ROOT / ".context/r055/native_labels.jsonl"))
    bld.add_argument("--open-cap", type=int, default=4000)
    bld.add_argument("--en-replay-min", type=int, default=2000)
    bld.add_argument("--en-replay-max", type=int, default=6000)
    bld.add_argument("--drop-ids", help="file of row ids to drop (with their twins): over-length rows")
    args = ap.parse_args(argv)
    {"label": cmd_label, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
