#!/usr/bin/env python3
"""R056 data: label inputs for the thinking teachers, then the training mix.

  labelin  generated nodes (.context/r056/gen*/<wf>/nodes.jsonl) -> .context/r056/label_in/{gemma,qwen}.jsonl
           gemma: agent_trace, customer, invoice, then security (reference only: security targets are Qwen-only);
           qwen:  security first, then agent_trace (coordinator: Qwen capacity to security, then trace).
  mix      see cmd_mix.
  python3 scripts/r056_build.py labelin
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R6 = ROOT / ".context/r056"
GENS = {"gen": "claude-haiku-5-5", "gen_gpt": "gpt-5.6-terra", "gen_gemma": "google/gemma-4-12B-it",
        "gen_qwen": "Qwen/Qwen3.8-27B-FP8"}  # output dir -> generator (coordinator: generator diversity)
WFS = ("security_incidents", "agent_trace_observability", "customer_service", "invoice_processing")


def jsonl(path, tolerant=False):
    """tolerant: skip a torn line (label files are pulled while the pod appends to them)."""
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if not tolerant:
                raise
    return out


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {"rows": len(rows), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def is_dev(case_id: str) -> bool:
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 10 == 0


def nodes(wf):
    return [{"id": n["request_id"], "state": n["state"], "questions": n["questions"], "case_id": n["case_id"],
             "generator": gen} for d, gen in GENS.items() for n in jsonl(R6 / d / wf / "nodes.jsonl")]


def cmd_labelin(args):
    sec, trace, cust, inv = (nodes(wf) for wf in WFS)
    out = {"gemma": write(R6 / "label_in/gemma.jsonl", trace + cust + inv + sec),  # security last: reference only
           "qwen": write(R6 / "label_in/qwen.jsonl", sec + trace)}  # security first, then agent_trace
    q = lambda rows: sum(len(r["questions"]) for r in rows)  # noqa: E731
    print(json.dumps({**out, "questions": {"security": q(sec), "agent_trace": q(trace), "customer": q(cust),
                                           "invoice": q(inv)}}))


# --------------------------------------------------------------------------- mix

R055_MIX = ROOT / ".context/r055/mix"
WF_CAPS = {"customer_service": 6000, "invoice_processing": 4500}  # questions; security/trace are teacher-limited
SHARE = {"wf": 0.60, "he": 0.25, "en": 0.10, "img": 0.05}  # of presentations


def _blank(v):
    return v is None or (isinstance(v, str) and not v.strip()) or v in ({}, [])


def _structured(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def wire_items(q):
    """Option (label, description) in the serving order (serve_typesafe_gemma_lora._items)."""
    c = q.get("criteria")
    if q["type"] == "noul":
        c = c or {}
        return [(k, c.get(k)) for k in [*c, *(k for k in ("true", "false") if k not in c)]]
    if q["type"] == "choice":
        return list(c.items())
    return [(str(i), d) for i, d in enumerate(c)]


def trainer_question(q):
    """Wire question -> trainer question whose option texts are exactly render_typesafe_decision_prompt's."""
    items = wire_items(q)
    if q["type"] == "score":
        crit = [f"Level {k}" if _blank(d) else _structured(d) for k, d in items]
    elif q["type"] == "noul":
        crit = {k: {"true": "Yes", "false": "No"}[k] + ("" if _blank(d) else f": {_structured(d)}") for k, d in items}
    else:
        crit = {k: k if _blank(d) else f"{k}: {_structured(d)}" for k, d in items}
    return {"type": q["type"], "instructions": q.get("instructions") or "", "criteria": crit}


def answer_probs(q, answer):
    keys = [k for k, _ in wire_items(q)]
    if q["type"] == "noul":
        p = float(answer["noul"])
        probs = [p if k == "true" else 1 - p for k in keys]
    else:
        probs = [max(0.0, float(answer.get("probabilities", {}).get(k, 0.0))) for k in keys]
    total = sum(probs)
    return [x / total for x in probs] if total > 0 else None


def teacher_target(wf, g, qw):
    """Coordinator rules: security = Qwen only; else mean(Gemma, Qwen), Gemma alone smoothed off exact 0/1."""
    if wf == "security_incidents":
        return (qw, "qwen") if qw else (None, None)
    if g and qw:
        return [(a + b) / 2 for a, b in zip(g, qw)], "mean"
    if g:
        if any(x in (0.0, 1.0) for x in g):  # == clip to [.02, .98] for noul
            g = [0.96 * x + 0.04 / len(g) for x in g]
        return g, "gemma"
    return (qw, "qwen") if qw else (None, None)


def wf_rows():
    labels = {name: {(r["id"], r["qid"]): r["answer"] for r in jsonl(R6 / "labels" / f"labels_{name}.jsonl", tolerant=True)
                     if "answer" in r} for name in ("gemma", "qwen", "qwen_trace")}
    qwen = {**labels["qwen"], **labels["qwen_trace"]}
    twins = {t["twin_id"]: t for t in jsonl(R6 / "twins/twins.jsonl")}
    rows, stats = [], {"teacher": {}, "unlabeled": 0}
    for d, gen in GENS.items():
        for wf in WFS:
            for n in jsonl(R6 / d / wf / "nodes.jsonl"):
                qs, tg, soft, teach = {}, {}, {}, []
                for qid, q in n["questions"].items():
                    key = (n["request_id"], qid)
                    g = answer_probs(q, labels["gemma"][key]) if key in labels["gemma"] else None
                    qw = answer_probs(q, qwen[key]) if key in qwen else None
                    t, who = teacher_target(wf, g, qw)
                    if t is None:
                        stats["unlabeled"] += 1
                        continue
                    stats["teacher"][f"{wf}/{who}"] = stats["teacher"].get(f"{wf}/{who}", 0) + 1
                    keys = [k for k, _ in wire_items(q)]
                    qs[qid], soft[qid] = trainer_question(q), t
                    tg[qid] = {"label": keys[max(range(len(t)), key=t.__getitem__)]}
                    teach.append(f"{qid}={who}")
                if not qs:
                    continue
                meta = {"workflow": wf, "case_id": n["case_id"], "generator": gen, "lang": "en", "node": n["node"],
                        "teachers": ",".join(teach), "split": "dev" if is_dev(n["case_id"]) else "train",
                        "r056_source": f"wf:{wf}"}
                tw = twins.get(n["request_id"])
                if tw:
                    meta["twin_id"] = n["request_id"]
                rows.append({"id": f"{n['case_id']}:{n['request_id']}", "state": n["state"], "questions": qs,
                             "targets": tg, "soft_targets": soft, "meta": meta})
                if tw:
                    rows.append({"id": f"{n['case_id']}:{n['request_id']}-he", "state": tw["state"],
                                 "questions": {q: trainer_question(tw["questions"][q]) for q in qs},
                                 "targets": tg, "soft_targets": soft, "meta": {**meta, "lang": "he"}})
    return rows, stats


def cap_cases(rows, rng):
    """Drop whole cases (with their twins) of over-quota workflows, keeping generator shares."""
    by_case = {}
    for r in rows:
        by_case.setdefault(r["meta"]["case_id"], []).append(r)
    keep = []
    for wf in WFS:
        cases = [c for c, rs in by_case.items() if rs[0]["meta"]["workflow"] == wf]
        rng.shuffle(cases)
        n = 0
        for c in cases:
            q = sum(len(r["questions"]) for r in by_case[c] if r["meta"]["lang"] == "en")
            if wf in WF_CAPS and n + q > WF_CAPS[wf]:
                continue
            n += q
            keep += by_case[c]
    return keep


def pres(rows):
    return sum(len(r["questions"]) for r in rows)


def contamination_index():
    """13-gram index of WorkflowEvals inputs (per-workflow >=5% boilerplate ignored), JevBench public, hejev-v0
    and the external Hebrew panel (R055 rule: 13-grams in >=20 test rows are boilerplate)."""
    import os
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, os.environ.get("WORKFLOWEVALS", str(ROOT / ".context/research_sources/WorkflowEvals")))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import r055_hebrew_open as ho
    from core.dataset import load_dataset

    index = set()
    for wf in WFS:
        bench = load_dataset(wf)
        df = {}
        for c in bench.inputs:
            for g in set().union(*(ho.grams(t) for t in ho.texts_of_task(bench.inputs[c]))):
                df[g] = df.get(g, 0) + 1
        index |= {g for g, c in df.items() if c < max(2, 0.05 * len(bench.inputs))}
    freq = {}
    tests = [*sorted((ROOT / ".context/jevbench-v154/datasets/public").glob("*.jsonl")),
             *sorted(ho.HEJEV.glob("*.jsonl")), *sorted(ho.EXT.glob("*.jsonl"))]
    for path in tests:
        for r in jsonl(path):
            for g in set().union(*(ho.grams(t) for t in ho.texts_of_task(r))):
                freq[g] = freq.get(g, 0) + 1
    index |= {g for g, c in freq.items() if c < 20}
    hit = lambda r: any(ho.grams(t) & index for t in ho.texts_of_task(  # noqa: E731
        r["state"] if r["meta"].get("r056_source", "").startswith("wf:") else {"s": r["state"], "q": r["questions"]}))
    return hit, [str(p.relative_to(ROOT)) for p in tests]


def images(n, out):
    """n ScienceQA image rows (R044 recipe: half real, a quarter mismatched image, a quarter gold dropped)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_r044_mm_replay as bm

    bm.COUNTS["visual"] = n
    bm.VISUAL_COUNTS.update(real=n // 2, mismatch=n // 4, gold_dropped=n - n // 2 - n // 4)
    train = ROOT / "data_v_full/train.jsonl"
    hashes, decisions, _ = bm._protected(sorted((ROOT / "data_v/eval").glob("*.jsonl")))
    rows, _, rejected = bm._visual(train, jsonl(train), hashes, decisions, out, 56)
    for r in rows:
        r["meta"] = {**r["meta"], "lang": "en", "r056_source": "img:ScienceQA"}
    return rows, rejected


def cmd_images(args):
    import shutil
    out = R6 / "img"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows, rejected = images(args.n, out)
    write(out / "rows.jsonl", rows)
    (out / "rejected.json").write_text(json.dumps(rejected))
    print(len(rows), rejected)


def cmd_mix(args):
    import random
    import shutil
    from datetime import datetime, timezone

    rng = random.Random(56)
    out = R6 / "mix"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    wf, stats = wf_rows()
    wf = cap_cases(wf, rng)
    hit, tests = contamination_index()
    wf_dev = [r for r in wf if r["meta"]["split"] == "dev"]
    wf_train = [r for r in wf if r["meta"]["split"] == "train"]
    total = pres(wf_train) / SHARE["wf"]

    groups = {}
    for r in jsonl(R055_MIX / "train.jsonl"):
        src = r["meta"].get("r055_source", "")
        if src == "r044:ScienceQA":
            continue
        kind = "en" if src.startswith("pool:") or src == "r044:None" else "he"
        groups.setdefault((kind, r["meta"].get("twin_id") or r["id"]), []).append(r)
    picked = {"he": [], "en": []}
    for (kind, _), rs in rng.sample(sorted(groups.items()), len(groups)):
        if pres(picked[kind]) < SHARE[kind] * total:
            picked[kind] += [{**r, "meta": {**r["meta"], "r056_source": f"r055:{r['meta'].get('r055_source')}"}}
                             for r in rs]
    img = jsonl(R6 / "img/rows.jsonl")  # prebuilt locally (`images`): the laptop uplink is too slow for data_v_full
    img_rejected = json.loads((R6 / "img/rejected.json").read_text())
    shutil.copytree(R6 / "img/images", out / "images")

    train = wf_train + picked["he"] + picked["en"] + img
    bad = {r["meta"].get("twin_id") or r["id"] for r in train if hit(r)}
    dropped = {}
    for r in train:
        if (r["meta"].get("twin_id") or r["id"]) in bad:
            dropped[r["meta"]["r056_source"].split(":")[0]] = dropped.get(r["meta"]["r056_source"].split(":")[0], 0) + 1
    train = [r for r in train if (r["meta"].get("twin_id") or r["id"]) not in bad]
    wf_dev = [r for r in wf_dev if not hit(r)]
    assert not any(hit(r) for r in train + wf_dev), "13-gram contamination left in the mix"
    rng.shuffle(train)
    rng.shuffle(wf_dev)
    dev_he = jsonl(R055_MIX / "dev_he.jsonl")[:300]
    files = {"train.jsonl": write(out / "train.jsonl", train), "dev_wf.jsonl": write(out / "dev_wf.jsonl", wf_dev),
             "dev_he.jsonl": write(out / "dev_he.jsonl", dev_he)}

    def count(rows, key):
        c = {}
        for r in rows:
            c[key(r)] = c.get(key(r), 0) + len(r["questions"])
        return dict(sorted(c.items()))

    sec = [p for r in wf_train if r["meta"]["workflow"] == "security_incidents" and r["meta"]["lang"] == "en"
           for q, t in r["soft_targets"].items() for p in ([t[0] if list(r["questions"][q]["criteria"])[0] == "true"
                                                             else t[1]] if r["questions"][q]["type"] == "noul" else [])]
    twin_keys = {(r["meta"]["twin_id"], q) for r in train if r["meta"].get("twin_id") for q in r["questions"]}
    twin_views = sum(len(r["questions"]) for r in train if r["meta"].get("twin_id"))
    manifest = {
        "task": "R056 training mix: thinking-teacher-labeled workflow nodes + R055 Hebrew + EN replay + images",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files,
        "presentations": pres(train), "slots_with_twins": pres(train) - twin_views + len(twin_keys),
        "share": {k: round(v / pres(train), 3) for k, v in count(train, lambda r: r["meta"]["r056_source"].split(":")[0]).items()},
        "train_by_source": count(train, lambda r: r["meta"]["r056_source"]),
        "train_by_lang": count(train, lambda r: r["meta"].get("lang")),
        "wf_train_by_generator": count(wf_train, lambda r: f"{r['meta']['workflow']}/{r['meta']['generator']}"),
        "wf_dev_by_generator": count(wf_dev, lambda r: f"{r['meta']['workflow']}/{r['meta']['generator']}"),
        "wf_cases": {"train": len({r["meta"]["case_id"] for r in wf_train}), "dev": len({r["meta"]["case_id"] for r in wf_dev})},
        "teacher_targets": stats["teacher"], "unlabeled_questions_dropped": stats["unlabeled"],
        "security_noul_targets": {"n": len(sec), "middle_band_0.15_0.75": round(sum(.15 <= p <= .75 for p in sec) / max(1, len(sec)), 4),
                                  "exact_0_or_1": round(sum(p in (0.0, 1.0) for p in sec) / max(1, len(sec)), 4)},
        "images": {"rows": len(img), "rejected": img_rejected},
        "contamination": {"refs": ["WorkflowEvals inputs (4 workflows, pinned dataset cache)", *tests],
                          "dropped_rows_by_kind": dropped, "final_assert": "passed",
                          "note": "WF rows: state only (questions are WorkflowEvals' public schema, reused on purpose)"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("labelin", "images", "mix"))
    ap.add_argument("--n", type=int, default=2000, help="images: ScienceQA rows (~5%% of ~40k presentations)")
    args = ap.parse_args(argv)
    {"labelin": cmd_labelin, "images": cmd_images, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    main()
