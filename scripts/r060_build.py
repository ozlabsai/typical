#!/usr/bin/env python3
"""R060 data: astra-relabelled R056 workflow rows + KL-to-base anchors (meta.anchor=true) + R056 image replay.

  anchors  tasksource slice (guychuk/typical-r060-data tasksource/) + R055 HE/MX rows + R056 English general replay
           -> .context/r060/mix/{anchors,anchor_dev}.jsonl (base_probs are added on the GPU by
           r060_precompute_base_probs.py). anchor_dev = tasksource dev sample + R056 dev_he, both anchored.
  mix      R056 mix workflow rows with new targets -> wf_train.jsonl / dev_wf.jsonl, R056 images -> img.jsonl.
           agent_trace: astra where labelled, else Qwen-only (Gemma dropped); security choice/score: astra where
           labelled, else unchanged; security noul, customer, invoice: unchanged. HE twins inherit the EN target.
  Both commands contamination-assert their output (WorkflowEvals, public JevBench, hejev-v0, external Hebrew panel,
  zoom-ai/contested-fast-decisions). Inputs: .context/r060/src/{r056/mix,cfd}, .context/r060/tasksource,
  .context/r055/mix, .context/r060a/{labels_astra_full,full_questions}.jsonl.

uv run --project .context/research_sources/WorkflowEvals python scripts/r060_build.py anchors|mix
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from r056_build import ROOT, contamination_index, jsonl, pres, write

SRC, OUT = ROOT / ".context/r060/src", ROOT / ".context/r060/mix"
R056 = SRC / "r056/mix"
CFD = SRC / "cfd/data/train.jsonl"
TRACE, SEC = "agent_trace_observability", "security_incidents"
# presentations (questions); ~35k total: workflow ~50%, anchors ~45%, images ~5%
ANCHOR_Q = {"tasksource": 7000, "he": 5500, "en": 3250}
WF_Q = {SEC: 8000, "customer_service": 2300, "invoice_processing": 1000}  # agent_trace: all of it
IMG_ROWS, TS_DEV_ROWS = 1750, 600


def group(r):
    return r["meta"].get("twin_id") or r["id"]


def sample_groups(rows, quota, rng):
    """Whole twin groups in random order until the question quota is met."""
    groups = {}
    for r in rows:
        groups.setdefault(group(r), []).append(r)
    out = []
    for _, rs in rng.sample(sorted(groups.items()), len(groups)):
        if pres(out) >= quota:
            break
        out += rs
    return out


def decontaminate(rows, hit):
    bad = {group(r) for r in rows if hit(r)}
    kept = [r for r in rows if group(r) not in bad]
    assert not any(hit(r) for r in kept), "13-gram contamination left"
    return kept, len(rows) - len(kept)


def cmd_anchors(args):
    rng = random.Random(60)
    hit, tests = contamination_index([CFD])
    ts = sample_groups(jsonl(ROOT / ".context/r060/tasksource/train.jsonl"), ANCHOR_Q["tasksource"], rng)
    he = sample_groups([r for r in jsonl(ROOT / ".context/r055/mix/train.jsonl")
                        if r["meta"].get("lang") in ("he", "mx") and not r.get("image")], ANCHOR_Q["he"], rng)
    en = sample_groups([r for r in jsonl(R056 / "train.jsonl")
                        if r["meta"].get("r056_source", "").startswith(("r055:pool:", "r055:r044:None"))],
                       ANCHOR_Q["en"], rng)
    tag = lambda rows, src: [{**r, "meta": {**r["meta"], "anchor": True, "r060_source": src}} for r in rows]  # noqa: E731
    anchors, dropped = decontaminate(tag(ts, "anchor:tasksource") + tag(he, "anchor:r055_he") + tag(en, "anchor:r056_en"), hit)
    dev = (tag(rng.sample(jsonl(ROOT / ".context/r060/tasksource/dev.jsonl"), TS_DEV_ROWS), "anchor:tasksource")
           + tag(jsonl(R056 / "dev_he.jsonl"), "anchor:r055_he"))
    dev, dev_dropped = decontaminate(dev, hit)
    summary = {"anchors.jsonl": write(OUT / "anchors.jsonl", anchors), "anchor_dev.jsonl": write(OUT / "anchor_dev.jsonl", dev),
               "questions_by_source": Counter(), "dev_questions_by_source": Counter(),
               "contamination": {"refs": tests, "dropped_rows": dropped, "dev_dropped_rows": dev_dropped}}
    for r in anchors:
        summary["questions_by_source"][r["meta"]["r060_source"]] += len(r["questions"])
    for r in dev:
        summary["dev_questions_by_source"][r["meta"]["r060_source"]] += len(r["questions"])
    (OUT / "anchors_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def option_keys(q):
    return list(q["criteria"]) if isinstance(q["criteria"], dict) else [str(i) for i in range(len(q["criteria"]))]


def vec(p, keys):
    v = [max(0.0, float(p.get(k, 0.0))) for k in keys]
    s = sum(v)
    return [x / s for x in v] if s > 0 else None


def retarget(rows, astra, qwen, stats):
    out = []
    for r in rows:
        wf = r["meta"].get("workflow")
        req = r["id"].split(":", 1)[1].removesuffix("-he")
        qs, soft, tg, teach = {}, {}, {}, []
        for qid, q in r["questions"].items():
            keys, key, t, who = option_keys(q), (req, qid), r["soft_targets"][qid], "r056"
            if wf == TRACE:
                t, who = (vec(astra[key], keys), "astra") if key in astra else \
                    (vec(qwen[key], keys), "qwen") if qwen.get(key) else (None, "none")
            elif wf == SEC and q["type"] != "noul" and key in astra:
                t, who = vec(astra[key], keys), "astra"
            stats[f"{wf}/{q['type']}/{who}/{r['meta']['lang']}"] += 1
            if t is None:
                continue
            qs[qid], soft[qid], tg[qid] = q, t, {"label": keys[max(range(len(t)), key=t.__getitem__)]}
            teach.append(f"{qid}={who}")
        if qs:
            out.append({**r, "questions": qs, "soft_targets": soft, "targets": tg,
                        "meta": {**r["meta"], "teachers": ",".join(teach)}})
    return out


def cmd_mix(args):
    rng = random.Random(60)
    astra = {(a["id"], a["qid"]): a["p"] for f in ("labels_astra_full.jsonl", "labels_astra_full_codex.jsonl")
             for a in jsonl(ROOT / ".context/r060a" / f)}  # OpenRouter, then Codex CLI for the rest (credits ran out)
    qwen = {(r["id"], r["qid"]): r["p_qwen"] for r in jsonl(ROOT / ".context/r060a/full_questions.jsonl")
            if r["workflow"] == TRACE}
    stats = Counter()
    train = [r for r in jsonl(R056 / "train.jsonl") if r["meta"].get("workflow")]
    dev = retarget(jsonl(R056 / "dev_wf.jsonl"), astra, qwen, Counter())
    train = retarget(train, astra, qwen, stats)
    by_wf = {}
    for r in train:
        by_wf.setdefault(r["meta"]["workflow"], []).append(r)
    wf_train = by_wf.pop(TRACE)
    for wf, rows in sorted(by_wf.items()):  # whole cases (EN + HE twins) per workflow quota
        cases = {}
        for r in rows:
            cases.setdefault(r["meta"]["case_id"], []).append(r)
        picked = []
        for _, rs in rng.sample(sorted(cases.items()), len(cases)):
            if pres(picked) >= WF_Q[wf]:
                break
            picked += rs
        wf_train += picked
    img = rng.sample([r for r in jsonl(R056 / "train.jsonl") if r["meta"].get("r056_source", "").startswith("img:")], IMG_ROWS)
    hit, tests = contamination_index([CFD])
    wf_train, dropped = decontaminate(wf_train, hit)
    img, img_dropped = decontaminate(img, hit)
    dev, dev_dropped = decontaminate(dev, hit)
    anchors = jsonl(OUT / "anchors.jsonl")
    assert not any(hit(r) for r in anchors), "anchors contaminated"
    count = lambda rows, key: dict(sorted(Counter(k for r in rows for k in [key(r)] * len(r["questions"])).items()))  # noqa: E731
    everything = wf_train + img + anchors
    twin_keys = {(r["meta"]["twin_id"], q) for r in everything if r["meta"].get("twin_id") for q in r["questions"]}
    twin_views = sum(len(r["questions"]) for r in everything if r["meta"].get("twin_id"))
    src = lambda r: r["meta"].get("r060_source") or r["meta"].get("r056_source", "?")  # noqa: E731
    manifest = {
        "task": "R060 mix: astra-relabelled R056 workflow rows + KL-to-base anchors + R056 image replay",
        "files": {"wf_train.jsonl": write(OUT / "wf_train.jsonl", wf_train), "img.jsonl": write(OUT / "img.jsonl", img),
                  "dev_wf.jsonl": write(OUT / "dev_wf.jsonl", dev),
                  "dev_he.jsonl": write(OUT / "dev_he.jsonl", jsonl(R056 / "dev_he.jsonl"))},
        "presentations": pres(everything), "slots_with_twins": pres(everything) - twin_views + len(twin_keys),
        "share": {k: round(pres([r for r in everything if src(r).split(":")[0] == k]) / pres(everything), 3)
                  for k in ("wf", "anchor", "img")},
        "train_by_source": count(everything, src), "train_by_lang": count(everything, lambda r: r["meta"].get("lang")),
        "target_teachers": dict(sorted(stats.items())),
        "contamination": {"refs": ["WorkflowEvals inputs (4 workflows)", *tests], "final_assert": "passed",
                          "dropped_rows": {"wf_train": dropped, "img": img_dropped, "dev_wf": dev_dropped}},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("anchors", "mix"))
    args = ap.parse_args(argv)
    {"anchors": cmd_anchors, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    main()
