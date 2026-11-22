"""Vision student evaluation and the end-to-end run on real sites.

score (on the GPU pod): base Gemma and the LoRA on vision_v0's test suites, every option's probability saved, plus the
    tuned model's answers on every keyframe (train, dev and test videos) for the end-to-end run.
  python scripts/helipad_vision_eval.py score --model <gemma dir> --adapter <lora dir> --data <vision_v0> --out <dir>
e2e (local, then one scoring pod): each real site in videos.json scored by the Plan 1 decision checkpoint without video,
    with the student's video findings and with the teacher's.
  uv run scripts/helipad_vision_eval.py e2e --student <dir>/frames_tuned.jsonl --teacher <frames_v0.jsonl>
      --videos $HELIPAD_PRIVATE/videos.json --sites <sites dir> --build <vision_v0>/build.json --ckpt <best.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from helipad import frames as fr  # noqa: E402

SUITES = ("test_real", "test_shuffled", "test_noimage", "test_mismatch")


def _stat(xs: list) -> dict:
    return {"value": sum(xs) / len(xs) if xs else None, "n": len(xs)}


def _teacher_top(row: dict, qid: str) -> str:
    return row["questions"][qid]["criteria"][row["targets"][qid]["label"]]


def _ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    groups = defaultdict(list)
    for conf, ok in pairs:
        groups[min(int(conf * bins), bins - 1)].append((conf, ok))
    return sum(abs(sum(ok for _, ok in g) / len(g) - sum(c for c, _ in g) / len(g)) * len(g) for g in groups.values()) / len(pairs)


def _auroc(neg: list[float], pos: list[float]) -> float | None:
    if not neg or not pos:
        return None
    return (sum(a < b for a in neg for b in pos) + .5 * sum(a == b for a in neg for b in pos)) / (len(neg) * len(pos))


def vision_report(rows: list[dict], preds: list[dict]) -> dict:
    """rows: grouped rows with meta.variant in real/shuffled/noimage/mismatch; preds[i] = {qid: {option: p}}.
    Agreement = the student's argmax equals the teacher's (real targets, also on shuffled / no-image rows); null recall on
    mismatch rows; answerability AUROC of P(null), mismatch (positive) vs real (negative); ECE per question on real rows."""
    qids = list(fr.FRAME_QUESTIONS)
    hits = defaultdict(lambda: defaultdict(list))   # variant -> qid -> [hit]
    calib, null_p, tops = defaultdict(list), {"real": [], "mismatch": []}, defaultdict(list)
    for r, p in zip(rows, preds):
        v = r["meta"]["variant"]
        for q in qids:
            top = fr.argmax(p[q])
            if v == "mismatch":
                hits[v][q].append(top == fr.NULL_OPTION)
            else:
                hits[v][q].append(top == _teacher_top(r, q))
            if v == "real":
                calib[q].append((p[q][top], top == _teacher_top(r, q)))
                tops[q].append(_teacher_top(r, q))
            if v in null_p:
                null_p[v].append(p[q][fr.NULL_OPTION])

    def per_q(v):
        return {**{q: _stat(hits[v][q]) for q in qids}, "overall": _stat([x for q in qids for x in hits[v][q]])}
    agreement = {v: per_q(v) for v in ("real", "shuffled", "noimage")}

    def minus(other):
        a, b = agreement["real"]["overall"], agreement[other]["overall"]
        ok = a["n"] and b["n"]
        return {"value": a["value"] - b["value"] if ok else None, "n": min(a["n"], b["n"])}
    majority = {q: _stat([t == max(set(tops[q]), key=tops[q].count) for t in tops[q]]) for q in qids}
    majority["overall"] = _stat([t == max(set(tops[q]), key=tops[q].count) for q in qids for t in tops[q]])
    return {"agreement": agreement, "null_recall": per_q("mismatch"),
            "real_minus_shuffled": minus("shuffled"), "real_minus_noimage": minus("noimage"),
            "answerability_auroc": {"value": _auroc(null_p["real"], null_p["mismatch"]),
                                    "n_pos": len(null_p["mismatch"]), "n_neg": len(null_p["real"])},
            "ece": {q: {"value": _ece(calib[q]), "n": len(calib[q])} for q in qids},
            "majority_baseline": majority}


# -- on the pod
def _load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def predict(model, processor, groups, rows: list[dict], root: Path) -> list[dict]:
    """Every option's probability per question, through the trainer's own prompt and image encoding (canonical order)."""
    import torch
    from scripts.train_gemma_decision_lora import encode, encode_image, flatten_row, option_scores_from_logits

    out = []
    with torch.inference_mode():
        for n, r in enumerate(rows, 1):
            answers = {}
            for qid, e in zip(r["questions"], flatten_row(r, root)):
                if e["image"]:
                    inputs, _, labels = encode_image(e, processor, 8192, False, include_labels=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                else:
                    ids, _, labels = encode(e, processor.tokenizer, 8192, False, include_labels=True)
                    inputs = {"input_ids": ids[None].to(model.device)}
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                probs = torch.softmax(option_scores_from_logits(logits, groups, len(labels), model.device), 0).tolist()
                text = dict(e["items"])
                answers[qid] = {text[k]: round(p, 5) for k, p in zip(labels, probs)}
            out.append(answers)
            if n % 50 == 0:
                print(f"{n}/{len(rows)}", flush=True)
    return out


def cmd_score(a) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from scripts.train_gemma_decision_lora import letter_token_groups

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cut = slice(a.limit or None)   # --limit: a smoke run on the first rows of every suite
    rows = [r for s in SUITES for r in _load_rows(Path(a.data) / f"{s}.jsonl")[cut]]
    frames = [r for s in ("train", "dev", "test_real") for r in _load_rows(Path(a.data) / f"{s}.jsonl")
              if r["meta"]["variant"] == "real"][cut]
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    groups = letter_token_groups(processor.tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                                        attn_implementation="sdpa").to("cuda").eval()
    reports = {}
    for name in ("base", "tuned") if a.adapter else ("base",):
        if name == "tuned":
            model = PeftModel.from_pretrained(model, a.adapter, local_files_only=True).eval()
        preds = predict(model, processor, groups, rows, Path(a.data))
        (out / f"preds_{name}.jsonl").write_text("".join(json.dumps({"id": r["id"], "variant": r["meta"]["variant"],
                                                                     "answers": p}, ensure_ascii=False) + "\n"
                                                         for r, p in zip(rows, preds)))
        reports[name] = vision_report(rows, preds)
        print(name, json.dumps(reports[name]["agreement"]["real"]["overall"]), flush=True)
    (out / "report.json").write_text(json.dumps(reports, indent=1))
    preds = predict(model, processor, groups, frames, Path(a.data))
    (out / "frames_tuned.jsonl").write_text("".join(json.dumps({"id": r["id"], "video": r["meta"]["video"], "t": r["meta"]["t"],
                                                                "answers": p}, ensure_ascii=False) + "\n"
                                                    for r, p in zip(frames, preds)))


# -- end to end on real sites
def site_cases(videos: dict, sites: list[dict], sites_dir: Path, answers: dict[str, dict[str, list]]) -> dict:
    """{public id: {"none": SiteCase, <source>: SiteCase with that source's video findings}}; answers = {source: {video:
    [{"t", "answers"}]}}. Dust is the worse of map and video; belly comes from video only."""
    from helipad.from_lz import case_from_lz

    by_id = {s["public_id"]: s for s in sites}
    vids = defaultdict(list)
    for v, m in videos.items():
        if m.get("site"):
            vids[m["site"]].append(v)
    out = {}
    for pid, vs in sorted(vids.items()):
        lz = sites_dir / by_id[pid]["slug"] / "lz.json"
        if not lz.exists():
            continue
        base = case_from_lz(json.loads(lz.read_text()), pid)
        out[pid] = {"none": base, "videos": vs}
        for src, per_video in answers.items():
            found = [fr.video_findings(per_video.get(v, []), v) for v in vs]
            dusts = [d for d in (base.dust, *(i["dust"] for _, i in found)) if d]
            bellies = [i["belly"] for _, i in found if i["belly"]]
            out[pid][src] = replace(base, name=f"{pid}+video({src})", obstacles=base.obstacles + [o for ob, _ in found for o in ob],
                                    dust=max(dusts, key=fr.DUST.index) if dusts else None,
                                    belly="obstacles" if "obstacles" in bellies else (bellies[0] if bellies else None))
            out[pid][f"{src}_info"] = [i for _, i in found]
    return out


def e2e_table(cases: dict, scored: dict, sites: list[dict], split_of: dict[str, str]) -> list[dict]:
    """One line per (site, source): rule verdict on the case, the model's verdict of record (from its item argmaxes) and its
    verdict question's full distribution, what changed against no video, and the expert direction where one exists."""
    from helipad.rules import derive_verdict, judge, verdict

    expert = {s["public_id"]: s.get("expert") for s in sites}
    lines = []
    for pid, c in cases.items():
        rec_none = None
        for src in ("none", "student", "teacher"):
            if src not in c:
                continue
            case = c[src]
            got = scored.get(case.name)
            if got is None:
                continue
            items = {k: v["argmax"].replace(" ", "_") for k, v in got["items"].items()}
            record = derive_verdict(items.values(), bool(case.core_missing))
            rec_none = record if src == "none" else rec_none
            rules = judge(case)
            line = {"site": pid, "source": src, "videos": [f"{v} ({split_of.get(v, '?')})" for v in c["videos"]],
                    "rule_verdict": verdict(case, rules), "record": record, "verdict_probs": got["verdict"]["probs"],
                    "changed_vs_no_video": None if src == "none" else record != rec_none,
                    "items": {k: got["items"][k]["probs"] for k in ("low_wires", "very_low_10", "low_50", "dust", "belly")}}
            if expert.get(pid):
                line["expert"] = expert[pid]
                line["expert_direction_match"] = (record == "approved") == (expert[pid] == "approved")
            lines.append(line)
    return lines


def cmd_e2e(a) -> None:
    from helipad.paths import SITES_JSON
    from helipad.rows import case_rows

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = {"student": defaultdict(list), "teacher": defaultdict(list)}
    for r in _load_rows(a.student):
        answers["student"][r["video"]].append({"t": r["t"], "answers": r["answers"]})
    for r in _load_rows(a.teacher):
        if not r.get("failed"):
            answers["teacher"][r["video"]].append({"t": r["t"], "answers": r["answers"]})
    sites = json.loads(SITES_JSON.read_text())
    cases = site_cases(json.loads(Path(a.videos).read_text()), sites, Path(a.sites), answers)
    rows = [row for c in cases.values() for k in ("none", "student", "teacher") for row in case_rows(c[k])]
    (out / "e2e.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "video_findings.json").write_text(json.dumps(
        {pid: {s: c[f"{s}_info"] for s in ("student", "teacher")} for pid, c in cases.items()}, indent=1, ensure_ascii=False))
    score = out / "score.json"
    if not score.exists():
        import helipad_score_pod
        helipad_score_pod.main([str(a.ckpt), str(score), "--eval-dir", str(out), "--files", "e2e.jsonl"])
    scored = json.loads(score.read_text())["e2e.jsonl"]["cases"]
    split_of = {v: s for s, vs in json.loads(Path(a.build).read_text())["videos"].items() for v in vs}
    table = e2e_table(cases, scored, sites, split_of)
    (out / "e2e_table.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    for line in table:
        print(json.dumps(line, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    for k in ("--model", "--data", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--adapter", help="without it: the base model only")
    s.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("e2e")
    for k in ("--student", "--teacher", "--videos", "--sites", "--build", "--ckpt", "--out"):
        e.add_argument(k, required=True)
    a = ap.parse_args(argv)
    {"score": cmd_score, "e2e": cmd_e2e}[a.cmd](a)


if __name__ == "__main__":
    main()
