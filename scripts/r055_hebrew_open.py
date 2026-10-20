#!/usr/bin/env python3
"""R055 D2-A/B/D: external Hebrew test panel, open Hebrew train rows, and 13-gram decontamination.

  panel   EVAL-ONLY panel -> .context/hebrew-bench/external/{belebele,hebnli,gmmlu,sentiment,winograd,heq}.jsonl
          in JevBench task format (same as hejev-v0/translated.jsonl; every row loads via jevbench Task.from_dict).
  train   open Hebrew train rows (trainer schema, meta.lang/meta.source/meta.license) ->
          .context/he-train/d2/open_train.jsonl: MASSIVE he-IL intents, HeQ answerable + span choice,
          HebrewSentiment (omilab, deduped against its test split).
  decon   drop every open_train / native_unlabeled row whose model input shares a 13-gram with hejev-v0
          or the external panel (in place; report in .context/he-train/d2/decon_report.json).

  uv run python scripts/r055_hebrew_open.py panel train decon
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".context" / "jevbench-v154"))
from jevbench.tasks import Task  # noqa: E402

EXT = ROOT / ".context" / "hebrew-bench" / "external"
HEJEV = ROOT / ".context" / "hebrew-bench" / "hejev-v0"
D2 = ROOT / ".context" / "he-train" / "d2"
RAW = D2 / "raw"
SEED = 55
LETTERS = "abcd"
EVAL_ONLY = "EVAL-ONLY external Hebrew test panel (R055). Never train, distill, or select training data on it."
OMILAB = "https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew/raw/master/data/token_{}.tsv"
HEQ = "https://raw.githubusercontent.com/NNLP-IL/Hebrew-Question-Answering-Dataset/main/data/data%20v1.1/{}%20v1.1.json"
SENT = {"pos": ("positive", "התגובה מביעה עמדה חיובית / תמיכה"),
        "neg": ("negative", "התגובה מביעה עמדה שלילית / ביקורת"),
        "off": ("off_topic", "התגובה ניטרלית או אינה עניינית לנושא הפוסט")}
SENT_LABELS = ["pos", "neg", "off"]  # omilab ClassLabel order: pos, neg, off-topic

LICENSES = {
    "belebele": "CC-BY-SA-4.0 (facebook/belebele)",
    "hebnli": "UNCLEAR: no repo license; MultiNLI-derived (OANC + CC-BY-SA-3.0 fiction), HF copy 'other'. Eval-only.",
    "gmmlu": "Apache-2.0 (CohereLabs/Global-MMLU)",
    "sentiment": "MIT (omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew)",
    "winograd": "CC-BY-4.0 per AlephBench card (original He-WSC by V. Shwartz: license unstated)",
    "heq": "CC-BY-4.0 (NNLP-IL/Hebrew-Question-Answering-Dataset v1.1)",
    "massive": "CC-BY-4.0 (AmazonScience/massive he-IL)",
}


def fetch(url: str, name: str) -> Path:
    import urllib.request

    path = RAW / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=300).read())
    return path


def hf(repo: str, file: str, **kw) -> Path:
    return Path(hf_hub_download(repo, file, repo_type="dataset", **kw))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows):5d} -> {path.relative_to(ROOT)}")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[֑-ׇ]", "", text)).strip()


def sentiment_rows() -> dict[str, list[tuple[str, str]]]:
    """omilab token split -> {split: [(text, label)]}, exact-normalized duplicates removed, test deduped vs train."""
    out = {}
    for split in ("train", "test"):
        rows = []
        for line in fetch(OMILAB.format(split), f"omilab/token_{split}.tsv").read_text().splitlines():
            if "\t" in line:
                text, label = line.rsplit("\t", 1)
                rows.append((norm(text), SENT_LABELS[int(label)]))
        out[split] = rows
    train_texts = {t for t, _ in out["train"]}
    test, seen = [], set()
    for t, lab in out["test"]:
        if t not in train_texts and t not in seen and len(t) > 5:
            seen.add(t)
            test.append((t, lab))
    out["test"], train, seen = test, [], {t for t, _ in test}
    for t, lab in out["train"]:  # also keep train free of anything left in test
        if t not in seen and len(t) > 5:
            seen.add(t)
            train.append((t, lab))
    out["train"] = train
    return out


def heq_split(split: str) -> list[dict]:
    data = json.loads(fetch(HEQ.format(split), f"heq/{split}_v1.1.json").read_text())["data"]
    return [{"context": p["context"], "title": a["title"], "source": a.get("source"), **q}
            for a in data for p in a["paragraphs"] for q in p["qas"]]


def balanced(groups: dict, n: int, rng: random.Random) -> list:
    """Round-robin over shuffled groups until n items (stratified, as balanced as availability allows)."""
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
    out = []
    while len(out) < n and any(pools.values()):
        for k in list(pools):
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out


# --------------------------------------------------------------------------- A: external panel


def task(tid, family, state, qtype, instructions, criteria, labels, expected, source, license_, **extra):
    row = {"id": tid, "family": family, "state": state, "labels": labels, "expected": expected, "split": "public",
           "group": None, "question": {"type": qtype, "instructions": instructions, "criteria": criteria},
           "provenance": {"source": source, "license": license_, "lang": "he", "eval_only": True,
                          "eval_only_rule": EVAL_ONLY, "exclude_reason": None, **extra}}
    Task.from_dict(row)
    return row


def mc4(tid, family, state, options, gold_idx, source, license_, instructions, **extra):
    criteria = {LETTERS[i]: str(o).strip() for i, o in enumerate(options)}
    return task(tid, family, state, "choice", instructions, criteria, list(LETTERS[:len(options)]),
                LETTERS[gold_idx], source, license_, **extra)


def cmd_panel(rng: random.Random) -> dict:
    counts = {}
    # Belebele heb_Hebr: all 900
    rows = [mc4(f"ext-belebele-{i:03d}", "extraction", f"קטע:\n{r['flores_passage']}\n\nשאלה: {r['question']}",
                [r[f"mc_answer{k}"] for k in range(1, 5)], int(r["correct_answer_num"]) - 1,
                f"facebook/belebele heb_Hebr {r['link']}#{r['question_number']}", LICENSES["belebele"],
                "בחרו את התשובה הנכונה לשאלה על פי הקטע בלבד.")
            for i, r in enumerate(jsonl(hf("facebook/belebele", "data/heb_Hebr.jsonl")))]
    write(EXT / "belebele.jsonl", rows)
    counts["belebele"] = len(rows)

    # HebNLI human-gold test: all 884
    nli = {"entailment": "ההשערה נובעת בהכרח מההנחה", "neutral": "ההשערה אינה נובעת מההנחה וגם אינה סותרת אותה",
           "contradiction": "ההשערה סותרת את ההנחה"}
    rows = [task(f"ext-hebnli-{r['pairID']}", "adequacy",
                 f"הנחה: {r['translation1'].strip()}\nהשערה: {r['translation2'].strip()}", "choice",
                 "מה היחס הלוגי בין ההנחה להשערה?", nli, list(nli), r["hebrew_label"],
                 "HebArabNlpProject/HebNLI HebNLI_test.jsonl (884-pair human-verified gold test)", LICENSES["hebnli"],
                 genre=r["genre"])
            for r in jsonl(hf("HebArabNlpProject/HebNLI", "HebNLI_test.jsonl")) if r.get("hebrew_label") in nli]
    write(EXT / "hebnli.jsonl", rows)
    counts["hebnli"] = len(rows)

    # Global-MMLU he test: 600 stratified by subject
    g = pd.read_parquet(hf("CohereLabs/Global-MMLU", "he/test-00000-of-00001.parquet"))
    picks = balanced({s: d.to_dict("records") for s, d in g.groupby("subject")}, 600, rng)
    rows = [mc4(f"ext-gmmlu-{r['sample_id'].replace('/', '-')}", "extraction", r["question"],
                [r[f"option_{k}"] for k in "abcd"], "ABCD".index(r["answer"]),
                f"CohereLabs/Global-MMLU he test {r['sample_id']}", LICENSES["gmmlu"],
                "בחרו את התשובה הנכונה.", subject=r["subject"], cultural_sensitivity=r["cultural_sensitivity_label"])
            for r in sorted(picks, key=lambda r: r["sample_id"])]
    write(EXT / "gmmlu.jsonl", rows)
    counts["gmmlu"] = len(rows)

    # HebrewSentiment (omilab) test, deduped vs train: 500, class-balanced as far as availability allows
    sent = sentiment_rows()
    by = defaultdict(list)
    for i, (t, lab) in enumerate(sent["test"]):
        by[lab].append((i, t, lab))
    crit = {v[0]: v[1] for v in SENT.values()}
    rows = [task(f"ext-sentiment-{i:04d}", "intent", f"תגובה בדף הפייסבוק של נשיא המדינה:\n{t}", "choice",
                 "מה הסנטימנט של התגובה כלפי הנשיא / הפוסט?", crit, list(crit), SENT[lab][0],
                 "omilab hebrew_sentiment token_test.tsv (deduped vs token_train)", LICENSES["sentiment"])
            for i, t, lab in sorted(balanced(by, 500, rng))]
    write(EXT / "sentiment.jsonl", rows)
    counts["sentiment"] = len(rows)
    counts["sentiment_test_after_dedupe"] = len(sent["test"])

    # Winograd-he (AlephBench): all 278
    rows = [task(f"ext-winograd-{i:03d}", "extraction", f"{r['context']}\nשאלה: {r['question']}", "choice",
                 "למי מתייחס הכינוי בשאלה? בחרו את הפירוש הנכון.", {"a": r["option1"], "b": r["option2"]},
                 ["a", "b"], "ab"[int(r["label"]) - 1], "HebArabNlpProject/AlephBench data/Winograd/test.jsonl",
                 LICENSES["winograd"])
            for i, r in enumerate(jsonl(hf("HebArabNlpProject/AlephBench", "data/Winograd/test.jsonl")))]
    write(EXT / "winograd.jsonl", rows)
    counts["winograd"] = len(rows)

    # HeQ test: 400 noul "answerable from the passage?", 200/200
    qs = heq_split("test")
    picks = balanced({imp: [q for q in qs if q["is_impossible"] == imp] for imp in (True, False)}, 400, rng)
    noul = {"true": "התשובה לשאלה מופיעה במפורש בקטע", "false": "הקטע אינו מכיל את התשובה לשאלה"}
    rows = [task(f"ext-heq-{q['id']}", "adequacy", f"קטע ({q['title']}):\n{q['context']}\n\nשאלה: {q['question']}",
                 "noul", "האם ניתן לענות על השאלה על סמך הקטע בלבד?", noul, ["no", "yes"],
                 "no" if q["is_impossible"] else "yes", "NNLP-IL HeQ v1.1 test", LICENSES["heq"],
                 heq_source=q["source"])
            for q in sorted(picks, key=lambda q: q["id"])]
    write(EXT / "heq.jsonl", rows)
    counts["heq"] = len(rows)
    readme(counts)
    return counts


def readme(counts: dict) -> None:
    lab = lambda f: dict(Counter(json.loads(line)["expected"] for line in (EXT / f"{f}.jsonl").open()))  # noqa: E731
    lines = ["# R055 external Hebrew test panel (EVAL-ONLY)", "",
             f"{EVAL_ONLY} Built by `scripts/r055_hebrew_open.py panel` (seed {SEED}). JevBench task format "
             "(same as `hejev-v0/translated.jsonl`); every row loads via `jevbench.tasks.Task.from_dict`.", "",
             "| file | rows | type | source / sampling | license | label counts |", "|---|---|---|---|---|---|"]
    spec = {"belebele": ("choice-4", "facebook/belebele heb_Hebr, all", LICENSES["belebele"]),
            "hebnli": ("choice-3", "HebNLI human-verified gold test (HebArabNlpProject/HebNLI HebNLI_test.jsonl), all",
                       LICENSES["hebnli"]),
            "gmmlu": ("choice-4", "Global-MMLU `he` test, 600 stratified round-robin over 57 subjects",
                      LICENSES["gmmlu"]),
            "sentiment": ("choice-3", f"omilab token_test, exact-dedup vs token_train and itself "
                                      f"({counts['sentiment_test_after_dedupe']} left), 500 class-balanced where "
                                      "possible; labels positive/negative/off_topic", LICENSES["sentiment"]),
            "winograd": ("choice-2", "AlephBench Winograd (He-WSC), all", LICENSES["winograd"]),
            "heq": ("noul", "HeQ v1.1 test (GitHub; the Etelis/HeQ_v1 HF mirror has is_impossible inverted), "
                            "200 answerable + 200 unanswerable; 'answerable from the passage?'", LICENSES["heq"])}
    for f, (t, s, lic) in spec.items():
        lines.append(f"| {f}.jsonl | {counts[f]} | {t} | {s} | {lic} | {lab(f)} |")
    lines += ["", "HebNLI caveat: the NNLP-IL repo has no license; the data is a Gemini translation of MultiNLI "
              "(OANC portions + CC-BY-SA-3.0 fiction), and the HF copy is tagged 'other'. Use for evaluation only; "
              "do not redistribute without review.", ""]
    (EXT / "README.md").write_text("\n".join(lines))
    print(f"README -> {(EXT / 'README.md').relative_to(ROOT)}")


# --------------------------------------------------------------------------- B: open train rows


def train_row(rid, source, license_, state, qtype, instructions, criteria, label, **meta):
    return {"id": rid, "state": state,
            "questions": {"decision": {"type": qtype, "instructions": instructions, "criteria": criteria}},
            "targets": {"decision": {"label": label}},
            "meta": {"lang": "he", "source": source, "license": license_, **meta}}


def massive_rows(n: int, rng: random.Random) -> list[dict]:
    script = hf("AmazonScience/massive", "massive.py").read_text()
    intents = ast.literal_eval(re.search(r"_INTENTS = (\[.*?\])", script, re.S).group(1))
    df = pd.read_parquet(hf("AmazonScience/massive", "he-IL/train/0000.parquet", revision="refs/convert/parquet"))
    df["name"] = df["intent"].map(lambda i: intents[i])
    scen = {name: name.split("_")[0] for name in intents}
    by_scen = defaultdict(set)
    for name in intents:
        by_scen[scen[name]].add(name)
    picks = balanced({k: d.to_dict("records") for k, d in df.groupby("name")}, n, rng)
    rows = []
    for r in picks:
        others = sorted(set(intents) - by_scen[scen[r["name"]]])
        options = sorted(by_scen[scen[r["name"]]] | set(rng.sample(others, 3)))[:12]
        if r["name"] not in options:
            options = sorted(set(options[:11]) | {r["name"]})
        rng.shuffle(options)
        rows.append(train_row(f"r055-massive-he-{r['id']}", "AmazonScience/massive he-IL train", LICENSES["massive"],
                              f"פנייה לעוזר קולי: {r['utt']}", "choice", "מהי הכוונה של המשתמש בפנייה?",
                              {o: o.replace("_", " ") for o in options}, r["name"]))
    return rows


def spans(context: str, gold: str, k: int, rng: random.Random, preferred: list[str]) -> list[str]:
    """k distractor spans: other answers / wrong answers from the same passage first, then random word windows."""
    out = []
    for s in rng.sample(preferred, len(preferred)):
        s = s.strip()
        if s and s not in out and s != gold and s not in gold and gold not in s:
            out.append(s)
    words = context.split()
    width = max(1, len(gold.split()))
    for _ in range(50):
        if len(out) >= k or len(words) <= width:
            break
        i = rng.randrange(0, len(words) - width + 1)
        s = " ".join(words[i:i + width]).strip(".,;:()\"'")
        if s and s not in out and s != gold and s not in gold and gold not in s:
            out.append(s)
    return out[:k]


def heq_rows(n: int, rng: random.Random) -> list[dict]:
    qs = heq_split("train")
    by_ctx = defaultdict(list)
    for q in qs:
        by_ctx[q["context"]].append(q)
    noul = {"no": "הקטע אינו מכיל את התשובה לשאלה", "yes": "התשובה לשאלה מופיעה במפורש בקטע"}
    picks = balanced({imp: [q for q in qs if q["is_impossible"] == imp] for imp in (True, False)}, n // 2, rng)
    rows = [train_row(f"r055-heq-ans-{q['id']}", "NNLP-IL HeQ v1.1 train", LICENSES["heq"],
                      f"קטע ({q['title']}):\n{q['context']}\n\nשאלה: {q['question']}", "noul",
                      "האם ניתן לענות על השאלה על סמך הקטע בלבד?", noul, "no" if q["is_impossible"] else "yes")
            for q in picks]
    used = {q["id"] for q in picks}
    pool = [q for q in qs if not q["is_impossible"] and q["answers"] and q["id"] not in used]
    for q in rng.sample(pool, len(pool)):
        if len(rows) >= n:
            break
        gold = q["answers"][0]["text"].strip()
        preferred = [a["text"] for o in by_ctx[q["context"]] if o["id"] != q["id"] for a in o["answers"]]
        preferred += [a["text"] for a in q.get("wrong_answers") or []]
        options = spans(q["context"], gold, 3, rng, preferred) + [gold]
        if len(options) < 4:
            continue
        rng.shuffle(options)
        rows.append(train_row(f"r055-heq-span-{q['id']}", "NNLP-IL HeQ v1.1 train", LICENSES["heq"],
                              f"קטע ({q['title']}):\n{q['context']}\n\nשאלה: {q['question']}", "choice",
                              "איזה קטע טקסט מתוך הפסקה עונה על השאלה?",
                              {LETTERS[i]: o for i, o in enumerate(options)}, LETTERS[options.index(gold)]))
    return rows


def sentiment_train(n: int, rng: random.Random) -> list[dict]:
    by = defaultdict(list)
    for i, (t, lab) in enumerate(sentiment_rows()["train"]):
        by[lab].append((i, t, lab))
    crit = {v[0]: v[1] for v in SENT.values()}
    return [train_row(f"r055-sentiment-{i:05d}", "omilab hebrew_sentiment token_train (deduped vs test)",
                      LICENSES["sentiment"], f"תגובה בדף הפייסבוק של נשיא המדינה:\n{t}", "choice",
                      "מה הסנטימנט של התגובה כלפי הנשיא / הפוסט?", crit, SENT[lab][0])
            for i, t, lab in sorted(balanced(by, n, rng))]


def cmd_train(rng: random.Random) -> dict:
    parts = {"massive": massive_rows(3000, rng), "heq": heq_rows(3000, rng), "sentiment": sentiment_train(2000, rng)}
    rows = [r for p in parts.values() for r in p]
    assert len({r["id"] for r in rows}) == len(rows)
    write(D2 / "open_train.jsonl", rows)
    return {k: len(v) for k, v in parts.items()}


# --------------------------------------------------------------------------- D: decontamination


def grams(text: str, n: int = 13) -> set[int]:
    words = re.findall(r"\w+", norm(text).lower())
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def texts_of_task(x) -> list[str]:
    """Every string leaf outside provenance (task rows and hejev qa.jsonl records alike)."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [t for k, v in x.items() if k != "provenance" for t in texts_of_task(v)]
    if isinstance(x, list):
        return [t for v in x for t in texts_of_task(v)]
    return []


def texts_of_row(r: dict) -> list[str]:
    out = [r["state"]]
    for q in r["questions"].values():
        crit = q.get("criteria") or {}
        out += [q.get("instructions") or "", *map(str, crit.values() if isinstance(crit, dict) else crit)]
    return out


def cmd_decon() -> dict:
    tests = [p for p in sorted(HEJEV.glob("*.jsonl")) + sorted(EXT.glob("*.jsonl"))]
    freq = Counter()
    for p in tests:
        for r in jsonl(p):
            freq.update(set().union(*(grams(t) for t in texts_of_task(r))))
    # ponytail: drop n-grams shared by >= 20 test items (our own prompt boilerplate), not content
    index = {g for g, c in freq.items() if c < 20}
    assert grams(jsonl(EXT / "belebele.jsonl")[0]["state"]) & index, "self-check: a test passage must hit the index"
    report = {"test_files": [str(p.relative_to(ROOT)) for p in tests], "index_13grams": len(index),
              "boilerplate_13grams_ignored": len(freq) - len(index)}
    for name in ("open_train.jsonl", "native_unlabeled.jsonl"):
        path = D2 / name
        if not path.exists():
            report[name] = "missing"
            continue
        rows = jsonl(path)
        keep, dropped = [], Counter()
        for r in rows:
            if any(grams(t) & index for t in texts_of_row(r)):
                dropped[r["meta"]["source"]] += 1
            else:
                keep.append(r)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
        report[name] = {"before": len(rows), "after": len(keep), "dropped_by_source": dict(dropped)}
        print(f"decon {name}: {len(rows)} -> {len(keep)} {dict(dropped)}")
    (D2 / "decon_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=("panel", "train", "decon"))
    args = ap.parse_args(argv)
    for stage in args.stages:
        rng = random.Random(SEED)
        out = {"panel": cmd_panel, "train": cmd_train}.get(stage)
        print(stage, json.dumps(out(rng) if out else cmd_decon(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
