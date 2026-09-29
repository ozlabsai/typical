"""Build the first multimodal and Hebrew pilot corpora.

Examples:
  uv run scripts/pilot_corpora.py hebrew --out data_he
  uv run scripts/pilot_corpora.py scienceqa --out data_v --accept-noncommercial
"""
import argparse
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pcdm"))
from data import cls_examples, words, write_jsonl  # noqa: E402


MASSIVE = "hf://datasets/mteb/amazon_massive_intent/{split}/he.json.gz"
HEBREW_QUERIES = (
    "מהי כוונת המשתמש?",
    "מה המשתמש רוצה לעשות?",
    "איזו כוונה מתאימה לבקשה?",
    "מה מטרת ההודעה?",
)


def build_hebrew(out: Path, seed: int = 0, limit: int = 0):
    raw = {split: list(load_dataset("json", data_files=MASSIVE.format(split=split), split="train"))
           for split in ("train", "validation", "test")}
    labels = sorted({row["label_text"] for rows in raw.values() for row in rows})
    label_id = {label: i for i, label in enumerate(labels)}
    names = [words(label) for label in labels]
    rng = random.Random(seed)
    result = {}
    for split, rows in raw.items():
        if limit:
            rows = rows[:limit]
        pairs = [(row["text"], label_id[row["label_text"]]) for row in rows]
        k_choice = None if split == "train" else lambda _rng, n: (min(10, n), True)
        examples = cls_examples(pairs, "massive_he", "intent", names, rng, k_choice=k_choice,
                                qidx=0, paraphrase=False, sibling_null=split == "train")
        for example in examples:
            example["query"] = rng.choice(HEBREW_QUERIES)
            example["meta"].update({"fam_bucket": "H", "lang": "he", "source": "MASSIVE"})
        result[split] = examples
    write_jsonl(out / "train.jsonl", result["train"])
    write_jsonl(out / "val.jsonl", result["validation"])
    write_jsonl(out / "eval" / "massive_he_test.jsonl", result["test"])
    return {split: len(rows) for split, rows in result.items()}


def _scienceqa_split(split: str, image_dir: Path, json_dir: Path, null_fraction: float, seed: int, limit: int):
    rows, rng = [], random.Random(seed)
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)
    for index, row in enumerate(dataset):
        if limit and len(rows) >= limit:
            break
        if row["image"] is None or row["task"] != "closed choice":
            continue
        image_path = image_dir / f"{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(image_path)
        candidates, gold = row["choices"], int(row["answer"])
        example = {
            "state": row["hint"].strip() or "Use the image as evidence.",
            "query": row["question"], "candidates": candidates,
            "target": [float(i == gold) for i in range(len(candidates))],
            "p_null": 0.0, "task": "scienceqa_image", "label": gold,
            "image": os.path.relpath(image_path, json_dir),
            "meta": {"fam_bucket": "V", "qtype": "choice", "source": "ScienceQA",
                     "license": "CC-BY-NC-SA-4.0"},
        }
        rows.append(example)
        if split == "train" and len(candidates) > 2 and rng.random() < null_fraction:
            kept = [candidate for i, candidate in enumerate(candidates) if i != gold]
            null = dict(example, candidates=kept, target=[1 / len(kept)] * len(kept), p_null=1.0, label=-1)
            null["meta"] = dict(example["meta"], null_aug=True)
            rows.append(null)
    return rows


def build_scienceqa(out: Path, seed: int = 0, limit: int = 0, null_fraction: float = 0.1):
    train = _scienceqa_split("train", out / "images" / "train", out, null_fraction, seed, limit)
    val = _scienceqa_split("validation", out / "images" / "validation", out, 0, seed + 1, limit)
    test = _scienceqa_split("test", out / "images" / "test", out / "eval", 0, seed + 2, limit)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "eval" / "scienceqa_test.jsonl", test)
    shuffled = [dict(row, image=test[(i + 1) % len(test)]["image"]) for i, row in enumerate(test)]
    write_jsonl(out / "eval" / "scienceqa_test_shuffled.jsonl", shuffled)
    write_jsonl(out / "eval" / "scienceqa_test_noimage.jsonl", [{k: v for k, v in row.items() if k != "image"}
                                                                  for row in test])
    return {"train": len(train), "validation": len(val), "test": len(test)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="corpus", required=True)
    for name in ("hebrew", "scienceqa"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--limit", type=int, default=0)
    sub.choices["scienceqa"].add_argument("--null-fraction", type=float, default=0.1)
    sub.choices["scienceqa"].add_argument("--accept-noncommercial", action="store_true")
    args = parser.parse_args()
    if args.corpus == "scienceqa":
        if not args.accept_noncommercial:
            parser.error("ScienceQA is CC-BY-NC-SA-4.0; pass --accept-noncommercial for research-only weights")
        counts = build_scienceqa(args.out, args.seed, args.limit, args.null_fraction)
    else:
        counts = build_hebrew(args.out, args.seed, args.limit)
    print(", ".join(f"{split}={count}" for split, count in counts.items()))


if __name__ == "__main__":
    main()
