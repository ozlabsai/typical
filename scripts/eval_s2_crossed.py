"""Score the locked S2 crossed panel with the trainer's exact letter readout."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_panel(path):
    try:
        from scripts.train_gemma_decision_lora import flatten_row
    except ModuleNotFoundError:
        from train_gemma_decision_lora import flatten_row

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 2048:
        raise ValueError(f"crossed panel must contain exactly 2,048 rows; got {len(rows)}")
    examples = []
    for row in rows:
        flattened = flatten_row(row)
        if len(flattened) != 1:
            raise ValueError(f"{row.get('id')}: expected exactly one flattened question")
        example = flattened[0]
        meta = row.get("meta") or {}
        if row["questions"]["decision"].get("type") != "noul":
            raise ValueError(f"{row.get('id')}: crossed eval only accepts Noul questions")
        example.update({
            "row_id": row["id"], "source_group": meta["source_group"],
            "source_id": meta["source_id"], "question_style": meta["question_style"],
            "criteria_style": meta["criteria_style"], "polarity": meta["polarity"],
            "context_view": meta["context_view"], "placement": meta["decisive_placement"],
            "option_order": tuple(key for key, _ in example["items"]),
            "gold": row["targets"]["decision"]["label"],
        })
        examples.append(example)
    cells = Counter((example["question_style"], example["criteria_style"],
                     example["polarity"], example["context_view"],
                     "_".join(example["option_order"])) for example in examples)
    collapsed_cells = Counter((example["question_style"], example["criteria_style"],
                               example["context_view"], "_".join(example["option_order"]))
                              for example in examples)
    labels = Counter(example["gold"] for example in examples)
    if len({example["source_group"] for example in examples}) != 64:
        raise ValueError("crossed panel must contain 64 rubric groups")
    if len(cells) != 32 or set(cells.values()) != {64}:
        raise ValueError("crossed panel must contain 64 examples in each of 32 polarity-aware cells")
    if len(collapsed_cells) != 16 or set(collapsed_cells.values()) != {128}:
        raise ValueError("crossed panel must collapse to 128 examples in each of 16 cells")
    if labels != {"no": 1024, "yes": 1024}:
        raise ValueError(f"crossed panel labels are not balanced: {dict(labels)}")
    pairs = defaultdict(list)
    for example in examples:
        key = (example["source_id"], example["question_style"], example["criteria_style"],
               example["polarity"], example["context_view"], example["placement"])
        pairs[key].append(example["option_order"])
    if any(Counter(pair) != {("no", "yes"): 1, ("yes", "no"): 1} for pair in pairs.values()):
        raise ValueError("crossed panel contains incomplete or duplicate option-order pairs")
    return examples


def aggregate_predictions(records):
    if not records:
        raise ValueError("cannot aggregate an empty prediction set")

    def metrics(rows):
        n = len(rows)
        return {
            "n": n,
            "accuracy": sum(row["correct"] for row in rows) / n,
            "nll": sum(row["nll"] for row in rows) / n,
            "brier": sum(row["brier"] for row in rows) / n,
        }

    cells = defaultdict(list)
    groups = defaultdict(list)
    labels = defaultdict(list)
    orders = defaultdict(list)
    failures = []
    for row in records:
        cell = (row["question_style"], row["criteria_style"], row["polarity"],
                row["context_view"], "_".join(row["option_order"]))
        cells[cell].append(row)
        groups[row["source_group"]].append(row)
        labels[row["gold"]].append(row)
        pair = (row["source_id"], row["question_style"], row["criteria_style"],
                row["polarity"], row["context_view"], row["placement"])
        orders[pair].append(row)
        if not row["correct"]:
            failures.append({key: row[key] for key in
                             ("example_id", "source_group", "source_id", "gold", "predicted", "p_yes")})

    pairs = []
    for key, pair_rows in orders.items():
        by_order = {tuple(row["option_order"]): row for row in pair_rows}
        if (len(pair_rows) != 2 or
                set(by_order) != {("no", "yes"), ("yes", "no")}):
            raise ValueError(f"order pair incomplete for {key}")
        first, second = by_order[("no", "yes")], by_order[("yes", "no")]
        pairs.append({"both_correct": first["correct"] and second["correct"],
                      "p_yes_delta": abs(first["p_yes"] - second["p_yes"])})

    n_groups = len(groups)
    return {
        "overall": metrics(records),
        "cells": {"|".join(key): metrics(rows) for key, rows in sorted(cells.items())},
        "per_condition_recall": {
            label: {"n": len(rows), "recall": sum(row["correct"] for row in rows) / len(rows)}
            for label, rows in sorted(labels.items())
        },
        "per_group_all_correct": {
            "n_groups": n_groups,
            "all_correct_groups": sum(all(row["correct"] for row in rows) for rows in groups.values()),
            "all_correct_rate": sum(all(row["correct"] for row in rows) for rows in groups.values()) / n_groups,
        },
        "order_robustness": {
            "n_pairs": len(pairs),
            "both_correct_rate": sum(pair["both_correct"] for pair in pairs) / len(pairs),
            "mean_abs_aligned_p_yes_delta": sum(pair["p_yes_delta"] for pair in pairs) / len(pairs),
        },
        "failures": failures,
    }


def score_adapter(model_name, adapter_name, adapter_path, examples, max_length):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    try:
        from scripts.train_gemma_decision_lora import encode, letter_token_groups
    except ModuleNotFoundError:
        from train_gemma_decision_lora import encode, letter_token_groups

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=dtype, local_files_only=True, attn_implementation="sdpa"
    )
    if adapter_path.lower() not in {"base", "none", "-"}:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.to(device).eval()
    model.config.use_cache = False
    token_groups = letter_token_groups(tokenizer)
    predictions, lengths = [], []

    with torch.inference_mode():
        for index, example in enumerate(examples):
            input_ids, _ = encode(example, tokenizer, max_length, shuffle=False)
            lengths.append(len(input_ids))
            ids = input_ids.unsqueeze(0).to(device)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=1)
            logits = output.logits[0, -1].float()
            option_scores = torch.stack([
                torch.logsumexp(logits[token_ids.to(device)], dim=0)
                for token_ids in token_groups[:len(example["items"])]
            ])
            probs = F.softmax(option_scores, dim=0).cpu().tolist()
            label_probs = {item[0]: prob for item, prob in zip(example["items"], probs)}
            predicted = max(label_probs, key=label_probs.__getitem__)
            p_yes = label_probs.get("yes", 0.0)
            gold = example["gold"]
            p_gold = max(label_probs[gold], 1e-12)
            y_yes = float(gold == "yes")
            predictions.append({
                "example_id": example["row_id"], "source_group": example["source_group"],
                "source_id": example["source_id"], "question_style": example["question_style"],
                "criteria_style": example["criteria_style"], "polarity": example["polarity"],
                "context_view": example["context_view"], "placement": example["placement"],
                "option_order": example["option_order"], "gold": gold, "predicted": predicted,
                "p_yes": p_yes, "correct": predicted == gold, "nll": -math.log(p_gold),
                "brier": (p_yes - y_yes) ** 2,
            })
            if (index + 1) % 128 == 0:
                print(f"{adapter_name}: scored {index + 1}/{len(examples)}", flush=True)
    result = aggregate_predictions(predictions)
    result.update({"name": adapter_name, "adapter": adapter_path,
                   "prompt_tokens": {"n": len(lengths), "min": min(lengths),
                                     "max": max(lengths), "mean": sum(lengths) / len(lengths)}})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_adapter(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be NAME=PATH (use NAME=base for no adapter)")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("adapter must be NAME=PATH")
    return name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, required=True,
                        metavar="NAME=PATH", help="use NAME=base to score the unadapted base")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len({name for name, _ in args.adapter}) != len(args.adapter):
        parser.error("adapter names must be unique")
    examples = load_panel(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for name, path in args.adapter:
        results.append(score_adapter(args.model, name, path, examples, args.max_length))
        out.write_text(json.dumps({
            "model": args.model, "data": args.data,
            "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "rows": len(examples), "max_length": args.max_length,
            "complete": len(results) == len(args.adapter), "results": results,
        }, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
