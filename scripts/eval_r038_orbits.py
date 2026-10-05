"""Score R038's source-disjoint semantic-orbit evaluation and audit its gates."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from scripts.score_action_panel import score_group
from scripts.train_gemma_decision_lora import letter_token_groups


def audit(rows, predictions):
    """Return factual-twin and renderer-robustness metrics from aligned scores."""
    by_id = {item["id"]: item for item in predictions}
    scored, twins, orders = [], defaultdict(list), defaultdict(list)
    format_polarity, labels = defaultdict(list), defaultdict(list)
    for row in rows:
        item = by_id.get(row["id"])
        if item is None:
            raise ValueError(f"missing prediction for {row['id']}")
        probabilities = item["probabilities"]
        p_yes, label = probabilities["yes"], row["targets"]["decision"]["label"]
        gold_yes = label == "yes"
        correct = (p_yes >= 0.5) == gold_yes
        meta = row["meta"]
        scored.append({"id": row["id"], "correct": correct, "p_yes": p_yes,
                       "nll": -math.log(max(p_yes if gold_yes else 1 - p_yes, 1e-12)),
                       "brier": (p_yes - float(gold_yes)) ** 2})
        twin_key = (meta["source_id"], meta["polarity"], meta["wording"],
                    meta["criteria_style"], meta["yes_first"])
        order_key = twin_key[:-1] + (meta["world"],)
        twins[twin_key].append(correct)
        orders[order_key].append((p_yes if meta["polarity"] == "positive" else 1 - p_yes, correct))
        format_polarity[(meta["wording"], meta["criteria_style"], meta["polarity"])].append(correct)
        labels[label].append(correct)
    if any(len(values) != 2 for values in twins.values()) or any(len(values) != 2 for values in orders.values()):
        raise ValueError("incomplete counterfactual or option-order pairs")
    accuracy = sum(item["correct"] for item in scored) / len(scored)
    return {
        "n": len(scored), "accuracy": accuracy,
        "nll": sum(item["nll"] for item in scored) / len(scored),
        "brier": sum(item["brier"] for item in scored) / len(scored),
        "counterfactual_twins_both_correct": sum(all(values) for values in twins.values()) / len(twins),
        "both_orders_correct": sum(all(item[1] for item in values) for values in orders.values()) / len(orders),
        "mean_semantic_order_delta": sum(abs(values[0][0] - values[1][0]) for values in orders.values()) / len(orders),
        "label_recall": {label: sum(values) / len(values) for label, values in sorted(labels.items())},
        "format_polarity_accuracy": {"/".join(key): sum(values) / len(values)
                                     for key, values in sorted(format_polarity.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to(device).eval()
    with torch.no_grad():
        predictions = [{"id": row["id"], "probabilities": score_group(
            model, tokenizer, groups, row, device, args.max_length)["q00"]} for row in rows]
    report = audit(rows, predictions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as target:
        for item in predictions:
            target.write(json.dumps(item, sort_keys=True) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
