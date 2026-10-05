"""Score grouped action-panel decisions and replay their executable policy.

The model is queried only for typed option probabilities.  Final actions and
the error taxonomy are deterministic post-processing, so calibration can be
fit on a disjoint panel split without retraining a checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForImageTextToText, AutoTokenizer

from scripts.build_action_panel import action_for
from scripts.train_gemma_decision_lora import encode, flatten_row, letter_token_groups


def classify(row, probabilities):
    """Replay a group's policy and identify its first actionable error class."""
    threshold = row["meta"]["decision_threshold"]
    fields = {qid: values["yes"] >= threshold for qid, values in probabilities.items()}
    raw = {qid: values["yes"] >= 0.5 for qid, values in probabilities.items()}
    gold = {qid: row["targets"][qid]["label"] == "yes" for qid in row["questions"]}
    predicted_action, gold_action = action_for(_named(row, fields)), row["meta"]["action_gold"]
    if predicted_action == gold_action:
        category = "correct"
    elif raw != gold:
        category = "ranking"
    elif fields != gold:
        category = "threshold"
    else:
        category = "composition"
    return predicted_action, gold_action, category


def _named(row, values):
    names = row["meta"]["action_trace"]
    return {name: values[f"q{index:02d}"] for index, name in enumerate(names)}


def summarize(rows, prediction_rows):
    by_id = {item["id"]: item for item in prediction_rows}
    counts = {name: 0 for name in ("correct", "ranking", "threshold", "composition", "invalid")}
    correct = 0
    for row in rows:
        item = by_id.get(row["id"])
        if item is None or set(item.get("probabilities", {})) != set(row["questions"]):
            counts["invalid"] += 1
            continue
        _, gold, category = classify(row, item["probabilities"])
        item["gold_action"], item["error_category"] = gold, category
        counts[category] += 1
        correct += category == "correct"
    return {"n": len(rows), "exact_action_accuracy": correct / len(rows) if rows else 0.0,
            "error_categories": counts}


@torch.no_grad()
def score_group(model, tokenizer, groups, row, device, max_length):
    examples = flatten_row(row)
    encoded = [encode(example, tokenizer, max_length, shuffle=False, include_labels=True)
               for example in examples]
    ids = pad_sequence([item[0] for item in encoded], batch_first=True,
                       padding_value=tokenizer.pad_token_id)
    mask = ids.ne(tokenizer.pad_token_id)
    positions = torch.tensor([len(item[0]) - 1 for item in encoded])
    unique_positions = torch.unique(positions, sorted=True)
    output = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False,
                   logits_to_keep=unique_positions.to(device))
    position_offsets = {position.item(): index for index, position in enumerate(unique_positions)}
    probabilities = {}
    for index, ((_, _, labels), position) in enumerate(zip(encoded, positions)):
        logits = output.logits[index, position_offsets[position.item()]].float()
        scores = torch.stack([torch.logsumexp(logits[token_ids.to(device)], dim=0)
                              for token_ids in groups[:len(labels)]])
        qid = f"q{index:02d}"
        probabilities[qid] = {label: float(value) for label, value in
                              zip(labels, F.softmax(scores, dim=0).tolist())}
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = rows[:args.limit] if args.limit else rows
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.config.use_cache = False
    model.to(device).eval()
    predictions = []
    for row in rows:
        probabilities = score_group(model, tokenizer, groups, row, device, args.max_length)
        predicted, gold, category = classify(row, probabilities)
        predictions.append({"id": row["id"], "probabilities": probabilities,
                            "predicted_action": predicted, "gold_action": gold,
                            "error_category": category})
    report = summarize(rows, predictions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as target:
        for item in predictions:
            target.write(json.dumps(item, sort_keys=True) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
