"""Probe bare versus canonical rendering on paired action-panel fields."""

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from scripts.score_action_panel import score_group
from scripts.train_gemma_decision_lora import letter_token_groups


def select_approved(rows, per_label):
    """Select equal approval labels from the bare finance renderer."""
    selected = {"no": [], "yes": []}
    for row in rows:
        if row["meta"]["family"] != "finance" or row["meta"]["template"] != "terse":
            continue
        label = row["targets"]["q01"]["label"]
        if len(selected[label]) < per_label:
            selected[label].append(row)
    if any(len(items) != per_label for items in selected.values()):
        raise ValueError("insufficient balanced approval cases")
    return selected["no"] + selected["yes"]


def render(row, *, canonical, yes_first):
    """Keep one field and vary only its prompt renderer and option order."""
    item = copy.deepcopy(row)
    question = item["questions"]["q01"]
    if canonical:
        question["instructions"] = f"Is the following statement true? {question['instructions']}"
        question["criteria"] = {"no": "False", "yes": "True"}
    else:
        question["criteria"] = {"no": "No", "yes": "Yes"}
    if yes_first:
        question["criteria"] = dict(reversed(list(question["criteria"].items())))
    item["questions"] = {"q00": question}
    item["targets"] = {"q00": item["targets"]["q01"]}
    return item


def summarize(records):
    by_condition, paired = defaultdict(list), defaultdict(list)
    for item in records:
        key = f"{item['renderer']}_{item['label']}"
        by_condition[key].append(item["correct"])
        paired[(item["id"], item["renderer"])].append(item)
    report = {key: {"n": len(values), "accuracy": sum(values) / len(values)}
              for key, values in sorted(by_condition.items())}
    for renderer in ("bare", "canonical"):
        pairs = [values for (item_id, style), values in paired.items()
                 if style == renderer and len(values) == 2]
        deltas = [abs(values[0]["p_yes"] - values[1]["p_yes"]) for values in pairs]
        report[f"{renderer}_mean_order_delta"] = sum(deltas) / len(deltas) if deltas else None
        for label in ("no", "yes"):
            label_pairs = [values for values in pairs if values[0]["label"] == label]
            report[f"{renderer}_{label}_both_orders"] = (
                sum(all(item["correct"] for item in values) for values in label_pairs) / len(label_pairs)
                if label_pairs else None
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--per-label", type=int, default=65)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    data = Path(args.data)
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    rows = select_approved(rows, args.per_label)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to(device).eval()
    records = []
    with torch.no_grad():
        for row in rows:
            label = row["targets"]["q01"]["label"]
            for canonical in (False, True):
                for yes_first in (False, True):
                    probabilities = score_group(model, tokenizer, groups,
                                                render(row, canonical=canonical, yes_first=yes_first),
                                                device, args.max_length)["q00"]
                    records.append({"id": row["id"], "label": label,
                                    "renderer": "canonical" if canonical else "bare",
                                    "yes_first": yes_first, "p_yes": probabilities["yes"],
                                    "correct": (probabilities["yes"] >= 0.5) == (label == "yes")})
    report = summarize(records)
    report["input_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    report["per_label"] = args.per_label
    canonical_no = report["canonical_no_both_orders"]
    bare_yes, canonical_yes = report["bare_yes_both_orders"], report["canonical_yes_both_orders"]
    report["gates"] = {
        "bare_negative_recovery": report["bare_no_both_orders"] >= 0.90,
        "bare_positive_retention": report["bare_yes_both_orders"] >= 0.95,
        "bare_order_delta_at_most_.05": report["bare_mean_order_delta"] <= 0.05,
        "negative_recovery": canonical_no >= 0.8,
        "positive_loss_at_most_2": (bare_yes - canonical_yes) * args.per_label <= 2,
        "canonical_order_delta_at_most_.05": report["canonical_mean_order_delta"] <= 0.05,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as target:
        for item in records:
            target.write(json.dumps(item, sort_keys=True) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
