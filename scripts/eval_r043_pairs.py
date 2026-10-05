"""Score every R043 counterfactual presentation and its executable action."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


EXPECTED_PAIRS = 256
EXPECTED_PRESENTATIONS = 512


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _probabilities(item, row_id):
    values = item.get("probabilities")
    if not isinstance(values, dict) or set(values) != {"no", "yes"}:
        raise ValueError(f"{row_id}: prediction must contain exactly no/yes probabilities")
    try:
        no, yes = float(values["no"]), float(values["yes"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{row_id}: probabilities must be numeric") from error
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (no, yes)):
        raise ValueError(f"{row_id}: invalid probability")
    if not math.isclose(no + yes, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{row_id}: probabilities must sum to one")
    return no, yes


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def _slice(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["source_id"]].append(row)
    if any(len(items) != 2 for items in pairs.values()):
        raise ValueError("every source must have exactly two counterfactual presentations")
    return {
        "pairs": len(pairs),
        "presentations": len(rows),
        "accuracy": _mean(rows, "correct"),
        "exact_action_accuracy": _mean(rows, "action_correct"),
        "twin_both_correct": sum(all(item["correct"] for item in items)
                                 for items in pairs.values()) / len(pairs),
        "nll": _mean(rows, "nll"),
        "brier": _mean(rows, "brier"),
    }


def audit(rows, predictions):
    """Validate the frozen panel and return full-denominator semantic/action metrics."""
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"expected {EXPECTED_PAIRS} evaluation pairs, got {len(rows)}")
    if any(row.get("pair_mode") != "counterfactual" for row in rows):
        raise ValueError("evaluation contains a non-counterfactual pair")
    views = [view for pair in rows for view in pair.get("paired_views", [])]
    if len(views) != EXPECTED_PRESENTATIONS:
        raise ValueError(f"expected {EXPECTED_PRESENTATIONS} presentations, got {len(views)}")
    ids = [view.get("id") for view in views]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("presentation IDs must be present and unique")
    prediction_ids = [item.get("id") for item in predictions]
    if None in prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction IDs must be present and unique")
    if set(ids) != set(prediction_ids) or len(predictions) != EXPECTED_PRESENTATIONS:
        raise ValueError("predictions must cover every presentation exactly once")
    by_id = {item["id"]: item for item in predictions}

    scored = []
    source_ids = set()
    for pair in rows:
        pair_views = pair["paired_views"]
        if len(pair_views) != 2:
            raise ValueError(f"{pair.get('id')}: pair must have exactly two views")
        pair_sources = {view.get("meta", {}).get("source_id") for view in pair_views}
        if len(pair_sources) != 1 or None in pair_sources:
            raise ValueError(f"{pair.get('id')}: counterfactual views must share one source")
        source_id = pair_sources.pop()
        if source_id in source_ids:
            raise ValueError(f"duplicate source ID: {source_id}")
        source_ids.add(source_id)
        if {view.get("targets", {}).get("decision", {}).get("label") for view in pair_views} != {"no", "yes"}:
            raise ValueError(f"{pair.get('id')}: pair labels must flip")
        for view in pair_views:
            meta = view.get("meta", {})
            family, polarity = meta.get("family"), meta.get("polarity")
            action_gold = meta.get("action_gold")
            if not family or polarity not in {"positive", "negative"} or action_gold not in {"execute", "withhold"}:
                raise ValueError(f"{view['id']}: incomplete executable-action metadata")
            no, yes = _probabilities(by_id[view["id"]], view["id"])
            gold = view["targets"]["decision"]["label"]
            queried_action = "execute" if polarity == "positive" else "withhold"
            expected = "yes" if queried_action == action_gold else "no"
            if gold != expected:
                raise ValueError(f"{view['id']}: label and executable action disagree")
            predicted_label = "yes" if yes >= no else "no"
            predicted_action = (queried_action if predicted_label == "yes" else
                                ("withhold" if queried_action == "execute" else "execute"))
            gold_yes = gold == "yes"
            scored.append({
                "id": view["id"], "source_id": source_id, "family": family,
                "polarity": polarity, "yes_first": bool(meta.get("yes_first")),
                "correct": predicted_label == gold,
                "action_correct": predicted_action == action_gold,
                "nll": -math.log(max(yes if gold_yes else no, 1e-12)),
                "brier": (yes - float(gold_yes)) ** 2,
            })

    result = _slice(scored)
    families = defaultdict(list)
    orders = defaultdict(list)
    polarities = defaultdict(list)
    for row in scored:
        families[row["family"]].append(row)
        orders["yes_first" if row["yes_first"] else "no_first"].append(row)
        polarities[row["polarity"]].append(row)
    if len(families) != 8 or any(len(items) != 64 for items in families.values()):
        raise ValueError("expected eight balanced 32-pair families")
    if set(orders) != {"no_first", "yes_first"} or set(polarities) != {"negative", "positive"}:
        raise ValueError("option order and polarity must both expose two cells")
    result["families"] = {key: _slice(value) for key, value in sorted(families.items())}
    result["option_order"] = {key: _slice(value) for key, value in sorted(orders.items())}
    result["polarity"] = {key: _slice(value) for key, value in sorted(polarities.items())}
    result["option_order_accuracy_delta"] = abs(
        result["option_order"]["yes_first"]["accuracy"] -
        result["option_order"]["no_first"]["accuracy"])
    result["polarity_accuracy_delta"] = abs(
        result["polarity"]["positive"]["accuracy"] -
        result["polarity"]["negative"]["accuracy"])
    result["source_metrics"] = {
        source_id: {
            "family": items[0]["family"],
            "exact_action_accuracy": _mean(items, "action_correct"),
            "twin_both_correct": float(all(item["correct"] for item in items)),
        }
        for source_id, items in sorted((source, [row for row in scored if row["source_id"] == source])
                                       for source in source_ids)
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if not args.model_revision.strip():
        raise ValueError("model revision must not be empty")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from scripts.score_action_panel import score_group
    from scripts.train_gemma_decision_lora import letter_token_groups

    rows = _read(args.data)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for pair in rows:
            for view in pair.get("paired_views", []):
                probabilities = score_group(
                    model, tokenizer, groups, view, device, args.max_length)["q00"]
                predictions.append({"id": view["id"], "probabilities": probabilities})
    report = audit(rows, predictions)
    report["input_sha256"] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    report["model_revision"] = args.model_revision
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as target:
        for item in predictions:
            target.write(json.dumps(item, sort_keys=True) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
