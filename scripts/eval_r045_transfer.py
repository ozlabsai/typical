"""Score the held-out R045 compositional transfer panel."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.build_r045_compositional_transfer import execute


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _probs(item, row_id):
    values = item.get("probabilities")
    if not isinstance(values, dict) or set(values) != {"no", "yes"}:
        raise ValueError(f"{row_id}: prediction must contain exactly no/yes probabilities")
    no, yes = (float(values[label]) for label in ("no", "yes"))
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (no, yes)):
        raise ValueError(f"{row_id}: invalid probability")
    if not math.isclose(no + yes, 1, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{row_id}: probabilities must sum to one")
    return no, yes


def _summary(items):
    return {
        "presentations": len(items),
        "accuracy": sum(item["correct"] for item in items) / len(items),
        "nll": sum(item["nll"] for item in items) / len(items),
        "brier": sum(item["brier"] for item in items) / len(items),
    }


def audit(rows, predictions):
    """Validate complete predictions and report pair/language/composition transfer."""
    views = [view for row in rows for view in row.get("paired_views", [])]
    ids = [view.get("id") for view in views]
    prediction_ids = [item.get("id") for item in predictions]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("presentation IDs must be present and unique")
    if None in prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction IDs must be present and unique")
    if set(ids) != set(prediction_ids) or len(predictions) != len(views):
        raise ValueError("predictions must cover every presentation exactly once")
    by_id = {item["id"]: item for item in predictions}
    scored, pair_scores, sources, scenarios = [], {}, defaultdict(list), defaultdict(list)
    for row in rows:
        if row.get("pair_mode") != "counterfactual" or len(row.get("paired_views", [])) != 2:
            raise ValueError(f"{row.get('id')}: expected one two-view counterfactual pair")
        pair = []
        for view in row["paired_views"]:
            meta = view.get("meta", {})
            gold_value = execute(meta["program"], meta["clauses"])
            gold = view["targets"]["decision"]["label"]
            if gold != ("yes" if gold_value else "no") or gold_value is not meta.get("gold"):
                raise ValueError(f"{view['id']}: target disagrees with executable gold")
            no, yes = _probs(by_id[view["id"]], view["id"])
            predicted = "yes" if yes >= no else "no"
            probability = yes if gold == "yes" else no
            item = {"id": view["id"], "source_id": meta["source_id"],
                    "scenario_id": meta["scenario_id"], "decisive_kind": meta["decisive_kind"],
                    "language": meta["language"], "composition": "+".join(meta["composition"]),
                    "yes_first": next(iter(view["questions"]["decision"]["criteria"])) == "yes",
                    "correct": predicted == gold, "nll": -math.log(max(probability, 1e-12)),
                    "brier": (yes - float(gold == "yes")) ** 2}
            scored.append(item)
            pair.append(item)
            sources[item["source_id"]].append(item)
            scenarios[item["scenario_id"]].append(item)
        if {view["targets"]["decision"]["label"] for view in row["paired_views"]} != {"no", "yes"}:
            raise ValueError(f"{row['id']}: counterfactual labels must flip")
        pair_scores[row["id"]] = all(item["correct"] for item in pair)
    if any(len(items) != 4 for items in sources.values()):
        raise ValueError("each logical source must contain English and Hebrew counterfactual pairs")
    if any(len(items) != 4 * len(set(items[0]["composition"].split("+")))
           or {item["decisive_kind"] for item in items} != set(items[0]["composition"].split("+"))
           for items in scenarios.values()):
        raise ValueError("each scenario must contain one complete bilingual pair per clause")

    def slices(key):
        grouped = defaultdict(list)
        for item in scored:
            grouped[str(item[key])].append(item)
        return {name: _summary(items) for name, items in sorted(grouped.items())}

    result = _summary(scored)
    result.update({
        "pairs": len(rows),
        "pair_both_correct": sum(pair_scores.values()) / len(pair_scores),
        "logical_sources": len(sources),
        "source_all_four_correct": sum(all(item["correct"] for item in items)
                                       for items in sources.values()) / len(sources),
        "scenarios": len(scenarios),
        "scenario_all_pivots_correct": sum(all(item["correct"] for item in items)
                                           for items in scenarios.values()) / len(scenarios),
        "language": slices("language"),
        "composition": slices("composition"),
        "option_order": slices("yes_first"),
    })
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

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from scripts.score_action_panel import score_group
    from scripts.train_gemma_decision_lora import letter_token_groups

    rows = _read(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to("cuda").eval()
    predictions = []
    with torch.no_grad():
        for row in rows:
            for view in row["paired_views"]:
                predictions.append({"id": view["id"], "probabilities":
                                    score_group(model, tokenizer, groups, view, torch.device("cuda"),
                                                args.max_length)["q00"]})
    report = audit(rows, predictions)
    report.update({"input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                   "model_revision": args.model_revision})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions.jsonl").write_text("".join(
        json.dumps(item, sort_keys=True) + "\n" for item in predictions))
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
