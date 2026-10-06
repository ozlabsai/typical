"""Score generic grouped Boolean policies with exact WorkflowEvals-style outcomes.

The model predicts semantic Boolean fields.  A validated deterministic policy
then produces a canonical primary action and exact action set.  Model execution
uses the shared singleton Jev-style scorer so BF16 results do not depend on a
question batch shape.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.exact_policy_executor import (
    CanonicalOutcome,
    canonicalize_outcome,
    compile_policy,
    execute_policy,
)


PANEL = "exact_policy_v1"
SLICE_KEYS = (
    "split",
    "language",
    "option_order",
    "policy_mode",
    "policy_topology",
    "topology_slice",
    "renderer_family",
    "source_family",
)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _row_contract(row):
    """Validate one panel row and return its executable contract."""
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("exact-policy row needs a nonempty string id")
    meta = row.get("meta")
    if not isinstance(meta, dict) or meta.get("panel") != PANEL:
        raise ValueError(f"{row['id']}: wrong exact-policy panel")
    questions, targets = row.get("questions"), row.get("targets")
    if not isinstance(questions, dict) or not questions or not isinstance(targets, dict):
        raise ValueError(f"{row['id']}: questions and targets must be nonempty objects")
    if set(questions) != set(targets):
        raise ValueError(f"{row['id']}: target coverage differs from questions")
    qids = [f"q{index:02d}" for index in range(len(questions))]
    if list(questions) != qids or set(targets) != set(qids):
        raise ValueError(f"{row['id']}: qids must be ordered and contiguous")

    fields = meta.get("field_order")
    mapping = meta.get("qid_to_field")
    if (not isinstance(fields, list) or len(fields) != len(qids)
            or len(set(fields)) != len(fields)
            or any(not isinstance(field, str) or not field for field in fields)):
        raise ValueError(f"{row['id']}: invalid field_order")
    expected_mapping = dict(zip(qids, fields))
    if mapping != expected_mapping:
        raise ValueError(f"{row['id']}: qid_to_field differs from field_order")

    threshold = meta.get("decision_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError(f"{row['id']}: invalid decision threshold")
    compiled = compile_policy(meta.get("policy"))
    if tuple(fields) != compiled.fields:
        raise ValueError(f"{row['id']}: policy fields differ from field_order")
    if meta.get("policy_mode") is not None and meta["policy_mode"] != compiled.mode:
        raise ValueError(f"{row['id']}: policy_mode metadata disagrees with policy")

    trace = meta.get("action_trace")
    if (not isinstance(trace, dict) or set(trace) != set(fields)
            or any(type(value) is not bool for value in trace.values())):
        raise ValueError(f"{row['id']}: invalid action_trace")
    target_trace = {}
    for qid, field in mapping.items():
        question, target = questions[qid], targets[qid]
        criteria = question.get("criteria") if isinstance(question, dict) else None
        label = target.get("label") if isinstance(target, dict) else None
        if (not isinstance(criteria, dict) or set(criteria) != {"no", "yes"}
                or label not in {"no", "yes"}):
            raise ValueError(f"{row['id']}/{qid}: fields require semantic no/yes labels")
        target_trace[field] = label == "yes"
    if target_trace != trace:
        raise ValueError(f"{row['id']}: targets disagree with action_trace")
    gold = canonicalize_outcome(meta.get("gold_outcome"))
    if execute_policy(compiled, trace) != gold:
        raise ValueError(f"{row['id']}: gold outcome disagrees with executable policy")
    return compiled, gold, mapping, float(threshold)


def _probability_contract(row, probabilities):
    if not isinstance(probabilities, dict) or set(probabilities) != set(row["questions"]):
        raise ValueError(f"{row['id']}: prediction qids differ from questions")
    clean = {}
    for qid in row["questions"]:
        values = probabilities[qid]
        if not isinstance(values, dict) or set(values) != {"no", "yes"}:
            raise ValueError(f"{row['id']}/{qid}: probabilities need no and yes")
        parsed = {}
        for label in ("no", "yes"):
            value = values[label]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{row['id']}/{qid}: invalid probability")
            parsed[label] = float(value)
        if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{row['id']}/{qid}: probabilities do not sum to one")
        clean[qid] = parsed
    return clean


def classify(row, probabilities):
    """Execute one prediction and identify its first field-level error class."""
    policy, gold, mapping, threshold = _row_contract(row)
    probabilities = _probability_contract(row, probabilities)
    predicted_trace = {
        field: probabilities[qid]["yes"] >= threshold for qid, field in mapping.items()
    }
    raw_trace = {
        field: probabilities[qid]["yes"] >= 0.5 for qid, field in mapping.items()
    }
    predicted = execute_policy(policy, predicted_trace)
    gold_trace = row["meta"]["action_trace"]
    if predicted == gold:
        category = "correct"
    elif raw_trace != gold_trace:
        category = "ranking"
    elif predicted_trace != gold_trace:
        category = "threshold"
    else:
        # This should be unreachable for validated deterministic rows, but it
        # remains explicit so any future executor mismatch fails visibly.
        category = "composition"
    return {
        "predicted_outcome": predicted,
        "gold_outcome": gold,
        "predicted_trace": predicted_trace,
        "gold_trace": gold_trace,
        "error_category": category,
    }


def _summary(records):
    n = len(records)
    exact = sum(record["predicted_outcome"] == record["gold_outcome"] for record in records)
    action_sets = sum(record["predicted_outcome"].actions == record["gold_outcome"].actions
                      for record in records)
    primary_records = [record for record in records
                       if record["predicted_outcome"].primary is not None
                       and record["gold_outcome"].primary is not None]
    primary = sum(record["predicted_outcome"].primary == record["gold_outcome"].primary
                  for record in primary_records)
    fields = sum(len(record["gold_trace"]) for record in records)
    correct_fields = sum(
        sum(record["predicted_trace"][field] == value
            for field, value in record["gold_trace"].items())
        for record in records
    )
    vectors = sum(record["predicted_trace"] == record["gold_trace"] for record in records)
    return {
        "n": n,
        "exact_outcome_accuracy": _rate(exact, n),
        "exact_action_set_accuracy": _rate(action_sets, n),
        "primary_action_eligible": len(primary_records),
        "primary_action_accuracy": _rate(primary, len(primary_records)),
        "field_accuracy": _rate(correct_fields, fields),
        "field_vector_accuracy": _rate(vectors, n),
    }


def summarize(rows, prediction_rows):
    """Return rendered, slice, endpoint, and independent-cluster metrics."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("exact-policy summary needs nonempty rows")
    row_ids = [row.get("id") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("exact-policy row ids must be unique")
    for row in rows:
        _row_contract(row)
    if not isinstance(prediction_rows, list):
        raise ValueError("predictions must be a list")
    prediction_ids = [item.get("id") for item in prediction_rows if isinstance(item, dict)]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction ids must be unique")
    by_id = {item.get("id"): item for item in prediction_rows if isinstance(item, dict)}

    records, invalid = [], 0
    category_counts = Counter()
    for row in rows:
        item = by_id.get(row["id"])
        try:
            if item is None:
                raise ValueError("missing prediction")
            record = classify(row, item.get("probabilities"))
        except ValueError:
            invalid += 1
            category_counts["invalid"] += 1
            continue
        record["row"] = row
        records.append(record)
        category_counts[record["error_category"]] += 1

    rendered = _summary(records)
    # Invalid or missing predictions receive zero credit in every headline
    # metric.  Keeping a separate valid-only denominator would make a partial
    # run look artificially strong.
    correct_outcomes = category_counts["correct"]
    correct_sets = sum(record["predicted_outcome"].actions
                       == record["gold_outcome"].actions for record in records)
    correct_vectors = sum(record["predicted_trace"] == record["gold_trace"]
                          for record in records)
    expected_fields = sum(len(row["meta"]["action_trace"]) for row in rows)
    correct_fields = sum(
        sum(record["predicted_trace"][field] == value
            for field, value in record["gold_trace"].items())
        for record in records
    )
    gold_primary_eligible = sum(
        canonicalize_outcome(row["meta"]["gold_outcome"]).primary is not None for row in rows
    )
    correct_primary = sum(
        record["gold_outcome"].primary is not None
        and record["predicted_outcome"].primary == record["gold_outcome"].primary
        for record in records
    )
    rendered.update({
        "expected_n": len(rows),
        "n": len(rows),
        "valid_predictions": len(records),
        "invalid": invalid,
        "coverage": len(records) / len(rows),
        "exact_outcome_accuracy": _rate(correct_outcomes, len(rows)),
        "exact_action_set_accuracy": _rate(correct_sets, len(rows)),
        "primary_action_eligible": gold_primary_eligible,
        "primary_action_accuracy": _rate(correct_primary, gold_primary_eligible),
        "field_accuracy": _rate(correct_fields, expected_fields),
        "field_vector_accuracy": _rate(correct_vectors, len(rows)),
        "error_categories": {name: category_counts[name] for name in
                             ("correct", "ranking", "threshold", "composition", "invalid")},
    })
    slices = {}
    for key in SLICE_KEYS:
        grouped = defaultdict(list)
        for record in records:
            value = record["row"]["meta"].get(key)
            if value is not None:
                grouped[str(value)].append(record)
        if grouped:
            slices[key] = {value: _summary(items) for value, items in sorted(grouped.items())}

    endpoints = defaultdict(list)
    clusters = defaultdict(list)
    for record in records:
        meta = record["row"]["meta"]
        cluster = meta.get("semantic_group_id")
        side = meta.get("counterfactual_side")
        if cluster is not None:
            clusters[str(cluster)].append(record)
            if side is not None:
                endpoints[(str(cluster), str(side))].append(record)
    endpoint_exact = sum(all(item["predicted_outcome"] == item["gold_outcome"]
                             for item in items) for items in endpoints.values())
    cluster_exact = sum(all(item["predicted_outcome"] == item["gold_outcome"]
                            for item in items) for items in clusters.values())
    return {
        "rendered": rendered,
        "semantic_endpoints": {
            "n": len(endpoints), "all_views_exact_accuracy": _rate(endpoint_exact, len(endpoints))
        },
        "independent_clusters": {
            "n": len(clusters), "all_views_exact_accuracy": _rate(cluster_exact, len(clusters))
        },
        "slices": slices,
    }


def score_rows_scalar(model, tokenizer, groups, rows, device, max_length, observations=None):
    """Run one canonical model forward for every field question."""
    from scripts.singleton_option_scorer import singleton_option_probabilities
    from scripts.train_gemma_decision_lora import encode, flatten_row

    predictions = []
    for row in rows:
        _, _, mapping, _ = _row_contract(row)
        views = flatten_row(row)
        by_qid = {view["id"].removeprefix(f"{row['id']}/"): view for view in views}
        if set(by_qid) != set(mapping) or any(view.get("image") for view in views):
            raise ValueError(f"{row['id']}: scalar scorer needs every text field exactly once")
        probabilities = {}
        for qid in mapping:
            ids, _, labels = encode(
                by_qid[qid], tokenizer, max_length, shuffle=False, include_labels=True
            )
            values = singleton_option_probabilities(
                model, ids.unsqueeze(0).to(device), groups, labels, observations
            )
            probabilities[qid] = values
        result = classify(row, probabilities)
        predictions.append({
            "id": row["id"],
            "probabilities": probabilities,
            "predicted_outcome": result["predicted_outcome"].as_dict(),
            "gold_outcome": result["gold_outcome"].as_dict(),
            "predicted_trace": result["predicted_trace"],
            "error_category": result["error_category"],
        })
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from scripts.singleton_option_scorer import ForwardObservations
    from scripts.train_gemma_decision_lora import letter_token_groups

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for row in rows:
        _row_contract(row)
    out = Path(args.out)
    if out.exists():
        raise ValueError("refusing to overwrite an exact-policy score directory")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.config.use_cache = False
    device = torch.device("cuda")
    model.to(device).eval()
    observations = ForwardObservations(require_sdpa=True)
    predictions = score_rows_scalar(
        model, tokenizer, groups, rows, device, args.max_length, observations
    )
    report = summarize(rows, predictions)
    report["execution"] = {
        "mode": "singleton",
        "batch_histogram": observations.report(),
        "sdpa_observed": observations.sdpa_observed,
        "max_length": args.max_length,
    }
    out.mkdir(parents=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as target:
        for item in predictions:
            target.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
