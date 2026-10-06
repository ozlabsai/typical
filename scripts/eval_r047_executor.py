"""Evaluate executable clause decisions and their final-decision composition."""

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _probabilities(item, row_id):
    values = item.get("probabilities")
    if not isinstance(values, dict) or set(values) != {"no", "yes"}:
        raise ValueError(f"{row_id}: prediction must contain exactly no/yes probabilities")
    no, yes = (float(values[label]) for label in ("no", "yes"))
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (no, yes)):
        raise ValueError(f"{row_id}: invalid probability")
    if not math.isclose(no + yes, 1, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{row_id}: probabilities must sum to one")
    return no, yes


def execute_clause(clause):
    """Independent executable gold; intentionally does not import the data builder."""
    kind = clause["kind"]
    if kind == "decimal":
        net = sum((Decimal(value) for value in clause["credits"]), Decimal("0"))
        return net - Decimal(clause["fee"]) >= Decimal(clause["threshold"])
    if kind == "fraction":
        return Fraction(clause["numerator"], clause["denominator"]) >= Fraction(*clause["threshold"])
    if kind == "datetime":
        local = datetime.fromisoformat(clause["local_time"]).replace(
            tzinfo=ZoneInfo(clause["timezone"]))
        deadline = datetime.fromisoformat(clause["deadline"]).replace(tzinfo=timezone.utc)
        return local.astimezone(timezone.utc) <= deadline
    if kind == "ledger":
        balance = sum((Decimal(event["delta"]) for event in clause["events"]), Decimal("0"))
        return balance >= Decimal(clause["threshold"])
    if kind == "conditional":
        target = next(group for group in clause["groups"]
                      if group["name"] == clause["condition_group"])
        probability = Fraction(target["success"], target["success"] + target["failure"])
        return probability >= Fraction(*clause["threshold"])
    if kind == "policy":
        applicable = [rule for rule in clause["rules"] if rule["active"] and
                      rule["scope"] in {"all", clause["request_scope"]}]
        winner = max(applicable, key=lambda rule: (rule["priority"], rule["effect"] == "deny"))
        return winner["effect"] == "allow"
    raise ValueError(f"unknown clause kind: {kind}")


def execute_program(program, clauses):
    by_kind = {clause["kind"]: clause for clause in clauses}
    if set(by_kind) != set(program.get("composition", [])):
        raise ValueError("program and evidence composition differ")
    return all(execute_clause(by_kind[kind]) for kind in program["composition"])


def clause_view(view, index):
    """Project one executable clause into the unchanged decision-serving schema."""
    meta = view.get("meta") or {}
    clauses = meta.get("clauses")
    if not isinstance(clauses, list) or not 0 <= index < len(clauses):
        raise ValueError(f"{view.get('id')}: missing executable clause {index}")
    clause = clauses[index]
    language = meta.get("language")
    if language not in {"en", "he"}:
        raise ValueError(f"{view.get('id')}: unsupported language")
    question = view.get("questions", {}).get("decision")
    if not isinstance(question, dict) or not isinstance(question.get("criteria"), dict):
        raise ValueError(f"{view.get('id')}: malformed decision question")
    instructions = (f"Evaluate only numbered condition {index + 1}. Does it satisfy its stated requirement?"
                    if language == "en" else
                    f"יש להעריך רק את תנאי מספר {index + 1}. האם הוא עומד בדרישה המוצהרת שלו?")
    result = copy.deepcopy(view)
    result["id"] = f"{view['id']}::clause-{index + 1}"
    result["questions"] = {"decision": {"type": "choice", "instructions": instructions,
                                          "criteria": copy.deepcopy(question["criteria"])}}
    result["targets"] = {"decision": {"label": "yes" if execute_clause(clause) else "no"}}
    result["meta"].update({"r047_parent_view_id": view["id"],
                           "r047_clause_index": index,
                           "r047_clause_kind": clause["kind"]})
    return result


def _score(item, gold, row_id):
    no, yes = _probabilities(item, row_id)
    predicted = "yes" if yes >= no else "no"
    probability = yes if gold == "yes" else no
    return {"gold": gold, "predicted": predicted, "correct": predicted == gold,
            "yes_logit": math.log(max(yes, 1e-12) / max(no, 1e-12)),
            "nll": -math.log(max(probability, 1e-12)),
            "brier": (yes - float(gold == "yes")) ** 2}


def _rate(values):
    return sum(values) / len(values) if values else None


def _balanced_accuracy(items):
    recalls = []
    for label in ("no", "yes"):
        group = [item for item in items if item["gold"] == label]
        if group:
            recalls.append(_rate([item["correct"] for item in group]))
    return _rate(recalls)


def _clause_summary(items):
    return {"n": len(items), "accuracy": _rate([item["correct"] for item in items]),
            "balanced_accuracy": _balanced_accuracy(items),
            "no_n": sum(item["gold"] == "no" for item in items),
            "yes_n": sum(item["gold"] == "yes" for item in items)}


def _final_summary(items):
    return {"presentations": len(items),
            "accuracy": _rate([item["correct"] for item in items]),
            "balanced_accuracy": _balanced_accuracy(items),
            "nll": _rate([item["nll"] for item in items]),
            "brier": _rate([item["brier"] for item in items])}


def audit(rows, predictions, min_clauses=1):
    """Fail closed and summarize final, per-clause, vector, and coherence metrics."""
    selected = [row for row in rows if len((row.get("meta") or {}).get("composition", []))
                >= min_clauses]
    if not selected:
        raise ValueError("no rows satisfy min_clauses")
    expected = []
    for row in selected:
        views = row.get("paired_views")
        if row.get("pair_mode") != "counterfactual" or not isinstance(views, list) or len(views) != 2:
            raise ValueError(f"{row.get('id')}: expected one two-view counterfactual pair")
        for view in views:
            clauses = (view.get("meta") or {}).get("clauses", [])
            expected.append(view["id"])
            expected.extend(clause_view(view, index)["id"] for index in range(len(clauses)))
    prediction_ids = [item.get("id") for item in predictions]
    if None in prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction IDs must be present and unique")
    if set(prediction_ids) != set(expected) or len(prediction_ids) != len(expected):
        raise ValueError("predictions must cover every final and clause view exactly once")
    by_id = {item["id"]: item for item in predictions}

    finals, clauses, pairs, vectors, coherences = [], [], [], [], []
    for row in selected:
        pair = []
        for view in row["paired_views"]:
            meta = view["meta"]
            final_value = execute_program(meta["program"], meta["clauses"])
            final_gold = view["targets"]["decision"]["label"]
            if final_gold != ("yes" if final_value else "no") or final_value is not meta.get("gold"):
                raise ValueError(f"{view['id']}: final target disagrees with executable gold")
            final = _score(by_id[view["id"]], final_gold, view["id"])
            final.update({"id": view["id"], "language": meta["language"],
                          "decisive_kind": meta["decisive_kind"]})
            finals.append(final)
            pair.append(final)
            view_clauses = []
            for index, clause in enumerate(meta["clauses"]):
                projected = clause_view(view, index)
                clause_gold = projected["targets"]["decision"]["label"]
                item = _score(by_id[projected["id"]], clause_gold, projected["id"])
                item.update({"id": projected["id"], "parent_id": view["id"],
                             "language": meta["language"], "kind": clause["kind"]})
                clauses.append(item)
                view_clauses.append(item)
            vectors.append(all(item["correct"] for item in view_clauses))
            composed = "yes" if all(item["predicted"] == "yes" for item in view_clauses) else "no"
            coherences.append(final["predicted"] == composed)
        if {item["gold"] for item in pair} != {"no", "yes"}:
            raise ValueError(f"{row['id']}: final counterfactual labels must flip")
        positive = next(item for item in pair if item["gold"] == "yes")
        negative = next(item for item in pair if item["gold"] == "no")
        pairs.append({"direction": positive["yes_logit"] > negative["yes_logit"],
                      "both_correct": all(item["correct"] for item in pair),
                      "language": pair[0]["language"],
                      "decisive_kind": pair[0]["decisive_kind"]})

    clause_groups = {}
    for key in ("kind", "language"):
        groups = defaultdict(list)
        for item in clauses:
            groups[item[key]].append(item)
        clause_groups[key] = {name: _clause_summary(items) for name, items in sorted(groups.items())}
    pair_groups = {}
    for key in ("decisive_kind", "language"):
        groups = defaultdict(list)
        for item in pairs:
            groups[item[key]].append(item)
        pair_groups[key] = {name: {"pairs": len(items),
                                  "direction_rate": _rate([item["direction"] for item in items]),
                                  "pair_both_correct": _rate([item["both_correct"] for item in items])}
                            for name, items in sorted(groups.items())}
    result = {"rows": len(selected), "final": _final_summary(finals),
              "final_pairs": {"pairs": len(pairs),
                              "direction_rate": _rate([item["direction"] for item in pairs]),
                              "pair_both_correct": _rate([item["both_correct"] for item in pairs]),
                              **pair_groups},
              "clauses": {**_clause_summary(clauses), **clause_groups,
                          "vectors": len(vectors), "vector_accuracy": _rate(vectors)},
              "coherence": {"n": len(coherences), "accuracy": _rate(coherences)}}
    return result


def m0_gate(report, required_kinds, threshold=0.98):
    """Machine-check the preregistered can-fit threshold without holdout selection."""
    failures = []
    clause = report["clauses"]
    pairs = report["final_pairs"]
    if clause["balanced_accuracy"] < threshold:
        failures.append("overall clause balanced accuracy")
    if pairs["direction_rate"] < threshold:
        failures.append("overall final direction")
    if report["final"]["balanced_accuracy"] < threshold:
        failures.append("overall final balanced accuracy")
    if pairs["pair_both_correct"] < threshold:
        failures.append("overall final pair-both correctness")
    for kind in required_kinds:
        if kind not in clause["kind"] or clause["kind"][kind]["balanced_accuracy"] < threshold:
            failures.append(f"{kind} clause balanced accuracy")
        if (kind not in pairs["decisive_kind"]
                or pairs["decisive_kind"][kind]["direction_rate"] < threshold):
            failures.append(f"{kind} final direction")
    return {"passed": not failures, "threshold": threshold,
            "required_kinds": list(required_kinds), "failures": failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--min-clauses", type=int, default=1)
    parser.add_argument("--m0-threshold", type=float, default=0.98)
    parser.add_argument("--required-kind", action="append", default=[])
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from scripts.score_action_panel import score_group
    from scripts.train_gemma_decision_lora import letter_token_groups

    rows = _read(args.data)
    selected = [row for row in rows if len((row.get("meta") or {}).get("composition", []))
                >= args.min_clauses]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to("cuda").eval()
    predictions = []
    with torch.no_grad():
        for row in selected:
            for view in row["paired_views"]:
                to_score = [view] + [clause_view(view, index)
                                     for index in range(len(view["meta"]["clauses"]))]
                for projected in to_score:
                    predictions.append({"id": projected["id"], "probabilities":
                                        score_group(model, tokenizer, groups, projected,
                                                    torch.device("cuda"), args.max_length)["q00"]})
    report = audit(rows, predictions, args.min_clauses)
    if args.required_kind:
        report["m0_gate"] = m0_gate(report, args.required_kind, args.m0_threshold)
    report.update({"input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                   "model_revision": args.model_revision})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions.jsonl").write_text("".join(
        json.dumps(item, sort_keys=True) + "\n" for item in predictions))
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
