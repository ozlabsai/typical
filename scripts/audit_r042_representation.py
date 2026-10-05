"""Compare R042 record-card and prose arms on both frozen eval renderers."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


METRICS = ("accuracy", "twins", "nll", "brier")
REPORT_METRICS = {
    "accuracy": "accuracy",
    "twins": "counterfactual_twins_both_correct",
    "nll": "nll",
    "brier": "brier",
    "order_delta": "mean_semantic_order_delta",
}


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _report(path):
    return json.loads(Path(path).read_text())


def _meta(row):
    try:
        return row["meta"]
    except KeyError as exc:
        raise ValueError(f"{row.get('id', '<unknown>')}: row lacks meta") from exc


def _gold(row):
    try:
        return row["targets"]["decision"]["label"]
    except KeyError as exc:
        raise ValueError(f"{row.get('id', '<unknown>')}: row lacks decision label") from exc


def _aligned(rows, predictions):
    row_ids = [row.get("id") for row in rows]
    prediction_ids = [item.get("id") for item in predictions]
    if None in row_ids or len(row_ids) != len(set(row_ids)):
        raise ValueError("evaluation row IDs must be unique")
    if None in prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction IDs must be unique")
    if set(row_ids) != set(prediction_ids):
        raise ValueError("row and prediction IDs differ")
    by_id = {item["id"]: item for item in predictions}
    result = []
    for row in rows:
        try:
            p_yes = float(by_id[row["id"]]["probabilities"]["yes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{row['id']}: prediction lacks numeric yes probability") from exc
        if not 0 <= p_yes <= 1 or not math.isfinite(p_yes):
            raise ValueError(f"{row['id']}: invalid yes probability")
        meta, gold = _meta(row), _gold(row)
        required = ("source_id", "world", "polarity", "wording", "criteria_style", "yes_first")
        if any(key not in meta for key in required) or gold not in {"yes", "no"}:
            raise ValueError(f"{row['id']}: incomplete R042 evaluation metadata")
        gold_yes = gold == "yes"
        result.append({"id": row["id"], "source_id": meta["source_id"], "world": meta["world"],
                       "polarity": meta["polarity"], "wording": meta["wording"],
                       "criteria_style": meta["criteria_style"], "yes_first": meta["yes_first"],
                       "p_yes": p_yes, "correct": (p_yes >= .5) == gold_yes,
                       "nll": -math.log(max(p_yes if gold_yes else 1 - p_yes, 1e-12)),
                       "brier": (p_yes - float(gold_yes)) ** 2})
    return result


def _source_metrics(rows):
    groups = {}
    order_groups = {}
    target_groups = {}
    for row in rows:
        source = groups.setdefault(row["source_id"], {"correct": [], "nll": [], "brier": [], "twins": {}})
        source["correct"].append(row["correct"])
        source["nll"].append(row["nll"])
        source["brier"].append(row["brier"])
        twin_key = (row["polarity"], row["wording"], row["criteria_style"], row["yes_first"])
        source["twins"].setdefault(twin_key, []).append(row["correct"])
        order_key = (row["source_id"], row["polarity"], row["wording"], row["criteria_style"], row["world"])
        order_groups.setdefault(order_key, []).append(row)
        # R037's locked failure is the false-world negative answer to a positive bare claim
        # with generic No/Yes options (e.g. "approval is recorded" when it is not).
        if (row["wording"] == "bare" and row["criteria_style"] == "generic"
                and row["polarity"] == "positive" and row["world"] in (False, 0)):
            target_groups.setdefault(order_key, []).append(row)

    if any(len(values) != 2 for source in groups.values() for values in source["twins"].values()):
        raise ValueError("incomplete counterfactual twins")
    if any(len(values) != 2 or {row["yes_first"] for row in values} != {False, True}
           for values in order_groups.values()):
        raise ValueError("incomplete option-order pairs")
    if len(target_groups) != 64 or any(
            len(values) != 2 or {row["yes_first"] for row in values} != {False, True}
            for values in target_groups.values()):
        raise ValueError("expected 64 complete frozen-defect option-order pairs")

    metrics = {}
    for source_id, values in groups.items():
        metrics[source_id] = {
            "accuracy": sum(values["correct"]) / len(values["correct"]),
            "twins": sum(all(pair) for pair in values["twins"].values()) / len(values["twins"]),
            "nll": sum(values["nll"]) / len(values["nll"]),
            "brier": sum(values["brier"]) / len(values["brier"]),
        }
    return metrics, {key: float(all(row["correct"] for row in values)) for key, values in target_groups.items()}


def _summary(rows):
    sources, _ = _source_metrics(rows)
    order_groups = {}
    for row in rows:
        key = (row["source_id"], row["polarity"], row["wording"], row["criteria_style"], row["world"])
        order_groups.setdefault(key, []).append(row)
    return {
        "n": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "twins": sum(item["twins"] for item in sources.values()) / len(sources),
        "nll": sum(row["nll"] for row in rows) / len(rows),
        "brier": sum(row["brier"] for row in rows) / len(rows),
        "order_delta": sum(abs((pair[0]["p_yes"] if pair[0]["polarity"] == "positive" else 1 - pair[0]["p_yes"]) -
                               (pair[1]["p_yes"] if pair[1]["polarity"] == "positive" else 1 - pair[1]["p_yes"]))
                           for pair in order_groups.values()) / len(order_groups),
    }


def _validate_report(report, summary):
    if report.get("n") != summary["n"]:
        raise ValueError("report row count does not match predictions")
    for metric, report_key in REPORT_METRICS.items():
        if report_key not in report or not math.isclose(report[report_key], summary[metric], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"report {report_key} does not match predictions")


def _bootstrap(left, right, *, samples, seed):
    if set(left) != set(right):
        raise ValueError("treatment/control source IDs differ")
    source_ids = sorted(left)
    result = {}
    rng = random.Random(seed)
    for metric in METRICS:
        values = [left[source][metric] - right[source][metric] for source in source_ids]
        draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
        result[f"{metric}_delta"] = sum(values) / len(values)
        result[f"{metric}_delta_lower_95"] = draws[int(.025 * samples)]
    result["sources"] = len(source_ids)
    result["nll_ratio"] = sum(left[source]["nll"] for source in source_ids) / sum(right[source]["nll"] for source in source_ids)
    result["brier_ratio"] = sum(left[source]["brier"] for source in source_ids) / sum(right[source]["brier"] for source in source_ids)
    return result


def _target_delta(treatment, control, *, samples, seed):
    if set(treatment) != set(control):
        raise ValueError("treatment/control targeted pair IDs differ")
    treatment_by_source, control_by_source = defaultdict(list), defaultdict(list)
    for key in treatment:
        treatment_by_source[key[0]].append(treatment[key])
        control_by_source[key[0]].append(control[key])
    if set(treatment_by_source) != set(control_by_source):
        raise ValueError("treatment/control targeted source IDs differ")
    values = [sum(treatment_by_source[source]) / len(treatment_by_source[source]) -
              sum(control_by_source[source]) / len(control_by_source[source])
              for source in sorted(treatment_by_source)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return {"card_both_orders": sum(treatment.values()) / len(treatment),
            "prose_both_orders": sum(control.values()) / len(control),
            "paired_delta": sum(values) / len(values),
            "paired_delta_lower_95": draws[int(.025 * samples)],
            "sources": len(values)}


def _compare_renderer(card_rows, prose_rows, card_report, prose_report, *, samples, seed):
    card_summary, prose_summary = _summary(card_rows), _summary(prose_rows)
    _validate_report(card_report, card_summary)
    _validate_report(prose_report, prose_summary)
    card_sources, card_target = _source_metrics(card_rows)
    prose_sources, prose_target = _source_metrics(prose_rows)
    delta = _bootstrap(card_sources, prose_sources, samples=samples, seed=seed)
    target = _target_delta(card_target, prose_target, samples=samples, seed=seed)
    gates = {
        "accuracy_delta_at_least_.03": delta["accuracy_delta"] >= .03,
        "accuracy_delta_lower_positive": delta["accuracy_delta_lower_95"] > 0,
        "twins_delta_at_least_.05": delta["twins_delta"] >= .05,
        "twins_delta_lower_positive": delta["twins_delta_lower_95"] > 0,
        "nll_not_over_1pct_worse": delta["nll_ratio"] <= 1.01,
        "brier_not_over_1pct_worse": delta["brier_ratio"] <= 1.01,
        "card_bare_generic_frozen_defect_both_orders_at_least_58_of_64": target["card_both_orders"] >= 58 / 64,
        "card_order_delta_at_most_.05": card_summary["order_delta"] <= .05,
        "targeted_paired_gain_at_least_10pp": target["paired_delta"] >= .10,
        "targeted_paired_gain_lower_positive": target["paired_delta_lower_95"] > 0,
    }
    return {"card": card_summary, "prose": prose_summary, "deltas": delta, "targeted": target,
            "gates": gates}


def audit(card_data, prose_data, card_card_predictions, card_prose_predictions,
          prose_card_predictions, prose_prose_predictions, card_card_report, card_prose_report,
          prose_card_report, prose_prose_report, *, samples=10_000, seed=42):
    card_data, prose_data = _read(card_data), _read(prose_data)
    rendered = {
        "card_renderer": (_aligned(card_data, _read(card_card_predictions)),
                          _aligned(card_data, _read(prose_card_predictions)),
                          _report(card_card_report), _report(prose_card_report)),
        "prose_renderer": (_aligned(prose_data, _read(card_prose_predictions)),
                           _aligned(prose_data, _read(prose_prose_predictions)),
                           _report(card_prose_report), _report(prose_prose_report)),
    }
    card_sources = {row["id"]: row["source_id"] for row in rendered["card_renderer"][0]}
    prose_sources = {row["id"]: row["source_id"] for row in rendered["prose_renderer"][0]}
    if Counter(card_sources.values()) != Counter(prose_sources.values()):
        raise ValueError("card/prose renderer source IDs or source coverage differ")
    result = {name: _compare_renderer(*values, samples=samples, seed=seed)
              for name, values in rendered.items()}
    result["gates"] = {f"{renderer}_{name}": passed
                       for renderer, report in result.items() if renderer != "gates"
                       for name, passed in report["gates"].items()}
    result["promotion_pass"] = all(result["gates"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-data", required=True)
    parser.add_argument("--prose-data", required=True)
    for arm in ("card", "prose"):
        for renderer in ("card", "prose"):
            parser.add_argument(f"--{arm}-{renderer}-predictions", required=True)
            parser.add_argument(f"--{arm}-{renderer}-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    result = audit(args.card_data, args.prose_data, args.card_card_predictions, args.card_prose_predictions,
                   args.prose_card_predictions, args.prose_prose_predictions, args.card_card_report,
                   args.card_prose_report, args.prose_card_report, args.prose_prose_report,
                   samples=args.bootstrap_samples)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
