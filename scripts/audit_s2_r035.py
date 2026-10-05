"""Apply the preregistered R035 mechanism gates against cached Arm D."""

import argparse
import hashlib
import json
from pathlib import Path


def _read_json(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _only_result(payload, expected_rows):
    if not payload.get("complete") or payload.get("rows") != expected_rows:
        raise ValueError(f"crossed result must be complete with {expected_rows} rows")
    if len(payload.get("results", [])) != 1:
        raise ValueError("crossed result must contain exactly one adapter")
    result = payload["results"][0]
    predictions = result.get("predictions", [])
    if len(predictions) != expected_rows:
        raise ValueError(f"expected {expected_rows} predictions, got {len(predictions)}")
    ids = [row["example_id"] for row in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction IDs must be unique")
    return result


def _accuracy(rows):
    if not rows:
        raise ValueError("metric slice is empty")
    return sum(row["correct"] for row in rows) / len(rows)


def _slice(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def _overall(rows):
    n = len(rows)
    return {"accuracy": _accuracy(rows),
            "nll": sum(row["nll"] for row in rows) / n,
            "brier": sum(row["brier"] for row in rows) / n}


def _interface_middle(rows):
    common = {"polarity": "negative", "context_view": "full", "placement": "middle"}
    arm_a = _slice(rows, question_style="interrogative", criteria_style="explicit", **common)
    arm_b = _slice(rows, question_style="declarative", criteria_style="generic", **common)
    key = lambda row: (row["source_id"], tuple(row["option_order"]))
    if {key(row) for row in arm_a} != {key(row) for row in arm_b}:
        raise ValueError("full/negative/middle A/B interface pairs are not aligned")
    return {"a_accuracy": _accuracy(arm_a), "b_accuracy": _accuracy(arm_b),
            "gap": _accuracy(arm_a) - _accuracy(arm_b), "n_pairs": len(arm_a)}


def _strong_cell_accuracies(rows):
    cells = {}
    for row in rows:
        if row["question_style"] == "interrogative" and row["criteria_style"] == "explicit":
            key = "|".join((row["polarity"], row["context_view"], "_".join(row["option_order"])))
            cells.setdefault(key, []).append(row)
    return {key: _accuracy(cell) for key, cell in sorted(cells.items())}


def _training_summary(path, expected_steps=250):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_step = {row["step"]: row for row in rows if isinstance(row.get("step"), int) and row["step"] > 0}
    if set(by_step) != set(range(1, expected_steps + 1)):
        raise ValueError(f"training metrics must contain every step 1..{expected_steps}")
    tail = [by_step[step] for step in range(expected_steps - 99, expected_steps + 1)]
    active = sum(row.get("pair_consistency_active", 0) for row in tail)
    seen = sum(row.get("pair_consistency_seen", 0) for row in tail)
    if not seen:
        raise ValueError("no paired consistency slots were seen in the final 100 steps")
    final = by_step[expected_steps]
    if "validation" not in final:
        raise ValueError("final training metric lacks validation")
    return {"active": active, "seen": seen, "rate": active / seen,
            "validation": final["validation"]}


def _final_validation(path, expected_steps=250):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row.get("step") == expected_steps and "validation" in row]
    if len(matches) != 1:
        raise ValueError(f"baseline metrics need one validated step {expected_steps}")
    return matches[0]["validation"]


def audit(baseline_crossed, candidate_crossed, baseline_metrics, candidate_metrics,
          expected_rows=2048, expected_steps=250):
    baseline_payload, candidate_payload = _read_json(baseline_crossed), _read_json(candidate_crossed)
    if baseline_payload.get("data_sha256") != candidate_payload.get("data_sha256"):
        raise ValueError("baseline and candidate used different crossed data")
    baseline = _only_result(baseline_payload, expected_rows)
    candidate = _only_result(candidate_payload, expected_rows)
    base_rows, cand_rows = baseline["predictions"], candidate["predictions"]
    if {row["example_id"] for row in base_rows} != {row["example_id"] for row in cand_rows}:
        raise ValueError("baseline and candidate prediction identities differ")

    target = {"question_style": "declarative", "criteria_style": "generic",
              "polarity": "negative"}
    full_no = {**target, "context_view": "full", "gold": "no"}
    base_target, cand_target = _accuracy(_slice(base_rows, **target)), _accuracy(_slice(cand_rows, **target))
    base_full_no, cand_full_no = _accuracy(_slice(base_rows, **full_no)), _accuracy(_slice(cand_rows, **full_no))
    base_interface, cand_interface = _interface_middle(base_rows), _interface_middle(cand_rows)
    base_strong, cand_strong = _strong_cell_accuracies(base_rows), _strong_cell_accuracies(cand_rows)
    if set(base_strong) != set(cand_strong):
        raise ValueError("candidate strong-cell coverage differs from baseline")
    strong_deltas = {key: cand_strong[key] - value for key, value in base_strong.items()}

    base_overall, cand_overall = _overall(base_rows), _overall(cand_rows)
    base_validation = _final_validation(baseline_metrics, expected_steps)
    training = _training_summary(candidate_metrics, expected_steps)
    cand_validation = training["validation"]
    retention_deltas = {name: cand_validation[name]["accuracy"] - base_validation[name]["accuracy"]
                        for name in ("wh", "replay")}
    base_order = baseline["order_robustness"]["mean_abs_aligned_p_yes_delta"]
    cand_order = candidate["order_robustness"]["mean_abs_aligned_p_yes_delta"]

    deltas = {
        "target_accuracy": cand_target - base_target,
        "full_negative_no_recall": cand_full_no - base_full_no,
        "interface_gap_shrink": base_interface["gap"] - cand_interface["gap"],
        "interface_a_accuracy": cand_interface["a_accuracy"] - base_interface["a_accuracy"],
        "interface_b_accuracy": cand_interface["b_accuracy"] - base_interface["b_accuracy"],
        "order_delta": cand_order - base_order,
        "wh_accuracy": retention_deltas["wh"],
        "replay_accuracy": retention_deltas["replay"],
    }
    gates = {
        "active_pair_rate_at_least_0.50": training["rate"] >= 0.50,
        "target_accuracy_gain_at_least_0.05": deltas["target_accuracy"] >= 0.05,
        "full_negative_no_recall_gain_at_least_0.10": deltas["full_negative_no_recall"] >= 0.10,
        "interface_gap_shrink_at_least_0.10": deltas["interface_gap_shrink"] >= 0.10,
        "interface_b_gain_at_least_0.10": deltas["interface_b_accuracy"] >= 0.10,
        "interface_a_drop_at_most_0.02": deltas["interface_a_accuracy"] >= -0.02,
        "each_strong_cell_drop_at_most_0.02": min(strong_deltas.values()) >= -0.02,
        "wh_drop_at_most_0.02": deltas["wh_accuracy"] >= -0.02,
        "replay_drop_at_most_0.02": deltas["replay_accuracy"] >= -0.02,
        "order_delta_does_not_worsen": cand_order <= base_order,
        "overall_nll_increase_at_most_5pct": cand_overall["nll"] <= 1.05 * base_overall["nll"],
        "overall_brier_increase_at_most_5pct": cand_overall["brier"] <= 1.05 * base_overall["brier"],
    }
    return {
        "baseline_crossed_sha256": _sha256(baseline_crossed),
        "candidate_crossed_sha256": _sha256(candidate_crossed),
        "baseline_metrics_sha256": _sha256(baseline_metrics),
        "candidate_metrics_sha256": _sha256(candidate_metrics),
        "crossed_data_sha256": baseline_payload["data_sha256"],
        "active_pair_coverage_final_100_steps": {key: training[key] for key in ("active", "seen", "rate")},
        "baseline": {"target_accuracy": base_target, "full_negative_no_recall": base_full_no,
                     "interface": base_interface, "overall": base_overall, "order_delta": base_order},
        "candidate": {"target_accuracy": cand_target, "full_negative_no_recall": cand_full_no,
                      "interface": cand_interface, "overall": cand_overall, "order_delta": cand_order},
        "deltas": deltas, "strong_cell_deltas": strong_deltas,
        "gates": gates, "mechanism_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-crossed", required=True)
    parser.add_argument("--candidate-crossed", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.baseline_crossed, args.candidate_crossed,
                   args.baseline_metrics, args.candidate_metrics)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
