"""Apply R040's locked complete-versus-partial renderer-probe gates."""

import argparse
import json
import random
from pathlib import Path


LOCKED_R037_SHA256 = "a477f8347c0d2a6ac4898e7c6e119f69450669b44b2943d1d55aa3089b27a297"
PAIR_CELLS = ("bare_no", "bare_yes", "canonical_no", "canonical_yes")


def _validate_report(report):
    if report.get("input_sha256") != LOCKED_R037_SHA256 or report.get("per_label") != 65:
        raise ValueError("renderer report did not use the locked 65-per-label R037 panel")
    for cell in PAIR_CELLS:
        if report.get(cell, {}).get("n") != 130:
            raise ValueError(f"renderer report has incomplete {cell} predictions")


def _paired(records, renderer, label):
    grouped = {}
    for record in records:
        if record.get("renderer") == renderer and record.get("label") == label:
            grouped.setdefault(record["id"], []).append(record)
    if len(grouped) != 65:
        raise ValueError(f"expected 65 {renderer}/{label} probe cases")
    values = {}
    for item_id, pair in grouped.items():
        if len(pair) != 2 or {item.get("yes_first") for item in pair} != {False, True}:
            raise ValueError(f"{item_id}: probe pair is incomplete")
        values[item_id] = float(all(item.get("correct") is True for item in pair))
    return values


def _paired_delta(treatment, control, *, samples=10_000, seed=42):
    if set(treatment) != set(control):
        raise ValueError("treatment/control probe IDs differ")
    values = [treatment[item_id] - control[item_id] for item_id in sorted(treatment)]
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return sum(values) / len(values), bootstrap[int(.025 * samples)]


def compare(treatment, control, treatment_records, control_records):
    _validate_report(treatment)
    _validate_report(control)
    report = {}
    for cell in PAIR_CELLS:
        renderer, label = cell.split("_", 1)
        delta, lower = _paired_delta(_paired(treatment_records, renderer, label),
                                     _paired(control_records, renderer, label))
        report[f"{cell}_paired_delta"] = delta
        report[f"{cell}_paired_delta_lower_95"] = lower
    for metric in ("bare_mean_order_delta", "canonical_mean_order_delta"):
        report[f"{metric}_delta"] = treatment[metric] - control[metric]
    report["gates"] = {
        "complete_bare_negative_at_least_59_of_65": treatment["bare_no_both_orders"] >= 59 / 65,
        "complete_bare_positive_at_least_62_of_65": treatment["bare_yes_both_orders"] >= 62 / 65,
        "complete_bare_order_delta_at_most_.05": treatment["bare_mean_order_delta"] <= .05,
        "complete_canonical_negative_at_least_62_of_65": treatment["canonical_no_both_orders"] >= 62 / 65,
        "complete_canonical_positive_at_least_62_of_65": treatment["canonical_yes_both_orders"] >= 62 / 65,
        "complete_canonical_order_delta_at_most_.05": treatment["canonical_mean_order_delta"] <= .05,
        "bare_negative_delta_at_least_10pp": report["bare_no_paired_delta"] >= .10,
        "bare_negative_delta_lower_positive": report["bare_no_paired_delta_lower_95"] > 0,
        "bare_positive_not_over_2_of_65_worse": report["bare_yes_paired_delta"] >= -2 / 65,
        "canonical_negative_not_over_2_of_65_worse": report["canonical_no_paired_delta"] >= -2 / 65,
        "canonical_positive_not_over_2_of_65_worse": report["canonical_yes_paired_delta"] >= -2 / 65,
        "bare_order_delta_not_over_.01_worse": report["bare_mean_order_delta_delta"] <= .01,
        "canonical_order_delta_not_over_.01_worse": report["canonical_mean_order_delta_delta"] <= .01,
    }
    return report


def _read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment-predictions", required=True)
    parser.add_argument("--control-predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compare(json.loads(Path(args.treatment).read_text()), json.loads(Path(args.control).read_text()),
                     _read_records(args.treatment_predictions), _read_records(args.control_predictions))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
