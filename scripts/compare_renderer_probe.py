"""Apply R040's locked complete-versus-partial renderer-probe gates."""

import argparse
import json
from pathlib import Path


def compare(treatment, control):
    metrics = (
        "bare_no_both_orders", "bare_yes_both_orders", "bare_mean_order_delta",
        "canonical_no_both_orders", "canonical_yes_both_orders", "canonical_mean_order_delta",
    )
    if any(metric not in treatment or metric not in control for metric in metrics):
        raise ValueError("renderer reports are incomplete")
    report = {f"{metric}_delta": treatment[metric] - control[metric] for metric in metrics}
    report["gates"] = {
        "complete_bare_negative_at_least_59_of_65": treatment["bare_no_both_orders"] >= 59 / 65,
        "complete_bare_positive_at_least_62_of_65": treatment["bare_yes_both_orders"] >= 62 / 65,
        "complete_bare_order_delta_at_most_.05": treatment["bare_mean_order_delta"] <= .05,
        "complete_canonical_negative_at_least_62_of_65": treatment["canonical_no_both_orders"] >= 62 / 65,
        "complete_canonical_positive_at_least_62_of_65": treatment["canonical_yes_both_orders"] >= 62 / 65,
        "complete_canonical_order_delta_at_most_.05": treatment["canonical_mean_order_delta"] <= .05,
        "complete_bare_negative_exceeds_partial": report["bare_no_both_orders_delta"] > 0,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    treatment, control = (json.loads(Path(path).read_text()) for path in (args.treatment, args.control))
    report = compare(treatment, control)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
