"""Compare two frozen R045 prediction files on independent scenarios."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.eval_r045_transfer import audit


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _successes(rows, predictions):
    by_id = {item["id"]: item for item in predictions}
    groups = {"scenario": defaultdict(list), "source": defaultdict(list),
              "pair": defaultdict(list), "language": defaultdict(list),
              "composition_scenario": defaultdict(lambda: defaultdict(list))}
    for row in rows:
        for view in row["paired_views"]:
            meta = view["meta"]
            gold = view["targets"]["decision"]["label"]
            values = by_id[view["id"]]["probabilities"]
            correct = ("yes" if float(values["yes"]) >= float(values["no"]) else "no") == gold
            groups["scenario"][meta["scenario_id"]].append(correct)
            groups["source"][meta["source_id"]].append(correct)
            groups["pair"][row["id"]].append(correct)
            groups["language"][meta["language"]].append(correct)
            composition = "+".join(meta["composition"])
            groups["composition_scenario"][composition][meta["scenario_id"]].append(correct)
    return {
        "scenario": {key: all(values) for key, values in groups["scenario"].items()},
        "source": {key: all(values) for key, values in groups["source"].items()},
        "pair": {key: all(values) for key, values in groups["pair"].items()},
        "language": {key: sum(values) for key, values in groups["language"].items()},
        "composition_scenario": {composition: {key: all(values) for key, values in scenarios.items()}
                                 for composition, scenarios in groups["composition_scenario"].items()},
    }


def _one_sided_sign(gains, losses):
    discordant = gains + losses
    if not discordant:
        return 1.0
    return sum(math.comb(discordant, value) for value in range(gains, discordant + 1)) / 2 ** discordant


def compare(rows, left, right, left_name="left", right_name="right"):
    left_report, right_report = audit(rows, left), audit(rows, right)
    left_success, right_success = _successes(rows, left), _successes(rows, right)
    scenario_ids = set(left_success["scenario"])
    if scenario_ids != set(right_success["scenario"]):
        raise ValueError("scenario coverage differs")
    gains = sum(not left_success["scenario"][key] and right_success["scenario"][key]
                for key in scenario_ids)
    losses = sum(left_success["scenario"][key] and not right_success["scenario"][key]
                 for key in scenario_ids)

    def complete_delta(kind):
        return sum(right_success[kind].values()) - sum(left_success[kind].values())

    compositions = {}
    for composition in sorted(left_success["composition_scenario"]):
        left_values = left_success["composition_scenario"][composition]
        right_values = right_success["composition_scenario"][composition]
        if set(left_values) != set(right_values):
            raise ValueError(f"composition coverage differs: {composition}")
        compositions[composition] = {
            left_name: sum(left_values.values()), right_name: sum(right_values.values()),
            "delta": sum(right_values.values()) - sum(left_values.values()),
            "scenarios": len(left_values),
        }
    return {
        "left": {"name": left_name, "report": left_report},
        "right": {"name": right_name, "report": right_report},
        "paired_scenarios": {"gains": gains, "losses": losses,
                             "ties": len(scenario_ids) - gains - losses,
                             "one_sided_p": _one_sided_sign(gains, losses)},
        "deltas": {"successful_scenarios": complete_delta("scenario"),
                   "complete_sources": complete_delta("source"),
                   "correct_pairs": complete_delta("pair"),
                   "correct_presentations": round(
                       (right_report["accuracy"] - left_report["accuracy"])
                       * right_report["presentations"]),
                   "english_correct": right_success["language"]["en"] - left_success["language"]["en"],
                   "hebrew_correct": right_success["language"]["he"] - left_success["language"]["he"]},
        "composition_scenarios": compositions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = compare(_read(args.data), _read(args.left), _read(args.right),
                     args.left_name, args.right_name)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
