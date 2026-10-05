"""Paired source-bootstrap comparison for R038 orbit predictions."""

import argparse
import json
import random
from pathlib import Path


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _source_metrics(rows, predictions):
    by_id = {item["id"]: item for item in predictions}
    groups = {}
    for row in rows:
        item = by_id.get(row["id"])
        if item is None:
            raise ValueError(f"missing prediction for {row['id']}")
        correct = (item["probabilities"]["yes"] >= .5) == (row["targets"]["decision"]["label"] == "yes")
        meta = row["meta"]
        source = groups.setdefault(meta["source_id"], {"correct": [], "twins": {}})
        source["correct"].append(correct)
        key = (meta["polarity"], meta["wording"], meta["criteria_style"], meta["yes_first"])
        source["twins"].setdefault(key, []).append(correct)
    return {source_id: {"accuracy": sum(values["correct"]) / len(values["correct"]),
                        "twins": sum(all(pair) for pair in values["twins"].values()) / len(values["twins"])}
            for source_id, values in groups.items()}


def compare(rows, treatment, control, *, samples=10_000, seed=42):
    left, right = _source_metrics(rows, treatment), _source_metrics(rows, control)
    if set(left) != set(right):
        raise ValueError("treatment/control source mismatch")
    source_ids = sorted(left)
    deltas = {name: [left[source][name] - right[source][name] for source in source_ids]
              for name in ("accuracy", "twins")}
    rng = random.Random(seed)
    report = {}
    for name, values in deltas.items():
        bootstrap = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
        report[f"{name}_delta"] = sum(values) / len(values)
        report[f"{name}_delta_lower_95"] = bootstrap[int(.025 * samples)]
    report["sources"] = len(source_ids)
    report["gates"] = {
        "accuracy_delta_at_least_.03": report["accuracy_delta"] >= .03,
        "accuracy_delta_lower_positive": report["accuracy_delta_lower_95"] > 0,
        "twins_delta_at_least_.05": report["twins_delta"] >= .05,
        "twins_delta_lower_positive": report["twins_delta_lower_95"] > 0,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compare(_read(args.data), _read(args.treatment), _read(args.control))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
