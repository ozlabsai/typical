"""Summarize fit and transfer across the R046 counterfactual ladder."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _rate(values):
    return sum(values) / len(values) if values else None


def summarize(rows, predictions):
    by_id = {item["id"]: item["probabilities"] for item in predictions}
    expected = {view["id"] for row in rows for view in row["paired_views"]}
    if set(by_id) != expected or len(by_id) != len(expected):
        raise ValueError("predictions must cover every presentation exactly once")

    presentations, pairs = [], []
    for row in rows:
        views = row["paired_views"]
        if len(views) != 2 or {v["targets"]["decision"]["label"] for v in views} != {"no", "yes"}:
            raise ValueError(f"{row['id']}: expected one no/yes counterfactual pair")
        scored = []
        for view in views:
            probabilities = by_id[view["id"]]
            no, yes = float(probabilities["no"]), float(probabilities["yes"])
            if not math.isclose(no + yes, 1.0, abs_tol=1e-6):
                raise ValueError(f"{view['id']}: probabilities must sum to one")
            gold = view["targets"]["decision"]["label"]
            gold_probability = probabilities[gold]
            item = {
                "correct": ("yes" if yes >= no else "no") == gold,
                "nll": -math.log(max(float(gold_probability), 1e-12)),
                "brier": (yes - float(gold == "yes")) ** 2,
                "yes_logit": math.log(max(yes, 1e-12) / max(no, 1e-12)),
                "gold": gold,
            }
            presentations.append(item)
            scored.append(item)
        positive = next(item for item in scored if item["gold"] == "yes")
        negative = next(item for item in scored if item["gold"] == "no")
        delta = positive["yes_logit"] - negative["yes_logit"]
        meta = views[0]["meta"]
        pairs.append({
            "both_correct": all(item["correct"] for item in scored),
            "delta": delta,
            "direction": delta > 0,
            "margin": delta >= 2.0,
            "hinge": max(0.0, 2.0 - delta),
            "source_id": meta["source_id"],
            "scenario_id": meta["scenario_id"],
            "language": meta["language"],
            "decisive_kind": meta["decisive_kind"],
            "renderer": meta["renderer"],
        })

    def complete(key):
        groups = defaultdict(list)
        for pair in pairs:
            groups[pair[key]].append(pair)
        return len(groups), _rate([all(item["both_correct"] for item in group)
                                   for group in groups.values()])

    def pair_summary(items):
        return {
            "pairs": len(items),
            "pair_both_correct": _rate([item["both_correct"] for item in items]),
            "direction_rate": _rate([item["direction"] for item in items]),
            "margin_rate": _rate([item["margin"] for item in items]),
            "mean_delta": _rate([item["delta"] for item in items]),
            "mean_hinge": _rate([item["hinge"] for item in items]),
        }

    result = pair_summary(pairs)
    result.update({
        "presentations": len(presentations),
        "accuracy": _rate([item["correct"] for item in presentations]),
        "nll": _rate([item["nll"] for item in presentations]),
        "brier": _rate([item["brier"] for item in presentations]),
    })
    result["sources"], result["source_complete"] = complete("source_id")
    result["scenarios"], result["scenario_complete"] = complete("scenario_id")
    for key in ("language", "decisive_kind", "renderer"):
        groups = defaultdict(list)
        for pair in pairs:
            groups[pair[key]].append(pair)
        result[key] = {name: pair_summary(items) for name, items in sorted(groups.items())}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--predictions", action="append", required=True,
                        help="NAME=predictions.jsonl; repeat for each model")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    rungs = {name: _read(data_dir / f"{name}.jsonl")
             for name in ("seen", "unused", "cal", "test")}
    rungs["seen_atomic"] = [row for row in rungs["seen"]
                            if len(row["meta"]["composition"]) == 1]
    rungs["seen_composed"] = [row for row in rungs["seen"]
                              if len(row["meta"]["composition"]) > 1]

    report = {"models": {}}
    for spec in args.predictions:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            parser.error("--predictions must be NAME=PATH")
        predictions = _read(path)
        report["models"][name] = {
            rung: summarize(rows, [item for item in predictions
                                   if item["id"] in {view["id"] for row in rows
                                                      for view in row["paired_views"]}])
            for rung, rows in rungs.items()
        }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
