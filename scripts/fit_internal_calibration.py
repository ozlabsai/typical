"""Fit per-question-type temperature calibration to saved internal JSONL records.

Record: {qtype, target: label or {label: weight}, probabilities: {label: p}}.
Usage: uv run python scripts/fit_internal_calibration.py internal.jsonl [--noul-intercept]
"""
import argparse
import json
import math
from collections import defaultdict


def _target(row, labels):
    target = row["target"]
    if not isinstance(target, dict):
        label = str(target)
        if label not in labels:
            raise ValueError(f"target {target!r} is not in probabilities")
        return {key: float(key == label) for key in labels}
    if not isinstance(target, dict) or set(target) != set(labels):
        raise ValueError("target distribution must use the probability labels")
    values = {label: float(target[label]) for label in labels}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("target weights must be finite and nonnegative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("target distribution must have positive mass")
    return {label: value / total for label, value in values.items()}


def _prepare(rows):
    groups = defaultdict(list)
    for row in rows:
        kind, probs = row.get("qtype"), row.get("probabilities")
        if not isinstance(kind, str) or not kind or not isinstance(probs, dict) or not probs:
            raise ValueError("each record needs qtype and a nonempty probabilities object")
        probs = {str(key): float(value) for key, value in probs.items()}
        if any(not math.isfinite(p) or p < 0 for p in probs.values()) or sum(probs.values()) <= 0:
            raise ValueError("probabilities must be finite, nonnegative, and have positive mass")
        total = sum(probs.values())
        probs = {key: value / total for key, value in probs.items()}
        groups[kind].append((probs, _target(row, probs)))
    return groups


def _scaled(probs, temperature, intercept=0.0):
    logits = {key: math.log(max(value, 1e-300)) / temperature +
              (intercept if key.lower() in {"true", "yes"} else 0.0)
              for key, value in probs.items()}
    peak = max(logits.values())
    weights = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _metrics(rows, temperature, intercept=0.0):
    nll = brier = 0.0
    for probs, target in rows:
        calibrated = _scaled(probs, temperature, intercept)
        nll -= sum(weight * math.log(max(calibrated[key], 1e-300)) for key, weight in target.items())
        brier += sum((calibrated[key] - target[key]) ** 2 for key in target)
    return {"n": len(rows), "nll": nll / len(rows), "brier": brier / len(rows)}


def _golden_minimize(fn, low=math.log(0.05), high=math.log(20.0)):
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left, f_right = fn(math.exp(left)), fn(math.exp(right))
    for _ in range(100):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = fn(math.exp(left))
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = fn(math.exp(right))
    candidates = [(fn(1.0), 1.0), (f_left, math.exp(left)), (f_right, math.exp(right))]
    return min(candidates)[1]


def fit(rows, noul_intercept=False):
    report = {}
    for kind, items in sorted(_prepare(rows).items()):
        baseline = _metrics(items, 1.0)
        binary = len(items[0][0]) == 2 and any(
            key.lower() in {"true", "yes"} for key in items[0][0])
        grid = ([i / 20 for i in range(-40, 41)]
                if noul_intercept and kind == "noul" and binary else [0.0])
        choices = []
        for intercept in grid:
            temperature = _golden_minimize(lambda t: _metrics(items, t, intercept)["nll"])
            choices.append((_metrics(items, temperature, intercept)["nll"], temperature, intercept))
        _, temperature, intercept = min(choices)
        report[kind] = {"temperature": temperature, "logit_intercept_true": intercept,
                        "before": baseline, "after": _metrics(items, temperature, intercept)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--noul-intercept", action="store_true",
                        help="also grid-search a true-class logit intercept from -2 to 2")
    args = parser.parse_args()
    with open(args.records) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    print(json.dumps(fit(rows, args.noul_intercept), indent=2))


if __name__ == "__main__":
    main()
