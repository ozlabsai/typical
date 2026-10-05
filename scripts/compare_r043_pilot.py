"""Compare R043 base/A/B/C/D with source-paired causal bootstrap contrasts."""

import argparse
import json
import math
import random
from pathlib import Path

from scripts.eval_r043_pairs import audit


METRICS = ("exact_action_accuracy", "twin_both_correct")


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load(path):
    return json.loads(Path(path).read_text())


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_report(expected, report, name, input_sha256, model_revision):
    if report.get("input_sha256") != input_sha256:
        raise ValueError(f"{name}: evaluation input hash mismatch")
    if report.get("model_revision") != model_revision:
        raise ValueError(f"{name}: model revision mismatch")
    for key in ("pairs", "presentations", "accuracy", "exact_action_accuracy",
                "twin_both_correct", "nll", "brier", "option_order_accuracy_delta",
                "polarity_accuracy_delta"):
        if key not in report:
            raise ValueError(f"{name}: missing report metric {key}")
        left, right = expected[key], report[key]
        if isinstance(left, float):
            if not math.isclose(left, _finite(right, f"{name}.{key}"), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{name}: report metric {key} does not match predictions")
        elif left != right:
            raise ValueError(f"{name}: report metric {key} does not match predictions")
    if expected["families"] != report.get("families") or expected["source_metrics"] != report.get("source_metrics"):
        raise ValueError(f"{name}: report diagnostics do not match predictions")


def _bootstrap(arms, coefficients, *, samples, seed):
    if not isinstance(samples, int) or samples < 100:
        raise ValueError("bootstrap samples must be an integer of at least 100")
    source_sets = [set(values) for values in arms.values()]
    if not source_sets or any(values != source_sets[0] for values in source_sets[1:]):
        raise ValueError("arm source IDs differ")
    if set(arms) != set(coefficients):
        raise ValueError("bootstrap arms and coefficients differ")
    source_ids = sorted(source_sets[0])
    result = {"sources": len(source_ids)}
    for offset, metric in enumerate(METRICS):
        values = [sum(coefficients[arm] * arms[arm][source][metric] for arm in arms)
                  for source in source_ids]
        rng = random.Random(seed + offset)
        draws = sorted(sum(rng.choice(values) for _ in values) / len(values)
                       for _ in range(samples))
        result[metric] = {
            "delta": sum(values) / len(values),
            "lower_95": draws[int(.025 * samples)],
            "upper_95": draws[min(samples - 1, int(.975 * samples))],
        }
    return result


def compare(eval_rows, predictions, reports, *, input_sha256, model_revision,
            samples=10_000, seed=43):
    required = {"base", "a", "b", "c", "d"}
    if set(predictions) != required or set(reports) != required:
        raise ValueError("comparison requires exactly base/A/B/C/D")
    scored = {arm: audit(eval_rows, values) for arm, values in predictions.items()}
    for arm in scored:
        _validate_report(scored[arm], reports[arm], arm, input_sha256, model_revision)

    deltas = {
        "b_minus_a": _bootstrap(
            {arm: scored[arm]["source_metrics"] for arm in ("b", "a")},
            {"b": 1, "a": -1}, samples=samples, seed=seed),
        "c_minus_d": _bootstrap(
            {arm: scored[arm]["source_metrics"] for arm in ("c", "d")},
            {"c": 1, "d": -1}, samples=samples, seed=seed + 100),
        "difference_in_differences": _bootstrap(
            {arm: scored[arm]["source_metrics"] for arm in ("b", "a", "c", "d")},
            {"b": 1, "a": -1, "c": -1, "d": 1}, samples=samples, seed=seed + 200),
        "b_minus_c_raw": _bootstrap(
            {arm: scored[arm]["source_metrics"] for arm in ("b", "c")},
            {"b": 1, "c": -1}, samples=samples, seed=seed + 300),
    }
    gates = {}
    for comparison in ("b_minus_a", "difference_in_differences"):
        values = deltas[comparison]
        for metric in METRICS:
            gates[f"{comparison}_{metric}_at_least_.02"] = values[metric]["delta"] >= .02
            gates[f"{comparison}_{metric}_lower_95_positive"] = values[metric]["lower_95"] > 0
    for family, b_values in scored["b"]["families"].items():
        for metric in METRICS:
            gates[f"b_{family}_{metric}_not_over_.02_below_a"] = (
                b_values[metric] >= scored["a"]["families"][family][metric] - .02)
    for control in ("a", "c"):
        for metric in ("nll", "brier"):
            gates[f"b_{metric}_not_over_1pct_worse_than_{control}"] = (
                scored["b"][metric] <= scored[control][metric] * 1.01)
    gates["b_option_order_accuracy_delta_at_most_.01"] = (
        scored["b"]["option_order_accuracy_delta"] <= .01)
    gates["b_polarity_accuracy_delta_at_most_.01"] = (
        scored["b"]["polarity_accuracy_delta"] <= .01)
    for polarity in ("negative", "positive"):
        gates[f"b_{polarity}_accuracy_not_over_.02_below_a"] = (
            scored["b"]["polarity"][polarity]["accuracy"] >=
            scored["a"]["polarity"][polarity]["accuracy"] - .02)
    return {
        "estimand": "source-paired (B-A) - (C-D); B/A true pairs, C/D matched shuffled pairs",
        "model_revision": model_revision,
        "input_sha256": input_sha256,
        "arms": scored,
        "deltas": deltas,
        "gates": gates,
        "promotion_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-revision", required=True)
    for arm in ("base", "a", "b", "c", "d"):
        parser.add_argument(f"--{arm}-predictions", required=True)
        parser.add_argument(f"--{arm}-report", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.model_revision.strip():
        raise ValueError("model revision must not be empty")
    data_path = Path(args.data)
    import hashlib
    input_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    predictions = {arm: _read(getattr(args, f"{arm}_predictions"))
                   for arm in ("base", "a", "b", "c", "d")}
    reports = {arm: _load(getattr(args, f"{arm}_report"))
               for arm in ("base", "a", "b", "c", "d")}
    result = compare(_read(data_path), predictions, reports, input_sha256=input_sha256,
                     model_revision=args.model_revision, samples=args.bootstrap_samples, seed=args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
