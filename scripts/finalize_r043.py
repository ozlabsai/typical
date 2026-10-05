"""Fail-closed terminal gate for the R043 causal pair pilot."""

import argparse
import json
import math
from pathlib import Path


ARMS = ("base", "a", "b")
SUITES = ("wf", "wh", "replay")
ACTION_MODES = ("raw", "calibrated")


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _accuracy(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite accuracy in [0, 1]")
    return float(value)


def _retention(report, name):
    report = _mapping(report, name)
    result = {}
    for suite in SUITES:
        values = _mapping(report.get(suite), f"{name}.{suite}")
        if values.get("n") != 128 or values.get("skipped_overlength") != 0:
            raise ValueError(f"{name}.{suite} must score all 128 rows without skipping")
        result[suite] = _accuracy(values.get("accuracy"), f"{name}.{suite}.accuracy")
    return result


def _action(report, name):
    report = _mapping(report, name)
    result = {}
    for mode in ACTION_MODES:
        values = _mapping(report.get(mode), f"{name}.{mode}")
        if values.get("n") != 256:
            raise ValueError(f"{name}.{mode}.n must be 256")
        invalid = _mapping(values.get("error_categories"), f"{name}.{mode}.error_categories").get("invalid")
        if invalid != 0:
            raise ValueError(f"{name}.{mode} contains invalid predictions")
        result[mode] = _accuracy(values.get("exact_action_accuracy"),
                                 f"{name}.{mode}.exact_action_accuracy")
    return result


def _mmhe(report):
    report = _mapping(report, "b_mmhe")
    for arm in ("base", "candidate"):
        values = _mapping(report.get(arm), f"b_mmhe.{arm}")
        for suite in ("he", "en", "real", "shuffled", "noimage", "mismatch", "gold_dropped"):
            if _mapping(values.get(suite), f"b_mmhe.{arm}.{suite}").get("n") != 128:
                raise ValueError(f"b_mmhe.{arm}.{suite}.n must be 128")
    comparison = _mapping(report.get("comparison"), "b_mmhe.comparison")
    gates = _mapping(comparison.get("gates"), "b_mmhe.comparison.gates")
    if not gates or not all(type(value) is bool for value in gates.values()):
        raise ValueError("b_mmhe comparison gates must be nonempty booleans")
    promotion = _bool(comparison.get("promotion_pass"), "b_mmhe.comparison.promotion_pass")
    if promotion != all(gates.values()):
        raise ValueError("b_mmhe promotion result disagrees with its gates")
    return promotion


def _gate_report(report, name):
    report = _mapping(report, name)
    gates = _mapping(report.get("gates"), f"{name}.gates")
    promotion = _bool(report.get("promotion_pass"), f"{name}.promotion_pass")
    if not gates or not all(type(value) is bool for value in gates.values()):
        raise ValueError(f"{name}.gates must be nonempty booleans")
    if promotion != all(gates.values()):
        raise ValueError(f"{name} promotion result disagrees with its gates")
    return promotion


def finalize(comparison, training_audit, assets, retentions, actions, b_mmhe, *, model_revision):
    if not model_revision:
        raise ValueError("model revision must not be empty")
    comparison = _mapping(comparison, "comparison")
    if comparison.get("model_revision") != model_revision:
        raise ValueError("comparison model revision mismatch")
    comparison_gates = _mapping(comparison.get("gates"), "comparison.gates")
    if not comparison_gates or not all(type(value) is bool for value in comparison_gates.values()):
        raise ValueError("comparison gates must be nonempty booleans")
    comparison_pass = _bool(comparison.get("promotion_pass"), "comparison.promotion_pass")
    if comparison_pass != all(comparison_gates.values()):
        raise ValueError("comparison promotion result disagrees with its gates")
    if set(retentions) != set(ARMS) or set(actions) != set(ARMS):
        raise ValueError("retention and action reports require exactly base/A/B")
    retention = {arm: _retention(retentions[arm], f"{arm}_retention") for arm in ARMS}
    action = {arm: _action(actions[arm], f"{arm}_action") for arm in ARMS}
    assets = _mapping(assets, "assets")
    if set(_mapping(assets.get("assets"), "assets.assets")) != {
            "model", "val-wf", "val-wh", "val-replay", "action-cal", "action-eval",
            "he-left", "he-right", "mm"}:
        raise ValueError("assets report does not contain the exact frozen asset set")

    gates = {"causal_pair_promotion_pass": comparison_pass,
             "training_audit_pass": _gate_report(training_audit, "training_audit"),
             "assets_preflight_pass": _bool(assets.get("promotion_pass"), "assets.promotion_pass"),
             "b_mmhe_promotion_pass": _mmhe(b_mmhe)}
    for suite in SUITES:
        for control in ("base", "a"):
            gates[f"b_{suite}_loses_at_most_.02_vs_{control}"] = (
                retention["b"][suite] >= retention[control][suite] - .02)
    for mode in ACTION_MODES:
        for control in ("base", "a"):
            gates[f"b_{mode}_action_loses_at_most_2_of_256_vs_{control}"] = (
                action["b"][mode] >= action[control][mode] - 2 / 256)
    return {
        "model_revision": model_revision,
        "retention_accuracy": retention,
        "action_accuracy": action,
        "gates": gates,
        "promotion_pass": all(gates.values()),
    }


def _load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--model-revision", required=True)
    for arm in ARMS:
        parser.add_argument(f"--{arm}-retention", required=True)
        parser.add_argument(f"--{arm}-action", required=True)
    parser.add_argument("--b-mmhe", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = finalize(
        _load(args.comparison),
        _load(args.training_audit),
        _load(args.assets),
        {arm: _load(getattr(args, f"{arm}_retention")) for arm in ARMS},
        {arm: _load(getattr(args, f"{arm}_action")) for arm in ARMS},
        _load(args.b_mmhe), model_revision=args.model_revision)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
