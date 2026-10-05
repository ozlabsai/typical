"""Fail-closed promotion gate for the R042 representation experiment."""

import argparse
import json
import math
from pathlib import Path


SUITES = ("wf", "wh", "replay")
ACTION_MODES = ("raw", "calibrated")
GENERIC_LOSS = .02
ACTION_LOSS = 2 / 256


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
    return {suite: _accuracy(_mapping(report.get(suite), f"{name}.{suite}").get("accuracy"),
                             f"{name}.{suite}.accuracy") for suite in SUITES}


def _mmhe_pass(report, name):
    comparison = _mapping(_mapping(report, name).get("comparison"), f"{name}.comparison")
    return _bool(comparison.get("promotion_pass"), f"{name}.comparison.promotion_pass")


def _action(report, name):
    report = _mapping(report, name)
    result = {}
    for mode in ACTION_MODES:
        summary = _mapping(report.get(mode), f"{name}.{mode}")
        if summary.get("n") != 256:
            raise ValueError(f"{name}.{mode}.n must be 256")
        result[mode] = _accuracy(summary.get("exact_action_accuracy"),
                                 f"{name}.{mode}.exact_action_accuracy")
    return result


def finalize(representation, renderer, base_retention, card_retention, prose_retention,
             card_mmhe, prose_mmhe, base_action, card_action, prose_action):
    """Return R042's single promotion decision after validating all input schemas."""
    representation = _mapping(representation, "representation")
    renderer = _mapping(renderer, "renderer")
    representation_pass = _bool(representation.get("promotion_pass"), "representation.promotion_pass")
    renderer_gates = _mapping(renderer.get("gates"), "renderer.gates")
    if not renderer_gates:
        raise ValueError("renderer.gates must not be empty")
    renderer_pass = all(_bool(value, f"renderer.gates.{key}") for key, value in renderer_gates.items())

    base = _retention(base_retention, "base_retention")
    card = _retention(card_retention, "card_retention")
    prose = _retention(prose_retention, "prose_retention")
    base_actions = _action(base_action, "base_action")
    card_actions = _action(card_action, "card_action")
    prose_actions = _action(prose_action, "prose_action")

    gates = {
        "representation_promotion_pass": representation_pass,
        "every_renderer_gate_passes": renderer_pass,
        "card_mmhe_promotion_pass": _mmhe_pass(card_mmhe, "card_mmhe"),
        "prose_mmhe_promotion_pass": _mmhe_pass(prose_mmhe, "prose_mmhe"),
    }
    for arm, values in (("card", card), ("prose", prose)):
        for suite in SUITES:
            gates[f"{arm}_{suite}_accuracy_loses_at_most_.02"] = values[suite] >= base[suite] - GENERIC_LOSS
    for mode in ACTION_MODES:
        gates[f"card_{mode}_action_loses_at_most_2_of_256_vs_base"] = (
            card_actions[mode] >= base_actions[mode] - ACTION_LOSS)
        gates[f"card_{mode}_action_not_over_2_of_256_below_prose"] = (
            card_actions[mode] >= prose_actions[mode] - ACTION_LOSS)
    return {"generic_accuracy": {"base": base, "card": card, "prose": prose},
            "action_accuracy": {"base": base_actions, "card": card_actions, "prose": prose_actions},
            "gates": gates, "promotion_pass": all(gates.values())}


def _load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--base-retention", required=True)
    parser.add_argument("--card-retention", required=True)
    parser.add_argument("--prose-retention", required=True)
    parser.add_argument("--card-mmhe", required=True)
    parser.add_argument("--prose-mmhe", required=True)
    parser.add_argument("--base-action", required=True)
    parser.add_argument("--card-action", required=True)
    parser.add_argument("--prose-action", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = finalize(*[_load(getattr(args, name)) for name in (
        "representation", "renderer", "base_retention", "card_retention", "prose_retention",
        "card_mmhe", "prose_mmhe", "base_action", "card_action", "prose_action")])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
