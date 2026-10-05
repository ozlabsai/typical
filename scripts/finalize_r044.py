"""Fail-closed terminal gate for R044's matched pixel-replay continuation pilot."""

import argparse
import json
import math
from pathlib import Path


ARMS = ("parent", "a", "b")
RETENTION_SUITES = ("wf", "wh", "replay")
ACTION_MODES = ("raw", "calibrated")
MM_SUITES = ("he", "en", "real", "mismatch", "gold_dropped")
RENDERER_CELLS = ("bare_no", "bare_yes", "canonical_no", "canonical_yes")
PAIR_METRICS = ("accuracy", "exact_action_accuracy", "twin_both_correct")


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value, name, *, lower=0, upper=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
    return float(value)


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _retention(report, name):
    report = _mapping(report, name)
    result = {}
    for suite in RETENTION_SUITES:
        values = _mapping(report.get(suite), f"{name}.{suite}")
        if values.get("n") != 128 or values.get("skipped_overlength") != 0:
            raise ValueError(f"{name}.{suite} must score all 128 rows without skipping")
        result[suite] = _number(values.get("accuracy"), f"{name}.{suite}.accuracy")
    return result


def _action(report, name):
    report = _mapping(report, name)
    result = {}
    for mode in ACTION_MODES:
        values = _mapping(report.get(mode), f"{name}.{mode}")
        if values.get("n") != 256:
            raise ValueError(f"{name}.{mode}.n must be 256")
        if _mapping(values.get("error_categories"), f"{name}.{mode}.error_categories").get("invalid") != 0:
            raise ValueError(f"{name}.{mode} contains invalid predictions")
        result[mode] = _number(values.get("exact_action_accuracy"),
                               f"{name}.{mode}.exact_action_accuracy")
    return result


def _renderer(report, name):
    report = _mapping(report, name)
    if report.get("per_label") != 65:
        raise ValueError(f"{name} must use 65 cases per label")
    result = {}
    for cell in RENDERER_CELLS:
        if _mapping(report.get(cell), f"{name}.{cell}").get("n") != 130:
            raise ValueError(f"{name}.{cell}.n must be 130")
        result[cell] = _number(report.get(f"{cell}_both_orders"),
                               f"{name}.{cell}_both_orders")
    return result


def _mmhe(report, name):
    report = _mapping(report, name)
    result = {}
    for endpoint in ("base", "candidate"):
        values = _mapping(report.get(endpoint), f"{name}.{endpoint}")
        for suite in ("he", "en", "real", "shuffled", "noimage", "mismatch", "gold_dropped"):
            summary = _mapping(values.get(suite), f"{name}.{endpoint}.{suite}")
            if summary.get("n") != 128:
                raise ValueError(f"{name}.{endpoint}.{suite}.n must be 128")
    comparison = _mapping(report.get("comparison"), f"{name}.comparison")
    gates = _mapping(comparison.get("gates"), f"{name}.comparison.gates")
    if not gates or not all(type(value) is bool for value in gates.values()):
        raise ValueError(f"{name}.comparison.gates must contain booleans")
    absolute_pass = _boolean(comparison.get("promotion_pass"), f"{name}.comparison.promotion_pass")
    if absolute_pass != all(gates.values()):
        raise ValueError(f"{name} promotion result disagrees with its gates")
    for suite in MM_SUITES:
        result[suite] = _number(report["candidate"][suite].get("accuracy"),
                                f"{name}.candidate.{suite}.accuracy")
    return result, absolute_pass


def _pair(report, name):
    report = _mapping(report, name)
    if report.get("pairs") != 256 or report.get("presentations") != 512:
        raise ValueError(f"{name} must contain 256 pairs and 512 presentations")
    return {metric: _number(report.get(metric), f"{name}.{metric}") for metric in PAIR_METRICS}


def _training(config, metrics, name, parent_adapter):
    config = _mapping(config, f"{name}_config")
    expected = {"steps": 100, "microbatch": 1, "grad_accum": 8, "lr": 5e-6,
                "min_lr": 5e-7, "pair_flip_weight": .25,
                "pair_flip_margin": 1.0, "pair_flip_start_step": 0,
                "stage_counterfactual_pairs": False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{name}_config.{key} must be {value!r}")
    if config.get("init_adapter") != parent_adapter:
        raise ValueError(f"{name}_config.init_adapter does not match the frozen parent")
    trains = config.get("train")
    if not isinstance(trains, list) or len(trains) != 1 or Path(trains[0]).name != f"arm-{name}.jsonl":
        raise ValueError(f"{name}_config must train only arm-{name}.jsonl")
    if not isinstance(metrics, list) or len(metrics) != 101 or metrics[0].get("step") != 0:
        raise ValueError(f"{name}_metrics must contain baseline plus exactly 100 steps")
    steps = metrics[1:]
    if [row.get("step") for row in steps] != list(range(1, 101)):
        raise ValueError(f"{name}_metrics steps are incomplete")
    if any(row.get("pair_flip_weight") != .25 for row in steps):
        raise ValueError(f"{name}_metrics did not apply flip loss throughout training")
    if sum(row.get("pair_flip_seen", -1000) for row in steps) != 100:
        raise ValueError(f"{name}_metrics must expose each of 100 pairs exactly once")
    return True


def _manifest(report):
    report = _mapping(report, "manifest")
    if report.get("experiment") != "r044_mm_replay":
        raise ValueError("manifest experiment mismatch")
    if report.get("logical_rows") != {"arm-a": 800, "arm-b": 800}:
        raise ValueError("manifest must contain two 800-row arms")
    leakage = _mapping(report.get("leakage_checks"), "manifest.leakage_checks")
    required = {"selected_base_image_hashes_unique": True,
                "selected_source_ids_unique": True,
                "protected_image_overlap": 0,
                "protected_decision_overlap": 0,
                "arms_equal_without_visual_image": True}
    if any(leakage.get(key) != value for key, value in required.items()):
        raise ValueError("manifest leakage or arm-matching check failed")
    return True


def finalize(manifest, configs, metrics, retentions, actions, renderers, mmhe,
             *, parent_adapter, pairs=None):
    """Validate every report and return promotion plus the narrower causal claim."""
    if not parent_adapter:
        raise ValueError("parent adapter must not be empty")
    if any(set(group) != set(ARMS) for group in (retentions, actions, renderers, mmhe)):
        raise ValueError("retention, action, renderer and MM/HE reports require parent/A/B")
    if set(configs) != {"a", "b"} or set(metrics) != {"a", "b"}:
        raise ValueError("training evidence requires exactly A/B")
    if pairs is None or set(pairs) != set(ARMS):
        raise ValueError("pair reports require parent/A/B")
    _manifest(manifest)
    training = {arm: _training(configs[arm], metrics[arm], arm, parent_adapter)
                for arm in ("a", "b")}
    retention = {arm: _retention(retentions[arm], f"{arm}_retention") for arm in ARMS}
    action = {arm: _action(actions[arm], f"{arm}_action") for arm in ARMS}
    renderer = {arm: _renderer(renderers[arm], f"{arm}_renderer") for arm in ARMS}
    parsed_mmhe = {arm: _mmhe(mmhe[arm], f"{arm}_mmhe") for arm in ARMS}
    mm = {arm: value[0] for arm, value in parsed_mmhe.items()}
    absolute_mmhe = {arm: value[1] for arm, value in parsed_mmhe.items()}

    gates = {"manifest_pass": True, "a_training_pass": training["a"],
             "b_training_pass": training["b"],
             "b_absolute_mmhe_promotion_pass": absolute_mmhe["b"]}
    for suite in RETENTION_SUITES:
        gates[f"b_{suite}_loses_at_most_.02_vs_parent"] = (
            retention["b"][suite] >= retention["parent"][suite] - .02)
    for mode in ACTION_MODES:
        gates[f"b_{mode}_action_loses_at_most_1_of_256_vs_parent"] = (
            action["b"][mode] >= action["parent"][mode] - 1 / 256)
    for cell in RENDERER_CELLS:
        gates[f"b_{cell}_loses_at_most_1_of_65_vs_parent"] = (
            renderer["b"][cell] >= renderer["parent"][cell] - 1 / 65)
    b_mm_gates = {}
    a_mm_gates = {}
    for suite in MM_SUITES:
        b_mm_gates[suite] = mm["b"][suite] >= mm["parent"][suite] - 1 / 128
        a_mm_gates[suite] = mm["a"][suite] >= mm["parent"][suite] - 1 / 128
        gates[f"b_{suite}_loses_at_most_1_of_128_vs_parent"] = b_mm_gates[suite]

    pair_values = {arm: _pair(pairs[arm], f"{arm}_pairs") for arm in ARMS}
    for metric in PAIR_METRICS:
        gates[f"b_pair_{metric}_loses_at_most_.01_vs_parent"] = (
            pair_values["b"][metric] >= pair_values["parent"][metric] - .01)

    b_mm_pass = absolute_mmhe["b"] and all(b_mm_gates.values())
    a_mm_pass = absolute_mmhe["a"] and all(a_mm_gates.values())
    real_gain = round((mm["b"]["real"] - mm["a"]["real"]) * 128)
    gold_gain = round((mm["b"]["gold_dropped"] - mm["a"]["gold_dropped"]) * 128)
    causal_gates = {
        "a_fails_mm_promotion": not a_mm_pass,
        "b_passes_mm_promotion": b_mm_pass,
        "b_minus_a_net_gain_at_least_4_correct": real_gain + gold_gain >= 4,
        "b_real_not_worse_than_a": real_gain >= 0,
        "b_gold_dropped_not_worse_than_a": gold_gain >= 0,
    }
    return {
        "parent_adapter": parent_adapter,
        "retention_accuracy": retention,
        "action_accuracy": action,
        "renderer_both_order_accuracy": renderer,
        "mmhe_accuracy": mm,
        "absolute_mmhe_promotion": absolute_mmhe,
        "pair_metrics": pair_values,
        "gates": gates,
        "promotion_pass": all(gates.values()),
        "causal_image_replay": {"real_correct_gain_b_minus_a": real_gain,
                                "gold_dropped_correct_gain_b_minus_a": gold_gain,
                                "gates": causal_gates,
                                "claim_pass": all(causal_gates.values())},
    }


def _load(path):
    return json.loads(Path(path).read_text())


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parent-adapter", required=True)
    for arm in ("a", "b"):
        parser.add_argument(f"--{arm}-config", required=True)
        parser.add_argument(f"--{arm}-metrics", required=True)
    for arm in ARMS:
        for report in ("retention", "action", "renderer", "mmhe"):
            parser.add_argument(f"--{arm}-{report}", required=True)
        parser.add_argument(f"--{arm}-pairs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = finalize(
        _load(args.manifest),
        {arm: _load(getattr(args, f"{arm}_config")) for arm in ("a", "b")},
        {arm: _jsonl(getattr(args, f"{arm}_metrics")) for arm in ("a", "b")},
        {arm: _load(getattr(args, f"{arm}_retention")) for arm in ARMS},
        {arm: _load(getattr(args, f"{arm}_action")) for arm in ARMS},
        {arm: _load(getattr(args, f"{arm}_renderer")) for arm in ARMS},
        {arm: _load(getattr(args, f"{arm}_mmhe")) for arm in ARMS},
        parent_adapter=args.parent_adapter,
        pairs={arm: _load(getattr(args, f"{arm}_pairs")) for arm in ARMS},
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["promotion_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
