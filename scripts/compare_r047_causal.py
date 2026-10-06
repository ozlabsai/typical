"""Fail-closed paired comparison of two sealed R047 causal-panel predictions."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.eval_r047_executor import causal_coverage_report, validate_sealed_causal_predictions


PRIMITIVES = ("conditional", "datetime", "decimal", "fraction", "ledger", "policy")
MIN_BOOTSTRAP_REPLICATES = 20_000


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_payload(payload):
    try:
        records = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid sealed prediction JSONL") from error
    return validate_sealed_causal_predictions(records)


def _prediction(record):
    no, yes = (float(record["probabilities"][label]) for label in ("no", "yes"))
    return "yes" if yes > no else "no" if no > yes else "tie"


def _scored(record):
    predicted = _prediction(record)
    yes = float(record["probabilities"]["yes"])
    probability = yes if record["gold"] == "yes" else 1 - yes
    return {**record, "predicted": predicted, "correct": predicted == record["gold"],
            "yes_logit": math.log(max(yes, 1e-12) / max(1 - yes, 1e-12)),
            "nll": -math.log(max(probability, 1e-12)),
            "brier": (yes - float(record["gold"] == "yes")) ** 2}


def _summary(items):
    no = [item for item in items if item["gold"] == "no"]
    yes = [item for item in items if item["gold"] == "yes"]
    tnr = sum(item["correct"] for item in no) / len(no) if no else None
    tpr = sum(item["correct"] for item in yes) / len(yes) if yes else None
    recalls = [value for value in (tnr, tpr) if value is not None]
    return {
        "n": len(items),
        "accuracy": sum(item["correct"] for item in items) / len(items),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "tn": sum(item["predicted"] == "no" for item in no),
        "fp": sum(item["predicted"] != "no" for item in no),
        "tp": sum(item["predicted"] == "yes" for item in yes),
        "fn": sum(item["predicted"] != "yes" for item in yes),
        "tnr": tnr,
        "tpr": tpr,
        "nll": sum(item["nll"] for item in items) / len(items),
        "brier": sum(item["brier"] for item in items) / len(items),
    }


def _slices(items, key):
    groups = defaultdict(list)
    for item in items:
        value = item[key]
        name = "+".join(value) if isinstance(value, list) else str(value)
        groups[name].append(item)
    return {name: _summary(group) for name, group in sorted(groups.items())}


def _view_groups(scored):
    groups = defaultdict(list)
    for item in scored:
        groups[item["semantic_view_id"]].append(item)
    return groups


def validate_panel_coverage(records):
    """Require a complete truth-cube × language × order panel in every cluster."""
    validate_sealed_causal_predictions(records)
    clusters = defaultdict(list)
    for record in records:
        clusters[record["semantic_cluster_id"]].append(record)
    for cluster_id, items in clusters.items():
        compositions = {tuple(item["composition"]) for item in items}
        scenarios = {item["scenario_id"] for item in items}
        if len(compositions) != 1 or len(scenarios) != 1:
            raise ValueError(f"{cluster_id}: cluster composition/scenario is not stable")
        composition = next(iter(compositions))
        k = len(composition)
        prompt_families = {item["prompt_family"] for item in items
                           if item["prediction_role"] == "clause"}
        if len(prompt_families) != 1:
            raise ValueError(f"{cluster_id}: cluster must use exactly one clause prompt family")
        views = defaultdict(list)
        for item in items:
            views[item["semantic_view_id"]].append(item)
        observed = set()
        for view_id, view in views.items():
            signatures = {(tuple(item["truth_pattern"]), item["language"],
                           item["option_order"], item["seen_edge_count"])
                          for item in view}
            if len(signatures) != 1:
                raise ValueError(f"{view_id}: semantic view metadata is inconsistent")
            signature = next(iter(signatures))
            finals = [item for item in view if item["prediction_role"] == "final"]
            clauses = [item for item in view if item["prediction_role"] == "clause"]
            if (len(finals) != 1 or len(clauses) != k
                    or {item["primitive"] for item in clauses} != set(composition)):
                raise ValueError(f"{view_id}: semantic view lacks one final and every clause")
            observed.add(signature[:3])
        expected = {(tuple(bool(mask & (1 << index)) for index in range(k)), language, order)
                    for mask in range(2 ** k) for language in ("en", "he")
                    for order in ("no_yes", "yes_no")}
        if observed != expected or len(views) != len(expected):
            raise ValueError(f"{cluster_id}: incomplete truth-cube/language/order coverage")
    return clusters


def _direction_items(scored):
    finals = [item for item in scored if item["prediction_role"] == "final"]
    groups = defaultdict(dict)
    for item in finals:
        key = (item["semantic_cluster_id"], item["language"], item["option_order"])
        pattern = tuple(item["truth_pattern"])
        if pattern in groups[key]:
            raise ValueError(f"duplicate final truth pattern in direction group {key}")
        groups[key][pattern] = item
    directions = []
    for (_, language, order), patterns in groups.items():
        sample = next(iter(patterns.values()))
        composition = sample["composition"]
        positive_pattern = tuple(True for _ in composition)
        if positive_pattern not in patterns:
            raise ValueError("direction group lacks all-true final")
        positive = patterns[positive_pattern]
        for index, primitive in enumerate(composition):
            negative_pattern = tuple(position != index for position in range(len(composition)))
            if negative_pattern not in patterns:
                raise ValueError("direction group lacks one-false counterfactual")
            negative = patterns[negative_pattern]
            directions.append({"primitive": primitive, "language": language,
                               "option_order": order,
                               "correct": positive["yes_logit"] > negative["yes_logit"]})
    return directions


def _option_invariance(scored):
    groups = defaultdict(list)
    for item in scored:
        key = (item["semantic_cluster_id"], item["scenario_id"],
               tuple(item["composition"]), tuple(item["truth_pattern"]), item["language"],
               item["prompt_family"], item["prediction_role"], item["primitive"],
               item["decisive_role"], item["seen_edge_count"])
        groups[key].append(item)
    agreements, deltas, divergences = [], [], []
    for key, pair in groups.items():
        if len(pair) != 2 or {item["option_order"] for item in pair} != {
                "no_yes", "yes_no"}:
            raise ValueError(f"semantic option-order pair is incomplete: {key}")
        left, right = pair
        p, q = (float(item["probabilities"]["yes"]) for item in pair)
        middle = (p + q) / 2

        def kl(value, reference):
            result = 0.0
            for first, second in ((value, reference), (1 - value, 1 - reference)):
                if first:
                    result += first * math.log(first / max(second, 1e-12))
            return result

        agreements.append(left["predicted"] == right["predicted"])
        deltas.append(abs(p - q))
        divergences.append((kl(p, middle) + kl(q, middle)) / 2)
    return {"pairs": len(groups), "semantic_agreement": sum(agreements) / len(agreements),
            "yes_probability_abs_delta_mean": sum(deltas) / len(deltas),
            "yes_probability_abs_delta_max": max(deltas),
            "jsd_mean_nats": sum(divergences) / len(divergences),
            "jsd_max_nats": max(divergences)}


def causal_metrics(records, arity):
    """Recompute all causal metrics solely from sealed prediction records."""
    validate_panel_coverage(records)
    scored = [_scored(item) for item in records if item["arity"] == arity]
    if not scored:
        raise ValueError(f"sealed predictions contain no K={arity} records")
    clauses = [item for item in scored if item["prediction_role"] == "clause"]
    finals = [item for item in scored if item["prediction_role"] == "final"]
    by_primitive = _slices(clauses, "primitive")
    clause_macro_ba = sum(item["balanced_accuracy"] for item in by_primitive.values()) / len(
        by_primitive)
    vectors, coherences, complete_views = [], [], []
    for view_id, view in _view_groups(scored).items():
        final = [item for item in view if item["prediction_role"] == "final"]
        view_clauses = [item for item in view if item["prediction_role"] == "clause"]
        if len(final) != 1 or len(view_clauses) != arity:
            raise ValueError(f"{view_id}: incomplete scored semantic view")
        vectors.append(all(item["correct"] for item in view_clauses))
        complete_views.append(all(item["correct"] for item in view))
        composed = "yes" if all(item["predicted"] == "yes" for item in view_clauses) else "no"
        coherences.append(final[0]["predicted"] == composed)
    directions = _direction_items(scored)
    direction_groups = defaultdict(list)
    for item in directions:
        direction_groups[item["primitive"]].append(item["correct"])
    crossed = defaultdict(list)
    for item in clauses:
        key = f"{item['primitive']}|{item['language']}|{item['option_order']}"
        crossed[key].append(item)
    cluster_groups = defaultdict(list)
    for item in scored:
        cluster_groups[item["semantic_cluster_id"]].append(item)
    complete_clusters = [all(item["correct"] for item in items)
                         for items in cluster_groups.values()]
    return {
        "arity": arity,
        "semantic_clusters": len({item["semantic_cluster_id"] for item in scored}),
        "clauses": {**_summary(clauses), "macro_balanced_accuracy": clause_macro_ba,
                    "primitive": by_primitive,
                    "primitive_language": _slices(
                        [{**item, "primitive_language":
                          f"{item['primitive']}|{item['language']}"} for item in clauses],
                        "primitive_language"),
                    "language": _slices(clauses, "language"),
                    "composition": _slices(clauses, "composition"),
                    "option_order": _slices(clauses, "option_order"),
                    "prompt_family": _slices(clauses, "prompt_family"),
                    "decisive_role": _slices(clauses, "decisive_role"),
                    "truth_hamming_weight": _slices(clauses, "truth_hamming_weight"),
                    "seen_edge_count": _slices(clauses, "seen_edge_count"),
                    "primitive_language_order": {
                        name: _summary(items) for name, items in sorted(crossed.items())}},
        "vectors": {"n": len(vectors), "accuracy": sum(vectors) / len(vectors)},
        "coherence": {"n": len(coherences), "accuracy": sum(coherences) / len(coherences)},
        "complete_semantic_views": {
            "n": len(complete_views), "correct": sum(complete_views),
            "accuracy": sum(complete_views) / len(complete_views)},
        "complete_semantic_clusters": {
            "n": len(complete_clusters), "correct": sum(complete_clusters),
            "accuracy": sum(complete_clusters) / len(complete_clusters)},
        "final": {**_summary(finals),
                  "language": _slices(finals, "language"),
                  "composition": _slices(finals, "composition"),
                  "option_order": _slices(finals, "option_order"),
                  "prompt_family": _slices(finals, "prompt_family"),
                  "truth_hamming_weight": _slices(finals, "truth_hamming_weight"),
                  "seen_edge_count": _slices(finals, "seen_edge_count")},
        "direction": {primitive: {"n": len(values), "rate": sum(values) / len(values)}
                      for primitive, values in sorted(direction_groups.items())},
        "option_invariance": _option_invariance(scored),
    }


def _assert_matching_structure(left, right):
    if [item["id"] for item in left] != [item["id"] for item in right]:
        raise ValueError("arm prediction order/coverage differs")
    ignored = {"probabilities", "record_sha256"}
    for first, second in zip(left, right):
        if ({key: value for key, value in first.items() if key not in ignored}
                != {key: value for key, value in second.items() if key not in ignored}):
            raise ValueError(f"arm semantic metadata differs for {first['id']}")


def _cluster_feature(records):
    scored = [_scored(item) for item in records]
    values = []
    clauses = [item for item in scored if item["prediction_role"] == "clause"]
    finals = [item for item in scored if item["prediction_role"] == "final"]
    for primitive in PRIMITIVES:
        subset = [item for item in clauses if item["primitive"] == primitive]
        yes = [item for item in subset if item["gold"] == "yes"]
        no = [item for item in subset if item["gold"] == "no"]
        values.extend((sum(item["correct"] for item in yes), len(yes),
                       sum(item["correct"] for item in no), len(no)))
    vectors, coherences = [], []
    for view in _view_groups(scored).values():
        final = next(item for item in view if item["prediction_role"] == "final")
        view_clauses = [item for item in view if item["prediction_role"] == "clause"]
        vectors.append(all(item["correct"] for item in view_clauses))
        composed = "yes" if all(item["predicted"] == "yes" for item in view_clauses) else "no"
        coherences.append(final["predicted"] == composed)
    values.extend((sum(vectors), len(vectors), sum(coherences), len(coherences)))
    final_yes = [item for item in finals if item["gold"] == "yes"]
    final_no = [item for item in finals if item["gold"] == "no"]
    values.extend((sum(item["correct"] for item in final_yes), len(final_yes),
                   sum(item["correct"] for item in final_no), len(final_no)))
    values.extend((sum(item["nll"] for item in finals), sum(item["brier"] for item in finals),
                   len(finals), sum(item["nll"] for item in clauses),
                   sum(item["brier"] for item in clauses), len(clauses)))
    directions = _direction_items(scored)
    for primitive in PRIMITIVES:
        subset = [item for item in directions if item["primitive"] == primitive]
        values.extend((sum(item["correct"] for item in subset), len(subset)))
    return np.asarray(values, dtype=np.float64)


def _feature_metrics(matrix):
    """Turn one or many additive feature vectors into metric arrays."""
    one = matrix.ndim == 1
    values = matrix[None, :] if one else matrix
    primitive_bas = []
    for index in range(len(PRIMITIVES)):
        tp, pos, tn, neg = (values[:, index * 4 + offset] for offset in range(4))
        with np.errstate(divide="ignore", invalid="ignore"):
            primitive_bas.append((tp / pos + tn / neg) / 2)
    macro_ba = np.nanmean(np.stack(primitive_bas, axis=1), axis=1)
    cursor = len(PRIMITIVES) * 4
    vector = values[:, cursor] / values[:, cursor + 1]
    coherence = values[:, cursor + 2] / values[:, cursor + 3]
    cursor += 4
    final_ba = ((values[:, cursor] / values[:, cursor + 1]
                 + values[:, cursor + 2] / values[:, cursor + 3]) / 2)
    cursor += 4
    final_nll = values[:, cursor] / values[:, cursor + 2]
    final_brier = values[:, cursor + 1] / values[:, cursor + 2]
    clause_nll = values[:, cursor + 3] / values[:, cursor + 5]
    clause_brier = values[:, cursor + 4] / values[:, cursor + 5]
    cursor += 6
    result = {"clause_macro_ba": macro_ba, "vector_accuracy": vector,
              "coherence": coherence, "final_ba": final_ba,
              "final_nll": final_nll, "final_brier": final_brier,
              "clause_nll": clause_nll, "clause_brier": clause_brier}
    for index, primitive in enumerate(PRIMITIVES):
        numerator, denominator = values[:, cursor + 2 * index], values[:, cursor + 2 * index + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"direction_{primitive}"] = numerator / denominator
    return {key: value[0] for key, value in result.items()} if one else result


def _bootstrap(control, treatment, arity, replicates, seed):
    if replicates < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError(f"paired bootstrap requires at least {MIN_BOOTSTRAP_REPLICATES} replicates")
    control_groups, treatment_groups = defaultdict(list), defaultdict(list)
    for item in control:
        if item["arity"] == arity:
            control_groups[item["semantic_cluster_id"]].append(item)
    for item in treatment:
        if item["arity"] == arity:
            treatment_groups[item["semantic_cluster_id"]].append(item)
    if set(control_groups) != set(treatment_groups):
        raise ValueError(f"K={arity} semantic-cluster coverage differs")
    strata = defaultdict(list)
    for cluster_id, items in control_groups.items():
        strata[tuple(items[0]["composition"])].append(cluster_id)
    width = len(_cluster_feature(next(iter(control_groups.values()))))
    control_total = np.zeros((replicates, width), dtype=np.float64)
    treatment_total = np.zeros_like(control_total)
    rng = np.random.Generator(np.random.PCG64(seed + arity))
    for composition in sorted(strata):
        cluster_ids = sorted(strata[composition])
        control_values = np.stack([_cluster_feature(control_groups[key]) for key in cluster_ids])
        treatment_values = np.stack([_cluster_feature(treatment_groups[key]) for key in cluster_ids])
        sampled = rng.integers(0, len(cluster_ids), size=(replicates, len(cluster_ids)))
        control_total += control_values[sampled].sum(axis=1)
        treatment_total += treatment_values[sampled].sum(axis=1)
    control_metrics = _feature_metrics(control_total)
    treatment_metrics = _feature_metrics(treatment_total)
    result = {}
    for name in control_metrics:
        delta = treatment_metrics[name] - control_metrics[name]
        finite = delta[np.isfinite(delta)]
        if not len(finite):
            continue
        result[name] = {"replicates": replicates, "seed": seed + arity,
                        "delta_mean": float(np.mean(finite)),
                        "delta_lcb_95": float(np.quantile(finite, .025, method="lower")),
                        "delta_ucb_95": float(np.quantile(finite, .975, method="higher"))}
    return result


def _delta(treatment, control, path):
    left, right = treatment, control
    for key in path:
        left, right = left[key], right[key]
    return left - right


def _gate(control, treatment, reference, bootstrap):
    deltas = {
        "clause_macro_ba": _delta(treatment, control,
                                   ("clauses", "macro_balanced_accuracy")),
        "vector_accuracy": _delta(treatment, control, ("vectors", "accuracy")),
        "coherence": _delta(treatment, control, ("coherence", "accuracy")),
        "final_ba": _delta(treatment, control, ("final", "balanced_accuracy")),
    }
    failures = []
    if deltas["clause_macro_ba"] < .05:
        failures.append("clause macro balanced-accuracy delta below +5pp")
    if bootstrap["clause_macro_ba"]["delta_lcb_95"] <= 0:
        failures.append("clause macro balanced-accuracy bootstrap LCB is not above zero")
    if deltas["vector_accuracy"] < .08:
        failures.append("clause-vector delta below +8pp")
    if deltas["coherence"] < .10:
        failures.append("coherence delta below +10pp")
    if deltas["final_ba"] < .05:
        failures.append("final balanced-accuracy delta below +5pp")
    for primitive in ("decimal", "ledger"):
        if (primitive not in control["direction"] or primitive not in treatment["direction"]
                or treatment["direction"][primitive]["rate"]
                <= control["direction"][primitive]["rate"]):
            failures.append(f"{primitive} direction did not improve")
    dimension_deltas = {}
    for section in ("clauses", "final"):
        dimension_deltas[section] = {}
        for dimension in ("language", "option_order"):
            dimension_deltas[section][dimension] = {}
            if set(control[section][dimension]) != set(treatment[section][dimension]):
                raise ValueError(f"{section} {dimension} coverage differs")
            for name in sorted(control[section][dimension]):
                effect = _delta(treatment[section][dimension], control[section][dimension],
                                (name, "balanced_accuracy"))
                dimension_deltas[section][dimension][name] = effect
                if effect < 0:
                    failures.append(f"{section} {dimension} {name} point effect is negative")
    order_interaction = {}
    for section in ("clauses", "final"):
        order_effects = dimension_deltas[section]["option_order"]
        if set(order_effects) != {"no_yes", "yes_no"}:
            raise ValueError(f"{section} option-order coverage is incomplete")
        order_interaction[section] = order_effects["yes_no"] - order_effects["no_yes"]
        for order, effect in order_effects.items():
            if effect <= 0:
                failures.append(f"{section} mechanism gain is not positive for order {order}")
    cell_deltas = {}
    for name in sorted(control["clauses"]["primitive_language_order"]):
        if name not in treatment["clauses"]["primitive_language_order"]:
            raise ValueError(f"treatment lacks robustness cell {name}")
        cell_deltas[name] = _delta(
            treatment["clauses"]["primitive_language_order"],
            control["clauses"]["primitive_language_order"],
            (name, "balanced_accuracy"))
        if cell_deltas[name] < -.02:
            failures.append(f"robustness cell {name} regressed by more than 2pp")
    complete_deltas = {}
    for section in ("complete_semantic_views", "complete_semantic_clusters"):
        complete_deltas[section] = {
            "correct": treatment[section]["correct"] - control[section]["correct"],
            "accuracy": treatment[section]["accuracy"] - control[section]["accuracy"],
        }
        if any(value < 0 for value in complete_deltas[section].values()):
            failures.append(f"{section} correctness regressed")
    calibration_ratios = {}
    for baseline_name, baseline_report in (("control", control), ("reference", reference)):
        calibration_ratios[baseline_name] = {}
        for section in ("clauses", "final"):
            calibration_ratios[baseline_name][section] = {}
            for metric in ("nll", "brier"):
                baseline = baseline_report[section][metric]
                candidate = treatment[section][metric]
                ratio = candidate / baseline if baseline else (1.0 if candidate == 0 else math.inf)
                calibration_ratios[baseline_name][section][metric] = ratio
                if ratio > 1.05:
                    failures.append(
                        f"{section} {metric} exceeds 1.05x {baseline_name}")
    return {"passed": not failures, "deltas": deltas, "cell_deltas": cell_deltas,
            "dimension_deltas": dimension_deltas,
            "treatment_x_order_interaction": order_interaction,
            "complete_correctness_deltas": complete_deltas,
            "calibration_ratios": calibration_ratios, "failures": failures}


def compare_payloads(first_payload, second_payload, assignment_payload, assignment_sha256,
                     first_sha256, second_sha256, replicates=MIN_BOOTSTRAP_REPLICATES,
                     seed=470047, reference_payload=None, reference_sha256=None):
    """Hash and validate both blinded arms before revealing the frozen assignment."""
    actual = (_sha(first_payload), _sha(second_payload))
    if actual != (first_sha256, second_sha256):
        raise ValueError("sealed prediction file hash mismatch")
    if actual[0] == actual[1]:
        raise ValueError("causal arms must have distinct sealed prediction hashes")
    first, second = _read_payload(first_payload), _read_payload(second_payload)
    _assert_matching_structure(first, second)
    validate_panel_coverage(first)
    validate_panel_coverage(second)
    blinded = dict(zip(sorted(actual), ("arm_0", "arm_1")))

    if reference_payload is None or reference_sha256 is None:
        raise ValueError("final causal gate requires frozen reference predictions and hash")
    if _sha(reference_payload) != reference_sha256:
        raise ValueError("frozen reference prediction hash mismatch")
    reference = _read_payload(reference_payload)
    _assert_matching_structure(first, reference)
    validate_panel_coverage(reference)

    if _sha(assignment_payload) != assignment_sha256:
        raise ValueError("frozen arm-assignment hash mismatch")
    try:
        assignment = json.loads(assignment_payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid frozen arm assignment") from error
    if set(assignment) != {"control_sha256", "treatment_sha256"}:
        raise ValueError("arm assignment must contain exactly control/treatment hashes")
    if {assignment["control_sha256"], assignment["treatment_sha256"]} != set(actual):
        raise ValueError("arm assignment does not match both sealed prediction hashes")
    by_hash = {actual[0]: first, actual[1]: second}
    control = by_hash[assignment["control_sha256"]]
    treatment = by_hash[assignment["treatment_sha256"]]

    strata, gates = {}, {}
    for arity in (2, 3):
        control_metrics = causal_metrics(control, arity)
        treatment_metrics = causal_metrics(treatment, arity)
        reference_metrics = causal_metrics(reference, arity)
        bootstrap = _bootstrap(control, treatment, arity, replicates, seed)
        gate = _gate(control_metrics, treatment_metrics, reference_metrics, bootstrap)
        strata[f"K{arity}"] = {"control": control_metrics, "treatment": treatment_metrics,
                               "reference": reference_metrics,
                               "paired_bootstrap": bootstrap, "gate": gate}
        gates[f"K{arity}"] = gate["passed"]
    return {
        "schema_version": "r047-causal-comparison-v1",
        "blinding": {"prediction_hashes_sorted": sorted(actual), "blind_labels": blinded,
                     "assignment_sha256": assignment_sha256,
                     "revealed_only_after_both_hashes_validated": True},
        "arms": assignment,
        "reference_sha256": reference_sha256,
        "coverage": {"control": causal_coverage_report(control),
                     "treatment": causal_coverage_report(treatment),
                     "reference": causal_coverage_report(reference)},
        "bootstrap": {"method": "paired stratified semantic-cluster percentile",
                      "unit": "whole semantic cluster", "strata": "arity_x_composition",
                      "replicates": replicates, "base_seed": seed,
                      "k2_k3_never_pooled": True},
        "strata": strata,
        "co_primary_gate": {"passed": all(gates.values()), "strata": gates,
                            "rule": "intersection-union; K2 and K3 must each pass"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--first-sha256", required=True)
    parser.add_argument("--second-sha256", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=MIN_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=470047)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    first, second = Path(args.first).read_bytes(), Path(args.second).read_bytes()
    assignment = Path(args.assignment).read_bytes()
    reference = Path(args.reference).read_bytes()
    result = compare_payloads(
        first, second, assignment, args.assignment_sha256, args.first_sha256,
        args.second_sha256, args.bootstrap_replicates, args.bootstrap_seed,
        reference, args.reference_sha256)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
