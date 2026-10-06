"""Fail-closed paired comparison of two sealed R047 causal-panel predictions."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.eval_r047_executor import (
    _canonical_json,
    causal_parity_subset,
    causal_coverage_report,
    validate_causal_run_receipt,
    validate_causal_panel_payload,
    validate_sealed_causal_predictions,
)


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
                               "correct": positive["yes_logit"] > negative["yes_logit"],
                               "pair_both_correct": positive["correct"] and negative["correct"]})
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
    def direction_slices(key):
        groups = defaultdict(list)
        for item in directions:
            name = key(item)
            groups[name].append(item)
        return {name: {"n": len(items),
                       "rate": sum(item["correct"] for item in items) / len(items),
                       "pair_both_correct": sum(item["pair_both_correct"] for item in items)
                       / len(items)}
                for name, items in sorted(groups.items())}
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
        "direction": {
            "primitive": direction_slices(lambda item: item["primitive"]),
            "language": direction_slices(lambda item: item["language"]),
            "option_order": direction_slices(lambda item: item["option_order"]),
            "primitive_language_order": direction_slices(
                lambda item: f"{item['primitive']}|{item['language']}|{item['option_order']}"),
            "hamming1_pairs": {"n": len(directions),
                               "both_correct": sum(item["pair_both_correct"]
                                                   for item in directions),
                               "pair_both_correct": sum(item["pair_both_correct"]
                                                        for item in directions) / len(directions)},
        },
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
    final_accuracy = ((values[:, cursor] + values[:, cursor + 2])
                      / (values[:, cursor + 1] + values[:, cursor + 3]))
    cursor += 4
    final_nll = values[:, cursor] / values[:, cursor + 2]
    final_brier = values[:, cursor + 1] / values[:, cursor + 2]
    clause_nll = values[:, cursor + 3] / values[:, cursor + 5]
    clause_brier = values[:, cursor + 4] / values[:, cursor + 5]
    cursor += 6
    result = {"clause_macro_ba": macro_ba, "vector_accuracy": vector,
              "coherence": coherence, "final_ba": final_ba,
              "final_accuracy": final_accuracy,
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
        "final_accuracy": _delta(treatment, control, ("final", "accuracy")),
    }
    failures = []
    secondary_values = {
        "vector_accuracy": (control["vectors"]["accuracy"],
                            treatment["vectors"]["accuracy"], .08),
        "coherence": (control["coherence"]["accuracy"],
                      treatment["coherence"]["accuracy"], .10),
        "final_accuracy": (control["final"]["accuracy"],
                           treatment["final"]["accuracy"], .05),
    }
    if deltas["clause_macro_ba"] < .05:
        failures.append("clause macro balanced-accuracy delta below +5pp")
    if control["clauses"]["macro_balanced_accuracy"] > .95:
        failures.append("uninformative_primary_ceiling: harder panel required")
    if bootstrap["clause_macro_ba"]["delta_lcb_95"] <= 0:
        failures.append("clause macro balanced-accuracy bootstrap LCB is not above zero")
    secondary_gate_contract = {}
    for name, (baseline, candidate, target) in secondary_values.items():
        if baseline <= 1 - target:
            passed = deltas[name] >= target
            secondary_gate_contract[name] = {
                "branch": "superiority", "target_delta": target,
                "control": baseline, "treatment": candidate, "passed": passed}
        else:
            lcb = bootstrap[name]["delta_lcb_95"]
            passed = candidate >= .98 and deltas[name] >= 0 and lcb > -.01
            secondary_gate_contract[name] = {
                "branch": "ceiling_noninferiority", "treatment_floor": .98,
                "delta_floor": 0, "bootstrap_lcb_floor_exclusive": -.01,
                "observed_delta_lcb_95": lcb, "control": baseline,
                "treatment": candidate, "passed": passed}
        if not passed:
            failures.append(f"{name} failed preregistered secondary {secondary_gate_contract[name]['branch']}")
    for primitive in ("decimal", "ledger"):
        if (primitive not in control["direction"]["primitive"]
                or primitive not in treatment["direction"]["primitive"]
                or treatment["direction"]["primitive"][primitive]["rate"]
                < min(1.0, control["direction"]["primitive"][primitive]["rate"] + 1e-12)):
            failures.append(f"{primitive} direction did not improve")
    direction_cell_deltas = {}
    for dimension in ("primitive", "language", "option_order", "primitive_language_order"):
        direction_cell_deltas[dimension] = {}
        if set(control["direction"][dimension]) != set(treatment["direction"][dimension]):
            raise ValueError(f"direction {dimension} coverage differs")
        for name in sorted(control["direction"][dimension]):
            effect = _delta(treatment["direction"][dimension],
                            control["direction"][dimension], (name, "rate"))
            direction_cell_deltas[dimension][name] = effect
            if effect < -.02:
                failures.append(
                    f"direction {dimension} cell {name} regressed by more than 2pp")
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
            control_value = control[section]["option_order"][order]["balanced_accuracy"]
            if effect <= 0 and control_value < 1:
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
    hamming1_delta = {
        "both_correct": (treatment["direction"]["hamming1_pairs"]["both_correct"]
                         - control["direction"]["hamming1_pairs"]["both_correct"]),
        "pair_both_correct": (treatment["direction"]["hamming1_pairs"]["pair_both_correct"]
                              - control["direction"]["hamming1_pairs"]["pair_both_correct"]),
    }
    if any(value < 0 for value in hamming1_delta.values()):
        failures.append("Hamming-1 pair-both correctness regressed")
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
            "secondary_gate_contract": secondary_gate_contract,
            "secondary_ceiling_rule": (
                "superiority below ceiling; otherwise treatment>=.98, delta>=0, "
                "paired cluster-bootstrap LCB95>-.01"),
            "direction_cell_deltas": direction_cell_deltas,
            "dimension_deltas": dimension_deltas,
            "treatment_x_order_interaction": order_interaction,
            "complete_correctness_deltas": complete_deltas,
            "hamming1_pair_both_delta": hamming1_delta,
            "calibration_ratios": calibration_ratios, "failures": failures}


ASSIGNMENT_FIELDS = {
    "schema_version", "control_slot", "treatment_slot", "control_train_sha256",
    "treatment_train_sha256", "reference_train_sha256", "reference_adapter_sha256",
    "parent_adapter_sha256", "model_revision", "model_inventory_sha256",
    "recipe_sha256", "panel_sha256", "panel_manifest_sha256",
}


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _bind_predictions_to_panel(records, panel_rows):
    if [item["id"] for item in records] != [row.get("id") for row in panel_rows]:
        raise ValueError("sealed predictions do not exactly cover the raw panel in order")
    for record, row in zip(records, panel_rows):
        expected = _sha(_canonical_json(row).encode())
        if record["row_sha256"] != expected:
            raise ValueError(f"{record['id']}: sealed row hash differs from raw panel")


def _load_evaluator_report(payload, expected_sha256, name, predictions_payload,
                           panel_sha256, manifest_sha256, expected):
    if not _valid_sha(expected_sha256) or _sha(payload) != expected_sha256:
        raise ValueError(f"{name} evaluator-report hash mismatch")
    try:
        report = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {name} evaluator report") from error
    checks = {
        "predictions_sha256": _sha(predictions_payload),
        "input_sha256": panel_sha256,
        "panel_manifest_sha256": manifest_sha256,
        "adapter_sha256": expected["adapter_sha256"],
        "model_revision": expected["model_revision"],
        "train_sha256": expected["train_sha256"],
        "recipe_sha256": expected["recipe_sha256"],
        "parent_adapter_sha256": expected["parent_adapter_sha256"],
        "run_receipt_sha256": expected["run_receipt_sha256"],
        "model_inventory_sha256": expected["model_inventory_sha256"],
        "records": expected["rows"],
        "panel_rows": expected["rows"],
        "semantic_clusters": expected["clusters"],
        "scenarios": expected["clusters"],
    }
    failures = [key for key, value in checks.items() if report.get(key) != value]
    if failures:
        raise ValueError(f"{name} evaluator report provenance/count mismatch: {failures}")
    if (not _valid_sha(report.get("model_tree_sha256"))
            or not isinstance(report.get("model_tree_files"), int)
            or isinstance(report.get("model_tree_files"), bool)
            or report["model_tree_files"] <= 0):
        raise ValueError(f"{name} evaluator model-tree receipt is invalid")
    parity = report.get("batch_scalar_parity")
    parity_fields = {"status", "rows", "subset_ids_sha256", "exact_discrete_outputs",
                     "max_abs_probability_delta", "tolerance", "selection"}
    if (not isinstance(parity, dict) or set(parity) != parity_fields
            or parity.get("status") != "pass" or parity.get("exact_discrete_outputs") is not True
            or not isinstance(parity.get("rows"), int) or isinstance(parity.get("rows"), bool)
            or parity["rows"] <= 0 or not _valid_sha(parity.get("subset_ids_sha256"))
            or parity.get("tolerance") != 1e-5
            or not isinstance(parity.get("max_abs_probability_delta"), (int, float))
            or not math.isfinite(parity["max_abs_probability_delta"])
            or parity["max_abs_probability_delta"] > 1e-5):
        raise ValueError(f"{name} evaluator batched/scalar parity contract failed")
    if (parity["rows"] != expected["parity_rows"]
            or parity["subset_ids_sha256"] != expected["parity_subset_ids_sha256"]):
        raise ValueError(f"{name} evaluator parity subset differs from frozen panel")
    return report


def _load_run_receipt(payload, expected_sha256, expected_slot, expected):
    if not _valid_sha(expected_sha256) or _sha(payload) != expected_sha256:
        raise ValueError(f"{expected_slot} run-receipt hash mismatch")
    receipt = validate_causal_run_receipt(payload)
    if expected_slot in {"slot_a", "slot_b"} and receipt["checkpoint_step"] != 400:
        raise ValueError(f"{expected_slot} run receipt is not the frozen step-400 checkpoint")
    checks = {"slot": expected_slot, "train_sha256": expected["train_sha256"],
              "recipe_sha256": expected["recipe_sha256"],
              "parent_adapter_sha256": expected["parent_adapter_sha256"],
              "model_revision": expected["model_revision"],
              "model_inventory_sha256": expected["model_inventory_sha256"]}
    failures = [key for key, value in checks.items() if receipt.get(key) != value]
    if failures:
        raise ValueError(f"{expected_slot} run receipt provenance mismatch: {failures}")
    return receipt


def compare_payloads(*, slot_a_predictions, slot_b_predictions, reference_predictions,
                     slot_a_report, slot_b_report, reference_report, panel_payload,
                     panel_manifest_payload, assignment_payload, slot_a_run_receipt,
                     slot_b_run_receipt, reference_run_receipt, slot_a_report_sha256,
                     slot_b_report_sha256, reference_report_sha256,
                     slot_a_run_receipt_sha256, slot_b_run_receipt_sha256,
                     reference_run_receipt_sha256, panel_manifest_sha256, assignment_sha256,
                     replicates=MIN_BOOTSTRAP_REPLICATES, seed=470047,
                     _require_frozen_panel=True):
    """Compare preassigned slots only after binding panel, manifests, reports, and provenance."""
    if _sha(assignment_payload) != assignment_sha256:
        raise ValueError("frozen slot-assignment hash mismatch")
    try:
        assignment = json.loads(assignment_payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid frozen slot assignment") from error
    if set(assignment) != ASSIGNMENT_FIELDS:
        raise ValueError("slot assignment schema differs from the frozen contract")
    if (assignment["schema_version"] != "r047-causal-slot-assignment-v1"
            or {assignment["control_slot"], assignment["treatment_slot"]}
            != {"slot_a", "slot_b"}):
        raise ValueError("invalid pre-frozen control/treatment slot assignment")
    hash_fields = ASSIGNMENT_FIELDS - {
        "schema_version", "control_slot", "treatment_slot", "model_revision"}
    if any(not _valid_sha(assignment[field]) for field in hash_fields):
        raise ValueError("slot assignment contains an invalid SHA-256")
    if not isinstance(assignment["model_revision"], str) or not assignment["model_revision"]:
        raise ValueError("slot assignment model revision is invalid")
    if assignment["reference_adapter_sha256"] != assignment["parent_adapter_sha256"]:
        raise ValueError("frozen reference adapter must be the exact parent adapter")

    if (_sha(panel_manifest_payload) != panel_manifest_sha256
            or assignment["panel_manifest_sha256"] != panel_manifest_sha256):
        raise ValueError("frozen panel-manifest hash mismatch")
    if _sha(panel_payload) != assignment["panel_sha256"]:
        raise ValueError("raw panel hash differs from frozen slot assignment")
    panel_rows, panel_manifest = validate_causal_panel_payload(
        panel_payload, panel_manifest_payload, require_frozen=_require_frozen_panel)
    expected_clusters = panel_manifest["clusters"]["causal_primary"]
    expected_rows = panel_manifest["outputs"]["primary"]["rows"]
    parity_rows = causal_parity_subset(panel_rows)
    parity_subset_ids_sha256 = _sha(
        ("\n".join(row["id"] for row in parity_rows) + "\n").encode())

    slot_records = {"slot_a": _read_payload(slot_a_predictions),
                    "slot_b": _read_payload(slot_b_predictions)}
    reference = _read_payload(reference_predictions)
    for records in (*slot_records.values(), reference):
        _bind_predictions_to_panel(records, panel_rows)
        validate_panel_coverage(records)
    _assert_matching_structure(slot_records["slot_a"], slot_records["slot_b"])
    _assert_matching_structure(slot_records["slot_a"], reference)

    common = {"model_revision": assignment["model_revision"],
              "recipe_sha256": assignment["recipe_sha256"],
              "parent_adapter_sha256": assignment["parent_adapter_sha256"],
              "model_inventory_sha256": assignment["model_inventory_sha256"],
              "parity_rows": len(parity_rows),
              "parity_subset_ids_sha256": parity_subset_ids_sha256,
              "rows": expected_rows, "clusters": expected_clusters}
    slot_expectations = {}
    for role in ("control", "treatment"):
        slot = assignment[f"{role}_slot"]
        slot_expectations[slot] = {
            **common, "train_sha256": assignment[f"{role}_train_sha256"]}
    receipt_inputs = {"slot_a": (slot_a_run_receipt, slot_a_run_receipt_sha256),
                      "slot_b": (slot_b_run_receipt, slot_b_run_receipt_sha256)}
    receipts = {}
    for slot in ("slot_a", "slot_b"):
        receipts[slot] = _load_run_receipt(
            *receipt_inputs[slot], slot, slot_expectations[slot])
        slot_expectations[slot].update(
            adapter_sha256=receipts[slot]["adapter_sha256"],
            run_receipt_sha256=receipt_inputs[slot][1])
    reference_expectation = {
        **common, "train_sha256": assignment["reference_train_sha256"]}
    receipts["reference"] = _load_run_receipt(
        reference_run_receipt, reference_run_receipt_sha256, "reference",
        reference_expectation)
    if receipts["reference"]["adapter_sha256"] != assignment["reference_adapter_sha256"]:
        raise ValueError("reference run receipt adapter differs from frozen parent")
    reference_expectation.update(
        adapter_sha256=receipts["reference"]["adapter_sha256"],
        run_receipt_sha256=reference_run_receipt_sha256)
    reports = {
        "slot_a": _load_evaluator_report(
            slot_a_report, slot_a_report_sha256, "slot_a", slot_a_predictions,
            assignment["panel_sha256"], panel_manifest_sha256, slot_expectations["slot_a"]),
        "slot_b": _load_evaluator_report(
            slot_b_report, slot_b_report_sha256, "slot_b", slot_b_predictions,
            assignment["panel_sha256"], panel_manifest_sha256, slot_expectations["slot_b"]),
        "reference": _load_evaluator_report(
            reference_report, reference_report_sha256, "reference", reference_predictions,
            assignment["panel_sha256"], panel_manifest_sha256,
            reference_expectation),
    }
    parity_signatures = {(report["batch_scalar_parity"]["rows"],
                          report["batch_scalar_parity"]["subset_ids_sha256"])
                         for report in reports.values()}
    if len(parity_signatures) != 1:
        raise ValueError("evaluator reports used different parity subsets")
    model_tree_signatures = {(report["model_tree_sha256"], report["model_tree_files"])
                             for report in reports.values()}
    if len(model_tree_signatures) != 1:
        raise ValueError("evaluator reports used different base-model trees")
    control = slot_records[assignment["control_slot"]]
    treatment = slot_records[assignment["treatment_slot"]]

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
        "schema_version": "r047-causal-comparison-v2",
        "blinding": {"assignment_sha256": assignment_sha256,
                     "control_slot": assignment["control_slot"],
                     "treatment_slot": assignment["treatment_slot"],
                     "pre_frozen_slot_assignment": True},
        "provenance": {"assignment": assignment,
                       "panel_manifest_sha256": panel_manifest_sha256,
                       "evaluator_report_sha256": {
                           "slot_a": slot_a_report_sha256, "slot_b": slot_b_report_sha256,
                           "reference": reference_report_sha256},
                       "run_receipt_sha256": {
                           "slot_a": slot_a_run_receipt_sha256,
                           "slot_b": slot_b_run_receipt_sha256,
                           "reference": reference_run_receipt_sha256},
                       "prediction_sha256": {
                           "slot_a": reports["slot_a"]["predictions_sha256"],
                           "slot_b": reports["slot_b"]["predictions_sha256"],
                           "reference": reports["reference"]["predictions_sha256"]}},
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
    parser.add_argument("--slot-a-predictions", required=True)
    parser.add_argument("--slot-b-predictions", required=True)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--slot-a-report", required=True)
    parser.add_argument("--slot-b-report", required=True)
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--slot-a-report-sha256", required=True)
    parser.add_argument("--slot-b-report-sha256", required=True)
    parser.add_argument("--reference-report-sha256", required=True)
    parser.add_argument("--slot-a-run-receipt", required=True)
    parser.add_argument("--slot-b-run-receipt", required=True)
    parser.add_argument("--reference-run-receipt", required=True)
    parser.add_argument("--slot-a-run-receipt-sha256", required=True)
    parser.add_argument("--slot-b-run-receipt-sha256", required=True)
    parser.add_argument("--reference-run-receipt-sha256", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--panel-manifest-sha256", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=MIN_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=470047)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = compare_payloads(
        slot_a_predictions=Path(args.slot_a_predictions).read_bytes(),
        slot_b_predictions=Path(args.slot_b_predictions).read_bytes(),
        reference_predictions=Path(args.reference_predictions).read_bytes(),
        slot_a_report=Path(args.slot_a_report).read_bytes(),
        slot_b_report=Path(args.slot_b_report).read_bytes(),
        reference_report=Path(args.reference_report).read_bytes(),
        panel_payload=Path(args.panel).read_bytes(),
        panel_manifest_payload=Path(args.panel_manifest).read_bytes(),
        assignment_payload=Path(args.assignment).read_bytes(),
        slot_a_run_receipt=Path(args.slot_a_run_receipt).read_bytes(),
        slot_b_run_receipt=Path(args.slot_b_run_receipt).read_bytes(),
        reference_run_receipt=Path(args.reference_run_receipt).read_bytes(),
        slot_a_report_sha256=args.slot_a_report_sha256,
        slot_b_report_sha256=args.slot_b_report_sha256,
        reference_report_sha256=args.reference_report_sha256,
        slot_a_run_receipt_sha256=args.slot_a_run_receipt_sha256,
        slot_b_run_receipt_sha256=args.slot_b_run_receipt_sha256,
        reference_run_receipt_sha256=args.reference_run_receipt_sha256,
        panel_manifest_sha256=args.panel_manifest_sha256,
        assignment_sha256=args.assignment_sha256,
        replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
