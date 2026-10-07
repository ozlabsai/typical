"""Fail-closed paired comparison for R049 calibrated-decision Stage-1 arms.

The comparator recomputes exact-policy outcomes and proper losses from field
probabilities. Rendered rows are descriptive; inference resamples canonical
``semantic_signature`` units so translated, reordered, counterfactual, and
repeated latent-case views are not treated as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.build_r048_action_panel as r048_builder
from scripts.score_exact_policy_panel import NLL_PROBABILITY_FLOOR, classify


SCHEMA = "r049_calibrated_comparison_v3"
DEFAULT_BOOTSTRAP_SEED = 490_049
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
MIN_BOOTSTRAP_REPLICATES = 100
METRICS = (
    "complete_outcome_accuracy",
    "categorical_brier",
    "complete_outcome_nll",
    "field_accuracy",
)
SLICE_KEYS = (
    "language",
    "option_order",
    "policy_mode",
    "topology_slice",
    "policy_topology",
)
FAMILY_KEYS = ("program_family", "action_family")
INFERENCE_KEY = "semantic_signature"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return math.fsum(values) / len(values)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_id_digest(ids: Iterable[str]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":")).encode()
    return _sha256(payload)


def _checked_hash(value: str, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"{label} must be a 64-character hexadecimal sha256")
    return value.lower()


def _gold_probabilities(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    questions, targets = row.get("questions"), row.get("targets")
    if not isinstance(questions, dict) or not isinstance(targets, dict):
        raise ValueError("questions and targets must be objects")
    probabilities = {}
    for qid in questions:
        target = targets.get(qid)
        label = target.get("label") if isinstance(target, dict) else None
        if label not in {"no", "yes"}:
            raise ValueError(f"{qid}: target needs a semantic no/yes label")
        probabilities[qid] = {
            "no": float(label == "no"),
            "yes": float(label == "yes"),
        }
    return probabilities


def _slice_value(row: dict[str, Any], key: str) -> str | None:
    meta = row["meta"]
    if key == "policy_mode":
        value = meta.get(key)
        policy = meta.get("policy")
        executable = policy.get("mode") if isinstance(policy, dict) else None
        if value is not None and executable is not None and value != executable:
            raise ValueError("policy_mode conflicts with executable policy")
        value = value if value is not None else executable
    else:
        value = meta.get(key)
    if value is None:
        return None
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{key} slice value must be scalar")
    return str(value)


def _validate_panel(rows: list[dict[str, Any]]) -> list[str]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("panel must be a nonempty list")
    versions = {
        row.get("meta", {}).get("panel_version")
        for row in rows if isinstance(row, dict) and isinstance(row.get("meta"), dict)
    }
    versions.discard(None)
    if versions:
        if versions != {r048_builder.PANEL_VERSION}:
            raise ValueError("panel versions must match the canonical R048 version")
        splits = {row.get("meta", {}).get("split") for row in rows}
        if len(splits) != 1 or None in splits:
            raise ValueError("canonical R048 comparison requires exactly one split")
        r048_builder.validate_split(rows, str(next(iter(splits))), len(rows))
    identifiers = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"panel row {index} must be an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"panel row {index} needs a nonempty string id")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("panel row ids must be unique")

    semantic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta")
        cluster = meta.get("semantic_group_id") if isinstance(meta, dict) else None
        if not isinstance(cluster, str) or not cluster:
            raise ValueError(
                f"{row['id']}: a nonempty semantic_group_id is required"
            )
        signature = meta.get(INFERENCE_KEY)
        if not isinstance(signature, str) or not signature:
            raise ValueError(
                f"{row['id']}: a nonempty {INFERENCE_KEY} is required for inference"
            )
        if ("counterfactual_side" not in meta
                or isinstance(meta["counterfactual_side"], bool)
                or not isinstance(meta["counterfactual_side"], int)):
            raise ValueError(
                f"{row['id']}: an integer counterfactual_side is required"
            )
        for key in ("policy_topology", "topology_slice"):
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f"{row['id']}: a nonempty {key} is required")
        for key in FAMILY_KEYS:
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f"{row['id']}: a nonempty {key} is required")
        try:
            result = classify(row, _gold_probabilities(row))
            if result["predicted_outcome"] != result["gold_outcome"]:
                raise ValueError("gold targets do not reproduce the gold outcome")
            for key in (*SLICE_KEYS, *FAMILY_KEYS):
                _slice_value(row, key)
        except ValueError as error:
            raise ValueError(f"{row['id']}: invalid panel contract: {error}") from error
        semantic_groups[cluster].append(row)

    for group_id, group in semantic_groups.items():
        invariant_keys = (
            INFERENCE_KEY,
            "field_order",
            "qid_to_field",
            "decision_threshold",
            "policy",
            "policy_mode",
            "policy_topology",
            "topology_slice",
            "program_family",
            "action_family",
        )
        inconsistent = [
            key for key in invariant_keys
            if len({
                json.dumps(row["meta"].get(key), ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
                for row in group
            }) != 1
        ]
        if inconsistent:
            raise ValueError(
                f"{group_id}: semantic-group invariants disagree: "
                + ", ".join(inconsistent)
            )
        variant_keys = [
            (row["meta"]["counterfactual_side"], row["meta"].get("language"),
             row["meta"].get("option_order"))
            for row in group
        ]
        if len(variant_keys) != len(set(variant_keys)):
            raise ValueError(
                f"{group_id}: duplicate counterfactual/language/option-order variant"
            )
        for side in {row["meta"]["counterfactual_side"] for row in group}:
            side_rows = [row for row in group
                         if row["meta"]["counterfactual_side"] == side]
            for key in ("action_trace", "gold_outcome", "facts"):
                values = {
                    json.dumps(row["meta"].get(key), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
                    for row in side_rows
                }
                if len(values) != 1:
                    raise ValueError(
                        f"{group_id}: side {side} has inconsistent {key}"
                    )
            targets = {
                json.dumps(row["targets"], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
                for row in side_rows
            }
            if len(targets) != 1:
                raise ValueError(
                    f"{group_id}: side {side} has inconsistent semantic targets"
                )
    return identifiers


def _index_predictions(
    prediction_rows: list[dict[str, Any]],
    label: str,
    panel_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(prediction_rows, list):
        raise ValueError(f"{label} predictions must be a list")
    indexed = {}
    for index, prediction in enumerate(prediction_rows):
        if not isinstance(prediction, dict):
            raise ValueError(f"{label} prediction {index} must be an object")
        identifier = prediction.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{label} prediction {index} needs a nonempty string id")
        if identifier in indexed:
            raise ValueError(f"duplicate {label} prediction id: {identifier}")
        indexed[identifier] = prediction
    unexpected = set(indexed) - panel_ids
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:5])
        raise ValueError(f"unexpected {label} prediction ids: {preview}")
    return indexed


def _invalid_record(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "row": row,
        "valid": False,
        "invalid_reason": reason,
        "complete_outcome_accuracy": 0.0,
        "categorical_brier": 2.0,
        "complete_outcome_nll": -math.log(NLL_PROBABILITY_FLOOR),
        "field_accuracy": 0.0,
    }


def _score_arm(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        prediction = predictions.get(row["id"])
        if prediction is None:
            records.append(_invalid_record(row, "missing_prediction"))
            continue
        try:
            result = classify(row, prediction.get("probabilities"))
        except (ValueError, OverflowError, TypeError):
            records.append(_invalid_record(row, "invalid_prediction"))
            continue
        gold_trace = result["gold_trace"]
        correct_fields = sum(
            result["predicted_trace"][field] == truth
            for field, truth in gold_trace.items()
        )
        records.append({
            "row": row,
            "valid": True,
            "invalid_reason": None,
            "predicted_outcome_key": json.dumps(
                result["predicted_outcome"].as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "complete_outcome_accuracy": float(
                result["predicted_outcome"] == result["gold_outcome"]
            ),
            "categorical_brier": result["categorical_brier"],
            "complete_outcome_nll": result["complete_outcome_nll"],
            "field_accuracy": correct_fields / len(gold_trace),
        })
    return records


def _metric_means(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("metric summary needs at least one record")
    return {metric: _mean(record[metric] for record in records) for metric in METRICS}


def _by_cluster(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["row"]["meta"][INFERENCE_KEY]].append(record)
    return {
        cluster: _metric_means(items) for cluster, items in sorted(grouped.items())
    }


def _cluster_means(cluster_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: _mean(values[metric] for values in cluster_metrics.values())
        for metric in METRICS
    }


def _arm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    clusters = _by_cluster(records)
    return {
        "rendered": {
            **_metric_means(records),
            "n": len(records),
            "valid": sum(record["valid"] for record in records),
            "invalid": sum(not record["valid"] for record in records),
        },
        "equal_semantic_cluster": {
            **_cluster_means(clusters),
            "n": len(clusters),
        },
    }


def _paired_delta(
    control: dict[str, float], candidate: dict[str, float]
) -> dict[str, float | None]:
    absolute_brier = control["categorical_brier"] - candidate["categorical_brier"]
    return {
        "complete_outcome_accuracy": (
            candidate["complete_outcome_accuracy"]
            - control["complete_outcome_accuracy"]
        ),
        "categorical_brier_absolute_improvement": absolute_brier,
        "categorical_brier_relative_improvement": (
            absolute_brier / control["categorical_brier"]
            if control["categorical_brier"] > 0 else None
        ),
        "complete_outcome_nll_degradation": (
            candidate["complete_outcome_nll"] - control["complete_outcome_nll"]
        ),
        "field_accuracy": candidate["field_accuracy"] - control["field_accuracy"],
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    control_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if (isinstance(replicates, bool) or not isinstance(replicates, int)
            or replicates < MIN_BOOTSTRAP_REPLICATES):
        raise ValueError(
            f"bootstrap_replicates must be an integer >= {MIN_BOOTSTRAP_REPLICATES}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    control = _by_cluster(control_records)
    candidate = _by_cluster(candidate_records)
    if set(control) != set(candidate) or not control:
        raise ValueError("paired bootstrap requires identical nonempty semantic clusters")
    clusters = sorted(control)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    undefined_relative = 0
    for _ in range(replicates):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        control_mean = {
            metric: _mean(control[cluster][metric] for cluster in selected)
            for metric in METRICS
        }
        candidate_mean = {
            metric: _mean(candidate[cluster][metric] for cluster in selected)
            for metric in METRICS
        }
        delta = _paired_delta(control_mean, candidate_mean)
        for metric, value in delta.items():
            if value is None:
                if metric == "categorical_brier_relative_improvement":
                    undefined_relative += 1
                continue
            samples[metric].append(value)

    point = _paired_delta(_cluster_means(control), _cluster_means(candidate))
    report = {}
    for metric, value in point.items():
        values = samples[metric]
        complete = len(values) == replicates
        report[metric] = {
            "point_estimate": value,
            "lower_95": _percentile(values, 0.025) if complete else None,
            "upper_95": _percentile(values, 0.975) if complete else None,
            "replicates": replicates,
            "valid_replicates": len(values),
            "seed": seed,
            "resampling_unit": INFERENCE_KEY,
            "cluster_weighting": "equal",
            "interval": "paired percentile 95%",
        }
    report["categorical_brier_relative_improvement"][
        "undefined_zero_control_replicates"
    ] = undefined_relative
    return report


def _comparison(
    control_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if len(control_records) != len(candidate_records) or not control_records:
        raise ValueError("paired comparison requires equal nonempty arms")
    control_summary = _arm_summary(control_records)
    candidate_summary = _arm_summary(candidate_records)
    return {
        "n_rendered": len(control_records),
        "n_semantic_clusters": len(_by_cluster(control_records)),
        "control": control_summary,
        "candidate": candidate_summary,
        "paired_delta": {
            "rendered": _paired_delta(
                control_summary["rendered"], candidate_summary["rendered"]
            ),
            "equal_semantic_cluster": _paired_delta(
                control_summary["equal_semantic_cluster"],
                candidate_summary["equal_semantic_cluster"],
            ),
        },
        "paired_cluster_bootstrap": _bootstrap(
            control_records,
            candidate_records,
            replicates=replicates,
            seed=seed,
        ),
    }


def _filter_records(
    records: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    return [record for record in records if predicate(record["row"])]


def _slice_report(
    rows: list[dict[str, Any]],
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    key: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = sorted({value for row in rows if (value := _slice_value(row, key)) is not None})
    report = {}
    for value in values:
        predicate = lambda row, expected=value: _slice_value(row, key) == expected
        report[value] = _comparison(
            _filter_records(control, predicate),
            _filter_records(candidate, predicate),
            replicates=replicates,
            seed=seed,
        )
    return report


def _coverage(
    panel_ids: list[str],
    predictions: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    missing = set(panel_ids) - set(predictions)
    invalid = [record for record in records if not record["valid"]]
    return {
        "expected": len(panel_ids),
        "provided": len(predictions),
        "exact_id_coverage": not missing and len(predictions) == len(panel_ids),
        "missing": len(missing),
        "missing_ids_sha256": _canonical_id_digest(missing),
        "missing_ids_preview": sorted(missing)[:20],
        "invalid": len(invalid),
        "invalid_ids_sha256": _canonical_id_digest(
            record["row"]["id"] for record in invalid
        ),
        "valid": len(records) - len(invalid),
    }


def _family_nonregression(slices: dict[str, Any]) -> dict[str, Any]:
    report = {}
    for key in FAMILY_KEYS:
        values = slices[key]
        if not values:
            report[key] = {
                "status": "not_evaluated",
                "reason": f"{key} is absent from panel metadata",
                "threshold": -0.02,
                "passed": None,
                "worst_delta": None,
            }
            continue
        deltas = {
            value: comparison["paired_delta"]["equal_semantic_cluster"][
                "complete_outcome_accuracy"
            ]
            for value, comparison in values.items()
        }
        worst_value = min(deltas, key=lambda value: (deltas[value], value))
        report[key] = {
            "status": "evaluated",
            "metric": "equal-cluster complete-outcome accuracy candidate-control",
            "threshold": -0.02,
            "passed": deltas[worst_value] >= -0.02,
            "worst_delta": deltas[worst_value],
            "worst_slice": worst_value,
            "deltas": deltas,
        }
    return report


def _option_order_robustness(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure semantic agreement across option-order twins, fail closed."""
    grouped: dict[tuple[str, str, int, str], dict[str, str | None]] = defaultdict(dict)
    for record in records:
        meta = record["row"]["meta"]
        key = (
            meta[INFERENCE_KEY],
            meta["semantic_group_id"],
            meta["counterfactual_side"],
            meta["language"],
        )
        grouped[key][str(meta["option_order"])] = record.get("predicted_outcome_key")
    complete = {
        key: values for key, values in grouped.items()
        if set(values) == {"no_yes", "yes_no"}
        and all(value is not None for value in values.values())
    }
    agreement_by_signature: dict[str, list[bool]] = defaultdict(list)
    for key, values in complete.items():
        agreement_by_signature[key[0]].append(
            values["no_yes"] == values["yes_no"]
        )
    agreement = (
        _mean(_mean(matches) for matches in agreement_by_signature.values())
        if agreement_by_signature else None
    )
    by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_order[str(record["row"]["meta"]["option_order"])].append(record)
    accuracies = {
        order: _cluster_means(_by_cluster(items))["complete_outcome_accuracy"]
        for order, items in sorted(by_order.items())
    }
    spread = (
        max(accuracies.values()) - min(accuracies.values())
        if len(accuracies) >= 2 else None
    )
    return {
        "expected_pairs": len(grouped),
        "complete_pairs": len(complete),
        "semantic_agreement": agreement,
        "semantic_agreement_unit": INFERENCE_KEY,
        "semantic_signatures": len(agreement_by_signature),
        "order_accuracy": accuracies,
        "order_accuracy_spread": spread,
    }


def _heldout_topology_gate(slices: dict[str, Any]) -> dict[str, Any]:
    heldout = {
        name: report for name, report in slices["topology_slice"].items()
        if name.casefold().startswith("heldout")
    }
    if not heldout:
        return {
            "passed": False,
            "reason": "no held-out topology slice was evaluated",
            "slices": {},
        }
    checks = {}
    for name, report in heldout.items():
        delta = report["paired_delta"]["equal_semantic_cluster"][
            "complete_outcome_accuracy"
        ]
        lower = report["paired_cluster_bootstrap"][
            "complete_outcome_accuracy"
        ]["lower_95"]
        checks[name] = {
            "accuracy_delta": delta,
            "accuracy_lower_95": lower,
            "passed": delta >= 0.03 and lower is not None and lower > 0,
        }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "requirements": "accuracy delta >= 0.03 and paired 95% lower bound > 0",
        "slices": checks,
    }


def _internal_gate(
    overall: dict[str, Any],
    coverage: dict[str, Any],
    families: dict[str, Any],
    slices: dict[str, Any],
    option_order: dict[str, Any],
) -> dict[str, Any]:
    delta = overall["paired_delta"]["equal_semantic_cluster"]
    bootstrap = overall["paired_cluster_bootstrap"]
    accuracy_ci = bootstrap["complete_outcome_accuracy"]
    brier_ci = bootstrap["categorical_brier_relative_improvement"]
    nll_ci = bootstrap["complete_outcome_nll_degradation"]
    field_ci = bootstrap["field_accuracy"]

    family_evaluated = [item for item in families.values()
                        if item["status"] == "evaluated"]
    family_passed = (
        len(family_evaluated) == len(FAMILY_KEYS)
        and all(item["passed"] for item in family_evaluated)
    )
    heldout = _heldout_topology_gate(slices)
    zero_invalid = all(
        coverage[arm]["invalid"] == 0 and coverage[arm]["exact_id_coverage"]
        for arm in ("control", "candidate")
    )
    requirements = {
        "complete_outcome_accuracy_point": {
            "passed": delta["complete_outcome_accuracy"] >= 0.03,
            "observed": delta["complete_outcome_accuracy"],
            "operator": ">=", "threshold": 0.03,
        },
        "complete_outcome_accuracy_lcb": {
            "passed": accuracy_ci["lower_95"] is not None
            and accuracy_ci["lower_95"] > 0,
            "observed": accuracy_ci["lower_95"],
            "operator": ">", "threshold": 0.0,
        },
        "categorical_brier_relative_improvement": {
            "passed": delta["categorical_brier_relative_improvement"] is not None
            and delta["categorical_brier_relative_improvement"] >= 0.05,
            "observed": delta["categorical_brier_relative_improvement"],
            "operator": ">=", "threshold": 0.05,
        },
        "categorical_brier_ci_excludes_zero": {
            "passed": brier_ci["lower_95"] is not None
            and brier_ci["lower_95"] > 0,
            "observed": brier_ci["lower_95"],
            "operator": ">", "threshold": 0.0,
        },
        "nll_degradation": {
            "passed": nll_ci["upper_95"] is not None
            and nll_ci["upper_95"] <= 0.02,
            "observed": nll_ci["upper_95"],
            "operator": "<=", "threshold": 0.02,
        },
        "field_accuracy_point": {
            "passed": delta["field_accuracy"] >= -0.01,
            "observed": delta["field_accuracy"],
            "operator": ">=", "threshold": -0.01,
        },
        "field_accuracy_lcb": {
            "passed": field_ci["lower_95"] is not None
            and field_ci["lower_95"] >= -0.02,
            "observed": field_ci["lower_95"],
            "operator": ">=", "threshold": -0.02,
        },
        "family_nonregression": {
            "passed": family_passed,
            "observed": {
                key: value["worst_delta"] for key, value in families.items()
            },
            "operator": ">= where available", "threshold": -0.02,
            "not_evaluated_families": [
                key for key, value in families.items()
                if value["status"] == "not_evaluated"
            ],
        },
        "heldout_topology_transfer": {
            "passed": heldout["passed"],
            "observed": heldout,
            "operator": "all held-out slices pass",
            "threshold": {"accuracy_delta": 0.03, "accuracy_lower_95": 0.0},
        },
        "option_permutation_agreement": {
            "passed": option_order["semantic_agreement"] is not None
            and option_order["complete_pairs"] == option_order["expected_pairs"]
            and option_order["semantic_agreement"] >= 0.99,
            "observed": option_order,
            "operator": ">= with complete pairs", "threshold": 0.99,
        },
        "option_order_accuracy_spread": {
            "passed": option_order["order_accuracy_spread"] is not None
            and option_order["order_accuracy_spread"] <= 0.01,
            "observed": option_order["order_accuracy_spread"],
            "operator": "<=", "threshold": 0.01,
        },
        "zero_invalid_predictions": {
            "passed": zero_invalid,
            "observed": {
                arm: {
                    "invalid": coverage[arm]["invalid"],
                    "exact_id_coverage": coverage[arm]["exact_id_coverage"],
                }
                for arm in ("control", "candidate")
            },
            "operator": "==", "threshold": 0,
        },
    }
    failures = [name for name, item in requirements.items() if not item["passed"]]
    return {
        "passed": not failures,
        "scope": "internal exact-policy Stage-1 screen only",
        "requirements": requirements,
        "failures": failures,
    }


def compare_rows(
    panel_rows: list[dict[str, Any]],
    control_prediction_rows: list[dict[str, Any]],
    candidate_prediction_rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two prediction sets with fail-closed, cluster-aware metrics."""
    panel_ids = _validate_panel(panel_rows)
    panel_id_set = set(panel_ids)
    control_predictions = _index_predictions(
        control_prediction_rows, "control", panel_id_set
    )
    candidate_predictions = _index_predictions(
        candidate_prediction_rows, "candidate", panel_id_set
    )
    control = _score_arm(panel_rows, control_predictions)
    candidate = _score_arm(panel_rows, candidate_predictions)
    coverage = {
        "control": _coverage(panel_ids, control_predictions, control),
        "candidate": _coverage(panel_ids, candidate_predictions, candidate),
    }
    overall = _comparison(
        control,
        candidate,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    slices = {
        key: _slice_report(
            panel_rows,
            control,
            candidate,
            key,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        for key in (*SLICE_KEYS, *FAMILY_KEYS)
    }
    families = _family_nonregression(slices)
    option_order = _option_order_robustness(candidate)
    gate = _internal_gate(overall, coverage, families, slices, option_order)
    return {
        "schema": SCHEMA,
        "provenance": provenance or {},
        "contract": {
            "invalid_prediction_policy": (
                "zero outcome/field accuracy, categorical Brier 2, and gold "
                "outcome probability floored at 1e-12 for NLL"
            ),
            "point_estimate_unit": f"equal-weight {INFERENCE_KEY}",
            "rendered_metrics_role": "descriptive_only",
            "brier_improvement_direction": "control_minus_candidate",
            "nll_degradation_direction": "candidate_minus_control",
            "bootstrap": {
                "method": "paired semantic-cluster percentile",
                "seed": bootstrap_seed,
                "replicates": bootstrap_replicates,
            },
        },
        "coverage": coverage,
        "overall": overall,
        "slices": slices,
        "family_nonregression": families,
        "option_order_robustness": option_order,
        "internal_stage1_gate": gate,
        "external_gates": {
            "retention": {
                "status": "not_evaluated",
                "reason": "requires the frozen Hebrew/multimodal retention evaluator",
            },
            "workflowevals": {
                "status": "not_evaluated",
                "reason": "requires a separate official WorkflowEvals run",
            },
        },
        "promotion": {
            "status": "not_evaluated",
            "internal_stage1_gate_passed": gate["passed"],
            "reason": "retention and WorkflowEvals gates are external to this comparator",
        },
    }


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label} JSONL: {path}") from error
    rows = []
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} JSONL line {line_number} is invalid") from error
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} JSONL is empty")
    return rows, payload


def _verify_manifest(
    manifest_path: Path,
    panel_path: Path,
    panel_payload: bytes,
    panel_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read a valid panel manifest") from error
    if (not isinstance(manifest, dict)
            or manifest.get("panel") != "exact_policy_v1"
            or manifest.get("panel_version") != r048_builder.PANEL_VERSION):
        raise ValueError(
            f"panel manifest is not canonical R048 {r048_builder.PANEL_VERSION}"
        )
    expected_generator = _sha256(Path(r048_builder.__file__).read_bytes())
    if manifest.get("generator_sha256") != expected_generator:
        raise ValueError("panel manifest generator differs from current canonical builder")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("panel manifest needs an outputs object")
    matches = [outputs[key] for key in (panel_path.name, panel_path.stem) if key in outputs]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError("panel manifest must identify the panel by filename or stem")
    expectation = matches[0]
    expected_sha = _checked_hash(expectation.get("sha256"), "manifest panel sha256")
    observed_sha = _sha256(panel_payload)
    if observed_sha != expected_sha:
        raise ValueError("panel sha256 differs from panel manifest")
    if expectation.get("rows") != len(panel_rows):
        raise ValueError("panel row count differs from panel manifest")
    return {
        "sha256": _sha256(payload),
        "verified": True,
        "panel_sha256": expected_sha,
        "panel_rows": len(panel_rows),
    }


def compare_files(
    panel_path: str | Path,
    control_path: str | Path,
    candidate_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected_hashes: dict[str, str] | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Read immutable inputs, verify optional expectations, and compare arms."""
    paths = {
        "panel": Path(panel_path),
        "control": Path(control_path),
        "candidate": Path(candidate_path),
    }
    loaded = {name: _read_jsonl(path, name) for name, path in paths.items()}
    hashes = {name: _sha256(payload) for name, (_, payload) in loaded.items()}
    expected_hashes = expected_hashes or {}
    unexpected_expectations = set(expected_hashes) - set(paths)
    if unexpected_expectations:
        raise ValueError("expected_hashes contains unsupported input names")
    for name, expected in expected_hashes.items():
        expected = _checked_hash(expected, f"{name} expected sha256")
        if hashes[name] != expected:
            raise ValueError(f"{name} sha256 mismatch")
    if manifest_path is None:
        raise ValueError("a canonical R048 panel manifest is required")
    provenance: dict[str, Any] = {
        "inputs": {
            name: {
                "sha256": hashes[name],
                "bytes": len(loaded[name][1]),
                "rows": len(loaded[name][0]),
                "expected_sha256_verified": name in expected_hashes,
            }
            for name in paths
        },
        "manifest": None,
    }
    provenance["manifest"] = _verify_manifest(
        Path(manifest_path), paths["panel"], loaded["panel"][1], loaded["panel"][0]
    )
    panel_rows = loaded["panel"][0]
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("meta"), dict)
        or row["meta"].get("panel_version") != r048_builder.PANEL_VERSION
        for row in panel_rows
    ):
        raise ValueError("file comparison requires canonical R048 panel row versions")
    return compare_rows(
        panel_rows,
        loaded["control"][0],
        loaded["candidate"][0],
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        provenance=provenance,
    )


def write_report_atomic(report: dict[str, Any], output_path: str | Path) -> str:
    """Publish deterministic JSON with an atomic no-overwrite hard-link."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison report: {output}")
    payload = (json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite comparison report: {output}"
            ) from error
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-sha256")
    parser.add_argument("--control-sha256")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    args = parser.parse_args(argv)
    expected_hashes = {
        name: value for name, value in {
            "panel": args.panel_sha256,
            "control": args.control_sha256,
            "candidate": args.candidate_sha256,
        }.items() if value is not None
    }
    report = compare_files(
        args.panel,
        args.control_predictions,
        args.candidate_predictions,
        manifest_path=args.manifest,
        expected_hashes=expected_hashes,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    digest = write_report_atomic(report, args.out)
    print(json.dumps({
        "output_sha256": digest,
        "internal_stage1_gate_passed": report["internal_stage1_gate"]["passed"],
        "promotion_status": report["promotion"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
