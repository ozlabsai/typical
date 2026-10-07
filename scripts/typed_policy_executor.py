"""Typed decision fields for the existing exact Boolean policy executor.

Each Choice, Noul, or Score field declares a nonempty proper subset of its
semantic labels as the Boolean event consumed by ``exact_policy_executor``.
This keeps policy execution unchanged while allowing the field readout to stay
categorical and order-independent.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from scripts.exact_policy_executor import (
    canonicalize_outcome,
    compile_policy,
    execute_policy,
)


PANEL = "typed_policy_v1"
QUESTION_TYPES = frozenset(("choice", "noul", "score"))
OPTION_ORDERS = frozenset(("canonical", "reversed"))


def _labels(question: Mapping[str, Any], identifier: str) -> tuple[str, ...]:
    criteria = question.get("criteria")
    if (not isinstance(criteria, Mapping) or len(criteria) < 2
            or any(not isinstance(label, str) or not label for label in criteria)
            or any(not isinstance(description, str) or not description.strip()
                   for description in criteria.values())):
        raise ValueError(
            f"{identifier}: typed criteria must map at least two nonempty labels to descriptions")
    return tuple(criteria)


def validate_typed_policy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one typed-policy presentation and return its compiled contract."""
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("typed-policy id must be a nonempty string")
    if not isinstance(row.get("state"), str) or not row["state"].strip():
        raise ValueError(f"{identifier}: typed-policy state must be nonempty text")
    if row.get("image") is not None or "paired_views" in row or "auxiliary_views" in row:
        raise ValueError(f"{identifier}: typed-policy groups must be unwrapped text-only rows")

    meta = row.get("meta")
    if not isinstance(meta, Mapping) or meta.get("panel") != PANEL:
        raise ValueError(f"{identifier}: malformed {PANEL} metadata")
    group_id = meta.get("semantic_group_id")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError(f"{identifier}: missing semantic_group_id")
    if meta.get("option_order") not in OPTION_ORDERS:
        raise ValueError(f"{identifier}: option_order must be canonical or reversed")

    questions, targets = row.get("questions"), row.get("targets")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError(f"{identifier}: typed-policy group needs questions")
    if not isinstance(targets, Mapping) or set(targets) != set(questions):
        raise ValueError(f"{identifier}: typed-policy targets must exactly match qids")
    qids = tuple(f"q{index:02d}" for index in range(len(questions)))
    if tuple(questions) != qids:
        raise ValueError(f"{identifier}: typed-policy qids must be ordered q00..qNN")

    compiled_policy = compile_policy(meta.get("policy"))
    fields = meta.get("field_order")
    if not isinstance(fields, list) or fields != list(compiled_policy.fields):
        raise ValueError(f"{identifier}: field_order must exactly match policy fields")
    qid_to_field = meta.get("qid_to_field")
    expected_mapping = dict(zip(qids, fields))
    if not isinstance(qid_to_field, Mapping) or dict(qid_to_field) != expected_mapping:
        raise ValueError(f"{identifier}: qid_to_field must exactly map ordered qids to fields")

    raw_events = meta.get("field_events")
    if not isinstance(raw_events, Mapping) or set(raw_events) != set(fields):
        raise ValueError(f"{identifier}: field_events must exactly match policy fields")
    raw_score_orders = meta.get("score_orders")
    if not isinstance(raw_score_orders, Mapping):
        raise ValueError(f"{identifier}: score_orders must be an object")

    soft_targets = row.get("soft_targets", {})
    if not isinstance(soft_targets, Mapping) or not set(soft_targets) <= set(qids):
        raise ValueError(f"{identifier}: soft_targets must be keyed by known qids")

    field_labels: dict[str, tuple[str, ...]] = {}
    positive_labels: dict[str, frozenset[str]] = {}
    question_types: dict[str, str] = {}
    normalized_soft_targets: dict[str, dict[str, float]] = {}
    score_qids = set()
    target_labels: dict[str, str] = {}
    for qid, field in expected_mapping.items():
        question = questions[qid]
        if not isinstance(question, Mapping):
            raise ValueError(f"{identifier}/{qid}: question must be an object")
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"{identifier}/{qid}: unsupported typed question type")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError(f"{identifier}/{qid}: instructions must be nonempty text")
        labels = _labels(question, f"{identifier}/{qid}")
        if len(labels) > 26:
            raise ValueError(f"{identifier}/{qid}: too many typed options")
        if question_type == "noul" and set(labels) != {"no", "yes"}:
            raise ValueError(f"{identifier}/{qid}: Noul labels must be exactly no/yes")

        target = targets[qid]
        if (not isinstance(target, Mapping) or set(target) != {"label"}
                or target.get("label") not in labels):
            raise ValueError(f"{identifier}/{qid}: target label must name one criterion")
        target_labels[field] = target["label"]

        event = raw_events[field]
        if not isinstance(event, Mapping) or set(event) != {"positive_labels"}:
            raise ValueError(f"{identifier}/{qid}: event must contain only positive_labels")
        positives = event["positive_labels"]
        if (not isinstance(positives, list) or not positives
                or any(not isinstance(label, str) for label in positives)
                or len(set(positives)) != len(positives)
                or not set(positives) < set(labels)):
            raise ValueError(
                f"{identifier}/{qid}: positive_labels must be a nonempty proper label subset")

        if question_type == "score":
            score_qids.add(qid)
            order = raw_score_orders.get(qid)
            if (not isinstance(order, list) or len(order) != len(labels)
                    or len(set(order)) != len(order) or set(order) != set(labels)):
                raise ValueError(
                    f"{identifier}/{qid}: score order must contain every score label exactly once")

        if qid in soft_targets:
            distribution = soft_targets[qid]
            if (not isinstance(distribution, Mapping) or set(distribution) != set(labels)):
                raise ValueError(
                    f"{identifier}/{qid}: soft target must map every semantic label")
            values = dict(distribution)
            if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 for value in values.values())
                    or sum(values.values()) <= 0):
                raise ValueError(f"{identifier}/{qid}: invalid soft target distribution")
            total = float(sum(values.values()))
            normalized_soft_targets[qid] = {
                label: float(values[label]) / total for label in labels
            }

        field_labels[field] = labels
        positive_labels[field] = frozenset(positives)
        question_types[field] = question_type

    if set(raw_score_orders) != score_qids:
        raise ValueError(f"{identifier}: score_orders must name exactly the Score qids")

    threshold = meta.get("decision_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError(f"{identifier}: invalid typed-policy decision_threshold")
    action_trace = meta.get("action_trace")
    expected_trace = {
        field: target_labels[field] in positive_labels[field] for field in fields
    }
    if (not isinstance(action_trace, Mapping) or dict(action_trace) != expected_trace
            or any(type(value) is not bool for value in action_trace.values())):
        raise ValueError(f"{identifier}: typed-policy action_trace disagrees with targets")
    try:
        gold_outcome = canonicalize_outcome(meta.get("gold_outcome"))
    except ValueError as error:
        raise ValueError(f"{identifier}: invalid typed-policy gold_outcome: {error}") from error
    if execute_policy(compiled_policy, expected_trace) != gold_outcome:
        raise ValueError(f"{identifier}: typed-policy gold_outcome disagrees with execution")

    return {
        "id": identifier,
        "semantic_group_id": group_id,
        "option_order": meta["option_order"],
        "qids": qids,
        "qid_to_field": expected_mapping,
        "field_labels": field_labels,
        "positive_labels": positive_labels,
        "question_types": question_types,
        "soft_targets": normalized_soft_targets,
        "compiled_policy": compiled_policy,
        "action_trace": expected_trace,
        "gold_outcome": gold_outcome,
        "decision_threshold": float(threshold),
    }


def validate_typed_policy_siblings(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require exactly one canonical/reversed pair for every semantic group."""
    groups: dict[str, list[tuple[Mapping[str, Any], dict[str, Any]]]] = defaultdict(list)
    identifiers = set()
    for row in rows:
        contract = validate_typed_policy_row(row)
        if contract["id"] in identifiers:
            raise ValueError(f"duplicate typed-policy id: {contract['id']}")
        identifiers.add(contract["id"])
        groups[contract["semantic_group_id"]].append((row, contract))

    for group_id, siblings in groups.items():
        if len(siblings) != 2 or {item[1]["option_order"] for item in siblings} != OPTION_ORDERS:
            raise ValueError(
                f"{group_id}: typed-policy group needs one canonical and one reversed sibling")
        by_order = {contract["option_order"]: (row, contract)
                    for row, contract in siblings}
        canonical, canonical_contract = by_order["canonical"]
        reversed_row, reversed_contract = by_order["reversed"]
        canonical_semantics = dict(canonical)
        reversed_semantics = dict(reversed_row)
        for value in (canonical_semantics, reversed_semantics):
            value.pop("id", None)
            value.pop("questions", None)
            value.pop("meta", None)
        if canonical_semantics != reversed_semantics:
            raise ValueError(
                f"{group_id}: typed-policy sibling top-level semantics differ")
        if canonical_contract["qids"] != reversed_contract["qids"]:
            raise ValueError(f"{group_id}: typed-policy siblings have different qids")

        canonical_meta = dict(canonical["meta"])
        reversed_meta = dict(reversed_row["meta"])
        canonical_meta.pop("option_order", None)
        reversed_meta.pop("option_order", None)
        if canonical_meta != reversed_meta:
            raise ValueError(f"{group_id}: typed-policy sibling metadata is not semantic-identical")

        for qid in canonical_contract["qids"]:
            left = canonical["questions"][qid]
            right = reversed_row["questions"][qid]
            if (left.get("type") != right.get("type")
                    or left.get("instructions") != right.get("instructions")
                    or dict(left.get("criteria", {})) != dict(right.get("criteria", {}))):
                raise ValueError(
                    f"{group_id}/{qid}: typed-policy sibling question semantics differ")
            if tuple(left["criteria"]) != tuple(reversed(tuple(right["criteria"]))):
                raise ValueError(
                    f"{group_id}/{qid}: typed-policy sibling option orders are not reversed")


def typed_event_logit(
    option_scores: torch.Tensor,
    labels: Sequence[str],
    positive_labels: Sequence[str] | frozenset[str],
) -> torch.Tensor:
    """Return log P(positive labels) - log P(complement) from option logits."""
    if (not isinstance(option_scores, torch.Tensor) or option_scores.ndim != 1
            or not option_scores.is_floating_point()
            or not bool(torch.isfinite(option_scores.detach()).all().item())):
        raise ValueError("typed event option scores must be a finite floating vector")
    if (not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))
            or len(labels) != option_scores.numel()
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("typed event labels must uniquely align with option scores")
    positives = set(positive_labels)
    if not positives or not positives < set(labels):
        raise ValueError("typed event positive labels must be a nonempty proper subset")
    positive_indices = torch.tensor(
        [index for index, label in enumerate(labels) if label in positives],
        device=option_scores.device,
        dtype=torch.long,
    )
    negative_indices = torch.tensor(
        [index for index, label in enumerate(labels) if label not in positives],
        device=option_scores.device,
        dtype=torch.long,
    )
    scores = option_scores.float()
    return (torch.logsumexp(scores.index_select(0, positive_indices), dim=0)
            - torch.logsumexp(scores.index_select(0, negative_indices), dim=0))


def typed_policy_truth_logits(
    field_scores: Mapping[str, torch.Tensor],
    field_labels: Mapping[str, Sequence[str]],
    positive_labels: Mapping[str, Sequence[str] | frozenset[str]],
    *,
    decision_threshold: float,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Convert categorical field scores into relaxed Boolean policy logits."""
    if (not isinstance(field_scores, Mapping) or not isinstance(field_labels, Mapping)
            or not isinstance(positive_labels, Mapping)
            or set(field_scores) != set(field_labels) or set(field_scores) != set(positive_labels)
            or not field_scores):
        raise ValueError("typed policy field scores, labels, and events must align")
    if (isinstance(decision_threshold, bool)
            or not isinstance(decision_threshold, (int, float))
            or not math.isfinite(decision_threshold) or not 0 < decision_threshold < 1):
        raise ValueError("decision_threshold must be finite and strictly between zero and one")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    threshold_logit = math.log(decision_threshold) - math.log1p(-decision_threshold)
    return {
        field: (typed_event_logit(field_scores[field], field_labels[field],
                                  positive_labels[field]) - threshold_logit) / temperature
        for field in field_scores
    }


def hard_typed_assignment(
    field_scores: Mapping[str, torch.Tensor],
    field_labels: Mapping[str, Sequence[str]],
    positive_labels: Mapping[str, Sequence[str] | frozenset[str]],
    decision_threshold: float,
) -> dict[str, bool]:
    """Apply the serving boundary to semantic event probabilities."""
    logits = typed_policy_truth_logits(
        field_scores, field_labels, positive_labels,
        decision_threshold=decision_threshold, temperature=1.0,
    )
    return {field: bool(value.detach().item() >= 0) for field, value in logits.items()}
