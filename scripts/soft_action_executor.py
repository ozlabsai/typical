"""Differentiable, training-only relaxation of the ``action_v1`` policy.

The executor consumes semantic ``no``/``yes`` option scores.  It deliberately
does not know option positions: callers provide the label order for every
field.  Serving continues to use :func:`action_for`.
"""

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


ACTIONS = ("deny", "review", "escalate", "allow")
FIELD_NAMES = ("safe", "approved", "within_limit", "timely", "evidence", "covered")
REQUIRED_FIELDS = frozenset(("safe", "approved"))


def action_for(fields):
    """Map semantic field booleans to the policy's exact final action."""
    if not fields["safe"]:
        return "deny"
    if not fields["approved"] or any(
        not value
        for name, value in fields.items()
        if name not in {"safe", "approved", "within_limit"}
    ):
        return "review"
    if "within_limit" in fields and not fields["within_limit"]:
        return "escalate"
    return "allow"


def _validate_scalar(name, value, *, unit_interval=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if unit_interval and not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_inputs(field_scores, field_labels):
    if not isinstance(field_scores, Mapping) or not isinstance(field_labels, Mapping):
        raise ValueError("field scores and labels must be mappings")
    score_fields, label_fields = set(field_scores), set(field_labels)
    if score_fields != label_fields:
        raise ValueError("field scores and labels must contain the same fields")
    missing = REQUIRED_FIELDS - score_fields
    if missing:
        raise ValueError(f"required fields are missing: {sorted(missing)}")
    unsupported = score_fields - set(FIELD_NAMES)
    if unsupported:
        raise ValueError(f"unsupported fields: {sorted(unsupported)}")

    device = None
    for name in FIELD_NAMES:
        if name not in field_scores:
            continue
        scores, labels = field_scores[name], field_labels[name]
        if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or scores.numel() != 2:
            raise ValueError(f"{name} must have exactly two scores")
        if not scores.is_floating_point():
            raise ValueError(f"{name} scores must be floating point")
        if not bool(torch.isfinite(scores.detach()).all().item()):
            raise ValueError(f"{name} scores must be finite")
        if device is None:
            device = scores.device
        elif scores.device != device:
            raise ValueError("all field scores must be on the same device")
        if (not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))
                or len(labels) != 2 or set(labels) != {"no", "yes"}):
            raise ValueError(f"{name} labels must contain exactly no/yes")


def semantic_yes_probabilities(
    field_scores,
    field_labels,
    *,
    decision_threshold,
    temperature,
):
    """Return thresholded semantic truth probabilities as float32 scalars.

    If ``m = score_yes - score_no``, the hard evaluator predicts true when
    ``sigmoid(m) >= decision_threshold``.  This function relaxes that exact
    boundary as ``sigmoid((m - logit(threshold)) / temperature)``.
    """
    logits = _semantic_truth_logits(
        field_scores,
        field_labels,
        decision_threshold=decision_threshold,
        temperature=temperature,
    )
    return {name: torch.sigmoid(logit) for name, logit in logits.items()}


def _semantic_truth_logits(
    field_scores,
    field_labels,
    *,
    decision_threshold,
    temperature,
):
    _validate_inputs(field_scores, field_labels)
    threshold = _validate_scalar(
        "decision_threshold", decision_threshold, unit_interval=True
    )
    temperature = _validate_scalar("temperature", temperature, positive=True)
    threshold_logit = math.log(threshold) - math.log1p(-threshold)

    logits = {}
    for name in FIELD_NAMES:
        if name not in field_scores:
            continue
        labels = field_labels[name]
        scores = field_scores[name].float()
        margin = scores[labels.index("yes")] - scores[labels.index("no")]
        logits[name] = (margin - threshold_logit) / temperature
    return logits


def _log1mexp(value):
    """Stable ``log(1 - exp(value))`` for a scalar ``value <= 0``."""
    cutoff = -math.log(2.0)
    return torch.where(
        value < cutoff,
        torch.log1p(-torch.exp(value)),
        torch.log(-torch.expm1(value)),
    )


def soft_action_distribution(
    field_scores,
    field_labels,
    *,
    decision_threshold,
    temperature,
):
    """Return the exact independent-Bernoulli ``action_v1`` distribution.

    The output order is :data:`ACTIONS`.  The four gates form a partition:
    unsafe; safe but not review-clear; safe/review-clear but over limit; and
    safe/review-clear/within limit.
    """
    logits = _semantic_truth_logits(
        field_scores,
        field_labels,
        decision_threshold=decision_threshold,
        temperature=temperature,
    )
    log_safe = F.logsigmoid(logits["safe"])
    log_unsafe = F.logsigmoid(-logits["safe"])
    log_review_clear = torch.zeros((), dtype=torch.float32, device=log_safe.device)
    for name in FIELD_NAMES:
        if name in logits and name not in {"safe", "within_limit"}:
            log_review_clear = log_review_clear + F.logsigmoid(logits[name])
    if "within_limit" in logits:
        log_within_limit = F.logsigmoid(logits["within_limit"])
        log_over_limit = F.logsigmoid(-logits["within_limit"])
    else:
        log_within_limit = torch.zeros((), dtype=torch.float32, device=log_safe.device)
        log_over_limit = torch.full((), -torch.inf, dtype=torch.float32,
                                    device=log_safe.device)

    log_actions = torch.stack((
        log_unsafe,
        log_safe + _log1mexp(log_review_clear),
        log_safe + log_review_clear + log_over_limit,
        log_safe + log_review_clear + log_within_limit,
    ))
    distribution = F.softmax(log_actions, dim=0)
    _validate_distribution(distribution)
    return distribution


def _validate_distribution(distribution):
    if (not isinstance(distribution, torch.Tensor) or distribution.ndim != 1
            or distribution.numel() != len(ACTIONS) or not distribution.is_floating_point()):
        raise ValueError(f"action distribution must have shape ({len(ACTIONS)},)")
    detached = distribution.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError("action distribution must be finite")
    if bool((detached < 0).any().item()):
        raise ValueError("action distribution must be nonnegative")
    if not math.isclose(float(detached.sum().item()), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("action distribution must sum to one")


def action_cross_entropy(distribution, gold_action):
    """Return primary-action cross entropy for an ``action_v1`` distribution."""
    if gold_action not in ACTIONS:
        raise ValueError(f"unsupported gold action: {gold_action!r}")
    _validate_distribution(distribution)
    probability = distribution[ACTIONS.index(gold_action)]
    return -torch.log(probability.clamp_min(torch.finfo(distribution.dtype).tiny))
