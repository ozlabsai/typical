import itertools
import math

import pytest
import torch

from scripts.soft_action_executor import (
    ACTIONS,
    action_cross_entropy,
    action_for,
    semantic_yes_probabilities,
    soft_action_distribution,
)


FIELD_NAMES = ("safe", "approved", "within_limit", "timely", "evidence", "covered")


def _scores(fields, *, reverse=False, magnitude=80.0, requires_grad=False):
    scores, labels = {}, {}
    for name, truth in fields.items():
        no_yes = torch.tensor(
            [-magnitude, magnitude] if truth else [magnitude, -magnitude],
            requires_grad=requires_grad,
        )
        if reverse:
            scores[name], labels[name] = no_yes.flip(0), ["yes", "no"]
        else:
            scores[name], labels[name] = no_yes, ["no", "yes"]
    return scores, labels


def test_boolean_corners_match_hard_action_oracle_exhaustively():
    for size in range(2, len(FIELD_NAMES) + 1):
        names = FIELD_NAMES[:size]
        for values in itertools.product((False, True), repeat=size):
            fields = dict(zip(names, values))
            scores, labels = _scores(fields)
            distribution = soft_action_distribution(
                scores, labels, decision_threshold=0.75, temperature=0.2
            )
            assert ACTIONS[distribution.argmax().item()] == action_for(fields)


@pytest.mark.parametrize("magnitude", [0.0, 1e2, 1e6])
def test_distribution_is_finite_nonnegative_and_normalized_at_extreme_logits(magnitude):
    fields = {name: index % 2 == 0 for index, name in enumerate(FIELD_NAMES)}
    scores, labels = _scores(fields, magnitude=magnitude)
    distribution = soft_action_distribution(
        scores, labels, decision_threshold=0.15, temperature=0.01
    )
    assert distribution.dtype == torch.float32
    assert distribution.shape == (4,)
    assert torch.isfinite(distribution).all()
    assert (distribution >= 0).all()
    torch.testing.assert_close(distribution.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_semantic_option_order_does_not_change_probabilities_or_actions():
    fields = {"safe": True, "approved": False, "within_limit": False, "timely": True}
    scores, labels = _scores(fields, magnitude=1.25)
    reversed_scores, reversed_labels = _scores(fields, reverse=True, magnitude=1.25)
    kwargs = {"decision_threshold": 0.6, "temperature": 0.7}
    ordinary = semantic_yes_probabilities(scores, labels, **kwargs)
    reversed_order = semantic_yes_probabilities(reversed_scores, reversed_labels, **kwargs)
    for name in fields:
        torch.testing.assert_close(ordinary[name], reversed_order[name])
    torch.testing.assert_close(
        soft_action_distribution(scores, labels, **kwargs),
        soft_action_distribution(reversed_scores, reversed_labels, **kwargs),
    )


def test_distribution_matches_exact_action_v1_partition():
    truth_probabilities = {
        "safe": 0.8,
        "approved": 0.7,
        "within_limit": 0.4,
        "timely": 0.6,
    }
    scores = {
        name: torch.tensor([0.0, math.log(probability / (1.0 - probability))])
        for name, probability in truth_probabilities.items()
    }
    labels = {name: ["no", "yes"] for name in scores}
    distribution = soft_action_distribution(
        scores, labels, decision_threshold=0.5, temperature=1.0
    )
    expected = torch.tensor([0.2, 0.464, 0.2016, 0.1344])
    torch.testing.assert_close(distribution, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("gold", ACTIONS)
def test_action_loss_has_finite_nonzero_gradients_for_decisive_fields(gold):
    fields = {"safe": True, "approved": True, "within_limit": True, "timely": True}
    scores, labels = _scores(fields, magnitude=0.3, requires_grad=True)
    distribution = soft_action_distribution(
        scores, labels, decision_threshold=0.5, temperature=0.8
    )
    loss = action_cross_entropy(distribution, gold)
    loss.backward()
    assert torch.isfinite(loss)
    decisive = {
        "deny": {"safe"},
        "review": {"safe", "approved", "timely"},
        "escalate": set(fields),
        "allow": set(fields),
    }[gold]
    for name in decisive:
        assert scores[name].grad is not None
        assert torch.isfinite(scores[name].grad).all()
        assert scores[name].grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda scores, labels: (scores.pop("safe"), labels.pop("safe")), "required fields"),
        (lambda scores, labels: (scores.__setitem__("mystery", torch.zeros(2)),
                                 labels.__setitem__("mystery", ["no", "yes"])), "unsupported fields"),
        (lambda scores, labels: labels.__setitem__("safe", ["yes", "yes"]), "exactly no/yes"),
        (lambda scores, labels: labels.pop("safe"), "same fields"),
        (lambda scores, labels: scores.__setitem__("safe", torch.zeros(3)), "two scores"),
        (lambda scores, labels: scores.__setitem__("safe", torch.tensor([0.0, float("nan")])), "finite"),
    ],
)
def test_malformed_field_inputs_fail_closed(mutator, message):
    fields = {"safe": True, "approved": True}
    scores, labels = _scores(fields)
    mutator(scores, labels)
    with pytest.raises(ValueError, match=message):
        soft_action_distribution(scores, labels, decision_threshold=0.5, temperature=1.0)


@pytest.mark.parametrize(
    ("threshold", "temperature"),
    [(0.0, 1.0), (1.0, 1.0), (float("nan"), 1.0), (0.5, 0.0), (0.5, -1.0),
     (0.5, float("inf"))],
)
def test_invalid_threshold_or_temperature_fails_closed(threshold, temperature):
    scores, labels = _scores({"safe": True, "approved": True})
    with pytest.raises(ValueError):
        soft_action_distribution(
            scores, labels, decision_threshold=threshold, temperature=temperature
        )


def test_action_loss_rejects_bad_gold_and_bad_distributions():
    valid = torch.full((4,), 0.25)
    with pytest.raises(ValueError, match="gold action"):
        action_cross_entropy(valid, "maybe")
    for invalid in (
        torch.ones(3) / 3,
        torch.tensor([0.4, 0.4, 0.4, -0.2]),
        torch.tensor([0.2, 0.2, 0.2, 0.2]),
        torch.tensor([0.25, 0.25, float("nan"), 0.25]),
    ):
        with pytest.raises(ValueError):
            action_cross_entropy(invalid, "allow")
