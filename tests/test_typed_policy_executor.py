import copy
import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.exact_policy_executor import (
    exact_outcome_log_distribution,
    execute_policy,
    outcome_nll,
)
from scripts.train_gemma_decision_lora import (
    grouped_typed_policy_slot_loss,
    load_examples,
    loss_for_examples,
)
from scripts.typed_policy_executor import (
    typed_event_logit,
    typed_policy_truth_logits,
    validate_typed_policy_row,
    validate_typed_policy_siblings,
)


def _action(name):
    return {"name": name, "payload": {}}


def _policy():
    return {
        "fields": ["route", "eligible", "severity"],
        "mode": "priority",
        "rules": [
            {"when": {"field": "severity"}, "action": _action("urgent")},
            {"when": {"field": "eligible"}, "action": _action("approve")},
            {"when": {"field": "route"}, "action": _action("review")},
        ],
        "fallback": _action("deny"),
    }


def typed_row(*, reverse=False):
    policy = _policy()
    specs = [
        ("choice", "Choose route.", {"auto": "Automatic", "manual": "Manual",
                                      "deny": "Deny"}, "manual", ["manual", "deny"]),
        ("noul", "Is the request eligible?", {"no": "No", "yes": "Yes"},
         "yes", ["yes"]),
        ("score", "Assign severity.", {"low": "Low", "medium": "Medium",
                                       "high": "High", "critical": "Critical"},
         "high", ["high", "critical"]),
    ]
    questions = {}
    targets = {}
    events = {}
    score_orders = {}
    truths = {}
    for index, (kind, instructions, criteria, target, positive) in enumerate(specs):
        qid = f"q{index:02d}"
        field = policy["fields"][index]
        if reverse:
            criteria = dict(reversed(tuple(criteria.items())))
        questions[qid] = {
            "type": kind,
            "instructions": instructions,
            "criteria": criteria,
        }
        targets[qid] = {"label": target}
        events[field] = {"positive_labels": positive}
        truths[field] = target in positive
        if kind == "score":
            score_orders[qid] = ["low", "medium", "high", "critical"]
    return {
        "id": f"typed-case-{'reversed' if reverse else 'canonical'}",
        "state": "Apply the policy to this complete record.",
        "questions": questions,
        "targets": targets,
        "soft_targets": {
            "q00": {"auto": 0.1, "manual": 0.8, "deny": 0.1},
        },
        "meta": {
            "panel": "typed_policy_v1",
            "semantic_group_id": "semantic-case-1",
            "option_order": "reversed" if reverse else "canonical",
            "field_order": list(policy["fields"]),
            "qid_to_field": {f"q{index:02d}": field
                             for index, field in enumerate(policy["fields"])},
            "field_events": events,
            "score_orders": score_orders,
            "action_trace": truths,
            "policy": policy,
            "gold_outcome": execute_policy(policy, truths).as_dict(),
            "decision_threshold": 0.5,
        },
    }


class _TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_):
        return messages[-1]["content"]

    def __call__(self, text, **_):
        return {"input_ids": torch.tensor([[len(text) % 7 + 1]], dtype=torch.long)}


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-0.4, 0.5, 8))

    def forward(self, input_ids, **_):
        batch = input_ids.shape[0]
        return SimpleNamespace(logits=self.logits.expand(batch, 1, -1))


def test_loader_preserves_typed_siblings_as_executable_slots(tmp_path):
    rows = [typed_row(), typed_row(reverse=True)]
    path = tmp_path / "typed.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    canonical, reversed_group = load_examples([path])

    assert canonical["action_kind"] == "typed_policy_v1"
    assert canonical["action_qid_to_field"] == {
        "q00": "route", "q01": "eligible", "q02": "severity",
    }
    assert canonical["action_positive_labels"] == {
        "route": frozenset(("manual", "deny")),
        "eligible": frozenset(("yes",)),
        "severity": frozenset(("high", "critical")),
    }
    assert all(view["fixed_order"] for view in canonical["action_views"])
    assert canonical["action_views"][0]["probabilities"] == [0.1, 0.8, 0.1]
    assert [label for label, _ in reversed_group["action_views"][2]["items"]] == [
        "critical", "high", "medium", "low",
    ]


def test_trainer_executes_typed_policy_branch_end_to_end(tmp_path):
    rows = [typed_row(), typed_row(reverse=True)]
    path = tmp_path / "typed.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    groups = [torch.tensor([index]) for index in range(1, 5)]
    model = _TinyModel()

    loss, _, stats = loss_for_examples(
        model, load_examples([path]), _TinyTokenizer(), groups, 100,
        torch.device("cpu"), shuffle=True, action_loss_weight=0.7)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.logits.grad is not None and torch.isfinite(model.logits.grad).all()
    assert stats["action_groups_seen"] == 2
    assert stats["action_loss_kind"] == "nll"
    assert stats["exact_policy_outcome_loss"] == "nll"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row["meta"]["qid_to_field"].__setitem__("q00", "severity"),
         "qid_to_field"),
        (lambda row: row["meta"]["field_events"]["route"].__setitem__(
            "positive_labels", []), "nonempty proper"),
        (lambda row: row["meta"]["field_events"]["eligible"].__setitem__(
            "positive_labels", ["no", "yes"]), "nonempty proper"),
        (lambda row: row["meta"]["score_orders"].__setitem__(
            "q02", ["critical", "high", "medium"]), "score order"),
        (lambda row: row["questions"]["q00"].__setitem__("type", "binary"),
         "unsupported typed question"),
        (lambda row: row["questions"]["q01"]["criteria"].__setitem__("maybe", "Maybe"),
         "Noul labels"),
        (lambda row: row["meta"]["action_trace"].__setitem__("route", False),
         "action_trace disagrees"),
    ],
)
def test_typed_contract_fails_closed(mutate, message):
    row = typed_row()
    mutate(row)
    with pytest.raises(ValueError, match=message):
        validate_typed_policy_row(row)


def test_typed_siblings_require_exact_semantics_and_reversed_order():
    canonical, reversed_row = typed_row(), typed_row(reverse=True)
    validate_typed_policy_siblings([canonical, reversed_row])

    with pytest.raises(ValueError, match="one canonical and one reversed"):
        validate_typed_policy_siblings([canonical])

    changed = copy.deepcopy(reversed_row)
    changed["questions"]["q00"]["criteria"] = copy.deepcopy(
        canonical["questions"]["q00"]["criteria"])
    with pytest.raises(ValueError, match="option orders are not reversed"):
        validate_typed_policy_siblings([canonical, changed])

    changed = copy.deepcopy(reversed_row)
    changed["state"] += " Changed."
    with pytest.raises(ValueError, match="top-level semantics differ"):
        validate_typed_policy_siblings([canonical, changed])


def test_typed_event_logit_is_set_logsumexp_and_has_finite_gradients():
    scores = torch.tensor([-0.7, 0.2, 1.1, -0.4], requires_grad=True)
    labels = ["low", "medium", "high", "critical"]

    actual = typed_event_logit(scores, labels, {"high", "critical"})
    expected = torch.logsumexp(scores[2:], 0) - torch.logsumexp(scores[:2], 0)
    torch.testing.assert_close(actual, expected)
    actual.square().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_typed_event_logit_is_semantically_invariant_to_option_reversal():
    labels = ["auto", "manual", "deny"]
    scores = torch.tensor([-0.7, 0.2, 1.1])
    direct = typed_event_logit(scores, labels, {"manual", "deny"})
    reversed_value = typed_event_logit(
        scores.flip(0), list(reversed(labels)), {"manual", "deny"})
    torch.testing.assert_close(direct, reversed_value, rtol=0, atol=0)


def test_typed_exact_distribution_matches_brute_force_categorical_worlds():
    contract = validate_typed_policy_row(typed_row())
    labels = contract["field_labels"]
    scores = {
        "route": torch.tensor([-0.3, 0.8, 0.1], requires_grad=True),
        "eligible": torch.tensor([0.4, -0.2], requires_grad=True),
        "severity": torch.tensor([-0.9, 0.2, 0.7, -0.1], requires_grad=True),
    }
    truth_logits = typed_policy_truth_logits(
        scores, labels, contract["positive_labels"],
        decision_threshold=0.5, temperature=1.0)
    exact = {
        outcome: value.exp()
        for outcome, value in exact_outcome_log_distribution(
            contract["compiled_policy"], truth_logits).items()
    }

    brute = {outcome: torch.zeros((), dtype=torch.float32) for outcome in exact}
    probabilities = {field: torch.softmax(scores[field], 0) for field in scores}
    fields = tuple(contract["compiled_policy"].fields)
    for chosen in itertools.product(*(labels[field] for field in fields)):
        assignment = {
            field: label in contract["positive_labels"][field]
            for field, label in zip(fields, chosen)
        }
        probability = torch.stack([
            probabilities[field][labels[field].index(label)]
            for field, label in zip(fields, chosen)
        ]).prod()
        outcome = execute_policy(contract["compiled_policy"], assignment)
        brute[outcome] = brute[outcome] + probability

    assert exact.keys() == brute.keys()
    for outcome in exact:
        torch.testing.assert_close(exact[outcome], brute[outcome], atol=1e-6, rtol=1e-6)
    loss = outcome_nll(
        exact_outcome_log_distribution(contract["compiled_policy"], truth_logits),
        contract["gold_outcome"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in scores.values())


def test_zero_outcome_weight_preserves_categorical_field_graph_exactly():
    contract = validate_typed_policy_row(typed_row())
    scores = {
        field: torch.randn(len(contract["field_labels"][field]), requires_grad=True)
        for field in contract["compiled_policy"].fields
    }
    field_losses = [scores[field].square().mean()
                    for field in contract["compiled_policy"].fields]
    expected = torch.stack(field_losses).mean()

    actual, _, outcome_loss = grouped_typed_policy_slot_loss(
        field_losses, scores, contract["field_labels"], contract["positive_labels"],
        contract["compiled_policy"], contract["gold_outcome"], 0.5, 0.0, 1.0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(outcome_loss)
    actual_gradients = torch.autograd.grad(actual, tuple(scores.values()), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, tuple(scores.values()))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert torch.isfinite(actual_gradient).all()
