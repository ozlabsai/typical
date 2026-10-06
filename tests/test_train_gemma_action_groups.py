import copy
import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

import scripts.train_gemma_decision_lora as trainer
from scripts.exact_policy_executor import execute_policy
from scripts.train_gemma_decision_lora import (
    flatten_row,
    grouped_action_slot_loss,
    grouped_exact_policy_slot_loss,
    hard_exact_outcome_from_scores,
    hard_action_from_scores,
    load_examples,
    loss_for_examples,
    validate_action_objective,
)


def action_row(*, reverse=False):
    fields = {"safe": True, "approved": True, "within_limit": False}
    questions = {}
    targets = {}
    for index, (field, truth) in enumerate(fields.items()):
        criteria = {"no": "No", "yes": "Yes"}
        if reverse:
            criteria = dict(reversed(list(criteria.items())))
        qid = f"q{index:02d}"
        questions[qid] = {
            "type": "noul", "instructions": f"Is {field} true?", "criteria": criteria,
        }
        targets[qid] = {"label": "yes" if truth else "no"}
    return {
        "id": "action-case",
        "state": "Use the recorded facts.",
        "questions": questions,
        "targets": targets,
        "meta": {
            "panel": "action_v1",
            "field_order": list(fields),
            "action_trace": fields,
            "action_gold": "escalate",
            "decision_threshold": 0.6,
        },
    }


def _action(name, **payload):
    return {"name": name, "payload": payload}


def priority_policy():
    return {
        "fields": ["account_verified", "בתוך_התקציב"],
        "mode": "priority",
        "rules": [
            {"when": {"not": {"field": "account_verified"}},
             "action": _action("deny", reason="identity")},
            {"when": {"not": {"field": "בתוך_התקציב"}},
             "action": _action("review", queue="limits")},
        ],
        "fallback": _action("allow", queue="automatic"),
    }


def collect_policy(*, ranked=True):
    return {
        "fields": ["notify_customer", "notify_risk"],
        "mode": "collect",
        "rules": [
            {"when": {"field": "notify_customer"},
             "action": _action("email", template="status"),
             "primary_rank": None},
            {"when": {"field": "notify_risk"},
             "action": _action("alert", queue="risk"),
             "primary_rank": 0 if ranked else None},
        ],
        "fallback": None,
    }


def exact_policy_row(*, policy=None, truths=None, reverse=False):
    policy = copy.deepcopy(policy or priority_policy())
    truths = dict(truths or {
        "account_verified": True, "בתוך_התקציב": False,
    })
    questions = {}
    targets = {}
    for index, field in enumerate(policy["fields"]):
        criteria = {"no": "No", "yes": "Yes"}
        if reverse:
            criteria = dict(reversed(list(criteria.items())))
        qid = f"q{index:02d}"
        questions[qid] = {
            "type": "noul", "instructions": f"Is {field} true?", "criteria": criteria,
        }
        targets[qid] = {"label": "yes" if truths[field] else "no"}
    return {
        "id": "exact-policy-case",
        "state": "Apply the exact policy to the recorded facts.",
        "questions": questions,
        "targets": targets,
        "meta": {
            "panel": "exact_policy_v1",
            "field_order": list(policy["fields"]),
            "action_trace": truths,
            "policy": policy,
            "gold_outcome": execute_policy(policy, truths).as_dict(),
            "decision_threshold": 0.6,
            "audit_family": "extra metadata is intentionally tolerated",
        },
    }


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_):
        return messages[-1]["content"]

    def __call__(self, text, **_):
        token = int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) % 100 + 1
        return {"input_ids": torch.tensor([[token]], dtype=torch.long)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.linspace(-0.3, 0.4, 8))

    def forward(self, input_ids, **_):
        token = input_ids[:, 0].float()
        vocab = torch.arange(8, device=input_ids.device).float()
        base = token[:, None] * vocab[None, :] * 0.001
        logits = base + self.theta[None, :] * (token[:, None] % 17 + 1) / 17
        return SimpleNamespace(logits=logits[:, None, :])


def test_loader_preserves_action_row_as_one_logical_slot(tmp_path):
    path = tmp_path / "action.jsonl"
    path.write_text(json.dumps(action_row()) + "\n")

    group, = load_examples([path])

    assert group["id"] == "action-case"
    assert group["action_qids"] == ["q00", "q01", "q02"]
    assert group["action_qid_to_field"] == {
        "q00": "safe", "q01": "approved", "q02": "within_limit",
    }
    assert [view["id"] for view in group["action_views"]] == [
        "action-case/q00", "action-case/q01", "action-case/q02",
    ]
    assert group["action_gold"] == "escalate"
    assert group["decision_threshold"] == 0.6


@pytest.mark.parametrize(
    ("policy", "truths", "expected_primary", "expected_actions"),
    [
        (priority_policy(), {"account_verified": True, "בתוך_התקציב": False},
         "review", {"review"}),
        (collect_policy(), {"notify_customer": True, "notify_risk": True},
         "alert", {"alert", "email"}),
        (collect_policy(ranked=False),
         {"notify_customer": True, "notify_risk": False}, None, {"email"}),
        (collect_policy(), {"notify_customer": False, "notify_risk": False},
         None, set()),
    ],
)
def test_loader_compiles_exact_policy_once_and_accepts_complete_outcomes(
        tmp_path, policy, truths, expected_primary, expected_actions):
    path = tmp_path / "exact.jsonl"
    path.write_text(json.dumps(exact_policy_row(policy=policy, truths=truths)) + "\n")

    group, = load_examples([path])

    assert group["action_kind"] == "exact_policy_v1"
    assert tuple(group["compiled_policy"].fields) == tuple(policy["fields"])
    assert group["gold_outcome"].primary is None or (
        group["gold_outcome"].primary.name == expected_primary)
    assert {action.name for action in group["gold_outcome"].actions} == expected_actions
    assert group["action_trace"] == truths
    assert group["decision_threshold"] == 0.6


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row["meta"]["policy"].__setitem__("mode", "unknown"),
         "policy mode"),
        (lambda row: row["meta"]["field_order"].__setitem__(1, "undeclared"),
         "exactly match policy fields"),
        (lambda row: row["meta"]["action_trace"].__setitem__("account_verified", False),
         "trace or gold outcome mismatch"),
        (lambda row: row["meta"]["gold_outcome"].__setitem__("actions", []),
         "gold_outcome"),
        (lambda row: row["questions"]["q00"]["criteria"].__setitem__("maybe", "Maybe"),
         "semantic no/yes"),
        (lambda row: row["questions"]["q00"].__setitem__("type", "other"),
         "semantic no/yes"),
        (lambda row: row["questions"]["q00"].__setitem__("instructions", "  "),
         "semantic no/yes"),
        (lambda row: row.__setitem__("image", "case.png"), "text-only"),
        (lambda row: row.__setitem__("auxiliary_views", []), "cannot be paired"),
    ],
)
def test_exact_policy_loader_rejects_malformed_groups(tmp_path, mutate, message):
    row = exact_policy_row()
    mutate(row)
    path = tmp_path / "invalid-exact.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=message):
        load_examples([path])


@pytest.mark.parametrize("wrapper_key", ["paired_views", "auxiliary_views"])
def test_loader_rejects_exact_policy_rows_inside_wrappers(tmp_path, wrapper_key):
    row = exact_policy_row()
    wrapper = {"id": "wrapped-exact", wrapper_key: [row, copy.deepcopy(row)]}
    path = tmp_path / "wrapped-exact.jsonl"
    path.write_text(json.dumps(wrapper) + "\n")
    with pytest.raises(ValueError, match="cannot be paired wrappers"):
        load_examples([path])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.__setitem__("id", None), "id must be a nonempty string"),
        (lambda row: row["targets"].pop("q01"), "targets must exactly match"),
        (lambda row: (row["questions"].__setitem__("q09", row["questions"].pop("q01")),
                      row["targets"].__setitem__("q09", row["targets"].pop("q01"))),
         "contiguous"),
        (lambda row: row["meta"]["field_order"].__setitem__(1, "safe"), "unique"),
        (lambda row: row["meta"]["field_order"].pop(), "exactly map"),
        (lambda row: row["meta"].__setitem__("decision_threshold", 1.0),
         "decision_threshold"),
        (lambda row: row["meta"].__setitem__("decision_threshold", float("nan")),
         "decision_threshold"),
        (lambda row: row["meta"].__setitem__("action_gold", "maybe"), "action_gold"),
        (lambda row: row["meta"]["action_trace"].__setitem__("safe", False),
         "trace or gold"),
        (lambda row: row.__setitem__("image", "case.png"), "text-only"),
        (lambda row: row.__setitem__("paired_views", []), "cannot be paired"),
    ],
)
def test_action_loader_rejects_malformed_groups(tmp_path, mutate, message):
    row = action_row()
    mutate(row)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=message):
        load_examples([path])


@pytest.mark.parametrize("wrapper_key", ["paired_views", "auxiliary_views"])
def test_action_loader_rejects_action_rows_inside_paired_wrappers(tmp_path, wrapper_key):
    row = action_row()
    wrapper = {"id": "wrapped-action", wrapper_key: [row, copy.deepcopy(row)]}
    path = tmp_path / "wrapped.jsonl"
    path.write_text(json.dumps(wrapper) + "\n")
    with pytest.raises(ValueError, match="cannot be paired wrappers"):
        load_examples([path])


def test_loader_rejects_duplicate_json_qids_instead_of_silently_overwriting(tmp_path):
    row = action_row()
    payload = json.dumps(row)
    question = json.dumps(row["questions"]["q00"])
    duplicate_questions = f'"questions":{{"q00":{question},"q00":{question}}}'
    start = payload.index('"questions":')
    end = payload.index(', "targets":')
    payload = payload[:start] + duplicate_questions + payload[end:]
    path = tmp_path / "duplicate.jsonl"
    path.write_text(payload + "\n")
    with pytest.raises(ValueError, match="duplicate JSON key 'q00'"):
        load_examples([path])


@pytest.mark.parametrize("weight,temperature", [(-1.0, 1.0), (float("nan"), 1.0),
                                                  (0.0, 0.0), (0.0, float("inf")),
                                                  ("0.5", 1.0), (None, 1.0),
                                                  (0.0, "1.0"), (0.0, None),
                                                  (False, 1.0), (0.0, True)])
def test_action_objective_cli_values_fail_closed(weight, temperature):
    with pytest.raises(ValueError):
        validate_action_objective(weight, temperature)


def test_evaluate_forwards_action_objective_configuration(monkeypatch):
    seen = []

    def fake_loss(*_args, **kwargs):
        seen.append(kwargs)
        stats = {
            "canonical_views_seen": 1, "canonical_views_correct": 1.0,
            "canonical_ce_sum": 0.25, "auxiliary_views_seen": 0,
            "auxiliary_views_correct": 0.0, "auxiliary_ce_sum": 0.0,
            "action_groups_seen": 1, "action_groups_correct": 1, "action_ce": 0.4,
        }
        return torch.tensor(0.75), 1.0, stats

    class ModeModel:
        def eval(self):
            return self

        def train(self):
            return self

    monkeypatch.setattr(trainer, "loss_for_examples", fake_loss)
    args = SimpleNamespace(eval_samples=1, microbatch=1, max_length=100,
                           action_loss_weight=0.7, action_temperature=0.8)

    report = trainer.evaluate(
        ModeModel(), [object()], args, object(), object(), torch.device("cpu"))

    assert seen == [{"shuffle": False, "processor": None,
                     "action_loss_weight": 0.7, "action_temperature": 0.8}]
    assert report["nll"] == 0.75
    assert report["action_ce"] == 0.4


def test_grouped_action_loss_uses_semantic_labels_not_option_positions():
    ordinary = {
        "safe": torch.tensor([-0.4, 0.8], requires_grad=True),
        "approved": torch.tensor([0.2, -0.3], requires_grad=True),
    }
    reversed_scores = {name: scores.detach().flip(0).requires_grad_()
                       for name, scores in ordinary.items()}
    field_losses = [torch.tensor(0.2, requires_grad=True),
                    torch.tensor(0.8, requires_grad=True)]
    kwargs = dict(gold_action="review", decision_threshold=0.45,
                  action_loss_weight=0.7, action_temperature=0.9)

    direct, direct_distribution, _ = grouped_action_slot_loss(
        field_losses, ordinary, {name: ["no", "yes"] for name in ordinary}, **kwargs)
    reversed_loss, reversed_distribution, _ = grouped_action_slot_loss(
        field_losses, reversed_scores, {name: ["yes", "no"] for name in ordinary}, **kwargs)

    torch.testing.assert_close(direct_distribution, reversed_distribution)
    torch.testing.assert_close(direct, reversed_loss)


def test_grouped_action_training_loss_has_finite_gradients_at_confident_scores():
    scores = {
        name: torch.tensor([-10.0, 10.0], requires_grad=True)
        for name in ("safe", "approved", "within_limit", "timely")
    }
    field_losses = [torch.tensor(0.1, requires_grad=True) for _ in scores]

    loss, distribution, action_loss = grouped_action_slot_loss(
        field_losses, scores, {name: ["no", "yes"] for name in scores},
        gold_action="review", decision_threshold=0.5,
        action_loss_weight=1.0, action_temperature=0.1)
    loss.backward()

    assert torch.isfinite(loss) and torch.isfinite(action_loss)
    assert torch.isfinite(distribution).all()
    for score in scores.values():
        assert score.grad is not None and torch.isfinite(score.grad).all()


@pytest.mark.parametrize(
    ("policy", "truths"),
    [
        (priority_policy(), {"account_verified": True, "בתוך_התקציב": False}),
        (collect_policy(), {"notify_customer": True, "notify_risk": True}),
        (collect_policy(ranked=False),
         {"notify_customer": True, "notify_risk": False}),
        (collect_policy(), {"notify_customer": False, "notify_risk": False}),
    ],
)
def test_grouped_exact_policy_loss_supports_priority_collect_and_null_outcomes(
        policy, truths):
    scores = {
        name: torch.tensor([-0.45, 0.35], requires_grad=True)
        for name in policy["fields"]
    }
    labels = {name: ["no", "yes"] for name in scores}
    gold = execute_policy(policy, truths)
    field_losses = [torch.tensor(0.2, requires_grad=True) for _ in scores]

    loss, distribution, exact_loss = grouped_exact_policy_slot_loss(
        field_losses, scores, labels, policy, gold,
        decision_threshold=0.55, action_loss_weight=0.8, action_temperature=0.7)
    loss.backward()

    assert gold in distribution
    assert torch.isfinite(loss) and torch.isfinite(exact_loss)
    torch.testing.assert_close(sum(distribution.values()), torch.tensor(1.0))
    for values in scores.values():
        assert values.grad is not None and torch.isfinite(values.grad).all()


def test_grouped_exact_policy_loss_is_invariant_to_option_reversal():
    policy = collect_policy()
    truths = {"notify_customer": True, "notify_risk": True}
    gold = execute_policy(policy, truths)
    direct_scores = {
        field: torch.tensor([-0.3 + index, 0.7 - index], requires_grad=True)
        for index, field in enumerate(policy["fields"])
    }
    reversed_scores = {
        field: values.detach().flip(0).requires_grad_()
        for field, values in direct_scores.items()
    }
    field_losses = [torch.tensor(0.1), torch.tensor(0.4)]
    kwargs = dict(
        policy=policy, gold_outcome=gold, decision_threshold=0.61,
        action_loss_weight=0.9, action_temperature=0.8)

    direct_loss, direct_distribution, _ = grouped_exact_policy_slot_loss(
        field_losses, direct_scores,
        {field: ["no", "yes"] for field in direct_scores}, **kwargs)
    reversed_loss, reversed_distribution, _ = grouped_exact_policy_slot_loss(
        field_losses, reversed_scores,
        {field: ["yes", "no"] for field in reversed_scores}, **kwargs)

    torch.testing.assert_close(direct_loss, reversed_loss)
    assert direct_distribution.keys() == reversed_distribution.keys()
    for outcome in direct_distribution:
        torch.testing.assert_close(
            direct_distribution[outcome], reversed_distribution[outcome])


def test_hard_exact_outcome_replays_thresholded_policy_execution():
    policy = collect_policy()
    scores = {
        "notify_customer": torch.tensor([-1.0, 1.0]),
        "notify_risk": torch.tensor([1.0, -1.0]),
    }
    labels = {field: ["no", "yes"] for field in scores}

    outcome = hard_exact_outcome_from_scores(scores, labels, policy, 0.6)

    assert outcome == execute_policy(
        policy, {"notify_customer": True, "notify_risk": False})


def test_action_accuracy_replays_hard_serving_policy_not_relaxed_distribution_mode():
    margin = torch.tensor(0.4054651081081642)  # semantic yes probability 0.60
    scores = {name: torch.stack((torch.tensor(0.0), margin))
              for name in ("safe", "approved", "within_limit", "timely", "evidence", "covered")}
    labels = {name: ["no", "yes"] for name in scores}
    field_losses = [torch.tensor(0.1) for _ in scores]

    _, relaxed, _ = grouped_action_slot_loss(
        field_losses, scores, labels, gold_action="allow", decision_threshold=0.5,
        action_loss_weight=0.0, action_temperature=1.0)

    # Independent relaxed Bernoulli gates favor review, while serving thresholds
    # every semantic field true and therefore executes allow.
    assert trainer.ACTIONS[relaxed.argmax().item()] == "review"
    assert hard_action_from_scores(scores, labels, 0.5) == "allow"


def test_zero_action_weight_exactly_matches_grouped_mean_field_ce_and_gradient(tmp_path):
    row = action_row()
    path = tmp_path / "action.jsonl"
    path.write_text(json.dumps(row) + "\n")
    group, = load_examples([path])
    groups = [torch.tensor([1]), torch.tensor([2])]
    grouped_model, flat_model = TinyModel(), TinyModel()
    flat_model.load_state_dict(grouped_model.state_dict())

    actual, _, _ = loss_for_examples(
        grouped_model, [group], TinyTokenizer(), groups, 100, torch.device("cpu"),
        shuffle=False, action_loss_weight=0.0)
    expected, _, _ = loss_for_examples(
        flat_model, flatten_row(row), TinyTokenizer(), groups, 100, torch.device("cpu"),
        shuffle=False)
    actual.backward()
    expected.backward()

    assert torch.equal(actual, expected)
    assert torch.equal(grouped_model.theta.grad, flat_model.theta.grad)


def test_zero_exact_policy_weight_exactly_matches_flat_field_ce_and_gradient(tmp_path):
    row = exact_policy_row(reverse=True)
    path = tmp_path / "exact.jsonl"
    path.write_text(json.dumps(row) + "\n")
    group, = load_examples([path])
    groups = [torch.tensor([1]), torch.tensor([2])]
    grouped_model, flat_model = TinyModel(), TinyModel()
    flat_model.load_state_dict(grouped_model.state_dict())

    actual, _, _ = loss_for_examples(
        grouped_model, [group], TinyTokenizer(), groups, 100, torch.device("cpu"),
        shuffle=False, action_loss_weight=0.0)
    expected, _, _ = loss_for_examples(
        flat_model, flatten_row(row), TinyTokenizer(), groups, 100, torch.device("cpu"),
        shuffle=False)
    actual.backward()
    expected.backward()

    assert torch.equal(actual, expected)
    assert torch.equal(grouped_model.theta.grad, flat_model.theta.grad)


def test_positive_action_weight_changes_finite_tiny_model_loss_and_gradients(tmp_path):
    path = tmp_path / "action.jsonl"
    path.write_text(json.dumps(action_row()) + "\n")
    group, = load_examples([path])
    groups = [torch.tensor([1]), torch.tensor([2])]
    model = TinyModel()

    field_only, _, _ = loss_for_examples(
        model, [group], TinyTokenizer(), groups, 100, torch.device("cpu"),
        shuffle=False, action_loss_weight=0.0)
    field_only.backward()
    field_gradient = model.theta.grad.detach().clone()
    model.theta.grad = None

    combined, _, stats = loss_for_examples(
        model, [group], TinyTokenizer(), groups, 100, torch.device("cpu"),
        shuffle=False, action_loss_weight=0.75, action_temperature=0.8)
    combined.backward()

    assert torch.isfinite(combined)
    assert combined > field_only.detach()
    assert torch.isfinite(model.theta.grad).all()
    assert not torch.equal(model.theta.grad, field_gradient)
    assert stats["action_groups_seen"] == 1
    assert stats["action_ce"] > 0


def test_positive_exact_policy_weight_runs_cached_policy_and_reports_outcome_accuracy(
        tmp_path, monkeypatch):
    path = tmp_path / "exact.jsonl"
    path.write_text(json.dumps(exact_policy_row()) + "\n")
    group, = load_examples([path])
    compiled_identity = id(group["compiled_policy"])
    seen = []
    original = trainer.exact_outcome_log_distribution

    def observe(policy, truth_logits):
        seen.append(id(policy))
        return original(policy, truth_logits)

    monkeypatch.setattr(trainer, "exact_outcome_log_distribution", observe)
    model = TinyModel()
    loss, _, stats = loss_for_examples(
        model, [group], TinyTokenizer(), [torch.tensor([1]), torch.tensor([2])],
        100, torch.device("cpu"), shuffle=False, action_loss_weight=0.75,
        action_temperature=0.8)
    loss.backward()

    assert seen == [compiled_identity]
    assert torch.isfinite(loss) and torch.isfinite(model.theta.grad).all()
    assert stats["action_groups_seen"] == 1
    assert stats["action_groups_correct"] in {0, 1}
    assert math.isfinite(stats["action_ce"])


@pytest.mark.parametrize("kwargs", [
    {"pair_consistency_weight": 0.1},
    {"pair_flip_weight": 0.1},
    {"return_components": True},
])
def test_grouped_actions_reject_pair_and_guard_objectives(tmp_path, kwargs):
    path = tmp_path / "action.jsonl"
    path.write_text(json.dumps(action_row()) + "\n")
    group, = load_examples([path])
    with pytest.raises(ValueError, match="incompatible"):
        loss_for_examples(
            TinyModel(), [group], TinyTokenizer(), [torch.tensor([1]), torch.tensor([2])],
            100, torch.device("cpu"), shuffle=False, **kwargs)
