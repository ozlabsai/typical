import itertools
import math
from dataclasses import replace

import pytest
import torch

from scripts.exact_policy_executor import (
    CanonicalAction,
    CanonicalOutcome,
    CompiledPolicy,
    action_set_nll,
    canonicalize_action,
    canonicalize_outcome,
    compile_policy,
    exact_outcome_distribution,
    exact_outcome_log_distribution,
    execute_policy,
    outcome_nll,
    primary_action_nll,
)
from scripts.soft_action_executor import ACTIONS, action_for, soft_action_log_distribution


def action(name, payload=None):
    return {"name": name, "payload": payload}


def priority_policy():
    return {
        "fields": ["a", "b"],
        "mode": "priority",
        "rules": [
            {
                "when": {"all": [{"field": "a"}, {"not": {"field": "b"}}]},
                "action": action("a_without_b", {"code": 1}),
            },
            {
                "when": {"any": [{"field": "b"}, {"const": False}]},
                "action": action("has_b", {"code": 2}),
            },
        ],
        "fallback": action("fallback", {"code": 0}),
    }


def collect_policy(rule_order=("a", "b")):
    specs = {
        "a": {
            "when": {"field": "a"},
            "action": action("audit", {"labels": [1, "one", True, None]}),
            "primary_rank": 10,
        },
        "b": {
            "when": {"field": "b"},
            "action": action("block", {"reason": {"kind": "policy", "score": 2.5}}),
            "primary_rank": 0,
        },
    }
    return {
        "fields": ["a", "b"],
        "mode": "collect",
        "rules": [specs[name] for name in rule_order],
        "fallback": action("allow", {}),
    }


def action_v1_program(fields=("safe", "approved", "within_limit", "timely",
                              "evidence", "covered")):
    review_fields = [name for name in fields
                     if name not in {"safe", "within_limit"}]
    rules = [
        {"when": {"not": {"field": "safe"}}, "action": action("deny")},
        {
            "when": {"any": [{"not": {"field": name}} for name in review_fields]},
            "action": action("review"),
        },
    ]
    if "within_limit" in fields:
        rules.append({"when": {"not": {"field": "within_limit"}},
                      "action": action("escalate")})
    return {"fields": list(fields), "mode": "priority", "rules": rules,
            "fallback": action("allow")}


def by_primary_name(distribution):
    return {outcome.primary.name: value for outcome, value in distribution.items()}


def test_priority_truth_table_covers_all_any_not_constant_and_fallback():
    policy = compile_policy(priority_policy())
    expected = {
        (False, False): "fallback",
        (False, True): "has_b",
        (True, False): "a_without_b",
        (True, True): "has_b",
    }
    for values, name in expected.items():
        result = execute_policy(policy, dict(zip(policy.fields, values)))
        assert result.primary.name == name
        assert result.actions == (result.primary,)


def test_priority_order_resolves_overlapping_rules():
    program = priority_policy()
    program["rules"] = [
        {"when": {"field": "a"}, "action": action("first")},
        {"when": {"field": "b"}, "action": action("second")},
    ]
    result = execute_policy(compile_policy(program), {"a": True, "b": True})
    assert result.primary.name == "first"


def test_collect_mode_canonicalizes_actions_and_uses_explicit_primary_rank():
    left = execute_policy(compile_policy(collect_policy(("a", "b"))),
                          {"a": True, "b": True})
    right = execute_policy(compile_policy(collect_policy(("b", "a"))),
                           {"a": True, "b": True})
    assert left == right
    assert left.primary.name == "block"
    assert [item.name for item in left.actions] == ["audit", "block"]
    assert left.actions[0].payload == {"labels": [1, "one", True, None]}
    assert execute_policy(compile_policy(collect_policy()),
                          {"a": False, "b": False}).primary.name == "allow"


def test_action_and_outcome_canonicalization_preserve_json_types_and_ignore_order():
    first = action("notify", {"z": [1, 1.0, False], "a": None})
    second = action("block", {"reason": "risk"})
    canonical = canonicalize_outcome({"primary": second, "actions": [second, first]})
    reversed_outcome = canonicalize_outcome(
        {"primary": second, "actions": [first, second]})
    assert canonical == reversed_outcome
    assert canonical.as_dict()["actions"][1]["payload"] == {"z": [1, 1.0, False], "a": None}


def test_empty_and_null_primary_outcomes_are_canonical_and_round_trip():
    empty = canonicalize_outcome({"primary": None, "actions": []})
    assert empty == CanonicalOutcome(primary=None, actions=())
    assert empty.as_dict() == {"primary": None, "actions": []}

    audit = canonicalize_action(action("audit"))
    no_primary = canonicalize_outcome({"primary": None, "actions": [audit.as_dict()]})
    assert no_primary == CanonicalOutcome(primary=None, actions=(audit,))
    assert sorted((no_primary, empty)) == [empty, no_primary]

    with pytest.raises(ValueError, match="belong"):
        CanonicalOutcome(primary=audit, actions=())


@pytest.mark.parametrize(
    "primary,actions,match",
    [
        ("not-an-action", (), "primary must be"),
        (canonicalize_action(action("ok")), [canonicalize_action(action("ok"))],
         "tuple of CanonicalAction"),
        (canonicalize_action(action("ok")), ({"name": "bad", "payload": None},),
         "tuple of CanonicalAction"),
    ],
)
def test_direct_canonical_outcome_construction_fails_closed(primary, actions, match):
    with pytest.raises(ValueError, match=match):
        CanonicalOutcome(primary=primary, actions=actions)


def test_exact_distribution_normalizes_and_semantic_field_key_order_is_irrelevant():
    policy = compile_policy(collect_policy())
    logits = {"a": torch.tensor(0.7), "b": torch.tensor(-1.2)}
    reversed_logits = {"b": logits["b"], "a": logits["a"]}
    first = exact_outcome_distribution(policy, logits)
    second = exact_outcome_distribution(policy, reversed_logits)
    assert sum(first.values()).item() == pytest.approx(1.0, abs=1e-6)
    assert first.keys() == second.keys()
    for outcome in first:
        assert first[outcome].item() == pytest.approx(second[outcome].item(), abs=1e-7)


def test_gold_nll_helpers_marginalize_complete_primary_and_action_set_targets():
    policy = compile_policy(collect_policy())
    distribution = exact_outcome_log_distribution(
        policy, {"a": torch.tensor(0.2), "b": torch.tensor(0.4)})
    both = execute_policy(policy, {"a": True, "b": True})
    assert outcome_nll(distribution, both).item() == pytest.approx(-distribution[both].item())

    primary_matches = [value for outcome, value in distribution.items()
                       if outcome.primary == both.primary]
    assert primary_action_nll(distribution, both.primary).item() == pytest.approx(
        -torch.logsumexp(torch.stack(primary_matches), 0).item())
    assert action_set_nll(distribution, list(reversed(both.actions))).item() == pytest.approx(
        -distribution[both].item())


def test_collect_null_fallback_and_unranked_rules_cover_typesafe_outcomes_and_nlls():
    program = {
        "fields": ["handoff", "notify"],
        "mode": "collect",
        "rules": [
            {"when": {"field": "handoff"}, "action": action("handoff"),
             "primary_rank": None},
            {"when": {"field": "notify"}, "action": action("notify"),
             "primary_rank": None},
        ],
        "fallback": None,
    }
    policy = compile_policy(program)
    empty = execute_policy(policy, {"handoff": False, "notify": False})
    both = execute_policy(policy, {"handoff": True, "notify": True})
    assert empty == CanonicalOutcome(primary=None, actions=())
    assert both.primary is None
    assert [item.name for item in both.actions] == ["handoff", "notify"]

    distribution = exact_outcome_log_distribution(
        policy, {"handoff": torch.tensor(-0.2), "notify": torch.tensor(0.7)}
    )
    null_primary_values = [value for outcome, value in distribution.items()
                           if outcome.primary is None]
    assert primary_action_nll(distribution, None).item() == pytest.approx(
        -torch.logsumexp(torch.stack(null_primary_values), 0).item()
    )
    assert action_set_nll(distribution, []).item() == pytest.approx(
        -distribution[empty].item()
    )
    assert outcome_nll(
        distribution, {"primary": None, "actions": []}
    ).item() == pytest.approx(-distribution[empty].item())


def test_collect_uses_only_ranked_matches_to_choose_optional_primary():
    program = collect_policy()
    program["rules"][0]["primary_rank"] = None
    policy = compile_policy(program)
    only_unranked = execute_policy(policy, {"a": True, "b": False})
    mixed = execute_policy(policy, {"a": True, "b": True})
    assert only_unranked.primary is None
    assert mixed.primary is not None and mixed.primary.name == "block"


def test_confident_logits_have_finite_exact_nll_and_gradients():
    policy = compile_policy(collect_policy())
    logits = {"a": torch.tensor(100.0, requires_grad=True),
              "b": torch.tensor(-100.0, requires_grad=True)}
    distribution = exact_outcome_log_distribution(policy, logits)
    rare_gold = execute_policy(policy, {"a": False, "b": True})
    loss = outcome_nll(distribution, rare_gold)
    loss.backward()
    assert torch.isfinite(loss)
    for value in logits.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad)


@pytest.mark.parametrize(
    "dtype,magnitude",
    [(torch.float32, torch.finfo(torch.float32).max), (torch.float64, 1e300)],
)
def test_extreme_finite_logits_are_promoted_or_preserved(dtype, magnitude):
    policy = compile_policy(collect_policy())
    logits = {"a": torch.tensor(magnitude, dtype=dtype, requires_grad=True),
              "b": torch.tensor(-magnitude, dtype=dtype, requires_grad=True)}
    distribution = exact_outcome_log_distribution(policy, logits)
    assert all(torch.isfinite(value) for value in distribution.values())
    loss = outcome_nll(distribution, execute_policy(policy, {"a": False, "b": True}))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None and torch.isfinite(value.grad)
               for value in logits.values())


def test_action_v1_truth_table_matches_existing_hard_executor():
    fields = ("safe", "approved", "within_limit", "timely", "evidence", "covered")
    policy = compile_policy(action_v1_program(fields))
    for values in itertools.product((False, True), repeat=len(fields)):
        assignment = dict(zip(fields, values))
        assert execute_policy(policy, assignment).primary.name == action_for(assignment)


def test_action_v1_exact_marginals_match_existing_closed_form():
    fields = ("safe", "approved", "within_limit", "timely", "evidence", "covered")
    policy = compile_policy(action_v1_program(fields))
    generator = torch.Generator().manual_seed(4801)
    for _ in range(12):
        logits = {name: torch.randn((), generator=generator) * 3 for name in fields}
        generic = by_primary_name(exact_outcome_log_distribution(policy, logits))
        scores = {name: torch.stack((torch.zeros_like(value), value))
                  for name, value in logits.items()}
        labels = {name: ["no", "yes"] for name in fields}
        closed = soft_action_log_distribution(
            scores, labels, decision_threshold=0.5, temperature=1.0)
        assert set(generic) == set(ACTIONS)
        for index, name in enumerate(ACTIONS):
            assert generic[name].item() == pytest.approx(closed[index].item(), abs=2e-6)


@pytest.mark.parametrize("field_count", [1, 13])
def test_field_count_bounds_are_enforced(field_count):
    fields = [f"f{index}" for index in range(field_count)]
    program = {
        "fields": fields,
        "mode": "collect",
        "rules": [{"when": {"field": name}, "action": action(f"a{index}"),
                   "primary_rank": index}
                  for index, name in enumerate(fields)],
        "fallback": action("fallback"),
    }
    with pytest.raises(ValueError, match="2-12"):
        compile_policy(program)


def test_twelve_field_policy_enumerates_complete_truth_table():
    fields = [f"f{index}" for index in range(12)]
    program = {
        "fields": fields,
        "mode": "collect",
        "rules": [{"when": {"field": name}, "action": action(f"a{index}"),
                   "primary_rank": index}
                  for index, name in enumerate(fields)],
        "fallback": action("fallback"),
    }
    policy = compile_policy(program)
    assert len(policy.truth_outcomes) == 4096
    assert execute_policy(policy, {name: True for name in fields}).primary.name == "a0"


@pytest.mark.parametrize(
    "payload,match",
    [
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        ({1: "bad"}, "keys must be strings"),
        (("tuple",), "non-JSON"),
        ({"bad": {1, 2}}, "non-JSON"),
    ],
)
def test_invalid_payloads_are_rejected(payload, match):
    with pytest.raises(ValueError, match=match):
        canonicalize_action(action("bad", payload))


@pytest.mark.parametrize(
    "value,match",
    [
        (action("bad\ud800"), "surrogate"),
        (action("bad", "\ud800"), "surrogate"),
        (action("bad", {"\ud800": "value"}), "surrogate"),
        (action("bad", "x" * 65_537), "character limit"),
        (action("bad", 1 << 4096), "bit-length"),
    ],
)
def test_payload_and_action_resource_limits_reject_adversarial_values(value, match):
    with pytest.raises(ValueError, match=match):
        canonicalize_action(value)


def test_cyclic_payload_is_rejected():
    payload = []
    payload.append(payload)
    with pytest.raises(ValueError, match="cycle"):
        canonicalize_action(action("bad", payload))


def test_policy_rule_and_condition_resource_limits_are_enforced():
    too_many_rules = collect_policy()
    too_many_rules["rules"] = [
        {"when": {"field": "a"}, "action": action(f"a{index}"),
         "primary_rank": index}
        for index in range(129)
    ]
    with pytest.raises(ValueError, match="at most 128"):
        compile_policy(too_many_rules)

    too_wide = priority_policy()
    too_wide["rules"][0]["when"] = {
        "all": [{"field": "a"} for _ in range(65)]
    }
    with pytest.raises(ValueError, match="arity limit"):
        compile_policy(too_wide)

    nested = {"field": "a"}
    for _ in range(33):
        nested = {"not": nested}
    too_deep = priority_policy()
    too_deep["rules"][0]["when"] = nested
    with pytest.raises(ValueError, match="depth limit"):
        compile_policy(too_deep)


def test_policy_total_payload_bytes_are_bounded():
    program = {
        "fields": ["a", "b"],
        "mode": "collect",
        "rules": [
            {"when": {"field": "a" if index % 2 else "b"},
             "action": action(f"action-{index}", "x" * 65_536),
             "primary_rank": index}
            for index in range(17)
        ],
        "fallback": None,
    }
    with pytest.raises(ValueError, match="policy action payloads exceed"):
        compile_policy(program)


def test_direct_canonical_action_construction_still_validates_canonical_json():
    with pytest.raises(ValueError, match="nonempty trimmed"):
        CanonicalAction(" bad ", "{}")
    with pytest.raises(ValueError, match="valid finite JSON"):
        CanonicalAction("bad", "NaN")
    with pytest.raises(ValueError, match="not canonical"):
        CanonicalAction("bad", '{"z": 1, "a": 2}')


def test_duplicate_actions_ranks_and_outcome_members_are_rejected():
    duplicate_actions = collect_policy()
    duplicate_actions["rules"][1]["action"] = duplicate_actions["rules"][0]["action"]
    with pytest.raises(ValueError, match="duplicate rule actions"):
        compile_policy(duplicate_actions)

    duplicate_ranks = collect_policy()
    duplicate_ranks["rules"][1]["primary_rank"] = 10
    with pytest.raises(ValueError, match="ranks must be unique"):
        compile_policy(duplicate_ranks)

    duplicated = action("same")
    with pytest.raises(ValueError, match="duplicate actions"):
        canonicalize_outcome({"primary": duplicated, "actions": [duplicated, duplicated]})


def test_provided_but_unreachable_collect_fallback_is_rejected():
    program = {
        "fields": ["a", "b"],
        "mode": "collect",
        "rules": [
            {"when": {"const": True}, "action": action("always"), "primary_rank": 2},
            {"when": {"field": "a"}, "action": action("a"), "primary_rank": 0},
            {"when": {"field": "b"}, "action": action("b"), "primary_rank": 1},
        ],
        "fallback": action("dead"),
    }
    with pytest.raises(ValueError, match="collect fallback is unreachable"):
        compile_policy(program)


def test_direct_compiled_policy_construction_cannot_bypass_validation():
    valid = compile_policy(priority_policy())
    malformed = [
        replace(valid, fields=("a",)),
        replace(valid, mode="unknown"),
        replace(valid, mode=[]),
        replace(valid, rules=("not-a-rule",)),
        replace(valid, fallback="not-an-action"),
        replace(valid, truth_outcomes=valid.truth_outcomes[:-1]),
        replace(valid, truth_outcomes=("not-an-outcome",) * 4),
        replace(valid, truth_outcomes=tuple(reversed(valid.truth_outcomes))),
    ]
    for policy in malformed:
        assert isinstance(policy, CompiledPolicy)
        with pytest.raises(ValueError, match="compiled"):
            execute_policy(policy, {"a": False, "b": False})


def test_ambiguous_or_incomplete_programs_are_rejected():
    no_fallback = priority_policy()
    no_fallback["fallback"] = None
    with pytest.raises(ValueError, match="requires a fallback"):
        compile_policy(no_fallback)

    unused = priority_policy()
    unused["rules"] = [{"when": {"field": "a"}, "action": action("only_a")}]
    with pytest.raises(ValueError, match="unused"):
        compile_policy(unused)

    dead = priority_policy()
    dead["rules"].append({"when": {"const": False}, "action": action("never")})
    with pytest.raises(ValueError, match="unreachable rules"):
        compile_policy(dead)

    masked = {
        "fields": ["a", "b"],
        "mode": "priority",
        "rules": [
            {"when": {"any": [{"field": "a"}, {"field": "b"}]},
             "action": action("first")},
            {"when": {"field": "a"}, "action": action("masked")},
        ],
        "fallback": action("fallback"),
    }
    with pytest.raises(ValueError, match="unreachable rules"):
        compile_policy(masked)

    noncausal = priority_policy()
    noncausal["rules"] = [
        {"when": {"any": [{"field": "a"}, {"field": "b"}, {"const": True}]},
         "action": action("always")},
    ]
    with pytest.raises(ValueError, match="unreachable|outcome-causal"):
        compile_policy(noncausal)


def test_assignment_logits_and_gold_targets_fail_closed():
    policy = compile_policy(priority_policy())
    with pytest.raises(ValueError, match="exactly match"):
        execute_policy(policy, {"a": True})
    with pytest.raises(ValueError, match="Boolean"):
        execute_policy(policy, {"a": 1, "b": False})
    with pytest.raises(ValueError, match="floating scalar"):
        exact_outcome_log_distribution(policy, {"a": torch.tensor([0.0]),
                                                "b": torch.tensor(0.0)})
    with pytest.raises(ValueError, match="finite"):
        exact_outcome_log_distribution(policy, {"a": torch.tensor(math.nan),
                                                "b": torch.tensor(0.0)})
    distribution = exact_outcome_log_distribution(
        policy, {"a": torch.tensor(0.0), "b": torch.tensor(0.0)})
    impossible = CanonicalOutcome(
        primary=canonicalize_action(action("impossible")),
        actions=(canonicalize_action(action("impossible")),),
    )
    with pytest.raises(ValueError, match="structurally impossible"):
        outcome_nll(distribution, impossible)
