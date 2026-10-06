"""Exact differentiable execution for small deterministic Boolean policies.

Policies operate on two to twelve semantic Boolean fields.  They either choose
the first matching rule (``priority``) or collect every matching rule
(``collect``).  The latter assigns an explicit rank to every action so its
primary action remains deterministic even though the action set is canonical
and order-independent.

The probabilistic executor enumerates every truth assignment and marginalizes
their probability in log space.  It is intended for training and evaluation of
small workflow policies; serving can call :func:`execute_policy` directly.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


MIN_FIELDS = 2
MAX_FIELDS = 12
_CONDITION_KEYS = frozenset(("field", "all", "any", "not", "const"))


@dataclass(frozen=True, order=True)
class CanonicalAction:
    """Hashable action with a canonical, typed JSON payload."""

    name: str
    payload_json: str

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name.strip()
                or self.name != self.name.strip()):
            raise ValueError("action name must be a nonempty trimmed string")
        if not isinstance(self.payload_json, str):
            raise ValueError("canonical action payload must be JSON text")
        try:
            payload = json.loads(
                self.payload_json,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value}")
                ),
            )
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("canonical action payload must be valid finite JSON") from error
        _validate_json(payload)
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if canonical != self.payload_json:
            raise ValueError("canonical action payload JSON is not canonical")

    @property
    def payload(self) -> Any:
        return json.loads(self.payload_json)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "payload": self.payload}


@dataclass(frozen=True, order=True)
class CanonicalOutcome:
    """A deterministic primary action and an order-independent action set."""

    primary: CanonicalAction
    actions: tuple[CanonicalAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.primary, CanonicalAction):
            raise ValueError("outcome primary must be a CanonicalAction")
        if (not isinstance(self.actions, tuple)
                or any(not isinstance(action, CanonicalAction) for action in self.actions)):
            raise ValueError("outcome actions must be a tuple of CanonicalAction values")
        if not self.actions:
            raise ValueError("an outcome must contain at least one action")
        if tuple(sorted(self.actions)) != self.actions:
            raise ValueError("outcome actions must be in canonical order")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("outcome actions must be unique")
        if self.primary not in self.actions:
            raise ValueError("the primary action must belong to the action set")

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "actions": [action.as_dict() for action in self.actions],
        }


# Compiled conditions are immutable tuples.  Their first item is one of the
# public condition operators; nested operands follow it.
Condition = tuple[Any, ...]


@dataclass(frozen=True)
class _Rule:
    condition: Condition
    action: CanonicalAction
    primary_rank: int | None = None


@dataclass(frozen=True)
class CompiledPolicy:
    """Validated policy plus its complete deterministic truth table."""

    fields: tuple[str, ...]
    mode: str
    rules: tuple[_Rule, ...]
    fallback: CanonicalAction | None
    truth_outcomes: tuple[CanonicalOutcome, ...] = field(repr=False)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_json(value: Any, path: str = "payload", active: set[int] | None = None) -> None:
    """Reject values that are not unambiguous finite JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, (list, dict)):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try:
            if isinstance(value, list):
                for index, item in enumerate(value):
                    _validate_json(item, f"{path}[{index}]", active)
            else:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{path} object keys must be strings")
                    _validate_json(item, f"{path}.{key}", active)
        finally:
            active.remove(identity)
        return
    raise ValueError(f"{path} contains a non-JSON value")


def canonicalize_action(value: Any) -> CanonicalAction:
    if isinstance(value, CanonicalAction):
        return value
    action = _require_mapping(value, "action")
    if set(action) != {"name", "payload"}:
        raise ValueError("an action must contain exactly name and payload")
    name = action["name"]
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ValueError("action name must be a nonempty trimmed string")
    payload = action["payload"]
    _validate_json(payload)
    payload_json = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return CanonicalAction(name=name, payload_json=payload_json)


def canonicalize_outcome(value: Any) -> CanonicalOutcome:
    if isinstance(value, CanonicalOutcome):
        return value
    outcome = _require_mapping(value, "outcome")
    if set(outcome) != {"primary", "actions"}:
        raise ValueError("an outcome must contain exactly primary and actions")
    raw_actions = outcome["actions"]
    if (not isinstance(raw_actions, Sequence)
            or isinstance(raw_actions, (str, bytes)) or not raw_actions):
        raise ValueError("outcome actions must be a nonempty array")
    actions = tuple(canonicalize_action(action) for action in raw_actions)
    if len(set(actions)) != len(actions):
        raise ValueError("duplicate actions are not allowed")
    return CanonicalOutcome(
        primary=canonicalize_action(outcome["primary"]),
        actions=tuple(sorted(actions)),
    )


def _compile_condition(value: Any, declared_fields: frozenset[str],
                       referenced: set[str], active: set[int]) -> Condition:
    condition = _require_mapping(value, "condition")
    identity = id(condition)
    if identity in active:
        raise ValueError("condition contains a cycle")
    if len(condition) != 1 or not set(condition) <= _CONDITION_KEYS:
        raise ValueError("condition must contain exactly one supported operator")
    active.add(identity)
    try:
        operator, operand = next(iter(condition.items()))
        if operator == "field":
            if not isinstance(operand, str) or operand not in declared_fields:
                raise ValueError(f"condition references undeclared field {operand!r}")
            referenced.add(operand)
            return ("field", operand)
        if operator == "const":
            if type(operand) is not bool:
                raise ValueError("const condition must be Boolean")
            return ("const", operand)
        if operator == "not":
            return ("not", _compile_condition(operand, declared_fields, referenced, active))
        if (not isinstance(operand, Sequence) or isinstance(operand, (str, bytes))
                or not operand):
            raise ValueError(f"{operator} condition needs a nonempty array")
        return (
            operator,
            tuple(_compile_condition(item, declared_fields, referenced, active)
                  for item in operand),
        )
    finally:
        active.remove(identity)


def _condition_value(condition: Condition, assignment: Mapping[str, bool]) -> bool:
    operator, operand = condition
    if operator == "field":
        return assignment[operand]
    if operator == "const":
        return operand
    if operator == "not":
        return not _condition_value(operand, assignment)
    values = (_condition_value(item, assignment) for item in operand)
    return all(values) if operator == "all" else any(values)


def _canonical_outcome(primary: CanonicalAction,
                       actions: Sequence[CanonicalAction]) -> CanonicalOutcome:
    canonical = tuple(sorted(actions))
    if len(set(canonical)) != len(canonical):
        raise ValueError("a policy produced duplicate actions")
    return CanonicalOutcome(primary=primary, actions=canonical)


def _execute_compiled(policy: CompiledPolicy, assignment: Mapping[str, bool]) -> CanonicalOutcome:
    if policy.mode == "priority":
        for rule in policy.rules:
            if _condition_value(rule.condition, assignment):
                return _canonical_outcome(rule.action, (rule.action,))
        # Priority policies always have a fallback after compilation.
        return _canonical_outcome(policy.fallback, (policy.fallback,))  # type: ignore[arg-type]

    matched = [rule for rule in policy.rules if _condition_value(rule.condition, assignment)]
    if not matched:
        if policy.fallback is None:
            raise ValueError("collect policy has no action for this assignment")
        return _canonical_outcome(policy.fallback, (policy.fallback,))
    primary = min(matched, key=lambda rule: rule.primary_rank).action
    return _canonical_outcome(primary, tuple(rule.action for rule in matched))


def compile_policy(program: Mapping[str, Any]) -> CompiledPolicy:
    """Validate and compile a JSON-shaped deterministic policy."""
    program = _require_mapping(program, "policy")
    if set(program) != {"fields", "mode", "rules", "fallback"}:
        raise ValueError("policy must contain exactly fields, mode, rules, and fallback")
    fields = program["fields"]
    if (not isinstance(fields, Sequence) or isinstance(fields, (str, bytes))
            or not MIN_FIELDS <= len(fields) <= MAX_FIELDS
            or any(not isinstance(name, str) or not name.strip() or name != name.strip()
                   for name in fields)
            or len(set(fields)) != len(fields)):
        raise ValueError(f"policy fields must be {MIN_FIELDS}-{MAX_FIELDS} unique names")
    fields = tuple(fields)
    declared_fields = frozenset(fields)
    mode = program["mode"]
    if not isinstance(mode, str) or mode not in {"priority", "collect"}:
        raise ValueError("policy mode must be priority or collect")
    raw_rules = program["rules"]
    if (not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes))
            or not raw_rules):
        raise ValueError("policy rules must be a nonempty array")

    referenced: set[str] = set()
    rules = []
    actions = []
    ranks = []
    expected_rule_keys = ({"when", "action"} if mode == "priority"
                          else {"when", "action", "primary_rank"})
    for index, raw_rule in enumerate(raw_rules):
        rule = _require_mapping(raw_rule, f"rule {index}")
        if set(rule) != expected_rule_keys:
            raise ValueError(f"rule {index} has invalid keys")
        condition = _compile_condition(rule["when"], declared_fields, referenced, set())
        action = canonicalize_action(rule["action"])
        rank = None
        if mode == "collect":
            rank = rule["primary_rank"]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ValueError("collect primary ranks must be nonnegative integers")
            ranks.append(rank)
        rules.append(_Rule(condition=condition, action=action, primary_rank=rank))
        actions.append(action)
    if referenced != declared_fields:
        raise ValueError(f"policy fields are unused: {sorted(declared_fields - referenced)}")
    if len(set(actions)) != len(actions):
        raise ValueError("duplicate rule actions are not allowed")
    if mode == "collect" and len(set(ranks)) != len(ranks):
        raise ValueError("collect primary ranks must be unique")

    raw_fallback = program["fallback"]
    if mode == "priority" and raw_fallback is None:
        raise ValueError("priority policy requires a fallback action")
    fallback = None if raw_fallback is None else canonicalize_action(raw_fallback)
    if fallback is not None and fallback in actions:
        raise ValueError("fallback action duplicates a rule action")

    # Build and validate every outcome once.  This catches uncovered collect
    # assignments and makes later differentiable marginalization deterministic.
    provisional = CompiledPolicy(fields, mode, tuple(rules), fallback, ())
    outcomes = []
    reachable = [False] * len(rules)
    fallback_reachable = False
    for bits in range(1 << len(fields)):
        assignment = {name: bool(bits & (1 << index))
                      for index, name in enumerate(fields)}
        matches = [_condition_value(rule.condition, assignment) for rule in rules]
        if mode == "priority":
            if any(matches):
                reachable[matches.index(True)] = True
            fallback_reachable |= not any(matches)
        else:
            for index, matches_rule in enumerate(matches):
                reachable[index] |= matches_rule
            if not any(matches):
                fallback_reachable = True
        outcomes.append(_execute_compiled(provisional, assignment))
    if not all(reachable):
        dead = [index for index, value in enumerate(reachable) if not value]
        raise ValueError(f"policy has unreachable rules: {dead}")
    if mode == "priority" and not fallback_reachable:
        raise ValueError("priority fallback is unreachable")
    if mode == "collect" and fallback is not None and not fallback_reachable:
        raise ValueError("collect fallback is unreachable")
    for field_index, name in enumerate(fields):
        bit = 1 << field_index
        if not any(outcomes[index] != outcomes[index | bit]
                   for index in range(1 << len(fields)) if not index & bit):
            raise ValueError(f"policy field is not outcome-causal: {name!r}")
    return CompiledPolicy(fields, mode, tuple(rules), fallback, tuple(outcomes))


def _validate_compiled_condition(condition: Any, fields: frozenset[str],
                                 referenced: set[str]) -> None:
    if not isinstance(condition, tuple) or len(condition) != 2:
        raise ValueError("compiled policy contains a malformed condition")
    operator, operand = condition
    if operator == "field":
        if not isinstance(operand, str) or operand not in fields:
            raise ValueError("compiled policy condition has an undeclared field")
        referenced.add(operand)
        return
    if operator == "const":
        if type(operand) is not bool:
            raise ValueError("compiled policy const condition must be Boolean")
        return
    if operator == "not":
        _validate_compiled_condition(operand, fields, referenced)
        return
    if operator not in {"all", "any"} or not isinstance(operand, tuple) or not operand:
        raise ValueError("compiled policy contains an unsupported condition")
    for item in operand:
        _validate_compiled_condition(item, fields, referenced)


def _validate_compiled_policy(policy: CompiledPolicy) -> None:
    """Fail closed when callers directly construct the public frozen dataclass."""
    fields = policy.fields
    if (not isinstance(fields, tuple) or not MIN_FIELDS <= len(fields) <= MAX_FIELDS
            or any(not isinstance(name, str) or not name.strip() or name != name.strip()
                   for name in fields)
            or len(set(fields)) != len(fields)):
        raise ValueError("compiled policy has invalid fields")
    if not isinstance(policy.mode, str) or policy.mode not in {"priority", "collect"}:
        raise ValueError("compiled policy has an invalid mode")
    if not isinstance(policy.rules, tuple) or not policy.rules:
        raise ValueError("compiled policy has invalid rules")
    if (policy.fallback is not None and
            not isinstance(policy.fallback, CanonicalAction)):
        raise ValueError("compiled policy has an invalid fallback")
    if policy.mode == "priority" and policy.fallback is None:
        raise ValueError("compiled priority policy requires a fallback")
    expected_outcomes = 1 << len(fields)
    if (not isinstance(policy.truth_outcomes, tuple)
            or len(policy.truth_outcomes) != expected_outcomes
            or any(not isinstance(outcome, CanonicalOutcome)
                   for outcome in policy.truth_outcomes)):
        raise ValueError("compiled policy has an invalid truth-outcome table")

    declared = frozenset(fields)
    referenced: set[str] = set()
    actions = []
    ranks = []
    for rule in policy.rules:
        if not isinstance(rule, _Rule) or not isinstance(rule.action, CanonicalAction):
            raise ValueError("compiled policy has an invalid rule")
        _validate_compiled_condition(rule.condition, declared, referenced)
        if policy.mode == "priority":
            if rule.primary_rank is not None:
                raise ValueError("compiled priority rule cannot have a primary rank")
        else:
            rank = rule.primary_rank
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ValueError("compiled collect rule has an invalid primary rank")
            ranks.append(rank)
        actions.append(rule.action)
    if referenced != declared:
        raise ValueError("compiled policy has unused fields")
    if len(set(actions)) != len(actions):
        raise ValueError("compiled policy has duplicate rule actions")
    if policy.mode == "collect" and len(set(ranks)) != len(ranks):
        raise ValueError("compiled policy has duplicate primary ranks")
    if policy.fallback is not None and policy.fallback in actions:
        raise ValueError("compiled policy fallback duplicates a rule action")

    reachable = [False] * len(policy.rules)
    fallback_reachable = False
    recomputed = []
    for bits in range(expected_outcomes):
        assignment = {name: bool(bits & (1 << index))
                      for index, name in enumerate(fields)}
        matches = [_condition_value(rule.condition, assignment) for rule in policy.rules]
        if policy.mode == "priority":
            if any(matches):
                reachable[matches.index(True)] = True
            else:
                fallback_reachable = True
        else:
            for index, matches_rule in enumerate(matches):
                reachable[index] |= matches_rule
            if not any(matches):
                fallback_reachable = True
                if policy.fallback is None:
                    raise ValueError("compiled collect policy has an uncovered assignment")
        recomputed.append(_execute_compiled(policy, assignment))
    if not all(reachable):
        raise ValueError("compiled policy has unreachable rules")
    if policy.mode == "priority" and not fallback_reachable:
        raise ValueError("compiled priority fallback is unreachable")
    if policy.mode == "collect" and policy.fallback is not None and not fallback_reachable:
        raise ValueError("compiled collect fallback is unreachable")
    if tuple(recomputed) != policy.truth_outcomes:
        raise ValueError("compiled policy truth-outcome table is inconsistent")
    for field_index, name in enumerate(fields):
        bit = 1 << field_index
        if not any(recomputed[index] != recomputed[index | bit]
                   for index in range(expected_outcomes) if not index & bit):
            raise ValueError(f"compiled policy field is not outcome-causal: {name!r}")


def _as_compiled(policy: CompiledPolicy | Mapping[str, Any]) -> CompiledPolicy:
    if isinstance(policy, CompiledPolicy):
        _validate_compiled_policy(policy)
        return policy
    return compile_policy(policy)


def execute_policy(policy: CompiledPolicy | Mapping[str, Any],
                   assignment: Mapping[str, bool]) -> CanonicalOutcome:
    """Execute a policy on one exact semantic truth assignment."""
    policy = _as_compiled(policy)
    if not isinstance(assignment, Mapping) or set(assignment) != set(policy.fields):
        raise ValueError("assignment fields must exactly match policy fields")
    if any(type(value) is not bool for value in assignment.values()):
        raise ValueError("assignment values must be Boolean")
    return _execute_compiled(policy, assignment)


def _validate_truth_logits(policy: CompiledPolicy,
                           truth_logits: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(truth_logits, Mapping) or set(truth_logits) != set(policy.fields):
        raise ValueError("truth logits must exactly match policy fields")
    result = {}
    device = None
    for name in policy.fields:
        value = truth_logits[name]
        if (not isinstance(value, torch.Tensor) or value.ndim != 0
                or not value.is_floating_point()):
            raise ValueError(f"truth logit {name!r} must be a floating scalar tensor")
        if not bool(torch.isfinite(value.detach()).item()):
            raise ValueError(f"truth logit {name!r} must be finite")
        if device is None:
            device = value.device
        elif value.device != device:
            raise ValueError("all truth logits must use the same device")
        result[name] = value.float()
    return result


def exact_outcome_log_distribution(
    policy: CompiledPolicy | Mapping[str, Any],
    truth_logits: Mapping[str, torch.Tensor],
) -> dict[CanonicalOutcome, torch.Tensor]:
    """Exactly marginalize independent field logits into policy outcomes.

    ``truth_logits`` are semantic true-vs-false logits keyed by field name, so
    option positions never enter this interface.  Returned values are scalar
    float32 log probabilities and remain differentiable with respect to every
    input logit.
    """
    policy = _as_compiled(policy)
    logits = _validate_truth_logits(policy, truth_logits)
    stacked = torch.stack([logits[name] for name in policy.fields])
    assignment_ids = torch.arange(1 << len(policy.fields), device=stacked.device)
    bit_positions = torch.arange(len(policy.fields), device=stacked.device)
    bits = ((assignment_ids[:, None] >> bit_positions[None, :]) & 1).bool()
    log_true = F.logsigmoid(stacked)
    log_false = F.logsigmoid(-stacked)
    assignment_log_probabilities = torch.where(bits, log_true, log_false).sum(dim=1)

    indices_by_outcome: dict[CanonicalOutcome, list[int]] = defaultdict(list)
    for index, outcome in enumerate(policy.truth_outcomes):
        indices_by_outcome[outcome].append(index)
    result = {}
    for outcome in sorted(indices_by_outcome):
        indices = torch.tensor(indices_by_outcome[outcome], device=stacked.device)
        result[outcome] = torch.logsumexp(
            assignment_log_probabilities.index_select(0, indices), dim=0
        )
    log_total = torch.logsumexp(torch.stack(tuple(result.values())), dim=0)
    if not bool(torch.isfinite(log_total.detach()).item()):
        raise ValueError("outcome log distribution is not finite")
    # Renormalization removes only accumulated floating-point error and gives
    # callers an exact log distribution even for large truth tables.
    return {outcome: value - log_total for outcome, value in result.items()}


def exact_outcome_distribution(
    policy: CompiledPolicy | Mapping[str, Any],
    truth_logits: Mapping[str, torch.Tensor],
) -> dict[CanonicalOutcome, torch.Tensor]:
    return {outcome: value.exp() for outcome, value in
            exact_outcome_log_distribution(policy, truth_logits).items()}


def _validate_log_distribution(
    log_distribution: Mapping[CanonicalOutcome, torch.Tensor],
) -> dict[CanonicalOutcome, torch.Tensor]:
    if not isinstance(log_distribution, Mapping) or not log_distribution:
        raise ValueError("outcome log distribution must be a nonempty mapping")
    result = {}
    device = None
    for outcome, value in log_distribution.items():
        if not isinstance(outcome, CanonicalOutcome):
            raise ValueError("outcome log distribution has a noncanonical key")
        if (not isinstance(value, torch.Tensor) or value.ndim != 0
                or not value.is_floating_point()
                or not bool(torch.isfinite(value.detach()).item())):
            raise ValueError("outcome log probabilities must be finite floating scalars")
        if device is None:
            device = value.device
        elif value.device != device:
            raise ValueError("outcome log probabilities must use the same device")
        result[outcome] = value
    total = torch.logsumexp(torch.stack(tuple(result.values())), dim=0).detach()
    if not math.isclose(float(total.item()), 0.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("outcome log probabilities must be normalized")
    return result


def outcome_nll(log_distribution: Mapping[CanonicalOutcome, torch.Tensor],
                gold_outcome: CanonicalOutcome | Mapping[str, Any]) -> torch.Tensor:
    """Negative log likelihood of one complete primary/action-set outcome."""
    distribution = _validate_log_distribution(log_distribution)
    gold = canonicalize_outcome(gold_outcome)
    if gold not in distribution:
        raise ValueError("gold outcome is structurally impossible under the policy")
    return -distribution[gold]


def primary_action_nll(log_distribution: Mapping[CanonicalOutcome, torch.Tensor],
                       gold_action: CanonicalAction | Mapping[str, Any]) -> torch.Tensor:
    """NLL after marginalizing all outcomes with the same primary action."""
    distribution = _validate_log_distribution(log_distribution)
    gold = canonicalize_action(gold_action)
    matches = [value for outcome, value in distribution.items() if outcome.primary == gold]
    if not matches:
        raise ValueError("gold primary action is structurally impossible under the policy")
    return -torch.logsumexp(torch.stack(matches), dim=0)


def action_set_nll(log_distribution: Mapping[CanonicalOutcome, torch.Tensor],
                   gold_actions: Sequence[CanonicalAction | Mapping[str, Any]]) -> torch.Tensor:
    """NLL of an exact action set, marginalizing over its possible primaries."""
    distribution = _validate_log_distribution(log_distribution)
    if (not isinstance(gold_actions, Sequence) or isinstance(gold_actions, (str, bytes))
            or not gold_actions):
        raise ValueError("gold actions must be a nonempty array")
    actions = tuple(sorted(canonicalize_action(action) for action in gold_actions))
    if len(set(actions)) != len(actions):
        raise ValueError("duplicate gold actions are not allowed")
    matches = [value for outcome, value in distribution.items() if outcome.actions == actions]
    if not matches:
        raise ValueError("gold action set is structurally impossible under the policy")
    return -torch.logsumexp(torch.stack(matches), dim=0)
