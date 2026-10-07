#!/usr/bin/env python3
"""Deterministically replay the frozen R052 typed-policy graph.

This executor is intentionally small and hostile to ambiguous inputs.  It reads
canonical JSON, uses only exact rational arithmetic, and treats missing evidence
as unknown rather than false.  It does not execute Python expressions or call a
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

LEDGER_VERSION = "r052-fact-ledger-v1"
RUBRIC_VERSION = "r052-policy-rubric-v1"
RECEIPT_VERSION = "r052-policy-replay-receipt-v1"
SCHEMA_VERSION = "r052-policy-replay-schema-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
FORBIDDEN_FACT_WORDS = re.compile(r"(?:gold|answer|correct)", re.IGNORECASE)
MAX_AST_DEPTH = 16
MAX_AST_NODES = 256
MAX_AST_ARGS = 64
MAX_REMOVAL_CHUNKS = 12


class PolicyError(ValueError):
    """An input violates the frozen replay contract."""


class Signal(Enum):
    MISSING = "missing"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class TypedValue:
    type: str
    unit: str | None
    value: bool | str | Fraction


@dataclass(frozen=True)
class DecisionResult:
    status: str
    output: str | None
    rules: tuple[tuple[str, bool | Signal], ...]


def canonical_json(value: Any) -> bytes:
    """Return the one accepted JSON encoding."""
    _reject_non_json_numbers(value, "JSON")
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _reject_non_json_numbers(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PolicyError(f"{label} must not contain non-finite numbers")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PolicyError(f"{label} object keys must be strings")
            _reject_non_json_numbers(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_non_json_numbers(item, label)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise PolicyError(f"{label} contains a non-JSON value")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_canonical_json(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PolicyError(f"{label} contains non-finite constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyError(f"{label} must be valid UTF-8 JSON") from exc
    _reject_non_json_numbers(value, label)
    if raw != canonical_json(value):
        raise PolicyError(f"{label} must use canonical JSON bytes")
    return value


def load_canonical_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[Any, str]:
    expected_sha256 = _sha(expected_sha256, f"expected {label} sha256")
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise PolicyError(f"{label} must be a non-empty regular file")
    raw = path.read_bytes()
    observed = sha256_bytes(raw)
    if observed != expected_sha256:
        raise PolicyError(f"{label} differs from its external sha256 expectation")
    return parse_canonical_json(raw, label), observed


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PolicyError(f"{label} keys must be exactly {sorted(keys)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"{label} must be non-empty text")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _text(value, label)
    if IDENTIFIER_RE.fullmatch(value) is None:
        raise PolicyError(f"{label} is not an opaque identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PolicyError(f"{label} must be a lowercase SHA-256")
    return value


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise PolicyError(
            f"{label} must be {'a non-empty' if nonempty else 'an'} array"
        )
    result = [_identifier(item, f"{label} item") for item in value]
    if len(result) != len(set(result)):
        raise PolicyError(f"{label} must not contain duplicates")
    return result


def _fact_declaration(value: Any, label: str) -> tuple[str, str | None]:
    value = _exact(value, {"type", "unit"}, label)
    fact_type = value["type"]
    if fact_type not in {"boolean", "integer", "decimal", "string"}:
        raise PolicyError(f"{label}.type is invalid")
    unit = value["unit"]
    if unit is not None:
        unit = _text(unit, f"{label}.unit")
    if fact_type in {"boolean", "string"} and unit is not None:
        raise PolicyError(f"{label} non-numeric facts cannot have a unit")
    return fact_type, unit


def _typed_value(
    raw: Any, declaration: tuple[str, str | None], label: str
) -> TypedValue:
    fact_type, unit = declaration
    if fact_type == "boolean":
        if type(raw) is not bool:
            raise PolicyError(f"{label} must be boolean")
        parsed: bool | str | Fraction = raw
    elif fact_type == "integer":
        if type(raw) is not int:
            raise PolicyError(f"{label} must be an integer")
        parsed = Fraction(raw)
    elif fact_type == "decimal":
        if not isinstance(raw, str) or DECIMAL_RE.fullmatch(raw) is None:
            raise PolicyError(f"{label} must be a canonical decimal string")
        parsed = Fraction(raw)
    else:
        parsed = _text(raw, label)
    return TypedValue(fact_type, unit, parsed)


def _panel_index(
    panel: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    if not isinstance(panel, dict) or not isinstance(panel.get("cases"), list):
        raise PolicyError("panel must contain a cases array")
    cases: dict[str, dict[str, Any]] = {}
    chunks: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for case in panel["cases"]:
        if not isinstance(case, dict):
            raise PolicyError("panel cases must be objects")
        case_key = _identifier(case.get("case_key"), "panel case_key")
        if case_key in cases:
            raise PolicyError("panel case keys must be unique")
        if not isinstance(case.get("chunks"), list) or not isinstance(
            case.get("questions"), list
        ):
            raise PolicyError(f"panel case {case_key} lacks chunks/questions")
        cases[case_key] = case
        local_chunks: set[str] = set()
        for chunk in case["chunks"]:
            if not isinstance(chunk, dict):
                raise PolicyError("panel chunks must be objects")
            key = _identifier(chunk.get("chunk_key"), "panel chunk_key")
            if key in local_chunks or key in chunks:
                raise PolicyError("panel chunk keys must be globally unique")
            local_chunks.add(key)
            content_hash = _sha(
                chunk.get("content_sha256"), "panel chunk content_sha256"
            )
            required_chunk_fields = {
                "kind",
                "modality",
                "text",
                "token_start",
                "token_end",
                "media",
                "derivation",
            }
            if not required_chunk_fields <= set(chunk):
                raise PolicyError("panel chunk lacks content-hash fields")
            media = chunk["media"]
            media_payload = None
            if media is not None:
                if not isinstance(media, dict):
                    raise PolicyError("panel chunk media must be an object or null")
                media_payload = {
                    "mime_type": media.get("mime_type"),
                    "sha256": media.get("sha256"),
                }
            chunk_payload = {
                "kind": chunk["kind"],
                "modality": chunk["modality"],
                "text": chunk["text"],
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "media": media_payload,
                "derivation": chunk["derivation"],
            }
            if sha256_json(chunk_payload) != content_hash:
                raise PolicyError("panel chunk content_sha256 is not content-derived")
            if not isinstance(chunk.get("immutable"), bool):
                raise PolicyError("panel chunk immutable must be boolean")
            chunks[key] = (case, chunk)
        local_questions: set[str] = set()
        for question in case["questions"]:
            if not isinstance(question, dict):
                raise PolicyError("panel questions must be objects")
            key = _identifier(question.get("question_key"), "panel question_key")
            if key in local_questions:
                raise PolicyError("panel question keys must be unique within a case")
            local_questions.add(key)
            content_hash = _sha(
                question.get("content_sha256"), "panel question content_sha256"
            )
            required_question_fields = {
                "text",
                "options",
                "decision_type",
                "policy_role",
            }
            if not required_question_fields <= set(question):
                raise PolicyError("panel question lacks content-hash fields")
            question_payload = {
                "text": question["text"],
                "options": question["options"],
                "decision_type": question["decision_type"],
                "policy_role": question["policy_role"],
            }
            if sha256_json(question_payload) != content_hash:
                raise PolicyError(
                    "panel question content_sha256 is not content-derived"
                )
    return cases, chunks


def validate_ledger(
    ledger: Any,
    panel: Any,
    panel_sha256: str,
) -> tuple[dict[str, tuple[str, str | None]], list[dict[str, Any]]]:
    ledger = _exact(
        ledger,
        {"schema_version", "panel_sha256", "facts", "bindings"},
        "fact ledger",
    )
    if (
        ledger["schema_version"] != LEDGER_VERSION
        or ledger["panel_sha256"] != panel_sha256
    ):
        raise PolicyError("fact ledger version or panel binding is invalid")
    facts_raw = ledger["facts"]
    if not isinstance(facts_raw, dict) or not facts_raw:
        raise PolicyError("fact ledger facts must be a non-empty object")
    facts: dict[str, tuple[str, str | None]] = {}
    for fact_id, declaration in facts_raw.items():
        _identifier(fact_id, "fact id")
        if FORBIDDEN_FACT_WORDS.search(fact_id):
            raise PolicyError("fact ids must be opaque and question-blind")
        facts[fact_id] = _fact_declaration(declaration, f"fact {fact_id}")

    cases, chunks = _panel_index(panel)
    bindings = ledger["bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise PolicyError("fact ledger bindings must be a non-empty array")
    seen_bindings: set[tuple[str, str]] = set()
    observed_facts: set[str] = set()
    for index, binding in enumerate(bindings):
        label = f"binding {index}"
        binding = _exact(
            binding,
            {
                "case_key",
                "chunk_key",
                "chunk_content_sha256",
                "observations",
                "adjudication_receipt_sha256",
            },
            label,
        )
        case_key = _identifier(binding["case_key"], f"{label}.case_key")
        chunk_key = _identifier(binding["chunk_key"], f"{label}.chunk_key")
        pair = (case_key, chunk_key)
        if pair in seen_bindings:
            raise PolicyError("ledger has duplicate case/chunk bindings")
        seen_bindings.add(pair)
        if (
            case_key not in cases
            or chunk_key not in chunks
            or chunks[chunk_key][0] is not cases[case_key]
        ):
            raise PolicyError(f"{label} references a foreign case/chunk pair")
        chunk = chunks[chunk_key][1]
        if binding["chunk_content_sha256"] != chunk["content_sha256"]:
            raise PolicyError(f"{label} chunk content hash mismatch")
        _sha(binding["chunk_content_sha256"], f"{label}.chunk_content_sha256")
        _sha(binding["adjudication_receipt_sha256"], f"{label}.adjudication receipt")
        observations = binding["observations"]
        if not isinstance(observations, list) or not observations:
            raise PolicyError(f"{label}.observations must be a non-empty array")
        seen_local: set[str] = set()
        for number, observation in enumerate(observations):
            obs_label = f"{label}.observation {number}"
            observation = _exact(observation, {"fact", "value", "locator"}, obs_label)
            fact_id = _identifier(observation["fact"], f"{obs_label}.fact")
            if fact_id not in facts or fact_id in seen_local:
                raise PolicyError(f"{obs_label} fact is unknown or duplicated")
            seen_local.add(fact_id)
            observed_facts.add(fact_id)
            _typed_value(observation["value"], facts[fact_id], f"{obs_label}.value")
            locator = observation["locator"]
            if not isinstance(locator, dict):
                raise PolicyError(f"{obs_label}.locator must be an object")
            if locator.get("kind") == "text_span":
                _exact(
                    locator,
                    {"kind", "char_start", "char_end", "span_sha256"},
                    f"{obs_label}.locator",
                )
                start, end = locator["char_start"], locator["char_end"]
                text = chunk.get("text")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or start < 0
                    or end <= start
                    or not isinstance(text, str)
                    or end > len(text)
                ):
                    raise PolicyError(f"{obs_label} text span is invalid")
                digest = _sha(locator["span_sha256"], f"{obs_label}.span_sha256")
                if digest != sha256_bytes(text[start:end].encode("utf-8")):
                    raise PolicyError(f"{obs_label} text span hash mismatch")
            elif locator.get("kind") == "image_region":
                _exact(locator, {"kind", "region_sha256"}, f"{obs_label}.locator")
                _sha(locator["region_sha256"], f"{obs_label}.region_sha256")
                if chunk.get("modality") != "image" or chunk.get("media") is None:
                    raise PolicyError(
                        f"{obs_label} image locator requires real image bytes"
                    )
            else:
                raise PolicyError(f"{obs_label}.locator kind is invalid")
    if observed_facts != set(facts):
        raise PolicyError("every declared fact must have at least one binding")
    return facts, bindings


def _validate_ast(
    node: Any,
    allowed_facts: set[str],
    declarations: dict[str, tuple[str, str | None]],
    state: list[int],
    depth: int = 1,
) -> None:
    if depth > MAX_AST_DEPTH:
        raise PolicyError("policy AST exceeds the maximum depth")
    state[0] += 1
    if state[0] > MAX_AST_NODES:
        raise PolicyError("policy AST exceeds the maximum node count")
    if not isinstance(node, dict) or not isinstance(node.get("op"), str):
        raise PolicyError("policy AST nodes must be operator objects")
    op = node["op"]
    if op == "fact":
        _exact(node, {"op", "id"}, "fact AST node")
        fact_id = _identifier(node["id"], "AST fact id")
        if fact_id not in allowed_facts:
            raise PolicyError("policy rules may reference only declared group fact ids")
    elif op == "const":
        _exact(node, {"op", "type", "unit", "value"}, "const AST node")
        declaration = _fact_declaration(
            {"type": node["type"], "unit": node["unit"]}, "AST constant"
        )
        _typed_value(node["value"], declaration, "AST constant value")
    elif op in {"eq", "lt", "le", "gt", "ge"}:
        _exact(node, {"op", "left", "right"}, f"{op} AST node")
        _validate_ast(node["left"], allowed_facts, declarations, state, depth + 1)
        _validate_ast(node["right"], allowed_facts, declarations, state, depth + 1)
    elif op == "not":
        _exact(node, {"op", "arg"}, "not AST node")
        _validate_ast(node["arg"], allowed_facts, declarations, state, depth + 1)
    elif op in {"all", "any", "count_true"}:
        _exact(node, {"op", "args"}, f"{op} AST node")
        args = node["args"]
        if not isinstance(args, list) or not args or len(args) > MAX_AST_ARGS:
            raise PolicyError(f"{op} requires 1..{MAX_AST_ARGS} arguments")
        for arg in args:
            _validate_ast(arg, allowed_facts, declarations, state, depth + 1)
    elif op == "ratio":
        _exact(node, {"op", "numerator", "denominator"}, "ratio AST node")
        _validate_ast(node["numerator"], allowed_facts, declarations, state, depth + 1)
        _validate_ast(
            node["denominator"], allowed_facts, declarations, state, depth + 1
        )
    else:
        raise PolicyError(f"unsupported policy AST operator: {op}")


def _fact_environment(
    facts: dict[str, tuple[str, str | None]],
    bindings: list[dict[str, Any]],
    case_key: str,
    visible_chunks: set[str],
) -> dict[str, TypedValue | Signal]:
    values: dict[str, list[TypedValue]] = {fact_id: [] for fact_id in facts}
    for binding in bindings:
        if (
            binding["case_key"] != case_key
            or binding["chunk_key"] not in visible_chunks
        ):
            continue
        for observation in binding["observations"]:
            fact_id = observation["fact"]
            values[fact_id].append(
                _typed_value(observation["value"], facts[fact_id], f"fact {fact_id}")
            )
    environment: dict[str, TypedValue | Signal] = {}
    for fact_id, candidates in values.items():
        if not candidates:
            environment[fact_id] = Signal.MISSING
        elif any(candidate != candidates[0] for candidate in candidates[1:]):
            environment[fact_id] = Signal.CONFLICT
        else:
            environment[fact_id] = candidates[0]
    return environment


def _as_bool(value: Any) -> bool | Signal:
    if isinstance(value, Signal):
        return value
    if type(value) is bool:
        return value
    if not isinstance(value, TypedValue) or value.type != "boolean":
        return Signal.CONFLICT
    return bool(value.value)


def _comparable(left: TypedValue, right: TypedValue, ordered: bool) -> bool:
    numeric = {left.type, right.type} <= {"integer", "decimal"}
    if numeric:
        return left.unit == right.unit
    return not ordered and left.type == right.type and left.unit == right.unit


def _eval(
    node: dict[str, Any], env: dict[str, TypedValue | Signal]
) -> TypedValue | bool | Signal:
    op = node["op"]
    if op == "fact":
        return env[node["id"]]
    if op == "const":
        return _typed_value(node["value"], (node["type"], node["unit"]), "constant")
    if op in {"eq", "lt", "le", "gt", "ge"}:
        left, right = _eval(node["left"], env), _eval(node["right"], env)
        if Signal.CONFLICT in {left, right}:
            return Signal.CONFLICT
        if Signal.MISSING in {left, right}:
            return Signal.MISSING
        if not isinstance(left, TypedValue) or not isinstance(right, TypedValue):
            return Signal.CONFLICT
        if not _comparable(left, right, op != "eq"):
            return Signal.CONFLICT
        operations = {
            "eq": lambda: left.value == right.value,
            "lt": lambda: left.value < right.value,
            "le": lambda: left.value <= right.value,
            "gt": lambda: left.value > right.value,
            "ge": lambda: left.value >= right.value,
        }
        return operations[op]()
    if op == "not":
        value = _as_bool(_eval(node["arg"], env))
        return value if isinstance(value, Signal) else not value
    if op in {"all", "any"}:
        values = [_as_bool(_eval(arg, env)) for arg in node["args"]]
        if Signal.CONFLICT in values:
            return Signal.CONFLICT
        decisive = op != "all"
        if decisive in values:
            return decisive
        if Signal.MISSING in values:
            return Signal.MISSING
        return not decisive
    if op == "count_true":
        values = [_as_bool(_eval(arg, env)) for arg in node["args"]]
        if Signal.CONFLICT in values:
            return Signal.CONFLICT
        if Signal.MISSING in values:
            return Signal.MISSING
        return TypedValue(
            "integer", None, Fraction(sum(value is True for value in values))
        )
    if op == "ratio":
        numerator = _eval(node["numerator"], env)
        denominator = _eval(node["denominator"], env)
        if Signal.CONFLICT in {numerator, denominator}:
            return Signal.CONFLICT
        if Signal.MISSING in {numerator, denominator}:
            return Signal.MISSING
        if (
            not isinstance(numerator, TypedValue)
            or not isinstance(denominator, TypedValue)
            or numerator.type not in {"integer", "decimal"}
            or denominator.type not in {"integer", "decimal"}
            or numerator.unit != denominator.unit
            or denominator.value == 0
        ):
            return Signal.CONFLICT
        return TypedValue("decimal", None, numerator.value / denominator.value)
    raise AssertionError(op)


def _resolve(
    decision: dict[str, Any], env: dict[str, TypedValue | Signal]
) -> DecisionResult:
    results: list[tuple[str, bool | Signal]] = []
    for rule in decision["rules"]:
        results.append((rule["output"], _as_bool(_eval(rule["when"], env))))
    values = [value for _, value in results]
    if Signal.CONFLICT in values or sum(value is True for value in values) > 1:
        return DecisionResult("invalid", None, tuple(results))
    true_outputs = [output for output, value in results if value is True]
    if len(true_outputs) == 1 and all(
        value is False for _, value in results if _ != true_outputs[0]
    ):
        return DecisionResult("resolved", true_outputs[0], tuple(results))
    return DecisionResult("unresolved", None, tuple(results))


def classify_transition(full: DecisionResult, removed: DecisionResult) -> str:
    if full.status == "invalid" or removed.status == "invalid":
        return "invalid"
    if full.status == "resolved" and removed.status == "resolved":
        return "no_change" if full.output == removed.output else "label_flip"
    if full.status == "resolved" and removed.status == "unresolved":
        return "resolution_loss"
    return "no_change" if full == removed else "invalid"


def _result_json(result: DecisionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "output": result.output,
        "rules": [
            {
                "output": output,
                "truth": value.value if isinstance(value, Signal) else value,
            }
            for output, value in sorted(result.rules)
        ],
    }


def _environment_json(
    env: dict[str, TypedValue | Signal], allowed: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fact_id in sorted(allowed):
        value = env[fact_id]
        if isinstance(value, Signal):
            result[fact_id] = {"status": value.value}
        else:
            encoded: Any = value.value
            if isinstance(encoded, Fraction):
                encoded = f"{encoded.numerator}/{encoded.denominator}"
            result[fact_id] = {
                "status": "known",
                "type": value.type,
                "unit": value.unit,
                "value": encoded,
            }
    return result


def _assert_result(result: DecisionResult, assertion: Any, label: str) -> None:
    if not isinstance(assertion, dict):
        raise PolicyError(f"{label} must be an object")
    status = assertion.get("status")
    if status == "resolved":
        _exact(assertion, {"status", "output"}, label)
        if result.status != "resolved" or result.output != assertion["output"]:
            raise PolicyError(f"{label} does not match replay")
    elif status == "unresolved":
        _exact(assertion, {"status"}, label)
        if result.status != "unresolved":
            raise PolicyError(f"{label} does not match replay")
    else:
        raise PolicyError(f"{label} may assert only resolved or unresolved")


def _question(case: dict[str, Any], key: str) -> dict[str, Any]:
    matches = [
        question
        for question in case["questions"]
        if question.get("question_key") == key
    ]
    if len(matches) != 1:
        raise PolicyError("rubric question binding must resolve exactly once")
    return matches[0]


def _ordered_subset(full: list[str], included: set[str]) -> list[str]:
    return [item for item in full if item in included]


def _validate_images(
    cases: dict[str, dict[str, Any]], bindings: list[dict[str, Any]]
) -> None:
    bound_chunks = {binding["chunk_key"] for binding in bindings}
    for case in cases.values():
        groups: dict[str, list[dict[str, Any]]] = {}
        for chunk in case["chunks"]:
            control = chunk.get("image_control")
            if control is None:
                continue
            if not isinstance(control, dict) or set(control) != {
                "group_key",
                "variant",
            }:
                raise PolicyError(
                    "image controls must contain exactly group_key and variant"
                )
            groups.setdefault(
                _identifier(control["group_key"], "image group key"), []
            ).append(chunk)
        for chunks in groups.values():
            by_variant = {chunk["image_control"]["variant"]: chunk for chunk in chunks}
            if set(by_variant) != {"real", "shuffled", "missing"} or len(chunks) != 3:
                raise PolicyError(
                    "image control groups require one real, shuffled, and missing chunk"
                )
            if len({chunk.get("text") for chunk in chunks}) != 1:
                raise PolicyError(
                    "image control text must be neutral and identical across variants"
                )
            if (
                by_variant["real"].get("media") is None
                or by_variant["shuffled"].get("media") is None
            ):
                raise PolicyError("real/shuffled image controls require image bytes")
            if by_variant["missing"].get("media") is not None:
                raise PolicyError("missing image controls cannot carry image bytes")
            if (
                by_variant["shuffled"]["chunk_key"] in bound_chunks
                or by_variant["missing"]["chunk_key"] in bound_chunks
            ):
                raise PolicyError(
                    "shuffled/missing image controls must not bind policy facts"
                )


def _validate_hebrew(
    groups: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    traces: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for group in groups:
        presentations = group["presentations"]
        for presentation in presentations:
            case = cases[presentation["case_key"]]
            if case.get("language") != "he":
                continue
            twin = case.get("twin_group_key")
            english = [
                other
                for other in presentations
                if cases[other["case_key"]].get("language") == "en"
                and cases[other["case_key"]].get("twin_group_key") == twin
            ]
            if len(english) != 1:
                raise PolicyError(
                    "every Hebrew presentation requires exactly one English semantic twin"
                )
            he_trace = traces[(group["group_id"], presentation["presentation_id"])]
            en_trace = traces[(group["group_id"], english[0]["presentation_id"])]
            if he_trace["full_fact_state"] != en_trace["full_fact_state"]:
                raise PolicyError("Hebrew/English twins disagree on full fact values")
            he_chunks = {chunk.get("twin_anchor"): chunk for chunk in case["chunks"]}
            en_case = cases[english[0]["case_key"]]
            en_chunks = {chunk.get("twin_anchor"): chunk for chunk in en_case["chunks"]}
            if (
                None in he_chunks
                or None in en_chunks
                or set(he_chunks) != set(en_chunks)
            ):
                raise PolicyError(
                    "Hebrew/English chunk semantic graph is not one-to-one"
                )
            for anchor, he_chunk in he_chunks.items():
                en_chunk = en_chunks[anchor]
                if any(
                    he_chunk.get(field) != en_chunk.get(field)
                    for field in ("kind", "modality", "immutable")
                ):
                    raise PolicyError("Hebrew/English chunk semantic graph drift")
                he_control, en_control = (
                    he_chunk.get("image_control"),
                    en_chunk.get("image_control"),
                )
                if (he_control is None) != (en_control is None) or (
                    he_control is not None
                    and he_control.get("variant") != en_control.get("variant")
                ):
                    raise PolicyError("Hebrew/English image-control graph drift")
            for decision_id, binding in presentation["question_bindings"].items():
                he_question = _question(case, binding["question_key"])
                en_binding = english[0]["question_bindings"].get(decision_id)
                if en_binding is None:
                    raise PolicyError(
                        "Hebrew/English twins must bind identical decisions"
                    )
                en_question = _question(en_case, en_binding["question_key"])
                if (
                    he_question.get("twin_anchor") != en_question.get("twin_anchor")
                    or he_question.get("decision_type")
                    != en_question.get("decision_type")
                    or he_question.get("policy_role") != en_question.get("policy_role")
                    or len(he_question.get("options", []))
                    != len(en_question.get("options", []))
                ):
                    raise PolicyError("Hebrew/English question semantic anchors drift")


def replay(
    panel: Any,
    ledger: Any,
    rubric: Any,
    *,
    panel_sha256: str,
    ledger_sha256: str,
    rubric_sha256: str,
    evaluator_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    """Validate and replay all frozen presentations, returning a deterministic receipt."""
    for value, label in (
        (panel_sha256, "panel"),
        (ledger_sha256, "fact ledger"),
        (rubric_sha256, "rubric"),
        (evaluator_sha256, "evaluator"),
        (schema_sha256, "schema"),
    ):
        _sha(value, f"{label} sha256")
    if (
        sha256_json(panel) != panel_sha256
        or sha256_json(ledger) != ledger_sha256
        or sha256_json(rubric) != rubric_sha256
    ):
        raise PolicyError("in-memory artifact does not match its supplied content hash")
    facts, bindings = validate_ledger(ledger, panel, panel_sha256)
    cases, _ = _panel_index(panel)
    _validate_images(cases, bindings)
    rubric = _exact(
        rubric,
        {"schema_version", "panel_sha256", "fact_ledger_sha256", "semantic_groups"},
        "policy rubric",
    )
    if (
        rubric["schema_version"] != RUBRIC_VERSION
        or rubric["panel_sha256"] != panel_sha256
        or rubric["fact_ledger_sha256"] != ledger_sha256
    ):
        raise PolicyError("policy rubric artifact bindings are invalid")
    groups = rubric["semantic_groups"]
    if not isinstance(groups, list) or not groups:
        raise PolicyError("policy rubric semantic_groups must be non-empty")
    seen_groups: set[str] = set()
    seen_presentations: set[str] = set()
    receipt_groups: list[dict[str, Any]] = []
    twin_traces: dict[tuple[str, str], dict[str, Any]] = {}

    for group in groups:
        group = _exact(
            group, {"group_id", "facts", "decisions", "presentations"}, "semantic group"
        )
        group_id = _identifier(group["group_id"], "semantic group id")
        if group_id in seen_groups:
            raise PolicyError("semantic group ids must be unique")
        seen_groups.add(group_id)
        group_facts_raw = group["facts"]
        if not isinstance(group_facts_raw, dict) or not group_facts_raw:
            raise PolicyError("semantic group facts must be a non-empty object")
        group_facts: set[str] = set()
        for fact_id, declaration in group_facts_raw.items():
            _identifier(fact_id, "semantic group fact id")
            if (
                fact_id not in facts
                or _fact_declaration(declaration, f"group fact {fact_id}")
                != facts[fact_id]
            ):
                raise PolicyError(
                    "semantic group fact declarations must exactly match the ledger"
                )
            group_facts.add(fact_id)

        decisions_raw = group["decisions"]
        if not isinstance(decisions_raw, list) or not decisions_raw:
            raise PolicyError("semantic group decisions must be a non-empty array")
        decisions: dict[str, dict[str, Any]] = {}
        for decision in decisions_raw:
            decision = _exact(
                decision,
                {"decision_id", "type", "outputs", "order", "rules"},
                "decision",
            )
            decision_id = _identifier(decision["decision_id"], "decision id")
            if decision_id in decisions:
                raise PolicyError("decision ids must be unique within a semantic group")
            if decision["type"] not in {"choice", "noul", "score"}:
                raise PolicyError("decision type is invalid")
            outputs = decision["outputs"]
            if not isinstance(outputs, list) or len(outputs) < 2:
                raise PolicyError("decisions need at least two outputs")
            output_ids: list[str] = []
            for output in outputs:
                output = _exact(output, {"id", "label"}, "decision output")
                output_ids.append(_identifier(output["id"], "output id"))
                _text(output["label"], "output label")
            if len(output_ids) != len(set(output_ids)):
                raise PolicyError("decision output ids must be unique")
            order = _string_list(decision["order"], "decision order", nonempty=True)
            if order != output_ids:
                raise PolicyError(
                    "decision order must list every output exactly in declared order"
                )
            if decision["type"] == "noul" and set(output_ids) != {"yes", "no"}:
                raise PolicyError(
                    "Noul decisions require explicit yes and no output ids"
                )
            rules = decision["rules"]
            if not isinstance(rules, list) or len(rules) != len(output_ids):
                raise PolicyError("decisions require exactly one rule per output")
            rule_outputs: list[str] = []
            for rule in rules:
                rule = _exact(rule, {"output", "when"}, "decision rule")
                rule_outputs.append(_identifier(rule["output"], "rule output"))
                _validate_ast(rule["when"], group_facts, facts, [0])
            if set(rule_outputs) != set(output_ids) or len(rule_outputs) != len(
                set(rule_outputs)
            ):
                raise PolicyError("rules must cover outputs exactly once")
            decisions[decision_id] = decision

        presentations = group["presentations"]
        if not isinstance(presentations, list) or not presentations:
            raise PolicyError("semantic group presentations must be non-empty")
        group_receipt_presentations: list[dict[str, Any]] = []
        for presentation in presentations:
            presentation = _exact(
                presentation,
                {
                    "presentation_id",
                    "case_key",
                    "question_bindings",
                    "views",
                    "ablations",
                },
                "presentation",
            )
            presentation_id = _identifier(
                presentation["presentation_id"], "presentation id"
            )
            if presentation_id in seen_presentations:
                raise PolicyError("presentation ids must be globally unique")
            seen_presentations.add(presentation_id)
            case_key = _identifier(presentation["case_key"], "presentation case_key")
            if case_key not in cases:
                raise PolicyError("presentation references an unknown case")
            case = cases[case_key]
            question_bindings = presentation["question_bindings"]
            if not isinstance(question_bindings, dict) or set(question_bindings) != set(
                decisions
            ):
                raise PolicyError(
                    "question_bindings must exactly cover group decisions"
                )
            questions: dict[str, dict[str, Any]] = {}
            for decision_id, binding in question_bindings.items():
                binding = _exact(
                    binding,
                    {"question_key", "question_content_sha256"},
                    "question binding",
                )
                question = _question(
                    case, _identifier(binding["question_key"], "question binding key")
                )
                if binding["question_content_sha256"] != question["content_sha256"]:
                    raise PolicyError("question binding content hash mismatch")
                if question.get("decision_type") != decisions[decision_id]["type"]:
                    raise PolicyError("question and policy decision types disagree")
                questions[decision_id] = question

            views = _exact(
                presentation["views"],
                {"full", "removed", "decisive_only", "distractor_only"},
                "presentation views",
            )
            parsed_views = {
                name: _string_list(value, f"{name} view", nonempty=True)
                for name, value in views.items()
            }
            full = [chunk["chunk_key"] for chunk in case["chunks"]]
            if parsed_views["full"] != full:
                raise PolicyError(
                    "full view must exactly preserve every case chunk in order"
                )
            immutable = {
                chunk["chunk_key"]
                for chunk in case["chunks"]
                if chunk.get("immutable") is True
                or chunk.get("kind") == "global_policy"
            }
            if not immutable or any(
                chunk.get("kind") == "global_policy"
                and chunk.get("immutable") is not True
                for chunk in case["chunks"]
            ):
                raise PolicyError("global policy chunks must be immutable")
            for name, chunk_keys in parsed_views.items():
                if not set(chunk_keys) <= set(full) or not immutable <= set(chunk_keys):
                    raise PolicyError(
                        f"{name} view omits policy or references foreign chunks"
                    )

            view_envs = {
                name: _fact_environment(facts, bindings, case_key, set(chunk_keys))
                for name, chunk_keys in parsed_views.items()
            }
            decision_traces: list[dict[str, Any]] = []
            ablations = presentation["ablations"]
            if not isinstance(ablations, list) or len(ablations) != len(decisions):
                raise PolicyError(
                    "presentations require exactly one ablation per decision"
                )
            seen_ablations: set[str] = set()
            for ablation in ablations:
                ablation = _exact(
                    ablation,
                    {
                        "decision_id",
                        "remove_chunk_keys",
                        "required_transition",
                        "assert_full",
                        "assert_removed",
                    },
                    "ablation",
                )
                decision_id = _identifier(
                    ablation["decision_id"], "ablation decision id"
                )
                if decision_id not in decisions or decision_id in seen_ablations:
                    raise PolicyError(
                        "ablation decisions must cover group decisions once"
                    )
                seen_ablations.add(decision_id)
                remove = _string_list(
                    ablation["remove_chunk_keys"], "ablation removal", nonempty=True
                )
                if len(remove) > MAX_REMOVAL_CHUNKS:
                    raise PolicyError(
                        f"ablation removal exceeds the {MAX_REMOVAL_CHUNKS}-chunk proof bound"
                    )
                remove_set = set(remove)
                question = questions[decision_id]
                decisive = _string_list(
                    question.get("decisive_chunk_keys"),
                    "question decisive chunks",
                    nonempty=True,
                )
                distractors = _string_list(
                    question.get("known_distractor_chunk_keys"),
                    "question distractor chunks",
                )
                if remove != decisive or remove_set & immutable:
                    raise PolicyError(
                        "ablation must remove all and only decisive non-policy chunks"
                    )
                image_fact_chunks = {
                    binding["chunk_key"]
                    for binding in bindings
                    if binding["case_key"] == case_key
                    and any(
                        observation["fact"] in group_facts
                        for observation in binding["observations"]
                    )
                    and next(
                        chunk
                        for chunk in case["chunks"]
                        if chunk["chunk_key"] == binding["chunk_key"]
                    ).get("image_control")
                    is not None
                }
                if not image_fact_chunks <= remove_set:
                    raise PolicyError(
                        "real image policy facts must be decisive evidence"
                    )
                if set(distractors) != set(full) - immutable - remove_set:
                    raise PolicyError(
                        "question must label every remaining retrievable chunk as a distractor"
                    )
                expected_views = {
                    "removed": _ordered_subset(full, set(full) - remove_set),
                    "decisive_only": _ordered_subset(full, immutable | remove_set),
                    "distractor_only": _ordered_subset(
                        full, immutable | set(distractors)
                    ),
                }
                for name, expected in expected_views.items():
                    if parsed_views[name] != expected:
                        raise PolicyError(
                            f"{name} view disagrees with the frozen ablation"
                        )
                decision = decisions[decision_id]
                resolved = {
                    name: _resolve(decision, env) for name, env in view_envs.items()
                }
                _assert_result(resolved["full"], ablation["assert_full"], "assert_full")
                _assert_result(
                    resolved["removed"], ablation["assert_removed"], "assert_removed"
                )
                transition = classify_transition(resolved["full"], resolved["removed"])
                if (
                    ablation["required_transition"]
                    not in {"label_flip", "resolution_loss"}
                    or transition != ablation["required_transition"]
                ):
                    raise PolicyError(
                        "ablation does not produce its required causal transition"
                    )
                if (
                    resolved["decisive_only"].status != "resolved"
                    or resolved["decisive_only"].output != resolved["full"].output
                ):
                    raise PolicyError(
                        "decisive-only evidence does not reproduce the full result"
                    )
                if (
                    resolved["distractor_only"].status == "resolved"
                    and resolved["distractor_only"].output == resolved["full"].output
                ):
                    raise PolicyError(
                        "distractor-only evidence reproduces the full result"
                    )
                for subset_size in range(len(remove)):
                    for subset in combinations(remove, subset_size):
                        subset_env = _fact_environment(
                            facts, bindings, case_key, set(full) - set(subset)
                        )
                        subset_result = _resolve(decision, subset_env)
                        if (
                            classify_transition(resolved["full"], subset_result)
                            == ablation["required_transition"]
                        ):
                            raise PolicyError(
                                "decisive removal is not inclusion-minimal"
                            )
                for distractor in distractors:
                    without = _fact_environment(
                        facts, bindings, case_key, set(full) - {distractor}
                    )
                    without_result = _resolve(decision, without)
                    if (
                        without_result.status != resolved["full"].status
                        or without_result.output != resolved["full"].output
                    ):
                        raise PolicyError(
                            "removing a distractor changes the policy result"
                        )
                decision_traces.append(
                    {
                        "decision_id": decision_id,
                        "views": {
                            name: _result_json(result)
                            for name, result in sorted(resolved.items())
                        },
                        "transition": transition,
                        "inclusion_minimal": True,
                        "distractor_removal_invariant": True,
                    }
                )
            trace = {
                "presentation_id": presentation_id,
                "case_key": case_key,
                "full_fact_state": _environment_json(view_envs["full"], group_facts),
                "fact_states": {
                    name: _environment_json(environment, group_facts)
                    for name, environment in sorted(view_envs.items())
                },
                "decisions": sorted(
                    decision_traces, key=lambda item: item["decision_id"]
                ),
            }
            twin_traces[(group_id, presentation_id)] = trace
            group_receipt_presentations.append(trace)
        receipt_groups.append(
            {
                "group_id": group_id,
                "presentations": sorted(
                    group_receipt_presentations,
                    key=lambda item: item["presentation_id"],
                ),
            }
        )
    _validate_hebrew(groups, cases, twin_traces)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_VERSION,
        "inputs": {
            "panel_sha256": panel_sha256,
            "fact_ledger_sha256": ledger_sha256,
            "rubric_sha256": rubric_sha256,
            "evaluator_sha256": evaluator_sha256,
            "schema_sha256": schema_sha256,
        },
        "semantic_groups": sorted(receipt_groups, key=lambda item: item["group_id"]),
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--fact-ledger", type=Path, required=True)
    parser.add_argument("--fact-ledger-sha256", required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--rubric-sha256", required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--schema-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        panel, panel_hash = load_canonical_json(args.panel, args.panel_sha256, "panel")
        ledger, ledger_hash = load_canonical_json(
            args.fact_ledger, args.fact_ledger_sha256, "fact ledger"
        )
        rubric, rubric_hash = load_canonical_json(
            args.rubric, args.rubric_sha256, "rubric"
        )
        schema, schema_hash = load_canonical_json(
            args.schema, args.schema_sha256, "policy replay schema"
        )
        if (
            not isinstance(schema, dict)
            or schema.get("x-schema-version") != SCHEMA_VERSION
        ):
            raise PolicyError("policy replay schema identity is invalid")
        evaluator_hash = sha256_bytes(Path(__file__).read_bytes())
        if evaluator_hash != _sha(args.evaluator_sha256, "expected evaluator sha256"):
            raise PolicyError(
                "running evaluator differs from its external sha256 expectation"
            )
        receipt = replay(
            panel,
            ledger,
            rubric,
            panel_sha256=panel_hash,
            ledger_sha256=ledger_hash,
            rubric_sha256=rubric_hash,
            evaluator_sha256=evaluator_hash,
            schema_sha256=schema_hash,
        )
        if args.output.exists() or args.output.is_symlink():
            raise PolicyError("output already exists; refusing to overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(receipt))
    except (OSError, PolicyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
