"""Build the deterministic, counterfactual R048 exact-policy v1.6 panel."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from itertools import combinations, product
from pathlib import Path
from typing import Any

from scripts.exact_policy_executor import compile_policy, execute_policy


# Surface is a coherent renderer/source pair, not two independent treatments.
DEFAULT_ROWS = {"train": 1792, "cal": 896, "eval": 2688}
SPLITS = tuple(DEFAULT_ROWS)
OPTION_ORDERS = ("no_yes", "yes_no")
LANGUAGES = ("en", "he")
CLUSTER_SIZE = 8
PANEL_VERSION = "1.6"
FUZZY_THRESHOLD = 0.92
SPLIT_OFFSETS = {"train": 0, "cal": 100_000, "eval": 200_000}

SPLIT_FAMILIES = {
    "train": {
        "renderer": ("case_ledger", "fact_sheet"),
        "source": ("claims_intake", "purchase_control"),
        "value_regime": ("raw_operands", "derived_residuals"),
        "operator": ("all_of_3", "any_of_3"),
    },
    "cal": {
        "renderer": ("review_dossier", "verification_note"),
        "source": ("grant_review", "regulatory_check"),
        "value_regime": ("raw_operands", "derived_residuals"),
        "operator": ("all_of_3", "any_of_3"),
    },
    "eval": {
        "renderer": ("incident_digest", "assessment_brief"),
        "source": ("service_recovery", "risk_underwriting"),
        "value_regime": ("raw_operands", "derived_residuals"),
        "operator": ("all_of_3", "any_of_3", "at_least_2_of_3"),
    },
}
CHECK_NAMES = ("primary", "partner", "archive")
PREDICATE_PROGRAMS = {
    "all_of_3": {"combine": "all", "checks": list(CHECK_NAMES)},
    "any_of_3": {"combine": "any", "checks": list(CHECK_NAMES)},
    "at_least_2_of_3": {"combine": "at_least", "checks": list(CHECK_NAMES), "count": 2},
}

FIELDS = (
    "identity_verified", "policy_eligible", "amount_within_limit",
    "submission_timely", "evidence_complete", "region_covered",
    "consent_valid", "account_active",
)
PRIMITIVES = {
    "identity_verified": "record_match",
    "policy_eligible": "categorical_membership",
    "amount_within_limit": "decimal_comparison",
    "submission_timely": "temporal_comparison",
    "evidence_complete": "ledger_completeness",
    "region_covered": "set_membership",
    "consent_valid": "expiry_comparison",
    "account_active": "status_lookup",
}

QUESTION_TEXT = {
    "en": {
        "identity_verified": "Does identity verification meet its stated checklist rule?",
        "policy_eligible": "Does policy eligibility meet its stated checklist rule?",
        "amount_within_limit": "Does amount approval meet its stated checklist rule?",
        "submission_timely": "Does submission timing meet its stated checklist rule?",
        "evidence_complete": "Does evidence completeness meet its stated checklist rule?",
        "region_covered": "Does region coverage meet its stated checklist rule?",
        "consent_valid": "Does consent validity meet its stated checklist rule?",
        "account_active": "Does account activity meet its stated checklist rule?",
    },
    "he": {
        "identity_verified": "האם אימות הזהות עומד בכלל הבדיקות המוצהר?",
        "policy_eligible": "האם הזכאות למסלול עומדת בכלל הבדיקות המוצהר?",
        "amount_within_limit": "האם אישור הסכום עומד בכלל הבדיקות המוצהר?",
        "submission_timely": "האם מועד ההגשה עומד בכלל הבדיקות המוצהר?",
        "evidence_complete": "האם שלמות הראיות עומדת בכלל הבדיקות המוצהר?",
        "region_covered": "האם הכיסוי האזורי עומד בכלל הבדיקות המוצהר?",
        "consent_valid": "האם תוקף ההסכמה עומד בכלל הבדיקות המוצהר?",
        "account_active": "האם פעילות החשבון עומדת בכלל הבדיקות המוצהר?",
    },
}

RENDERER_COPY = {
    "case_ledger": {
        "en": ("Claims case ledger", "Ledger determination"),
        "he": ("יומן תיק תביעות", "הכרעת היומן"),
    },
    "fact_sheet": {
        "en": ("Purchase-control fact sheet", "Fact-sheet finding"),
        "he": ("דף עובדות לבקרת רכש", "ממצא דף העובדות"),
    },
    "review_dossier": {
        "en": ("Grant review dossier", "Dossier conclusion"),
        "he": ("תיק סקירת מענק", "מסקנת תיק הסקירה"),
    },
    "verification_note": {
        "en": ("Regulatory verification note", "Verification-note verdict"),
        "he": ("מזכר אימות רגולטורי", "קביעת מזכר האימות"),
    },
    "incident_digest": {
        "en": ("Service-recovery incident digest", "Incident-digest assessment"),
        "he": ("תקציר אירוע לשיקום שירות", "הערכת תקציר האירוע"),
    },
    "assessment_brief": {
        "en": ("Risk-underwriting assessment brief", "Assessment-brief decision"),
        "he": ("תמצית הערכת חיתום סיכון", "החלטת תמצית ההערכה"),
    },
}

SOURCE_COPY = {
    "claims_intake": {"en": "claims intake", "he": "קליטת תביעות"},
    "purchase_control": {"en": "purchase control", "he": "בקרת רכש"},
    "grant_review": {"en": "grant review", "he": "בדיקת מענקים"},
    "regulatory_check": {"en": "regulatory check", "he": "בדיקה רגולטורית"},
    "service_recovery": {"en": "service recovery", "he": "שיקום שירות"},
    "risk_underwriting": {"en": "risk underwriting", "he": "חיתום סיכונים"},
}


def _action(name: str, field: str | None = None) -> dict[str, Any]:
    payload = {"queue": name}
    if field is not None:
        payload["reason_field"] = field
    return {"name": name, "payload": payload}


def _policy(mode: str, fields: list[str], topology: str) -> dict[str, Any]:
    coarse_name = f"{mode}_coarse_gate"
    fieldwise_name = f"{mode}_fieldwise"
    if topology == coarse_name:
        any_failure = {"any": [{"not": {"field": field}} for field in fields]}
        if mode == "priority":
            rules = [{"when": any_failure,
                      "action": _action("review_any_failure")}]
        else:
            rules = [{"when": any_failure, "action": _action("review_any_failure"),
                      "primary_rank": None}]
            # Preserve genuinely multi-action collect outcomes without making
            # the two-field development mapping injective.
            if len(fields) >= 3:
                rules.append({
                    "when": {"all": [{"not": {"field": field}} for field in fields]},
                    "action": _action("escalate_total_failure"),
                    "primary_rank": 0,
                })
    elif topology == fieldwise_name:
        rules = []
        for rank, field in enumerate(fields):
            rule = {
                "when": {"not": {"field": field}},
                "action": _action(f"remediate_{field}", field),
            }
            if mode == "collect":
                rule["primary_rank"] = None if rank % 3 == 0 else rank
            rules.append(rule)
    else:
        raise ValueError(f"unknown {mode} policy topology: {topology}")
    fallback = _action("approve") if mode == "priority" else None
    return {"fields": fields, "mode": mode, "rules": rules, "fallback": fallback}


def _iso_day(offset: int) -> str:
    return (date(2024, 1, 1) + timedelta(days=offset)).isoformat()


def _make_native_fact(field: str, truth: bool, scenario: int,
                      variant: int) -> dict[str, Any]:
    """Create a field-native operand, not a renamed numeric threshold."""
    if field not in PRIMITIVES:
        raise ValueError(f"unknown field: {field}")
    delta = 1 + scenario % 9
    token = f"{scenario:07d}"
    if field == "identity_verified":
        birth_date = _iso_day(365 + scenario % 9000)
        values = {
            "registry_record": {"legal_name": f"Person {token}", "birth_date": birth_date},
            "submitted_record": {
                "legal_name": (f"Person {token}" if truth or variant % 3 == 0
                               else f"Other {token}"),
                "birth_date": (birth_date if truth or variant % 3 == 1
                               else _iso_day(366 + scenario % 9000)),
            },
        }
    elif field == "policy_eligible":
        values = {
            "policy_tier": (("standard", "enhanced")[variant % 2] if truth
                            else f"excluded-{scenario % 17}"),
            "eligible_tiers": ["standard", "enhanced"],
        }
    elif field == "amount_within_limit":
        limit = Decimal(1000 + scenario % 7000) / Decimal("10")
        amount = (limit - Decimal(delta if variant % 2 else 0) / 10 if truth
                  else limit + Decimal(delta) / 10)
        values = {"requested_amount": f"{amount:.2f}", "limit_amount": f"{limit:.2f}",
                  "currency": "ILS"}
    elif field == "submission_timely":
        deadline_offset = 10_000 + scenario % 5000
        values = {
            "submitted_on": _iso_day(deadline_offset - (delta if variant % 2 else 0) if truth
                                     else deadline_offset + delta),
            "deadline_on": _iso_day(deadline_offset),
        }
    elif field == "evidence_complete":
        required = ["identity", "receipt", f"case-{scenario % 997:03d}"]
        values = {"required_items": required,
                  "present_items": required if truth else required[:1 + variant % 2]}
    elif field == "region_covered":
        values = {"recorded_region": (("north", "central", "south")[variant % 3] if truth
                                      else f"offshore-{scenario % 19}"),
                  "covered_regions": ["north", "central", "south"]}
    elif field == "consent_valid":
        review_offset = 15_000 + scenario % 4000
        values = {
            "review_on": _iso_day(review_offset),
            "expires_on": _iso_day(review_offset + (delta if variant % 2 else 0) if truth
                                   else review_offset - delta),
        }
    else:  # account_active
        account = f"acct-{token}"
        values = {"account_ref": account,
                  "status_by_account": {account: ("active" if truth
                                                   else ("closed", "suspended")[variant % 2])}}
    return values


def _native_truth(field: str, fact: dict[str, Any]) -> bool:
    if field == "identity_verified":
        return fact["registry_record"] == fact["submitted_record"]
    if field == "policy_eligible":
        return fact["policy_tier"] in fact["eligible_tiers"]
    if field == "amount_within_limit":
        try:
            return Decimal(fact["requested_amount"]) <= Decimal(fact["limit_amount"])
        except (InvalidOperation, TypeError) as error:
            raise ValueError("invalid decimal operand") from error
    if field == "submission_timely":
        return date.fromisoformat(fact["submitted_on"]) <= date.fromisoformat(
            fact["deadline_on"])
    if field == "evidence_complete":
        return set(fact["required_items"]) <= set(fact["present_items"])
    if field == "region_covered":
        return fact["recorded_region"] in fact["covered_regions"]
    if field == "consent_valid":
        return date.fromisoformat(fact["expires_on"]) >= date.fromisoformat(
            fact["review_on"])
    if field == "account_active":
        return fact["status_by_account"].get(fact["account_ref"]) == "active"
    raise ValueError(f"unknown field: {field}")


def _program_truth(program: dict[str, Any], values: tuple[bool, ...]) -> bool:
    if program not in PREDICATE_PROGRAMS.values() or len(values) != len(CHECK_NAMES):
        raise ValueError("invalid three-check predicate program")
    if any(type(value) is not bool for value in values):
        raise ValueError("native check values must be Boolean")
    if program["combine"] == "all":
        return all(values)
    if program["combine"] == "any":
        return any(values)
    return sum(values) >= program["count"]


def _predicate_spec(field: str, value_regime: str, operator: str) -> dict[str, Any]:
    if value_regime not in {"raw_operands", "derived_residuals"}:
        raise ValueError("unknown operand schema")
    if operator not in PREDICATE_PROGRAMS:
        raise ValueError("unknown predicate operator")
    program = copy.deepcopy(PREDICATE_PROGRAMS[operator])
    return {"primitive": PRIMITIVES[field], "input_schema": value_regime,
            "program": program}


def _program_signature(program: dict[str, Any]) -> tuple[bool, ...]:
    return tuple(_program_truth(program, values)
                 for values in product((False, True), repeat=len(CHECK_NAMES)))


def _operator_witnesses(operator: str) -> list[dict[str, Any]]:
    """Executable witnesses distinguish every DSL operator on the same domain."""
    program = PREDICATE_PROGRAMS[operator]
    witnesses = []
    for other in PREDICATE_PROGRAMS.values():
        if other == program:
            continue
        values = next(values for values in product((False, True), repeat=len(CHECK_NAMES))
                      if _program_truth(program, values) != _program_truth(other, values))
        witnesses.append({"program": copy.deepcopy(program), "other_program": copy.deepcopy(other),
                          "check_truths": list(values),
                          "result": _program_truth(program, values),
                          "other_result": _program_truth(other, values)})
    return witnesses


def _native_residual(field: str, operands: dict[str, Any]) -> int | str:
    """Derive an auditable measurement from raw operands, never a target label."""
    if field == "identity_verified":
        return sum(operands["registry_record"][key] != operands["submitted_record"][key]
                   for key in ("legal_name", "birth_date"))
    if field in {"policy_eligible", "region_covered", "account_active"}:
        return int(not _native_truth(field, operands))
    if field == "amount_within_limit":
        return str(Decimal(operands["limit_amount"]) - Decimal(operands["requested_amount"]))
    if field == "submission_timely":
        return (date.fromisoformat(operands["deadline_on"])
                - date.fromisoformat(operands["submitted_on"])).days
    if field == "consent_valid":
        return (date.fromisoformat(operands["expires_on"])
                - date.fromisoformat(operands["review_on"])).days
    if field == "evidence_complete":
        return len(set(operands["required_items"]) - set(operands["present_items"]))
    raise ValueError("unknown residual primitive")


RESIDUAL_RULES = {
    "identity_verified": ("different record attributes", "שדות רשומה שאינם תואמים", "== 0"),
    "policy_eligible": ("tier membership violations", "חריגות שיוך למסלול", "== 0"),
    "amount_within_limit": ("limit minus requested, ILS", "תקרה פחות הסכום המבוקש, ILS", ">= 0"),
    "submission_timely": ("deadline minus submitted, days", "מועד אחרון פחות מועד הגשה, ימים", ">= 0"),
    "evidence_complete": ("missing required items", "פריטים נדרשים שחסרים", "== 0"),
    "region_covered": ("region membership violations", "חריגות שיוך לאזור", "== 0"),
    "consent_valid": ("expiry minus review, days", "מועד תפוגה פחות מועד בדיקה, ימים", ">= 0"),
    "account_active": ("inactive registry entries", "רשומות מרשם שאינן פעילות", "== 0"),
}


def _check_values(field: str, fact: dict[str, Any]) -> tuple[bool, ...]:
    spec = fact["predicate_spec"]
    values = []
    for name in CHECK_NAMES:
        check = fact["checks"][name]
        if spec["input_schema"] == "raw_operands":
            values.append(_native_truth(field, check))
        elif spec["input_schema"] == "derived_residuals":
            if set(check) != {"residual", "provenance"}:
                raise ValueError("invalid derived residual schema")
            expected_residual = _native_residual(field, check["provenance"])
            if (type(check["residual"]) is not type(expected_residual)
                    or check["residual"] != expected_residual):
                raise ValueError("residual disagrees with raw provenance")
            residual = Decimal(str(check["residual"]))
            values.append(residual >= 0 if RESIDUAL_RULES[field][2] == ">= 0"
                          else residual == 0)
        else:
            raise ValueError("unknown input schema")
    return tuple(values)


def _make_fact(field: str, truth: bool, scenario: int, value_regime: str,
               operator: str, case_variant: int = 0, causal: bool = False) -> dict[str, Any]:
    spec = _predicate_spec(field, value_regime, operator)
    choice = int(_sha([field, scenario, case_variant, "check pattern"]), 16)
    if causal:
        pairs = []
        for low in product((False, True), repeat=len(CHECK_NAMES)):
            for index, value in enumerate(low):
                high = tuple(True if position == index else item
                             for position, item in enumerate(low))
                if (not value and not _program_truth(spec["program"], low)
                        and _program_truth(spec["program"], high)):
                    pairs.append((low, high))
        bits = pairs[choice % len(pairs)][int(truth)]
    else:
        choices = [bits for bits in product((False, True), repeat=len(CHECK_NAMES))
                   if _program_truth(spec["program"], bits) == truth]
        bits = choices[choice % len(choices)]
    checks = {}
    for index, (name, value) in enumerate(zip(CHECK_NAMES, bits)):
        variant = int(_sha([field, scenario, case_variant, name]), 16)
        operands = _make_native_fact(field, value, scenario + index * 997, variant)
        checks[name] = (operands if value_regime == "raw_operands" else
                        {"residual": _native_residual(field, operands), "provenance": operands})
    return {"primitive": PRIMITIVES[field], "value_regime": value_regime,
            "operator_composition": operator, "predicate_spec": spec, "checks": checks}


def _fact_truth(field: str, fact: dict[str, Any]) -> bool:
    if (fact.get("primitive") != PRIMITIVES.get(field)
            or fact.get("predicate_spec") != _predicate_spec(
                field, fact["value_regime"], fact["operator_composition"])
            or set(fact["checks"]) != set(CHECK_NAMES)):
        raise ValueError("invalid executable predicate specification")
    return _program_truth(fact["predicate_spec"]["program"], _check_values(field, fact))


def _render_native_fact(field: str, fact: dict[str, Any], language: str) -> str:
    subject = QUESTION_TEXT[language][field].rstrip("?")
    if language == "en":
        renderers = {
            "identity_verified": lambda: (
                f"registry {fact['registry_record']}; submitted {fact['submitted_record']}; "
                "valid only on exact record agreement"),
            "policy_eligible": lambda: (
                f"tier {fact['policy_tier']}; eligible tiers {fact['eligible_tiers']}"),
            "amount_within_limit": lambda: (
                f"requested {fact['requested_amount']} {fact['currency']}; limit "
                f"{fact['limit_amount']} {fact['currency']}; requested <= limit"),
            "submission_timely": lambda: (
                f"submitted {fact['submitted_on']}; deadline {fact['deadline_on']}; "
                "submitted date <= deadline"),
            "evidence_complete": lambda: (
                f"required {fact['required_items']}; present {fact['present_items']}; "
                "every required item must be present"),
            "region_covered": lambda: (
                f"recorded {fact['recorded_region']}; covered {fact['covered_regions']}"),
            "consent_valid": lambda: (
                f"review {fact['review_on']}; expiry {fact['expires_on']}; expiry >= review"),
            "account_active": lambda: (
                f"account {fact['account_ref']}; registry status "
                f"{fact['status_by_account'][fact['account_ref']]}; must equal active"),
        }
        detail = renderers[field]()
        return f"{subject} — {detail}."
    renderers = {
        "identity_verified": lambda: (
            f"רשומת מרשם {fact['registry_record']}; רשומה שהוגשה {fact['submitted_record']}; "
            "נדרש תיאום מלא"),
        "policy_eligible": lambda: (
            f"מסלול {fact['policy_tier']}; מסלולים זכאים {fact['eligible_tiers']}"),
        "amount_within_limit": lambda: (
            f"סכום מבוקש {fact['requested_amount']} {fact['currency']}; תקרה "
            f"{fact['limit_amount']} {fact['currency']}; הסכום <= התקרה"),
        "submission_timely": lambda: (
            f"הוגש בתאריך {fact['submitted_on']}; מועד אחרון {fact['deadline_on']}; "
            "תאריך ההגשה <= המועד האחרון"),
        "evidence_complete": lambda: (
            f"פריטים נדרשים {fact['required_items']}; פריטים קיימים {fact['present_items']}; "
            "כל פריט נדרש חייב להימצא"),
        "region_covered": lambda: (
            f"אזור רשום {fact['recorded_region']}; אזורים מכוסים {fact['covered_regions']}"),
        "consent_valid": lambda: (
            f"מועד בדיקה {fact['review_on']}; תפוגה {fact['expires_on']}; תפוגה >= בדיקה"),
        "account_active": lambda: (
            f"חשבון {fact['account_ref']}; מצב במרשם "
            f"{fact['status_by_account'][fact['account_ref']]}; נדרש מצב active"),
    }
    detail = renderers[field]()
    return f"{subject} — {detail}."


def _render_fact(field: str, fact: dict[str, Any], language: str) -> str:
    rules = {
        "all": {"en": "Every named check must pass; a single failed check rejects this field.",
                "he": "כל בדיקה בשם חייבת לעבור; כישלון של בדיקה אחת פוסל את התנאי."},
        "any": {"en": "One successful named check is sufficient; reject only when none passes.",
                "he": "די בבדיקה אחת בשם שעוברת; יש לפסול רק אם אף בדיקה אינה עוברת."},
        "at_least": {"en": "Require a majority: at least two of the three named checks must pass.",
                     "he": "נדרש רוב: לפחות שתיים משלוש הבדיקות בשם חייבות לעבור."},
    }
    spec = fact["predicate_spec"]
    lines = [f"{QUESTION_TEXT[language][field].rstrip('?')} — "
             + rules[spec["program"]["combine"]][language]]
    for name in CHECK_NAMES:
        check = fact["checks"][name]
        if spec["input_schema"] == "raw_operands":
            detail = _render_native_fact(field, check, language).partition(" — ")[2]
        else:
            label = RESIDUAL_RULES[field][0 if language == "en" else 1]
            detail = f"{label}: {check['residual']}; {RESIDUAL_RULES[field][2]}."
        lines.append(f"  {name}: {detail}")
    return "\n".join(lines)


def _text(language: str, fields: list[str], facts: dict[str, dict[str, Any]],
          renderer_family: str, source_family: str) -> str:
    header = RENDERER_COPY[renderer_family][language][0]
    source = SOURCE_COPY[source_family][language]
    opener = (f"{header}. Source register: {source}. Apply every stated operator."
              if language == "en" else
              f"{header}. מקור הרשומה: {source}. יש להחיל כל כלל חישוב מפורש.")
    return "\n".join([opener] +
                     [_render_fact(field, facts[field], language) for field in fields])


def _question(field: str, language: str, option_order: str,
              renderer_family: str) -> dict[str, Any]:
    labels = (("no", "לא"), ("yes", "כן")) if language == "he" else (
        ("no", "No"), ("yes", "Yes"))
    if option_order == "yes_no":
        labels = tuple(reversed(labels))
    prefix = RENDERER_COPY[renderer_family][language][1]
    instructions = f"{prefix}: {QUESTION_TEXT[language][field]}"
    return {"type": "noul", "instructions": instructions, "criteria": dict(labels)}


def _policy_families(mode: str, topology: str) -> tuple[str, str]:
    expected = {
        ("priority", "priority_coarse_gate"): ("coarse_disjunction", "single_review_gate"),
        ("collect", "collect_coarse_gate"): ("coarse_disjunction", "aggregate_review_set"),
        ("priority", "priority_fieldwise"): ("ordered_field_conditions", "first_remediation"),
        ("collect", "collect_fieldwise"): ("independent_field_conditions", "remediation_set"),
    }
    try:
        return expected[(mode, topology)]
    except KeyError as error:
        raise ValueError(f"unknown policy family: {mode}/{topology}") from error


def _separation_witness(mode: str, fields: list[str],
                        heldout_policy: dict[str, Any]) -> dict[str, Any]:
    """Return a one-field witness that held-out mapping refines development."""
    development = _policy(mode, fields, f"{mode}_coarse_gate")
    development_table = compile_policy(development).truth_outcomes
    heldout_table = compile_policy(heldout_policy).truth_outcomes
    assignments = [dict(zip(fields, values))
                   for values in product((False, True), repeat=len(fields))]
    for left_index, left in enumerate(assignments):
        for right in assignments[left_index + 1:]:
            if sum(left[field] != right[field] for field in fields) != 1:
                continue
            left_id = sum(int(left[field]) << bit for bit, field in enumerate(fields))
            right_id = sum(int(right[field]) << bit for bit, field in enumerate(fields))
            development_left = development_table[left_id].as_dict()
            development_right = development_table[right_id].as_dict()
            heldout_left = heldout_table[left_id].as_dict()
            heldout_right = heldout_table[right_id].as_dict()
            if development_left == development_right and heldout_left != heldout_right:
                return {
                    "trace_a": left,
                    "trace_b": right,
                    "development_outcome": development_left,
                    "heldout_outcome_a": heldout_left,
                    "heldout_outcome_b": heldout_right,
                }
    raise ValueError(f"{mode}: held-out topology has no causal separation witness")


def _cluster_spec(split: str, cluster: int, seed: int,
                  case_variant: int = 0) -> dict[str, Any]:
    scenario = seed * 1_000_000 + SPLIT_OFFSETS[split] + cluster
    field_count = 2 + cluster % 7
    # Exhaust a local evidence neighborhood before moving to another causal
    # field. A finite native-case pool must never be enlarged with fresh IDs.
    flipped_field = FIELDS[(cluster + seed + case_variant // 32) % len(FIELDS)]
    field_sets = list(combinations([field for field in FIELDS if field != flipped_field],
                                  field_count - 1))
    choice = int(_sha([seed, cluster, case_variant, "fields"]), 16)
    fields = list(field_sets[choice % len(field_sets)])
    pivot_position = (cluster // 7 + seed) % field_count
    fields.insert(pivot_position, flipped_field)
    cell = cluster // 7
    mode = ("priority", "collect")[(cell + seed) % 2]
    families = SPLIT_FAMILIES[split]
    cell //= 2
    if split == "eval" and cell % 2:
        topology_slice = "heldout"
        topology = f"{mode}_fieldwise"
    else:
        topology_slice = "same" if split == "eval" else "development"
        topology = f"{mode}_coarse_gate"
    if split == "eval":
        cell //= 2
    operator = families["operator"][cell % len(families["operator"])]
    cell //= len(families["operator"])
    value_regime = families["value_regime"][cell % 2]
    surface = (cell // 2) % 2
    renderer = families["renderer"][surface]
    source = families["source"][surface]

    policy = _policy(mode, fields, topology)
    compiled = compile_policy(policy)
    program_family, action_family = _policy_families(mode, topology)
    witness = (_separation_witness(mode, fields, policy)
               if topology_slice == "heldout" else None)
    other_fields = [field for field in fields if field != flipped_field]
    candidates = []
    for bits in range(1 << len(other_fields)):
        base = {field: bool(bits & (1 << index))
                for index, field in enumerate(other_fields)}
        low = {**base, flipped_field: False}
        high = {**base, flipped_field: True}
        low_id = sum(int(low[field]) << bit for bit, field in enumerate(fields))
        high_id = sum(int(high[field]) << bit for bit, field in enumerate(fields))
        if compiled.truth_outcomes[low_id] != compiled.truth_outcomes[high_id]:
            candidates.append((low, high))
    if not candidates:
        raise ValueError(f"policy does not expose a causal twin for {flipped_field}")
    low, high = candidates[int(_sha([seed, cluster, case_variant, "endpoints"]), 16)
                           % len(candidates)]
    facts_by_side = {}
    false_side = (cluster // 2 + seed) % 2
    for side in (0, 1):
        truths = low if side == false_side else high
        facts_by_side[side] = {
            field: _make_fact(field, truths[field], scenario + position * 37,
                              value_regime, operator, case_variant,
                              causal=field == flipped_field)
            for position, field in enumerate(fields)
        }
    encoder_signature = _encoder_semantic_signature(fields, flipped_field, facts_by_side)
    semantic_signature = _semantic_signature(fields, flipped_field, facts_by_side, policy)
    exact_cluster_signature = _exact_cluster_signature(
        semantic_signature, facts_by_side,
    )
    return {
        "scenario": scenario, "fields": fields, "mode": mode,
        "flipped_field": flipped_field, "facts_by_side": facts_by_side,
        "topology": topology, "topology_slice": topology_slice,
        "renderer": renderer, "source": source, "value_regime": value_regime,
        "operator": operator, "semantic_signature": semantic_signature,
        "encoder_semantic_signature": encoder_signature,
        "grammar_signature": _grammar_signature(fields, facts_by_side[0]),
        "case_variant": case_variant,
        "exact_cluster_signature": exact_cluster_signature,
        "program_family": program_family, "action_family": action_family,
        "topology_separation_witness": witness,
    }


def _row(split: str, index: int, seed: int,
         spec: dict[str, Any] | None = None) -> dict[str, Any]:
    cluster, variant = divmod(index, CLUSTER_SIZE)
    side = variant % 2
    language = LANGUAGES[(variant // 2) % 2]
    option_order = OPTION_ORDERS[(variant // 4) % 2]
    spec = spec or _cluster_spec(split, cluster, seed)
    fields = spec["fields"]
    facts = spec["facts_by_side"][side]
    truths = {field: _fact_truth(field, facts[field]) for field in fields}
    policy = _policy(spec["mode"], fields, spec["topology"])
    qid_to_field = {f"q{position:02d}": field for position, field in enumerate(fields)}
    questions = {qid: _question(field, language, option_order, spec["renderer"])
                 for qid, field in qid_to_field.items()}
    targets = {qid: {"label": "yes" if truths[field] else "no"}
               for qid, field in qid_to_field.items()}
    outcome = execute_policy(policy, truths).as_dict()
    return {
        "id": f"r048-{split}-{seed:04d}-{index:06d}",
        "state": _text(language, fields, facts, spec["renderer"], spec["source"]),
        "questions": questions,
        "targets": targets,
        "meta": {
            "panel": "exact_policy_v1",
            "panel_version": PANEL_VERSION,
            "split": split,
            "language": language,
            "option_order": option_order,
            "counterfactual_cluster_id": f"r048-{split}-{seed:04d}-c{cluster:06d}",
            "semantic_group_id": f"r048-{split}-{seed:04d}-c{cluster:06d}",
            "counterfactual_side": side,
            "scenario_id": spec["scenario"],
            "flipped_field": spec["flipped_field"],
            "field_order": fields,
            "qid_to_field": qid_to_field,
            "decision_threshold": 0.5,
            "facts": facts,
            "primitive_by_field": {field: PRIMITIVES[field] for field in fields},
            "action_trace": truths,
            "policy": policy,
            "gold_outcome": outcome,
            "policy_mode": spec["mode"],
            "policy_topology": spec["topology"],
            "topology_slice": spec["topology_slice"],
            "program_family": spec["program_family"],
            "action_family": spec["action_family"],
            "topology_separation_witness": spec["topology_separation_witness"],
            "renderer_family": spec["renderer"],
            "source_family": spec["source"],
            "value_regime": spec["value_regime"],
            "operator_composition": spec["operator"],
            "topology_transfer_scope": "external_policy_outcome_mapping_only",
            "external_policy_topology": spec["topology"],
            "external_policy_topology_slice": spec["topology_slice"],
            "semantic_signature": spec["semantic_signature"],
            "encoder_semantic_signature": spec["encoder_semantic_signature"],
            "encoder_endpoint_signature": _encoder_endpoint_signature(fields, facts),
            "grammar_signature": spec["grammar_signature"],
            "generation_attempt": spec["case_variant"],
            "operator_slice": ("heldout" if spec["operator"] == "at_least_2_of_3"
                               else "anchor"),
            "operator_separation_witnesses": _operator_witnesses(spec["operator"]),
            "exact_cluster_signature": spec["exact_cluster_signature"],
            "latent_signature": spec["semantic_signature"],
            "factor_cell": f"{spec['mode']}|n{len(fields)}|p{fields.index(spec['flipped_field'])}|{spec['topology_slice']}",
        },
    }


def build(split: str, count: int, seed: int = 48) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if (isinstance(count, bool) or not isinstance(count, int)
            or count < 2 * CLUSTER_SIZE or count % CLUSTER_SIZE):
        raise ValueError(f"count must be a multiple of {CLUSTER_SIZE} and at least 16")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9999:
        raise ValueError("seed must be an integer from 0 through 9999")
    rows = []
    seen = set()
    seen_endpoints = set()
    for cluster in range(count // CLUSTER_SIZE):
        for attempt in range(4096):
            spec = _cluster_spec(split, cluster, seed, attempt)
            signature = spec["encoder_semantic_signature"]
            endpoints = {_encoder_endpoint_signature(spec["fields"], facts)
                         for facts in spec["facts_by_side"].values()}
            # Reserve both endpoints before rendering, since distinct causal
            # pairs can otherwise share a single leaked encoder input.
            if (all(int(key, 16) % len(SPLITS) == SPLITS.index(split) for key in endpoints)
                    and signature not in seen and endpoints.isdisjoint(seen_endpoints)):
                seen.add(signature)
                seen_endpoints.update(endpoints)
                break
        else:
            raise ValueError(f"{split}: exhausted unique semantic cases at cluster {cluster}")
        rows.extend(_row(split, cluster * CLUSTER_SIZE + variant, seed, spec)
                    for variant in range(CLUSTER_SIZE))
    return rows


def _normalize_text(value: str) -> str:
    value = re.sub(r"\d+(?:\.\d+)?", "<number>", value.casefold())
    return " ".join(re.findall(r"[\w<>]+", value, flags=re.UNICODE))


def _model_input(row: dict[str, Any]) -> str:
    questions = []
    for qid, question in row["questions"].items():
        options = " | ".join(f"{key}: {value}" for key, value in question["criteria"].items())
        questions.append(f"{qid} {question['instructions']} {options}")
    return row["state"] + "\n" + "\n".join(questions)


def _sha(value: Any) -> str:
    payload = (value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(payload.encode()).hexdigest()


def _complete_cluster_serialization_signature(group: list[dict[str, Any]]) -> str:
    """Hash all user-visible and executable cluster content, excluding IDs."""
    excluded_meta = {
        "split", "counterfactual_cluster_id", "semantic_group_id", "scenario_id",
        "semantic_signature", "exact_cluster_signature", "latent_signature",
        "encoder_semantic_signature", "grammar_signature", "generation_attempt",
        "encoder_endpoint_signature",
    }
    projected = []
    for row in sorted(group, key=lambda item: (
            item["meta"]["counterfactual_side"], item["meta"]["language"],
            item["meta"]["option_order"])):
        projected.append({
            "state": row["state"],
            "questions": row["questions"],
            "targets": row["targets"],
            "meta": {key: value for key, value in row["meta"].items()
                     if key not in excluded_meta},
        })
    return _sha(projected)


def _partition_signature(policy: dict[str, Any]) -> tuple[int, ...]:
    """Truth-partition signature, invariant to action names and payloads."""
    compiled = compile_policy(policy)
    classes: dict[str, int] = {}
    result = []
    for outcome in compiled.truth_outcomes:
        key = json.dumps(outcome.as_dict(), ensure_ascii=False, sort_keys=True)
        classes.setdefault(key, len(classes))
        result.append(classes[key])
    return tuple(result)


def _case_evidence(field: str, fact: dict[str, Any]) -> dict[str, Any]:
    """Preserve operative relations, excluding IDs, magnitudes and presentation.

    This is the information shared by raw and residual inputs. For example,
    equality at a monetary boundary differs from positive headroom, and one
    missing ledger item differs from two. Hidden provenance attributes are
    deliberately not used to manufacture distinct encoder cases.
    """
    relations = []
    for name in CHECK_NAMES:
        check = fact["checks"][name]
        operands = (check if fact["predicate_spec"]["input_schema"] == "raw_operands"
                    else check["provenance"])
        residual = Decimal(str(_native_residual(field, operands)))
        if RESIDUAL_RULES[field][2] == ">= 0":
            relations.append((residual > 0) - (residual < 0))
        else:
            relations.append(int(residual))
    return {"primitive": PRIMITIVES[field],
            "truth_function": _program_signature(fact["predicate_spec"]["program"]),
            "check_names": CHECK_NAMES, "native_relations": relations}


def _encoder_semantic_signature(fields: list[str], flipped_field: str,
                                 facts_by_side: dict[int, dict[str, Any]]) -> str:
    endpoints = [_encoder_endpoint_signature(fields, facts)
                 for facts in facts_by_side.values()]
    return _sha({"flipped_field": flipped_field,
                 "endpoints": sorted(endpoints)})


def _encoder_endpoint_signature(fields: list[str], facts: dict[str, Any]) -> str:
    return _sha({field: _case_evidence(field, facts[field]) for field in sorted(fields)})


def _grammar_signature(fields: list[str], facts: dict[str, Any]) -> str:
    return _sha({field: facts[field]["predicate_spec"] for field in sorted(fields)})


def _semantic_signature(fields: list[str], flipped_field: str,
                        facts_by_side: dict[int, dict[str, Any]],
                        policy: dict[str, Any]) -> str:
    compiled = compile_policy(policy)
    outcomes = []
    canonical_fields = sorted(fields)
    # Identity is the exact assignment -> outcome function, not its unlabeled
    # partition. Canonical enumeration permits field-order invariance only when
    # primary actions and full action names/typed payloads are also identical.
    for values in product((False, True), repeat=len(fields)):
        trace = dict(zip(canonical_fields, values))
        assignment = sum(int(trace[field]) << bit for bit, field in enumerate(compiled.fields))
        outcomes.append(compiled.truth_outcomes[assignment].as_dict())
    return _sha({"encoder_case": _encoder_semantic_signature(fields, flipped_field, facts_by_side),
                 "policy_outcomes": outcomes})


def _normalized_model_input(row: dict[str, Any], include_surface: bool = True) -> str:
    text = _model_input(row)
    if not include_surface:
        questions = [question["instructions"].partition(": ")[2]
                     + " " + str(list(question["criteria"].items()))
                     for question in row["questions"].values()]
        text = "\n".join(row["state"].splitlines()[1:] + questions)
    # Plain number erasure collapses x == limit and x < limit, and ledger
    # missing-count one and two. Preserve these visible primitive relations.
    evidence = {field: _case_evidence(field, fact)
                for field, fact in row["meta"]["facts"].items()}
    return _normalize_text(text) + "\n" + json.dumps(evidence, sort_keys=True)


def _exact_cluster_signature(
    semantic_signature: str,
    facts_by_side: dict[int, dict[str, dict[str, Any]]],
) -> str:
    """Hash the full pair operands in addition to their latent semantics."""
    return _sha({
        "semantic_signature": semantic_signature,
        "endpoint_facts": facts_by_side,
    })


def _fingerprints(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "model_input_exact": {_sha(_model_input(row)) for row in rows},
        "model_input_normalized": {_sha(_normalized_model_input(row)) for row in rows},
        "encoder_input_normalized": {_sha(_normalized_model_input(row, False)) for row in rows},
        "encoder_semantic_signature": {row["meta"]["encoder_semantic_signature"] for row in rows},
        "encoder_endpoint_signature": {row["meta"]["encoder_endpoint_signature"] for row in rows},
        "semantic_signature": {row["meta"]["semantic_signature"] for row in rows},
        "grammar_signature": {row["meta"]["grammar_signature"] for row in rows},
        "policy": {_sha(row["meta"]["policy"]) for row in rows},
    }


def _token_shingles(value: str) -> frozenset[str]:
    tokens = _normalize_text(value).split()
    if len(tokens) < 3:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[index:index + 3])
                     for index in range(len(tokens) - 2))


def _fuzzy_report(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    # Exhaustive over unique model inputs; duplicate option views are retained
    # in the row count but do not inflate the near-duplicate count.
    left_inputs = sorted({_model_input(row) for row in left})
    right_inputs = sorted({_model_input(row) for row in right})
    left_sets = [_token_shingles(value) for value in left_inputs]
    right_sets = [_token_shingles(value) for value in right_inputs]
    vocabulary = {token: index for index, token in enumerate(sorted(
        set().union(*left_sets, *right_sets)))}
    # Exact bit-set Jaccard retains the exhaustive audit without millions of
    # temporary Python set unions on a balanced corpus.
    left_masks = [sum(1 << vocabulary[token] for token in values) for values in left_sets]
    right_masks = [sum(1 << vocabulary[token] for token in values) for values in right_sets]
    maximum = 0.0
    above = 0
    for left_mask in left_masks:
        for right_mask in right_masks:
            union = (left_mask | right_mask).bit_count()
            score = (left_mask & right_mask).bit_count() / union if union else 1.0
            maximum = max(maximum, score)
            above += score >= FUZZY_THRESHOLD
    return {
        "method": "exhaustive unique-model-input token-trigram Jaccard after case/number normalization",
        "threshold": FUZZY_THRESHOLD,
        "left_unique_inputs": len(left_inputs),
        "right_unique_inputs": len(right_inputs),
        "comparisons": len(left_inputs) * len(right_inputs),
        "maximum_similarity": round(maximum, 6),
        "pairs_at_or_above_threshold": above,
        "pair_rate_at_or_above_threshold": (
            above / (len(left_inputs) * len(right_inputs))
            if left_inputs and right_inputs else 0.0
        ),
        "role": "descriptive template diagnostic; anchor grammars intentionally overlap",
    }


def _validate_separation_witness(meta: dict[str, Any]) -> None:
    witness = meta["topology_separation_witness"]
    if meta["topology_slice"] != "heldout":
        if witness is not None:
            raise ValueError("non-held-out topology must not carry a separation witness")
        return
    fields = meta["field_order"]
    if not isinstance(witness, dict) or set(witness) != {
        "trace_a", "trace_b", "development_outcome", "heldout_outcome_a",
        "heldout_outcome_b",
    }:
        raise ValueError("held-out topology needs a complete separation witness")
    left, right = witness["trace_a"], witness["trace_b"]
    if (not isinstance(left, dict) or not isinstance(right, dict)
            or set(left) != set(fields) or set(right) != set(fields)
            or any(type(value) is not bool for value in (*left.values(), *right.values()))
            or sum(left[field] != right[field] for field in fields) != 1):
        raise ValueError("held-out separation witness is not a causal pair")
    development = _policy(meta["policy_mode"], fields,
                          f"{meta['policy_mode']}_coarse_gate")
    development_left = execute_policy(development, left).as_dict()
    development_right = execute_policy(development, right).as_dict()
    heldout_left = execute_policy(meta["policy"], left).as_dict()
    heldout_right = execute_policy(meta["policy"], right).as_dict()
    if (development_left != development_right
            or heldout_left == heldout_right
            or witness != {
                "trace_a": left,
                "trace_b": right,
                "development_outcome": development_left,
                "heldout_outcome_a": heldout_left,
                "heldout_outcome_b": heldout_right,
            }):
        raise ValueError("held-out topology does not separate the witness")


def validate_split(rows: list[dict[str, Any]], split: str,
                   expected_rows: int | None = None) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{split}: expected {expected_rows} rows, got {len(rows)}")
    ids: set[str] = set()
    modes: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    field_sizes: Counter[int] = Counter()
    flipped_fields: Counter[str] = Counter()
    primitive_labels: dict[str, Counter[bool]] = defaultdict(Counter)
    topology_slices: Counter[str] = Counter()
    topology_modes: Counter[str] = Counter()
    factor_cells: Counter[str] = Counter()
    crossed_cells: Counter[tuple[int, str, str, str, str]] = Counter()
    flipped_field_modes: Counter[tuple[str, str]] = Counter()
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if set(row) != {"id", "state", "questions", "targets", "meta"}:
            raise ValueError(f"{split}: rows must contain exactly the panel schema")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"{split}: row ids must be unique nonempty strings")
        ids.add(identifier)
        match = re.fullmatch(rf"r048-{re.escape(split)}-(\d{{4}})-(\d{{6}})", identifier)
        if match is None:
            raise ValueError(f"{identifier}: malformed row id")
        seed, row_index = (int(value) for value in match.groups())
        cluster_index, expected_variant = divmod(row_index, CLUSTER_SIZE)
        meta = row.get("meta")
        if not isinstance(meta, dict) or meta.get("panel") != "exact_policy_v1":
            raise ValueError(f"{identifier}: invalid panel metadata")
        required = {
            "field_order", "qid_to_field", "decision_threshold", "action_trace",
            "policy", "gold_outcome", "facts", "primitive_by_field", "language",
            "option_order", "counterfactual_cluster_id", "counterfactual_side",
            "flipped_field", "scenario_id", "semantic_group_id", "policy_mode",
            "policy_topology", "topology_slice", "renderer_family", "source_family",
            "value_regime", "operator_composition", "topology_transfer_scope",
            "external_policy_topology", "external_policy_topology_slice",
            "semantic_signature", "exact_cluster_signature", "latent_signature",
            "factor_cell", "program_family", "action_family",
            "topology_separation_witness",
            "encoder_semantic_signature", "grammar_signature", "generation_attempt",
            "encoder_endpoint_signature",
            "operator_slice", "operator_separation_witnesses",
        }
        if (not required <= set(meta) or meta.get("split") != split
                or meta.get("panel_version") != PANEL_VERSION):
            raise ValueError(f"{identifier}: incomplete panel metadata")
        expected_cluster_id = f"r048-{split}-{seed:04d}-c{cluster_index:06d}"
        expected_side = expected_variant % 2
        expected_language = LANGUAGES[(expected_variant // 2) % 2]
        expected_order = OPTION_ORDERS[(expected_variant // 4) % 2]
        expected_scenario = seed * 1_000_000 + SPLIT_OFFSETS[split] + cluster_index
        if (meta["counterfactual_cluster_id"] != expected_cluster_id
                or meta["semantic_group_id"] != expected_cluster_id
                or meta["counterfactual_side"] != expected_side
                or meta["language"] != expected_language
                or meta["option_order"] != expected_order
                or meta["scenario_id"] != expected_scenario):
            raise ValueError(f"{identifier}: row identity metadata mismatch")
        policy = compile_policy(meta["policy"])
        if (list(policy.fields) != meta["field_order"]
                or not 2 <= len(policy.fields) <= 8):
            raise ValueError(f"{identifier}: policy/field order mismatch")
        topology = meta["policy_topology"]
        if (meta["policy_mode"] != policy.mode
                or meta["external_policy_topology"] != topology
                or meta["external_policy_topology_slice"] != meta["topology_slice"]
                or meta["topology_transfer_scope"]
                != "external_policy_outcome_mapping_only"):
            raise ValueError(f"{identifier}: scorer policy metadata mismatch")
        if meta["policy"] != _policy(policy.mode, list(policy.fields), topology):
            raise ValueError(f"{identifier}: external policy topology mismatch")
        expected_program, expected_action = _policy_families(policy.mode, topology)
        if (meta["program_family"] != expected_program
                or meta["action_family"] != expected_action):
            raise ValueError(f"{identifier}: invalid program/action family")
        try:
            _validate_separation_witness(meta)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{identifier}: invalid topology separation witness") from error
        qids = [f"q{index:02d}" for index in range(len(policy.fields))]
        expected_mapping = dict(zip(qids, policy.fields))
        if meta["qid_to_field"] != expected_mapping:
            raise ValueError(f"{identifier}: invalid qid_to_field")
        if (set(row.get("questions", {})) != set(qids)
                or set(row.get("targets", {})) != set(qids)):
            raise ValueError(f"{identifier}: invalid questions or targets")
        if meta["language"] not in LANGUAGES or meta["option_order"] not in OPTION_ORDERS:
            raise ValueError(f"{identifier}: invalid language or option order")
        families = SPLIT_FAMILIES[split]
        if ((meta["renderer_family"], meta["source_family"]) not in set(zip(
                families["renderer"], families["source"]))
                or meta["value_regime"] not in families["value_regime"]
                or meta["operator_composition"] not in families["operator"]):
            raise ValueError(f"{identifier}: invalid split family cell")
        expected_operator_slice = ("heldout" if meta["operator_composition"] == "at_least_2_of_3"
                                   else "anchor")
        if (meta["operator_slice"] != expected_operator_slice
                or meta["operator_separation_witnesses"] != _operator_witnesses(
                    meta["operator_composition"])):
            raise ValueError(f"{identifier}: invalid operator separation witnesses")
        for qid, field in expected_mapping.items():
            question = row["questions"][qid]
            expected_criteria = (["no", "yes"] if meta["option_order"] == "no_yes"
                                 else ["yes", "no"])
            if (question != _question(field, meta["language"], meta["option_order"],
                                      meta["renderer_family"])
                    or list(question.get("criteria", {})) != expected_criteria
                    or set(row["targets"][qid]) != {"label"}
                    or row["targets"][qid]["label"] not in {"no", "yes"}):
                raise ValueError(f"{identifier}/{qid}: invalid semantic question")
        truths = {expected_mapping[qid]: row["targets"][qid].get("label") == "yes"
                  for qid in qids}
        facts = meta["facts"]
        if (not isinstance(facts, dict) or set(facts) != set(policy.fields)
                or meta["primitive_by_field"] != {
                    field: PRIMITIVES[field] for field in policy.fields}):
            raise ValueError(f"{identifier}: invalid primitive facts")
        try:
            fact_truths = {field: _fact_truth(field, facts[field]) for field in policy.fields}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{identifier}: invalid primitive facts") from error
        if truths != meta["action_trace"] or truths != fact_truths:
            raise ValueError(f"{identifier}: action trace mismatch")
        if (isinstance(meta["scenario_id"], bool)
                or not isinstance(meta["scenario_id"], int)
                or type(meta["generation_attempt"]) is not int
                or not 0 <= meta["generation_attempt"] < 4096
                or facts != {
                    field: _make_fact(
                        field, truths[field], meta["scenario_id"] + position * 37,
                        meta["value_regime"], meta["operator_composition"],
                        meta["generation_attempt"], causal=field == meta["flipped_field"])
                    for position, field in enumerate(policy.fields)
                }):
            raise ValueError(f"{identifier}: primitive facts are not canonical")
        if row.get("state") != _text(
                meta["language"], list(policy.fields), facts,
                meta["renderer_family"], meta["source_family"]):
            raise ValueError(f"{identifier}: state/facts mismatch")
        expected_endpoint = _encoder_endpoint_signature(list(policy.fields), facts)
        if (meta["encoder_endpoint_signature"] != expected_endpoint
                or int(expected_endpoint, 16) % len(SPLITS) != SPLITS.index(split)):
            raise ValueError(f"{identifier}: invalid or cross-split encoder endpoint")
        if execute_policy(policy, truths).as_dict() != meta["gold_outcome"]:
            raise ValueError(f"{identifier}: gold outcome mismatch")
        if (isinstance(meta["decision_threshold"], bool)
                or meta["decision_threshold"] != 0.5):
            raise ValueError(f"{identifier}: invalid decision threshold")
        expected_factor = (
            f"{policy.mode}|n{len(policy.fields)}|"
            f"p{list(policy.fields).index(meta['flipped_field'])}|"
            f"{meta['topology_slice']}"
        )
        if meta["factor_cell"] != expected_factor:
            raise ValueError(f"{identifier}: invalid factor cell")
        modes[policy.mode] += 1
        outcome = meta["gold_outcome"]
        outcomes[f"{len(outcome['actions'])}:{outcome['primary'] is None}"] += 1
        field_sizes[len(policy.fields)] += 1
        flipped_fields[meta["flipped_field"]] += 1
        topology_slices[meta["topology_slice"]] += 1
        topology_modes[f"{meta['topology_slice']}:{policy.mode}"] += 1
        factor_cells[meta["factor_cell"]] += 1
        crossed_cells[(len(policy.fields), policy.mode,
                       meta["topology_slice"], meta["language"],
                       meta["option_order"])] += 1
        flipped_field_modes[(meta["flipped_field"], policy.mode)] += 1
        clusters[meta["counterfactual_cluster_id"]].append(row)
        for field, truth in truths.items():
            primitive_labels[PRIMITIVES[field]][truth] += 1
    if set(modes) != {"priority", "collect"}:
        raise ValueError(f"{split}: both policy modes are required")
    if len(rows) >= 7 * CLUSTER_SIZE and set(field_sizes) != set(range(2, 9)):
        raise ValueError(f"{split}: field sizes 2-8 are required")
    expected_variants = {
        (side, language, order) for side in (0, 1)
        for language in LANGUAGES for order in OPTION_ORDERS
    }
    for cluster_id, group in clusters.items():
        if len(group) != CLUSTER_SIZE or {
            (row["meta"]["counterfactual_side"], row["meta"]["language"],
             row["meta"]["option_order"]) for row in group
        } != expected_variants:
            raise ValueError(f"{cluster_id}: incomplete counterfactual cluster")
        if len({row["meta"]["semantic_signature"] for row in group}) != 1:
            raise ValueError(f"{cluster_id}: inconsistent semantic signature")
        invariant_keys = (
            "scenario_id", "field_order", "policy", "flipped_field",
            "policy_topology", "topology_slice", "renderer_family", "source_family",
            "value_regime", "operator_composition", "program_family", "action_family",
            "topology_separation_witness", "exact_cluster_signature",
            "encoder_semantic_signature", "grammar_signature", "generation_attempt",
            "operator_slice", "operator_separation_witnesses",
        )
        if any(len({_sha(row["meta"][key]) for row in group}) != 1
               for key in invariant_keys):
            raise ValueError(f"{cluster_id}: inconsistent cluster invariants")
        representative = group[0]["meta"]
        for side in (0, 1):
            side_rows = [row for row in group
                         if row["meta"]["counterfactual_side"] == side]
            if len({_sha(row["meta"]["facts"]) for row in side_rows}) != 1:
                raise ValueError(f"{cluster_id}: inconsistent endpoint facts")
        facts_by_side = {side: next(row["meta"]["facts"] for row in group
                                    if row["meta"]["counterfactual_side"] == side)
                         for side in (0, 1)}
        expected_encoder = _encoder_semantic_signature(
            representative["field_order"], representative["flipped_field"], facts_by_side)
        expected_signature = _semantic_signature(
            representative["field_order"], representative["flipped_field"],
            facts_by_side, representative["policy"])
        expected_exact_signature = _exact_cluster_signature(
            expected_signature,
            facts_by_side,
        )
        if (representative["semantic_signature"] != expected_signature
                or representative["encoder_semantic_signature"] != expected_encoder
                or representative["grammar_signature"] != _grammar_signature(
                    representative["field_order"], facts_by_side[0])
                or representative["exact_cluster_signature"] != expected_exact_signature
                or representative["latent_signature"] != expected_signature):
            raise ValueError(f"{cluster_id}: invalid split-neutral semantic signature")
        by_view: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_view[(row["meta"]["language"], row["meta"]["option_order"])].append(row)
        for pair in by_view.values():
            left, right = sorted(pair, key=lambda item: item["meta"]["counterfactual_side"])
            left_meta, right_meta = left["meta"], right["meta"]
            changed_truths = {field for field in left_meta["action_trace"]
                              if left_meta["action_trace"][field]
                              != right_meta["action_trace"][field]}
            changed_facts = {field for field in left_meta["facts"]
                             if left_meta["facts"][field] != right_meta["facts"][field]}
            if (changed_truths != {left_meta["flipped_field"]}
                    or changed_facts != changed_truths
                    or left_meta["gold_outcome"] == right_meta["gold_outcome"]):
                raise ValueError(f"{cluster_id}: non-causal counterfactual pair")
            field = left_meta["flipped_field"]
            if sum(left_meta["facts"][field]["checks"][name]
                   != right_meta["facts"][field]["checks"][name] for name in CHECK_NAMES) != 1:
                raise ValueError(f"{cluster_id}: twin must change exactly one native check")
    exact_clusters: dict[str, str] = {}
    complete_serializations: dict[str, str] = {}
    encoder_cases: set[str] = set()
    policy_cases: set[str] = set()
    encoder_endpoints: set[str] = set()
    for cluster_id, group in clusters.items():
        signature = group[0]["meta"]["exact_cluster_signature"]
        previous = exact_clusters.setdefault(signature, cluster_id)
        if previous != cluster_id:
            raise ValueError(
                f"{split}: duplicate exact semantic case clusters {previous} and {cluster_id}"
            )
        serialization = _complete_cluster_serialization_signature(group)
        previous = complete_serializations.setdefault(serialization, cluster_id)
        if previous != cluster_id:
            raise ValueError(
                f"{split}: duplicate complete cluster serializations {previous} and {cluster_id}"
            )
        encoder_key = group[0]["meta"]["encoder_semantic_signature"]
        policy_key = group[0]["meta"]["semantic_signature"]
        if encoder_key in encoder_cases or policy_key in policy_cases:
            raise ValueError(f"{split}: repeated latent semantic case at {cluster_id}")
        encoder_cases.add(encoder_key)
        policy_cases.add(policy_key)
        endpoints = {row["meta"]["encoder_endpoint_signature"] for row in group}
        if len(endpoints) != 2 or not endpoints.isdisjoint(encoder_endpoints):
            raise ValueError(f"{split}: repeated encoder endpoint at {cluster_id}")
        encoder_endpoints.update(endpoints)
    if len(rows) >= 16 * CLUSTER_SIZE:
        if set(flipped_fields) != set(FIELDS):
            raise ValueError(f"{split}: every field needs causal flips")
        if max(flipped_fields.values()) > 2 * min(flipped_fields.values()):
            raise ValueError(f"{split}: causal roles are not adequately balanced")
        if (set(primitive_labels) != set(PRIMITIVES.values())
                or any(set(labels) != {False, True}
                       for labels in primitive_labels.values())):
            raise ValueError(f"{split}: primitives are degenerate")
        if set(flipped_field_modes) != {
            (field, mode) for field in FIELDS for mode in ("priority", "collect")
        }:
            raise ValueError(f"{split}: causal field roles are not crossed with policy mode")
    if split == "eval":
        if set(topology_slices) != {"same", "heldout"}:
            raise ValueError("eval: same and held-out topology slices are required")
        if set(topology_modes) != {
            "same:priority", "same:collect", "heldout:priority", "heldout:collect",
        }:
            raise ValueError("eval: policy mode/topology factors are not crossed")
        if any((row["meta"]["topology_slice"] == "same")
               != row["meta"]["policy_topology"].endswith("_coarse_gate")
               for row in rows):
            raise ValueError("eval: topology slice label does not match policy topology")
        for row in rows:
            meta = row["meta"]
            if meta["topology_slice"] == "heldout":
                development = _policy(meta["policy_mode"], meta["field_order"],
                                      f"{meta['policy_mode']}_coarse_gate")
                if _partition_signature(meta["policy"]) == _partition_signature(development):
                    raise ValueError("eval: held-out policy has an identity-equivalent partition")
    elif set(topology_slices) != {"development"}:
        raise ValueError(f"{split}: only development topology is allowed")
    elif any(not row["meta"]["policy_topology"].endswith("_coarse_gate")
             for row in rows):
        raise ValueError(f"{split}: held-out policy topology leaked into development")
    if len(rows) >= 400:
        slices = ({"same", "heldout"} if split == "eval"
                  else {"development"})
        expected_cross = {
            (size, mode, topology_slice, language, order)
            for size in range(2, 9) for mode in ("priority", "collect")
            for topology_slice in slices for language in LANGUAGES
            for order in OPTION_ORDERS
        }
        if set(crossed_cells) != expected_cross:
            raise ValueError(f"{split}: experimental factors are not fully crossed")
    design_cells = Counter(
        (len(group[0]["meta"]["field_order"]), group[0]["meta"]["policy_mode"],
         group[0]["meta"]["operator_composition"], group[0]["meta"]["value_regime"],
         group[0]["meta"]["renderer_family"], group[0]["meta"]["source_family"],
         group[0]["meta"]["topology_slice"])
        for group in clusters.values())
    surfaces = list(zip(SPLIT_FAMILIES[split]["renderer"], SPLIT_FAMILIES[split]["source"]))
    expected_design = {
        (size, mode, operator, regime, renderer, source, topology)
        for size in range(2, 9) for mode in ("priority", "collect")
        for operator in SPLIT_FAMILIES[split]["operator"]
        for regime in SPLIT_FAMILIES[split]["value_regime"]
        for renderer, source in surfaces
        for topology in ({"same", "heldout"} if split == "eval" else {"development"})}
    if len(clusters) >= len(expected_design):
        if set(design_cells) != expected_design:
            raise ValueError(f"{split}: operator/schema/surface factors are not fully crossed")
        positions = {(len(group[0]["meta"]["field_order"]),
                      group[0]["meta"]["field_order"].index(group[0]["meta"]["flipped_field"]))
                     for group in clusters.values()}
        if positions != {(size, position) for size in range(2, 9) for position in range(size)}:
            raise ValueError(f"{split}: causal pivots do not cover interior positions")
    collect_rows = [row for row in rows if row["meta"]["policy"]["mode"] == "collect"]
    if len(rows) >= 400:
        shapes = {(len(row["meta"]["gold_outcome"]["actions"]),
                   row["meta"]["gold_outcome"]["primary"] is None)
                  for row in collect_rows}
        if not ({(0, True), (1, True)} <= shapes
                and any(size > 1 for size, _ in shapes)):
            raise ValueError(f"{split}: collect outcome coverage is incomplete")
    return {
        "rows": len(rows),
        "generated_counterfactual_clusters": len(clusters),
        "unique_split_neutral_semantic_signatures": len({
            row["meta"]["semantic_signature"] for row in rows}),
        "unique_exact_semantic_case_clusters": len(exact_clusters),
        "unique_encoder_semantic_cases": len(encoder_cases),
        "unique_encoder_endpoints": len(encoder_endpoints),
        "unique_complete_cluster_serializations": len(complete_serializations),
        "counterfactual_endpoint_pairs": len(clusters) * 4,
        "semantic_endpoints": len(clusters) * 2,
        "mode_counts": dict(sorted(modes.items())),
        "field_size_counts": {str(key): value for key, value in sorted(field_sizes.items())},
        "flipped_field_counts": dict(sorted(flipped_fields.items())),
        "primitive_label_counts": {
            primitive: {str(label).lower(): count for label, count in sorted(labels.items())}
            for primitive, labels in sorted(primitive_labels.items())
        },
        "outcome_shape_counts": dict(sorted(outcomes.items())),
        "topology_slice_counts": dict(sorted(topology_slices.items())),
        "topology_mode_counts": dict(sorted(topology_modes.items())),
        "factor_cells": len(factor_cells),
        "fully_crossed_cells": len(crossed_cells),
        "design_cells": len(design_cells),
        "expected_design_cells": len(expected_design),
        "design_cell_min_repetitions": min(design_cells.values()),
        "design_cell_max_repetitions": max(design_cells.values()),
        "grammar_signatures": len({row["meta"]["grammar_signature"] for row in rows}),
        "fingerprints": {name: {"unique": len(values)}
                         for name, values in _fingerprints(rows).items()},
    }


def validate_corpus(corpus: dict[str, list[dict[str, Any]]],
                    row_counts: dict[str, int] | None = None) -> dict[str, Any]:
    expected = row_counts or {split: len(rows) for split, rows in corpus.items()}
    if set(corpus) != set(SPLITS) or set(expected) != set(SPLITS):
        raise ValueError("corpus must contain exactly train, cal, and eval splits")
    reports = {split: validate_split(corpus[split], split, expected[split])
               for split in corpus}
    fingerprints = {split: _fingerprints(rows) for split, rows in corpus.items()}
    overlaps = {}
    fuzzy = {}
    ordered_splits = [split for split in SPLITS if split in corpus]
    for left_index, left in enumerate(ordered_splits):
        for right in ordered_splits[left_index + 1:]:
            pair = f"{left}__{right}"
            overlaps[pair] = {
                name: len(fingerprints[left][name] & fingerprints[right][name])
                for name in fingerprints[left]
            }
            if any(overlaps[pair][key] for key in (
                    "model_input_exact", "model_input_normalized", "encoder_input_normalized",
                    "encoder_semantic_signature", "encoder_endpoint_signature", "semantic_signature")):
                raise ValueError(f"{pair}: protected fingerprints cross splits")
            fuzzy[pair] = _fuzzy_report(corpus[left], corpus[right])
    return {
        "splits": reports,
        "cross_split_overlaps": overlaps,
        "fuzzy_model_input": fuzzy,
        "all_protected_fingerprints_disjoint": True,
        "normalized_input_contract": (
            "case/number normalization preserving native comparison signs and mismatch/missing counts"
        ),
        "fuzzy_overlap_role": "descriptive; shared anchor grammars are intentional",
    }


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    # Key insertion order is part of the benchmark contract: sorting keys
    # would silently turn the yes/no intervention back into a positional cue.
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid persisted JSONL: {path}") from error


def write_corpus(out: Path, seed: int = 48,
                 row_counts: dict[str, int] | None = None) -> dict[str, Any]:
    counts = dict(row_counts or DEFAULT_ROWS)
    if set(counts) != set(SPLITS):
        raise ValueError("row counts must contain exactly train, cal, and eval")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"refusing to overwrite nonempty directory: {out}")
    corpus = {split: build(split, count, seed) for split, count in counts.items()}
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, rows in corpus.items():
        payload = _jsonl_bytes(rows)
        (out / f"{split}.jsonl").write_bytes(payload)
        outputs[split] = {"rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
    persisted = {split: _read_jsonl(out / f"{split}.jsonl") for split in SPLITS}
    for split in SPLITS:
        persisted_bytes = (out / f"{split}.jsonl").read_bytes()
        if persisted[split] != corpus[split] or _jsonl_bytes(persisted[split]) != persisted_bytes:
            raise ValueError(f"{split}: persisted JSONL failed exact round-trip validation")
    # The manifest is bound to an audit of bytes read back from disk, not to
    # the pre-serialization Python objects.
    audit = validate_corpus(persisted, counts)
    manifest = {
        "panel": "exact_policy_v1",
        "panel_version": PANEL_VERSION,
        "seed": seed,
        "row_counts": counts,
        "cluster_size": CLUSTER_SIZE,
        "source_contract": {
            "synthetic_only": True,
            "external_benchmark_content": False,
            "external_policy_supervision": True,
            "topology_transfer_scope": "external_policy_outcome_mapping_only",
            "semantic_encoder_policy_input": False,
            "decision_threshold": 0.5,
            "heldout_predicate_operator": "at_least_2_of_3",
            "case_reservation": "both canonical encoder endpoints sha256 modulo three splits",
            "policy_identity": "canonical sorted-field exact assignment-to-outcome truth table",
        },
        "headroom": {
            "status": "deferred",
            "reason": ("Generate unconsumed canonical endpoints and verify them against all frozen "
                       "split bytes after the core is frozen. Changing the split modulus would "
                       "require refreezing every split; an eval subset consumes final cases."),
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": outputs,
        "audit": audit,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.out, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
