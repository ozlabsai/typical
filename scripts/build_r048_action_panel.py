"""Build the deterministic, counterfactual R048 exact-policy v1.2 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.exact_policy_executor import compile_policy, execute_policy


DEFAULT_ROWS = {"train": 1600, "cal": 400, "eval": 2000}
SPLITS = tuple(DEFAULT_ROWS)
OPTION_ORDERS = ("no_yes", "yes_no")
LANGUAGES = ("en", "he")
CLUSTER_SIZE = 8
PANEL_VERSION = "1.3"
FUZZY_THRESHOLD = 0.92
FUZZY_MAX_PAIR_RATE = 0.0001
FUZZY_MAX_SIMILARITY = 0.98
SPLIT_OFFSETS = {"train": 0, "cal": 100_000, "eval": 200_000}

SPLIT_FAMILIES = {
    "train": {
        "renderer": ("case_ledger", "fact_sheet"),
        "source": ("claims_intake", "purchase_control"),
        "value_regime": ("observed_threshold", "bounded_interval"),
        "operator": ("direct_le", "closed_interval"),
    },
    "cal": {
        "renderer": ("review_dossier", "verification_note"),
        "source": ("grant_review", "regulatory_check"),
        "value_regime": ("signed_headroom", "violation_count"),
        "operator": ("nonnegative_margin", "zero_violations"),
    },
    "eval": {
        "renderer": ("incident_digest", "assessment_brief"),
        "source": ("service_recovery", "risk_underwriting"),
        "value_regime": ("conjunctive_checks", "weighted_gate"),
        "operator": ("all_checks", "sum_gate"),
    },
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
        "identity_verified": "Do the identity records agree?",
        "policy_eligible": "Is the case in an eligible policy tier?",
        "amount_within_limit": "Is the requested amount within the stated limit?",
        "submission_timely": "Was the submission made by the deadline?",
        "evidence_complete": "Are all required evidence items present?",
        "region_covered": "Is the recorded region covered?",
        "consent_valid": "Is the consent still valid on the review date?",
        "account_active": "Is the account active?",
    },
    "he": {
        "identity_verified": "האם פרטי הזהות תואמים זה לזה?",
        "policy_eligible": "האם התיק שייך למסלול זכאי?",
        "amount_within_limit": "האם הסכום המבוקש אינו חורג מהתקרה?",
        "submission_timely": "האם ההגשה בוצעה עד המועד האחרון?",
        "evidence_complete": "האם כל מסמכי הראיות הנדרשים קיימים?",
        "region_covered": "האם האזור הרשום נכלל בכיסוי?",
        "consent_valid": "האם ההסכמה עדיין בתוקף במועד הבדיקה?",
        "account_active": "האם החשבון פעיל?",
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
    direct_name = f"{mode}_fieldwise"
    grouped_name = f"{mode}_grouped_partition"
    if topology == direct_name:
        rules = []
        for rank, field in enumerate(fields):
            rule = {
                "when": {"not": {"field": field}},
                "action": _action(f"remediate_{field}", field),
            }
            if mode == "collect":
                rule["primary_rank"] = None if rank % 3 == 0 else rank
            rules.append(rule)
    elif topology == grouped_name:
        any_failure = {"any": [{"not": {"field": field}} for field in fields]}
        if mode == "priority":
            rules = [{"when": any_failure,
                      "action": _action("review_combined_failure")}]
        else:
            all_failure = {"all": [{"not": {"field": field}} for field in fields]}
            rules = [
                {"when": any_failure, "action": _action("review_any_failure"),
                 "primary_rank": None},
                {"when": all_failure, "action": _action("escalate_total_failure"),
                 "primary_rank": 0},
            ]
    else:
        raise ValueError(f"unknown {mode} policy topology: {topology}")
    fallback = _action("approve") if mode == "priority" else None
    return {"fields": fields, "mode": mode, "rules": rules, "fallback": fallback}


def _make_fact(field: str, truth: bool, scenario: int, value_regime: str,
               operator: str) -> dict[str, Any]:
    if field not in PRIMITIVES:
        raise ValueError(f"unknown field: {field}")
    delta = 1 + scenario % 9
    base = 20 + scenario % 71
    common = {"primitive": PRIMITIVES[field], "value_regime": value_regime,
              "operator_composition": operator}
    if operator == "direct_le":
        values = {"observed": base - delta if truth else base + delta,
                  "boundary": base}
    elif operator == "closed_interval":
        values = {"lower": base - 10,
                  "observed": base if truth else base + 10 + delta,
                  "upper": base + 10}
    elif operator == "nonnegative_margin":
        values = {"headroom": delta if truth else -delta}
    elif operator == "zero_violations":
        values = {"violations": 0 if truth else delta}
    elif operator == "all_checks":
        values = {"checks": [True, truth, True]}
    elif operator == "sum_gate":
        limit = base + delta
        values = {"components": [base // 2, base - base // 2 if truth else base],
                  "limit": limit}
    else:
        raise ValueError(f"unknown operator composition: {operator}")
    return {**common, **values}


def _fact_truth(field: str, fact: dict[str, Any]) -> bool:
    if fact.get("primitive") != PRIMITIVES.get(field):
        raise ValueError(f"{field}: primitive mismatch")
    operator = fact.get("operator_composition")
    if operator == "direct_le":
        return fact["observed"] <= fact["boundary"]
    if operator == "closed_interval":
        return fact["lower"] <= fact["observed"] <= fact["upper"]
    if operator == "nonnegative_margin":
        return fact["headroom"] >= 0
    if operator == "zero_violations":
        return fact["violations"] == 0
    if operator == "all_checks":
        return all(fact["checks"])
    if operator == "sum_gate":
        return sum(fact["components"]) <= fact["limit"]
    raise ValueError(f"{field}: unknown operator composition")


def _render_fact(field: str, fact: dict[str, Any], language: str) -> str:
    subject = QUESTION_TEXT[language][field].rstrip("?")
    operator = fact["operator_composition"]
    if language == "en":
        if operator == "direct_le":
            detail = f"observed {fact['observed']}; boundary {fact['boundary']}; observed <= boundary"
        elif operator == "closed_interval":
            detail = (f"lower {fact['lower']}; observation {fact['observed']}; upper "
                      f"{fact['upper']}; lower <= observation <= upper")
        elif operator == "nonnegative_margin":
            detail = f"signed headroom {fact['headroom']}; acceptable when headroom >= 0"
        elif operator == "zero_violations":
            detail = f"violation count {fact['violations']}; acceptable only at zero"
        elif operator == "all_checks":
            detail = f"three independent checks {fact['checks']}; acceptable only when all pass"
        else:
            detail = (f"weighted components {fact['components']}; gate {fact['limit']}; "
                      "acceptable when their sum <= gate")
        return f"{subject} — {detail}."
    if operator == "direct_le":
        detail = f"ערך נצפה {fact['observed']}; גבול {fact['boundary']}; הערך הנצפה <= הגבול"
    elif operator == "closed_interval":
        detail = (f"סף תחתון {fact['lower']}; תצפית {fact['observed']}; סף עליון "
                  f"{fact['upper']}; הסף התחתון <= התצפית <= הסף העליון")
    elif operator == "nonnegative_margin":
        detail = f"מרווח חתום {fact['headroom']}; תקין כאשר המרווח >= 0"
    elif operator == "zero_violations":
        detail = f"מספר חריגות {fact['violations']}; תקין רק כאשר המספר אפס"
    elif operator == "all_checks":
        checks = ["עבר" if value else "נכשל" for value in fact["checks"]]
        detail = f"שלוש בדיקות עצמאיות {checks}; תקין רק כאשר כולן עברו"
    else:
        detail = (f"רכיבים משוקללים {fact['components']}; שער {fact['limit']}; "
                  "תקין כאשר סכומם <= השער")
    return f"{subject} — {detail}."


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


def _cluster_spec(split: str, cluster: int, seed: int) -> dict[str, Any]:
    scenario = seed * 1_000_000 + SPLIT_OFFSETS[split] + cluster
    field_count = 2 + cluster % 7
    start = (cluster + seed) % len(FIELDS)
    flipped_field = FIELDS[start]
    fields = [FIELDS[(start + offset) % len(FIELDS)] for offset in range(field_count)]
    if (cluster // 14 + seed) % 2:
        fields.reverse()
    # Blocks of seven cross both policy modes with every 2--8 field size.
    mode = ("priority", "collect")[(cluster // 7 + seed) % 2]
    if split == "eval" and (cluster // 2) % 2:
        topology_slice = "heldout"
        topology = f"{mode}_grouped_partition"
    else:
        topology_slice = "same" if split == "eval" else "development"
        topology = f"{mode}_fieldwise"
    family_index = (cluster // 7 + cluster + seed) % 2
    families = SPLIT_FAMILIES[split]
    renderer = families["renderer"][family_index]
    source = families["source"][family_index]
    value_regime = families["value_regime"][family_index]
    operator = families["operator"][family_index]

    policy = _policy(mode, fields, topology)
    other_fields = [field for field in fields if field != flipped_field]
    candidates = []
    for bits in range(1 << len(other_fields)):
        base = {field: bool(bits & (1 << index))
                for index, field in enumerate(other_fields)}
        low = {**base, flipped_field: False}
        high = {**base, flipped_field: True}
        if execute_policy(policy, low) != execute_policy(policy, high):
            candidates.append((low, high))
    if not candidates:
        raise ValueError(f"policy does not expose a causal twin for {flipped_field}")
    low, high = candidates[(cluster // 2 + seed) % len(candidates)]
    facts_by_side = {}
    false_side = (cluster // 2 + seed) % 2
    truths_by_side = {}
    for side in (0, 1):
        truths = low if side == false_side else high
        truths_by_side[side] = truths
        facts_by_side[side] = {
            field: _make_fact(field, truths[field], scenario + position * 37,
                              value_regime, operator)
            for position, field in enumerate(fields)
        }
    semantic_signature = _semantic_signature(
        fields, mode, flipped_field, truths_by_side, renderer, source,
        value_regime, operator, policy,
    )
    return {
        "scenario": scenario, "fields": fields, "mode": mode,
        "flipped_field": flipped_field, "facts_by_side": facts_by_side,
        "topology": topology, "topology_slice": topology_slice,
        "renderer": renderer, "source": source, "value_regime": value_regime,
        "operator": operator, "semantic_signature": semantic_signature,
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
            "renderer_family": spec["renderer"],
            "source_family": spec["source"],
            "value_regime": spec["value_regime"],
            "operator_composition": spec["operator"],
            "topology_transfer_scope": "external_policy_outcome_mapping_only",
            "external_policy_topology": spec["topology"],
            "external_policy_topology_slice": spec["topology_slice"],
            "semantic_signature": spec["semantic_signature"],
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
    for cluster in range(count // CLUSTER_SIZE):
        spec = _cluster_spec(split, cluster, seed)
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


def _semantic_signature(fields: list[str], mode: str, flipped_field: str,
                        truths_by_side: dict[int, dict[str, bool]],
                        renderer_family: str, source_family: str,
                        value_regime: str, operator_composition: str,
                        policy: dict[str, Any]) -> str:
    """Hash semantics only: deliberately excludes split, IDs, and raw values."""
    return _sha({
        "fields": fields,
        "mode": mode,
        "flipped_field": flipped_field,
        "endpoint_truths": truths_by_side,
        "renderer_family": renderer_family,
        "source_family": source_family,
        "value_regime": value_regime,
        "operator_composition": operator_composition,
        "policy_partition": _partition_signature(policy),
    })


def _fingerprints(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "model_input_exact": {_sha(_model_input(row)) for row in rows},
        "model_input_normalized": {_sha(_normalize_text(_model_input(row))) for row in rows},
        "semantic_signature": {row["meta"]["semantic_signature"] for row in rows},
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
    maximum = 0.0
    above = 0
    for left_set in left_sets:
        for right_set in right_sets:
            union = len(left_set | right_set)
            score = len(left_set & right_set) / union if union else 1.0
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
        "preregistered_max_pair_rate": FUZZY_MAX_PAIR_RATE,
        "preregistered_max_similarity": FUZZY_MAX_SIMILARITY,
    }


def validate_split(rows: list[dict[str, Any]], split: str,
                   expected_rows: int | None = None) -> dict[str, Any]:
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
            "semantic_signature", "latent_signature", "factor_cell",
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
        family_cell = (meta["renderer_family"], meta["source_family"],
                       meta["value_regime"], meta["operator_composition"])
        if family_cell not in set(zip(
                families["renderer"], families["source"],
                families["value_regime"], families["operator"])):
            raise ValueError(f"{identifier}: invalid split family cell")
        for qid, field in expected_mapping.items():
            question = row["questions"][qid]
            if (question != _question(field, meta["language"], meta["option_order"],
                                      meta["renderer_family"])
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
                or facts != {
                    field: _make_fact(
                        field, truths[field], meta["scenario_id"] + position * 37,
                        meta["value_regime"], meta["operator_composition"])
                    for position, field in enumerate(policy.fields)
                }):
            raise ValueError(f"{identifier}: primitive facts are not canonical")
        if row.get("state") != _text(
                meta["language"], list(policy.fields), facts,
                meta["renderer_family"], meta["source_family"]):
            raise ValueError(f"{identifier}: state/facts mismatch")
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
            "value_regime", "operator_composition",
        )
        if any(len({_sha(row["meta"][key]) for row in group}) != 1
               for key in invariant_keys):
            raise ValueError(f"{cluster_id}: inconsistent cluster invariants")
        representative = group[0]["meta"]
        truths_by_side = {}
        for side in (0, 1):
            side_rows = [row for row in group
                         if row["meta"]["counterfactual_side"] == side]
            if len({_sha(row["meta"]["facts"]) for row in side_rows}) != 1:
                raise ValueError(f"{cluster_id}: inconsistent endpoint facts")
            truths_by_side[side] = side_rows[0]["meta"]["action_trace"]
        expected_signature = _semantic_signature(
            representative["field_order"], representative["policy_mode"],
            representative["flipped_field"], truths_by_side,
            representative["renderer_family"], representative["source_family"],
            representative["value_regime"], representative["operator_composition"],
            representative["policy"],
        )
        if (representative["semantic_signature"] != expected_signature
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
               != row["meta"]["policy_topology"].endswith("_fieldwise")
               for row in rows):
            raise ValueError("eval: topology slice label does not match policy topology")
        for row in rows:
            meta = row["meta"]
            if meta["topology_slice"] == "heldout":
                direct = _policy(meta["policy_mode"], meta["field_order"],
                                 f"{meta['policy_mode']}_fieldwise")
                if _partition_signature(meta["policy"]) == _partition_signature(direct):
                    raise ValueError("eval: held-out policy has an identity-equivalent partition")
    elif set(topology_slices) != {"development"}:
        raise ValueError(f"{split}: only development topology is allowed")
    elif any(not row["meta"]["policy_topology"].endswith("_fieldwise")
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
            if (overlaps[pair]["model_input_exact"]
                    or overlaps[pair]["model_input_normalized"]
                    or overlaps[pair]["semantic_signature"]):
                raise ValueError(f"{pair}: protected fingerprints cross splits")
            fuzzy[pair] = _fuzzy_report(corpus[left], corpus[right])
            if (fuzzy[pair]["pair_rate_at_or_above_threshold"] > FUZZY_MAX_PAIR_RATE
                    or fuzzy[pair]["maximum_similarity"] > FUZZY_MAX_SIMILARITY):
                raise ValueError(f"{pair}: fuzzy model-input overlap exceeds preregistered cap")
    return {
        "splits": reports,
        "cross_split_overlaps": overlaps,
        "fuzzy_model_input": fuzzy,
        "all_protected_fingerprints_disjoint": True,
        "fuzzy_overlap_passes_preregistered_cap": True,
    }


def write_corpus(out: Path, seed: int = 48,
                 row_counts: dict[str, int] | None = None) -> dict[str, Any]:
    counts = dict(row_counts or DEFAULT_ROWS)
    if set(counts) != set(SPLITS):
        raise ValueError("row counts must contain exactly train, cal, and eval")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"refusing to overwrite nonempty directory: {out}")
    corpus = {split: build(split, count, seed) for split, count in counts.items()}
    audit = validate_corpus(corpus, counts)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, rows in corpus.items():
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                          for row in rows).encode()
        (out / f"{split}.jsonl").write_bytes(payload)
        outputs[split] = {"rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
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
