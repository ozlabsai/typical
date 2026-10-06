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
PANEL_VERSION = "1.2"
FUZZY_THRESHOLD = 0.88
SPLIT_OFFSETS = {"train": 0, "cal": 100_000, "eval": 200_000}

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


def _action(name: str, field: str | None = None) -> dict[str, Any]:
    payload = {"queue": name}
    if field is not None:
        payload["reason_field"] = field
    return {"name": name, "payload": payload}


def _failure_condition(field: str, topology: str) -> dict[str, Any]:
    direct = {"not": {"field": field}}
    if topology == "direct":
        return direct
    if topology == "guarded_any":
        return {"any": [direct, {"const": False}]}
    if topology == "demorgan":
        return {"not": {"all": [{"field": field}, {"const": True}]}}
    raise ValueError(f"unknown policy topology: {topology}")


def _policy(mode: str, fields: list[str], topology: str) -> dict[str, Any]:
    rules = []
    for rank, field in enumerate(fields):
        rule = {
            "when": _failure_condition(field, topology),
            "action": _action(f"remediate_{field}", field),
        }
        if mode == "collect":
            # Some reachable singleton sets intentionally have no primary.
            rule["primary_rank"] = None if rank % 3 == 0 else rank
        rules.append(rule)
    fallback = _action("approve") if mode == "priority" else None
    return {"fields": fields, "mode": mode, "rules": rules, "fallback": fallback}


def _make_fact(field: str, truth: bool, scenario: int) -> dict[str, Any]:
    delta = 1 + scenario % 9
    if field == "identity_verified":
        return {"primitive": PRIMITIVES[field], "name_match": True,
                "birth_date_match": truth}
    if field == "policy_eligible":
        return {"primitive": PRIMITIVES[field], "tier": "standard" if truth else "excluded",
                "eligible_tiers": ["standard", "enhanced"]}
    if field == "amount_within_limit":
        limit = 100 + scenario % 701
        return {"primitive": PRIMITIVES[field], "amount": limit - delta if truth else limit + delta,
                "limit": limit}
    if field == "submission_timely":
        deadline = 5 + scenario % 25
        return {"primitive": PRIMITIVES[field],
                "submitted_day": deadline - delta if truth else deadline + delta,
                "deadline_day": deadline}
    if field == "evidence_complete":
        required = 2 + scenario % 7
        return {"primitive": PRIMITIVES[field],
                "present": required if truth else required - 1, "required": required}
    if field == "region_covered":
        return {"primitive": PRIMITIVES[field], "region": "north" if truth else "offshore",
                "covered_regions": ["north", "central", "south"]}
    if field == "consent_valid":
        review_day = 30 + scenario % 300
        return {"primitive": PRIMITIVES[field], "review_day": review_day,
                "expiry_day": review_day + delta if truth else review_day - delta}
    if field == "account_active":
        return {"primitive": PRIMITIVES[field], "status": "active" if truth else "closed"}
    raise ValueError(f"unknown field: {field}")


def _fact_truth(field: str, fact: dict[str, Any]) -> bool:
    if fact.get("primitive") != PRIMITIVES.get(field):
        raise ValueError(f"{field}: primitive mismatch")
    if field == "identity_verified":
        return fact["name_match"] is True and fact["birth_date_match"] is True
    if field == "policy_eligible":
        return fact["tier"] in fact["eligible_tiers"]
    if field == "amount_within_limit":
        return fact["amount"] <= fact["limit"]
    if field == "submission_timely":
        return fact["submitted_day"] <= fact["deadline_day"]
    if field == "evidence_complete":
        return fact["present"] >= fact["required"]
    if field == "region_covered":
        return fact["region"] in fact["covered_regions"]
    if field == "consent_valid":
        return fact["expiry_day"] >= fact["review_day"]
    if field == "account_active":
        return fact["status"] == "active"
    raise ValueError(f"unknown field: {field}")


def _render_fact(field: str, fact: dict[str, Any], language: str) -> str:
    if language == "en":
        renderers = {
            "identity_verified": lambda: f"Identity checks — name match: yes; birth-date match: {'yes' if fact['birth_date_match'] else 'no'}.",
            "policy_eligible": lambda: f"Policy tier: {fact['tier']}; eligible tiers: standard, enhanced.",
            "amount_within_limit": lambda: f"Requested amount: {fact['amount']}; maximum allowed: {fact['limit']} (eligible when amount <= maximum).",
            "submission_timely": lambda: f"Submission day: {fact['submitted_day']}; deadline day: {fact['deadline_day']} (timely when submission <= deadline).",
            "evidence_complete": lambda: f"Evidence ledger: {fact['present']} present out of {fact['required']} required.",
            "region_covered": lambda: f"Recorded region: {fact['region']}; covered regions: north, central, south.",
            "consent_valid": lambda: f"Review day: {fact['review_day']}; consent expiry day: {fact['expiry_day']} (valid when expiry >= review).",
            "account_active": lambda: f"Account status in the registry: {fact['status']}.",
        }
    else:
        renderers = {
            "identity_verified": lambda: f"בדיקות זהות — התאמת שם: כן; התאמת תאריך לידה: {'כן' if fact['birth_date_match'] else 'לא'}.",
            "policy_eligible": lambda: f"מסלול הפוליסה: {'רגיל' if fact['tier'] == 'standard' else 'מוחרג'}; המסלולים הזכאים הם רגיל ומורחב.",
            "amount_within_limit": lambda: f"הסכום המבוקש: {fact['amount']}; התקרה: {fact['limit']} (זכאות כאשר הסכום <= התקרה).",
            "submission_timely": lambda: f"יום ההגשה: {fact['submitted_day']}; היום האחרון: {fact['deadline_day']} (בזמן כאשר ההגשה <= המועד).",
            "evidence_complete": lambda: f"פנקס הראיות: קיימים {fact['present']} מתוך {fact['required']} פריטים נדרשים.",
            "region_covered": lambda: f"האזור הרשום: {'צפון' if fact['region'] == 'north' else 'מחוץ לתחום'}; האזורים המכוסים: צפון, מרכז ודרום.",
            "consent_valid": lambda: f"יום הבדיקה: {fact['review_day']}; יום פקיעת ההסכמה: {fact['expiry_day']} (בתוקף כאשר הפקיעה >= הבדיקה).",
            "account_active": lambda: f"מצב החשבון במרשם: {'פעיל' if fact['status'] == 'active' else 'סגור'}.",
        }
    return renderers[field]()


def _text(language: str, fields: list[str], facts: dict[str, dict[str, Any]],
          scenario: int) -> str:
    headers = ((f"Case record {scenario}: review the independently recorded facts.",
                f"Case record {scenario}: determine each property from the records below.")
               if language == "en" else
               (f"רשומת תיק {scenario}: בדקו את העובדות שנרשמו בנפרד.",
                f"רשומת תיק {scenario}: קבעו כל מאפיין לפי הרשומות שלהלן."))
    return "\n".join([headers[scenario % len(headers)]] +
                     [_render_fact(field, facts[field], language) for field in fields])


def _question(field: str, language: str, option_order: str) -> dict[str, Any]:
    labels = (("no", "לא"), ("yes", "כן")) if language == "he" else (
        ("no", "No"), ("yes", "Yes"))
    if option_order == "yes_no":
        labels = tuple(reversed(labels))
    instructions = QUESTION_TEXT[language][field]
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
    flip_position = fields.index(flipped_field)
    if split == "eval" and (cluster // 2) % 2:
        topology_slice = "heldout_topology"
        topology = ("guarded_any", "demorgan")[(cluster // 4) % 2]
    else:
        topology_slice = "same_topology" if split == "eval" else "development"
        topology = "direct"

    base_truths = {}
    if mode == "priority":
        for index, field in enumerate(fields):
            base_truths[field] = True if index <= flip_position else (scenario + index) % 3 != 0
    else:
        false_budget = (cluster // 2) % 3
        for index, field in enumerate(fields):
            base_truths[field] = not (index != flip_position and
                                      (index + scenario) % field_count < false_budget)
        base_truths[flipped_field] = True
    facts_by_side = {}
    false_side = (cluster // 2 + seed) % 2
    for side in (0, 1):
        truths = dict(base_truths)
        truths[flipped_field] = side != false_side
        facts_by_side[side] = {
            field: _make_fact(field, truths[field], scenario + position * 37)
            for position, field in enumerate(fields)
        }
    signature_payload = {
        "record": scenario, "mode": mode, "fields": fields,
        "flipped_field": flipped_field,
        "facts_by_side": facts_by_side,
    }
    latent_signature = hashlib.sha256(json.dumps(
        signature_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "scenario": scenario, "fields": fields, "mode": mode,
        "flipped_field": flipped_field, "facts_by_side": facts_by_side,
        "topology": topology, "topology_slice": topology_slice,
        "latent_signature": latent_signature,
    }


def _row(split: str, index: int, seed: int) -> dict[str, Any]:
    cluster, variant = divmod(index, CLUSTER_SIZE)
    side = variant % 2
    language = LANGUAGES[(variant // 2) % 2]
    option_order = OPTION_ORDERS[(variant // 4) % 2]
    spec = _cluster_spec(split, cluster, seed)
    fields = spec["fields"]
    facts = spec["facts_by_side"][side]
    truths = {field: _fact_truth(field, facts[field]) for field in fields}
    policy = _policy(spec["mode"], fields, spec["topology"])
    qid_to_field = {f"q{position:02d}": field for position, field in enumerate(fields)}
    questions = {qid: _question(field, language, option_order)
                 for qid, field in qid_to_field.items()}
    targets = {qid: {"label": "yes" if truths[field] else "no"}
               for qid, field in qid_to_field.items()}
    outcome = execute_policy(policy, truths).as_dict()
    return {
        "id": f"r048-{split}-{seed:04d}-{index:06d}",
        "state": _text(language, fields, facts, spec["scenario"]),
        "questions": questions,
        "targets": targets,
        "meta": {
            "panel": "exact_policy_v1",
            "panel_version": PANEL_VERSION,
            "split": split,
            "language": language,
            "option_order": option_order,
            "counterfactual_cluster_id": f"r048-{split}-{seed:04d}-c{cluster:06d}",
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
            "external_policy_topology": spec["topology"],
            "external_policy_topology_slice": spec["topology_slice"],
            "latent_signature": spec["latent_signature"],
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
    return [_row(split, index, seed) for index in range(count)]


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


def _fingerprints(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "model_input_exact": {_sha(_model_input(row)) for row in rows},
        "model_input_normalized": {_sha(_normalize_text(_model_input(row))) for row in rows},
        "latent_signature": {row["meta"]["latent_signature"] for row in rows},
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
            "flipped_field", "scenario_id", "external_policy_topology",
            "external_policy_topology_slice", "latent_signature", "factor_cell",
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
                or meta["counterfactual_side"] != expected_side
                or meta["language"] != expected_language
                or meta["option_order"] != expected_order
                or meta["scenario_id"] != expected_scenario):
            raise ValueError(f"{identifier}: row identity metadata mismatch")
        policy = compile_policy(meta["policy"])
        if (list(policy.fields) != meta["field_order"]
                or not 2 <= len(policy.fields) <= 8):
            raise ValueError(f"{identifier}: policy/field order mismatch")
        topology = meta["external_policy_topology"]
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
        for qid, field in expected_mapping.items():
            question = row["questions"][qid]
            if (question != _question(field, meta["language"], meta["option_order"])
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
                        field, truths[field], meta["scenario_id"] + position * 37)
                    for position, field in enumerate(policy.fields)
                }):
            raise ValueError(f"{identifier}: primitive facts are not canonical")
        if row.get("state") != _text(
                meta["language"], list(policy.fields), facts, meta["scenario_id"]):
            raise ValueError(f"{identifier}: state/facts mismatch")
        if execute_policy(policy, truths).as_dict() != meta["gold_outcome"]:
            raise ValueError(f"{identifier}: gold outcome mismatch")
        if (isinstance(meta["decision_threshold"], bool)
                or meta["decision_threshold"] != 0.5):
            raise ValueError(f"{identifier}: invalid decision threshold")
        expected_factor = (
            f"{policy.mode}|n{len(policy.fields)}|"
            f"p{list(policy.fields).index(meta['flipped_field'])}|"
            f"{meta['external_policy_topology_slice']}"
        )
        if meta["factor_cell"] != expected_factor:
            raise ValueError(f"{identifier}: invalid factor cell")
        modes[policy.mode] += 1
        outcome = meta["gold_outcome"]
        outcomes[f"{len(outcome['actions'])}:{outcome['primary'] is None}"] += 1
        field_sizes[len(policy.fields)] += 1
        flipped_fields[meta["flipped_field"]] += 1
        topology_slices[meta["external_policy_topology_slice"]] += 1
        topology_modes[f"{meta['external_policy_topology_slice']}:{policy.mode}"] += 1
        factor_cells[meta["factor_cell"]] += 1
        crossed_cells[(len(policy.fields), policy.mode,
                       meta["external_policy_topology_slice"], meta["language"],
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
        if len({row["meta"]["latent_signature"] for row in group}) != 1:
            raise ValueError(f"{cluster_id}: inconsistent latent signature")
        invariant_keys = (
            "scenario_id", "field_order", "policy", "flipped_field",
            "external_policy_topology", "external_policy_topology_slice",
        )
        if any(len({_sha(row["meta"][key]) for row in group}) != 1
               for key in invariant_keys):
            raise ValueError(f"{cluster_id}: inconsistent cluster invariants")
        representative = group[0]["meta"]
        facts_by_side = {}
        for side in (0, 1):
            side_rows = [row for row in group
                         if row["meta"]["counterfactual_side"] == side]
            if len({_sha(row["meta"]["facts"]) for row in side_rows}) != 1:
                raise ValueError(f"{cluster_id}: inconsistent endpoint facts")
            facts_by_side[side] = side_rows[0]["meta"]["facts"]
        expected_signature = _sha({
            "record": representative["scenario_id"],
            "mode": representative["policy"]["mode"],
            "fields": representative["field_order"],
            "flipped_field": representative["flipped_field"],
            "facts_by_side": facts_by_side,
        })
        if representative["latent_signature"] != expected_signature:
            raise ValueError(f"{cluster_id}: invalid latent signature")
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
        if set(topology_slices) != {"same_topology", "heldout_topology"}:
            raise ValueError("eval: same and held-out topology slices are required")
        if set(topology_modes) != {
            "same_topology:priority", "same_topology:collect",
            "heldout_topology:priority", "heldout_topology:collect",
        }:
            raise ValueError("eval: policy mode/topology factors are not crossed")
        if any((row["meta"]["external_policy_topology_slice"] == "same_topology")
               != (row["meta"]["external_policy_topology"] == "direct")
               for row in rows):
            raise ValueError("eval: topology slice label does not match policy topology")
    elif set(topology_slices) != {"development"}:
        raise ValueError(f"{split}: only development topology is allowed")
    elif any(row["meta"]["external_policy_topology"] != "direct" for row in rows):
        raise ValueError(f"{split}: held-out policy topology leaked into development")
    if len(rows) >= 400:
        slices = ({"same_topology", "heldout_topology"} if split == "eval"
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
    if len(rows) >= 7 * CLUSTER_SIZE:
        shapes = {(len(row["meta"]["gold_outcome"]["actions"]),
                   row["meta"]["gold_outcome"]["primary"] is None)
                  for row in collect_rows}
        if not ({(0, True), (1, True)} <= shapes
                and any(size > 1 for size, _ in shapes)):
            raise ValueError(f"{split}: collect outcome coverage is incomplete")
    return {
        "rows": len(rows),
        "independent_latent_clusters": len(clusters),
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
                    or overlaps[pair]["latent_signature"]):
                raise ValueError(f"{pair}: protected fingerprints cross splits")
            fuzzy[pair] = _fuzzy_report(corpus[left], corpus[right])
    return {
        "splits": reports,
        "cross_split_overlaps": overlaps,
        "fuzzy_model_input": fuzzy,
        "all_protected_fingerprints_disjoint": True,
        "fuzzy_is_diagnostic_not_a_disjointness_claim": True,
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
