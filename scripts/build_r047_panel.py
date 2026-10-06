"""Build the generator-disjoint R047 causal panel and legacy smoke slice.

The causal outputs are flat, one-question rows.  ``semantic_cluster_id`` is the
bootstrap unit and ``semantic_view_id`` joins the final prediction to its
clause predictions.  Benchmark data is deliberately not an input to this
builder; leakage checks belong to :mod:`scripts.preflight_r047_panel`.
"""

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from scripts.build_r045_compositional_transfer import COMPOSITIONS, _pair, execute
except ModuleNotFoundError:  # Supports ``uv run python scripts/build_r047_panel.py``.
    from build_r045_compositional_transfer import COMPOSITIONS, _pair, execute


KINDS = ("decimal", "fraction", "datetime", "ledger", "conditional", "policy")
SEEN_K2 = (("decimal", "policy"), ("ledger", "conditional"))
UNSEEN_K2 = tuple(c for c in itertools.combinations(KINDS, 2) if c not in SEEN_K2)
K3 = tuple(itertools.combinations(KINDS, 3))
FROZEN_COUNTS = {"unseen_k2": 24, "k3": 16, "seen_k2": 24}
PROMPT_FAMILIES = ("criterion_probe", "evidence_check", "binary_audit")
GENERATOR_FAMILY_ID = "r047-primary-generator-v1"
SEED = 4701

ENTITIES = (
    "Juniper Quay", "Lapis Station", "Mica Annex", "Nimbus Yard",
    "Onyx Depot", "Pollen Wharf", "Quartz Hub", "Raven Terminal",
)
RENDERERS = ("clearance_memo", "verification_docket", "operations_worksheet",
             "compliance_transcript")
EN_TEMPLATES = {
    "clearance_memo": "Clearance memo — subject {entity}.\n{evidence}\nResolve the requested checks.",
    "verification_docket": "Verification docket for {entity}:\n{evidence}\nReturn the audit decisions.",
    "operations_worksheet": "Operations worksheet [{entity}]\n{evidence}\nComplete the binary review.",
    "compliance_transcript": "Compliance transcript concerning {entity}.\n{evidence}\nEvaluate the record.",
}
HE_TEMPLATES = {
    "clearance_memo": "מזכר אישור — נושא: {entity}.\n{evidence}\nיש להכריע בבדיקות המבוקשות.",
    "verification_docket": "תיק אימות עבור {entity}:\n{evidence}\nיש להחזיר את החלטות הביקורת.",
    "operations_worksheet": "גיליון תפעול [{entity}]\n{evidence}\nיש להשלים בדיקה בינארית.",
    "compliance_transcript": "תמליל ציות בנוגע אל {entity}.\n{evidence}\nיש להעריך את הרשומה.",
}
RULE = {
    "en": "Joint rule: clearance is granted exactly when every numbered requirement is true.",
    "he": "כלל משולב: האישור ניתן בדיוק כאשר כל הדרישות הממוספרות נכונות.",
}
FINAL_QUESTION = {
    "en": "Taking every numbered requirement together, is clearance granted?",
    "he": "כאשר בוחנים את כל הדרישות הממוספרות יחד, האם ניתן אישור?",
}
CLAUSE_QUESTIONS = {
    "criterion_probe": {
        "en": "For this record alone, is numbered requirement {position} true?",
        "he": "לפי הרשומה בלבד, האם דרישה ממוספרת {position} נכונה?",
    },
    "evidence_check": {
        "en": "Does the evidence establish numbered requirement {position}?",
        "he": "האם הראיות מבססות את דרישה ממוספרת {position}?",
    },
    "binary_audit": {
        "en": "Mark whether numbered requirement {position} is satisfied.",
        "he": "יש לסמן אם דרישה ממוספרת {position} מתקיימת.",
    },
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def _jsonl(rows):
    # Dict insertion order is part of the choice-presentation contract.  In
    # particular, recursively sorting keys would silently erase option-order
    # counterbalancing (``no,yes`` sorts identically to ``yes,no``).
    return b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
                    for row in rows)


def _quantized(value):
    return str(Decimal(value).quantize(Decimal("0.01")))


def _panel_clause(kind, variant, passed):
    """Create a fresh-regime clause executable by the shared R045 oracle."""
    margin = (Decimal("0.00"), Decimal("0.19"), Decimal("1.37"),
              Decimal("4.61"))[variant % 4]
    signed_margin = margin if passed else -(margin or Decimal("0.23"))
    if kind == "decimal":
        threshold = Decimal(50_000 + variant * 7) + Decimal("0.43")
        fee = Decimal(31 + variant % 17) + Decimal("0.29")
        tail = [Decimal("-13.17"), Decimal("8.41"), Decimal("21.73")]
        if variant % 2:
            tail.append(Decimal("-4.09"))
        first = threshold + fee + signed_margin - sum(tail)
        return {"kind": kind, "credits": [_quantized(first), *map(_quantized, tail)],
                "fee": _quantized(fee), "threshold": _quantized(threshold),
                "unit": "clearance-points"}
    if kind == "fraction":
        denominator = 211 + 2 * (variant % 89)
        threshold = (2, 5) if variant % 3 == 0 else ((3, 7) if variant % 3 == 1 else (5, 8))
        boundary = (threshold[0] * denominator + threshold[1] - 1) // threshold[1]
        return {"kind": kind, "numerator": boundary + (variant % 2 if passed else -1),
                "denominator": denominator, "threshold": list(threshold)}
    if kind == "datetime":
        zones = ("Asia/Tokyo", "Australia/Perth", "America/Los_Angeles",
                 "Africa/Johannesburg", "America/Sao_Paulo")
        zone = zones[variant % len(zones)]
        instant = datetime(2055 + variant % 3, 2 + variant % 9, 2 + variant % 23,
                           (variant * 7) % 24, 17, tzinfo=timezone.utc)
        local = instant.astimezone(ZoneInfo(zone)).replace(tzinfo=None)
        delta = timedelta(minutes=(0 if passed and variant % 3 == 0 else
                                   47 if passed else -53))
        deadline = (instant + delta).replace(tzinfo=None)
        return {"kind": kind, "local_time": local.isoformat(), "timezone": zone,
                "deadline": deadline.isoformat()}
    if kind == "ledger":
        threshold = Decimal(70_000 + variant * 11)
        desired = threshold + signed_margin
        length = 4 + variant % 4
        values = [Decimal(90_000 + variant), Decimal("-1723.31"), Decimal("491.27")]
        values.extend(Decimal(((-1) ** i) * (37 + i * 13)) + Decimal("0.11")
                      for i in range(length - 3))
        values[-1] += desired - sum(values)
        events = [{"event_id": f"txn-{variant:05d}-{i:02d}", "delta": _quantized(value)}
                  for i, value in enumerate(values)]
        shift = (variant * 3) % len(events)
        events = events[shift:] + events[:shift]
        return {"kind": kind, "events": events, "threshold": _quantized(threshold),
                "currency": "audit-units"}
    if kind == "conditional":
        total = 307 + variant % 97
        threshold = (4, 9) if variant % 2 else (7, 13)
        boundary = (threshold[0] * total + threshold[1] - 1) // threshold[1]
        success = boundary + (variant % 2 if passed else -1)
        groups = [
            {"name": "reviewed", "success": success, "failure": total - success},
            {"name": "unreviewed", "success": 23 + variant % 11, "failure": 19 + variant % 7},
            {"name": "deferred", "success": 5 + variant % 5, "failure": 29 + variant % 13},
        ]
        shift = variant % len(groups)
        return {"kind": kind, "condition_group": "reviewed",
                "groups": groups[shift:] + groups[:shift], "threshold": list(threshold)}
    if kind == "policy":
        base = 500 + variant % 131
        allow_priority = base + (17 if passed else 9)
        deny_priority = base + (16 if passed else 10)
        rules = [
            {"name": "fallback-bar", "priority": 1, "effect": "deny", "active": True,
             "scope": "all"},
            {"name": f"permit-{variant}", "priority": allow_priority, "effect": "allow",
             "active": True, "scope": "clearance"},
            {"name": f"bar-{variant}", "priority": deny_priority, "effect": "deny",
             "active": True, "scope": "clearance"},
            {"name": "inactive-superrule", "priority": base + 200, "effect": "deny",
             "active": False, "scope": "clearance"},
            {"name": "foreign-superrule", "priority": base + 300, "effect": "deny",
             "active": True, "scope": "shipping"},
        ]
        shift = variant % len(rules)
        return {"kind": kind, "rules": rules[shift:] + rules[:shift],
                "request_scope": "clearance",
                "precedence": "active_scope_then_highest_priority_then_deny"}
    raise ValueError(f"unknown clause kind: {kind}")


def _describe(clause, language):
    kind = clause["kind"]
    if language == "en":
        if kind == "decimal":
            body = (f"combine entries {'; '.join(clause['credits'])}, then subtract charge "
                    f"{clause['fee']} {clause['unit']}; result must be at least {clause['threshold']}")
        elif kind == "fraction":
            body = (f"compare measured share {clause['numerator']} of {clause['denominator']} "
                    f"with floor {clause['threshold'][0]} of {clause['threshold'][1]}")
        elif kind == "datetime":
            body = (f"translate {clause['local_time']} from {clause['timezone']} to UTC; "
                    f"it must not follow {clause['deadline']} UTC")
        elif kind == "ledger":
            entries = " | ".join(f"{e['event_id']}={e['delta']}" for e in clause["events"])
            body = (f"total every posting ({entries}); total must reach {clause['threshold']} "
                    f"{clause['currency']}")
        elif kind == "conditional":
            groups = " | ".join(f"{g['name']}({g['success']} pass,{g['failure']} fail)"
                                for g in clause["groups"])
            body = (f"within {clause['condition_group']} only, pass/(pass+fail) must reach "
                    f"{clause['threshold'][0]}/{clause['threshold'][1]}; cohorts: {groups}")
        else:
            rules = " | ".join(f"{r['name']}:{r['effect']}:{r['priority']}:"
                               f"{'on' if r['active'] else 'off'}:{r['scope']}"
                               for r in clause["rules"])
            body = (f"for scope {clause['request_scope']}, ignore inactive/foreign rules, then use "
                    f"the greatest priority (deny wins ties): {rules}")
        return f"{kind} check — {body}."
    if kind == "decimal":
        body = (f"יש לחבר את הרשומות {'; '.join(clause['credits'])} ואז להפחית חיוב "
                f"{clause['fee']} {clause['unit']}; התוצאה חייבת להיות לפחות {clause['threshold']}")
    elif kind == "fraction":
        body = (f"יש להשוות חלק נמדד {clause['numerator']} מתוך {clause['denominator']} "
                f"לסף {clause['threshold'][0]} מתוך {clause['threshold'][1]}")
    elif kind == "datetime":
        body = (f"יש להמיר את {clause['local_time']} מאזור {clause['timezone']} ל-UTC; "
                f"אסור שהזמן יהיה אחרי {clause['deadline']} UTC")
    elif kind == "ledger":
        entries = " | ".join(f"{e['event_id']}={e['delta']}" for e in clause["events"])
        body = (f"יש לסכם כל רישום ({entries}); הסכום חייב להגיע אל {clause['threshold']} "
                f"{clause['currency']}")
    elif kind == "conditional":
        groups = " | ".join(f"{g['name']}({g['success']} הצלחה,{g['failure']} כישלון)"
                            for g in clause["groups"])
        body = (f"רק בתוך {clause['condition_group']}, היחס הצלחה/(הצלחה+כישלון) חייב להגיע אל "
                f"{clause['threshold'][0]}/{clause['threshold'][1]}; קבוצות: {groups}")
    else:
        rules = " | ".join(f"{r['name']}:{r['effect']}:{r['priority']}:"
                           f"{'פעיל' if r['active'] else 'כבוי'}:{r['scope']}"
                           for r in clause["rules"])
        body = (f"עבור תחום {clause['request_scope']}, יש להתעלם מכללים כבויים או זרים ואז לבחור "
                f"בעדיפות הגבוהה ביותר (דחייה מנצחת בשוויון): {rules}")
    return f"בדיקת {kind} — {body}."


def _cluster_plan(counts=None):
    counts = FROZEN_COUNTS if counts is None else counts
    plan = []
    for stratum, compositions in (("unseen_k2", UNSEEN_K2), ("k3", K3),
                                  ("seen_k2", SEEN_K2)):
        for composition_index, composition in enumerate(compositions):
            for scenario_index in range(counts[stratum]):
                plan.append((stratum, composition, composition_index, scenario_index))
    return plan


def _criteria(language, option_order):
    labels = {"yes": "Yes" if language == "en" else "כן",
              "no": "No" if language == "en" else "לא"}
    keys = ("yes", "no") if option_order == "yes_no" else ("no", "yes")
    return {key: labels[key] for key in keys}


def _row(cluster_meta, state, clauses, truth, language, option_order, role,
         clause_index=None):
    composition = cluster_meta["composition"]
    truth_code = "".join("1" if value else "0" for value in truth)
    semantic_view_id = (f"{cluster_meta['semantic_cluster_id']}::{truth_code}::"
                        f"{language}::{option_order}")
    if role == "final":
        instruction, primitive, decisive_role, family = (
            FINAL_QUESTION[language], None, "final", "final")
        gold = execute(cluster_meta["program"], clauses)
        suffix = "final"
    else:
        family = cluster_meta["clause_prompt_family"]
        instruction = CLAUSE_QUESTIONS[family][language].format(position=clause_index + 1)
        primitive = composition[clause_index]
        decisive_role = ("decisive" if all(truth[i] for i in range(len(truth))
                                            if i != clause_index) else "non_decisive")
        gold = truth[clause_index]
        suffix = f"clause-{clause_index}"
    row_id = f"{semantic_view_id}::{suffix}"
    common = {
        **cluster_meta,
        "semantic_view_id": semantic_view_id,
        "language": language,
        "option_order": option_order,
        "prompt_family": family,
        "truth_pattern": list(truth),
        "truth_hamming_weight": sum(truth),
        "prediction_role": role,
        "primitive": primitive,
        "decisive_role": decisive_role,
        "clause_index": clause_index,
        "clauses": clauses,
        "gold": gold,
        "gold_source": "stdlib-independent-executable-ast",
    }
    return {"id": row_id, "state": state,
            "questions": {"decision": {"type": "choice", "instructions": instruction,
                                         "criteria": _criteria(language, option_order)}},
            "targets": {"decision": {"label": "yes" if gold else "no"}},
            "meta": common}


def _cluster_rows(stratum, composition, composition_index, scenario_index, seed):
    arity = len(composition)
    cluster_id = (f"r047p-{seed}-{stratum}-c{composition_index:02d}-"
                  f"s{scenario_index:03d}")
    prompt_family = PROMPT_FAMILIES[(composition_index + scenario_index) % len(PROMPT_FAMILIES)]
    renderer = RENDERERS[(composition_index * 3 + scenario_index) % len(RENDERERS)]
    entity = ENTITIES[(composition_index + scenario_index * 5) % len(ENTITIES)]
    seen_edge_count = sum(tuple(pair) in SEEN_K2 for pair in itertools.combinations(composition, 2))
    program = {"language": "r047-panel-ast-v1", "composition": list(composition),
               "operator": "all"}
    common = {
        "source": "r047_panel_generated",
        "panel_stratum": stratum,
        "semantic_cluster_id": cluster_id,
        "scenario_id": cluster_id,
        "cluster_lineage": {"root": cluster_id, "parent": None,
                            "generator_family_id": GENERATOR_FAMILY_ID},
        "generator_family_id": GENERATOR_FAMILY_ID,
        "generator_version": 1,
        "composition": list(composition),
        "arity": arity,
        "seen_edge_count": seen_edge_count,
        "near_transfer_diagnostic": stratum == "seen_k2",
        "clause_prompt_family": prompt_family,
        "renderer": renderer,
        "template_id": f"r047-panel-{renderer}-v1",
        "rendered_contract": "r047-primary-explicit-all-v1",
        "entity_vocab": entity,
        "numeric_regime": "r047-high-range-variable-margin-v1",
        "program": program,
        "seed": seed,
    }
    rows = []
    variant = int(_sha(f"{seed}:{cluster_id}".encode())[:10], 16)
    for truth in itertools.product((False, True), repeat=arity):
        clauses = [_panel_clause(kind, variant + position * 997, truth[position])
                   for position, kind in enumerate(composition)]
        if [clause["kind"] for clause in clauses] != list(composition):
            raise AssertionError("clause order must match composition")
        if [execute({"language": "single", "composition": [clause["kind"]], "operator": "all"},
                    [clause]) for clause in clauses] != list(truth):
            raise AssertionError("clause generator disagrees with shared executable oracle")
        for language in ("en", "he"):
            condition = "Requirement" if language == "en" else "דרישה"
            evidence = "\n".join(f"{condition} {i + 1}: {_describe(clause, language)}"
                                 for i, clause in enumerate(clauses))
            template = EN_TEMPLATES[renderer] if language == "en" else HE_TEMPLATES[renderer]
            state = template.format(entity=entity, evidence=f"{RULE[language]}\n{evidence}")
            for option_order in ("yes_no", "no_yes"):
                rows.append(_row(common, state, clauses, truth, language, option_order, "final"))
                rows.extend(_row(common, state, clauses, truth, language, option_order, "clause", i)
                            for i in range(arity))
    return rows


def _legacy_smoke(seed=47, domain_offset=2):
    rows, index = [], sum(len(COMPOSITIONS[split]) for split in ("train", "cal")) * 12
    for composition in COMPOSITIONS["test"]:
        for case_index in range(12):
            for row in _pair("test", composition, index, seed, case_index, domain_offset):
                row["meta"]["panel_stratum"] = "legacy_near_transfer_smoke"
                row["meta"]["causal_panel"] = False
                for view in row["paired_views"]:
                    language = view["meta"]["language"]
                    view["questions"]["decision"]["instructions"] = FINAL_QUESTION[language]
                    view["meta"]["panel_stratum"] = "legacy_near_transfer_smoke"
                    view["meta"]["causal_panel"] = False
                    view["meta"]["question_repair"] = "composition-neutral-v1"
                rows.append(row)
            index += 1
    return rows


def build(out, seed=SEED, counts=None):
    if seed < 0:
        raise ValueError("seed must be non-negative")
    effective_counts = dict(FROZEN_COUNTS if counts is None else counts)
    if set(effective_counts) != set(FROZEN_COUNTS) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in effective_counts.values()):
        raise ValueError(f"counts must be positive integers for {sorted(FROZEN_COUNTS)}")
    primary, near = [], []
    cluster_counts = {"unseen_k2": 0, "k3": 0, "seen_k2": 0}
    for stratum, composition, composition_index, scenario_index in _cluster_plan(effective_counts):
        target = near if stratum == "seen_k2" else primary
        target.extend(_cluster_rows(stratum, composition, composition_index, scenario_index, seed))
        cluster_counts[stratum] += 1
    legacy = _legacy_smoke()
    outputs = {"primary": primary, "near-transfer": near, "legacy-smoke": legacy}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {name: _jsonl(rows) for name, rows in outputs.items()}
    for name, payload in payloads.items():
        (out / f"{name}.jsonl").write_bytes(payload)
    manifest = {
        "experiment": "r047_primary_causal_panel",
        "version": 1,
        "seed": seed,
        "builder_sha256": _sha(Path(__file__).read_bytes()),
        "generator_family_id": GENERATOR_FAMILY_ID,
        "contract": {
            "frozen_production": counts is None,
            "flat_single_question_rows": True,
            "semantic_cluster_bootstrap_unit": True,
            "full_boolean_truth_cube": True,
            "languages": ["en", "he"],
            "option_orders": ["yes_no", "no_yes"],
            "clause_prompt_families": list(PROMPT_FAMILIES),
            "one_clause_prompt_family_per_cluster": True,
            "independently_executable_gold": True,
            "legacy_smoke_excluded_from_causal_panel": True,
        },
        "counts_per_composition": effective_counts,
        "clusters": {
            "unseen_k2": cluster_counts["unseen_k2"],
            "k3": cluster_counts["k3"],
            "seen_k2_near_transfer": cluster_counts["seen_k2"],
            "causal_primary": cluster_counts["unseen_k2"] + cluster_counts["k3"],
        },
        "compositions": {"unseen_k2": [list(value) for value in UNSEEN_K2],
                         "k3": [list(value) for value in K3],
                         "seen_k2_near_transfer": [list(value) for value in SEEN_K2]},
        "outputs": {name: {"file": f"{name}.jsonl", "rows": len(rows),
                           "sha256": _sha(payloads[name])}
                    for name, rows in outputs.items()},
        "legacy_smoke": {"seed": 47, "domain_offset": 2, "split": "test",
                         "question_repair": "composition-neutral-v1",
                         "causal_gate_eligible": False},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out, args.seed), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
