"""Generate a small, executable, leakage-separated R045 transfer corpus."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo


CASES_PER_COMPOSITION = 12
COMPOSITIONS = {
    "train": (("decimal",), ("fraction",), ("datetime",), ("ledger",),
               ("conditional",), ("policy",), ("decimal", "policy"),
               ("ledger", "conditional")),
    "cal": (("datetime", "ledger"), ("fraction", "policy")),
    "test": (("decimal", "ledger"), ("fraction", "conditional"),
              ("datetime", "policy"), ("ledger", "policy"),
              ("decimal", "datetime", "policy"),
              ("ledger", "fraction", "conditional")),
}
SHARED_NUISANCE = {"base": 1_000, "entities": ("Aster", "Birch", "Cedar", "Dune"),
                   "renderers": ("ledger_note", "case_brief")}
SPLITS = {split: dict(SHARED_NUISANCE) for split in ("train", "cal", "test")}
FRACTIONS = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
YEARS = (2041, 2042)
PRIORITIES = (20, 40, 60)
EN_TEMPLATES = {
    "ledger_note": "Record for {entity}: {evidence}\nRequest: may the service proceed?",
    "case_brief": "Dossier {entity}. Evidence: {evidence}\nDecision requested: proceed?",
    "review_card": "Review {entity} | {evidence}\nShould the service proceed?",
    "audit_slip": "Audit subject {entity}; evidence follows: {evidence}\nProceed?",
    "dispatch_form": "Dispatch file {entity}: {evidence}\nAuthorize the service?",
    "archive_note": "Archive entry {entity}. {evidence}\nIs proceeding authorized?",
}
HE_TEMPLATES = {
    "ledger_note": "רשומה עבור {entity}: {evidence}\nהאם השירות רשאי להמשיך?",
    "case_brief": "תיק {entity}. ראיות: {evidence}\nהאם להמשיך?",
    "review_card": "בדיקת {entity} | {evidence}\nהאם השירות רשאי להמשיך?",
    "audit_slip": "נושא הביקורת {entity}; ראיות: {evidence}\nלהמשיך?",
    "dispatch_form": "תיק משלוח {entity}: {evidence}\nהאם לאשר את השירות?",
    "archive_note": "רשומת ארכיון {entity}. {evidence}\nהאם ההמשך מורשה?",
}
HE_KIND = {"decimal": "סכום", "fraction": "יחס", "datetime": "מועד", "ledger": "יומן אירועים",
           "conditional": "הסתברות מותנית", "policy": "כלל מדיניות"}
HE_EFFECT = {"allow": "אישור", "deny": "דחייה"}
HE_SCOPE = {"all": "הכול", "service": "שירות", "unrelated": "לא-קשור"}
DECISION_RULE = {
    "en": "Decision rule: authorize if and only if EVERY numbered condition below is satisfied.",
    "he": "כלל החלטה: יש לאשר אם ורק אם כל התנאים הממוספרים להלן מתקיימים.",
}
BENCHMARK_RE = re.compile(r"jev[\W_]*bench|workflow[\W_]*evals", re.IGNORECASE)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _check_sources(paths=(), contents=()):
    for path in paths:
        if BENCHMARK_RE.search(str(path)):
            raise ValueError(f"prohibited benchmark source path: {path}")
    for content in contents:
        if BENCHMARK_RE.search(str(content)):
            raise ValueError("prohibited benchmark source content")


def _evaluate(clause):
    kind = clause["kind"]
    if kind == "decimal":
        net = sum((Decimal(value) for value in clause["credits"]), Decimal("0")) - Decimal(clause["fee"])
        return net >= Decimal(clause["threshold"])
    if kind == "fraction":
        return Fraction(clause["numerator"], clause["denominator"]) >= Fraction(
            clause["threshold"][0], clause["threshold"][1])
    if kind == "datetime":
        local = datetime.fromisoformat(clause["local_time"]).replace(
            tzinfo=ZoneInfo(clause["timezone"]))
        deadline = datetime.fromisoformat(clause["deadline"]).replace(tzinfo=timezone.utc)
        return local.astimezone(timezone.utc) <= deadline
    if kind == "ledger":
        balance = sum((Decimal(event["delta"]) for event in clause["events"]), Decimal("0"))
        return balance >= Decimal(clause["threshold"])
    if kind == "conditional":
        target = next(group for group in clause["groups"] if group["name"] == clause["condition_group"])
        probability = Fraction(target["success"], target["success"] + target["failure"])
        return probability >= Fraction(clause["threshold"][0], clause["threshold"][1])
    if kind == "policy":
        applicable = [rule for rule in clause["rules"] if rule["active"] and
                      rule["scope"] in {"all", clause["request_scope"]}]
        winner = max(applicable, key=lambda rule: (rule["priority"], rule["effect"] == "deny"))
        return winner["effect"] == "allow"
    raise ValueError(f"unknown clause kind: {kind}")


def execute(program, clauses):
    """Execute an all-of composition; ties in policy precedence resolve to deny."""
    by_kind = {clause["kind"]: clause for clause in clauses}
    if set(by_kind) != set(program["composition"]):
        raise ValueError("program and evidence composition differ")
    return all(_evaluate(by_kind[kind]) for kind in program["composition"])


def _clause(kind, split, variant, passed, temporal_index=0):
    config = SPLITS[split]
    base = config["base"] + (variant % 24) * 10
    fraction = FRACTIONS[variant % len(FRACTIONS)]
    if kind == "decimal":
        threshold = Decimal(base) + Decimal(5 + variant % 91) / 100
        second = Decimal(3 + variant % 4) + Decimal(7 + variant % 83) / 100
        fee = Decimal(5 + (variant // 11) % 5) + Decimal(3 + (variant // 7) % 89) / 100
        neutral_first = threshold + fee - second
        first = neutral_first + (Decimal("0") if passed and variant % 2 == 0 else
                                 Decimal("0.07") if passed else Decimal("-0.07"))
        return {"kind": kind, "credits": [str(first), str(second)], "fee": str(fee),
                "threshold": str(threshold), "unit": "credits"}
    if kind == "fraction":
        denominator = (40, 53, 72, 83)[variant % 4]
        threshold_num = ((fraction.numerator * denominator + fraction.denominator - 1)
                         // fraction.denominator)
        return {"kind": kind, "numerator": threshold_num if passed else threshold_num - 1,
                "denominator": denominator, "threshold": [fraction.numerator, fraction.denominator]}
    if kind == "datetime":
        group, mode = divmod(temporal_index, 2)
        local_day = group % 20 + 1
        year = YEARS[group % len(YEARS)]
        timezone_name = ("Pacific/Auckland", "America/New_York",
                         "Asia/Jerusalem", "Europe/Paris")[group % 4]
        instant = datetime(year, 4, local_day, group * 3 % 24, tzinfo=timezone.utc)
        local = instant.astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
        if mode == 0:
            offset = 30 if passed else -30
        else:
            offset = 23 * 60 + 30 if passed else -(23 * 60 + 30)
        deadline = instant + timedelta(minutes=offset)
        return {"kind": kind, "local_time": local.isoformat(),
                "timezone": timezone_name,
                "deadline": deadline.replace(tzinfo=None).isoformat()}
    if kind == "ledger":
        threshold = Decimal(base + 5)
        desired = threshold + (1 if passed else -1)
        opening = threshold + 17 + variant % 5
        correction = Decimal(-11 - variant % 3)
        adjustment = desired - opening - correction
        return {"kind": kind, "events": [
                    {"event_id": "opening", "delta": str(opening)},
                    {"event_id": "correction", "delta": str(correction)},
                    {"event_id": "adjustment", "delta": str(adjustment)}],
                "threshold": str(threshold), "currency": "units"}
    if kind == "conditional":
        condition_count = (41, 59, 73, 89)[variant % 4]
        threshold_num = ((fraction.numerator * condition_count + fraction.denominator - 1)
                         // fraction.denominator)
        success = threshold_num if passed else threshold_num - 1
        return {"kind": kind, "condition_group": "eligible", "groups": [
                    {"name": "eligible", "success": success,
                     "failure": condition_count - success},
                    {"name": "other", "success": 7 + variant % 5,
                     "failure": 3 + variant % 7}],
                "threshold": [fraction.numerator, fraction.denominator]}
    if kind == "policy":
        priority = PRIORITIES[variant % len(PRIORITIES)]
        allow_name = ("allow-alpha", "allow-beta")[variant % 2]
        deny_name = ("deny-alpha", "deny-beta")[(variant // 2) % 2]
        deny_priority = priority + 5 + variant % 7
        allow_priority = deny_priority + 1 if passed else deny_priority - (1 if variant % 2 == 0 else 0)
        rules = [
            {"name": "default", "priority": 0, "effect": "deny", "active": True,
             "scope": "all"},
            {"name": allow_name, "priority": allow_priority, "effect": "allow", "active": True,
             "scope": "service"},
            {"name": deny_name, "priority": deny_priority, "effect": "deny", "active": True,
             "scope": "service"},
            {"name": "allow-spare", "priority": priority + 3, "effect": "allow", "active": True,
             "scope": "service"},
            {"name": "deny-spare", "priority": priority + 4, "effect": "deny", "active": True,
             "scope": "service"},
            {"name": "scope-decoy", "priority": priority + 20, "effect": "deny",
             "active": True, "scope": "unrelated"},
        ]
        shift = variant % len(rules)
        return {"kind": kind, "rules": rules[shift:] + rules[:shift],
                "request_scope": "service", "precedence": "scope_then_highest_priority_then_deny"}
    raise ValueError(f"unknown clause kind: {kind}")


def _describe(clause, lang):
    kind = clause["kind"]
    if lang == "en":
        if kind == "decimal":
            detail = (f"credits {' + '.join(clause['credits'])}, fee {clause['fee']} {clause['unit']}; "
                      f"requirement: credits minus fee >= "
                      f"{clause['threshold']} {clause['unit']}")
        elif kind == "fraction":
            detail = (f"observed ratio {clause['numerator']}/{clause['denominator']}; requirement: "
                      f"ratio >= {clause['threshold'][0]}/{clause['threshold'][1]}")
        elif kind == "datetime":
            detail = (f"local event time {clause['local_time']} in {clause['timezone']}; requirement: "
                      f"converted UTC time <= {clause['deadline']} UTC")
        elif kind == "ledger":
            entries = ", ".join(f"{event['event_id']}:{event['delta']}" for event in clause["events"])
            detail = (f"ordered deltas [{entries}]; requirement: sum of all deltas >= "
                      f"{clause['threshold']} {clause['currency']}")
        elif kind == "conditional":
            groups = ", ".join(f"{group['name']}[success={group['success']},failure={group['failure']}]"
                               for group in clause["groups"])
            detail = (f"groups {groups}; condition on {clause['condition_group']}; requirement: "
                      f"success/(success+failure) >= {clause['threshold'][0]}/{clause['threshold'][1]}")
        else:
            rules = ", ".join(f"{rule['name']}={rule['effect']}@{rule['priority']}"
                               f" {'active' if rule['active'] else 'inactive'} scope={rule['scope']}"
                               for rule in clause["rules"])
            detail = (f"request scope {clause['request_scope']}; matching active rules only; highest numeric "
                      f"priority wins; equal-priority deny wins; {rules}")
        return f"{kind}: {detail}"

    if kind == "decimal":
        detail = (f"זיכויים {' + '.join(clause['credits'])}, עמלה {clause['fee']} {clause['unit']}; "
                  f"דרישה: זיכויים פחות עמלה >= "
                  f"{clause['threshold']} {clause['unit']}")
    elif kind == "fraction":
        detail = (f"יחס נצפה {clause['numerator']}/{clause['denominator']}; דרישה: יחס >= "
                  f"{clause['threshold'][0]}/{clause['threshold'][1]}")
    elif kind == "datetime":
        detail = (f"זמן אירוע מקומי {clause['local_time']} באזור {clause['timezone']}; דרישה: "
                  f"הזמן לאחר המרה ל-UTC <= {clause['deadline']} UTC")
    elif kind == "ledger":
        entries = ", ".join(f"{event['event_id']}:{event['delta']}" for event in clause["events"])
        detail = (f"שינויים לפי הסדר [{entries}]; דרישה: סכום כל השינויים >= "
                  f"{clause['threshold']} {clause['currency']}")
    elif kind == "conditional":
        groups = ", ".join(f"{group['name']}[הצלחה={group['success']},כישלון={group['failure']}]"
                           for group in clause["groups"])
        detail = (f"קבוצות {groups}; יש להתנות על {clause['condition_group']}; דרישה: "
                  f"הצלחה/(הצלחה+כישלון) >= {clause['threshold'][0]}/{clause['threshold'][1]}")
    else:
        rules = ", ".join(f"{rule['name']}={HE_EFFECT[rule['effect']]}@{rule['priority']}"
                           f" {'פעיל' if rule['active'] else 'לא-פעיל'} תחום={HE_SCOPE[rule['scope']]}"
                           for rule in clause["rules"])
        detail = (f"תחום הבקשה {HE_SCOPE[clause['request_scope']]}; רק כללים פעילים בתחום המתאים; העדיפות המספרית "
                  f"הגבוהה מנצחת; בשוויון דחייה מנצחת; {rules}")
    return f"{HE_KIND[kind]}: {detail}"


def _pair(split, composition, index, seed, case_index=None):
    case_index = index % CASES_PER_COMPOSITION if case_index is None else case_index
    config = SPLITS[split]
    source_id = f"r045-{seed}-{split}-{index:05d}"
    composition_index = index // CASES_PER_COMPOSITION
    scenario_index = case_index // len(composition)
    scenario_id = f"r045-{seed}-{split}-c{composition_index:02d}-s{scenario_index:02d}"
    entity = config["entities"][(scenario_index + composition_index) % len(config["entities"])]
    renderer = config["renderers"][(scenario_index + seed) % len(config["renderers"])]
    decisive = case_index % len(composition)
    variant = int(_digest(f"{seed}:{split}:{composition}:{scenario_index}".encode())[:8], 16)
    program = {"language": "r045-composition-v1", "composition": list(composition), "operator": "all"}
    views_by_lang = {}
    for lang in ("en", "he"):
        views = []
        for passed in (True, False):
            clauses = [_clause(kind, split, variant + position * 17,
                               passed if position == decisive else True, scenario_index)
                       for position, kind in enumerate(composition)]
            shift = variant % len(clauses)
            clauses = clauses[shift:] + clauses[:shift]
            gold = execute(program, clauses)
            criteria = {"yes": "Yes" if lang == "en" else "כן",
                        "no": "No" if lang == "en" else "לא"}
            if (index + (lang == "he")) % 2:
                criteria = dict(reversed(list(criteria.items())))
            condition = "Condition" if lang == "en" else "תנאי"
            evidence = "\n".join(f"{condition} {position}: {_describe(clause, lang)}"
                                 for position, clause in enumerate(clauses, 1))
            evidence = f"{DECISION_RULE[lang]}\n{evidence}"
            template = EN_TEMPLATES[renderer] if lang == "en" else HE_TEMPLATES[renderer]
            state = template.format(entity=entity, evidence=evidence)
            opaque = _digest(f"{source_id}:{lang}:{passed}".encode())[:10]
            view_id = f"{source_id}-{lang}-{opaque}"
            views.append({"id": view_id, "state": state,
                          "questions": {"decision": {"type": "choice",
                              "instructions": ("Does the policy authorize proceeding?" if lang == "en"
                                               else "האם המדיניות מאשרת להמשיך?"),
                              "criteria": criteria}},
                          "targets": {"decision": {"label": "yes" if gold else "no"}},
                          "meta": {"source": "r045_generated", "source_id": source_id,
                                   "split": split, "composition": list(composition), "program": program,
                                   "clauses": clauses, "gold": gold, "entity_vocab": entity,
                                   "renderer": renderer, "language": lang, "twin_id": source_id,
                                   "scenario_id": scenario_id,
                                   "decisive_kind": composition[decisive], "seed": seed,
                                   "rendered_contract": "explicit-all-of-v2"}})
        if {view["targets"]["decision"]["label"] for view in views} != {"yes", "no"}:
            raise AssertionError(f"counterfactual failed to flip: {source_id}/{lang}")
        if int(_digest(f"{source_id}:{lang}:order".encode()), 16) % 2:
            views.reverse()
        views_by_lang[lang] = views
    result = []
    for lang, views in views_by_lang.items():
        if any(execute(view["meta"]["program"], view["meta"]["clauses"])
               != view["meta"]["gold"] for view in views):
            raise AssertionError(f"executable gold mismatch: {source_id}/{lang}")
        result.append({"id": f"{source_id}-{lang}", "pair_mode": "counterfactual",
                       "source_ids": [source_id, source_id], "paired_views": views,
                       "meta": {"source": "r045_generated", "source_id": source_id,
                                "split": split, "language": lang, "twin_id": source_id,
                                "composition": list(composition), "renderer": renderer,
                                "entity_vocab": entity, "scenario_id": scenario_id,
                                "decisive_kind": composition[decisive]}})
    return result


def build(out, seed=45, source_paths=(), source_contents=()):
    _check_sources(source_paths, source_contents)
    generated = {split: [] for split in SPLITS}
    index = 0
    for split, compositions in COMPOSITIONS.items():
        for composition_index, composition in enumerate(compositions):
            for case_index in range(CASES_PER_COMPOSITION):
                generated[split].extend(_pair(split, composition, index, seed, case_index))
                index += 1
    # Every generated group stays whole, and its language twins share the same split.
    by_name, output_hashes = {}, {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in generated.items():
        payload = _jsonl(rows)
        (out / f"{split}.jsonl").write_bytes(payload)
        by_name[split] = {"rows": len(rows), "pairs": len(rows),
                          "logical_sources": len({row["meta"]["source_id"] for row in rows}),
                          "scenarios": len({row["meta"]["scenario_id"] for row in rows}),
                          "sha256": _digest(payload)}
        output_hashes[split] = _digest(payload)
    manifest = {"experiment": "r045_compositional_transfer", "seed": seed,
                "generator": "stdlib-only", "builder_sha256": _digest(Path(__file__).read_bytes()),
                "source_inputs": [], "rejected_source_count": 0,
                "splits": by_name, "compositions": {split: [list(c) for c in COMPOSITIONS[split]]
                                                        for split in SPLITS},
                "leakage_contract": {"disjoint_compositions": True, "disjoint_instances": True,
                                     "matched_nuisance_pools": True,
                                     "counterfactual_pairs_atomic": True, "language_twins_atomic": True},
                "outputs": output_hashes}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    (out / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--source-path", action="append", default=[],
                        help="Prohibited benchmark sources are rejected; the builder reads no sources.")
    parser.add_argument("--source-content", action="append", default=[],
                        help="Prohibited benchmark text/labels are rejected; the builder reads no sources.")
    args = parser.parse_args(argv)
    manifest = build(args.out, args.seed, args.source_path, args.source_content)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
