"""Generate a small, executable, leakage-separated R045 transfer corpus."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo


CASES_PER_COMPOSITION = 8
COMPOSITIONS = {
    "train": (("decimal",), ("fraction",), ("datetime",), ("ledger",),
               ("conditional",), ("policy",), ("decimal", "policy"),
               ("ledger", "conditional")),
    "cal": (("datetime", "ledger"), ("fraction", "policy")),
    "test": (("decimal", "ledger"), ("fraction", "conditional"),
              ("datetime", "policy"), ("ledger", "policy")),
}
SPLITS = {
    "train": {"base": 100, "fraction": Fraction(1, 4), "year": 2031, "priority": 20,
              "entities": ("Aster", "Birch", "Cedar", "Dune"), "renderers": ("ledger_note", "case_brief")},
    "cal": {"base": 1_000, "fraction": Fraction(1, 2), "year": 2041, "priority": 40,
           "entities": ("Fjord", "Grove", "Harbor", "Indigo"), "renderers": ("review_card", "audit_slip")},
    "test": {"base": 10_000, "fraction": Fraction(3, 4), "year": 2051, "priority": 60,
            "entities": ("Juniper", "Kestrel", "Lumen", "Mesa"), "renderers": ("dispatch_form", "archive_note")},
}
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
BENCHMARK_RE = re.compile(r"jev[\W_]*bench|workflow[\W_]*evals", re.IGNORECASE)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()


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
        return Decimal(clause["amount"]) >= Decimal(clause["threshold"])
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
        probability = Fraction(clause["joint_count"], clause["condition_count"])
        return probability >= Fraction(clause["threshold"][0], clause["threshold"][1])
    if kind == "policy":
        applicable = [rule for rule in clause["rules"] if rule["active"]]
        winner = max(applicable, key=lambda rule: (rule["priority"], rule["effect"] == "deny"))
        return winner["effect"] == "allow"
    raise ValueError(f"unknown clause kind: {kind}")


def execute(program, clauses):
    """Execute an all-of composition; ties in policy precedence resolve to deny."""
    by_kind = {clause["kind"]: clause for clause in clauses}
    if set(by_kind) != set(program["composition"]):
        raise ValueError("program and evidence composition differ")
    return all(_evaluate(by_kind[kind]) for kind in program["composition"])


def _clause(kind, split, index, passed):
    config = SPLITS[split]
    base = config["base"] + index * 10
    fraction = config["fraction"]
    if kind == "decimal":
        threshold = Decimal(base + 5)
        return {"kind": kind, "amount": str(threshold + (1 if passed else -1)),
                "threshold": str(threshold), "unit": f"credit_{split}"}
    if kind == "fraction":
        denominator = 100
        threshold_num = fraction.numerator * denominator // fraction.denominator
        return {"kind": kind, "numerator": threshold_num + (10 if passed else -10),
                "denominator": denominator, "threshold": [fraction.numerator, fraction.denominator]}
    if kind == "datetime":
        local_day = index % 20 + 1
        local_hour = 12 if passed else 20
        timezone_name = "Asia/Jerusalem" if index % 2 else "Europe/Paris"
        return {"kind": kind, "local_time": f"{config['year']}-04-{local_day:02d}T{local_hour:02d}:00:00",
                "timezone": timezone_name,
                "deadline": f"{config['year']}-04-{local_day:02d}T16:00:00"}
    if kind == "ledger":
        threshold = Decimal(base + 5)
        return {"kind": kind, "events": [{"event_id": f"evt-{split}-{index}-opening", "delta": str(threshold + (1 if passed else -1))},
                                           {"event_id": f"evt-{split}-{index}-adjustment", "delta": "0"}],
                "threshold": str(threshold), "currency": f"unit_{split}"}
    if kind == "conditional":
        threshold_num = fraction.numerator * 100 // fraction.denominator
        return {"kind": kind, "joint_count": threshold_num + (10 if passed else -10),
                "condition_count": 100, "threshold": [fraction.numerator, fraction.denominator]}
    if kind == "policy":
        return {"kind": kind, "rules": [
            {"name": f"default-{split}", "priority": 0, "effect": "deny", "active": True},
            {"name": f"baseline-{split}", "priority": config["priority"], "effect": "allow", "active": True},
            {"name": f"exception-{split}", "priority": config["priority"] + 10, "effect": "deny", "active": not passed},
        ], "precedence": "highest_priority_then_deny"}
    raise ValueError(f"unknown clause kind: {kind}")


def _describe(clause, lang):
    kind = clause["kind"]
    if kind == "decimal":
        detail = f"{clause['amount']} {clause['unit']}; threshold {clause['threshold']}"
    elif kind == "fraction":
        detail = f"{clause['numerator']}/{clause['denominator']}; threshold {clause['threshold'][0]}/{clause['threshold'][1]}"
    elif kind == "datetime":
        detail = f"{clause['local_time']} {clause['timezone']}; deadline {clause['deadline']} UTC"
    elif kind == "ledger":
        entries = ", ".join(event["delta"] for event in clause["events"])
        detail = f"events [{entries}]; balance threshold {clause['threshold']} {clause['currency']}"
    elif kind == "conditional":
        detail = (f"joint/condition {clause['joint_count']}/{clause['condition_count']}; "
                  f"threshold {clause['threshold'][0]}/{clause['threshold'][1]}")
    else:
        rules = ", ".join(f"{rule['name']}={rule['effect']}@{rule['priority']}"
                           f"{' active' if rule['active'] else ' inactive'}" for rule in clause["rules"])
        detail = f"highest priority wins; ties deny; {rules}"
    if lang == "he":
        return f"{HE_KIND[kind]}: {detail}"
    return f"{kind}: {detail}"


def _pair(split, composition, index, seed):
    config = SPLITS[split]
    source_id = f"r045-{seed}-{split}-{index:05d}"
    entity = config["entities"][index % len(config["entities"])]
    renderer = config["renderers"][index % len(config["renderers"])]
    decisive = index % len(composition)
    program = {"language": "r045-composition-v1", "composition": list(composition), "operator": "all"}
    views_by_lang = {}
    for lang in ("en", "he"):
        views = []
        for passed in (True, False):
            clauses = [_clause(kind, split, index, passed if position == decisive else True)
                       for position, kind in enumerate(composition)]
            gold = execute(program, clauses)
            criteria = {"yes": "Yes" if lang == "en" else "כן",
                        "no": "No" if lang == "en" else "לא"}
            if (index + (lang == "he")) % 2:
                criteria = dict(reversed(list(criteria.items())))
            state = EN_TEMPLATES[renderer].format(entity=entity, evidence="; ".join(
                _describe(clause, lang) for clause in clauses)) if lang == "en" else HE_TEMPLATES[renderer].format(
                    entity=entity, evidence="; ".join(_describe(clause, lang) for clause in clauses))
            view_id = f"{source_id}-{lang}-{'pass' if passed else 'fail'}"
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
                                   "decisive_kind": composition[decisive], "seed": seed}})
        if {view["targets"]["decision"]["label"] for view in views} != {"yes", "no"}:
            raise AssertionError(f"counterfactual failed to flip: {source_id}/{lang}")
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
                                "entity_vocab": entity}})
    return result


def build(out, seed=45, source_paths=(), source_contents=()):
    _check_sources(source_paths, source_contents)
    generated = {split: [] for split in SPLITS}
    index = 0
    for split, compositions in COMPOSITIONS.items():
        for composition_index, composition in enumerate(compositions):
            for case_index in range(CASES_PER_COMPOSITION):
                generated[split].extend(_pair(split, composition, index, seed))
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
                          "sha256": _digest(payload)}
        output_hashes[split] = _digest(payload)
    manifest = {"experiment": "r045_compositional_transfer", "seed": seed,
                "generator": "stdlib-only", "builder_sha256": _digest(Path(__file__).read_bytes()),
                "source_inputs": [], "rejected_source_count": 0,
                "splits": by_name, "compositions": {split: [list(c) for c in COMPOSITIONS[split]]
                                                        for split in SPLITS},
                "leakage_contract": {"disjoint_compositions": True, "disjoint_value_ranges": True,
                                     "disjoint_thresholds": True, "disjoint_entity_vocabularies": True,
                                     "disjoint_renderer_templates": True,
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
