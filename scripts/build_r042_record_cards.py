"""Build R042's matched record-card versus prose representation orbits.

Each source has one queried fact and three distractors.  The two training arms
share every logical fact, question, label, and option position; only the state
renderer differs.  Held-out sources use a disjoint predicate vocabulary.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ASSERTIONS = ("bare", "explicit", "interrogative")
OPTION_SURFACES = (("generic", "No", "Yes"), ("boolean", "False", "True"))
TRAIN_KINDS = ("archive", "cask", "drum", "forge", "harp", "lantern", "mosaic", "niche", "orb", "plinth", "quiver")
EVAL_KINDS = ("binnacle", "cairn", "diorama", "fountain", "grotto", "horizon", "isthmus", "keystone")
TRAIN_PROPERTIES = (("alignment", "aligned", "misaligned"), ("seal", "sealed", "unsealed"),
                    ("charge", "charged", "uncharged"), ("stability", "stable", "unstable"))
EVAL_PROPERTIES = (("calibration", "calibrated", "uncalibrated"), ("filter", "filtered", "unfiltered"),
                   ("pairing", "paired", "unpaired"), ("indexing", "indexed", "unindexed"),
                   ("clearance", "cleared", "uncleared"), ("registration", "registered", "unregistered"),
                   ("balance", "balanced", "unbalanced"), ("routing", "routed", "unrouted"))
FORBIDDEN = ("approval", "manager", "case", "request", "risk", "policy", "evidence", "coverage",
             "amount", "deadline", "reviewer", "document", "signature", "signed", "unsigned")


def _spec(split, source_id):
    kinds, properties = (TRAIN_KINDS, TRAIN_PROPERTIES) if split == "train" else (EVAL_KINDS, EVAL_PROPERTIES)
    kind = kinds[source_id // len(properties)]
    property_ = properties[source_id % len(properties)]
    return kind, property_, properties


def _facts(split, source_id, world, factor_index):
    kind, queried_property, properties = _spec(split, source_id)
    primary = f"{kind}-{source_id + 1:03d}"
    distractors = [
        (f"{kind}-aux{index}-{source_id + 1:03d}", properties[(source_id + index) % len(properties)],
         bool((factor_index >> (index - 1)) & 1))
        for index in range(1, 4)
    ]
    position = factor_index % 4
    entries = [(primary, queried_property, world, True), *[(subject, prop, value, False)
                                                            for subject, prop, value in distractors]]
    queried = entries.pop(0)
    entries.insert(position, queried)
    return tuple({"subject": subject, "field": prop[0], "value": int(value), "queried": queried}
                 for subject, prop, value, queried in entries), position


def _render(facts, renderer):
    values = {field: (yes, no) for field, yes, no in (*TRAIN_PROPERTIES, *EVAL_PROPERTIES)}
    clauses = [f"{fact['subject']} — {fact['field']}: "
               f"{values[fact['field']][0 if fact['value'] else 1]}."
               for fact in facts]
    if renderer == "card":
        return "Facts card:\n" + "\n".join(f"{index + 1}. {clause}" for index, clause in enumerate(clauses))
    if renderer == "prose":
        return " ".join(clauses)
    raise ValueError(f"unknown renderer: {renderer}")


def _claim(facts, polarity):
    queried = next(fact for fact in facts if fact["queried"])
    values = {field: (yes, no) for field, yes, no in (*TRAIN_PROPERTIES, *EVAL_PROPERTIES)}
    return f"{queried['subject']} is {values[queried['field']][0 if polarity == 'positive' else 1]}."


def _instruction(claim, wording, polarity):
    if wording == "bare":
        return claim
    if wording == "explicit":
        return f"Is this statement true? {claim}"
    if wording == "interrogative":
        return f"Is it true that {claim}"
    raise ValueError(f"unknown wording: {wording}")


def _row(source_id, split, renderer, world, polarity, wording_index, option_index, yes_first):
    factor_index = (((int(world) * 2 + (polarity == "negative")) * 3 + wording_index) * 2 + option_index) * 2 + yes_first
    facts, queried_position = _facts(split, source_id, world, factor_index)
    wording = ASSERTIONS[wording_index]
    criteria_style, no, yes = OPTION_SURFACES[option_index]
    criteria = {"yes": yes, "no": no} if yes_first else {"no": no, "yes": yes}
    label = "yes" if world == (polarity == "positive") else "no"
    kind, property_, _ = _spec(split, source_id)
    return {
        "id": (f"r042-{renderer}-{split}-{source_id:03d}-w{int(world)}-p{polarity[0]}-"
               f"a{wording_index}-c{option_index}-o{'yn' if yes_first else 'ny'}"),
        "state": _render(facts, renderer),
        "questions": {"decision": {"type": "noul", "instructions": _instruction(_claim(facts, polarity), wording, polarity),
                                      "criteria": criteria}},
        "targets": {"decision": {"label": label}},
        "meta": {"source": "r042_programmatic", "source_id": f"{split}-{source_id:03d}",
                 "split": split, "predicate": f"{kind}:{property_[0]}", "lexicon": split,
                 "world": int(world), "polarity": polarity, "wording": wording,
                 "criteria_style": criteria_style, "yes_first": bool(yes_first), "fixed_option_order": True,
                 "renderer": renderer, "queried_position": queried_position, "facts": facts},
    }


def _rows_for_source(source_id, split, renderer):
    return [_row(source_id, split, renderer, world, polarity, wording_index, option_index, yes_first)
            for world in (False, True) for polarity in ("positive", "negative")
            for wording_index in range(len(ASSERTIONS)) for option_index in range(len(OPTION_SURFACES))
            for yes_first in (False, True)]


def _by_source(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["meta"]["source_id"], []).append(row)
    return grouped


def rendered_signature(row):
    question = row["questions"]["decision"]
    return row["state"], question["instructions"], tuple(question["criteria"].items()), row["targets"]["decision"]["label"]


def _logical_signature(row):
    meta = {key: value for key, value in row["meta"].items() if key != "renderer"}
    return meta, row["questions"], row["targets"]


def build(train_sources=44, eval_sources=64):
    if train_sources != 44 or eval_sources != 64:
        raise ValueError("R042 is preregistered for exactly 44 train and 64 held-out sources")
    card = [row for source_id in range(train_sources) for row in _rows_for_source(source_id, "train", "card")]
    prose = [row for source_id in range(train_sources) for row in _rows_for_source(source_id, "train", "prose")]
    card_eval = [row for source_id in range(eval_sources) for row in _rows_for_source(source_id, "eval", "card")]
    prose_eval = [row for source_id in range(eval_sources) for row in _rows_for_source(source_id, "eval", "prose")]
    validate(card, prose, card_eval, prose_eval)
    return card, prose, card_eval, prose_eval


def _validate_orbit(rows, expected_sources):
    if len(rows) != expected_sources * 48:
        raise ValueError("invalid R042 source orbit size")
    for source_rows in _by_source(rows).values():
        if len(source_rows) != 48:
            raise ValueError("incomplete source orbit")
        labels = Counter(row["targets"]["decision"]["label"] for row in source_rows)
        positions = Counter(row["meta"]["queried_position"] for row in source_rows)
        if labels != {"yes": 24, "no": 24} or positions != {0: 12, 1: 12, 2: 12, 3: 12}:
            raise ValueError("labels or queried positions are unbalanced")
        if any(Counter(row["targets"]["decision"]["label"] for row in source_rows
                       if row["meta"]["queried_position"] == position) != {"yes": 6, "no": 6}
               for position in range(4)):
            raise ValueError("queried position predicts label")
        if Counter((row["meta"]["yes_first"], row["targets"]["decision"]["label"])
                   for row in source_rows) != {(False, "yes"): 12, (False, "no"): 12,
                                                (True, "yes"): 12, (True, "no"): 12}:
            raise ValueError("option position predicts label")
        if any(sum(fact["queried"] for fact in row["meta"]["facts"]) != 1
               or len(row["meta"]["facts"]) != 4 for row in source_rows):
            raise ValueError("each state needs one queried fact and three distractors")
        distractors = {}
        for row in source_rows:
            label = row["targets"]["decision"]["label"]
            for fact in row["meta"]["facts"]:
                if not fact["queried"]:
                    distractors.setdefault((fact["field"], fact["value"]), []).append(label)
        if any(Counter(labels) != {"yes": 12, "no": 12} for labels in distractors.values()):
            raise ValueError("distractor truth predicts label")


def validate(card, prose, card_eval, prose_eval):
    _validate_orbit(card, 44)
    _validate_orbit(prose, 44)
    _validate_orbit(card_eval, 64)
    _validate_orbit(prose_eval, 64)
    if [ _logical_signature(row) for row in card ] != [ _logical_signature(row) for row in prose ]:
        raise ValueError("card and prose arms differ beyond state renderer")
    if [ _logical_signature(row) for row in card_eval ] != [ _logical_signature(row) for row in prose_eval ]:
        raise ValueError("evaluation renderers differ beyond state renderer")
    train_sources = {row["meta"]["source_id"] for row in card}
    eval_sources = {row["meta"]["source_id"] for row in card_eval}
    train_predicates = {row["meta"]["predicate"] for row in card}
    eval_predicates = {row["meta"]["predicate"] for row in card_eval}
    if train_sources & eval_sources or train_predicates & eval_predicates:
        raise ValueError("train/evaluation source or predicate overlap")
    if set(map(rendered_signature, card + prose)) & set(map(rendered_signature, card_eval + prose_eval)):
        raise ValueError("rendered train/evaluation overlap")
    rendered = "\n".join(row["state"] + "\n" + row["questions"]["decision"]["instructions"]
                           for row in card + prose + card_eval + prose_eval).lower()
    if any(word in rendered for word in FORBIDDEN):
        raise ValueError("R042 uses action-panel or R040 vocabulary")


def _digest(rows):
    return hashlib.sha256("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()).hexdigest()


def _write(path, rows):
    with Path(path).open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_replay(replay, card, prose):
    signatures = {rendered_signature(row) for row in card + prose}
    for row in replay:
        meta = row.get("meta") or {}
        question = row.get("questions", {}).get("decision", {})
        comparable = (isinstance(question.get("criteria"), dict)
                      and all(isinstance(value, str) for value in question["criteria"].values())
                      and isinstance(row.get("state"), str)
                      and isinstance(question.get("instructions"), str)
                      and isinstance(row.get("targets", {}).get("decision", {}).get("label"), str))
        if (meta.get("panel") == "action_v1" or str(row.get("id", "")).startswith(("r040-", "r042-"))
                or (comparable and rendered_signature(row) in signatures)):
            raise ValueError("R042 replay must exclude action-panel, R040/R042, and duplicate rendered rows")
    return [*replay, *card], [*replay, *prose]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".context/r042-record-cards")
    parser.add_argument("--replay")
    parser.add_argument("--replay-count", type=int, default=0)
    args = parser.parse_args()
    card, prose, card_eval, prose_eval = build()
    replay = []
    if args.replay:
        replay = [json.loads(line) for line in Path(args.replay).read_text().splitlines() if line.strip()]
        replay = replay[:args.replay_count] if args.replay_count else replay
        if not replay:
            raise ValueError("replay input produced no rows")
        card, prose = add_replay(replay, card, prose)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    named = (("card", card), ("prose", prose), ("eval-card", card_eval), ("eval-prose", prose_eval))
    for name, rows in named:
        _write(out / f"{name}.jsonl", rows)
    manifest = {"train_sources": 44, "eval_sources": 64, "replay_rows": len(replay),
                "orbit_rows_per_arm": 44 * 48,
                "rows": {name: len(rows) for name, rows in named},
                "sha256": {name: _digest(rows) for name, rows in named}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
