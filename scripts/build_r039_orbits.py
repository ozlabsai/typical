"""Build R039's source-disjoint, format-balanced semantic interaction arms."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


TOPICS = (
    ("risk score", "points"), ("requested amount", "credits"),
    ("response time", "minutes"), ("inventory count", "units"),
    ("temperature reading", "degrees"), ("error rate", "basis points"),
    ("review age", "days"), ("shipment weight", "kilograms"),
)
TRAIN_STATE = (
    "Recorded {topic}: {value} {unit}. The permitted maximum is {limit} {unit}.",
    "The case file lists {topic} = {value} {unit}; its ceiling is {limit} {unit}.",
    "Audit note: {value} {unit} for {topic}, against a cap of {limit} {unit}.",
    "For this request, {topic} is {value} {unit}; policy allows at most {limit} {unit}.",
)
EVAL_STATE = (
    "The verified {topic} measures {value} {unit}; the policy threshold is {limit} {unit}.",
    "Review record: {topic} has value {value} {unit}, while the limit is {limit} {unit}.",
    "A maximum of {limit} {unit} applies to the reported {topic} of {value} {unit}.",
    "The report gives {topic} as {value} {unit}. Its authorized bound is {limit} {unit}.",
)
TRAIN_CLAIM = (
    "Is the recorded {topic} within the policy limit?",
    "Does the stated {topic} satisfy the permitted maximum?",
    "Is the {topic} no greater than the allowed ceiling?",
    "Does this case remain below the specified {topic} cap?",
)
EVAL_CLAIM = (
    "Does the verified {topic} meet its threshold?",
    "Is the reported {topic} at or below the authorized bound?",
    "Does the listed {topic} comply with the maximum?",
    "Is this {topic} inside the policy ceiling?",
)


def _facts(facts, source_id, split, world):
    offset = source_id + (31 if split == "eval" else 0)
    topic, unit = TOPICS[offset % len(TOPICS)]
    value, limit = facts[(split, source_id, world)]
    return topic, unit, value, limit


def _row(facts, source_id, split, world, state_variant, claim_variant, yes_first, copy_index, arm):
    topic, unit, value, limit = _facts(facts, source_id, split, world)
    state_templates = EVAL_STATE if split == "eval" else TRAIN_STATE
    claim_templates = EVAL_CLAIM if split == "eval" else TRAIN_CLAIM
    criteria = {"yes": "Yes", "no": "No"}
    if not yes_first:
        criteria = dict(reversed(list(criteria.items())))
    return {
        "id": (f"r039-{arm}-{split}-{source_id:03d}-w{world}-s{state_variant}-"
               f"c{claim_variant}-o{'yn' if yes_first else 'ny'}-r{copy_index}"),
        "state": state_templates[state_variant].format(topic=topic, unit=unit, value=value, limit=limit),
        "questions": {"decision": {"type": "noul",
                                      "instructions": claim_templates[claim_variant].format(topic=topic),
                                      "criteria": criteria}},
        "targets": {"decision": {"label": "yes" if world == 0 else "no"}},
        "meta": {"source": "r039_programmatic", "source_id": f"{split}-{source_id:03d}",
                 "world": world, "polarity": "positive", "wording": f"claim-{claim_variant}",
                 "criteria_style": f"state-{state_variant}", "yes_first": yes_first,
                 "fixed_option_order": True, "numeric_value": value, "numeric_limit": limit,
                 "arm": arm, "copy_index": copy_index},
    }


def _selected_pairs():
    """A balanced 4x4 half-orbit: every state/claim surface appears twice."""
    return {(state, claim) for state in range(4) for claim in (state, (state + 1) % 4)}


def _rows_for_source(facts, source_id, split, arm):
    selected = _selected_pairs()
    rows = []
    for world in (0, 1):
        for yes_first in (False, True):
            for state_variant in range(4):
                for claim_variant in range(4):
                    copies = (1 if arm in {"complete", "eval"} else
                              2 if (state_variant, claim_variant) in selected else 0)
                    for copy_index in range(copies):
                        rows.append(_row(facts, source_id, split, world, state_variant, claim_variant,
                                         yes_first, copy_index, arm))
    return rows


def _operand_shortcut_accuracy(train, evaluation):
    """Worst held-out accuracy of fixed one-operand threshold or residue rules."""
    results = []
    for field in (0, 1):
        values = sorted({row[field] for row in train})
        candidates = [values[0] - 1, *values, values[-1] + 1]
        rules = [(cutoff, upward) for cutoff in candidates for upward in (False, True)]
        rule = max(rules, key=lambda item: sum(((row[field] >= item[0]) == item[1]) == row[2]
                                               for row in train))
        results.append(sum(((row[field] >= rule[0]) == rule[1]) == row[2]
                           for row in evaluation) / len(evaluation))
        for modulus in range(2, 18):
            default = sum(row[2] for row in train) * 2 >= len(train)
            lookup = {}
            for residue in range(modulus):
                labels = [row[2] for row in train if row[field] % modulus == residue]
                lookup[residue] = sum(labels) * 2 >= len(labels) if labels else default
            results.append(sum(lookup[row[field] % modulus] == row[2] for row in evaluation) / len(evaluation))
    return max(results)


def _fact_table(train_sources, eval_sources, seed):
    for attempt in range(10_000):
        rng = random.Random(f"r039:{seed}:{attempt}")
        facts, train, evaluation = {}, [], []
        for split, count, target in (("train", train_sources, train), ("eval", eval_sources, evaluation)):
            for source_id in range(count):
                for world in (0, 1):
                    while True:
                        value, limit = rng.randrange(100, 900), rng.randrange(100, 900)
                        if value != limit and (value < limit) == (world == 0):
                            facts[(split, source_id, world)] = (value, limit)
                            target.append((value, limit, world == 0))
                            break
        if _operand_shortcut_accuracy(train, evaluation) <= .75:
            return facts
    raise ValueError("could not construct an operand-balanced fact table")


def build(train_sources=32, eval_sources=32, seed=42):
    if eval_sources < 32:
        raise ValueError("R039 needs at least 32 evaluation sources for a stable shortcut audit")
    facts = _fact_table(train_sources, eval_sources, seed)
    complete, partial, evaluation = [], [], []
    for source_id in range(train_sources):
        complete.extend(_rows_for_source(facts, source_id, "train", "complete"))
        partial.extend(_rows_for_source(facts, source_id, "train", "partial"))
    for source_id in range(eval_sources):
        evaluation.extend(_rows_for_source(facts, source_id, "eval", "eval"))
    validate(complete, partial, evaluation)
    return complete, partial, evaluation


def _by_source(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["meta"]["source_id"], []).append(row)
    return grouped


def rendered_signature(row):
    """The actual model-facing prompt and label, excluding opaque bookkeeping metadata."""
    question = row["questions"]["decision"]
    return (row["state"], question["instructions"], tuple(question["criteria"].items()),
            row["targets"]["decision"]["label"])


def operand_shortcut_accuracy(train, evaluation):
    """Worst held-out accuracy of simple one-operand threshold or residue rules."""
    canonical = lambda rows: [row for row in rows if row["meta"]["criteria_style"] == "state-0"
                              and row["meta"]["wording"] == "claim-0" and not row["meta"]["yes_first"]
                              and row["meta"]["copy_index"] == 0]
    as_pairs = lambda rows: [(row["meta"]["numeric_value"], row["meta"]["numeric_limit"],
                              row["targets"]["decision"]["label"] == "yes") for row in canonical(rows)]
    return _operand_shortcut_accuracy(as_pairs(train), as_pairs(evaluation))


def validate(complete, partial, evaluation):
    if len(complete) != len(partial) or len(complete) != 64 * len(_by_source(complete)):
        raise ValueError("arms must have equal physical rows and complete 64-view source orbits")
    if {row["meta"]["source_id"] for row in complete} & {row["meta"]["source_id"] for row in evaluation}:
        raise ValueError("source overlap between training and evaluation")
    if set(map(rendered_signature, complete)) & set(map(rendered_signature, evaluation)):
        raise ValueError("rendered train/eval overlap")
    if operand_shortcut_accuracy(complete, evaluation) > .75:
        raise ValueError("operand-only preflight exceeds chance tolerance")
    full_by_source, partial_by_source = _by_source(complete), _by_source(partial)
    if set(full_by_source) != set(partial_by_source):
        raise ValueError("arm source mismatch")
    for source_id, full_rows in full_by_source.items():
        partial_rows = partial_by_source[source_id]
        full_pairs = Counter((row["meta"]["criteria_style"], row["meta"]["wording"])
                             for row in full_rows)
        half_pairs = Counter((row["meta"]["criteria_style"], row["meta"]["wording"])
                             for row in partial_rows)
        if len(full_rows) != 64 or len(partial_rows) != 64 or len(full_pairs) != 16 or len(half_pairs) != 8:
            raise ValueError("invalid source orbit size or interaction coverage")
        for rows in (full_rows, partial_rows):
            if Counter(row["targets"]["decision"]["label"] for row in rows) != Counter({"yes": 32, "no": 32}):
                raise ValueError("source label imbalance")
            for yes_first in (False, True):
                labels = {row["targets"]["decision"]["label"] for row in rows
                          if row["meta"]["yes_first"] == yes_first}
                if labels != {"yes", "no"}:
                    raise ValueError("option order predicts the label")
        for state in range(4):
            if sum(pair[0] == f"state-{state}" for pair in half_pairs) != 2:
                raise ValueError("partial arm does not balance state surfaces")
            if sum(pair[1] == f"claim-{state}" for pair in half_pairs) != 2:
                raise ValueError("partial arm does not balance claim surfaces")
    for full, half in zip(complete, partial):
        if (full["meta"]["source_id"], full["meta"]["world"], full["meta"]["yes_first"]) != (
                half["meta"]["source_id"], half["meta"]["world"], half["meta"]["yes_first"]):
            raise ValueError("arms do not share the same source/label/order curriculum")


def _write(path, rows):
    with Path(path).open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def _digest(rows):
    return hashlib.sha256("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()).hexdigest()


def add_replay(replay, complete, partial):
    return [*replay, *complete], [*replay, *partial]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".context/r039-orbits")
    parser.add_argument("--train-sources", type=int, default=32)
    parser.add_argument("--eval-sources", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay")
    parser.add_argument("--replay-count", type=int, default=0)
    args = parser.parse_args()
    complete, partial, evaluation = build(args.train_sources, args.eval_sources, args.seed)
    replay = []
    if args.replay:
        replay = [json.loads(line) for line in Path(args.replay).read_text().splitlines() if line.strip()]
        replay = replay[:args.replay_count] if args.replay_count else replay
        if not replay:
            raise ValueError("replay input produced no rows")
        complete, partial = add_replay(replay, complete, partial)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("complete", complete), ("partial", partial), ("eval", evaluation)):
        _write(out / f"{name}.jsonl", rows)
    manifest = {"seed": args.seed, "train_sources": args.train_sources, "eval_sources": args.eval_sources,
                "replay_rows": len(replay), "orbit_rows_per_arm": len(complete) - len(replay),
                "rows": {name: len(rows) for name, rows in (("complete", complete), ("partial", partial),
                                                              ("eval", evaluation))},
                "sha256": {name: _digest(rows) for name, rows in (("complete", complete), ("partial", partial),
                                                                     ("eval", evaluation))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
