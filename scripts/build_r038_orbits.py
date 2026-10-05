"""Build matched complete- versus partial-semantic-orbit decision data."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


PREDICATES = {
    "safe": ("Risk score recorded: 20/100. Policy limit: 50/100.",
             "Risk score recorded: 90/100. Policy limit: 50/100.",
             "The recorded risk score is within the policy limit.",
             "The recorded risk score exceeds the policy limit."),
    "approved": ("Manager approval: recorded in the case file.",
                 "Manager approval: not recorded in the case file.",
                 "A manager approval is recorded for this request.",
                 "No manager approval is recorded for this request."),
    "within_limit": ("Requested amount: $120. Authorized limit: $150.",
                     "Requested amount: $190. Authorized limit: $150.",
                     "The requested amount is within the authorized limit.",
                     "The requested amount exceeds the authorized limit."),
    "timely": ("Request age: 3 days. Maximum age: 7 days.",
               "Request age: 10 days. Maximum age: 7 days.",
               "The request is within the allowed age window.",
               "The request is older than the allowed age window."),
    "evidence": ("Required evidence packet: complete.", "Required evidence packet: missing.",
                 "All required evidence is present.", "Required evidence is missing."),
    "covered": ("Policy coverage: confirmed for this request.",
                "Policy coverage: not confirmed for this request.",
                "The request is covered by the policy.", "The request is not covered by the policy."),
    "deadline": ("Submission time: before the stated deadline.",
                 "Submission time: after the stated deadline.",
                 "The submission was made before the deadline.",
                 "The submission was made after the deadline."),
    "owner": ("Assigned reviewer: an authorized owner is recorded.",
              "Assigned reviewer: no authorized owner is recorded.",
              "An authorized owner is assigned to this request.",
              "No authorized owner is assigned to this request."),
}


def _state(source_id, predicate, world):
    fact = PREDICATES[predicate][world]
    return "\n".join((
        f"Case reference: {source_id}.", "Record status: verified.",
        "Use only the recorded facts; do not infer missing evidence.", fact,
    ))


def _view(source_id, predicate, world, polarity, wording, criteria_style, yes_first, arm, copy_index):
    positive, negative = PREDICATES[predicate][2:]
    statement = positive if polarity == "positive" else negative
    instruction = (f"Is the following statement true? {statement}" if wording == "interrogative"
                   else statement)
    target = world == 0
    if polarity == "negative":
        target = not target
    criteria = ({"no": "False", "yes": "True"} if criteria_style == "explicit"
                else {"no": "No", "yes": "Yes"})
    if yes_first:
        criteria = dict(reversed(list(criteria.items())))
    return {
        "id": (f"r038-{arm}-{source_id}-{predicate}-w{world}-p{polarity[0]}-"
               f"q{wording[0]}-c{criteria_style[0]}-o{'yn' if yes_first else 'ny'}-r{copy_index}"),
        "state": _state(source_id, predicate, world),
        "questions": {"decision": {"type": "noul", "instructions": instruction, "criteria": criteria}},
        "targets": {"decision": {"label": "yes" if target else "no"}},
        "meta": {"source": "r038_programmatic", "source_id": source_id, "predicate": predicate,
                 "world": world, "polarity": polarity, "wording": wording,
                 "criteria_style": criteria_style, "yes_first": yes_first, "arm": arm,
                 "copy_index": copy_index},
    }


def _variants(source_id, predicate, arm):
    rows = []
    for world in (0, 1):
        for polarity in ("positive", "negative"):
            for wording in ("interrogative", "declarative"):
                for criteria_style in ("explicit", "generic"):
                    for yes_first in (False, True):
                        parity = world + (polarity == "negative") + (wording == "declarative")
                        parity += (criteria_style == "generic") + yes_first
                        copies = 1 if arm in {"complete", "eval"} else (2 if parity % 2 == 0 else 0)
                        for copy_index in range(copies):
                            rows.append(_view(source_id, predicate, world, polarity, wording,
                                              criteria_style, yes_first, arm, copy_index))
    return rows


def build(train_sources=64, eval_sources=16, seed=42):
    if train_sources % len(PREDICATES) or eval_sources % len(PREDICATES):
        raise ValueError("source counts must divide evenly across predicates")
    predicates, rng = tuple(PREDICATES), random.Random(seed)
    full, half, evaluation = [], [], []
    for split, count in (("train", train_sources), ("eval", eval_sources)):
        for index in range(count):
            predicate = predicates[index % len(predicates)]
            source_id = f"{split}-{index:03d}"
            if split == "train":
                full.extend(_variants(source_id, predicate, "complete"))
                half.extend(_variants(source_id, predicate, "partial"))
            else:
                evaluation.extend(_variants(source_id, predicate, "eval"))
    rng.shuffle(full)
    rng.shuffle(half)
    rng.shuffle(evaluation)
    validate(full, half, evaluation)
    return full, half, evaluation


def _factors(rows):
    return Counter(tuple(row["meta"][name] for name in
                         ("world", "polarity", "wording", "criteria_style", "yes_first")) for row in rows)


def validate(full, half, evaluation):
    train_ids = {row["meta"]["source_id"] for row in full}
    eval_ids = {row["meta"]["source_id"] for row in evaluation}
    if train_ids & eval_ids:
        raise ValueError("source overlap between training and evaluation")
    if len(full) != len(half) or len(full) != 32 * len(train_ids):
        raise ValueError("arms must have equal physical views and complete 32-view source orbits")
    if any(len(rows) != 32 for rows in _by_source(full).values()):
        raise ValueError("complete arm lacks a full orbit")
    if any(len(rows) != 32 or len(_factors(rows)) != 16 for rows in _by_source(half).values()):
        raise ValueError("partial arm must repeat exactly 16 logical views")
    for name in ("world", "polarity", "wording", "criteria_style", "yes_first"):
        if Counter(row["meta"][name] for row in full) != Counter(row["meta"][name] for row in half):
            raise ValueError(f"unmatched {name} marginal")
    if Counter(row["targets"]["decision"]["label"] for row in full) != Counter(
            row["targets"]["decision"]["label"] for row in half):
        raise ValueError("unmatched target marginal")


def _by_source(rows):
    result = {}
    for row in rows:
        result.setdefault(row["meta"]["source_id"], []).append(row)
    return result


def _write(path, rows):
    with Path(path).open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def _digest(rows):
    return hashlib.sha256("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()).hexdigest()


def add_replay(replay, complete, partial):
    """Prefix identical retained examples to both matched experimental arms."""
    return [*replay, *complete], [*replay, *partial]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".context/r038-orbits")
    parser.add_argument("--train-sources", type=int, default=64)
    parser.add_argument("--eval-sources", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", help="optional JSONL prefix shared byte-for-byte by both arms")
    parser.add_argument("--replay-count", type=int, default=0)
    args = parser.parse_args()
    full, half, evaluation = build(args.train_sources, args.eval_sources, args.seed)
    replay = []
    if args.replay:
        replay = [json.loads(line) for line in Path(args.replay).read_text().splitlines() if line.strip()]
        replay = replay[:args.replay_count] if args.replay_count else replay
        if not replay:
            raise ValueError("replay input produced no rows")
        full, half = add_replay(replay, full, half)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("complete", full), ("partial", half), ("eval", evaluation)):
        _write(out / f"{name}.jsonl", rows)
    manifest = {"seed": args.seed, "train_sources": args.train_sources, "eval_sources": args.eval_sources,
                "replay_rows": len(replay), "replay_sha256": _digest(replay) if replay else None,
                "orbit_rows_per_arm": len(full) - len(replay),
                "rows": {name: len(rows) for name, rows in (("complete", full), ("partial", half),
                                                              ("eval", evaluation))},
                "sha256": {name: _digest(rows) for name, rows in (("complete", full), ("partial", half),
                                                                     ("eval", evaluation))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
