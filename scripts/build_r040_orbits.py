"""Build matched R040 Boolean-renderer coverage orbits.

The complete arm crosses Boolean truth, assertion polarity, assertion surface,
option semantics, and option order.  The partial arm repeats a balanced half of
the surfaces, keeping sources, labels, order exposure, and row count matched.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


TRAIN_STATE = (
    "Case {entity}. Manager approval: {status}.",
    "Review file for {entity}: the manager's approval is {status}.",
    "The case record for {entity} lists manager approval as {status}.",
)
EVAL_STATE = (
    "Audit entry for {entity}: management approval was {status}.",
    "For {entity}, the verified record says manager approval is {status}.",
    "The reviewed file for {entity} reports a {status} manager approval.",
)
ASSERTIONS = ("bare", "explicit", "interrogative")
OPTION_SURFACES = (
    ("generic", "No", "Yes"),
    ("boolean", "False", "True"),
)


def _claim(polarity):
    return "A manager approval is recorded." if polarity == "positive" else "No manager approval is recorded."


def _instructions(surface, polarity):
    claim = _claim(polarity)
    if surface == "bare":
        return claim
    if surface == "explicit":
        return f"Is the following statement true? {claim}"
    if surface == "interrogative":
        return "Is a manager approval recorded?" if polarity == "positive" else "Is manager approval absent?"
    raise ValueError(f"unknown assertion surface: {surface}")


def _row(source_id, split, world, polarity, assertion_index, option_index, yes_first, copy_index, arm):
    assertion_surface = ASSERTIONS[assertion_index]
    option_surface, no, yes = OPTION_SURFACES[option_index]
    criteria = {"no": no, "yes": yes}
    if yes_first:
        criteria = dict(reversed(list(criteria.items())))
    state_templates = EVAL_STATE if split == "eval" else TRAIN_STATE
    status = "recorded" if world else "not recorded"
    entity = f"case-{split}-{source_id:03d}"
    label = "yes" if world == (polarity == "positive") else "no"
    return {
        "id": (f"r040-{arm}-{split}-{source_id:03d}-w{int(world)}-p{polarity[0]}-"
               f"a{assertion_index}-c{option_index}-o{'yn' if yes_first else 'ny'}-r{copy_index}"),
        "state": state_templates[source_id % len(state_templates)].format(entity=entity, status=status),
        "questions": {"decision": {"type": "noul", "instructions": _instructions(assertion_surface, polarity),
                                   "criteria": criteria}},
        "targets": {"decision": {"label": label}},
        "meta": {"source": "r040_programmatic", "source_id": f"{split}-{source_id:03d}",
                 "world": int(world), "polarity": polarity, "field": "approved",
                 "assertion_true": label == "yes", "wording": assertion_surface,
                 "criteria_style": option_surface, "yes_first": yes_first, "fixed_option_order": True,
                 "arm": arm, "copy_index": copy_index},
    }


def _partial_surface(assertion_index, option_index, yes_first):
    """Balanced fractional surface orbit, independent of truth and assertion polarity."""
    selected_orders = ((False, True), (True, False), (False, True))
    return yes_first == selected_orders[assertion_index][option_index]


def _rows_for_source(source_id, split, arm):
    rows = []
    for world in (False, True):
        for polarity in ("positive", "negative"):
            for assertion_index in range(len(ASSERTIONS)):
                for option_index in range(len(OPTION_SURFACES)):
                    for yes_first in (False, True):
                        copies = 1 if arm in {"complete", "eval"} else 2 if _partial_surface(
                            assertion_index, option_index, yes_first) else 0
                        for copy_index in range(copies):
                            rows.append(_row(source_id, split, world, polarity, assertion_index, option_index,
                                             yes_first, copy_index, arm))
    return rows


def _by_source(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["meta"]["source_id"], []).append(row)
    return grouped


def rendered_signature(row):
    question = row["questions"]["decision"]
    return row["state"], question["instructions"], tuple(question["criteria"].items()), row["targets"]["decision"]["label"]


def build(train_sources=43, eval_sources=32):
    if eval_sources < 32:
        raise ValueError("R040 needs at least 32 held-out sources")
    complete = [row for source_id in range(train_sources)
                for row in _rows_for_source(source_id, "train", "complete")]
    partial = [row for source_id in range(train_sources)
               for row in _rows_for_source(source_id, "train", "partial")]
    evaluation = [row for source_id in range(eval_sources)
                  for row in _rows_for_source(source_id, "eval", "eval")]
    validate(complete, partial, evaluation)
    return complete, partial, evaluation


def validate(complete, partial, evaluation):
    if len(complete) != len(partial) or len(complete) != 48 * len(_by_source(complete)):
        raise ValueError("arms must be matched 48-view source orbits")
    if {row["meta"]["source_id"] for row in complete} & {row["meta"]["source_id"] for row in evaluation}:
        raise ValueError("source overlap between train and evaluation")
    if set(map(rendered_signature, complete)) & set(map(rendered_signature, evaluation)):
        raise ValueError("rendered train/eval overlap")
    full_by_source, partial_by_source = _by_source(complete), _by_source(partial)
    if set(full_by_source) != set(partial_by_source):
        raise ValueError("arm source mismatch")
    for source_id, full_rows in full_by_source.items():
        partial_rows = partial_by_source[source_id]
        surfaces = lambda rows: Counter((row["meta"]["wording"], row["meta"]["criteria_style"],
                                         row["meta"]["yes_first"]) for row in rows)
        if len(full_rows) != 48 or len(partial_rows) != 48:
            raise ValueError("invalid source orbit size")
        if len(surfaces(full_rows)) != 12 or len(surfaces(partial_rows)) != 6:
            raise ValueError("invalid complete or partial surface coverage")
        for rows in (full_rows, partial_rows):
            if Counter(row["targets"]["decision"]["label"] for row in rows) != Counter({"yes": 24, "no": 24}):
                raise ValueError("source label imbalance")
            for field in ("world", "polarity", "wording", "criteria_style", "yes_first"):
                values = {}
                for row in rows:
                    values.setdefault(row["meta"][field], set()).add(row["targets"]["decision"]["label"])
                if any(labels != {"yes", "no"} for labels in values.values()):
                    raise ValueError(f"{field} predicts the label")
        for assertion_index in range(len(ASSERTIONS)):
            selected = [surface for surface in surfaces(partial_rows) if surface[0] == ASSERTIONS[assertion_index]]
            if len(selected) != 2:
                raise ValueError("partial arm does not balance assertion surfaces")
    for full, half in zip(complete, partial):
        if tuple(full["meta"][key] for key in ("source_id", "world", "polarity")) != tuple(
                half["meta"][key] for key in ("source_id", "world", "polarity")):
            raise ValueError("arms do not share the same source/truth/polarity curriculum")


def _write(path, rows):
    with Path(path).open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def _digest(rows):
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def add_replay(replay, complete, partial):
    return [*replay, *complete], [*replay, *partial]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".context/r040-orbits")
    parser.add_argument("--train-sources", type=int, default=43)
    parser.add_argument("--eval-sources", type=int, default=32)
    parser.add_argument("--replay")
    parser.add_argument("--replay-count", type=int, default=0)
    args = parser.parse_args()
    complete, partial, evaluation = build(args.train_sources, args.eval_sources)
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
    manifest = {"train_sources": args.train_sources, "eval_sources": args.eval_sources,
                "replay_rows": len(replay), "orbit_rows_per_arm": len(complete) - len(replay),
                "rows": {name: len(rows) for name, rows in (("complete", complete), ("partial", partial),
                                                              ("eval", evaluation))},
                "sha256": {name: _digest(rows) for name, rows in (("complete", complete), ("partial", partial),
                                                                     ("eval", evaluation))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
