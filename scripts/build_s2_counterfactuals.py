"""Reconstruct the S1 2k presentation slice and build matched S2 wording arms."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


S1_COUNT = 1488
PRESENTATIONS = 2000
DONOR_GROUPS = 64
OPTIONS = ("no", "yes")
GENERIC = {"no": "No", "yes": "Yes"}
ARM_A_QUESTION = (
    "Is the requested action permitted under the stated policy?"
)
ARM_B_QUESTION = "The requested action is permitted under the stated policy."
EVAL_QUESTIONS = {
    ("positive", "interrogative"): "Do the governing rules authorize this request?",
    ("positive", "declarative"): "Under these rules, authorization is granted for this request.",
    ("negative", "interrogative"): "Is it true that the governing rules do not authorize this request?",
    ("negative", "declarative"): "The governing rules do not authorize this request.",
}
TRAIN_FULL_CHARS = 12_500
EVAL_FULL_CHARS = 40_000
POSITIONS = ("beginning", "middle", "end")
CLOSED_WORLD_CAVEAT = "Any requirement or exception not established by these facts counts as unmet."
DISTRACTOR_LABEL = "Unrelated policy and facts (not governing this request):"


def read_jsonl(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hash_rows(rows):
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def flatten_rows(paths):
    """One-question rows in exactly the same file/question iteration as the trainer."""
    rows = []
    for source_index, path in enumerate(paths):
        for source_row_index, row in enumerate(read_jsonl(path)):
            for qid, question in row["questions"].items():
                criteria = question.get("criteria")
                if question["type"] == "score":
                    labels = [str(index) for index in range(len(criteria))]
                else:
                    labels = [str(label) for label in criteria]
                if len(labels) > 26:
                    continue
                label = str(row["targets"][qid]["label"])
                if len(labels) < 2:
                    raise ValueError(f"{row.get('id')}/{qid}: unsupported option count")
                if label not in labels:
                    raise ValueError(f"{row.get('id')}/{qid}: target is not an option")
                probabilities = row.get("soft_targets", {}).get(qid)
                if probabilities is None:
                    probabilities = [float(value == label) for value in labels]
                if (len(probabilities) != len(labels) or any(value < 0 for value in probabilities)
                        or sum(probabilities) <= 0):
                    raise ValueError(f"{row.get('id')}/{qid}: invalid soft target")
                item = {
                    "id": f"s1-{source_index}-{source_row_index}-{row.get('id')}/{qid}",
                    "state": row["state"],
                    "questions": {"decision": {
                        "type": question["type"],
                        "instructions": question.get("instructions") or "",
                        "criteria": criteria,
                    }},
                    "targets": {"decision": {"label": label}},
                }
                if probabilities is not None:
                    item["soft_targets"] = {"decision": probabilities}
                item["meta"] = dict(row.get("meta") or {})
                item["meta"].update({"s1_source_id": f"{row.get('id')}/{qid}",
                                     "s1_source_file": Path(path).name,
                                     "s1_source_row_index": source_row_index})
                rows.append(item)
    return rows


def s1_schedule(paths, seed=42, count=PRESENTATIONS):
    rows = flatten_rows(paths)
    if len(rows) < count:
        raise ValueError(f"expected at least {count} flattened S1 rows, got {len(rows)}")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[index] for index in order[:count]], rows


def _group(row):
    meta = row.get("meta") or {}
    if (row.get("task") != "wf_policy_permit" or meta.get("family") != "policy_permit"
            or meta.get("qtype") != "noul" or not meta.get("rubric_group")
            or row.get("candidates") != ["no", "yes"]):
        return None
    target = row.get("target") or []
    if len(target) != 2 or sum(target) != 1 or any(v not in (0, 1, 0.0, 1.0) for v in target):
        return None
    return str(meta["rubric_group"])


def _label(row):
    return "yes" if row["target"][1] == 1 else "no"


def _source_id(row):
    return str(row.get("id") or
               f"{row.get('meta', {}).get('rubric_group')}-variant-"
               f"{row.get('meta', {}).get('rubric_variant', 'unknown')}")


def _question_parts(row):
    """Split the stored renderer output into instruction and visible yes/no criteria."""
    query = row["query"]
    try:
        instructions, rubric = query.split("\nyes: ", 1)
        yes_text, no_text = rubric.split("  no: ", 1)
    except ValueError as error:
        raise ValueError(f"{row.get('id')}: cannot parse the stored Noul query") from error
    return instructions, {"yes": yes_text.strip(), "no": no_text.strip()}


def _extract_policy(row):
    """Use the corpus's exact style function to remove style from prefix, not heuristics."""
    try:
        from scripts.workflow_corpus import NOUL_POLICY_SPEC, NOUL_STYLES
    except ModuleNotFoundError:
        from workflow_corpus import NOUL_POLICY_SPEC, NOUL_STYLES

    style = row["meta"]["rubric_style"]
    style_instruction = dict(NOUL_STYLES)[style](NOUL_POLICY_SPEC)[0]
    instructions, _ = _question_parts(row)
    suffix = " " + style_instruction
    if not instructions.endswith(suffix):
        raise ValueError(f"{row.get('id')}: policy/style boundary did not match {style!r}")
    policy = instructions[:-len(suffix)].strip()
    if not policy:
        raise ValueError(f"{row.get('id')}: empty extracted policy")
    return policy


def _by_group(train_rows, val_rows):
    held_out = {_group(row) for row in val_rows} - {None}
    groups = defaultdict(dict)
    for row in train_rows:
        group = _group(row)
        if group and group not in held_out:
            label = _label(row)
            groups[group].setdefault(label, []).append(row)
    return {gid: variants for gid, variants in groups.items()
            if set(variants) == {"no", "yes"}}, held_out


def _choose_groups(groups, count, seed):
    eligible = sorted(groups)
    if len(eligible) < count:
        raise ValueError(f"need {count} disjoint policy groups; found {len(eligible)}")
    return random.Random(seed).sample(eligible, count)


def _first_variant(rows):
    return sorted(rows, key=lambda row: (row.get("meta", {}).get("rubric_variant", -1), row.get("id", "")))[0]


def _context_block(source):
    policy = _extract_policy(source)
    return (f"State:\n{source['state']}\n\nPolicy:\n{policy}\n\n"
            f"Decision rule:\n{CLOSED_WORLD_CAVEAT}")


def _distractor_block(row):
    return (f"{DISTRACTOR_LABEL}\nPolicy: {_extract_policy(row)}\n"
            f"Facts: {row['state']}")


def _view_state(source, distractors=(), placement="beginning", target_chars=0):
    decisive = _context_block(source)
    if not target_chars:
        return decisive, []
    blocks = [_distractor_block(row) for row in distractors]
    if not blocks:
        raise ValueError("full context requires unrelated policy-group distractors")
    context = decisive
    cursor = 0
    while len(context) < target_chars:
        context += "\n\n" + blocks[cursor % len(blocks)]
        cursor += 1
    used_distractors = [distractors[index % len(distractors)] for index in range(cursor)]
    distractor_text = "\n\n".join(blocks[index % len(blocks)] for index in range(cursor))
    if placement == "beginning":
        state = decisive + "\n\n" + distractor_text
        return state, [_group(row) for row in used_distractors]
    if placement == "end":
        state = distractor_text + "\n\n" + decisive
        return state, [_group(row) for row in used_distractors]
    before_count = cursor // 2
    before = "\n\n".join(blocks[index % len(blocks)] for index in range(before_count))
    after = "\n\n".join(blocks[index % len(blocks)] for index in range(before_count, cursor))
    state = before + "\n\n" + decisive + "\n\n" + after
    return state, [_group(row) for row in used_distractors]


def _make_view(source, arm, view, option_order, distractors, index, group,
               question_style=None, criteria_style=None, polarity="positive",
               placement="beginning", context_chars=0):
    policy = _extract_policy(source)
    question_style = question_style or ("interrogative" if arm == "A" else "declarative")
    criteria_style = criteria_style or ("explicit" if arm == "A" else "generic")
    if arm == "crossed_eval":
        instruction = EVAL_QUESTIONS[(polarity, question_style)]
    elif question_style == "interrogative":
        instruction = ARM_A_QUESTION.replace("permitted", "not permitted") if polarity == "negative" else ARM_A_QUESTION
    else:
        instruction = ARM_B_QUESTION.replace("permitted", "not permitted") if polarity == "negative" else ARM_B_QUESTION
    labels = list(option_order)
    target = _label(source)
    if polarity == "negative":
        target = "yes" if target == "no" else "no"
    criteria = _explicit_criteria(source, polarity)
    state, used_distractor_groups = _view_state(
        source, distractors if view == "full" else (), placement,
        context_chars if view == "full" else 0,
    )
    row = {
        "id": f"s2-{arm.lower()}-{index:04d}-{group}-{target}-{view}-{'ny' if labels[0]=='no' else 'yn'}",
        "state": state,
        "questions": {"decision": {
            "type": "noul", "instructions": instruction,
            "criteria": {label: (criteria[label] if criteria_style == "explicit" else GENERIC[label])
                         for label in labels},
        }},
        "targets": {"decision": {"label": target}},
        "meta": {"source": "programmatic_policy_counterfactual", "source_group": group,
                 "source_id": _source_id(source), "source_label": _label(source),
                 "policy": policy, "s2_arm": arm,
                 "question_style": question_style, "criteria_style": criteria_style,
                 "context_view": view, "polarity": polarity, "decisive_placement": placement,
                 "distractor_groups": used_distractor_groups,
                 "fixed_option_order": True},
    }
    return row


def _explicit_criteria(source, polarity):
    """Invert semantic conditions first, then render the source style's matching markers."""
    try:
        from scripts.workflow_corpus import NOUL_POLICY_SPEC, NOUL_STYLES
    except ModuleNotFoundError:
        from workflow_corpus import NOUL_POLICY_SPEC, NOUL_STYLES
    spec = dict(NOUL_POLICY_SPEC)
    if polarity == "negative":
        spec["yes"], spec["no"] = spec["no"], spec["yes"]
    style = source["meta"]["rubric_style"]
    rendered = dict(NOUL_STYLES)[style](spec)[1]
    return {"yes": rendered["true"], "no": rendered["false"]}


def _donor_rows(groups, selected_groups, seed):
    eligible = sorted(set(groups) - set(selected_groups))
    if not eligible:
        raise ValueError("need unselected disjoint groups to supply full-view distractors")
    rotated = list(eligible)
    random.Random(seed + 1).shuffle(rotated)
    rows = []
    for group_index, group in enumerate(selected_groups):
        offset = (group_index * 37) % len(rotated)
        distractors = [_first_variant(groups[g]["no"])
                       for g in (rotated[offset:] + rotated[:offset])[:128]]
        polarity = "positive" if group_index < DONOR_GROUPS // 2 else "negative"
        placement = POSITIONS[group_index % len(POSITIONS)]
        for label in ("no", "yes"):
            source = _first_variant(groups[group][label])
            for view in ("compact", "full"):
                for order in (OPTIONS, tuple(reversed(OPTIONS))):
                    rows.append((source, view, order, distractors, group_index, group,
                                 polarity, placement))
    return rows, {"available_distractor_groups": len(eligible)}


def crossed_eval_rows(train_rows, val_rows, seed):
    eval_groups, held_out = _by_group(val_rows, [])
    train_group_ids = {_group(row) for row in train_rows} - {None}
    overlap = train_group_ids & set(eval_groups)
    if overlap:
        raise ValueError(f"policy group leakage between train and eval: {sorted(overlap)[:3]}")
    selected = _choose_groups(eval_groups, DONOR_GROUPS, seed + 10_000)
    remaining = sorted(set(eval_groups) - set(selected))
    if not remaining:
        raise ValueError("need unselected validation groups for full-view distractors")
    rotated = list(remaining)
    random.Random(seed + 10_001).shuffle(rotated)
    rows = []
    for group_index, group in enumerate(selected):
        offset = (group_index * 7) % len(rotated)
        distractor_rows = [_first_variant(eval_groups[g]["no"])
                           for g in (rotated[offset:] + rotated[:offset])]
        polarity = "positive" if group_index < DONOR_GROUPS // 2 else "negative"
        placement = POSITIONS[group_index % len(POSITIONS)]
        for label in ("no", "yes"):
            source = _first_variant(eval_groups[group][label])
            for question_style in ("interrogative", "declarative"):
                for criteria_style in ("explicit", "generic"):
                    for order in (OPTIONS, tuple(reversed(OPTIONS))):
                        for view in ("compact", "full"):
                            rows.append(_make_view(
                                source, "crossed_eval", view, order, distractor_rows,
                                len(rows), group, question_style, criteria_style,
                                polarity, placement, EVAL_FULL_CHARS,
                            ))
    counts = Counter((row["meta"]["question_style"], row["meta"]["criteria_style"],
                      "no_yes" if next(iter(row["questions"]["decision"]["criteria"])) == "no" else "yes_no",
                      row["meta"]["context_view"]) for row in rows)
    if len(rows) != 2048 or len(counts) != 16 or set(counts.values()) != {128}:
        raise AssertionError("crossed eval must have 16 cells with 128 rows each")
    return rows, {"groups": selected, "excluded_train_groups": len(overlap),
                  "eval_group_pool": len(eval_groups), "label_counts": {
                      label: sum(row["targets"]["decision"]["label"] == label for row in rows)
                      for label in OPTIONS},
                  "polarity_group_counts": {"positive": DONOR_GROUPS // 2,
                                            "negative": DONOR_GROUPS // 2},
                  "placement_group_counts": dict(Counter(POSITIONS[i % len(POSITIONS)]
                                                          for i in range(DONOR_GROUPS))),
                  "distractor_groups": sorted({group for row in rows
                                                for group in row["meta"]["distractor_groups"]}),
                  "cell_counts": {"each_of_16_cells": 128}}


def build(train_paths, donor_train, donor_val, seed=42):
    schedule, full_s1 = s1_schedule(train_paths, seed)
    groups, held_out = _by_group(donor_train, donor_val)
    selected = _choose_groups(groups, DONOR_GROUPS, seed)
    donor_specs, distractor_pool_stats = _donor_rows(groups, selected, seed)
    arm_a = list(schedule[:S1_COUNT])
    arm_b = list(schedule[:S1_COUNT])
    for index, (source, view, order, distractor_rows, group_index, group, polarity, placement) in enumerate(donor_specs):
        arm_a.append(_make_view(source, "A", view, order, distractor_rows, index, group,
                                polarity=polarity, placement=placement, context_chars=TRAIN_FULL_CHARS))
        arm_b.append(_make_view(source, "B", view, order, distractor_rows, index, group,
                                polarity=polarity, placement=placement, context_chars=TRAIN_FULL_CHARS))
    if len(arm_a) != PRESENTATIONS or len(arm_b) != PRESENTATIONS:
        raise AssertionError("S2 arms must contain exactly 2,000 presentations")
    eval_rows, eval_stats = crossed_eval_rows(donor_train, donor_val, seed)
    train_distractor_groups = {group for row in arm_a[S1_COUNT:]
                               for group in row["meta"]["distractor_groups"]}
    train_eval_source_groups = train_distractor_groups | set(selected)
    eval_source_groups = set(eval_stats["groups"]) | set(eval_stats["distractor_groups"])
    source_group_overlap = train_eval_source_groups & eval_source_groups
    if source_group_overlap:
        raise ValueError(f"train/eval context group overlap: {sorted(source_group_overlap)[:3]}")
    manifest = {
        "seed": seed, "s1_source_examples_flattened": len(full_s1),
        "promotion_scope": "policy_permit only; no non-permission family transfer panel",
        "shared_s1_examples_count": S1_COUNT, "matched_noul_views_per_arm": PRESENTATIONS - S1_COUNT,
        "total_presentations_per_arm": PRESENTATIONS,
        "s1_presentation_schedule_seed": seed,
        "shared_s1_example_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in arm_a[:S1_COUNT]).encode()).hexdigest(),
        "shared_s1_example_rows_sha256": _hash_rows(arm_a[:S1_COUNT]),
        "shared_s1_example_ids": [row["id"] for row in arm_a[:S1_COUNT]],
        "shared_s1_example_source_refs": [row["meta"] for row in arm_a[:S1_COUNT]],
        "donor_group_count": len(selected), "selected_groups": selected,
        "selected_train_donor_ids": list(dict.fromkeys(
            row["meta"]["source_id"] for row in arm_a[S1_COUNT:])),
        "excluded_validation_groups": len(held_out), "eligible_donor_groups": len(groups),
        "donor_labels": dict(Counter(r["targets"]["decision"]["label"] for r in arm_a[S1_COUNT:])),
        "option_orders": dict(Counter("no_yes" if row[2][0] == "no" else "yes_no" for row in donor_specs)),
        "context_views": dict(Counter(row[1] for row in donor_specs)),
        "polarity_groups": dict(Counter(row[6] for row in donor_specs[::8])),
        "placement_groups": dict(Counter(row[7] for row in donor_specs[::8])),
        "context_targets_approx_tokens": {"train_full": TRAIN_FULL_CHARS // 4,
                                          "eval_full": EVAL_FULL_CHARS // 4},
        "distractor_pool_groups": distractor_pool_stats["available_distractor_groups"],
        "train_distractor_groups": sorted(train_distractor_groups),
        "train_eval_context_group_intersection": len(source_group_overlap),
    }
    manifest["crossed_eval"] = eval_stats
    manifest["train_eval_group_intersection"] = 0
    manifest["output_sha256"] = {"arm_a": _hash_rows(arm_a), "arm_b": _hash_rows(arm_b),
                                 "crossed_eval": _hash_rows(eval_rows)}
    return arm_a, arm_b, eval_rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-input", action="append", required=True,
                        help="repeat in exact S1 concatenation order")
    parser.add_argument("--programmatic-train", required=True)
    parser.add_argument("--programmatic-val", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    arm_a, arm_b, eval_rows, manifest = build([Path(p) for p in args.s1_input],
                                   read_jsonl(args.programmatic_train),
                                   read_jsonl(args.programmatic_val), args.seed)
    source_paths = [*args.s1_input, args.programmatic_train, args.programmatic_val]
    manifest["source_inputs_sha256"] = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "arm-a.jsonl", arm_a)
    write_jsonl(out / "arm-b.jsonl", arm_b)
    write_jsonl(out / "crossed-eval.jsonl", eval_rows)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {key: value for key, value in manifest.items()
               if key not in {"shared_s1_example_ids", "shared_s1_example_source_refs", "selected_groups",
                              "selected_train_donor_ids", "train_distractor_groups"}}
    summary["crossed_eval"] = {key: value for key, value in manifest["crossed_eval"].items()
                               if key != "groups" and key != "distractor_groups"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
