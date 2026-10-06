"""Train our own typed-decision LoRA on a fresh Gemma 4 multimodal base.

The loss is computed only over the supplied option letters at the single answer
position, matching the Cygnet/Winnow serving readout without inheriting either
model's weights.
"""

import argparse
import json
import math
import random
import time
from numbers import Real
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from scripts.exact_policy_executor import (
    canonicalize_outcome,
    compile_policy,
    exact_outcome_log_distribution,
    execute_policy,
    outcome_brier,
    outcome_nll,
)
from scripts.soft_action_executor import (
    ACTIONS,
    FIELD_NAMES,
    REQUIRED_FIELDS,
    action_for,
    action_log_cross_entropy,
    semantic_truth_logits,
    soft_action_log_distribution,
)


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)
TARGET_MODULES = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
GROUPED_ACTION_PANELS = frozenset(("action_v1", "exact_policy_v1"))


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-adapter")
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--val", action="append", default=[], help="NAME=JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--save-steps", default="",
                        help="Comma-separated additional checkpoint steps")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-consistency-weight", type=float, default=0.0)
    parser.add_argument("--pair-consistency-min-gold-probability", type=float, default=0.80)
    parser.add_argument("--pair-consistency-clip", type=float, default=4.0)
    parser.add_argument("--pair-consistency-beta", type=float, default=1.0)
    parser.add_argument("--project-consistency-against-guard", action="store_true")
    parser.add_argument("--pair-flip-weight", type=float, default=0.0)
    parser.add_argument("--pair-flip-margin", type=float, default=1.0)
    parser.add_argument("--pair-flip-start-step", type=int, default=0)
    parser.add_argument("--stage-counterfactual-pairs", action="store_true")
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parser.add_argument("--action-temperature", type=float, default=1.0)
    parser.add_argument("--exact-policy-outcome-loss", choices=("nll", "brier"),
                        default="nll")
    return parser.parse_args()


def parse_save_steps(value, total_steps):
    if not value:
        return set()
    try:
        steps = {int(item) for item in value.split(",") if item}
    except ValueError as error:
        raise ValueError("save steps must be comma-separated integers") from error
    if any(step < 1 or step > total_steps for step in steps):
        raise ValueError("save steps must fall within the training schedule")
    return steps


def flatten_row(row, source_dir=None):
    image = row.get("image")
    if image and source_dir is not None:
        image = Path(image)
        image = image if image.is_absolute() else Path(source_dir) / image
        image = str(image.resolve())
    examples = []
    for qid, question in row["questions"].items():
        criteria = question.get("criteria")
        if question["type"] == "score":
            items = [(str(index), value) for index, value in enumerate(criteria)]
        else:
            items = list(criteria.items())
        if len(items) > len(LETTERS):
            continue
        if len(items) < 2:
            raise ValueError(f"{row.get('id')}/{qid}: unsupported option count")
        label = str(row["targets"][qid]["label"])
        labels = [str(key) for key, _ in items]
        if label not in labels:
            raise ValueError(f"{row.get('id')}/{qid}: target is not an option")
        probabilities = row.get("soft_targets", {}).get(qid)
        if probabilities is None:
            probabilities = [float(value == label) for value in labels]
        if len(probabilities) != len(items) or any(value < 0 for value in probabilities):
            raise ValueError(f"{row.get('id')}/{qid}: invalid soft target")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError(f"{row.get('id')}/{qid}: empty soft target")
        examples.append({
            "id": f"{row.get('id')}/{qid}",
            "state": row["state"],
            "instructions": question.get("instructions") or "",
            "items": [(str(key), value) for key, value in items],
            "probabilities": [float(value) / total for value in probabilities],
            "fixed_order": bool((row.get("meta") or {}).get("fixed_option_order")),
            "gold_label": label,
            "image": image,
        })
    return examples


def _action_group(row, source_dir=None):
    """Preserve an action-panel row as one grouped optimizer slot."""
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("action group id must be a nonempty string")
    meta = row.get("meta")
    if not isinstance(meta, dict) or meta.get("panel") != "action_v1":
        raise ValueError(f"{identifier}: malformed action_v1 metadata")
    if "image" in row and row["image"] is not None:
        raise ValueError(f"{identifier}: action groups must be text-only")
    if "paired_views" in row or "auxiliary_views" in row:
        raise ValueError(f"{identifier}: action groups cannot be paired or auxiliary wrappers")
    questions, targets = row.get("questions"), row.get("targets")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{identifier}: action group needs questions")
    if not isinstance(targets, dict) or set(targets) != set(questions):
        raise ValueError(f"{identifier}: action group targets must exactly match its qids")
    expected_qids = [f"q{index:02d}" for index in range(len(questions))]
    if set(questions) != set(expected_qids):
        raise ValueError(f"{identifier}: action qids must be contiguous q00..qNN")
    field_order = meta.get("field_order")
    if (not isinstance(field_order, list) or len(field_order) != len(expected_qids)
            or any(not isinstance(field, str) or not field for field in field_order)):
        raise ValueError(f"{identifier}: field_order must exactly map action-group qids")
    if len(set(field_order)) != len(field_order):
        raise ValueError(f"{identifier}: action fields must be unique")
    fields = set(field_order)
    if not REQUIRED_FIELDS <= fields or not fields <= set(FIELD_NAMES):
        raise ValueError(f"{identifier}: action fields are missing or unsupported")
    qid_to_field = dict(zip(expected_qids, field_order))
    explicit_mapping = meta.get("qid_to_field")
    if explicit_mapping is not None and explicit_mapping != qid_to_field:
        raise ValueError(f"{identifier}: qid_to_field conflicts with field_order")
    threshold = meta.get("decision_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError(f"{identifier}: invalid action decision_threshold")
    action_gold = meta.get("action_gold")
    if action_gold not in ACTIONS:
        raise ValueError(f"{identifier}: invalid action_gold")
    action_trace = meta.get("action_trace")
    truths = {qid_to_field[qid]: targets[qid].get("label") == "yes"
              for qid in expected_qids if isinstance(targets[qid], dict)}
    if (not isinstance(action_trace, dict) or set(action_trace) != fields
            or any(type(value) is not bool for value in action_trace.values())
            or len(truths) != len(expected_qids) or action_trace != truths
            or action_for(truths) != action_gold):
        raise ValueError(f"{identifier}: action trace or gold action mismatch")
    views = flatten_row(row, source_dir)
    by_qid = {}
    prefix = f"{identifier}/"
    for view in views:
        if not view["id"].startswith(prefix):
            raise ValueError(f"{identifier}: cannot recover action qid")
        qid = view["id"][len(prefix):]
        if qid in by_qid:
            raise ValueError(f"{identifier}: duplicate action qid {qid!r}")
        if set(label for label, _ in view["items"]) != {"no", "yes"}:
            raise ValueError(f"{identifier}/{qid}: action fields require semantic no/yes labels")
        by_qid[qid] = view
    if set(by_qid) != set(qid_to_field):
        raise ValueError(f"{identifier}: flattened action qids do not match qid_to_field")
    return {
        "id": identifier,
        "action_views": [by_qid[qid] for qid in expected_qids],
        "action_qids": expected_qids,
        "action_qid_to_field": dict(qid_to_field),
        "action_gold": action_gold,
        "decision_threshold": float(threshold),
    }


def _exact_policy_group(row, source_dir=None):
    """Validate and cache one generic exact-policy optimizer slot."""
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("exact-policy group id must be a nonempty string")
    meta = row.get("meta")
    if not isinstance(meta, dict) or meta.get("panel") != "exact_policy_v1":
        raise ValueError(f"{identifier}: malformed exact_policy_v1 metadata")
    if "image" in row and row["image"] is not None:
        raise ValueError(f"{identifier}: exact-policy groups must be text-only")
    if "paired_views" in row or "auxiliary_views" in row:
        raise ValueError(f"{identifier}: exact-policy groups cannot be paired or auxiliary wrappers")

    questions, targets = row.get("questions"), row.get("targets")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{identifier}: exact-policy group needs questions")
    if not isinstance(targets, dict) or set(targets) != set(questions):
        raise ValueError(
            f"{identifier}: exact-policy group targets must exactly match its qids")
    expected_qids = [f"q{index:02d}" for index in range(len(questions))]
    if set(questions) != set(expected_qids):
        raise ValueError(f"{identifier}: exact-policy qids must be contiguous q00..qNN")

    # Compilation is intentionally a loader operation.  The immutable result is
    # retained on the example and reused by every optimizer step.
    compiled_policy = compile_policy(meta.get("policy"))
    field_order = meta.get("field_order")
    if (not isinstance(field_order, list)
            or field_order != list(compiled_policy.fields)):
        raise ValueError(
            f"{identifier}: field_order must exactly match policy fields")
    qid_to_field = dict(zip(expected_qids, field_order))
    explicit_mapping = meta.get("qid_to_field")
    if explicit_mapping is not None and explicit_mapping != qid_to_field:
        raise ValueError(f"{identifier}: qid_to_field conflicts with field_order")

    threshold = meta.get("decision_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError(f"{identifier}: invalid exact-policy decision_threshold")
    action_trace = meta.get("action_trace")
    if (not isinstance(action_trace, dict)
            or set(action_trace) != set(compiled_policy.fields)
            or any(type(value) is not bool for value in action_trace.values())):
        raise ValueError(f"{identifier}: invalid exact-policy action_trace")
    target_truths = {}
    for qid in expected_qids:
        question = questions[qid]
        if (not isinstance(question, dict) or question.get("type") != "noul"
                or not isinstance(question.get("instructions"), str)
                or not question["instructions"].strip()
                or not isinstance(question.get("criteria"), dict)
                or set(question["criteria"]) != {"no", "yes"}
                or any(not isinstance(value, str) or not value.strip()
                       for value in question["criteria"].values())):
            raise ValueError(
                f"{identifier}/{qid}: exact-policy fields require semantic no/yes options")
        target = targets[qid]
        if not isinstance(target, dict) or target.get("label") not in {"no", "yes"}:
            raise ValueError(
                f"{identifier}/{qid}: exact-policy target must be semantic no/yes")
        target_truths[qid_to_field[qid]] = target["label"] == "yes"
    try:
        gold_outcome = canonicalize_outcome(meta.get("gold_outcome"))
    except ValueError as error:
        raise ValueError(f"{identifier}: invalid gold_outcome: {error}") from error
    if (target_truths != action_trace
            or execute_policy(compiled_policy, action_trace) != gold_outcome):
        raise ValueError(f"{identifier}: exact-policy trace or gold outcome mismatch")

    views = flatten_row(row, source_dir)
    by_qid = {}
    prefix = f"{identifier}/"
    for view in views:
        if not view["id"].startswith(prefix):
            raise ValueError(f"{identifier}: cannot recover exact-policy qid")
        qid = view["id"][len(prefix):]
        if qid in by_qid:
            raise ValueError(f"{identifier}: duplicate exact-policy qid {qid!r}")
        if set(label for label, _ in view["items"]) != {"no", "yes"}:
            raise ValueError(
                f"{identifier}/{qid}: exact-policy fields require semantic no/yes labels")
        by_qid[qid] = view
    if set(by_qid) != set(qid_to_field):
        raise ValueError(
            f"{identifier}: flattened exact-policy qids do not match qid_to_field")
    return {
        "id": identifier,
        "action_kind": "exact_policy_v1",
        "action_views": [by_qid[qid] for qid in expected_qids],
        "action_qids": expected_qids,
        "action_qid_to_field": qid_to_field,
        "action_trace": dict(action_trace),
        "compiled_policy": compiled_policy,
        "gold_outcome": gold_outcome,
        "decision_threshold": float(threshold),
    }


def load_examples(paths):
    examples = []
    for path in paths:
        path = Path(path)
        with path.open() as source:
            for line in source:
                if line.strip():
                    row = json.loads(line, object_pairs_hook=_unique_json_object)
                    meta = row.get("meta")
                    if isinstance(meta, dict) and meta.get("panel") == "action_v1":
                        examples.append(_action_group(row, path.parent))
                        continue
                    if isinstance(meta, dict) and meta.get("panel") == "exact_policy_v1":
                        examples.append(_exact_policy_group(row, path.parent))
                        continue
                    wrapped_views = [
                        view for key in ("paired_views", "auxiliary_views")
                        if isinstance(row.get(key), list) for view in row[key]
                    ]
                    if any(isinstance(view, dict)
                           and isinstance(view.get("meta"), dict)
                           and view["meta"].get("panel") in GROUPED_ACTION_PANELS
                           for view in wrapped_views):
                        raise ValueError(f"{row.get('id')}: action groups cannot be paired wrappers")
                    if "auxiliary_views" in row and "paired_views" not in row:
                        raise ValueError(
                            f"{row.get('id')}: auxiliary_views are only valid on a paired slot")
                    if "paired_views" in row:
                        if (not isinstance(row["paired_views"], list) or
                                len(row["paired_views"]) != 2):
                            raise ValueError(f"{row.get('id')}: paired slot must have exactly two views")
                        if any(not isinstance(view, dict) for view in row["paired_views"]):
                            raise ValueError(f"{row.get('id')}: paired views must be objects")
                        views = [flatten_row(view, path.parent) for view in row["paired_views"]]
                        if any(len(view) != 1 for view in views):
                            raise ValueError(f"{row.get('id')}: each paired view must flatten to one example")
                        if any("auxiliary_views" in view for view in row["paired_views"]):
                            raise ValueError(
                                f"{row.get('id')}: auxiliary_views belong on the paired row")
                        auxiliary_rows = row.get("auxiliary_views", [])
                        if not isinstance(auxiliary_rows, list):
                            raise ValueError(f"{row.get('id')}: auxiliary_views must be a list")
                        if any(not isinstance(view, dict) for view in auxiliary_rows):
                            raise ValueError(f"{row.get('id')}: auxiliary views must be objects")
                        auxiliary = [flatten_row(view, path.parent) for view in auxiliary_rows]
                        if any(len(view) != 1 for view in auxiliary):
                            raise ValueError(
                                f"{row.get('id')}: each auxiliary view must flatten to one example")
                        auxiliary = [view[0] for view in auxiliary]
                        if any(view.get("image") for view in auxiliary):
                            raise ValueError(f"{row.get('id')}: auxiliary views must be text-only")
                        all_view_ids = [view[0]["id"] for view in views] + [
                            view["id"] for view in auxiliary]
                        if len(all_view_ids) != len(set(all_view_ids)):
                            raise ValueError(f"{row.get('id')}: paired and auxiliary view ids must be unique")
                        paired_metas = [(raw.get("meta") or {}) for raw in row["paired_views"]]
                        pair_mode = row.get("pair_mode")
                        if pair_mode is not None:
                            if pair_mode not in ("counterfactual", "shuffled_counterfactual"):
                                raise ValueError(f"{row.get('id')}: unsupported pair mode {pair_mode!r}")
                            labels = [[label for label, _ in view[0]["items"]] for view in views]
                            if any(len(view_labels) != 2 for view_labels in labels):
                                raise ValueError(f"{row.get('id')}: counterfactual views need exactly two options")
                            if set(labels[0]) != set(labels[1]):
                                raise ValueError(
                                    f"{row.get('id')}: counterfactual views need identical option labels")
                            if views[0][0]["gold_label"] == views[1][0]["gold_label"]:
                                raise ValueError(
                                    f"{row.get('id')}: counterfactual views need opposite gold labels")
                            source_ids = row.get("source_ids")
                            if not isinstance(source_ids, list) or len(source_ids) != 2:
                                raise ValueError(f"{row.get('id')}: counterfactual pair needs two source_ids")
                            view_source_ids = [meta.get("source_id") for meta in paired_metas]
                            if any(source_id is None for source_id in source_ids + view_source_ids):
                                raise ValueError(f"{row.get('id')}: counterfactual source_ids cannot be null")
                            if view_source_ids != source_ids:
                                raise ValueError(
                                    f"{row.get('id')}: wrapper and view source_ids must match")
                            if pair_mode == "counterfactual" and source_ids[0] != source_ids[1]:
                                raise ValueError(
                                    f"{row.get('id')}: counterfactual views must share a source_id")
                            if pair_mode == "shuffled_counterfactual":
                                families = [meta.get("family") for meta in paired_metas]
                                if source_ids[0] == source_ids[1]:
                                    raise ValueError(
                                        f"{row.get('id')}: shuffled counterfactual needs distinct source_ids")
                                if None in families or families[0] != families[1]:
                                    raise ValueError(
                                        f"{row.get('id')}: shuffled counterfactual needs one shared family")
                            example = {"id": row["id"],
                                       "paired_views": [view[0] for view in views],
                                       "pair_mode": pair_mode,
                                       "pair_source_ids": source_ids}
                            if auxiliary:
                                example["auxiliary_views"] = auxiliary
                            examples.append(example)
                            continue
                        arms = [meta.get("s2_arm") for meta in paired_metas]
                        polarities = [meta.get("polarity") for meta in paired_metas]
                        if arms == [None, None]:
                            arms = ["A", "B"]
                        if polarities == [None, None]:
                            polarities = ["positive", "positive"]
                        if arms != ["A", "B"]:
                            raise ValueError(f"{row.get('id')}: paired views must be ordered A then B")
                        if polarities[0] not in ("positive", "negative") or polarities[1] != polarities[0]:
                            raise ValueError(f"{row.get('id')}: paired views need matching polarity")
                        example = {"id": row["id"],
                                   "paired_views": [view[0] for view in views],
                                   "paired_polarity": polarities[0],
                                   "paired_view_arms": arms}
                        if auxiliary:
                            example["auxiliary_views"] = auxiliary
                        examples.append(example)
                    else:
                        examples.extend(flatten_row(row, path.parent))
    return examples


def logical_slot_views(example):
    """Return canonical views followed by optional training-only auxiliary views."""
    action = example.get("action_views")
    if action is not None:
        return action
    paired = example.get("paired_views")
    if paired is None:
        return [example]
    auxiliary = example.get("auxiliary_views")
    return paired if not auxiliary else [*paired, *auxiliary]


def examples_have_images(examples):
    return any(view.get("image") for example in examples
               for view in logical_slot_views(example))


def validate_multimodal_examples(examples, microbatch):
    if not examples_have_images(examples):
        return
    if microbatch != 1:
        raise ValueError("multimodal training requires --microbatch 1")
    for example in examples:
        views = logical_slot_views(example)
        if any(view.get("image") for view in views) and len(views) != 1:
            raise ValueError(f"{example['id']}: image examples must be singleton logical slots")


def mean_logical_slot_loss(view_losses):
    """Average views within each logical slot, then average logical slots."""
    if not view_losses or any(not losses for losses in view_losses):
        raise ValueError("each logical slot needs at least one view loss")
    return torch.stack([torch.stack(losses).mean() for losses in view_losses]).mean()


def semantic_margin(option_scores, item_labels, polarity):
    """Polarity-aligned semantic margin, independent of option order."""
    if polarity not in ("positive", "negative"):
        raise ValueError(f"unsupported pair polarity: {polarity}")
    try:
        no_index, yes_index = item_labels.index("no"), item_labels.index("yes")
    except ValueError as error:
        raise ValueError("paired views must expose no/yes item labels") from error
    sign = 1.0 if polarity == "positive" else -1.0
    return sign * (option_scores[yes_index] - option_scores[no_index])


def semantic_gold_margin(option_scores, item_labels, gold_label):
    """Gold-minus-other score for a binary decision, independent of option order."""
    if len(item_labels) != 2 or len(set(item_labels)) != 2:
        raise ValueError("counterfactual margin requires exactly two option labels")
    try:
        gold_index = item_labels.index(gold_label)
    except ValueError as error:
        raise ValueError("counterfactual gold label is not an option") from error
    return option_scores[gold_index] - option_scores[1 - gold_index]


def pair_flip_term(margin_a, margin_b, margin):
    if margin <= 0:
        raise ValueError("pair flip margin must be positive")
    return F.relu(2 * margin - (margin_a + margin_b))


def pair_consistency_term(anchor_margin, view_b_margin, anchor_gold_probability,
                          min_gold_probability=0.80, clip=4.0, beta=1.0):
    """Return gated Smooth-L1 and whether anchor A passed the confidence gate."""
    if not 0 <= min_gold_probability <= 1 or clip <= 0 or beta <= 0:
        raise ValueError("invalid pair consistency configuration")
    eligible = bool(anchor_gold_probability.detach().item() >= min_gold_probability)
    if not eligible:
        return view_b_margin.new_zeros(()), False
    target = anchor_margin.detach().clamp(-clip, clip)
    return F.smooth_l1_loss(view_b_margin, target, beta=beta), True


def pair_slot_loss(view_losses, consistency, weight):
    """Mean logical-slot CE plus optional consistency; preserve the zero-weight path."""
    base = torch.stack(view_losses).mean()
    if weight == 0 or consistency is None:
        return base
    return base + weight * consistency


def validate_action_objective(action_loss_weight, action_temperature):
    if (isinstance(action_loss_weight, bool) or not isinstance(action_loss_weight, Real)
            or not math.isfinite(action_loss_weight)
            or action_loss_weight < 0):
        raise ValueError("action loss weight must be finite and nonnegative")
    if (isinstance(action_temperature, bool) or not isinstance(action_temperature, Real)
            or not math.isfinite(action_temperature)
            or action_temperature <= 0):
        raise ValueError("action temperature must be finite and positive")


def validate_exact_policy_outcome_objective(
        loss_kind, decision_threshold, action_temperature):
    if loss_kind not in {"nll", "brier"}:
        raise ValueError("exact-policy outcome loss must be 'nll' or 'brier'")
    if (loss_kind == "brier"
            and (decision_threshold != 0.5 or action_temperature != 1.0)):
        raise ValueError(
            "Brier calibration requires decision_threshold=0.5 and "
            "action_temperature=1.0")


def validate_action_training_mix(examples, action_loss_weight, pair_consistency_weight,
                                 pair_flip_weight, project_consistency_against_guard,
                                 stage_counterfactual_pairs):
    has_actions = any("action_views" in example for example in examples)
    if has_actions and (pair_consistency_weight > 0 or pair_flip_weight > 0
                        or project_consistency_against_guard or stage_counterfactual_pairs):
        raise ValueError("grouped action slots are incompatible with pair and guard objectives")
    if action_loss_weight > 0 and not has_actions:
        raise ValueError("positive action loss weight requires grouped action_v1 training rows")


def grouped_action_slot_loss(field_losses, field_scores, field_labels, gold_action,
                             decision_threshold, action_loss_weight, action_temperature):
    """Mean field CE plus optional executable action CE for one grouped row."""
    validate_action_objective(action_loss_weight, action_temperature)
    if not field_losses:
        raise ValueError("action group needs at least one field loss")
    field_loss = torch.stack(field_losses).mean()
    log_distribution = soft_action_log_distribution(
        field_scores, field_labels, decision_threshold=decision_threshold,
        temperature=action_temperature)
    action_loss = action_log_cross_entropy(log_distribution, gold_action)
    # Keep weight zero exactly on the grouped field-CE graph.
    loss = (field_loss if action_loss_weight == 0
            else field_loss + action_loss_weight * action_loss)
    return loss, log_distribution.exp(), action_loss


def grouped_exact_policy_slot_loss(
        field_losses, field_scores, field_labels, policy, gold_outcome,
        decision_threshold, action_loss_weight, action_temperature,
        exact_policy_outcome_loss="nll"):
    """Mean field CE plus a proper score over exact complete outcomes."""
    validate_action_objective(action_loss_weight, action_temperature)
    validate_exact_policy_outcome_objective(
        exact_policy_outcome_loss, decision_threshold, action_temperature)
    if not field_losses:
        raise ValueError("exact-policy group needs at least one field loss")
    field_loss = torch.stack(field_losses).mean()
    truth_logits = semantic_truth_logits(
        field_scores, field_labels, decision_threshold=decision_threshold,
        temperature=action_temperature)
    log_distribution = exact_outcome_log_distribution(policy, truth_logits)
    # Compute both experimental arms from the identical exact distribution so
    # NLL and Brier runs have matched executor work and differ only in score.
    exact_losses = {
        "nll": outcome_nll(log_distribution, gold_outcome),
        "brier": outcome_brier(log_distribution, gold_outcome),
    }
    exact_loss = exact_losses[exact_policy_outcome_loss]
    # Compute the complete outcome objective in both arms for matched compute,
    # while preserving the field-only graph and gradients exactly at weight 0.
    loss = (field_loss if action_loss_weight == 0
            else field_loss + action_loss_weight * exact_loss)
    return loss, {outcome: value.exp()
                  for outcome, value in log_distribution.items()}, exact_loss


def hard_action_from_scores(field_scores, field_labels, decision_threshold):
    """Replay serving's semantic threshold and Boolean executor for telemetry."""
    truths = {}
    for field, scores in field_scores.items():
        labels = field_labels[field]
        yes_probability = F.softmax(scores.detach().float(), dim=0)[labels.index("yes")]
        truths[field] = bool(yes_probability.item() >= decision_threshold)
    return action_for(truths)


def hard_exact_outcome_from_scores(field_scores, field_labels, policy,
                                   decision_threshold):
    """Threshold semantic scores and run the deterministic generic policy."""
    truths = {}
    for field, scores in field_scores.items():
        labels = field_labels[field]
        yes_probability = F.softmax(scores.detach().float(), dim=0)[labels.index("yes")]
        truths[field] = bool(yes_probability.item() >= decision_threshold)
    return execute_policy(policy, truths)


def pair_consistency_stats(active, seen):
    return {"pair_consistency_active": active,
            "pair_consistency_seen": seen,
            "pair_consistency_rate": active / seen if seen else 0.0}


def validate_pair_objectives(pair_consistency_weight, project_consistency_against_guard,
                             pair_flip_weight, pair_flip_margin, pair_flip_start_step=0,
                             steps=None):
    if pair_flip_weight < 0:
        raise ValueError("pair flip weight must be nonnegative")
    if pair_flip_margin <= 0:
        raise ValueError("pair flip margin must be positive")
    if pair_flip_start_step < 0:
        raise ValueError("pair flip start step must be nonnegative")
    if steps is not None and pair_flip_weight > 0 and pair_flip_start_step > steps:
        raise ValueError("pair flip start step cannot exceed training steps")
    if pair_flip_weight > 0 and (pair_consistency_weight > 0 or
                                 project_consistency_against_guard):
        raise ValueError("pair flip is incompatible with pair consistency and guard projection")


def pair_flip_weight_at_step(weight, start_step, step):
    return weight if step >= start_step else 0.0


def staged_counterfactual_order(examples, steps, microbatch, grad_accum, start_step):
    logical_batch = microbatch * grad_accum
    if logical_batch <= 0 or not 1 <= start_step <= steps:
        raise ValueError("staged counterfactual schedule needs positive batches and a valid start step")
    singles = [index for index, example in enumerate(examples)
               if "paired_views" not in example]
    pairs = [index for index, example in enumerate(examples)
             if example.get("pair_mode") in ("counterfactual", "shuffled_counterfactual")]
    if len(singles) + len(pairs) != len(examples):
        raise ValueError("staged counterfactual schedule only supports singles and counterfactual pairs")
    expected_singles = (start_step - 1) * logical_batch
    expected_pairs = (steps - start_step + 1) * logical_batch
    if len(singles) != expected_singles or len(pairs) != expected_pairs:
        raise ValueError(
            f"staged counterfactual counts need singles={expected_singles}, pairs={expected_pairs}; "
            f"got singles={len(singles)}, pairs={len(pairs)}")
    if len(examples) != steps * logical_batch:
        raise ValueError("staged counterfactual schedule must cover exactly one epoch")
    random.shuffle(singles)
    random.shuffle(pairs)
    return singles + pairs


def guard_logical_slot_loss(view_losses, paired_view_counts=None):
    """Mean slot loss, omitting canonical B while preserving every other CE coefficient."""
    if not view_losses or any(not losses for losses in view_losses):
        raise ValueError("each logical slot needs at least one view loss")
    if paired_view_counts is None:
        paired_view_counts = [2 if len(losses) == 2 else 1 for losses in view_losses]
    if len(paired_view_counts) != len(view_losses):
        raise ValueError("paired-view counts must align with logical slots")
    slots = []
    for losses, canonical_count in zip(view_losses, paired_view_counts):
        if canonical_count not in (1, 2) or len(losses) < canonical_count:
            raise ValueError("each slot needs one singleton or two canonical paired views")
        if canonical_count == 1:
            if len(losses) != 1:
                raise ValueError("singleton logical slots cannot have auxiliary views")
            slots.append(losses[0])
        else:
            # D averages A, B, and every auxiliary view. The guard drops only B and
            # keeps the same denominator, so every retained CE coefficient is unchanged.
            slots.append(torch.stack([losses[0], *losses[2:]]).sum() / len(losses))
    return torch.stack(slots).mean()


def view_role_stats(canonical_correct, canonical_count, canonical_ce_sum,
                    auxiliary_correct, auxiliary_count, auxiliary_ce_sum):
    return {
        "canonical_views_seen": canonical_count,
        "canonical_views_correct": canonical_correct,
        "canonical_view_accuracy": (canonical_correct / canonical_count
                                    if canonical_count else 0.0),
        "canonical_ce_sum": canonical_ce_sum,
        "canonical_ce": canonical_ce_sum / canonical_count if canonical_count else 0.0,
        "auxiliary_views_seen": auxiliary_count,
        "auxiliary_views_correct": auxiliary_correct,
        "auxiliary_view_accuracy": (auxiliary_correct / auxiliary_count
                                    if auxiliary_count else 0.0),
        "auxiliary_ce_sum": auxiliary_ce_sum,
        "auxiliary_ce": auxiliary_ce_sum / auxiliary_count if auxiliary_count else 0.0,
    }


def accumulate_view_stats_(totals, stats):
    for key in ("canonical_views_seen", "canonical_views_correct", "canonical_ce_sum",
                "auxiliary_views_seen", "auxiliary_views_correct", "auxiliary_ce_sum"):
        totals[key] += stats[key]


def project_conflicting_gradient(cons_grads, guard_grads, params=None, eps=1e-12):
    """Globally project consistency gradient off guard when their dot is negative."""
    if len(cons_grads) != len(guard_grads):
        raise ValueError("gradient lists must be aligned")
    params = params or [c if c is not None else g for c, g in zip(cons_grads, guard_grads)]
    if len(params) != len(cons_grads) or any(p is None for p in params):
        raise ValueError("a parameter reference is required for every gradient slot")
    cons = [torch.zeros_like(p, dtype=torch.float32) if g is None else g.float()
            for p, g in zip(params, cons_grads)]
    guard = [torch.zeros_like(p, dtype=torch.float32) if g is None else g.float()
             for p, g in zip(params, guard_grads)]
    dot = sum((a * b).sum() for a, b in zip(cons, guard))
    cons_energy = sum((a * a).sum() for a in cons)
    guard_energy = sum((b * b).sum() for b in guard)
    conflict = bool(dot.item() < 0)
    if conflict:
        coefficient = dot / (guard_energy + eps)
        projected = [a - coefficient * b for a, b in zip(cons, guard)]
    else:
        projected = cons
    removed_energy = sum(((a - b) ** 2).sum() for a, b in zip(cons, projected))
    removed_fraction = (removed_energy / cons_energy if cons_energy.item() > 0
                        else cons_energy.new_zeros(()))
    cosine = dot / (torch.sqrt(cons_energy * guard_energy) + eps)
    stats = {"projection_guard_cons_dot": float(dot.item()),
             "projection_guard_cons_cosine": float(cosine.item()),
             "projection_conflict": conflict,
             "projection_cons_energy": float(cons_energy.item()),
             "projection_removed_energy": float(removed_energy.item()),
             "projection_removed_energy_fraction": float(removed_fraction.item())}
    return projected, stats


def projection_early_stop(step, cumulative_removed_energy, cumulative_cons_energy):
    return step == 50 and cumulative_removed_energy / max(cumulative_cons_energy, 1e-30) < 0.01


def accumulate_dense_gradients_(buffers, grads, objective):
    """Accumulate one dense objective or fail before zero grads change optimizer semantics."""
    if len(buffers) != len(grads) or any(grad is None for grad in grads):
        missing = sum(grad is None for grad in grads)
        raise RuntimeError(f"{objective} did not reach {missing} trainable parameters")
    for buffer, grad in zip(buffers, grads):
        buffer.add_(grad.detach().float())


def backward_into_dense_gradients_(loss, params, buffers, objective, retain_graph=False):
    """Copy one backward pass into dense buffers, then release parameter grads."""
    for parameter in params:
        parameter.grad = None
    loss.backward(retain_graph=retain_graph)
    accumulate_dense_gradients_(buffers, [parameter.grad for parameter in params], objective)
    for parameter in params:
        parameter.grad = None


def render(example, order=None):
    order = list(range(len(example["items"]))) if order is None else list(order)
    state = example["state"]
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    instructions = example["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    lines = [state.rstrip(), "", instructions.rstrip(), "", "Options:"]
    for output_index, source_index in enumerate(order):
        lines.append(f"{LETTERS[output_index]}. {example['items'][source_index][1]}")
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    probabilities = [example["probabilities"][source_index] for source_index in order]
    return "\n".join(lines), probabilities


def letter_token_groups(tokenizer):
    # vLLM's structured `choice` permits the exact uppercase strings. Include
    # every vocabulary token that decodes to that string (Gemma has a direct
    # token and a byte fallback), but not lowercase or leading-space aliases
    # that serving masks out.
    groups = {letter: [] for letter in LETTERS}
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        if text in groups:
            groups[text].append(token_id)
    if any(not groups[letter] for letter in LETTERS):
        raise ValueError("tokenizer cannot represent every option letter")
    return [torch.tensor(groups[letter]) for letter in LETTERS]


def encode(example, tokenizer, max_length, shuffle, include_labels=False):
    order = list(range(len(example["items"])))
    if shuffle and not example.get("fixed_order", False):
        random.shuffle(order)
    prompt, probabilities = render(example, order)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(ids) > max_length:
        raise ValueError(f"{example['id']}: {len(ids)} tokens exceeds {max_length}")
    encoded = (ids, torch.tensor(probabilities))
    if include_labels:
        labels = [example["items"][source_index][0] for source_index in order]
        return (*encoded, labels)
    return encoded


def encode_image(example, processor, max_length, shuffle, include_labels=False):
    order = list(range(len(example["items"])))
    if shuffle and not example.get("fixed_order", False):
        random.shuffle(order)
    prompt, probabilities = render(example, order)
    text = processor.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": [
             {"type": "image"}, {"type": "text", "text": prompt},
         ]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    with Image.open(example["image"]) as image:
        inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
    if inputs["input_ids"].shape[-1] > max_length:
        raise ValueError(f"{example['id']}: {inputs['input_ids'].shape[-1]} tokens exceeds {max_length}")
    encoded = (inputs, torch.tensor(probabilities))
    if include_labels:
        labels = [example["items"][source_index][0] for source_index in order]
        return (*encoded, labels)
    return encoded


def loss_for_examples(model, examples, tokenizer, groups, max_length, device, shuffle,
                      pair_consistency_weight=0.0, pair_consistency_min_gold_probability=0.80,
                      pair_consistency_clip=4.0, pair_consistency_beta=1.0,
                      pair_flip_weight=0.0, pair_flip_margin=1.0,
                      return_components=False, processor=None, action_loss_weight=0.0,
                      action_temperature=1.0, exact_policy_outcome_loss="nll"):
    validate_pair_objectives(pair_consistency_weight, return_components,
                             pair_flip_weight, pair_flip_margin)
    validate_action_objective(action_loss_weight, action_temperature)
    if exact_policy_outcome_loss not in {"nll", "brier"}:
        raise ValueError("exact-policy outcome loss must be 'nll' or 'brier'")
    action_examples = [example for example in examples if "action_views" in example]
    if action_examples and (pair_consistency_weight > 0 or pair_flip_weight > 0
                            or return_components):
        raise ValueError("grouped action slots are incompatible with pair and guard objectives")
    logical_views = [logical_slot_views(example) for example in examples]
    flat_examples = [view for views in logical_views for view in views]
    image_examples = [example for example in flat_examples if example.get("image")]
    if image_examples:
        if len(examples) != 1 or len(flat_examples) != 1:
            raise ValueError("image examples must be singleton logical slots with --microbatch 1")
        if processor is None:
            raise ValueError("image example requires a multimodal processor")
        encoded = [encode_image(flat_examples[0], processor, max_length, shuffle,
                                include_labels=True)]
        model_inputs = {name: value.to(device) for name, value in encoded[0][0].items()}
        positions = torch.tensor([model_inputs["input_ids"].shape[-1] - 1])
    else:
        encoded = [encode(example, tokenizer, max_length, shuffle, include_labels=True)
                   for example in flat_examples]
        ids = pad_sequence([item[0] for item in encoded], batch_first=True,
                           padding_value=tokenizer.pad_token_id)
        mask = ids.ne(tokenizer.pad_token_id)
        positions = torch.tensor([len(item[0]) - 1 for item in encoded])
        model_inputs = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
    union = torch.unique(positions, sorted=True)
    output = model(**model_inputs, use_cache=False, logits_to_keep=union.to(device))
    offsets = {position.item(): index for index, position in enumerate(union)}
    view_losses = []
    view_correct = []
    view_scores = []
    view_gold_probabilities = []
    for batch_index, ((_, probabilities, labels), position, example) in enumerate(
            zip(encoded, positions, flat_examples)):
        logits = output.logits[batch_index, offsets[position.item()]].float()
        option_scores = torch.stack([
            torch.logsumexp(logits[token_ids.to(device)], dim=0)
            for token_ids in groups[:len(probabilities)]
        ])
        view_losses.append(-(probabilities.to(device) * F.log_softmax(option_scores, dim=0)).sum())
        view_correct.append(float(option_scores.argmax().item() == probabilities.argmax().item()))
        view_scores.append((option_scores, labels))
        gold_index = labels.index(example["gold_label"])
        view_gold_probabilities.append(F.softmax(option_scores.detach(), dim=0)[gold_index])
    slot_losses = []
    guard_slot_losses = []
    guard_paired_view_counts = []
    consistency_slot_losses = []
    slot_correct = []
    canonical_views_seen = auxiliary_views_seen = 0
    canonical_views_correct = auxiliary_views_correct = 0.0
    canonical_ce_losses = []
    auxiliary_ce_losses = []
    consistency_seen = 0
    consistency_active = 0
    flip_losses = []
    flip_active = 0
    action_losses = []
    action_loss_kinds = []
    action_correct = 0
    cursor = 0
    for logical_example, views in zip(examples, logical_views):
        end = cursor + len(views)
        slot_view_losses = view_losses[cursor:end]
        canonical_count = (len(views) if "action_views" in logical_example
                           else 2 if "paired_views" in logical_example else 1)
        auxiliary_count = len(views) - canonical_count
        canonical_views_seen += canonical_count
        canonical_views_correct += sum(view_correct[cursor:cursor + canonical_count])
        canonical_ce_losses.extend(slot_view_losses[:canonical_count])
        auxiliary_views_seen += auxiliary_count
        auxiliary_views_correct += sum(view_correct[cursor + canonical_count:end])
        auxiliary_ce_losses.extend(slot_view_losses[canonical_count:])
        if "action_views" in logical_example:
            qids = logical_example["action_qids"]
            if len(qids) != len(views) or set(qids) != set(logical_example["action_qid_to_field"]):
                raise ValueError(f"{logical_example['id']}: malformed action field mapping")
            field_scores = {}
            field_labels = {}
            for local_index, qid in enumerate(qids):
                field = logical_example["action_qid_to_field"][qid]
                scores, labels = view_scores[cursor + local_index]
                field_scores[field] = scores
                field_labels[field] = labels
            if logical_example.get("action_kind") == "exact_policy_v1":
                action_slot, _, action_loss = grouped_exact_policy_slot_loss(
                    slot_view_losses, field_scores, field_labels,
                    logical_example["compiled_policy"], logical_example["gold_outcome"],
                    logical_example["decision_threshold"], action_loss_weight,
                    action_temperature, exact_policy_outcome_loss)
                predicted_action = hard_exact_outcome_from_scores(
                    field_scores, field_labels, logical_example["compiled_policy"],
                    logical_example["decision_threshold"])
                action_correct += int(predicted_action == logical_example["gold_outcome"])
                action_loss_kinds.append(exact_policy_outcome_loss)
            else:
                action_slot, _, action_loss = grouped_action_slot_loss(
                    slot_view_losses, field_scores, field_labels,
                    logical_example["action_gold"], logical_example["decision_threshold"],
                    action_loss_weight, action_temperature)
                predicted_action = hard_action_from_scores(
                    field_scores, field_labels, logical_example["decision_threshold"])
                action_correct += int(predicted_action == logical_example["action_gold"])
                action_loss_kinds.append("nll")
            slot_losses.append([action_slot])
            action_losses.append(action_loss)
        elif logical_example.get("pair_mode") is not None:
            if len(logical_example.get("paired_views", [])) != 2:
                raise ValueError(f"{logical_example['id']}: counterfactual pair needs two views")
            # The canonical final-decision pair is always first. Auxiliary views affect
            # only the slot CE average and can never become pair-flip endpoints.
            margin_a = semantic_gold_margin(
                *view_scores[cursor], flat_examples[cursor]["gold_label"])
            margin_b = semantic_gold_margin(
                *view_scores[cursor + 1], flat_examples[cursor + 1]["gold_label"])
            flip = pair_flip_term(margin_a, margin_b, pair_flip_margin)
            flip_losses.append(flip)
            flip_active += int(flip.detach().item() > 0)
            if pair_flip_weight > 0:
                slot_losses.append([pair_slot_loss(slot_view_losses, flip, pair_flip_weight)])
            else:
                # Preserve the exact CE graph while still running both matched-compute views.
                slot_losses.append(slot_view_losses)
            if return_components:
                raise ValueError("counterfactual pair flip is incompatible with guard projection")
        elif ("paired_views" in logical_example and
              (pair_consistency_weight > 0 or return_components)):
            if logical_example.get("paired_view_arms") != ["A", "B"]:
                raise ValueError(f"{logical_example['id']}: paired views must be ordered A then B")
            polarity = logical_example["paired_polarity"]
            margin_a = semantic_margin(*view_scores[cursor], polarity)
            margin_b = semantic_margin(*view_scores[cursor + 1], polarity)
            consistency, eligible = pair_consistency_term(
                margin_a, margin_b, view_gold_probabilities[cursor],
                pair_consistency_min_gold_probability, pair_consistency_clip,
                pair_consistency_beta,
            )
            consistency_seen += 1
            if eligible:
                consistency_active += 1
            else:
                consistency = None
            if return_components:
                slot_losses.append(slot_view_losses)
                guard_slot_losses.append(slot_view_losses)
                guard_paired_view_counts.append(2)
                consistency_slot_losses.append(
                    consistency if consistency is not None else slot_view_losses[0].new_zeros(())
                )
            else:
                slot_losses.append([pair_slot_loss(slot_view_losses, consistency,
                                                   pair_consistency_weight)])
        else:
            # Preserve the existing CE graph and exact aggregation when disabled.
            slot_losses.append(slot_view_losses)
            if return_components:
                guard_slot_losses.append(slot_view_losses)
                guard_paired_view_counts.append(canonical_count)
                consistency_slot_losses.append(slot_view_losses[0].new_zeros(()))
        slot_correct.append(sum(view_correct[cursor:end]) / len(views))
        cursor = end
    stats = pair_consistency_stats(consistency_active, consistency_seen)
    stats.update({"pair_flip_seen": len(flip_losses),
                  "pair_flip_active": flip_active,
                  "pair_flip_loss": (sum(loss.detach().item() for loss in flip_losses) /
                                     len(flip_losses) if flip_losses else 0.0)})
    if action_losses:
        mean_action_loss = (sum(item.detach().item() for item in action_losses) /
                            len(action_losses))
        stats.update({"action_groups_seen": len(action_losses),
                      "action_groups_correct": action_correct,
                      "action_accuracy": action_correct / len(action_losses),
                      "action_loss": mean_action_loss})
        loss_kinds = set(action_loss_kinds)
        stats["action_loss_kind"] = (next(iter(loss_kinds))
                                     if len(loss_kinds) == 1 else "mixed")
        if loss_kinds == {"nll"}:
            stats["action_ce"] = mean_action_loss
        elif loss_kinds == {"brier"}:
            stats["action_brier"] = mean_action_loss
        if any(example.get("action_kind") == "exact_policy_v1"
               for example in action_examples):
            stats["exact_policy_outcome_loss"] = exact_policy_outcome_loss
    canonical_ce_sum = torch.stack(canonical_ce_losses).detach().sum().item()
    auxiliary_ce_sum = (torch.stack(auxiliary_ce_losses).detach().sum().item()
                        if auxiliary_ce_losses else 0.0)
    stats.update(view_role_stats(
        canonical_views_correct, canonical_views_seen, canonical_ce_sum,
        auxiliary_views_correct, auxiliary_views_seen, auxiliary_ce_sum))
    loss = mean_logical_slot_loss(slot_losses)
    if return_components:
        components = {"d_loss": loss,
                      "guard_loss": guard_logical_slot_loss(
                          guard_slot_losses, guard_paired_view_counts),
                      "consistency_loss": torch.stack(consistency_slot_losses).mean()}
        return loss, sum(slot_correct), stats, components
    return loss, sum(slot_correct), stats


@torch.no_grad()
def evaluate(model, examples, args, tokenizer, groups, device, processor=None):
    model.eval()
    losses = []
    correct = 0
    count = 0
    view_totals = {"canonical_views_seen": 0, "canonical_views_correct": 0.0,
                   "canonical_ce_sum": 0.0, "auxiliary_views_seen": 0,
                   "auxiliary_views_correct": 0.0, "auxiliary_ce_sum": 0.0}
    action_seen = action_correct = 0
    action_loss_sum = 0.0
    action_loss_kinds = set()
    for start in range(0, min(args.eval_samples, len(examples)), args.microbatch):
        batch = examples[start:start + args.microbatch]
        loss, batch_correct, stats = loss_for_examples(
            model, batch, tokenizer, groups, args.max_length, device, shuffle=False,
            processor=processor,
            action_loss_weight=getattr(args, "action_loss_weight", 0.0),
            action_temperature=getattr(args, "action_temperature", 1.0),
            exact_policy_outcome_loss=getattr(
                args, "exact_policy_outcome_loss", "nll"))
        losses.append(loss.item())
        correct += batch_correct
        count += len(batch)
        accumulate_view_stats_(view_totals, stats)
        action_seen += stats.get("action_groups_seen", 0)
        action_correct += stats.get("action_groups_correct", 0)
        action_loss_sum += stats.get(
            "action_loss", stats.get("action_ce", 0.0)
        ) * stats.get("action_groups_seen", 0)
        if stats.get("action_groups_seen", 0):
            action_loss_kinds.add(stats.get("action_loss_kind", "nll"))
    model.train()
    objective_loss = sum(losses) / len(losses)
    result = {"objective_loss": objective_loss, "accuracy": correct / count, "n": count}
    # Preserve the established key for CE/NLL runs, but do not mislabel an
    # objective containing a Brier term as negative log likelihood.
    if "brier" not in action_loss_kinds:
        result["nll"] = objective_loss
    result.update(view_role_stats(
        view_totals["canonical_views_correct"], view_totals["canonical_views_seen"],
        view_totals["canonical_ce_sum"], view_totals["auxiliary_views_correct"],
        view_totals["auxiliary_views_seen"], view_totals["auxiliary_ce_sum"]))
    if action_seen:
        result.update({"action_groups_seen": action_seen,
                       "action_groups_correct": action_correct,
                       "action_accuracy": action_correct / action_seen,
                       "action_loss": action_loss_sum / action_seen,
                       "action_loss_kind": (next(iter(action_loss_kinds))
                                            if len(action_loss_kinds) == 1 else "mixed")})
        if action_loss_kinds == {"nll"}:
            result["action_ce"] = action_loss_sum / action_seen
        elif action_loss_kinds == {"brier"}:
            result["action_brier"] = action_loss_sum / action_seen
    return result


def save(model, optimizer, scheduler, step, out):
    path = out / f"step-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
               path / "trainer.pt")


def load_tokenizer_and_processor(model_name, multimodal):
    if multimodal:
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        return processor.tokenizer, processor
    return AutoTokenizer.from_pretrained(model_name, local_files_only=True), None


def attach_lora(model, init_adapter, rank, alpha):
    if init_adapter:
        return PeftModel.from_pretrained(
            model, init_adapter, is_trainable=True, local_files_only=True)
    return get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES,
    ))


def validate_trainable_path(model):
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    bad = [name for name in trainable
           if any(part in name for part in ("vision", "visual", "projector", "embedder"))]
    if bad:
        raise RuntimeError(f"LoRA unexpectedly touched the multimodal path: {bad[:3]}")
    if not any("language_model" in name for name in trainable):
        raise RuntimeError("LoRA has no trainable language-model parameters")


def main():
    args = parse_args()
    explicit_save_steps = parse_save_steps(args.save_steps, args.steps)
    validate_action_objective(args.action_loss_weight, args.action_temperature)
    if args.exact_policy_outcome_loss not in {"nll", "brier"}:
        raise ValueError("exact-policy outcome loss must be 'nll' or 'brier'")
    if args.pair_consistency_weight < 0:
        raise ValueError("pair consistency weight must be nonnegative")
    if not 0 <= args.pair_consistency_min_gold_probability <= 1:
        raise ValueError("pair consistency minimum gold probability must be in [0, 1]")
    if args.pair_consistency_clip <= 0 or args.pair_consistency_beta <= 0:
        raise ValueError("pair consistency clip and beta must be positive")
    if args.project_consistency_against_guard and args.pair_consistency_weight <= 0:
        raise ValueError("guard projection requires a positive pair consistency weight")
    validate_pair_objectives(args.pair_consistency_weight,
                             args.project_consistency_against_guard,
                             args.pair_flip_weight, args.pair_flip_margin,
                             args.pair_flip_start_step, args.steps)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_examples(args.train)
    validate_action_training_mix(
        train, args.action_loss_weight, args.pair_consistency_weight,
        args.pair_flip_weight, args.project_consistency_against_guard,
        args.stage_counterfactual_pairs)
    has_action_training = any("action_views" in example for example in train)
    validation = {name: load_examples([path]) for name, path in
                  (item.split("=", 1) for item in args.val)}
    validate_multimodal_examples(train, args.microbatch)
    for rows in validation.values():
        validate_multimodal_examples(rows, args.microbatch)
    staged_order = (staged_counterfactual_order(
        train, args.steps, args.microbatch, args.grad_accum, args.pair_flip_start_step)
        if args.stage_counterfactual_pairs else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    multimodal = examples_have_images(train) or any(
        examples_have_images(rows) for rows in validation.values())
    tokenizer, processor = load_tokenizer_and_processor(args.model, multimodal)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = attach_lora(model, args.init_adapter, args.rank, args.alpha)
    validate_trainable_path(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    device = torch.device("cuda")
    model.to(device).train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"examples": len(train), "trainable_parameters": trainable,
                      "total_parameters": total, "trainable_fraction": trainable / total}), flush=True)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay, fused=True,
    )
    warmup = max(1, round(args.steps * 0.03))
    floor = args.min_lr / args.lr

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    if staged_order is None:
        order = list(range(len(train)))
        random.shuffle(order)
    else:
        order = staged_order
    cursor = 0
    optimizer.zero_grad(set_to_none=True)
    metrics = out / "metrics.jsonl"
    started = time.time()
    baseline = {"step": 0, "validation": {
        name: evaluate(model, rows, args, tokenizer, groups, device, processor)
        for name, rows in validation.items()
    }}
    metrics.write_text(json.dumps(baseline) + "\n")
    print(json.dumps(baseline), flush=True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.project_consistency_against_guard:
        grad_d = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        grad_guard = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        grad_cons = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    projection_conflicts = 0
    projection_active_steps = 0
    cumulative_removed_energy = 0.0
    cumulative_cons_energy = 0.0
    staged_singles_seen = staged_pairs_seen = 0
    for step in range(1, args.steps + 1):
        accumulated = 0.0
        flip_weight = pair_flip_weight_at_step(
            args.pair_flip_weight, args.pair_flip_start_step, step)
        pair_active = pair_seen = 0
        flip_active = flip_seen = 0
        flip_loss_sum = 0.0
        action_groups_seen = action_groups_correct = 0
        action_loss_sum = 0.0
        action_loss_kinds = set()
        view_totals = {"canonical_views_seen": 0, "canonical_views_correct": 0.0,
                       "canonical_ce_sum": 0.0, "auxiliary_views_seen": 0,
                       "auxiliary_views_correct": 0.0, "auxiliary_ce_sum": 0.0}
        staged_step_singles = staged_step_pairs = 0
        if args.project_consistency_against_guard:
            for buffers in (grad_d, grad_guard, grad_cons):
                for buffer in buffers:
                    buffer.zero_()
            for _ in range(args.grad_accum):
                if cursor + args.microbatch > len(order):
                    if args.stage_counterfactual_pairs:
                        raise RuntimeError("staged counterfactual schedule exhausted early")
                    random.shuffle(order)
                    cursor = 0
                batch = [train[index] for index in order[cursor:cursor + args.microbatch]]
                cursor += args.microbatch
                if args.stage_counterfactual_pairs:
                    batch_pairs = sum(example.get("pair_mode") is not None for example in batch)
                    staged_step_pairs += batch_pairs
                    staged_step_singles += len(batch) - batch_pairs
                _, _, pair_stats, components = loss_for_examples(
                    model, batch, tokenizer, groups, args.max_length, device, shuffle=True,
                    pair_consistency_weight=args.pair_consistency_weight,
                    pair_consistency_min_gold_probability=args.pair_consistency_min_gold_probability,
                    pair_consistency_clip=args.pair_consistency_clip,
                    pair_consistency_beta=args.pair_consistency_beta,
                    pair_flip_weight=flip_weight,
                    pair_flip_margin=args.pair_flip_margin,
                    return_components=True, processor=processor,
                    action_loss_weight=args.action_loss_weight,
                    action_temperature=args.action_temperature,
                    exact_policy_outcome_loss=args.exact_policy_outcome_loss)
                pair_active += pair_stats["pair_consistency_active"]
                pair_seen += pair_stats["pair_consistency_seen"]
                flip_active += pair_stats["pair_flip_active"]
                flip_seen += pair_stats["pair_flip_seen"]
                flip_loss_sum += pair_stats["pair_flip_loss"] * pair_stats["pair_flip_seen"]
                action_groups_seen += pair_stats.get("action_groups_seen", 0)
                action_groups_correct += pair_stats.get("action_groups_correct", 0)
                action_loss_sum += (pair_stats.get(
                    "action_loss", pair_stats.get("action_ce", 0.0))
                    * pair_stats.get("action_groups_seen", 0))
                if pair_stats.get("action_groups_seen", 0):
                    action_loss_kinds.add(pair_stats.get("action_loss_kind", "nll"))
                accumulate_view_stats_(view_totals, pair_stats)
                scale = 1.0 / args.grad_accum
                backward_into_dense_gradients_(
                    components["d_loss"] * scale, params, grad_d, "D loss", retain_graph=True)
                backward_into_dense_gradients_(
                    components["guard_loss"] * scale, params, grad_guard, "guard loss",
                    retain_graph=components["consistency_loss"].requires_grad)
                if components["consistency_loss"].requires_grad:
                    backward_into_dense_gradients_(
                        components["consistency_loss"] * scale, params, grad_cons,
                        "consistency loss")
                accumulated += (components["d_loss"].item() +
                                args.pair_consistency_weight * components["consistency_loss"].item())
                del components
            if pair_active:
                projected_cons, projection_stats = project_conflicting_gradient(
                    grad_cons, grad_guard, params=params)
            else:
                projected_cons = grad_cons
                projection_stats = {
                    "projection_guard_cons_dot": 0.0,
                    "projection_guard_cons_cosine": 0.0,
                    "projection_conflict": False,
                    "projection_cons_energy": 0.0,
                    "projection_removed_energy": 0.0,
                    "projection_removed_energy_fraction": 0.0,
                }
            for parameter, d_grad, cons_grad in zip(params, grad_d, projected_cons):
                if pair_active:
                    parameter.grad = (d_grad + args.pair_consistency_weight * cons_grad).to(parameter.dtype)
                else:
                    parameter.grad = d_grad.to(parameter.dtype)
            del projected_cons, d_grad, cons_grad
            if projection_stats["projection_cons_energy"] > 0:
                projection_active_steps += 1
                projection_conflicts += int(projection_stats["projection_conflict"])
                cumulative_cons_energy += projection_stats["projection_cons_energy"]
                cumulative_removed_energy += projection_stats["projection_removed_energy"]
            projection_stats["projection_cumulative_conflict_fraction"] = (
                projection_conflicts / projection_active_steps if projection_active_steps else 0.0)
            projection_stats["projection_cumulative_removed_energy_fraction"] = (
                cumulative_removed_energy / cumulative_cons_energy if cumulative_cons_energy else 0.0)
        else:
            projection_stats = None
            for _ in range(args.grad_accum):
                if cursor + args.microbatch > len(order):
                    if args.stage_counterfactual_pairs:
                        raise RuntimeError("staged counterfactual schedule exhausted early")
                    random.shuffle(order)
                    cursor = 0
                batch = [train[index] for index in order[cursor:cursor + args.microbatch]]
                cursor += args.microbatch
                if args.stage_counterfactual_pairs:
                    batch_pairs = sum(example.get("pair_mode") is not None for example in batch)
                    staged_step_pairs += batch_pairs
                    staged_step_singles += len(batch) - batch_pairs
                loss, _, pair_stats = loss_for_examples(
                    model, batch, tokenizer, groups, args.max_length, device, shuffle=True,
                    pair_consistency_weight=args.pair_consistency_weight,
                    pair_consistency_min_gold_probability=args.pair_consistency_min_gold_probability,
                    pair_consistency_clip=args.pair_consistency_clip,
                    pair_consistency_beta=args.pair_consistency_beta,
                    pair_flip_weight=flip_weight,
                    pair_flip_margin=args.pair_flip_margin, processor=processor,
                    action_loss_weight=args.action_loss_weight,
                    action_temperature=args.action_temperature,
                    exact_policy_outcome_loss=args.exact_policy_outcome_loss)
                pair_active += pair_stats["pair_consistency_active"]
                pair_seen += pair_stats["pair_consistency_seen"]
                flip_active += pair_stats["pair_flip_active"]
                flip_seen += pair_stats["pair_flip_seen"]
                flip_loss_sum += pair_stats["pair_flip_loss"] * pair_stats["pair_flip_seen"]
                action_groups_seen += pair_stats.get("action_groups_seen", 0)
                action_groups_correct += pair_stats.get("action_groups_correct", 0)
                action_loss_sum += (pair_stats.get(
                    "action_loss", pair_stats.get("action_ce", 0.0))
                    * pair_stats.get("action_groups_seen", 0))
                if pair_stats.get("action_groups_seen", 0):
                    action_loss_kinds.add(pair_stats.get("action_loss_kind", "nll"))
                accumulate_view_stats_(view_totals, pair_stats)
                (loss / args.grad_accum).backward()
                accumulated += loss.item()
        if args.stage_counterfactual_pairs:
            logical_batch = args.microbatch * args.grad_accum
            expected_pairs = logical_batch if step >= args.pair_flip_start_step else 0
            if (staged_step_pairs != expected_pairs or
                    staged_step_singles != logical_batch - expected_pairs):
                raise RuntimeError(f"staged counterfactual exposure mismatch at step {step}")
            staged_singles_seen += staged_step_singles
            staged_pairs_seen += staged_step_pairs
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        metric = {"step": step, "train_loss": accumulated / args.grad_accum,
                  "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
                  "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                  "allocated_vram_gb": torch.cuda.memory_allocated() / 2**30,
                  "reserved_vram_gb": torch.cuda.memory_reserved() / 2**30,
                  "pair_consistency_active": pair_active,
                  "pair_consistency_seen": pair_seen,
                  "pair_consistency_rate": pair_active / pair_seen if pair_seen else 0.0,
                  "pair_flip_active": flip_active,
                  "pair_flip_seen": flip_seen,
                  "pair_flip_loss": flip_loss_sum / flip_seen if flip_seen else 0.0,
                  "pair_flip_weight": flip_weight,
                  "pair_flip_weight_configured": args.pair_flip_weight}
        if has_action_training:
            metric.update({"action_groups_seen": action_groups_seen,
                           "action_groups_correct": action_groups_correct,
                           "action_accuracy": (action_groups_correct / action_groups_seen
                                               if action_groups_seen else 0.0),
                           "action_loss": (action_loss_sum / action_groups_seen
                                           if action_groups_seen else 0.0),
                           "action_loss_kind": (
                               next(iter(action_loss_kinds))
                               if len(action_loss_kinds) == 1 else "mixed"),
                           "action_loss_weight": args.action_loss_weight,
                           "action_temperature": args.action_temperature,
                           "exact_policy_outcome_loss":
                               args.exact_policy_outcome_loss})
            if action_loss_kinds == {"nll"}:
                metric["action_ce"] = metric["action_loss"]
            elif action_loss_kinds == {"brier"}:
                metric["action_brier"] = metric["action_loss"]
        metric.update(view_role_stats(
            view_totals["canonical_views_correct"], view_totals["canonical_views_seen"],
            view_totals["canonical_ce_sum"], view_totals["auxiliary_views_correct"],
            view_totals["auxiliary_views_seen"], view_totals["auxiliary_ce_sum"]))
        if args.stage_counterfactual_pairs:
            metric.update({"staged_single_slots": staged_step_singles,
                           "staged_counterfactual_slots": staged_step_pairs,
                           "staged_singles_seen": staged_singles_seen,
                           "staged_counterfactual_pairs_seen": staged_pairs_seen})
        early_stop = False
        if projection_stats is not None:
            metric.update(projection_stats)
            early_stop = projection_early_stop(
                step, cumulative_removed_energy, cumulative_cons_energy)
            if early_stop:
                metric["early_stop"] = "projection_removed_energy_fraction_below_0.01_at_step_50"
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metric["validation"] = {
                name: evaluate(model, rows, args, tokenizer, groups, device, processor)
                for name, rows in validation.items()
            }
        with metrics.open("a") as target:
            target.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if (step % args.save_every == 0 or step in explicit_save_steps
                or step == args.steps or early_stop):
            save(model, optimizer, scheduler, step, out)
        if early_stop:
            return


if __name__ == "__main__":
    main()
