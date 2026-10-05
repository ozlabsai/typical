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
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)
TARGET_MODULES = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


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
    return parser.parse_args()


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


def load_examples(paths):
    examples = []
    for path in paths:
        path = Path(path)
        with path.open() as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if "paired_views" in row:
                        if len(row["paired_views"]) != 2:
                            raise ValueError(f"{row.get('id')}: paired slot must have exactly two views")
                        views = [flatten_row(view, path.parent) for view in row["paired_views"]]
                        if any(len(view) != 1 for view in views):
                            raise ValueError(f"{row.get('id')}: each paired view must flatten to one example")
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
                            examples.append({"id": row["id"],
                                             "paired_views": [view[0] for view in views],
                                             "pair_mode": pair_mode,
                                             "pair_source_ids": source_ids})
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
                        examples.append({"id": row["id"],
                                         "paired_views": [view[0] for view in views],
                                         "paired_polarity": polarities[0],
                                         "paired_view_arms": arms})
                    else:
                        examples.extend(flatten_row(row, path.parent))
    return examples


def examples_have_images(examples):
    return any(view.get("image") for example in examples
               for view in example.get("paired_views", [example]))


def validate_multimodal_examples(examples, microbatch):
    if not examples_have_images(examples):
        return
    if microbatch != 1:
        raise ValueError("multimodal training requires --microbatch 1")
    for example in examples:
        views = example.get("paired_views", [example])
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
    """Mean paired CE plus optional consistency; preserve the exact zero-weight path."""
    base = torch.stack(view_losses).mean()
    if weight == 0 or consistency is None:
        return base
    return base + weight * consistency


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


def guard_logical_slot_loss(view_losses):
    """Mean slot loss, retaining only A at half weight for paired slots."""
    if not view_losses or any(not losses for losses in view_losses):
        raise ValueError("each logical slot needs at least one view loss")
    slots = [losses[0] * 0.5 if len(losses) == 2 else losses[0]
             for losses in view_losses]
    return torch.stack(slots).mean()


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
                      return_components=False, processor=None):
    validate_pair_objectives(pair_consistency_weight, return_components,
                             pair_flip_weight, pair_flip_margin)
    logical_views = [example.get("paired_views", [example]) for example in examples]
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
    consistency_slot_losses = []
    slot_correct = []
    consistency_seen = 0
    consistency_active = 0
    flip_losses = []
    flip_active = 0
    cursor = 0
    for logical_example, views in zip(examples, logical_views):
        end = cursor + len(views)
        slot_view_losses = view_losses[cursor:end]
        if logical_example.get("pair_mode") is not None:
            if len(views) != 2:
                raise ValueError(f"{logical_example['id']}: counterfactual pair needs two views")
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
        elif len(views) == 2 and (pair_consistency_weight > 0 or return_components):
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
                consistency_slot_losses.append(slot_view_losses[0].new_zeros(()))
        slot_correct.append(sum(view_correct[cursor:end]) / len(views))
        cursor = end
    stats = pair_consistency_stats(consistency_active, consistency_seen)
    stats.update({"pair_flip_seen": len(flip_losses),
                  "pair_flip_active": flip_active,
                  "pair_flip_loss": (sum(loss.detach().item() for loss in flip_losses) /
                                     len(flip_losses) if flip_losses else 0.0)})
    loss = mean_logical_slot_loss(slot_losses)
    if return_components:
        components = {"d_loss": loss,
                      "guard_loss": guard_logical_slot_loss(guard_slot_losses),
                      "consistency_loss": torch.stack(consistency_slot_losses).mean()}
        return loss, sum(slot_correct), stats, components
    return loss, sum(slot_correct), stats


@torch.no_grad()
def evaluate(model, examples, args, tokenizer, groups, device, processor=None):
    model.eval()
    losses = []
    correct = 0
    count = 0
    for start in range(0, min(args.eval_samples, len(examples)), args.microbatch):
        batch = examples[start:start + args.microbatch]
        loss, batch_correct, _ = loss_for_examples(model, batch, tokenizer, groups,
                                                   args.max_length, device, shuffle=False,
                                                   processor=processor)
        losses.append(loss.item())
        correct += batch_correct
        count += len(batch)
    model.train()
    return {"nll": sum(losses) / len(losses), "accuracy": correct / count, "n": count}


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
                    return_components=True, processor=processor)
                pair_active += pair_stats["pair_consistency_active"]
                pair_seen += pair_stats["pair_consistency_seen"]
                flip_active += pair_stats["pair_flip_active"]
                flip_seen += pair_stats["pair_flip_seen"]
                flip_loss_sum += pair_stats["pair_flip_loss"] * pair_stats["pair_flip_seen"]
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
                    pair_flip_margin=args.pair_flip_margin, processor=processor)
                pair_active += pair_stats["pair_consistency_active"]
                pair_seen += pair_stats["pair_consistency_seen"]
                flip_active += pair_stats["pair_flip_active"]
                flip_seen += pair_stats["pair_flip_seen"]
                flip_loss_sum += pair_stats["pair_flip_loss"] * pair_stats["pair_flip_seen"]
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
        if step % args.save_every == 0 or step == args.steps or early_stop:
            save(model, optimizer, scheduler, step, out)
        if early_stop:
            return


if __name__ == "__main__":
    main()
