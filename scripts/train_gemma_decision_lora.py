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
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForImageTextToText, AutoTokenizer


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
    return parser.parse_args()


def flatten_row(row):
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
        })
    return examples


def load_examples(paths):
    examples = []
    for path in paths:
        with open(path) as source:
            for line in source:
                if line.strip():
                    examples.extend(flatten_row(json.loads(line)))
    return examples


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


def encode(example, tokenizer, max_length, shuffle):
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
    return ids, torch.tensor(probabilities)


def loss_for_examples(model, examples, tokenizer, groups, max_length, device, shuffle):
    encoded = [encode(example, tokenizer, max_length, shuffle) for example in examples]
    ids = pad_sequence([item[0] for item in encoded], batch_first=True,
                       padding_value=tokenizer.pad_token_id)
    mask = ids.ne(tokenizer.pad_token_id)
    positions = torch.tensor([len(item[0]) - 1 for item in encoded])
    union = torch.unique(positions, sorted=True)
    output = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False,
                   logits_to_keep=union.to(device))
    offsets = {position.item(): index for index, position in enumerate(union)}
    losses = []
    correct = 0
    for batch_index, ((_, probabilities), position) in enumerate(zip(encoded, positions)):
        logits = output.logits[batch_index, offsets[position.item()]].float()
        option_scores = torch.stack([
            torch.logsumexp(logits[token_ids.to(device)], dim=0)
            for token_ids in groups[:len(probabilities)]
        ])
        losses.append(-(probabilities.to(device) * F.log_softmax(option_scores, dim=0)).sum())
        correct += int(option_scores.argmax().item() == probabilities.argmax().item())
    return torch.stack(losses).mean(), correct


@torch.no_grad()
def evaluate(model, examples, args, tokenizer, groups, device):
    model.eval()
    losses = []
    correct = 0
    count = 0
    for start in range(0, min(args.eval_samples, len(examples)), args.microbatch):
        batch = examples[start:start + args.microbatch]
        loss, batch_correct = loss_for_examples(model, batch, tokenizer, groups,
                                                args.max_length, device, shuffle=False)
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


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_examples(args.train)
    validation = {name: load_examples([path]) for name, path in
                  (item.split("=", 1) for item in args.val)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES,
    ))
    bad = [name for name, parameter in model.named_parameters()
           if parameter.requires_grad and any(part in name for part in
                                               ("vision", "visual", "projector", "embedder"))]
    if bad:
        raise RuntimeError(f"LoRA unexpectedly touched the multimodal path: {bad[:3]}")
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
    order = list(range(len(train)))
    random.shuffle(order)
    cursor = 0
    optimizer.zero_grad(set_to_none=True)
    metrics = out / "metrics.jsonl"
    started = time.time()
    baseline = {"step": 0, "validation": {
        name: evaluate(model, rows, args, tokenizer, groups, device)
        for name, rows in validation.items()
    }}
    metrics.write_text(json.dumps(baseline) + "\n")
    print(json.dumps(baseline), flush=True)
    for step in range(1, args.steps + 1):
        accumulated = 0.0
        for _ in range(args.grad_accum):
            if cursor + args.microbatch > len(order):
                random.shuffle(order)
                cursor = 0
            batch = [train[index] for index in order[cursor:cursor + args.microbatch]]
            cursor += args.microbatch
            loss, _ = loss_for_examples(model, batch, tokenizer, groups, args.max_length,
                                        device, shuffle=True)
            (loss / args.grad_accum).backward()
            accumulated += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        metric = {"step": step, "train_loss": accumulated / args.grad_accum,
                  "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
                  "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metric["validation"] = {
                name: evaluate(model, rows, args, tokenizer, groups, device)
                for name, rows in validation.items()
            }
        with metrics.open("a") as target:
            target.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(model, optimizer, scheduler, step, out)


if __name__ == "__main__":
    main()
