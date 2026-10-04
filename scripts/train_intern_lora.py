"""One-GPU reversible LoRA continuation for Intern-Decision-4B.

The script deliberately reuses Intern's canonical compiler and supervises only
the token immediately before each ``<decision>`` marker. It does not pack
requests, because Qwen3.5's hybrid linear attention needs sequence boundaries.

uv run scripts/train_intern_lora.py --model /path/to/Intern-Decision-4B \
  --train train.jsonl --val workflow=val.jsonl --intern-repo /path/to/Intern-Decision \
  --out runs/intern-g1
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", action="append", default=[], help="NAME=JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--intern-repo", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def setup_schema(repo):
    sys.path.insert(0, str(Path(repo).resolve()))
    from src.inputs.schema import DECISION_TOKEN, compile_row
    return DECISION_TOKEN, compile_row


def encode(row, tokenizer, compile_row, marker_id, max_length):
    compiled = compile_row(row)
    text = tokenizer.apply_chat_template(
        compiled.messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(ids) > max_length:
        raise ValueError(f"{row.get('id')}: {len(ids)} tokens exceeds {max_length}")
    positions = (ids == marker_id).nonzero().flatten() - 1
    if len(positions) != len(compiled.fields) or (positions < 0).any():
        raise ValueError(f"{row.get('id')}: decision marker mismatch")
    targets = []
    for field in compiled.fields:
        symbols = compiled.symbols[field]
        token_ids = [tokenizer.encode(symbol, add_special_tokens=False) for symbol in symbols]
        if any(len(value) != 1 for value in token_ids):
            raise ValueError("answer symbols must each be one token")
        probabilities = row.get("soft_targets", {}).get(field)
        if probabilities is None:
            probabilities = [float(symbol == compiled.targets[field]) for symbol in symbols]
        if len(probabilities) != len(symbols) or any(value < 0 for value in probabilities):
            raise ValueError(f"{row.get('id')}/{field}: invalid soft target")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError(f"{row.get('id')}/{field}: empty soft target")
        targets.append((torch.tensor([value[0] for value in token_ids]),
                        torch.tensor([value / total for value in probabilities])))
    return ids, positions, targets


def loss_for_rows(model, rows, tokenizer, compile_row, marker_id, max_length, device):
    encoded = [encode(row, tokenizer, compile_row, marker_id, max_length) for row in rows]
    ids = pad_sequence([item[0] for item in encoded], batch_first=True, padding_value=tokenizer.pad_token_id)
    mask = ids.ne(tokenizer.pad_token_id)
    union = torch.unique(torch.cat([item[1] for item in encoded]), sorted=True)
    output = model(
        input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False,
        logits_to_keep=union.to(device),
    )
    offsets = {position.item(): index for index, position in enumerate(union)}
    losses = []
    for batch_index, (_, positions, targets) in enumerate(encoded):
        logits = output.logits[batch_index, [offsets[position.item()] for position in positions]].float()
        for field_logits, (candidate_ids, probabilities) in zip(logits, targets):
            log_probabilities = F.log_softmax(field_logits, dim=-1)[candidate_ids.to(device)]
            losses.append(-(probabilities.to(device) * log_probabilities).sum())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, rows, samples, args, tokenizer, compile_row, marker_id, device):
    if samples <= 0:
        return None
    model.eval()
    losses = []
    for start in range(0, min(samples, len(rows)), args.microbatch):
        loss = loss_for_rows(model, rows[start:start + args.microbatch], tokenizer, compile_row,
                             marker_id, args.max_length, device)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save(model, optimizer, scheduler, step, out):
    path = out / f"step-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, path / "trainer.pt")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    decision_token, compile_row = setup_schema(args.intern_repo)
    train = load_rows(args.train)
    validation = {name: load_rows(path) for name, path in (item.split("=", 1) for item in args.val)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    marker_id = tokenizer.convert_tokens_to_ids(decision_token)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES,
    ))
    bad = [name for name, parameter in model.named_parameters()
           if parameter.requires_grad and any(part in name for part in ("visual", "vision", "projector"))]
    if bad:
        raise RuntimeError(f"LoRA unexpectedly touched the vision path: {bad[:3]}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    device = torch.device("cuda")
    model.to(device).train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"trainable_parameters": trainable, "total_parameters": total_parameters,
                      "trainable_fraction": trainable / total_parameters}), flush=True)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr,
        weight_decay=args.weight_decay, fused=True,
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
    metrics_path = out / "metrics.jsonl"
    started = time.time()
    baseline = {"step": 0, "validation": {
        name: evaluate(model, rows, args.eval_samples, args, tokenizer, compile_row, marker_id, device)
        for name, rows in validation.items()
    }}
    metrics_path.write_text(json.dumps(baseline) + "\n")
    print(json.dumps(baseline), flush=True)
    for step in range(1, args.steps + 1):
        total = 0.0
        for _ in range(args.grad_accum):
            if cursor + args.microbatch > len(order):
                random.shuffle(order)
                cursor = 0
            batch = [train[index] for index in order[cursor:cursor + args.microbatch]]
            cursor += args.microbatch
            loss = loss_for_rows(model, batch, tokenizer, compile_row, marker_id, args.max_length, device)
            (loss / args.grad_accum).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        metric = {"step": step, "train_loss": total / args.grad_accum,
                  "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
                  "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metric["validation"] = {
                name: evaluate(model, rows, args.eval_samples, args, tokenizer, compile_row, marker_id, device)
                for name, rows in validation.items()
            }
        with metrics_path.open("a") as target:
            target.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(model, optimizer, scheduler, step, out)


if __name__ == "__main__":
    main()
