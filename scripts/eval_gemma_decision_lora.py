"""Evaluate a Gemma decision adapter with the training-time option scorer."""

import argparse
import json
import random

import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from transformers import AutoModelForImageTextToText, AutoTokenizer

from train_gemma_decision_lora import encode, evaluate, letter_token_groups, load_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--val", action="append", required=True, help="NAME=JSONL")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.scale_layer(args.adapter_scale)
    model.to("cuda").eval()
    eval_args = argparse.Namespace(
        eval_samples=args.samples, microbatch=1, max_length=args.max_length
    )
    groups = letter_token_groups(tokenizer)
    rng = random.Random(args.seed)
    results = {}
    for item in args.val:
        name, path = item.split("=", 1)
        examples = load_examples([path])
        rng.shuffle(examples)
        selected = []
        skipped = 0
        for example in examples:
            try:
                encode(example, tokenizer, args.max_length, shuffle=False)
            except ValueError as error:
                if "tokens exceeds" not in str(error):
                    raise
                skipped += 1
                continue
            selected.append(example)
            if len(selected) == args.samples:
                break
        result = evaluate(model, selected, eval_args, tokenizer, groups, "cuda")
        result["skipped_overlength"] = skipped
        results[name] = result
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
