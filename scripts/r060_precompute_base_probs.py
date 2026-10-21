#!/usr/bin/env python3
"""R060: cache the frozen base's option distribution into anchored rows, so the trainer's KL-to-base anchor
skips its extra no-grad forward. Writes meta.base_probs = {qid: {option label: p}} on rows with meta[--on]
truthy; every other row passes through unchanged. Canonical option order, no adapter, causal attention (the
trainer's adapter-off forward under --bidirectional-prompt would differ; don't mix the two).

GPU: uv run python -m scripts.r060_precompute_base_probs --model M --input train.jsonl --out train.base.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

from scripts.train_gemma_decision_lora import (
    encode, encode_image, flatten_row, letter_token_groups, load_tokenizer_and_processor,
    option_scores_from_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--on", default="anchor", help="meta key selecting anchored rows")
    args = parser.parse_args()
    source = Path(args.input)
    rows = [json.loads(line) for line in source.open() if line.strip()]
    tokenizer, processor = load_tokenizer_and_processor(args.model, any(row.get("image") for row in rows))
    groups = letter_token_groups(tokenizer)
    device = torch.device("cuda")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa").to(device).eval()
    anchored = 0
    with open(args.out, "w") as target, torch.no_grad():
        for row in rows:
            meta = row.get("meta") or {}
            if meta.get(args.on):  # ponytail: top-level rows only; paired/auxiliary views are not cached
                meta["base_probs"] = {}
                for example in flatten_row(row, source.parent):
                    if example["image"]:
                        inputs, _, labels = encode_image(example, processor, 10**9, False, include_labels=True)
                        inputs = {name: value.to(device) for name, value in inputs.items()}
                    else:
                        ids, _, labels = encode(example, tokenizer, 10**9, False, include_labels=True)
                        inputs = {"input_ids": ids[None].to(device)}
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    probs = torch.softmax(option_scores_from_logits(logits, groups, len(labels), device), 0)
                    qid = example["id"].removeprefix(f"{row.get('id')}/")
                    meta["base_probs"][qid] = dict(zip(labels, probs.tolist()))
                row["meta"] = meta
                anchored += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "anchored": anchored, "out": args.out}))


if __name__ == "__main__":
    main()
