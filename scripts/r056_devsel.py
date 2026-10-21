#!/usr/bin/env python3
"""R056 checkpoint selection + calibration data: teacher cross-entropy of one LoRA checkpoint on the
teacher-labeled dev split, with the serving forward (B=1, last logit, option-letter groups).

Writes {"summary": {nll, accuracy_vs_teacher_argmax, n, skipped, by_workflow}, "rows": [{id, student, teacher}]}.
  PYTHONPATH=. python scripts/r056_devsel.py --model <gemma dir> --adapter <step dir> --dev dev_wf.jsonl --out x.json
"""
import argparse
import json
import math
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from scripts.singleton_option_scorer import singleton_option_probabilities
from scripts.train_gemma_decision_lora import enable_bidirectional_prompt, encode, letter_token_groups, load_examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--ids", help="R060-pre: score only the example ids listed in this file (the proxy set)")
    ap.add_argument("--bidirectional-prompt", action="store_true", help="R057: adapter trained with the prompt mask")
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
                                                        attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model = model.to("cuda").eval()
    if args.bidirectional_prompt:
        enable_bidirectional_prompt(model)
    keep = set(open(args.ids).read().split()) if args.ids else None
    groups = letter_token_groups(tokenizer)
    rows, skipped = [], 0
    for ex in load_examples([args.dev]):
        if keep is not None and ex["id"] not in keep:
            continue
        try:
            ids, teacher = encode(ex, tokenizer, args.max_length, shuffle=False)
        except ValueError:
            skipped += 1
            continue
        labels = [k for k, _ in ex["items"]]
        with torch.inference_mode():
            student = singleton_option_probabilities(model, ids[None].to("cuda"), groups, labels)
        rows.append({"id": ex["id"], "student": [student[k] for k in labels], "teacher": teacher.tolist()})
    by = defaultdict(list)
    for r in rows:
        ce = -sum(t * math.log(max(s, 1e-12)) for t, s in zip(r["teacher"], r["student"]))
        hit = max(range(len(r["teacher"])), key=r["teacher"].__getitem__) == max(range(len(r["student"])), key=r["student"].__getitem__)
        r["ce"], r["hit"] = ce, hit
        by["all"].append(r)
        by[r["id"].split("-")[1]].append(r)  # r056-<workflow>-NNNN-...
    summary = {k: {"nll": sum(r["ce"] for r in v) / len(v), "acc": sum(r["hit"] for r in v) / len(v), "n": len(v)}
               for k, v in by.items()}
    summary["skipped_overlength"] = skipped
    json.dump({"adapter": args.adapter, "summary": summary, "rows": rows}, open(args.out, "w"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
