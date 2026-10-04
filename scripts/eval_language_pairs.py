"""Evaluate aligned language views from one native checkpoint.

uv run scripts/eval_language_pairs.py --model runs/tm2 --left data_he/eval/massive_he_test.jsonl \
  --right data_he/eval/massive_en_test_paired.jsonl --out runs/tm2/massive_he_en.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "pcdm"), str(ROOT)]

import bench  # noqa: E402
from metrics import summarize  # noqa: E402
from train import eval_dataset, load_jsonl  # noqa: E402


def paired_metrics(left, right, left_examples, right_examples):
    mean = np.clip((left + right) / 2, 1e-12, 1)
    left_safe, right_safe = np.clip(left, 1e-12, 1), np.clip(right, 1e-12, 1)
    jsd = 0.5 * (left_safe * (np.log(left_safe) - np.log(mean))).sum(-1) + \
        0.5 * (right_safe * (np.log(right_safe) - np.log(mean))).sum(-1)
    left_pred, right_pred = left.argmax(-1), right.argmax(-1)
    gold = np.array([left.shape[1] - 1 if ex["label"] == -1 else ex["label"] for ex in left_examples])
    return {
        "n": len(gold),
        "prediction_agreement": float((left_pred == right_pred).mean()),
        "both_correct": float(((left_pred == gold) & (right_pred == gold)).mean()),
        "either_correct": float(((left_pred == gold) | (right_pred == gold)).mean()),
        "mean_jsd": float(jsd.mean()),
        "p95_jsd": float(np.quantile(jsd, 0.95)),
        "mean_max_probability_delta": float(np.abs(left - right).max(-1).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="native run directory containing best.pt")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    left_examples, right_examples = load_jsonl(args.left), load_jsonl(args.right)
    assert len(left_examples) == len(right_examples) > 0
    for left, right in zip(left_examples, right_examples):
        assert left["meta"]["pair"] == right["meta"]["pair"]
        assert left["candidates"] == right["candidates"] and left["label"] == right["label"]

    checkpoint = Path(args.model) / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False).get("args", {})
    head, model = bench.load_ours(args.model, None, args.device)
    temperature = 1.0
    result_path = Path(args.model) / "results.json"
    if result_path.exists():
        temperature = float(json.loads(result_path.read_text()).get("T", 1.0))

    left_probs, left_target, left_label = eval_dataset(
        head, model, None, left_examples, args.batch_size, Ts=(temperature,), readout="native",
        max_state=saved.get("max_state", 256))
    right_probs, right_target, right_label = eval_dataset(
        head, model, None, right_examples, args.batch_size, Ts=(temperature,), readout="native",
        max_state=saved.get("max_state", 256))
    left_probs, right_probs = left_probs[temperature], right_probs[temperature]
    result = {
        "model": args.model,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "temperature": temperature,
        "left": {"path": args.left, "metrics": summarize(left_probs, left_target, left_label)},
        "right": {"path": args.right, "metrics": summarize(right_probs, right_target, right_label)},
        "paired": paired_metrics(left_probs, right_probs, left_examples, right_examples),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
