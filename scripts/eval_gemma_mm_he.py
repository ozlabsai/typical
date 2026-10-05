"""Read-only Hebrew and pixel-conditioned retention audit for Gemma decision LoRAs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.train_gemma_decision_lora import SYSTEM, letter_token_groups


NULL_OPTION = "None of the options is supported by the image."
MM_FILES = {
    "real": "scienceqa_test.jsonl", "shuffled": "scienceqa_test_shuffled.jsonl",
    "noimage": "scienceqa_test_noimage.jsonl", "mismatch": "scienceqa_test_mismatch.jsonl",
    "gold_dropped": "scienceqa_test_gold_dropped.jsonl",
}


def _read(path, *, multimodal):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    cases = []
    for index, row in enumerate(rows):
        candidates = list(row["candidates"])
        label = row["label"]
        if multimodal:
            candidates.append(NULL_OPTION)
            label = len(candidates) - 1 if row.get("p_null", 0) else label
        if not 0 <= label < len(candidates):
            raise ValueError(f"{path}:{index}: invalid decision label")
        image = row.get("image")
        cases.append({"id": row.get("meta", {}).get("pair", f"{Path(path).stem}-{index}"),
                      "state": row["state"], "query": row["query"], "candidates": candidates,
                      "label": label, "null": bool(row.get("p_null", 0)),
                      "image": str(Path(path).parent / image) if image else None})
    return cases


def _prompt(case, processor):
    options = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(case["candidates"]))
    text = "\n\n".join((case["state"].rstrip(), case["query"].rstrip(),
                           f"Options:\n{options}",
                           "Answer with the letter of exactly one option, and nothing else:"))
    content = ([{"type": "image"}] if case["image"] else []) + [{"type": "text", "text": text}]
    return processor.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@torch.no_grad()
def _score(model, processor, groups, case, device, max_length):
    text = _prompt(case, processor)
    kwargs = {"text": [text], "return_tensors": "pt"}
    if case["image"]:
        with Image.open(case["image"]) as image:
            kwargs["images"] = [image.convert("RGB")]
            inputs = processor(**kwargs)
    else:
        inputs = processor(**kwargs)
    if inputs["input_ids"].shape[-1] > max_length:
        raise ValueError(f"{case['id']}: {inputs['input_ids'].shape[-1]} tokens exceeds {max_length}")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    scores = torch.stack([torch.logsumexp(logits[token_ids.to(device)], dim=0)
                          for token_ids in groups[:len(case["candidates"])]] )
    return F.softmax(scores, dim=0).tolist()


def _summary(cases, predictions):
    correct, nll, brier, confidence = [], [], [], []
    for case, probabilities in zip(cases, predictions):
        gold = case["label"]
        correct.append(int(max(range(len(probabilities)), key=probabilities.__getitem__) == gold))
        nll.append(-math.log(max(probabilities[gold], 1e-12)))
        brier.append(sum((value - float(index == gold)) ** 2 for index, value in enumerate(probabilities)))
        confidence.append(max(probabilities))
    ece = 0.0
    for bucket in range(10):
        indices = [index for index, value in enumerate(confidence) if min(9, int(value * 10)) == bucket]
        if indices:
            ece += len(indices) / len(cases) * abs(sum(confidence[i] for i in indices) / len(indices) -
                                                    sum(correct[i] for i in indices) / len(indices))
    null_rows = [index for index, case in enumerate(cases) if case["null"]]
    return {"n": len(cases), "accuracy": sum(correct) / len(cases), "nll": sum(nll) / len(cases),
            "brier": sum(brier) / len(cases), "ece": ece,
            "null_recall": (sum(max(range(len(predictions[i])), key=predictions[i].__getitem__) == cases[i]["label"]
                                for i in null_rows) / len(null_rows) if null_rows else None)}


def _paired(left, right):
    if len(left) != len(right):
        raise ValueError("paired sets have different lengths")
    jsd, agreement = [], []
    for left_values, right_values in zip(left, right):
        if len(left_values) != len(right_values):
            raise ValueError("paired sets have different choice cardinality")
        mean = [(a + b) / 2 for a, b in zip(left_values, right_values)]
        jsd.append(.5 * sum(a * math.log(max(a, 1e-12) / max(m, 1e-12)) +
                            b * math.log(max(b, 1e-12) / max(m, 1e-12))
                            for a, b, m in zip(left_values, right_values, mean)))
        agreement.append(max(range(len(left_values)), key=left_values.__getitem__) ==
                         max(range(len(right_values)), key=right_values.__getitem__))
    return {"mean_jsd": sum(jsd) / len(jsd), "prediction_agreement": sum(agreement) / len(agreement)}


def _auroc(negative, positive):
    wins = sum(a < b for a in negative for b in positive) + .5 * sum(a == b for a in negative for b in positive)
    return wins / (len(negative) * len(positive))


def _run(model, processor, groups, suites, device, max_length):
    predictions = {name: [_score(model, processor, groups, case, device, max_length) for case in cases]
                   for name, cases in suites.items()}
    report = {name: _summary(suites[name], values) for name, values in predictions.items()}
    report["he_pair"] = _paired(predictions["he"], predictions["en"])
    null_scores = lambda name: [values[-1] for values in predictions[name]]
    report["answerability_auroc"] = _auroc(null_scores("real"), null_scores("mismatch") + null_scores("gold_dropped"))
    return report


def _comparison(base, candidate):
    loss = lambda suite: base[suite]["accuracy"] - candidate[suite]["accuracy"]
    return {"he_errors_lost": loss("he") * base["he"]["n"],
            "en_errors_lost": loss("en") * base["en"]["n"],
            "he_nll_delta": candidate["he"]["nll"] - base["he"]["nll"],
            "he_jsd_delta": candidate["he_pair"]["mean_jsd"] - base["he_pair"]["mean_jsd"],
            "real_errors_lost": loss("real") * base["real"]["n"],
            "real_minus_shuffled": candidate["real"]["accuracy"] - candidate["shuffled"]["accuracy"],
            "real_minus_noimage": candidate["real"]["accuracy"] - candidate["noimage"]["accuracy"],
            "answerability_auroc": candidate["answerability_auroc"],
            "mismatch_null_recall_lost": base["mismatch"]["null_recall"] - candidate["mismatch"]["null_recall"],
            "gold_dropped_null_recall_lost": base["gold_dropped"]["null_recall"] - candidate["gold_dropped"]["null_recall"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--he-left", required=True)
    parser.add_argument("--he-right", required=True)
    parser.add_argument("--mm-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    suites = {"he": _read(args.he_left, multimodal=False), "en": _read(args.he_right, multimodal=False)}
    suites.update({name: _read(Path(args.mm_dir) / filename, multimodal=True) for name, filename in MM_FILES.items()})
    if [case["id"] for case in suites["he"]] != [case["id"] for case in suites["en"]]:
        raise ValueError("Hebrew/English rows are not aligned")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(processor.tokenizer)
    if any(len(cases) != 128 for cases in suites.values()):
        raise ValueError("locked retention screen requires exactly 128 rows per suite")
    device = torch.device("cuda")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa").to(device).eval()
    base = _run(base_model, processor, groups, suites, device, args.max_length)
    del base_model
    torch.cuda.empty_cache()
    candidate_model = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"), args.adapter,
        local_files_only=True).to(device).eval()
    candidate = _run(candidate_model, processor, groups, suites, device, args.max_length)
    comparison = _comparison(base, candidate)
    comparison["gates"] = {"he_errors_lost_at_most_one": comparison["he_errors_lost"] <= 1,
                           "en_errors_lost_at_most_one": comparison["en_errors_lost"] <= 1,
                           "he_nll_delta_at_most_.05": comparison["he_nll_delta"] <= .05,
                           "he_jsd_delta_at_most_.01": comparison["he_jsd_delta"] <= .01,
                           "real_errors_lost_at_most_one": comparison["real_errors_lost"] <= 1,
                           "real_minus_shuffled_at_least_.10": comparison["real_minus_shuffled"] >= .10,
                           "real_minus_noimage_at_least_.10": comparison["real_minus_noimage"] >= .10,
                           "answerability_auroc_at_least_.75": comparison["answerability_auroc"] >= .75,
                           "mismatch_null_recall_lost_at_most_one": comparison["mismatch_null_recall_lost"] <= 1 / 128,
                           "gold_dropped_null_recall_lost_at_most_one": comparison["gold_dropped_null_recall_lost"] <= 1 / 128}
    report = {"inputs": {"he_left_sha256": hashlib.sha256(Path(args.he_left).read_bytes()).hexdigest(),
                         "he_right_sha256": hashlib.sha256(Path(args.he_right).read_bytes()).hexdigest(),
                         "mm": {name: hashlib.sha256((Path(args.mm_dir) / filename).read_bytes()).hexdigest()
                                for name, filename in MM_FILES.items()}},
              "base": base, "candidate": candidate, "comparison": comparison}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
