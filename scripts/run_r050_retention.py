"""Trusted R050 Hebrew and visual retention comparison against R044-R."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_r047_model_inventory import (
    FROZEN_BASE_REVISION,
    MODEL_ID,
    verify_inventory,
)
from scripts.build_r050_replay_pools import _atomic_publish
from scripts.preflight_r043_assets import file_hash, tree_hash
from scripts.run_r050_eval import validate_arm_completion
from scripts.singleton_option_scorer import validate_runtime_fingerprint
from scripts.train_gemma_decision_lora import SYSTEM


SCHEMA = "r050_protected_retention_receipt_v1"
COMPLETION_SCHEMA = "r050_protected_retention_completion_v1"
ASSET_INVENTORY_SCHEMA = "r050_replay_pools_v1"
MAX_LENGTH = 8192
ROWS_PER_SUITE = 128
SUITES = ("he", "en", "real", "shuffled", "noimage", "mismatch", "gold_dropped")
MM_SUITES = SUITES[2:]
NULL_OPTION = "None of the options is supported by the image."
SOURCE_SHA256 = {
    "he": "604c99aaf56449fe5f836c4901e4ca400ff2625f0d5edc12aec8e35875e8956a",
    "en": "4f0b50a65baa20dd3cbbc54e095950b52cb905a16b9a04a998c5debc441a9daa",
    "real": "9ac7793c12484c30f01102a4daeac9c16576e48310185d3967b13d91193214c3",
    "shuffled": "604f0df603b20add7cb73c0767df9c5e8dfdf935aa24077554dd8016f3cd2715",
    "noimage": "c20efc463b80d08d687d38794722d8b5abaa1b8554a1146e9bd4c92d497d92e3",
    "mismatch": "f37916443f30bc61e101e08b66e18588ffba96881d16e71f21aac1efbb288ace",
    "gold_dropped": "daf6136912819a713d3a6904323ebc166cd12e077a1aff172559834a06b5807c",
}
FROZEN_MODEL_INVENTORY_SHA256 = "5c97d1ed2179841c5824733d6b1c83c2de300d8458a6eb22e50846a18ec7e605"
FROZEN_MODEL_TREE_SHA256 = "30ef898307067e89a7349ede93920d2914105053b02fdc53b8c9dfc57fcb896b"
FROZEN_MODEL_TREE_FILES = 9
FROZEN_PARENT_TREE_SHA256 = "44e1998efd10275a6f0ab20cc6ccbd97cb9a8ad9b8c3f620620a5f6fdcc91f99"
FROZEN_PARENT_TREE_FILES = 4
FROZEN_ASSET_INVENTORY_SHA256 = "d556492b58922a8f3a7e51b3f907611404e9c02fc92c0084f7e09e422a3f4e3b"
FROZEN_SCIENCEQA_IMAGE_TREE_SHA256 = "9d08678f915a5d28132153fda3063e4f0d7c74a4fff167c5621ac6afbf3e5e7f"
FROZEN_SCIENCEQA_IMAGE_TREE_FILES = 1949


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"{label} must be a 64-character hexadecimal sha256")
    return value.lower()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(payload: bytes, label: str):
    def nonfinite(value):
        raise ValueError(f"{label} contains nonfinite JSON: {value}")
    try:
        return json.loads(payload, object_pairs_hook=_object, parse_constant=nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _row_schema(row: dict[str, Any], suite: str, index: int) -> None:
    required = {"state", "query", "candidates", "target", "p_null", "task", "label", "meta"}
    if not isinstance(row, dict) or not required <= set(row):
        raise ValueError(f"{suite}:{index}: incomplete flat decision schema")
    candidates, target, label = row["candidates"], row["target"], row["label"]
    if (not isinstance(row["state"], str) or not row["state"].strip()
            or not isinstance(row["query"], str) or not row["query"].strip()
            or not isinstance(candidates, list)
            or not (1 if suite == "gold_dropped" else 2) <= len(candidates) <= 25
            or any(not isinstance(value, str) or not value.strip() for value in candidates)
            or len(set(candidates)) != len(candidates)
            or not isinstance(target, list) or len(target) != len(candidates)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in target)
            or not isinstance(row["meta"], dict)):
        raise ValueError(f"{suite}:{index}: malformed flat decision")
    null_expected = suite in {"mismatch", "gold_dropped"}
    if (isinstance(row["p_null"], bool)
            or not isinstance(row["p_null"], (int, float))
            or not math.isfinite(row["p_null"])
            or row["p_null"] not in {0, 1}
            or bool(row["p_null"]) != null_expected):
        raise ValueError(f"{suite}:{index}: null target disagrees with suite")
    if null_expected:
        if label != -1:
            raise ValueError(f"{suite}:{index}: null suite label must be -1")
    elif (type(label) is not int or not 0 <= label < len(candidates)
          or not math.isclose(sum(target), 1.0, abs_tol=1e-9)
          or target[label] != 1.0):
        raise ValueError(f"{suite}:{index}: invalid categorical gold label")
    image = row.get("image")
    if suite in {"he", "en", "noimage"}:
        if image is not None:
            raise ValueError(f"{suite}:{index}: suite must not contain images")
    elif not isinstance(image, str) or not image:
        raise ValueError(f"{suite}:{index}: suite requires an image reference")


def _case(row: dict[str, Any], source: Path, suite: str) -> dict[str, Any]:
    candidates = list(row["candidates"])
    label = row["label"]
    null = bool(row["p_null"])
    if suite in MM_SUITES:
        candidates.append(NULL_OPTION)
        if null:
            label = len(candidates) - 1
    image = row.get("image")
    image_path = (source.parent / image).resolve() if image else None
    return {"id": row["meta"]["pair"], "state": row["state"], "query": row["query"],
            "candidates": candidates, "label": label, "null": null,
            "image": str(image_path) if image_path else None}


def validate_sources(paths: dict[str, str | Path], expected_hashes: dict[str, str],
                     asset_inventory_path: str | Path,
                     expected_asset_inventory_sha256: str) -> tuple[
                         dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Validate literal source anchors, prefix selection, alignment, and image bytes."""
    if set(paths) != set(SUITES) or set(expected_hashes) != set(SUITES):
        raise ValueError("source paths and expected hashes must name all seven suites")
    loaded, bindings = {}, {}
    for suite in SUITES:
        if _hash(expected_hashes[suite], f"expected {suite} sha256") != SOURCE_SHA256[suite]:
            raise ValueError(f"{suite}: caller expectation is not the literal frozen anchor")
        path = Path(paths[suite])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{suite}: source must be a regular file")
        payload = path.read_bytes()
        if _sha(payload) != SOURCE_SHA256[suite]:
            raise ValueError(f"{suite}: source bytes differ from the literal frozen anchor")
        rows = [_parse(line, f"{suite} row") for line in payload.splitlines() if line.strip()]
        if len(rows) < ROWS_PER_SUITE:
            raise ValueError(f"{suite}: source has fewer than 128 rows")
        selected = rows[:ROWS_PER_SUITE]
        for index, row in enumerate(selected):
            _row_schema(row, suite, index)
            if not isinstance(row["meta"].get("pair"), str) or not row["meta"]["pair"]:
                raise ValueError(f"{suite}:{index}: missing pair identity")
        loaded[suite] = selected
        bindings[suite] = {"path": str(path.resolve()), "sha256": _sha(payload),
                           "source_rows": len(rows), "selected_rows": ROWS_PER_SUITE,
                           "selection": "first_128_in_frozen_source_order"}
    he_ids = [row["meta"]["pair"] for row in loaded["he"]]
    if len(set(he_ids)) != ROWS_PER_SUITE:
        raise ValueError("Hebrew/English pair identities must be unique")
    if he_ids != [row["meta"]["pair"] for row in loaded["en"]]:
        raise ValueError("Hebrew and English pair identities are not aligned")
    for left, right in zip(loaded["he"], loaded["en"]):
        if left["candidates"] != right["candidates"] or left["label"] != right["label"]:
            raise ValueError("Hebrew/English semantic decisions are not aligned")
    real_ids = [row["meta"]["pair"] for row in loaded["real"]]
    if len(set(real_ids)) != ROWS_PER_SUITE:
        raise ValueError("ScienceQA pair identities must be unique")
    if any(real_ids != [row["meta"]["pair"] for row in loaded[name]] for name in MM_SUITES[1:]):
        raise ValueError("ScienceQA suite identities are not aligned")
    variants = {"real": "base", "shuffled": "shuffled_forced_answer", "noimage": "base",
                "mismatch": "image_mismatch", "gold_dropped": "remove_gold"}
    for index in range(ROWS_PER_SUITE):
        real = loaded["real"][index]
        for name in MM_SUITES:
            row = loaded[name][index]
            if row["meta"].get("variant") != variants[name]:
                raise ValueError(f"{name}:{index}: unexpected variant contract")
        for name in ("shuffled", "noimage", "mismatch"):
            row = loaded[name][index]
            if (row["state"], row["query"], row["candidates"], row.get("target")) != (
                    real["state"], real["query"], real["candidates"], real.get("target")):
                raise ValueError(f"real/{name}:{index}: decision content is not aligned")
        dropped = loaded["gold_dropped"][index]
        expected_candidates = [value for position, value in enumerate(real["candidates"])
                               if position != real["label"]]
        if (dropped["state"], dropped["query"], dropped["candidates"]) != (
                real["state"], real["query"], expected_candidates):
            raise ValueError(f"gold_dropped:{index}: gold removal is invalid")
        if (loaded["mismatch"][index]["image"] != loaded["shuffled"][index]["image"]
                or dropped["image"] != real["image"]):
            raise ValueError(f"ScienceQA:{index}: image-control assignment is invalid")

    inventory_path = Path(asset_inventory_path)
    inventory_payload = inventory_path.read_bytes()
    if (_hash(expected_asset_inventory_sha256, "expected asset inventory sha256")
            != FROZEN_ASSET_INVENTORY_SHA256
            or _sha(inventory_payload) != FROZEN_ASSET_INVENTORY_SHA256):
        raise ValueError("asset inventory differs from the literal frozen anchor")
    inventory = _parse(inventory_payload, "asset inventory")
    if not isinstance(inventory, dict) or inventory.get("schema") != ASSET_INVENTORY_SCHEMA:
        raise ValueError("asset inventory schema is invalid")
    protection = inventory.get("inputs", {}).get("scienceqa_protection")
    if not isinstance(protection, list):
        raise ValueError("asset inventory lacks ScienceQA protection entries")
    by_source = {entry.get("sha256"): entry for entry in protection if isinstance(entry, dict)}
    image_bindings = {}
    for suite in MM_SUITES:
        entry = by_source.get(SOURCE_SHA256[suite])
        images = entry.get("images") if isinstance(entry, dict) else None
        expected_files = 0 if suite == "noimage" else FROZEN_SCIENCEQA_IMAGE_TREE_FILES
        expected_tree = (_sha(b"") if suite == "noimage"
                         else FROZEN_SCIENCEQA_IMAGE_TREE_SHA256)
        if (not isinstance(images, dict) or images.get("files") != expected_files
                or images.get("tree_sha256") != expected_tree
                or not isinstance(images.get("files_by_path"), dict)):
            raise ValueError(f"{suite}: asset inventory tree binding is invalid")
        image_bindings[suite] = images
    selected_images = {}
    for suite in MM_SUITES:
        for row in loaded[suite]:
            reference = row.get("image")
            if not reference:
                continue
            expected = image_bindings[suite]["files_by_path"].get(reference)
            path = (Path(paths[suite]).parent / reference).resolve()
            if (not isinstance(expected, str) or path.is_symlink() or not path.is_file()
                    or file_hash(path) != expected):
                raise ValueError(f"{suite}: referenced image differs from authenticated inventory")
            selected_images[str(path)] = expected
    selected_tree_payload = "".join(
        f"{name}\0{digest}\n" for name, digest in sorted(selected_images.items())).encode()
    cases = {suite: [_case(row, Path(paths[suite]), suite) for row in loaded[suite]]
             for suite in SUITES}
    return cases, {"sources": bindings,
                   "asset_inventory": {"path": str(inventory_path.resolve()),
                                       "sha256": _sha(inventory_payload)},
                   "full_scienceqa_image_tree": {
                       "sha256": FROZEN_SCIENCEQA_IMAGE_TREE_SHA256,
                       "files": FROZEN_SCIENCEQA_IMAGE_TREE_FILES},
                   "selected_images": {"files": len(selected_images),
                                       "tree_sha256": _sha(selected_tree_payload),
                                       "files_by_absolute_path": dict(sorted(selected_images.items()))}}


def verify_model(model_path, inventory_path, expected_inventory_sha256):
    if (_hash(expected_inventory_sha256, "expected model inventory sha256")
            != FROZEN_MODEL_INVENTORY_SHA256
            or file_hash(inventory_path) != FROZEN_MODEL_INVENTORY_SHA256):
        raise ValueError("model inventory differs from the literal frozen anchor")
    inventory, digest = verify_inventory(inventory_path, model_path, FROZEN_BASE_REVISION)
    if (digest != FROZEN_MODEL_INVENTORY_SHA256
            or inventory.get("tree_sha256") != FROZEN_MODEL_TREE_SHA256
            or inventory.get("files") != FROZEN_MODEL_TREE_FILES
            or inventory.get("model_id") != MODEL_ID):
        raise ValueError("base model is not the literal frozen Gemma inventory/tree")
    return {"model_id": MODEL_ID, "model_revision": FROZEN_BASE_REVISION,
            "model_inventory_sha256": digest,
            "base_model_tree_sha256": inventory["tree_sha256"],
            "base_model_tree_files": inventory["files"]}


def verify_adapter(path, expected_tree_sha256, expected_tree_files, label):
    expected = _hash(expected_tree_sha256, f"expected {label} tree sha256")
    if (isinstance(expected_tree_files, bool) or type(expected_tree_files) is not int
            or expected_tree_files <= 0):
        raise ValueError(f"expected {label} file count must be positive")
    path = Path(path)
    if path.is_symlink() or not path.is_dir() or any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError(f"{label} must be a regular directory tree")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (path / name).is_file() or (path / name).stat().st_size <= 0:
            raise ValueError(f"{label} is missing {name}")
    observed, files = tree_hash(path)
    if (observed, files) != (expected, expected_tree_files):
        raise ValueError(f"{label} differs from the externally expected tree")
    config = _parse((path / "adapter_config.json").read_bytes(), f"{label} config")
    if config.get("peft_type") != "LORA" or config.get("revision") not in {None, FROZEN_BASE_REVISION}:
        raise ValueError(f"{label} configuration is unsupported")
    base = config.get("base_model_name_or_path")
    if base != MODEL_ID and (not isinstance(base, str) or Path(base).name != FROZEN_BASE_REVISION):
        raise ValueError(f"{label} declares a different base model")
    return {"path": str(path.resolve()), "adapter_tree_sha256": observed,
            "adapter_tree_files": files,
            "adapter_weights_sha256": file_hash(path / "adapter_model.safetensors"),
            "adapter_config_sha256": file_hash(path / "adapter_config.json")}


def _prompt(case, processor):
    options = "\n".join(f"{chr(65 + index)}. {choice}"
                        for index, choice in enumerate(case["candidates"]))
    text = "\n\n".join((case["state"].rstrip(), case["query"].rstrip(),
                         f"Options:\n{options}",
                         "Answer with the letter of exactly one option, and nothing else:"))
    content = ([{"type": "image"}] if case["image"] else []) + [{"type": "text", "text": text}]
    return processor.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def infer_adapter(model_path, adapter_path, suites):
    """Load one adapter and run one BF16 SDPA CUDA forward per retained case."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from scripts.singleton_option_scorer import (
        ForwardObservations,
        collect_runtime_fingerprint,
        probabilities_from_option_scores,
    )
    from scripts.train_gemma_decision_lora import letter_token_groups

    if not torch.cuda.is_available():
        raise ValueError("trusted retention inference requires an actual CUDA GPU")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    groups = letter_token_groups(processor.tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("base parameters did not load entirely in BF16")
    model = PeftModel.from_pretrained(
        model, str(adapter_path), local_files_only=True, is_trainable=False)
    model.config.use_cache = False
    device = torch.device("cuda:0")
    model.to(device).eval()
    observations = ForwardObservations(require_sdpa=True)
    predictions = {}
    with torch.inference_mode():
        for suite, cases in suites.items():
            values = []
            for case in cases:
                text = _prompt(case, processor)
                kwargs = {"text": [text], "return_tensors": "pt"}
                if case["image"]:
                    with Image.open(case["image"]) as image:
                        kwargs["images"] = [image.convert("RGB")]
                        inputs = processor(**kwargs)
                else:
                    inputs = processor(**kwargs)
                if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
                    raise ValueError("retention scorer requires singleton input_ids")
                if inputs["input_ids"].shape[1] > MAX_LENGTH:
                    raise ValueError(f"{suite}/{case['id']}: prompt exceeds max length")
                inputs = {name: value.to(device) for name, value in inputs.items()}
                observations.batch_histogram[1] += 1
                def forward():
                    return model(**inputs, use_cache=False, logits_to_keep=1)
                if not observations.sdpa_observed:
                    with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                        output = forward()
                    observations.sdpa_observed = any(
                        "scaled_dot_product_attention" in event.key
                        for event in profiler.key_averages())
                    if not observations.sdpa_observed:
                        raise ValueError("SDPA profiler canary did not observe attention")
                else:
                    output = forward()
                observations.logits_dtypes.add(str(output.logits.dtype).removeprefix("torch."))
                logits = output.logits[0, -1].float()
                scores = [float(torch.logsumexp(logits[group.to(device)], dim=0).item())
                          for group in groups[:len(case["candidates"])]]
                probabilities = probabilities_from_option_scores(
                    list(range(len(case["candidates"]))), scores)
                vector = [probabilities[index] for index in range(len(probabilities))]
                values.append({"id": case["id"], "probabilities": vector,
                               "predicted_index": max(range(len(vector)), key=vector.__getitem__),
                               "label": case["label"], "null_probability": vector[-1]})
            predictions[suite] = values
    runtime = collect_runtime_fingerprint(model, observations, device)
    return predictions, runtime, observations.report()


def validate_predictions(suites, predictions, label):
    if not isinstance(predictions, dict) or set(predictions) != set(SUITES):
        raise ValueError(f"{label}: predictions must contain all seven suites")
    clean = {}
    for suite, cases in suites.items():
        values = predictions[suite]
        if not isinstance(values, list) or len(values) != ROWS_PER_SUITE:
            raise ValueError(f"{label}/{suite}: prediction cardinality is invalid")
        if [item.get("id") for item in values if isinstance(item, dict)] != [case["id"] for case in cases]:
            raise ValueError(f"{label}/{suite}: prediction IDs/order are invalid")
        clean[suite] = []
        for case, item in zip(cases, values):
            probs = item.get("probabilities")
            if (not isinstance(probs, list) or len(probs) != len(case["candidates"])
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or not 0 <= value <= 1 for value in probs)
                    or not math.isclose(math.fsum(probs), 1.0, rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f"{label}/{suite}/{case['id']}: invalid probabilities")
            predicted = max(range(len(probs)), key=probs.__getitem__)
            if (item.get("predicted_index") != predicted or item.get("label") != case["label"]
                    or item.get("null_probability") != probs[-1]):
                raise ValueError(f"{label}/{suite}/{case['id']}: claimed output mismatch")
            clean[suite].append([float(value) for value in probs])
    return clean


def _summary(cases, predictions):
    correct, nll, brier, confidence = [], [], [], []
    for case, probabilities in zip(cases, predictions):
        gold = case["label"]
        correct.append(int(max(range(len(probabilities)), key=probabilities.__getitem__) == gold))
        nll.append(-math.log(max(probabilities[gold], 1e-12)))
        brier.append(math.fsum((value - float(index == gold)) ** 2
                               for index, value in enumerate(probabilities)))
        confidence.append(max(probabilities))
    null_indices = [index for index, case in enumerate(cases) if case["null"]]
    null_correct = sum(max(range(len(predictions[index])),
                           key=predictions[index].__getitem__) == cases[index]["label"]
                       for index in null_indices)
    return {"n": len(cases), "correct": sum(correct), "accuracy": sum(correct) / len(cases),
            "nll": math.fsum(nll) / len(cases), "brier": math.fsum(brier) / len(cases),
            "null_n": len(null_indices), "null_correct": null_correct,
            "null_recall": null_correct / len(null_indices) if null_indices else None}


def _paired_jsd(left, right):
    values = []
    for a, b in zip(left, right):
        if len(a) != len(b):
            raise ValueError("Hebrew pair probability cardinality differs")
        mean = [(x + y) / 2 for x, y in zip(a, b)]
        values.append(.5 * math.fsum(
            x * math.log(max(x, 1e-12) / max(m, 1e-12))
            + y * math.log(max(y, 1e-12) / max(m, 1e-12))
            for x, y, m in zip(a, b, mean)))
    return math.fsum(values) / len(values)


def _auroc(negative, positive):
    if not negative or not positive:
        raise ValueError("AUROC needs nonempty positive and negative classes")
    wins = sum(a < b for a in negative for b in positive)
    ties = sum(a == b for a in negative for b in positive)
    return (wins + .5 * ties) / (len(negative) * len(positive))


def _relative(candidate, parent):
    if parent < 0 or candidate < 0 or not math.isfinite(parent + candidate):
        raise ValueError("proper losses must be finite and nonnegative")
    if parent == 0:
        return {"parent": parent, "candidate": candidate, "relative_regression": 0.0 if candidate == 0 else None,
                "passed": candidate == 0, "zero_parent_behavior": "candidate must also equal zero"}
    regression = (candidate - parent) / parent
    return {"parent": parent, "candidate": candidate, "relative_regression": regression,
            "passed": regression <= .05, "zero_parent_behavior": "not_applicable"}


def compare(suites, parent_predictions, candidate_predictions):
    parent_probs = validate_predictions(suites, parent_predictions, "parent")
    candidate_probs = validate_predictions(suites, candidate_predictions, "candidate")
    parent = {name: _summary(suites[name], parent_probs[name]) for name in SUITES}
    candidate = {name: _summary(suites[name], candidate_probs[name]) for name in SUITES}
    parent_jsd = _paired_jsd(parent_probs["he"], parent_probs["en"])
    candidate_jsd = _paired_jsd(candidate_probs["he"], candidate_probs["en"])
    null_scores = lambda name: [values[-1] for values in candidate_probs[name]]
    candidate_auroc = _auroc(
        null_scores("real") + null_scores("shuffled") + null_scores("noimage"),
        null_scores("mismatch") + null_scores("gold_dropped"))
    proper = {suite: {metric: _relative(candidate[suite][metric], parent[suite][metric])
                      for metric in ("nll", "brier")} for suite in SUITES}
    gates = {
        "he_accuracy_errors_lost_at_most_one": parent["he"]["correct"] - candidate["he"]["correct"] <= 1,
        "en_accuracy_errors_lost_at_most_one": parent["en"]["correct"] - candidate["en"]["correct"] <= 1,
        "real_accuracy_errors_lost_at_most_one": parent["real"]["correct"] - candidate["real"]["correct"] <= 1,
        "he_pair_jsd_delta_at_most_0.01": candidate_jsd - parent_jsd <= .01,
        "he_nll_delta_at_most_0.05": candidate["he"]["nll"] - parent["he"]["nll"] <= .05,
        "mismatch_null_recall_loss_at_most_one_of_128": (
            parent["mismatch"]["null_correct"] - candidate["mismatch"]["null_correct"] <= 1),
        "gold_dropped_null_recall_loss_at_most_one_of_128": (
            parent["gold_dropped"]["null_correct"] - candidate["gold_dropped"]["null_correct"] <= 1),
        "candidate_real_minus_shuffled_at_least_0.10": (
            candidate["real"]["accuracy"] - candidate["shuffled"]["accuracy"] >= .10),
        "candidate_real_minus_noimage_at_least_0.10": (
            candidate["real"]["accuracy"] - candidate["noimage"]["accuracy"] >= .10),
        "candidate_answerability_auroc_at_least_0.75": candidate_auroc >= .75,
        "every_suite_nll_and_brier_relative_regression_at_most_5pct": all(
            value["passed"] for metrics in proper.values() for value in metrics.values()),
    }
    return {"parent": {**parent, "he_pair_mean_jsd": parent_jsd},
            "candidate": {**candidate, "he_pair_mean_jsd": candidate_jsd,
                          "answerability_auroc": candidate_auroc},
            "proper_loss_relative_regression": proper, "gates": gates,
            "promotion_pass": all(gates.values()),
            "comparison_unit": "R044-R parent adapter versus candidate adapter on identical rows"}


def _code_hashes():
    names = ("run_r050_retention.py", "singleton_option_scorer.py",
             "train_gemma_decision_lora.py", "build_r047_model_inventory.py",
             "preflight_r043_assets.py", "run_r050_eval.py", "run_r050_arm.py",
             "build_r050_replay_pools.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def _write(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def publish(out, parent_predictions, candidate_predictions, report, receipt):
    out = Path(out).absolute()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite retention output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        def prediction_payload(predictions):
            return b"".join(_json({"suite": suite, **item}) + b"\n"
                            for suite in SUITES for item in predictions[suite])
        payloads = {"parent_predictions.jsonl": prediction_payload(parent_predictions),
                    "candidate_predictions.jsonl": prediction_payload(candidate_predictions),
                    "report.json": _json(report) + b"\n"}
        outputs = {name: {"sha256": _sha(payload), "bytes": len(payload)}
                   for name, payload in payloads.items()}
        sealed = {**receipt, "outputs": outputs}
        payloads["receipt.json"] = _json(sealed) + b"\n"
        for name, payload in payloads.items():
            _write(stage / name, payload)
        marker = {"schema": COMPLETION_SCHEMA, "status": "complete",
                  "receipt_sha256": _sha(payloads["receipt.json"]),
                  "promotion_pass": report["promotion_pass"], "outputs": outputs}
        _write(stage / "complete.json", _json(marker) + b"\n")
        _atomic_publish(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sealed


def run(*, paths, expected_hashes, asset_inventory, expected_asset_inventory_sha256,
        model, model_inventory, expected_model_inventory_sha256,
        parent_adapter, candidate_adapter, expected_candidate_tree_sha256,
        expected_candidate_tree_files, candidate_completion,
        expected_candidate_completion_sha256, out):
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite retention output: {out}")
    suites, inputs = validate_sources(
        paths, expected_hashes, asset_inventory, expected_asset_inventory_sha256)
    model_binding = verify_model(model, model_inventory, expected_model_inventory_sha256)
    parent = verify_adapter(parent_adapter, FROZEN_PARENT_TREE_SHA256,
                            FROZEN_PARENT_TREE_FILES, "R044-R parent adapter")
    candidate = verify_adapter(candidate_adapter, expected_candidate_tree_sha256,
                               expected_candidate_tree_files, "candidate adapter")
    completion_assets = {**model_binding, **candidate}
    completion = validate_arm_completion(
        candidate_completion, expected_candidate_completion_sha256,
        adapter_path=candidate_adapter, assets=completion_assets,
        model_path=model, inventory_path=model_inventory)
    if completion is None:
        raise ValueError("a candidate arm completion receipt is required")
    code = _code_hashes()
    parent_predictions, parent_runtime, parent_histogram = infer_adapter(
        model, parent_adapter, suites)
    candidate_predictions, candidate_runtime, candidate_histogram = infer_adapter(
        model, candidate_adapter, suites)
    validate_runtime_fingerprint(parent_runtime)
    validate_runtime_fingerprint(candidate_runtime)
    expected_forwards = len(SUITES) * ROWS_PER_SUITE
    if parent_histogram != {"1": expected_forwards} or candidate_histogram != {"1": expected_forwards}:
        raise ValueError("each adapter must run exactly one singleton forward per protected case")
    report = compare(suites, parent_predictions, candidate_predictions)
    check_suites, check_inputs = validate_sources(
        paths, expected_hashes, asset_inventory, expected_asset_inventory_sha256)
    if (check_suites != suites or check_inputs != inputs or _code_hashes() != code
            or verify_model(model, model_inventory, expected_model_inventory_sha256) != model_binding
            or verify_adapter(parent_adapter, FROZEN_PARENT_TREE_SHA256,
                              FROZEN_PARENT_TREE_FILES, "R044-R parent adapter") != parent
            or verify_adapter(candidate_adapter, expected_candidate_tree_sha256,
                              expected_candidate_tree_files, "candidate adapter") != candidate
            or validate_arm_completion(
                candidate_completion, expected_candidate_completion_sha256,
                adapter_path=candidate_adapter, assets=completion_assets,
                model_path=model, inventory_path=model_inventory) != completion):
        raise ValueError("retention inputs, assets, lineage, or code changed during inference")
    receipt = {"schema": SCHEMA, "role": "protected_retention", "inputs": inputs,
               "model": model_binding, "parent_adapter": parent,
               "candidate_adapter": candidate, "candidate_completion": completion,
               "code_sha256": code,
               "runtime_fingerprints": {"parent": parent_runtime,
                                        "candidate": candidate_runtime},
               "execution": {"mode": "singleton_native_letter", "max_length": MAX_LENGTH,
                             "forwards_per_adapter": expected_forwards,
                             "parent_batch_histogram": parent_histogram,
                             "candidate_batch_histogram": candidate_histogram,
                             "base_dtype": "bfloat16", "attention": "sdpa",
                             "temperature_fitting": False}}
    return report, publish(out, parent_predictions, candidate_predictions, report, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for suite in SUITES:
        parser.add_argument(f"--{suite}", required=True, type=Path)
        parser.add_argument(f"--expected-{suite}-sha256", required=True)
    for name in ("asset-inventory", "model", "model-inventory", "parent-adapter",
                 "candidate-adapter", "candidate-completion", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-asset-inventory-sha256", required=True)
    parser.add_argument("--expected-model-inventory-sha256", required=True)
    parser.add_argument("--expected-candidate-tree-sha256", required=True)
    parser.add_argument("--expected-candidate-tree-files", required=True, type=int)
    parser.add_argument("--expected-candidate-completion-sha256", required=True)
    args = vars(parser.parse_args(argv))
    paths = {suite: args.pop(suite) for suite in SUITES}
    expected = {suite: args.pop(f"expected_{suite}_sha256") for suite in SUITES}
    report, _ = run(paths=paths, expected_hashes=expected, **args)
    print(json.dumps({"promotion_pass": report["promotion_pass"],
                      "gates": report["gates"]}, sort_keys=True))
    return 0 if report["promotion_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
