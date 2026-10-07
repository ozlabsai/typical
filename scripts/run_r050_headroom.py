"""Run hash-anchored R050 parent headroom inference and seal its sufficiency receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_r048_action_panel as r048
from scripts import build_r050_headroom as headroom
from scripts import score_exact_policy_panel as scorer
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, MODEL_ID, verify_inventory
from scripts.build_r050_replay_pools import _atomic_publish
from scripts.preflight_r043_assets import file_hash, tree_hash
from scripts.singleton_option_scorer import validate_runtime_fingerprint


REPORT_SCHEMA = "r050_headroom_report_v1"
RECEIPT_SCHEMA = "r050_headroom_receipt_v1"
MAX_LENGTH = 8192
MODES = ("priority", "collect")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"{label}: externally expected sha256 must be 64 hexadecimal characters")
    return value.lower()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(payload):
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    try:
        return json.loads(payload, object_pairs_hook=_object, parse_constant=nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON payload") from error


def validate_headroom(data_payload, manifest_payload, expected_data_sha256, expected_manifest_sha256):
    """Authenticate externally frozen bytes, then verify the canonical full tail."""
    for label, payload, expected in (("headroom data", data_payload, expected_data_sha256),
                                     ("headroom manifest", manifest_payload, expected_manifest_sha256)):
        if _sha(payload) != _hash(expected, label):
            raise ValueError(f"{label}: externally expected sha256 mismatch")
    manifest = _parse(manifest_payload)
    builder_hash = file_hash(r048.__file__)
    if (not isinstance(manifest, dict) or manifest.get("schema") != headroom.SCHEMA
            or manifest.get("role") != "headroom" or manifest.get("panel") != scorer.PANEL
            or manifest.get("panel_version") != r048.PANEL_VERSION
            or manifest.get("generator_sha256") != builder_hash
            or manifest.get("headroom_builder_sha256") != file_hash(headroom.__file__)):
        raise ValueError("manifest role/schema/panel version or current builder binding is invalid")
    if (type(manifest.get("cluster_size")) is not int or manifest["cluster_size"] != r048.CLUSTER_SIZE
            or manifest.get("row_counts") != {"headroom": headroom.HEADROOM_ROWS}
            or type(manifest.get("seed")) is not int or not 0 <= manifest["seed"] <= 9999):
        raise ValueError("manifest headroom counts/seed are invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"headroom"}:
        raise ValueError("manifest must bind only the headroom output")
    output = outputs["headroom"]
    if (not isinstance(output, dict) or type(output.get("rows")) is not int
            or output["rows"] != headroom.HEADROOM_ROWS or output.get("sha256") != _sha(data_payload)):
        raise ValueError("manifest output row/hash binding is invalid")
    bindings = manifest.get("bindings")
    if (not isinstance(bindings, dict) or bindings.get("r048_builder_sha256") != builder_hash
            or not isinstance(bindings.get("core_manifest"), dict)
            or not isinstance(bindings.get("core_files"), dict)
            or set(bindings["core_files"]) != set(r048.SPLITS)):
        raise ValueError("manifest frozen-core bindings are invalid")
    _hash(bindings["core_manifest"].get("sha256"), "core manifest")
    for split, count in r048.DEFAULT_ROWS.items():
        item = bindings["core_files"][split]
        if not isinstance(item, dict) or type(item.get("rows")) is not int or item["rows"] != count:
            raise ValueError("manifest frozen-core row binding is invalid")
        _hash(item.get("sha256"), f"core {split}")
    construction = manifest.get("construction")
    prefix = r048.DEFAULT_ROWS["eval"]
    if (not isinstance(construction, dict) or construction.get("canonical_row_split") != "eval"
            or construction.get("frozen_prefix_rows") != prefix
            or construction.get("expanded_eval_rows") != prefix + headroom.HEADROOM_ROWS
            or construction.get("tail_clusters") != headroom.HEADROOM_CLUSTERS
            or construction.get("prefix_objects_and_bytes_identical") is not True):
        raise ValueError("manifest does not describe the separate unconsumed headroom tail")
    rows = [_parse(line) for line in data_payload.splitlines() if line.strip()]
    if len(rows) != headroom.HEADROOM_ROWS or any(not isinstance(row, dict) for row in rows):
        raise ValueError("headroom JSONL must contain the complete expected object rows")
    r048.validate_split(rows, "eval", headroom.HEADROOM_ROWS)
    for row in rows:
        scorer._row_contract(row)
    expected_ids = [f"r048-eval-{manifest['seed']:04d}-{index:06d}"
                    for index in range(prefix, prefix + headroom.HEADROOM_ROWS)]
    if [row["id"] for row in rows] != expected_ids or r048._jsonl_bytes(rows) != data_payload:
        raise ValueError("headroom IDs/order/serialized bytes do not match the canonical tail")
    clusters = _clusters(rows)
    if len(clusters) != headroom.HEADROOM_CLUSTERS:
        raise ValueError("headroom independent cluster count differs from the frozen design")
    return rows, manifest


def _clusters(rows):
    """Validate units independently of prediction values; never infer units from IDs."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("headroom needs nonempty rows")
    grouped, identifiers, signatures = defaultdict(list), set(), set()
    for row in rows:
        scorer._row_contract(row)
        if row["id"] in identifiers:
            raise ValueError("headroom row IDs must be unique")
        identifiers.add(row["id"])
        meta = row["meta"]
        group = meta.get("semantic_group_id")
        if not isinstance(group, str) or not group or meta.get("policy_mode") not in MODES:
            raise ValueError("headroom needs semantic groups and both recognized policy modes")
        if meta.get("semantic_signature") != _hash(meta.get("semantic_signature"), "semantic signature"):
            raise ValueError("semantic signatures must use canonical lowercase SHA256")
        if not 2 <= len(meta["field_order"]) <= 8 or type(meta.get("counterfactual_side")) is not int:
            raise ValueError("headroom group field count/side is invalid")
        order = meta.get("option_order")
        if order not in r048.OPTION_ORDERS or any(list(question["criteria"]) != order.split("_")
                                                for question in row["questions"].values()):
            raise ValueError("headroom option order metadata disagrees with serialized choices")
        grouped[group].append(row)
    variants = {(side, language, order) for side in (0, 1)
                for language in r048.LANGUAGES for order in r048.OPTION_ORDERS}
    result = {}
    for group, items in grouped.items():
        keys = [(row["meta"]["counterfactual_side"], row["meta"].get("language"), row["meta"]["option_order"])
                for row in items]
        if len(items) != r048.CLUSTER_SIZE or set(keys) != variants:
            raise ValueError("each headroom semantic group must be a complete unique eight-row orbit")
        for key in ("semantic_signature", "policy_mode", "policy", "field_order", "qid_to_field",
                    "decision_threshold", "policy_topology", "topology_slice"):
            if len({_json(row["meta"].get(key)) for row in items}) != 1:
                raise ValueError(f"headroom group has inconsistent {key}")
        signature = items[0]["meta"]["semantic_signature"]
        if signature in signatures:
            raise ValueError("independent semantic signatures may not be repeated across concrete groups")
        signatures.add(signature)
        endpoints = {}
        for side in (0, 1):
            side_rows = [row for row in items if row["meta"]["counterfactual_side"] == side]
            for key in ("action_trace", "gold_outcome", "facts"):
                if len({_json(row["meta"].get(key)) for row in side_rows}) != 1:
                    raise ValueError(f"headroom endpoint has inconsistent {key}")
            endpoints[side] = side_rows[0]["meta"]
        if (sum(endpoints[0]["action_trace"][field] != endpoints[1]["action_trace"][field]
                for field in endpoints[0]["field_order"]) != 1
                or endpoints[0]["gold_outcome"] == endpoints[1]["gold_outcome"]):
            raise ValueError("headroom twins must flip exactly one field and the complete outcome")
        result[signature] = items
    return result


def evaluate_headroom(rows, predictions):
    """Recompute complete outcomes and fail closed before any sufficiency decision."""
    clusters = _clusters(rows)
    if not isinstance(predictions, list) or any(not isinstance(item, dict) for item in predictions):
        raise ValueError("predictions must contain object rows")
    identifiers = [item.get("id") for item in predictions]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("prediction IDs must be nonempty strings")
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != {row["id"] for row in rows}:
        raise ValueError("prediction coverage must be exact and IDs unique; no missing or extra rows")
    by_id, errors = dict(zip(identifiers, predictions)), {}
    for row in rows:
        item = by_id[row["id"]]
        try:
            _json(item)
            classified = scorer.classify(row, item.get("probabilities"))
            for key, expected in classified.items():
                if key in {"predicted_outcome", "gold_outcome"}:
                    expected = expected.as_dict()
                if key in item and _json(item[key]) != _json(expected):
                    raise ValueError(f"claimed {key} differs from recomputed prediction")
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{row['id']}: invalid prediction: {error}") from error
        errors[row["id"]] = classified["predicted_outcome"] != classified["gold_outcome"]
    total_errors = sum(errors.values())
    def counts(units):
        selected = [row for items in units for row in items]
        return {"rows": len(selected), "row_errors": sum(errors[row["id"]] for row in selected),
                "independent_clusters": len(units),
                "failed_independent_clusters": sum(any(errors[row["id"]] for row in items) for items in units)}
    modes = {mode: counts([items for items in clusters.values() if items[0]["meta"]["policy_mode"] == mode])
             for mode in MODES}
    large = counts([items for items in clusters.values() if len(items[0]["meta"]["field_order"]) >= 5])
    requirements = {"full_coverage": True, "zero_invalid": True,
                    "complete_outcome_error_at_least_ten_percent": total_errors * 10 >= len(rows),
                    **{f"{mode}_has_failed_independent_cluster": modes[mode]["failed_independent_clusters"] >= 1
                       for mode in MODES},
                    "five_or_more_fields_has_failed_independent_cluster": large["failed_independent_clusters"] >= 1}
    return {"schema": REPORT_SCHEMA, "role": "headroom", "headroom_sufficient": all(requirements.values()),
            "requirements": requirements, "failures": [key for key, passed in requirements.items() if not passed],
            "expected_rows": len(rows), "valid_predictions": len(predictions), "invalid": 0, "coverage": 1.0,
            "overall": {**counts(list(clusters.values())), "complete_outcome_error_rate": total_errors / len(rows)},
            "by_policy_mode": modes, "fields_at_least_five": large,
            "cluster_unit": "semantic_signature; one complete eight-row orbit per unique signature",
            "cluster_failure": "at least one rendered view has an incorrect complete executable outcome",
            "overall_error_unit": "rendered rows, equal weight; complete primary action and action set",
            "failed_semantic_signatures": sorted(signature for signature, items in clusters.items()
                                                  if any(errors[row["id"]] for row in items))}


def verify_assets(model_path, inventory_path, adapter_path, *, expected_inventory_sha256,
                  expected_adapter_tree_sha256, expected_adapter_tree_files):
    inventory_expectation = _hash(expected_inventory_sha256, "model inventory")
    if file_hash(inventory_path) != inventory_expectation:
        raise ValueError("externally expected model inventory sha256 mismatch")
    _parse(Path(inventory_path).read_bytes())
    inventory, inventory_hash = verify_inventory(inventory_path, model_path, FROZEN_BASE_REVISION)
    if inventory_hash != inventory_expectation:
        raise ValueError("model inventory changed during verification")
    digest = _hash(expected_adapter_tree_sha256, "parent adapter tree")
    if type(expected_adapter_tree_files) is not int or expected_adapter_tree_files <= 0:
        raise ValueError("externally expected adapter tree file count must be a positive integer")
    adapter = Path(adapter_path).absolute()
    if not adapter.is_dir() or adapter.is_symlink() or any(item.is_symlink() for item in adapter.rglob("*")):
        raise ValueError("parent adapter must be a directory without symbolic links")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (adapter / name).is_file() or (adapter / name).stat().st_size <= 0:
            raise ValueError("parent adapter needs nonempty safetensors weights and config")
    observed, count = tree_hash(adapter)
    if (observed, count) != (digest, expected_adapter_tree_files):
        raise ValueError("parent adapter tree differs from external hash/file-count expectation")
    config = _parse((adapter / "adapter_config.json").read_bytes())
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError("parent adapter must contain a LoRA configuration")
    if config.get("revision") not in {None, FROZEN_BASE_REVISION}:
        raise ValueError("parent adapter declares a different base revision")
    base = config.get("base_model_name_or_path")
    if base != MODEL_ID and (not isinstance(base, str) or Path(base).name != FROZEN_BASE_REVISION):
        raise ValueError("parent adapter declares a different base model")
    return {"model_id": MODEL_ID, "model_revision": FROZEN_BASE_REVISION,
            "model_inventory_sha256": inventory_hash, "base_model_tree_sha256": inventory["tree_sha256"],
            "base_model_tree_files": inventory["files"], "adapter_tree_sha256": observed,
            "adapter_tree_files": count, "adapter_weights_sha256": file_hash(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": file_hash(adapter / "adapter_config.json")}


def infer_rows(rows, model_path, adapter_path):
    """The only GPU path: frozen local BF16 base, PEFT parent, singleton SDPA."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from scripts.singleton_option_scorer import ForwardObservations, collect_runtime_fingerprint
    from scripts.train_gemma_decision_lora import letter_token_groups

    if not torch.cuda.is_available():
        raise ValueError("trusted headroom inference requires an actual CUDA GPU")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("base parameters did not load entirely in BF16")
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True, is_trainable=False)
    model.config.use_cache = False
    device = torch.device("cuda:0")
    model.to(device).eval()
    observations = ForwardObservations(require_sdpa=True)
    with torch.inference_mode():
        predictions = scorer.score_rows_scalar(model, tokenizer, groups, rows, device, MAX_LENGTH, observations)
    runtime = collect_runtime_fingerprint(model, observations, device)
    return predictions, runtime, observations.report()


def _code_hashes():
    names = ("run_r050_headroom.py", "build_r050_headroom.py", "build_r048_action_panel.py",
             "score_exact_policy_panel.py", "singleton_option_scorer.py", "exact_policy_executor.py",
             "train_gemma_decision_lora.py", "build_r047_model_inventory.py", "preflight_r043_assets.py",
             "build_r050_replay_pools.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def _write(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise IOError(f"output read-back mismatch: {path.name}")


def publish(out, predictions, report, receipt):
    out = Path(out).absolute()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite headroom run: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        prediction_payload = b"".join(_json(item) + b"\n" for item in predictions)
        payloads = {"predictions.jsonl": prediction_payload, "report.json": _json(report) + b"\n"}
        receipt = {**receipt, "outputs": {name: {"sha256": _sha(payload), "bytes": len(payload)}
                                         for name, payload in payloads.items()}}
        payloads["receipt.json"] = _json(receipt) + b"\n"
        for name, payload in payloads.items():
            _write(stage / name, payload)
        # Written last; a directory only becomes visible after all files and the marker exist.
        marker = {"schema": "r050_headroom_completion_v1", "receipt_sha256": _sha(payloads["receipt.json"]),
                  "outputs": receipt["outputs"], "headroom_sufficient": report["headroom_sufficient"]}
        _write(stage / "complete.json", _json(marker) + b"\n")
        _atomic_publish(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # This invocation's explicit private staging directory only.
    return receipt


def run(data, manifest, model, inventory, adapter, out, *, expected_data_sha256,
        expected_manifest_sha256, expected_inventory_sha256, expected_adapter_tree_sha256,
        expected_adapter_tree_files):
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite headroom run: {out}")
    data_payload, manifest_payload = Path(data).read_bytes(), Path(manifest).read_bytes()
    rows, checked_manifest = validate_headroom(data_payload, manifest_payload,
                                              expected_data_sha256, expected_manifest_sha256)
    asset_kwargs = {"expected_inventory_sha256": expected_inventory_sha256,
                    "expected_adapter_tree_sha256": expected_adapter_tree_sha256,
                    "expected_adapter_tree_files": expected_adapter_tree_files}
    assets = verify_assets(model, inventory, adapter, **asset_kwargs)
    code_hashes = _code_hashes()
    predictions, runtime, histogram = infer_rows(rows, model, adapter)
    validate_runtime_fingerprint(runtime)
    forwards = sum(len(row["questions"]) for row in rows)
    if histogram != {"1": forwards}:
        raise ValueError("observed forwards must match every field exactly once at batch size one")
    report = evaluate_headroom(rows, predictions)
    if (file_hash(data) != _sha(data_payload) or file_hash(manifest) != _sha(manifest_payload)
            or _code_hashes() != code_hashes or verify_assets(model, inventory, adapter, **asset_kwargs) != assets):
        raise ValueError("verified inputs, code, base, or parent changed during inference")
    receipt = {"schema": RECEIPT_SCHEMA, "role": "headroom", "panel_version": r048.PANEL_VERSION,
               "headroom_sufficient": report["headroom_sufficient"], "assets": assets,
               "headroom_data_sha256": _sha(data_payload), "headroom_manifest_sha256": _sha(manifest_payload),
               "headroom_core_bindings": checked_manifest["bindings"], "code_sha256": code_hashes,
               "runtime_fingerprint": runtime, "runtime_fingerprint_sha256": _sha(_json(runtime)),
               "execution": {"mode": "singleton_scalar", "max_length": MAX_LENGTH, "batch_histogram": histogram,
                             "expected_field_forwards": forwards, "base_dtype": "bfloat16", "attention": "sdpa"}}
    return report, publish(out, predictions, report, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "model", "inventory", "adapter", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("data", "manifest", "adapter-tree"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-inventory-sha256", "--expected-model-inventory-sha256",
                        dest="expected_inventory_sha256", required=True)
    parser.add_argument("--expected-adapter-tree-files", required=True, type=int)
    args = vars(parser.parse_args(argv))
    report, _ = run(**args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["headroom_sufficient"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
