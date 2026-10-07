"""Run trusted singleton inference on the frozen full R048 v1.6 eval panel.

This runner reports the scorer summary only.  It neither fits a temperature nor
applies a promotion gate; matched-arm comparison remains a separate sealed step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts import build_r048_action_panel as r048
from scripts import compare_r049_calibrated as comparator
from scripts import run_r050_headroom as trusted_runtime
from scripts import score_exact_policy_panel as scorer
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION
from scripts.build_r050_replay_pools import _atomic_publish
from scripts.preflight_r043_assets import file_hash
from scripts.run_r050_arm import COMPLETION_SCHEMA as ARM_COMPLETION_SCHEMA, validate_recipe
from scripts.singleton_option_scorer import validate_runtime_fingerprint


SCHEMA = "r050_full_eval_receipt_v1"
COMPLETION_SCHEMA = "r050_full_eval_completion_v1"
MAX_LENGTH = 8192
EVAL_ROWS = 2688
EVAL_CLUSTERS = 336
FROZEN_EVAL_SHA256 = "b764082343c68ec29bd94d9ca041c3e922b5aa421d40c06519683b804fd3a5e7"
FROZEN_MANIFEST_SHA256 = "9a477c41999abf2dab8c5e00f156ed96c01d318b9fd54364802be0c3e6cd95bb"
FROZEN_INVENTORY_SHA256 = "5c97d1ed2179841c5824733d6b1c83c2de300d8458a6eb22e50846a18ec7e605"
FROZEN_BASE_TREE_SHA256 = "30ef898307067e89a7349ede93920d2914105053b02fdc53b8c9dfc57fcb896b"
FROZEN_BASE_TREE_FILES = 9
FROZEN_OUTPUTS = {
    "train": {"rows": 1792,
              "sha256": "086e461c1342ca05a9ddcc64bae2b35853f7bac1468598f861eb2df6817e6234"},
    "cal": {"rows": 896,
            "sha256": "1cbb44c4ea92b9502c5274599dd6bde83e07cc41f3f55ed59774f3841581cf81"},
    "eval": {"rows": EVAL_ROWS, "sha256": FROZEN_EVAL_SHA256},
}
PROTECTED_FINGERPRINTS = {
    "encoder_endpoint_signature", "encoder_input_normalized",
    "encoder_semantic_signature", "model_input_exact", "model_input_normalized",
    "semantic_signature",
}


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


def validate_full_panel(data_payload: bytes, manifest_payload: bytes,
                        expected_data_sha256: str,
                        expected_manifest_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate literal anchors and validate the complete canonical eval split."""
    if _hash(expected_data_sha256, "expected eval panel sha256") != FROZEN_EVAL_SHA256:
        raise ValueError("expected eval panel sha256 is not the frozen literal anchor")
    if _hash(expected_manifest_sha256, "expected core manifest sha256") != FROZEN_MANIFEST_SHA256:
        raise ValueError("expected core manifest sha256 is not the frozen literal anchor")
    if _sha(data_payload) != FROZEN_EVAL_SHA256:
        raise ValueError("eval panel bytes differ from the frozen literal anchor")
    if _sha(manifest_payload) != FROZEN_MANIFEST_SHA256:
        raise ValueError("core manifest bytes differ from the frozen literal anchor")
    manifest = _parse(manifest_payload, "core manifest")
    if (not isinstance(manifest, dict) or set(manifest) != {
            "panel", "panel_version", "seed", "row_counts", "cluster_size",
            "source_contract", "headroom", "generator_sha256", "outputs", "audit"}
            or manifest.get("panel") != scorer.PANEL
            or manifest.get("panel_version") != r048.PANEL_VERSION
            or manifest.get("seed") != 48
            or manifest.get("row_counts") != r048.DEFAULT_ROWS
            or manifest.get("cluster_size") != r048.CLUSTER_SIZE
            or manifest.get("outputs") != FROZEN_OUTPUTS
            or manifest.get("generator_sha256") != file_hash(r048.__file__)):
        raise ValueError("core manifest is not the canonical full v1.6 corpus manifest")
    audit = manifest.get("audit")
    if not isinstance(audit, dict) or audit.get("all_protected_fingerprints_disjoint") is not True:
        raise ValueError("core manifest does not attest disjoint protected fingerprints")
    overlaps = audit.get("cross_split_overlaps")
    expected_pairs = {"train__cal", "train__eval", "cal__eval"}
    if not isinstance(overlaps, dict) or set(overlaps) != expected_pairs:
        raise ValueError("core manifest cross-split audit is incomplete")
    for pair, values in overlaps.items():
        if (not isinstance(values, dict) or not PROTECTED_FINGERPRINTS <= set(values)
                or any(values[name] != 0 for name in PROTECTED_FINGERPRINTS)):
            raise ValueError(f"core manifest protected overlap is nonzero: {pair}")
    split_audit = audit.get("splits", {}).get("eval")
    if (not isinstance(split_audit, dict)
            or split_audit.get("rows") != EVAL_ROWS
            or split_audit.get("generated_counterfactual_clusters") != EVAL_CLUSTERS
            or split_audit.get("unique_exact_semantic_case_clusters") != EVAL_CLUSTERS
            or split_audit.get("unique_split_neutral_semantic_signatures") != EVAL_CLUSTERS
            or split_audit.get("expected_design_cells") != EVAL_CLUSTERS
            or split_audit.get("design_cells") != EVAL_CLUSTERS):
        raise ValueError("core manifest eval audit does not describe the full 336-cluster panel")
    rows = [_parse(line, "eval row") for line in data_payload.splitlines() if line.strip()]
    if len(rows) != EVAL_ROWS or any(not isinstance(row, dict) for row in rows):
        raise ValueError("eval JSONL must contain exactly 2,688 object rows")
    r048.validate_split(rows, "eval", EVAL_ROWS)
    for row in rows:
        scorer._row_contract(row)
    if r048._jsonl_bytes(rows) != data_payload:
        raise ValueError("eval rows do not round-trip to the canonical serialized bytes")
    expected_ids = [f"r048-eval-0048-{index:06d}" for index in range(EVAL_ROWS)]
    if [row.get("id") for row in rows] != expected_ids:
        raise ValueError("eval row identities/order are not the complete canonical sequence")
    groups = defaultdict(list)
    signatures = set()
    for row in rows:
        meta = row["meta"]
        groups[meta.get("semantic_group_id")].append(row)
        signatures.add(meta.get("semantic_signature"))
    if (None in groups or None in signatures or len(groups) != EVAL_CLUSTERS
            or len(signatures) != EVAL_CLUSTERS
            or any(len(items) != r048.CLUSTER_SIZE for items in groups.values())):
        raise ValueError("eval panel is not exactly 336 complete independent clusters")
    return rows, manifest


def validate_predictions(rows: list[dict[str, Any]],
                         predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Require exact coverage and recompute every claimed prediction field."""
    if not isinstance(predictions, list) or any(not isinstance(item, dict) for item in predictions):
        raise ValueError("predictions must be a list of objects")
    expected_ids = [row["id"] for row in rows]
    identifiers = [item.get("id") for item in predictions]
    if identifiers != expected_ids or len(set(identifiers)) != len(identifiers):
        raise ValueError("predictions must cover every panel row once in canonical order")
    for row, item in zip(rows, predictions):
        try:
            _json(item)
            result = scorer.classify(row, item.get("probabilities"))
            claims = {
                "predicted_outcome": result["predicted_outcome"].as_dict(),
                "gold_outcome": result["gold_outcome"].as_dict(),
                "predicted_trace": result["predicted_trace"],
                "error_category": result["error_category"],
                "complete_outcome_nll": result["complete_outcome_nll"],
                "categorical_brier": result["categorical_brier"],
                "top_outcome_confidence": result["top_outcome_confidence"],
                "top_outcome_correct": result["top_outcome_correct"],
                "gold_outcome_probability": result["gold_outcome_probability"],
            }
            for key, expected in claims.items():
                if key in item and _json(item[key]) != _json(expected):
                    raise ValueError(f"claimed {key} differs from recomputation")
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{row['id']}: invalid prediction: {error}") from error
    report = scorer.summarize(rows, predictions)
    if (report.get("rendered", {}).get("valid_predictions") != len(rows)
            or report["rendered"].get("invalid") != 0
            or report["rendered"].get("coverage") != 1.0):
        raise ValueError("scorer did not accept every prediction")
    return report


def validate_arm_completion(path: str | Path | None, expected_sha256: str | None,
                            *, adapter_path: str | Path, assets: dict[str, Any],
                            model_path: str | Path,
                            inventory_path: str | Path) -> dict[str, Any] | None:
    """Optionally authenticate the training receipt and its recipe/model lineage."""
    if (path is None) != (expected_sha256 is None):
        raise ValueError("arm completion and its external sha256 must be supplied together")
    if path is None:
        return None
    completion_path = Path(path)
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ValueError("arm completion must be a regular file")
    observed = file_hash(completion_path)
    if observed != _hash(expected_sha256, "expected arm completion sha256"):
        raise ValueError("arm completion sha256 mismatch")
    completion = _parse(completion_path.read_bytes(), "arm completion")
    if (not isinstance(completion, dict)
            or completion.get("schema") != ARM_COMPLETION_SCHEMA
            or completion.get("status") != "complete"):
        raise ValueError("arm completion receipt is not complete")
    checkpoint = completion.get("artifacts", {}).get("checkpoint")
    expected_checkpoint = {
        "path": str(Path(adapter_path).resolve()),
        "tree_sha256": assets["adapter_tree_sha256"],
        "files": assets["adapter_tree_files"],
        "adapter_sha256": assets["adapter_weights_sha256"],
        "adapter_config_sha256": assets["adapter_config_sha256"],
    }
    if checkpoint != expected_checkpoint:
        raise ValueError("arm completion does not bind the exact candidate checkpoint tree")
    recipe_path = completion.get("recipe_path")
    recipe_sha = completion.get("recipe_sha256")
    recipe, verified_recipe_sha = validate_recipe(recipe_path, recipe_sha)
    model = recipe.get("bindings", {}).get("model")
    if (not isinstance(model, dict) or not isinstance(model.get("path"), str)
            or Path(model["path"]).resolve() != Path(model_path).resolve()
            or model.get("revision") != FROZEN_BASE_REVISION
            or model.get("tree_sha256") != assets["base_model_tree_sha256"]
            or model.get("files") != assets["base_model_tree_files"]
            or not isinstance(model.get("inventory_path"), str)
            or Path(model["inventory_path"]).resolve() != Path(inventory_path).resolve()
            or model.get("inventory_sha256") != assets["model_inventory_sha256"]):
        raise ValueError("arm completion recipe does not bind the fixed base/inventory")
    if completion.get("recipe_sha256") != verified_recipe_sha:
        raise ValueError("arm completion recipe hash drift")
    return {"path": str(completion_path.resolve()), "sha256": observed,
            "arm_id": completion.get("arm_id"), "recipe_sha256": verified_recipe_sha}


def _code_hashes() -> dict[str, str]:
    names = (
        "run_r050_eval.py", "run_r050_headroom.py", "build_r048_action_panel.py",
        "score_exact_policy_panel.py", "compare_r049_calibrated.py",
        "singleton_option_scorer.py", "exact_policy_executor.py",
        "train_gemma_decision_lora.py", "build_r047_model_inventory.py",
        "preflight_r043_assets.py", "run_r050_arm.py",
    )
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise IOError(f"output read-back mismatch: {path.name}")


def publish(out: str | Path, predictions: list[dict[str, Any]], report: dict[str, Any],
            receipt: dict[str, Any]) -> dict[str, Any]:
    """Atomically publish a complete immutable directory with marker written last."""
    out = Path(out).absolute()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite full eval run: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        prediction_payload = b"".join(_json(item) + b"\n" for item in predictions)
        payloads = {"predictions.jsonl": prediction_payload,
                    "report.json": _json(report) + b"\n"}
        outputs = {name: {"sha256": _sha(payload), "bytes": len(payload)}
                   for name, payload in payloads.items()}
        sealed = {**receipt, "outputs": outputs,
                  "prediction_sha256": outputs["predictions.jsonl"]["sha256"]}
        payloads["receipt.json"] = _json(sealed) + b"\n"
        for name, payload in payloads.items():
            _write_new(stage / name, payload)
        marker = {
            "schema": COMPLETION_SCHEMA,
            "status": "complete",
            "receipt_sha256": _sha(payloads["receipt.json"]),
            "prediction_sha256": outputs["predictions.jsonl"]["sha256"],
            "report_sha256": outputs["report.json"]["sha256"],
        }
        _write_new(stage / "complete.json", _json(marker) + b"\n")
        _atomic_publish(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sealed


def run(data, manifest, model, inventory, adapter, out, *, expected_data_sha256,
        expected_manifest_sha256, expected_inventory_sha256,
        expected_adapter_tree_sha256, expected_adapter_tree_files,
        arm_completion=None, expected_arm_completion_sha256=None):
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite full eval run: {out}")
    data_payload = Path(data).read_bytes()
    manifest_payload = Path(manifest).read_bytes()
    rows, checked_manifest = validate_full_panel(
        data_payload, manifest_payload, expected_data_sha256, expected_manifest_sha256)
    if (_hash(expected_inventory_sha256, "expected model inventory sha256")
            != FROZEN_INVENTORY_SHA256):
        raise ValueError("expected model inventory sha256 is not the frozen literal anchor")
    asset_kwargs = {
        "expected_inventory_sha256": expected_inventory_sha256,
        "expected_adapter_tree_sha256": expected_adapter_tree_sha256,
        "expected_adapter_tree_files": expected_adapter_tree_files,
    }
    assets = trusted_runtime.verify_assets(model, inventory, adapter, **asset_kwargs)
    if (assets.get("base_model_tree_sha256") != FROZEN_BASE_TREE_SHA256
            or assets.get("base_model_tree_files") != FROZEN_BASE_TREE_FILES):
        raise ValueError("base model tree differs from the frozen literal anchor")
    completion = validate_arm_completion(
        arm_completion, expected_arm_completion_sha256, adapter_path=adapter, assets=assets,
        model_path=model, inventory_path=inventory)
    code_hashes = _code_hashes()
    predictions, runtime, histogram = trusted_runtime.infer_rows(rows, model, adapter)
    validate_runtime_fingerprint(runtime)
    forwards = sum(len(row["questions"]) for row in rows)
    if histogram != {"1": forwards}:
        raise ValueError("observed singleton forwards do not cover every field exactly once")
    report = validate_predictions(rows, predictions)
    report["evaluation_contract"] = {
        "role": "frozen_full_eval",
        "promotion_gate": "not_applied",
        "temperature_fitting": "not_performed",
        "checkpoint_selection": "not_performed",
    }
    if (file_hash(data) != _sha(data_payload)
            or file_hash(manifest) != _sha(manifest_payload)
            or _code_hashes() != code_hashes
            or trusted_runtime.verify_assets(model, inventory, adapter, **asset_kwargs) != assets
            or validate_arm_completion(
                arm_completion, expected_arm_completion_sha256, adapter_path=adapter,
                assets=assets, model_path=model, inventory_path=inventory) != completion):
        raise ValueError("panel, assets, completion lineage, or code changed during inference")
    receipt = {
        "schema": SCHEMA,
        "role": "frozen_full_eval",
        "panel_version": r048.PANEL_VERSION,
        "inputs": {
            "eval": {"path": str(Path(data).resolve()), "sha256": _sha(data_payload),
                     "rows": len(rows), "clusters": EVAL_CLUSTERS},
            "core_manifest": {"path": str(Path(manifest).resolve()),
                              "sha256": _sha(manifest_payload),
                              "outputs": checked_manifest["outputs"]},
        },
        "assets": assets,
        "arm_completion": completion,
        "code_sha256": code_hashes,
        "scorer_sha256": code_hashes["score_exact_policy_panel.py"],
        "comparator_schema": comparator.SCHEMA,
        "comparator_sha256": code_hashes["compare_r049_calibrated.py"],
        "runtime_fingerprint": runtime,
        "runtime_fingerprint_sha256": _sha(_json(runtime)),
        "execution": {
            "mode": "singleton_scalar", "max_length": MAX_LENGTH,
            "batch_histogram": histogram, "expected_field_forwards": forwards,
            "base_dtype": "bfloat16", "attention": "sdpa",
            "temperature_fitting": False, "promotion_gate_applied": False,
        },
    }
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
    parser.add_argument("--arm-completion", type=Path)
    parser.add_argument("--expected-arm-completion-sha256")
    args = vars(parser.parse_args(argv))
    report, receipt = run(**args)
    print(json.dumps({"rendered": report["rendered"],
                      "prediction_sha256": receipt["prediction_sha256"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
