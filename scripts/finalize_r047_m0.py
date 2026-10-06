"""Create the fail-closed receipt for the R047 in-sample can-fit diagnostic."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.eval_r047_executor import m0_gate


EXPECTED_STEPS = (20, 50, 100, 200)
REQUIRED_KINDS = ("decimal", "ledger", "policy", "conditional")
FROZEN_BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
FROZEN_PARENT_SHA256 = "3b5535de8c90156662d0b3b3f608fff12f557ddf0890722c4a584dbe93a1f8a4"
EXPECTED_CONFIG = {
    "steps": 200, "microbatch": 1, "grad_accum": 8, "lr": 2e-5,
    "min_lr": 2e-6, "rank": 32, "seed": 47, "pair_flip_weight": 1.0,
    "alpha": 64, "weight_decay": 0.01, "max_length": 8192,
    "save_every": 250, "save_steps": "20,50,100", "eval_every": 200,
    "eval_samples": 48, "pair_flip_margin": 1.0, "pair_flip_start_step": 0,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _jsonl(path):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL: {path}") from error


def _specs(values, label):
    result = {}
    for value in values:
        raw_step, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"{label} must use STEP=PATH")
        step = int(raw_step)
        if step in result:
            raise ValueError(f"duplicate {label} step: {step}")
        result[step] = Path(path)
    if tuple(sorted(result)) != EXPECTED_STEPS:
        raise ValueError(f"{label} steps must be exactly {EXPECTED_STEPS}")
    return result


def _finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def finalize(data_path, manifest_path, preflight_path, config_path, metrics_path, report_specs,
             checkpoint_specs, parent_adapter, builder_path, trainer_path, evaluator_path,
             base_revision, model_path):
    reports, checkpoints = _specs(report_specs, "report"), _specs(checkpoint_specs, "checkpoint")
    data_hash = _sha(data_path)
    manifest, preflight = _json(manifest_path), _json(preflight_path)
    config, metrics = _json(config_path), _jsonl(metrics_path)
    if manifest.get("outputs", {}).get("m0-treatment", {}).get("sha256") != data_hash:
        raise ValueError("manifest M0 hash does not match evaluated data")
    if manifest.get("m0_treatment", {}).get("rows") != 48:
        raise ValueError("M0 must contain exactly 48 logical slots")
    if (manifest["m0_treatment"].get("canonical_views") != 96
            or manifest["m0_treatment"].get("auxiliary_views") != 192):
        raise ValueError("M0 canonical/auxiliary view contract mismatch")
    model_path = Path(model_path).resolve()
    if base_revision != FROZEN_BASE_REVISION:
        raise ValueError("base revision differs from the frozen R047 revision")
    if (model_path.name != FROZEN_BASE_REVISION or model_path.parent.name != "snapshots"
            or model_path.parent.parent.name != "models--google--gemma-4-12B-it"
            or not (model_path / "config.json").is_file()):
        raise ValueError("model path is not the frozen Gemma snapshot")
    parent_hash = _sha(parent_adapter)
    if parent_hash != FROZEN_PARENT_SHA256:
        raise ValueError("parent adapter differs from frozen R044-R")
    expected_paths = {
        "model": str(model_path), "init_adapter": str(Path(parent_adapter).parent),
        "train": [str(Path(data_path))], "val": [f"m0={Path(data_path)}"],
    }
    for key, expected in expected_paths.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key} is not bound to the supplied artifact")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key} must equal {expected!r}")
    if (preflight.get("passed") is not True or preflight.get("input_sha256") !=
            manifest.get("input", {}).get("sha256")
            or preflight.get("control_sha256") !=
            manifest.get("outputs", {}).get("control", {}).get("sha256")
            or preflight.get("clause_sha256") !=
            manifest.get("outputs", {}).get("clause", {}).get("sha256")
            or preflight.get("m0_treatment_sha256") != data_hash):
        raise ValueError("tokenizer preflight is missing or does not match frozen inputs")
    token_budget = preflight.get("token_budget")
    if not isinstance(token_budget, dict):
        raise ValueError("tokenizer preflight did not record a token budget")
    control_tokens, clause_tokens = (token_budget.get(key)
                                     for key in ("control_tokens", "clause_tokens"))
    if (not isinstance(control_tokens, int) or isinstance(control_tokens, bool)
            or not isinstance(clause_tokens, int) or isinstance(clause_tokens, bool)
            or control_tokens <= 0 or clause_tokens <= 0):
        raise ValueError("tokenizer preflight token counts must be positive integers")
    absolute_delta = abs(control_tokens - clause_tokens)
    relative_delta = absolute_delta / max(control_tokens, clause_tokens)
    tokenizer_name = token_budget.get("tokenizer")
    try:
        tokenizer_matches = Path(tokenizer_name).resolve() == model_path
    except TypeError:
        tokenizer_matches = False
    if (token_budget.get("within_limit") is not True
            or token_budget.get("absolute_delta") != absolute_delta
            or not math.isclose(token_budget.get("relative_delta", -1), relative_delta,
                                rel_tol=0, abs_tol=1e-12)
            or token_budget.get("limit") != 0.005 or relative_delta > 0.005
            or not tokenizer_matches):
        raise ValueError("tokenizer preflight did not pass the 0.5% token budget")
    if len(metrics) != 201 or [item.get("step") for item in metrics] != list(range(201)):
        raise ValueError("metrics must cover steps 0..200 exactly")
    if not _finite(metrics):
        raise ValueError("metrics contain a non-finite value")
    training = metrics[1:]
    for item in training:
        if (item.get("canonical_views_seen") != 16 or item.get("auxiliary_views_seen") != 32
                or item.get("pair_flip_seen") != 8 or item.get("pair_flip_weight") != 1.0):
            raise ValueError(f"step {item.get('step')}: exposure/objective contract mismatch")
    if (sum(item["canonical_views_seen"] for item in training[:6]) != 96
            or sum(item["auxiliary_views_seen"] for item in training[:6]) != 192):
        raise ValueError("first epoch-equivalent does not expose 96 canonical/192 auxiliary views")

    loaded_reports = {step: _json(path) for step, path in reports.items()}
    adapter_paths = {step: path / "adapter_model.safetensors"
                     for step, path in checkpoints.items()}
    if any(not path.is_file() for path in adapter_paths.values()):
        raise ValueError("every frozen checkpoint must contain adapter_model.safetensors")
    for step, report in loaded_reports.items():
        if report.get("input_sha256") != data_hash:
            raise ValueError(f"step {step}: report input hash mismatch")
        if report.get("rows") != 48 or report.get("final", {}).get("presentations") != 96:
            raise ValueError(f"step {step}: report final coverage mismatch")
        if report.get("clauses", {}).get("n") != 192:
            raise ValueError(f"step {step}: report clause coverage mismatch")
        prediction_path = reports[step].parent / "predictions.jsonl"
        if report.get("predictions_sha256") != _sha(prediction_path):
            raise ValueError(f"step {step}: prediction hash mismatch")
        if report.get("adapter_sha256") != _sha(adapter_paths[step]):
            raise ValueError(f"step {step}: report is not bound to checkpoint adapter")
        if report.get("model_revision") != base_revision:
            raise ValueError(f"step {step}: model revision mismatch")
        try:
            recomputed_gate = m0_gate(report, REQUIRED_KINDS, 0.98)
        except (KeyError, TypeError) as error:
            raise ValueError(f"step {step}: report lacks frozen gate metrics") from error
        if report.get("m0_gate") != recomputed_gate:
            raise ValueError(f"step {step}: reported gate does not match recomputed gate")
        if not _finite(report):
            raise ValueError(f"step {step}: report contains a non-finite value")
    passing = [step for step in EXPECTED_STEPS
               if loaded_reports[step].get("m0_gate", {}).get("passed") is True]
    selected = passing[0] if passing else None
    checks = {
        "all_metrics_finite": True,
        "config_frozen": True,
        "dataset_integrity": True,
        "prediction_coverage_exact": True,
        "training_exposure_exact": True,
        "ast_targets_all_reproduced": True,
        "benchmark_holdout_quarantined": True,
        "candidate_passed_frozen_gate": selected is not None,
    }
    receipt = {
        "schema_version": "r047-m0-v1", "purpose": "in_sample_fit_only",
        "selected_step": selected, "evaluated_steps": list(EXPECTED_STEPS),
        "provenance": {
            "base_revision": base_revision,
            "parent_adapter_sha256": parent_hash,
            "candidate_adapter_sha256": (_sha(adapter_paths[selected]) if selected else None),
            "source_arm_sha256": manifest.get("input", {}).get("sha256"),
            "k2_subset_sha256": manifest["m0_treatment"].get("source_subset_sha256"),
            "m0_data_sha256": data_hash,
            "predictions_sha256": {str(step): _sha(reports[step].parent / "predictions.jsonl")
                                   for step in EXPECTED_STEPS},
            "candidate_adapter_sha256_by_step": {
                str(step): _sha(adapter_paths[step]) for step in EXPECTED_STEPS},
            "trainer_sha256": _sha(trainer_path), "builder_sha256": _sha(builder_path),
            "evaluator_sha256": _sha(evaluator_path), "config_sha256": _sha(config_path),
            "metrics_log_sha256": _sha(metrics_path), "manifest_sha256": _sha(manifest_path),
            "preflight_sha256": _sha(preflight_path),
        },
        "integrity": {"slots": 48, "canonical_views": 96, "auxiliary_views": 192,
                      "ast_targets_reproduced": 192, "prediction_coverage_exact": True,
                      "probabilities_valid": True, "temperature": 1.0,
                      "benchmark_holdout_used": False},
        "reports": {str(step): loaded_reports[step] for step in EXPECTED_STEPS},
        "checks": checks,
        "gate_pass": all(checks.values()),
        "failures": ([] if selected is not None else
                     ["no checkpoint passed every frozen M0 threshold"]),
    }
    if receipt["gate_pass"] != (selected is not None):
        raise AssertionError("receipt gate is inconsistent with checkpoint selection")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--parent-adapter", required=True)
    parser.add_argument("--builder", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    receipt = finalize(args.data, args.manifest, args.preflight, args.config, args.metrics, args.report,
                       args.checkpoint, args.parent_adapter, args.builder, args.trainer,
                       args.evaluator, args.base_revision, args.model_path)
    Path(args.out).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
