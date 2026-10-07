"""Run one frozen R050 arm and seal its artifacts with a completion receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, verify_inventory
from scripts.preflight_r043_assets import file_hash, tree_hash
from scripts.prepare_r050_arms import (
    EXPECTED_SLOTS,
    EXPECTED_STRATA,
    SCHEMA,
    TRAINING,
    _canonical,
    _reject_constant,
    _same_number,
    _unique_object,
    panel_factor_cells_sha256,
    read_json,
    require_hash,
    validate_mix,
    validate_parent,
    validate_probe,
)


COMPLETION_SCHEMA = "r050-matched-arm-completion-v1"


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    return False


def _same_path(value: Any, expected: str | Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == Path(expected).resolve()


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (set(left) == set(right)
                and all(_strict_equal(left[key], right[key]) for key in left))
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_recipe(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    path, digest = require_hash(path, expected_sha256, "arm recipe")
    recipe, parsed = read_json(path, "arm recipe")
    if parsed != digest or set(recipe) != {
            "schema", "arm_id", "run_order", "objective", "training",
            "epoch_contract", "bindings"} or recipe.get("schema") != SCHEMA:
        raise ValueError("arm recipe schema is invalid")
    if (not isinstance(recipe.get("arm_id"), str) or not recipe["arm_id"].startswith("slot_")
            or type(recipe.get("run_order")) is not int or recipe["run_order"] not in {1, 2, 3}
            or not _strict_equal(recipe.get("training"), TRAINING)):
        raise ValueError("arm identity or matched training configuration is invalid")
    epoch = recipe.get("epoch_contract")
    if epoch != {
            "optimizer_slots": EXPECTED_SLOTS,
            "slots_per_update": TRAINING["microbatch"] * TRAINING["grad_accum"],
            "updates": TRAINING["steps"], "passes": 1,
            "slot_order": "trainer seeded global shuffle"}:
        raise ValueError("arm recipe does not consume the mix exactly once")
    objective = recipe.get("objective")
    if not isinstance(objective, dict) or set(objective) != {
            "objective", "action_loss_weight", "exact_policy_outcome_loss"}:
        raise ValueError("arm objective schema is invalid")
    valid_names = {
        "field_ce": (0.0, "nll"),
        "field_ce_plus_exact_outcome_nll": (None, "nll"),
        "field_ce_plus_exact_outcome_brier": (None, "brier"),
    }
    if objective.get("objective") not in valid_names:
        raise ValueError("unknown arm objective")
    expected_weight, expected_kind = valid_names[objective["objective"]]
    weight = objective.get("action_loss_weight")
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0
            or objective.get("exact_policy_outcome_loss") != expected_kind
            or (expected_weight is None and weight <= 0)
            or (expected_weight is not None and not _same_number(weight, expected_weight))):
        raise ValueError("wrong action objective in arm recipe")
    if not isinstance(recipe.get("bindings"), dict):
        raise ValueError("arm recipe bindings are missing")
    return recipe, digest


def verify_bindings(recipe: dict[str, Any]) -> None:
    bindings = recipe["bindings"]
    required = {"mix", "model", "gradient_probe", "training_panel", "panel_manifest",
                "panel_builder", "parent_adapter", "trainer", "gradient_probe_code"}
    if set(bindings) != required or any(not isinstance(bindings[name], dict) for name in required):
        raise ValueError("arm recipe input binding set is invalid")
    mix_binding = bindings["mix"]
    mix = validate_mix(
        mix_binding.get("path"), mix_binding.get("manifest_path"),
        mix_binding.get("sha256"), mix_binding.get("manifest_sha256"))
    if mix != mix_binding:
        raise ValueError("training mix binding drift")
    panel_binding = bindings["training_panel"]
    panel, panel_sha = require_hash(
        panel_binding.get("path"), panel_binding.get("sha256"), "training panel")
    if mix["executable_factor_cells_sha256"] != panel_factor_cells_sha256(panel):
        raise ValueError("training mix no longer covers the complete panel design")
    if str(panel) != panel_binding.get("path"):
        raise ValueError("training panel path drift")
    panel_manifest_binding = bindings["panel_manifest"]
    panel_manifest, panel_manifest_sha = require_hash(
        panel_manifest_binding.get("path"), panel_manifest_binding.get("sha256"),
        "panel manifest")
    builder_binding = bindings["panel_builder"]
    builder, builder_sha = require_hash(
        builder_binding.get("path"), builder_binding.get("sha256"), "panel builder")
    trainer_binding = bindings["trainer"]
    trainer, trainer_sha = require_hash(
        trainer_binding.get("path"), trainer_binding.get("sha256"), "trainer")
    probe_binding = bindings["gradient_probe_code"]
    probe, probe_sha = require_hash(
        probe_binding.get("path"), probe_binding.get("sha256"), "gradient probe")
    model_binding = bindings["model"]
    inventory_path, inventory_sha = require_hash(
        model_binding.get("inventory_path"), model_binding.get("inventory_sha256"),
        "model inventory")
    model_path = Path(model_binding.get("path", "")).resolve()
    inventory, verified_sha = verify_inventory(inventory_path, model_path, FROZEN_BASE_REVISION)
    expected_model = {"path": str(model_path), **inventory,
                      "inventory_path": str(inventory_path),
                      "inventory_sha256": inventory_sha}
    if verified_sha != inventory_sha or model_binding != expected_model:
        raise ValueError("base model binding drift")
    parent_binding = bindings["parent_adapter"]
    parent = validate_parent(parent_binding.get("path"), parent_binding.get("tree_sha256"),
                             parent_binding.get("files"))
    if parent != parent_binding:
        raise ValueError("parent adapter binding drift")
    probe_receipt_binding = bindings["gradient_probe"]
    _, coefficients, receipt_sha = validate_probe(
        probe_receipt_binding.get("path"), probe_receipt_binding.get("sha256"),
        panel=panel, panel_sha=panel_sha, panel_manifest=panel_manifest,
        panel_manifest_sha=panel_manifest_sha, builder=builder, builder_sha=builder_sha,
        inventory_path=inventory_path, inventory_sha=inventory_sha, inventory=inventory,
        parent=parent, trainer=trainer, trainer_sha=trainer_sha, probe=probe,
        probe_sha=probe_sha)
    if receipt_sha != probe_receipt_binding.get("sha256"):
        raise ValueError("gradient receipt binding drift")
    objective = recipe["objective"]
    expected = {
        "field_ce": 0.0,
        "field_ce_plus_exact_outcome_nll": coefficients["lambda_nll"],
        "field_ce_plus_exact_outcome_brier": coefficients["lambda_brier"],
    }[objective["objective"]]
    if not _same_number(objective["action_loss_weight"], expected):
        raise ValueError("arm coefficient differs from the frozen gradient probe")


def trainer_config(recipe: dict[str, Any], output: Path) -> dict[str, Any]:
    bindings, objective = recipe["bindings"], recipe["objective"]
    return {
        "model": bindings["model"]["path"],
        "init_adapter": bindings["parent_adapter"]["path"],
        "train": [bindings["mix"]["path"]],
        "val": [],
        "out": str(output.resolve()),
        **recipe["training"],
        "action_loss_weight": objective["action_loss_weight"],
        "exact_policy_outcome_loss": objective["exact_policy_outcome_loss"],
    }


def trainer_command(recipe: dict[str, Any], output: Path) -> list[str]:
    config = trainer_config(recipe, output)
    command = [sys.executable, recipe["bindings"]["trainer"]["path"],
               "--model", config["model"], "--init-adapter", config["init_adapter"],
               "--train", config["train"][0], "--out", config["out"]]
    names = (
        "steps", "microbatch", "grad_accum", "max_length", "lr", "min_lr",
        "weight_decay", "rank", "alpha", "save_every", "save_steps", "eval_every",
        "eval_samples", "seed", "pair_consistency_weight",
        "pair_consistency_min_gold_probability", "pair_consistency_clip",
        "pair_consistency_beta", "pair_flip_weight", "pair_flip_margin",
        "pair_flip_start_step", "action_loss_weight", "action_temperature",
        "exact_policy_outcome_loss",
    )
    for name in names:
        command.extend((f"--{name.replace('_', '-')}", str(config[name])))
    if config["project_consistency_against_guard"]:
        command.append("--project-consistency-against-guard")
    if config["stage_counterfactual_pairs"]:
        command.append("--stage-counterfactual-pairs")
    return command


def validate_config(path: Path, expected: dict[str, Any]) -> str:
    value, digest = read_json(path, "trainer config")
    if set(value) != set(expected):
        raise ValueError("trainer config keys drifted from the arm recipe")
    for name, expected_value in expected.items():
        actual = value[name]
        if name in {"model", "init_adapter", "out"}:
            matched = _same_path(actual, expected_value)
        elif name == "train":
            matched = (isinstance(actual, list) and len(actual) == 1
                       and _same_path(actual[0], expected_value[0]))
        elif type(expected_value) in {int, float} and not isinstance(expected_value, bool):
            matched = _same_number(actual, expected_value)
        else:
            matched = type(actual) is type(expected_value) and actual == expected_value
        if not matched:
            raise ValueError(f"trainer config drift: {name}")
    return digest


def validate_metrics(path: Path, recipe: dict[str, Any]) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("metrics must be a nonempty regular file")
    rows = []
    try:
        for line in path.read_bytes().splitlines():
            if line.strip():
                value = json.loads(line, object_pairs_hook=_unique_object,
                                   parse_constant=_reject_constant)
                if not isinstance(value, dict) or not _finite(value):
                    raise ValueError("metrics contain a nonfinite or invalid value")
                rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid metrics JSONL") from error
    if len(rows) != TRAINING["steps"] + 1 or [row.get("step") for row in rows] != list(
            range(TRAINING["steps"] + 1)):
        raise ValueError("metrics do not contain contiguous steps 0 through 400")
    if rows[0].get("validation") != {}:
        raise ValueError("training validation channel must remain empty")
    objective = recipe["objective"]
    executable_seen = 0
    for row in rows[1:]:
        if (isinstance(row.get("train_loss"), bool)
                or not isinstance(row.get("train_loss"), (int, float))
                or not _same_number(row.get("action_loss_weight"),
                                    objective["action_loss_weight"])
                or row.get("exact_policy_outcome_loss") != objective["exact_policy_outcome_loss"]
                or row.get("action_loss_kind") != objective["exact_policy_outcome_loss"]
                or type(row.get("action_groups_seen")) is not int
                or row["action_groups_seen"] < 0):
            raise ValueError("metrics report the wrong or incomplete action objective")
        executable_seen += row["action_groups_seen"]
    if executable_seen != EXPECTED_STRATA["executable"]:
        raise ValueError("metrics do not account for all 1,600 executable slots exactly once")
    return file_hash(path), len(rows)


def validate_checkpoint(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir() or path.name != "step-000400":
        raise ValueError("exact step-400 checkpoint is absent")
    if any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("checkpoint cannot contain symbolic links")
    required = {"adapter_model.safetensors", "adapter_config.json", "trainer.pt"}
    if any(not (path / name).is_file() or (path / name).stat().st_size <= 0 for name in required):
        raise ValueError("step-400 checkpoint is missing a required nonempty payload")
    digest, files = tree_hash(path)
    return {
        "path": str(path.resolve()), "tree_sha256": digest, "files": files,
        "adapter_sha256": file_hash(path / "adapter_model.safetensors"),
        "adapter_config_sha256": file_hash(path / "adapter_config.json"),
    }


def _atomic_new(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite completion receipt: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def run(recipe_path: str | Path, expected_recipe_sha256: str,
        out: str | Path) -> dict[str, Any]:
    recipe, recipe_sha = validate_recipe(recipe_path, expected_recipe_sha256)
    verify_bindings(recipe)
    output = Path(out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending_log = output.parent / f".{output.name}.{recipe['arm_id']}.train.log"
    if pending_log.exists():
        raise FileExistsError(f"refusing to overwrite pending training log: {pending_log}")
    command = trainer_command(recipe, output)
    environment = os.environ.copy()
    environment.update({"PYTHONHASHSEED": str(TRAINING["seed"]),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    with pending_log.open("xb") as log:
        completed = subprocess.run(
            command, cwd=str(Path(recipe["bindings"]["trainer"]["path"]).parent.parent),
            env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise RuntimeError(
            f"trainer exited with status {completed.returncode}; log preserved at {pending_log}")
    if not output.is_dir():
        raise ValueError("trainer returned success without creating its output directory")
    log_path = output / "train.log"
    if log_path.exists():
        raise ValueError("trainer unexpectedly created the reserved training log path")
    try:
        os.link(pending_log, log_path)
    except FileExistsError as error:
        raise ValueError("refusing to overwrite the training log") from error
    pending_log.unlink()
    expected_config = trainer_config(recipe, output)
    config_sha = validate_config(output / "config.json", expected_config)
    metrics_sha, metrics_records = validate_metrics(output / "metrics.jsonl", recipe)
    checkpoint = validate_checkpoint(output / "step-000400")
    if log_path.stat().st_size <= 0:
        raise ValueError("training log is empty")
    completion_path = output / "completion.json"
    if completion_path.exists():
        raise ValueError("completion receipt already exists")
    receipt = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "arm_id": recipe["arm_id"],
        "run_order": recipe["run_order"],
        "recipe_path": str(Path(recipe_path).resolve()),
        "recipe_sha256": recipe_sha,
        "command_sha256": hashlib.sha256(_canonical(command)).hexdigest(),
        "objective": recipe["objective"],
        "epoch_contract": recipe["epoch_contract"],
        "artifacts": {
            "config": {"path": str((output / "config.json").resolve()),
                       "sha256": config_sha},
            "metrics": {"path": str((output / "metrics.jsonl").resolve()),
                        "sha256": metrics_sha, "records": metrics_records,
                        "final_step": TRAINING["steps"]},
            "log": {"path": str(log_path.resolve()), "sha256": file_hash(log_path)},
            "checkpoint": checkpoint,
        },
    }
    _atomic_new(completion_path, _canonical(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    receipt = run(args.recipe, args.expected_recipe_sha256, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
