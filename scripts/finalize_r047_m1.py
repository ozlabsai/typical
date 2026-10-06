"""Emit a fail-closed, hash-bound receipt for one blinded R047-M1 run."""

import argparse
import json
import math
from pathlib import Path

from scripts.build_r045_arms import _valid_row
from scripts.build_r047_clause_arms import _summary
from scripts.build_r047_model_inventory import (
    FROZEN_BASE_REVISION,
    verify_inventory,
)
from scripts.preflight_r043_assets import file_hash, tree_hash


SCHEMA_VERSION = "r047-causal-run-receipt-v1"
RECIPE_SCHEMA_VERSION = "r047-m1-recipe-v1"
SLOTS = {"slot_a", "slot_b"}
FROZEN_PARENT_SHA256 = "3b5535de8c90156662d0b3b3f608fff12f557ddf0890722c4a584dbe93a1f8a4"
FROZEN_ARM_MANIFEST_SHA256 = "c18214db7befdfb93aa818b445bf9606a5f4333a2b7f0024cb802e8399460cef"
KNOWN_ARM_SHA256 = {
    "control": "179b6497fb6050dc5974b2545b1fed198fc0c00b200481e6e29c5bf1f8e27694",
    "treatment": "10366cc7656a89331ff85fd88e304637f5487f82b50f7ccd69be30436357a586",
}
EXPECTED_STEPS = 400
EPOCH_STEPS = 100
EXPECTED_EPOCHS = 4
EXPECTED_SLOTS_PER_EPOCH = 800
EXPECTED_ARM = {
    "logical_rows": 800,
    "base_presentations": 900,
    "pair_slots": 100,
    "augmented_rows": 48,
    "auxiliary_views": 192,
    "effective_presentations": 1092,
}
EXPECTED_EPOCH_EXPOSURE = {
    "canonical_views_seen": 900,
    "auxiliary_views_seen": 192,
    "pair_flip_seen": 100,
}
PATH_CONFIG_KEYS = {"model", "init_adapter", "train", "val", "out"}
TRAINER_CONFIG_KEYS = {
    "steps", "microbatch", "grad_accum", "max_length", "lr", "min_lr",
    "weight_decay", "rank", "alpha", "save_every", "save_steps", "eval_every",
    "eval_samples", "seed", "pair_consistency_weight",
    "pair_consistency_min_gold_probability", "pair_consistency_clip",
    "pair_consistency_beta", "project_consistency_against_guard", "pair_flip_weight",
    "pair_flip_margin", "pair_flip_start_step", "stage_counterfactual_pairs",
}
RECIPE_FIELDS = {
    "schema_version", "base_revision", "parent_adapter_sha256",
    "model_inventory_sha256", "arm_manifest_sha256", "arm_sha256", "trainer_sha256",
    "trainer_config",
}
FROZEN_TRAINING_INVARIANTS = {
    "steps": EXPECTED_STEPS,
    "microbatch": 1,
    "grad_accum": 8,
    "max_length": 8192,
    "lr": 5e-6,
    "min_lr": 5e-7,
    "weight_decay": 0.01,
    "rank": 32,
    "alpha": 64,
    "save_every": 400,
    "save_steps": "",
    "eval_every": 400,
    "eval_samples": 128,
    "seed": 47,
    "pair_consistency_weight": 0.0,
    "pair_consistency_min_gold_probability": 0.8,
    "pair_consistency_clip": 4.0,
    "pair_consistency_beta": 1.0,
    "project_consistency_against_guard": False,
    "pair_flip_weight": 1.0,
    "pair_flip_margin": 1.0,
    "pair_flip_start_step": 0,
    "stage_counterfactual_pairs": False,
}
INTEGER_CONFIG_KEYS = {
    "steps", "microbatch", "grad_accum", "max_length", "rank", "alpha", "save_every",
    "eval_every", "eval_samples", "seed", "pair_flip_start_step",
}
FLOAT_CONFIG_KEYS = {
    "lr", "min_lr", "weight_decay", "pair_consistency_weight",
    "pair_consistency_min_gold_probability", "pair_consistency_clip",
    "pair_consistency_beta", "pair_flip_weight", "pair_flip_margin",
}
BOOLEAN_CONFIG_KEYS = {"project_consistency_against_guard", "stage_counterfactual_pairs"}


def _json(path, label):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _jsonl(path, label):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSONL") from error


def _same(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def _finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    return False


def _same_path(left, right):
    return isinstance(left, str) and Path(left).resolve() == Path(right).resolve()


def _nonempty_file_hash(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return file_hash(path)


def _validate_train(train_path, manifest_path):
    train_hash = _nonempty_file_hash(train_path, "train payload")
    if train_hash not in set(KNOWN_ARM_SHA256.values()):
        raise ValueError("train payload is not either frozen R047 arm")
    if file_hash(manifest_path) != FROZEN_ARM_MANIFEST_SHA256:
        raise ValueError("arm manifest differs from the frozen R047 manifest")
    manifest = _json(manifest_path, "arm manifest")
    arm = next(name for name, digest in KNOWN_ARM_SHA256.items() if digest == train_hash)
    output = (manifest.get("outputs") or {}).get("clause" if arm == "treatment" else arm)
    summary = (manifest.get("arms") or {}).get("clause" if arm == "treatment" else arm)
    if (not isinstance(output, dict) or output.get("sha256") != train_hash
            or output.get("rows") != EXPECTED_ARM["logical_rows"]
            or not isinstance(summary, dict)
            or any(summary.get(key) != value for key, value in EXPECTED_ARM.items())):
        raise ValueError("arm manifest does not bind the selected frozen arm")
    rows = _jsonl(train_path, "train")
    if (len(rows) != EXPECTED_ARM["logical_rows"]
            or any(not _valid_row(row) for row in rows)
            or any(_summary(rows).get(key) != value for key, value in EXPECTED_ARM.items())):
        raise ValueError("train payload does not satisfy the frozen 800-slot arm contract")
    return train_hash


def _validate_recipe(recipe_path, model_inventory_sha256, trainer_path):
    recipe = _json(recipe_path, "recipe")
    if (set(recipe) != RECIPE_FIELDS
            or recipe.get("schema_version") != RECIPE_SCHEMA_VERSION
            or recipe.get("base_revision") != FROZEN_BASE_REVISION
            or recipe.get("parent_adapter_sha256") != FROZEN_PARENT_SHA256
            or recipe.get("model_inventory_sha256") != model_inventory_sha256
            or recipe.get("arm_manifest_sha256") != FROZEN_ARM_MANIFEST_SHA256
            or recipe.get("arm_sha256") != KNOWN_ARM_SHA256
            or recipe.get("trainer_sha256") != _nonempty_file_hash(
                trainer_path, "trainer implementation")):
        raise ValueError("recipe provenance differs from the frozen R047-M1 contract")
    config = recipe.get("trainer_config")
    if not isinstance(config, dict) or set(config) != TRAINER_CONFIG_KEYS:
        raise ValueError("recipe trainer_config does not exactly cover trainer hyperparameters")
    if (any(type(config[key]) is not int for key in INTEGER_CONFIG_KEYS)
            or any(type(config[key]) not in (int, float) or not math.isfinite(config[key])
                   for key in FLOAT_CONFIG_KEYS)
            or any(type(config[key]) is not bool for key in BOOLEAN_CONFIG_KEYS)
            or not isinstance(config["save_steps"], str)):
        raise ValueError("recipe trainer_config contains invalid value types")
    if any(not _same(config.get(key), value)
           for key, value in FROZEN_TRAINING_INVARIANTS.items()):
        raise ValueError("recipe violates a frozen R047-M1 training invariant")
    if EPOCH_STEPS * config["microbatch"] * config["grad_accum"] != EXPECTED_SLOTS_PER_EPOCH:
        raise ValueError("recipe does not provide four exact 800-slot epochs")
    if not _finite(recipe):
        raise ValueError("recipe contains a non-finite value")
    return recipe, file_hash(recipe_path)


def _validate_config(config_path, recipe, train_path, parent_adapter, model_path,
                     checkpoint_path):
    config = _json(config_path, "trainer config")
    if set(config) != PATH_CONFIG_KEYS | TRAINER_CONFIG_KEYS:
        raise ValueError("trainer config keys differ from the frozen trainer schema")
    if any(not _same(config.get(key), value)
           for key, value in recipe["trainer_config"].items()):
        raise ValueError("trainer config differs from the hash-bound recipe")
    trains = config.get("train")
    if (not isinstance(trains, list) or len(trains) != 1
            or not _same_path(trains[0], train_path)):
        raise ValueError("trainer config is not bound to the supplied train payload")
    if (not _same_path(config.get("model"), model_path)
            or not _same_path(config.get("init_adapter"), Path(parent_adapter).parent)
            or not _same_path(config.get("out"), Path(checkpoint_path).parent)):
        raise ValueError("trainer config artifact paths do not match supplied artifacts")
    validation = config.get("val")
    if not isinstance(validation, list) or any(
            not isinstance(item, str) or "=" not in item or not item.split("=", 1)[0]
            or not Path(item.split("=", 1)[1]).is_file() for item in validation):
        raise ValueError("trainer config validation inputs are invalid")
    return file_hash(config_path)


def _validate_metrics(metrics_path):
    metrics = _jsonl(metrics_path, "metrics")
    if (len(metrics) != EXPECTED_STEPS + 1
            or [item.get("step") if isinstance(item, dict) else None for item in metrics]
            != list(range(EXPECTED_STEPS + 1))):
        raise ValueError("metrics must cover contiguous steps 0..400 exactly")
    if not _finite(metrics):
        raise ValueError("metrics contain a non-finite value")
    training = metrics[1:]
    for item in training:
        step = item["step"]
        if "early_stop" in item:
            raise ValueError(f"step {step}: early-stop evidence is forbidden")
        for key in EXPECTED_EPOCH_EXPOSURE:
            value = item.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"step {step}: {key} must be a nonnegative integer")
        if (not _same(item.get("pair_flip_weight"), 1.0)
                or not _same(item.get("pair_flip_weight_configured"), 1.0)):
            raise ValueError(f"step {step}: pair-flip objective differs from recipe")
    for epoch in range(EXPECTED_EPOCHS):
        window = training[epoch * EPOCH_STEPS:(epoch + 1) * EPOCH_STEPS]
        for key, expected in EXPECTED_EPOCH_EXPOSURE.items():
            if sum(item[key] for item in window) != expected:
                raise ValueError(f"epoch {epoch + 1}: {key} exposure must equal {expected}")
    return file_hash(metrics_path)


def _validate_checkpoint(checkpoint_path):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir() or checkpoint.name != "step-000400":
        raise ValueError("final checkpoint must be the exact step-000400 directory")
    required = {"adapter_model.safetensors", "adapter_config.json", "trainer.pt"}
    if any(not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size <= 0
           for name in required):
        raise ValueError("final checkpoint is missing a required nonempty payload")
    if any(item.is_symlink() for item in checkpoint.rglob("*")):
        raise ValueError("final checkpoint must not contain symbolic links")
    payload_hash, _ = tree_hash(checkpoint)
    return file_hash(checkpoint / "adapter_model.safetensors"), payload_hash


def finalize(*, slot, train_path, arm_manifest_path, recipe_path, config_path, metrics_path,
             log_path, checkpoint_path, parent_adapter, model_inventory_path, trainer_path,
             base_revision, model_path):
    if slot not in SLOTS:
        raise ValueError("R047-M1 slot must be slot_a or slot_b")
    if base_revision != FROZEN_BASE_REVISION:
        raise ValueError("base revision differs from the frozen R047 revision")
    _, inventory_hash = verify_inventory(model_inventory_path, model_path, base_revision)
    parent_hash = _nonempty_file_hash(parent_adapter, "parent adapter")
    if parent_hash != FROZEN_PARENT_SHA256:
        raise ValueError("parent adapter differs from frozen R044-R")
    train_hash = _validate_train(train_path, arm_manifest_path)
    recipe, recipe_hash = _validate_recipe(recipe_path, inventory_hash, trainer_path)
    config_hash = _validate_config(
        config_path, recipe, train_path, parent_adapter, model_path, checkpoint_path)
    metrics_hash = _validate_metrics(metrics_path)
    log_hash = _nonempty_file_hash(log_path, "training log")
    adapter_hash, checkpoint_hash = _validate_checkpoint(checkpoint_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "slot": slot,
        "status": "complete",
        "checkpoint_step": EXPECTED_STEPS,
        "train_sha256": train_hash,
        "recipe_sha256": recipe_hash,
        "parent_adapter_sha256": parent_hash,
        "model_revision": base_revision,
        "model_inventory_sha256": inventory_hash,
        "adapter_sha256": adapter_hash,
        "checkpoint_payload_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "metrics_sha256": metrics_hash,
        "log_sha256": log_hash,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", required=True, choices=sorted(SLOTS))
    parser.add_argument("--train", required=True)
    parser.add_argument("--arm-manifest", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-adapter", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    receipt = finalize(
        slot=args.slot, train_path=args.train, arm_manifest_path=args.arm_manifest,
        recipe_path=args.recipe, config_path=args.config, metrics_path=args.metrics,
        log_path=args.log, checkpoint_path=args.checkpoint,
        parent_adapter=args.parent_adapter, model_inventory_path=args.model_inventory,
        trainer_path=args.trainer, base_revision=args.base_revision, model_path=args.model_path)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
