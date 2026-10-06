"""Emit a fail-closed, hash-bound receipt for one blinded R047-M1 run."""

import argparse
import json
import math
from pathlib import Path

from scripts.build_r045_arms import _tree_hash as _arm_image_tree_hash, _valid_row
from scripts.build_r047_clause_arms import _summary
from scripts.build_r047_clause_arms import FROZEN_R045_ARM_B_SHA256
from scripts.build_r047_model_inventory import (
    FROZEN_BASE_REVISION,
    verify_inventory,
)
from scripts.preflight_r043_assets import file_hash, tree_hash


SCHEMA_VERSION = "r047-causal-run-receipt-v1"
RECIPE_SCHEMA_VERSION = "r047-m1-recipe-v1"
SLOTS = {"slot_a", "slot_b"}
FROZEN_PARENT_SHA256 = "3b5535de8c90156662d0b3b3f608fff12f557ddf0890722c4a584dbe93a1f8a4"
FROZEN_PARENT_TREE_SHA256 = "44e1998efd10275a6f0ab20cc6ccbd97cb9a8ad9b8c3f620620a5f6fdcc91f99"
FROZEN_PARENT_TREE_FILES = 4
FROZEN_PARENT_CONFIG_SHA256 = "ba83cbbcc1e793843372bc1f080e272df8f5bb06c9580ba4998c7d03691bb98a"
FROZEN_ARM_MANIFEST_SHA256 = "c18214db7befdfb93aa818b445bf9606a5f4333a2b7f0024cb802e8399460cef"
FROZEN_IMAGE_TREE_SHA256 = "56adbf7b9aa3101c06f3f254a0d8f62a9c41b786d29aad71b58a5e912c5f5394"
FROZEN_IMAGE_TREE_FILES = 200
FROZEN_ARM_PREFLIGHT_SHA256 = (
    "d3708564062d6533c12fedc0fd24139850756c64de58f991a71f70831d0ff02c")
FROZEN_M0_TREATMENT_SHA256 = (
    "f4e82bcfb272158c5b988238ec6ab56e0a5afa7d77143c1972fa7b7f0d0a647f")
TOKEN_DELTA_LIMIT = 0.005
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
    "parent_adapter_tree_sha256", "parent_adapter_tree_files",
    "parent_adapter_config_sha256", "model_inventory_sha256", "arm_manifest_sha256",
    "arm_sha256", "image_tree_sha256", "image_tree_files", "trainer_sha256",
    "arm_preflight_sha256",
    "panel_sha256", "panel_manifest_sha256", "panel_preflight_sha256",
    "inventory_manifest_sha256", "evaluator_sha256", "comparator_sha256",
    "precheck_sha256",
    "bootstrap_replicates", "bootstrap_seed", "trainer_config",
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


def _validate_parent(parent_adapter):
    adapter = Path(parent_adapter)
    if adapter.name != "adapter_model.safetensors":
        raise ValueError("parent adapter must be the checkpoint adapter_model.safetensors")
    checkpoint = adapter.parent
    config = checkpoint / "adapter_config.json"
    if not checkpoint.is_dir() or not config.is_file():
        raise ValueError("parent checkpoint is missing adapter_config.json")
    if any(item.is_symlink() for item in checkpoint.rglob("*")):
        raise ValueError("parent checkpoint must not contain symbolic links")
    weights_hash = _nonempty_file_hash(adapter, "parent adapter")
    config_hash = _nonempty_file_hash(config, "parent adapter config")
    tree_digest, tree_files = tree_hash(checkpoint)
    if (weights_hash != FROZEN_PARENT_SHA256
            or config_hash != FROZEN_PARENT_CONFIG_SHA256
            or tree_digest != FROZEN_PARENT_TREE_SHA256
            or tree_files != FROZEN_PARENT_TREE_FILES):
        raise ValueError("parent checkpoint differs from the complete frozen R044-R tree")
    return {
        "parent_adapter_sha256": weights_hash,
        "parent_adapter_config_sha256": config_hash,
        "parent_adapter_tree_sha256": tree_digest,
        "parent_adapter_tree_files": tree_files,
    }


def _validate_image_tree(manifest, manifest_path, image_root):
    root = Path(image_root).resolve()
    expected_root = (Path(manifest_path).resolve().parent / "images").resolve()
    if root != expected_root or not root.is_dir():
        raise ValueError("image root is not the frozen arm manifest image directory")
    images = manifest.get("images")
    entries = images.get("files_by_sha256") if isinstance(images, dict) else None
    if (not isinstance(entries, dict) or images.get("files") != FROZEN_IMAGE_TREE_FILES
            or images.get("tree_sha256") != FROZEN_IMAGE_TREE_SHA256):
        raise ValueError("arm manifest image contract differs from the frozen R047 tree")
    actual_files = sorted(item for item in root.rglob("*") if item.is_file())
    if (len(actual_files) != FROZEN_IMAGE_TREE_FILES
            or any(item.is_symlink() for item in root.rglob("*"))):
        raise ValueError("staged image tree file count or file type is invalid")
    expected_paths = {info.get("path") for info in entries.values()
                      if isinstance(info, dict)}
    actual_paths = {item.relative_to(root.parent).as_posix() for item in actual_files}
    if len(entries) != FROZEN_IMAGE_TREE_FILES or expected_paths != actual_paths:
        raise ValueError("staged image tree paths differ from the arm manifest")
    for digest, info in entries.items():
        if (info.get("sha256") != digest
                or file_hash(root.parent / info["path"]) != digest):
            raise ValueError("staged image payload is not content-addressed as declared")
    image_hash, count = _arm_image_tree_hash(root.parent, entries)
    if image_hash != FROZEN_IMAGE_TREE_SHA256 or count != FROZEN_IMAGE_TREE_FILES:
        raise ValueError("staged image tree differs from the frozen R047 tree")
    return {"image_tree_sha256": image_hash, "image_tree_files": count}


def _validate_train(train_path, manifest_path, image_root):
    train_hash = _nonempty_file_hash(train_path, "train payload")
    if train_hash not in set(KNOWN_ARM_SHA256.values()):
        raise ValueError("train payload is not either frozen R047 arm")
    if file_hash(manifest_path) != FROZEN_ARM_MANIFEST_SHA256:
        raise ValueError("arm manifest differs from the frozen R047 manifest")
    manifest = _json(manifest_path, "arm manifest")
    image_binding = _validate_image_tree(manifest, manifest_path, image_root)
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
    return train_hash, image_binding


def _evaluation_bindings(panel_path, panel_manifest_path, panel_preflight_path,
                         inventory_manifest_path, evaluator_path, comparator_path,
                         precheck_path):
    hashes = {
        "panel_sha256": _nonempty_file_hash(panel_path, "causal panel"),
        "panel_manifest_sha256": _nonempty_file_hash(
            panel_manifest_path, "causal panel manifest"),
        "panel_preflight_sha256": _nonempty_file_hash(
            panel_preflight_path, "causal panel preflight"),
        "inventory_manifest_sha256": _nonempty_file_hash(
            inventory_manifest_path, "causal inventory manifest"),
        "evaluator_sha256": _nonempty_file_hash(evaluator_path, "causal evaluator"),
        "comparator_sha256": _nonempty_file_hash(comparator_path, "causal comparator"),
        "precheck_sha256": _nonempty_file_hash(precheck_path, "causal precheck"),
    }
    manifest = _json(panel_manifest_path, "causal panel manifest")
    preflight = _json(panel_preflight_path, "causal panel preflight")
    primary = (manifest.get("outputs") or {}).get("primary")
    preflight_primary = (preflight.get("outputs") or {}).get("primary")
    if (not isinstance(primary, dict) or not isinstance(preflight_primary, dict)
            or primary.get("sha256") != hashes["panel_sha256"]
            or preflight_primary.get("sha256") != hashes["panel_sha256"]
            or preflight.get("panel_manifest_sha256") != hashes["panel_manifest_sha256"]
            or preflight.get("inventory_manifest_sha256") !=
            hashes["inventory_manifest_sha256"]
            or preflight.get("status") != "pass"
            or preflight.get("benchmark_content_emitted") is not False):
        raise ValueError("causal panel/preflight/inventory binding is invalid")
    return hashes


def _validate_arm_preflight(path):
    """Validate the frozen structural/token-budget audit that justified both arms."""
    digest = _nonempty_file_hash(path, "arm preflight")
    report = _json(path, "arm preflight")
    required = {
        "passed", "input_sha256", "control_sha256", "clause_sha256",
        "m0_treatment_sha256", "control", "clause", "token_budget",
    }
    budget = report.get("token_budget")
    budget_fields = {
        "tokenizer", "control_tokens", "clause_tokens", "absolute_delta",
        "relative_delta", "limit", "within_limit",
    }
    if (digest != FROZEN_ARM_PREFLIGHT_SHA256
            or set(report) != required
            or report.get("passed") is not True
            or report.get("input_sha256") != FROZEN_R045_ARM_B_SHA256
            or report.get("control_sha256") != KNOWN_ARM_SHA256["control"]
            or report.get("clause_sha256") != KNOWN_ARM_SHA256["treatment"]
            or report.get("m0_treatment_sha256") != FROZEN_M0_TREATMENT_SHA256
            or not isinstance(report.get("control"), dict)
            or not isinstance(report.get("clause"), dict)
            or any(report[name].get(key) != value for name in ("control", "clause")
                   for key, value in EXPECTED_ARM.items())
            or not isinstance(budget, dict) or set(budget) != budget_fields):
        raise ValueError("arm preflight differs from the frozen R047 contract")
    control_tokens = budget.get("control_tokens")
    clause_tokens = budget.get("clause_tokens")
    delta = budget.get("absolute_delta")
    relative = budget.get("relative_delta")
    tokenizer = budget.get("tokenizer")
    expected_suffix = (
        f"models--google--gemma-4-12B-it/snapshots/{FROZEN_BASE_REVISION}")
    if (type(control_tokens) is not int or control_tokens <= 0
            or type(clause_tokens) is not int or clause_tokens <= 0
            or type(delta) is not int
            or delta != abs(control_tokens - clause_tokens)
            or type(relative) not in (int, float) or not math.isfinite(relative)
            or not math.isclose(relative, delta / max(control_tokens, clause_tokens),
                                rel_tol=1e-12, abs_tol=1e-12)
            or not _same(budget.get("limit"), TOKEN_DELTA_LIMIT)
            or relative > TOKEN_DELTA_LIMIT
            or budget.get("within_limit") is not True
            or not isinstance(tokenizer, str)
            or not Path(tokenizer).as_posix().endswith(expected_suffix)):
        raise ValueError("arm preflight token-budget evidence is invalid")
    return digest


def _validate_recipe(recipe_path, model_inventory_sha256, trainer_path, parent_binding,
                     image_binding, evaluation_binding, arm_preflight_sha256):
    recipe = _json(recipe_path, "recipe")
    if (set(recipe) != RECIPE_FIELDS
            or recipe.get("schema_version") != RECIPE_SCHEMA_VERSION
            or recipe.get("base_revision") != FROZEN_BASE_REVISION
            or recipe.get("model_inventory_sha256") != model_inventory_sha256
            or recipe.get("arm_manifest_sha256") != FROZEN_ARM_MANIFEST_SHA256
            or recipe.get("arm_sha256") != KNOWN_ARM_SHA256
            or any(recipe.get(key) != value for key, value in parent_binding.items())
            or any(recipe.get(key) != value for key, value in image_binding.items())
            or any(recipe.get(key) != value for key, value in evaluation_binding.items())
            or recipe.get("arm_preflight_sha256") != arm_preflight_sha256
            or recipe.get("bootstrap_replicates") != 20_000
            or recipe.get("bootstrap_seed") != 470_047
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
    if config.get("val") != []:
        raise ValueError("trainer config validation channel must be exactly empty")
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
             log_path, checkpoint_path, parent_adapter, image_root, model_inventory_path,
             trainer_path, panel_path, panel_manifest_path, panel_preflight_path,
             inventory_manifest_path, evaluator_path, comparator_path,
             precheck_path, arm_preflight_path, base_revision, model_path):
    if slot not in SLOTS:
        raise ValueError("R047-M1 slot must be slot_a or slot_b")
    if base_revision != FROZEN_BASE_REVISION:
        raise ValueError("base revision differs from the frozen R047 revision")
    _, inventory_hash = verify_inventory(model_inventory_path, model_path, base_revision)
    parent_binding = _validate_parent(parent_adapter)
    train_hash, image_binding = _validate_train(train_path, arm_manifest_path, image_root)
    evaluation_binding = _evaluation_bindings(
        panel_path, panel_manifest_path, panel_preflight_path, inventory_manifest_path,
        evaluator_path, comparator_path, precheck_path)
    arm_preflight_sha256 = _validate_arm_preflight(arm_preflight_path)
    recipe, recipe_hash = _validate_recipe(
        recipe_path, inventory_hash, trainer_path, parent_binding, image_binding,
        evaluation_binding, arm_preflight_sha256)
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
        "parent_adapter_sha256": parent_binding["parent_adapter_sha256"],
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
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--panel-preflight", required=True)
    parser.add_argument("--inventory-manifest", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--precheck", required=True)
    parser.add_argument("--arm-preflight", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    receipt = finalize(
        slot=args.slot, train_path=args.train, arm_manifest_path=args.arm_manifest,
        recipe_path=args.recipe, config_path=args.config, metrics_path=args.metrics,
        log_path=args.log, checkpoint_path=args.checkpoint,
        parent_adapter=args.parent_adapter, image_root=args.image_root,
        model_inventory_path=args.model_inventory, trainer_path=args.trainer,
        panel_path=args.panel, panel_manifest_path=args.panel_manifest,
        panel_preflight_path=args.panel_preflight,
        inventory_manifest_path=args.inventory_manifest, evaluator_path=args.evaluator,
        comparator_path=args.comparator, precheck_path=args.precheck,
        arm_preflight_path=args.arm_preflight,
        base_revision=args.base_revision,
        model_path=args.model_path)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
