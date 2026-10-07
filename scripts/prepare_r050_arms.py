"""Validate and freeze the three matched R050 calibrated-executor arms.

The resulting recipes are intentionally boring: the only arm-varying fields are
the exact-policy objective kind and its precomputed coefficient.  Every input is
content bound, including the images referenced by the multimodal replay slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, verify_inventory
from scripts.build_r050_mix import SCHEMA as MIX_SCHEMA
from scripts.preflight_r043_assets import file_hash, tree_hash
from scripts.probe_r050_gradient_scale import SCHEMA_VERSION as PROBE_SCHEMA


SCHEMA = "r050-matched-arm-recipe-v1"
SUITE_SCHEMA = "r050-matched-arm-suite-v1"
EXPECTED_SLOTS = 3200
EXPECTED_STRATA = {
    "executable": 1600,
    "broad": 800,
    "hebrew": 400,
    "multimodal": 400,
}
TRAINING = {
    "steps": 400,
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
    "seed": 50,
    "pair_consistency_weight": 0.0,
    "pair_consistency_min_gold_probability": 0.8,
    "pair_consistency_clip": 4.0,
    "pair_consistency_beta": 1.0,
    "project_consistency_against_guard": False,
    "pair_flip_weight": 0.0,
    "pair_flip_margin": 1.0,
    "pair_flip_start_step": 0,
    "stage_counterfactual_pairs": False,
    "action_temperature": 1.0,
}


def _reject_constant(value: str):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, file_hash(path)


def checked_hash(value: str, label: str = "sha256") -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"{label} must be a 64-character hexadecimal sha256")
    return value.lower()


def require_hash(path: str | Path, expected: str, label: str) -> tuple[Path, str]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    actual = file_hash(path)
    if actual != checked_hash(expected, f"expected {label} sha256"):
        raise ValueError(f"{label} sha256 mismatch")
    return path.resolve(), actual


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _same_number(left: Any, right: Any) -> bool:
    return (type(left) in {int, float} and type(right) in {int, float}
            and not isinstance(left, bool) and not isinstance(right, bool)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12))


def _safe_image_path(root: Path, reference: str) -> Path:
    relative = PurePosixPath(reference)
    if (not reference or relative.is_absolute() or ".." in relative.parts
            or relative.parts[0] in {"train.jsonl", "manifest.json"}):
        raise ValueError("mix image path is unsafe")
    target = root.joinpath(*relative.parts)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"mix image is missing or symbolic: {reference}")
    resolved = target.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("mix image escapes the frozen mix directory") from error
    return resolved


def validate_mix(mix_path: str | Path, manifest_path: str | Path,
                 expected_mix: str, expected_manifest: str) -> dict[str, Any]:
    mix_path, mix_sha = require_hash(mix_path, expected_mix, "training mix")
    manifest_path, manifest_sha = require_hash(
        manifest_path, expected_manifest, "training mix manifest")
    manifest, parsed_sha = read_json(manifest_path, "training mix manifest")
    if parsed_sha != manifest_sha or manifest.get("schema") != MIX_SCHEMA:
        raise ValueError("training mix manifest schema is invalid")
    output = manifest.get("outputs", {}).get("train")
    if (not isinstance(output, dict) or output.get("sha256") != mix_sha
            or output.get("rows") != EXPECTED_SLOTS
            or output.get("optimizer_slots") != EXPECTED_SLOTS
            or manifest.get("optimizer_slots") != EXPECTED_SLOTS):
        raise ValueError("training mix is not the frozen 3,200-slot output")
    strata = manifest.get("strata")
    if (not isinstance(strata, dict) or set(strata) != set(EXPECTED_STRATA)
            or any(not isinstance(strata[name], dict)
                   or strata[name].get("optimizer_slots") != count
                   for name, count in EXPECTED_STRATA.items())):
        raise ValueError("training mix strata are not exactly 1600/800/400/400")
    quarantine = manifest.get("quarantine")
    if (not isinstance(quarantine, dict)
            or quarantine.get("selected_id_overlap") != 0
            or quarantine.get("selected_hash_overlap") != 0):
        raise ValueError("training mix has quarantine overlap")
    images = manifest.get("images")
    if not isinstance(images, dict):
        raise ValueError("training mix image receipt is missing")
    entries = images.get("files_by_path")
    if (not isinstance(entries, dict) or images.get("files") != len(entries)
            or images.get("files") != EXPECTED_STRATA["multimodal"]):
        raise ValueError("training mix must bind exactly 400 multimodal images")
    actual_entries = {}
    for reference, expected in entries.items():
        actual_entries[reference] = file_hash(_safe_image_path(mix_path.parent, reference))
        if actual_entries[reference] != checked_hash(expected, "mix image sha256"):
            raise ValueError(f"mix image sha256 mismatch: {reference}")
    tree_payload = "".join(
        f"{name}\0{digest}\n" for name, digest in sorted(actual_entries.items())).encode()
    image_tree_sha = hashlib.sha256(tree_payload).hexdigest()
    if image_tree_sha != images.get("tree_sha256"):
        raise ValueError("training mix image tree mismatch")
    try:
        rows = [json.loads(line, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
                for line in mix_path.read_bytes().splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid training mix JSONL") from error
    if len(rows) != EXPECTED_SLOTS or any(not isinstance(row, dict) for row in rows):
        raise ValueError("training mix must serialize exactly 3,200 object rows")
    image_references = [row.get("image") for row in rows if "image" in row]
    if (len(image_references) != EXPECTED_STRATA["multimodal"]
            or any(not isinstance(reference, str) or not reference
                   for reference in image_references)
            or len(set(image_references)) != EXPECTED_STRATA["multimodal"]
            or set(image_references) != set(entries)):
        raise ValueError("training mix rows do not reference exactly the 400 bound images")
    executable = [row for row in rows
                  if (row.get("meta") or {}).get("panel") == "exact_policy_v1"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in executable:
        meta = row.get("meta")
        identifier = meta.get("semantic_group_id") if isinstance(meta, dict) else None
        if (not isinstance(identifier, str) or not identifier
                or meta.get("panel_version") != "1.6" or meta.get("split") != "train"):
            raise ValueError("executable mix rows must be frozen v1.6 training rows")
        groups.setdefault(identifier, []).append(row)
    if len(executable) != EXPECTED_STRATA["executable"] or len(groups) != 200:
        raise ValueError("training mix must contain exactly 200 executable semantic clusters")
    variants = {(side, language, order) for side in (0, 1)
                for language in ("en", "he") for order in ("no_yes", "yes_no")}
    cells, semantic_signatures = set(), set()
    for identifier, group in groups.items():
        if (len(group) != 8 or {(row["meta"].get("counterfactual_side"),
                                row["meta"].get("language"),
                                row["meta"].get("option_order")) for row in group} != variants):
            raise ValueError(f"incomplete executable semantic cluster: {identifier}")
        metas = [row["meta"] for row in group]
        cell = (len(metas[0].get("field_order", [])), metas[0].get("policy_mode"),
                metas[0].get("operator_composition"), metas[0].get("value_regime"),
                metas[0].get("renderer_family"), metas[0].get("source_family"),
                metas[0].get("topology_slice"))
        if any((len(meta.get("field_order", [])), meta.get("policy_mode"),
                meta.get("operator_composition"), meta.get("value_regime"),
                meta.get("renderer_family"), meta.get("source_family"),
                meta.get("topology_slice")) != cell for meta in metas):
            raise ValueError(f"executable factor cell changes within cluster: {identifier}")
        cells.add(cell)
        signatures = {meta.get("semantic_signature") for meta in metas}
        if (len(signatures) != 1 or not isinstance(next(iter(signatures)), str)
                or not next(iter(signatures))):
            raise ValueError(f"invalid executable semantic signature: {identifier}")
        semantic_signatures.update(signatures)
    if len(semantic_signatures) != 200:
        raise ValueError("executable semantic clusters must have 200 unique signatures")
    if len(cells) != 112:
        raise ValueError("training mix does not cover all 112 v1.6 executable factor cells")
    cell_sha = hashlib.sha256(_canonical(sorted(cells))).hexdigest()
    return {
        "path": str(mix_path), "sha256": mix_sha, "rows": EXPECTED_SLOTS,
        "optimizer_slots": EXPECTED_SLOTS,
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
        "image_tree_sha256": image_tree_sha, "image_tree_files": len(entries),
        "executable_semantic_clusters": len(groups),
        "executable_factor_cells": len(cells),
        "executable_factor_cells_sha256": cell_sha,
        "manifest": manifest,
    }


def validate_parent(path: str | Path, expected_tree: str,
                    expected_files: int) -> dict[str, Any]:
    original = Path(path)
    if original.is_symlink() or not original.is_dir():
        raise ValueError("parent adapter must be a regular directory")
    if (isinstance(expected_files, bool) or not isinstance(expected_files, int)
            or expected_files <= 0):
        raise ValueError("expected parent tree file count must be positive")
    if any(item.is_symlink() for item in original.rglob("*")):
        raise ValueError("parent adapter tree cannot contain symbolic links")
    for required in ("adapter_model.safetensors", "adapter_config.json"):
        if not (original / required).is_file():
            raise ValueError(f"parent adapter is missing {required}")
    digest, files = tree_hash(original)
    if digest != checked_hash(expected_tree, "expected parent tree sha256") or files != expected_files:
        raise ValueError("parent adapter tree differs from the external expectation")
    return {
        "path": str(original.resolve()), "tree_sha256": digest, "files": files,
        "adapter_sha256": file_hash(original / "adapter_model.safetensors"),
        "config_sha256": file_hash(original / "adapter_config.json"),
    }


def panel_factor_cells_sha256(panel: Path) -> str:
    """Return the complete v1.6 train design-cell digest from persisted rows."""
    try:
        rows = [json.loads(line, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
                for line in panel.read_bytes().splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid training panel JSONL") from error
    cells = set()
    for row in rows:
        meta = row.get("meta") if isinstance(row, dict) else None
        if (not isinstance(meta, dict) or meta.get("panel") != "exact_policy_v1"
                or meta.get("panel_version") != "1.6" or meta.get("split") != "train"):
            raise ValueError("training panel contains a non-v1.6 training row")
        cells.add((len(meta.get("field_order", [])), meta.get("policy_mode"),
                   meta.get("operator_composition"), meta.get("value_regime"),
                   meta.get("renderer_family"), meta.get("source_family"),
                   meta.get("topology_slice")))
    if len(cells) != 112:
        raise ValueError("training panel does not define exactly 112 factor cells")
    return hashlib.sha256(_canonical(sorted(cells))).hexdigest()


def validate_probe(receipt_path: str | Path, expected_receipt: str, *,
                   panel: Path, panel_sha: str, panel_manifest: Path,
                   panel_manifest_sha: str, builder: Path, builder_sha: str,
                   inventory_path: Path, inventory_sha: str,
                   inventory: dict[str, Any], parent: dict[str, Any],
                   trainer: Path, trainer_sha: str, probe: Path,
                   probe_sha: str) -> tuple[dict[str, Any], dict[str, float], str]:
    receipt_path, receipt_sha = require_hash(
        receipt_path, expected_receipt, "gradient probe receipt")
    receipt, parsed_sha = read_json(receipt_path, "gradient probe receipt")
    if (parsed_sha != receipt_sha or receipt.get("schema_version") != PROBE_SCHEMA
            or receipt.get("status") != "complete"):
        raise ValueError("gradient probe receipt is not complete or has the wrong schema")
    configuration = receipt.get("configuration")
    if (not isinstance(configuration, dict)
            or configuration.get("max_length") != TRAINING["max_length"]
            or not _same_number(configuration.get("target_ratio"), 0.25)
            or not _same_number(configuration.get("action_loss_weight"), 0.0)
            or not _same_number(configuration.get("action_temperature"), 1.0)
            or configuration.get("option_shuffle") is not False
            or configuration.get("optimizer_created") is not False):
        raise ValueError("gradient probe configuration is incompatible with R050")
    selection = receipt.get("selection")
    if (not isinstance(selection, dict) or selection.get("selected_signatures") != 64
            or selection.get("requested_signatures") != 64):
        raise ValueError("gradient probe did not use the frozen 64-signature sample")
    report = receipt.get("gradient_report")
    coefficients = report.get("coefficients") if isinstance(report, dict) else None
    norms = report.get("norms") if isinstance(report, dict) else None
    if (not isinstance(coefficients, dict) or set(coefficients) != {"lambda_nll", "lambda_brier"}
            or not isinstance(norms, dict)):
        raise ValueError("gradient probe coefficient report is invalid")
    checked_coefficients = {}
    for name, loss_name in (("lambda_nll", "outcome_nll"),
                            ("lambda_brier", "outcome_brier")):
        value = coefficients[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("gradient probe coefficients must be finite and positive")
        expected = 0.25 * norms.get("field_ce", float("nan")) / norms.get(loss_name, float("nan"))
        if not math.isfinite(expected) or not math.isclose(value, expected, rel_tol=1e-10):
            raise ValueError("gradient probe coefficient does not match its bound norms")
        checked_coefficients[name] = float(value)
    bindings = receipt.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("gradient probe bindings are missing")
    expected_files = {
        "training_panel": (panel, panel_sha),
        "panel_manifest": (panel_manifest, panel_manifest_sha),
        "model_inventory": (inventory_path, inventory_sha),
    }
    for name, (path, digest) in expected_files.items():
        binding = bindings.get(name)
        if (not isinstance(binding, dict) or Path(binding.get("path", "")).resolve() != path
                or binding.get("sha256") != digest):
            raise ValueError(f"gradient probe {name} binding mismatch")
    if bindings["training_panel"].get("rows") != sum(
            1 for line in panel.read_bytes().splitlines() if line.strip()):
        raise ValueError("gradient probe training panel row count mismatch")
    manifest_binding = bindings["panel_manifest"]
    if (manifest_binding.get("generator_sha256") != builder_sha
            or bindings.get("panel_builder_sha256") != builder_sha
            or bindings.get("trainer_sha256") != trainer_sha
            or bindings.get("probe_sha256") != probe_sha):
        raise ValueError("gradient probe code or panel-builder binding mismatch")
    model_binding = bindings["model_inventory"]
    if (model_binding.get("tree_sha256") != inventory["tree_sha256"]
            or model_binding.get("revision") != FROZEN_BASE_REVISION):
        raise ValueError("gradient probe model-tree binding mismatch")
    if bindings.get("parent_adapter") != {
            "path": parent["path"], "tree_sha256": parent["tree_sha256"],
            "files": parent["files"]}:
        raise ValueError("gradient probe parent-adapter binding mismatch")
    return receipt, checked_coefficients, receipt_sha


def prepare(*, mix_path, mix_manifest_path, model_path, model_inventory_path,
            gradient_receipt_path, panel_path, panel_manifest_path, panel_builder_path,
            parent_adapter_path, trainer_path, probe_path, out,
            expected_mix_sha256, expected_mix_manifest_sha256,
            expected_model_inventory_sha256, expected_gradient_receipt_sha256,
            expected_panel_sha256, expected_panel_manifest_sha256,
            expected_builder_sha256, expected_trainer_sha256, expected_probe_sha256,
            expected_parent_tree_sha256, expected_parent_tree_files,
            assignment_seed: int = 50_050) -> dict[str, Any]:
    """Freeze all recipes only after the complete trust boundary validates."""
    if isinstance(assignment_seed, bool) or not isinstance(assignment_seed, int):
        raise ValueError("assignment seed must be an integer")
    output = Path(out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite arm suite: {output}")
    mix = validate_mix(mix_path, mix_manifest_path, expected_mix_sha256,
                       expected_mix_manifest_sha256)
    panel, panel_sha = require_hash(panel_path, expected_panel_sha256, "training panel")
    if mix["executable_factor_cells_sha256"] != panel_factor_cells_sha256(panel):
        raise ValueError("training mix does not cover the panel's complete 112-cell design")
    panel_manifest, panel_manifest_sha = require_hash(
        panel_manifest_path, expected_panel_manifest_sha256, "panel manifest")
    builder, builder_sha = require_hash(
        panel_builder_path, expected_builder_sha256, "panel builder")
    trainer, trainer_sha = require_hash(trainer_path, expected_trainer_sha256, "trainer")
    probe, probe_sha = require_hash(probe_path, expected_probe_sha256, "gradient probe")
    inventory_path, inventory_sha = require_hash(
        model_inventory_path, expected_model_inventory_sha256, "model inventory")
    model_path = Path(model_path).resolve()
    inventory, verified_inventory_sha = verify_inventory(
        inventory_path, model_path, FROZEN_BASE_REVISION)
    if verified_inventory_sha != inventory_sha:
        raise ValueError("model inventory changed during verification")
    parent = validate_parent(parent_adapter_path, expected_parent_tree_sha256,
                             expected_parent_tree_files)
    _, coefficients, receipt_sha = validate_probe(
        gradient_receipt_path, expected_gradient_receipt_sha256,
        panel=panel, panel_sha=panel_sha, panel_manifest=panel_manifest,
        panel_manifest_sha=panel_manifest_sha, builder=builder, builder_sha=builder_sha,
        inventory_path=inventory_path, inventory_sha=inventory_sha, inventory=inventory,
        parent=parent, trainer=trainer, trainer_sha=trainer_sha, probe=probe,
        probe_sha=probe_sha)
    mix_manifest = mix.pop("manifest")
    inputs = mix_manifest.get("inputs", {})
    if (not isinstance(inputs.get("executable"), dict)
            or inputs["executable"].get("sha256") != panel_sha
            or not isinstance(inputs.get("panel_manifest"), dict)
            or inputs["panel_manifest"].get("sha256") != panel_manifest_sha
            or mix_manifest.get("panel_generator_sha256") != builder_sha
            or mix_manifest.get("code_sha256", {}).get(trainer.name) != trainer_sha):
        raise ValueError("training mix is not bound to the supplied panel and trainer")

    common_bindings = {
        "mix": mix,
        "model": {"path": str(model_path), **inventory,
                  "inventory_path": str(inventory_path),
                  "inventory_sha256": inventory_sha},
        "gradient_probe": {"path": str(Path(gradient_receipt_path).resolve()),
                           "sha256": receipt_sha},
        "training_panel": {"path": str(panel), "sha256": panel_sha},
        "panel_manifest": {"path": str(panel_manifest), "sha256": panel_manifest_sha},
        "panel_builder": {"path": str(builder), "sha256": builder_sha},
        "parent_adapter": parent,
        "trainer": {"path": str(trainer), "sha256": trainer_sha},
        "gradient_probe_code": {"path": str(probe), "sha256": probe_sha},
    }
    objectives = [
        {"objective": "field_ce", "action_loss_weight": 0.0,
         "exact_policy_outcome_loss": "nll"},
        {"objective": "field_ce_plus_exact_outcome_nll",
         "action_loss_weight": coefficients["lambda_nll"],
         "exact_policy_outcome_loss": "nll"},
        {"objective": "field_ce_plus_exact_outcome_brier",
         "action_loss_weight": coefficients["lambda_brier"],
         "exact_policy_outcome_loss": "brier"},
    ]
    commitment_input = hashlib.sha256(_canonical({
        "bindings": common_bindings, "seed": assignment_seed})).hexdigest()
    ranked = sorted(objectives, key=lambda objective: hashlib.sha256(
        _canonical([assignment_seed, commitment_input, objective["objective"]])).hexdigest())
    recipes = []
    for order, objective in enumerate(ranked, 1):
        opaque = hashlib.sha256(_canonical(
            ["r050-arm", assignment_seed, commitment_input, order])).hexdigest()[:12]
        recipe = {
            "schema": SCHEMA,
            "arm_id": f"slot_{opaque}",
            "run_order": order,
            "objective": objective,
            "training": dict(TRAINING),
            "epoch_contract": {
                "optimizer_slots": EXPECTED_SLOTS,
                "slots_per_update": TRAINING["microbatch"] * TRAINING["grad_accum"],
                "updates": TRAINING["steps"],
                "passes": 1,
                "slot_order": "trainer seeded global shuffle",
            },
            "bindings": common_bindings,
        }
        payload = _canonical(recipe)
        recipes.append((recipe, payload, hashlib.sha256(payload).hexdigest()))
    assignment_commitment = hashlib.sha256(_canonical([
        {"arm_id": recipe["arm_id"], "run_order": recipe["run_order"],
         "objective": recipe["objective"]} for recipe, _, _ in recipes])).hexdigest()
    suite = {
        "schema": SUITE_SCHEMA,
        "assignment_seed": assignment_seed,
        "assignment_commitment": assignment_commitment,
        "common_binding_sha256": commitment_input,
        "arms": [{"arm_id": recipe["arm_id"], "run_order": recipe["run_order"],
                  "recipe": f"{recipe['arm_id']}.json", "sha256": digest}
                 for recipe, _, digest in recipes],
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for recipe, payload, _ in recipes:
            with (output / f"{recipe['arm_id']}.json").open("xb") as handle:
                handle.write(payload)
        with (output / "manifest.json").open("xb") as handle:
            handle.write(_canonical(suite))
    except Exception:
        # Preserve any partial output for forensic inspection; never rewrite it.
        raise
    return suite


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mix", "mix-manifest", "model", "model-inventory", "gradient-receipt",
                 "panel", "panel-manifest", "panel-builder", "parent-adapter", "trainer",
                 "probe", "out"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("mix", "mix-manifest", "model-inventory", "gradient-receipt", "panel",
                 "panel-manifest", "builder", "trainer", "probe", "parent-tree"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-parent-tree-files", required=True, type=int)
    parser.add_argument("--assignment-seed", type=int, default=50_050)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    values = vars(args)
    suite = prepare(
        mix_path=values["mix"], mix_manifest_path=values["mix_manifest"],
        model_path=values["model"], model_inventory_path=values["model_inventory"],
        gradient_receipt_path=values["gradient_receipt"], panel_path=values["panel"],
        panel_manifest_path=values["panel_manifest"],
        panel_builder_path=values["panel_builder"],
        parent_adapter_path=values["parent_adapter"], trainer_path=values["trainer"],
        probe_path=values["probe"], out=values["out"],
        expected_mix_sha256=values["expected_mix_sha256"],
        expected_mix_manifest_sha256=values["expected_mix_manifest_sha256"],
        expected_model_inventory_sha256=values["expected_model_inventory_sha256"],
        expected_gradient_receipt_sha256=values["expected_gradient_receipt_sha256"],
        expected_panel_sha256=values["expected_panel_sha256"],
        expected_panel_manifest_sha256=values["expected_panel_manifest_sha256"],
        expected_builder_sha256=values["expected_builder_sha256"],
        expected_trainer_sha256=values["expected_trainer_sha256"],
        expected_probe_sha256=values["expected_probe_sha256"],
        expected_parent_tree_sha256=values["expected_parent_tree_sha256"],
        expected_parent_tree_files=values["expected_parent_tree_files"],
        assignment_seed=values["assignment_seed"])
    print(json.dumps(suite, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
