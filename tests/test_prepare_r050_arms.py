import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, build_inventory
from scripts.build_r050_mix import SCHEMA as MIX_SCHEMA
from scripts.preflight_r043_assets import file_hash, tree_hash
from scripts.prepare_r050_arms import SCHEMA, prepare, validate_mix
from scripts.probe_r050_gradient_scale import SCHEMA_VERSION as PROBE_SCHEMA


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _sha(path):
    return file_hash(path)


def _meta(cluster, cell, side, language, order):
    return {
        "panel": "exact_policy_v1", "panel_version": "1.6", "split": "train",
        "semantic_group_id": f"cluster-{cluster:03d}", "counterfactual_side": side,
        "semantic_signature": hashlib.sha256(f"semantic-{cluster}".encode()).hexdigest(),
        "language": language, "option_order": order, "field_order": ["a", "b"],
        "policy_mode": "priority", "operator_composition": "all_of_3",
        "value_regime": "raw_operands", "renderer_family": "renderer",
        "source_family": f"source-{cell:03d}", "topology_slice": "development",
    }


def _fixture(tmp_path, *, cells=112):
    trainer = _write(tmp_path / "train_gemma_decision_lora.py", "# trainer\n")
    probe = _write(tmp_path / "probe_r050_gradient_scale.py", "# probe\n")
    builder = _write(tmp_path / "build_r048_action_panel.py", "# builder\n")
    panel_rows = [{"id": f"panel-{cell}", "meta": _meta(cell, cell, 0, "en", "no_yes")}
                  for cell in range(112)]
    panel = _write(tmp_path / "panel.jsonl", "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in panel_rows))
    panel_manifest = _write(tmp_path / "panel-manifest.json", {
        "generator_sha256": _sha(builder),
        "outputs": {"train": {"sha256": _sha(panel), "rows": len(panel_rows)}},
    })

    model = (tmp_path / "models--google--gemma-4-12B-it" / "snapshots"
             / FROZEN_BASE_REVISION)
    _write(model / "config.json", {})
    _write(model / "model.safetensors", b"model")
    inventory_value = build_inventory(model)
    inventory = _write(tmp_path / "model-inventory.json", inventory_value)
    parent = tmp_path / "parent" / "step-000100"
    _write(parent / "adapter_model.safetensors", b"parent")
    _write(parent / "adapter_config.json", {})
    parent_tree, parent_files = tree_hash(parent)

    mix_root = tmp_path / "mix"
    executable = []
    variants = [(side, language, order) for side in (0, 1)
                for language in ("en", "he") for order in ("no_yes", "yes_no")]
    for cluster in range(200):
        for side, language, order in variants:
            executable.append({"id": f"e-{cluster}-{side}-{language}-{order}",
                               "meta": _meta(cluster, cluster % cells, side, language, order)})
    replay = ([{"id": f"b-{index}", "meta": {"kind": "broad"}} for index in range(800)]
              + [{"id": f"h-{index}", "meta": {"kind": "hebrew"}} for index in range(400)])
    image_entries = {}
    multimodal = []
    for index in range(400):
        reference = f"images/{index:03d}.png"
        image = _write(mix_root / reference, f"image-{index}".encode())
        image_entries[reference] = _sha(image)
        multimodal.append({"id": f"m-{index}", "image": reference,
                           "meta": {"kind": "multimodal"}})
    rows = executable + replay + multimodal
    mix = _write(mix_root / "train.jsonl", "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    image_tree = hashlib.sha256("".join(
        f"{name}\0{digest}\n" for name, digest in sorted(image_entries.items())).encode()).hexdigest()
    mix_manifest_value = {
        "schema": MIX_SCHEMA,
        "optimizer_slots": 3200,
        "outputs": {"train": {"sha256": _sha(mix), "rows": 3200,
                               "optimizer_slots": 3200}},
        "strata": {name: {"optimizer_slots": count} for name, count in
                   {"executable": 1600, "broad": 800, "hebrew": 400,
                    "multimodal": 400}.items()},
        "quarantine": {"selected_id_overlap": 0, "selected_hash_overlap": 0},
        "images": {"files": 400, "tree_sha256": image_tree,
                   "files_by_path": image_entries},
        "inputs": {"executable": {"sha256": _sha(panel)},
                   "panel_manifest": {"sha256": _sha(panel_manifest)}},
        "panel_generator_sha256": _sha(builder),
        "code_sha256": {trainer.name: _sha(trainer)},
    }
    mix_manifest = _write(mix_root / "manifest.json", mix_manifest_value)
    gradient = _write(tmp_path / "gradient.json", {
        "schema_version": PROBE_SCHEMA, "status": "complete",
        "selection": {"requested_signatures": 64, "selected_signatures": 64},
        "gradient_report": {
            "norms": {"field_ce": 2.0, "outcome_nll": 4.0, "outcome_brier": 8.0},
            "coefficients": {"lambda_nll": 0.125, "lambda_brier": 0.0625}},
        "configuration": {"max_length": 8192, "target_ratio": 0.25,
                          "action_loss_weight": 0.0, "action_temperature": 1.0,
                          "option_shuffle": False, "optimizer_created": False},
        "bindings": {
            "training_panel": {"path": str(panel.resolve()), "sha256": _sha(panel),
                               "rows": len(panel_rows)},
            "panel_manifest": {"path": str(panel_manifest.resolve()),
                               "sha256": _sha(panel_manifest),
                               "generator_sha256": _sha(builder)},
            "panel_builder_sha256": _sha(builder),
            "model_inventory": {"path": str(inventory.resolve()),
                                "sha256": _sha(inventory), **inventory_value},
            "parent_adapter": {"path": str(parent.resolve()),
                               "tree_sha256": parent_tree, "files": parent_files},
            "trainer_sha256": _sha(trainer), "probe_sha256": _sha(probe),
        },
    })
    kwargs = {
        "mix_path": mix, "mix_manifest_path": mix_manifest, "model_path": model,
        "model_inventory_path": inventory, "gradient_receipt_path": gradient,
        "panel_path": panel, "panel_manifest_path": panel_manifest,
        "panel_builder_path": builder, "parent_adapter_path": parent,
        "trainer_path": trainer, "probe_path": probe, "out": tmp_path / "suite",
        "expected_mix_sha256": _sha(mix),
        "expected_mix_manifest_sha256": _sha(mix_manifest),
        "expected_model_inventory_sha256": _sha(inventory),
        "expected_gradient_receipt_sha256": _sha(gradient),
        "expected_panel_sha256": _sha(panel),
        "expected_panel_manifest_sha256": _sha(panel_manifest),
        "expected_builder_sha256": _sha(builder),
        "expected_trainer_sha256": _sha(trainer),
        "expected_probe_sha256": _sha(probe),
        "expected_parent_tree_sha256": parent_tree,
        "expected_parent_tree_files": parent_files,
    }
    return kwargs


def test_prepare_freezes_three_matched_preassigned_arms(tmp_path):
    values = _fixture(tmp_path)
    suite = prepare(**values)
    assert [arm["run_order"] for arm in suite["arms"]] == [1, 2, 3]
    assert len({arm["arm_id"] for arm in suite["arms"]}) == 3
    recipes = [json.loads((Path(values["out"]) / arm["recipe"]).read_text())
               for arm in suite["arms"]]
    assert all(recipe["schema"] == SCHEMA for recipe in recipes)
    assert {recipe["objective"]["objective"] for recipe in recipes} == {
        "field_ce", "field_ce_plus_exact_outcome_nll",
        "field_ce_plus_exact_outcome_brier"}
    common = [{key: recipe[key] for key in ("training", "epoch_contract", "bindings")}
              for recipe in recipes]
    assert common[0] == common[1] == common[2]
    assert {recipe["objective"]["action_loss_weight"] for recipe in recipes} == {
        0.0, 0.125, 0.0625}
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(**values)


def test_mix_rejects_missing_factor_cell_or_changed_pixels(tmp_path):
    values = _fixture(tmp_path, cells=111)
    with pytest.raises(ValueError, match="112.*factor cells"):
        validate_mix(values["mix_path"], values["mix_manifest_path"],
                     values["expected_mix_sha256"],
                     values["expected_mix_manifest_sha256"])

    values = _fixture(tmp_path / "pixels")
    (Path(values["mix_path"]).parent / "images/000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image sha256 mismatch"):
        prepare(**values)


def test_prepare_rejects_probe_coefficient_or_external_hash_drift(tmp_path):
    values = _fixture(tmp_path)
    receipt = json.loads(Path(values["gradient_receipt_path"]).read_text())
    receipt["gradient_report"]["coefficients"]["lambda_nll"] = 9.0
    _write(values["gradient_receipt_path"], receipt)
    values["expected_gradient_receipt_sha256"] = _sha(values["gradient_receipt_path"])
    with pytest.raises(ValueError, match="coefficient does not match"):
        prepare(**values)

    values = _fixture(tmp_path / "hash")
    values["expected_mix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="training mix sha256 mismatch"):
        prepare(**values)
