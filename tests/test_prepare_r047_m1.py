import hashlib
import json
from pathlib import Path

import pytest

import scripts.compare_r047_causal as comparator
import scripts.finalize_r047_m1 as finalizer
import scripts.prepare_r047_m1 as preparer
from scripts.eval_r047_executor import validate_reference_provenance


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def _fixture(tmp_path, monkeypatch):
    values = {
        "parent_adapter_sha256": hashlib.sha256(b"parent").hexdigest(),
        "parent_adapter_config_sha256": hashlib.sha256(b"config").hexdigest(),
        "parent_adapter_tree_sha256": hashlib.sha256(b"tree").hexdigest(),
        "parent_adapter_tree_files": 4,
        "image_tree_sha256": hashlib.sha256(b"images").hexdigest(),
        "image_tree_files": 200,
    }
    model_hash = hashlib.sha256(b"model-inventory").hexdigest()
    preflight_hash = hashlib.sha256(b"preflight").hexdigest()
    inventory_hash = hashlib.sha256(b"inventory").hexdigest()
    binding = {
        "panel_sha256": hashlib.sha256(b"panel").hexdigest(),
        "panel_manifest_sha256": hashlib.sha256(b"panel-manifest").hexdigest(),
        "panel_preflight_sha256": preflight_hash,
        "inventory_manifest_sha256": inventory_hash,
        "evaluator_sha256": hashlib.sha256(b"evaluator").hexdigest(),
        "comparator_sha256": hashlib.sha256(b"comparator").hexdigest(),
    }
    control = _write(tmp_path / "control.jsonl", "control\n")
    treatment = _write(tmp_path / "treatment.jsonl", "treatment\n")
    source = _write(tmp_path / "checkpoint.sha256", "r044 provenance\n")
    trainer = _write(tmp_path / "trainer.py", "# trainer\n")
    monkeypatch.setattr(preparer, "FROZEN_R044_CHECKPOINT_PROVENANCE_SHA256",
                        hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(comparator, "FROZEN_PANEL_PREFLIGHT_SHA256", preflight_hash)
    monkeypatch.setattr(comparator, "FROZEN_INVENTORY_MANIFEST_SHA256", inventory_hash)
    monkeypatch.setattr(preparer, "verify_inventory", lambda *args: ({}, model_hash))
    monkeypatch.setattr(finalizer, "_validate_parent", lambda _: values.copy())
    monkeypatch.setattr(
        finalizer, "_validate_train",
        lambda path, *_: (hashlib.sha256(Path(path).read_bytes()).hexdigest(), {
            "image_tree_sha256": values["image_tree_sha256"],
            "image_tree_files": values["image_tree_files"],
        }),
    )
    monkeypatch.setattr(finalizer, "_evaluation_bindings", lambda *args: binding.copy())
    checked = []

    def validate_recipe(path, *_):
        recipe = json.loads(Path(path).read_text())
        assert set(recipe) == finalizer.RECIPE_FIELDS
        assert recipe["trainer_config"] == finalizer.FROZEN_TRAINING_INVARIANTS
        checked.append(path)

    monkeypatch.setattr(finalizer, "_validate_recipe", validate_recipe)
    return {
        "out": tmp_path / "prepared",
        "model_inventory_path": tmp_path / "model-inventory.json",
        "model_path": tmp_path / "model",
        "parent_adapter": tmp_path / "parent" / "adapter_model.safetensors",
        "arm_manifest_path": tmp_path / "arms-manifest.json",
        "control_train_path": control,
        "treatment_train_path": treatment,
        "image_root": tmp_path / "images",
        "trainer_path": trainer,
        "panel_path": tmp_path / "panel.jsonl",
        "panel_manifest_path": tmp_path / "panel-manifest.json",
        "panel_preflight_path": tmp_path / "preflight.json",
        "inventory_manifest_path": tmp_path / "inventory.json",
        "evaluator_path": tmp_path / "evaluator.py",
        "comparator_path": tmp_path / "comparator.py",
        "source_checkpoint_provenance": source,
    }, checked


def test_prepare_emits_exact_pretraining_consumer_artifacts(tmp_path, monkeypatch):
    args, checked = _fixture(tmp_path, monkeypatch)

    result = preparer.prepare(**args)

    recipe = json.loads((args["out"] / preparer.RECIPE_NAME).read_text())
    reference = json.loads((args["out"] / preparer.REFERENCE_NAME).read_text())
    assignment = json.loads((args["out"] / preparer.ASSIGNMENT_NAME).read_text())
    assert checked
    assert set(recipe) == finalizer.RECIPE_FIELDS
    assert recipe["trainer_config"] == finalizer.FROZEN_TRAINING_INVARIANTS
    assert "val" not in recipe["trainer_config"]  # finalizer owns the later exact config val=[] check.
    assert validate_reference_provenance(json.dumps(reference).encode()) == reference
    preparer._validate_assignment(assignment)
    assert assignment["control_slot"] == "slot_b"
    assert assignment["treatment_slot"] == "slot_a"
    assert assignment["recipe_sha256"] == result["recipe_sha256"]
    assert assignment["reference_provenance_sha256"] == result["reference_provenance_sha256"]


def test_prepare_fails_before_writing_when_reference_receipt_is_wrong(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    _write(args["source_checkpoint_provenance"], "tampered\n")

    with pytest.raises(ValueError, match="R044 checkpoint provenance"):
        preparer.prepare(**args)

    assert not args["out"].exists()


def test_prepare_refuses_to_replace_a_frozen_preassignment(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    args["out"].mkdir()
    _write(args["out"] / preparer.RECIPE_NAME, "{}\n")

    with pytest.raises(ValueError, match="already exist"):
        preparer.prepare(**args)
