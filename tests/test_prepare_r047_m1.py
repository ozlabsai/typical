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
    arm_preflight_hash = hashlib.sha256(b"arm-preflight").hexdigest()
    binding = {
        "panel_sha256": hashlib.sha256(b"panel").hexdigest(),
        "panel_manifest_sha256": hashlib.sha256(b"panel-manifest").hexdigest(),
        "panel_preflight_sha256": preflight_hash,
        "inventory_manifest_sha256": inventory_hash,
        "evaluator_sha256": hashlib.sha256(b"evaluator").hexdigest(),
        "comparator_sha256": hashlib.sha256(b"comparator").hexdigest(),
        "precheck_sha256": hashlib.sha256(b"precheck").hexdigest(),
    }
    control = _write(tmp_path / "control.jsonl", "control\n")
    treatment = _write(tmp_path / "treatment.jsonl", "treatment\n")
    monkeypatch.setattr(finalizer, "KNOWN_ARM_SHA256", {
        "control": hashlib.sha256(control.read_bytes()).hexdigest(),
        "treatment": hashlib.sha256(treatment.read_bytes()).hexdigest(),
    })
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
    monkeypatch.setattr(finalizer, "_validate_arm_preflight",
                        lambda _: arm_preflight_hash)
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
        "precheck_path": tmp_path / "precheck.py",
        "arm_preflight_path": tmp_path / "arm-preflight.json",
        "source_checkpoint_provenance": source,
    }, checked


def test_prepare_emits_exact_pretraining_consumer_artifacts(tmp_path, monkeypatch):
    args, checked = _fixture(tmp_path, monkeypatch)

    result = preparer.prepare(**args)

    recipe = json.loads((args["out"] / preparer.RECIPE_NAME).read_text())
    reference = json.loads((args["out"] / preparer.REFERENCE_NAME).read_text())
    assignment = json.loads((args["out"] / preparer.ASSIGNMENT_NAME).read_text())
    preparation = json.loads((args["out"] / preparer.PREPARATION_NAME).read_text())
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
    assert recipe["arm_preflight_sha256"] == assignment["arm_preflight_sha256"]
    assert preparation["status"] == "complete"
    assert preparer.PREPARATION_NAME == "r047-preparation-receipt-v2.json"
    assert preparation["schema_version"] == "r047-m1-preparation-receipt-v2"
    for field, filename in (("scalar_scorer_sha256", "singleton_option_scorer.py"),
                            ("launcher_sha256", "launch_r047_m1_arm.py"),
                            ("atomic_publisher_sha256", "atomic_publish.py")):
        assert preparation[field] == hashlib.sha256(
            Path(preparer.__file__).with_name(filename).read_bytes()).hexdigest()
        assert assignment[field] == preparation[field]
    assert preparation["superseded_preparation_receipt_sha256"] == (
        "5619af841c44bdd210cfef7cf122979b96c2b04e314b83769fb82204c532c070")
    assert preparation["protocol_amendment_sha256"] == hashlib.sha256(
        preparer.PROTOCOL_AMENDMENT_PATH.read_bytes()).hexdigest()
    assert preparation["recipe_sha256"] == result["recipe_sha256"]
    assert preparation["reference_provenance_sha256"] == result[
        "reference_provenance_sha256"]
    assert preparation["assignment_sha256"] == result["assignment_sha256"]
    assert preparation["precheck_sha256"] == assignment["precheck_sha256"]
    assert result["preparation_receipt_sha256"] == hashlib.sha256(
        (args["out"] / preparer.PREPARATION_NAME).read_bytes()).hexdigest()


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


def test_prepare_rejects_complete_role_and_train_swap(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    preparer.prepare(**args)
    assignment = json.loads((args["out"] / preparer.ASSIGNMENT_NAME).read_text())
    assignment["control_slot"], assignment["treatment_slot"] = (
        assignment["treatment_slot"], assignment["control_slot"])
    assignment["control_train_sha256"], assignment["treatment_train_sha256"] = (
        assignment["treatment_train_sha256"], assignment["control_train_sha256"])
    with pytest.raises(ValueError, match="frozen comparator contract"):
        preparer._validate_assignment(assignment)

    assignment = json.loads((args["out"] / preparer.ASSIGNMENT_NAME).read_text())
    assignment["control_train_sha256"], assignment["treatment_train_sha256"] = (
        assignment["treatment_train_sha256"], assignment["control_train_sha256"])
    with pytest.raises(ValueError, match="frozen comparator contract"):
        preparer._validate_assignment(assignment)


def test_prepare_late_failure_is_atomic_and_rerunnable(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    validate_assignment = preparer._validate_assignment
    monkeypatch.setattr(
        preparer, "_validate_assignment",
        lambda _: (_ for _ in ()).throw(ValueError("late assignment failure")))
    with pytest.raises(ValueError, match="late assignment failure"):
        preparer.prepare(**args)
    assert not args["out"].exists()
    assert not list(args["out"].parent.glob(f".{args['out'].name}.tmp-*"))

    monkeypatch.setattr(preparer, "_validate_assignment", validate_assignment)
    result = preparer.prepare(**args)
    assert result["assignment"] == str(args["out"] / preparer.ASSIGNMENT_NAME)


@pytest.mark.parametrize("missing", [
    "singleton_option_scorer.py", "launch_r047_m1_arm.py", "atomic_publish.py"])
def test_prepare_requires_fixed_sibling_source_implementations(tmp_path, monkeypatch, missing):
    args, _ = _fixture(tmp_path, monkeypatch)
    script = _write(tmp_path / "scripts" / "prepare_r047_m1.py", "# preparer\n")
    for filename in ("singleton_option_scorer.py", "launch_r047_m1_arm.py",
                     "atomic_publish.py"):
        if filename != missing:
            _write(script.with_name(filename), "# source\n")
    monkeypatch.setattr(preparer, "__file__", str(script))
    with pytest.raises(ValueError, match="implementation must be a nonempty regular file"):
        preparer.prepare(**args)
    assert not args["out"].exists()
    assert not list(args["out"].parent.glob(f".{args['out'].name}.tmp-*"))


@pytest.mark.parametrize("payload", [None, ""])
def test_prepare_requires_nonempty_protocol_amendment(tmp_path, monkeypatch, payload):
    args, _ = _fixture(tmp_path, monkeypatch)
    amendment = tmp_path / "amendment.md"
    if payload is not None:
        _write(amendment, payload)
    monkeypatch.setattr(preparer, "PROTOCOL_AMENDMENT_PATH", amendment)
    with pytest.raises(ValueError, match="protocol amendment must be a nonempty regular file"):
        preparer.prepare(**args)
    assert not args["out"].exists()
    assert not list(args["out"].parent.glob(f".{args['out'].name}.tmp-*"))
