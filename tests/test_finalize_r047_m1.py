import hashlib
import json
from pathlib import Path

import pytest

import scripts.finalize_r047_m1 as finalizer
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, build_inventory
from scripts.eval_r047_executor import RUN_RECEIPT_FIELDS, validate_causal_run_receipt


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


def _view(row_id, label):
    return {
        "id": row_id,
        "state": "state",
        "questions": {"q": {"type": "choice", "criteria": {"no": "n", "yes": "y"}}},
        "targets": {"q": {"label": label}},
        "meta": {"source_id": row_id.rsplit("-", 1)[0]},
    }


def _rows():
    rows = []
    for index in range(100):
        source = f"pair-{index:03d}"
        row = {
            "id": source,
            "pair_mode": "counterfactual",
            "source_ids": [source, source],
            "paired_views": [_view(f"{source}-no", "no"), _view(f"{source}-yes", "yes")],
        }
        if index < 48:
            row["meta"] = {"composition": ["decimal", "ledger"]}
            row["auxiliary_views"] = [
                {
                    "targets": {"q": {"label": "yes" if aux_index else "no"}},
                    "meta": {
                        "language": "en" if aux_index % 2 == 0 else "he",
                        "program": {"operator": "all"},
                        "r047_auxiliary": {"reference_clause_kind": "decimal"},
                    },
                }
                for aux_index in range(4)
            ]
        rows.append(row)
    rows.extend(_view(f"single-{index:03d}", "yes") for index in range(700))
    return rows


def _trainer_config():
    values = {
        "steps": 400, "microbatch": 1, "grad_accum": 8, "max_length": 8192,
        "lr": 2e-5, "min_lr": 2e-6, "weight_decay": .01, "rank": 32,
        "alpha": 64, "save_every": 400, "save_steps": "", "eval_every": 400,
        "eval_samples": 128, "seed": 47, "pair_consistency_weight": 0.0,
        "pair_consistency_min_gold_probability": .8, "pair_consistency_clip": 4.0,
        "pair_consistency_beta": 1.0, "project_consistency_against_guard": False,
        "pair_flip_weight": 1.0, "pair_flip_margin": 1.0,
        "pair_flip_start_step": 0, "stage_counterfactual_pairs": False,
    }
    assert set(values) == finalizer.TRAINER_CONFIG_KEYS
    return values


def _metrics():
    result = [{"step": 0, "validation": {}}]
    for step in range(1, 401):
        within_epoch = (step - 1) % 100
        result.append({
            "step": step,
            "train_loss": 0.1,
            "canonical_views_seen": 9,
            "auxiliary_views_seen": 2 if within_epoch < 92 else 1,
            "pair_flip_seen": 1,
            "pair_flip_weight": 1.0,
            "pair_flip_weight_configured": 1.0,
        })
    return result


def _fixture(tmp_path, monkeypatch):
    model = (tmp_path / "models--google--gemma-4-12B-it" / "snapshots"
             / FROZEN_BASE_REVISION)
    _write(model / "config.json", {})
    _write(model / "model.safetensors", b"model")
    inventory = _write(tmp_path / "model-inventory.json", build_inventory(model))
    inventory_hash = hashlib.sha256(inventory.read_bytes()).hexdigest()

    parent = _write(tmp_path / "parent" / "adapter_model.safetensors", b"parent")
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_SHA256", parent_hash)

    rows = _rows()
    train = _write(
        tmp_path / "clause.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    train_hash = hashlib.sha256(train.read_bytes()).hexdigest()
    control_hash = hashlib.sha256(b"control").hexdigest()
    arm_hashes = {"control": control_hash, "treatment": train_hash}
    monkeypatch.setattr(finalizer, "KNOWN_ARM_SHA256", arm_hashes)
    summary = finalizer._summary(rows)
    manifest = _write(tmp_path / "manifest.json", {
        "outputs": {"control": {"rows": 800, "sha256": control_hash},
                    "clause": {"rows": 800, "sha256": train_hash}},
        "arms": {"control": summary, "clause": summary},
    })
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(finalizer, "FROZEN_ARM_MANIFEST_SHA256", manifest_hash)

    trainer = _write(tmp_path / "train_gemma_decision_lora.py", "# frozen trainer\n")
    recipe = _write(tmp_path / "recipe.json", {
        "schema_version": finalizer.RECIPE_SCHEMA_VERSION,
        "base_revision": FROZEN_BASE_REVISION,
        "parent_adapter_sha256": parent_hash,
        "model_inventory_sha256": inventory_hash,
        "arm_manifest_sha256": manifest_hash,
        "arm_sha256": arm_hashes,
        "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
        "trainer_config": _trainer_config(),
    })
    checkpoint = tmp_path / "run" / "step-000400"
    adapter = _write(checkpoint / "adapter_model.safetensors", b"adapter")
    _write(checkpoint / "adapter_config.json", {})
    _write(checkpoint / "trainer.pt", b"trainer-state")
    config = _write(tmp_path / "run" / "config.json", {
        **_trainer_config(),
        "model": str(model),
        "init_adapter": str(parent.parent),
        "train": [str(train)],
        "val": [],
        "out": str(checkpoint.parent),
    })
    metrics = _write(
        tmp_path / "run" / "metrics.jsonl",
        "".join(json.dumps(item) + "\n" for item in _metrics()),
    )
    log = _write(tmp_path / "run" / "train.log", "completed\n")
    values = {
        "slot": "slot_a", "train_path": train, "arm_manifest_path": manifest,
        "recipe_path": recipe, "config_path": config, "metrics_path": metrics,
        "log_path": log, "checkpoint_path": checkpoint, "parent_adapter": parent,
        "model_inventory_path": inventory, "base_revision": FROZEN_BASE_REVISION,
        "model_path": model, "trainer_path": trainer,
    }
    return values, adapter


def _update_json(path, update):
    value = json.loads(Path(path).read_text())
    update(value)
    _write(path, value)


def test_finalize_emits_exact_evaluator_receipt_contract(tmp_path, monkeypatch):
    values, adapter = _fixture(tmp_path, monkeypatch)
    receipt = finalizer.finalize(**values)
    assert set(receipt) == RUN_RECEIPT_FIELDS
    assert receipt["checkpoint_step"] == 400
    assert receipt["status"] == "complete"
    adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert receipt["adapter_sha256"] == adapter_hash
    assert validate_causal_run_receipt(json.dumps(receipt).encode(), adapter_hash) == receipt


@pytest.mark.parametrize("mutation,match", [
    (lambda values: _update_json(
        values["config_path"], lambda item: item.__setitem__("grad_accum", 4)),
     "hash-bound recipe"),
    (lambda values: _update_json(
        values["recipe_path"], lambda item: item["trainer_config"].__setitem__("steps", 399)),
     "training invariant"),
    (lambda values: Path(values["log_path"]).write_text(""), "nonempty regular file"),
])
def test_finalize_fails_closed_on_recipe_config_or_log_drift(
        tmp_path, monkeypatch, mutation, match):
    values, _ = _fixture(tmp_path, monkeypatch)
    mutation(values)
    with pytest.raises(ValueError, match=match):
        finalizer.finalize(**values)


def test_finalize_rejects_boolean_disguised_as_integer(tmp_path, monkeypatch):
    values, _ = _fixture(tmp_path, monkeypatch)
    _update_json(values["recipe_path"],
                 lambda item: item["trainer_config"].__setitem__("microbatch", True))
    with pytest.raises(ValueError, match="invalid value types"):
        finalizer.finalize(**values)


def test_finalize_rejects_incomplete_nonfinite_or_wrong_epoch_metrics(tmp_path, monkeypatch):
    values, _ = _fixture(tmp_path, monkeypatch)
    metrics = _metrics()
    metrics[101]["train_loss"] = float("nan")
    _write(values["metrics_path"], "".join(json.dumps(item) + "\n" for item in metrics))
    with pytest.raises(ValueError, match="non-finite"):
        finalizer.finalize(**values)

    metrics[101]["train_loss"] = 0.1
    metrics[100]["auxiliary_views_seen"] -= 1
    _write(values["metrics_path"], "".join(json.dumps(item) + "\n" for item in metrics))
    with pytest.raises(ValueError, match="epoch 1.*auxiliary_views_seen"):
        finalizer.finalize(**values)

    _write(values["metrics_path"], "".join(json.dumps(item) + "\n" for item in metrics[:-1]))
    with pytest.raises(ValueError, match="contiguous steps"):
        finalizer.finalize(**values)


def test_finalize_rejects_model_and_checkpoint_payload_drift(tmp_path, monkeypatch):
    values, _ = _fixture(tmp_path, monkeypatch)
    _write(Path(values["model_path"]) / "model.safetensors", b"changed")
    with pytest.raises(ValueError, match="differs from model inventory"):
        finalizer.finalize(**values)

    values, _ = _fixture(tmp_path / "fresh", monkeypatch)
    (Path(values["checkpoint_path"]) / "trainer.pt").unlink()
    with pytest.raises(ValueError, match="missing a required"):
        finalizer.finalize(**values)


def test_checkpoint_tree_hash_binds_every_payload(tmp_path, monkeypatch):
    values, _ = _fixture(tmp_path, monkeypatch)
    before = finalizer.finalize(**values)["checkpoint_payload_sha256"]
    _write(Path(values["checkpoint_path"]) / "README.md", "metadata\n")
    after = finalizer.finalize(**values)["checkpoint_payload_sha256"]
    assert before != after


def test_finalize_rejects_unknown_train_bytes_and_reference_slot(tmp_path, monkeypatch):
    values, _ = _fixture(tmp_path, monkeypatch)
    values["slot"] = "reference"
    with pytest.raises(ValueError, match="slot_a or slot_b"):
        finalizer.finalize(**values)
    values["slot"] = "slot_b"
    with Path(values["train_path"]).open("a") as target:
        target.write("{}\n")
    with pytest.raises(ValueError, match="not either frozen R047 arm"):
        finalizer.finalize(**values)
