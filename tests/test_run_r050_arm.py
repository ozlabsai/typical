import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_r050_arm as runner
from scripts.prepare_r050_arms import SCHEMA, TRAINING, _canonical


def _recipe(tmp_path, objective="field_ce_plus_exact_outcome_brier"):
    trainer = tmp_path / "train_gemma_decision_lora.py"
    trainer.write_text("# fake trainer\n")
    kinds = {
        "field_ce": (0.0, "nll"),
        "field_ce_plus_exact_outcome_nll": (0.125, "nll"),
        "field_ce_plus_exact_outcome_brier": (0.0625, "brier"),
    }
    weight, kind = kinds[objective]
    value = {
        "schema": SCHEMA, "arm_id": "slot_0123456789ab", "run_order": 2,
        "objective": {"objective": objective, "action_loss_weight": weight,
                      "exact_policy_outcome_loss": kind},
        "training": dict(TRAINING),
        "epoch_contract": {"optimizer_slots": 3200, "slots_per_update": 8,
                           "updates": 400, "passes": 1,
                           "slot_order": "trainer seeded global shuffle"},
        "bindings": {
            "trainer": {"path": str(trainer.resolve())},
            "model": {"path": str((tmp_path / "model").resolve())},
            "parent_adapter": {"path": str((tmp_path / "parent").resolve())},
            "mix": {"path": str((tmp_path / "mix.jsonl").resolve())},
        },
    }
    path = tmp_path / "recipe.json"
    path.write_bytes(_canonical(value))
    return path, value, hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_success(recipe, *, metrics_kind=None, returncode=0):
    def invoke(command, **kwargs):
        kwargs["stdout"].write(b"fake trainer output\n")
        kwargs["stdout"].flush()
        if returncode:
            return SimpleNamespace(returncode=returncode)
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        (output / "config.json").write_text(
            json.dumps(runner.trainer_config(recipe, output)) + "\n")
        kind = metrics_kind or recipe["objective"]["exact_policy_outcome_loss"]
        rows = [{"step": 0, "validation": {}}]
        rows.extend({
            "step": step, "train_loss": 0.1, "action_groups_seen": 4,
            "action_loss_weight": recipe["objective"]["action_loss_weight"],
            "exact_policy_outcome_loss": recipe["objective"]["exact_policy_outcome_loss"],
            "action_loss_kind": kind,
        } for step in range(1, 401))
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        checkpoint = output / "step-000400"
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        (checkpoint / "adapter_config.json").write_text("{}\n")
        (checkpoint / "trainer.pt").write_bytes(b"state")
        return SimpleNamespace(returncode=0)
    return invoke


def test_runner_invokes_exact_recipe_and_writes_completion_last(tmp_path, monkeypatch):
    recipe_path, recipe, digest = _recipe(tmp_path)
    monkeypatch.setattr(runner, "verify_bindings", lambda value: None)
    monkeypatch.setattr(runner.subprocess, "run", _fake_success(recipe))
    output = tmp_path / "run"

    receipt = runner.run(recipe_path, digest, output)

    assert receipt["status"] == "complete"
    assert receipt["artifacts"]["metrics"]["final_step"] == 400
    assert receipt["artifacts"]["checkpoint"]["path"].endswith("step-000400")
    assert (output / "completion.json").read_bytes() == _canonical(receipt)
    assert (output / "train.log").read_text() == "fake trainer output\n"
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run(recipe_path, digest, output)


def test_runner_rejects_wrong_action_objective_in_metrics(tmp_path, monkeypatch):
    recipe_path, recipe, digest = _recipe(tmp_path)
    monkeypatch.setattr(runner, "verify_bindings", lambda value: None)
    monkeypatch.setattr(runner.subprocess, "run",
                        _fake_success(recipe, metrics_kind="nll"))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="wrong or incomplete action objective"):
        runner.run(recipe_path, digest, output)
    assert not (output / "completion.json").exists()


def test_runner_preserves_nonzero_log_and_never_seals(tmp_path, monkeypatch):
    recipe_path, recipe, digest = _recipe(tmp_path, objective="field_ce")
    monkeypatch.setattr(runner, "verify_bindings", lambda value: None)
    monkeypatch.setattr(runner.subprocess, "run", _fake_success(recipe, returncode=7))
    with pytest.raises(RuntimeError, match="status 7"):
        runner.run(recipe_path, digest, tmp_path / "run")
    pending = tmp_path / ".run.slot_0123456789ab.train.log"
    assert pending.read_text() == "fake trainer output\n"
    assert not (tmp_path / "run" / "completion.json").exists()


def test_recipe_rejects_rehashed_wrong_objective_and_boolean_integer(tmp_path):
    recipe_path, recipe, _ = _recipe(tmp_path)
    recipe["objective"]["exact_policy_outcome_loss"] = "nll"
    recipe_path.write_bytes(_canonical(recipe))
    digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="wrong action objective"):
        runner.validate_recipe(recipe_path, digest)

    recipe["objective"]["exact_policy_outcome_loss"] = "brier"
    recipe["training"]["microbatch"] = True
    recipe_path.write_bytes(_canonical(recipe))
    digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="matched training configuration"):
        runner.validate_recipe(recipe_path, digest)
