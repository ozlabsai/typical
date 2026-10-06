import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_compare_r047_causal as causal_fixtures  # noqa: E402
import test_precheck_r047_causal as precheck_fixtures  # noqa: E402

from scripts import finalize_r047_m1 as finalizer  # noqa: E402
from scripts import launch_r047_m1_arm as launcher  # noqa: E402
from scripts import precheck_r047_causal as precheck  # noqa: E402
from scripts import compare_r047_causal as comparator  # noqa: E402
from scripts import eval_r047_executor as evaluator  # noqa: E402


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if isinstance(payload, bytes) else _payload(payload))
    return path


def _fixture(tmp_path, monkeypatch):
    artifacts = causal_fixtures._artifacts()
    assignment = json.loads(artifacts["assignment"])
    provenance = json.loads(artifacts["reference_provenance"])
    recipe = {
        "schema_version": finalizer.RECIPE_SCHEMA_VERSION,
        "trainer_sha256": _sha(b"trainer"),
        "trainer_config": dict(finalizer.FROZEN_TRAINING_INVARIANTS),
        "arm_sha256": {
            "control": assignment["control_train_sha256"],
            "treatment": assignment["treatment_train_sha256"],
        },
        "model_inventory_sha256": assignment["model_inventory_sha256"],
        "parent_adapter_sha256": assignment["parent_adapter_sha256"],
        "parent_adapter_config_sha256": provenance["adapter_config_sha256"],
        "parent_adapter_tree_sha256": provenance["adapter_tree_sha256"],
        "parent_adapter_tree_files": provenance["adapter_tree_files"],
        "image_tree_sha256": "4" * 64,
        "image_tree_files": 1,
    }
    recipe_payload = _payload(recipe)
    assignment["recipe_sha256"] = _sha(recipe_payload)
    assignment_payload = _payload(assignment)
    artifacts = {**artifacts, "assignment": assignment_payload}

    slot_b_receipt = json.loads(artifacts["receipts"]["slot_b"])
    slot_b_receipt["recipe_sha256"] = assignment["recipe_sha256"]
    slot_b_receipt_payload = _payload(slot_b_receipt)
    slot_b_report = json.loads(artifacts["reports"]["slot_b"])
    slot_b_report["recipe_sha256"] = assignment["recipe_sha256"]
    slot_b_report["run_receipt_sha256"] = _sha(slot_b_receipt_payload)
    artifacts["receipts"]["slot_b"] = slot_b_receipt_payload
    artifacts["reports"]["slot_b"] = _payload(slot_b_report)

    preparation = precheck_fixtures._preparation_payload(artifacts, recipe_payload)
    reference_args = {
        **precheck_fixtures._reference_args(artifacts, recipe_payload),
        "preparation_receipt_payload": preparation,
        "preparation_receipt_sha256": _sha(preparation),
    }
    reference = precheck.precheck_reference(**reference_args)
    launch_chain = precheck_fixtures._launch_chain_inputs(
        artifacts, reference, recipe_payload, preparation)
    control = precheck.precheck_control(**{
        **reference_args,
        **launch_chain,
        "reference_receipt_payload": precheck._receipt_payload(reference),
        "reference_receipt_sha256": _sha(precheck._receipt_payload(reference)),
        "slot_b_predictions": artifacts["predictions"]["slot_b"],
        "slot_b_report": artifacts["reports"]["slot_b"],
        "slot_b_report_sha256": _sha(artifacts["reports"]["slot_b"]),
        "slot_b_run_receipt": slot_b_receipt_payload,
        "slot_b_run_receipt_sha256": _sha(slot_b_receipt_payload),
    })

    reference_paths = {
        "assignment": _write(tmp_path / "assignment.json", assignment_payload),
        "reference_predictions": _write(tmp_path / "reference-predictions.jsonl",
                                         artifacts["predictions"]["reference"]),
        "reference_report": _write(tmp_path / "reference-report.json",
                                    artifacts["reports"]["reference"]),
        "reference_provenance": _write(tmp_path / "reference-provenance.json",
                                        artifacts["reference_provenance"]),
        "recipe": _write(tmp_path / "recipe.json", recipe_payload),
        "preparation_receipt": _write(tmp_path / "preparation.json", preparation),
        "panel": _write(tmp_path / "panel.jsonl", artifacts["panel"]),
        "panel_manifest": _write(tmp_path / "panel-manifest.json", artifacts["manifest"]),
        "panel_preflight": _write(tmp_path / "panel-preflight.json", artifacts["preflight"]),
        "inventory_manifest": _write(tmp_path / "inventory.json", artifacts["inventory"]),
        "arm_preflight": _write(tmp_path / "arm-preflight.json", artifacts["arm_preflight"]),
        "reference_receipt": _write(tmp_path / "reference-receipt.json",
                                    precheck._receipt_payload(reference)),
    }
    control_paths = {
        "slot_b_predictions": _write(tmp_path / "slot-b-predictions.jsonl",
                                      artifacts["predictions"]["slot_b"]),
        "slot_b_report": _write(tmp_path / "slot-b-report.json",
                                 artifacts["reports"]["slot_b"]),
        "slot_b_run_receipt": _write(tmp_path / "slot-b-run-receipt.json",
                                      slot_b_receipt_payload),
        "slot_b_launch_intent": _write(tmp_path / "slot-b-launch-intent.json",
                                        launch_chain["slot_b_launch_intent_payload"]),
        "slot_b_launch_completion": _write(tmp_path / "slot-b-launch-completion.json",
                                            launch_chain["slot_b_launch_completion_payload"]),
        "control_receipt": _write(tmp_path / "control-receipt.json",
                                  precheck._receipt_payload(control)),
    }
    paths = {
        "control_train": _write(tmp_path / "control.jsonl", b"control\n"),
        "treatment_train": _write(tmp_path / "treatment.jsonl", b"treatment\n"),
        "trainer": _write(tmp_path / "scripts" / "train_gemma_decision_lora.py", b"trainer"),
        "model_inventory": _write(tmp_path / "model-inventory.json", b"{}\n"),
        "model": tmp_path / "model", "parent": _write(
            tmp_path / "parent" / "step-000100" / "adapter_model.safetensors", b"parent"),
        "arm_manifest": _write(tmp_path / "arm-manifest.json", b"{}\n"),
        "image_root": tmp_path / "images", "source": _write(
            tmp_path / "source.json", b"reference-provenance"),
        "evaluator": Path(evaluator.__file__),
        "comparator": Path(comparator.__file__),
        "precheck": Path(precheck.__file__),
    }
    paths["model"].mkdir()
    paths["image_root"].mkdir()
    original_file_hash = launcher.file_hash

    def fake_hash(path):
        path = Path(path)
        if path == paths["control_train"]:
            return assignment["control_train_sha256"]
        if path == paths["treatment_train"]:
            return assignment["treatment_train_sha256"]
        return original_file_hash(path)

    monkeypatch.setattr(launcher, "file_hash", fake_hash)
    monkeypatch.setattr(launcher, "verify_inventory",
                        lambda *args: ({}, assignment["model_inventory_sha256"]))
    parent_binding = {key: recipe[key] for key in (
        "parent_adapter_sha256", "parent_adapter_config_sha256",
        "parent_adapter_tree_sha256", "parent_adapter_tree_files")}
    monkeypatch.setattr(finalizer, "_validate_parent", lambda path: parent_binding)
    monkeypatch.setattr(finalizer, "_validate_train", lambda path, *args: (
        fake_hash(path), {"image_tree_sha256": recipe["image_tree_sha256"],
                          "image_tree_files": recipe["image_tree_files"]}))
    monkeypatch.setattr(finalizer, "finalize", lambda **kwargs: {
        "schema_version": "r047-causal-run-receipt-v1", "slot": kwargs["slot"],
        "status": "complete", "adapter_sha256": "a" * 64,
    })
    return reference_paths, control_paths, paths, assignment


def _arguments(tmp_path, fixture, slot="slot_b"):
    reference, control, paths, _ = fixture
    return {
        "slot": slot, "out": tmp_path / f"out-{slot}",
        "train_path": paths["control_train"] if slot == "slot_b" else paths["treatment_train"],
        "arm_manifest_path": paths["arm_manifest"], "image_root": paths["image_root"],
        "model_inventory_path": paths["model_inventory"], "model_path": paths["model"],
        "parent_adapter": paths["parent"], "trainer_path": paths["trainer"],
        "evaluator_path": paths["evaluator"], "comparator_path": paths["comparator"],
        "precheck_path": paths["precheck"],
        "source_checkpoint_provenance_path": paths["source"],
        "reference_paths": reference,
        "control_paths": None if slot == "slot_b" else control,
        "require_frozen_panel": False, "require_frozen_assignment": False,
    }


class Completed:
    stdout = "trainer completed\n"
    stderr = ""


def test_control_launch_reruns_reference_and_executes_exact_frozen_command(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    calls = []

    def run(command, cwd):
        run_out = Path(command[command.index("--out") + 1])
        intent_path = run_out.parent / "launch-intent.json"
        assert intent_path.is_file()
        intent = json.loads(intent_path.read_text())
        preparation = json.loads(fixture[0]["preparation_receipt"].read_bytes())
        assert intent["schema_version"] == "r047-m1-launch-intent-v2"
        assert intent["preparation_receipt_sha256"] == _sha(
            fixture[0]["preparation_receipt"].read_bytes())
        assert intent["launcher_sha256"] == preparation["launcher_sha256"] == _sha(
            Path(launcher.__file__).read_bytes())
        assert intent["prerequisite_receipt_sha256"] == {
            "reference_receipt_sha256": _sha(fixture[0]["reference_receipt"].read_bytes())}
        calls.append((command, cwd))
        return Completed()

    result = launcher.launch(**_arguments(tmp_path, fixture),
                             _run=run)
    assert len(calls) == 1
    command = calls[0][0]
    assert command[1] == str(fixture[2]["trainer"].resolve())
    assert command[command.index("--steps") + 1] == "400"
    assert command[command.index("--max-length") + 1] == "8192"
    assert command[command.index("--pair-flip-weight") + 1] == "1.0"
    assert "--val" not in command
    completion = json.loads(Path(result["completion"]).read_text())
    assert completion["schema_version"] == "r047-m1-launch-completion-v2"
    assert completion["status"] == "trainer_completed"
    assert completion["launch_intent_sha256"] == _sha(Path(result["intent"]).read_bytes())
    assert completion["run_receipt_sha256"] == _sha(Path(result["run_receipt"]).read_bytes())
    assert set(completion["prerequisite_receipt_sha256"]) == {"reference_receipt_sha256"}


def test_treatment_requires_control_and_revalidates_both_receipts(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args = _arguments(tmp_path, fixture, "slot_a")
    with pytest.raises(ValueError, match="complete immutable control evidence"):
        launcher.launch(**{**args, "control_paths": None}, _run=lambda *_: Completed())
    result = launcher.launch(**args, _run=lambda *_: Completed())
    completion = json.loads(Path(result["completion"]).read_text())
    assert set(completion["prerequisite_receipt_sha256"]) == {
        "reference_receipt_sha256", "control_receipt_sha256"}


@pytest.mark.parametrize("receipt", ["reference", "control"])
def test_substituted_or_rehashed_prerequisite_receipt_fails(tmp_path, monkeypatch, receipt):
    fixture = _fixture(tmp_path, monkeypatch)
    args = _arguments(tmp_path, fixture, "slot_a" if receipt == "control" else "slot_b")
    path = (fixture[0]["reference_receipt"] if receipt == "reference"
            else fixture[1]["control_receipt"])
    value = json.loads(path.read_text())
    value["status"] = "pass-but-substituted"
    value.pop("binding_sha256")
    precheck._with_binding(value)
    path.write_bytes(_payload(value))
    with pytest.raises(ValueError, match=f"immutable {receipt} receipt"):
        launcher.launch(**args, _run=lambda *_: Completed())


def test_missing_reference_receipt_fails_before_invocation(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[0]["reference_receipt"].unlink()
    with pytest.raises(ValueError, match="reference receipt"):
        launcher.launch(**_arguments(tmp_path, fixture), _run=lambda *_: Completed())


def test_wrong_slot_train_mapping_fails(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args = _arguments(tmp_path, fixture)
    args["train_path"] = fixture[2]["treatment_train"]
    with pytest.raises(ValueError, match="slot/train payload"):
        launcher.launch(**args, _run=lambda *_: Completed())


def test_failure_cleans_scratch_and_is_rerunnable(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args = _arguments(tmp_path, fixture)

    def fail(command, cwd):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        launcher.launch(**args, _run=fail)
    assert not Path(args["out"]).exists()
    assert not list(tmp_path.glob(f".{Path(args['out']).name}.tmp-*"))
    launcher.launch(**args, _run=lambda *_: Completed())
    assert Path(args["out"]).is_dir()


def test_no_overwrite_even_after_complete_run(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args = _arguments(tmp_path, fixture)
    launcher.launch(**args, _run=lambda *_: Completed())
    with pytest.raises(ValueError, match="overwrite"):
        launcher.launch(**args, _run=lambda *_: Completed())


@pytest.mark.parametrize("receipt", ["reference", "control"])
def test_v1_prerequisite_receipts_fail_before_trainer(tmp_path, monkeypatch, receipt):
    fixture = _fixture(tmp_path, monkeypatch)
    args = _arguments(tmp_path, fixture, "slot_a" if receipt == "control" else "slot_b")
    path = fixture[0]["reference_receipt"] if receipt == "reference" else fixture[1]["control_receipt"]
    value = json.loads(path.read_bytes())
    value["schema_version"] = f"r047-causal-{receipt}-precheck-v1"
    value.pop("binding_sha256")
    precheck._with_binding(value)
    path.write_bytes(_payload(value))
    calls = []
    with pytest.raises(ValueError, match=f"immutable {receipt} receipt"):
        launcher.launch(**args, _run=lambda *_: calls.append(True) or Completed())
    assert not calls


@pytest.mark.parametrize("artifact", ["slot_b_launch_intent", "slot_b_launch_completion"])
def test_treatment_requires_exact_control_launch_chain(tmp_path, monkeypatch, artifact):
    fixture = _fixture(tmp_path, monkeypatch)
    args = _arguments(tmp_path, fixture, "slot_a")
    incomplete = dict(fixture[1])
    del incomplete[artifact]
    with pytest.raises(ValueError, match="complete immutable control evidence"):
        launcher.launch(**{**args, "control_paths": incomplete}, _run=lambda *_: Completed())
    path = fixture[1][artifact]
    value = json.loads(path.read_bytes())
    value["command_plan_sha256"] = "f" * 64
    path.write_bytes(_payload(value))
    calls = []
    with pytest.raises(ValueError, match="frozen launch chain"):
        launcher.launch(**args, _run=lambda *_: calls.append(True) or Completed())
    assert not calls


def test_running_launcher_must_match_preparation_before_trainer(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    original_hash = launcher.file_hash
    monkeypatch.setattr(launcher, "file_hash", lambda path: (
        "f" * 64 if Path(path) == Path(launcher.__file__) else original_hash(path)))
    calls = []
    with pytest.raises(ValueError, match="running launcher"):
        launcher.launch(**_arguments(tmp_path, fixture),
                        _run=lambda *_: calls.append(True) or Completed())
    assert not calls
