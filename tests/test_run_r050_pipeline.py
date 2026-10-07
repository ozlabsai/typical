import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_r050_pipeline as pipeline
from scripts.preflight_r043_assets import file_hash, tree_hash


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pipeline._canonical(value))
    return path


def _publish(directory, schema, report, receipt_extra=None, marker_extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True)
    report_path = _write_json(directory / "report.json", report)
    predictions = directory / "predictions.jsonl"
    predictions.write_text('{"id":"fixture"}\n')
    outputs = {
        "report.json": {"sha256": file_hash(report_path), "bytes": report_path.stat().st_size},
        "predictions.jsonl": {"sha256": file_hash(predictions), "bytes": predictions.stat().st_size},
    }
    receipt = {**(receipt_extra or {}), "outputs": outputs}
    receipt_path = _write_json(directory / "receipt.json", receipt)
    marker = {"schema": schema, "status": "complete",
              "receipt_sha256": file_hash(receipt_path), "outputs": outputs,
              **(marker_extra or {})}
    _write_json(directory / "complete.json", marker)


def _config(tmp_path):
    contracts = pipeline._runtime_contracts()
    dummy = tmp_path / "inputs" / "dummy"
    suites = {name: tmp_path / "inputs" / f"{name}.jsonl"
              for name in contracts["retention_suites"]}
    return pipeline.PipelineConfig(
        out=tmp_path / "pipeline", headroom_data=dummy,
        headroom_data_sha256="1" * 64, headroom_manifest=dummy,
        headroom_manifest_sha256="2" * 64, train_panel=dummy,
        train_panel_sha256="3" * 64, eval_panel=dummy,
        eval_panel_sha256="4" * 64, panel_manifest=dummy,
        panel_manifest_sha256="5" * 64, mix=dummy, mix_sha256="6" * 64,
        mix_manifest=dummy, mix_manifest_sha256="7" * 64, model=dummy,
        model_inventory=dummy, model_inventory_sha256="8" * 64,
        parent_adapter=dummy, parent_tree_sha256="9" * 64, parent_tree_files=4,
        retention_asset_inventory=dummy,
        retention_asset_inventory_sha256="a" * 64,
        retention_suites=suites,
        retention_suite_hashes={name: hashlib.sha256(name.encode()).hexdigest()
                                for name in suites},
        code_hashes={name: hashlib.sha256(name.encode()).hexdigest()
                     for name in pipeline.CODE_FILES},
        python=Path(sys.executable))


class FakeGPU:
    def __init__(self, config, *, headroom=True, comparison=None, retention=None, gains=None):
        self.config = config
        self.headroom = headroom
        self.comparison = comparison or {"arm_nll": True, "arm_brier": True}
        self.retention = retention or {"arm_nll": True, "arm_brier": True}
        self.gains = gains or {"arm_nll": (0.1, 0.1), "arm_brier": (0.2, 0.05)}
        self.calls = []
        self.objectives = {
            "arm_nll": "field_ce_plus_exact_outcome_nll",
            "arm_brier": "field_ce_plus_exact_outcome_brier",
            "arm_control": "field_ce",
        }

    @staticmethod
    def _arg(command, name):
        return Path(command[command.index(name) + 1])

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        script = Path(command[1]).name
        output = self._arg(command, "--out")
        returncode = 0
        if script == "run_r050_headroom.py":
            _publish(output, "r050_headroom_completion_v1",
                     {"headroom_sufficient": self.headroom},
                     {"headroom_sufficient": self.headroom},
                     {"headroom_sufficient": self.headroom})
            returncode = 0 if self.headroom else 2
        elif script == "probe_r050_gradient_scale.py":
            _write_json(output, {"schema_version": "r050-gradient-scale-probe-v1",
                                 "status": "complete"})
        elif script == "prepare_r050_arms.py":
            output.mkdir(parents=True)
            arms = []
            for order, arm_id in enumerate(("arm_nll", "arm_brier", "arm_control"), 1):
                recipe = {"schema": "r050-matched-arm-recipe-v1", "arm_id": arm_id,
                          "run_order": order,
                          "objective": {"objective": self.objectives[arm_id]}}
                path = _write_json(output / f"{arm_id}.json", recipe)
                arms.append({"arm_id": arm_id, "run_order": order,
                             "recipe": path.name, "sha256": file_hash(path)})
            _write_json(output / "manifest.json",
                        {"schema": "r050-matched-arm-suite-v1", "arms": arms})
        elif script == "run_r050_arm.py":
            recipe = json.loads(self._arg(command, "--recipe").read_text())
            output.mkdir(parents=True)
            adapter = output / "step-000400"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter / "adapter_config.json").write_text("{}\n")
            digest, files = tree_hash(adapter)
            _write_json(output / "completion.json", {
                "schema": "r050-matched-arm-completion-v1", "status": "complete",
                "arm_id": recipe["arm_id"], "run_order": recipe["run_order"],
                "recipe_sha256": self._arg(command, "--expected-recipe-sha256").name,
                "artifacts": {"checkpoint": {"path": str(adapter.resolve()),
                                                "tree_sha256": digest, "files": files}},
            })
        elif script == "run_r050_eval.py":
            completion = self._arg(command, "--arm-completion")
            prediction_payload = b'{"id":"fixture"}\n'
            output.mkdir(parents=True)
            (output / "predictions.jsonl").write_bytes(prediction_payload)
            report_path = _write_json(output / "report.json", {"rendered": {}})
            outputs = {
                "predictions.jsonl": {"sha256": hashlib.sha256(prediction_payload).hexdigest(),
                                      "bytes": len(prediction_payload)},
                "report.json": {"sha256": file_hash(report_path), "bytes": report_path.stat().st_size},
            }
            receipt = _write_json(output / "receipt.json", {
                "arm_completion": {"sha256": file_hash(completion)}, "outputs": outputs})
            _write_json(output / "complete.json", {
                "schema": "r050_full_eval_completion_v1", "status": "complete",
                "receipt_sha256": file_hash(receipt), "outputs": outputs,
                "prediction_sha256": outputs["predictions.jsonl"]["sha256"],
                "report_sha256": outputs["report.json"]["sha256"],
            })
        elif script == "compare_r049_calibrated.py":
            candidate = self._arg(command, "--candidate-predictions").parent.name
            outcome, brier = self.gains[candidate]
            _write_json(output, {
                "schema": pipeline._runtime_contracts()["comparator_schema"],
                "provenance": {"inputs": {
                    "panel": {"sha256": command[command.index("--panel-sha256") + 1]},
                    "control": {"sha256": command[command.index("--control-sha256") + 1]},
                    "candidate": {"sha256": command[command.index("--candidate-sha256") + 1]},
                }},
                "internal_stage1_gate": {"passed": self.comparison[candidate]},
                "overall": {"paired_delta": {"equal_semantic_cluster": {
                    "complete_outcome_accuracy": outcome,
                    "categorical_brier_absolute_improvement": brier,
                }}},
            })
        elif script == "run_r050_retention.py":
            candidate = self._arg(command, "--candidate-adapter").parent.name
            passed = self.retention[candidate]
            _publish(output, pipeline._runtime_contracts()["retention_completion_schema"],
                     {"promotion_pass": passed}, {}, {"promotion_pass": passed})
            returncode = 0 if passed else 2
        else:
            raise AssertionError(script)
        return SimpleNamespace(returncode=returncode)


def _run(monkeypatch, config, fake):
    binding = config.binding()
    monkeypatch.setattr(pipeline, "preflight", lambda *_args, **_kwargs: {
        "binding": binding, "binding_sha256": pipeline._sha(pipeline._canonical(binding))})
    return pipeline.Pipeline(config, subprocess_run=fake).run()


def test_insufficient_headroom_seals_and_stops_before_probe(monkeypatch, tmp_path):
    config = _config(tmp_path)
    fake = FakeGPU(config, headroom=False)
    result = _run(monkeypatch, config, fake)
    assert result["status"] == "headroom_insufficient" and result["winner"] is None
    assert [Path(call[0][1]).name for call in fake.calls] == ["run_r050_headroom.py"]
    assert (config.out / "complete.json").is_file()


def test_partial_output_is_never_adopted(monkeypatch, tmp_path):
    config = _config(tmp_path)
    binding = config.binding()
    config.out.mkdir()
    _write_json(config.out / "state.json", {"schema": pipeline.SCHEMA,
                "binding": binding, "binding_sha256": pipeline._sha(pipeline._canonical(binding)),
                "stages": {}})
    (config.out / "headroom").mkdir()
    fake = FakeGPU(config)
    monkeypatch.setattr(pipeline, "preflight", lambda *_args, **_kwargs: {
        "binding": binding, "binding_sha256": pipeline._sha(pipeline._canonical(binding))})
    with pytest.raises(FileExistsError, match="partial"):
        pipeline.Pipeline(config, subprocess_run=fake).run()
    assert not fake.calls


def test_resume_revalidates_hashes_and_rejects_drift(monkeypatch, tmp_path):
    config = _config(tmp_path)
    fake = FakeGPU(config, headroom=False)
    _run(monkeypatch, config, fake)
    (config.out / "headroom" / "report.json").write_text("{}\n")
    with pytest.raises((ValueError, KeyError), match="completion|sufficiency|drift"):
        _run(monkeypatch, config, FakeGPU(config, headroom=False))


def test_control_is_unblinded_from_recipe_after_every_eval_and_argv_is_exact(monkeypatch, tmp_path):
    config = _config(tmp_path)
    fake = FakeGPU(config)
    _run(monkeypatch, config, fake)
    names = [Path(call[0][1]).name for call in fake.calls]
    assert names[:8] == ["run_r050_headroom.py", "probe_r050_gradient_scale.py",
                         "prepare_r050_arms.py", "run_r050_arm.py", "run_r050_arm.py",
                         "run_r050_arm.py", "run_r050_eval.py", "run_r050_eval.py"]
    assert names[8] == "run_r050_eval.py"
    compare_calls = [call for call, _ in fake.calls if Path(call[1]).name == "compare_r049_calibrated.py"]
    assert len(compare_calls) == 2
    assert all(Path(call[call.index("--control-predictions") + 1]).parent.name == "arm_control"
               for call in compare_calls)
    first = fake.calls[0]
    assert first[1]["check"] is False and "shell" not in first[1]
    assert first[0] == [str(config.python.resolve()),
                        str((config.scripts / "run_r050_headroom.py").resolve()),
                        "--data", str(config.headroom_data), "--manifest", str(config.headroom_manifest),
                        "--model", str(config.model), "--inventory", str(config.model_inventory),
                        "--adapter", str(config.parent_adapter), "--out", str(config.out.resolve() / "headroom"),
                        "--expected-data-sha256", config.headroom_data_sha256,
                        "--expected-manifest-sha256", config.headroom_manifest_sha256,
                        "--expected-inventory-sha256", config.model_inventory_sha256,
                        "--expected-adapter-tree-sha256", config.parent_tree_sha256,
                        "--expected-adapter-tree-files", str(config.parent_tree_files)]


@pytest.mark.parametrize(
    ("comparison", "retention", "gains", "winner"),
    [
        ({"arm_nll": True, "arm_brier": False}, {"arm_nll": True},
         {"arm_nll": (0.1, 0.1), "arm_brier": (9.0, 9.0)}, "arm_nll"),
        ({"arm_nll": True, "arm_brier": True}, {"arm_nll": True, "arm_brier": True},
         {"arm_nll": (0.2, 0.1), "arm_brier": (0.2, 0.2)}, "arm_brier"),
        ({"arm_nll": False, "arm_brier": True}, {"arm_brier": False},
         {"arm_nll": (0.1, 0.1), "arm_brier": (0.2, 0.2)}, None),
    ],
)
def test_deterministic_winner_for_one_two_or_no_eligible(
        monkeypatch, tmp_path, comparison, retention, gains, winner):
    config = _config(tmp_path)
    fake = FakeGPU(config, comparison=comparison, retention=retention, gains=gains)
    result = _run(monkeypatch, config, fake)
    assert result["winner"] == winner
    retention_calls = [call for call, _ in fake.calls
                       if Path(call[1]).name == "run_r050_retention.py"]
    assert len(retention_calls) == sum(comparison.values())
    assert not any("jevbench" in " ".join(call).lower() or "workfloweval" in " ".join(call).lower()
                   for call, _ in fake.calls)


def test_atomic_new_has_one_winner_and_never_overwrites(tmp_path):
    target = tmp_path / "complete.json"
    pipeline._atomic_new(target, {"winner": "first"})
    with pytest.raises(FileExistsError, match="overwrite"):
        pipeline._atomic_new(target, {"winner": "second"})
    assert json.loads(target.read_text()) == {"winner": "first"}
