import hashlib
import fcntl
import json
import sys
from dataclasses import replace
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
        code_hashes={"run_r050_pipeline.py": file_hash(pipeline.__file__),
                     **pipeline.FROZEN_CODE_SHA256},
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

    def _code(self, *names):
        return {name: self.config.code_hashes[name] for name in names}

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        script = Path(command[1]).name
        output = self._arg(command, "--out")
        returncode = 0
        if script == "run_r050_headroom.py":
            _publish(output, "r050_headroom_completion_v1",
                     {"headroom_sufficient": self.headroom},
                     {"headroom_sufficient": self.headroom, "code_sha256": self._code(
                         "run_r050_headroom.py", "build_r050_headroom.py",
                         "build_r048_action_panel.py", "score_exact_policy_panel.py",
                         "singleton_option_scorer.py", "exact_policy_executor.py",
                         "train_gemma_decision_lora.py", "build_r047_model_inventory.py",
                         "preflight_r043_assets.py", "build_r050_replay_pools.py")},
                     {"headroom_sufficient": self.headroom})
            returncode = 0 if self.headroom else 2
        elif script == "probe_r050_gradient_scale.py":
            thread_contract = {"before": 128, "requested": 1, "during": 1, "after": 128}
            _write_json(output, {"schema_version": "r050-gradient-scale-probe-v1",
                                 "status": "complete", "bindings": {
                                     "trainer_sha256": self.config.code_hashes["train_gemma_decision_lora.py"],
                                     "probe_sha256": self.config.code_hashes["probe_r050_gradient_scale.py"],
                                     "panel_builder_sha256": self.config.code_hashes["build_r048_action_panel.py"],
                                 }, "configuration": {"cpu_intraop_threads": 1},
                                 "gradient_report": {"cpu_thread_contract": thread_contract},
                                 "environment": {"cpu_thread_contract": thread_contract}})
        elif script == "prepare_r050_arms.py":
            output.mkdir(parents=True)
            arms = []
            for order, arm_id in enumerate(("arm_nll", "arm_brier", "arm_control"), 1):
                recipe = {"schema": "r050-matched-arm-recipe-v1", "arm_id": arm_id,
                          "run_order": order,
                          "objective": {"objective": self.objectives[arm_id]},
                          "bindings": {
                              "panel_builder": {"sha256": self.config.code_hashes["build_r048_action_panel.py"]},
                              "trainer": {"sha256": self.config.code_hashes["train_gemma_decision_lora.py"]},
                              "gradient_probe_code": {"sha256": self.config.code_hashes["probe_r050_gradient_scale.py"]},
                          }}
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
                "arm_completion": {"sha256": file_hash(completion)}, "outputs": outputs,
                "code_sha256": self._code(
                    "run_r050_eval.py", "run_r050_headroom.py", "build_r048_action_panel.py",
                    "score_exact_policy_panel.py", "compare_r049_calibrated.py",
                    "singleton_option_scorer.py", "exact_policy_executor.py",
                    "train_gemma_decision_lora.py", "build_r047_model_inventory.py",
                    "preflight_r043_assets.py", "run_r050_arm.py")})
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
                     {"promotion_pass": passed}, {"code_sha256": self._code(
                         "run_r050_retention.py", "singleton_option_scorer.py",
                         "train_gemma_decision_lora.py", "build_r047_model_inventory.py",
                         "preflight_r043_assets.py", "run_r050_eval.py", "run_r050_arm.py",
                         "build_r050_replay_pools.py")}, {"promotion_pass": passed})
            returncode = 0 if passed else 2
        else:
            raise AssertionError(script)
        return SimpleNamespace(returncode=returncode)


def _run(monkeypatch, config, fake):
    binding = config.binding()
    monkeypatch.setattr(pipeline, "preflight", lambda *_args, **_kwargs: {
        "binding": binding, "binding_sha256": pipeline._sha(pipeline._canonical(binding))})
    return pipeline.Pipeline(config, subprocess_run=fake).run()


def _state(config, stages):
    binding = config.binding()
    config.out.mkdir(parents=True, exist_ok=True)
    _write_json(config.out / "state.json", {
        "schema": pipeline.SCHEMA, "binding": binding,
        "binding_sha256": pipeline._sha(pipeline._canonical(binding)), "stages": stages})


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


def test_running_stage_adopts_only_fully_valid_atomic_output(monkeypatch, tmp_path):
    config = _config(tmp_path)
    _state(config, {"01-headroom": {"status": "running"}})
    fake = FakeGPU(config, headroom=False)
    fake([str(config.python), str(config.scripts / "run_r050_headroom.py"),
          "--out", str(config.out / "headroom")], check=False)
    fake.calls.clear()
    result = _run(monkeypatch, config, fake)
    assert result["status"] == "headroom_insufficient"
    assert not fake.calls
    state = json.loads((config.out / "state.json").read_text())
    assert state["stages"]["01-headroom"]["status"] == "complete"


@pytest.mark.parametrize("make_partial", [False, True])
def test_running_stage_rejects_missing_or_partial_output(
        monkeypatch, tmp_path, make_partial):
    config = _config(tmp_path)
    _state(config, {"01-headroom": {"status": "running"}})
    if make_partial:
        (config.out / "headroom").mkdir()
        _write_json(config.out / "headroom" / "receipt.json", {"partial": True})
    fake = FakeGPU(config)
    match = "partial or invalid" if make_partial else "before publishing"
    with pytest.raises(RuntimeError, match=match):
        _run(monkeypatch, config, fake)
    assert not fake.calls


def test_observed_failed_stage_is_forensic_and_never_adopted(monkeypatch, tmp_path):
    config = _config(tmp_path)
    _state(config, {"01-headroom": {"status": "failed"}})
    fake = FakeGPU(config, headroom=False)
    fake([str(config.python), str(config.scripts / "run_r050_headroom.py"),
          "--out", str(config.out / "headroom")], check=False)
    fake.calls.clear()
    with pytest.raises(RuntimeError, match="new output root"):
        _run(monkeypatch, config, fake)
    assert not fake.calls


def test_stale_lock_inode_and_preledger_kill_window_are_recoverable(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.out.mkdir()
    (config.out / ".pipeline.lock").write_text("stale pid data is not authority\n")
    fake = FakeGPU(config, headroom=False)
    result = _run(monkeypatch, config, fake)
    assert result["status"] == "headroom_insufficient"
    assert (config.out / ".pipeline.lock").is_file()


def test_live_flock_rejects_concurrent_invocation(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.out.mkdir()
    lock = (config.out / ".pipeline.lock").open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="active"):
            _run(monkeypatch, config, FakeGPU(config, headroom=False))
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


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
    assert first[0] == [str(config.python.absolute()),
                        str((config.scripts / "run_r050_headroom.py").resolve()),
                        "--data", str(config.headroom_data), "--manifest", str(config.headroom_manifest),
                        "--model", str(config.model), "--inventory", str(config.model_inventory),
                        "--adapter", str(config.parent_adapter), "--out", str(config.out.resolve() / "headroom"),
                        "--expected-data-sha256", config.headroom_data_sha256,
                        "--expected-manifest-sha256", config.headroom_manifest_sha256,
                        "--expected-inventory-sha256", config.model_inventory_sha256,
                        "--expected-adapter-tree-sha256", config.parent_tree_sha256,
                        "--expected-adapter-tree-files", str(config.parent_tree_files)]


def test_child_command_preserves_virtualenv_interpreter_symlink(tmp_path):
    config = _config(tmp_path)
    target = tmp_path / "system-python"
    target.write_text("interpreter fixture\n")
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(target)
    config = replace(config, python=interpreter)

    command = pipeline.Pipeline(config)._command("run_r050_headroom.py")

    assert command[0] == str(interpreter.absolute())
    assert Path(command[0]).is_symlink()
    assert config.binding()["python"] == str(interpreter.absolute())


def test_probe_evidence_requires_restored_single_thread_contract(tmp_path):
    config = _config(tmp_path)
    config.out.mkdir()
    output = config.out / "gradient-probe.json"
    fake = FakeGPU(config)
    fake([str(config.python), str(config.scripts / "probe_r050_gradient_scale.py"),
          "--out", str(output)], check=False)
    receipt = json.loads(output.read_text())
    receipt["gradient_report"]["cpu_thread_contract"]["after"] = 1
    _write_json(output, receipt)

    with pytest.raises(ValueError, match="CPU thread contract"):
        pipeline.Pipeline(config)._probe_evidence()


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


@pytest.mark.parametrize(("metric", "value"), [
    ("complete_outcome_accuracy", "NaN"),
    ("complete_outcome_accuracy", "1e999"),
    ("categorical_brier_absolute_improvement", "-1e999"),
])
def test_comparison_rejects_nonfinite_tie_break_metrics(
        monkeypatch, tmp_path, metric, value):
    config = _config(tmp_path)
    runner = pipeline.Pipeline(config)
    arm = {"arm_id": "control"}
    candidate = {"arm_id": "candidate"}
    monkeypatch.setattr(runner, "_arms", lambda: [arm, candidate])
    monkeypatch.setattr(runner, "_eval_evidence", lambda _arm: {"prediction_sha256": "f" * 64})
    output = config.out / "comparisons" / "candidate-vs-control.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        '{"schema":"r049_calibrated_comparison_v5","provenance":{"inputs":{'
        f'"panel":{{"sha256":"{config.eval_panel_sha256}"}},'
        '"control":{"sha256":"' + "f" * 64 + '"},'
        '"candidate":{"sha256":"' + "f" * 64 + '"}}},'
        '"internal_stage1_gate":{"passed":true},'
        '"overall":{"paired_delta":{"equal_semantic_cluster":{'
        f'"complete_outcome_accuracy":{value if metric == "complete_outcome_accuracy" else "0.1"},'
        f'"categorical_brier_absolute_improvement":{value if metric != "complete_outcome_accuracy" else "0.1"}'
        '}}}}\n')
    with pytest.raises(ValueError, match="nonfinite|tie-break"):
        runner._comparison_evidence("candidate", "control")


def test_frozen_code_trust_covers_current_transitive_child_closure():
    expected = {
        "build_r043_causal_pairs.py", "build_r044_mm_replay.py",
        "build_r047_model_inventory.py", "build_r048_action_panel.py",
        "build_r050_headroom.py", "build_r050_mix.py", "build_r050_replay_pools.py",
        "compare_r049_calibrated.py", "exact_policy_executor.py",
        "preflight_r043_assets.py", "prepare_r050_arms.py",
        "probe_r050_gradient_scale.py", "run_r050_arm.py", "run_r050_eval.py",
        "run_r050_headroom.py", "run_r050_retention.py", "score_exact_policy_panel.py",
        "singleton_option_scorer.py", "soft_action_executor.py",
        "train_gemma_decision_lora.py",
    }
    assert set(pipeline.FROZEN_CODE_SHA256) == expected
    scripts = Path(pipeline.__file__).parent
    assert all(file_hash(scripts / name) == digest
               for name, digest in pipeline.FROZEN_CODE_SHA256.items())


def test_child_receipt_cannot_reanchor_modified_code(tmp_path):
    config = _config(tmp_path)
    runner = pipeline.Pipeline(config)
    names = ("run_r050_headroom.py", "score_exact_policy_panel.py")
    valid = {name: config.code_hashes[name] for name in names}
    runner._validate_code_receipt({"code_sha256": valid}, names)
    forged = {**valid, "score_exact_policy_panel.py": "0" * 64}
    with pytest.raises(ValueError, match="pipeline binding"):
        runner._validate_code_receipt({"code_sha256": forged}, names)


def _install_hash_drift(monkeypatch, filename, active):
    real_hash = pipeline.file_hash

    def observed(path):
        if active["value"] and Path(path).name == filename:
            return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(pipeline, "file_hash", observed)


def test_frozen_dependency_mutation_during_child_aborts_before_next_stage(
        monkeypatch, tmp_path):
    config = _config(tmp_path)
    fake = FakeGPU(config, headroom=True)
    active = {"value": False}
    _install_hash_drift(monkeypatch, "score_exact_policy_panel.py", active)

    def mutating_child(*args, **kwargs):
        result = fake(*args, **kwargs)
        active["value"] = True
        return result

    with pytest.raises(ValueError, match="changed during pipeline execution"):
        _run(monkeypatch, config, mutating_child)
    assert [Path(call[0][1]).name for call in fake.calls] == ["run_r050_headroom.py"]
    assert (config.out / "headroom" / "complete.json").is_file()
    assert not (config.out / "gradient-probe.json").exists()
    state = json.loads((config.out / "state.json").read_text())
    assert state["stages"]["01-headroom"]["status"] == "failed"


def test_running_stage_adoption_rejects_code_drift(monkeypatch, tmp_path):
    config = _config(tmp_path)
    _state(config, {"01-headroom": {"status": "running"}})
    fake = FakeGPU(config, headroom=False)
    fake([str(config.python), str(config.scripts / "run_r050_headroom.py"),
          "--out", str(config.out / "headroom")], check=False)
    fake.calls.clear()
    active = {"value": True}
    _install_hash_drift(monkeypatch, "soft_action_executor.py", active)
    with pytest.raises(RuntimeError, match="partial or invalid") as caught:
        _run(monkeypatch, config, fake)
    assert isinstance(caught.value.__cause__, ValueError)
    assert "changed during pipeline execution" in str(caught.value.__cause__)
    assert not fake.calls


def test_orchestrator_mutation_during_invocation_is_also_rejected(
        monkeypatch, tmp_path):
    config = _config(tmp_path)
    fake = FakeGPU(config, headroom=False)
    active = {"value": False}
    _install_hash_drift(monkeypatch, "run_r050_pipeline.py", active)

    def mutating_child(*args, **kwargs):
        result = fake(*args, **kwargs)
        active["value"] = True
        return result

    with pytest.raises(ValueError, match="changed during pipeline execution"):
        _run(monkeypatch, config, mutating_child)
    assert len(fake.calls) == 1
