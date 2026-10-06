import hashlib
import json
from pathlib import Path

import pytest

import scripts.finalize_r047_m0 as finalizer
from scripts.eval_r047_executor import m0_gate
from scripts.finalize_r047_m0 import EXPECTED_CONFIG, EXPECTED_STEPS, REQUIRED_KINDS


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value) + "\n")
    return path


def _update_json(path, **changes):
    value = json.loads(Path(path).read_text())
    value.update(changes)
    return _write(path, value)


def _fixture(tmp_path, passing_step=50):
    data = _write(tmp_path / "m0.jsonl", "{}\n")
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = _write(tmp_path / "manifest.json", {
        "input": {"sha256": "source"},
        "outputs": {"m0-treatment": {"sha256": data_hash}},
        "m0_treatment": {"rows": 48, "canonical_views": 96, "auxiliary_views": 192,
                         "source_subset_sha256": "subset"}})
    model = tmp_path / "model"
    parent = _write(tmp_path / "parent" / "adapter_model.safetensors", "parent")
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    config = _write(tmp_path / "config.json", {
        **EXPECTED_CONFIG, "model": str(model), "init_adapter": str(parent.parent),
        "train": [str(data)], "val": [f"m0={data}"],
    })
    preflight = _write(tmp_path / "preflight.json", {
        "passed": True, "input_sha256": "source", "m0_treatment_sha256": data_hash,
        "token_budget": {"within_limit": True, "relative_delta": 0.001},
    })
    metrics = [{"step": 0}]
    metrics.extend({"step": step, "canonical_views_seen": 16, "auxiliary_views_seen": 32,
                    "pair_flip_seen": 8, "pair_flip_weight": 1.0,
                    "canonical_ce": 0.1, "auxiliary_ce": 0.1}
                   for step in range(1, 201))
    metrics_path = _write(tmp_path / "metrics.jsonl",
                          "".join(json.dumps(item) + "\n" for item in metrics))
    reports, checkpoints = [], []
    for step in EXPECTED_STEPS:
        directory = tmp_path / f"eval-{step}"
        predictions = _write(directory / "predictions.jsonl", "{}\n")
        checkpoint = tmp_path / f"step-{step:06d}"
        adapter = _write(checkpoint / "adapter_model.safetensors", f"adapter-{step}")
        rate = 1.0 if step == passing_step else 0.5
        report = {"input_sha256": data_hash, "rows": 48,
                  "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                  "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                  "model_revision": finalizer.FROZEN_BASE_REVISION,
                  "final": {"presentations": 96, "balanced_accuracy": rate},
                  "clauses": {"n": 192, "balanced_accuracy": rate,
                              "kind": {kind: {"balanced_accuracy": rate}
                                       for kind in REQUIRED_KINDS}},
                  "final_pairs": {"direction_rate": rate, "pair_both_correct": rate,
                                  "decisive_kind": {
                                      kind: {"direction_rate": rate}
                                      for kind in REQUIRED_KINDS}}}
        report["m0_gate"] = m0_gate(report, REQUIRED_KINDS, 0.98)
        reports.append(f"{step}={_write(directory / 'report.json', report)}")
        checkpoints.append(f"{step}={checkpoint}")
    files = {name: _write(tmp_path / name, name) for name in
             ("builder.py", "trainer.py", "evaluator.py")}
    values = dict(data_path=data, manifest_path=manifest, preflight_path=preflight,
                config_path=config,
                metrics_path=metrics_path, report_specs=reports, checkpoint_specs=checkpoints,
                parent_adapter=parent, builder_path=files["builder.py"],
                trainer_path=files["trainer.py"], evaluator_path=files["evaluator.py"],
                base_revision=finalizer.FROZEN_BASE_REVISION, model_path=model)
    return values, parent_hash


def test_finalize_selects_earliest_passing_frozen_checkpoint(tmp_path, monkeypatch):
    values, parent_hash = _fixture(tmp_path)
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_SHA256", parent_hash)
    receipt = finalizer.finalize(**values)
    assert receipt["selected_step"] == 50
    assert receipt["gate_pass"]
    assert receipt["purpose"] == "in_sample_fit_only"


def test_finalize_records_a_clean_gate_failure(tmp_path, monkeypatch):
    (values, parent_hash) = _fixture(tmp_path, passing_step=None)
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_SHA256", parent_hash)
    receipt = finalizer.finalize(**values)
    assert receipt["selected_step"] is None
    assert not receipt["gate_pass"]
    assert receipt["failures"]


@pytest.mark.parametrize("mutation,match", [
    (lambda values: _update_json(values["config_path"], seed=1), "config seed"),
    (lambda values: _write(values["metrics_path"], "{}\n"), "steps 0..200"),
])
def test_finalize_fails_closed_on_recipe_or_log_drift(tmp_path, monkeypatch, mutation, match):
    values, parent_hash = _fixture(tmp_path)
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_SHA256", parent_hash)
    mutation(values)
    with pytest.raises(ValueError, match=match):
        finalizer.finalize(**values)


def test_finalize_recomputes_gate_and_binds_predictions_to_adapter(tmp_path, monkeypatch):
    values, parent_hash = _fixture(tmp_path)
    monkeypatch.setattr(finalizer, "FROZEN_PARENT_SHA256", parent_hash)
    first_report = Path(values["report_specs"][0].split("=", 1)[1])
    report = json.loads(first_report.read_text())
    report["m0_gate"]["passed"] = True
    _write(first_report, report)
    with pytest.raises(ValueError, match="recomputed gate"):
        finalizer.finalize(**values)
