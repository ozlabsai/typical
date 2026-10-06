import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_r047_m0 import EXPECTED_CONFIG, EXPECTED_STEPS, finalize


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value) + "\n")
    return path


def _fixture(tmp_path, passing_step=50):
    data = _write(tmp_path / "m0.jsonl", "{}\n")
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = _write(tmp_path / "manifest.json", {
        "input": {"sha256": "source"},
        "outputs": {"m0-treatment": {"sha256": data_hash}},
        "m0_treatment": {"rows": 48, "canonical_views": 96, "auxiliary_views": 192,
                         "source_subset_sha256": "subset"}})
    config = _write(tmp_path / "config.json", EXPECTED_CONFIG)
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
        report = {"input_sha256": data_hash, "rows": 48,
                  "final": {"presentations": 96}, "clauses": {"n": 192},
                  "m0_gate": {"passed": step == passing_step}}
        reports.append(f"{step}={_write(directory / 'report.json', report)}")
        _write(directory / "predictions.jsonl", "{}\n")
        checkpoint = tmp_path / f"step-{step:06d}"
        _write(checkpoint / "adapter_model.safetensors", f"adapter-{step}")
        checkpoints.append(f"{step}={checkpoint}")
    files = {name: _write(tmp_path / name, name) for name in
             ("parent", "builder.py", "trainer.py", "evaluator.py")}
    return dict(data_path=data, manifest_path=manifest, config_path=config,
                metrics_path=metrics_path, report_specs=reports, checkpoint_specs=checkpoints,
                parent_adapter=files["parent"], builder_path=files["builder.py"],
                trainer_path=files["trainer.py"], evaluator_path=files["evaluator.py"],
                base_revision="revision")


def test_finalize_selects_earliest_passing_frozen_checkpoint(tmp_path):
    receipt = finalize(**_fixture(tmp_path))
    assert receipt["selected_step"] == 50
    assert receipt["gate_pass"]
    assert receipt["purpose"] == "in_sample_fit_only"


def test_finalize_records_a_clean_gate_failure(tmp_path):
    values = _fixture(tmp_path, passing_step=None)
    receipt = finalize(**values)
    assert receipt["selected_step"] is None
    assert not receipt["gate_pass"]
    assert receipt["failures"]


@pytest.mark.parametrize("mutation,match", [
    (lambda values: _write(values["config_path"], {**EXPECTED_CONFIG, "seed": 1}), "config seed"),
    (lambda values: _write(values["metrics_path"], "{}\n"), "steps 0..200"),
])
def test_finalize_fails_closed_on_recipe_or_log_drift(tmp_path, mutation, match):
    values = _fixture(tmp_path)
    mutation(values)
    with pytest.raises(ValueError, match=match):
        finalize(**values)
