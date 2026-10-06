import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.compare_r049_calibrated import (
    compare_files,
    compare_rows,
    write_report_atomic,
)
from scripts.exact_policy_executor import execute_policy


def _action(name):
    return {"name": name, "payload": {"queue": name}}


def _row(row_id, cluster, *, family="standard", language="en", order="no_yes"):
    policy = {
        "fields": ["risk", "evidence"],
        "mode": "priority",
        "rules": [
            {"when": {"field": "risk"}, "action": _action("review")},
            {"when": {"field": "evidence"}, "action": _action("retain")},
        ],
        "fallback": _action("approve"),
    }
    trace = {"risk": True, "evidence": False}
    labels = ("no", "yes") if order == "no_yes" else ("yes", "no")
    questions = {
        f"q{index:02d}": {
            "type": "noul",
            "instructions": f"is {field} true?",
            "criteria": {label: label.upper() for label in labels},
        }
        for index, field in enumerate(policy["fields"])
    }
    targets = {
        "q00": {"label": "yes"},
        "q01": {"label": "no"},
    }
    return {
        "id": row_id,
        "state": "record",
        "questions": questions,
        "targets": targets,
        "meta": {
            "panel": "exact_policy_v1",
            "split": "eval",
            "semantic_group_id": cluster,
            "counterfactual_side": 0,
            "language": language,
            "option_order": order,
            "policy_mode": "priority",
            "topology_slice": "heldout_topology",
            "policy_topology": "priority_grouped_partition",
            "program_family": family,
            "action_family": family,
            "field_order": policy["fields"],
            "qid_to_field": {"q00": "risk", "q01": "evidence"},
            "decision_threshold": 0.5,
            "action_trace": trace,
            "policy": policy,
            "gold_outcome": execute_policy(policy, trace).as_dict(),
        },
    }


def _prediction(row, correct, confidence=1.0):
    risk = confidence if correct else 1.0 - confidence
    evidence = 1.0 - confidence if correct else confidence
    return {
        "id": row["id"],
        "probabilities": {
            "q00": {"no": 1.0 - risk, "yes": risk},
            "q01": {"no": 1.0 - evidence, "yes": evidence},
        },
    }


def _compare(rows, control, candidate, *, replicates=400):
    return compare_rows(
        rows,
        control,
        candidate,
        bootstrap_seed=490049,
        bootstrap_replicates=replicates,
    )


def test_perfect_gain_passes_internal_stage1_gate():
    rows = [_row(f"r{index}", f"c{index}") for index in range(6)]
    control = [_prediction(row, False) for row in rows]
    candidate = [_prediction(row, True) for row in rows]

    report = _compare(rows, control, candidate)

    delta = report["overall"]["paired_delta"]["equal_semantic_cluster"]
    assert delta["complete_outcome_accuracy"] == 1.0
    assert delta["categorical_brier_absolute_improvement"] == 2.0
    assert delta["categorical_brier_relative_improvement"] == 1.0
    assert delta["complete_outcome_nll_degradation"] < 0
    assert delta["field_accuracy"] == 1.0
    assert report["internal_stage1_gate"]["passed"] is True
    assert report["promotion"]["status"] == "not_evaluated"
    assert report["external_gates"]["retention"]["status"] == "not_evaluated"
    assert report["external_gates"]["workflowevals"]["status"] == "not_evaluated"


def test_missing_and_invalid_predictions_are_scored_fail_closed():
    rows = [_row("a", "c0"), _row("b", "c1"), _row("c", "c2")]
    control = [_prediction(row, False, 0.9) for row in rows]
    invalid = _prediction(rows[1], True)
    invalid["probabilities"]["q00"] = {"no": 0.4, "yes": 0.4}
    candidate = [_prediction(rows[0], True), invalid]

    report = _compare(rows, control, candidate)

    coverage = report["coverage"]["candidate"]
    assert coverage["exact_id_coverage"] is False
    assert coverage["missing"] == 1
    assert coverage["invalid"] == 2
    assert report["overall"]["candidate"]["rendered"]["complete_outcome_accuracy"] == pytest.approx(1 / 3)
    requirement = report["internal_stage1_gate"]["requirements"]["zero_invalid_predictions"]
    assert requirement["passed"] is False
    assert report["internal_stage1_gate"]["passed"] is False


def test_cluster_estimate_and_bootstrap_do_not_treat_views_as_independent():
    rows = [_row(f"many-{index}", "many") for index in range(9)]
    rows.append(_row("one", "one"))
    control = [_prediction(row, row["id"] == "one") for row in rows]
    candidate = [_prediction(row, row["id"] != "one") for row in rows]

    report = _compare(rows, control, candidate, replicates=600)

    paired = report["overall"]["paired_delta"]
    assert paired["rendered"]["complete_outcome_accuracy"] == pytest.approx(0.8)
    assert paired["equal_semantic_cluster"]["complete_outcome_accuracy"] == 0.0
    interval = report["overall"]["paired_cluster_bootstrap"]["complete_outcome_accuracy"]
    assert interval["lower_95"] == -1.0
    assert interval["upper_95"] == 1.0
    assert interval["resampling_unit"] == "semantic_group_id"


def test_worse_soft_predictions_fail_brier_and_nll_gates():
    rows = [_row(f"r{index}", f"c{index}") for index in range(6)]
    control = [_prediction(row, True, 0.9) for row in rows]
    candidate = [_prediction(row, True, 0.51) for row in rows]

    report = _compare(rows, control, candidate)
    requirements = report["internal_stage1_gate"]["requirements"]

    assert requirements["categorical_brier_relative_improvement"]["passed"] is False
    assert requirements["categorical_brier_ci_excludes_zero"]["passed"] is False
    assert requirements["nll_degradation"]["passed"] is False


def test_field_accuracy_regression_fails_both_field_gates():
    rows = [_row(f"r{index}", f"c{index}") for index in range(6)]
    control = [_prediction(row, True) for row in rows]
    candidate = [_prediction(row, False) for row in rows]

    requirements = _compare(rows, control, candidate)["internal_stage1_gate"]["requirements"]

    assert requirements["field_accuracy_point"]["passed"] is False
    assert requirements["field_accuracy_lcb"]["passed"] is False


def test_program_or_action_family_loss_blocks_internal_gate():
    rows = [
        _row("bad", "bad", family="rare"),
        *[_row(f"good-{index}", f"good-{index}", family="common") for index in range(5)],
    ]
    control = [_prediction(row, row["meta"]["program_family"] == "rare") for row in rows]
    candidate = [_prediction(row, row["meta"]["program_family"] != "rare") for row in rows]

    report = _compare(rows, control, candidate)
    requirement = report["internal_stage1_gate"]["requirements"]["family_nonregression"]

    assert report["family_nonregression"]["program_family"]["worst_delta"] == -1.0
    assert report["family_nonregression"]["action_family"]["worst_delta"] == -1.0
    assert requirement["passed"] is False
    assert report["internal_stage1_gate"]["passed"] is False


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_file_comparison_binds_manifest_hashes_and_is_deterministic(tmp_path):
    rows = [_row(f"r{index}", f"c{index}") for index in range(4)]
    panel = tmp_path / "eval.jsonl"
    control_path = tmp_path / "control.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(panel, rows)
    _write_jsonl(control_path, [_prediction(row, False) for row in rows])
    _write_jsonl(candidate_path, [_prediction(row, True) for row in rows])
    panel_sha = hashlib.sha256(panel.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "outputs": {"eval": {"rows": len(rows), "sha256": panel_sha}},
    }), encoding="utf-8")

    first = compare_files(
        panel,
        control_path,
        candidate_path,
        manifest_path=manifest,
        expected_hashes={"panel": panel_sha},
        bootstrap_seed=17,
        bootstrap_replicates=200,
    )
    second = compare_files(
        panel,
        control_path,
        candidate_path,
        manifest_path=manifest,
        expected_hashes={"panel": panel_sha},
        bootstrap_seed=17,
        bootstrap_replicates=200,
    )

    assert first == second
    assert first["provenance"]["manifest"]["verified"] is True
    assert "timestamp" not in json.dumps(first)


def test_atomic_output_refuses_overwrite_and_has_deterministic_bytes(tmp_path):
    rows = [_row(f"r{index}", f"c{index}") for index in range(3)]
    report = _compare(
        rows,
        [_prediction(row, False) for row in rows],
        [_prediction(row, True) for row in rows],
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_report_atomic(report, first)
    write_report_atomic(report, second)

    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report_atomic(report, first)


def test_duplicate_or_extra_prediction_ids_fail_validation():
    row = _row("r0", "c0")
    prediction = _prediction(row, True)
    with pytest.raises(ValueError, match="duplicate control prediction id"):
        _compare([row], [prediction, prediction], [prediction])
    extra = dict(prediction, id="not-in-panel")
    with pytest.raises(ValueError, match="unexpected candidate prediction ids"):
        _compare([row], [prediction], [prediction, extra])


def test_canonical_v13_cluster_and_topology_keys_are_required():
    row = _row("r0", "c0")
    prediction = _prediction(row, True)
    row["meta"]["external_policy_topology"] = row["meta"].pop("policy_topology")
    row["meta"]["external_policy_topology_slice"] = row["meta"].pop("topology_slice")

    with pytest.raises(ValueError, match="policy_topology is required"):
        _compare([row], [prediction], [prediction])


def test_absent_optional_families_are_not_invented_or_blocking():
    rows = [_row(f"r{index}", f"c{index}") for index in range(4)]
    for row in rows:
        row["meta"].pop("program_family")
        row["meta"].pop("action_family")
    report = _compare(
        rows,
        [_prediction(row, False) for row in rows],
        [_prediction(row, True) for row in rows],
    )

    assert report["family_nonregression"]["program_family"]["status"] == "not_evaluated"
    assert report["family_nonregression"]["action_family"]["status"] == "not_evaluated"
    assert report["internal_stage1_gate"]["requirements"]["family_nonregression"][
        "passed"
    ] is True


def test_hash_mismatch_fails_before_comparison(tmp_path):
    row = _row("r0", "c0")
    panel = tmp_path / "eval.jsonl"
    control = tmp_path / "control.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(panel, [row])
    _write_jsonl(control, [_prediction(row, False)])
    _write_jsonl(candidate, [_prediction(row, True)])

    with pytest.raises(ValueError, match="panel sha256 mismatch"):
        compare_files(
            panel,
            control,
            candidate,
            expected_hashes={"panel": "0" * 64},
            bootstrap_seed=1,
            bootstrap_replicates=100,
        )


def test_direct_script_help_bootstraps_repository_imports(tmp_path):
    script = Path("scripts/compare_r049_calibrated.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Fail-closed paired comparison" in result.stdout
