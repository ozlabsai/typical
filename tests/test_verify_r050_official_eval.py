from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.gemma_decision_runtime import (
    artifact_identity,
    runtime_environment_identity,
)
from scripts.serve_typesafe_gemma_lora import build_startup_receipt
from scripts.verify_r050_official_eval import (
    BASE_REVISION,
    FROZEN_RUNTIME,
    WORKFLOW_PATCH_BYTES,
    WORKFLOW_PATCH_SHA256,
    VerificationError,
    WorkflowSpec,
    _dataset_hash,
    _validate_workflow_schemas,
    _workflow_patch_identity,
    fingerprint,
    verify_jev_run,
    verify_preserved_evidence,
    verify_server_receipt,
    verify_workflow_run,
    workflow_parity,
)

MODEL_ID_PREFIX = "typical-r050-sha256-"
BASE_URL = "http://127.0.0.1:8090"


def write_json(path: Path, value, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n")


def test_frozen_workflow_patch_identity_matches_disclosed_bytes():
    patch = Path("patches/workflowevals-typical-provider-0ac3b8ad.patch")
    assert patch.stat().st_size == WORKFLOW_PATCH_BYTES
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == WORKFLOW_PATCH_SHA256


def test_patch_identity_rejects_extra_edit_inside_disclosed_file(tmp_path, monkeypatch):
    from scripts import verify_r050_official_eval as verifier

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    source = checkout / "core" / "clients.py"
    source.parent.mkdir()
    base_text = "".join(f"value_{index} = {index}\n" for index in range(20))
    source.write_text(base_text)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=checkout, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True
    ).stdout.strip()
    patched_lines = base_text.splitlines(keepends=True)
    patched_lines[10] = "value_10 = 'patched'\n"
    source.write_text("".join(patched_lines))
    patch_bytes = subprocess.run(
        ["git", "diff", "--binary", "--full-index"],
        cwd=checkout,
        check=True,
        capture_output=True,
    ).stdout
    patch = tmp_path / "change.patch"
    patch.write_bytes(patch_bytes)
    source.write_text(base_text)
    subprocess.run(["git", "apply", str(patch)], cwd=checkout, check=True)
    source.write_text("value_0 = 'not disclosed'\n" + "".join(source.read_text().splitlines(keepends=True)[1:]))
    monkeypatch.setattr(verifier, "WORKFLOWEVALS_COMMIT", commit)
    monkeypatch.setattr(verifier, "WORKFLOW_PATCH_SHA256", hashlib.sha256(patch_bytes).hexdigest())
    monkeypatch.setattr(verifier, "WORKFLOW_PATCH_BYTES", len(patch_bytes))
    with pytest.raises(VerificationError, match="patched file differs"):
        _workflow_patch_identity(checkout, patch)


def startup_fixture(tmp_path: Path) -> tuple[Path, dict]:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text('{"model":"gemma"}\n')
    (adapter / "adapter_config.json").write_text('{"rank":32}\n')
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    base_hash = artifact_identity(str(base), None)["tree_sha256"]
    adapter_hash = artifact_identity(str(adapter), None)["tree_sha256"]
    receipt = build_startup_receipt(
        model_id=f"{MODEL_ID_PREFIX}{adapter_hash[:16]}",
        base_model=str(base),
        base_revision=BASE_REVISION,
        adapter=str(adapter),
        adapter_revision=None,
        expected_model_tree_sha256=base_hash,
        expected_adapter_tree_sha256=adapter_hash,
        **FROZEN_RUNTIME,
    )
    path = tmp_path / "startup.json"
    write_json(path, receipt)
    return path, receipt


def test_startup_receipt_rehashes_live_canonical_trees_and_runtime(tmp_path):
    path, receipt = startup_fixture(tmp_path)
    assert verify_server_receipt(path) == receipt

    receipt["runtime"]["dtype"] = "float16"
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = fingerprint(body)
    write_json(path, receipt)
    with pytest.raises(VerificationError, match="frozen R050 runtime"):
        verify_server_receipt(path)


def test_startup_receipt_detects_tree_mutation_even_with_unchanged_receipt(tmp_path):
    path, receipt = startup_fixture(tmp_path)
    (Path(receipt["adapter"]["source"]) / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(VerificationError, match="live tree differs"):
        verify_server_receipt(path)


def test_startup_receipt_rejects_runtime_dependency_substitution(tmp_path):
    path, receipt = startup_fixture(tmp_path)
    receipt["environment"]["dependencies"]["torch"]["installed"] = "0.0.0"
    receipt["receipt_sha256"] = fingerprint(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    write_json(path, receipt)
    with pytest.raises(VerificationError, match="runtime environment"):
        verify_server_receipt(path)


def workflow_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, WorkflowSpec, str]:
    from scripts import verify_r050_official_eval as verifier

    model_id = f"{MODEL_ID_PREFIX}{'a' * 16}"
    directory = tmp_path / "r050-tiny"
    case_ids = ["case-1", "מקרה-2"]
    declared = [{"case_id": case_id, "content_hash": f"hash-{case_id}"} for case_id in case_ids]
    dataset_revision = fingerprint({case["case_id"]: case["content_hash"] for case in declared})
    dataset_source = "https://huggingface.co/datasets/example/tree/" + "d" * 40
    spec = WorkflowSpec(
        2,
        ("p1", "p2"),
        "exact_actions",
        "d" * 40,
        dataset_revision,
        dataset_source,
    )
    monkeypatch.setitem(verifier.WORKFLOWS, "tiny_workflow", spec)
    prices = {"_comment": "unknown Typical cost"}
    dependencies = {
        "typesafe-sdk": "0.7.2",
        "system-one-adapter": "0.2.1",
        "pydantic-ai-slim": "2.41.0",
        "pydantic": "2.13.5",
    }
    model_config = {
        "provider": "typical",
        "model": model_id,
        "thinking": "off",
        "question_mode": "all",
        "base_url": BASE_URL,
        "timeout_s": 1800.0,
        "max_output_tokens": 64000,
        "malformed_retries": 1,
    }
    configuration = {
        "model": model_config,
        "policy_ids": list(spec.policy_ids),
        "prices": prices,
        "dataset_revision": dataset_revision,
        "dataset_source": dataset_source,
        "policies": [{"policy_id": policy} for policy in spec.policy_ids],
        "infra_revision": "infra-revision",
        "workflow_revision": "workflow-revision",
        "dependency_lock": "lock-revision",
        "dependencies": dependencies,
        "structured_outputs": True,
        "answer_mode": "probabilities",
        "normalize_probabilities": True,
    }
    run_cases = []
    for case_id in case_ids:
        questions = {"q": {"type": "noul"}}
        call = {
            "call_id": "1",
            "status": "ok",
            "error": None,
            "retries": 0,
            "questions": questions,
            "response": {"model": model_id, "answers": {"q": {"type": "noul", "noul": 0.5}}},
        }
        case = {
            "case_id": case_id,
            "case_hash": f"hash-{case_id}",
            "input_status": "verified",
            "decisions": [{"policy_id": policy, "status": "ok"} for policy in spec.policy_ids],
            "execution": {
                "policy_ids": list(spec.policy_ids),
                "status": "ok",
                "reason": None,
                "calls": 1,
            },
            "calls": [call],
        }
        run_cases.append(case)
        write_json(directory / "cases" / f"{fingerprint(case_id)}.json", case)
    run = {
        "run_id": "r050-tiny",
        "model": {"name": f"typical:{model_id}", "lab": "other", "effort": None},
        "configuration": configuration,
        "configuration_hash": fingerprint(configuration),
        "cases": run_cases,
    }
    results = {
        "eval_id": "tiny_workflow",
        "dataset_revision": dataset_revision,
        "policies": [{"policy_id": policy} for policy in spec.policy_ids],
        "metrics": [{"metric_id": spec.metric, "comparison": spec.metric}],
        "cases": declared,
        "runs": [run],
        "references": {"consensus": "labels-consensus"},
    }
    manifest = {
        "run_id": run["run_id"],
        "configuration_hash": run["configuration_hash"],
        "configuration": configuration,
    }
    score_rows = [
        {
            "run_id": run["run_id"],
            "label_set_id": "labels-consensus",
            "case_id": case_id,
            "policy_id": policy,
            "status": "scored",
            "unverified": False,
            "metrics": {spec.metric: True},
        }
        for case_id in case_ids
        for policy in spec.policy_ids
    ]
    scores = {
        "eval_id": "tiny_workflow",
        "dataset_revision": dataset_revision,
        "allow_unverified": False,
        "failure_handling": "count_prediction_errors_as_wrong",
        "series": [{
            "run_id": run["run_id"],
            "label_set_id": "labels-consensus",
            "cases": spec.cases,
            "decisions": spec.decisions,
            "status_counts": {"scored": spec.decisions},
            "unverified_decisions": 0,
            "metrics": {spec.metric: 1.0},
            "metric_denominators": {spec.metric: spec.decisions},
        }],
        "scores": score_rows,
    }
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "results.json", results)
    write_json(directory / "scores.json", scores)
    return directory, spec, model_id


def workflow_contract() -> dict:
    return {
        "infra_revision": "infra-revision",
        "workflow_revisions": {"tiny_workflow": "workflow-revision"},
        "dependency_lock": "lock-revision",
        "dependencies": {
            "typesafe-sdk": "0.7.2",
            "system-one-adapter": "0.2.1",
            "pydantic-ai-slim": "2.41.0",
            "pydantic": "2.13.5",
        },
        "prices": {"_comment": "unknown Typical cost"},
    }


def test_complete_workflow_evidence_passes(tmp_path, monkeypatch):
    directory, spec, model_id = workflow_fixture(tmp_path, monkeypatch)
    result = verify_workflow_run(
        directory,
        spec=spec,
        model_id=model_id,
        base_url=BASE_URL,
        timeout_s=1800,
        configuration_contract=workflow_contract(),
    )
    assert {key: value for key, value in result.items() if key != "evidence"} == {
        "workflow": "tiny_workflow",
        "run_id": "r050-tiny",
        "cases": 2,
        "decisions": 4,
        "metric": 1.0,
    }
    assert result["evidence"]["cases"]["files"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_id", "missing or extra run case IDs"),
        ("duplicate_id", "duplicate tiny_workflow run case IDs"),
        ("wrong_count", "wrong expected case count"),
        ("failed_execution", "failed execution"),
        ("retry", "nonzero retries"),
        ("model_substitution", "model substitution"),
        ("attempt_archive", "retry attempt artifacts"),
        ("nonfinite_metric", "non-finite JSON number"),
        ("infra_drift", "infrastructure revision drift"),
        ("workflow_drift", "workflow revision drift"),
        ("lock_drift", "dependency lock drift"),
        ("dependency_drift", "dependency versions drift"),
        ("pricing_drift", "pricing configuration drift"),
        ("aggregate_tamper", "aggregate results differ from durable checkpoints"),
        ("series_tamper", "aggregate metric differs from consensus rows"),
    ],
)
def test_workflow_verifier_fails_closed(tmp_path, monkeypatch, mutation, message):
    directory, spec, model_id = workflow_fixture(tmp_path, monkeypatch)
    results = json.loads((directory / "results.json").read_text())
    scores = json.loads((directory / "scores.json").read_text())
    if mutation == "missing_id":
        results["runs"][0]["cases"].pop()
    elif mutation == "duplicate_id":
        results["runs"][0]["cases"][1]["case_id"] = "case-1"
    elif mutation == "wrong_count":
        results["cases"].pop()
    elif mutation == "failed_execution":
        results["runs"][0]["cases"][0]["execution"]["status"] = "error"
    elif mutation == "retry":
        results["runs"][0]["cases"][0]["calls"][0]["retries"] = 1
    elif mutation == "model_substitution":
        results["runs"][0]["cases"][0]["calls"][0]["response"]["model"] = "jev-1.13.0"
    elif mutation == "attempt_archive":
        write_json(directory / "attempts" / "old.json", {})
    elif mutation == "nonfinite_metric":
        scores["series"][0]["metrics"][spec.metric] = float("nan")
    elif mutation == "infra_drift":
        results["runs"][0]["configuration"]["infra_revision"] = "wrong"
    elif mutation == "workflow_drift":
        results["runs"][0]["configuration"]["workflow_revision"] = "wrong"
    elif mutation == "lock_drift":
        results["runs"][0]["configuration"]["dependency_lock"] = "wrong"
    elif mutation == "dependency_drift":
        results["runs"][0]["configuration"]["dependencies"]["pydantic"] = "0"
    elif mutation == "pricing_drift":
        results["runs"][0]["configuration"]["prices"]["typical:fake"] = [0, 0]
    elif mutation == "aggregate_tamper":
        results["runs"][0]["cases"][0]["details"] = {"tampered": True}
    elif mutation == "series_tamper":
        scores["series"][0]["metrics"][spec.metric] = 0.5
    if mutation in {"failed_execution", "retry", "model_substitution"}:
        case = results["runs"][0]["cases"][0]
        write_json(directory / "cases" / f"{fingerprint(case['case_id'])}.json", case)
    write_json(directory / "results.json", results)
    write_json(directory / "scores.json", scores, allow_nan=True)
    with pytest.raises(VerificationError, match=message):
        verify_workflow_run(
            directory,
            spec=spec,
            model_id=model_id,
            base_url=BASE_URL,
            timeout_s=1800,
            configuration_contract=workflow_contract(),
        )


def test_pinned_workflow_schema_rejects_incomplete_handrolled_bundle(tmp_path, monkeypatch):
    checkout = Path(".context/research_sources/WorkflowEvals").resolve()
    if not checkout.is_dir():
        pytest.skip("pinned WorkflowEvals checkout is unavailable")
    directory, _, _ = workflow_fixture(tmp_path, monkeypatch)
    with pytest.raises(VerificationError, match="schema/scorer validation failed"):
        _validate_workflow_schemas(checkout, [directory])


def jev_fixture(tmp_path: Path) -> tuple[Path, dict, dict, list[dict], dict]:
    directory = tmp_path / "jev"
    expected = {
        "harness_commit": "harness-commit",
        "harness_files": {
            "jevbench/__init__.py": "e" * 64,
            "jevbench/budget.py": "f" * 64,
            "jevbench/tasks.py": "a" * 64,
            "jevbench/runner.py": "b" * 64,
            "jevbench/scoring.py": "c" * 64,
            "jevbench/adapters/base.py": "d" * 64,
        },
        "code": {
            "jevbench_gemma_lora.py": "1" * 64,
            "gemma_decision_runtime.py": "2" * 64,
            "singleton_option_scorer.py": "3" * 64,
            "preflight_r043_assets.py": "4" * 64,
            "train_gemma_decision_lora.py": "0" * 64,
        },
        "environment": runtime_environment_identity(),
        "task_inputs": [{
            "name": "easy.jsonl",
            "sha256": "5" * 64,
            "count": 2,
            "dataset_hash": "6" * 64,
        }],
        "selected_dataset_hash": "7" * 64,
        "task_ids": ["task-1", "task-2"],
    }
    server = {
        "base_model": {
            "source": "/base",
            "revision": BASE_REVISION,
            "artifact": {"kind": "tree", "tree_sha256": "8" * 64},
        },
        "adapter": {
            "source": "/adapter",
            "revision": None,
            "artifact": {"kind": "tree", "tree_sha256": "9" * 64},
        },
        "environment": runtime_environment_identity(),
    }
    manifest = {
        "adapter": "typical_gemma_lora",
        "harness_commit": expected["harness_commit"],
        "harness_code": expected["harness_files"],
        "code": expected["code"],
        "model_revision": BASE_REVISION,
        "model": "/base",
        "model_artifact": server["base_model"]["artifact"],
        "lora_adapter": "/adapter",
        "lora_revision": None,
        "lora_artifact": server["adapter"]["artifact"],
        "environment": expected["environment"],
        "model_context": 32768,
        **FROZEN_RUNTIME,
        "seed": 42,
        "task_inputs": [{
            "path": "datasets/public/easy.jsonl",
            "sha256": "5" * 64,
            "count": 2,
            "dataset_hash": "6" * 64,
        }],
        "selected_dataset_hash": expected["selected_dataset_hash"],
        "counts": {"planned": 2, "attempted": 2, "unattempted": 0, "valid": 2, "strict_valid": 2},
        "statuses": {"ok": 2},
        "status_codes": {"200": 2},
        "started_utc": "2026-10-07T10:00:00+00:00",
        "finished_utc": "2026-10-07T10:01:00+00:00",
        "load_s": 1.5,
        "torch_version": "2.14.0",
        "gpu": "Test GPU",
        "score_scope": "per-item harness evidence only; no sealed or headline score computed",
    }
    write_json(directory / "manifest.json", manifest)
    rows = []
    tasks = []
    frozen_scores = {}
    ledger = [{"event": "cap", "cap_usd": 15, "ts": 1.0}]
    for index, task_id in enumerate(expected["task_ids"], 1):
        task = {
            "id": task_id,
            "family": "routing",
            "state": {"ticket": task_id},
            "question": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"a": "first", "b": "second"},
            },
            "labels": ["a", "b"],
            "expected": "b",
            "split": "public",
            "group": None,
            "provenance": {},
        }
        tasks.append(task)
        runtime = {
            "input_tokens": 17,
            "max_context": 16384,
            "truncated": False,
            "probability_origin": "native-option-logits",
        }
        raw_record = {
            "request": {
                "state": task["state"],
                "questions": {"decision": task["question"]},
                "labels": task["labels"],
            },
            "response": {
                "model": "/base+/adapter",
                "probabilities": {"a": 0.25, "b": 0.75},
                "usage": {"input_tokens": 17, "output_tokens": 0},
                "runtime": runtime,
            },
            "http_status": 200,
        }
        raw = json.dumps(raw_record, ensure_ascii=False, separators=(",", ":")).encode()
        raw_path = directory / "raw" / f"{hashlib.sha256(task_id.encode()).hexdigest()}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        rows.append({
            "task_id": task_id,
            "family": task["family"],
            "split": task["split"],
            "group": task["group"],
            "ts": float(index),
            "status": "ok",
            "ok": True,
            "valid": True,
            "correct": True,
            "predicted": "b",
            "ordinal_ev": None,
            "probs": {"a": 0.25, "b": 0.75},
            "probs_as_returned": {"a": 0.25, "b": 0.75},
            "strict_valid": True,
            "renormalized": False,
            "status_code": 200,
            "probs_source": "native",
            "model": "/base+/adapter",
            "error": None,
            "schema_error": None,
            "latency_s": 0.1,
            "usage": {"input_tokens": 17, "output_tokens": 0},
            "cost_usd": None,
            "cost_basis": "local_gpu_no_provider_tariff",
            "reserved_usd": 0.0,
            "charged_usd": 0.0,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "runtime": runtime,
        })
        frozen_scores[task_id] = {
            "valid": True,
            "strict_valid": True,
            "renormalized": False,
            "probs": {"a": 0.25, "b": 0.75},
            "predicted": "b",
            "correct": True,
            "ordinal_ev": None,
            "error": None,
        }
        reservation = f"reservation-{index}"
        ledger.extend([
            {
                "event": "reserve",
                "reservation_id": reservation,
                "reserved_usd": 0.0,
                "ts": float(index),
                "meta": {"task_id": task_id, "adapter": "typical_gemma_lora"},
            },
            {
                "event": "settle",
                "reservation_id": reservation,
                "charged_usd": 0.0,
                "ts": float(index),
                "meta": {"task_id": task_id, "basis": "local_gpu_no_provider_tariff"},
            },
        ])
    (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
    return directory, expected, server, tasks, frozen_scores


def test_complete_jev_evidence_passes(tmp_path):
    directory, expected, server, tasks, frozen_scores = jev_fixture(tmp_path)
    result = verify_jev_run(directory, expected, server, tasks, frozen_scores)
    assert {key: value for key, value in result.items() if key != "evidence"} == {
        "scope": "legacy_public_231_diagnostic",
        "official_v1_5_4_parity": "unmeasured",
        "tasks": 2,
        "dataset_hash": expected["selected_dataset_hash"],
    }
    assert result["evidence"]["raw"]["files"] == 2


def test_final_receipt_detects_post_verification_evidence_mutation(tmp_path, monkeypatch):
    workflow_dir, spec, model_id = workflow_fixture(tmp_path / "workflow", monkeypatch)
    workflow = verify_workflow_run(
        workflow_dir,
        spec=spec,
        model_id=model_id,
        base_url=BASE_URL,
        timeout_s=1800,
        configuration_contract=workflow_contract(),
    )
    jev_dir, expected, server, tasks, frozen_scores = jev_fixture(tmp_path / "jev-fixture")
    jev = verify_jev_run(jev_dir, expected, server, tasks, frozen_scores)
    receipt = {
        "schema_version": "typical.r050_official_eval_verification.v1",
        "workflowevals": [workflow],
        "jevbench": jev,
    }
    receipt["receipt_sha256"] = fingerprint(receipt)
    receipt_path = tmp_path / "verification.json"
    write_json(receipt_path, receipt)
    assert verify_preserved_evidence(receipt_path, [workflow_dir], jev_dir)["evidence_preserved"]

    raw_path = next((jev_dir / "raw").glob("*.json"))
    original = raw_path.read_text()
    raw_path.write_text(original + " ")
    with pytest.raises(VerificationError, match="JevBench evidence changed"):
        verify_preserved_evidence(receipt_path, [workflow_dir], jev_dir)
    raw_path.write_text(original)

    (jev_dir / "unexpected-attempt.json").write_text("{}\n")
    with pytest.raises(VerificationError, match="JevBench evidence changed"):
        verify_preserved_evidence(receipt_path, [workflow_dir], jev_dir)


def test_workflow_parity_reports_honest_pass_and_below_target_evidence():
    passing = [
        {"workflow": "security_incidents", "metric": 0.70},
        {"workflow": "agent_trace_observability", "metric": 0.72},
        {"workflow": "invoice_processing", "metric": 0.70},
        {"workflow": "customer_service", "metric": 0.80},
    ]
    assert workflow_parity(passing)["workflow_parity_pass"] is True
    below = [dict(item) for item in passing]
    below[2]["metric"] = 0.60
    report = workflow_parity(below)
    assert report["workflow_parity_pass"] is False
    assert report["floors"]["invoice_processing"] == {
        "metric": 0.60,
        "target": 0.618,
        "pass": False,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("harness", "harness provenance"),
        ("runner", "runner/runtime provenance"),
        ("environment", "runtime environment"),
        ("model_source", "model source"),
        ("missing_manifest_field", "manifest schema"),
        ("tasks", "task inputs"),
        ("failure", "contains failures"),
        ("duplicate", "duplicate JevBench result task IDs"),
        ("missing_probs", "lacks probabilities"),
        ("wrong_correct", "correct differs from rescoring"),
        ("raw_semantics", "raw request/response mismatch"),
        ("raw_probs", "raw request/response mismatch"),
        ("row_model", "model mismatch"),
        ("runtime_tolerance", "runtime tolerance"),
        ("ledger", "ledger has wrong event count"),
        ("ledger_basis", "ledger settlement mismatch"),
        ("ledger_timestamp", "ledger timestamp order mismatch"),
    ],
)
def test_jev_verifier_fails_closed_on_provenance_and_results(tmp_path, mutation, message):
    directory, expected, server, tasks, frozen_scores = jev_fixture(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    if mutation == "harness":
        manifest["harness_commit"] = "wrong"
    elif mutation == "runner":
        manifest["code"]["jevbench_gemma_lora.py"] = "f" * 64
    elif mutation == "environment":
        manifest["environment"]["dependencies"]["torch"] = "0.0.0"
    elif mutation == "model_source":
        manifest["model"] = "/substituted-base"
    elif mutation == "missing_manifest_field":
        del manifest["gpu"]
    elif mutation == "tasks":
        manifest["task_inputs"][0]["sha256"] = "f" * 64
    elif mutation == "failure":
        manifest["statuses"] = {"failed": 1, "ok": 1}
    elif mutation == "duplicate":
        rows = (directory / "results.jsonl").read_text().splitlines()
        first = json.loads(rows[0])
        second = json.loads(rows[1])
        second["task_id"] = first["task_id"]
        (directory / "results.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    elif mutation in {
        "missing_probs", "wrong_correct", "raw_semantics", "raw_probs",
        "row_model", "runtime_tolerance",
    }:
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        if mutation == "missing_probs":
            rows[0]["probs_as_returned"] = None
        elif mutation == "wrong_correct":
            rows[0]["correct"] = False
        elif mutation == "row_model":
            rows[0]["model"] = "/substituted-base+/adapter"
        elif mutation == "runtime_tolerance":
            rows[0]["probs_as_returned"] = {"a": 0.250001, "b": 0.75}
            task_id = rows[0]["task_id"]
            raw_path = directory / "raw" / f"{hashlib.sha256(task_id.encode()).hexdigest()}.json"
            raw = json.loads(raw_path.read_text())
            raw["response"]["probabilities"] = rows[0]["probs_as_returned"]
            raw_path.write_text(json.dumps(raw, separators=(",", ":")))
            rows[0]["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        else:
            task_id = rows[0]["task_id"]
            raw_path = directory / "raw" / f"{hashlib.sha256(task_id.encode()).hexdigest()}.json"
            raw = json.loads(raw_path.read_text())
            if mutation == "raw_semantics":
                raw["request"]["state"] = {"tampered": True}
            else:
                raw["response"]["probabilities"] = {"a": 0.75, "b": 0.25}
            raw_path.write_text(json.dumps(raw, separators=(",", ":")))
            rows[0]["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif mutation == "ledger":
        rows = (directory / "ledger.jsonl").read_text().splitlines()
        (directory / "ledger.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    elif mutation == "ledger_basis":
        rows = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()]
        rows[2]["meta"]["basis"] = "forged"
        (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif mutation == "ledger_timestamp":
        rows = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()]
        rows[2]["ts"] = 99.0
        (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(VerificationError, match=message):
        verify_jev_run(directory, expected, server, tasks, frozen_scores)


def test_local_dataset_hash_matches_current_official_jevbench_implementation():
    checkout = Path(".context/jevbench-v154").resolve()
    if not checkout.is_dir():
        pytest.skip("pinned JevBench checkout is unavailable")
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from jevbench.tasks import dataset_hash, load_jsonl

    path = checkout / "datasets/public/easy.jsonl"
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert _dataset_hash(raw) == dataset_hash(load_jsonl(str(path)))


def test_jev_dataset_hash_checked_in_unicode_golden_vector():
    task = {
        "id": "golden-שלום",
        "family": "intent",
        "state": {"טקסט": "שלום"},
        "question": {
            "type": "choice",
            "instructions": "בחר",
            "criteria": {"א": "ראשון", "ב": "שני"},
        },
        "labels": ["א", "ב"],
        "expected": "ב",
        "split": "public",
        "group": None,
        "provenance": {"source": "golden"},
    }
    assert _dataset_hash([task]) == "dc945905abc2a870802566997fd6bfc7ea01b9e1c027ad6b95ea731ea04390fe"
