#!/usr/bin/env python3
"""Fail-closed provenance and coverage checks for the frozen R050 handoff.

``preflight`` freezes the server, harness, patch, and public JevBench inputs before
any score is observed. ``postrun`` proves that the resulting WorkflowEvals and
JevBench evidence used exactly those inputs and contains no hidden failures.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gemma_decision_runtime import (
    artifact_identity,
    runtime_environment_identity,
)
from scripts.preflight_r043_assets import file_hash

WORKFLOWEVALS_COMMIT = "0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb"
WORKFLOW_PATCH_SHA256 = "07106bf888aa53a2770b83503c58635708db32042ecb4b034a41963b1ba43908"
WORKFLOW_PATCH_BYTES = 17_920
JEVBENCH_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
FROZEN_RUNTIME = {
    "max_context": 16384,
    "dtype": "bfloat16",
    "device": "cuda",
    "attn_implementation": "sdpa",
}
JEV_INPUT_COUNTS = {"easy.jsonl": 48, "original.jsonl": 72, "hard.jsonl": 111}


@dataclass(frozen=True)
class WorkflowSpec:
    cases: int
    policy_ids: tuple[str, ...]
    metric: str
    dataset_source_revision: str
    dataset_revision: str
    dataset_source: str

    @property
    def decisions(self) -> int:
        return self.cases * len(self.policy_ids)


WORKFLOWS = {
    "security_incidents": WorkflowSpec(
        240,
        ("playbook",),
        "exact_actions",
        "fbe1ea5c69cf494157fd23f2002a0d9d9a418443",
        "d7728a6fcaaf58667ddee86b5c3ab622bfcbc59e380d8303bf97742b709e3d39",
        "https://huggingface.co/datasets/typesafe/evalsafe-security-incidents/tree/fbe1ea5c69cf494157fd23f2002a0d9d9a418443",
    ),
    "agent_trace_observability": WorkflowSpec(
        111,
        ("balanced", "cautious"),
        "primary_action",
        "8635540973910a92465fe2bc53e195375aa6e1a8",
        "70524509cd285387ecebdd6aec3e535d3a8b4afaac47f53479651befe85ba436",
        "https://huggingface.co/datasets/typesafe/evalsafe-agent-trace-observability/tree/8635540973910a92465fe2bc53e195375aa6e1a8",
    ),
    "invoice_processing": WorkflowSpec(
        150,
        ("startup", "enterprise", "high_volume_retailer"),
        "exact_actions",
        "6beeb2d2acd65c086c835022f5f4d7434114cafc",
        "d239f4c8b6ab7c35c234582b5cd21731299bd6ac605c72ed28f8f7476659928c",
        "https://huggingface.co/datasets/typesafe/evalsafe-invoice-processing/tree/6beeb2d2acd65c086c835022f5f4d7434114cafc",
    ),
    "customer_service": WorkflowSpec(
        204,
        ("A", "B", "C"),
        "exact_actions",
        "b1342f5a704587dbc465867c38c2694348ff86e4",
        "cd47359afd54badb35fa7e14746b41e38fd4408c2e485c184010ab4b269ab9c6",
        "https://huggingface.co/datasets/typesafe/evalsafe-customer-service/tree/b1342f5a704587dbc465867c38c2694348ff86e4",
    ),
}
WORKFLOW_THRESHOLDS = {
    "security_incidents": 0.617,
    "agent_trace_observability": 0.716,
    "invoice_processing": 0.618,
    "customer_service": 0.760,
}
WORKFLOW_MEAN_THRESHOLD = 0.678

SHA256 = re.compile(r"[0-9a-f]{64}")


class VerificationError(ValueError):
    """The handoff is incomplete, inconsistent, or not the frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON number: {value}")


def _finite(value: Any, where: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise VerificationError(f"non-finite number at {where}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{where}[{index}]")


def load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read valid JSON from {path}: {error}") from error
    _finite(value, str(path))
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise VerificationError(f"invalid JSON at {path}:{number}: {error}") from error
        require(isinstance(row, dict), f"{path}:{number} must contain an object")
        _finite(row, f"{path}:{number}")
        rows.append(row)
    return rows


def fingerprint(value: Any) -> str:
    wire = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(wire.encode()).hexdigest()


def _receipt_hash(receipt: dict[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return fingerprint(body)


def _git(path: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=path,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr.decode(errors="replace") if binary else result.stderr
        raise VerificationError(f"git {' '.join(args)} failed in {path}: {stderr.strip()}")
    return result.stdout


def _git_head(path: Path) -> str:
    return str(_git(path, "rev-parse", "HEAD")).strip()


def _unique(values: list[Any], what: str) -> set[Any]:
    try:
        unique = set(values)
    except TypeError as error:
        raise VerificationError(f"{what} must be hashable") from error
    require(len(unique) == len(values), f"duplicate {what}")
    return unique


def _artifact_tree(artifact: Any, name: str) -> str:
    require(isinstance(artifact, dict), f"{name} artifact is absent")
    digest = artifact.get("canonical_sha256")
    require(artifact.get("kind") == "tree", f"{name} must be a local canonical tree")
    require(SHA256.fullmatch(str(digest)) is not None, f"{name} canonical hash is invalid")
    require(artifact.get("tree_sha256") == digest, f"{name} tree/canonical hashes differ")
    require(type(artifact.get("files")) is int and artifact["files"] > 0, f"{name} file count is invalid")
    require(type(artifact.get("bytes")) is int and artifact["bytes"] > 0, f"{name} byte count is invalid")
    return str(digest)


def verify_server_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json(path)
    require(isinstance(receipt, dict), "startup receipt must be an object")
    require(receipt.get("schema_version") == "typical.typesafe_startup.v1", "wrong startup receipt schema")
    require(receipt.get("receipt_sha256") == _receipt_hash(receipt), "startup receipt hash mismatch")
    model_id = receipt.get("model_id")
    require(isinstance(model_id, str) and model_id, "startup receipt model ID is absent")
    base = receipt.get("base_model")
    adapter = receipt.get("adapter")
    require(isinstance(base, dict) and isinstance(adapter, dict), "startup artifacts are absent")
    require(base.get("revision") == BASE_REVISION, "wrong frozen base revision")
    _artifact_tree(base.get("artifact"), "base model")
    adapter_hash = _artifact_tree(adapter.get("artifact"), "adapter")
    require(
        re.search(r"-sha256-([0-9a-f]{12,64})$", model_id) is not None
        and adapter_hash.startswith(model_id.rsplit("-sha256-", 1)[1]),
        "model ID is not content-addressed to the adapter",
    )
    require(receipt.get("runtime") == FROZEN_RUNTIME, "startup runtime is not the frozen R050 runtime")

    for name, entry in (("base model", base), ("adapter", adapter)):
        source = entry.get("source")
        require(isinstance(source, str) and Path(source).is_dir(), f"{name} source is not a local tree")
        require(
            artifact_identity(source, entry.get("revision")) == entry["artifact"],
            f"{name} live tree differs from startup receipt",
        )

    scripts = Path(__file__).resolve().parent
    code_paths = {
        "runtime_sha256": scripts / "gemma_decision_runtime.py",
        "server_sha256": scripts / "serve_typesafe_gemma_lora.py",
        "scorer_sha256": scripts / "singleton_option_scorer.py",
        "training_helper_sha256": scripts / "train_gemma_decision_lora.py",
    }
    code = receipt.get("code")
    require(isinstance(code, dict), "startup code receipt is absent")
    require(set(code) == set(code_paths), "startup code receipt has missing or extra entries")
    for field, source in code_paths.items():
        require(code.get(field) == file_hash(source), f"startup {field} differs from current code")
    require(
        receipt.get("environment") == runtime_environment_identity(),
        "startup runtime environment differs from current lock",
    )
    return receipt


def _jev_task_record(row: dict[str, Any]) -> dict[str, Any]:
    required = {"id", "family", "state", "question", "labels", "split"}
    require(required <= set(row), "JevBench task is missing required fields")
    return {
        "id": row["id"],
        "family": row["family"],
        "state": row["state"],
        "question": row["question"],
        "labels": row["labels"],
        "expected": row.get("expected"),
        "split": row["split"],
        "group": row.get("group"),
        "provenance": row.get("provenance", {}),
    }


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    blobs = [json.dumps(_jev_task_record(row), ensure_ascii=False, sort_keys=True) for row in rows]
    for blob in sorted(blobs):
        digest.update(blob.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _workflow_patch_identity(checkout: Path, patch: Path) -> dict[str, Any]:
    require(_git_head(checkout) == WORKFLOWEVALS_COMMIT, "wrong WorkflowEvals upstream commit")
    require(file_hash(patch) == WORKFLOW_PATCH_SHA256, "wrong disclosed WorkflowEvals patch hash")
    require(patch.stat().st_size == WORKFLOW_PATCH_BYTES, "wrong disclosed WorkflowEvals patch size")
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch.resolve())],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    require(reverse.returncode == 0, "WorkflowEvals checkout is not the disclosed applied patch")
    patch_paths = {
        line.split(" b/", 1)[1]
        for line in patch.read_text().splitlines()
        if line.startswith("diff --git a/") and " b/" in line
    }
    status = str(_git(checkout, "status", "--porcelain=v1", "--untracked-files=all"))
    changed_paths = {
        line[3:].split(" -> ", 1)[-1]
        for line in status.splitlines()
        if len(line) >= 4
    }
    require(changed_paths == patch_paths, "WorkflowEvals checkout has patch drift or extra files")
    with tempfile.TemporaryDirectory(prefix="r050-workflow-patch-") as temporary:
        expected_root = Path(temporary)
        for relative in patch_paths:
            source = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=checkout,
                check=False,
                capture_output=True,
            )
            if source.returncode == 0:
                destination = expected_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.stdout)
        applied = subprocess.run(
            ["git", "apply", str(patch.resolve())],
            cwd=expected_root,
            check=False,
            capture_output=True,
            text=True,
        )
        require(applied.returncode == 0, f"cannot reconstruct disclosed patch: {applied.stderr.strip()}")
        for relative in patch_paths:
            actual = checkout / relative
            expected = expected_root / relative
            require(
                actual.is_file() == expected.is_file()
                and (not actual.is_file() or actual.read_bytes() == expected.read_bytes()),
                f"WorkflowEvals patched file differs from disclosure: {relative}",
            )
    return {
        "upstream_commit": WORKFLOWEVALS_COMMIT,
        "patch_sha256": WORKFLOW_PATCH_SHA256,
        "patch_bytes": WORKFLOW_PATCH_BYTES,
    }


def _python_revision(root: Path) -> str:
    """Reproduce WorkflowEvals' code revision without importing the checkout."""
    return fingerprint(
        {
            str(path.relative_to(root)): path.read_text()
            for path in sorted(root.rglob("*.py"))
            if not any(part in path.parts for part in (".venv", "tests", "__pycache__"))
        }
    )


def _locked_dependencies(lock_path: Path) -> dict[str, str]:
    lock = tomllib.loads(lock_path.read_text())
    packages = {
        package.get("name"): package.get("version")
        for package in lock.get("package", [])
        if isinstance(package, dict)
    }
    expected = {}
    for name in ("typesafe-sdk", "system-one-adapter", "pydantic-ai-slim", "pydantic"):
        version = packages.get(name)
        require(isinstance(version, str) and version, f"WorkflowEvals lock lacks {name}")
        expected[name] = version
    return expected


def _workflow_configuration_contract(checkout: Path) -> dict[str, Any]:
    lock_path = checkout / "uv.lock"
    prices = load_json(checkout / "core" / "pricing.json")
    require(isinstance(prices, dict), "WorkflowEvals pricing must be an object")
    return {
        "infra_revision": _python_revision(checkout / "core"),
        "workflow_revisions": {
            name: _python_revision(checkout / "evals" / name) for name in WORKFLOWS
        },
        "dependency_lock": fingerprint(lock_path.read_text()),
        "dependencies": _locked_dependencies(lock_path),
        "prices": prices,
    }


def _validate_workflow_schemas(checkout: Path, directories: list[Path]) -> None:
    """Use the frozen upstream Pydantic contracts and scorer as the authority."""
    program = """
import sys
from pathlib import Path
from core.io import load_results
from core.scoring import ScoreData, build_score_data
for raw in sys.argv[1:]:
    directory = Path(raw)
    results = load_results(directory)
    scores = ScoreData.model_validate_json((directory / 'scores.json').read_text())
    rebuilt = build_score_data(results, policy_ids=[p.policy_id for p in results.policies])
    if rebuilt != scores:
        raise ValueError(f'{directory}: scores differ from frozen scorer recomputation')
"""
    result = subprocess.run(
        ["uv", "run", "--frozen", "python", "-c", program, *(str(path.resolve()) for path in directories)],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, f"WorkflowEvals schema/scorer validation failed: {result.stderr.strip()}")


def _jev_preflight_identity(
    jevbench_dir: Path,
    jev_runner: Path,
    task_files: list[Path],
) -> dict[str, Any]:
    jevbench_dir = jevbench_dir.resolve()
    require(_git_head(jevbench_dir) == JEVBENCH_COMMIT, "wrong current JevBench main commit")
    require(
        not str(_git(jevbench_dir, "status", "--porcelain=v1", "--untracked-files=all")).strip(),
        "JevBench checkout must be clean",
    )
    require(
        jev_runner.resolve() == Path(__file__).resolve().with_name("jevbench_gemma_lora.py"),
        "wrong Typical JevBench runner path",
    )
    require(len(task_files) == len(JEV_INPUT_COUNTS), "JevBench requires exactly three public task files")
    names = [path.name for path in task_files]
    require(len(names) == len(set(names)) and set(names) == set(JEV_INPUT_COUNTS), "wrong JevBench public task filenames")
    all_rows: list[dict[str, Any]] = []
    inputs = []
    for path in task_files:
        resolved = path.resolve()
        require(
            resolved.parent == jevbench_dir / "datasets" / "public",
            f"JevBench input is outside the pinned public dataset: {path}",
        )
        rows = load_jsonl(resolved)
        require(len(rows) == JEV_INPUT_COUNTS[path.name], f"wrong {path.name} task count")
        _unique([row.get("id") for row in rows], f"{path.name} task IDs")
        inputs.append(
            {
                "path": str(resolved),
                "name": path.name,
                "sha256": file_hash(resolved),
                "count": len(rows),
                "dataset_hash": _dataset_hash(rows),
            }
        )
        all_rows.extend(rows)
    task_ids = _unique([row.get("id") for row in all_rows], "JevBench task IDs")
    require(len(task_ids) == sum(JEV_INPUT_COUNTS.values()), "wrong total JevBench public task count")
    scripts = Path(__file__).resolve().parent
    code = {
        "jevbench_gemma_lora.py": file_hash(jev_runner.resolve()),
        "gemma_decision_runtime.py": file_hash(scripts / "gemma_decision_runtime.py"),
        "singleton_option_scorer.py": file_hash(scripts / "singleton_option_scorer.py"),
        "preflight_r043_assets.py": file_hash(scripts / "preflight_r043_assets.py"),
        "train_gemma_decision_lora.py": file_hash(scripts / "train_gemma_decision_lora.py"),
    }
    harness_files = {
        relative: file_hash(jevbench_dir / relative)
        for relative in (
            "jevbench/__init__.py",
            "jevbench/budget.py",
            "jevbench/tasks.py",
            "jevbench/runner.py",
            "jevbench/scoring.py",
            "jevbench/adapters/base.py",
        )
    }
    return {
        "harness_commit": JEVBENCH_COMMIT,
        "harness_files": harness_files,
        "code": code,
        "environment": runtime_environment_identity(),
        "task_inputs": inputs,
        "selected_dataset_hash": _dataset_hash(all_rows),
        "task_ids": sorted(task_ids),
    }


def build_preflight(
    *,
    startup_receipt: Path,
    workflowevals_dir: Path,
    workflow_patch: Path,
    jevbench_dir: Path,
    jev_runner: Path,
    task_files: list[Path],
    base_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    server = verify_server_receipt(startup_receipt)
    require(base_url == "http://127.0.0.1:8090", "official endpoint must be the frozen loopback URL")
    require(timeout_s == 1800, "official WorkflowEvals timeout must be 1800 seconds")
    workflow = _workflow_patch_identity(workflowevals_dir.resolve(), workflow_patch.resolve())
    workflow["configuration_contract"] = _workflow_configuration_contract(
        workflowevals_dir.resolve()
    )

    jev = _jev_preflight_identity(jevbench_dir, jev_runner, task_files)
    receipt = {
        "schema_version": "typical.r050_official_eval_preflight.v1",
        "server": {
            "startup_receipt_sha256": file_hash(startup_receipt),
            "startup_receipt": server,
        },
        "workflowevals": {
            **workflow,
            "base_url": base_url,
            "timeout_s": timeout_s,
            "workflows": {
                name: {
                    "cases": spec.cases,
                    "policy_ids": list(spec.policy_ids),
                    "decisions": spec.decisions,
                    "metric": spec.metric,
                    "dataset_source_revision": spec.dataset_source_revision,
                    "dataset_revision": spec.dataset_revision,
                    "dataset_source": spec.dataset_source,
                }
                for name, spec in WORKFLOWS.items()
            },
        },
        "jevbench": jev,
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def _check_configuration(
    config: Any,
    *,
    workflow: str,
    spec: WorkflowSpec,
    model_id: str,
    base_url: str,
    timeout_s: float,
    contract: dict[str, Any],
) -> None:
    require(isinstance(config, dict), f"{workflow}: configuration is absent")
    model = config.get("model")
    require(isinstance(model, dict), f"{workflow}: model configuration is absent")
    expected_model = {
        "provider": "typical",
        "model": model_id,
        "thinking": "off",
        "question_mode": "all",
        "base_url": base_url,
        "timeout_s": timeout_s,
        "max_output_tokens": 64000,
        "malformed_retries": 1,
    }
    require(model == expected_model, f"{workflow}: model/runtime configuration mismatch")
    require(config.get("policy_ids") == list(spec.policy_ids), f"{workflow}: policy scope mismatch")
    require(config.get("structured_outputs") is True, f"{workflow}: structured outputs disabled")
    require(config.get("answer_mode") == "probabilities", f"{workflow}: wrong answer mode")
    require(config.get("normalize_probabilities") is True, f"{workflow}: probability normalization disabled")
    require(config.get("dataset_source") == spec.dataset_source, f"{workflow}: wrong dataset source")
    require(config.get("dataset_revision") == spec.dataset_revision, f"{workflow}: wrong dataset revision")
    require(config.get("prices") == contract.get("prices"), f"{workflow}: pricing configuration drift")
    require(config.get("infra_revision") == contract.get("infra_revision"), f"{workflow}: infrastructure revision drift")
    require(
        config.get("workflow_revision") == contract.get("workflow_revisions", {}).get(workflow),
        f"{workflow}: workflow revision drift",
    )
    require(config.get("dependency_lock") == contract.get("dependency_lock"), f"{workflow}: dependency lock drift")
    require(config.get("dependencies") == contract.get("dependencies"), f"{workflow}: dependency versions drift")
    require(not any(str(key).startswith("typical:") for key in config["prices"]), f"{workflow}: unverified Typical price present")


def _checkpoint_cases(directory: Path) -> dict[str, dict[str, Any]]:
    paths = sorted((directory / "cases").glob("*.json"))
    cases: dict[str, dict[str, Any]] = {}
    for path in paths:
        value = load_json(path)
        require(isinstance(value, dict), f"{path}: checkpoint must be an object")
        case_id = value.get("case_id")
        require(isinstance(case_id, str) and case_id, f"{path}: checkpoint case ID is invalid")
        require(case_id not in cases, f"duplicate checkpoint case ID {case_id!r}")
        cases[case_id] = value
        expected_name = fingerprint(case_id)
        require(path.stem == expected_name, f"{path}: checkpoint filename does not bind its case ID")
    return cases


def _workflow_evidence_identity(directory: Path) -> dict[str, Any]:
    return {
        "run_tree": artifact_identity(str(directory), None),
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "results_sha256": file_hash(directory / "results.json"),
        "scores_sha256": file_hash(directory / "scores.json"),
        "cases": artifact_identity(str(directory / "cases"), None),
    }


def _jev_evidence_identity(directory: Path) -> dict[str, Any]:
    return {
        "run_tree": artifact_identity(str(directory), None),
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "results_sha256": file_hash(directory / "results.jsonl"),
        "ledger_sha256": file_hash(directory / "ledger.jsonl"),
        "raw": artifact_identity(str(directory / "raw"), None),
    }


def verify_workflow_run(
    directory: Path,
    *,
    spec: WorkflowSpec,
    model_id: str,
    base_url: str,
    timeout_s: float,
    configuration_contract: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_json(directory / "manifest.json")
    results = load_json(directory / "results.json")
    scores = load_json(directory / "scores.json")
    require(isinstance(results, dict) and isinstance(scores, dict), f"{directory}: malformed output")
    workflow = results.get("eval_id")
    require(isinstance(workflow, str), f"{directory}: eval ID is absent")
    require(workflow in WORKFLOWS and WORKFLOWS[workflow] == spec, f"{directory}: unexpected workflow")

    declared = results.get("cases")
    runs = results.get("runs")
    policies = results.get("policies")
    require(isinstance(declared, list) and isinstance(runs, list) and len(runs) == 1, f"{workflow}: malformed run bundle")
    require(isinstance(policies, list), f"{workflow}: policies are absent")
    require([policy.get("policy_id") for policy in policies] == list(spec.policy_ids), f"{workflow}: policy definitions mismatch")
    metric_definitions = results.get("metrics")
    require(isinstance(metric_definitions, list), f"{workflow}: metric definitions are absent")
    metric_ids = [metric.get("metric_id") for metric in metric_definitions]
    _unique(metric_ids, f"{workflow} metric IDs")
    require(spec.metric in metric_ids, f"{workflow}: required metric definition {spec.metric!r} is absent")
    required_metric = next(metric for metric in metric_definitions if metric.get("metric_id") == spec.metric)
    require(required_metric.get("comparison") == spec.metric, f"{workflow}: required metric comparison mismatch")
    declared_ids = [case.get("case_id") for case in declared]
    expected_ids = _unique(declared_ids, f"{workflow} declared case IDs")
    require(len(expected_ids) == spec.cases, f"{workflow}: wrong expected case count")
    require(
        fingerprint({case["case_id"]: case.get("content_hash") for case in declared})
        == spec.dataset_revision,
        f"{workflow}: case ID/content-hash digest differs from frozen dataset",
    )

    run = runs[0]
    expected_identity = {"name": f"typical:{model_id}", "lab": "other", "effort": None}
    require(run.get("model") == expected_identity, f"{workflow}: run model identity mismatch")
    config = run.get("configuration")
    _check_configuration(
        config,
        workflow=workflow,
        spec=spec,
        model_id=model_id,
        base_url=base_url,
        timeout_s=timeout_s,
        contract=configuration_contract,
    )
    require(config.get("policies") == policies, f"{workflow}: policy configuration drift")
    require(run.get("configuration_hash") == fingerprint(config), f"{workflow}: configuration hash mismatch")
    require(
        manifest == {
            "run_id": run.get("run_id"),
            "configuration_hash": run.get("configuration_hash"),
            "configuration": config,
        },
        f"{workflow}: manifest/run mismatch",
    )
    require(
        results.get("dataset_revision") == spec.dataset_revision == config.get("dataset_revision"),
        f"{workflow}: dataset revision mismatch",
    )

    cases = run.get("cases")
    require(isinstance(cases, list), f"{workflow}: run cases are absent")
    run_ids = [case.get("case_id") for case in cases]
    _unique(run_ids, f"{workflow} run case IDs")
    require(set(run_ids) == expected_ids and len(run_ids) == spec.cases, f"{workflow}: missing or extra run case IDs")
    checkpoint_cases = _checkpoint_cases(directory)
    require(set(checkpoint_cases) == expected_ids and len(checkpoint_cases) == spec.cases, f"{workflow}: missing or extra checkpoints")
    require(
        checkpoint_cases == {case["case_id"]: case for case in cases},
        f"{workflow}: aggregate results differ from durable checkpoints",
    )
    attempts = list((directory / "attempts").glob("*.json")) if (directory / "attempts").exists() else []
    require(not attempts, f"{workflow}: retry attempt artifacts are present")

    expected_policies = set(spec.policy_ids)
    for case in cases:
        case_id = case.get("case_id")
        expected_hash = next(item.get("content_hash") for item in declared if item.get("case_id") == case_id)
        require(case.get("input_status") == "verified" and case.get("case_hash") == expected_hash, f"{workflow}/{case_id}: stale or unverified input")
        decisions = case.get("decisions")
        require(isinstance(decisions, list), f"{workflow}/{case_id}: decisions are absent")
        decision_ids = [decision.get("policy_id") for decision in decisions]
        _unique(decision_ids, f"{workflow}/{case_id} decision policy IDs")
        require(set(decision_ids) == expected_policies, f"{workflow}/{case_id}: missing or extra decisions")
        require(all(decision.get("status") == "ok" for decision in decisions), f"{workflow}/{case_id}: failed decision")
        execution = case.get("execution")
        require(isinstance(execution, dict), f"{workflow}/{case_id}: execution is absent")
        require(execution.get("status") == "ok" and execution.get("reason") is None, f"{workflow}/{case_id}: failed execution")
        require(set(execution.get("policy_ids", [])) == expected_policies, f"{workflow}/{case_id}: execution policy mismatch")
        calls = case.get("calls")
        require(isinstance(calls, list) and execution.get("calls") == len(calls), f"{workflow}/{case_id}: call count mismatch")
        for call in calls:
            require(call.get("status") == "ok" and call.get("error") is None, f"{workflow}/{case_id}: failed call")
            require(call.get("retries") in (None, 0), f"{workflow}/{case_id}: nonzero retries")
            questions = call.get("questions")
            response = call.get("response")
            require(isinstance(questions, dict) and questions, f"{workflow}/{case_id}: empty call")
            require(isinstance(response, dict), f"{workflow}/{case_id}: missing response")
            require(response.get("model") == model_id, f"{workflow}/{case_id}: model substitution")
            answers = response.get("answers")
            require(isinstance(answers, dict) and set(answers) == set(questions), f"{workflow}/{case_id}: response question IDs mismatch")

    require(scores.get("eval_id") == workflow, f"{workflow}: score eval mismatch")
    require(scores.get("dataset_revision") == results.get("dataset_revision"), f"{workflow}: score dataset mismatch")
    require(scores.get("allow_unverified") is False, f"{workflow}: unverified scores enabled")
    require(scores.get("failure_handling") == "count_prediction_errors_as_wrong", f"{workflow}: wrong failure handling")
    consensus = results.get("references", {}).get("consensus")
    require(isinstance(consensus, str) and consensus, f"{workflow}: consensus reference is absent")
    run_id = run.get("run_id")
    expected_pairs = {(case_id, policy_id) for case_id in expected_ids for policy_id in spec.policy_ids}
    rows = [
        row
        for row in scores.get("scores", [])
        if row.get("run_id") == run_id and row.get("label_set_id") == consensus
    ]
    pairs = [(row.get("case_id"), row.get("policy_id")) for row in rows]
    _unique(pairs, f"{workflow} consensus score pairs")
    require(set(pairs) == expected_pairs and len(rows) == spec.decisions, f"{workflow}: incomplete consensus score coverage")
    for row in rows:
        require(row.get("status") == "scored" and row.get("unverified") is False, f"{workflow}: failed consensus score")
        require(type(row.get("metrics", {}).get(spec.metric)) is bool, f"{workflow}: missing consensus metric")
    series = [
        item
        for item in scores.get("series", [])
        if item.get("run_id") == run_id and item.get("label_set_id") == consensus
    ]
    require(len(series) == 1, f"{workflow}: consensus series is absent or duplicated")
    series = series[0]
    metric = series.get("metrics", {}).get(spec.metric)
    require(isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric), f"{workflow}: non-finite or absent metric")
    require(0 <= metric <= 1, f"{workflow}: metric is outside [0,1]")
    require(series.get("cases") == spec.cases and series.get("decisions") == spec.decisions, f"{workflow}: wrong scored counts")
    require(series.get("status_counts") == {"scored": spec.decisions}, f"{workflow}: nonzero score failures")
    require(series.get("unverified_decisions") == 0, f"{workflow}: unverified decisions")
    require(series.get("metric_denominators", {}).get(spec.metric) == spec.decisions, f"{workflow}: wrong metric denominator")
    recomputed_metric = math.fsum(
        int(row["metrics"][spec.metric]) for row in rows
    ) / spec.decisions
    require(
        math.isclose(metric, recomputed_metric, rel_tol=1e-12, abs_tol=1e-12),
        f"{workflow}: aggregate metric differs from consensus rows",
    )
    require(directory.name == run_id, f"{workflow}: directory basename must equal run ID")
    return {
        "workflow": workflow,
        "run_id": run_id,
        "cases": spec.cases,
        "decisions": spec.decisions,
        "metric": metric,
        "evidence": _workflow_evidence_identity(directory),
    }


JEV_RESULT_FIELDS = {
    "task_id", "family", "split", "group", "ts", "status", "ok", "valid", "correct",
    "predicted", "ordinal_ev", "probs", "probs_as_returned", "strict_valid",
    "renormalized", "probs_source", "model", "error", "schema_error", "status_code",
    "latency_s", "usage", "cost_usd", "cost_basis", "reserved_usd", "charged_usd",
    "raw_sha256", "runtime",
}
JEV_MANIFEST_FIELDS = {
    "adapter", "model", "model_revision", "model_artifact", "lora_adapter",
    "lora_revision", "lora_artifact", "code", "harness_commit", "harness_code",
    "environment", "task_inputs", "selected_dataset_hash", "max_context", "model_context",
    "dtype", "device", "attn_implementation", "seed", "counts", "statuses",
    "status_codes", "started_utc", "finished_utc", "load_s", "torch_version", "gpu",
    "score_scope",
}


def _validate_native_probabilities(task: dict[str, Any], probabilities: Any) -> dict[str, float]:
    labels = task.get("labels")
    require(isinstance(labels, list) and labels, f"JevBench task {task.get('id')} has invalid labels")
    require(isinstance(probabilities, dict), f"JevBench task {task.get('id')} lacks probabilities")
    require(list(probabilities) == labels, f"JevBench task {task.get('id')} probability order/labels differ")
    clean: dict[str, float] = {}
    for label, value in probabilities.items():
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1,
            f"JevBench task {task.get('id')} has invalid probability",
        )
        clean[label] = float(value)
    require(
        abs(math.fsum(clean.values()) - 1.0) <= 1e-9,
        f"JevBench task {task.get('id')} probabilities do not sum to one at runtime tolerance",
    )
    return clean


def _verify_jev_ledger(directory: Path, results: list[dict[str, Any]]) -> None:
    task_ids = [str(result["task_id"]) for result in results]
    rows = load_jsonl(directory / "ledger.jsonl")
    require(len(rows) == 1 + 2 * len(task_ids), "JevBench ledger has wrong event count")
    cap = rows[0]
    require(
        set(cap) == {"event", "cap_usd", "ts"}
        and cap["event"] == "cap"
        and cap["cap_usd"] == 15
        and isinstance(cap["ts"], (int, float))
        and math.isfinite(cap["ts"]),
        "JevBench ledger lacks the exact initial cap",
    )
    reservation_ids: set[str] = set()
    for index, task_id in enumerate(task_ids):
        reserve, settle = rows[1 + 2 * index:3 + 2 * index]
        reservation_id = reserve.get("reservation_id")
        require(
            set(reserve) == {"event", "reservation_id", "reserved_usd", "ts", "meta"}
            and reserve.get("event") == "reserve"
            and isinstance(reservation_id, str)
            and reservation_id
            and reservation_id not in reservation_ids
            and reserve.get("reserved_usd") == 0
            and reserve.get("meta") == {"task_id": task_id, "adapter": "typical_gemma_lora"}
            and isinstance(reserve.get("ts"), (int, float))
            and math.isfinite(reserve["ts"]),
            f"JevBench ledger reservation mismatch for {task_id}",
        )
        reservation_ids.add(reservation_id)
        require(
            set(settle) == {"event", "reservation_id", "charged_usd", "ts", "meta"}
            and settle.get("event") == "settle"
            and settle.get("reservation_id") == reservation_id
            and settle.get("charged_usd") == 0
            and settle.get("meta")
            == {"task_id": task_id, "basis": "local_gpu_no_provider_tariff"}
            and isinstance(settle.get("ts"), (int, float))
            and math.isfinite(settle["ts"]),
            f"JevBench ledger settlement mismatch for {task_id}",
        )
        require(
            cap["ts"] <= reserve["ts"] <= settle["ts"] <= results[index]["ts"],
            f"JevBench ledger timestamp order mismatch for {task_id}",
        )


def _validate_jev_with_frozen_scorer(
    checkout: Path,
    task_files: list[Path],
    results_path: Path,
) -> dict[str, dict[str, Any]]:
    program = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from jevbench.scoring import score_task
from jevbench.tasks import load_jsonl
tasks = {}
for path in sys.argv[3:]:
    for task in load_jsonl(path):
        if task.id in tasks:
            raise ValueError(f'duplicate task: {task.id}')
        tasks[task.id] = task
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
if {row.get('task_id') for row in rows} != set(tasks) or len(rows) != len(tasks):
    raise ValueError('result/task scope mismatch')
scores = {}
for row in rows:
    score = score_task(row.get('probs_as_returned'), tasks[row['task_id']])
    scores[row['task_id']] = score
print(json.dumps(scores, sort_keys=True, allow_nan=False))
"""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "python",
            "-c",
            program,
            str(checkout.resolve()),
            str(results_path.resolve()),
            *(str(path.resolve()) for path in task_files),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, f"JevBench frozen scorer validation failed: {result.stderr.strip()}")
    try:
        scores = json.loads(result.stdout, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise VerificationError("JevBench frozen scorer returned invalid evidence") from error
    require(isinstance(scores, dict), "JevBench frozen scorer returned invalid evidence")
    return scores


def verify_jev_run(
    directory: Path,
    expected: dict[str, Any],
    server: dict[str, Any],
    task_rows: list[dict[str, Any]],
    frozen_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = load_json(directory / "manifest.json")
    require(isinstance(manifest, dict), "JevBench manifest must be an object")
    require(set(manifest) == JEV_MANIFEST_FIELDS, "JevBench manifest schema mismatch")
    require(manifest.get("harness_commit") == expected.get("harness_commit"), "wrong JevBench harness provenance")
    require(manifest.get("harness_code") == expected.get("harness_files"), "wrong JevBench harness code provenance")
    require(manifest.get("code") == expected.get("code"), "wrong JevBench runner/runtime provenance")
    require(manifest.get("environment") == expected.get("environment"), "wrong JevBench runtime environment")
    require(manifest.get("environment") == server.get("environment"), "JevBench/server runtime environments differ")
    require(manifest.get("adapter") == "typical_gemma_lora", "wrong JevBench adapter name")
    require(manifest.get("model") == server["base_model"]["source"], "wrong JevBench model source")
    require(manifest.get("model_revision") == server["base_model"]["revision"] == BASE_REVISION, "wrong JevBench base revision")
    require(manifest.get("lora_adapter") == server["adapter"]["source"], "wrong JevBench adapter source")
    require(manifest.get("lora_revision") == server["adapter"]["revision"], "wrong JevBench adapter revision")
    require(manifest.get("model_artifact") == server["base_model"]["artifact"], "JevBench base artifact mismatch")
    require(manifest.get("lora_artifact") == server["adapter"]["artifact"], "JevBench adapter artifact mismatch")
    for field, value in FROZEN_RUNTIME.items():
        require(manifest.get(field) == value, f"JevBench {field} mismatch")
    require(manifest.get("seed") == 42, "wrong JevBench seed")
    try:
        started = dt.datetime.fromisoformat(manifest["started_utc"])
        finished = dt.datetime.fromisoformat(manifest["finished_utc"])
    except (TypeError, ValueError) as error:
        raise VerificationError("JevBench manifest timestamps are invalid") from error
    require(
        started.tzinfo is not None and finished.tzinfo is not None and started <= finished,
        "JevBench manifest timestamp order is invalid",
    )
    require(
        isinstance(manifest.get("load_s"), (int, float))
        and not isinstance(manifest["load_s"], bool)
        and math.isfinite(manifest["load_s"])
        and manifest["load_s"] >= 0,
        "JevBench model load time is invalid",
    )
    require(
        type(manifest.get("model_context")) is int
        and manifest["model_context"] >= FROZEN_RUNTIME["max_context"],
        "JevBench model context is invalid",
    )
    require(
        isinstance(manifest.get("torch_version"), str)
        and bool(manifest["torch_version"].strip())
        and isinstance(manifest.get("gpu"), str)
        and bool(manifest["gpu"].strip()),
        "JevBench runtime telemetry is incomplete",
    )
    require(
        manifest.get("score_scope")
        == "per-item harness evidence only; no sealed or headline score computed",
        "JevBench score scope mismatch",
    )

    actual_inputs = manifest.get("task_inputs")
    require(isinstance(actual_inputs, list), "JevBench task provenance is absent")
    def normalized(item: dict[str, Any]) -> dict[str, Any]:
        return {key: item.get(key) for key in ("sha256", "count", "dataset_hash")}
    actual_by_name = {Path(str(item.get("path"))).name: normalized(item) for item in actual_inputs}
    require(len(actual_by_name) == len(actual_inputs), "duplicate JevBench task input")
    frozen_by_name = {
        item["name"]: normalized(item) for item in expected.get("task_inputs", [])
    }
    require(actual_by_name == frozen_by_name, "wrong JevBench task inputs")
    require(manifest.get("selected_dataset_hash") == expected.get("selected_dataset_hash"), "wrong JevBench selected task hash")

    count = len(expected.get("task_ids", []))
    counts = manifest.get("counts")
    require(
        counts == {
            "planned": count,
            "attempted": count,
            "unattempted": 0,
            "valid": count,
            "strict_valid": count,
        },
        "JevBench run is incomplete or invalid",
    )
    require(manifest.get("statuses") == {"ok": count}, "JevBench contains failures")
    require(manifest.get("status_codes") == {"200": count}, "JevBench contains non-200 results")
    rows = load_jsonl(directory / "results.jsonl")
    tasks = {task.get("id"): _jev_task_record(task) for task in task_rows}
    require(len(tasks) == len(task_rows), "duplicate JevBench source task IDs")
    require(set(tasks) == set(expected["task_ids"]), "JevBench source task scope mismatch")
    require(set(frozen_scores) == set(tasks), "JevBench frozen score scope mismatch")
    ids = [row.get("task_id") for row in rows]
    _unique(ids, "JevBench result task IDs")
    require(set(ids) == set(expected["task_ids"]) and len(rows) == count, "missing or extra JevBench task IDs")
    raw_dir = directory / "raw"
    raw_paths = list(raw_dir.glob("*.json"))
    require(len(raw_paths) == count, "wrong JevBench raw evidence count")
    expected_model = f"{manifest['model']}+{manifest['lora_adapter']}"
    for row in rows:
        task_id = row.get("task_id")
        require(set(row) == JEV_RESULT_FIELDS, f"JevBench task {task_id} result schema mismatch")
        task = tasks.get(task_id)
        require(task is not None, f"unknown JevBench task {task_id}")
        native_probs = _validate_native_probabilities(task, row.get("probs_as_returned"))
        score = frozen_scores[task_id]
        require(
            row.get("status") == "ok"
            and row.get("ok") is True
            and row.get("valid") is True
            and row.get("strict_valid") is True
            and row.get("status_code") == 200,
            f"JevBench task {task_id} failed strict validation",
        )
        require(row.get("probs_source") == "native", f"JevBench task {task_id} lacks native probabilities")
        require(row.get("probs_as_returned") == native_probs, f"JevBench task {task_id} native probabilities changed")
        for field in ("valid", "strict_valid", "renormalized", "probs", "predicted", "correct", "ordinal_ev"):
            value = score.get(field)
            actual = row.get(field)
            if isinstance(value, float):
                require(isinstance(actual, (int, float)) and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12), f"JevBench task {task_id} {field} differs from rescoring")
            else:
                require(actual == value, f"JevBench task {task_id} {field} differs from rescoring")
        require(row.get("schema_error") == score.get("error"), f"JevBench task {task_id} schema_error differs from rescoring")
        require(
            (row.get("family"), row.get("split"), row.get("group"))
            == (task["family"], task["split"], task["group"]),
            f"JevBench task {task_id} metadata mismatch",
        )
        require(row.get("model") == expected_model, f"JevBench task {task_id} model mismatch")
        usage = row.get("usage")
        require(
            isinstance(usage, dict)
            and type(usage.get("input_tokens")) is int
            and usage["input_tokens"] >= 0
            and usage.get("output_tokens") == 0,
            f"JevBench task {task_id} usage mismatch",
        )
        runtime = {
            "input_tokens": usage["input_tokens"],
            "max_context": FROZEN_RUNTIME["max_context"],
            "truncated": False,
            "probability_origin": "native-option-logits",
        }
        require(row.get("runtime") == runtime, f"JevBench task {task_id} runtime mismatch")
        require(
            isinstance(row.get("ts"), (int, float))
            and math.isfinite(row["ts"])
            and isinstance(row.get("latency_s"), (int, float))
            and math.isfinite(row["latency_s"])
            and row["latency_s"] >= 0,
            f"JevBench task {task_id} timing is invalid",
        )
        require(
            row.get("cost_usd") is None
            and row.get("cost_basis") == "local_gpu_no_provider_tariff"
            and row.get("reserved_usd") == 0
            and row.get("charged_usd") == 0,
            f"JevBench task {task_id} local cost evidence mismatch",
        )
        raw = raw_dir / f"{hashlib.sha256(str(task_id).encode()).hexdigest()}.json"
        require(raw.is_file() and file_hash(raw) == row.get("raw_sha256"), f"JevBench task {task_id} raw evidence mismatch")
        raw_record = load_json(raw)
        require(
            raw_record
            == {
                "request": {
                    "state": task["state"],
                    "questions": {"decision": task["question"]},
                    "labels": task["labels"],
                },
                "response": {
                    "model": expected_model,
                    "probabilities": native_probs,
                    "usage": usage,
                    "runtime": runtime,
                },
                "http_status": 200,
            },
            f"JevBench task {task_id} raw request/response mismatch",
        )
    _verify_jev_ledger(directory, rows)
    return {
        "scope": "legacy_public_231_diagnostic",
        "official_v1_5_4_parity": "unmeasured",
        "tasks": count,
        "dataset_hash": expected["selected_dataset_hash"],
        "evidence": _jev_evidence_identity(directory),
    }


def workflow_parity(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {item["workflow"]: item["metric"] for item in summaries}
    require(set(metrics) == set(WORKFLOW_THRESHOLDS), "workflow parity summary is incomplete")
    mean = math.fsum(metrics.values()) / len(metrics)
    floors = {
        workflow: {
            "metric": metrics[workflow],
            "target": target,
            "pass": metrics[workflow] >= target,
        }
        for workflow, target in WORKFLOW_THRESHOLDS.items()
    }
    return {
        "floors": floors,
        "equal_workflow_mean": mean,
        "equal_workflow_mean_target": WORKFLOW_MEAN_THRESHOLD,
        "workflow_parity_pass": mean >= WORKFLOW_MEAN_THRESHOLD
        and all(item["pass"] for item in floors.values()),
    }


def verify_preflight_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json(path)
    require(isinstance(receipt, dict), "preflight receipt must be an object")
    require(receipt.get("schema_version") == "typical.r050_official_eval_preflight.v1", "wrong preflight schema")
    require(receipt.get("receipt_sha256") == _receipt_hash(receipt), "preflight receipt hash mismatch")
    return receipt


def verify_postrun(
    *,
    preflight_path: Path,
    startup_receipt: Path,
    workflow_runs: list[Path],
    workflowevals_dir: Path,
    jevbench_dir: Path,
    jev_runner: Path,
    jev_run: Path,
) -> dict[str, Any]:
    preflight = verify_preflight_receipt(preflight_path)
    server = verify_server_receipt(startup_receipt)
    require(file_hash(startup_receipt) == preflight["server"]["startup_receipt_sha256"], "startup receipt file changed after preflight")
    require(server == preflight["server"]["startup_receipt"], "startup receipt content changed after preflight")
    current_workflow = _workflow_patch_identity(
        workflowevals_dir.resolve(),
        Path(__file__).resolve().parents[1]
        / "patches"
        / "workflowevals-typical-provider-0ac3b8ad.patch",
    )
    frozen_workflow = {
        key: preflight["workflowevals"][key]
        for key in ("upstream_commit", "patch_sha256", "patch_bytes")
    }
    require(current_workflow == frozen_workflow, "WorkflowEvals harness changed after preflight")
    current_contract = _workflow_configuration_contract(workflowevals_dir.resolve())
    require(
        current_contract == preflight["workflowevals"].get("configuration_contract"),
        "WorkflowEvals configuration contract changed after preflight",
    )
    _validate_workflow_schemas(workflowevals_dir.resolve(), workflow_runs)
    current_jev = _jev_preflight_identity(
        jevbench_dir,
        jev_runner,
        [Path(item["path"]) for item in preflight["jevbench"].get("task_inputs", [])],
    )
    require(current_jev == preflight.get("jevbench"), "JevBench sources changed after preflight")
    require(len(workflow_runs) == len(WORKFLOWS), "exactly four WorkflowEvals runs are required")
    by_eval: dict[str, Path] = {}
    for directory in workflow_runs:
        result = load_json(directory / "results.json")
        eval_id = result.get("eval_id") if isinstance(result, dict) else None
        require(isinstance(eval_id, str) and eval_id not in by_eval, "duplicate or absent workflow eval ID")
        by_eval[eval_id] = directory
    require(set(by_eval) == set(WORKFLOWS), "missing or extra workflow run")

    workflow_receipt = preflight["workflowevals"]
    summaries = [
        verify_workflow_run(
            by_eval[name],
            spec=spec,
            model_id=server["model_id"],
            base_url=workflow_receipt["base_url"],
            timeout_s=workflow_receipt["timeout_s"],
            configuration_contract=current_contract,
        )
        for name, spec in WORKFLOWS.items()
    ]
    require(sum(item["cases"] for item in summaries) == 705, "wrong aggregate workflow case count")
    require(sum(item["decisions"] for item in summaries) == 1524, "wrong aggregate workflow decision count")
    require(len({item["run_id"] for item in summaries}) == 1, "workflow runs use different run IDs")
    parity = workflow_parity(summaries)
    jev_task_rows = [
        row
        for item in current_jev["task_inputs"]
        for row in load_jsonl(Path(item["path"]))
    ]
    frozen_scores = _validate_jev_with_frozen_scorer(
        jevbench_dir.resolve(),
        [Path(item["path"]) for item in current_jev["task_inputs"]],
        jev_run / "results.jsonl",
    )
    jev = verify_jev_run(
        jev_run,
        preflight["jevbench"],
        server,
        jev_task_rows,
        frozen_scores,
    )
    report = {
        "schema_version": "typical.r050_official_eval_verification.v1",
        "preflight_receipt_sha256": file_hash(preflight_path),
        "startup_receipt_sha256": file_hash(startup_receipt),
        "model_id": server["model_id"],
        "base_tree_sha256": server["base_model"]["artifact"]["tree_sha256"],
        "adapter_tree_sha256": server["adapter"]["artifact"]["tree_sha256"],
        "workflowevals": summaries,
        "workflow_parity": parity,
        "jevbench": jev,
        "integrity_pass": True,
    }
    report["receipt_sha256"] = _receipt_hash(report)
    return report


def verify_preserved_evidence(
    verification_receipt: Path,
    workflow_runs: list[Path],
    jev_run: Path,
) -> dict[str, Any]:
    """Recheck that every byte committed by a final receipt remains unchanged."""
    receipt = load_json(verification_receipt)
    require(isinstance(receipt, dict), "verification receipt must be an object")
    require(
        receipt.get("schema_version") == "typical.r050_official_eval_verification.v1",
        "wrong verification receipt schema",
    )
    require(receipt.get("receipt_sha256") == _receipt_hash(receipt), "verification receipt hash mismatch")
    expected_workflows = receipt.get("workflowevals")
    require(isinstance(expected_workflows, list), "verification receipt lacks workflow evidence")
    expected_by_name = {item.get("workflow"): item.get("evidence") for item in expected_workflows}
    require(len(expected_by_name) == len(expected_workflows), "duplicate workflow evidence in receipt")
    actual_by_name: dict[str, dict[str, Any]] = {}
    for directory in workflow_runs:
        result = load_json(directory / "results.json")
        name = result.get("eval_id") if isinstance(result, dict) else None
        require(isinstance(name, str) and name not in actual_by_name, "duplicate or absent workflow evidence")
        actual_by_name[name] = _workflow_evidence_identity(directory)
    require(set(actual_by_name) == set(expected_by_name), "preserved workflow scope mismatch")
    require(actual_by_name == expected_by_name, "WorkflowEvals evidence changed after verification")
    require(
        _jev_evidence_identity(jev_run) == receipt.get("jevbench", {}).get("evidence"),
        "JevBench evidence changed after verification",
    )
    report = {
        "schema_version": "typical.r050_evidence_preservation.v1",
        "verification_receipt_sha256": file_hash(verification_receipt),
        "evidence_preserved": True,
    }
    report["receipt_sha256"] = _receipt_hash(report)
    return report


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    except FileExistsError as error:
        raise VerificationError(f"refusing to overwrite immutable receipt: {path}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="freeze evidence before observing scores")
    preflight.add_argument("--startup-receipt", type=Path, required=True)
    preflight.add_argument("--workflowevals-dir", type=Path, required=True)
    preflight.add_argument("--workflow-patch", type=Path, required=True)
    preflight.add_argument("--jevbench-dir", type=Path, required=True)
    preflight.add_argument("--jev-runner", type=Path, required=True)
    preflight.add_argument("--task-file", type=Path, action="append", required=True)
    preflight.add_argument("--base-url", default="http://127.0.0.1:8090")
    preflight.add_argument("--timeout", type=float, default=1800)
    preflight.add_argument("--out", type=Path, required=True)
    postrun = subparsers.add_parser("postrun", help="verify complete evaluation evidence")
    postrun.add_argument("--preflight", type=Path, required=True)
    postrun.add_argument("--startup-receipt", type=Path, required=True)
    postrun.add_argument("--workflow-run", type=Path, action="append", required=True)
    postrun.add_argument("--workflowevals-dir", type=Path, required=True)
    postrun.add_argument("--jevbench-dir", type=Path, required=True)
    postrun.add_argument("--jev-runner", type=Path, required=True)
    postrun.add_argument("--jev-run", type=Path, required=True)
    postrun.add_argument("--out", type=Path, required=True)
    preserved = subparsers.add_parser("preserved", help="recheck evidence committed by a final receipt")
    preserved.add_argument("--verification", type=Path, required=True)
    preserved.add_argument("--workflow-run", type=Path, action="append", required=True)
    preserved.add_argument("--jev-run", type=Path, required=True)
    preserved.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            report = build_preflight(
                startup_receipt=args.startup_receipt,
                workflowevals_dir=args.workflowevals_dir,
                workflow_patch=args.workflow_patch,
                jevbench_dir=args.jevbench_dir,
                jev_runner=args.jev_runner,
                task_files=args.task_file,
                base_url=args.base_url,
                timeout_s=args.timeout,
            )
        elif args.command == "postrun":
            report = verify_postrun(
                preflight_path=args.preflight,
                startup_receipt=args.startup_receipt,
                workflow_runs=args.workflow_run,
                workflowevals_dir=args.workflowevals_dir,
                jevbench_dir=args.jevbench_dir,
                jev_runner=args.jev_runner,
                jev_run=args.jev_run,
            )
        else:
            report = verify_preserved_evidence(
                args.verification,
                args.workflow_run,
                args.jev_run,
            )
        _write_new(args.out, report)
    except VerificationError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
