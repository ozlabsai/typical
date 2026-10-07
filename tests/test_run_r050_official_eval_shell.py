from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_r050_official_eval.sh"
RUNBOOK = ROOT / "docs" / "research" / "R050_OFFICIAL_EVAL_RUNBOOK_20261007.md"


def _source() -> str:
    return SCRIPT.read_text()


def test_wrapper_is_executable_strict_and_syntax_valid() -> None:
    assert os.access(SCRIPT, os.X_OK)
    source = _source()
    assert "set -euo pipefail" in source
    assert "umask 077" in source
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_wait_mode_is_explicit_bounded_and_precedes_any_gpu_work() -> None:
    source = _source()
    assert "/workspace/g1/typical-eval-frozen" in source
    assert "R050_WAIT_FOR_COMPLETE" in source
    assert "R050_WAIT_TIMEOUT_SECONDS:-43200" in source
    assert "wait_for_pipeline_completion" in source
    wait_call = source.index("wait_for_pipeline_completion\n")
    assert wait_call < source.index('jq -er .status "$COMPLETE"')
    assert wait_call < source.index('mkdir "$EVAL_ROOT"')
    assert "pipeline coordinator is not active" in source
    assert "pipeline completed without an eligible winner" in source


def test_wrapper_holds_lock_and_claims_adapter_once() -> None:
    source = _source()
    assert 'mkdir "$EVAL_ROOT"' in source
    assert 'flock -n "$CLAIM_LOCK_FD"' in source
    assert 'mkdir "$ADAPTER_CLAIM"' in source
    assert source.index('flock -n "$CLAIM_LOCK_FD"') < source.index('mkdir "$ADAPTER_CLAIM"')
    assert source.index('preflight-receipt.json"') < source.index('mkdir "$ADAPTER_CLAIM"')
    assert source.index('mkdir "$ADAPTER_CLAIM"') < source.index("run_workflow security_incidents")
    assert "adapter-once-claim.txt" in source


def test_wrapper_binds_pipeline_training_completion() -> None:
    source = _source()
    assert '04-train-$WINNER' in source
    assert ".evidence.completion_sha256" in source
    assert 'sha256sum "$ARM_COMPLETION"' in source


def test_harness_is_warmed_before_server_and_readiness_is_bounded() -> None:
    source = _source()
    server = source.index("scripts/serve_typesafe_gemma_lora.py")
    assert source.index("python -m unittest discover -v") < server
    assert source.count("--dry-run") == 1
    assert source.index("--dry-run") < server
    assert "for _ in $(seq 1 120)" in source
    assert 'Authorization: Bearer $TYPICAL_API_KEY' in source


def test_scored_workflows_are_single_worker_and_logs_are_sealed() -> None:
    source = _source()
    for workflow in (
        "security_incidents",
        "agent_trace_observability",
        "invoice_processing",
        "customer_service",
    ):
        assert source.count(f"run_workflow {workflow} ") == 1
    assert '"$EVAL_ROOT/workflow-$workflow.log"' in source
    assert '--workers 1 --question-workers 1' in source
    assert '--questions all' in source


def test_server_shutdown_gpu_gate_and_distinct_preservation_receipts() -> None:
    source = _source()
    assert "kill -TERM" in source and "kill -KILL" in source
    assert source.index("wait_for_gpu_release\n\nlog") < source.rindex(
        "scripts/jevbench_gemma_lora.py"
    )
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "'.workflow_parity.workflow_parity_pass == true'" in source
    assert "preservation-before-archive-receipt.json" in source
    assert "preservation-after-archive-receipt.json" in source
    assert '"${archived_workflow_run_args[@]}"' in source


def test_runbook_uses_one_detached_wrapper_process() -> None:
    runbook = RUNBOOK.read_text()
    assert "scripts/run_r050_official_eval.sh" in runbook
    assert "tmux new-session -d" in runbook
    assert "single normative execution path" in runbook
    assert "/workspace/g1/typical-eval-frozen" in runbook
    assert "R050_WAIT_FOR_COMPLETE=1" in runbook
    assert "985a08e" not in runbook
    assert '= "$EXPECTED_TYPICAL_COMMIT"' in runbook
