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
    assert 'LOCAL_ROOT="${R050_LOCAL_ROOT:-/root/r050-eval}"' in source
    assert 'TYPICAL_REPO="${TYPICAL_REPO:-$LOCAL_ROOT/typical}"' in source
    assert "R050_WAIT_FOR_COMPLETE" in source
    assert "R050_WAIT_TIMEOUT_SECONDS:-86400" in source
    assert "wait_for_pipeline_completion" in source
    wait_call = source.index("wait_for_pipeline_completion\n")
    assert wait_call < source.index('jq -er .status "$COMPLETE"')
    assert wait_call < source.index('mkdir "$EVAL_ROOT"')
    assert "pipeline coordinator is not active" in source
    assert "pipeline completed without an eligible winner" in source


def test_reauthenticates_private_staging_after_pipeline_quiescence() -> None:
    source = _source()
    assert source.count("validate_staged_repositories") >= 3  # definition plus two calls
    quiescence = source.index("wait_for_pipeline_quiescence\n")
    second_validation = source.index("validate_staged_repositories\n", quiescence)
    assert quiescence < second_validation < source.index('jq -er .status "$COMPLETE"')
    assert "_workflow_patch_identity" in source
    assert "require_private_checkout" in source
    assert "group/other writable" in source


def test_wrapper_holds_lock_and_claims_adapter_once() -> None:
    source = _source()
    assert 'mkdir "$EVAL_ROOT"' in source
    assert 'flock -n "$CLAIM_LOCK_FD"' in source
    assert 'mkdir "$ADAPTER_CLAIM"' in source
    assert source.index('flock -n "$CLAIM_LOCK_FD"') < source.index('mkdir "$ADAPTER_CLAIM"')
    assert source.index('preflight-receipt.json"') < source.index('mkdir "$ADAPTER_CLAIM"')
    assert source.index('mkdir "$EVAL_ROOT"') < source.index('mkdir "$ADAPTER_CLAIM"')
    assert source.index('mkdir "$ADAPTER_CLAIM"') < source.index("run_workflow security_incidents")
    assert "adapter-once-claim.txt" in source
    assert 'CLAIM_ROOT="$R050_ROOT/.official-eval-claims"' in source


def test_preclaim_failures_are_private_recoverable_and_bounded() -> None:
    source = _source()
    assert "R050_PRECLAIM_MAX_ATTEMPTS:-3" in source
    assert "((max_attempts <= 3))" in source
    assert "mktemp -d" in source
    assert "failed-preclaim" in source
    assert "R050_ATTEMPT_CHILD" in source
    assert 'if [[ -e "$ADAPTER_CLAIM" ]]' in source
    assert "refusing to retry after adapter claim" in source
    assert source.index("preclaim attempt failed") < source.index("sleep \"$retry_delay\"")


def test_preclaim_cleanup_requires_exact_reservation_ownership(tmp_path: Path) -> None:
    source = _source()
    assert "R050_RESERVATION_TOKEN" in source
    assert '.typical-preclaim-reservation' in source
    ownership_check = source.index(
        '"$(cat "$EVAL_ROOT/.typical-preclaim-reservation")" != "$reservation_token"'
    )
    removal = source.index('rm "$EVAL_ROOT/.typical-preclaim-reservation"')
    assert ownership_check < removal
    assert "refusing to remove evaluation root not owned by this preclaim attempt" in source

    start = source.index("retain_failed_preclaim() {")
    end = source.index("\npublish_completed_attempt() {", start)
    function_source = source[start:end]
    subprocess.run(
        [
            "bash",
            "-c",
            f"""
set -u
LOCAL_ROOT="$1/local"
EVAL_ROOT="$1/eval"
mkdir -p "$LOCAL_ROOT" "$EVAL_ROOT" "$1/attempt-foreign"
printf '%s\n' foreign >"$EVAL_ROOT/.typical-preclaim-reservation"
log() {{ :; }}
{function_source}
if retain_failed_preclaim "$1/attempt-foreign" 1 owned; then
  exit 20
fi
test -d "$EVAL_ROOT"
test "$(cat "$EVAL_ROOT/.typical-preclaim-reservation")" = foreign

mkdir "$1/attempt-owned"
printf '%s\n' owned >"$EVAL_ROOT/.typical-preclaim-reservation"
retain_failed_preclaim "$1/attempt-owned" 2 owned
test ! -e "$EVAL_ROOT"
""",
            "bash",
            str(tmp_path),
        ],
        check=True,
    )


def test_postclaim_recovery_republishes_only_verified_evidence() -> None:
    source = _source()
    assert "R050_PUBLISH_ONLY_ATTEMPT_ROOT" in source
    assert "publish_with_retries" in source
    branch = source.index('if [[ -n "${R050_PUBLISH_ONLY_ATTEMPT_ROOT:-}" ]]')
    claim = source.index('test -d "$ADAPTER_CLAIM"', branch)
    publish = source.index('publish_with_retries "$ATTEMPT_ROOT"', claim)
    scored = source.index("run_workflow security_incidents")
    assert branch < claim < publish < scored
    assert "scored work will not rerun" in source


def test_publication_retry_is_bounded_and_preserves_failure_status() -> None:
    source = _source()
    start = source.index("publish_with_retries() {")
    end = source.index('\nif [[ -n "${R050_PUBLISH_ONLY_ATTEMPT_ROOT:-}" ]]', start)
    function_source = source[start:end]
    subprocess.run(
        [
            "bash",
            "-c",
            f"""
set -u
R050_PUBLICATION_MAX_ATTEMPTS=3
R050_PUBLICATION_RETRY_SECONDS=1
calls=0
log() {{ :; }}
sleep() {{ :; }}
publish_completed_attempt() {{
  calls=$((calls + 1))
  if ((calls == 3)); then return 0; fi
  return 7
}}
{function_source}
publish_with_retries ignored
test "$calls" = 3

calls=0
publish_completed_attempt() {{ calls=$((calls + 1)); return 7; }}
if publish_with_retries ignored; then exit 20; else status=$?; fi
test "$status" = 7
test "$calls" = 3

R050_PUBLICATION_MAX_ATTEMPTS=4
if publish_with_retries ignored; then exit 21; else status=$?; fi
test "$status" = 2
""",
        ],
        check=True,
    )


def test_frozen_input_manifest_is_path_independent() -> None:
    source = _source()
    assert 'cp "$TYPICAL_REPO/uv.lock" "$ATTEMPT_ROOT/typical-uv.lock"' in source
    manifest = source.index("sha256sum r050-state.json")
    assert source.rfind('cd "$ATTEMPT_ROOT"', 0, manifest) != -1
    assert '"$TYPICAL_REPO/uv.lock" \\\n  >"$ATTEMPT_ROOT/frozen-input-sha256.txt"' not in source


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
    assert '"$ATTEMPT_ROOT/workflow-$workflow.log"' in source
    assert '"$EVAL_ROOT/workflow-$workflow.log"' not in source
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


def test_persistent_root_is_populated_only_after_scoring_and_verification() -> None:
    source = _source()
    assert 'bash "$SCRIPT_PATH"; then\n      RESERVATION_TOKEN="$reservation_token"' in source
    ready = source.index('>"$ATTEMPT_ROOT/official-eval-ready.json.tmp"')
    verification = source.index("scripts/verify_r050_official_eval.py postrun")
    after_archive = source.index("preservation-after-archive-receipt.json", verification)
    assert verification < after_archive < ready
    assert 'cp -a "$ATTEMPT_ROOT/." "$EVAL_ROOT/"' in source
    assert "R050_LAUNCH_LOG" in source
    assert 'cp "$R050_LAUNCH_LOG" "$EVAL_ROOT/launch.log"' in source
    assert 'sha256sum "$EVAL_ROOT/launch.log"' in source
    for field in (
        "verification_receipt_sha256",
        "preservation_before_archive_sha256",
        "preservation_after_archive_sha256",
    ):
        assert field in source
    validation = source.index('for binding in \\\n')
    publication = source.index('cp -a "$ATTEMPT_ROOT/." "$EVAL_ROOT/"')
    assert validation < publication


def test_runbook_uses_one_detached_wrapper_process() -> None:
    runbook = RUNBOOK.read_text()
    assert "scripts/run_r050_official_eval.sh" in runbook
    assert "tmux new-session -d" in runbook
    assert "single normative execution path" in runbook
    assert "/root/r050-eval/typical" in runbook
    assert "/root/r050-eval/WorkflowEvals" in runbook
    assert "/root/r050-eval/jevbench" in runbook
    assert "R050_WAIT_FOR_COMPLETE=1" in runbook
    assert "985a08e" not in runbook
    assert '= "$EXPECTED_TYPICAL_COMMIT"' in runbook
    assert "chmod -R go-w" in runbook
    assert 'umask 077; exec env' in runbook
    assert "/root/r050-eval/r050-official-eval.launch.log" in runbook
