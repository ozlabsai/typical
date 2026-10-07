#!/usr/bin/env bash
# Execute the frozen R050 external evaluation as one fail-closed, unattended attempt.

set -euo pipefail
umask 077
((BASH_VERSINFO[0] >= 4)) || {
  printf 'bash 4 or newer is required\n' >&2
  exit 1
}

: "${EXPECTED_TYPICAL_COMMIT:?set EXPECTED_TYPICAL_COMMIT to the reviewed staged commit}"
[[ "$EXPECTED_TYPICAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'EXPECTED_TYPICAL_COMMIT must be a lowercase 40-character commit\n' >&2
  exit 1
}

LOCAL_ROOT="${R050_LOCAL_ROOT:-/root/r050-eval}"
TYPICAL_REPO="${TYPICAL_REPO:-$LOCAL_ROOT/typical}"
R050_ROOT="${R050_ROOT:-/workspace/g1/r050-ab512c3/repo/.context/r050-gpu-run-a77fea5-resume2}"
EVAL_ROOT="${EVAL_ROOT:-/workspace/g1/r050-official-eval-20261007}"
WORKFLOWEVALS="${WORKFLOWEVALS:-$LOCAL_ROOT/WorkflowEvals}"
JEVBENCH="${JEVBENCH:-$LOCAL_ROOT/jevbench}"
R050_LAUNCH_LOG="${R050_LAUNCH_LOG:-$LOCAL_ROOT/r050-official-eval.launch.log}"
SERVER_PORT="${R050_SERVER_PORT:-8090}"
TYPICAL_BASE_URL="http://127.0.0.1:${SERVER_PORT}"

WORKFLOWEVALS_COMMIT="0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb"
JEVBENCH_COMMIT="bb05a335bc809e61b20c0f745d25499a82b326fc"
WORKFLOW_PATCH_SHA256="07106bf888aa53a2770b83503c58635708db32042ecb4b034a41963b1ba43908"
TARGET_URL="https://benchmarkheaven.com/api/jevbench/v1.5.4"

SERVER_PID=""

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

server_group_alive() {
  [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null
}

stop_server() {
  if ! server_group_alive; then
    SERVER_PID=""
    return
  fi
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! server_group_alive; then
      wait "$SERVER_PID" 2>/dev/null || true
      SERVER_PID=""
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

wait_for_gpu_release() {
  local pids
  for _ in $(seq 1 120); do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    if [[ -z "$pids" ]]; then
      return 0
    fi
    sleep 5
  done
  printf 'GPU still has compute owners after bounded wait\n' >&2
  return 1
}

cleanup() {
  local status=$?
  stop_server
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_command() {
  command -v "$1" >/dev/null || {
    printf 'required command is absent: %s\n' "$1" >&2
    exit 1
  }
}

require_private_path() {
  local path=$1
  local mode
  mode="$(stat -c '%a' "$path")"
  if ((8#$mode & 8#022)); then
    printf 'code/evidence staging path is group/other writable: %s (mode %s)\n' \
      "$path" "$mode" >&2
    return 1
  fi
}

require_private_checkout() {
  local checkout=$1
  local listing
  local status=0
  require_private_path "$checkout"
  listing="$(mktemp "$LOCAL_ROOT/.permission-scan.XXXXXX")"
  if ! find "$checkout" -xdev \
    \( -path "$checkout/.git" -o -path "$checkout/.venv" \
       -o -path "$checkout/.pytest_cache" -o -path '*/__pycache__' \
       -o -path "$checkout/runs" \) -prune -o \
    \( -type f -o -type d \) -print0 >"$listing"; then
    rm -f "$listing"
    return 1
  fi
  while IFS= read -r -d '' path; do
    require_private_path "$path" || status=1
  done <"$listing"
  rm -f "$listing"
  return "$status"
}

validate_workflow_patch() {
  cd "$TYPICAL_REPO"
  uv run --frozen python - "$WORKFLOWEVALS" "$PATCH" >/dev/null <<'PY'
import sys
from pathlib import Path

from scripts.verify_r050_official_eval import _workflow_patch_identity

_workflow_patch_identity(Path(sys.argv[1]), Path(sys.argv[2]))
PY
}

validate_staged_repositories() {
  test "$(git -C "$TYPICAL_REPO" rev-parse HEAD)" = "$EXPECTED_TYPICAL_COMMIT"
  test -z "$(git -C "$TYPICAL_REPO" status --porcelain=v1 --untracked-files=all)"
  test "$(git -C "$WORKFLOWEVALS" rev-parse HEAD)" = "$WORKFLOWEVALS_COMMIT"
  test "$(git -C "$JEVBENCH" rev-parse HEAD)" = "$JEVBENCH_COMMIT"
  test -z "$(git -C "$JEVBENCH" status --porcelain=v1 --untracked-files=all)"
  test "$(sha256sum "$PATCH" | cut -d' ' -f1)" = "$WORKFLOW_PATCH_SHA256"
  validate_workflow_patch
  require_private_checkout "$TYPICAL_REPO"
  require_private_checkout "$WORKFLOWEVALS"
  require_private_checkout "$JEVBENCH"
  require_private_path "$R050_LAUNCH_LOG"
}

pipeline_lock_free() {
  exec 8>>"$R050_ROOT/.pipeline.lock"
  if flock -n 8; then
    flock -u 8
    exec 8>&-
    return 0
  fi
  exec 8>&-
  return 1
}

wait_for_pipeline_completion() {
  local wait_mode="${R050_WAIT_FOR_COMPLETE:-0}"
  local timeout="${R050_WAIT_TIMEOUT_SECONDS:-86400}"
  local interval="${R050_WAIT_INTERVAL_SECONDS:-60}"
  [[ "$wait_mode" = 0 || "$wait_mode" = 1 ]]
  [[ "$timeout" =~ ^[1-9][0-9]*$ ]]
  [[ "$interval" =~ ^[1-9][0-9]*$ ]]

  if [[ -e "$COMPLETE" ]]; then
    jq -e 'type == "object" and has("status") and has("winner")' "$COMPLETE" >/dev/null
    return
  fi
  if [[ "$wait_mode" != 1 ]]; then
    printf 'pipeline completion is absent; set R050_WAIT_FOR_COMPLETE=1 to wait\n' >&2
    return 1
  fi

  local deadline=$((SECONDS + timeout))
  local remaining
  local delay
  log "waiting up to $timeout seconds for the pipeline completion marker"
  while ((SECONDS < deadline)); do
    if [[ -e "$COMPLETE" ]]; then
      jq -e 'type == "object" and has("status") and has("winner")' "$COMPLETE" >/dev/null
      log "pipeline completion marker published"
      return
    fi
    if pipeline_lock_free; then
      # Close the race where completion was published just before the lock was released.
      if [[ -e "$COMPLETE" ]]; then
        jq -e 'type == "object" and has("status") and has("winner")' "$COMPLETE" >/dev/null
        log "pipeline completion marker published"
        return
      fi
      printf 'pipeline coordinator is not active and completion is absent\n' >&2
      return 1
    fi
    remaining=$((deadline - SECONDS))
    delay=$interval
    if ((delay > remaining)); then
      delay=$remaining
    fi
    sleep "$delay"
  done
  printf 'timed out waiting for pipeline completion after %s seconds\n' "$timeout" >&2
  return 1
}

wait_for_pipeline_quiescence() {
  for _ in $(seq 1 120); do
    if pipeline_lock_free; then
      return
    fi
    sleep 1
  done
  printf 'pipeline completion exists but coordinator lock was not released\n' >&2
  return 1
}

for command in curl find flock git jq mktemp nvidia-smi openssl setsid sha256sum stat uv; do
  require_command "$command"
done

for directory in "$LOCAL_ROOT" "$TYPICAL_REPO" "$R050_ROOT" "$WORKFLOWEVALS" "$JEVBENCH"; do
  [[ "$directory" = /* ]] && [[ -d "$directory" ]] || {
    printf 'required absolute directory is absent: %s\n' "$directory" >&2
    exit 1
  }
done
[[ "$EVAL_ROOT" = /* ]] || {
  printf 'EVAL_ROOT must be absolute\n' >&2
  exit 1
}
[[ "$R050_LAUNCH_LOG" = "$LOCAL_ROOT/"* ]] && [[ -f "$R050_LAUNCH_LOG" ]] || {
  printf 'R050_LAUNCH_LOG must be an existing file below the private local root\n' >&2
  exit 1
}

STATE="$R050_ROOT/state.json"
COMPLETE="$R050_ROOT/complete.json"
PATCH="$TYPICAL_REPO/patches/workflowevals-typical-provider-0ac3b8ad.patch"

log "validating frozen staged repositories"
validate_staged_repositories

wait_for_pipeline_completion
wait_for_pipeline_quiescence
log "re-authenticating staged repositories after pipeline quiescence"
validate_staged_repositories

log "validating frozen pipeline completion"
PIPELINE_STATUS="$(jq -er .status "$COMPLETE")"
WINNER="$(jq -r '.winner // empty' "$COMPLETE")"
if [[ "$PIPELINE_STATUS" != complete || -z "$WINNER" ]]; then
  log "pipeline completed without an eligible winner; no GPU evaluation will run"
  exit 2
fi
test "$(jq -c .official_evaluations_run "$COMPLETE")" = '[]'
jq -e --arg winner "$WINNER" '.eligible_candidates | index($winner) != null' "$COMPLETE" >/dev/null
ARM_COMPLETION="$R050_ROOT/runs/$WINNER/completion.json"
ADAPTER="$(jq -er .artifacts.checkpoint.path "$ARM_COMPLETION")"
ADAPTER_SHA="$(jq -er .artifacts.checkpoint.tree_sha256 "$ARM_COMPLETION")"
ADAPTER_FILES="$(jq -er .artifacts.checkpoint.files "$ARM_COMPLETION")"
[[ "$ADAPTER_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$ADAPTER_FILES" =~ ^[1-9][0-9]*$ ]]
EXPECTED_COMPLETION_SHA="$(
  jq -er --arg stage "04-train-$WINNER" \
    '.stages[$stage] | select(.status == "complete") | .evidence.completion_sha256' "$STATE"
)"
test "$(sha256sum "$ARM_COMPLETION" | cut -d' ' -f1)" = "$EXPECTED_COMPLETION_SHA"

BASE_MODEL="$(jq -er .binding.paths.model "$STATE")"
MODEL_INVENTORY="$(jq -er .binding.paths.model_inventory "$STATE")"
MODEL_INVENTORY_SHA="$(jq -er .binding.hashes.model_inventory "$STATE")"
BASE_MODEL_SHA="$(jq -er .tree_sha256 "$MODEL_INVENTORY")"
BASE_REVISION="$(jq -er .revision "$MODEL_INVENTORY")"
MODEL_ID="typical-r050-sha256-${ADAPTER_SHA:0:16}"

# The once-only authority is tied to the immutable pipeline root, independent
# of any operator-selected evidence destination.
CLAIM_ROOT="$R050_ROOT/.official-eval-claims"
mkdir -p "$CLAIM_ROOT"
CLAIM_LOCK="$CLAIM_ROOT/$ADAPTER_SHA.lock"
ADAPTER_CLAIM="$CLAIM_ROOT/$ADAPTER_SHA.claimed"

retain_failed_preclaim() {
  local attempt_root=$1
  local attempt_number=$2
  local reservation_token=$3
  local stamp
  local destination
  stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$-a$attempt_number"
  destination="$LOCAL_ROOT/failed-preclaim/$stamp"
  mkdir -p "$destination"
  if [[ -d "$attempt_root" ]]; then
    mv "$attempt_root" "$destination/evidence"
  fi
  if [[ -e "$EVAL_ROOT" ]]; then
    if [[ ! -f "$EVAL_ROOT/.typical-preclaim-reservation" ]] || \
      [[ "$(cat "$EVAL_ROOT/.typical-preclaim-reservation")" != "$reservation_token" ]]; then
      printf 'refusing to remove evaluation root not owned by this preclaim attempt: %s\n' \
        "$EVAL_ROOT" >&2
      return 1
    fi
    cp "$EVAL_ROOT/.typical-preclaim-reservation" "$destination/reservation-token.txt"
    rm "$EVAL_ROOT/.typical-preclaim-reservation"
    rmdir "$EVAL_ROOT"
  fi
  log "preclaim attempt failed; retained evidence at $destination"
}

publish_completed_attempt() {
  local ATTEMPT_ROOT=$1
  local ready="$ATTEMPT_ROOT/official-eval-ready.json"
  local adapter_sha
  local actual_sha
  local binding
  local field
  local filename
  local run_tag
  test -f "$ready" || return
  test -d "$ADAPTER_CLAIM" || return
  test -f "$EVAL_ROOT/.typical-preclaim-reservation" || return
  test "$(cat "$EVAL_ROOT/.typical-preclaim-reservation")" = "$RESERVATION_TOKEN" || return
  jq -e --arg expected "$ADAPTER_SHA" '
    (keys | sort) == ["adapter_tree_sha256", "integrity_pass", "preservation_after_archive_sha256", "preservation_before_archive_sha256", "run_tag", "schema_version", "status", "verification_receipt_sha256", "workflow_parity_pass"]
    and .schema_version == "typical.r050_official_eval_ready.v1"
    and .status == "ready_for_persistent_publication"
    and .adapter_tree_sha256 == $expected
    and (.run_tag | type == "string" and length > 0)
    and (.verification_receipt_sha256 | test("^[0-9a-f]{64}$"))
    and (.preservation_before_archive_sha256 | test("^[0-9a-f]{64}$"))
    and (.preservation_after_archive_sha256 | test("^[0-9a-f]{64}$"))
    and .integrity_pass == true
    and .workflow_parity_pass == true
  ' "$ready" >/dev/null || return
  for binding in \
    "verification_receipt_sha256:verification-receipt.json" \
    "preservation_before_archive_sha256:preservation-before-archive-receipt.json" \
    "preservation_after_archive_sha256:preservation-after-archive-receipt.json"; do
    field="${binding%%:*}"
    filename="${binding#*:}"
    actual_sha="$(sha256sum "$ATTEMPT_ROOT/$filename" | cut -d' ' -f1)" || return
    test "$(jq -er --arg field "$field" '.[$field]' "$ready")" = "$actual_sha" || return
  done
  adapter_sha="$(jq -er .adapter_tree_sha256 "$ready")" || return
  run_tag="$(jq -er .run_tag "$ready")" || return
  test "$(sed -n '1p' "$ADAPTER_CLAIM/attempt.txt")" = "$EVAL_ROOT" || return
  test "$(sed -n '2p' "$ADAPTER_CLAIM/attempt.txt")" = "$run_tag" || return
  test "$(sed -n '4p' "$ADAPTER_CLAIM/attempt.txt")" = "$RESERVATION_TOKEN" || return
  log "publishing completed evidence to the persistent evaluation root"
  cp -a "$ATTEMPT_ROOT/." "$EVAL_ROOT/" || return
  cp "$R050_LAUNCH_LOG" "$EVAL_ROOT/launch.log" || return
  sha256sum "$EVAL_ROOT/launch.log" >"$EVAL_ROOT/launch-log-sha256.txt" || return
  (
    set -e
    cd "$EVAL_ROOT"
    find . -type f ! -name archive-sha256.txt \
      ! -name official-eval-complete.json ! -name official-eval-complete.json.tmp -print0 \
      | sort -z \
      | xargs -0 sha256sum >archive-sha256.txt
  ) || return
  jq -n --arg adapter_sha "$adapter_sha" --arg run_tag "$run_tag" \
    '{schema_version:"typical.r050_official_eval_completion.v1",status:"complete",adapter_tree_sha256:$adapter_sha,run_tag:$run_tag,integrity_pass:true,workflow_parity_pass:true}' \
    >"$EVAL_ROOT/official-eval-complete.json.tmp" || return
  mv "$EVAL_ROOT/official-eval-complete.json.tmp" \
    "$EVAL_ROOT/official-eval-complete.json" || return
}

publish_with_retries() {
  local attempt_root=$1
  local max_attempts="${R050_PUBLICATION_MAX_ATTEMPTS:-3}"
  local retry_delay="${R050_PUBLICATION_RETRY_SECONDS:-30}"
  local attempt
  local status=1
  [[ "$max_attempts" =~ ^[1-3]$ ]] || return 2
  [[ "$retry_delay" =~ ^[1-9][0-9]*$ ]] || return 2
  for attempt in $(seq 1 "$max_attempts"); do
    if publish_completed_attempt "$attempt_root"; then
      return 0
    else
      status=$?
    fi
    log "publication attempt $attempt of $max_attempts failed; scored work will not rerun"
    if ((attempt < max_attempts)); then
      sleep "$retry_delay"
    fi
  done
  return "$status"
}

if [[ -n "${R050_PUBLISH_ONLY_ATTEMPT_ROOT:-}" ]]; then
  ATTEMPT_ROOT="$(cd "$R050_PUBLISH_ONLY_ATTEMPT_ROOT" && pwd -P)"
  [[ "$ATTEMPT_ROOT" = "$LOCAL_ROOT/attempts/"* ]] && [[ -d "$ATTEMPT_ROOT" ]]
  exec {CLAIM_LOCK_FD}>>"$CLAIM_LOCK"
  flock -n "$CLAIM_LOCK_FD"
  test -d "$ADAPTER_CLAIM"
  RESERVATION_TOKEN="$(sed -n '4p' "$ADAPTER_CLAIM/attempt.txt")"
  [[ "$RESERVATION_TOKEN" =~ ^[0-9a-f]{64}$ ]]
  publish_with_retries "$ATTEMPT_ROOT"
  exit 0
fi

if [[ "${R050_ATTEMPT_CHILD:-0}" != 1 ]]; then
  max_attempts="${R050_PRECLAIM_MAX_ATTEMPTS:-3}"
  retry_delay="${R050_PRECLAIM_RETRY_SECONDS:-60}"
  [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]
  ((max_attempts <= 3))
  [[ "$retry_delay" =~ ^[1-9][0-9]*$ ]]
  test ! -e "$EVAL_ROOT"
  exec {CLAIM_LOCK_FD}>>"$CLAIM_LOCK"
  flock -n "$CLAIM_LOCK_FD"
  test ! -e "$ADAPTER_CLAIM"
  mkdir -p "$LOCAL_ROOT/attempts" "$LOCAL_ROOT/failed-preclaim"
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  for attempt_number in $(seq 1 "$max_attempts"); do
    attempt_root="$(mktemp -d "$LOCAL_ROOT/attempts/r050-preclaim.XXXXXX")"
    reservation_token="$(openssl rand -hex 32)"
    log "starting private preclaim attempt $attempt_number of $max_attempts"
    if env \
      EXPECTED_TYPICAL_COMMIT="$EXPECTED_TYPICAL_COMMIT" \
      R050_ATTEMPT_CHILD=1 \
      R050_ATTEMPT_ROOT="$attempt_root" \
      R050_CLAIM_LOCK_HELD=1 \
      R050_PRECLAIM_ATTEMPT="$attempt_number" \
      R050_RESERVATION_TOKEN="$reservation_token" \
      R050_LOCAL_ROOT="$LOCAL_ROOT" \
      R050_LAUNCH_LOG="$R050_LAUNCH_LOG" \
      R050_ROOT="$R050_ROOT" \
      EVAL_ROOT="$EVAL_ROOT" \
      TYPICAL_REPO="$TYPICAL_REPO" \
      WORKFLOWEVALS="$WORKFLOWEVALS" \
      JEVBENCH="$JEVBENCH" \
      R050_SERVER_PORT="$SERVER_PORT" \
      bash "$SCRIPT_PATH"; then
      RESERVATION_TOKEN="$reservation_token"
      if publish_with_retries "$attempt_root"; then
        exit 0
      else
        status=$?
      fi
      printf 'verified scored evidence remains private at %s; resume with R050_PUBLISH_ONLY_ATTEMPT_ROOT\n' \
        "$attempt_root" >&2
      exit "$status"
    else
      status=$?
    fi
    if [[ -e "$ADAPTER_CLAIM" ]]; then
      printf 'refusing to retry after adapter claim; private evidence remains at %s\n' \
        "$attempt_root" >&2
      exit "$status"
    fi
    retain_failed_preclaim "$attempt_root" "$attempt_number" "$reservation_token"
    if ((attempt_number == max_attempts)); then
      printf 'all %s bounded preclaim attempts failed\n' "$max_attempts" >&2
      exit "$status"
    fi
    sleep "$retry_delay"
  done
fi

[[ "${R050_CLAIM_LOCK_HELD:-0}" = 1 ]]
ATTEMPT_ROOT="${R050_ATTEMPT_ROOT:?child attempt root is required}"
RESERVATION_TOKEN="${R050_RESERVATION_TOKEN:?child reservation token is required}"
[[ "$RESERVATION_TOKEN" =~ ^[0-9a-f]{64}$ ]]
[[ "$ATTEMPT_ROOT" = "$LOCAL_ROOT/attempts/"* ]] && [[ -d "$ATTEMPT_ROOT" ]]
test ! -e "$ADAPTER_CLAIM"
test ! -e "$EVAL_ROOT"

cp "$STATE" "$ATTEMPT_ROOT/r050-state.json"
cp "$COMPLETE" "$ATTEMPT_ROOT/r050-complete.json"
cp "$ARM_COMPLETION" "$ATTEMPT_ROOT/r050-winner-completion.json"
cp "$TYPICAL_REPO/uv.lock" "$ATTEMPT_ROOT/typical-uv.lock"
git -C "$TYPICAL_REPO" rev-parse HEAD >"$ATTEMPT_ROOT/typical-commit.txt"
(
  cd "$ATTEMPT_ROOT"
  sha256sum r050-state.json r050-complete.json r050-winner-completion.json \
    typical-uv.lock >frozen-input-sha256.txt
)

log "authenticating local model and winner adapter"
cd "$TYPICAL_REPO"
uv run --frozen python - "$MODEL_INVENTORY" "$BASE_MODEL" \
  "$MODEL_INVENTORY_SHA" "$ADAPTER" "$ADAPTER_SHA" "$ADAPTER_FILES" \
  >"$ATTEMPT_ROOT/artifact-validation.log" 2>&1 <<'PY'
import sys
from pathlib import Path

from scripts.build_r047_model_inventory import verify_inventory
from scripts.preflight_r043_assets import tree_hash

inventory, inventory_sha = verify_inventory(Path(sys.argv[1]), Path(sys.argv[2]))
assert inventory_sha == sys.argv[3]
adapter_sha, adapter_files = tree_hash(Path(sys.argv[4]))
assert adapter_sha == sys.argv[5]
assert adapter_files == int(sys.argv[6])
assert inventory["tree_sha256"]
PY

log "freezing the released external target"
curl --fail --location --silent --show-error "$TARGET_URL" \
  --retry 4 --retry-all-errors --retry-delay 5 \
  --output "$ATTEMPT_ROOT/jevbench-v1.5.4-target.json"
uv run --frozen python - "$ATTEMPT_ROOT/jevbench-v1.5.4-target.json" \
  >"$ATTEMPT_ROOT/target-validation.log" 2>&1 <<'PY'
import json
import math
import sys
from pathlib import Path

target = json.loads(Path(sys.argv[1]).read_text())
assert (target["revision"], target["protocol"], target["status"], target["run_kind"]) == (
    "v1.5.4", "jevbench::v1.5", "released", "official"
)
assert target["source_sha256"] == "452885de2a84cd5b9ed393d541fd8d6c6a540f9d7f762f9d2df9349383f26173"
assert target["method_sha256"] == "c25d3d8b8512e4d93370a9e0c99705d19b2a9389956ca33b8a4bd2b0ec501c07"
assert target["headline_method_addendum_sha256"] == "752ddccc4e191c81412631ee3121563421c4b9077cb6c3f2243ed1d1545cfab0"
assert target["sample"] == {"open": 904, "sealed": 720, "total": 1624, "published_open": 601}
assert target["headline"] == "A" and set(target["types"].values()) == {1 / 3}
jev = next(row for row in target["systems"] if row["key"] == "jev-1.13.0")
assert jev["status"] == {"status": "complete", "rows": 1624, "missing": 0, "answered_ok": 1624}
assert math.isclose(jev["jevbench_score"], 72.13292897895896, rel_tol=0, abs_tol=1e-12)
assert math.isclose(jev["axes"]["intelligence"], 71.99974990450261, rel_tol=0, abs_tol=1e-12)
PY

log "verifying and warming the patched WorkflowEvals harness"
uv run --frozen python - "$WORKFLOWEVALS" "$PATCH" \
  >"$ATTEMPT_ROOT/workflow-patch-validation.log" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

from scripts.verify_r050_official_eval import _workflow_patch_identity

print(json.dumps(_workflow_patch_identity(Path(sys.argv[1]), Path(sys.argv[2])), sort_keys=True))
PY

cd "$WORKFLOWEVALS"
uv lock --check
uv run --frozen python -m unittest discover -v >"$ATTEMPT_ROOT/workflow-tests.log" 2>&1

TYPICAL_API_KEY="$(openssl rand -hex 32)"
export TYPICAL_API_KEY TYPICAL_BASE_URL
RUN_TAG="typical-r050-${ADAPTER_SHA:0:16}-$(date -u +%Y%m%dT%H%M%SZ)-a${R050_PRECLAIM_ATTEMPT}"
printf '%s\n' "$RUN_TAG" >"$ATTEMPT_ROOT/run-tag.txt"

execute_workflow() {
  local workflow=$1
  local revision=$2
  local dry_run=$3
  local log_file=$4
  local -a invocation=(uv run --frozen python run.py "$workflow" \
    --model "typical:$MODEL_ID" --name "$RUN_TAG" \
    --base-url "$TYPICAL_BASE_URL" --workers 1 --question-workers 1 \
    --questions all --timeout 1800 --dataset-revision "$revision")
  if [[ "$dry_run" = true ]]; then
    "${invocation[@]}" --dry-run >>"$log_file" 2>&1
  else
    "${invocation[@]}" >"$log_file" 2>&1
  fi
}

while IFS= read -r spec; do
  read -r workflow revision <<<"$spec"
  test ! -e "$WORKFLOWEVALS/runs/$workflow/$RUN_TAG"
  execute_workflow "$workflow" "$revision" true "$ATTEMPT_ROOT/workflow-dry-run.log"
done <<'SPECS'
security_incidents fbe1ea5c69cf494157fd23f2002a0d9d9a418443
agent_trace_observability 8635540973910a92465fe2bc53e195375aa6e1a8
invoice_processing 6beeb2d2acd65c086c835022f5f4d7434114cafc
customer_service b1342f5a704587dbc465867c38c2694348ff86e4
SPECS

log "starting the authenticated frozen model server"
cd "$TYPICAL_REPO"
wait_for_gpu_release
uv run --frozen python - "$SERVER_PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY

setsid uv run --frozen python scripts/serve_typesafe_gemma_lora.py \
  --model "$BASE_MODEL" \
  --model-revision "$BASE_REVISION" \
  --expected-model-tree-sha256 "$BASE_MODEL_SHA" \
  --adapter "$ADAPTER" \
  --expected-adapter-tree-sha256 "$ADAPTER_SHA" \
  --model-id "$MODEL_ID" \
  --receipt "$ATTEMPT_ROOT/server-startup-receipt.json" \
  --max-context 16384 \
  --dtype bfloat16 \
  --device cuda \
  --attn-implementation sdpa \
  --host 127.0.0.1 \
  --port "$SERVER_PORT" >"$ATTEMPT_ROOT/server.log" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 120); do
  if ! server_group_alive; then
    printf 'model server exited before readiness\n' >&2
    exit 1
  fi
  if [[ -s "$ATTEMPT_ROOT/server-startup-receipt.json" ]] && \
    curl --fail --silent --show-error \
      --header "Authorization: Bearer $TYPICAL_API_KEY" \
      "$TYPICAL_BASE_URL/v1/models" \
      --output "$ATTEMPT_ROOT/server-models.tmp.json"; then
    READY=1
    break
  fi
  sleep 5
done
test "$READY" = 1
jq -e --arg model "$MODEL_ID" \
  '.models | length == 1 and .[0].name == $model' \
  "$ATTEMPT_ROOT/server-models.tmp.json" >/dev/null
mv "$ATTEMPT_ROOT/server-models.tmp.json" "$ATTEMPT_ROOT/server-models.json"

log "freezing preflight identities"
uv run --frozen python scripts/verify_r050_official_eval.py preflight \
  --startup-receipt "$ATTEMPT_ROOT/server-startup-receipt.json" \
  --workflowevals-dir "$WORKFLOWEVALS" \
  --workflow-patch "$PATCH" \
  --jevbench-dir "$JEVBENCH" \
  --jev-runner "$TYPICAL_REPO/scripts/jevbench_gemma_lora.py" \
  --task-file "$JEVBENCH/datasets/public/easy.jsonl" \
  --task-file "$JEVBENCH/datasets/public/original.jsonl" \
  --task-file "$JEVBENCH/datasets/public/hard.jsonl" \
  --base-url "$TYPICAL_BASE_URL" \
  --timeout 1800 \
  --out "$ATTEMPT_ROOT/preflight-receipt.json" \
  >"$ATTEMPT_ROOT/preflight.log" 2>&1

# This atomic directory is the irreversible exactly-once boundary. It is made
# only after every non-scored preparation passes and immediately before the
# first scored request.
mkdir "$EVAL_ROOT"
printf '%s\n' "$RESERVATION_TOKEN" >"$EVAL_ROOT/.typical-preclaim-reservation.tmp"
mv "$EVAL_ROOT/.typical-preclaim-reservation.tmp" \
  "$EVAL_ROOT/.typical-preclaim-reservation"
mkdir "$ADAPTER_CLAIM"
printf '%s\n%s\n%s\n%s\n' "$EVAL_ROOT" "$RUN_TAG" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RESERVATION_TOKEN" \
  >"$ADAPTER_CLAIM/attempt.txt"
cp "$ADAPTER_CLAIM/attempt.txt" "$ATTEMPT_ROOT/adapter-once-claim.txt"

run_workflow() {
  local workflow=$1
  local revision=$2
  execute_workflow "$workflow" "$revision" false "$ATTEMPT_ROOT/workflow-$workflow.log"
}

log "running four sealed WorkflowEvals workflows exactly once"
cd "$WORKFLOWEVALS"
run_workflow security_incidents fbe1ea5c69cf494157fd23f2002a0d9d9a418443
run_workflow agent_trace_observability 8635540973910a92465fe2bc53e195375aa6e1a8
run_workflow invoice_processing 6beeb2d2acd65c086c835022f5f4d7434114cafc
run_workflow customer_service b1342f5a704587dbc465867c38c2694348ff86e4

log "stopping model server and waiting for complete GPU release"
stop_server
unset TYPICAL_API_KEY

wait_for_gpu_release

log "running the frozen 231-row public JevBench diagnostic"
cd "$TYPICAL_REPO"
test ! -e "$ATTEMPT_ROOT/jevbench-public-231"
uv run --frozen python scripts/jevbench_gemma_lora.py \
  --jevbench-dir "$JEVBENCH" \
  --task-file "$JEVBENCH/datasets/public/easy.jsonl" \
  --task-file "$JEVBENCH/datasets/public/original.jsonl" \
  --task-file "$JEVBENCH/datasets/public/hard.jsonl" \
  --model "$BASE_MODEL" \
  --model-revision "$BASE_REVISION" \
  --expected-model-tree-sha256 "$BASE_MODEL_SHA" \
  --adapter "$ADAPTER" \
  --expected-adapter-tree-sha256 "$ADAPTER_SHA" \
  --out "$ATTEMPT_ROOT/jevbench-public-231" \
  --max-context 16384 \
  --dtype bfloat16 \
  --device cuda \
  --attn-implementation sdpa \
  --seed 42 >"$ATTEMPT_ROOT/jevbench-public-231.log" 2>&1

declare -a workflow_runs
declare -a workflow_run_args
for workflow in security_incidents agent_trace_observability invoice_processing customer_service; do
  workflow_runs+=("$WORKFLOWEVALS/runs/$workflow/$RUN_TAG")
  workflow_run_args+=(--workflow-run "$WORKFLOWEVALS/runs/$workflow/$RUN_TAG")
done

log "verifying integrity and frozen parity gates"
uv run --frozen python scripts/verify_r050_official_eval.py postrun \
  --preflight "$ATTEMPT_ROOT/preflight-receipt.json" \
  --startup-receipt "$ATTEMPT_ROOT/server-startup-receipt.json" \
  --workflowevals-dir "$WORKFLOWEVALS" \
  "${workflow_run_args[@]}" \
  --jevbench-dir "$JEVBENCH" \
  --jev-runner "$TYPICAL_REPO/scripts/jevbench_gemma_lora.py" \
  --jev-run "$ATTEMPT_ROOT/jevbench-public-231" \
  --out "$ATTEMPT_ROOT/verification-receipt.json" \
  >"$ATTEMPT_ROOT/postrun.log" 2>&1
jq -e '.integrity_pass == true' "$ATTEMPT_ROOT/verification-receipt.json" >/dev/null
jq -e '.workflow_parity.workflow_parity_pass == true' \
  "$ATTEMPT_ROOT/verification-receipt.json" >/dev/null

uv run --frozen python scripts/verify_r050_official_eval.py preserved \
  --verification "$ATTEMPT_ROOT/verification-receipt.json" \
  "${workflow_run_args[@]}" \
  --jev-run "$ATTEMPT_ROOT/jevbench-public-231" \
  --out "$ATTEMPT_ROOT/preservation-before-archive-receipt.json" \
  >"$ATTEMPT_ROOT/preservation-before-archive.log" 2>&1
jq -e '.evidence_preserved == true' \
  "$ATTEMPT_ROOT/preservation-before-archive-receipt.json" >/dev/null

log "copying complete run trees into the attempt archive"
ARCHIVE="$ATTEMPT_ROOT/archive"
mkdir "$ARCHIVE"
declare -a archived_workflow_run_args
for index in "${!workflow_runs[@]}"; do
  workflow="${workflow_runs[$index]}"
  workflow_name="$(basename "$(dirname "$workflow")")"
  destination="$ARCHIVE/workflow-runs/$workflow_name/$RUN_TAG"
  mkdir -p "$(dirname "$destination")"
  cp -a "$workflow" "$destination"
  archived_workflow_run_args+=(--workflow-run "$destination")
done
cp -a "$ATTEMPT_ROOT/jevbench-public-231" "$ARCHIVE/jevbench-public-231"

uv run --frozen python scripts/verify_r050_official_eval.py preserved \
  --verification "$ATTEMPT_ROOT/verification-receipt.json" \
  "${archived_workflow_run_args[@]}" \
  --jev-run "$ARCHIVE/jevbench-public-231" \
  --out "$ATTEMPT_ROOT/preservation-after-archive-receipt.json" \
  >"$ATTEMPT_ROOT/preservation-after-archive.log" 2>&1
jq -e '.evidence_preserved == true' \
  "$ATTEMPT_ROOT/preservation-after-archive-receipt.json" >/dev/null

VERIFICATION_RECEIPT_SHA="$(sha256sum "$ATTEMPT_ROOT/verification-receipt.json" | cut -d' ' -f1)"
PRESERVATION_BEFORE_SHA="$(sha256sum "$ATTEMPT_ROOT/preservation-before-archive-receipt.json" | cut -d' ' -f1)"
PRESERVATION_AFTER_SHA="$(sha256sum "$ATTEMPT_ROOT/preservation-after-archive-receipt.json" | cut -d' ' -f1)"
jq -n --arg adapter_sha "$ADAPTER_SHA" --arg run_tag "$RUN_TAG" \
  --arg verification_sha "$VERIFICATION_RECEIPT_SHA" \
  --arg preservation_before_sha "$PRESERVATION_BEFORE_SHA" \
  --arg preservation_after_sha "$PRESERVATION_AFTER_SHA" \
  '{schema_version:"typical.r050_official_eval_ready.v1",status:"ready_for_persistent_publication",adapter_tree_sha256:$adapter_sha,run_tag:$run_tag,verification_receipt_sha256:$verification_sha,preservation_before_archive_sha256:$preservation_before_sha,preservation_after_archive_sha256:$preservation_after_sha,integrity_pass:true,workflow_parity_pass:true}' \
  >"$ATTEMPT_ROOT/official-eval-ready.json.tmp"
mv "$ATTEMPT_ROOT/official-eval-ready.json.tmp" "$ATTEMPT_ROOT/official-eval-ready.json"

log "official local evaluation verified in private staging"
