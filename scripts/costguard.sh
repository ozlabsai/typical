#!/bin/zsh

# Guard one RunPod pod. A recent .context/costguard.hold pauses one check, never two.
set -u

POD_ID=${COSTGUARD_POD_ID:-ujiq3ud4vnhf80}
INTERVAL_SECONDS=${COSTGUARD_INTERVAL_SECONDS:-1800}
SSH_KEY=${COSTGUARD_SSH_KEY:-/Users/guynachshon/.runpod/ssh/RunPod-Key-Go}
SCRIPT_DIR=${0:A:h}
ROOT_DIR=${SCRIPT_DIR:h}
CONTEXT_DIR=${COSTGUARD_CONTEXT_DIR:-$ROOT_DIR/.context}
LOG_FILE=${COSTGUARD_LOG_FILE:-$CONTEXT_DIR/costguard.log}
PID_FILE="$CONTEXT_DIR/costguard.pid"
LOCK_DIR="$CONTEXT_DIR/costguard.lock"
HOLD_FILE="$CONTEXT_DIR/costguard.hold"
SAFE_TO_STOP_FILE=${COSTGUARD_SAFE_TO_STOP_FILE:-$CONTEXT_DIR/costguard.safe-to-stop.$POD_ID}

mkdir -p "$CONTEXT_DIR"

log() {
  print -r -- "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $*" | tee -a "$LOG_FILE"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    log "guard already running pid=$old_pid"
    exit 0
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || { log "cannot clear stale lock $LOCK_DIR"; exit 1; }
  mkdir "$LOCK_DIR" || exit 1
fi

print -r -- $$ > "$PID_FILE"
cleanup() {
  rm -f "$PID_FILE"
  rmdir "$LOCK_DIR" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 0' INT TERM

check_pod() {
  local pod_json pod_status volume_in_gb ip port remote metrics util memory active artifacts processes stopped_status local_transfer

  pod_json=$(runpodctl pod get "$POD_ID" -o json 2>/dev/null) || {
    log "pod=$POD_ID state=UNKNOWN reason=runpodctl_failed action=none"
    return
  }
  pod_status=$(jq -r '.desiredStatus // "UNKNOWN"' <<< "$pod_json")
  if [[ "$pod_status" != "RUNNING" ]]; then
    log "pod=$POD_ID state=$pod_status action=none"
    return
  fi
  volume_in_gb=$(jq -r '.volumeInGb // "UNKNOWN"' <<< "$pod_json")

  if [[ -f "$HOLD_FILE" ]] && find "$HOLD_FILE" -mmin -25 -print -quit | grep -q .; then
    log "pod=$POD_ID state=RUNNING action=hold reason=recent_costguard_hold"
    return
  fi

  ip=$(jq -r '.ssh.ip // empty' <<< "$pod_json")
  port=$(jq -r '.ssh.port // empty' <<< "$pod_json")
  if [[ -z "$ip" || -z "$port" ]]; then
    log "pod=$POD_ID state=RUNNING action=none reason=ssh_not_ready"
    return
  fi

  # A completed GPU job can still be actively copying its checkpoints locally.
  # Keep this one pod alive until that transfer exits; the next scheduled check
  # will stop it if nothing else is active.
  local_transfer=$(pgrep -af "scp .*${ip}" 2>/dev/null || true)
  if [[ -n "$local_transfer" ]]; then
    log "pod=$POD_ID state=RUNNING action=hold reason=local_artifact_transfer"
    return
  fi

  remote=$(ssh -i "$SSH_KEY" -p "$port" -o BatchMode=yes -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=accept-new "root@$ip" '
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1)
      memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1)
      processes=$(ps -eo pid=,comm=,args= | awk -v self=$$ '\''
        $1 == self { next }
        $2 ~ /^(python|python3|uv|vllm|torchrun|accelerate|deepspeed|VLLM|hf|rsync|scp|curl|wget|git|huggingface-cli)/ { print; next }
        $2 ~ /^(bash|zsh|sh)$/ && $0 ~ /\/workspace\/.*(train|eval|bench|serve|launch|run_|setup_)/ { print }
      '\'')
      active=$(printf "%s\n" "$processes" | awk "NF { n++ } END { print n+0 }")
      artifacts=$(find /workspace/g1 -mindepth 1 -maxdepth 3 -type f -print 2>/dev/null | wc -l | tr -d " ")
      printf "METRICS|%s|%s|%s|%s\n%s\n" "${util:-0}" "${memory:-0}" "$active" "$artifacts" "$processes"
    ' 2>/dev/null) || {
      log "pod=$POD_ID state=RUNNING action=none reason=ssh_failed"
      return
    }

  metrics=${remote%%$'\n'*}
  if [[ "$remote" == *$'\n'* ]]; then
    processes=${remote#*$'\n'}
  else
    processes=''
  fi
  IFS='|' read -r _ util memory active artifacts <<< "$metrics"
  if [[ ! "$util" =~ '^[0-9]+$' || ! "$active" =~ '^[0-9]+$' || ! "$artifacts" =~ '^[0-9]+$' ]]; then
    log "pod=$POD_ID state=RUNNING action=none reason=invalid_probe"
    return
  fi

  if (( active > 0 || util >= 5 )); then
    log "pod=$POD_ID state=RUNNING gpu=${util}% memory=${memory}MiB active=$active action=keep"
    [[ -n "$processes" ]] && print -r -- "$processes" | sed 's/^/  process: /' >> "$LOG_FILE"
    return
  fi
  if [[ "$volume_in_gb" == 0 && ! -f "$SAFE_TO_STOP_FILE" ]]; then
    log "pod=$POD_ID state=RUNNING gpu=${util}% memory=${memory}MiB active=0 action=hold reason=ephemeral_without_archive_receipt receipt=$SAFE_TO_STOP_FILE"
    return
  fi
  if (( artifacts == 0 )); then
    log "pod=$POD_ID state=RUNNING gpu=${util}% active=0 action=none reason=no_persisted_artifacts"
    return
  fi

  ssh -i "$SSH_KEY" -p "$port" -o BatchMode=yes -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=accept-new "root@$ip" 'sync' 2>/dev/null || {
      log "pod=$POD_ID state=RUNNING action=none reason=artifact_sync_failed"
      return
    }
  if runpodctl pod stop "$POD_ID" >/dev/null 2>&1; then
    sleep 2
    stopped_status=$(runpodctl pod get "$POD_ID" -o json 2>/dev/null | jq -r '.desiredStatus // "UNKNOWN"')
    log "pod=$POD_ID state=RUNNING gpu=${util}% memory=${memory}MiB active=0 artifacts=$artifacts action=STOP_REQUESTED state_after=$stopped_status"
  else
    log "pod=$POD_ID state=RUNNING action=none reason=stop_failed"
  fi
}

wait_for_next_check() {
  local deadline remaining
  deadline=$(( $(date +%s) + INTERVAL_SECONDS ))
  while (( (remaining = deadline - $(date +%s)) > 0 )); do
    (( remaining > 60 )) && remaining=60
    sleep "$remaining"
  done
}

log "guard started pid=$$ pod=$POD_ID interval=${INTERVAL_SECONDS}s"
while true; do
  check_pod
  [[ ${COSTGUARD_ONCE:-0} == 1 ]] && break
  # Short sleeps make a suspended Mac catch up immediately after waking.
  wait_for_next_check
done
