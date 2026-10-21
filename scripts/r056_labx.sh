#!/usr/bin/env bash
# R056 streaming labeler without the HTTP shim: r056_labx.sh <name> gemma|qwen <gpu>...
# Re-runs scripts/r056_label.py on $IN/<name>.jsonl (growing, pushed by r056_push.sh) until $O6/STOP_<name>.
source "$(dirname "$0")/r056_pod.sh" true
name=$1 kind=$2; shift 2
[[ $kind == gemma ]] && m=${G[0]} || m=${Q[0]}
urls=(); for g in "$@"; do urls+=(--base-url "http://127.0.0.1:802$g/v1"); done
until [[ -e $O6/STOP_$name ]]; do
  (cd "$WE" && uv run --frozen python "$R/scripts/r056_label.py" --data "$IN/$name.jsonl" --out "$O6/labels_$name.jsonl" \
    --model "vllm:$m" "${urls[@]}" --workers "${WORKERS:-64}" --workflowevals "$WE") >>"$O6/logs/label-$name.log" 2>&1
  echo "$(date -u +%FT%TZ) pass done" >>"$O6/logs/label-$name.log"
  sleep 60
done
