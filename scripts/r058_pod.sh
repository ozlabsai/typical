#!/usr/bin/env bash
# R058 on one GPU pod (docs/research/R058_IDEA_HEADS.md). Repo staged at /workspace/r054 (the Phase-0 layout),
# inputs in /workspace/g1/r058/in, results in /workspace/g1/r058 (the cost guard watches /workspace/g1).
#   rsetup                  pinned WorkflowEvals + typical patch, repo venv, pinned Gemma weights
#   cache                   R056 student features of in/rows_train.jsonl and in/rows_dev.jsonl (adapter: in/adapter)
#   train                   the four heads on the cached features (CPU tensors; minutes)
#   wf <head>               serve student+head (or `student`), official WorkflowEvals 705 (pinned, typical:, all)
set -euo pipefail
source "$(dirname "$0")/r054_pod_phase0.sh" true  # env (.env, HF_HOME, ulimit -n 65536), R, WE, REV, wait_url
O8=/workspace/g1/r058; IN=$O8/in
G=(google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7)
mkdir -p "$O8/logs"

rsetup() {
  git clone -q https://github.com/typesafe-ai/WorkflowEvals.git "$WE" && git -C "$WE" checkout -q 0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
  git -C "$WE" apply "$R/patches/workflowevals-typical-provider-0ac3b8ad.patch"
  (cd "$WE" && uv sync --locked -q)
  (cd "$R" && uv sync --frozen --no-dev -q)
  uvx --from huggingface_hub hf download "${G[0]}" --revision "${G[1]}" >"$O8/logs/dl-gemma.log" 2>&1
  touch "$O8/SETUP_DONE"
}

h8() { (cd "$R" && PYTHONPATH=$R WORKFLOWEVALS=$WE uv run --no-sync python scripts/r058_heads.py "$@"); }

cache() {
  local s
  for s in dev train; do
    h8 cache "$IN/rows_$s.jsonl" "$O8/cache_$s" --model "${G[0]}" --model-revision "${G[1]}" --adapter "$IN/adapter" \
      >>"$O8/logs/cache-$s.log" 2>&1
  done
  touch "$O8/CACHE_DONE"
}

train() {
  h8 train "$O8/cache_train" "$O8/cache_dev" "$O8/heads" ${TEMPS:+--temperatures "$IN/adapter/temperatures.json"} \
    >"$O8/logs/train.log" 2>&1
  touch "$O8/TRAIN_DONE"
}

wf() {  # head -- one server, the four workflows in parallel (R056 `wf` settings)
  local head=$1 sha id pid w pids=()
  sha=$(cd "$R" && uv run --no-sync python -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$IN/adapter")
  id=typical-r058-$head-sha256-${sha:0:12}; echo "$id" >"$O8/served-id-$head.txt"
  (cd "$R" && PYTHONPATH=$R exec uv run --no-sync python scripts/r058_heads.py serve "$O8/heads/heads.pt" "$head" \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$IN/adapter" --model-id "$id" --port 8090 \
    >"$O8/logs/serve-$head.log" 2>&1) &
  pid=$!
  wait_url http://127.0.0.1:8090/v1/models $pid || { tail -30 "$O8/logs/serve-$head.log"; return 1; }
  for w in "${!REV[@]}"; do
    (cd "$WE" && TYPICAL_API_KEY=local TYPICAL_BASE_URL=http://127.0.0.1:8090 uv run --frozen python run.py "$w" \
      --model "typical:$id" --name "r058-$head" --dataset-revision "${REV[$w]}" --questions all --workers 4 \
      --question-workers 1 --timeout 1800 --retry-errors >"$O8/logs/wf-$head-$w.log" 2>&1
     rsync -a "$WE/runs/$w/r058-$head/" "$O8/wf-$head/$w/") & pids+=($!)
  done
  for w in "${pids[@]}"; do wait "$w" || true; done
  pkill -f "r058_heads.py serve" || true
  for _ in $(seq 60); do (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )) && break; sleep 3; done
  touch "$O8/WF_DONE-$head"
}

"$@"
