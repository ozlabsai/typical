#!/usr/bin/env bash
# R054 Phase 0 on one GPU pod. Results land in /workspace/g1/r054/<tag>/ (the cost guard watches /workspace/g1).
#   setup                                   venvs, pinned WorkflowEvals (+ typical + vllm patches), jevbench, weights
#   llm <tag> <hf_id> <rev> <thinking> <reasoning_parser> [extra vllm args...]
#                                           vLLM + WorkflowEvals `vllm:` provider (questions one) + JevBench 231
#   d1 <tag> <hf_id> <rev>                  d1 native readout as `typical:` (questions all) + JevBench 231
# Env: LIMIT=N runs only N cases per workflow (smoke); FRACTION=k runs the first ceil(n/k) cases in case-ID order
#      (budget-limited subset); JEV=0 skips JevBench.
set -euo pipefail
ulimit -n 65536  # 4 workflows x workers x question-workers open many sockets; the 1024 default crashed run.py
set -a; source /workspace/.env; set +a
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
R=/workspace/r054; OUT=/workspace/g1/r054; WE=$R/WorkflowEvals; JB=$R/jevbench
mkdir -p "$R/logs" "$OUT"
declare -A REV=([security_incidents]=fbe1ea5c69cf494157fd23f2002a0d9d9a418443
  [agent_trace_observability]=8635540973910a92465fe2bc53e195375aa6e1a8
  [invoice_processing]=6beeb2d2acd65c086c835022f5f4d7434114cafc
  [customer_service]=b1342f5a704587dbc465867c38c2694348ff86e4)
declare -A NCASES=([security_incidents]=240 [agent_trace_observability]=111 [invoice_processing]=150 [customer_service]=204)

setup() {
  git clone -q https://github.com/typesafe-ai/WorkflowEvals.git "$WE" && git -C "$WE" checkout -q 0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
  git -C "$WE" apply "$R/patches/workflowevals-typical-provider-0ac3b8ad.patch" "$R/patches/workflowevals-r054-vllm-provider.patch"
  (cd "$WE" && uv sync --locked -q && uv run --frozen python -m unittest discover 2>&1 | tail -1)
  git clone -q https://github.com/fstandhartinger/jevbench.git "$JB" && git -C "$JB" checkout -q bb05a335bc809e61b20c0f745d25499a82b326fc
  (cd "$JB" && uv sync -q)
  uv venv -q "$R/venv-vllm" && uv pip install -q --python "$R/venv-vllm/bin/python" vllm==0.31.0 ninja  # the version Phase 0 ran
  uv venv -q "$R/venv-d1" && uv pip install -q --python "$R/venv-d1/bin/python" "transformers>=5.14" torch torchvision pillow accelerate
  "$R/venv-vllm/bin/vllm" --version
}

wait_url() { for _ in $(seq 1 360); do curl -fsS "$1" >/dev/null 2>&1 && return 0; kill -0 "$2" 2>/dev/null || return 1; sleep 5; done; return 1; }

run_wf() {  # tag model_spec extra-run.py-args...
  local tag=$1 spec=$2; shift 2
  local pids=()
  for wf in "${!REV[@]}"; do
    (cd "$WE" && uv run --frozen python run.py "$wf" --model "$spec" --name "r054-$tag" \
      --dataset-revision "${REV[$wf]}" ${LIMIT:+--limit $LIMIT} \
      ${FRACTION:+--limit $(( (NCASES[$wf] + FRACTION - 1) / FRACTION ))} "$@" >"$R/logs/$tag-$wf.log" 2>&1) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
}

run_jev() {  # tag endpoint model -- public 231 in parallel chunks (jevbench's runner is sequential); resumable
  local tag=$1 ep=$2 model=$3 dir=$OUT/$1/jevbench
  [[ ${JEV:-1} == 1 ]] || return 0
  mkdir -p "$dir"
  # Keep answered rows of earlier attempts (as <tier>.done<N>.jsonl), write the rest as task chunks of 12.
  local round
  round=$(python3 - "$dir" "$JB/datasets/public" "${LIMIT:-0}" <<'PY'
import glob, json, os, sys
d, src, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
for f in glob.glob(f"{d}/tasks-*.jsonl"):
    os.remove(f)
done, n = set(), len(glob.glob(f"{d}/*.done*.jsonl"))
for f in sorted(glob.glob(f"{d}/*.jsonl")):
    name = os.path.basename(f)
    if name.startswith(("easy", "original", "hard")) and ".done" not in name:
        ok = [l for l in open(f) if json.loads(l)["ok"]]
        open(f"{d}/{name.split('-')[0].split('.')[0]}.done{n}-{name}", "w").writelines(ok)
        os.remove(f)
    if name.startswith(("easy", "original", "hard")):
        done |= {json.loads(l)["task_id"] for l in open(f if os.path.exists(f) else os.devnull)}
for f in glob.glob(f"{d}/*.done*.jsonl"):
    done |= {json.loads(l)["task_id"] for l in open(f)}
for tier in ("easy", "original", "hard"):
    rows = [l for l in open(f"{src}/{tier}.jsonl") if json.loads(l)["id"] not in done][: limit or None]
    for k in range(0, len(rows), 12):
        open(f"{d}/tasks-{tier}-{k // 12:02d}.jsonl", "w").writelines(rows[k:k + 12])
print(n)
PY
)
  local pids=()
  for tasks in "$dir"/tasks-*.jsonl; do
    [[ -e "$tasks" ]] || continue
    local shard; shard=$(basename "$tasks" .jsonl); shard=${shard#tasks-}
    # ponytail: the CLI exposes no timeout; thinking calls need more than the adapter's 120 s default.
    (cd "$JB" && uv run --no-sync python -c "
import sys, jevbench.adapters.typesafe as t
d = list(t.TypeSafeAdapter.__init__.__defaults__); d[3] = 3600.0; t.TypeSafeAdapter.__init__.__defaults__ = tuple(d)
from jevbench.cli import main; sys.exit(main())" run \
      --tasks "$tasks" --adapter typesafe --endpoint "$ep" --key-env '' \
      --model "$model" --cost-basis self_hosted_gpu --reserve-usd 0 --results "$dir/$shard-r$round.jsonl" \
      --raw-dir "$dir/raw/$shard-r$round" --ledger "$dir/ledger/$shard-r$round.jsonl" \
      --manifest "$dir/manifest/$shard-r$round.json" \
      --run-label "r054-$tag" --delay-s 0 >"$R/logs/$tag-jev-$shard.log" 2>&1) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
}

collect() {  # tag
  for wf in "${!REV[@]}"; do
    [[ -d "$WE/runs/$wf/r054-$1" ]] && rsync -a "$WE/runs/$wf/r054-$1/" "$OUT/$1/$wf/"
  done
  cp "$R"/logs/"$1"-* "$OUT/$1/" 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader >"$OUT/$1/gpu.txt"
  date -u +%FT%TZ >"$OUT/$1/DONE"
}

llm() {
  local tag=$1 model=$2 rev=$3 thinking=$4 parser=$5; shift 5
  mkdir -p "$OUT/$tag"; date -u +%FT%TZ >"$OUT/$tag/START"
  PATH="$R/venv-vllm/bin:$PATH" "$R/venv-vllm/bin/vllm" serve "$model" --revision "$rev" --served-model-name "$model" --host 127.0.0.1 --port 8000 \
    --max-model-len 65536 --gpu-memory-utilization 0.92 --enable-prefix-caching --reasoning-parser "$parser" \
    --max-num-seqs 256 --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' "$@" >"$R/logs/$tag-vllm.log" 2>&1 &
  local vllm=$!
  trap "kill $vllm 2>/dev/null || true" RETURN
  wait_url http://127.0.0.1:8000/v1/models $vllm || { tail -50 "$R/logs/$tag-vllm.log"; return 1; }
  date -u +%FT%TZ >"$OUT/$tag/SERVER_READY"
  export VLLM_BASE_URL=http://127.0.0.1:8000/v1
  (cd "$WE" && exec uv run --frozen python "$R/scripts/r054_systemone_server.py" llm "vllm:$model" --thinking "$thinking" \
    --base-url "$VLLM_BASE_URL" --port 8010 --workflowevals "$WE" >"$R/logs/$tag-shim.log" 2>&1) &
  local shim=$!
  wait_url http://127.0.0.1:8010/v1/models $shim || { tail -50 "$R/logs/$tag-shim.log"; return 1; }
  run_jev "$tag" http://127.0.0.1:8010 "vllm:$model" &
  local jev=$!
  run_wf "$tag" "vllm:$model" --thinking "$thinking" --questions one --workers 16 --question-workers 8 \
    --max-output-tokens 16384 --timeout 3600 --retry-errors
  wait $jev || true
  kill $shim 2>/dev/null || true
  grep -h "prompt_tokens\|generation_tokens" "$R/logs/$tag-vllm.log" | tail -2 >"$OUT/$tag/vllm-throughput.txt" || true
  collect "$tag"
}

d1() {
  local tag=$1 model=$2 rev=$3 name=${2##*/}-${3:0:7}
  mkdir -p "$OUT/$tag"; date -u +%FT%TZ >"$OUT/$tag/START"
  "$R/venv-d1/bin/python" "$R/scripts/r054_systemone_server.py" d1 "$model" --revision "$rev" --served-name "$name" \
    --port 8011 >"$R/logs/$tag-shim.log" 2>&1 &
  local shim=$!
  trap "kill $shim 2>/dev/null || true" RETURN
  wait_url http://127.0.0.1:8011/v1/models $shim || { tail -50 "$R/logs/$tag-shim.log"; return 1; }
  export TYPICAL_API_KEY=local TYPICAL_BASE_URL=http://127.0.0.1:8011
  run_jev "$tag" http://127.0.0.1:8011 "$name" &
  local jev=$!
  run_wf "$tag" "typical:$name" --questions all --workers 4 --question-workers 1 --timeout 1800 --retry-errors
  wait $jev || true
  collect "$tag"
}

"$@"
