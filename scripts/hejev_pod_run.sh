#!/usr/bin/env bash
# Hebrew JevBench v0 (EVAL-ONLY) on one GPU pod, on the R054 Phase 0 stack (scripts/r054_pod_phase0.sh).
# Results: /workspace/g1/hejev/<model>/<split>/NNN.jsonl, one file per 12-task chunk (hejev_score.py merges a dir).
# Splits: en (public 231) | translated | mixed | native. Thinking models and d1 skip en (Phase 0 runs are reused).
#   hsetup   phase0 setup + cygnet-recipe@10b4099 + repo venv (R044-R server) + Winnow build/GGUF + weights
#   chain    cygnet, gemma_think, r044r, winnow, d1_native, qwen_think in turn
#   <one of the above model functions>
set -euo pipefail
source "$(dirname "$0")/r054_pod_phase0.sh" true  # env, ulimit -n 65536, setup(), wait_url(); `true` = no-op dispatch
OUT=/workspace/g1/hejev; H=$R/hejev; W=/workspace/winnow-g0
G=(google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7)
Q=(Qwen/Qwen3.8-27B-FP8 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a)
D=(LiquidAI/d1-3B 051bcc464b01b9f92942b364d9586b0ef5912432)
R044R_SHA=198185cfb4188fcbf24f1793377c268f184913aa2fd1e09b0d2280ecb28c76c9  # tree: README, adapter_config, adapter_model
mkdir -p "$OUT/logs"
# ponytail: jevbench's CLI exposes no timeout; thinking calls need more than the adapter's 120 s default (as Phase 0).
JEV_MAIN='import sys, jevbench.adapters.typesafe as t
d = list(t.TypeSafeAdapter.__init__.__defaults__); d[3] = 3600.0; t.TypeSafeAdapter.__init__.__defaults__ = tuple(d)
from jevbench.cli import main; sys.exit(main())'

hsetup() {
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  (cd "$R" && uv sync --frozen --no-dev && touch "$R/.venv-ready") >"$OUT/logs/setup-venv.log" 2>&1 &
  CUDACXX=/usr/local/cuda/bin/nvcc bash "$R/prepare_winnow.sh" >"$OUT/logs/setup-winnow.log" 2>&1 &
  git clone -q https://github.com/blockbrain-ai/cygnet-recipe.git "$R/cygnet-recipe"
  git -C "$R/cygnet-recipe" checkout -q 10b4099a61e7dca6a82032b54c74dc0a8c496d4b  # HEAD when G0 ran (2026-10-04)
  setup
  local dl=() m
  for m in "${G[*]}" "${Q[*]}" "${D[*]}"; do
    set -- $m; "$R/venv-vllm/bin/hf" download "$1" --revision "$2" >"$OUT/logs/dl-${1##*/}.log" 2>&1 & dl+=($!)
  done
  for m in "${dl[@]}"; do wait "$m"; done
}

down() {  # stop every server of this script and wait for the GPU to drain
  local pat='vllm serve|r054_systemone_server|decision_server.py|serve_typesafe_gemma_lora|winnow-server|scripts/serve.py'
  pkill -f "$pat" || true
  for _ in $(seq 60); do
    (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )) && return 0; sleep 3
  done
  pkill -9 -f "$pat|VLLM" || true; sleep 5
}

up() {  # url pid log -- wait for a server or show why it died
  wait_url "$1" "$2" || { tail -40 "$3"; down; return 1; }
}

jev() {  # tag endpoint model split... -- one pass, then one retry pass for chunks with failed rows
  date -u +%FT%TZ >"$OUT/$1.START"
  jev_pass "$@"; jev_pass "$@"
  nvidia-smi --query-gpu=name --format=csv,noheader >"$OUT/$1/gpu.txt"
  date -u +%FT%TZ >"$OUT/$1.DONE"
}

jev_pass() {  # every incomplete chunk of every split in parallel; complete chunks are kept
  local tag=$1 ep=$2 model=$3 split dir t c; shift 3
  local pids=()
  for split in "$@"; do
    dir=$OUT/$tag/$split; mkdir -p "$dir"/{tasks,raw,ledger,manifest,logs,failed}
    if [[ ! -e $dir/tasks/000.jsonl ]]; then
      if [[ $split == en ]]; then cat "$JB"/datasets/public/{original,easy,hard}.jsonl; else cat "$H/$split.jsonl"; fi |
        split -l 12 -d -a 3 --additional-suffix=.jsonl - "$dir/tasks/"
    fi
    for t in "$dir"/tasks/*.jsonl; do
      c=$(basename "$t" .jsonl)
      if [[ -e $dir/$c.jsonl ]]; then
        [[ $(wc -l <"$dir/$c.jsonl") == $(wc -l <"$t") ]] && ! grep -q '"ok": false' "$dir/$c.jsonl" && continue
        mv "$dir/$c.jsonl" "$dir/failed/$c.$(date +%s).jsonl"
        rm -rf "$dir/raw/$c" "$dir/ledger/$c.jsonl" "$dir/manifest/$c.json"
      fi
      (cd "$JB" && uv run --no-sync python -c "$JEV_MAIN" run --tasks "$t" --adapter typesafe --endpoint "$ep" \
        --key-env '' --model "$model" --cost-basis self_hosted_gpu --reserve-usd 0 --results "$dir/$c.jsonl" \
        --raw-dir "$dir/raw/$c" --ledger "$dir/ledger/$c.jsonl" --manifest "$dir/manifest/$c.json" \
        --run-label "hejev-$tag-$split" --delay-s 0 >"$dir/logs/$c.log" 2>&1) &
      pids+=($!)
      sleep 0.3  # ponytail: ThreadingHTTPServer's listen backlog is 5; a 70-connect burst got resets (cygnet run)
    done
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
}

cygnet() {  # frozen Gemma single pass, Cygnet letter readout at its fitted T=3.4 (.context/run_g0.sh)
  local name=cygnet-gemma-4-12B-it-707f0a3-T3.4 log=$OUT/logs/cygnet
  "$R/venv-vllm/bin/vllm" serve "${G[0]}" --revision "${G[1]}" --served-model-name cygnet --host 127.0.0.1 \
    --port 8890 --max-model-len 16384 --gpu-memory-utilization 0.90 >"$log-vllm.log" 2>&1 &
  up http://127.0.0.1:8890/v1/models $! "$log-vllm.log" || return 1
  SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions SHIM_MODEL=cygnet SHIM_TEMPERATURE=3.4 CYGNET_PORT=8010 \
    CYGNET_MODEL_NAME=$name python3 "$R/cygnet-recipe/shim/decision_server.py" >"$log-shim.log" 2>&1 &
  up http://127.0.0.1:8010/v1/models $! "$log-shim.log" || return 1
  jev cygnet http://127.0.0.1:8010 "$name" en translated mixed native
  down
}

think() {  # tag model rev reasoning_parser [extra vllm args] -- Phase 0 `llm` serving + vllm: systemone shim, no WF
  local tag=$1 model=$2 rev=$3 parser=$4 log=$OUT/logs/$1; shift 4
  PATH="$R/venv-vllm/bin:$PATH" "$R/venv-vllm/bin/vllm" serve "$model" --revision "$rev" --served-model-name "$model" \
    --host 127.0.0.1 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.92 --enable-prefix-caching \
    --reasoning-parser "$parser" --max-num-seqs 256 \
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' "$@" >"$log-vllm.log" 2>&1 &
  up http://127.0.0.1:8000/v1/models $! "$log-vllm.log" || return 1
  (cd "$WE" && exec uv run --frozen python "$R/scripts/r054_systemone_server.py" llm "vllm:$model" --thinking default \
    --base-url http://127.0.0.1:8000/v1 --port 8010 --workflowevals "$WE" >"$log-shim.log" 2>&1) &
  up http://127.0.0.1:8010/v1/models $! "$log-shim.log" || return 1
  jev "$tag" http://127.0.0.1:8010 "vllm:$model" ${SPLITS:-translated mixed native}
  grep -h "prompt_tokens\|generation_tokens\|throughput" "$log-vllm.log" | tail -2 >"$OUT/$tag/vllm-throughput.txt" || true
  down
}
gemma_think() { think gemma4-12b-think "${G[@]}" gemma4; }
qwen_think() { think qwen38-27b-think "${Q[@]}" qwen3 --gdn-prefill-backend triton --max-num-batched-tokens 16384; }

r044r() {  # Typical R044-R LoRA, native readout via the TypeSafe-compatible server
  local id=typical-r044r-sha256-${R044R_SHA:0:12} log=$OUT/logs/r044r
  for _ in $(seq 120); do [[ -e $R/.venv-ready ]] && break; sleep 10; done
  mkdir -p "$OUT/r044r"
  (cd "$R" && TYPICAL_API_KEY= exec uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$R/r044r" --expected-adapter-tree-sha256 "$R044R_SHA" \
    --model-id "$id" --receipt "$OUT/r044r/server-startup-receipt.json" --max-context 16384 --port 8090 \
    >"$log-server.log" 2>&1) &
  up http://127.0.0.1:8090/v1/models $! "$log-server.log" || return 1
  jev r044r http://127.0.0.1:8090 "$id" en translated mixed native
  down
}

winnow() {  # Winnow-12B Q8_0 GGUF on its pinned server (.context/run_winnow_g0.sh)
  local log=$OUT/logs/winnow
  for _ in $(seq 180); do [[ -e $W/PREPARED ]] && break; sleep 10; done
  echo "b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea  $W/models/Winnow-12B-Q8_0.gguf" | sha256sum -c -
  (cd "$W/inference" && exec uv run --no-project python scripts/serve.py --model "$W/models/Winnow-12B-Q8_0.gguf" \
    --context 16384 --decision-context 16384 --cache q8_0 --decision-parallel 4 --chat-parallel 1 --memory auto \
    >"$log-server.log" 2>&1) &
  up http://127.0.0.1:8091/health $! "$log-server.log" || return 1
  jev winnow-12b http://127.0.0.1:8091 Winnow-12B en translated mixed native
  down
}

d1_native() {  # LiquidAI d1-3B shipped model.system_one (Phase 0 `d1` backend)
  local log=$OUT/logs/d1
  "$R/venv-d1/bin/python" "$R/scripts/r054_systemone_server.py" d1 "${D[0]}" --revision "${D[1]}" \
    --served-name d1-3B-051bcc4 --port 8011 >"$log-shim.log" 2>&1 &
  up http://127.0.0.1:8011/v1/models $! "$log-shim.log" || return 1
  jev d1-3b http://127.0.0.1:8011 d1-3B-051bcc4 translated mixed native
  down
}

chain() {
  local m
  for m in ${MODELS:-cygnet gemma_think r044r winnow d1_native qwen_think}; do
    echo "$(date -u +%FT%TZ) start $m"; down; "$m" || echo "$(date -u +%FT%TZ) FAILED rc=$? $m"
    echo "$(date -u +%FT%TZ) end $m"
  done
  echo CHAIN_DONE
}

"$@"
