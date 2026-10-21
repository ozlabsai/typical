#!/usr/bin/env bash
# R055 readout check on one GPU pod: the R055 LoRAs under Cygnet's letter readout (vLLM 0.30.0 + Gemma@707f0a3 +
# cygnet-recipe@10b4099 decision_server.py, flags as .context/run_g0.sh) vs their native readout. The temperature
# and the readout choice use the HE dev split only (scripts/r055_readout.py); test splits run after the choice.
# Repo staged at /workspace/r054 (as hejev_pod_run.sh), adapters at $R/serve-{a,b}, tasks at $R/hejev/<split>.jsonl.
#   xsetup                     jevbench@bb05a335 + vLLM 0.30.0 venv + repo venv + Gemma weights
#   native <split>...          native-readout server per arm (both on the GPU at once) -> jev (tag native-<arm>)
#   lora_up                    one vLLM 0.30.0: Gemma + both LoRAs (served as r055-a / r055-b)
#   cyg <arm> <T> <split>...   Cygnet shim at T over the arm's LoRA -> jev (tag cyg-<arm>-T<T>)
set -euo pipefail
source "$(dirname "$0")/hejev_pod_run.sh" true  # env, jev(), down(), up(), G pins
OUT=/workspace/g1/readout; H=$R/hejev
mkdir -p "$OUT/logs"
declare -A SHA=([a]=10c880b02a85766d8335de052dd1420b8e69b81e97b99a5e903b29c49c3cb9e8
  [b]=c54ad4caca12c006bca352737a4d4b98343a80ff203b5af92d389bac7fc28e30)

xsetup() {
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  local p=() x rc=0
  (git clone -q https://github.com/fstandhartinger/jevbench.git "$JB" && git -C "$JB" checkout -q bb05a335bc809e61b20c0f745d25499a82b326fc \
    && cd "$JB" && uv sync -q) >"$OUT/logs/setup-jevbench.log" 2>&1 & p+=($!)
  (uv venv -q "$R/venv-vllm030" && uv pip install -q --python "$R/venv-vllm030/bin/python" vllm==0.30.0) \
    >"$OUT/logs/setup-vllm030.log" 2>&1 & p+=($!)
  (cd "$R" && uv sync --frozen --no-dev) >"$OUT/logs/setup-repo.log" 2>&1 & p+=($!)
  uvx --from huggingface_hub hf download "${G[0]}" --revision "${G[1]}" >"$OUT/logs/dl-gemma.log" 2>&1 & p+=($!)
  for x in "${p[@]}"; do wait "$x" || rc=1; done; return $rc
}

native() {  # split... -- as r055_pod.sh evalarm, both arms at once
  local arm pids=() port
  for arm in a b; do
    port=$((8095 + ${#pids[@]}))
    (cd "$R" && TYPICAL_API_KEY= exec uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
      --model "${G[0]}" --model-revision "${G[1]}" --adapter "$R/serve-$arm" --expected-adapter-tree-sha256 "${SHA[$arm]}" \
      --model-id "typical-r055-$arm-sha256-${SHA[$arm]:0:12}" --receipt "$OUT/native-$arm-receipt.json" --max-context 16384 \
      --port $port >"$OUT/logs/native-$arm.log" 2>&1) & pids+=($!)
  done
  up http://127.0.0.1:8095/v1/models ${pids[0]} "$OUT/logs/native-a.log"
  up http://127.0.0.1:8096/v1/models ${pids[1]} "$OUT/logs/native-b.log"
  local j=() x
  jev native-a http://127.0.0.1:8095 "typical-r055-a-sha256-${SHA[a]:0:12}" "$@" & j+=($!)
  jev native-b http://127.0.0.1:8096 "typical-r055-b-sha256-${SHA[b]:0:12}" "$@" & j+=($!)
  for x in "${j[@]}"; do wait "$x" || true; done
  down
}

lora_up() {  # .context/run_g0.sh's serve line + the two adapters
  "$R/venv-vllm030/bin/vllm" serve "${G[0]}" --revision "${G[1]}" --served-model-name cygnet --host 127.0.0.1 \
    --port 8890 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules r055-a="$R/serve-a" r055-b="$R/serve-b" \
    >"$OUT/logs/vllm.log" 2>&1 &
  up http://127.0.0.1:8890/v1/models $! "$OUT/logs/vllm.log"
  curl -fsS http://127.0.0.1:8890/v1/models >"$OUT/vllm-models.json"
}

cyg() {  # arm T split...
  local arm=$1 t=$2; shift 2
  local port=$((8010 + $(printf %d "'$arm") - 97)) id=typical-r055-$arm-sha256-${SHA[$arm]:0:12}-cygnet-T$t log=$OUT/logs/cyg-$arm-T$t
  SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions SHIM_MODEL=r055-$arm SHIM_TEMPERATURE=$t CYGNET_PORT=$port \
    CYGNET_MODEL_NAME=$id python3 "$R/cygnet-recipe/shim/decision_server.py" >"$log-shim.log" 2>&1 &
  local pid=$!
  wait_url http://127.0.0.1:$port/v1/models $pid || { tail -40 "$log-shim.log"; return 1; }
  jev "cyg-$arm-T$t" http://127.0.0.1:$port "$id" "$@"
  kill $pid
}

"$@"
