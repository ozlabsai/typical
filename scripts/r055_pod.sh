#!/usr/bin/env bash
# R055 GPU half on one 2-GPU pod (docs/research/hebrew/R055_HEBREW_SPRINT.md). Repo staged at /workspace/r054,
# results in /workspace/g1/r055 (the cost guard watches /workspace/g1). Reuses the hejev/Phase-0 stack.
#   rsetup                      Phase-0 setup (WorkflowEvals+patches, jevbench, vLLM 0.31) + pinned train venv
#                               + repo venv + Gemma/Qwen weights
#   label <deadline-utc>        Qwen3.8-27B-FP8 thinking, one vLLM per GPU, label native_unlabeled.jsonl
#   train <arm> <gpu> <steps> [trainer args]   fresh Gemma-4-12B-it LoRA on .context/r055/mix
#   evalarm <arm> <gpu> <step>  native-readout server -> hejev (en/HE/MX/native) + external panel; then retention
#   psetup2                     peers' runtimes for the external panel (cygnet, winnow, clef, startlux, R044-R)
#   peers <gpu> <fn>...         hejev peer functions on the external panel only, on one GPU
set -euo pipefail
source "$(dirname "$0")/hejev_pod_run.sh" true  # env, phase0 setup(), jev(), peer functions, G/Q/... pins
O5=/workspace/g1/r055; MIX=$R/.context/r055/mix; PY=/workspace/g1/.venv-pinned/bin/python
GM=$HF_HOME/hub/models--google--gemma-4-12B-it/snapshots/${G[1]}
EXT="belebele hebnli gmmlu sentiment winograd heq"
mkdir -p "$O5/logs"

rsetup() {
  local p=()
  (setup) >"$O5/logs/setup-phase0.log" 2>&1 & p+=($!)
  (uv venv -q /workspace/g1/.venv-pinned && uv pip install -q --python $PY torch==2.14.0 torchvision==0.29.0 \
    transformers==5.17.0 peft==0.21.0 accelerate==1.13.0 safetensors==0.8.0 sentencepiece==0.2.2) \
    >"$O5/logs/setup-pinned.log" 2>&1 & p+=($!)
  (cd "$R" && uv sync --frozen --no-dev && touch "$R/.venv-ready") >"$O5/logs/setup-repo.log" 2>&1 & p+=($!)
  local m; for m in "${G[*]}" "${Q[*]}"; do
    set -- $m; uvx --from huggingface_hub hf download "$1" --revision "$2" >"$O5/logs/dl-${1##*/}.log" 2>&1 & p+=($!)
  done
  local rc=0 x; for x in "${p[@]}"; do wait "$x" || rc=1; done
  test -f "$GM/config.json"; return $rc
}

killtree() { local c; for c in $(pgrep -P "$1"); do killtree "$c"; done; kill "$1" 2>/dev/null || true; }

label() {  # deadline -- vLLM on 8020/8021: RunPod nginx owns 8001
  local pids=() g
  for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g PATH="$R/venv-vllm/bin:$PATH" "$R/venv-vllm/bin/vllm" serve "${Q[0]}" --revision "${Q[1]}" \
      --served-model-name "${Q[0]}" --host 127.0.0.1 --port 802$g --max-model-len 65536 --gpu-memory-utilization 0.92 \
      --enable-prefix-caching --reasoning-parser qwen3 --max-num-seqs 256 --gdn-prefill-backend triton \
      --max-num-batched-tokens 16384 --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
      >"$O5/logs/qwen$g-vllm.log" 2>&1 & pids+=($!)
  done
  for g in 0 1; do wait_url http://127.0.0.1:802$g/v1/models ${pids[$g]} || { tail -40 "$O5/logs/qwen$g-vllm.log"; return 1; }; done
  for g in 0 1; do
    (cd "$WE" && exec uv run --frozen python "$R/scripts/r054_systemone_server.py" llm "vllm:${Q[0]}" --thinking default \
      --base-url http://127.0.0.1:802$g/v1 --port 801$g --workflowevals "$WE" >"$O5/logs/qwen$g-shim.log" 2>&1) & pids+=($!)
  done
  for g in 0 1; do wait_url http://127.0.0.1:801$g/v1/models ${pids[$((g + 2))]}; done
  date -u +%FT%TZ >"$O5/label.START"
  python3 "$R/scripts/r055_build_mix.py" label --data "$R/.context/he-train/d2/native_unlabeled.jsonl" \
    --out "$O5/native_labels.jsonl" --endpoint http://127.0.0.1:8010 --endpoint http://127.0.0.1:8011 --deadline "$1" --workers ${WORKERS:-128} \
    >"$O5/logs/label.log" 2>&1 || true
  date -u +%FT%TZ >"$O5/label.DONE"
  grep -h "throughput" "$O5"/logs/qwen*-vllm.log | tail -4 >"$O5/label-throughput.txt" || true
  for g in "${pids[@]}"; do killtree "$g"; done
}

train() {  # arm gpu steps [extra trainer args]
  local arm=$1 gpu=$2 steps=$3; shift 3
  mkdir -p "$O5/arm-$arm"
  cd "$R" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$R $PY -m scripts.train_gemma_decision_lora --model "$GM" \
    --train "$MIX/train.jsonl" --val he_dev="$MIX/dev_he.jsonl" --val en_dev="$MIX/dev_en.jsonl" --out "$O5/arm-$arm" \
    --steps "$steps" --rank 32 --alpha 64 --lr 5e-5 --min-lr 5e-6 --microbatch 1 --grad-accum 8 --max-length 8192 \
    --weight-decay .01 --seed 55 --eval-every 250 --save-every 500 --eval-samples 256 "$@" >"$O5/arm-$arm/train.log" 2>&1
  echo done >"$O5/arm-$arm/TRAIN_DONE"
}

evalarm() {  # arm gpu step -- serve the checkpoint's 3-file adapter tree, as R044-R was shipped
  local arm=$1 gpu=$2 step=$3 port=$((8095 + $2)) srv=$O5/serve-$1 sha id pid
  rm -rf "$srv"; mkdir -p "$srv"
  cp "$O5/arm-$arm/step-$(printf %06d "$step")"/{adapter_config.json,adapter_model.safetensors} "$srv/"
  [[ -e $O5/arm-$arm/step-$(printf %06d "$step")/README.md ]] && cp "$O5/arm-$arm/step-$(printf %06d "$step")/README.md" "$srv/"
  sha=$(cd "$R" && $PY -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$srv")
  id=typical-r055-$arm-sha256-${sha:0:12}; echo "$id $sha step=$step" >"$O5/arm-$arm/served-id.txt"
  (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu TYPICAL_API_KEY= exec uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$srv" --expected-adapter-tree-sha256 "$sha" --model-id "$id" \
    --receipt "$O5/arm-$arm/server-startup-receipt.json" --max-context 16384 --port $port >"$O5/logs/serve-$arm.log" 2>&1) &
  pid=$!
  wait_url http://127.0.0.1:$port/v1/models $pid || { tail -40 "$O5/logs/serve-$arm.log"; return 1; }
  jev "r055-$arm" http://127.0.0.1:$port "$id" en translated mixed native
  jev "r055-$arm" http://127.0.0.1:$port "$id" $EXT
  killtree $pid
  (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$R $PY -m scripts.eval_gemma_mm_he --model "$GM" --adapter "$srv" \
    --he-left data_he/eval/massive_he_test.jsonl --he-right data_he/eval/massive_en_test_paired.jsonl \
    --mm-dir data_v/eval --out "$O5/arm-$arm/mmhe.json" >"$O5/logs/mmhe-$arm.log" 2>&1)
  echo done >"$O5/arm-$arm/EVAL_DONE"
}

psetup2() {  # runtimes the hejev peer functions need (no jevbench re-clone, no torchcast)
  local p=()
  [[ -d $R/cygnet-recipe ]] || { git clone -q https://github.com/blockbrain-ai/cygnet-recipe.git "$R/cygnet-recipe" &&
    git -C "$R/cygnet-recipe" checkout -q 10b4099a61e7dca6a82032b54c74dc0a8c496d4b; }
  CUDACXX=/usr/local/cuda/bin/nvcc bash "$R/prepare_winnow.sh" >"$O5/logs/setup-winnow.log" 2>&1 & p+=($!)
  [[ -d $R/startlux ]] || git clone -q https://github.com/StartLuxLabs/StartLux-Decision.git "$R/startlux"
  git -C "$R/startlux" checkout -q $SL_CODE
  uv pip install -q --python "$R/venv-vllm/bin/python" "$R/startlux" >"$O5/logs/setup-startlux.log" 2>&1 & p+=($!)
  (uv venv -q "$R/venv-clef" && uv pip install -q --torch-backend auto --python "$R/venv-clef/bin/python" torch torchvision \
    transformers==5.10.2 pillow accelerate flash-linear-attention) >"$O5/logs/setup-venv-clef.log" 2>&1 & p+=($!)
  local hf=(uvx --from huggingface_hub hf download)
  "${hf[@]}" "${CF[0]}" --revision "${CF[1]}" >"$O5/logs/dl-clef.log" 2>&1 & p+=($!)
  "${hf[@]}" "${SL[0]}" --revision "${SL[1]}" --local-dir "$R/models/${SL[0]##*/}" >"$O5/logs/dl-startlux.log" 2>&1 & p+=($!)
  local rc=0 x; for x in "${p[@]}"; do wait "$x" || rc=1; done; return $rc
}

peers() {  # gpu fn... -- peer functions on one GPU, external panel only; down() scoped to this chain
  export CUDA_VISIBLE_DEVICES=$1; local gpu=$1 m; shift
  eval "jev_orig() $(declare -f jev | tail -n +2)"
  jev() { jev_orig "$1" "$2" "$3" $EXT; }
  eval "$(declare -f startlux | sed "s/--gpus 0/--gpus $gpu/")"  # its launcher sets CUDA_VISIBLE_DEVICES itself
  down() {  # ponytail: kill this chain's process tree, not every server on the pod (the other GPU is busy)
    local c me=$BASHPID; for c in $(pgrep -P $me); do killtree "$c"; done  # $BASHPID inside $() is the subshell
    for _ in $(seq ${DRAIN:-60}); do  # DRAIN=small when a trainer shares the GPU
      (( $(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) < 2000 )) && return 0; sleep 3
    done
  }
  for m in "$@"; do
    echo "$(date -u +%FT%TZ) start $m gpu$gpu"; "$m" || echo "$(date -u +%FT%TZ) FAILED rc=$? $m"
    echo "$(date -u +%FT%TZ) end $m"
  done
}

"$@"
