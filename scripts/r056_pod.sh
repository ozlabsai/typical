#!/usr/bin/env bash
# R056 (R054 Phase 1) on one 4-GPU pod. Repo staged at /workspace/r054, results in /workspace/g1/r056
# (the cost guard watches /workspace/g1). Reuses the Phase-0 / hejev / R055 stack (r055_pod.sh rsetup, jev, ...).
#   rsetup                          (R055) WorkflowEvals+patches, jevbench, vLLM 0.31, pinned train venv, weights
#   srv <gpu> gemma|qwen            vLLM thinking teacher on GPU g (:802g) + vllm: systemone shim (:801g)
#   stopsrv <gpu>                   stop GPU g's teacher + shim
#   lab <name> <deadline> <gpu>...  label $IN/<name>.jsonl through the shims of those GPUs -> $O6/labels_<name>.jsonl
#   train <steps>                   torchrun x4: fresh Gemma-4-12B-it LoRA on $R/.context/r056/mix
#   devsel <step>...                dev teacher-NLL (+ per-question probs) of saved checkpoints, one GPU each
#   serve <gpu> <step> <port>       native-readout server of a checkpoint (+ temperatures.json if present)
#   wf <port> <id> <workflow>...    WorkflowEvals (pinned revisions, typical: provider, questions all)
#   evalhe <gpu> <step>             JevBench public 231 + hejev (HE/MX/native) + external panel, then retention
set -euo pipefail
source "$(dirname "$0")/r055_pod.sh" true  # env, ulimit, setup/rsetup, wait_url, killtree, jev, G/Q pins, PY, GM
O6=/workspace/g1/r056; MIX6=$R/.context/r056/mix; IN=$R/.context/r056/label_in
mkdir -p "$O6/logs" "$O6/pids"

srv() {  # gpu model
  local g=$1 m p
  if [[ $2 == gemma ]]; then m=("${G[@]}"); p=(--reasoning-parser gemma4)
  else m=("${Q[@]}"); p=(--reasoning-parser qwen3 --gdn-prefill-backend triton --max-num-batched-tokens 16384); fi
  CUDA_VISIBLE_DEVICES=$g PATH="$R/venv-vllm/bin:$PATH" setsid "$R/venv-vllm/bin/vllm" serve "${m[0]}" --revision "${m[1]}" \
    --served-model-name "${m[0]}" --host 127.0.0.1 --port 802$g --max-model-len 65536 --gpu-memory-utilization 0.92 \
    --enable-prefix-caching --max-num-seqs 256 "${p[@]}" \
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' >"$O6/logs/$2$g-vllm.log" 2>&1 &
  echo $! >"$O6/pids/vllm$g"
  wait_url http://127.0.0.1:802$g/v1/models "$(cat "$O6/pids/vllm$g")" || { tail -30 "$O6/logs/$2$g-vllm.log"; return 1; }
  (cd "$WE" && exec setsid uv run --frozen python "$R/scripts/r054_systemone_server.py" llm "vllm:${m[0]}" --thinking default \
    --base-url http://127.0.0.1:802$g/v1 --port 801$g --workflowevals "$WE" >"$O6/logs/$2$g-shim.log" 2>&1) &
  echo $! >"$O6/pids/shim$g"
  wait_url http://127.0.0.1:801$g/v1/models "$(cat "$O6/pids/shim$g")"
  echo "$2" >"$O6/pids/model$g"; echo "srv $2 gpu$g up"
}

stopsrv() {
  local f; for f in "$O6/pids/shim$1" "$O6/pids/vllm$1"; do [[ -e $f ]] && { killtree "$(cat "$f")" || true; rm -f "$f"; }; done
  for _ in $(seq 60); do (( $(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits) < 2000 )) && return 0; sleep 3; done
  pkill -9 -f "port 802$1" || true
}

lab() {  # name deadline gpu...
  local name=$1 deadline=$2 g eps=(); shift 2
  for g in "$@"; do eps+=(--endpoint "http://127.0.0.1:801$g"); done
  python3 "$R/scripts/r055_build_mix.py" label --data "$IN/$name.jsonl" --out "$O6/labels_$name.jsonl" "${eps[@]}" \
    --deadline "$deadline" --workers $((64 * $#)) >>"$O6/logs/label-$name.log" 2>&1
  echo "$(date -u +%FT%TZ) LAB_DONE $name" >>"$O6/logs/label-$name.log"
}

train() {  # steps [extra trainer args]
  local steps=$1; shift
  mkdir -p "$O6/train"
  cd "$R" && PYTHONPATH=$R /workspace/g1/.venv-pinned/bin/torchrun --standalone --nproc_per_node 4 \
    -m scripts.train_gemma_decision_lora --model "$GM" --train "$MIX6/train.jsonl" \
    --val wf_dev="$MIX6/dev_wf.jsonl" --val he_dev="$MIX6/dev_he.jsonl" --out "$O6/train" --steps "$steps" \
    --rank 32 --alpha 64 --lr 5e-5 --min-lr 5e-6 --microbatch 1 --grad-accum 8 --max-length 16384 --weight-decay .01 \
    --seed 56 --eval-every 200 --save-every 100 --eval-samples 96 --twin-consistency-weight 0.5 --skip-overlength "$@" \
    >"$O6/train/train.log" 2>&1
  echo done >"$O6/train/TRAIN_DONE"
}

devsel() {  # step... -> $O6/devsel/step-N.json, one GPU per checkpoint
  mkdir -p "$O6/devsel"
  local i=0 s pids=()
  for s in "$@"; do
    (cd "$R" && CUDA_VISIBLE_DEVICES=$((i % 4)) PYTHONPATH=$R $PY scripts/r056_devsel.py --model "$GM" \
      --adapter "$O6/train/step-$(printf %06d "$s")" --dev "$MIX6/dev_wf.jsonl" --out "$O6/devsel/step-$s.json" \
      >"$O6/logs/devsel-$s.log" 2>&1) & pids+=($!)
    i=$((i + 1))
    (( i % 4 == 0 )) && { for p in "${pids[@]}"; do wait "$p" || true; done; pids=(); }
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
}

serve() {  # gpu step port -- 2-file adapter tree (+ temperatures.json) as R055; id typical:r056-sha256-<12>
  local gpu=$1 step=$2 port=$3 srv=$O6/serve sha id
  mkdir -p "$srv"
  cp "$O6/train/step-$(printf %06d "$step")"/{adapter_config.json,adapter_model.safetensors} "$srv/"
  [[ -e $O6/temperatures.json ]] && cp "$O6/temperatures.json" "$srv/"
  sha=$(cd "$R" && $PY -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$srv")
  id=r056-sha256-${sha:0:12}; echo "$id $sha step=$step" >"$O6/served-id.txt"
  (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu TYPICAL_API_KEY= exec setsid uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$srv" --expected-adapter-tree-sha256 "$sha" --model-id "$id" \
    ${TEMPS:+--temperatures "$srv/temperatures.json"} --receipt "$O6/server-startup-receipt-$port.json" \
    --max-context 16384 --port "$port" >"$O6/logs/serve-$port.log" 2>&1) &
  echo $! >"$O6/pids/serve$port"
  wait_url http://127.0.0.1:$port/v1/models "$(cat "$O6/pids/serve$port")" || { tail -30 "$O6/logs/serve-$port.log"; return 1; }
  echo "$id"
}

wf() {  # port id workflow... -- the R050/Phase-0 harness path, typical: provider, questions all
  local port=$1 id=$2 w pids=(); shift 2
  for w in "$@"; do
    (cd "$WE" && TYPICAL_API_KEY=local TYPICAL_BASE_URL=http://127.0.0.1:$port exec uv run --frozen python run.py "$w" \
      --model "typical:$id" --name r056 --dataset-revision "${REV[$w]}" --questions all --workers 4 \
      --question-workers 1 --timeout 1800 --retry-errors >"$O6/logs/wf-$w.log" 2>&1
     rsync -a "$WE/runs/$w/r056/" "$O6/wf/$w/") & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
}

evalhe() {  # gpu step -- JevBench 231 + hejev + external (as R055 evalarm), then MASSIVE/ScienceQA retention
  local gpu=$1 step=$2 port=$((8095 + $1)) id
  id=$(serve "$gpu" "$step" "$port" | tail -1)
  OUT=$O6/hejev jev r056 http://127.0.0.1:$port "$id" en translated mixed native
  OUT=$O6/hejev jev r056 http://127.0.0.1:$port "$id" $EXT
  killtree "$(cat "$O6/pids/serve$port")" || true
  (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$R $PY -m scripts.eval_gemma_mm_he --model "$GM" --adapter "$O6/serve" \
    --he-left data_he/eval/massive_he_test.jsonl --he-right data_he/eval/massive_en_test_paired.jsonl \
    --mm-dir data_v/eval --out "$O6/mmhe.json" >"$O6/logs/mmhe.log" 2>&1)
  echo done >"$O6/EVALHE_DONE"
}

"$@"
