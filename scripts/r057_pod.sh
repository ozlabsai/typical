#!/usr/bin/env bash
# R057 on one 2-GPU pod: R055-A's run (r055_pod.sh train: same mix, args, seed) plus --bidirectional-prompt, then
# R055's evaluation with the mask on at serving. Repo (commit af5bc7b+) staged at /workspace/r054 like R055;
# results in /workspace/g1/r057 (the cost guard watches /workspace/g1).
#   xsetup                          pinned train venv, repo venv, jevbench, WorkflowEvals+typical patch, Gemma
#   train r057 <gpu> 2200 --bidirectional-prompt     (r055_pod.sh train)
#   evalarm <name> <gpu> <adapter-dir|arm-step>      hejev en/HE/MX/native + external panel, then MASSIVE/image
#                                   retention; BIDIR=1 serves and screens with the mask; MMHE=0 skips retention
#   wf <name> <gpu> <adapter-dir>   full WorkflowEvals (4 workflows, typical provider), BIDIR as above
set -euo pipefail
source "$(dirname "$0")/r055_pod.sh" true  # env, train(), jev(), killtree(), wait_url(), G pins, PY, GM, EXT
O5=/workspace/g1/r057; WE=$R/WorkflowEvals
mkdir -p "$O5/logs" "$O5/hejev"

xsetup() {
  local p=() x rc=0
  (uv venv -q /workspace/g1/.venv-pinned && uv pip install -q --python $PY torch==2.14.0 torchvision==0.29.0 \
    transformers==5.17.0 peft==0.21.0 accelerate==1.13.0 safetensors==0.8.0 sentencepiece==0.2.2) \
    >"$O5/logs/setup-pinned.log" 2>&1 & p+=($!)
  (cd "$R" && uv sync --frozen --no-dev) >"$O5/logs/setup-repo.log" 2>&1 & p+=($!)
  (git clone -q https://github.com/fstandhartinger/jevbench.git "$JB" && git -C "$JB" checkout -q bb05a335bc809e61b20c0f745d25499a82b326fc \
    && cd "$JB" && uv sync -q) >"$O5/logs/setup-jevbench.log" 2>&1 & p+=($!)
  (git clone -q https://github.com/typesafe-ai/WorkflowEvals.git "$WE" && git -C "$WE" checkout -q 0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb \
    && git -C "$WE" apply "$R/patches/workflowevals-typical-provider-0ac3b8ad.patch" && cd "$WE" && uv sync --locked -q) \
    >"$O5/logs/setup-we.log" 2>&1 & p+=($!)
  uvx --from huggingface_hub hf download "${G[0]}" --revision "${G[1]}" >"$O5/logs/dl-gemma.log" 2>&1 & p+=($!)
  for x in "${p[@]}"; do wait "$x" || rc=1; done
  test -f "$GM/config.json"; return $rc
}

stage() {  # name src -> $O5/serve-<name> (2-3 file adapter tree), echoes its tree sha256
  local srv=$O5/serve-$1; rm -rf "$srv"; mkdir -p "$srv"
  cp "$2"/{adapter_config.json,adapter_model.safetensors} "$srv/"; [[ -e $2/README.md ]] && cp "$2/README.md" "$srv/"
  cd "$R" && $PY -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$srv"
}

serve() {  # name gpu port key -- background server for $O5/serve-<name>; sets SPID and SID
  local sha; sha=$(stage "$1" "$SRC")
  SID=typical-$1-sha256-${sha:0:12}; echo "$SID $sha src=$SRC bidir=${BIDIR:-0}" >"$O5/$1.served-id.txt"
  (cd "$R" && CUDA_VISIBLE_DEVICES=$2 TYPICAL_API_KEY=$4 exec uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$O5/serve-$1" --expected-adapter-tree-sha256 "$sha" \
    --model-id "$SID" --receipt "$O5/$1.receipt-$3.json" --max-context 16384 --port "$3" \
    ${BIDIR:+--bidirectional-prompt} >"$O5/logs/serve-$1-$3.log" 2>&1) &
  SPID=$!
  # not wait_url: a keyed server answers 401 to a bare probe (R057's first WF runs waited 30 min on that and failed)
  for _ in $(seq 360); do
    curl -fsS ${4:+-H "Authorization: Bearer $4"} "http://127.0.0.1:$3/v1/models" >/dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || break; sleep 5
  done
  tail -40 "$O5/logs/serve-$1-$3.log"; return 1
}

evalarm() {  # name gpu src
  local name=$1 gpu=$2; SRC=$3
  [[ ${BIDIR:-0} == 1 ]] || unset BIDIR
  serve "$name" "$gpu" $((8095 + gpu)) ""
  OUT=$O5/hejev jev "$name" "http://127.0.0.1:$((8095 + gpu))" "$SID" en translated mixed native $EXT
  killtree $SPID
  if [[ ${MMHE:-1} == 1 ]]; then
    (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$R $PY -m scripts.eval_gemma_mm_he --model "$GM" --adapter "$O5/serve-$name" \
      --he-left data_he/eval/massive_he_test.jsonl --he-right data_he/eval/massive_en_test_paired.jsonl \
      --mm-dir data_v/eval --out "$O5/$name.mmhe.json" ${BIDIR:+--bidirectional-prompt} >"$O5/logs/mmhe-$name.log" 2>&1)
  fi
  echo done >"$O5/$name.EVAL_DONE"
}

wf() {  # name gpu src -- the typical provider with joint questions, as R050's official runbook (workers 4 here)
  local name=$1 gpu=$2 port=$((8097 + $2)) key wfn pids=(); SRC=$3
  [[ ${BIDIR:-0} == 1 ]] || unset BIDIR
  key=$(openssl rand -hex 16)
  serve "$name-wf" "$gpu" $port "$key"
  for wfn in "${!REV[@]}"; do
    (cd "$WE" && TYPICAL_API_KEY=$key exec uv run --frozen python run.py "$wfn" --model "typical:$SID" --name "r057-$name" \
      --base-url "http://127.0.0.1:$port" --questions all --workers 4 --question-workers 1 --timeout 1800 \
      --dataset-revision "${REV[$wfn]}" >"$O5/logs/wf-$name-$wfn.log" 2>&1) & pids+=($!)
  done
  local x; for x in "${pids[@]}"; do wait "$x" || true; done
  killtree $SPID
  for wfn in "${!REV[@]}"; do mkdir -p "$O5/wf-$name"; cp -r "$WE/runs/$wfn/r057-$name" "$O5/wf-$name/$wfn" 2>/dev/null || true; done
  echo done >"$O5/wf-$name.DONE"
}

"$@"
