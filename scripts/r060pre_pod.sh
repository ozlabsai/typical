#!/usr/bin/env bash
# R060-pre on one 4-GPU pod, eval only (docs/research/R060PRE_BLEND_PROXY.md). Repo staged at /workspace/r054
# (+ data_he/eval, data_v/{eval,images/test}, hejev/ splits, .context/r060pre/proxy_ids.txt); results in
# /workspace/g1/r060pre (the cost guard watches /workspace/g1).
#   psetup      r057 xsetup (pinned + repo venvs, jevbench, WorkflowEvals+typical patch, Gemma) + adapters + dev split;
#               R056 with every lora_B × λ -> adapters/lam000, lam050, lam075 (R056 itself = lam100)
#   proxy       dev teacher-NLL on the proxy ids, every adapter, 2 per GPU; then MASSIVE/image retention λ .5/.75
#   evalpair    λ .5 and .75 at once, one replica of each per GPU behind its own r056_lb.py:
#               WorkflowEvals 705 (R056 `wf` settings) || JevBench 231 + hejev + external panel
#   evalbase    λ 0, two replicas per GPU: WorkflowEvals 705 (JevBench/hejev at λ 0 are R057's base numbers)
#   all         psetup, proxy, evalpair, evalbase -> ALL_DONE
set -euo pipefail
source "$(dirname "$0")/r057_pod.sh" true  # xsetup, jev, killtree, wait_url, G, GM, PY, EXT, REV, WE
O=/workspace/g1/r060pre; A=$O/adapters; O5=$O
mkdir -p "$O/logs" "$O/pids" "$O/receipts" "$O/proxy" "$O/hejev"

psetup() {
  xsetup
  local hf=(uvx --from huggingface_hub hf download)
  "${hf[@]}" guychuk/typical-r056 adapter_config.json adapter_model.safetensors temperatures.json --local-dir "$O/dl/r056"
  "${hf[@]}" guychuk/typical-r057-bidir --include 'r057h-step2000/*' --local-dir "$O/dl/r057"
  "${hf[@]}" guychuk/typical-r060pre-adapters --local-dir "$O/dl/r060pre"
  "${hf[@]}" guychuk/typical-r056-data mix/dev_wf.jsonl --repo-type dataset --local-dir "$O/data"
  local s
  for s in lam000:0 lam050:0.5 lam075:0.75 lam100:1; do
    (cd "$R" && $PY scripts/r060pre_proxy.py scale "$O/dl/r056" "$A/${s%%:*}" "${s#*:}")
  done
  for s in r044r r055a r055b; do (cd "$R" && $PY scripts/r060pre_proxy.py scale "$O/dl/r060pre/$s" "$A/$s" 1); done
  (cd "$R" && $PY scripts/r060pre_proxy.py scale "$O/dl/r057/r057h-step2000" "$A/r057h" 1)
  touch "$O/SETUP_DONE"
}

dsel() {  # name gpu [devsel args] -- proxy teacher-NLL rows -> $O/proxy/<name>.json
  local n=$1 g=$2; shift 2
  (cd "$R" && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$R $PY scripts/r056_devsel.py --model "$GM" --adapter "$A/$n" \
    --dev "$O/data/mix/dev_wf.jsonl" --ids "$R/.context/r060pre/proxy_ids.txt" --out "$O/proxy/$n.json" "$@" \
    >"$O/logs/devsel-$n.log" 2>&1) || echo "FAILED devsel $n"
}

mmhe() {  # name gpu
  (cd "$R" && CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=$R $PY -m scripts.eval_gemma_mm_he --model "$GM" --adapter "$A/$1" \
    --he-left data_he/eval/massive_he_test.jsonl --he-right data_he/eval/massive_en_test_paired.jsonl \
    --mm-dir data_v/eval --out "$O/$1.mmhe.json" >"$O/logs/mmhe-$1.log" 2>&1) || echo "FAILED mmhe $1"
}

proxy() {
  { dsel lam000 0; mmhe lam050 0; } & { dsel lam050 0; } &
  { dsel lam075 1; mmhe lam075 1; } & { dsel lam100 1; } &
  { dsel r044r 2; } & { dsel r055a 2; } &
  { dsel r055b 3; } & { dsel r057h 3 --bidirectional-prompt; } &
  wait
  touch "$O/PROXY_DONE"
}

rep() {  # name gpu port -- one native-readout replica of $A/<name>, R056's temperatures (+ new served id)
  local d=$A/$1 sha id
  sha=$(cd "$R" && $PY -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$d")
  id=typical-r060-$1-sha256-${sha:0:12}; echo "$id $sha" >"$O/$1.served-id.txt"
  (cd "$R" && CUDA_VISIBLE_DEVICES=$2 TYPICAL_API_KEY= exec setsid uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$d" --expected-adapter-tree-sha256 "$sha" --model-id "$id" \
    --temperatures "$d/temperatures.json" --receipt "$O/receipts/$1-$3.json" --max-context 16384 --port "$3" \
    >"$O/logs/serve-$1-$3.log" 2>&1) &
  echo $! >"$O/pids/serve$3"
}

fleet() {  # name lbport port:gpu... -- replicas behind r056_lb.py on lbport
  local n=$1 lb=$2 urls=() pg u; shift 2
  for pg in "$@"; do rep "$n" "${pg#*:}" "${pg%%:*}"; urls+=("http://127.0.0.1:${pg%%:*}"); done
  for u in "${urls[@]}"; do
    wait_url "$u/v1/models" "$(cat "$O/pids/serve${u##*:}")" || { tail -20 "$O/logs/serve-$n-${u##*:}.log"; return 1; }
  done
  setsid python3 "$R/scripts/r056_lb.py" "$lb" "${urls[@]}" >"$O/logs/lb-$n.log" 2>&1 &
  echo $! >"$O/pids/lb$lb"
  wait_url "http://127.0.0.1:$lb/v1/models" "$(cat "$O/pids/lb$lb")"
}

stopall() {
  local f; for f in "$O"/pids/serve* "$O"/pids/lb*; do [[ -e $f ]] && { killtree "$(cat "$f")" || true; rm -f "$f"; }; done
  for _ in $(seq 60); do
    (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) < 2000 )) && return 0; sleep 3
  done
}

wfrun() {  # name lbport workers -- R056 `wf`: pinned revisions, typical:, questions all
  local n=$1 id w pids=(); read -r id _ <"$O/$n.served-id.txt"
  for w in "${!REV[@]}"; do
    (cd "$WE" && TYPICAL_API_KEY=local TYPICAL_BASE_URL=http://127.0.0.1:$2 uv run --frozen python run.py "$w" \
      --model "typical:$id" --name "r060-$n" --dataset-revision "${REV[$w]}" --questions all --workers "$3" \
      --question-workers 1 --timeout 3600 --retry-errors >"$O/logs/wf-$n-$w.log" 2>&1
     mkdir -p "$O/wf-$n"; rsync -a "$WE/runs/$w/r060-$n/" "$O/wf-$n/$w/") & pids+=($!)
  done
  for w in "${pids[@]}"; do wait "$w" || true; done
  touch "$O/WF_DONE-$n"
}

hev() {  # name lbport -- JevBench 231 + hejev translated/mixed/native + external panel
  local id; read -r id _ <"$O/$1.served-id.txt"
  OUT=$O/hejev jev "$1" "http://127.0.0.1:$2" "$id" en translated mixed native $EXT
}

evalpair() {
  fleet lam050 8090 8100:0 8101:1 8102:2 8103:3
  fleet lam075 8091 8110:0 8111:1 8112:2 8113:3
  wfrun lam050 8090 4 & wfrun lam075 8091 4 & { hev lam050 8090; hev lam075 8091; } &
  wait
  stopall
}

evalbase() {
  fleet lam000 8090 8100:0 8101:0 8102:1 8103:1 8104:2 8105:2 8106:3 8107:3
  wfrun lam000 8090 8
  stopall
}

all() {
  local s
  for s in psetup proxy evalpair evalbase; do
    echo "$(date -u +%FT%TZ) start $s"; "$s" || echo "$(date -u +%FT%TZ) FAILED rc=$? $s"
  done
  echo "$(date -u +%FT%TZ) ALL_DONE" | tee "$O/ALL_DONE"
}

"$@"
