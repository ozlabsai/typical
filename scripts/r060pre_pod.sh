#!/usr/bin/env bash
# R060-pre on one multi-GPU pod (2 or 4), eval only (docs/research/R060PRE_BLEND_PROXY.md). Repo staged at /workspace/r054
# (+ data_he/eval, data_v/{eval,images/test}, hejev/ splits, .context/r060pre/proxy_ids.txt); results in
# /workspace/g1/r060pre (the cost guard watches /workspace/g1).
#   psetup      r057 xsetup (pinned + repo venvs, jevbench, WorkflowEvals+typical patch, Gemma) + adapters + dev split;
#               R056 with every lora_B × λ -> adapters/lam000, lam050, lam075 (R056 itself = lam100)
#   proxy       dev teacher-NLL on the proxy ids, every adapter, 2 per GPU; then MASSIVE/image retention λ .5/.75
#   evalpair    λ .5 and .75 at once, one replica of each per GPU behind its own r056_lb.py:
#               WorkflowEvals 705 (R056 `wf` settings) || JevBench 231 + hejev + external panel
#   evalbase    λ 0, two replicas per GPU: WorkflowEvals 705 (JevBench/hejev at λ 0 are R057's base numbers)
#   all         psetup, proxy, evalpair, evalbase -> ALL_DONE
#   finish      (own process) after ALL_DONE: zoom-ai/contested-fast-decisions for λ .5/.75/0/1 (CONTESTED_DONE),
#               then merge: M1/M2 rank-concat merges of R056 + R055-B (proxy, retention, hejev/EN/external), their
#               WF only if `r060pre_proxy.py gate` passes -> MERGE_DONE
set -euo pipefail
source "$(dirname "$0")/r057_pod.sh" true  # xsetup, jev, killtree, wait_url, G, GM, PY, EXT, REV, WE
O=/workspace/g1/r060pre; A=$O/adapters; O5=$O; NG=$(nvidia-smi -L | wc -l)
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

proxy() {  # the jobs round-robin over 2 slots per GPU
  local jobs=("dsel lam000" "dsel lam050" "dsel lam075" "dsel lam100" "dsel r044r" "dsel r055a" "dsel r055b"
              "dsel r057h --bidirectional-prompt" "mmhe lam050" "mmhe lam075") k j
  for ((k = 0; k < 2 * NG; k++)); do
    (for ((j = k; j < ${#jobs[@]}; j += 2 * NG)); do set -- ${jobs[j]}; "$1" "$2" $((k % NG)) "${@:3}"; done) &
  done
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

ports() {  # base per_gpu -> port:gpu list
  local g k; for ((g = 0; g < NG; g++)); do for ((k = 0; k < $2; k++)); do echo "$(($1 + 10 * g + k)):$g"; done; done
}

evalpair() {  # WF workers = replicas, i.e. 4 requests in flight per replica as R056
  fleet lam050 8090 $(ports 8100 1)
  fleet lam075 8091 $(ports 8200 1)
  wfrun lam050 8090 "$NG" & wfrun lam075 8091 "$NG" & { hev lam050 8090; hev lam075 8091; } &
  wait
  stopall
}

evalbase() {
  fleet lam000 8090 $(ports 8100 2)
  wfrun lam000 8090 $((2 * NG))
  stopall
}

cfd() {  # name port -- zoom-ai/contested-fast-decisions (EVAL ONLY) through one served adapter
  local id; read -r id _ <"$O/$1.served-id.txt"
  (cd "$R" && uv run --no-sync python scripts/contested_eval.py --data "$O/data/cfd/data/train.jsonl" \
    --endpoint "http://127.0.0.1:$2" --model "$id" --out "$O/contested/$1.jsonl" >"$O/logs/contested-$1.log" 2>&1) ||
    echo "FAILED contested $1"
}


merge() {  # M1 = R056 + .5·R055-B, M2 = .75·R056 + .5·R055-B (rank concat, fixed in advance): proxy, retention,
  # JevBench/hejev/external; full WF only if `r060pre_proxy.py gate` passes (coordinator rule, budget)
  (cd "$R" && $PY scripts/r060pre_proxy.py merge "$O/dl/r056" 1 "$A/r055b" .5 "$A/m1" &&
    $PY scripts/r060pre_proxy.py merge "$O/dl/r056" .75 "$A/r055b" .5 "$A/m2")
  dsel m1 0 & dsel m2 1 & mmhe m1 0 & mmhe m2 1 & wait
  fleet m1 8090 $(ports 8100 1)
  fleet m2 8091 $(ports 8200 1)
  hev m1 8090 & hev m2 8091 & wait
  stopall
  local win
  win=$(cd "$R" && $PY scripts/r060pre_proxy.py gate "$O/proxy" "$O/hejev" --out "$O/merge-gate.json" | tail -1)
  echo "merge gate: ${win:-fail}"
  if [[ $win == m1 || $win == m2 ]]; then fleet "$win" 8090 $(ports 8100 2); wfrun "$win" 8090 $((2 * NG)); stopall; fi
  touch "$O/MERGE_DONE"
}

cfdpair() {  # a b -- contested on dedicated replicas, a on GPU 0 and b on GPU 1
  fleet "$1" 8090 8100:0
  fleet "$2" 8091 8110:$((NG - 1))
  cfd "$1" 8090 & cfd "$2" 8091 & wait
  stopall
}

finish() {  # separate process, after the chain's ALL_DONE: contested-fast-decisions (λ .5, .75, 0, 1), then merge
  # bench/gate on the pod read jevbench + the hejev splits from the local-checkout paths
  ln -sfn "$JB" "$R/.context/jevbench-v154"; mkdir -p "$R/.context/hebrew-bench"
  ln -sfn "$R/hejev" "$R/.context/hebrew-bench/hejev-v0"; ln -sfn "$R/hejev" "$R/.context/hebrew-bench/external"
  until [[ -e $O/ALL_DONE ]]; do sleep 30; done
  cfdpair lam050 lam075
  cfdpair lam000 lam100
  touch "$O/CONTESTED_DONE"
  merge
}

all() {
  local s
  for s in psetup proxy evalpair evalbase; do
    echo "$(date -u +%FT%TZ) start $s"; "$s" || echo "$(date -u +%FT%TZ) FAILED rc=$? $s"
  done
  echo "$(date -u +%FT%TZ) ALL_DONE" | tee "$O/ALL_DONE"
}

"$@"
