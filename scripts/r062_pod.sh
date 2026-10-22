#!/usr/bin/env bash
# R062 on one 4-GPU pod: Qwen3.8-27B-FP8 thinking labels on the Hebrew pool (scripts/r062_data.py), then ONE fresh
# Gemma-4-12B-it LoRA, then eval + two merges. Repo staged at /workspace/r054 (+ data_he/eval, data_v, hejev/ splits);
# results in /workspace/g1/r062 (the cost guard watches /workspace/g1). Label input/mix are scp'd from the laptop.
#   r62setup  R055 rsetup (WorkflowEvals+patches, jevbench, vLLM 0.31, pinned train venv, Gemma+Qwen) + R056 images,
#             R056 adapter, contested-fast-decisions
#   label     4 Qwen vLLMs (one per GPU), scripts/r056_label.py in-process, re-run on $O/label_in.jsonl (re-pushed
#             from the laptop as translation grows) until STOP_LABEL / LABEL_DEADLINE / FINAL_POOL fully labelled
#   mix       scripts/r062_data.py mix on the pod (pool + labels + static), data -> HF typical-hebrew-train r062/
#   base      frozen-base option probs into the anchors (meta.base_probs), 4 GPU shards
#   train     torchrun x4, fresh LoRA r32, lr 5e-5 -> 5e-6, max-len 8192, twin JS .5, KL-to-base 1.0 on anchors
#   post      devsel (Hebrew dev teacher-NLL) on the last third -> type temperatures -> evals -> merges -> HF model
#   all       setup, label, mix, base, train, post, then finish: results -> HF, SAFE_TO_STOP marker, self-stop
# Env: LABEL_DEADLINE, COST_DEADLINE (epoch s).
set -euo pipefail
source "$(dirname "$0")/r055_pod.sh" true  # rsetup, killtree, wait_url, jev, G, Q, GM, PY, EXT, REV, WE, JB
O=/workspace/g1/r062; M=$O/mix; A=$O/adapters; HFD=(uvx --from huggingface_hub hf download)
mkdir -p "$O/logs" "$O/pids" "$O/receipts" "$O/labels" "$O/hejev" "$O/contested" "$M" "$A"
mark() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/markers.txt"; }

r62setup() {  # not `setup`: r055 rsetup calls the phase-0 `setup` (overriding it recursed into a fork storm)
  rsetup
  "${HFD[@]}" guychuk/typical-r056-data --repo-type dataset --include 'mix/images/*' --local-dir "$O/dl056" >/dev/null
  ln -sfn "$O/dl056/mix/images" "$M/images"
  "${HFD[@]}" guychuk/typical-r056 adapter_config.json adapter_model.safetensors temperatures.json --local-dir "$A/r056"
  "${HFD[@]}" zoom-ai/contested-fast-decisions --repo-type dataset --local-dir "$O/cfd" >/dev/null
  touch "$O/SETUP_DONE"
}

qsrv() {  # gpu -- Qwen thinking vLLM on :802g (R056 srv flags)
  CUDA_VISIBLE_DEVICES=$1 PATH="$R/venv-vllm/bin:$PATH" setsid "$R/venv-vllm/bin/vllm" serve "${Q[0]}" --revision "${Q[1]}" \
    --served-model-name "${Q[0]}" --host 127.0.0.1 --port 802$1 --max-model-len 65536 --gpu-memory-utilization 0.92 \
    --enable-prefix-caching --max-num-seqs 256 --reasoning-parser qwen3 --gdn-prefill-backend triton \
    --max-num-batched-tokens 16384 --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
    >"$O/logs/qwen$1-vllm.log" 2>&1 &
  echo $! >"$O/pids/vllm$1"
}

label() {  # until STOP_LABEL, LABEL_DEADLINE (epoch s), or FINAL_POOL pushed and >= 99.5% of its questions answered
  local g urls=()
  for g in 0 1 2 3; do qsrv $g; urls+=(--base-url "http://127.0.0.1:802$g/v1"); done
  for g in 0 1 2 3; do wait_url "http://127.0.0.1:802$g/v1/models" "$(cat "$O/pids/vllm$g")" || { tail -30 "$O/logs/qwen$g-vllm.log"; return 1; }; done
  mark "label servers up"
  while :; do
    [[ -e $O/STOP_LABEL ]] && break
    local left=$(( ${LABEL_DEADLINE:-4102444800} - $(date +%s) ))
    (( left > 60 )) || break
    [[ -e $O/FINAL_POOL ]] && python3 - "$O" <<'PY' && break
import glob, json, sys
o = sys.argv[1]
done = {(a["id"], a["qid"]) for f in glob.glob(f"{o}/labels/*.jsonl") for l in open(f) if '"answer"' in l
        for a in [json.loads(l)] if "answer" in a}
want = {(r["id"], q) for l in open(f"{o}/label_in.jsonl") for r in [json.loads(l)] for q in r["questions"]}
print(f"labelled {len(want & done)}/{len(want)}", flush=True)
sys.exit(0 if len(want - done) <= 0.005 * len(want) else 1)
PY
    cp "$O/label_in.jsonl" "$O/label_in.run.jsonl"  # the laptop may replace label_in.jsonl mid-run
    (cd "$WE" && timeout "$left" uv run --frozen python "$R/scripts/r056_label.py" --data "$O/label_in.run.jsonl" \
      --out "$O/labels/qwen.jsonl" --model "vllm:${Q[0]}" "${urls[@]}" --workers 96 --workflowevals "$WE") >>"$O/logs/label.log" 2>&1 || true
    sleep 20
  done
  for g in 0 1 2 3; do killtree "$(cat "$O/pids/vllm$g")" || true; done
  pkill -f "vllm serve" || true
  for _ in $(seq 60); do (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) < 2000 )) && break; sleep 3; done
  mark "LABEL_DONE $(cat "$O"/labels/*.jsonl | grep -c '"answer"') answers"
}

mix() {  # on the pod: pool + labels + static -> $M, then the data to HF (guychuk/typical-hebrew-train r062/)
  python3 "$R/scripts/r062_data.py" mix --root "$O" >"$O/logs/mix.log" 2>&1
  mark "mix: $(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["presentations"], m["slots_with_twins"], m["share"])' "$M/manifest.json")"
  local d=$O/hfdata; rm -rf "$d"; mkdir -p "$d/mix"
  cp "$O"/{pool.jsonl,label_in.jsonl,pool_manifest.json,ts_twins.jsonl} "$d/" 2>/dev/null || true
  cp -r "$O/labels" "$O/static" "$d/"; cp "$M"/{train_text,anchors,img,dev_he}.jsonl "$M/manifest.json" "$d/mix/"
  { set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
  uvx --from huggingface_hub hf upload guychuk/typical-hebrew-train "$d" r062 --repo-type dataset \
    --commit-message "R062 pool, Qwen3.8-27B-think labels, mix" >"$O/logs/hf-data.log" 2>&1 || mark "FAILED hf data upload"
}

base() {  # anchors -> $M/anchors.base.jsonl, one shard per GPU
  local g pids=() sh=$O/shards
  mkdir -p "$sh"; split -n l/4 -d -a 1 "$M/anchors.jsonl" "$sh/in."
  for g in 0 1 2 3; do
    (cd "$R" && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$R $PY -m scripts.r060_precompute_base_probs --model "$GM" \
      --input "$sh/in.$g" --out "$sh/out.$g" >"$O/logs/base-$g.log" 2>&1) & pids+=($!)
  done
  for g in "${pids[@]}"; do wait "$g"; done
  cat "$sh"/out.? >"$M/anchors.base.jsonl"
  cat "$M/train_text.jsonl" "$M/anchors.base.jsonl" "$M/img.jsonl" >"$M/train.jsonl"
  mark "base: $(grep -c base_probs "$M/anchors.base.jsonl") anchored rows"
}

train() {
  local steps; steps=$(python3 -c 'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["slots_with_twins"] / 32))' "$M/manifest.json")
  echo "$steps" >"$O/steps.txt"; mkdir -p "$O/train"; mark "train $steps steps"
  cd "$R" && PYTHONPATH=$R /workspace/g1/.venv-pinned/bin/torchrun --standalone --nproc_per_node 4 \
    -m scripts.train_gemma_decision_lora --model "$GM" --train "$M/train.jsonl" --val he_dev="$M/dev_he.jsonl" \
    --out "$O/train" --steps "$steps" --rank 32 --alpha 64 --lr 5e-5 --min-lr 5e-6 --microbatch 1 --grad-accum 8 \
    --max-length 8192 --weight-decay .01 --seed 62 --eval-every 100 --save-every 50 --eval-samples 128 \
    --twin-consistency-weight 0.5 --kl-to-base-weight 1.0 --kl-to-base-on anchor --skip-overlength \
    >"$O/train/train.log" 2>&1
  echo done >"$O/train/TRAIN_DONE"; mark "TRAIN_DONE"
}

devsel() {  # step... -> $O/devsel/step-N.json (Hebrew dev teacher-NLL), round-robin over 4 GPUs
  mkdir -p "$O/devsel"
  local s k j steps=("$@")
  for ((k = 0; k < 4; k++)); do
    (for ((j = k; j < ${#steps[@]}; j += 4)); do s=${steps[j]}
      (cd "$R" && CUDA_VISIBLE_DEVICES=$k PYTHONPATH=$R $PY scripts/r056_devsel.py --model "$GM" --max-length 8192 \
        --adapter "$O/train/step-$(printf %06d "$s")" --dev "$M/dev_he.jsonl" --out "$O/devsel/step-$s.json" \
        >"$O/logs/devsel-$s.log" 2>&1) || echo "FAILED devsel $s"
    done) &
  done
  wait
}

tree_sha() { (cd "$R" && $PY -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$1"); }

rep() {  # name gpu port -- one native-readout replica of $A/<name> (its temperatures.json)
  local d=$A/$1 sha id
  sha=$(tree_sha "$d"); id=$1-sha256-${sha:0:12}  # served as typical:<id>
  echo "$id $sha" >"$O/$1.served-id.txt"
  (cd "$R" && CUDA_VISIBLE_DEVICES=$2 TYPICAL_API_KEY= exec setsid uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$d" --expected-adapter-tree-sha256 "$sha" --model-id "$id" \
    --temperatures "$d/temperatures.json" --receipt "$O/receipts/$1-$3.json" --max-context 16384 --port "$3" \
    >"$O/logs/serve-$1-$3.log" 2>&1) &
  echo $! >"$O/pids/serve$3"
}

fleet() {  # name lbport port:gpu... -- replicas behind r056_lb.py
  local n=$1 lb=$2 urls=() pg u; shift 2
  for pg in "$@"; do rep "$n" "${pg#*:}" "${pg%%:*}"; urls+=("http://127.0.0.1:${pg%%:*}"); done
  for u in "${urls[@]}"; do
    wait_url "$u/v1/models" "$(cat "$O/pids/serve${u##*:}")" || { tail -20 "$O/logs/serve-$n-${u##*:}.log"; return 1; }
  done
  setsid python3 "$R/scripts/r056_lb.py" "$lb" "${urls[@]}" >"$O/logs/lb-$n.log" 2>&1 &
  echo $! >"$O/pids/lb$lb"
  wait_url "http://127.0.0.1:$lb/v1/models" "$(cat "$O/pids/lb$lb")"
}

resume() {  # after the 17:42Z crash: eval stage onward (no retrain), fast first; finish + self-stop even on failure
  set +eu
  trap finish EXIT
  local n ms=() p=() id win
  pkill -f serve_typesafe_gemma_lora; pkill -f r056_lb.py; sleep 10; rm -f "$O"/pids/serve* "$O"/pids/lb*
  fleet r062 8090 8100:0 8101:0 8110:1 8111:1 8120:2 8121:2 8130:3 8131:3 || mark "FAILED fleet r062"
  read -r id _ <"$O/r062.served-id.txt"; mark "serving $id"
  OUT=$O/hejev jev r062 http://127.0.0.1:8090 "$id" en translated mixed native; mark "EVAL r062 hejev+en done"
  OUT=$O/hejev jev r062 http://127.0.0.1:8090 "$id" $EXT; mark "EVAL r062 external done"
  (cd "$R" && uv run --no-sync python scripts/contested_eval.py --data "$O/cfd/data/train.jsonl" --endpoint http://127.0.0.1:8090 \
    --model "$id" --out "$O/contested/r062.jsonl" >"$O/logs/contested-r062.log" 2>&1); mark "EVAL r062 contested done"
  stopall
  mmhe r062 0 & p+=($!)
  for n in r062mr056 r062mr060; do [[ -e $A/$n/adapter_model.safetensors ]] && ms+=("$n"); done
  [[ ${#ms[@]} -ge 1 ]] && { fleet "${ms[0]}" 8091 8210:1 8211:1 && { hev "${ms[0]}" 8091 & p+=($!); }; }
  [[ ${#ms[@]} -ge 2 ]] && { fleet "${ms[1]}" 8092 8220:2 8221:2 8230:3 8231:3 && { hev "${ms[1]}" 8092 & p+=($!); }; }
  for n in "${p[@]}"; do wait "$n"; done; mark "EVAL r062 retention done; merges done"
  stopall
  mmhe "${ms[0]}" 1 & p=($!); [[ ${#ms[@]} -ge 2 ]] && { mmhe "${ms[1]}" 2 & p+=($!); }
  for n in "${p[@]}"; do wait "$n"; done; mark "merge retention done"
  bench r062 "${ms[@]}"
  win=$(python3 - "$O" "${ms[@]}" <<'PY'
import json, statistics as st, sys
o, names = sys.argv[1], sys.argv[2:]
b = json.load(open(f"{o}/bench.json"))
def cfd(n):
    rows = [json.loads(l) for l in open(f"{o}/contested/{n}.jsonl")]
    return sum(bool(r["correct"]) for r in rows) / len(rows)
score = {n: st.mean([b[n]["he"], b[n]["mx"], b[n]["native"], b[n]["en"], cfd(n)]) for n in names}
json.dump(score, open(f"{o}/merge-pick.json", "w"), indent=1)
print(max(score, key=score.get) if score else "")
PY
)
  mark "merge pick: ${win:-none} $(cat "$O/merge-pick.json" 2>/dev/null | tr -d '\n ')"
  if [[ -n $win ]] && (( $(date +%s) + 9000 <= ${COST_DEADLINE:-0} )); then
    fleet "$win" 8090 8100:0 8101:0 8110:1 8111:1 8120:2 8121:2 8130:3 8131:3 && wfrun "$win" 8090 8; stopall
    mark "WF $win done"
  else mark "merge WF skipped (budget)"; fi
  cp "$O/best-step.txt" "$O/steps.txt" "$O/calibration.json" "$O/train/train.log" "$O/train/metrics.jsonl" "$O"/*.served-id.txt "$O/devsel/" 2>/dev/null
  { set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
  uvx --from huggingface_hub hf upload guychuk/typical-r062 "$O/devsel" devsel --repo-type model --private >>"$O/logs/hf-model.log" 2>&1
  mark "POST_DONE"; touch "$O/POST_DONE"
}

stopall() {
  local f; for f in "$O"/pids/serve* "$O"/pids/lb*; do [[ -e $f ]] && { killtree "$(cat "$f")" || true; rm -f "$f"; }; done
  for _ in $(seq 60); do (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) < 2000 )) && return 0; sleep 3; done
}

hev() {  # name lbport -- JevBench 231 + hejev translated/mixed/native + external panel, then contested
  local id; read -r id _ <"$O/$1.served-id.txt"
  OUT=$O/hejev jev "$1" "http://127.0.0.1:$2" "$id" en translated mixed native $EXT || mark "FAILED jev $1"
  (cd "$R" && uv run --no-sync python scripts/contested_eval.py --data "$O/cfd/data/train.jsonl" \
    --endpoint "http://127.0.0.1:$2" --model "$id" --out "$O/contested/$1.jsonl" >"$O/logs/contested-$1.log" 2>&1) ||
    mark "FAILED contested $1"
  mark "EVAL $1 done"
}

wfrun() {  # name lbport workers -- R056 `wf`: pinned revisions, typical:, questions all
  local n=$1 id w pids=(); read -r id _ <"$O/$n.served-id.txt"
  for w in "${!REV[@]}"; do
    (cd "$WE" && TYPICAL_API_KEY=local TYPICAL_BASE_URL=http://127.0.0.1:$2 uv run --frozen python run.py "$w" \
      --model "typical:$id" --name "r062-$n" --dataset-revision "${REV[$w]}" --questions all --workers "$3" \
      --question-workers 1 --timeout 3600 --retry-errors >"$O/logs/wf-$n-$w.log" 2>&1
     mkdir -p "$O/wf-$n"; rsync -a "$WE/runs/$w/r062-$n/" "$O/wf-$n/$w/"; mark "WF $n $w done") & pids+=($!)
  done
  for w in "${pids[@]}"; do wait "$w" || true; done
}

mmhe() {  # name gpu
  (cd "$R" && CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=$R $PY -m scripts.eval_gemma_mm_he --model "$GM" --adapter "$A/$1" \
    --he-left data_he/eval/massive_he_test.jsonl --he-right data_he/eval/massive_en_test_paired.jsonl \
    --mm-dir data_v/eval --out "$O/$1.mmhe.json" >"$O/logs/mmhe-$1.log" 2>&1) || mark "FAILED mmhe $1"
}

merges() {  # rank-concat merges with R062 (+ R062 type temperatures, + the partner's per-question WF temperatures)
  local p
  "${HFD[@]}" guychuk/typical-r060 adapter_config.json adapter_model.safetensors temperatures.json --local-dir "$A/r060" \
    >"$O/logs/dl-r060.log" 2>&1 || rm -rf "$A/r060"
  for p in r056 r060; do
    [[ -e $A/$p/adapter_model.safetensors ]] || { mark "no $p adapter: merge skipped"; continue; }
    (cd "$R" && $PY scripts/r060pre_proxy.py merge "$A/r062" 1 "$A/$p" .5 "$A/r062m$p") || continue
    python3 - "$A/r062/temperatures.json" "$A/$p/temperatures.json" "$A/r062m$p/temperatures.json" <<'PY'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
json.dump({**{k: v for k, v in b.items() if not k.startswith("type:")}, **{k: v for k, v in a.items() if k.startswith("type:")}},
          open(sys.argv[3], "w"), indent=1, sort_keys=True)
PY
  done
}

bench() {  # name... -> $O/bench.json (r060pre_proxy bench: EN 231, hejev, external)
  ln -sfn "$JB" "$R/.context/jevbench-v154"; mkdir -p "$R/.context/hebrew-bench"
  ln -sfn "$R/hejev" "$R/.context/hebrew-bench/hejev-v0"; ln -sfn "$R/hejev" "$R/.context/hebrew-bench/external"
  local n runs=(); for n in "$@"; do runs+=("$n=$O/hejev/$n"); done
  (cd "$R" && $PY scripts/r060pre_proxy.py bench "${runs[@]}" --out "$O/bench.json") | tee "$O/bench.md"
}

post() {
  local last steps best win
  last=$(ls -d "$O"/train/step-* | sed 's/.*step-0*//' | sort -n | tail -n 1)
  steps=$(ls -d "$O"/train/step-* | sed 's/.*step-0*//' | sort -n | awk -v l="$last" '$1 >= 2 * l / 3')
  mark "devsel on: $(echo $steps)"
  devsel $steps
  best=$(python3 - "$O/devsel" <<'PY'
import glob, json, sys
r = {int(f.split("step-")[1][:-5]): json.load(open(f))["summary"]["all"] for f in glob.glob(sys.argv[1] + "/step-*.json")}
json.dump(r, open(sys.argv[1] + "/selection.json", "w"), indent=1, sort_keys=True)
print(min(r, key=lambda s: r[s]["nll"]))
PY
)
  echo "$best" >"$O/best-step.txt"
  python3 "$R/scripts/r056_calibrate.py" "$O/devsel/step-$best.json" "$O/temperatures.json" "$M/dev_he.jsonl" >"$O/calibration.json"
  rm -rf "$A/r062"; mkdir -p "$A/r062"
  cp "$O/train/step-$(printf %06d "$best")"/{adapter_config.json,adapter_model.safetensors} "$O/temperatures.json" "$A/r062/"
  mark "best step $best; $(head -c 300 "$O/calibration.json" | tr '\n' ' ')"
  { set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
  local up=(uvx --from huggingface_hub hf upload guychuk/typical-r062)
  "${up[@]}" "$A/r062" . --repo-type model --private >"$O/logs/hf-model.log" 2>&1 || mark "FAILED hf adapter upload"
  # R062 itself: 8 replicas, fast evals; merges are built on CPU meanwhile
  fleet r062 8090 8100:0 8101:0 8110:1 8111:1 8120:2 8121:2 8130:3 8131:3
  merges & local mp=$!
  hev r062 8090
  stopall; wait $mp || true
  mmhe r062 0 & local p=($!) n ms=()
  for n in r062mr056 r062mr060; do [[ -e $A/$n/adapter_model.safetensors ]] && ms+=("$n"); done
  if [[ ${#ms[@]} -ge 1 ]]; then fleet "${ms[0]}" 8091 8210:1 8211:1; hev "${ms[0]}" 8091 & p+=($!); fi
  if [[ ${#ms[@]} -ge 2 ]]; then fleet "${ms[1]}" 8092 8220:2 8221:2 8230:3 8231:3; hev "${ms[1]}" 8092 & p+=($!); fi
  for n in "${p[@]}"; do wait "$n" || true; done  # pid-scoped: a bare wait also waits on the servers (R060-pre hang)
  stopall
  for n in "${ms[@]}"; do mmhe "$n" 0; done
  bench r062 "${ms[@]}"
  # the better merge (mean of HE, MX, native, EN 231, contested) gets a full WorkflowEvals if the budget allows
  win=$(python3 - "$O" "${ms[@]}" <<'PY'
import json, statistics as st, sys
o, names = sys.argv[1], sys.argv[2:]
b = json.load(open(f"{o}/bench.json"))
def cfd(n):
    rows = [json.loads(l) for l in open(f"{o}/contested/{n}.jsonl")]
    return sum(bool(r["correct"]) for r in rows) / len(rows)
score = {n: st.mean([b[n]["he"], b[n]["mx"], b[n]["native"], b[n]["en"], cfd(n)]) for n in names}
json.dump(score, open(f"{o}/merge-pick.json", "w"), indent=1)
print(max(score, key=score.get) if score else "")
PY
) || win=""
  mark "merge pick: ${win:-none}"
  touch "$O/FAST_DONE"
  if [[ -n $win ]] && (( $(date +%s) + 5400 <= ${COST_DEADLINE:-0} )); then
    fleet "$win" 8090 8100:0 8101:0 8110:1 8111:1 8120:2 8121:2 8130:3 8131:3
    wfrun "$win" 8090 8; stopall
  else mark "merge WF skipped (budget)"; fi
  cp "$O/best-step.txt" "$O/steps.txt" "$O/calibration.json" "$O/train/train.log" "$O/train/metrics.jsonl" "$O"/*.served-id.txt "$O/devsel/" 2>/dev/null
  "${up[@]}" "$O/devsel" devsel --repo-type model --private >>"$O/logs/hf-model.log" 2>&1
  mark "POST_DONE"; touch "$O/POST_DONE"
}

finish() {  # (trap-safe: set +eu) everything the laptop needs -> private HF model repo results/, then the final copy marker + self-stop
  set +eu; trap - EXIT
  local r=$O/results
  rm -rf "$r"; mkdir -p "$r"
  rsync -a --exclude 'raw/' --exclude 'ledger/' --exclude 'tasks/' "$O/hejev" "$O/contested" "$r/" 2>/dev/null || true
  rsync -a "$O"/wf-* "$r/" 2>/dev/null || true
  cp "$O"/*.json "$O"/*.md "$O"/*.txt "$O"/steps.txt "$M/manifest.json" "$r/" 2>/dev/null || true
  cp "$O/train/train.log" "$O/train/metrics.jsonl" "$r/" 2>/dev/null || true
  mkdir -p "$r/logs"; cp "$O"/logs/{label,mix,hf-model,hf-data}.log "$O"/logs/wf-*.log "$r/logs/" 2>/dev/null || true
  for n in r062mr056 r062mr060; do [[ -e $A/$n/temperatures.json ]] && mkdir -p "$r/$n" && cp "$A/$n/adapter_config.json" "$A/$n/temperatures.json" "$r/$n/"; done
  { set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
  local k ok=0
  for k in 1 2 3; do
    uvx --from huggingface_hub hf upload guychuk/typical-r062 "$r" results --repo-type model --private >"$O/logs/hf-results.log" 2>&1 && { ok=1; break; }
    sleep 60
  done
  (( ok )) && mark "COPY_DONE (results on guychuk/typical-r062 results/)" || mark "FAILED results upload x3: stopping anyway"
  touch /workspace/g1/SAFE_TO_STOP
  # container env (RUNPOD_POD_ID, RUNPOD_API_KEY) is not visible in SSH-launched shells: read it from PID 1
  export $(tr '\0' '\n' </proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs) 2>/dev/null || true
  [[ -e /workspace/g1/HOLD ]] && { mark "HOLD present: not self-stopping"; return; }
  command -v runpodctl >/dev/null && [[ -n ${RUNPOD_POD_ID:-} ]] && { mark "self-stop $RUNPOD_POD_ID"; runpodctl stop pod "$RUNPOD_POD_ID"; }
}

all() {
  local s
  for s in r62setup label mix base train post; do
    mark "start $s"; $s || { mark "FAILED rc=$? $s"; [[ $s == post || $s == label ]] || break; }
  done
  mark "ALL_DONE"; touch "$O/ALL_DONE"
  finish
}

"$@"
