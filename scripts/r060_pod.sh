#!/usr/bin/env bash
# R060 on one 4-GPU pod: anchored continuation of R056 on astra-relabelled workflow rows (scripts/r060_build.py).
# Repo staged at /workspace/r054 (+ data_he/eval, data_v/{eval,images/test}, hejev/ splits); results in
# /workspace/g1/r060 (the cost guard watches /workspace/g1). Data from private HF guychuk/typical-r060-data mix/.
#   setup     r057 xsetup + R056 adapter (init) + R056 images + anchors + contested-fast-decisions
#   base      frozen-base option probs into the anchors (meta.base_probs), 4 GPU shards; then anchor_dev
#   waitmix   poll HF for mix/READY (pushed from the laptop once the astra labels are in), download the mix
#   train     torchrun x4 from the R056 adapter, KL-to-base on anchors, twin JS .5, steps = slots / 32
#   post      devsel (wf dev teacher-NLL + anchor-dev KL) on the last third -> temperatures -> eval -> HF model
#   all       setup, base, waitmix, train, post -> ALL_DONE
set -euo pipefail
source "$(dirname "$0")/r057_pod.sh" true  # xsetup, jev, killtree, wait_url, G, GM, PY, EXT, REV, WE
O=/workspace/g1/r060; O5=$O; M=$O/mix; HFD=(uvx --from huggingface_hub hf download)
mkdir -p "$O/logs" "$O/pids" "$O/receipts" "$M"

setup() {
  xsetup
  "${HFD[@]}" guychuk/typical-r056 adapter_config.json adapter_model.safetensors --local-dir "$O/init"
  "${HFD[@]}" guychuk/typical-r056-data --repo-type dataset --include 'mix/images/*' --local-dir "$O/dl056" >/dev/null
  ln -sfn "$O/dl056/mix/images" "$M/images"
  "${HFD[@]}" guychuk/typical-r060-data mix/anchors.jsonl mix/anchor_dev.jsonl --repo-type dataset --local-dir "$O/dl"
  "${HFD[@]}" zoom-ai/contested-fast-decisions --repo-type dataset --local-dir "$O/cfd" >/dev/null
  touch "$O/SETUP_DONE"
}

base() {  # name -- $O/dl/mix/<name>.jsonl -> $M/<name>.base.jsonl, one shard per GPU
  local n=$1 g pids=() sh=$O/shards/$1
  mkdir -p "$sh"; split -n l/4 -d -a 1 "$O/dl/mix/$n.jsonl" "$sh/in."
  for g in 0 1 2 3; do
    (cd "$R" && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$R $PY -m scripts.r060_precompute_base_probs --model "$GM" \
      --input "$sh/in.$g" --out "$sh/out.$g" >"$O/logs/base-$n-$g.log" 2>&1) & pids+=($!)
  done
  for g in "${pids[@]}"; do wait "$g"; done
  cat "$sh"/out.? >"$M/$n.base.jsonl"
  echo "$(date -u +%FT%TZ) base $n: $(grep -c base_probs "$M/$n.base.jsonl") anchored rows"
}

waitmix() {
  until "${HFD[@]}" guychuk/typical-r060-data mix/READY --repo-type dataset --local-dir "$O/dl" >/dev/null 2>&1; do sleep 60; done
  "${HFD[@]}" guychuk/typical-r060-data mix/wf_train.jsonl mix/img.jsonl mix/dev_wf.jsonl mix/dev_he.jsonl \
    mix/manifest.json --repo-type dataset --local-dir "$O/dl"
  cp "$O"/dl/mix/{wf_train,img,dev_wf,dev_he}.jsonl "$O/dl/mix/manifest.json" "$M/"
  cat "$M/anchors.base.jsonl" "$M/wf_train.jsonl" "$M/img.jsonl" >"$M/train.jsonl"
}

train() {  # [extra trainer args]
  local steps; steps=$(python3 -c 'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["slots_with_twins"] / 32))' "$M/manifest.json")
  echo "$steps" >"$O/steps.txt"; mkdir -p "$O/train"
  cd "$R" && PYTHONPATH=$R /workspace/g1/.venv-pinned/bin/torchrun --standalone --nproc_per_node 4 \
    -m scripts.train_gemma_decision_lora --model "$GM" --init-adapter "$O/init" --train "$M/train.jsonl" \
    --val wf_dev="$M/dev_wf.jsonl" --val anchor_dev="$M/anchor_dev.base.jsonl" --out "$O/train" --steps "$steps" \
    --rank 32 --alpha 64 --lr 3e-5 --min-lr 3e-6 --microbatch 1 --grad-accum 8 --max-length 16384 --weight-decay .01 \
    --seed 60 --eval-every 200 --save-every 100 --eval-samples 96 --twin-consistency-weight 0.5 \
    --kl-to-base-weight 1.0 --kl-to-base-on anchor --skip-overlength "$@" >"$O/train/train.log" 2>&1
  echo done >"$O/train/TRAIN_DONE"
}

dsel() {  # step dev gpu -> $O/devsel/<dev>-<step>.json
  (cd "$R" && CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=$R $PY scripts/r056_devsel.py --model "$GM" \
    --adapter "$O/train/step-$(printf %06d "$1")" --dev "$M/$2.jsonl" --out "$O/devsel/$2-$1.json" \
    >"$O/logs/devsel-$2-$1.log" 2>&1) || echo "FAILED devsel $2 $1"
}

devsel() {  # step... -- wf dev + anchor dev per checkpoint, jobs round-robin over 4 GPUs
  mkdir -p "$O/devsel"
  local jobs=() s k j
  for s in "$@"; do jobs+=("$s dev_wf" "$s anchor_dev.base"); done
  for ((k = 0; k < 4; k++)); do
    (for ((j = k; j < ${#jobs[@]}; j += 4)); do set -- ${jobs[j]}; dsel "$1" "$2" $k; done) &
  done
  wait
}

stage() {  # step -- adapter tree (+ temperatures.json); served id typical:r060-sha256-<12>
  local srv=$O/serve sha
  rm -rf "$srv"; mkdir -p "$srv"
  cp "$O/train/step-$(printf %06d "$1")"/{adapter_config.json,adapter_model.safetensors} "$srv/"
  [[ -e $O/temperatures.json ]] && cp "$O/temperatures.json" "$srv/"
  sha=$(cd "$R" && $PY -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$srv")
  echo "r060-sha256-${sha:0:12} $sha step=$1" >"$O/served-id.txt"
}

replicas() {  # per_gpu -- native-readout replicas of $O/serve (:8100+) behind r056_lb.py (:8090)
  local srv=$O/serve id sha g k port urls=()
  read -r id sha _ <"$O/served-id.txt"
  for g in 0 1 2 3; do for k in $(seq 0 $(($1 - 1))); do
    port=$((8100 + 10 * g + k)); urls+=(http://127.0.0.1:$port)
    (cd "$R" && CUDA_VISIBLE_DEVICES=$g TYPICAL_API_KEY= exec setsid uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
      --model "${G[0]}" --model-revision "${G[1]}" --adapter "$srv" --expected-adapter-tree-sha256 "$sha" --model-id "$id" \
      --temperatures "$srv/temperatures.json" --receipt "$O/receipts/$port.json" --max-context 16384 --port "$port" \
      >"$O/logs/serve-$port.log" 2>&1) &
    echo $! >"$O/pids/serve$port"
  done; done
  for u in "${urls[@]}"; do wait_url "$u/v1/models" "$(cat "$O/pids/serve${u##*:}")" || { tail -20 "$O/logs/serve-${u##*:}.log"; return 1; }; done
  setsid python3 "$R/scripts/r056_lb.py" 8090 "${urls[@]}" >"$O/logs/lb.log" 2>&1 &
  echo $! >"$O/pids/lb"
  wait_url http://127.0.0.1:8090/v1/models "$(cat "$O/pids/lb")"
}

stopall() {
  local f; for f in "$O"/pids/serve* "$O/pids/lb"; do [[ -e $f ]] && { killtree "$(cat "$f")" || true; rm -f "$f"; }; done
}

wf() {  # id -- R056 `wf`: pinned revisions, typical:, questions all
  local w pids=()
  for w in "${!REV[@]}"; do
    (cd "$WE" && TYPICAL_API_KEY=local TYPICAL_BASE_URL=http://127.0.0.1:8090 uv run --frozen python run.py "$w" \
      --model "typical:$1" --name r060 --dataset-revision "${REV[$w]}" --questions all --workers 8 \
      --question-workers 1 --timeout 3600 --retry-errors >"$O/logs/wf-$w.log" 2>&1
     mkdir -p "$O/wf/$w"; rsync -a "$WE/runs/$w/r060/" "$O/wf/$w/") & pids+=($!)
  done
  for w in "${pids[@]}"; do wait "$w" || true; done
}

evalall() {  # step -- WorkflowEvals 705 || JevBench 231 + hejev + external, then contested; retention last
  local id
  stage "$1"; replicas 2; read -r id _ <"$O/served-id.txt"
  mkdir -p "$O/hejev" "$O/wf" "$O/contested"
  date -u +%FT%TZ >"$O/EVAL.START"
  wf "$id" & local wfp=$!
  OUT=$O/hejev jev r060 http://127.0.0.1:8090 "$id" en translated mixed native $EXT
  (cd "$R" && uv run --no-sync python scripts/contested_eval.py --data "$O/cfd/data/train.jsonl" \
    --endpoint http://127.0.0.1:8090 --model "$id" --out "$O/contested/r060.jsonl" >"$O/logs/contested.log" 2>&1) ||
    echo "FAILED contested"
  wait $wfp || true
  stopall
  (cd "$R" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R $PY -m scripts.eval_gemma_mm_he --model "$GM" --adapter "$O/serve" \
    --he-left data_he/eval/massive_he_test.jsonl --he-right data_he/eval/massive_en_test_paired.jsonl \
    --mm-dir data_v/eval --out "$O/mmhe.json" >"$O/logs/mmhe.log" 2>&1) || echo "FAILED mmhe"
  date -u +%FT%TZ >"$O/EVAL.DONE"
}

post() {
  local last steps best
  last=$(ls -d "$O"/train/step-* | sed 's/.*step-0*//' | sort -n | tail -n 1)
  steps=$(ls -d "$O"/train/step-* | sed 's/.*step-0*//' | sort -n | awk -v l="$last" '$1 >= 2 * l / 3')
  echo "$(date -u +%FT%TZ) devsel on: $steps"
  devsel $steps
  best=$(python3 - "$O/devsel" <<'PY'
import glob, json, sys
d = sys.argv[1]
score = {}
for f in glob.glob(f"{d}/dev_wf-*.json"):
    s = f.rsplit("-", 1)[1][:-5]
    a = json.load(open(f"{d}/anchor_dev.base-{s}.json"))["summary"]["all"]
    w = json.load(open(f))["summary"]["all"]
    score[int(s)] = {"wf_nll": w["nll"], "wf_acc": w["acc"], "anchor_kl": a["kl"], "anchor_nll": a["nll"], "sum": w["nll"] + a["kl"]}
json.dump(score, open(f"{d}/selection.json", "w"), indent=1, sort_keys=True)
print(min(score, key=lambda s: score[s]["sum"]))
PY
)
  echo "$(date -u +%FT%TZ) best step $best"; echo "$best" >"$O/best-step.txt"
  python3 "$R/scripts/r056_calibrate.py" "$O/devsel/dev_wf-$best.json" "$O/temperatures.json" >"$O/calibration.json"
  evalall "$best"
  { set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
  local up=(uvx --from huggingface_hub hf upload guychuk/typical-r060)
  "${up[@]}" "$O/serve" . --repo-type model --private >"$O/logs/hf-model.log" 2>&1
  cp "$O/served-id.txt" "$O/calibration.json" "$O/best-step.txt" "$O/steps.txt" "$O/train/train.log" "$O/devsel/"
  "${up[@]}" "$O/devsel" devsel --repo-type model --private >>"$O/logs/hf-model.log" 2>&1
  echo "$(date -u +%FT%TZ) POST_DONE" | tee "$O/POST_DONE"
}

all() {
  local s
  for s in setup "base anchors" "base anchor_dev" waitmix train post; do
    echo "$(date -u +%FT%TZ) start $s"; $s || { echo "$(date -u +%FT%TZ) FAILED rc=$? $s"; [[ $s == train || $s == waitmix ]] && break; }
  done
  echo "$(date -u +%FT%TZ) ALL_DONE" | tee "$O/ALL_DONE"
}

"$@"
