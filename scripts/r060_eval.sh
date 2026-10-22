#!/usr/bin/env bash
# R060 post, reordered (coordinator): selection + calibration, then the FAST evals first on GPUs 0-1
# (EN 231 -> hejev translated/mixed/native -> contested -> external panel -> retention) while full WorkflowEvals 705
# runs on GPUs 2-3; GPUs 0-1 join the WF fleet once the fast evals are done (r056_lb.py @file backends).
#   setsid nohup bash scripts/r060_eval.sh run > /workspace/g1/r060/eval.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/r060_pod.sh" true  # O, M, stage, devsel, killtree, wait_url, jev, G, GM, PY, EXT, REV, WE
mark() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/eval-markers.txt"; }

select_best() {  # devsel jsons -> best step (wf dev NLL + anchor-dev KL), temperatures on its wf dev
  until (( $(ls "$O"/devsel/dev_wf-*.json "$O"/devsel/anchor_dev.base-*.json 2>/dev/null | wc -l) >= 8 )); do sleep 30; done
  python3 - "$O/devsel" >"$O/best-step.txt" <<'PY'
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
  python3 "$R/scripts/r056_calibrate.py" "$O/devsel/dev_wf-$(cat "$O/best-step.txt").json" "$O/temperatures.json" >"$O/calibration.json"
  mark "best step $(cat "$O/best-step.txt")"
}

rep() {  # gpu k -- one native-readout replica of $O/serve on :8100+10g+k; prints its URL once up
  local id sha port=$((8100 + 10 * $1 + $2))
  read -r id sha _ <"$O/served-id.txt"
  (cd "$R" && CUDA_VISIBLE_DEVICES=$1 TYPICAL_API_KEY= exec setsid uv run --no-sync python scripts/serve_typesafe_gemma_lora.py \
    --model "${G[0]}" --model-revision "${G[1]}" --adapter "$O/serve" --expected-adapter-tree-sha256 "$sha" --model-id "$id" \
    --temperatures "$O/serve/temperatures.json" --receipt "$O/receipts/$port.json" --max-context 16384 --port "$port" \
    >"$O/logs/serve-$port.log" 2>&1) &
  echo $! >"$O/pids/serve$port"
  wait_url "http://127.0.0.1:$port/v1/models" "$(cat "$O/pids/serve$port")" || { tail -20 "$O/logs/serve-$port.log" >&2; return 1; }
  echo "http://127.0.0.1:$port"
}

fleet() {  # file gpu... -- 2 replicas per GPU appended to the backend file as they come up
  local f=$1 g p pids=(); shift
  for g in "$@"; do rep "$g" 0 >>"$f" & pids+=($!); rep "$g" 1 >>"$f" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || true; done  # pid-scoped: a bare wait would also wait on the LBs
}

lb() {  # port file
  setsid python3 "$R/scripts/r056_lb.py" "$1" "@$2" >"$O/logs/lb-$1.log" 2>&1 &
  echo $! >"$O/pids/lb$1"
  wait_url "http://127.0.0.1:$1/v1/models" "$(cat "$O/pids/lb$1")"
}

stopgpu() {  # gpu -- its replicas
  local k; for k in 0 1; do local f=$O/pids/serve$((8100 + 10 * $1 + k)); [[ -e $f ]] && { killtree "$(cat "$f")" || true; rm -f "$f"; }; done
  for _ in $(seq 60); do (( $(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits) < 2000 )) && return 0; sleep 3; done
}

wfrun() {  # id -- R056 `wf` settings on the WF fleet (:8091)
  local w pids=()
  for w in "${!REV[@]}"; do
    (cd "$WE" && TYPICAL_API_KEY=local TYPICAL_BASE_URL=http://127.0.0.1:8091 uv run --frozen python run.py "$w" \
      --model "typical:$1" --name r060 --dataset-revision "${REV[$w]}" --questions all --workers 8 \
      --question-workers 1 --timeout 3600 --retry-errors >"$O/logs/wf-$w.log" 2>&1
     mkdir -p "$O/wf/$w"; rsync -a "$WE/runs/$w/r060/" "$O/wf/$w/"; mark "WF done $w") & pids+=($!)
  done
  for w in "${pids[@]}"; do wait "$w" || true; done
}

run() {
  local id wfp
  select_best
  stage "$(cat "$O/best-step.txt")"; read -r id _ <"$O/served-id.txt"
  mkdir -p "$O/hejev" "$O/wf" "$O/contested"; : >"$O/backends-wf.txt"; : >"$O/backends-fast.txt"
  date -u +%FT%TZ >"$O/EVAL.START"
  fleet "$O/backends-wf.txt" 2 3; lb 8091 "$O/backends-wf.txt"
  fleet "$O/backends-fast.txt" 0 1; lb 8090 "$O/backends-fast.txt"
  mark "fleets up: $id"
  wfrun "$id" & wfp=$!
  OUT=$O/hejev jev r060 http://127.0.0.1:8090 "$id" en || mark "FAILED jev en"; mark "FAST en"
  OUT=$O/hejev jev r060 http://127.0.0.1:8090 "$id" translated mixed native || mark "FAILED jev translated mixed native"; mark "FAST hejev"
  (cd "$R" && uv run --no-sync python scripts/contested_eval.py --data "$O/cfd/data/train.jsonl" \
    --endpoint http://127.0.0.1:8090 --model "$id" --out "$O/contested/r060.jsonl" >"$O/logs/contested.log" 2>&1) ||
    mark "FAILED contested"
  mark "FAST contested"
  OUT=$O/hejev jev r060 http://127.0.0.1:8090 "$id" $EXT || mark "FAILED jev $EXT"; mark "FAST external"
  killtree "$(cat "$O/pids/lb8090")" || true; stopgpu 0; stopgpu 1
  (rep 1 0 >>"$O/backends-wf.txt"; rep 1 1 >>"$O/backends-wf.txt") &  # GPU 1 joins WF during retention
  (cd "$R" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R $PY -m scripts.eval_gemma_mm_he --model "$GM" --adapter "$O/serve" \
    --he-left data_he/eval/massive_he_test.jsonl --he-right data_he/eval/massive_en_test_paired.jsonl \
    --mm-dir data_v/eval --out "$O/mmhe.json" >"$O/logs/mmhe.log" 2>&1) || mark "FAILED mmhe"
  mark "FAST retention"
  rep 0 0 >>"$O/backends-wf.txt" || true; rep 0 1 >>"$O/backends-wf.txt" || true; mark "GPU 0-1 joined WF"
  wait $wfp || true
  stopall; killtree "$(cat "$O/pids/lb8091")" || true
  date -u +%FT%TZ >"$O/EVAL.DONE"
  { set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
  local up=(uvx --from huggingface_hub hf upload guychuk/typical-r060)
  "${up[@]}" "$O/serve" . --repo-type model --private >"$O/logs/hf-model.log" 2>&1
  cp "$O/served-id.txt" "$O/calibration.json" "$O/best-step.txt" "$O/steps.txt" "$O/train/train.log" "$O/devsel/"
  "${up[@]}" "$O/devsel" devsel --repo-type model --private >>"$O/logs/hf-model.log" 2>&1
  mark "POST_DONE"; touch "$O/POST_DONE" "$O/ALL_DONE"
}

"$@"
