#!/usr/bin/env bash
# R056 after training, unattended: dev teacher-NLL selection -> per-question temperatures (dev only) -> full eval
# -> private HF model backup. Waits for $O6/train/TRAIN_DONE.   setsid nohup bash scripts/r056_post.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/r056_pod.sh true
until [[ -e $O6/train/TRAIN_DONE ]]; do sleep 60; done
last=$(ls -d "$O6"/train/step-* | sed 's/.*step-0*//' | sort -n | tail -n 1)
steps=$(ls -d "$O6"/train/step-* | sed 's/.*step-0*//' | sort -n | awk -v l="$last" '$1 >= l - 500')
echo "$(date -u +%FT%TZ) devsel on: $steps"
devsel $steps
best=$(python3 - "$O6/devsel" <<'PY'
import json, sys, glob
r = {int(f.split("step-")[1][:-5]): json.load(open(f))["summary"]["all"]["nll"] for f in glob.glob(sys.argv[1] + "/step-*.json")}
print(min(r, key=r.get))
PY
)
echo "$(date -u +%FT%TZ) best step $best"; echo "$best" >"$O6/best-step.txt"
python3 scripts/r056_calibrate.py "$O6/devsel/step-$best.json" "$O6/temperatures.json" >"$O6/calibration.json"
evalall "$best"
set -a; . /workspace/.env; set +a
uvx --from huggingface_hub hf upload guychuk/typical-r056 "$O6/serve" . --repo-type model --private >"$O6/logs/hf-model.log" 2>&1
uvx --from huggingface_hub hf upload guychuk/typical-r056 "$O6/devsel" devsel --repo-type model --private >>"$O6/logs/hf-model.log" 2>&1
cp "$O6/served-id.txt" "$O6/calibration.json" "$O6/best-step.txt" "$O6/devsel/" 2>/dev/null
uvx --from huggingface_hub hf upload guychuk/typical-r056 "$O6/devsel" devsel --repo-type model --private >>"$O6/logs/hf-model.log" 2>&1
echo "$(date -u +%FT%TZ) POST_DONE"
