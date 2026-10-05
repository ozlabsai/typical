#!/bin/zsh
# Run R040's frozen internal treatment/control sequence on an already-staged GPU workspace.
set -euo pipefail

ROOT=${R040_ROOT:-$PWD}
MODEL=${R040_MODEL:-$ROOT/hf-gemma/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}
RUN=${R040_RUN:-$ROOT/r040}
PROBE_SHA=a477f8347c0d2a6ac4898e7c6e119f69450669b44b2943d1d55aa3089b27a297
PYTHON=${R040_PYTHON:-$ROOT/.venv-pinned/bin/python}

cd "$ROOT"
[[ -f "$RUN/replay.jsonl" && -f "$RUN/renderer-probe-eval.jsonl" ]]
[[ ! -e "$RUN/complete" && ! -e "$RUN/partial" && ! -e "$RUN/internal-finalized" ]]
[[ $(sha256sum "$RUN/renderer-probe-eval.jsonl" | awk '{print $1}') == "$PROBE_SHA" ]]

"$PYTHON" -m scripts.build_r040_orbits --out "$RUN" --replay "$RUN/replay.jsonl"
for arm in complete partial; do
  "$PYTHON" -m scripts.train_gemma_decision_lora --model "$MODEL" --train "$RUN/$arm.jsonl" \
    --val r040="$RUN/eval.jsonl" --val wf="$RUN/val-wf.jsonl" --val wh="$RUN/val-wh.jsonl" \
    --val replay="$RUN/val-replay.jsonl" --out "$RUN/$arm" --steps 444 --microbatch 1 --grad-accum 8 \
    --max-length 8192 --lr 2e-5 --min-lr 2e-6 --weight-decay .01 --rank 32 --alpha 64 \
    --save-every 444 --eval-every 444 --eval-samples 128 --seed 42
  "$PYTHON" -m scripts.eval_r038_orbits --model "$MODEL" --adapter "$RUN/$arm/step-000444" \
    --data "$RUN/eval.jsonl" --out "$RUN/$arm-orbit-audit"
  "$PYTHON" -m scripts.probe_action_renderer --model "$MODEL" --adapter "$RUN/$arm/step-000444" \
    --data "$RUN/renderer-probe-eval.jsonl" --out "$RUN/$arm-renderer-probe" --per-label 65
done
"$PYTHON" -m scripts.compare_r038_orbits --data "$RUN/eval.jsonl" \
  --treatment "$RUN/complete-orbit-audit/predictions.jsonl" --control "$RUN/partial-orbit-audit/predictions.jsonl" \
  --out "$RUN/orbit-compare.json"
"$PYTHON" -m scripts.compare_renderer_probe --treatment "$RUN/complete-renderer-probe/report.json" \
  --control "$RUN/partial-renderer-probe/report.json" \
  --treatment-predictions "$RUN/complete-renderer-probe/predictions.jsonl" \
  --control-predictions "$RUN/partial-renderer-probe/predictions.jsonl" --out "$RUN/renderer-compare.json"
touch "$RUN/internal-finalized"
