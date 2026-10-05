#!/bin/zsh
# Run only R043's frozen internal causal-pair pilot and preservation screens.
set -euo pipefail

ROOT=${R043_ROOT:-$PWD}
RUN=${R043_RUN:-$ROOT/r043}
MODEL=${R043_MODEL:?set R043_MODEL to the staged fresh Gemma base}
PYTHON=${R043_PYTHON:-$ROOT/.venv-pinned/bin/python}
MANIFEST_SHA=${R043_MANIFEST_SHA:?set R043_MANIFEST_SHA to the frozen manifest hash}
ASSETS_MANIFEST_SHA=${R043_ASSETS_MANIFEST_SHA:?set R043_ASSETS_MANIFEST_SHA to the frozen assets manifest hash}
MODEL_REVISION=${R043_MODEL_REVISION:?set R043_MODEL_REVISION to the pinned base revision}
EXPECTED_MODEL_REVISION=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
ACTION_EVAL_SHA=a477f8347c0d2a6ac4898e7c6e119f69450669b44b2943d1d55aa3089b27a297
ACTION_CAL_SHA=82afc31f76e7a11dad0749de8b0ba29622740affc39f54af1353cf5a1f5b8f60
HE_LEFT=${R043_HE_LEFT:?set R043_HE_LEFT to the frozen Hebrew panel}
HE_RIGHT=${R043_HE_RIGHT:?set R043_HE_RIGHT to the paired English panel}
MM_DIR=${R043_MM_DIR:?set R043_MM_DIR to the frozen image-panel directory}

cd "$ROOT"
[[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]]
[[ -d "$MODEL" && -x "$PYTHON" ]]
[[ -f "$RUN/manifest.json" && -f "$RUN/assets-manifest.json" && -f "$RUN/action-cal.jsonl" && -f "$RUN/action-eval.jsonl" ]]
[[ -f "$RUN/val-wf.jsonl" && -f "$RUN/val-wh.jsonl" && -f "$RUN/val-replay.jsonl" ]]
[[ -f "$HE_LEFT" && -f "$HE_RIGHT" && -d "$MM_DIR" ]]
[[ ! -e "$RUN/a" && ! -e "$RUN/b" && ! -e "$RUN/c" && ! -e "$RUN/d" ]]
[[ ! -e "$RUN/pilot-finalized" && ! -e "$RUN/pilot-promoted" ]]
[[ $(sha256sum "$RUN/manifest.json" | awk '{print $1}') == "$MANIFEST_SHA" ]]
[[ $(sha256sum "$RUN/assets-manifest.json" | awk '{print $1}') == "$ASSETS_MANIFEST_SHA" ]]
[[ $(sha256sum "$RUN/action-cal.jsonl" | awk '{print $1}') == "$ACTION_CAL_SHA" ]]
[[ $(sha256sum "$RUN/action-eval.jsonl" | awk '{print $1}') == "$ACTION_EVAL_SHA" ]]
"$PYTHON" -m scripts.preflight_r043 --run "$RUN" > "$RUN/preflight.json"
"$PYTHON" -m scripts.preflight_r043_assets --manifest "$RUN/assets-manifest.json" \
  --model "$MODEL" --val-wf "$RUN/val-wf.jsonl" --val-wh "$RUN/val-wh.jsonl" \
  --val-replay "$RUN/val-replay.jsonl" --action-cal "$RUN/action-cal.jsonl" \
  --action-eval "$RUN/action-eval.jsonl" --he-left "$HE_LEFT" --he-right "$HE_RIGHT" \
  --mm "$MM_DIR" --out "$RUN/assets-preflight.json"

for arm in a b c d; do
  flip=0
  [[ "$arm" == b || "$arm" == c ]] && flip=.25
  "$PYTHON" -m scripts.train_gemma_decision_lora --model "$MODEL" --train "$RUN/arm-$arm.jsonl" \
    --val wf="$RUN/val-wf.jsonl" --val wh="$RUN/val-wh.jsonl" --val replay="$RUN/val-replay.jsonl" \
    --out "$RUN/$arm" --steps 350 --microbatch 1 --grad-accum 8 --max-length 8192 \
    --lr 2e-5 --min-lr 2e-6 --weight-decay .01 --rank 32 --alpha 64 \
    --save-every 350 --eval-every 350 --eval-samples 128 --seed 43 \
    --pair-flip-weight "$flip" --pair-flip-margin 1.0 --pair-flip-start-step 201 \
    --stage-counterfactual-pairs
done

"$PYTHON" -m scripts.audit_r043_training \
  --a-config "$RUN/a/config.json" --a-metrics "$RUN/a/metrics.jsonl" \
  --b-config "$RUN/b/config.json" --b-metrics "$RUN/b/metrics.jsonl" \
  --c-config "$RUN/c/config.json" --c-metrics "$RUN/c/metrics.jsonl" \
  --d-config "$RUN/d/config.json" --d-metrics "$RUN/d/metrics.jsonl" \
  --out "$RUN/training-audit.json"

for arm in base a b c d; do
  adapter=()
  [[ "$arm" != base ]] && adapter=(--adapter "$RUN/$arm/step-000350")
  "$PYTHON" -m scripts.eval_r043_pairs --model "$MODEL" --model-revision "$MODEL_REVISION" \
    "${adapter[@]}" --data "$RUN/eval.jsonl" --out "$RUN/$arm-pairs"
done

"$PYTHON" -m scripts.compare_r043_pilot --data "$RUN/eval.jsonl" \
  --model-revision "$MODEL_REVISION" \
  --base-predictions "$RUN/base-pairs/predictions.jsonl" --base-report "$RUN/base-pairs/report.json" \
  --a-predictions "$RUN/a-pairs/predictions.jsonl" --a-report "$RUN/a-pairs/report.json" \
  --b-predictions "$RUN/b-pairs/predictions.jsonl" --b-report "$RUN/b-pairs/report.json" \
  --c-predictions "$RUN/c-pairs/predictions.jsonl" --c-report "$RUN/c-pairs/report.json" \
  --d-predictions "$RUN/d-pairs/predictions.jsonl" --d-report "$RUN/d-pairs/report.json" \
  --seed 43 --out "$RUN/pair-comparison.json"

for arm in base a b; do
  adapter=()
  [[ "$arm" != base ]] && adapter=(--adapter "$RUN/$arm/step-000350")
  "$PYTHON" scripts/eval_gemma_decision_lora.py --model "$MODEL" "${adapter[@]}" \
    --val wf="$RUN/val-wf.jsonl" --val wh="$RUN/val-wh.jsonl" --val replay="$RUN/val-replay.jsonl" \
    --samples 128 --max-length 8192 --seed 43 > "$RUN/$arm-retention.json"
  "$PYTHON" -m scripts.score_action_panel --model "$MODEL" "${adapter[@]}" \
    --data "$RUN/action-cal.jsonl" --out "$RUN/$arm-action-cal"
  "$PYTHON" -m scripts.score_action_panel --model "$MODEL" "${adapter[@]}" \
    --data "$RUN/action-eval.jsonl" --calibration-data "$RUN/action-cal.jsonl" \
    --calibration-predictions "$RUN/$arm-action-cal/predictions.jsonl" --limit 256 \
    --out "$RUN/$arm-action"
done

"$PYTHON" -m scripts.eval_gemma_mm_he --model "$MODEL" --adapter "$RUN/b/step-000350" \
  --he-left "$HE_LEFT" --he-right "$HE_RIGHT" --mm-dir "$MM_DIR" --out "$RUN/b-mmhe.json"

if "$PYTHON" -m scripts.finalize_r043 --comparison "$RUN/pair-comparison.json" \
  --training-audit "$RUN/training-audit.json" --assets "$RUN/assets-preflight.json" \
  --model-revision "$MODEL_REVISION" \
  --base-retention "$RUN/base-retention.json" --a-retention "$RUN/a-retention.json" \
  --b-retention "$RUN/b-retention.json" \
  --base-action "$RUN/base-action/report.json" --a-action "$RUN/a-action/report.json" \
  --b-action "$RUN/b-action/report.json" \
  --b-mmhe "$RUN/b-mmhe.json" --out "$RUN/final-gates.json"; then
  touch "$RUN/pilot-promoted"
else
  status=$?
  [[ $status == 2 ]] || exit $status
fi
touch "$RUN/pilot-finalized"
