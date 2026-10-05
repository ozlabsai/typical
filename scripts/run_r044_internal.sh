#!/bin/zsh
# Run R044's matched text-only versus pixel-replay continuation pilot.
set -euo pipefail

ROOT=${R044_ROOT:-$PWD}
RUN=${R044_RUN:-$ROOT/r044}
MODEL=${R044_MODEL:?set R044_MODEL to the staged fresh Gemma base}
PARENT=${R044_PARENT_ADAPTER:?set R044_PARENT_ADAPTER to the frozen parent adapter}
PYTHON=${R044_PYTHON:-$ROOT/.venv-pinned/bin/python}
MANIFEST_SHA=${R044_MANIFEST_SHA:?set R044_MANIFEST_SHA to the frozen corpus manifest hash}
ASSETS_MANIFEST=${R044_ASSETS_MANIFEST:?set R044_ASSETS_MANIFEST to the frozen assets manifest}
ASSETS_MANIFEST_SHA=${R044_ASSETS_MANIFEST_SHA:?set R044_ASSETS_MANIFEST_SHA to its hash}
PARENT_ADAPTER_SHA=${R044_PARENT_ADAPTER_SHA:?set R044_PARENT_ADAPTER_SHA to the parent tree hash}
MODEL_REVISION=${R044_MODEL_REVISION:?set R044_MODEL_REVISION to the pinned base revision}
EXPECTED_MODEL_REVISION=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
HE_LEFT=${R044_HE_LEFT:?set R044_HE_LEFT to the frozen Hebrew panel}
HE_RIGHT=${R044_HE_RIGHT:?set R044_HE_RIGHT to the paired English panel}
MM_DIR=${R044_MM_DIR:?set R044_MM_DIR to the frozen image-panel directory}
PAIR_EVAL=${R044_PAIR_EVAL:?set R044_PAIR_EVAL to the frozen R043 pair panel}
PAIR_EVAL_SHA=${R044_PAIR_EVAL_SHA:?set R044_PAIR_EVAL_SHA to the pair panel hash}
RENDERER_EVAL_SHA=${R044_RENDERER_EVAL_SHA:?set R044_RENDERER_EVAL_SHA to the renderer panel hash}

cd "$ROOT"
[[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]]
[[ -d "$MODEL" && -d "$PARENT" && -x "$PYTHON" ]]
[[ -f "$RUN/manifest.json" && -f "$RUN/arm-a.jsonl" && -f "$RUN/arm-b.jsonl" ]]
[[ -f "$RUN/val-wf.jsonl" && -f "$RUN/val-wh.jsonl" && -f "$RUN/val-replay.jsonl" ]]
[[ -f "$RUN/action-cal.jsonl" && -f "$RUN/action-eval.jsonl" && -f "$RUN/renderer-probe-eval.jsonl" ]]
[[ -f "$HE_LEFT" && -f "$HE_RIGHT" && -d "$MM_DIR" ]]
[[ -f "$PAIR_EVAL" && -f "$ASSETS_MANIFEST" ]]
[[ ! -e "$RUN/a" && ! -e "$RUN/b" ]]
[[ ! -e "$RUN/pilot-finalized" && ! -e "$RUN/pilot-promoted" ]]
[[ $(sha256sum "$RUN/manifest.json" | awk '{print $1}') == "$MANIFEST_SHA" ]]
[[ $(sha256sum "$ASSETS_MANIFEST" | awk '{print $1}') == "$ASSETS_MANIFEST_SHA" ]]
[[ $(sha256sum "$PAIR_EVAL" | awk '{print $1}') == "$PAIR_EVAL_SHA" ]]
[[ $(sha256sum "$RUN/renderer-probe-eval.jsonl" | awk '{print $1}') == "$RENDERER_EVAL_SHA" ]]
[[ $("$PYTHON" -c 'import sys; from scripts.preflight_r043_assets import tree_hash; print(tree_hash(sys.argv[1])[0])' "$PARENT") == "$PARENT_ADAPTER_SHA" ]]
[[ $(wc -l < "$RUN/arm-a.jsonl" | tr -d ' ') == 800 ]]
[[ $(wc -l < "$RUN/arm-b.jsonl" | tr -d ' ') == 800 ]]
[[ $(sha256sum "$RUN/arm-a.jsonl" | awk '{print $1}') == $(jq -r '.outputs["arm-a"].sha256' "$RUN/manifest.json") ]]
[[ $(sha256sum "$RUN/arm-b.jsonl" | awk '{print $1}') == $(jq -r '.outputs["arm-b"].sha256' "$RUN/manifest.json") ]]
"$PYTHON" -c 'import json, sys; from pathlib import Path; from scripts.build_r044_mm_replay import _tree_hash; p=Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text()); assert list(_tree_hash(p/"images")) == [m["images"]["tree_sha256"], m["images"]["files"]]' "$RUN"
"$PYTHON" -m scripts.preflight_r043_assets --manifest "$ASSETS_MANIFEST" \
  --model "$MODEL" --val-wf "$RUN/val-wf.jsonl" --val-wh "$RUN/val-wh.jsonl" \
  --val-replay "$RUN/val-replay.jsonl" --action-cal "$RUN/action-cal.jsonl" \
  --action-eval "$RUN/action-eval.jsonl" --he-left "$HE_LEFT" --he-right "$HE_RIGHT" \
  --mm "$MM_DIR" --out "$RUN/assets-preflight.json"

# B runs first because pixel replay is the decision-relevant arm.
for arm in b a; do
  "$PYTHON" -m scripts.train_gemma_decision_lora --model "$MODEL" --init-adapter "$PARENT" \
    --train "$RUN/arm-$arm.jsonl" \
    --val wf="$RUN/val-wf.jsonl" --val wh="$RUN/val-wh.jsonl" --val replay="$RUN/val-replay.jsonl" \
    --out "$RUN/$arm" --steps 100 --microbatch 1 --grad-accum 8 --max-length 8192 \
    --lr 5e-6 --min-lr 5e-7 --weight-decay .01 --save-every 100 --eval-every 100 \
    --eval-samples 128 --seed 44 --pair-flip-weight .25 --pair-flip-margin 1 \
    --pair-flip-start-step 0
done

for arm in parent a b; do
  adapter="$PARENT"
  [[ "$arm" != parent ]] && adapter="$RUN/$arm/step-000100"
  "$PYTHON" scripts/eval_gemma_decision_lora.py --model "$MODEL" --adapter "$adapter" \
    --val wf="$RUN/val-wf.jsonl" --val wh="$RUN/val-wh.jsonl" --val replay="$RUN/val-replay.jsonl" \
    --samples 128 --max-length 8192 --seed 44 > "$RUN/$arm-retention.json"
  "$PYTHON" -m scripts.score_action_panel --model "$MODEL" --adapter "$adapter" \
    --data "$RUN/action-cal.jsonl" --out "$RUN/$arm-action-cal"
  "$PYTHON" -m scripts.score_action_panel --model "$MODEL" --adapter "$adapter" \
    --data "$RUN/action-eval.jsonl" --calibration-data "$RUN/action-cal.jsonl" \
    --calibration-predictions "$RUN/$arm-action-cal/predictions.jsonl" --limit 256 \
    --out "$RUN/$arm-action"
  "$PYTHON" -m scripts.probe_action_renderer --model "$MODEL" --adapter "$adapter" \
    --data "$RUN/renderer-probe-eval.jsonl" --out "$RUN/$arm-renderer" --per-label 65
  "$PYTHON" -m scripts.eval_gemma_mm_he --model "$MODEL" --adapter "$adapter" \
    --he-left "$HE_LEFT" --he-right "$HE_RIGHT" --mm-dir "$MM_DIR" \
    --out "$RUN/$arm-mmhe.json"
  "$PYTHON" -m scripts.eval_r043_pairs --model "$MODEL" --model-revision "$MODEL_REVISION" \
    --adapter "$adapter" --data "$PAIR_EVAL" --out "$RUN/$arm-pairs"
done

if "$PYTHON" -m scripts.finalize_r044 --manifest "$RUN/manifest.json" \
  --parent-adapter "$PARENT" \
  --a-config "$RUN/a/config.json" --a-metrics "$RUN/a/metrics.jsonl" \
  --b-config "$RUN/b/config.json" --b-metrics "$RUN/b/metrics.jsonl" \
  --parent-retention "$RUN/parent-retention.json" --a-retention "$RUN/a-retention.json" \
  --b-retention "$RUN/b-retention.json" \
  --parent-action "$RUN/parent-action/report.json" --a-action "$RUN/a-action/report.json" \
  --b-action "$RUN/b-action/report.json" \
  --parent-renderer "$RUN/parent-renderer/report.json" --a-renderer "$RUN/a-renderer/report.json" \
  --b-renderer "$RUN/b-renderer/report.json" \
  --parent-mmhe "$RUN/parent-mmhe.json" --a-mmhe "$RUN/a-mmhe.json" --b-mmhe "$RUN/b-mmhe.json" \
  --parent-pairs "$RUN/parent-pairs/report.json" \
  --a-pairs "$RUN/a-pairs/report.json" --b-pairs "$RUN/b-pairs/report.json" \
  --out "$RUN/final-gates.json"; then
  touch "$RUN/pilot-promoted"
else
  status=$?
  [[ $status == 2 ]] || exit $status
fi
touch "$RUN/pilot-finalized"
