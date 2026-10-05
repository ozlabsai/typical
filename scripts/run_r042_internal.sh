#!/bin/zsh
# Run R042's frozen record-card versus prose screen on a staged GPU workspace.
set -euo pipefail

ROOT=${R042_ROOT:-$PWD}
MODEL=${R042_MODEL:-$ROOT/hf-gemma/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}
RUN=${R042_RUN:-$ROOT/r042}
PYTHON=${R042_PYTHON:-$ROOT/.venv-pinned/bin/python}
PROBE_SHA=a477f8347c0d2a6ac4898e7c6e119f69450669b44b2943d1d55aa3089b27a297

cd "$ROOT"
[[ -f "$RUN/replay.jsonl" && -f "$RUN/renderer-probe-eval.jsonl" ]]
[[ ! -e "$RUN/card" && ! -e "$RUN/prose" && ! -e "$RUN/internal-finalized" ]]
[[ $(sha256sum "$RUN/renderer-probe-eval.jsonl" | awk '{print $1}') == "$PROBE_SHA" ]]

"$PYTHON" -m scripts.build_r042_record_cards --out "$RUN" --replay "$RUN/replay.jsonl"
[[ $(wc -l < "$RUN/replay.jsonl" | tr -d ' ') == 1488 ]]
[[ $(wc -l < "$RUN/card.jsonl" | tr -d ' ') == 3600 ]]
[[ $(wc -l < "$RUN/prose.jsonl" | tr -d ' ') == 3600 ]]
[[ $(jq -r '.rows.card, .rows.prose, .rows["eval-card"], .rows["eval-prose"]' "$RUN/manifest.json" | tr '\n' ' ') == '3600 3600 3072 3072 ' ]]
for arm in card prose; do
  "$PYTHON" -m scripts.train_gemma_decision_lora --model "$MODEL" --train "$RUN/$arm.jsonl" \
    --val wf="$RUN/val-wf.jsonl" --val wh="$RUN/val-wh.jsonl" --val replay="$RUN/val-replay.jsonl" \
    --out "$RUN/$arm" --steps 450 --microbatch 1 --grad-accum 8 --max-length 8192 \
    --lr 2e-5 --min-lr 2e-6 --weight-decay .01 --rank 32 --alpha 64 \
    --save-every 450 --eval-every 450 --eval-samples 128 --seed 42
  for renderer in card prose; do
    "$PYTHON" -m scripts.eval_r038_orbits --model "$MODEL" --adapter "$RUN/$arm/step-000450" \
      --data "$RUN/eval-$renderer.jsonl" --out "$RUN/$arm-$renderer-audit"
  done
  "$PYTHON" -m scripts.probe_action_renderer --model "$MODEL" --adapter "$RUN/$arm/step-000450" \
    --data "$RUN/renderer-probe-eval.jsonl" --out "$RUN/$arm-renderer-probe" --per-label 65
done

"$PYTHON" -m scripts.audit_r042_representation --card-data "$RUN/eval-card.jsonl" \
  --prose-data "$RUN/eval-prose.jsonl" \
  --card-card-predictions "$RUN/card-card-audit/predictions.jsonl" --card-card-report "$RUN/card-card-audit/report.json" \
  --card-prose-predictions "$RUN/card-prose-audit/predictions.jsonl" --card-prose-report "$RUN/card-prose-audit/report.json" \
  --prose-card-predictions "$RUN/prose-card-audit/predictions.jsonl" --prose-card-report "$RUN/prose-card-audit/report.json" \
  --prose-prose-predictions "$RUN/prose-prose-audit/predictions.jsonl" --prose-prose-report "$RUN/prose-prose-audit/report.json" \
  --out "$RUN/representation-compare.json"
"$PYTHON" -m scripts.compare_renderer_probe --treatment "$RUN/card-renderer-probe/report.json" \
  --control "$RUN/prose-renderer-probe/report.json" \
  --treatment-predictions "$RUN/card-renderer-probe/predictions.jsonl" \
  --control-predictions "$RUN/prose-renderer-probe/predictions.jsonl" --out "$RUN/renderer-compare.json"
touch "$RUN/internal-finalized"
