#!/usr/bin/env bash
# R056 step 1: workflow-shaped cases with claude-haiku-5-5 (CLI), in chunks so each chunk is persisted.
#   scripts/r056_gen.sh <workflow> <start> <end> <per_call> [chunk]
set -euo pipefail
cd "$(dirname "$0")/../.context/research_sources/WorkflowEvals"
wf=$1 start=$2 end=$3 per=$4 chunk=${5:-40}
for ((s = start; s < end; s += chunk)); do
  n=$(( end - s < chunk ? end - s : chunk ))
  uv run --frozen python ../../../scripts/r053_generate_cases.py --workflows "$wf" --start "$s" --n "$n" \
    --per-call "$per" --backend "${BACKEND:-cli}" --model "${MODEL:-claude-haiku-5-5}" --effort "${EFFORT:-medium}" --out "../../r056/${GENDIR:-gen}" --tag "${TAG:-r056}" \
    --max-usd "${MAX_USD:-150}" --workers "${WORKERS:-8}" || echo "chunk $s failed rc=$?"
done
echo "GEN_DONE $wf $start $end"
