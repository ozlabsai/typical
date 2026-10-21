#!/usr/bin/env bash
# R056: build the training mix ON THE POD (laptop uplink ~0.15 MB/s), then back it up to the private HF dataset.
#   r056_mixpod.sh   expects /workspace/r056_mixin.tgz (gen*/, twins/, r055 mix text, hebrew-bench, jevbench public)
#                    and /workspace/r056_img.tar (prebuilt ScienceQA rows + images)
set -euo pipefail
cd /workspace/r054
set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv HF_HUB_OFFLINE=0 WORKFLOWEVALS=/workspace/r054/WorkflowEvals
mkdir -p .context/r056/labels
cp /workspace/g1/r056/labels_gemma.jsonl /workspace/g1/r056/labels_qwen.jsonl .context/r056/labels/
tar -xf /workspace/r056_img.tar -C .context/r056/
tar -xzf /workspace/r056_mixin.tgz -C .
(cd WorkflowEvals && uv run --frozen --with pandas python ../scripts/r056_build.py mix) > /workspace/g1/r056/logs/mix.log 2>&1
cp .context/r056/mix/manifest.json /workspace/g1/r056/mix-manifest.json
# private HF backup: cases (meta.generator), teacher labels, train mix + dev splits, manifest
U=(uvx --from huggingface_hub hf upload guychuk/typical-r056-data --repo-type dataset --private)
D=.context/r056
for g in gen gen_gpt gen_gemma gen_qwen; do "${U[@]}" $D/$g $g --exclude "cache/*" >/dev/null; done
"${U[@]}" $D/twins twins --exclude "cache/*" >/dev/null
"${U[@]}" $D/labels labels >/dev/null
"${U[@]}" $D/mix mix >/dev/null
echo MIX_DONE
