#!/usr/bin/env bash
# Launch the hash-bound R050 internal GPU pipeline from the staged repository.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

REVISION=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
MODEL=${R050_MODEL:-/workspace/g1/hf-gemma/hub/models--google--gemma-4-12B-it/snapshots/$REVISION}
OUT=${R050_OUT:-$ROOT/.context/r050-gpu-run-0f5ceb5}
EXPECTED_PIPELINE_SHA256=a688a737160dee7ed6747259867f3a2ba6ae3f937e04a774192ce7f07b07288c

observed_pipeline_sha=$(sha256sum scripts/run_r050_pipeline.py | awk '{print $1}')
if [[ "$observed_pipeline_sha" != "$EXPECTED_PIPELINE_SHA256" ]]; then
  echo "run_r050_pipeline.py differs from the externally audited anchor" >&2
  exit 1
fi

code_files=(
  run_r050_pipeline.py
  build_r043_causal_pairs.py
  build_r044_mm_replay.py
  build_r047_model_inventory.py
  build_r048_action_panel.py
  build_r050_headroom.py
  build_r050_mix.py
  build_r050_replay_pools.py
  compare_r049_calibrated.py
  exact_policy_executor.py
  preflight_r043_assets.py
  prepare_r050_arms.py
  probe_r050_gradient_scale.py
  run_r050_arm.py
  run_r050_eval.py
  run_r050_headroom.py
  run_r050_retention.py
  score_exact_policy_panel.py
  singleton_option_scorer.py
  soft_action_executor.py
  train_gemma_decision_lora.py
)
code_args=()
for file_name in "${code_files[@]}"; do
  digest=$(sha256sum "scripts/$file_name" | awk '{print $1}')
  code_args+=(--expected-code-sha256 "$file_name=$digest")
done
mode_args=()
if [[ ${R050_DRY_RUN:-0} == 1 ]]; then
  mode_args+=(--dry-run)
fi

exec uv run --no-sync python scripts/run_r050_pipeline.py \
  "${mode_args[@]}" \
  --out "$OUT" \
  --headroom-data .context/r050-headroom-v2-audit/headroom.jsonl \
  --headroom-data-sha256 00c4010181b098f93c5ad7cf151dcd2f94cfe6acdcc9a888995a0b6b5f037f58 \
  --headroom-manifest .context/r050-headroom-v2-audit/manifest.json \
  --headroom-manifest-sha256 2a7b598fe9f41cd9d3c3c14792e19b86088f58f4d69fb32b0d38e92466a9d151 \
  --train-panel .context/r048-action-v1.6-audit/train.jsonl \
  --train-panel-sha256 086e461c1342ca05a9ddcc64bae2b35853f7bac1468598f861eb2df6817e6234 \
  --eval-panel .context/r048-action-v1.6-audit/eval.jsonl \
  --eval-panel-sha256 b764082343c68ec29bd94d9ca041c3e922b5aa421d40c06519683b804fd3a5e7 \
  --panel-manifest .context/r048-action-v1.6-audit/manifest.json \
  --panel-manifest-sha256 9a477c41999abf2dab8c5e00f156ed96c01d318b9fd54364802be0c3e6cd95bb \
  --mix .context/r050-training-mix-v2/train.jsonl \
  --mix-sha256 499dd9c9a8341e55d1d965bbd1ffc45c893880fcdab4d35d295d06324c1c2003 \
  --mix-manifest .context/r050-training-mix-v2/manifest.json \
  --mix-manifest-sha256 2b52da6fac16a803d5dd2fa14bcc54c68693f01754b3039561386602f287b53d \
  --model "$MODEL" \
  --model-inventory .context/r047-m1/model-inventory.json \
  --model-inventory-sha256 5c97d1ed2179841c5824733d6b1c83c2de300d8458a6eb22e50846a18ec7e605 \
  --parent-adapter .context/r044-rebuild/b/step-000100 \
  --parent-tree-sha256 44e1998efd10275a6f0ab20cc6ccbd97cb9a8ad9b8c3f620620a5f6fdcc91f99 \
  --parent-tree-files 4 \
  --retention-asset-inventory .context/r050-replay-pools/manifest.json \
  --retention-asset-inventory-sha256 d556492b58922a8f3a7e51b3f907611404e9c02fc92c0084f7e09e422a3f4e3b \
  --retention-suite he=data_he/eval/massive_he_test.jsonl \
  --retention-suite en=data_he/eval/massive_en_test_paired.jsonl \
  --retention-suite real=data_v/eval/scienceqa_test.jsonl \
  --retention-suite shuffled=data_v/eval/scienceqa_test_shuffled.jsonl \
  --retention-suite noimage=data_v/eval/scienceqa_test_noimage.jsonl \
  --retention-suite mismatch=data_v/eval/scienceqa_test_mismatch.jsonl \
  --retention-suite gold_dropped=data_v/eval/scienceqa_test_gold_dropped.jsonl \
  --retention-suite-sha256 he=c88f816b3fe4a8f27f6406cb8beee35716a8c9bbc7a0cf40425d3ae072370fee \
  --retention-suite-sha256 en=36d41a4e44e8c75e26a52a13ccaac2642879197a503ad3bf638ea45a98d003a4 \
  --retention-suite-sha256 real=05bc175eed370c6ec28ba6ce06cefbd0e881b516d1cfd3c1b32b7d91bd3fab11 \
  --retention-suite-sha256 shuffled=7f625f23343185eda85d685c30dd2d4dd4f27131c156a0232c377ee7453900cf \
  --retention-suite-sha256 noimage=a8af9435a9a5d6538dd8c15d05b93031fe1340a0d0ae337123f62875a27455c7 \
  --retention-suite-sha256 mismatch=850bc58e836fa9606b10303c7050fe079e0f063d106f1dd8441f10f0f643df13 \
  --retention-suite-sha256 gold_dropped=05e7684f73aa2a6bf2be45c7bb3c22d7eebcecd6d62438b98b8dc5f7e6915de2 \
  "${code_args[@]}"
