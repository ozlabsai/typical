# R054 Phase 0 chain as finally run (under tmux on the pod, after `r054_pod_phase0.sh setup`).
# Qwen3.8-27B-FP8 ran ~4x slower per question than Gemma on vLLM 0.31/H100, so it was cut to FRACTION=3 (budget).
cd /workspace/r054
S=scripts/r054_pod_phase0.sh
G="google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
Q="Qwen/Qwen3.8-27B-FP8 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
D="LiquidAI/d1-3B 051bcc464b01b9f92942b364d9586b0ef5912432"
step() { echo "$(date -u +%FT%TZ) start $*"; "$@"; echo "$(date -u +%FT%TZ) end rc=$? $*"; }
step bash $S llm gemma4-12b-think $G default gemma4
FRACTION=3 step bash $S llm qwen38-27b-think $Q default qwen3 --gdn-prefill-backend triton --max-num-batched-tokens 16384
step bash $S d1 d1-3b $D
echo CHAIN_DONE
