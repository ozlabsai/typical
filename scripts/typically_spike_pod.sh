#!/bin/bash
# typically spike, one GPU pod: released typical-small vs per-company fine-tunes (scripts/typically_spike.py).
# Expects the repo + data_co_a/ + data_co_b/ at /workspace/pcdm, secrets in /workspace/.env.
# bash scripts/typically_spike_pod.sh [1|2|3|e] > /workspace/spike.log 2>&1   (spike 2 needs runs/co_a/best.pt from spike 1;
# spike 3 needs data_co_a,d,e,f,g). `job <slug> [small|medium] [steps]` (scripts/typically_job.py): base eval + one fine-tune on
# data_co_<slug>, JOB=1 trims the evals. Every fine-tune's architecture flags are read from runs/base/best.pt (typically_job.train_flags).
set -eo pipefail
if [ "${1:-2}" = job ]; then BASE=${3:-small}; STEPS=${4:-400}; else BASE=small; fi
[[ "$BASE" =~ ^(small|medium)$ && "${STEPS:-400}" =~ ^[0-9]+$ ]] || { echo "job: bad base/steps '$BASE'/'${STEPS:-}'" >&2; exit 2; }   # same rule as typically_job.BASES / STEPS
cd /workspace/pcdm
{ set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
export UV_PROJECT_ENVIRONMENT=/workspace/venv
uv sync
if [[ "$(nvidia-smi)" == *"CUDA Version: 12.8"* ]]; then   # docs/plan/PROJECT.md torch/driver rule (no grep -q pipe: pipefail + SIGPIPE)
  uv pip install --python /workspace/venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
if [ "$BASE" = medium ]; then   # Qwen3.5 hybrid: without these two the Gated-DeltaNet layers fall back to a slow reference path (still correct)
  # separate calls: causal-conv1d has no wheel for torch 2.14+cu130 and its source build needs nvcc 13 (image has 12.8); fla is pure Triton
  uv pip install --python /workspace/venv/bin/python flash-linear-attention || echo "WARN: fla install failed"
  uv pip install --python /workspace/venv/bin/python --no-build-isolation causal-conv1d || echo "WARN: causal-conv1d build failed; the conv stays on the reference path"
  uv run --no-sync python -c "import fla" || echo "WARN: fla missing; continuing on the slower reference path"
fi
for d in v5 wf wh u; do   # one call per dir: --include takes ONE pattern, extra ones become filenames (and disable it)
  uv run --no-sync hf download guychuk/pcdm-data --repo-type dataset --include "$d/*" --local-dir data
done
for d in v5 wf wh u; do ln -sfn data/$d data_$d; done
uv run --no-sync hf download "OzLabs/typical-$BASE" best.pt --local-dir runs/base

EV="uv run --no-sync python scripts/eval_wf.py --mode native"
REG="data_wf/eval/wf_heldout_noul.jsonl data_wf/eval/wf_heldout_choice.jsonl data_wf/eval/wf_heldout_score.jsonl data_wf/eval/wf_rubric_flip.jsonl data_wh/eval/wh_heldout_family.jsonl"
evals() {  # $1 run dir, $2 company files (eval_wf reads $1/best.pt)
  $EV --run "$1" --files $2 --limit 0 --out "$1/eval_co.json"
  # ponytail: JOB=1 (job mode) skips the regression evals: a user's own model is judged on its own held-out rows only
  [ -n "$JOB" ] || $EV --run "$1" --files $REG --limit 500 --out "$1/eval_reg.json"   # ponytail: 500-row stratified regression sample
}

# the Release-1 v3 recipe (releases/typical-small.md), warm-started, company data = bucket C at half of every batch
train() {  # $1 company, $2 steps, $3 run name
  local flags   # assigned on its own line so set -e sees a failing flag builder; no value has a space, so $flags is safe unquoted
  flags=$(uv run --no-sync python scripts/typically_job.py --train-flags "$1" "$2" "$3" "$BASE")   # best.pt is picked on the company's own val (spike 3)
  uv run --no-sync python pcdm/train.py $flags
  mkdir -p "runs/$3_last"   # ponytail: last.pt carries opt/sched/rng; strip to what bench.load_ours reads (weights_only load) instead of symlinking
  uv run --no-sync python -c "import sys,torch; b=torch.load(sys.argv[1]+'/best.pt',weights_only=False); print(sys.argv[1],'best step',b['step']); \
l=torch.load(sys.argv[1]+'/last.pt',weights_only=False); torch.save({k:l[k] for k in ('tower','lora','step','best_val','args')},sys.argv[1]+'_last/best.pt'); \
print(sys.argv[1],'last step',l['step'])" "runs/$3"
  evals "runs/$3" "data_co_$1/eval/*.jsonl"
  [ -n "$JOB" ] || evals "runs/$3_last" "data_co_$1/eval/*.jsonl"   # ponytail: job mode skips the _last eval too (best.pt only)
}
if [ "${1:-2}" = job ]; then   # $2 = slug; the UI's "Teach it" (eval + train only that company's held-out rows)
  [[ "${2:-}" =~ ^[a-z0-9_]{1,40}$ ]] || { echo "job: bad slug '${2:-}'" >&2; exit 2; }   # same rule as typically_job.SLUG_RE
  [ -f "data_co_$2/train.jsonl" ] || { echo "job: data_co_$2/train.jsonl missing" >&2; exit 2; }
  JOB=1
  evals runs/base "data_co_$2/eval/*.jsonl"
  train "$2" "$STEPS" "co_$2"
  # the results page's base-vs-yours scores, while both models are on this GPU (else the server downloads + scores them itself)
  uv run --no-sync python scripts/typically_reveal.py --job "$2" "$BASE" || echo "WARN: reveal failed; the server will score it"
  exit 0   # success is the exit code: job.sh's EXIT trap writes /workspace/job.exit for the poller (no log-string protocol)
elif [ "${1:-2}" = e ]; then   # spike 2 arm e (eval tickets == data_co_a's, so base numbers carry over)
  train e 400 co_e
elif [ "${1:-2}" = 3 ]; then   # spike 3: best.pt on company val + final step; e/f/g vary the augmentation, d is the clean-import arm
  train a 400 a_sel
  train e 400 e_sel
  train f 400 co_f
  train g 400 co_g
  train d 400 d_sel
elif [ "${1:-2}" = 1 ]; then
  evals runs/base "data_co_a/eval/*.jsonl data_co_b/eval/*.jsonl"
  train a 400 co_a
  train b 400 co_b
  train a 100 co_a_s100
else   # spike 2: numbers in words (c), clean imported labels (d), held-out flip rules for base + spike-1 co_a
  [ -f runs/base/eval_co2.json ] || $EV --run runs/base --files data_co_a/eval/a_flip_heldout.jsonl data_co_c/eval/*.jsonl data_co_d/eval/*.jsonl --limit 0 --out runs/base/eval_co2.json
  [ -f runs/co_a/eval_co2.json ] || $EV --run runs/co_a --files data_co_a/eval/a_flip_heldout.jsonl --limit 0 --out runs/co_a/eval_co2.json
  train c 400 co_c
  train d 400 co_d
fi
echo SPIKE_DONE
