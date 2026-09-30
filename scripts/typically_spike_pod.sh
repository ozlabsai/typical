#!/bin/bash
# typically spike, one GPU pod: released typical-small vs per-company fine-tunes (scripts/typically_spike.py).
# Expects the repo + data_co_a/ + data_co_b/ at /workspace/pcdm, secrets in /workspace/.env.
# bash scripts/typically_spike_pod.sh [1|2|3|e] > /workspace/spike.log 2>&1   (spike 2 needs runs/co_a/best.pt from spike 1;
# spike 3 needs data_co_a,d,e,f,g)
set -eo pipefail
cd /workspace/pcdm
{ set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a
export UV_PROJECT_ENVIRONMENT=/workspace/venv
uv sync
if [[ "$(nvidia-smi)" == *"CUDA Version: 12.8"* ]]; then   # docs/plan/PROJECT.md torch/driver rule (no grep -q pipe: pipefail + SIGPIPE)
  uv pip install --python /workspace/venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
for d in v5 wf wh u; do   # one call per dir: --include takes ONE pattern, extra ones become filenames (and disable it)
  uv run --no-sync hf download guychuk/pcdm-data --repo-type dataset --include "$d/*" --local-dir data
done
for d in v5 wf wh u; do ln -sfn data/$d data_$d; done
uv run --no-sync hf download OzLabs/typical-small best.pt --local-dir runs/base

EV="uv run --no-sync python scripts/eval_wf.py --mode native"
REG="data_wf/eval/wf_heldout_noul.jsonl data_wf/eval/wf_heldout_choice.jsonl data_wf/eval/wf_heldout_score.jsonl data_wf/eval/wf_rubric_flip.jsonl data_wh/eval/wh_heldout_family.jsonl"
evals() {  # $1 run dir, $2 company files (eval_wf reads $1/best.pt)
  $EV --run "$1" --files $2 --limit 0 --out "$1/eval_co.json"
  $EV --run "$1" --files $REG --limit 500 --out "$1/eval_reg.json"   # ponytail: 500-row stratified regression sample
}

# the Release-1 v3 recipe (releases/typical-small.md), warm-started, company data = bucket C at half of every batch
train() {  # $1 company, $2 steps, $3 run name
  uv run --no-sync python pcdm/train.py --name "$3" --init_from runs/base/best.pt \
    --readout native --nc_head n3 --nc_render semif --null factored --noul_head bern --score_head choice \
    --ordinal_smooth 0.7 --tap_layer 20 --zscore --lora_r 16 --lora_layers 8 --max_state 1024 \
    --data data_v5 --extra_data "data_wf,data_wh,data_u,data_co_$1" --bucket_map "data_wh=W,data_u=U,data_co_$1=C" \
    --family_weights C:0.5,W:0.3,E:0.15,U:0.05 --null_aug W:0.20 \
    --steps "$2" --bs 64 --grad_accum 8 --val_every 50 --ckpt_every 100 --eval_every "$2" --eval_limit 200 --eval_bs 8 \
    --best_on "data_co_$1_val"   # pick best.pt on the company's own val, not general replay (spike 3; site/server.py build copies this)
  mkdir -p "runs/$3_last"   # ponytail: last.pt carries opt/sched/rng; strip to what bench.load_ours reads (weights_only load) instead of symlinking
  uv run --no-sync python -c "import sys,torch; b=torch.load(sys.argv[1]+'/best.pt',weights_only=False); print(sys.argv[1],'best step',b['step']); \
l=torch.load(sys.argv[1]+'/last.pt',weights_only=False); torch.save({k:l[k] for k in ('tower','lora','step','best_val','args')},sys.argv[1]+'_last/best.pt'); \
print(sys.argv[1],'last step',l['step'])" "runs/$3"
  evals "runs/$3" "data_co_$1/eval/*.jsonl"
  evals "runs/$3_last" "data_co_$1/eval/*.jsonl"
}
if [ "${1:-2}" = e ]; then   # spike 2 arm e (eval tickets == data_co_a's, so base numbers carry over)
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
