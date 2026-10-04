# Multimodal + Hebrew smoke — 2026-10-04

## Decision

The 200-step frozen-LoRA pilot proves that the existing Typical head can learn
pixel-dependent decisions and improves held-out Hebrew without a broad text-quality
collapse. It is not a promotion candidate yet: the pilot is only 128 rows per new
bucket, lacks a direct-VLM visual baseline and answerability/null controls, and loses
2--4 public JevBench items relative to `tm2`.

Proceed to the M2 baselines and a larger seed-0 E1. Do not add an attention module or
unfreeze LoRA yet.

## Reproducibility

- Source commit: `11b18e427c6094d4a112e491da4aad6ec834fc9f`
- GPU: NVIDIA H100 80GB HBM3
- Base checkpoint SHA-256:
  `4564b3b1be12142516f1081ab6d98e9bf528547d26251ba32e008c6d41567469`
- Smoke best checkpoint SHA-256:
  `fe82904a7e3f08fff2d3ee881666d3fc5aa72a7a0caf9388e772ee2d3c8b9cb1`
- Data revision: `guychuk/pcdm-data@946aee5c2a4c25b7df7da099a1510946b6349d2e`
- JevBench: the commit recorded in each run manifest under `runs/jev_p0_*_20261004/`
- New checkpoint: `runs/tm2_mm_he_smoke200_20261004/best.pt` (local, gitignored)

The run initialized from `tm2`, froze its top-eight rank-16 language LoRA, and trained
only the native decision head. The saved LoRA state has 124 tensors and a maximum
absolute difference of exactly `0.0` from `tm2`.

```bash
uv run pcdm/train.py \
  --name tm2_mm_he_smoke200_20261004 \
  --readout native --multimodal \
  --backbone Qwen/Qwen3.5-4B-Base --tap_layer 23 \
  --init_from runs/tm2/best.pt --freeze_lora \
  --nc_head n3 --nc_render letters_nonull \
  --null factored --noul_head bern --ordinal_smooth 0.7 --zscore \
  --data data_v5 \
  --extra_data data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u,data_v,data_he \
  --bucket_map data_wf_long=W,data_wh=W,data_u=U,data_v=V,data_he=H \
  --family_weights E:0.30,K:0.11,W:0.26,U:0.08,V:0.15,H:0.10 \
  --null_aug W:0.20 --best_on data_u_val,data_wh_val \
  --steps 200 --bs 64 --grad_accum 16 --lr 3e-4 \
  --val_every 50 --eval_every 200 --ckpt_every 100 \
  --eval_cap 128 --eval_limit 128 --eval_bs 8 \
  --max_state 2048 --drop_truncated --device cuda
```

The final step had validation NLL `0.2924`. Step 200 was also the best checkpoint on
the predeclared `data_u_val,data_wh_val` gate (`0.4013` mean raw NLL). Peak allocated
memory was about 43.6 GB. The V and H buckets both appeared in sampled accumulation
windows; the configured family mix was E/K/W/U/V/H = 30/11/26/8/15/10%.

## P0 readout control

All rows are the same JevBench public tasks. Brier and ECE are lower-is-better;
latency is warm per-task median on the same H100.

| Readout | Tier | Accuracy | Brier | ECE | p50 seconds |
|---|---:|---:|---:|---:|---:|
| `tm2` native | standard | .861 | .222 | .082 | .122 |
| frozen direct letters | standard | .847 | .237 | .072 | .095 |
| `tm2` native | easy | 1.000 | .018 | .032 | .123 |
| frozen direct letters | easy | .979 | .031 | .028 | .062 |
| `tm2` native | hard | .486 | .724 | .252 | .197 |
| frozen direct letters | hard | .468 | .617 | .151 | .125 |

Direct letters are faster and better calibrated on the hard tier, but do not expose
an accuracy reserve hidden by the native head. An endpoint swap was not run: `tm2`'s
head was trained on mean-span features, so changing the feature at inference would
not be a valid frozen-head control; training another endpoint head is deferred until
a readout gap justifies it.

## Multimodal and Hebrew result

| Evaluation (128 rows each) | `tm2` | smoke | Delta |
|---|---:|---:|---:|
| ScienceQA, real image | n/a | .906 | n/a |
| ScienceQA, no image | n/a | .641 | n/a |
| ScienceQA, shuffled image | n/a | .617 | n/a |
| MASSIVE Hebrew test | .727 | .797 | +.070 |
| Hebrew validation | .781 | .789 | +.008 |

The real-image gap is +28.9 points over shuffled images and +26.6 over no image,
comfortably above the predeclared 10-point screen. This establishes image dependence;
it does not establish robustness, because the pilot uses a small research-only
ScienceQA subset (CC-BY-NC-SA-4.0).

Across the 64 common non-Hebrew text suites with numeric accuracy, capped to the same
128 rows, macro accuracy moved from `.7411` to `.7482` (+.0071). Across all common
numeric suites, macro NLL moved `.8178 -> .7926`, Brier `.3292 -> .3153`, and ECE
`.1386 -> .1303`. The distribution is mixed (25 suites improved, 10 tied, 29
regressed), so the macro gain is not a substitute for the targeted regression tests.

On JevBench, the smoke checkpoint scored `.833/.1000/.450` on standard/easy/hard,
versus `.861/1.000/.486` for `tm2`. Paired item bootstrap 95% intervals for the
accuracy deltas were `[-.069, 0]` on standard and `[-.081, 0]` on hard; exact McNemar
tests gave `p=.50` and `p=.219`. The observed drops are not statistically resolved,
but should be treated as a warning and rechecked at larger scale and multiple seeds.

## Next gate

1. Implement the matched direct-Qwen3.5 visual letter baseline (R003).
2. Add gold-dropped and image-mismatch rows so visual null AUROC can be measured.
3. Add paired English/Hebrew views with group-safe splits and probability divergence.
4. Run the larger 20k--50k-row seed-0 E1 with the same frozen LoRA and broad replay.
5. Continue to seeds 1/2 only if image dependence, null AUROC, Hebrew, and targeted
   text/JevBench retention all pass.
