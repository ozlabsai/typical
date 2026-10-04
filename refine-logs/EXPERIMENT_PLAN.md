# Experiment Plan

**Problem**: Extend Typical to image-conditioned decisions and Hebrew without losing its calibrated, runtime-defined text behavior.
**Method Thesis**: Reuse Qwen3.5's native visual prefix and Typical's existing cached native head; add visual and Hebrew buckets instead of a new fusion architecture.
**Date**: 2026-09-29

**2026-10-04 research update:** The competitive audit in
[`docs/research/DECISION_MODELS_AUDIT.md`](../docs/research/DECISION_MODELS_AUDIT.md)
keeps the existing multimodal-prefix thesis. Before B1, add three same-backbone P0
baselines: current native head, frozen direct option-letter logits, and endpoint
pooling in the native head. Fit temperature on the same calibration split. Defer
cross-field attention until a dedicated multi-question interaction suite shows a
repeatable gap. If B2/B3 pass, prioritize grouped hard-negative refinement with
40% text replay and paired English/Hebrew probability consistency.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Typical can use pixels while retaining cached, calibrated decisions | This is the multimodal contribution | Real-image accuracy beats shuffled/no-image by >=10 points; reaches >=80% of direct Qwen3.5 accuracy; null AUROC >=.75; cache parity holds | B1, B2, B4 |
| C2: Hebrew training improves Hebrew decisions without material English regression | A multilingual mix is useful only if it transfers safely | Paired Hebrew gain over tm2 with 95% bootstrap CI excluding 0; English MASSIVE and established text suites stay within 1 point or their existing seed interval | B3 |
| Anti-claim: gains come from question/answer priors or generic continued training | Must prove the image and Hebrew bucket cause the gains | Shuffled/no-image controls and a no-H ablation | B2, B3 |

## Paper Storyline

- Main paper must prove: pixel dependence, calibrated null behavior, amortized visual prefill, and Hebrew gain without text regression.
- Appendix can support: three-seed intervals, resolution/token-budget sweep, and error examples.
- Experiments intentionally cut: video, audio, multi-image input, visual Score, vision adapters, and a new fusion tower.

## Experiment Blocks

### B1: Artifact and plumbing gate

- Claim tested: the implemented path is trainable and reproducible.
- Dataset / split / task: 128 ScienceQA image rows, 128 MASSIVE Hebrew rows, 128 text-replay rows.
- Compared systems: full one-pass vs cached suffix; red/blue and real/shuffled image controls.
- Metrics: cache probability max error, finite loss, train NLL reduction, image-token count.
- Setup details: Qwen3.5-4B, tap 23, tm2 initialization, frozen vision and LoRA, 200 steps, seed 0.
- Success criterion: `best.pt` restored; all corpora present; cache error <=2e-3 in bf16; loss decreases; changing the image changes predictions.
- Failure interpretation: plumbing/data bug; do not launch E1.
- Priority: MUST-RUN.

### B2: Frozen visual probe

- Claim tested: the existing native head can extract useful pixel evidence.
- Dataset / split / task: ScienceQA image train/validation/test; real, shuffled, and no-image test variants.
- Compared systems: direct Qwen3.5 letter scoring, tm2/no-image, E1 real-image, E1 shuffled-image.
- Metrics: among-K accuracy, NLL, Brier, ECE, null AUROC, real-minus-shuffled gap, prefix and warm-query latency.
- Setup details: initialize from tm2; train head only; V=.15 with text replay; max 262,144 pixels; seed 0 first, seeds 1/2 only after passing.
- Success criterion: >=10-point real/shuffled gap, >=80% of direct baseline accuracy, null AUROC >=.75, no established text-suite regression beyond 1 point or its prior interval.
- Failure interpretation: if direct VLM succeeds but E1 fails, unfreeze LoRA in E2; if both fail, replace the pilot data/task before changing architecture.
- Priority: MUST-RUN.

### B3: Hebrew isolation

- Claim tested: the H bucket, rather than generic continuation, causes Hebrew gains.
- Dataset / split / task: MASSIVE Hebrew test plus matched English MASSIVE and existing English suites.
- Compared systems: tm2, E1 with H=.10, matched E1 with H=0 and its weight returned to E.
- Metrics: accuracy, NLL, Brier, ECE, null recall; paired bootstrap deltas.
- Setup details: identical seed/steps/order outside the sampler weight change; English intent labels remain runtime candidates.
- Success criterion: Hebrew gain CI excludes 0; English MASSIVE and core text suites regress <=1 point or remain within their prior interval.
- Failure interpretation: reduce H weight or add bilingual candidate renderings; do not claim multilingual improvement.
- Priority: MUST-RUN.

### B4: Conditional LoRA continuation

- Claim tested: language LoRA adaptation is necessary only if the frozen head bottlenecks visual quality.
- Compared systems: best E1 vs E2 with top-eight LoRA trainable; vision stays frozen.
- Metrics: B2 metrics plus text retention and GPU cost.
- Success criterion: E2 materially improves visual NLL/accuracy without erasing the real/shuffled gap or moving text suites outside bounds.
- Failure interpretation: retain simpler E1; do not add vision adapters.
- Priority: NICE-TO-HAVE; run only if B2 misses the direct-VLM retention gate.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Risk |
|---|---|---|---|---|---|
| M0 | Restore inputs | R001 | tm2 checkpoint and all E/K/W/U/V/H data validate | CPU/download time | complete: 1,054,473 training rows schema-checked |
| M0.5 | Readout baselines | R001a-c | native head beats or complements direct-letter and endpoint variants on held-out NLL/Brier | <1 H100-hour | architecture gain may be prompt/readout only |
| M1 | Sanity | R002 | 200-step loss drop and visual/cache checks pass | <1 H100-hour | path or processor mismatch |
| M2 | Baselines | R003-R004 | direct VLM and tm2 baseline numbers recorded | 1-2 H100-hours | prompt baseline mismatch |
| M3 | E1 decision | R005, then R006-R007 | B2 gates pass on seed 0 before extra seeds | roughly 3-6 H100-hours/seed; recalibrate after 100 steps | visual data license and throughput |
| M4 | Hebrew isolation | R008 | paired Hebrew gain with English retention | roughly one E1 seed | mixture confound |
| M5 | Conditional E2 | R009 | only if E1 bottlenecks | measure from first 100 steps | unnecessary complexity |

## Compute and Data Budget

- Must-run: one smoke, two frozen baselines, one E1 seed, two confirming seeds only after the seed-0 gate, and one H=0 ablation.
- Estimated E1 cost is provisional; record tokens/sec and memory over the first 100 steps before reserving the remaining runs.
- Data preparation: restore E/K/W/U corpora; generate MASSIVE Hebrew; generate ScienceQA only for research-only weights.
- M0 inputs are restored locally. `runs/tm2/best.pt` came from the public `OzLabs/typical-medium`
  release and was verified from its payload as Qwen3.5-4B/tap-23; the exact training buckets came
  from pinned `guychuk/pcdm-data` revision `946aee5c2a4c25b7df7da099a1510946b6349d2e`.

## Risks and Mitigations

- ScienceQA is non-commercial: mark E1 weights research-only; audit a permissive replacement before release training.
- Language priors masquerade as vision: require real/shuffled/no-image controls.
- Hebrew gain harms English: matched H=0 ablation and English MASSIVE retention gate.
- Resolution dominates cost: keep 262,144-pixel cap; sweep only after E1 passes.

## Final Checklist

- [x] Main claims and anti-claim are explicit
- [x] Novelty is isolated by image counterfactuals
- [x] Simplicity is defended by making E2 conditional
- [x] VLM necessity is compared with direct Qwen3.5
- [x] Nice-to-have runs are separated from must-run runs
