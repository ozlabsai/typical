# Experiment Plan — Jev parity

**Problem:** Typical's 4B layer-23 shared head is below Jev on capability and regressed when visual/Hebrew examples rewrote that same frozen-backbone decision boundary.

**Method thesis:** First evaluate stronger untouched instruction checkpoints with final-token/pointer readouts. If training is needed, use grouped long-context decision data, soft-distribution retention, per-kind calibration, and explicit non-regression gates. Add Hebrew and multimodality only after text/workflow competence is locked.

**Date:** 2026-10-04

The prior multimodal/Hebrew plan and completed E1 history remain in `EXPERIMENT_PLAN_20260929_190948.md` and `docs/research/MM_HE_E1_20261004.md`. The evidence synthesis is `docs/research/JEV_PARITY_STRATEGY_20261004.md`.

## Claim map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1: Typical equals or beats Jev's decision capability | Official JevBench score ≥72.1 **and** Intelligence ≥72.0; public diagnostic ≥200/231, hard ≥81/111 | B0, B2, B5 |
| C2: Typical equals or beats Jev on real workflows | Official WorkflowEvals exact-actions mean ≥67.8 and each workflow within two points of Jev | B0, B1, B3, B5 |
| C3: The model is genuinely multimodal | Pinned image benchmark plus real/shuffled/no-image and mismatch-null controls | B0, B4, B5 |
| C4: Hebrew improves without erasing English | Paired Hebrew gain; English loss ≤1 point; calibration and pair JSD reported | B0, B4, B5 |
| C5: The result is fast and calibrated | Valid distributions; Brier/ECE no worse than Jev on comparable sets; latency/cost within 2× Jev | B0, B5, B6 |
| Anti-claim: public benchmark tuning or teacher imitation alone explains the result | Locked test quarantine, overlap audit, grouped splits, independent and distilled result tables | B1, B2, B3, B5 |

## Fixed targets

| Evaluation | Jev target | Promotion gate |
|---|---:|---:|
| JevBench official v1.5.x | score 72.1; Intelligence 72.0 | ≥ both; report open/sealed and type breakdown |
| JevBench public diagnostic | about 200/231; hard 81/111 | ≥200/231 and hard ≥81/111; stretch 203/231 and 85/111 |
| WorkflowEvals security | 61.7% | ≥59.7% |
| WorkflowEvals agent trace | 71.6% | ≥69.6% |
| WorkflowEvals invoice | 61.8% | ≥59.8% |
| WorkflowEvals customer | 76.0% | ≥74.0% |
| WorkflowEvals equal-weight mean | 67.8% | ≥67.8% |
| typed-decisions, grouped | accuracy .727 | ≥.727 plus KL/Brier/ECE no worse |

Image JevBench and the paired Hebrew suite are additional release gates, not substitutes for the text/workflow targets.

## Experiment blocks

### B0 — benchmark and evaluator correctness

- Fix all validation calls to use `args.max_state`; add a regression check.
- Run WorkflowEvals exactly as released: all questions in a case jointly, final workflow actions scored against consensus.
- Run typed-decisions as complete five-question cases, not capped rows.
- Pin repositories, datasets, prompts, model revisions, temperatures, and output manifests.
- Quarantine all benchmark examples from training and checkpoint selection.

**Pass:** Cygnet identity run reproduces 203/231 ± the documented single near-tie; all 705 WorkflowEvals cases complete without invalid vectors.

### B1 — untouched baseline matrix

Evaluate, without training:

1. Cygnet / pinned Gemma-4-12B-it.
2. Winnow-12B.
3. Intern-Decision-4B.
4. Existing `tm2` and already-trained `tl2`, using correct long context and request grouping.

Metrics: fixed targets above, p50/p95, cost, invalid/error count, Hebrew paired metrics, and image counterfactual metrics.

**Decision:** If one untouched checkpoint clears every primary gate, adopt it as the base and skip competence training. Otherwise promote the smallest model with the strongest locked generalization to B2/B3.

### B2 — readout/backbone selection

Compare on disjoint development data only:

- constrained final-token symbols;
- Strands-style answer/option pointer;
- current Typical native head.

Candidate backbones: Gemma-4-12B-it for the flagship, Qwen3.5-4B-it/Intern for the fast lane. Do not return to Qwen3.5-4B-Base layer 23 unless it wins this ablation.

**Pass:** chosen readout improves held-out accuracy/NLL/Brier, not formatting alone, and supports at least 16k context.

### B3 — text/workflow competence training

- 30k–100k first run, quality before scale.
- Sources after audit: Tasksource selected mix, Open-Jev v1.1, Sarge v3 `yuri_v3`; exclude Sarge uniform placeholders and duplicate Open-Jev rows.
- Length buckets include 8k–32k states; primitive-stratified and source-grouped batches.
- 30% option permutations; counterfactual fact changes; long policy/doc cases; staged hard negatives; 30–40% broad replay.
- Hard gold: full-vocabulary symbol CE. Soft distributions: `KL(target || model)`. Score: add `0.5 ×` ranked-probability score. Teacher signal applies only when teacher argmax agrees with verified gold.
- LoRA: flagship rank 32 over attention and MLP; fast lane rank 16/32; freeze vision tower/projector initially.
- Checkpoints advance on per-suite floors, never a mean validation loss alone.

**Stop rule:** if two successive gates fail to improve both hard capability and workflow dev without regression, change data/readout rather than extending steps.

### B4 — multimodal and Hebrew continuation

Starts only from a B1/B3 text winner.

- Separate or zero-initialized residual adapter.
- Frozen-teacher KL on text/workflow replay.
- Hebrew semantic triples: HE state/HE options, HE state/EN options, EN twin; grouped split and probability-consistency loss.
- Image rows include answerable, shuffled, no-image, gold-dropped, and genuinely mismatched evidence with verified null labels.
- At minimum freeze typed Noul and text-null parameters if the shared head remains.

**Pass:** Jev/workflow gates retain their declared floors; Hebrew improves with ≤1 English point lost; multimodal counterfactuals pass.

### B5 — frozen final evaluation

Freeze checkpoint, prompt, temperatures, thresholds, quantization, and serving settings before:

- one final public diagnostic run;
- official JevBench submission including sealed items;
- full official WorkflowEvals;
- grouped typed-decisions;
- Image JevBench/pinned image suite;
- paired Hebrew suite.

Report bootstrap intervals and separate independent vs teacher-distilled tables.

### B6 — serving optimization

Only after B5 quality passes: Q8, prefix packing, concurrent questions, and then profiled kernel/CUDA-graph work. Re-run B5 equivalence checks after quantization.

## Run order

| Run | Purpose | Advancement rule | Status |
|---|---|---|---|
| G0-CYGNET | Pinned Cygnet identity + 705-case WorkflowEvals | identity 203/231 ±1; zero request failures | RUNNING |
| G0-WINNOW | Same matrix for Winnow | compare quality/MM/HE/workflow against Cygnet | QUEUED |
| G0-INTERN4 | Same matrix for Intern-Decision-4B | establish fast/MM candidate | QUEUED |
| G0-TL2 | Evaluate existing 9B checkpoint before training | use correct context/grouping | QUEUED |
| G1-READOUT | Final-token vs pointer vs native | development-only winner | BLOCKED on G0 |
| G2-TEXT | First competence LoRA | ≥198/231 diagnostic and no workflow-dev regression | BLOCKED on G1 |
| G3-HARD | Long/hard-negative refinement | improves locked internal hard families | BLOCKED on G2 |
| G4-MMHE | Residual Hebrew/MM continuation | all retention gates | BLOCKED on text winner |
| G5-OFFICIAL | Frozen official submission/evals | all primary gates | BLOCKED on candidate |
| G6-SERVE | Q8/prefix optimization | prediction equivalence + latency/cost | BLOCKED on G5 quality |

## Compute budget

- G0: one H100, run baselines sequentially; persist results before stopping. Expected tens of GPU-hours at most, dominated by downloads and 705 workflow cases.
- First training gate: cap at 250 steps; estimate throughput/memory before authorizing the full run.
- Never run extra seeds until seed 0 clears all declared gates.
- Stop idle pods automatically after artifacts persist.

## Risks and mitigations

- **Public-set overfitting:** public JevBench is diagnostic only; no checkpoint selection.
- **Teacher imitation presented as intelligence:** disclose Sarge/Jev distillation and keep an independent-data table.
- **Long-context truncation:** reject over-limit cases instead of silent truncation; length-bucket training/evaluation.
- **Shared-head forgetting:** separate residuals and frozen-teacher retention; lower continuation LR.
- **Sealed Noul collapse:** type-specific held-out calibration and independent Noul data; do not infer sealed quality from public accuracy.
- **Hebrew translation label drift:** semantic twins and manual/rule checks before translated complex rows enter training.
- **MM benchmark shortcut:** require shuffled/no-image/mismatch controls.

## Completion checklist

- [x] Root cause identified and evaluator context bug fixed
- [x] Claims, anti-claims, and numeric targets declared
- [x] Official WorkflowEvals replaces the stale proxy as the primary workflow target
- [x] Untouched baselines precede training
- [ ] G0 baseline matrix complete
- [ ] Candidate clears locked development gates
- [ ] Official JevBench score and Intelligence equal/beat Jev
- [ ] Official WorkflowEvals mean equal/beats 67.8
- [ ] Multimodal and Hebrew release gates pass
- [ ] Serving latency/cost and calibration reported
