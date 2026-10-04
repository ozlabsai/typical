# Typical → Jev parity strategy

**Date:** 2026-10-04  
**Goal:** equal or beat Jev on JevBench and the official TypeSafe WorkflowEvals, while retaining a credible path to fast multimodal and Hebrew decisions.

## Decision

Stop extending the current frozen-4B shared head. Establish the strongest untouched public baselines first, then train only if none clears the locked targets.

The most credible flagship path is a native multimodal 12B instruction model with constrained final-token scoring and LoRA adaptation, following Cygnet and Winnow. A Qwen3.5-4B Intern-style model remains the fast lane. A 27B model is useful as a teacher or fallback, not the default serving target.

Custom attention is not the next experiment. Current leaders show that final LM logits or a pointer readout, a strong instruction backbone, good decision data, long context, and calibration dominate head complexity.

## Why E1 moved backward

E1 improved images and Hebrew because it trained the only component allowed to move: the shared `NativeHead`. It also damaged text for the same reason.

- All 124 LoRA tensors remained bit-identical. The shared head changed heavily: the main projections and typed Noul parameters moved roughly 46–51% in relative L2 norm.
- The head saw a new 25% visual/Hebrew mixture at a fresh `3e-4` learning-rate schedule. Ground-truth replay preserved sample ratios, not the old function; there was no frozen-`tm2` probability-retention loss.
- Mismatched image/question pairs were evaluation-only. Training taught missing-gold-option nulls, not visual inconsistency, explaining 10.5% mismatch null recall.
- Checkpoint selection averaged validation losses, so visual/Hebrew gains could hide a typed or Jev regression.
- A real evaluator bug made this worse: checkpoint-selection validation silently truncated states to 256 tokens instead of the configured 2,048. The training and final evaluation used 2,048. This is now fixed in `pcdm/train.py`.

Observed effect: public JevBench moved from `164/231` to `159/231`; the exact-row typed slice moved `.373 → .340`; image accuracy moved to `.949`; Hebrew moved `.727 → .820`.

## What the current evidence says

| System / recipe | Strongest relevant evidence | Lesson for Typical |
|---|---|---|
| [JevBench v1.5.6](https://www.benchmarkheaven.com/jev-models) | Jev: official score 72.1, Intelligence 72.0. Cygnet: 73.7 / 71.1. Winnow Q8: 73.2 / 74.4. Winnow is the open system that clears both Jev's composite and intelligence. | Require both official score and Intelligence parity; do not win only through speed/cost. |
| [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) | Frozen Gemma-4-12B-it, one constrained letter, temperature 3.4; `203/231` public and official rank #1. | The fastest evidence-backed baseline is stock 12B + correct prompt/readout, before training. |
| [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) | Gemma-4-12B-it, rank-32 LoRA across attention and MLP, about 26k curated/refined decisions, teacher loss only on teacher/gold agreement, replay; official Intelligence 74.4. | This is the primary text-quality recipe to reproduce or build from. |
| [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) | Qwen3.5-4B, native multimodality, final LM logits at JSON decision markers; self-reported `201/231`, typed decisions 80.55%, 44 ms on 4090. | Best fast/MM candidate; evaluate untouched before modifying our older layer-23 head. |
| [Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/) | Pointer scoring beat an earlier slot head; rank-16 language LoRA. | If a custom head survives, compare answer-position/option-position pointer scoring, not more slot capacity. |
| [JevK5](https://huggingface.co/alibiserikbay/JevK5) and [Plumb](https://github.com/crh225/plumb) | Long teacher documents, agreement filtering, hard mining, and broad replay improve public hard accuracy; current sealed Noul remains weak. | Long hard data matters, but public-set gains are not evidence of sealed generalization. |
| [Decision 2.0 Vega](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) | 27B final decision model; JevArena 74.0 and multi-question one-pass inference. | Useful teacher/upper bound; too large to be the first serving target. |
| [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) | Frozen 27B + rank-16 LoRA + fp32 head, KL to soft distributions, Score RPS, option permutations, per-kind temperatures; reported six-benchmark mean 84.07 vs Jev 83.85. | Copy the loss and batching details, while labeling this as Jev distillation rather than independent intelligence. |
| [CLM](https://github.com/Contrastive-LM/CLM) | Staged hard negatives, contrastive learning, replay, and cached actions. | Add hard-negative contrastive refinement only after the plain decision objective passes. |
| [Bandits](https://github.com/bandr-ai/bandits) | Outcome-grounded agent trace labels, lineage-preserving splits, quarantine of ambiguous cases. | Use post-action evidence for workflow data; do not force uncertain trace labels. |
| [Official WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals) | 705 public cases; exact final action sets after workflow code. Jev scores 61.7 security, 71.6 agent trace, 61.8 invoice, 76.0 customer, mean 67.8. | This is now the primary workflow benchmark, not our old row-wise proxy. |
| [LocalLLaMA typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Full five-question cases; Jev `.727` accuracy. Request grouping affects results. | Our first-1,500 row-wise `.515` number is not leaderboard-comparable. Re-evaluate grouped cases. |
| [Tasksource typed decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions), [Open-Jev v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1), [Sarge v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Broad grouped supervision, programmatic counterfactuals, and full teacher distributions. | These are the training pool after licensing, overlap, quality, and placeholder audits. |

## Success gates

The goal is not complete until a frozen submitted checkpoint meets every primary gate.

1. **Official JevBench v1.5.x:** score `≥72.1` and Intelligence `≥72.0`; additionally target Winnow's `74.4` Intelligence. Report open/sealed and Choice/Noul/Score separately.
2. **Public JevBench diagnostic:** at least `200/231`, with hard `≥81/111`; stretch target Cygnet `203/231`, hard `85/111`. The public set is diagnostic, never a checkpoint-selection set.
3. **Official WorkflowEvals:** equal-weight exact-actions mean `≥67.8`, with no workflow more than two points below Jev's published row.
4. **Typed-decisions:** grouped five-question protocol; accuracy `≥.727` plus KL, Brier, and ECE no worse than Jev.
5. **Multimodal:** submit to Image JevBench or run its pinned public protocol; meanwhile require real-image accuracy, shuffled/no-image counterfactuals, and mismatch abstention. ScienceQA alone is not the final MM claim.
6. **Hebrew:** locked paired Hebrew/English set; Hebrew improves over the selected base, English loses at most one point, and probability JSD/calibration are reported.
7. **Serving:** no invalid vectors; one-pass typed probabilities; measured p50/p95 and cost. A quality pass that is slower or costlier than 2× Jev is not the final fast model.

## Experiment sequence

### G0 — untouched baselines

Run exact, pinned evaluations before any training:

- Cygnet / frozen Gemma-4-12B-it.
- Winnow-12B Q8 or BF16 reference.
- Intern-Decision-4B.
- Existing `tm2` and already-trained `tl2` with corrected state length and grouped requests.

For each: JevBench public, official WorkflowEvals, grouped typed-decisions, Hebrew pairs, and an image benchmark. Promote no checkpoint based on a public benchmark result.

### G1 — flagship text decision model

Only if an untouched baseline does not already clear the full target:

- Base: Gemma-4-12B-it, or Winnow as an explicitly attributed initialization.
- Readout: constrained final-token candidate symbols first; the current Typical head and a Strands-style pointer are ablations.
- Adaptation: rank-32 LoRA on `q/k/v/o` and MLP projections; native vision path stays frozen initially.
- Context: train/evaluate length buckets through at least 16k; official agent-trace inputs reach 44k characters.
- Data: 30k–100k quality-filtered grouped cases, long policy/doc examples, option permutations, counterfactual fact changes, hard negatives, and 30–40% broad replay.
- Loss: hard-label full-vocabulary CE; `KL(target || model)` for valid soft distributions; `0.5 × RPS` on Score; teacher loss only when teacher argmax agrees with verified gold.
- Calibration: disjoint per-kind temperatures. No public benchmark labels.

### G2 — fast 4B lane

Start from Intern-Decision-4B or reproduce its final-marker objective on Qwen3.5-4B-it. Keep the vision tower/projector frozen and adapt the language model with rank-16/32 LoRA. This lane must pass sealed generalization before it can replace the 12B flagship; current 4B leaderboard entries often gain composite rank from speed/cost while losing badly on sealed Noul.

### G3 — Hebrew and multimodal retention

Do not mix these into the first competence run.

- Use paired Hebrew-state/Hebrew-options, Hebrew-state/English-options, and English semantic twins.
- Add a small cross-language probability-consistency loss.
- Use a separate or zero-initialized residual adapter and frozen-teacher KL replay so the English decision function starts exactly preserved.
- Include verified image-dependent mismatch/null rows, not only removed-option nulls.
- Advance only if Jev and workflow gates do not regress beyond their declared tolerance.

### G4 — serving optimization

After quality is frozen: Q8, shared-prefix packing, concurrent questions, CUDA graphs/fused kernels if profiling justifies them. OpenJev-Fast is an implementation reference, not a dependency.

## Data and leakage rules

- Quarantine JevBench, WorkflowEvals, typed-decisions test, and Image JevBench from training and checkpoint selection.
- Hash normalized state/question/options, preserve source group IDs, and near-deduplicate against every locked benchmark.
- Do not combine overlapping Open-Jev configurations blindly.
- Exclude Sarge `yuri_v1` uniform placeholders; treat `yuri_v3` as teacher distillation and disclose it.
- Keep translations, option permutations, and counterfactual siblings in the same split.
- Maintain two result tables: independent zero-shot/generalization and teacher-distilled/specialist.

## Immediate execution

G0 has started on one RTX PRO 6000: pinned Cygnet reproduction (`203/231` identity target), then all 705 official WorkflowEvals cases against the untouched checkpoint. This is higher-information and cheaper than another E1-style continuation. The GPU run must stop after results are persisted; no training begins until this baseline table is complete.
