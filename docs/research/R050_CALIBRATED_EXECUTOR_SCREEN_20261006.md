# R050-CDX: calibrated exact-outcome post-training screen

**Date:** 2026-10-07  
**Status:** GPU-ready internal screen; R047 failed, so R044-R is the parent. R048 v1.6,
comparator v5, the balanced headroom runner, and the unattended pipeline passed independent
adversarial review. GPU execution still begins with a separate headroom receipt. R048 v1.4 was
rejected because its value/operator holdout was nominal;
v1.5 was rejected because its semantic key aliased distinct collect policies and its comparison
trust boundary admitted a rewritten subset manifest.  
**Goal:** repair the field-to-action objective mismatch that keeps Typical below Jev on
WorkflowEvals while preserving its promoted Hebrew and real-pixel behavior.

## Hypothesis

Typical is trained mainly on individual typed fields but WorkflowEvals scores the complete
executable action outcome. Starting from one accepted parent, adding a strictly proper loss over
the exact outcome distribution will improve unseen-policy complete actions and calibration beyond
matched field supervision without degrading predicate truth, Hebrew, or visual grounding.

For deterministic policy `F`, Boolean field assignment `z`, and field probabilities `p_i`:

```text
P_theta(o | x, F) = sum_{z: F(z)=o} product_i p_i^z_i (1-p_i)^(1-z_i)
```

This distribution is exactly differentiable. Sampling the final decision with REINFORCE would
replace an exact gradient with a higher-variance estimator.

## Frozen chronology

1. Use R044-R. R047 failed its co-primary K2 and K3 gates, so its treatment is not a valid parent.
2. Freeze and independently audit R048 v1.6 from persisted JSONL bytes. R048 v1.3 is rejected for
   serialization, repetition, family-protection, and policy-holdout defects. R048 v1.4 repaired
   those mechanics but is also rejected as an advancement receipt: its declared value/operator
   families did not alter executable facts, and decorative family labels contaminated its
   semantic signature. R048 v1.5 repaired the executable factors, but is rejected because an
   unlabeled outcome partition merged distinct field-to-action mappings. V1.6 keys the complete
   canonical assignment-to-outcome function, including primary action and typed payloads.
3. Run an inference-only headroom split of 50--100 new semantic clusters, separate from final
   evaluation. Require at least 10% complete-outcome error overall and nontrivial errors for both
   policy modes and field count at least five. A saturated panel must be hardened, not trained on.
4. Seal all three one-seed checkpoints before evaluation. Evaluate with the v5 comparator, whose
   inference unit is the split-neutral `semantic_signature`, not fresh record or cluster IDs.
   Canonical pass/fail requires externally frozen hashes for both panel and manifest; the generic
   row API is diagnostic-only and cannot issue the Stage-1 gate.
   V5 additionally hard-requires the full 2,688-row/336-cluster canonical evaluation and exact
   default manifest cardinalities, closing a demonstrated 400-row cherry-pick false pass.
5. Repeat the control and one internal winner for two more seeds only after the first mechanism
   gate passes.

No official WorkflowEvals case, action, label, or public JevBench answer is used for optimization or
checkpoint selection.

## Matched arms

| Arm | Objective | Scientific question |
|---|---|---|
| A | field CE | continued-training control |
| B | field CE + exact-outcome NLL | does direct downstream alignment help? |
| C | field CE + categorical exact-outcome Brier | does a bounded proper outcome loss help? |

Choose the B/C coefficient with a training-only 64-signature gradient probe so the initial
outcome-objective gradient norm is 25% of the field-CE norm. The probe cycles evenly across both
counterfactual sides, EN/HE, and both option orders (eight examples per combination), rather than
estimating the coefficient from English/canonical-order views only. The arms must share parent,
examples, order, optimizer
slots, seed, token budget, updates, and serving readout. Outcome-only training is diagnostic at
most: a correct final action can mask compensating predicate errors.

Suggested 3,200-slot mixture:

- 50% executable R048 workflow groups;
- 25% broad text/action replay;
- 12.5% protected real-pixel replay;
- 12.5% grouped Hebrew replay.

Keep EN/HE twins, both option orders, and counterfactual sides in the same semantic split and
complete eight-row serialized block. Each panel row remains one exact-policy optimizer slot whose
field questions are grouped internally; this avoids inventing an unsupported cross-policy wrapper.
The 400-step, microbatch-1, accumulation-8 run consumes all 3,200 slots exactly once under the same
seeded global shuffle in all arms; serialization adjacency is not misreported as a joint optimizer
slot or block-aware sampler.
The current trainer also cannot group multiple image views into one slot, so the executable v1
protocol uses deterministic singleton real-image replay; removed-image, shuffled, mismatched, and
gold-evidence-dropped variants remain locked evaluation controls. Freeze the vision
tower/projector. Language-adapter updates can still change how image features are consumed, so all
pixel-control retention checks remain mandatory.

The actual replay pool is frozen from native sources at 400 Hebrew and 400 real-image slots. Hebrew
gold positions are exactly balanced (40 at each of ten positions). Multimodal selection excludes
validation/evaluation image hashes and decision tuples, uses unique source/image identities, and
copies 400 decodable image files into the immutable artifact. Its manifest SHA-256 is
`d556492b58922a8f3a7e51b3f907611404e9c02fc92c0084f7e09e422a3f4e3b`.

The final mix is `.context/r050-training-mix-v2`: 3,200 optimizer slots, 200 complete executable
orbits, 400 Hebrew rows, 400 real-image rows, and all 112 executable factor cells. Its training
SHA-256 is `499dd9c9a8341e55d1d965bbd1ffc45c893880fcdab4d35d295d06324c1c2003`;
its manifest SHA-256 is
`2b52da6fac16a803d5dd2fa14bcc54c68693f01754b3039561386602f287b53d`.
Twenty locked inputs quarantine 49,838 content hashes and 4,788 identities with zero selected
overlap. An independent rebuild reproduced every byte.

Retention uses the exact seven 128-row R042/R044 protected suites under `data_he/eval` and
`data_v/eval`, not the first 128 rows of the later full corpora. Historical launch logs and the
R042/R044 reports independently bind those hashes. This distinction matters: the later full
Hebrew corpora contain the same source identities but different sampled candidate decisions.
The selected ScienceQA pixels remain authenticated through the frozen full-corpus asset inventory.

The frozen pipeline runs headroom, the balanced gradient probe, opaque matched-arm preparation,
all three training arms, all three complete internal evaluations, post-evaluation unblinding,
candidate-versus-control comparison, protected retention for passing candidates, and a
deterministic winner choice. It never calls JevBench or WorkflowEvals. After crash-recovery and
between-stage code-integrity red-teaming, a dry run against the real frozen inputs completed with
zero GPU calls. The binding digest is path-sensitive and is therefore frozen again after remote
staging rather than copied from the local workspace.

## Internal advancement gate

A candidate must beat A on all of the following:

- complete-outcome accuracy `>= +3` points with semantic-signature paired 95% lower bound `> 0`;
- held-out separating-policy accuracy independently `>= +3` points with lower bound `> 0`;
- categorical complete-outcome Brier `>= 5%` relative improvement with lower bound `> 0`;
- complete-outcome NLL degradation upper bound `<= 0.02` nats;
- field-accuracy point loss `<= 1` point and paired lower bound `>= -2` points;
- every program and action family loses at most two points;
- semantic option-order agreement `>= 99%` and order-accuracy spread `<= 1` point;
- zero missing or invalid predictions;
- Hebrew, English, real-image, mismatch-null, and gold-dropped accuracy loss `<= 1` point;
- protected-suite Brier and NLL relative regression `<= 5%`.

The comparison is fail-closed when family metadata or a held-out policy slice is absent. Repeated
latent cases are one inference unit even when rendered with new values, IDs, translations, orders,
or counterfactual views.

Only an internal winner may scale to 30k--100k independently generated groups. It then needs a
paired WorkflowEvals gain of at least two points over R044-R's `52.81%` screen with no workflow
losing more than two points. Final workflow success remains the frozen Jev target (`>=67.8%` mean
and every workflow floor); JevBench requires a fresh official sealed submission rather than
optimizing the public diagnostic set.

## Why not online RL yet

The public TypeSafe name “Reinforcement Learning for Calibrated Decisions” does not disclose its
reward, optimizer, data recipe, or ablations. The independent OpenJev-RLCD implementation found
that calibrated readout training supplied most of its benefit; rationale REINFORCE added a modest
GSM8K-Verify gain but was statistically indistinguishable from readout continuation on MMLU-Pro.
It also reports empty-rationale collapse without an anchor and a score-function gradient 7--12
times larger than the useful pathwise gradient.

Online rationale RL becomes eligible only if a frozen four-rationale probe shows either `>=+5`
points single-rationale complete-outcome accuracy or `>=10%` lower NLL than direct scoring. Then
compare Stage-1 continuation, proper-score rationale RLCD with leave-one-out baselines and a frozen
KL anchor, correctness GRPO, and an anchor-only control. RLCD must beat continuation by at least two
exact-action points and 5% Brier.

Absolute executable truth is richer than pairwise preferences. DPO is therefore not the main
objective; factual twins, option permutations, EN/HE twins, and pixel controls remain bounded
contrastive auxiliaries introduced one factor at a time after an A/B/C winner.

## Architecture decision

Do not replace Gemma attention for R050. Kimi Delta Attention is an efficiency architecture whose
published result depends on native large-scale training or conversion distillation. FlashKDA's
optimized path targets SM90+, while the current A100 is SM80. Neither addresses the observed
field-to-action objective mismatch.

If R050 exposes high field accuracy but poor evidence binding, the next architecture probe is a
small option-to-evidence routing layer plus cross-field interaction layer, motivated by Clef and
Strands Decider. It must beat the constrained readout by five points on a frozen dispersed-evidence
probe before any invasive backbone change.

## Cost and falsification

At the current `$1.59/hour` A100 rate, expected cost is approximately:

- headroom inference: 0.5--1 GPU-hour;
- three 400-step arms plus internal evaluation: 12--16 GPU-hours (`$19--25`);
- two-seed control/winner confirmation: another 10--14 GPU-hours.

Reject the calibrated-executor hypothesis if the advantage disappears under latent-signature
clustering, does not transfer to separating policies, comes from post-hoc temperature fitting,
improves actions while degrading predicates, or violates any protected Hebrew/image floor.

## Primary evidence

- [OpenJev-RLCD](https://arxiv.org/abs/2609.38850) and its
  [reference implementation](https://github.com/ZimmyGao/openjev-rlcd)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [QRPO](https://arxiv.org/abs/2507.08068)
- [Semantic Loss](https://proceedings.mlr.press/v80/xu18h.html)
- [Verifiable Process Rewards](https://arxiv.org/abs/2605.10325)
- [mDPO](https://arxiv.org/abs/2406.11839)
- [Clef decision models](https://blog.cloudflare.com/clef-decision-models/)
- [Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/)
- [Kimi Linear](https://arxiv.org/abs/2510.26692) and
  [FlashKDA](https://github.com/MoonshotAI/FlashKDA)
