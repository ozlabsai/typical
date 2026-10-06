# Post-R047 strategy: protected grouped workflow training

**Date:** 2026-10-06  
**Status:** prospective; R047-C reference/control/treatment chronology remains authoritative  
**Goal:** close the gap from R044-R's `52.81%` WorkflowEvals mean to the locked `>=67.8%`
gate without losing the promoted multimodal and Hebrew behavior.

## Decision

The highest-value method after R047-C is **protected grouped workflow training with a
train-only differentiable executor**.

The current model is trained primarily on local field decisions, while the benchmark scores the
complete downstream action set. The missing path is:

```text
shared state -> grouped field distributions -> policy rules/thresholds -> exact action set
```

R047-C tests whether executable clause supervision repairs the first two links. A subsequent
matched experiment should test the remaining probability-to-policy link directly. It must train on
independently generated workflow programs, never copied WorkflowEvals cases, wording, actions, or
labels.

This is a positive candidate rather than another generic regularization screen. It explains why
prior broad LoRA, soft distillation, Boolean coverage, and pair-weight changes could improve local
metrics without closing the workflow gap.

## R047-C decision tree

1. **Full mechanism pass:** clause/vector/coherence/final transfer all pass.
   - If retention also passes, initialize the grouped workflow pilot from the treatment checkpoint.
   - If retention fails, retain the mechanism but discard the weights. Restart from R044-R with a
     fresh residual adapter and protected parent-logit replay.
2. **Clause gain without final/coherence gain:** treat this as an aggregation bottleneck. Make the
   soft-executor action loss the next isolated factor; do not merely increase clause dose.
3. **Only numeric primitives remain weak:** add machine-executable operand -> value -> predicate ->
   action traces for decimal, temporal, and ledger cases. Do not use prose chain-of-thought.
4. **Final gain without clause gain, or gain confined to one answer order/language:** treat as a
   shortcut/confound and do not scale those weights.
5. **Clean 6% dose null:** run only the preregistered 25% R047-C2 dose screen. If that also fails,
   drop indexed-local auxiliaries and proceed with grouped field plus action-set supervision.

A mechanism pass without the inherited multimodal/Hebrew receipt is research evidence only, not a
product initializer.

## Matched pilot

Use at least 1,600 executable workflow groups plus an equal number of protected replay slots. Each
group contains one shared state, 2--12 related typed questions, field labels, executable clause
values, a primary action, and the complete action set.

Hold out program topologies, generator families, templates, value regimes, threshold intervals,
and semantic clusters. Required families include conjunction/disjunction/negation,
priority/fallback, authorization, missing evidence, dates, amounts, ledgers, and typed action
payloads. Include minimal fact twins that change exactly one predicate or action.

Matched arms:

- **A:** grouped field loss, plus clause/coherence loss only if R047-C validates it.
- **B:** exact same groups, views, replay, seed, optimizer slots, and updates, with an additional
  `0.1 * action-set loss` routed through the soft executor.

Compile each deterministic policy to a training-only relaxation:

- thresholds -> temperature-controlled sigmoids;
- AND/OR/NOT -> stable differentiable t-norm operations;
- priority/fallback -> normalized gated routing;
- each action -> a soft activation probability.

For a workflow group:

```text
L_field  = hard-label CE/KL + 0.5 * RPS for ordinal fields
L_action = BCE(soft action indicators, gold action set)
           + CE(soft primary action, gold primary action)

L_total  = L_field
           + I[R047 validated] * lambda_process * L_clause/coherence
           + 0.1 * L_action
           + lambda_cf * L_counterfactual-margin
           + 0.05 * JS(aligned option permutations and HE/EN twins)
           + 0.05 * KL(parent || student) on protected replay
```

For finite action sets with executable rewards, exact reward-tilted targets are preferable to
sampling-based GRPO/PPO:

```text
q_beta(a | s) proportional to pi_ref(a | s) * exp(R(s, a) / beta)
```

This is a lower-variance proper target because all legal actions can be enumerated and scored.
Online RL is reserved for a slow multi-turn teacher only if exploration changes future state and a
verified best-of-eight oracle shows at least five points of headroom over direct decisions.

## Protected multimodal and Hebrew mix

Suggested optimizer-slot mix:

- 50% new executable workflow groups;
- 25% broad text/action replay;
- 12.5% real-pixel groups;
- 12.5% grouped Hebrew semantic triples.

Every visual semantic group must include real, shuffled/no-image, mismatched, evidence-removed, and
counterfactual-image views. Useful process labels are `evidence_visible`, `field_read`, and
`state_image_consistent`; raw image-present accuracy is not sufficient evidence of grounding.

Each Hebrew group includes HE state/HE options, HE state/EN options, and its English semantic twin.
Align distributions by semantic option ID rather than surface position. Keep complete groups in one
split and one optimizer slot.

Freeze the vision tower/projector initially. Use a fresh zero-initialized workflow residual adapter
from the accepted parent, with protected parent-logit replay. Select checkpoints by hard per-suite
floors; never average workflow, Hebrew, English, and image losses into one selector.

## Advancement gates

Advance arm B only if all conditions hold:

- independent held-out exact actions improve by at least 3 points with a scenario-clustered 95%
  lower confidence bound above zero;
- the gain persists on unseen program topologies and threshold ranges;
- no program family loses more than 2 points;
- field option accuracy does not fall, and NLL/Brier worsen by no more than 1%;
- Hebrew, English, real-image, mismatch-null, renderer, WF/WH, and replay remain within inherited
  floors.

Require at least an 8-point gain across two paired seeds before scaling to 30k--100k groups. Only a
frozen candidate then runs official WorkflowEvals, where success remains `>=67.8%` mean with every
workflow-specific floor satisfied.

## Architecture and efficiency decisions

Preserve the current constrained readout and external exact executor for the first pilot. Add a
joint head only if a frozen option-pointer/evidence-routing probe beats the current readout by at
least five points on long, dispersed-evidence groups. The smallest justified residual is one
option-to-state routing layer plus one field-interaction layer, width 256--512, zero-initialized on
top of existing logits.

Kimi-style hybrid linear attention is an efficiency branch, not the present quality intervention.
The published gains come from native pretraining or very large conversion distillation, not a small
post-hoc LoRA. Use prefix caching and grouped/shared-state execution first; reconsider a native
hybrid backbone only after the quality objective is frozen.

## Evidence basis

- [Cygnet recipe](https://github.com/blockbrain-ai/cygnet-recipe): frozen Gemma plus constrained
  one-token readout is a strong benchmark baseline, but public decision accuracy does not imply
  WorkflowEvals transfer.
- [Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/): option-aligned
  pointer evidence motivates a probe, not an immediate architecture replacement.
- [Clef decision models](https://blog.cloudflare.com/clef-decision-models/): grouped schema
  processing, option permutations, calibrated proper losses, and cross-field interaction are
  directionally relevant; its scale, data, head, and objective are confounded, so copy the
  experimental question rather than the whole head.
- [CLM](https://github.com/Contrastive-LM/CLM): broad contrastive competence followed by staged hard
  negatives and substantial replay supports introducing difficult counterfactuals after ordinary
  groups fit.
- [Bandits](https://github.com/bandr-ai/bandits): trace lineage, quarantine, and unambiguous outcome
  evidence are useful data-governance rules; reaction labels alone are not reliable workflow truth.
- [Open-Jev workflow controls](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/workflows.md):
  useful curriculum/control generators, not evidence on official WorkflowEvals.
- [Verifiable Process Rewards](https://arxiv.org/abs/2605.10325),
  [PROF](https://arxiv.org/abs/2509.03403), and
  [ThinkPRM](https://arxiv.org/abs/2504.16828): deterministic process truth and
  process/outcome-consistency filtering are preferable to directly optimizing a noisy learned PRM.
- [QRPO](https://arxiv.org/abs/2507.08068) and
  [A*-PO](https://arxiv.org/abs/2505.20686): closed-form or advantage-regression views support exact
  reward-tilted targets when the legal action set is enumerable.
- [Set-LLM](https://arxiv.org/abs/2505.15433) and
  [PIF](https://arxiv.org/abs/2406.01026): semantic-ID permutation alignment and explicit
  symbol/content binding should precede invasive set-attention changes.
- [Kimi Linear](https://arxiv.org/abs/2510.26692): native hybrid attention can materially improve
  long-context efficiency, but does not support low-cost post-hoc attention surgery as a workflow
  quality fix.

