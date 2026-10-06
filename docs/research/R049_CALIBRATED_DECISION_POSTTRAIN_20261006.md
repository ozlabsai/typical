# R049: calibrated-decision post-training for Typical

**Date:** 2026-10-06  
**Status:** prospective implementation and CPU validation; no GPU arm is authorized before the
R047 parent decision and R048-H headroom gate  
**Goal:** test whether a bounded proper score on complete executable outcomes improves
WorkflowEvals-style decisions beyond matched predicate supervision while preserving Hebrew and
multimodal behavior.

**Implementation status:** Stage-1 categorical Brier training, fail-closed threshold/temperature
validation, explicit objective telemetry, and complete-outcome NLL/Brier/ECE/AURC scoring are
implemented in commits `4058ba0`, `e729d1b`, `dc40fe4`, and `cb736e1`. Independent audits found no
blocking defects. Rendered rows carry calibration metrics; semantic endpoint/cluster aggregation is
explicitly a robustness summary and does not claim a joint calibrated confidence.

## Evidence boundary

The acronym `RLCD` refers to different methods and must not be used without expansion:

- TypeSafe calls Jev's undisclosed method **Reinforcement Learning for Calibrated Decisions**. Its
  public launch describes a new architecture, parallel sampler, typed distributions, and RLCD, but
  publishes no reward equation, optimizer, data recipe, or causal ablation.
- Yang et al.'s ICLR 2024 **Reinforcement Learning from Contrastive Distillation** constructs
  positive/negative prompted responses, trains a preference model, and runs KL-regularized PPO. It
  is useful evidence for constructing differentiated examples, not a calibrated-decision recipe.
- Gao and Wang's independent **OpenJev-RLCD** (arXiv:2609.38850, 2026-09-30) samples rationales,
  differentiates a proper score through the post-rationale answer distribution, then optionally
  reinforces rationales with a leave-one-out REINFORCE estimator. It explicitly does not claim to
  reproduce Jev's private algorithm.

OpenJev-RLCD found two important failure modes: end-to-end training without an anchor can collapse
to an empty rationale, and its score-function gradient was 7--12 times larger than the useful
pathwise gradient. Its Stage 1 supplied most of the gain; Stage 2 helped the computation-heavy
GSM8K verification task but was neutral against readout continuation on knowledge-heavy MMLU-Pro.
All headline results used development-set temperature scaling, and the evidence covers one 1.7B
model, two reasoning tasks, and no Hebrew, vision, WorkflowEvals, or JevBench.

Primary sources:

- TypeSafe launch: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe WorkflowEvals: https://evals.typesafe.ai/
- OpenJev-RLCD paper/code: https://arxiv.org/abs/2609.38850 and
  https://github.com/ZimmyGao/openjev-rlcd
- Calibration rewards: https://arxiv.org/abs/2507.16806
- Multimodal DPO failure and image-conditioned repair: https://arxiv.org/abs/2406.11839
- Proper scoring rules: https://doi.org/10.1198/016214506000001437
- Contrastive-distillation RLCD: https://arxiv.org/abs/2307.12950

## Hypothesis

> Starting from the same accepted parent, clause-local supervision plus a bounded proper score on
> the exact complete outcome distribution will improve confidence ranking and unseen-policy exact
> outcomes beyond matched field CE and exact-outcome NLL, without erasing Hebrew or visual
> grounding.

The proposed mechanism is precise. Executable counterfactuals identify which evidence changes a
predicate; field supervision prevents compensating predicate errors; exact policy marginalization
assigns downstream credit; categorical Brier penalizes confidently wrong complete outcomes while
remaining bounded.

Brier is not theoretically superior to cross-entropy: both are strictly proper. Their relative
optimization and generalization behavior is the experiment.

## Stage 1: exact calibrated-decision objective

For Boolean field assignment `z`, deterministic policy `F`, and model field probabilities, define
the architecture's factorized outcome distribution

```text
P_theta(o | x) = sum_{z: F(z)=o} product_i p_i^z_i (1-p_i)^(1-z_i)
```

and categorical complete-outcome Brier loss

```text
L_Brier = sum_o (P_theta(o | x) - 1[o = gold])^2.
```

This is exact for the declared independent-Bernoulli architecture, not proof that real predicates
are independent. Outcome calibration experiments freeze the semantic decision threshold at `0.5`
and action temperature at `1.0`; otherwise the executor probabilities are deliberately transformed
threshold surrogates rather than literal predicate probabilities.

Matched one-seed screen on one immutable corpus:

| Arm | Loss | Role |
|---|---|---|
| A | field CE | matched continued-training control |
| B | field CE + complete-outcome NLL | existing semantic-loss treatment |
| C | field CE + complete-outcome Brier | calibrated-decision hypothesis |
| D | outcome Brier only, short diagnostic | expose predicate masking; never promotable |

Choose fixed B/C coefficients from a training-only gradient probe so the initial outcome-objective
gradient norm is 25% of the field objective. Do not tune weights on official WorkflowEvals. Repeat
only A and the internal-validation winner for two additional seeds.

Outcome-only optimization is unsafe as a main recipe. A final action generally does not identify
every predicate, so compensating field errors can produce the right action. Every primary arm keeps
field/process truth, exact complete actions (primary, all secondary actions, and typed payloads), and
unseen separating-policy evaluation.

## Data and preservation

Split semantic cases by source/template/policy family before rendering. Keep every translation,
option order, counterfactual, image variant, and policy view for one case in one split. Include:

- one-fact decision-flipping twins and irrelevant changes that must preserve the outcome;
- priority, collect, fallback, empty-set, nullable-primary, multi-action, and typed-payload policies;
- held-out policy topologies that separate otherwise outcome-masked predicates;
- current-model high-confidence errors, capped so natural calibration priors are retained;
- natural Hebrew rather than translatedese, with semantic EN twins;
- real-image evidence changes, plus removed, shuffled, mismatched, and text-held-fixed controls;
- both option orders, field permutations, identifier renamings, and irrelevant opinion/instruction
  injection as robustness probes.

Freeze the vision tower/projector initially. Language-adapter updates can still change use of image
features, so protected pixel and Hebrew replay plus paired grounding tests remain mandatory.

## Stage 2: genuine online RL gate

Typical currently makes one parallel constrained decision and has no sampled rationale. For its
finite exact policy, direct autograd computes the expectation that REINFORCE would estimate with
higher variance. Therefore Stage 1 is RLCD-like calibrated-decision post-training, not a claim of
online RL or a reproduction of Jev.

Online RL becomes eligible only if a frozen validation-only probe with four bounded rationales
shows either:

- at least `+5` points single-rationale complete-outcome accuracy over direct scoring, or
- at least `10%` lower complete-outcome NLL,

while preserving Hebrew, vision, and latency constraints. Mixture-of-rationales improvement alone
does not qualify because it can reward ungrounded disagreement.

If eligible, fork the Stage-1 checkpoint into matched readout continuation, proper-score rationale
REINFORCE, correctness-reward GRPO, and anchored no-reward control. Use per-rationale proper scores,
leave-one-out baselines, a frozen Stage-1 KL anchor, and a separately tuned small score-function
coefficient. Advance online RL only if it beats readout continuation by `>=2` unseen-policy points
and `>=5%` relative Brier with paired confidence intervals excluding zero.

## Advancement gates

Stage 1 advances only if all hold against matched field CE:

- unseen-policy complete-outcome accuracy: `>=+3` points and scenario-clustered 95% lower bound
  above zero;
- complete-outcome Brier: `>=5%` relative improvement with clustered interval excluding zero;
- complete-outcome NLL degradation upper bound: `<=0.02` nats;
- field accuracy loss: `<=1` point, with paired lower bound `>=-2` points;
- every action/policy family loss: `<=2` points;
- validation-frozen coverage: `>=+5` points at a `10%` complete-outcome error budget, whose untouched
  test one-sided 95% upper bound remains `<=10%`;
- Hebrew, English, real-image, mismatch, and gold-dropped accuracy loss: `<=1` point each; their
  Brier/NLL may not worsen by more than `5%` relative;
- semantic option-permutation agreement `>=99%`, order accuracy spread `<=1` point, zero invalid
  predictions, and serving p95 regression `<=10%`.

Public WorkflowEvals remains a frozen downstream gate. A Stage-1 candidate must gain at least `+2`
points over R044-R's `52.81` mean with a paired case-cluster interval above zero and no workflow
losing more than two points before scale. The project goal remains Jev parity (`>=67.8` in the
frozen plan), not the `+2` screening threshold.

Kill the broad mechanism claim if gains vanish on unseen policies, exact outcomes improve while
predicate truth degrades, calibration gains come only from post-hoc temperature fitting, image
controls reveal text-only shortcuts, or gains disappear when all actions and payloads are scored.

## Interpretation discipline

- Report raw and validation-temperature-scaled NLL, Brier, AURC, and risk/coverage. ECE is a
  diagnostic whose answer depends on binning, not a sole gate.
- Bootstrap semantic cases, not rendered questions or multiple views of one case.
- Type safety means schema validity, not factual correctness or robustness to manipulation.
- WorkflowEvals measures agreement with external-model reference actions, not calibrated business
  truth; it cannot be the only calibration dataset.
- Attach repository/version to `JevBench`: several unrelated projects use that name, and coherence
  alone can reward an input-ignoring uniform predictor.
