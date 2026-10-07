# R050 diagnosis against the live JevBench v1.5.4 target

Date: 2026-10-07

## Evidence boundary

The authoritative scored artifact used here is the released first-party API response at
<https://benchmarkheaven.com/api/jevbench/v1.5.4>. It identifies protocol `jevbench::v1.5`,
source SHA-256 `452885de2a84cd5b9ed393d541fd8d6c6a540f9d7f762f9d2df9349383f26173`,
method SHA-256 `c25d3d8b8512e4d93370a9e0c99705d19b2a9389956ca33b8a4bd2b0ec501c07`,
headline-addendum SHA-256 `752ddccc4e191c81412631ee3121563421c4b9077cb6c3f2243ed1d1545cfab0`,
and 1,624 decisions: 904 open and 720 sealed. The public GitHub repository's README still
labels v1.4.2.2 current, although its v1.5 method files exist. The API artifact, not that stale
README label, defines this project's current parity target.

The disclosed post-leaderboard headline amendment changes the frozen proposal to equal weights
for the four axes and equal weights for Choice, Noul, and Score. All targets and interpretations
below use released headline A, not the earlier 40/20/20/20 or 50/25/25 proposals.

No sealed item, answer, or prediction is available to or used by Typical. The table contains only
published aggregate axes and chance-corrected type splits.

## What the leaders actually show

| System | Score | Intelligence | Calibration | Speed | Cost | Open / sealed Intelligence |
|---|---:|---:|---:|---:|---:|---:|
| Cygnet, frozen Gemma-4-12B-it | 73.7013 | 71.0924 | 87.0073 | 90.9726 | 56.4291 | 72.4639 / 69.7209 |
| Winnow-12B Q8, Gemma-4 LoRA | 73.2331 | 74.4347 | 84.0699 | 86.1396 | 56.5560 | 72.4759 / 76.3934 |
| Jev 1.13.0 | 72.1329 | 71.9997 | 88.0290 | 83.8141 | 54.7250 | 71.5492 / 72.4503 |

The matching open Intelligence of Cygnet and Winnow hides a large sealed difference. Their
published sealed chance-corrected type scores are:

| Type | Cygnet | Winnow | Winnow minus Cygnet |
|---|---:|---:|---:|
| Choice | 80.2873 | 82.0445 | +1.7572 |
| Noul | 55.6727 | 66.1455 | +10.4727 |
| Score | 73.2027 | 80.9903 | +7.7876 |

This is aggregate association, not causal evidence about Winnow's recipe. It nevertheless rules
out one tempting diagnosis: generic Choice accuracy is not the only important gap. Binary semantic
orientation, ordinal distributions, calibration, and sealed transfer matter materially.

The calibration components give a second constraint. Cygnet's aggregate calibration is strong,
but Winnow improves Noul Brier (`0.09627` to `0.08320`) and ordinal NRPS (`0.05563` to
`0.04821`) while losing Choice calibration. Jev remains strongest overall at `88.0290`. A training
change that raises hard choices but makes probabilities less proper can miss Jev despite higher raw
accuracy.

## Current best hypothesis

R050 is the correct next causal screen:

1. Keep the strong general multimodal Gemma-4-12B-it backbone and the already promoted Typical
   adapter lineage.
2. Compare matched control, categorical-NLL, and categorical-Brier continuations on the same
   independently generated executable policy mixture.
3. Train complete grouped outcomes rather than only independent fields, while retaining explicit
   Choice, Noul, and Score examples.
4. Keep Hebrew and real-image replay in every matched arm and require both retention batteries
   before promotion.
5. Select only on the frozen internal executable/action panel. WorkflowEvals and public JevBench
   remain one-shot external gates, never development losses.

Why this is preferable to immediate RL: for a bounded typed decision with known executable gold,
the complete probability of the correct joint outcome is differentiable. Exact NLL and Brier use
all counterfactual option probabilities without sampling variance. Policy-gradient estimators
would throw away information unless the reward depends on a latent, non-differentiable rollout.
R050 therefore tests the lower-variance estimator first.

Why this is preferable to replacing attention now: the v1.5.4 leader is frozen Gemma-4-12B-it,
which proves the backbone can already reach the target regime without Kimi Linear Attention or a
new attention stack. KDA may improve long-context efficiency, but conversion would change both
capacity and pretrained dynamics and require substantial recovery training. It is not the current
bottleneck hypothesis.

## Predeclared next actions by outcome

| R050 result | Diagnosis | Next experiment |
|---|---|---|
| A proper-loss arm wins action outcomes and passes Hebrew/MM retention | The grouped objective supplies useful signal | Freeze the winner; run WorkflowEvals once, the 231-row public diagnostic once, then request the official 1,624-row evaluation without adapting to either result |
| Accuracy improves but Brier/NLL or retention fails | The treatment is overconfident or destructive | Repeat winner and control with two fresh training seeds; add KL-to-parent only if the failure reproduces |
| Per-field accuracy is high but complete action sets remain low | Independent marginals do not model conjunction risk | Test a 2x2 matched screen: proper loss × group-DRO over complete policies, with worst-group and exact-set loss computed from executable programs |
| Noul orientation or Score ordering fails on independent synthetic panels | Type semantics are underidentified | Add balanced label-order counterfactual pairs and an ordinal cumulative-link auxiliary loss; retain the same option-logit serving contract |
| Long-state errors persist after controlling for type and arity | Evidence retrieval, not output learning, is limiting | Run the frozen pointer/evidence probe, then test a zero-initialized option-to-evidence residual module; architecture changes require an independent long-policy win before external evaluation |
| All matched arms are null | The current data/objective has no useful gradient at this parent | Run a small oracle-overfit test on held-out generated programs. If it cannot fit, change representation/capacity; if it fits but does not transfer, change generator diversity and group weighting |

## Conditional RLCD-like follow-up

An RL stage is justified only if the next task includes non-differentiable workflow execution or a
latent action sequence. The candidate method is **counterfactual setwise policy optimization**:

- sample a complete action set from the model's typed marginal distributions;
- execute it in a generated workflow simulator and score exact business outcome, constraint
  violations, and calibrated abstention;
- use the exact enumerated set reward as a control variate wherever the action space is tractable;
- optimize a clipped policy objective plus parent KL and per-type Brier regularization;
- group-DRO the reward by workflow family, action-set size, state length, language, and image use;
- prohibit WorkflowEvals/JevBench prompts, outputs, and scores from the rollout or reward stream.

This is not scheduled ahead of R050. It becomes scientifically useful only if exact supervised
proper loss succeeds locally but fails on simulator-level execution, or if important rewards cannot
be differentiated. Otherwise it is a higher-variance reimplementation of information we already
have.

## Success remains conjunctive

Typical reaches the project goal only when one immutable model simultaneously demonstrates:

- official JevBench v1.5.4 score at least `72.13292897895896` and Intelligence at least
  `71.99974990450261`, with 1,624/1,624 valid answers;
- WorkflowEvals equal-workflow mean at least `0.678`, every workflow at or above Jev's published
  row, and all 705 cases / 1,524 decisions present;
- the frozen Hebrew and multimodal retention gates from R050;
- reproducible artifact, code, runtime, and evaluation receipts.

Public-subset proximity, a faster composite alone, or a model that discards Hebrew/vision does not
satisfy the goal.
