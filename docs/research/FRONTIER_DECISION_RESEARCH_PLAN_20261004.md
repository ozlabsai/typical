# Typical frontier decision research plan

**Date:** 2026-10-04
**Goal:** build our own model from a fresh general multimodal foundation model and equal or beat Jev on official JevBench and WorkflowEvals, while retaining real image use and strong Hebrew behavior.

## Research contract

- Other decision models are references and optional disclosed teachers, never the initialization of the final model.
- The public 231-item JevBench subset and the 705 WorkflowEvals cases are evaluation-only. They do not select checkpoints, losses, temperatures, or data mixtures.
- Public accuracy is not a promotion gate: Winnow's `199/231` public result coexists with `74.4` official Intelligence, above Jev's `72.0`.
- Change one causal factor per screen. No extra seeds until seed 0 clears its predeclared internal gate.
- Quality is frozen before serving optimization. A faster failure is still a failure.
- A benchmark gain does not establish multimodality: images must be necessary to solve the multimodal rows.

## What has already been falsified

| Observation | What it rules out | What remains plausible |
|---|---|---|
| E1 changed only the shared head, improving image/Hebrew while Jev fell `164→159/231` and typed decisions `.373→.340`. | One unconstrained shared continuation safely adds MM/Hebrew. | Frozen text winner plus zero-initialized capability residuals and explicit retention losses. |
| Clean Intern-4B LoRA improved every internal NLL but fell `200→197/231`; soft distillation reached `199/231` and workflows `.588/.495/.469` on security/trace/invoice. | Lower training loss or more generic decision data predicts external workflow transfer. | Better lineage, evidence-necessary counterfactuals, downstream action loss, and a stronger fresh base. |
| Cygnet's untouched Gemma-4-12B gets `203/231`; Winnow's LoRA gets `199/231`. | A custom attention mechanism or head is required for public JevBench quality. | Data/objective improvements can help workflows, calibration, and sealed generalization. |
| G2 gets multi-hop `16/18` but temporal/numeric `4/15`; all strong baselines are weak on temporal/numeric. | The remaining error is a uniform semantic-capacity failure. | Targeted computation/data and bounded deliberation. |
| Best untouched WorkflowEvals mean is Winnow at `55.8%`, versus Jev `67.8%`. | Public JevBench parity implies workflow parity. | Probability-to-program calibration and executable workflow supervision. |

The current evidence does **not** isolate model capacity: base, data, objective, and scale differ across systems. It does show that generic fine-tuning and public-set accuracy are weak proxies for workflow exact actions.

## Completed control: S0 fresh-base option-logit LoRA

S0 starts from pinned `google/gemma-4-12B-it`, not Winnow or another decision checkpoint.

- exact Cygnet system/user scaffold;
- option probability from the LM's next-token logits, originally grouped over six uppercase/lowercase/leading-space aliases per letter;
- rank-32/alpha-64 language-only LoRA on attention and MLP projections;
- frozen vision path;
- option permutations during training;
- hard-label CE or forward soft-target CE on the supplied option set;
- 8,192-token cap, effective batch 8, `2e-5→2e-6` cosine schedule;
- 250-step screen over 92,331 available question examples (2,000 presentations seen).

This is a sensitivity control, not a full training claim. The text-only trainer proves that a multimodal base was preserved structurally; it does not train or validate image use because it supplies no pixels.

S0 also deliberately lacks base-logit retention, RPS, source-balanced validation, and executable action loss. Its first-128 validation is a smoke signal only. No positive or negative S0 result tests the full recipe below.

S0 finished all 250 updates. Re-scored with the production-compatible exact-uppercase token groups, it reached the same public diagnostic total as Cygnet, `203/231`, but failed transfer to official WorkflowEvals: security `.300`, agent trace `.536`, invoice exact `.298` (primary `.573`), customer `.716`, equal-weight mean `.462`. The invoice primary-action decline from Cygnet's `.813` proves this is not only a threshold/calibration failure.

The scorer audit found a real but practically negligible training/serving mismatch. Gemma has two token IDs that decode exactly to each uppercase option letter, while S0 optimized six aliases. vLLM structured choice masks out lowercase and leading-space aliases, but the legal aliases already held at least `99.995%` of the six-alias mass. S1 repeated S0 from the fresh base with only the legal-token scorer corrected. It did not recover the aggregate held-out tradeoff: on randomized 256-item panels, public-source/replay/workflow/hard-rule accuracy was `.652/.797/.832/.836`, versus S0's `.688/.785/.840/.832`. The scorer mismatch is rejected as the main cause. An adapter-scale sweep at `.25/.5/.75/1` also exposed no retention-preserving interior optimum.

### S0 failure localization

Invoice is dominated by one amplified field error, not a uniform reasoning decline. S0 answered `statement_not_invoice=true` on 143/150 cases, while the untouched Cygnet base answered false on all 150. Replaying the saved S0 probabilities through the pinned invoice workflow with only this field forced false (or replaced by the base probability) raised exact actions from `.298` to `.504` and primary actions from `.573` to `.798`. The change fixed 103 exact and 105 primary policy decisions, while breaking 10 and 4 respectively. Replacing only exact `.5` ties with base probabilities changed no actions. Calibration or scorer ties therefore cannot explain this failure.

The interface is out of training support in two concrete ways:

- the official question is a bare declarative, negated Noul proposition with `criteria=null`, rendered as generic `A. No / B. Yes`;
- all 19,823 training Noul questions have criteria, and none of the 2,967 generic No/Yes examples is a bare declarative proposition.

Invoice prompts are also much longer (median about 8.6k tokens, p95 about 11.5k) than the observed training support (maximum about 4.1k), while the shorter customer and trace workflows improved. Length is an amplifier in some settings, but the factorial probe did not isolate it as the first cause. On four S0-failing cases, full-state + bare-declarative + generic false-first `No/Yes` made S0 answer true on all four (`mean P(true)=.769`). Reversing option order, making the proposition interrogative, or supplying semantic option descriptions fixed all four. S1 reduced the full-state false-first error to two of four, but its compact-state version still failed all four (`mean P(true)=.900`). A benchmark-independent canonical fallback—`False/True` plus explicit statement semantics—fixed all four under both option orders and both context views. On the full 150-case invoice workflow, the same S1 weights scored `.396` exact actions and `.649` primary actions with the generic renderer versus `.504/.811` with canonical semantics (`+.109/+.162`). The latter exactly matches the earlier counterfactual replay and recovers the primary-action regression, but remains below Jev's `.618` invoice exact target. This is a symbol-binding/interface interaction, not evidence for an attention-capacity change.

The corpus has zero exact normalized state or full state/question/options matches with the 231 public JevBench rows; a near-duplicate/source-lineage audit remains required before any transfer claim.

## Hypothesis stack and experiment order

### H0 — Typed-interface coverage and symbol binding precede retention surgery

Random option shuffling did not teach the absent combination of a bare declarative proposition with generic binary labels. This matches published multiple-choice selector bias: semantic knowledge can survive while the letter/slot binding changes under option permutations. See [PriDe](https://arxiv.org/abs/2309.03882) and [PIF](https://arxiv.org/abs/2406.01026).

First evaluate a deterministic, task-agnostic fallback for `criteria=null`: preserve the proposition verbatim and render explicit semantic `False` and `True` descriptions. The transformation must apply to every Noul request and must not inspect workflow identity, case ID, or an expected answer. Validate it on an independent crossed panel before treating any official score as evidence.

Then run a matched data-only screen from the fresh base. Replace 512 of the fixed 2,000 S1 presentations with the same verified binary facts in two arms: an interrogative/explicit content control and a bare-declarative/generic robustness arm. Balance true/false labels, both option orders, negated propositions, compact/full context views, and hold out scenario groups. No KL, localization, new head, RL, or attention change enters this comparison.

The launched artifacts passed two independent semantic/leakage audits. Both arms share 1,488 examples and have 512 paired policy-permission views with identical states, targets, and fixed option orders. The full crossed panel has 2,048 disjoint views. Exact Gemma tokenization gives maxima of 3,181 training tokens and 9,298 evaluation tokens. This is presentation-matched rather than FLOP-matched: explicit criteria make Arm A's total tokens 1.15% higher. Claims remain scoped to policy permission and to the joint question-wording plus criteria intervention.

The paired result isolated convention interference. Arm A reached `.836` overall but collapsed on negated declarative/generic views (`4.69%`); Arm B reached `.911` and repaired that cell to `95.31%`, while negated explicit views fell from roughly `96–98%` to `78–79%`. B's paired 64-group advantage was `+7.42` points (case-bootstrap 95% interval `[+4.69,+10.06]`; one training seed). Exact rank-64 convex LoRA-delta mixtures were screened on all 1,024 compact views at the frozen interior weights `.25/.5/.75`. The best, `.75` B, reached `.943` overall but only `.875` in its worst format×polarity cell and `.914` on the target negated declarative/generic cell, missing the preregistered `.90/.93` gates. The grid was not refined.

Arm C used the unchanged 2,000-presentation budget: 1,488 shared rows plus 256 A-format and 256 B-format views. Independent audit caught two otherwise invisible data confounds before launch: a parity split that predicted the correct answer letter on `512/512` rows without reading facts, and an emission sort that moved `504/512` presentations. The corrected artifact balances both answer indices inside every visible format×context×polarity cell and preserves A/B presentation order position-wise. C improved overall crossed accuracy to `.931`, but failed the target negated declarative/generic cell at `.793`; its compact/full split was `.906/.680`, and the full false-first order fell to `.578`. WH/replay retention also fell to `.820/.859`. The failure is concentrated in long-context negation and does not establish that CE is inadequate: C halved each format's exposure and never showed both aligned formats for the same semantic view.

Arm D was the complete-view coverage screen. It kept the fresh base, rank-32 adapter, optimizer, seed, 250 updates, 1,488 shared logical slots, and 512 donor logical slots. Each donor slot computed `0.5 CE(A_i) + 0.5 CE(B_i)`, processing 2,512 physical views while preserving the shared/donor loss weights. It reached `.9380` overall but failed every decisive target gate: negated declarative/generic `.8164`, its two full-context orders `.7031/.7031`, declarative/generic aggregate `.8848`, worst format×polarity `.8164`, and probability order delta `.0521`. Replay `.8594` also missed its `.8628` floor. Relative to C, the overall gain was only `+.68` points (paired-group bootstrap 95% interval `[-.78,+2.20]`), while the target and target-full gains were both `+2.34` points with intervals crossing zero. Complete paired CE coverage is therefore insufficient; no extra epochs or coverage rebalance is warranted.

The canonical-teacher eligibility check passed. Frozen Arm A was scored on 128 training sources through compact, positive, interrogative, explicit views in both option orders. `121/128` sources met all locked conditions: both orders correct, minimum gold probability at least `.80`, and aligned `P(yes)` delta at most `.05`. Coverage was balanced across gold (`61/64` no, `60/64` yes) and training polarity (`62/64` negative, `59/64` positive). The panel and raw results hashes are `cacdd465...` and `3c81850c...`.

This is evidence that our own Arm A can supply a stable privileged view, not evidence that binary output distillation will transfer semantics. A binary teacher exposes only one case-specific scalar, so aligned label-preserving soft targets can collapse to ordinary confidence-dependent label smoothing; a within-label shuffled control may differ too little to falsify that explanation. No student run is authorized until the target distribution is measured and the objective plus negative control can distinguish semantic correspondence from regularization.

R035 isolates the smallest relational objective justified by the failure. It reruns matched Arm D from the same fresh base with the same data bytes, paired A/B inputs, rank-32 adapter, optimizer, seed, 250 updates, and evaluation. For each paired donor slot, map letter scores back to semantic `yes/no`, let `m = polarity_sign * (logit_yes - logit_no)`, and use the explicit/interrogative A margin as a one-way anchor for the declarative/generic B margin:

```text
teacher = clip(stop_gradient(m_A), -4, 4)
L = 0.5 CE_A + 0.5 CE_B
    + 0.25 * 1[P_A(gold) >= 0.80] * SmoothL1(m_B, teacher; beta=1)
```

Shared singleton slots remain unchanged. The gate avoids exporting an uncertain A prediction; stop-gradient prevents the weak B view from directly dragging A toward its failure. Cached D is the same-seed, same-environment matched CE-only control, so the intended experimental factor is only the added loss. CUDA deterministic algorithms were not enabled for D, so this remains a cached single-seed comparison rather than a bit-exact counterfactual; the material delta gates below are required. This follows the relational lesson from [PairCFR](https://aclanthology.org/2024.acl-long.646/) and the instruction-variation motivation of [consistency alignment](https://aclanthology.org/2024.lrec-main.782/), without importing their larger training machinery.

The mechanism gate is preregistered before launch: active pairs must exceed 50% over the final 100 updates; negative declarative/generic accuracy must improve by at least five points over D; full-context negative declarative/generic `no` recall must improve by at least ten points from its `.5000` baseline; the full/negative/middle A-to-B accuracy gap must shrink by at least ten points because B improves, not because A falls; no explicit/interrogative cell or WH/replay retention metric may fall by more than two points; order delta may not worsen; and overall NLL/Brier may not worsen by more than 5%. Mechanism success is not promotion: all prior absolute target, worst-cell, full-context, recall, order, and retention gates must still pass before official evaluation. Stop at step 250 with no coefficient search, EMA teacher, symmetric retry, or extra epochs.

If balanced augmentation improves order consistency but not counterfactual accuracy, test a swap-cancelled counterfactual margin: align both option orders to semantic true/false logits, penalize their log-odds disagreement, and require verified counterfactual twins to cross a margin. This is conditional; ordinary CE remains the default.

### H1 — Probability-to-program mismatch is a primary workflow bottleneck

Workflow code consumes probabilities through thresholds and action composition. Security, for example, uses Noul thresholds including `0.15`, `0.40–0.70`, and `0.75`; two models with the same argmax can therefore take different actions. This is a decision-focused-learning problem, not plain classification. [Smart Predict-then-Optimize](https://arxiv.org/abs/1710.08005) motivates optimizing downstream decision regret rather than only prediction error.

**First, no training:** audit that training and serving aggregate exactly the same allowed token surfaces, then replay saved probability vectors through independent synthetic threshold programs. Decompose every failure into:

1. wrong option ranking;
2. correct ranking but wrong side of a threshold;
3. correct field decisions but wrong composed action set;
4. invalid/missing output.

Fit scalar temperatures per decision kind and a binary affine-logit Noul calibrator on disjoint internal calibration data only. Evaluate threshold robustness over a sweep, not the official thresholds.

**Training ablation:** on independently generated executable workflows, compare:

```text
A: proper option CE/KL
B: A + 0.1 × smooth downstream action-regret loss
```

Promote B only if held-out executable-workflow exact actions improve by at least 3 points while raw NLL/Brier do not regress by more than 1%, option accuracy does not fall, and the gain persists across unseen threshold ranges.

### H2 — Evidence-necessary, lineage-preserving data beats generic distillation

The official workflow labels are consensus-reference actions, not realized environment outcomes. New training data must therefore have independent label lineage:

- programmatically executable policies and arithmetic;
- the exact evidence span or fact dependency that determines each answer;
- counterfactual twins where one decisive fact flips the label;
- near-tie and missing-evidence cases;
- temporal, numeric, probability, invoice-policy, and agent-trace emphasis;
- transformations and siblings kept in the same split.

For teacher probabilities, score four option permutations with two or more independent teachers, map back to candidate identity, and calibrate each teacher by kind. Use teacher loss only when its argmax agrees with verified gold and its permutation variance/disagreement is below the declared threshold. Teacher disagreement rows remain as gold-only hard examples.

The candidate target is:

```text
q = (1 - λ) one_hot(gold) + λ mean(calibrated teachers)
L = CE(q, p) + 0.5 RPS_score + 0.05 JS_permutation + 0.05 KL(base || student)_replay
```

Compare this recipe to S0 on identical presentations and seed. Promote only on held-out internal families; public benchmarks are run once after the candidate is frozen.

If a verified sufficient-evidence view beats the same base on full context by at least five points, test **evidence-privileged equivariant distillation** rather than adding attention. The frozen teacher sees only the verified evidence; the student sees the full context. Use teacher KL only when the teacher is gold-correct and confident, align option permutations back to semantic identity before a JS term, and require a minimal evidence counterfactual to flip the intended candidate. Kill this branch if the compact-evidence oracle has less than five points of headroom or a 10% pilot gains less than two held-out long-context points. This combines short-to-long self-distillation with symbol-binding and causal-evidence controls without adding serving cost.

### H3 — Diagnose the readout before adding attention

Run zero-training controls on S0:

- grouped questions versus one independent branch per question;
- question-order and option-order permutations;
- length and decisive-evidence-position buckets;
- final-letter logits versus a frozen-backbone Strands-style option-end pointer probe.

If grouped/single disagreement is above 1%, fix question isolation with shared-prefix independent suffix branches. If a small pointer decodes the correct answer from option/evidence states but the final token does not, the readout is lossy.

Only then test a **zero-initialized evidence residual**: one query/option-to-token cross-attention layer whose score is added to the unchanged final-letter logits. It must gain at least 3 internal workflow points, add no more than 10% latency, and preserve permutation, Hebrew, and image controls. Otherwise delete it.

Changing the pretrained backbone attention is rejected at this stage. Cygnet, Intern, Strands, Decision 2.0, and Clef all retain pretrained causal sequence mixing; their differences are readout, scale, data, and post-training.

### H4 — Bounded deliberation helps only if measurable headroom exists

Before training latent thoughts or using RL, measure a slow oracle on the independent hard set:

- direct one-token decision;
- explicit reasoning followed by a constrained decision;
- best-of-8 verified reasoning.

No deliberation experiment runs unless best-of-8 improves at least 5 absolute points on temporal/numeric and executable workflows. Pause tokens are not the first choice: their strongest published result trained the pause interface during both pretraining and fine-tuning, not a small LoRA continuation. See [Think Before You Speak](https://arxiv.org/abs/2310.02226).

If headroom exists, train a slow multi-step teacher inside an executable workflow simulator. RL/GRPO is used only where there is exploration, delayed outcome reward, and credit assignment. Its verified trajectories and action distributions are then distilled into the one-pass student. Online RL on the final one-letter contextual bandit is rejected: supervised proper scoring already provides the exact, lower-variance gradient over every action. [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) and [DPO](https://arxiv.org/abs/2305.18290) address different feedback structures.

### H5 — Multimodal skill requires visual necessity, not image presence

After a text checkpoint is frozen, use the actual Gemma processor and pixel tensors. Start with the vision tower frozen and the text adapter protected by frozen-teacher replay; unfreeze/project LoRA only if the frozen visual representation is not linearly usable.

Each image-dependent example needs:

- real image;
- shuffled image;
- gray/no-image control;
- mismatched image/state;
- evidence-removed or counterfactual image;
- a label that cannot be recovered from a caption shortcut.

The key gate is the **real-minus-control causal gap**, not ordinary image accuracy. Recent multimodal RL evidence shows blind training can recover much of a benchmark gain and that prolonged image training can reduce grounding; visually necessary counterfactual tasks remove that shortcut. See [Same Reward, Different Skills](https://arxiv.org/abs/2610.01908).

Promote only if real-image accuracy improves, gray/shuffled performance falls toward the declared control baseline, mismatch abstention rises, and all text gates remain within one point.

### H6 — Hebrew is a paired consistency problem, not a bulk-mixture percentage

Use native Hebrew plus English↔Hebrew semantic twins in three forms: Hebrew state/options, Hebrew state with English options, and English twins. Add low-weight JS consistency after mapping options by meaning, while retaining supervised Hebrew labels and frozen English-teacher KL.

Begin with a zero-initialized Hebrew residual adapter on top of the frozen text winner. Merge or conditionally activate it only after it proves:

- Hebrew improvement on a locked native set;
- English loss no greater than one point;
- cross-language option probability JSD improvement;
- no calibration or RTL/schema regression.

### H7 — Kimi Delta Attention is a scaling branch, not a competence patch

[Kimi Linear](https://arxiv.org/abs/2510.26692) establishes KDA in a hybrid pretrained architecture and reports KV-cache/throughput gains at extreme context. It does not establish that cheap attention surgery improves a decision fine-tune. Post-hoc linearization requires hundreds of millions to billions of distillation tokens and can preserve perplexity while damaging multiple-choice behavior and inducing option-position bias.

KDA begins only after quality freezes and only if profiling shows state access dominates cost for large `(context length × question count)`. Compare:

1. native Gemma attention with forked shared-prefix cache;
2. a native pretrained hybrid-KDA base with the same decision objective;
3. conversion only as a research arm with staged hidden/block/output distillation.

The KDA success signature is a lower latency/cache slope with context and question count at non-inferior quality. It is not expected to make the model smarter.

### H8 — Update geometry is measurable before inventing an optimizer

If the interface winner still forgets base behavior, compute each LoRA update's singular spectrum and overlap with the frozen weight's dominant subspace, then correlate those measurements with held-out regressions. Only test projected or gated updates if this no-training diagnostic predicts failures (target AUC at least `.65`) and simple base-output KL/localization does not already solve them.

[OPLoRA](https://arxiv.org/abs/2510.13003) projects LoRA updates away from dominant base-weight singular subspaces, while recent spectral work argues that a small number of update spikes can govern forgetting. These are conditional retention methods, not evidence that a new optimizer makes decisions smarter. Start with a matched localized-LoRA + non-target KL control; test a spectral projection or spike budget only as the next isolated factor. Changing Gemma's SwiGLU activation is rejected unless causal activation patching first shows a localized, decodable bottleneck: post-hoc activation replacement primarily targets sparsity/throughput and perturbs the pretrained function broadly.

## Ranked frontier backlog

| Rank | Method | Run trigger | Cheap falsifier / stop rule |
|---:|---|---|---|
| 0 | Exact scorer, typed canonicalization, permutation sentinels | none | stop when train/serve parity is exact and semantic order consistency is stable |
| 1 | Matched interface-coverage SFT | missing declarative/generic cell confirmed | require target-cell gain with polarity/order balance and ≤2-point retention loss |
| 2 | Localized LoRA + non-target base KL | interface winner still forgets | kill if it is not Pareto-better than broad LoRA and adapter-scale interpolation |
| 3 | Swap-cancelled counterfactual margin | CE improves consistency but not semantic flips | kill if order delta falls without at least two points of semantic gain |
| 4 | Evidence-privileged equivariant distillation | verified evidence-only view beats full context by ≥5 points | kill if teacher is not gold-correct/confident or 10% pilot gains <2 points |
| 5 | Shared candidate-wise scorer | hidden-state probe beats final-letter readout by ≥5 points | abandon if frozen probe has no headroom or K-way serving cost is unacceptable |
| 6 | Spectral projection/spike-budget LoRA | update geometry predicts regressions and simple KL fails | require diagnostic AUC ≥.65 and a matched-task retention improvement |
| 7 | Evidence-position curriculum | moving identical evidence changes accuracy materially | kill if errors are invariant to distance/distractor count |
| 8 | Verified slow-path distillation, then RL | best-of-8 + checker has ≥5 points of headroom | rejection-SFT first; RL must beat matched SFT and random-reward controls |
| 9 | Hebrew/MM protected residuals | text winner frozen | require native Hebrew transfer and image causal-control gaps, not translated/text shortcuts |
| 10 | Native KDA branch | quality frozen and profiling proves KV/attention cost dominates | speed only; require quality within one point and no option-position collapse |

## Kill / run / defer

| Run now | Conditional | Defer/kill |
|---|---|---|
| Canonical Noul inference control; matched interface-coverage data screen; independent permutation/counterfactual panel. | Base-output KL after the interface factor; swap-cancelled margin; pointer probe; zero-init evidence residual; downstream action-regret loss; MM and Hebrew residuals after text freeze. | More generic Intern steps; shared-head MM/HE continuation; pure DPO/GRPO on one-letter answers; cross-architecture attention-map imitation; KDA surgery before parity; checkpoint selection on public JevBench. |

## Ablation plan

| Order | Ablation | What it tests | Expected signature if it matters | Cost |
|---:|---|---|---|---:|
| 1 | HF-train vs vLLM-serve letter scorer | Whether we optimize probability mass that serving discards | Rotating the answer A–F changes scorer agreement or exposes token aliases | no GPU |
| 2 | Per-kind temperature / Noul affine calibration | Whether probabilities, rather than semantic ranking, drive workflow errors | Exact actions rise with nearly unchanged argmax accuracy; gains cluster near policy thresholds | cached logits |
| 3 | Full state vs verified sufficient evidence vs irrelevant evidence | Retrieval bottleneck versus reasoning bottleneck | Sufficient evidence helps and irrelevant/evidence-position perturbations explain failures | inference only |
| 4 | Adapter scale `0,.25,.5,.75,1` | Whether S0's useful delta can be separated from forgetting | An interior scale retains the base while preserving workflow gain | inference only |
| 5 | Exact-uppercase training scorer vs S0 | Whether optimizing serving-forbidden aliases caused the tradeoff | Legal-token NLL and retained action behavior recover at the same 2,000 presentations | one screen |
| 6 | Generic fallback vs semantic `False/True` canonicalization | Whether the failure is a renderer-level symbol-binding ambiguity | Order-robust recovery on an independent crossed panel with no task-specific text | inference only |
| 7 | Interrogative/explicit control vs bare-declarative/generic coverage | Whether supplying the missing interface contract is causal | Target cell +15 points or ≥95%, order-induced `P(true)` delta ≤.05, retention within 2 points | two matched screens |
| 8 | Base-logit KL vs interface winner | Whether output retention adds more than a uniformly weaker adapter | Better retention than scale interpolation and the data-only winner | one screen |
| 9 | Broad vs localized LoRA, both with corrected scorer | Whether always-on full-depth adaptation causes selective forgetting | Better retention at equal processed tokens without erasing learned gains | one screen |
| 10 | Final-token vs pointer vs pointer+evidence residual | Whether information exists in the backbone but is lost at readout | Residual helps long/dispersed-evidence cases beyond a parameter-matched MLP | conditional |
| 11 | Rejection-sampling SFT vs GRPO | Whether exploration adds more than verified imitation | GRPO beats matched SFT by 3 points at fixed rollout/inference budget | conditional |

Rows 1–5 are complete. Rows 6–7 are the next causal block; rows 8–11 run only if the preceding diagnostics establish their prerequisite. KDA is excluded from this ablation set until profiling shows attention is a material serving bottleneck.

## Promotion gates

A frozen final candidate must satisfy all of the following:

1. Official JevBench score `≥72.1` and Intelligence `≥72.0`; target Winnow Intelligence `74.4`.
2. Public diagnostic is reported with paired uncertainty and family breakdown, but has no pass/fail threshold and never selects a checkpoint.
3. WorkflowEvals equal-weight exact-action mean `≥67.8%`, no workflow more than two points below Jev.
4. Grouped typed-decisions accuracy `≥.727`, with non-inferior NLL/Brier/ECE.
5. Option permutation delta `≤1 point` and grouped/single disagreement `≤1%`.
6. Image causal controls and Hebrew paired gates above.
7. No invalid vectors; final p50/p95 and cost within the declared serving budget.

Public scores are diagnostic receipts, not evidence that a hypothesis caused generalization. Every promotion claim must include the internal gate, paired deltas, and the negative controls that could have falsified it.
