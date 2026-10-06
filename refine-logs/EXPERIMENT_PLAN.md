# Experiment Plan — Jev parity

**Problem:** Typical's 4B layer-23 shared head is below Jev on capability and regressed when visual/Hebrew examples rewrote that same frozen-backbone decision boundary.

**Method thesis:** First evaluate stronger untouched instruction checkpoints with final-token/pointer readouts. If training is needed, use grouped long-context decision data, soft-distribution retention, per-kind calibration, and explicit non-regression gates. Add Hebrew and multimodality only after text/workflow competence is locked.

**Date:** 2026-10-04

The prior multimodal/Hebrew plan and completed E1 history remain in `EXPERIMENT_PLAN_20260929_190948.md` and `docs/research/MM_HE_E1_20261004.md`. The evidence synthesis is `docs/research/JEV_PARITY_STRATEGY_20261004.md`.

## Claim map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1: Typical equals or beats Jev's decision capability | Official JevBench score ≥72.1 **and** Intelligence ≥72.0; public diagnostic reported but not used as a gate | B0, B2, B5 |
| C2: Typical equals or beats Jev on real workflows | Official WorkflowEvals exact-actions mean ≥67.8 and each workflow within two points of Jev | B0, B1, B3, B5 |
| C3: The model is genuinely multimodal | Pinned image benchmark plus real/shuffled/no-image and mismatch-null controls | B0, B4, B5 |
| C4: Hebrew improves without erasing English | Paired Hebrew gain; English loss ≤1 point; calibration and pair JSD reported | B0, B4, B5 |
| C5: The result is fast and calibrated | Valid distributions; Brier/ECE no worse than Jev on comparable sets; latency/cost within 2× Jev | B0, B5, B6 |
| Anti-claim: public benchmark tuning or teacher imitation alone explains the result | Locked test quarantine, overlap audit, grouped splits, independent and distilled result tables | B1, B2, B3, B5 |

## Fixed targets

| Evaluation | Jev target | Promotion gate |
|---|---:|---:|
| JevBench official v1.5.x | score 72.1; Intelligence 72.0 | ≥ both; report open/sealed and type breakdown |
| JevBench public diagnostic | Cygnet 203/231; Winnow 199/231 | report paired/family results; no pass/fail threshold |
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

**Pass:** chosen readout improves grouped, source-balanced held-out accuracy/NLL/Brier, not formatting alone, and supports at least 16k context.

### B3 — text/workflow competence training

- 30k–100k first run, quality before scale.
- Sources after audit: Tasksource selected mix, Open-Jev v1.1, Sarge v3 `yuri_v3`; exclude Sarge uniform placeholders and duplicate Open-Jev rows.
- Length buckets include 8k–32k states; primitive-stratified and source-grouped batches.
- 30% option permutations; counterfactual fact changes; long policy/doc cases; staged hard negatives; 30–40% broad replay.
- Hard gold: full-vocabulary symbol CE. Soft distributions: `KL(target || model)`. Score: add `0.5 ×` ranked-probability score. Teacher signal applies only when teacher argmax agrees with verified gold.
- LoRA: flagship rank 32 over attention and MLP; fast lane rank 16/32; freeze vision tower/projector initially.
- Checkpoints advance on per-suite floors, never a mean validation loss alone.
- Before another training run, audit training/serving candidate-token parity, workflow threshold margins, evidence position, and adapter-scale interpolation using frozen logits/checkpoints. S1 repaired the six-alias/two-alias mismatch but did not recover aggregate behavior; forbidden aliases held less than `.005%` mass. The dominant invoice regression is instead a bare-declarative + generic `No/Yes` + option-position interaction. Test that interface factor before retention methods.

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
| G0-CYGNET | Pinned Cygnet identity + 705-case WorkflowEvals | identity 203/231 ±1; zero request failures | DONE: 203/231; workflow mean 52.1 |
| G0-WINNOW | Same matrix for Winnow | compare workflow against Cygnet | DONE: 199/231; workflow mean 55.8 |
| G0-INTERN4 | Same matrix for Intern-Decision-4B | establish fast/MM candidate | DONE: 200/231; workflow mean 49.9 |
| G0-TL2 | Evaluate existing 9B checkpoint before training | use correct context/grouping | QUEUED |
| G1-READOUT | Final-token vs pointer vs native | development-only winner | BLOCKED on G0 |
| G2-TEXT | First competence LoRA | ≥200/231 diagnostic and no workflow-dev regression | FAILED: clean pilot 197/231; official workflow transfer below target |
| G2-SOFT | Filtered public soft-distillation LoRA | locked internal retention before any diagnostic | FAILED: 199/231; no transferable workflow gain |
| G2-S0 | Fresh Gemma broad-LoRA sensitivity control | measure own-lineage adaptation | FAILED: public 203/231; workflow mean 46.2; invoice exact/primary 29.8/57.3 |
| G2-S1 | Exact-serving scorer control | isolate six-alias/two-alias mismatch | FAILED: corrected support did not recover aggregate held-out behavior |
| G2-INTERFACE | Semantic fallback inference control, then matched format-coverage arms | isolate Noul symbol binding from new content | RUNNING: A/B localized convention interference; exact interpolation failed; audited mixed Arm C live |
| G3-RETENTION | Base-output KL, then localized LoRA as separate factors | improve retention beyond adapter-scale tradeoff | BLOCKED on G2-INTERFACE |
| G3-HARD | Long/hard-negative refinement | improves locked internal hard families | BLOCKED on corrected retention control |
| G4-MMHE | Residual Hebrew/MM continuation | all retention gates | BLOCKED on text winner |
| G5-OFFICIAL | Frozen official submission/evals | all primary gates | BLOCKED on candidate |
| G6-SERVE | Q8/prefix optimization | prediction equivalence + latency/cost | BLOCKED on G5 quality |

## Compute budget

- G0 completed sequentially on one RTX PRO 6000; artifacts persisted under `.context/g0-{cygnet,winnow,intern}` before stopping.
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
- [x] G0 baseline matrix complete for Cygnet, Winnow, and Intern
- [ ] Candidate clears locked development gates
- [ ] Official JevBench score and Intelligence equal/beat Jev
- [ ] Official WorkflowEvals mean equal/beats 67.8
- [ ] Multimodal and Hebrew release gates pass
- [ ] Serving latency/cost and calibration reported

## R047 — factorized executor curriculum (frozen 2026-10-06)

### Claim correction

R046 does not isolate a special conjunction or attention failure. Weight `1.0` still fitted only
`125/200` seen presentations; on the clean panel its strongest families were policy-bearing while
decimal and ledger counterfactual direction remained weak. Moreover, `0/32` complete scenarios is
compatible with ordinary row accuracy around `.615`: 24 scenarios require eight simultaneous
correct presentations and eight require twelve. Complete-scenario accuracy remains a promotion
metric, but the mechanistic claim is narrower:

> Executable supervision for each typed primitive and its evidence binding improves unseen
> primitive execution and final decisions more than spending the same supervised-answer budget on
> repeated final decisions.

The anti-claim is that any gain comes from rereading states, extra answer positions, longer
training, option order, language balance, or more optimizer weight per source program.

This is deterministic process supervision, not free-form chain-of-thought. It is directionally
supported by process-supervision and scratchpad results, but those papers do not establish the
effect in this classifier setting; R047 is the causal test. See Lightman et al.
(`https://arxiv.org/abs/2305.20050`), Uesato et al.
(`https://arxiv.org/abs/2211.14275`), and Nye et al.
(`https://arxiv.org/abs/2112.00114`).

### Frozen representation

- Start every candidate from exact R044-R, never from a rejected R045/R046 candidate.
- Preserve the canonical final question and constrained single-letter serving interface byte for
  byte. Training-only auxiliary views never appear at inference.
- Keep each program as one optimizer slot. Average all views inside the slot so multi-clause
  programs receive the same optimizer weight as control programs.
- For the 48 frozen K=2 training pair rows, keep the two original counterfactual final views and
  attach four training-only views: clause 1 and clause 2 on each state.
- Treatment targets are computed directly by `_evaluate(clause)`. Control has the same states,
  view count, option order, and slot order but its four extra views ask paraphrased whole-case audit
  questions and repeat `execute(program, clauses)`.
- Pair-flip operates only on the two canonical final views. Both arms use weight `1.0` so R047 does
  not silently change the R046 setting.
- The auxiliary implementation must be generic over K even though frozen training has 52 K=1 and
  48 K=2 rows. K=3 remains held-out transfer.
- Tokenizer preflight must show equal view counts and aggregate non-padding tokens within `0.5%`;
  report wall time and peak VRAM. Config differences are limited to arm path/mode and output path.
- Full-clause treatment targets are 75% yes while repeated-final control targets are 50% yes. This
  is part of the semantic intervention, must be disclosed, and requires balanced-accuracy and
  per-class reporting. A decisive-only target is rejected because it duplicates the final label and
  adds no intermediate information.

### M0 — can-fit sanity before causal training

Run the 48 K=2 treatment slots without replay from R044-R, rank-32 LoRA, seed 47, LR `2e-5`,
cosine floor `2e-6`, batch `1×8`, and checkpoints at 20/50/100/200 steps. This is a diagnostic
overfit run, not a candidate.

**Go gate:** by or before step 200, executable auxiliary balanced accuracy and final counterfactual
direction are each at least `98%` overall and for decimal, ledger, policy, and conditional. Final
balanced accuracy and pair-both correctness must also each reach `98%`; direction alone is
insufficient because both states can be ranked correctly while receiving the same wrong class.
Metrics must be finite, every clause target must independently reproduce from the AST, and logs must
account separately for canonical/auxiliary views and CE. Select the earliest passing checkpoint only
from steps 20/50/100/200. If it fails, do not run the matched arms. Inspect objective/readout
gradients and separately preregister any capacity change: the current trainer's `--rank` is ignored
when `--init-adapter` loads R044-R, so a valid higher-rank test requires an explicit merge-and-attach
or second-adapter procedure with a new parent hash. No holdout result may choose that retry.

**Result (2026-10-06): PASS at step 200.** Steps 20, 50, and 100 failed the frozen gate; step 200
reached `1.0` final balanced accuracy (`96/96`), `1.0` clause balanced accuracy (`192/192`), `1.0`
clause-vector accuracy, `1.0` final direction (`48/48`), `1.0` pair-both correctness, and `1.0`
coherence. Each of decimal, ledger, policy, and conditional reached `1.0` clause balanced accuracy
and `1.0` direction; the minimum pair-direction delta was `7.625`. The fail-closed receipt selected
only step 200 and all integrity checks passed. This establishes representation/capacity on the
in-sample panel only; it does not establish prospective transfer or benchmark parity. An initial
post-training launch failed because a scorer dependency was absent from the remote bundle; the
dependency was restored and only evaluation/finalization reran against unchanged checkpoint hashes.

### M1 — matched causal screen

Use the frozen 800-slot R045-B mixture in both arms. The 700 non-compositional slots and all 100
canonical counterfactual pair rows are byte-identical; only the 48 K=2 rows receive matched
training-only bundles. Train four deterministic epochs (`400` steps × accumulation `8`) from the
same R044-R parent with the M0-passing capacity. Control runs first, then treatment. Seed, order,
optimizer, scheduler, base revision, images, and every retention row are identical.

Before candidate exposure, freeze two panels. The existing seed-47/domain-offset-2 panel is retained
only as a 32-scenario **near-transfer smoke diagnostic**: despite 144 language rows, it has only 24
K=2 and 8 K=3 independent clusters, reuses the generator/template family, and confounds language
with answer order. It may detect obvious failure but cannot establish the mechanism gate.

The primary panel is exact template/value-regime disjoint and uses semantic clusters as its
independent unit. It deliberately shares the six executable AST semantics and oracle family with
R045; it does not claim full generator-family or fuzzy near-duplicate disjointness:

- 312 scenarios spanning all 13 unseen K=2 compositions (24 per composition);
- 320 scenarios spanning all 20 K=3 compositions (16 per composition);
- 48 scenarios on the two seen K=2 compositions, reported only as a near-transfer diagnostic;
- the full Boolean truth cube per scenario, crossing EN/HE with both answer orders and at least three
  held-out clause-query families;
- new entity, renderer, numeric-margin, ledger-order/length, date/timezone, fraction, conditional,
  and policy regimes, with independently executed gold and explicit generator-family provenance.

Freeze builder, evaluator, comparison, bootstrap RNG, arm, panel, manifest, and threshold hashes.
Reject exact or normalized overlap in states, questions, instructions, clause JSON, numeric
signatures, decisive pairs, complete AST/multisets, composition hyperedges, scenario/template/
generator lineage, and all quarantined benchmarks. The preflight proves these exact normalized
overlap classes only; it does not prove fuzzy near-duplicate absence. Language/order twins and the
complete truth cube remain in one semantic cluster. Resample whole clusters within K×composition
for at least 20,000 fixed paired-bootstrap replicates; never row-bootstrap or pool K=2 with K=3.

**Mechanism gate:** treatment minus control must pass separately on both unseen K=2 and K=3
co-primary strata (an intersection-union decision), satisfying all of:

- clause balanced accuracy `+5` points with scenario-clustered 95% bootstrap lower bound above 0;
- exact clause-vector accuracy `+8` points;
- final/clauses coherence (`final == AND(predicted clauses)`) `+10` points;
- counterfactual direction gains for both decimal and ledger, with no primitive or language losing
  more than 2 points;
- final presentation accuracy `+5` points, no regression in correct pairs/sources, and NLL/Brier at
  most 105% of control and the best frozen R044/R046 reference.

Report per primitive×language×K class confusion, truth-Hamming-weight and seen-edge slices, both
answer orders, semantic order agreement, probability delta/JSD, and treatment×order interaction.
Every language/order point effect must be nonnegative, no cell may lose more than two points, and a
gain driven by only one answer order fails. M1 is one paired optimizer-seed causal screen; a second
paired training seed is required in M2 before calling the mechanism replicated.

A final-only gain without clause gains is a shortcut and fails. Complete scenarios are reported but
are not the mechanism gate. Promotion-scale expectations require at least `.81` final presentation
accuracy before treating five complete scenarios as plausible.

The initial intervention reaches only `48/800` logical slots (`6%`). If M1 produces no clause
improvement, that result rejects this dose, not the mechanism family. One preregistered dose-response
screen, M1b, may compare `200/800` executable slots (`25%`) with a view-, token-, order-, and
optimizer-matched repeated-final control. M1b must clear the same `+5` clause-BA and `+8`
clause-vector gates; otherwise this form of process supervision is abandoned. If M1 improves clauses
but not final/coherence, skip M1b and test a matched aggregate-consistency objective instead. If only
decimal/ledger remain weak, test machine-executable arithmetic traces rather than increasing generic
clause dose.

### M2 — scale only a causally validated treatment

If M1 passes, build a source-grouped curriculum from multiple fresh seeds/domain offsets with at
least 1,600 executable program slots and an equal number of replay/retention slots. Preserve EN/HE
twins, full scenario groups, option permutations, real-pixel replay, and family balance. Compare the
same matched control/treatment at two fixed paired seeds. Scale next to 30k–100k grouped cases only
if operator direction reaches at least `90%` for every family and unseen-value/composition transfer
survives both seeds.

If clauses improve but final aggregation does not, R048 tests deterministic aggregate-consistency
or a small training-only aggregator. If arithmetic primitives remain weak after can-fit succeeds,
R048 instead adds machine-generated traces (`operands -> computed value -> predicate -> action`)
against a token-matched answer-only control. No teacher prose is allowed.

Only an M2 candidate that passes every inherited WF/WH/replay, renderer, action, real-image,
mismatch/null, Hebrew, and English gate receives fresh official JevBench v1.5.x and all-705-case
WorkflowEvals evaluation.

### Deferred methods

- **RL/GRPO/DPO:** deferred because every clause and final action already has exact dense gold;
  sampled rewards would add variance and credit-assignment ambiguity. Reconsider RLVR only for
  genuinely sequential verified workflows after supervised execution works. The first eligible
  pilot should enumerate the small typed action space to form a Rao-Blackwellized conditional
  advantage, with CE anchoring and KL to our own pre-RL Typical checkpoint; use sampled RLOO only if
  exact enumeration is impossible. Require at least `+3` exact-action points with a
  scenario-clustered lower confidence bound above zero and no field, Hebrew, English, or multimodal
  regression before scaling.
- **KDA/custom attention:** deferred because the failures occur on short and even seen examples.
  Kimi Linear (`https://arxiv.org/abs/2510.26692`) demonstrates a pretrained hybrid architecture
  and long-context efficiency, not a repair for this adaptation failure. Reopen attention research
  only if short-context primitive accuracy reaches `90%` while a controlled 1k/4k/16k/32k
  evidence-position sweep loses more than 10 points at 32k or retrieval falls below `90%`.
- **Activation/optimizer changes:** deferred until M0 learning curves establish a capacity or
  optimization defect. Answer-order permutations remain a guard, not the primary intervention.
