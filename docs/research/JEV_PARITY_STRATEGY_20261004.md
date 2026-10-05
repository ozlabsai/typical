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
| [JevBench v1.5.4](https://benchmarkheaven.com/api/jevbench/v1.5.4) | Jev: official score 72.13293, Intelligence 71.99975. Cygnet: 73.70127 / 71.09242. Winnow Q8: 73.23310 / 74.43469. Winnow is the open system that clears both Jev's composite and intelligence. | Require both official score and Intelligence parity; do not win only through speed/cost. |
| [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) | Frozen Gemma-4-12B-it, one constrained letter, temperature 3.4; `203/231` public and official rank #1. | The fastest evidence-backed baseline is stock 12B + correct prompt/readout, before training. |
| [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) | Gemma-4-12B-it, rank-32 LoRA across attention and MLP, about 26k curated/refined decisions, teacher loss only on teacher/gold agreement, replay; official Intelligence 74.4. | This is the primary text-quality recipe to reproduce or build from. |
| [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) | Qwen3.5-4B, native multimodality, final LM logits at JSON decision markers; self-reported `201/231`, typed decisions 80.55%, 44 ms on 4090. | Best fast/MM candidate; evaluate untouched before modifying our older layer-23 head. |
| [Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/) | Pointer scoring beat an earlier slot head; rank-16 language LoRA. | If a custom head survives, compare answer-position/option-position pointer scoring, not more slot capacity. |
| [JevK5](https://huggingface.co/alibiserikbay/JevK5) and [Plumb](https://github.com/crh225/plumb) | Long teacher documents, agreement filtering, hard mining, and broad replay improve public hard accuracy; current sealed Noul remains weak. | Long hard data matters, but public-set gains are not evidence of sealed generalization. |
| [Decision 2.0 Vega](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) | 27B final decision model; JevArena 74.0 and multi-question one-pass inference. | Useful teacher/upper bound; too large to be the first serving target. |
| [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) | Frozen 27B + rank-16 LoRA + fp32 head, KL to soft distributions, Score RPS, option permutations, per-kind temperatures; reported six-benchmark mean 84.07 vs Jev 83.85. | Copy the loss and batching details, while labeling this as Jev distillation rather than independent intelligence. |
| [CLM](https://github.com/Contrastive-LM/CLM) | Staged hard negatives, contrastive learning, replay, and cached actions. | Add hard-negative contrastive refinement only after the plain decision objective passes. |
| [Bandits](https://github.com/bandr-ai/bandits) | Outcome-grounded agent trace labels, lineage-preserving splits, quarantine of ambiguous cases. | Use post-action evidence for workflow data; do not force uncertain trace labels. |
| [Official WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals) | 705 public cases / 1,524 policy decisions. Jev scores 61.7 security, 71.6 agent trace, 61.8 invoice, 76.0 customer, mean 67.8. Security/invoice/customer use exact action sets; agent trace publishes primary-action agreement. | This is now the primary workflow benchmark, not our old row-wise proxy. |
| [LocalLLaMA typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Full five-question cases; Jev `.727` accuracy. Request grouping affects results. | Our first-1,500 row-wise `.515` number is not leaderboard-comparable. Re-evaluate grouped cases. |
| [Tasksource typed decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions), [Open-Jev v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1), [Sarge v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Broad grouped supervision, programmatic counterfactuals, and full teacher distributions. | These are the training pool after licensing, overlap, quality, and placeholder audits. |

## Success gates

The goal is not complete until a frozen submitted checkpoint meets every primary gate.

1. **Official JevBench v1.5.4:** full 1,624-item evaluator run with zero missing/invalid outputs, score `≥72.13293` and Intelligence `≥71.99975`; additionally target Winnow's `74.43469` Intelligence. Report open/sealed and Choice/Noul/Score separately. A defensible win needs the official paired-bootstrap lower bound above zero; equality needs a preregistered ±1-point equivalence interval on both score and Intelligence.
2. **Public JevBench diagnostic:** the legacy 231-item v1.2 subset and `scripts/jevbench_score.py` are diagnostics only, not v1.5 parity evidence. Report family breakdown, paired deltas, and uncertainty, with no promotion threshold or checkpoint selection.
3. **Official WorkflowEvals:** all 705 cases / 1,524 policy decisions with `questions=all`, full status/coverage accounting, equal-workflow published-metric mean `≥67.8`, and no workflow below Jev's published row. Use exact action sets for security/invoice/customer and primary-action agreement for agent trace.
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

G0 is complete for the three public leaders. These are pinned, untouched reference runtimes; the WorkflowEvals values are equal-weight exact-action/primary-action scores on all 705 current cases.

| Model | Public JevBench | Security | Agent trace | Invoice | Customer | Workflow mean |
|---|---:|---:|---:|---:|---:|---:|
| Cygnet | 203/231 | 33.3% | 52.3% | 53.1% | 69.8% | 52.1% |
| Winnow-12B Q8 | 199/231 | 39.2% | 62.6% | 47.8% | 73.5% | **55.8%** |
| Intern-Decision-4B | 200/231 | **52.1%** | 46.8% | 44.7% | 55.9% | 49.9% |
| Typical S0, fresh Gemma LoRA | 203/231 | 30.0% | 53.6% | 29.8% | 71.6% | 46.2% |
| Jev target | 200/231 public reference | 61.7% | 71.6% | 61.8% | 76.0% | **67.8%** |

This changes the optimization target: public JevBench is not predictive enough to be a gate. Typical S0 exactly matched Cygnet's public total while losing 5.9 workflow-mean points and 23.8 invoice primary-action points. Winnow is the strongest workflow baseline and remains twelve mean points below Jev. The two Intern pilots were useful negative controls, but the final lineage now starts from a fresh general multimodal base rather than a pretrained decision model.

### G1 — flagship text decision model

Only if an untouched baseline does not already clear the full target:

- Base: a fresh pinned Gemma-4-12B-it. Winnow is a reference/teacher only; Typical does not initialize from another decision model.
- Readout: constrained final-token candidate symbols first; the current Typical head and a Strands-style pointer are ablations.
- Adaptation: rank-32 LoRA on `q/k/v/o` and MLP projections; native vision path stays frozen initially.
- Context: train/evaluate length buckets through at least 16k; official agent-trace inputs reach 44k characters.
- Data: 30k–100k quality-filtered grouped cases, long policy/doc examples, option permutations, counterfactual fact changes, hard negatives, and 30–40% broad replay.
- Loss: hard-label full-vocabulary CE; `KL(target || model)` for valid soft distributions; `0.5 × RPS` on Score; teacher loss only when teacher argmax agrees with verified gold.
- Calibration: disjoint per-kind temperatures. No public benchmark labels.

### G2 — fast 4B lane

Start from Intern-Decision-4B or reproduce its final-marker objective on Qwen3.5-4B-it. Keep the vision tower/projector frozen and adapt the language model with rank-16/32 LoRA. This lane must pass sealed generalization before it can replace the 12B flagship; current 4B leaderboard entries often gain composite rank from speed/cost while losing badly on sealed Noul.

The first G1/G2 pilot is now locked:

- Released `internlm/Intern-Decision-4B`; rank-16 language LoRA, 32.46M trainable parameters (0.71%).
- Exact Intern JSON-marker objective, 4,096 tokens, effective batch 16, `5e-5` cosine LR, 250-step first gate.
- 40,000 grouped requests: 28,000 hard workflow/rule requests and 12,000 broad E/K replay requests.
- Final corpus SHA-256: `1ffd2262b14845e569002ae50fc7e0fb3333ade2b437dc5110fda94dfc796c44`.
- All requests fit without truncation: max 3,482 tokens, p99 3,044.
- Zero exact, normalized-stem, or word-trigram MinHash ≥0.8 overlap against 1,336 locked inputs: 231 JevBench, 705 WorkflowEvals, and 400 typed-decisions test cases.
- One real forward/backward smoke step passed at 10.05 GiB peak VRAM. The full gate uses separate workflow, hard-rule, and replay validation floors.

The referenced public corpora stay out of this first pilot. Tasksource has mixed row-level licensing; Open-Jev includes workflow-control sources close to the target domains; Sarge includes uniform placeholders and Jev-teacher distributions that the released hard-label objective cannot represent. They remain a filtered G2 curriculum only if the clean internal pilot misses its locked gate.

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

The pinned Gemma-4-12B S0 and exact-uppercase S1 controls are complete. S1 did not recover the held-out tradeoff, and serving-forbidden aliases contained less than `.005%` of the relevant probability mass; scorer mismatch is not the main cause. Failure localization instead found an unsupported Noul interface: bare declaratives with generic `No/Yes` labels interact sharply with option position, despite random option shuffling elsewhere in training. R042 therefore ran a task-agnostic semantic `False/True` renderer control and a matched data-only interface-coverage screen. Base-output KL and localized LoRA remain separate later factors. Hebrew and multimodal training begins only after text/workflow promotion and must exercise the real pixel path.

### R042 — representation control

R042 compared matched programmatic supervision rendered as record cards or as natural prose on the fresh Gemma lineage. The natural-prose checkpoint repaired the practical bare `No/Yes` defect: bare-negative accuracy was `128/130` and both-order correctness was `64/65`, while bare-positive and both canonical controls were `130/130`; mean bare order delta was `.0100`. The record-card checkpoint reached only `99/130` bare-negative accuracy and `48/65` both-order correctness, with mean order delta `.0271`. Cards are therefore rejected as the representation treatment; their near-perfect in-distribution card score did not transfer to the natural renderer.

The prose checkpoint also retained the three locked text slices: replay/WF/WH accuracy was `.7656/.8594/.7500`, versus `.7656/.8125/.6875` for the base. On the held-out executable action panel it scored `255/256` exact actions both raw and calibrated, versus base `228/256` raw and `224/256` calibrated; the card checkpoint scored `252/256` raw and `250/256` calibrated. These measurements make prose the text/action candidate, not a globally promoted checkpoint.

The causal Hebrew/multimodal battery blocked global promotion. Prose preserved Hebrew accuracy (`.9219`) while lowering Hebrew NLL (`.5591 → .2576`) and improved English accuracy (`.9219 → .9375`), but real-image accuracy fell from `.6641` to `.6328` (four additional errors) and gold-dropped null recall fell from `.6563` to `.6250` (four additional errors). Its mismatch null recall lost one example (`.7734 → .7656`) and still passed that gate. Cards also failed the battery: Hebrew lost three examples and both mismatch and gold-dropped null recall lost three examples. The full R042 promotion gate is therefore false.

R043 is the next experiment: a preregistered causal-pair objective pilot that separates ordinary CE, true-pair flip loss, and exact-stratum shuffled-pair controls. If its text/action candidate advances, multimodal replay through the real pixel path is required late in the curriculum before any global promotion; Hebrew and multimodal gates remain unchanged.

### R043 — pair-loss pilot stopped at the activity gate

R043 did not justify a pair-loss claim. Its treated arm completed, but the flip hinge was active on essentially no treated examples, so the proposed objective was normally a no-op. The matched controls were stopped rather than spending compute on an inactive factor. The completed arm also failed the locked renderer screen (`53/65` bare-negative both-order pairs versus `64/65` for the R042 parent) and regressed the three retention slices. R043 is therefore neither a promoted model nor evidence against pair-aware objectives in general; it is evidence that this particular hinge and sampling regime did not create useful training signal.

### R044 — causal multimodal replay promotion

R044 resumed from the R042 prose checkpoint and compared two matched one-epoch continuations over 800 logical rows. Both arms saw the same 100 causal pairs, 200 prose examples, 200 broad replay examples, 100 balanced Hebrew intents, and 200 visual-task rows. Arm A stripped every image; arm B consumed all 200 images through Gemma's real pixel path. The language LoRA was trainable while the vision tower and projector remained frozen.

Arm B passed every preregistered gate and is the first globally promoted checkpoint in this lineage. Against the R042 parent, real-image accuracy rose from `81/128` to `103/128`, gold-dropped null recall from `80/128` to `96/128`, Hebrew from `118/128` to `119/128`, and English remained `120/128`. Workflow/replay retention remained `108/128`, `103/128`, and `95/128`; the action panel remained `255/256`; the renderer stayed `64/65` on the only imperfect cell and `65/65` elsewhere. Exact pair actions improved from `242/256` to `246/256` and both-twin correctness from `228/256` to `236/256`.

The text-only arm did not explain the visual improvement: B beat A by 20 real-image and seven gold-dropped decisions, while A failed the absolute multimodal promotion gate. This supports a causal image-replay claim for the locked battery. It does not yet establish Image JevBench performance, broad Hebrew reasoning, or official workflow/JevBench parity.

### R044 public JevBench diagnostic

The frozen promoted adapter completed all 231 legacy public JevBench items with 231 strict-valid native probability vectors and `196/231` correct (`84.85%`), ECE `.0632`, p50 `.119 s`, and p95 `.674 s` on an A100. On the same item IDs, Jev 1.13 scored `200/231`. This is a net four-answer difference: Jev alone was correct on ten disagreements and Typical alone on six; exact paired McNemar `p=.4545`. It is not evidence of a reliable difference or official v1.5.4 equality.

The errors localize the next independent-data experiment. Temporal/numeric accounts for 12 of 35 misses, followed by long-policy (six), probability (four), and multi-hop (four). R044's apparent temporal/numeric coverage was shallow: its pair generator reused fixed dates, thresholds, and operators, so the strong internal pair score mainly demonstrates nuisance-record transfer rather than new calculation or rule composition. R045 therefore tests held-out values, thresholds, program compositions, entity vocabularies, and renderer grammars with executable gold. JevBench and WorkflowEvals remain excluded from data generation, calibration, checkpoint selection, and model selection.

### R044 official WorkflowEvals result

R044-B completed the pinned official workflow protocol with all 705 cases and 1,524 decisions, zero request errors, and complete consensus scoring. Its published-metric results were security exact actions `33.75%`, agent-trace primary action `54.95%`, invoice exact actions `51.78%`, and customer-service exact actions `70.75%`; the equal-workflow mean was `52.81%`, versus Jev's `67.8%`. All four preregistered workflow floors failed. The copied manifests, results, scores, and run log under `.context/r044-workflows` match the remote SHA-256 receipts.

This confirms that public JevBench proximity (`196/231` versus Jev's `200/231`) does not imply workflow parity. The gap is largest where an exact action set depends on several predicates distributed across the state. R044's internal action panel had short states and direct fields, while its training mixture contained almost entirely singleton questions. The next treatment therefore targets evidence binding and rule composition rather than attention replacement or RL.

### R045 compositional transfer preregistration

The first R045 generator was rejected before model exposure. It omitted model-visible operators, contained label shortcuts, coupled renderer to the decisive clause, used disjoint numeric ranges across splits, and did not make every clause causal. The repaired v3 corpus states the full rule in English and Hebrew, independently verifies English labels from rendered text, crosses option order and nuisance fields, uses real decimal/ledger/conditional/timezone/precedence calculations, and groups every clause pivot into one scenario. Benchmark source paths/content are rejected.

Frozen v3 hashes:

- builder: `079609abefc1367fdde2a6d7193e2061a29f0368e15ea81fe5d54d1b01741a04`
- train, 192 rows / 96 pivot sources: `47e5939607097854b5923d45080f4c7cbc766699455ccdcb04b9dd940e26e331`
- calibration, 48 rows / 24 pivot sources: `1690edcc41ca6a7f24b7499edf2f0f1446b47ec441f19edc0dc570205cfa14e7`
- test, 144 rows / 72 pivot sources / 32 complete scenarios: `d9a1f3d2e5eeab456faba5bcf87eb06b297912185f1c2ef2f648d98ca2392acd`

The primary metric is complete-scenario accuracy: every pivot, both languages, and both counterfactual outcomes must be correct. A continuation must gain at least four of 32 scenarios and 18 of 288 presentations over both its matched control and frozen R044, with a one-sided exact paired scenario test at `p ≤ .05`. It may not regress complete pivot sources, correct pairs, either language, or any composition by more than one scenario. R045 test NLL and Brier must remain within 105% of R044. If frozen R044 already reaches 28/32 scenarios or 271/288 presentations, this panel lacks enough headroom and cannot justify training by itself.

The training arms each contain 789 optimizer slots and 889 presentations. They share 389 retention rows; control A uses 400 clean R044 rows, while treatment B uses 300 repaired WH singletons and 100 complete-scenario R045 pair rows. They are deliberately labeled an equal-update curriculum comparison, not equal-token compute: A contains 294,589 state characters and B 548,755. A win must therefore be described as evidence for the whole curriculum package, not an isolated composition-loss or compute-efficiency claim. Existing R044 multimodal, Hebrew, English, action, renderer, pair, and retention floors remain mandatory.
