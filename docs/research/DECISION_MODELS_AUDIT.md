# Decision-model audit for a smarter, faster, multimodal, Hebrew Typical

**Date:** 2026-10-04

**Scope:** the 21 artifacts supplied by the project lead, plus inspection of released model code where available.

**Evidence policy:** architecture and recipe claims come from primary model cards, repositories, and author posts. Scores are treated as self-reported unless the source identifies an independent evaluation. The audit structure follows the evidence-first procedure described by [Scientific Agent Skills](https://arxiv.org/abs/2609.00065).

## Executive decision

Do **not** add a large attention module yet.

Typical already implements the most consistently successful pattern in these releases: a pretrained torso, runtime-defined option spans, a query representation, bilinear/elementwise candidate scoring, and no autoregressive answer generation. Its head is materially closer to [Decision 2.0's released candidate/query head](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B/blob/main/decision2/_vendor/dev2model/decision_model.py) and the [Strands pointer head](https://github.com/strands-labs/strands-decider) than to a fixed classifier.

The highest-confidence path is:

1. Establish a frozen, direct option-letter baseline before training anything.
2. Keep the current Qwen3.5 multimodal prefix and cached native head.
3. Improve training with grouped hard negatives, counterfactuals, outcome-grounded traces, and replay.
4. Make Hebrew a parallel-consistency problem, not merely a small Hebrew bucket.
5. Test one small cross-field-attention ablation only after the simpler head fails on multi-question interactions.

[Recommended Typical path diagram](figures/typical_mm_hebrew_path.md). Green is the retained serving path, blue is the proposed training signal, and orange is the conditional attention experiment.

## What the sources collectively show

### 1. The prompt/readout is a first-class baseline

[Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) uses frozen Gemma-4-12B-it, lettered options, one answer position, and a fitted temperature. Its repository reports first place on the independently evaluated JevBench v1.5.4 board, statistically tied with Winnow. [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) similarly renders a complete answer skeleton and reads candidate-symbol logits at masked positions in one causal pass. [Together's recipe](https://www.together.ai/blog/how-to-train-your-own-jev) shows that ordinary supervised fine-tuning on roughly 38K classification rows can learn the output format cheaply, although it does not establish calibrated probability quality.

**Implication for Typical:** add a locked direct-letter baseline with the same backbone, prompt, context limit, and temperature-fitting split as every trained run. If a trained head does not beat it on held-out capability and calibration, the extra architecture has not earned its complexity.

### 2. Candidate alignment beats fixed slots

[Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/) reports that a fixed slot head underperformed a pointer head that compares each option-position hidden state with an answer-position hidden state. [Decision 2.0 Vega](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) releases a 256-dimensional FP32 head combining a bilinear candidate/query term with a small nonlinear residual. Typical already uses projected query and contextual option-span representations with elementwise interaction, a dot product, and an MLP residual.

**Implication for Typical:** we already have the important mechanism. The cheap ablation is endpoint pooling versus the current mean-span pooling, not a replacement architecture.

### 3. Better negatives and better provenance are the main intelligence lever

[CLM](https://github.com/Contrastive-LM/CLM) trains state/action encoders with bidirectional InfoNCE, then refines on semantically plausible hard negatives and agent traces. Its authors report that staged hard-negative refinement outperformed introducing hard negatives from the start, and that replay preserved broad ability. [Winnow](https://huggingface.co/EldanRing/Winnow-12B) uses contrastive fact changes, replay, and teacher-distribution loss only when the teacher agrees with gold. [Bandits](https://github.com/bandr-ai/bandits) labels agent steps from the following tool result or user reaction, preserves evidence lineage, and quarantines ambiguous examples instead of forcing labels.

**Implication for Typical:** first spend compute on examples that distinguish plausible options. Add hard negatives as a second stage, keep 30–40% broad replay, and treat unresolved trace outcomes as null/unclear or quarantine. This is more likely to improve reasoning than adding head depth.

### 4. Multimodality works when the visual tokens enter the shared prefix

[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni), [Clef](https://huggingface.co/Cloudflare/clef), [Winnow](https://huggingface.co/EldanRing/Winnow-12B), and [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) all retain the base model's multimodal path rather than training a separate visual classifier. Clef processes text, JSON, images, and video in a single record; Jev-Omni exposes image, audio, and video through Gemma's native components. However, most published multimodal results are developer-reported and use different protocols.

**Implication for Typical:** the implemented Qwen3.5 visual-prefix route is the right first experiment. Prove pixel dependence with real/shuffled/no-image counterfactuals before touching the vision tower or adding fusion layers.

### 5. Attention is useful for interacting fields, not automatically for single decisions

[Clef's released joint head](https://huggingface.co/Cloudflare/clef/blob/main/joint_schema_model.py) is the clearest attention proposal: two 1024-wide evidence-routing layers let every option attend to the full backbone sequence, then four transformer-decoder layers allow question fields to interact before scoring. Cloudflare reports training it with label-smoothed CE, Brier loss, rank-256 LoRA, and schema/order permutations. The card does not isolate how much gain comes from the attention head versus the 27B backbone, data, LoRA, or loss.

**Implication for Typical:** cross-field attention is justified only for requests where answers genuinely depend on other fields. Test a single small layer after the existing head is measured on a dedicated cross-field suite. Do not copy Clef's six-layer, width-1024 head into the 4B pilot.

### 6. Serving architecture matters more than exotic training-time attention for speed

[OpenJev-Fast](https://yiqilyu.me/open-jev-fast/) reports a 6× speedup over its optimized PyTorch path using shared-prefix trees, packed question/candidate branches, fused kernels, tuned GEMMs, and CUDA graphs. It is B300-specific, BF16-only, single-request, and uses about 98 GiB, so the kernels are not a near-term Typical dependency. [Winnow](https://huggingface.co/EldanRing/Winnow-12B) demonstrates the practical value of quantization and cached state prefixes; [CLM](https://github.com/Contrastive-LM/CLM) additionally caches reusable candidate embeddings.

**Implication for Typical:** keep prefix caching and branch batching, benchmark warm-query latency, then test Q8. Cache static candidate embeddings only for repeated taxonomies; contextual option features must remain the default for arbitrary options.

## Hebrew plan

The current MASSIVE Hebrew bucket is a good smoke test, but it is insufficient as the final multilingual recipe because the state is Hebrew while the intent candidates remain English identifiers. That can teach cross-lingual intent mapping, but not robust Hebrew criteria, option descriptions, or calibration.

Use three paired views of the same semantic decision:

- Hebrew state + Hebrew option descriptions.
- Hebrew state + English option descriptions, matching current runtime behavior.
- English state + the same semantic option IDs.

Train the gold objective on every view and add a small symmetric probability-consistency loss between matched English/Hebrew rows. Keep semantic candidate IDs fixed while randomizing display language and option order. Group all translations and counterfactuals into the same split to prevent leakage. Audit tokenizer tokens per character and truncation rates separately for Hebrew.

For data, start with parallel MASSIVE identifiers and original, rule-generated Hebrew control tasks. Machine-translated complex tasks should enter only after label-preservation checks; fluent translation is not evidence that the decision boundary survived. Evaluate Hebrew and English paired accuracy, probability divergence, ECE/Brier, null recall, and English retention.

## Data-source decisions

| Artifact | What it contributes | Use in Typical | Main caution |
|---|---|---|---|
| [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) | LoRA + replay + gated teacher distributions; Q8 and prefix-cache evidence; multimodal runtime | Borrow the replay and teacher-agreement gate; use as an external benchmark | Private data; vision functionality is not a clean multimodal-quality ablation |
| [cygnet-recipe](https://github.com/blockbrain-ai/cygnet-recipe) | Frozen instruction model + single-token letter readout + temperature | Mandatory baseline | Backbone is 12B Gemma, so compare the method, not raw score |
| [Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/) | Small pointer head, rank-16 LoRA, documented iterations | Endpoint-pooling ablation and reproducible recipe reference | Mostly public-subset iteration; avoid tuning to JevBench |
| [Decision-2.0 Vega](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) | Bilinear + nonlinear candidate/query head; FP32 head | Confirms Typical's current head family | 27B self-reported comparisons; remote custom code must be reviewed before execution |
| [Bandits](https://github.com/bandr-ai/bandits) | Outcome-grounded trace mining, deterministic checks, quarantine, lineage-safe splits | Adapt its data principles for real agent-step training | Do not import sensitive traces; reaction-free errors have low recall |
| [Clef model](https://huggingface.co/Cloudflare/clef) / [training post](https://blog.cloudflare.com/clef-decision-models/) | Native multimodality, joint schema attention, Brier auxiliary loss, schema permutations | Cross-field attention ablation only after the simple baseline; consider Brier auxiliary | Attention contribution is not independently ablated; internal synthetic data |
| [Together $17 recipe](https://www.together.ai/blog/how-to-train-your-own-jev) | Small, diverse supervised mix and simple normalization | Sanity check that the end-to-end recipe can learn | Generated label output is not a calibrated decision head |
| [CLM repository](https://github.com/Contrastive-LM/CLM) | Contrastive state/action alignment, hard negatives, replay, cached candidates | Add a late hard-negative auxiliary objective; cache repeated labels | Very large private/synthetic recipe; released claims are not yet a common benchmark protocol |
| [CLM Notion site](https://contrastive-lm.notion.site/) | Scaling-law write-up linked by the repository | Treat as supplementary | The homepage was not retrievable during this audit; the repository mirrors the actionable claims |
| [OpenJev-Fast](https://yiqilyu.me/open-jev-fast/) | Prefix-tree packing, fused kernels, CUDA graphs | Borrow serving layout concepts | Hardware-specific and far outside the 4B pilot's memory envelope |
| [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | Gemma-native image/audio/video path with a 256-slot linear head | External multimodal comparator | Fixed slots and one-question calls are weaker architectural priors for runtime-defined options |
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Five workflows, 400 test cases / 2,000 decisions, soft teacher labels | Evaluation only | Gold is three teacher samples and measures teacher agreement, not objective correctness; training on its train split invalidates independent use |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 340M zero-shot operational classifier; dynamic label descriptions; multi-head output | Latency-floor baseline for routing/intent; compare multilingual sibling | English specialist, not a general reasoner or calibrated null model |
| [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) | Qwen3.5-4B multimodality with multi-field masked-token readout | Strong same-scale direct-logit baseline | Model-card scores are self-reported; symbol cap is part of the interface |
| [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | Frozen controlled tasks, group-aware splits, reconstruction metadata | Select exact-rule train configs and OOD controls | Configs overlap; mostly synthetic; do not sum them or expose metadata labels as input |
| [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | 656K train rows with soft distributions, calibration split, many domains | Research ablation after clean gold data | Most rows are teacher-distilled, including Jev outputs; teacher quirks and output-use terms require review |
| [Open-Jev-v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1) | 327K public rows, controlled counterfactuals, WANLI, detailed provenance | Use CC0 controls and separately attributed CC-BY rows if compatible | Source-dependent license; correlated views; public projection differs from exact training mixture |
| [tasksource typed decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) | About 2.5M decisions from 670 sources, group IDs, soft labels, per-row license metadata | Best broad-data candidate after `license_use=commercial`, quality, length, and benchmark filters | Mixed licenses and noisy judgments; its default mix is partly model-selected |
| [jev-tree-choice-cap](https://huggingface.co/datasets/reachjalil/jev-tree-choice-cap) | Hierarchical routing over 320 choices | Serving/evaluation test for very large option sets | Synthetic benchmark, not training data; current Hub viewer has a schema-cast failure |
| [INSTRUCT_JEV](https://huggingface.co/datasets/ctaxnagomi/INSTRUCT_JEV) | 119 instruction rows derived from API documentation | Documentation/evaluation examples only | Teaches Jev concepts and docs, not decision capability |

## Prioritized experiments

### P0 — baseline gate

Run the same Qwen3.5-4B checkpoint in three modes: current Typical native head, direct option-letter logits, and endpoint-pointer head. Fit one temperature per mode on the same calibration split. No new data and no attention.

**Pass condition:** the trained head must improve held-out NLL/Brier or capability, not only in-domain accuracy.

### P1 — multimodal and Hebrew proof

Keep the existing experiment plan: real/shuffled/no-image visual controls, matched H=0 Hebrew ablation, English retention, and cache parity. Add an English/Hebrew paired evaluation immediately; add consistency training only after the basic H bucket shows signal.

### P2 — hard-negative refinement

Starting from the best P1 checkpoint, create grouped option perturbations: near-synonyms, tempting policy exceptions, wrong tool with matching vocabulary, and image/text counterfactuals. Train with the existing CE plus a small candidate-contrastive term, retaining 40% broad replay.

**Pass condition:** gain on held-out adversarial candidates without regression on broad text, Hebrew, or visual controls.

### P3 — trace outcomes

Convert real or benchmark agent steps into state → next-action decisions using post-action evidence. Preserve lineage, keep retries together, and quarantine unclear outcomes. This should improve tool choice and verifier behavior more directly than generic instruction data.

### P4 — attention, only if earned

Create a suite where multiple fields share evidence and one field changes another's correct interpretation. Compare the existing per-field head with one narrow cross-field layer. Stop if the gain is within seed variance or warm-query latency regresses materially.

## What not to do yet

- Do not replace Qwen3.5-4B solely because a 12B or 27B model has a higher card score.
- Do not ingest every public decision dataset into one pool; several overlap, contain public evals, or encode teacher quirks.
- Do not use JevBench repeatedly as the architecture-development set.
- Do not train a separate visual tower or add audio/video before image dependence is proven.
- Do not copy OpenJev-Fast's custom kernels before the model and request shape are stable.

## Recommended next action

Amend the current M0–M2 sequence with the P0 direct-letter and endpoint baselines, then run the already-planned 128-row multimodal/Hebrew smoke. The first new training feature, if those gates pass, should be grouped hard-negative refinement—not a large attention head.
