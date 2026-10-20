# Hebrew decision-benchmark sources (researched 2026-10-20)

Scope: Hebrew eval sets that can become Typical decision items (choice / noul / score), Hebrew leaderboards and
published scores, and what decision models have shown on Hebrew. Sizes and licenses come from the HF API or the
dataset-viewer `/size` endpoint unless marked otherwise. "Unstated" means no license on the card or repo; treat it
as research-only until cleared.

Conversion key: **choice** = MC or k-way label, **noul** = yes/no, **score** = ordinal levels.

## 1. Hebrew-native / Hebrew-adapted datasets

| # | Dataset | URL | License | Size (splits) | Task | -> Typical |
|---|---|---|---|---|---|---|
| 1 | **AlephBench** (11 tasks, frozen prompts, per-row outputs for 6 models) | https://hf.co/datasets/HebArabNlpProject/AlephBench | CC-BY-4.0 (card; subsets inherit upstream terms) | 11,432 test items total | MC, NLI, classification, QA, math, MT | Best single source; see 1a |
| 1a | `arc_heb` | same repo | CC-BY-4.0 (ARC is CC-BY-SA-4.0 upstream) | 976 test | 4-way MC | choice(4) |
| 1a | `mmlu_heb` | same | CC-BY-4.0 (MMLU MIT upstream) | 1,004 test (57 subjects, balanced) | 4-way MC | choice(4) |
| 1a | `hellaswag_heb` | same | CC-BY-4.0 (HellaSwag MIT) | 878 test | 4-way completion | choice(4) |
| 1a | `copa_heb` | same | CC-BY-4.0 (COPA BSD-2) | 995 test | 2-way causal | choice(2) or noul per alternative |
| 1a | `Winograd` (Shwartz He-WSC) | same; orig https://www.cs.ubc.ca/~vshwartz/resources/winograd_he.jsonl | CC-BY-4.0 here; original unstated | 278 test | 2-way coreference | choice(2) |
| 1a | `HebNLI` (subset) | same | CC-BY-4.0 here | 2,000 test | 3-class NLI | choice(3), or score(contradiction<neutral<entailment) |
| 1a | `HebrewSentiment` (subset) | same | CC-BY-4.0 here | 2,170 test | 3-class sentiment | choice(3) / score(neg<neu<pos) |
| 1a | `Hebrew-QA` (HQAD) | same | CC-BY-4.0 here | 1,072 test | extractive QA | noul "is span X the answer?" (needs distractor spans) |
| 1a | `israeli_trivia` | same | CC-BY-4.0 here | 99 test | graded 4-way MC (correct/close/wrong/bad) | choice(4); grading gives a ready **score** variant |
| 1a | `gsm8k_heb`, `ted_heb` | same | CC-BY-4.0 here | 984 / 976 | open math / EN->HE MT | weak fit; skip or MC-ify |
| 2 | **HebNLI** (full, MNLI translated with Gemini; 884-pair human-verified gold test) | https://github.com/NNLP-IL/HebNLI ; HF mirror https://hf.co/datasets/HebArabNlpProject/HebNLI (private/gated) | No repo license; inherits MultiNLI terms (mixed OANC + CC-BY-SA-3.0 fiction). Commercial use plausible, needs review | 302,150 sentence pairs + 884 gold | 3-class NLI | choice(3) / noul "entailed?" |
| 3 | **LCHAIM** (ConTRoL translated, long-premise NLI) | https://hf.co/datasets/HebArabNlpProject/LCHAIM | MIT | 6,719 / 799 / 805 | 3-class long-context NLI | choice(3); hard (best LLM ~60%, human ~87%) |
| 4 | **HebrewSentiment** (Facebook comments on President Rivlin's page) | https://hf.co/datasets/omilab/hebrew_sentiment ; https://hf.co/datasets/HebArabNlpProject/HebrewSentiment | MIT (omilab card); "other" on HebArab copy | 12,804 comments | 3-class sentiment | choice(3) / score(3). Known train/test leakage in the original split; dedupe first |
| 5 | **HeQ** (Wikipedia + Geektime reading comprehension, has unanswerables) | https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset ; mirror https://hf.co/datasets/Etelis/HeQ_v1 | CC-BY-4.0 | 27,142 / 1,501 / 1,504 (30,147 QA) | extractive QA, SQuAD2-style | noul "answerable?"; choice over candidate spans |
| 6 | **ParaShoot** (few-shot QA, Wikipedia) | https://hf.co/datasets/imvladikon/parashoot ; https://github.com/omrikeren/ParaShoot | Unstated (no repo license) | 1,792 / 221 / 1,025 | extractive QA | as HeQ; research-only |
| 7 | **HeSum** (abstractive summarization, Shakuf articles) | https://hf.co/datasets/biunlp/HeSum | Unstated on HF | 8,000 / 1,000 / 1,000 | summarization | weak fit; could make noul "is summary faithful?" with perturbed summaries |
| 8 | **HebSummaries** (IAHLT) | https://hf.co/datasets/HebArabNlpProject/HebSummaries | Unstated on HF | 1K-10K | summarization | same as HeSum |
| 9 | **NEMO** (NER on Hebrew treebank) | https://hf.co/datasets/imvladikon/nemo_corpus ; https://github.com/OnlpLab/NEMO-Corpus | Unstated | 10K-100K tokens/sentences | token NER | choice "entity type of span X" (9 types) |
| 10 | **Hebrew Paraphrase** (IAHLT) | https://hf.co/datasets/HebArabNlpProject/Hebrew-Paraphrase-Dataset | CC-BY-SA-4.0 | 9,785 pairs | paraphrase generation | noul "paraphrase?" with negative sampling |
| 11 | **abstractive-qa-llm-eval** (grounding / hallucination) | https://hf.co/datasets/HebArabNlpProject/abstractive-qa-llm-eval | CC-BY-SA-4.0 | 132 | grounded QA | noul "is answer supported by doc?" |
| 12 | **Israeli law corpus** (Wikisource laws) | https://hf.co/datasets/gkour/israeli_law | CC-BY-SA-4.0 | 5,942 docs | raw legal text, no labels | source for synthetic legal decisions only |
| 13 | **Hebrew VAD lexicon** | https://hf.co/datasets/GiliGold/Hebrew_VAD_lexicon | CC-BY-SA-4.0 | 10K-100K words | valence/arousal/dominance ratings | score (binned VAD), word level only |
| 14 | **HeDetox** (offensive -> non-offensive pairs, RANLP 2025) | https://aclanthology.org/2025.ranlp-1.149 (data on GitHub, OffensiveLanguageResearchLab) | See repo | 600 pairs | detox | noul "offensive?" (balanced by construction) |
| 15 | **OLaH / Hebrew offensive tweets** (15,881 tweets, 5 classes) | https://core.ac.uk/works/146514513 | Research (check with authors) | 15,881 | offensive-language classification | choice(5) / noul |

Gaps (searched, nothing public found): Israeli **psychometric exam** (Psychometria) dataset; Israeli **bar exam** QA;
public **Hebrew medical** MCQ. Closest substitutes: Shlav Aleph internal-medicine study (133 Q, IMAJ, items not
released) https://www.ima.org.il/medicineimaj/Article.aspx?aId=6423 ; Bagrut chemistry MCQ benchmark (120 Q)
https://link.springer.com/10.1007/s10956-026-10310-y ; INCLUDE-Hebrew (below), which is in practice the
Israeli driving-theory test.

## 2. Multilingual sets with a Hebrew split

| Dataset | URL | License | Hebrew size | Task | -> Typical |
|---|---|---|---|---|---|
| **Global-MMLU** `he` (MMLU, human-reviewed translation, CS/CA culture tags) | https://hf.co/datasets/CohereLabs/Global-MMLU | Apache-2.0 | 14,042 test / 285 dev | 4-way MC | choice(4); use the culturally-sensitive (CS) subset for Israel-specific items |
| **INCLUDE** `Hebrew` (native regional exams) | https://hf.co/datasets/CohereLabs/include-base-44 (lite: include-lite-44) | Apache-2.0 | base 550 test + 10 val (500 driving licence + 50 logic); lite 250 | 4-way MC | choice(4); only native-Hebrew exam MCQ found |
| **Belebele** `heb_Hebr` (FLORES passage reading MC) | https://hf.co/datasets/facebook/belebele | CC-BY-SA-4.0 | 900 test | 4-way reading MC | choice(4); clean RC-with-options |
| **Global PIQA** `heb_hebr` (culturally native physical commonsense) | https://hf.co/datasets/mrlbenchmarks/global-piqa-nonparallel | CC-BY-SA-4.0 | 100 test | 2-way MC | choice(2) |
| **MASSIVE** `he-IL` (Alexa intents, 60 intents / 18 scenarios) | https://hf.co/datasets/AmazonScience/massive | CC-BY-4.0 | 11,514 / 2,033 / 2,974 | intent classification | choice(60) or two-level choice(scenario -> intent) |
| MMLU-ProX | https://hf.co/datasets/li-lab/MMLU-ProX | MIT | **no Hebrew** (29 langs) | - | - |
| MMMLU (OpenAI) | https://hf.co/datasets/openai/MMMLU | MIT | **no Hebrew** (14 langs) | - | - |
| MGSM | https://hf.co/datasets/juletxara/mgsm | CC-BY-SA-4.0 | **no Hebrew** | - | - |
| xP3 | https://hf.co/datasets/bigscience/xP3 | Apache-2.0 | **no Hebrew** (BLOOM languages) | - | - |

Note: vendor "multilingual" numbers (MMMLU, MMLU-ProX averages) contain **zero Hebrew items**, so they are no
substitute for a Hebrew measurement.

## 3. Hebrew leaderboards and published numbers

### 3a. AlephBench (HebArabNlpProject), snapshot 2026-05-11, chat mode, temp 0, letter-parsed answers
Source: https://hf.co/datasets/HebArabNlpProject/AlephBench (leaderboard.csv; code at github.com/HebArabNlpProject/heb_leaderboard)

| Model | Avg | MMLU | ARC | HellaSwag | COPA | HebNLI | Winograd | Sentiment | Trivia | Hebrew-QA F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| gemini-2.5-flash | 88.8 | 82.8 | 96.3 | 72.4 | 95.2 | 82.5 | 87.1 | 83.4 | 96.0 | 98.9 |
| **gemma-4-31b-it** | 88.1 | 81.9 | 95.8 | 74.0 | 94.2 | 85.2 | 86.0 | 81.4 | 89.4 | 98.9 |
| DictaLM-3.0-24B-Thinking | 85.2 | 79.0 | 92.5 | 69.8 | 95.1 | 82.0 | 77.2 | 75.2 | 86.4 | 97.8 |
| gpt-oss-120b | 84.2 | 85.1 | 93.7 | 66.1 | 91.0 | 79.5 | 67.6 | 81.4 | 88.4 | 97.2 |
| gemma-3-27b-it | 83.5 | 72.4 | 89.2 | 69.0 | 90.8 | 73.6 | 79.1 | 78.5 | 88.9 | 98.6 |
| Hebatron (31.6B MoE, 3B active) | 77.1 | 67.3 | 85.2 | 57.1 | 76.1 | 65.2 | 69.1 | 69.3 | 92.4 | 95.1 |

Per-row outputs are in `results/*.jsonl`, so the Typical item IDs can be aligned exactly. Accuracy only, with no
calibration or logprobs. No Qwen and no Gemma-4-12B.

### 3b. Hebrew LLM Chat Leaderboard (DICTA + MAFAT/NNLP-IL), updated 2026-08-31
Source: https://hf.co/datasets/hebrew-llm-leaderboard/chat-results (34 models) ; UI https://hf.co/spaces/hebrew-llm-leaderboard/chat-leaderboard ;
method in DictaLM-3.0 report https://arxiv.org/abs/2602.02104 . Only Winograd and Trivia are decision-shaped.

| Model | Winograd 0-shot | Trivia 0-shot | Nikud | Summary (pairwise) | Translation (pairwise) |
|---|---|---|---|---|---|
| gemini-3-pro-preview | 96.0 | 86.7 | 87.6 | 51.8 | 51.2 |
| gemini-3-flash-preview | 94.6 | 86.4 | 88.4 | 49.7 | 50.7 |
| glm-5.3-flash | 93.9 | 76.7 | 35.0 | 60.5 | 40.3 |
| **Qwen3.5-122B-A10B** | 93.2 | 70.1 | 67.1 | 46.2 | 31.0 |
| gpt-5.1 | 91.0 | 81.7 | 80.5 | 66.0 | 44.2 |
| claude-sonnet-4.5 | 88.8 | 69.0 | 63.0 | 55.1 | 33.8 |
| **Qwen3.5-397B-A17B** | 86.7 | 72.1 | 78.2 | 47.3 | 42.9 |
| **gemma-4-31B-it** | 86.0 | 58.5 | 73.4 | 48.5 | 33.3 |
| Llama-3.3-70B-Instruct | 83.5 | 60.1 | 4.0 | 37.8 | 19.3 |
| gemma-3-27b-it | 79.9 | 45.5 | 60.2 | 44.5 | 26.7 |
| DictaLM-3.0-24B-Thinking | 78.1 | 60.1 | 86.9 | 56.9 | 30.1 |
| **gemma-4-26B-A4B-it** | 78.1 | 44.5 | 68.1 | 46.0 | 27.3 |
| Qwen3-30B-A3B-Thinking-2507 | 77.0 | 43.5 | 15.8 | 12.4 | 1.0 |
| claude-haiku-4.5 | 76.3 | 52.8 | 62.5 | 44.0 | 13.7 |

The older base-model Hebrew leaderboard (HeQ, sentiment, Winograd, translation; few-shot) is at
https://hf.co/spaces/hebrew-llm-leaderboard/leaderboard (blog: https://huggingface.co/blog/leaderboard-hebrew).

### 3c. Target backbones (no Hebrew number published for either)
- **Gemma-4-12B-it**: the card (https://hf.co/google/gemma-4-12b-it) lists MMMLU 83.4 and MMLU-Pro 77.2. MMMLU has
  no Hebrew. Neither Hebrew leaderboard includes it. Proxies: same-family 26B-A4B (Winograd 78.1, Trivia 44.5)
  and 31B (AlephBench 88.1). Expect the 12B to land around or below 26B-A4B.
- **Qwen3.8-27B**: the card (https://hf.co/Qwen/Qwen3.8-27B) has no multilingual table. The closest published
  numbers are from the Qwen3.5-27B card (https://hf.co/Qwen/Qwen3.5-27B): MMMLU 85.9, MMLU-ProX 82.2, INCLUDE 81.6,
  Global PIQA 87.5, NOVA-63 58.1, PolyMATH 71.2. All are multi-language averages with little or no Hebrew. On the
  Hebrew chat board, Qwen3.5-122B-A10B scores Winograd 93.2 / Trivia 70.1 (well above Gemma-4-31B on Winograd).
- **DictaLM 3.0** (24B from Mistral-Small-3.1, 12B from Nemotron-Nano-V2, 1.7B from Qwen3-1.7B): Hebrew SOTA
  for its size per the report https://arxiv.org/abs/2602.02104 . On AlephBench it trails Gemma-4-31B.
- LCHAIM paper: best few-shot LLM (GPT-4o / Gemma-9B / DictaLM-2-7B) gets 60.1% mean accuracy, about 35 points
  below humans (Malul et al., Findings ACL 2025). https://hf.co/datasets/HebArabNlpProject/LCHAIM

## 4. Decision models on Hebrew / multilingual
No Hebrew result was found for any decision model. What exists:
- **Jev (TypeSafe)**: hosted, closed. One listing says English is the primary training language, and no
  multilingual eval has been published. https://openrouter.ai/blog/insights/what-is-jev/ ; https://infrabase.ai/inference-apis/typesafe-jev
- **Clef / Clef-flash (Cloudflare)**: Apache-2.0, built on **Qwen3.8-27B** / Qwen3.5-9B. The card covers 43 English
  benchmarks (MMLU 90.3, ANLI 69.8, ContractNLI 81.4, ForecastBench Brier 13.9) and no multilingual rows.
  https://hf.co/cloudflare/clef
- **Liquid d1-3B / d1-omni-600M**: LFM Open License v1.0, on LFM2.5-VL-3B. Its 16 card languages exclude Hebrew.
  It reports XNLI 85.0, PAWS-X 76.9, and MASSIVE-intent 87.3 "as decisions", but these are language-averaged with
  no per-language split, and XNLI/PAWS-X contain no Hebrew. https://hf.co/LiquidAI/d1-3B
- **Winnow-12B / E4B / E2B (EldanRing)**: Apache-2.0 GGUF fine-tunes of **Gemma-4-12B-it**. Decision Index 0.2.1 =
  50.02 (Language sub-score 56.0, English). https://hf.co/EldanRing/Winnow-12B
- **Cygnet (blockbrain recipe; Ollaya package)**: Apache-2.0, frozen **Gemma-4-12B-it** base. Co-leads JevBench
  v1.5.4 at 73.70 (Winnow-12B Q8 73.23, Jev 1.13.0 72.13; English).
  https://hf.co/ollaya-dev/cygnet ; https://benchlm.ai/benchmarks/jevbench
- **Decision Index v1.0 (Artificial Analysis)**: AA-Omniscience, HLE-MC, When2Call, and WildGuardTest, all
  English. Jev 1.13 = 58, GLiDE = 54, MS-Decision-1 = 54. https://artificialanalysis.ai/articles/artificial-analysis-decision-index
- "Laya multilingual" (Convai) is listed as multilingual on BenchLM, but no score is published. https://benchlm.ai/decision-models

Implication: a Hebrew decision benchmark with calibration metrics (ECE/Brier) would be the first such
measurement. The comparison baselines would be the **base LLMs' accuracy** from sections 3a/3b, plus our own runs
of Winnow/Cygnet (Gemma-4-12B) and Clef (Qwen3.8-27B) on the same items.
