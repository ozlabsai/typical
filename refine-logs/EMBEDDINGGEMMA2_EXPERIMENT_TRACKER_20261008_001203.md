# R052 Experiment Tracker — EmbeddingGemma 2

| Run ID | Purpose | System / variant | Split | Decisive metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R052-P0 | runtime and integrity preflight | pinned `google/embeddinggemma-2` `914f7f8…` | non-benchmark smoke | deterministic finite normalized vectors; hashes | MUST | READY AFTER R050 + GATED WEIGHT ACCESS | Frozen runtime: `runtimes/r052-embeddinggemma2/uv.lock` SHA256 `1efdac9d…`; verified Transformers 5.19.0 provider import. Do not mutate the project 5.17.0 lock. BF16/FP32 only |
| R052-R0 | lexical baseline | BM25, fixed chunks/top-k | untouched long/HE/MM panel | recall@k, MRR, distractor FPR | MUST | BLOCKED ON PANEL | matched retrieval budget |
| R052-R1 | native representation baseline | frozen Typical/Gemma pooled states | same | recall@k, MRR, latency | MUST | BLOCKED ON PANEL | no training |
| R052-R2 | open multimodal retrieval | EmbeddingGemma 2 768d | same | slice recall@k; HE twin agreement; image margin | MUST | BLOCKED ON PANEL | exact revision pinned |
| R052-R3 | Matryoshka efficiency | EmbeddingGemma 2 256d, renormalized | same | R2 quality delta; latency/storage | MUST | BLOCKED ON R2 | pass if <=1 recall point lost |
| R052-D0 | evidence ceiling | gold decisive exhibit | untouched decision panel | exact actions, NLL/Brier | MUST | BLOCKED ON PANEL | must beat direct by >=5 points |
| R052-D1 | causal downstream ablation | direct/BM25/EG2/oracle/remove-evidence | same | exact actions; clustered CI; retention | MUST | BLOCKED ON RETRIEVAL | no weight updates |
| R052-I | frozen router integration | EmbeddingGemma 2 + Typical | Workflow/HE/MM/long | workflow mean; preservation; p95 | MUST IF D1 PASSES | BLOCKED | self-hosted only |
| R052-L | localized router adaptation | verified-evidence LoRA/projection | independent held-out panel | >=2-point incremental gain | OPTIONAL | CONDITIONAL | no hosted Gemini labels |

Evidence: the open checkpoint's exact identity, modality/parameter/dimension/context claims, MRL
re-normalization rule, numerical-precision warning, generic published metrics, and current local
runtime compatibility path are verified. The provider-agnostic scorer has 22 passing CPU-only tests
and one cleanly skipped exact-provider integration test; no model weights were downloaded and no GPU
was used. It validates externally frozen causal annotations without claiming to replay their semantic
executor, resolves every provenance digest to supplied local artifacts, recomputes ordered contexts
and inline outcome diffs, preserves source order, reports all B1 slices plus genuine FPR and ordered
twin agreement, rejects output aliasing of or writes inside any supplied input artifact before
writing, separates model-load
and warm-scoring telemetry, and verifies the complete pinned model file inventory. Local model runs
also require a content-verified UV lock and exact Transformers wheel.

Hypothesis: decisive-evidence routing improves Typical, transfers reliably to Hebrew, preserves image
causality, and beats matched BM25/direct baselines. No R052 quality result exists yet. A real run still
requires the natural contamination-audited panel and preregistered semantic annotations, their frozen
local audit/builder/corpus artifacts, an isolated Transformers 5.19.0 lock and verified wheel, locally
cached gated exact-revision weights, and completion of the live frozen R050 run. Conditional later
uses are diversity/hard-negative mining and cross-lingual or cross-modal neighborhood consistency;
nearest-neighbor answers are explicitly out of scope.
