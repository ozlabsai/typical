# R052 Experiment Tracker — EmbeddingGemma 2

| Run ID | Purpose | System / variant | Split | Decisive metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R052-P0 | runtime and integrity preflight | pinned `google/embeddinggemma-2` `914f7f8…` | non-benchmark smoke | deterministic finite normalized vectors; hashes | MUST | PASS | Public ungated checkpoint cached; CPU/FP32 Hebrew query returned finite unit-normalized 768d vectors, bitwise identical on repeat. Snapshot tree `5d8ab1c…`; runtime lock `1efdac9d…`; no GPU used |
| R052-R0 | lexical baseline | BM25, fixed chunks/top-k | untouched long/HE/MM panel | recall@k, MRR, distractor FPR | MUST | DEV PILOT DONE / B1 PENDING | tracked English pilot BM25 complete recall@4 `.8333`; full reviewed panel remains pending |
| R052-R1 | native representation baseline | frozen Typical/Gemma pooled states | same | recall@k, MRR, latency | MUST | BLOCKED ON PANEL | no training |
| R052-R2 | open multimodal retrieval | EmbeddingGemma 2 768d | same | slice recall@k; HE twin agreement; image margin | MUST | DEV PILOT PASS / B1 PENDING | English candidate anchors `.9333` complete recall@4; no HE/MM/long claim |
| R052-R3 | Matryoshka efficiency | EmbeddingGemma 2 256d, renormalized | same | R2 quality delta; latency/storage | MUST | DEV PILOT PASS / B1 PENDING | 256d ties 768d at `.9333` complete recall@4 on the small English pilot |
| R052-D0 | evidence ceiling | gold decisive exhibit | untouched decision panel | exact actions, NLL/Brier | MUST | BLOCKED ON PANEL | must beat direct by >=5 points |
| R052-D1 | causal downstream ablation | direct/BM25/EG2/oracle/remove-evidence | same | exact actions; clustered CI; retention | MUST | BLOCKED ON RETRIEVAL | no weight updates |
| R052-I | frozen router integration | EmbeddingGemma 2 + Typical | Workflow/HE/MM/long | workflow mean; preservation; p95 | MUST IF D1 PASSES | BLOCKED | self-hosted only |
| R052-L | localized router adaptation | verified-evidence LoRA/projection | independent held-out panel | >=2-point incremental gain | OPTIONAL | CONDITIONAL | no hosted Gemini labels |

Evidence: the open checkpoint's exact identity, modality/parameter/dimension/context claims, MRL
re-normalization rule, numerical-precision warning, generic published metrics, and local runtime are
verified. Ten long natural NTSB/GovInfo reports are bound to immutable upstream URLs and exact
raw/text hashes by acquisition manifest
`6b1d9a30fed29dcf665cc2caa84e96afc023855a5cdf81ba4d7e425e58216b43`.
Source manifest v5 preserves that noncanonical manifest byte-for-byte, cross-checks every acquisition
and reviewed-source field, and atomically installs one privately sealed, exact-hash-validated tree
with a platform-gated no-replace rename. The panel
builder and scorer bind typed-policy replay, source order, stable semantic option IDs, order-invariant
retrieval queries, externally frozen causal annotations, all B1 slices, real FPR, and ordered twin
agreement. The combined focused suite passes 154 tests with one intentional missing-provider skip.

The independently audited v3 development receipt SHA256 is
`58e49a5acc20b03da755146a686cd6d8a54dba79624313f4fafadaa85a6f87af`. It binds the dedicated
runtime lock `1efdac9d…`, exact Transformers wheel, model tree, scorer, runner, 275 embedding inputs,
and a Python 3.11 isolated interpreter. Two runs are byte-identical. On 30 English questions over 85
short candidate anchors, production BM25 complete recall@4 is `.8333`; EmbeddingGemma 2 all-options
768d and 256d are both `.9333` with `.95` item recall@4. The paired outcome is only four dense wins
versus one BM25 win (`p=.375`), so this is directional development evidence only.

Hypothesis: decisive-evidence routing improves Typical, transfers reliably to Hebrew, preserves image
causality, and beats matched BM25/direct baselines. Mechanically audited drafts contain ten
Hebrew-twin groups and ten materialized image-control groups, but they are explicitly AI-authored and
unreviewed. The full 40-case/60-question B1 run still requires independent English semantic,
bilingual, image-rights, pixel-causality, and
contamination receipts plus frozen annotations. The exact-revision weights and runtime are no longer
a blocker; R050 is complete and its GPU pod is stopped. Conditional later uses are hybrid sparse
routing, evidence-privileged self-distillation, diversity/hard-negative mining, and cross-lingual or
cross-modal neighborhood consistency. Nearest-neighbor answers are explicitly out of scope.
