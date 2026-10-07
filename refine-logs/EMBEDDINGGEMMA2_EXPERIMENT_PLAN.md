# R052 Experiment Plan — EmbeddingGemma 2 evidence routing

**Problem:** Typical's remaining long-workflow errors may come from failing to bind decisive evidence
across long, multilingual, and multimodal state rather than from an inability to execute a policy once
the evidence is isolated.

**Method thesis:** A small, frozen, local multimodal embedder can act as explicit sparse evidence
attention: retrieve a bounded evidence union for every question in a case, preserve global policy
context, and let the unchanged Typical decision model execute the complete typed policy. This is a
retrieval hypothesis, not a claim that embeddings can perform compositional decisions.

**Date:** 2026-10-08

## Frozen component and boundary

- Model: `google/embeddinggemma-2`
- Revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- License: Apache-2.0
- Published: 2026-10-06
- Size: 740M total parameters; 270M text-only, 440M text+vision, 740M full text+vision+audio
- Output: 768 dimensions with supported Matryoshka truncation to 512, 256, or 128 dimensions
- Supported context: 8,192 tokens shared across interleaved text, images, video, and audio
- Runtime: BF16 or FP32 only; FP16 is explicitly rejected because it can silently produce degraded
  or non-finite embeddings.

Primary sources:

- <https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/>
- <https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md>
- <https://ai.google.dev/gemini-api/terms.md>
- <https://huggingface.co/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2>
- <https://pypi.org/project/transformers/5.19.0/>

This plan uses the downloadable open model, not the hosted `gemini-embedding-2` API. The hosted API
would send sealed inputs to Google, incur an API exposure/latency path, and its current terms restrict
using the service or its outputs to develop a competing model. Hosted embeddings therefore cannot be
teacher targets, training data, or part of the self-hosted JevBench entrant without a separate legal
and benchmark decision. The open checkpoint also remains subject to the Gemma Prohibited Use Policy
linked from its model card.

## Evidence versus hypothesis

**Primary-source evidence:** the exact open checkpoint is Apache-2.0, 740M at full modality, emits
768-dimensional vectors, supports 128/256/512-dimensional MRL truncation, accepts 8,192 shared
tokens, covers 100+ languages, and requires re-normalization after truncation. Its model card says
BF16/FP32 are safe and FP16 can silently degrade or return NaNs. Google reports multilingual MTEB-v2
61.36 at 768d and 60.41 at 256d, but no Hebrew-specific, decisive-evidence, JevBench, or workflow
decision result. The current Gemini API terms prohibit using the hosted service to develop competing
models; that restriction is why this branch is local-only.

**Project hypothesis, not established fact:** retrieving a per-question union of decisive evidence
will improve Typical's complete typed decisions, and one frozen embedder will retain parity across
English/Hebrew and real-image controls. Google's generic embedding results do not establish either
claim. B1/B2 and the oracle-headroom condition are designed to falsify them before any integration or
training.

**Current runtime evidence:** this repository's frozen UV environment resolves Transformers `5.17.0`
and has no `transformers.models.embedding_gemma2` module, so the local provider fails closed before
model loading. Compatible Transformers `5.19.0` is now released; its wheel SHA256 is
`afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518` and release source commit is
`c4c4605f961877852b89b97478536bb3a26b6fd1`. P0 must use a separate exact UV lock rather than mutate
the project lock. That runtime is now frozen at `runtimes/r052-embeddinggemma2/uv.lock`, SHA256
`1efdac9dfe879bf76da25c9a6abb3d6303fc62863cea7f048ec8d493a85425ac`; an isolated `uv run
--frozen` import verified Transformers `5.19.0` exposes `EmbeddingGemma2Model`. The remaining run
blockers are R050 ownership of the GPU, gated checkpoint access, and completing the natural audited
panel—not an unreleased runtime.

**Current panel evidence:** commit `5623b91` pins and independently re-verifies ten long natural
NTSB/GovInfo reports (10,523–37,579 extracted words each) with immutable upstream URLs, raw and
extracted-text hashes, and an exact `pdftotext 26.04.0` recipe. Its acquisition-manifest SHA256 is
`6b1d9a30fed29dcf665cc2caa84e96afc023855a5cdf81ba4d7e425e58216b43`; a fresh local verification
passes 10/10. Raw PDFs remain local and non-redistributable. Only reviewed US-government staff text
may enter the panel, and third-party passages and images remain excluded. Commit `b876cf9` adds the
two-stage `prepare-audit`/`finalize` builder, while commit `59fa20b` binds scoring to exact source,
chunk, translation-alignment, image-control, causal-annotation, and contamination receipts. The
combined acquisition/builder/scorer suite passes 58 tests with one intentional missing-provider
skip. This establishes reproducible machinery and source inputs, not a quality result or semantic
ground-truth claim.

Two secondary uses are conditional on B1/B2 succeeding: contamination-safe diversity sampling and
hard-negative mining for R051/RLCD (never embedding vectors as answer labels), and HE↔EN plus
text↔image neighborhood consistency as a data-quality check or auxiliary router loss. Nearest-answer
retrieval is explicitly rejected as Typical's decision mechanism: similarity cannot guarantee typed,
compositional policy execution.

## Claim map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: explicit evidence routing fixes a real bottleneck | It is a cheaper, safer alternative to changing the decision backbone's attention | Gold decisive-exhibit recall and downstream exact-action gains on untouched long cases, both with cluster-bootstrap lower bounds above zero | B0–B2 |
| C2: one local router can preserve Hebrew and multimodal evidence | It directly supports the project's two protected capabilities | No more than one-point English/Hebrew/image regression; decisive-image and HE/EN twin retrieval gates pass | B1–B3 |
| Anti-claim: a strong semantic embedder itself performs policy reasoning | Similarity can retrieve evidence while still failing negation, ordering, arithmetic, or complete action composition | Oracle-evidence ceiling is reported separately; EmbeddingGemma 2 must be compared with BM25 and the unchanged direct model | B1–B2 |

## Experiment blocks

### B0 — reproducibility and leakage preflight (MUST-RUN)

- Load only the exact pinned revision through an isolated UV environment with a compatible,
  exactly-locked Transformers release; do not mutate the live R050 environment.
- Verify model/config/tree hashes, Apache-2.0 license, BF16 finiteness, unit-normalized vectors, and
  deterministic rankings.
- Use only independently generated internal panels and license-cleared R051 sources. Reject benchmark
  paths, IDs, substrings, or upstream lineage before embedding.
- Cache document embeddings by normalized content hash. Never cache benchmark query embeddings across
  a sealed run.
- Keep the R050 and R051 preregistrations byte-identical; R052 cannot interpret or select their arms.
- Validate the panel with `scripts/score_r052_evidence_retrieval.py`: inline media only, exact content
  hashes, externally anchored contamination and causal-annotation manifests, recomputed ordered
  context/outcome hashes, no benchmark lineage/path/ID fields or markers, immutable global-policy
  chunks, paired HE/EN roles, and complete real/shuffled/missing-image controls. Every claimed
  annotation-builder, auditor, corpus, tokenizer, chunker, and vector-generator hash must resolve to
  a supplied local file or complete directory artifact. Precomputed vectors require content hashes
  and local/non-hosted provenance. The scorer always emits BM25 without a model download.
- The scorer validates the structure and content binding of causal annotations; it does **not** replay
  a domain-specific policy executor or establish semantic ground truth. A preregistered evaluator
  must create and freeze those outcome annotations before R052. This deliberately narrow boundary
  avoids treating a caller-supplied digest as proof that an executor actually ran.
- Verify exact hashes for weights, config, processor, module, pooling, and normalization files. Enable
  deterministic Torch algorithms. A local-provider run additionally requires the exact
  Transformers 5.19.0 wheel and a separately frozen UV runtime lock as content-verified artifacts.
  A cached-weight provider contract test skips cleanly when the compatible runtime, lock, wheel, or
  exact checkpoint is absent.

**Pass:** two identical runs produce the same top-k inventory and all vectors are finite and normalized.

### B1 — decisive-evidence retrieval diagnostic (MUST-RUN)

Panel: a minimum 40-case/60-question independent panel: ten long English dossiers with three typed
questions each, ten HE↔EN twin groups, and ten real-image cases with real/shuffled/missing controls
and at least five textual distractors. The ten acquired long sources span five domains and seed the
English dossiers; they do not replace reviewed Hebrew translations, first-party causal charts, or
externally authored annotations. Each long dossier must cover its full declared token range with
contiguous semantic chunks or the frozen 512/64 fallback, preserve explicit global-policy context,
expose at least 15 retrievable chunks, and contain one Choice, Noul, and Score question. It must
identify the minimal decisive exhibit and prove that removing it flips a field or complete policy
outcome.

Chunking is frozen before scoring: semantic section boundaries when present; otherwise 512-token
chunks with 64-token overlap. Every question is embedded separately with its options and policy role.
The case evidence set is the union of per-question top-k chunks plus immutable global policy/header
chunks. This prevents one easy question from suppressing evidence needed by another question.

Compare:

1. BM25 lexical retrieval;
2. frozen Typical/Gemma pooled-state similarity;
3. EmbeddingGemma 2 at 768 dimensions;
4. EmbeddingGemma 2 at 256 dimensions after required renormalization;
5. gold decisive-exhibit oracle.

Metrics: decisive recall@1/@2/@4, MRR, distractor false-positive rate, early/middle/late evidence
slices, source/domain slices, Hebrew/English twin rank agreement, correct-versus-shuffled image
margin, and wall time/VRAM.

The deterministic score receipt contains rankings, source-order evidence unions, exhaustive-negative
FPR, precision, ordered twin agreement, and long/language/modality/position/domain slices. Measured
wall time and peak CUDA memory are written to a clearly marked non-deterministic runtime sidecar.

**Pass:** 768d recall@4 at least .95 overall and .90 in each long, Hebrew, and image slice; at least
three points above both non-oracle baselines; 256d loses no more than one recall point; HE/EN twin
top-k agreement at least .95; shuffled/missing images do not retain the positive retrieval margin.

### B2 — retrieve-then-decide causal ablation (MUST-RUN)

Use one frozen Typical checkpoint and one untouched panel. Compare:

1. full state, direct decision;
2. matched-length BM25 evidence union;
3. matched-length EmbeddingGemma 2 evidence union;
4. gold oracle evidence union;
5. decisive exhibit removed from the EmbeddingGemma union.

The decision prompt, field order, typed-policy executor, calibration, and answer order are identical.
No model weights are updated. Report complete exact-action-set accuracy, complete-outcome NLL/Brier,
field accuracy by Choice/Noul/Score, both-order agreement, and evidence-position slices.

**Go criterion:** EmbeddingGemma routing improves long/dispersed exact actions by at least five points
over direct full context and three points over BM25, with semantic-cluster bootstrap 95% lower bounds
above zero; overall and every field regress by no more than one point; NLL by no more than .02; the
remove-evidence arm loses the expected field or outcome. Oracle evidence must show at least five
points of headroom over direct full context, otherwise retrieval is not the limiting mechanism.

### B3 — self-hosted integration (CONDITIONAL)

Run only after B2 passes. Keep EmbeddingGemma 2 frozen initially. Add a deterministic router in front
of Typical with these constraints:

- route only states above a frozen length threshold or containing non-text media;
- cache document chunks but embed the current query online;
- use 256d only if B1 proves parity with 768d;
- preserve all global policy clauses, section titles, option text, and negation-bearing sentences;
- union evidence across every question in the workflow before any decision is emitted;
- return to the unmodified direct path on router failure; record the fallback;
- never call an external service.

Re-run TypeSafe/WorkflowEvals, the locked long panel, Hebrew twins, real/shuffled/missing-image
controls, and latency. JevBench remains quarantined until the configuration freezes.

**Pass:** Workflow exact-action mean and long exact actions improve without weakening the existing
Hebrew/multimodal gates; zero invalid outputs; warm p95 latency increases no more than 10%. If quality
passes but latency does not, profile/co-reside/quantize the router before changing the decision model.

### B4 — evidence-supervised router adaptation (CONDITIONAL, LATER)

Only if B2 proves the mechanism and B3 misses recall in a localized domain may the open router receive
a small LoRA/projection update. Train only on verified decisive exhibits, cross-lingual twins,
real-versus-shuffled media, and hard same-dossier negatives. Compare frozen versus adapted router at
matched top-k and compute. Do not use hosted Gemini embeddings as labels.

**Pass:** at least two additional long exact-action points over frozen routing, no protected-slice
regression, and the same calibration/latency gates. Otherwise keep the frozen router.

## Run order and compute budget

| Run | Goal | Decision gate | Estimated compute | Status |
|---|---|---|---:|---|
| R052-P0 | pinned-model/runtime preflight | deterministic finite normalized vectors | CPU metadata + <15 GPU min | READY in isolated Transformers 5.19 lock after R050; gated weights required |
| R052-R | retrieval-only diagnostic | B1 recall and cross-modal/cross-lingual gates | <1 A100-hour | SOURCE/BUILDER READY; authored review and audit pending |
| R052-D | downstream no-training causal ablation | B2 exact-action and oracle-headroom gates | 1–2 A100-hours | BLOCKED on R052-R |
| R052-I | frozen-router integration | B3 quality, retention, latency | 1–3 A100-hours | BLOCKED on R052-D |
| R052-L | optional router LoRA | B4 incremental gain | 2–4 A100-hours | CONDITIONAL |

The live R050 evaluator owns the current GPU. R052 must not co-run with it because saturated compute
would alter latency and extend the frozen evaluation. The model can be staged only after R050's
artifacts are complete and durable.

## Why this precedes attention surgery

Embedding routing is explicit, inspectable sparse attention over evidence. It tests the same broad
hypothesis as an attention retrofit—whether the model fails to find distant evidence—without changing
pretrained Gemma attention or risking Hebrew/vision forgetting. If oracle evidence shows no headroom,
neither EmbeddingGemma routing nor KDA is justified; the bottleneck is policy execution/readout. If
oracle helps and learned retrieval fails, improve retrieval. If learned retrieval succeeds but the
decision remains wrong, train the typed policy executor rather than the router.

## Final checklist

- [x] Open local model identified and exact revision pinned
- [x] Hosted API and training-use boundary documented
- [x] Immutable natural long-source acquisition and panel machinery verified
- [ ] 40-case/60-question panel authored, human-reviewed where required, and contamination-audited
- [ ] Retrieval baselines and oracle run
- [ ] Retrieve-then-decide causal ablation passes
- [ ] Hebrew and multimodal preservation gates pass
- [ ] Workflow gain and latency gate pass
- [ ] One frozen self-hosted configuration evaluated externally
