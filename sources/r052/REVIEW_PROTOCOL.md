# R052 independent panel review protocol

This protocol separates source authoring from evaluation. A dossier is not eligible for R052-B1
until every gate below is complete and its receipt is content-addressed. Development-pilot scores do
not count as B1 and must use disjoint cases.

## 1. Rights and source review

For each acquisition-manifest snapshot, a reviewer records the exact UTF-8 character ranges retained
from the extracted text. The promotion tool, not the reviewer, slices those ranges and hashes the
result. Reviewers must confirm that retained text was prepared by the relevant US-government agency
and remove third-party text, photographs, illustrations, logos, and seals. The upstream PDF remains a
local verification input and is never declared redistributable.

Required receipt fields per source:

- acquisition snapshot ID and acquisition-manifest SHA-256;
- reviewer identity or pseudonymous reviewer ID and review date;
- ordered, non-overlapping retained character ranges;
- explicit staff-authored-text, third-party-exclusion, evaluation-use, and redistribution decisions;
- reviewed-text SHA-256 emitted by the promotion tool;
- unresolved concerns, which make the source ineligible.

## 2. Evidence and typed-policy review

Before questions and options are frozen, two reviewers who cannot see them map exact reviewed-text
spans or image regions to opaque typed fact IDs. They record value, unit, source chunk/content hash,
locator hash, and an adjudication receipt in a question-blind fact ledger. A separate author then
writes a bounded Choice/Noul/Score rule graph over those fact IDs; missing facts remain unknown and
never imply false. The ledger and rubric are frozen before retrieval scoring.

Two reviewers then independently assess each question. An adjudicator resolves disagreements without
seeing retrieval rankings. They verify:

- the answer follows from the declared decisive chunks;
- every other retrievable chunk is explicitly labeled as a distractor for that question;
- removing all and only the declared decisive chunks produces the preregistered replay transition;
- the removal is inclusion-minimal, decisive-only evidence reproduces the full result, distractor-only
  evidence does not, and removing distractors leaves the result unchanged;
- the answer cannot still be reconstructed by elimination, arithmetic using retained numbers, a
  duplicated statement, the question wording, title, or option wording;
- the question, options, and evidence do not quote or derive from JevBench, WorkflowEvals, Open-Jev,
  typed-decisions, or any other target benchmark lineage.

The standard-library replay executor recomputes outcomes from visible facts and exact rational
arithmetic. A full resolved result becoming unresolved is reported as `resolution_loss`, not a
semantic label flip; only two different resolved outputs are a `label_flip`. R052-B1 is described as
retrieval of replay-eligible decisive evidence. Causal claims about Typical are reserved for R052-B2,
where removal must measurably change the model's prediction.

## 3. Hebrew twin review

Each Hebrew case has exactly one English twin and a one-to-one mapping for policy clauses, questions,
options, chunks, decisive labels, distractor labels, and outcomes. One bilingual reviewer checks the
translation and another performs a blind back-translation comparison. Proper names and numeric units
remain invariant. Hebrew twins are additive and do not replace the ten long English dossiers.

## 4. Multimodal control review

Images are first-party deterministic causal charts or diagrams, never extracted third-party report
images. Every image question has matched real, shuffled, and missing variants with identical neutral
text and at least five text distractors. A reviewer confirms that only the real image contains the
decisive relationship and that the shuffled image does not leak it through labels, filenames, alt
text, dimensions, or ordering.

## 5. Freeze and contamination audit

The complete minimum panel contains 40 cases and 60 questions: ten long English dossiers with one
question of each decision type, ten English/Hebrew twin pairs, and ten image-control cases. Before any
retrieval method runs, freeze exact source, chunk, question, annotation, tokenizer, chunker, alignment,
image-control, and policy-executor hashes. Run the public-corpus audit, then ask the benchmark owner to
perform the sealed-corpus overlap check. Any post-score question, option, evidence, or distractor
change creates a new panel version and invalidates comparison with prior methods.
