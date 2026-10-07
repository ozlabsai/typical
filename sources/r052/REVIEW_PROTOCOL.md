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

Two reviewers independently assess each Choice, Noul, and Score question. An adjudicator resolves
disagreements without seeing retrieval rankings. They verify:

- the answer follows from the declared decisive chunks;
- every other retrievable chunk is explicitly labeled as a distractor for that question;
- removing all and only the declared decisive chunks changes the result under the frozen policy
  rubric, rather than merely lowering confidence;
- the answer cannot still be reconstructed by elimination, arithmetic using retained numbers, a
  duplicated statement, the question wording, title, or option wording;
- the question, options, and evidence do not quote or derive from JevBench, WorkflowEvals, Open-Jev,
  typed-decisions, or any other target benchmark lineage.

The rubric and replay executor are frozen before scoring. R052-B1 is described as decisive-evidence
retrieval; causal downstream claims are reserved for R052-B2 and require executor replay.

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
