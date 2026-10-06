# R047 reference evaluator incident — 2026-10-06

Status: failed closed before reference output publication; no control or treatment candidate was
trained or evaluated.

## Frozen package superseded

- Recipe SHA-256: `39aed1b8584f746d70d9779fc8ea359d4da4b5a6d0d787550699aa63aee04a23`
- Reference provenance SHA-256: `108afb5c1246668a4ffc49743bdc482303febdde52374deea55e6173ed039613`
- Assignment SHA-256: `a1d55063313da881396b640819a97f45fb54fd4feecd05b24bfff21fb89666b8`
- Preparation receipt SHA-256: `5619af841c44bdd210cfef7cf122979b96c2b04e314b83769fb82204c532c070`

These artifacts remain immutable but cannot satisfy any corrected reference or candidate gate.

## Reproduction

The first reference command stopped at the evaluator's batched/scalar parity gate:

`r047p-4701-unseen_k2-c00-s000::00::he::yes_no::final: batched/scalar discrete prediction differs`

Across the frozen 152-row parity subset, BF16/SDPA multi-row scoring differed from singleton
scoring by as much as `.147969` probability and changed six discrete/tie outputs. For one target:

- scalar/singleton: `p(yes)=.4687906504`
- same-length second row: `p(yes)=.5`
- shorter second row: `p(yes)=.5`
- longer second row: `p(yes)=.4378235042`

The same-length case falsifies padding as the cause. It is consistent with PyTorch's documented
non-equivalence of batched and per-slice floating-point computation:
https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations

Float32 singleton scoring was diagnostic only: the same target changed to `p(yes)=.5210618377`,
while a same-length batch differed by only `1.43e-6`. Adopting float32 after this observation would
change the estimand and is not allowed.

## Prospective correction

Before any new reference run, freeze a replacement package that requires:

- BF16 model load, SDPA request, `max_length=8192`, no cache, frozen row order;
- the same JevBench-style singleton forward primitive, with one `[1,L]` input per call, no mask,
  `logits_to_keep=1`, and last-logit float32 reduction;
- canonical- and reverse-order passes on the unchanged 152-row subset;
- exact discrete agreement and maximum probability delta at most `1e-5` after matching by ID;
- observed singleton-forward counts, SDPA profiler evidence, and an exact runtime fingerprint shared
  by reference, control, and treatment;
- assignment/preparation authority for the scorer, launcher, amendment, and atomic no-replace
  publisher, plus exact intent/completion evidence for both candidate arms;
- official comparison recomputation from the complete chronology, followed by a second raw-evidence
  recomputation inside the final Hebrew/multimodal retention gate;
- rejection of the former batched receipt schema and all superseded package hashes.

The panel, base model, tokenizer, parent adapter, training arms, trainer configuration, bootstrap,
and statistical gates remain unchanged.
