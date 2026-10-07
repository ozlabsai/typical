# Typical R050 JevBench sealed-evaluation request template

Date: 2026-10-07

Status: unsubmitted draft. Do not publish this request until the frozen R050 candidate has passed
the internal, Hebrew, multimodal, and WorkflowEvals gates and the evidence verifier has closed its
run. Publishing or commenting on GitHub requires explicit operator approval.

This structure follows recent self-hosted requests such as
[Kapteeni #102](https://github.com/fstandhartinger/jevbench/issues/102),
[Plumb #84](https://github.com/fstandhartinger/jevbench/issues/84), and
[Imajev #110](https://github.com/fstandhartinger/jevbench/issues/110). The official sealed run must
remain evaluator-controlled under the frozen
[JevBench v1.5 method](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5.md).

## Proposed title

`[bench request]: Typical R050 12B (Gemma 4 + LoRA, native option probabilities, TypeSafe wire format, multimodal + Hebrew, self-hosted)`

## Request body

Hello — requesting an evaluator-controlled JevBench v1.5.4 run for **Typical R050 12B**.

### What it is

- Base: `google/gemma-4-12b-it` at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Adaptation: rank-32 / alpha-64 LoRA on the language backbone; the vision path remains available.
- Readout: one causal prefill followed by a native softmax over the declared option-letter logits;
  zero generated answer tokens and no verbalized probabilities.
- Interface: authenticated TypeSafe-compatible `POST /v1/systemone`, plus `GET /v1/models`.
- Determinism: fixed model, adapter, runtime, seed, and scorer; overlength inputs are rejected and
  never silently truncated.
- Additional retained capabilities: natural Hebrew decision evaluation and real-image causal-use
  controls are required promotion gates, not model-card claims inferred from architecture.

### Frozen artifacts

Fill these fields only from the closed verification receipts:

| Artifact | Frozen identity |
|---|---|
| Typical evaluation-code commit | `{{TYPICAL_COMMIT}}` |
| Base-model canonical tree | `{{BASE_TREE_SHA256}}` (`{{BASE_TREE_FILES}}` files) |
| Adapter canonical tree | `{{ADAPTER_TREE_SHA256}}` (`{{ADAPTER_TREE_FILES}}` files) |
| Adapter/config publication revision | `{{PUBLIC_ARTIFACT_REVISION}}` |
| Runtime lock | `{{UV_LOCK_SHA256}}` |
| Startup receipt | `{{STARTUP_RECEIPT_SHA256}}` |
| Evaluation preflight receipt | `{{PREFLIGHT_RECEIPT_SHA256}}` |
| Final verification receipt | `{{VERIFICATION_RECEIPT_SHA256}}` |
| Evidence-preservation receipt | `{{PRESERVATION_RECEIPT_SHA256}}` |

Public artifacts: `{{PUBLIC_REPOSITORY_OR_BUNDLE_URL}}`.

### Reproduction

The evaluator receives an immutable bundle containing the pinned code, UV lock, canonical model
inventory, adapter identity, startup verifier, and server command. The normal launch is:

```bash
uv sync --frozen
uv run --frozen python scripts/serve_typesafe_gemma_lora.py \
  --model {{BASE_MODEL_PATH}} \
  --model-revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --expected-model-tree-sha256 {{BASE_TREE_SHA256}} \
  --adapter {{ADAPTER_PATH}} \
  --expected-adapter-tree-sha256 {{ADAPTER_TREE_SHA256}} \
  --model-id typical-r050-sha256-{{ADAPTER_SHA_PREFIX}} \
  --receipt server-startup-receipt.json \
  --max-context 16384 --dtype bfloat16 --device cuda \
  --attn-implementation sdpa --host 127.0.0.1 --port 8090
```

The server refuses an unauthenticated non-loopback bind. We can instead provide a frozen offline
snapshot or use an evaluator-owned pod, whichever keeps all sealed items outside our control.

### Closed local diagnostics

These are diagnostic and do not claim official v1.5.4 parity:

| Evaluation | Result |
|---|---:|
| Public JevBench valid decisions | `{{PUBLIC_VALID}} / 231` |
| Easy / standard / hard correct | `{{PUBLIC_EASY}} / 48`; `{{PUBLIC_STANDARD}} / 72`; `{{PUBLIC_HARD}} / 111` |
| Public top-label ECE / Brier / NLL | `{{PUBLIC_ECE}} / {{PUBLIC_BRIER}} / {{PUBLIC_NLL}}` |
| Serial p50 / p95 latency | `{{P50_SECONDS}} / {{P95_SECONDS}}` seconds |
| Mean input / output tokens | `{{MEAN_INPUT_TOKENS}} / 0` |
| WorkflowEvals security exact set | `{{SECURITY_SCORE}}` |
| WorkflowEvals trace primary action | `{{TRACE_SCORE}}` |
| WorkflowEvals invoice exact set | `{{INVOICE_SCORE}}` |
| WorkflowEvals customer exact set | `{{CUSTOMER_SCORE}}` |
| Equal-workflow mean | `{{WORKFLOW_MEAN}}` |
| Natural Hebrew preservation | `{{HEBREW_RESULT_AND_GATE}}` |
| Real-image causal-use preservation | `{{MULTIMODAL_RESULT_AND_GATE}}` |

No public or WorkflowEvals result selected the checkpoint, loss, temperature, prompt, or runtime.

### Training and benchmark-use disclosure

- This is our own model continuation from a general multimodal Gemma foundation; it is not a
  fine-tune of Jev, Winnow, Cygnet, Clef, Intern-Decision, or another decision checkpoint.
- Training uses programmatically verified typed decisions, executable policy groups, broad
  decision replay, natural Hebrew examples, and real-image examples.
- JevBench public and sealed rows are absent from training, calibration, checkpoint selection, and
  loss selection. Report the final exact/normalized/n-gram contamination audit here:
  `{{CONTAMINATION_AUDIT_SUMMARY_AND_SHA256}}`.
- The public 231-row suite was run once after the model and runtime were frozen. Its output did not
  trigger retraining or calibration.
- WorkflowEvals was run once after freezing and was not used as an optimization signal.
- All procedural deviations, including the recorded R050 operator-blinding deviation, are listed
  in `{{DEVIATION_LOG_URL_OR_HASH}}`.

### Limits and cost basis

- Context limit: 16,384 tokens; longer inputs are rejected.
- Option limit and supported typed questions: `{{VERIFIED_OPTION_LIMITS}}`; Choice, Noul, Score.
- Hardware and serial benchmark conditions: `{{GPU_AND_RUNTIME}}`.
- Cost convention and calculation: `{{COST_BASIS_WITH_DATE_AND_SOURCE}}`.
- License for base, adapter, data, and server: `{{LICENSES}}`.

We are happy to provide the frozen snapshot directly to the evaluator or reproduce the server on
an evaluator-controlled offline pod. Please prescribe the sealed-run transfer mechanism.

## Pre-publication gate

Every placeholder must be resolved from immutable receipts or published artifacts. Before sending:

1. re-run the preservation verifier against the archived run trees;
2. confirm WorkflowEvals parity and Hebrew/MM retention are true;
3. publish or transfer the exact adapter and code snapshot without changing their hashes;
4. independently reproduce the startup receipt on a clean pod;
5. obtain explicit user approval for the external GitHub message.

