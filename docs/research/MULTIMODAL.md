# Typical Multimodal: first investigation

**Date:** 2026-09-29  
**Status:** E0 plumbing implemented; E1 corpus/training pilot ready

## Verdict

The initial idea is right, and the shortest path is smaller than a model migration:
`tm2` already uses the natively multimodal `Qwen/Qwen3.5-4B-Base`. The current
backbone loader explicitly discards its vision tower and keeps only
`.language_model` (`pcdm/encode.py`). Keep the full wrapper, feed image tokens into
the state prefix, and reuse the existing native suffix renderer and decision head.

Do **not** build another fusion module or retrain from scratch first. Initialize from
`tm2`, freeze the pretrained vision tower, and continue training the existing LoRA +
native head on a small visual-decision mixture with text replay.

This first version should mean **visual Typical**, not every modality:

- state: text plus one image;
- queries and runtime candidates: text;
- outputs: the existing Choice, Noul, Score, and explicit null distributions;
- one visual prefill reused by many queries;
- no audio, multi-image, video, image candidates, or visual generation yet.

## Why the architecture fits

```text
image + state text -> Qwen3.5 vision encoder + language prefix -> shared state cache
                                                                  |
query + runtime text options -> existing native suffix/readout ----+-> P(options, null)
```

The load-bearing Typical invariant survives: expensive state perception happens
once; each decision is still a short causal suffix. The existing `NativeHead` already
reads the final decision state and pooled option spans, so it does not need a new
multimodal head. Visual information reaches both through causal attention to the
cached prefix.

The implementation boundary is narrow:

1. Keep `Qwen3_5Model.visual` instead of replacing the wrapper with
   `.language_model`.
2. Use `AutoProcessor` for interleaved image/text inputs and preserve
   `pixel_values`, `image_grid_thw`, and `mm_token_type_ids`.
3. Apply the existing tap/truncation and LoRA logic to
   `model.language_model.layers`; leave the vision tower frozen for the pilot.
4. Let text-only rows continue through the same wrapper without pixel inputs.
5. Extend the state schema with an optional image reference; candidates remain
   unchanged.
6. Carry Qwen3.5's multimodal RoPE delta into cached query suffixes and add a
   cached-vs-full parity check.

## Local feasibility probe

A no-training probe used the cached `Qwen/Qwen3.5-0.8B-Base` checkpoint and the
repo's current Transformers 5.17 environment:

- The full multimodal wrapper runs on MPS and produces image-dependent decision
  states: identical suffix text over two different repo images had final-state cosine
  similarity `0.869707`.
- A visual prefix followed by a cached text suffix matched the equivalent one-pass
  sequence at cosine `0.999919` in bf16 (`max_abs=0.183594`, `mean_abs=0.033264`).
- A naïve cached wrapper call failed with a position-length mismatch. Explicit suffix
  positions using the prefix's multimodal RoPE delta restored near-parity. This must
  be a regression test, not an undocumented serving trick.
- Image token budgeting is mandatory. On the tall site screenshot, the processor's
  default produced `12,480` image tokens. Capping `max_pixels` at 65,536 / 262,144 /
  1,048,576 produced 44 / 220 / 979 image tokens. Start at `max_pixels=262144`
  (roughly at most 256 visual tokens) and measure quality before raising it.
- `torchvision` is required by the processor but is not currently a project
  dependency. The probe used `uv run --with torchvision`; add it only when the
  implementation starts.

## Pilot data recipe

Add one visual bucket `V`; do not disturb the data model beyond an optional image.
Use only training splits and run the same license/leak audit as the current corpora.

Initial sources should cover the decision types without inventing a new objective:

- visual multiple choice: start with ScienceQA only for a research pilot. Its upstream
  data license is CC-BY-NC-SA-4.0, so it must not feed commercially released weights.
  Replace it with an audited permissive source before release training;
- visual Noul: object/attribute presence and contradiction rows derived from
  annotated images;
- null training: gold-option removal, mismatched image/question pairs, and images
  where the evidence is genuinely insufficient;
- text replay: the current `tm2` mixture, preserving its exact typed heads and
  ordinal targets.

For the first pilot, skip visual Score. Keep learning Score from the existing text
rows until there is a real ordinal visual dataset; synthetic severity labels would
test the generator more than the model.

Every visual evaluation item needs three counterfactuals:

1. correct image;
2. shuffled image;
3. no image or a corrupted image.

Without these controls, a high score can come from question and answer priors rather
than pixels.

## Experiment ladder

### E0 — plumbing and cache parity (done locally)

The full wrapper consumes images, hidden states respond to image changes, and cached
suffix execution is feasible with explicit multimodal positions. Reproduce this in a
small test before touching training code.

### E1 — frozen visual probe

- Backbone: `Qwen/Qwen3.5-4B-Base`, tap 23, initialized from `tm2`.
- Freeze: vision tower and language LoRA.
- Train: existing native head only, on roughly 20k--50k visual decision rows mixed
  with the current text replay (20% visual / 80% text batches).
- Resolution: one image, at most ~256 visual tokens.
- Mix: `E:.30,K:.11,W:.26,U:.08,V:.15,H:.10`. `H` is Hebrew MASSIVE
  (CC-BY-4.0); keep it at 10% initially so cross-lingual gains and English retention
  can be measured independently of the visual arm.
- Compare: direct Qwen3.5 letter scoring, real-image Typical, shuffled-image Typical,
  no-image Typical, and the text-only `tm2` suite.

Proceed only if all hold:

- real-image accuracy exceeds shuffled-image accuracy by at least 10 points;
- the head retains at least 80% of the direct VLM baseline's visual accuracy;
- null probability separates answerable from gold-dropped/mismatched rows
  (`AUROC >= .75` as an initial screen);
- established text metrics remain within their existing seed or confidence bounds.

### E2 — mixed continuation

Only if E1 passes, unfreeze the existing top-eight language LoRA layers and continue
from `tm2`. Give visual rows 20% of batches and scale the current E/K/W/U proportions
into the remaining 80%. Keep the vision tower frozen. Use the current best-checkpoint
logic, adding a visual validation NLL alongside the uncertainty/curriculum sets.

Pass if visual accuracy, NLL, Brier, ECE, null AUROC, and real-vs-shuffled image gap
all improve without moving the established text suites beyond their existing seed or
confidence bounds. Benchmark state prefill separately from warm per-query latency.

### E3 — only if E2 says the frozen vision tower is the bottleneck

Tune the vision projector or last vision blocks with a small adapter. Do not start
here: Qwen3.5 was pretrained with early-fused image/video tokens, so another visual
adapter is speculative until the frozen-tower arm fails.

## What is and is not novel

"A VLM that answers multiple-choice questions" is not novel. Prefix caching is also
standard causal-model behavior. The potentially defensible contribution is their
combination under Typical's contract:

- runtime-defined visual decisions rather than generated answers;
- one image encoding amortized across many heterogeneous queries;
- typed Choice/Noul/Score distributions with explicit visual insufficiency/null;
- calibration and selective-risk evaluation under image corruption and modality
  mismatch;
- direct comparison with generative VLM confidence at matched visual backbones.

Recent work reports that multimodal LMs remain overconfident and that ordinary
fine-tuning or scaling does not reliably fix calibration. That supports calibration
and null behavior as first-class hypotheses, but it also means novelty must be based
on measured selective risk and amortized serving—not the word "multimodal."

## Decision

Implement E0 as one regression test, then E1. A full multimodal retrain, video path,
or new fusion architecture is not justified before the frozen visual probe shows
that the native head can extract pixel-dependent decisions.

The implementation is opt-in and keeps text checkpoints compatible. Build the pilot
data and launch the frozen-LoRA arm with:

```bash
uv run scripts/pilot_corpora.py hebrew --out data_he
uv run scripts/pilot_corpora.py scienceqa --out data_v --accept-noncommercial
uv run pcdm/train.py --name tm2_mm_e1 --readout native --multimodal \
  --backbone Qwen/Qwen3.5-4B-Base --tap_layer 23 --init_from runs/tm2/best.pt \
  --freeze_lora --nc_head n3 --nc_render letters_nonull --null factored --zscore \
  --data data_v5 \
  --extra_data data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u,data_v,data_he \
  --bucket_map data_wf_long=W,data_wh=W,data_u=U,data_v=V,data_he=H \
  --family_weights E:.30,K:.11,W:.26,U:.08,V:.15,H:.10
```

Each visual example carries one optional `image` path. The batched trainer partitions
text and visual rows internally; serving accepts the same image through
`native_kv_decide(..., image=...)` and preserves Qwen3.5's M-RoPE offset across the
cached suffix.

## Primary references

- [Qwen3.5 base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) — native
  image/text model, base checkpoint, Apache-2.0.
- [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) —
  full multimodal wrapper, hybrid attention, and multimodal RoPE requirements.
- [Unveiling Uncertainty: Calibration of Multimodal LLMs](https://arxiv.org/abs/2412.14660) —
  multimodal miscalibration and IDK evaluation.
- [Overconfidence and Calibration in Medical VQA](https://arxiv.org/abs/2604.02543) —
  overconfidence persists across VLM families/scales; post-hoc calibration helps
  calibration but not discrimination.
- [MASSIVE dataset notice](https://github.com/alexa/massive/blob/main/NOTICE.md) —
  CC-BY-4.0 multilingual intent data used for the Hebrew bucket.
- [ScienceQA upstream README](https://github.com/lupantech/ScienceQA/blob/main/README.md) —
  dataset license is CC-BY-NC-SA-4.0; research pilot only.
