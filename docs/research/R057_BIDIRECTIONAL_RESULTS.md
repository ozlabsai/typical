# R057: StartLux-style bidirectional prompt attention on Typical (Gemma-4-12B-it), 2026-10-21

**Verdict: negative at our dose.** Neither stable bidirectional arm beats its causal control, R055-A. The
LR-matched arm diverged. At half LR, the best arm (R057h) scores 4–6 points below R055-A on HE, MX, native and EN 231,
and 1.8 points below on the external Hebrew panel. It also fails image retention (mismatch null-recall 59/128 vs 94).
The mask alone, with no training, breaks the frozen base by 25–36 points. The adapter has to relearn reading under the
new mask, and at R055's dose (17.6k presentations, LoRA r32) it does not catch up. StartLux trained its mask on ~1.7M
examples after a 2.7M-example full fine-tune. This result bounds "the mask at our dose", not the idea itself.

StartLux mechanics: `.context/r057/STARTLUX_NOTES.md`. Raw outputs: `.context/r057/pod/` (per-item hejev/external
rows, metrics, retention, receipts, logs). Status log: `.context/r057/STATUS.md`.

## Main table (accuracy; same protocol and columns as R055_RESULTS)

HE = translated (230), MX = mixed (230), native (119), EN = public 231. External = `.context/hebrew-bench/external/`.
HE−EN gap and agreement are measured on the 230 twin pairs (`scripts/hejev_score.py`). All rows use the native
readout through `serve_typesafe_gemma_lora.py`; the R057 rows and "base + mask" run with `--bidirectional-prompt`.
Every run had 0 failed rows.

| model | HE | MX | native | EN 231 | Belebele | HebNLI | G-MMLU | Sentiment | Winograd | HeQ | **ext. mean** | HE−EN | HE/EN agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R057h** (mask, LR 2.5e-5, step 2000) | .757 | .770 | .824 | .784 | .900 | .829 | .607 | .828 | .691 | .927 | **.797** | −.030 | .874 |
| **R057q** (mask, LR 1.25e-5, step 2200) | .739 | .704 | .731 | .723 | .883 | .814 | .590 | .836 | .698 | .927 | **.792** | +.013 | .878 |
| R057 (mask, LR 5e-5 = matched) | diverged at step ~493, dead by 750; not evaluated | | | | | | | | | | | | |
| **R055-A** (causal control, step 2200) | .813 | .822 | .874 | .848 | .901 | .848 | .617 | .828 | .770 | .925 | **.815** | −.035 | .883 |
| frozen base, causal (zero LoRA) | .852 | .835 | .815 | .874 | .890 | .796 | .635 | .766 | .737 | .917 | **.790** | −.022 | .935 |
| frozen base + mask (zero LoRA) | .557 | .474 | .403 | .511 | .542 | .421 | .390 | .632 | .590 | .682 | **.543** | +.048 | .670 |
| StartLux-9B-Instant (peer) | .878 | .874 | .916 | .896 | .917 | .903 | .703 | .792 | .863 | .927 | .851 | −.017 | .935 |
| torchcast-12b (peer) | .865 | .865 | .899 | .879 | .900 | .791 | .630 | .786 | .773 | .920 | .800 | −.013 | .939 |

The R055-A and peer rows are copied from `hebrew/R055_RESULTS.md`; the R055-A row was re-scored from its per-item
rows and matches. The other peers (Cygnet, Winnow, clef, R044-R, R055-B) are in that file. Every one of them scores
above R057h on both HE and EN.

- **R057h vs R055-A:** −5.6 HE, −5.2 MX, −5.0 native, −6.4 EN, −1.8 ext. mean. It ties on Belebele, Sentiment and HeQ
  and loses on Winograd (−7.9, 22 items), HebNLI (−1.9) and G-MMLU (−1.0).
- **Mask alone (frozen base):** −29.5 HE, −36.1 MX, −41.2 native, −36.3 EN, −24.7 ext. mean. The step-0 dev evaluation
  agrees: he_dev accuracy goes from .797 to .426. Gemma-4 was never trained to read a bidirectional prompt.
- The frozen base with the native readout is itself ahead of R055-A on HE (.852), MX and EN (.874). This is a new data
  point: R055 compared only against Cygnet's tempered readout.

## Retention (`scripts/eval_gemma_mm_he.py`, 128 per suite; candidate run with the mask, base causal)

| model | MASSIVE-he | MASSIVE-en | HE/EN JSD | SQA real | shuffled | no-image | mismatch null-recall | gold-dropped null-recall | answerability AUROC |
|---|---|---|---|---|---|---|---|---|---|
| base Gemma-4-12B-it (causal) | 118 | 118 | .026 | 85 | 26 | 46 | 99 | 84 | .850 |
| R055-A | 119 | 123 | .032 | 109 | 29 | 58 | 94 | 91 | .922 |
| **R057h** | 118 | 120 | .023 | 114 | 52 | 87 | **59** | 86 | .885 |
| **R057q** | 119 | 121 | .024 | 112 | 42 | 84 | 81 | 82 | .892 |

- **Text retention:** passes. MASSIVE is within 2/128 of the base.
- **Image retention:** fails. Real-image accuracy goes up (114 vs 109), but so do shuffled (52 vs 29) and no-image
  (87 vs 58). The real−no-image margin shrinks from +.40 (R055-A) to +.21 (R057h). Mismatch null-recall falls 35
  items (59 vs 94). With the image tokens inside one bidirectional block, R057h relies less on the pixels and
  answers from text priors more often.
- The mask works in the multimodal path. Image soft tokens are ordinary prompt tokens inside the block, and Gemma's own
  bidirectional rule for image blocks on sliding layers is a subset of the mask (280-token blocks < 1024 window).

## What was built (commit af5bc7b)

- `--bidirectional-prompt` on `train_gemma_decision_lora.py`, `serve_typesafe_gemma_lora.py` (through
  `GemmaDecisionRuntime`) and `eval_gemma_mm_he.py`.
- One mechanism serves all three: `enable_bidirectional_prompt(model)` registers a forward pre-hook on the inner
  `Gemma4UnifiedModel`. The hook replaces the 2D or absent `attention_mask` with the `{full_attention, sliding_attention}`
  mapping of 4D boolean masks, which Gemma-4's forward in transformers 5.17 accepts as-is (the same path `generate`
  uses).
- Rule for a row of n tokens (read-out slot n−1): q may read k iff k < n and (k ≤ q or (q < n−1 and k < n−1)).
  - The global layers get this rule as-is.
  - The sliding layers also keep a symmetric window |q−k| < 1024.
  - Padding is never a key, and right padding is checked.
- Difference from StartLux:
  - StartLux makes the prompt bidirectional only in its full-attention layers (its linear layers cannot be), and with
    p = L it includes the read-out token.
  - R057 does both layer types, per the brief, and keeps the read-out slot causal.
  - For the read-out row itself the two rules are identical.
- **Default off is byte-identical.** Without the flag no hook is registered, so the stock forward runs. The server
  receipt gains `prompt_attention` only when the flag is on.
  - Trainer `config.json` gains a `bidirectional_prompt` key.
- `tests/test_bidirectional_prompt_mask.py` runs on CPU with no download. It checks:
  - the brute-force rule;
  - on a tiny random Gemma-4: stock is causal; with the hook, a late prompt token reaches position 0, the read-out
    slot is no key for the prompt, padding is inert, and removing the hook restores the stock forward bit for bit.
- Runner: `scripts/r057_pod.sh`. It reuses `r055_pod.sh` `train` with R055-A's exact arguments (mix sha 1ae2445f…,
  seed 55, r32/α64, microbatch 1 × accum 8, max length 8192, wd .01, 2200 steps, eval every 250 / save every 500).

## Training, divergence and deviations

- **R057 (matched: R055-A's args plus the flag; lr 5e-5 → 5e-6)**
  - It learned fast: he_dev .426 → .770 at step 250.
  - It diverged at step ~493: train loss .58 → 1.4, train accuracy .80 → .35.
  - It never recovered. he_dev was .332 at step 500 and an identical .324 / NLL 1.057 at steps 750 and 1000, i.e. a
    uniform letter readout.
  - The LoRA weights at step 1000 are finite and normal-sized (max |w| .023, as R055-A's .026). The collapse is in the
    activations, not NaN weights.
  - Killed at step 1045 (10:12Z). R055-A had a similar blip at step 151 and recovered within 50 steps.
- **Deviation 1 (09:50Z): R057h** = R057 with half the LR (2.5e-5 → 2.5e-6), on the second GPU. This replaced the
  optional R055-A WorkflowEvals run, which was killed.
- **Deviation 2 (10:45Z): R057q** = quarter LR (1.25e-5 → 1.25e-6), on GPU0 once R057 was dead.
- Both stayed stable. Checkpoints were selected by the R055 rule (lowest he_dev NLL among saved steps; no test data):

  | arm | selected step | he_dev acc | he_dev NLL |
  |---|---|---|---|
  | R057h | 2000 | .871 | .398 |
  | R057q | 2200 | .859 | .426 |
  | R055-A (for comparison) | 2200 | .922 | .314 |

  At every eval step from 750 on, both bidirectional arms' he_dev is below R055-A's. R057h was briefly ahead at step 500
  (.801 vs .730), during R055-A's dip.
- **Single seed per arm.** The two lower LRs are fallbacks, not a sweep. The selected adapters are in the PRIVATE repo
  `guychuk/typical-r057-bidir` (`r057h-step2000/`, `r057q-step2200/`), pushed from the pod.
- **WorkflowEvals: not run.**
  - Both WF launches (R057h with the mask, R055-A causal) failed before sending a single case. The runner's readiness
    probe called the keyed server without its bearer token, got 401 for 30 minutes and gave up.
  - That probe is fixed in `scripts/r057_pod.sh`, but WF was not rerun: it was optional, and both arms already trail
    R055-A everywhere else.
- **Network:** the laptop's uplink to the pod ran at ~0.15–0.3 MB/s. The 525 MB R055-A adapter took ~45 min with a
  chunked, resumable, sha256-verified upload. A controller outage (13:40–16:55Z) left the GPUs idle from 15:29Z until
  the stop.

## Compute and cost

- Pod `s0lfh15iy7bjx7` (`typical-r057`): 2× A100-SXM4-80GB, secure cloud, $3.58/h.
- Up 07:33 → 17:01 UTC = 9.47 h, **≈ $33.9**. Stopped EXITED.
- Its own cost guard `com.typical.hanoi.costguard.r057` was booted out; the shared guard plist was not touched.
- Wall times:
  - Training: R057h 4.31 h, R057q 4.35 h, and R057 2.05 h until killed; both GPUs trained at once, ~5.5–7 s/step.
  - hejev + external: ~12 min per model. Retention screen: ~20 min.
  - About 1.5 h of GPU time was lost to the WF probe bug and the outage.
