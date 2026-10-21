# R055 results: Hebrew sprint, GPU half (2026-10-20/21)

Plan: `R055_HEBREW_SPRINT.md`. Runner: `scripts/r055_pod.sh`. Mix: `scripts/r055_build_mix.py`. All raw outputs are in
`.context/r055/` (`pod/hejev/<model>/<split>/` per-item rows, `pod/r055/` training logs, metrics, retention, served adapters).

**Verdict: not "best in Hebrew".** Neither arm beats every peer. StartLux-9B-Instant leads HE, MX, native and the
external mean. Torchcast-12b also beats both arms on MX and native. Arm A has the best external mean of the
remaining models (.815: above R044-R .801, clef .801, torchcast .800, Winnow .785 and Cygnet .784). Arm B has the
smallest HE−EN gap of all models (−.004). Both arms hold English at .848, which is within the 1-point guardrail of
R044-R's .853.

## 9–14B decision-model peer table (accuracy)

HE = `translated` (230), MX = `mixed` (230), native (119), EN = public 231. External = the eval-only panel in
`.context/hebrew-bench/external/` (README there). Gap and agreement are HE vs EN on the 230 twin pairs
(`scripts/hejev_score.py`). All rows use the same jevbench commit (bb05a335) and the typesafe adapter.

| model | HE | MX | native | EN 231 | Belebele | HebNLI | G-MMLU | Sentiment | Winograd | HeQ | **ext. mean** | HE−EN | HE/EN agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R055-A** (plain, step 2200) | .813 | .822 | .874 | .848 | .901 | .848 | .617 | .828 | .770 | .925 | **.815** | −.035 | .883 |
| **R055-B** (twin JS .5, step 1448) | .843 | .826 | .866 | .848 | .890 | .787 | .607 | .824 | .755 | .920 | **.797** | −.004 | .926 |
| R044-R | .830 | .843 | .866 | .853 | .896 | .851 | .623 | .780 | .737 | .917 | .801 | −.022 | .913 |
| Cygnet (frozen Gemma, T=3.4) | .839 | .839 | .882 | .879 | .899 | .743 | .647 | .760 | .745 | .910 | .784 | −.043 | .904 |
| Winnow-12B Q8_0 | .865 | .843 | .840 | .853 | .898 | .773 | .613 | .758 | .755 | .910 | .785 | +.013 | .939 |
| clef-flash | .774 | .752 | .782 | .823 | .888 | .846 | .728 | .640 | .806 | .895 | .801 | −.052 | .878 |
| StartLux-9B-Instant | **.878** | **.874** | **.916** | **.896** | **.917** | **.903** | .703 | .792 | **.863** | **.927** | **.851** | −.017 | .935 |
| torchcast-12b | .865 | .865 | .899 | .879 | .900 | .791 | .630 | .786 | .773 | .920 | .800 | −.013 | .939 |

Where the arms fall short of "best in Hebrew" (rule: beat every peer on HE, MX, native and the external mean,
with EN ≥ R044-R − 1 pt):
- **HE:** B .843 and A .813 are below StartLux (.878), Winnow (.865) and torchcast (.865). A is also below Cygnet and R044-R.
- **MX:** B .826 and A .822 are below every peer except clef.
- **Native:** A .874 and B .866 are below StartLux (.916), torchcast (.899) and Cygnet (.882).
- **External mean:** A .815 and B .797 are below StartLux (.851). A ranks second overall. It wins on Sentiment
  (.828, the best of all models) and is near the top on HeQ and Belebele. Both arms lose most on Global-MMLU,
  where clef leads (.728) and StartLux follows (.703).
- **EN guardrail:** passed. Both arms score .848 against R044-R's .853 (−0.4 pt).

Peer hejev numbers (HE/MX/native/EN) are the earlier same-protocol runs in `.context/hebrew-bench/runs/<model>/score.json`.
Peer external numbers and both arms' numbers all come from this pod.

## Retention: MASSIVE he/en and ScienceQA images (`scripts/eval_gemma_mm_he.py`, 128 per suite)

Accuracy shown as correct/128. "Base" is frozen Gemma-4-12B-it (recomputed in each run and identical across runs).
R044-R comes from `.context/r044-rebuild/mmhe.json`.

| model | MASSIVE-he | MASSIVE-en | HE/EN JSD | SQA real | shuffled | no-image | mismatch null-recall | gold-dropped null-recall | answerability AUROC |
|---|---|---|---|---|---|---|---|---|---|
| base Gemma-4-12B-it | 118 | 118 | .026 | 85 | 26 | 46 | 99 | 84 | .850 |
| R044-R | 120 | 120 | .018 | 103 | 27 | 65 | 98 | 94 | .914 |
| R055-A | 119 | 123 | .032 | 109 | 29 | 58 | 94 | 91 | .922 |
| R055-B | 122 | 123 | .019 | 95 | 45 | 78 | 78 | 87 | .872 |

- **Text retention:** passed. MASSIVE-he/en do not lose (≤ 2/128) vs R044-R; A is 119/123 (−1/+3 vs R044-R) and B is 122/123 (+2/+3).
- **Image retention:** fails vs R044-R. B answers 8 fewer real-image items (95 vs 103) and loses 20 mismatch
  null-recalls (78 vs 98). Both are beyond the 2/128 guardrail. B's real−shuffled margin is +.39 vs R044-R's
  +.59, so the images still drive answers. The likely cause is image replay at 200 of ~25k rows (0.8%), against
  25% in R044. A: A has the best real-image accuracy (109 vs 103) and AUROC (.922 vs .914). It still fails the null-recall guardrail narrowly: mismatch −4 (94 vs 98) and gold-dropped −3 (91 vs 94). A's HE/EN JSD is higher (.032 vs .018), which is consistent with its larger HE−EN gap on hejev. The image guardrail fails for both arms, so the image-replay share needs raising before either arm is promoted.

## Training

- **Base:** fresh `google/gemma-4-12B-it`@707f0a3 LoRA, with no `--init-adapter`. Settings: r32/α64, lr 5e-5→5e-6
  cosine, microbatch 1 × accum 8, max length 8192, wd .01, seed 55. Evaluated every 250 steps (256 samples each),
  saved every 500.
- **Mix:** `.context/r055/mix/manifest.json` has the sha256 of every input and output.
  - Train rows: 24,950 (en 10,846 / he 9,658 / mx 4,446), and 58.4% of presentations are HE-or-MX.
  - D1 twins: 13,338 rows from the post-fix D1, with train sha 4f27e1ac….
  - D2 open: 4,000, balanced 1,334/1,333/1,333 across MASSIVE-he, HeQ and HebrewSentiment.
  - Qwen-labeled native: 1,112 cases.
  - EN replay: 6,300 rows, 300 of them R044 text (200 tasksource replay + 100 MASSIVE-he). The other 6,000 are D1
    source-pool rows not used by D1, round-robin over wf, wh, u, kb and s2_policy.
  - ScienceQA image rows: 200.
  - EN replay was sized up from ~2k so the mix moves toward the ~50% target, capped at 6k. It ends at 58%, not 50%.
- **Dev for selection:** `dev_he` (660 rows) is the D1 dev HE+MX rows plus 118 held-out native cases (10% by case).
  `dev_en` is 271 D1-dev EN rows. Each arm's checkpoint was chosen by `he_dev` NLL among the saved steps; no test
  sets were used.
  - A, step 2200: he_dev NLL .314, acc .922.
  - B, step 1448: he_dev NLL .372, acc .879.
  - The validation loss had not flattened for either arm, so both picked their last step.
- **Contamination:** before training, a 13-gram check against hejev-v0 (all 4 files) and the 6 external files,
  using the same boilerplate rule as the D2 decon. It dropped 0 rows, and the hard assert passed.
- **Native labels:**
  - Labeler: Qwen3.8-27B-FP8 thinking on vLLM 0.31, `--reasoning-parser qwen3`. It used the WorkflowEvals
    system_one probability prompt with the schema appended (the Phase-0 protocol), one question per request, one
    vLLM per GPU.
  - Coverage: 2,545 of 2,558 questions were labeled. The 13 left were stopped at the time box.
  - Agreement with the generator's `meta.intended` is **95.25%** (2,424/2,545). Questions that disagreed were
    dropped, leaving 1,230 cases with ≥1 kept question.
  - Wall time was ~55 min. The first run hung after 12 min: one shim's threads stalled with 192 in-flight requests
    and the vLLM sockets in CLOSE-WAIT. It was relaunched with 128 workers; answered questions were kept.
- **Steps (matched presentations):**
  - A: 2,200 steps = 17,600 presentations.
  - B: 1,448 steps × 8 slots × 1.519 views/slot = 17,594 presentations.
  - Both are 0.88× the 20k target, cut so both would finish near 03:30.
  - A took 4.37 h (≈6.1 s/step on A100). B took 3.32 h (≈7.9 s/step), peak VRAM 30/42 GB.
  - B's twin JS fell to .0056 by the end.
- **Restart (deviation):** the first launch crashed B at step ~420. A D1 `s2_policy` HE/MX row tokenized to 9,991
  Gemma tokens, above the 8,192 limit (chars/4 underestimates Hebrew). A would have hit the same rows.
  - Both arms were restarted from scratch.
  - The 5 over-length twin groups (10 HE/MX rows plus their EN twins) were dropped with `--drop-ids`, and the
    steps were reduced as above.
  - First-run logs are in `.context/r055/pod/r055/arm-*.crashrun1/`.

## Compute and cost

- Pod `stakepm0cw4tg0`: 2× **A100-SXM4 80GB**, because 2× H100 HBM3 was unavailable in both secure and community
  cloud. $3.58/h, secure cloud.
- Up 20:58:34 → 03:58:52 UTC = 7.0 h. A second session ran 03:59:19 → 04:15:38 UTC = 0.27 h. The first stop
  had interrupted A's retention screen, and stop wipes this volume-less pod, so the session re-staged Gemma and
  re-ran only that screen. Total **≈14.6 GPU-h, ≈$26.0**. Both sessions ended EXITED.
- Rough split: setup ~15 min; labeling ~55 min; training 4.4 h (A) and 3.3 h (B). The peers' external runs were
  co-located with training (R044-R, clef and Winnow) or ran after B (Cygnet, StartLux, torchcast). Evaluation took
  ~20 min per arm.

## Deviations and notes

- **Early pod.** The pod was created at 20:58 UTC, before D1 was built, once native + open + external existed and
  the peers pod had EXITED. D1 is only needed at mix build, so this took about an hour off the critical path.
- **Served ids.** The served ids are `typical-r055-a-sha256-10c880b02a85` and `typical-r055-b-sha256-c54ad4caca12`.
  `serve_typesafe_gemma_lora.py` requires a `-sha256-<≥12 hex>` suffix bound to the adapter tree, so
  `typical:r055-a-<sha8>` could not be used. Each adapter is served as a 2-file tree (config + weights) at
  `.context/r055/pod/r055/serve-{a,b}/`.
- **Peers on the external panel:** all six were run here (the peers pod covered hejev only).
  - Winnow's server returned HTTP 429 "Decision queue is full" when about 300 chunks arrived at once. Its failed
    chunks (HeQ and part of Winograd) were rerun whole in a third pass.
  - StartLux's launcher pins `--gpus 0`. It was remapped to GPU1, because its first try landed on the training GPU
    and failed vLLM's free-memory check.
  - Clef ran without the flash-linear-attention fast path (torch fallback). This affects speed only.
  - R044-R, clef and Winnow shared a GPU with training. This affects only speed, not outputs.
- **D1 fix.** The coordinator's D1 fix (78 garbled-option twins removed at 21:50Z) was in place before the final mix
  was built. The mix manifest records the post-fix sha256 values.
