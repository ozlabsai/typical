# R062 results: Hebrew thinking-teacher distillation (R056 recipe on a Hebrew pool)

Runner: `scripts/r062_pod.sh` (one on-pod chain: label → mix → train → select → eval → HF → self-stop). Data:
`scripts/r062_data.py` (translate / labelin / prep / mix). Status log and raw outputs: `.context/r062/` (STATUS.md,
`hf/results/`, `pod/`). Private backups: model `guychuk/typical-r062` (adapter, temperatures, `devsel/`, `results/`),
data `guychuk/typical-hebrew-train` `r062/` (pool, label input, Qwen labels, static anchors/images, mix).

**Verdict: not best in Hebrew.** One fresh Gemma-4-12B-it LoRA, distilled from Qwen3.8-27B-thinking soft labels, gets
**MX .870**. That is the best mixed-language score of any Typical and is within .004 of StartLux (.874). HE (.852)
equals the frozen base and R056-λ.5. EN 231 holds at 200/231. Native (.840) and the external panel (.796) fall below
R056-λ.5 (.899 / .809). StartLux stays ahead on HE (−2.6), native (−7.6) and external (−5.5). Torchcast also leads on
HE (−1.3) and native (−5.9). Both all-in-one merges are dominated by R062 on Hebrew. R062+.5·R056 has the best
contested score of any Typical (.603), but no merge WorkflowEvals run was done (see Deviations). Every retention
gate passes, and image null-recall is far above the base.

## Results (served id `typical:r062-sha256-79401fd95f20`, step 573; jevbench bb05a335 typesafe; `hejev_score.py`)

| model | HE | MX | native | EN 231 | ext. mean | contested | HE−EN | HE/EN agree |
|---|---|---|---|---|---|---|---|---|
| **R062** | **.852** | **.870** | .840 | .866 (200) | .796 | .582 | −.017 | .887 |
| R062 + .5·R056 (merge) | .848 | .839 | .824 | .861 (199) | .797 | **.603** | −.013 | .909 |
| R062 + .5·R060 (merge) | .848 | .848 | .824 | .861 (199) | .793 | .584 | −.013 | .922 |
| R056 λ=.5 (best Typical before) | .852 | .843 | .899 | .874 (202) | .809 | .583 | −.022 | - |
| R060 | .826 | .817 | .849 | .857 (198) | .800 | .594 | −.030 | .896 |
| frozen base (native readout) | .852 | .835 | .815 | .874 (202) | .790 | .550 | −.022 | - |
| StartLux-9B-Instant | **.878** | **.874** | **.916** | **.896** | **.851** | .601 | −.017 | .935 |
| torchcast-12b | .865 | .865 | .899 | .879 | .800 | - | −.013 | .939 |

External panel, R062 / +.5·R056 / +.5·R060:
- belebele .894 / .891 / .894
- hebnli .795 / .800 / .808
- gmmlu .615 / .612 / .597
- sentiment .818 / .834 / .830
- winograd .723 / .734 / .712
- heq .932 / .910 / .917

R062's HE Brier is .210 (R044-R .241, StartLux .163).

**Retention** (`eval_gemma_mm_he.py`, 128 per suite; base recomputed in the same run):

| model | MASSIVE he/en | SQA real | shuffled | mismatch / gold-dropped null recall | AUROC | gates |
|---|---|---|---|---|---|---|
| base | 118 / 118 | 85 | 26 | 99 / 84 | .850 | |
| R062 | 118 / 120 | 110 | 3 | 124 / 119 | .985 | all 10 pass |
| R062 + .5·R056 | 118 / 120 | 111 | | 123 / 117 | .984 | pass |
| R062 + .5·R060 | 117 / 120 | 111 | | 125 / 119 | .987 | pass |

## Data

- **Pool.** 61,798 questions after decontamination: D1 twins 3 × 4,722; tasksource HE/MX/EN 3 × 14,235; native 2,558;
  open 2,369.
  - Decontamination used 13-gram + MinHash (char-5gram Jaccard > .5) against public JevBench (all of `datasets/`),
    the WorkflowEvals cases, hejev-v0, the external panel and contested-fast-decisions.
  - It dropped 631 groups, all omilab sentiment rows near-duplicating the external sentiment test.
- **Tasksource translation.** 12,000 English rows from 289 task families, round-robin by family, with choice, noul,
  score and procedural items, no wili rows, and ≤ 4k tokens each.
  - Translator: claude-haiku-5-5 CLI, 16 workers, sha256 cache. Notional cost **$18.4** against a $40 cap.
  - QA: deterministic checks on digits, snake_case IDs, option keys and target-in-options, with one fix retry.
  - Outcome: 11,952 accepted, 42 det_fail, 6 errors.
  - EN/HE/MX twins share `meta.twin_id`.
- **Option-key bug fixed in code.** `r055_translate_pool.from_legacy` now rejects out-of-range labels. Before the fix,
  label −1 picked `keys[-1]`.
- **Teacher.** Qwen3.8-27B-FP8 with thinking, vLLM 0.31, one instance per A100, in-process `r056_label.py`, system_one
  protocol, one question per call.
  - Labelled 27,184 answers in a 3 h box (2.5–2.9 q/s), with 0 errors at steady state.
  - The 2,545 R055 native labels (same teacher and protocol) were reused.
  - Soft target = Qwen's distribution; exact 0/1 is smoothed (p ← .96p + .04/K).
  - Teacher argmax agrees with the source gold on 83.4% (native intended labels: 95%).
- **Mix.** 31,434 presentations in 18,315 slots:
  - HE/MX .567: tasksource 7.9k, D1 5.2k, native 2.4k, open 2.3k
  - EN twins .21
  - EN tasksource anchors .167: 5,250 rows, disjoint from all 12k translation candidates and their groups, with
    `meta.base_probs`
  - ScienceQA images .056
  - Every prompt is rendered exactly as the server renders it (`r056_build.trainer_question`).
  - Dev: 879 HE/MX questions, held out 5% by twin_id/case.

## Training and selection

- **Training.** Fresh `google/gemma-4-12B-it`@707f0a3 LoRA r32/α64, lr 5e-5 → 5e-6 cosine, 573 steps × 32 slots
  (torchrun × 4, microbatch 1 × accum 8), max length 8192, twin JS .5 (final .0055), KL-to-base 1.0 on anchors from
  step 0, seed 62. It took 2.0 h at ~12.4 s/step on 4×A100, peak 41 GB.
- **In-run Hebrew dev NLL** (128 samples): 1.364 at step 0, .530 at 100, .491 at 300, .494 at 573.
- **Selection.** Hebrew dev teacher-NLL over the last third (879 questions):

  | step | 400 | 450 | 500 | 550 | 573 |
  |---|---|---|---|---|---|
  | dev NLL | .5553 | .5583 | .5526 | .5538 | **.5505** |

  Step 573 was selected, with accuracy vs teacher argmax .844. The curve is flat after ~step 300.
- **Temperatures.** Per-type temperatures (the server's `type:` key) were fitted on dev: choice 1.02, noul 1.04. Dev
  CE moved only from .5505 to .5497.

## Cost and time

- Pod `qr2izzf6nl9q9r`, 4×A100-SXM4-80GB secure, $7.16/h (no 4×H100 was available). It ran 12:23 → 18:37Z, ≈6.2 h,
  ≈**$45**, and self-stopped (EXITED).
- Wall time: labelling 12:43 → 15:35Z; mix and base probs 15:35 → 15:38Z; training 15:38 → 17:37Z; evals
  18:11 → 18:36Z.
- Notional Haiku spend: $18.4.

## Deviations

1. **Partial labelling.** Not every pool question was labelled: 27k of 62k, in a 3 h box. Labelling everything would
   have taken ~5.7 h on A100. The box still filled the planned mix shares, at 31k presentations instead of 35k.
2. **Hardware.** 4×A100 instead of 4×H100 (none available).
3. **Fork storm on the first launch.** My chain's `setup()` overrode the phase-0 `setup` that `rsetup` calls, which
   recursed. It was killed within 10 min and the function renamed to `r62setup`.
4. **Post-eval crash at 17:42Z.** An edit commented out the served-id write, so `hev` died on `set -u`. The pod idled
   ~26 min (my session was offline). The eval was resumed with a `set +eu` / trap-finish entry, without retraining.
5. **No merge WorkflowEvals.** The on-pod `bench` step failed (bench.md is empty), so the automatic merge pick was
   empty. The chain correctly skipped the WF run and self-stopped, which also wiped the volume-less pod.
   - Scored locally, the better merge is R062 + .5·R056: mean(HE, MX, native, EN, contested) .795 vs .793.
   - Its full WorkflowEvals (~2 h on 4×A100, ~$15) still needs a pod. Both adapters are on HF, and the merge is
     deterministic (`r060pre_proxy.py merge`, temperatures in `results/r062mr056/`).
6. **Pod chain changes.** On-pod self-stop (`runpodctl stop pod` from inside, env read from PID 1). `costguard.sh` now
   also honours a remote `/workspace/g1/SAFE_TO_STOP`.
