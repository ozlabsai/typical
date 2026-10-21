# R061: fast/deliberate dual mode (offline estimate)

Idea: serve R056 single pass by default (fast mode, also used for JevBench). When R056's own answer is uncertain,
answer that question with Gemma-4-12B-it + thinking instead (deliberate mode). WorkflowEvals scores accuracy only, so
routing a minority of questions might close the gap from R056 (.649) to Jev 1.13 (.678).

> **BENCHMARK DIAGNOSTIC.** Every k below is swept on WorkflowEvals labels, so no row is a model-selection result.
> A real router's k (or entropy threshold) has to be fixed on synthetic dev data.

Script: `scripts/r061_route_offline.py`. Outputs: `.context/r061-route-offline/{results.json,README.md}`, with all
three signals, both modes and the full subset tables. Runs on CPU in about 25 min.

## Verdict

**The step-1 gate (≥ +2 pts at some k ≤ 30%) is missed under the rule as specified** (unseen branches count as
wrong).
- On full WorkflowEvals, routing to Gemma-think is best at k = 5%: **.6616 (+1.3)**. No other cell beats +1.3 (3
  signals × 2 modes × 4 k ≤ 30).
- Unseen branches decide the call. Re-scoring each errored case with its own R056-alone or Gemma-think-alone outcome
  gives **+2.3 / +2.4 at k = 5%** (.671 / .673). That is still below Jev's .678. Only one cell reaches .683, and it
  is the maximum over 30 cells.
- So the effect is borderline: about +1 to +2.5 points, all from k ≈ 5%. Step 2 was not run (see the end of this doc
  for what would settle it).

Where the gain comes from (entropy, replace, k = 5%):

| workflow | change |
|---|---|
| invoice | +5.5 (.627 → .682), above both R056 and Gemma-think (.664) |
| security | +0.4 |
| trace | 0 |
| customer | −0.8 |

Routing more than ~5% hurts security fast: .604 → .558 at 10%, → .425 at 30%. R056's calibrated mid-band security
nouls (70% in [.15, .75]) get replaced by Gemma-think's polar ones (97% exact 0/1 in R056 labeling). On security,
R056's uncertainty marks questions where it is already better than the thinking model.

**Qwen3.8-27B-think, 235-case subset.** The rows clear +2 even with unseen = wrong: entropy +2.1 to +2.5 at k = 5%,
best cell margin/average/20 +3.7. These gains are not evidence for Qwen:
- Gemma-think as the deliberate mode on the same subset gets +3.3 at entropy/replace/5, but only +1.3 on the full set.
  The subset inflates routing gains by about 2 points.
- Qwen is not clearly better than Gemma here.
- Qwen is also a second weight set and about 4× slower.

## Method

- **Answers.** Fast: R056 on full WorkflowEvals (`.context/r056/pod/wf/<wf>/results.json`, `typical:` provider,
  `--questions all`). Deliberate: Gemma-4-12B-it think, full set (`.context/r054-phase0/gemma4-12b-think`, one
  question per call), and Qwen3.8-27B-FP8 think on its 235-case subset (`.context/r054-phase0/qwen38-27b-think`).
- **Replay.** Answers are keyed by (node, state fingerprint, question id). The official programs are replayed
  offline with `r053_calibration_headroom.execute_cases/score_cases` and scored against labels-consensus with
  `core.scoring`. Headline metrics: security exact, trace primary, invoice exact, customer exact; the mean is the
  equal-workflow mean.
- **Sanity.** k = 0 reproduces R056's .6487 and k = 100 (replace) reproduces Gemma-think's .6117. On the subset,
  R056 is .6702, and Qwen and Gemma reproduce .684 / .664. The saved runs omit `legend` on score answers, and the
  loader restores it as `{}`.
- **Uncertainty signals,** computed from R056's own answer only:
  - `entropy`: normalized entropy H/log K. This is the primary signal, fixed in advance from the replay-gate idea in
    `.context/novel-ideas/neuro_bio.md`.
  - `margin`: 1 − (p1 − p2).
  - `threshold`: closeness of the noul to a program gate; choice and score questions fall back to margin. The gates
    are .15/.60/.75 for `is_true_positive` and .5 for the other nouls. The per-policy floors (.6/.75/.85) are not
    encoded.
- **Routing.** Within each workflow, rank all questions R056 answered and route the top k% (k ∈ {0, 5, 10, 20, 30, 50,
  100}, fixed in advance).
  - *Replace* uses the thinking answer.
  - *Average* uses the mean distribution of the two answers; choice, score and confidence are rebuilt from it.
- **Unseen branches.** A routed answer can send the program to a state that neither saved run visited. A routed
  question can also have no thinking answer at a state R056 visited (18 cases at k = 5%, 62 at 30%). Both count as
  wrong in the main column. The bracket column re-scores those cases with their R056-alone or thinking-alone
  decisions.

## Results: full WorkflowEvals (705 cases), R056 → Gemma-4-12B think, entropy signal

| mode | k | routed % | security | trace | invoice | customer | **mean** (unseen = wrong) | case errors | mean, errored → R056-alone / think-alone | think tok / case |
|---|---|---|---|---|---|---|---|---|---|---|
| — | 0 | 0.0 | 0.604 | 0.635 | 0.627 | 0.729 | **0.6487** | 0 | 0.6487 / 0.6487 | 0 |
| replace | 5 | 5.0 | 0.608 | 0.635 | 0.682 | 0.721 | **0.6616** | 18 | 0.6714 / 0.6726 | 1592 |
| replace | 10 | 10.0 | 0.558 | 0.604 | 0.687 | 0.712 | **0.6403** | 35 | 0.6648 / 0.6634 | 3274 |
| replace | 20 | 20.0 | 0.496 | 0.595 | 0.704 | 0.724 | **0.6297** | 51 | 0.6638 / 0.6492 | 6091 |
| replace | 30 | 29.8 | 0.425 | 0.604 | 0.682 | 0.727 | **0.6095** | 62 | 0.6516 / 0.6276 | 8404 |
| replace | 50 | 49.7 | 0.375 | 0.586 | 0.680 | 0.730 | **0.5927** | 80 | 0.6444 / 0.6206 | 12019 |
| replace | 100 | 100.0 | 0.396 | 0.640 | 0.664 | 0.747 | **0.6117** | 0 | 0.6117 / 0.6117 | 18518 |
| average | 5 | 5.0 | 0.621 | 0.635 | 0.682 | 0.704 | **0.6606** | 18 | 0.6719 / 0.6741 | 1588 |
| average | 10 | 10.0 | 0.608 | 0.604 | 0.682 | 0.706 | **0.6500** | 28 | 0.6705 / 0.6733 | 3269 |
| average | 20 | 20.0 | 0.583 | 0.595 | 0.707 | 0.721 | **0.6513** | 42 | 0.6825 / 0.6739 | 6087 |
| average | 30 | 29.8 | 0.521 | 0.595 | 0.684 | 0.727 | **0.6317** | 53 | 0.6695 / 0.6560 | 8398 |
| average | 50 | 49.8 | 0.483 | 0.577 | 0.682 | 0.725 | **0.6169** | 73 | 0.6668 / 0.6521 | 12011 |
| average | 100 | 99.5 | 0.475 | 0.568 | 0.667 | 0.725 | **0.6087** | 72 | 0.6576 / 0.6473 | 17517 |

Best cell at k ≤ 30 for the secondary signals (gain over .6487; unseen = wrong, then the bracket):

| signal | best cell | gain, unseen = wrong | gain, bracket (→ R056-alone / → think-alone) |
|---|---|---|---|
| margin | average, 5% | +1.3 | +1.8 / +2.0 |
| threshold | average, 5% | +0.6 | +1.8 / +1.9 |

## Results: Qwen subset (235 cases*), R056 → Qwen3.8-27B think, entropy signal

| mode | k | routed % | security | trace | invoice | customer | **mean** (unseen = wrong) | case errors | mean, errored → R056-alone / think-alone | think tok / case |
|---|---|---|---|---|---|---|---|---|---|---|
| — | 0 | 0.0 | 0.613 | 0.658 | 0.607 | 0.804 | **0.6702** | 0 | 0.6702 / 0.6702 | 0 |
| replace | 5 | 5.0 | 0.613 | 0.671 | 0.707 | 0.775 | **0.6912** | 13 | 0.7019 / 0.7128 | 2541 |
| replace | 10 | 10.0 | 0.588 | 0.645 | 0.720 | 0.779 | **0.6829** | 19 | 0.7066 / 0.7177 | 4781 |
| replace | 20 | 19.9 | 0.575 | 0.632 | 0.753 | 0.775 | **0.6836** | 21 | 0.7117 / 0.7159 | 7825 |
| replace | 30 | 29.8 | 0.463 | 0.605 | 0.760 | 0.789 | **0.6542** | 28 | 0.6880 / 0.6897 | 10262 |
| replace | 100 | 99.8 | 0.487 | 0.724 | 0.740 | 0.784 | **0.6839** | 9† | 0.6955 / 0.6839 | 20071 |
| average | 5 | 5.0 | 0.625 | 0.671 | 0.707 | 0.779 | **0.6955** | 12 | 0.7062 / 0.7172 | 2541 |
| average | 10 | 10.0 | 0.613 | 0.645 | 0.720 | 0.784 | **0.6904** | 16 | 0.7110 / 0.7252 | 4781 |
| average | 20 | 19.9 | 0.588 | 0.658 | 0.740 | 0.784 | **0.6924** | 18 | 0.7198 / 0.7260 | 7825 |
| average | 30 | 29.9 | 0.562 | 0.658 | 0.740 | 0.814 | **0.6935** | 22 | 0.7223 / 0.7229 | 10262 |

\* 80/38/50/68 = 236 cases: the Qwen run's own case list. PHASE0 mentions both 37 and 38 trace cases.
† Qwen's own 9 invoice failures (16k output cap).

The paired Gemma-think-on-subset table is in `.context/r061-route-offline/README.md`. Its best cells are
entropy/replace/5 +3.3 and margin/replace/10 +3.5, the same size as Qwen's.

## Cost of the deliberate path (Gemma-think, entropy, full set)

Fast mode generates no tokens: it is one forward pass per node with the native option-letter readout. JevBench stays
in fast mode (196/231, unchanged). Deliberate-mode token counts come from the P0a run, one question per call:

| k | cases with ≥1 routed question | think tokens / routed question | think tokens per touched case, p50 / p90 | avg think tokens / case |
|---|---|---|---|---|
| 5% | 48% | 1,807 | 1,749 / 9,137 | 1,592 |
| 10% | 69% | 1,841 | 2,347 / 12,624 | 3,274 |
| 20% | 86% | 1,704 | 3,376 / 20,163 | 6,091 |
| 30% | 95% | 1,568 | 4,503 / 24,638 | 8,404 |

All-thinking costs 18.5k tokens per case (13.06M over 13,377 calls).

Latency, at an assumed ~75 tok/s single-stream decode for Gemma-12B on one H100 (not measured here):
- One routed question adds about 24 s.
- At k = 5%, half the cases see no added latency. The median touched case gains about 23 s and p90 about 2 min, if
  questions are sequential within a case.

Throughput: batched decode was about 2k tok/s in the P0a vLLM log. The k = 5% deliberate load for the whole benchmark
(1.1M tokens) is about 10 GPU-min.

## Step 2: not run

The gate is missed as specified, so no dev-split router and no full serving plan were written. Facts that matter if
the result is overridden:

1. **The end-to-end run is what settles it.** The open question is the 18 unseen-branch cases at k = 5%, and only a
   live run answers them. One H100 at $3.49/h:
   - setup: ~0.5 h;
   - R056 fast pass over WF: ~71M logged prompt tokens, about 1 h;
   - Gemma-think on the routed ~5–10% (1.1–2.3M output tokens): ~0.2–0.35 h;
   - a k = 0 parity check against .649.
   - **Total about 2–2.5 GPU-h, ≈ $8–10 ($15 with slack).**
2. **One weight set works on our stack.** vLLM 0.30.0 already served Gemma-4-12B@707f0a3 with r32 LoRAs
   (`scripts/r055_readout_pod.sh`, `--enable-lora --max-lora-rank 32`). Fast = request with `model=r056`
   (letter-logprob readout). Deliberate = the base model name with `enable_thinking` (the `vllm:` provider path).
   Adapter + thinking is a per-request option.
   - Caveat: R056's .649 was measured with the native HF readout plus `temperatures.json`. A vLLM letter readout
     needs the k = 0 parity check (R055 found the two readouts within noise).
3. **The proposed dev criterion is partly circular.** R056 dev teacher targets are Gemma-think for customer and
   invoice, and the Gemma/Qwen mean for trace. "Agreement with teacher labels when routing to Gemma-think" therefore
   rewards routing everything on those workflows.
   - Non-circular options: Qwen-only security targets; the R059 gold-style astra+fable labels (392 security
     questions); or astra on the 2,855 dev rows (~$110 at $0.038 per question, less if HE twins inherit).
   - Gemma-think labels already exist for 1,320 of the 2,855 dev rows (`.context/r056/pod/labels_gemma.jsonl`). R056
     dev answers are in `pod/devsel/step-1052.json`, so most of the dev selection is CPU-only. The 1,535 rows
     without Gemma labels (mostly security) need about 2.8M think tokens, roughly 25 GPU-min.
4. **The only clear lever is invoice,** where Gemma-think beats R056 (.664 vs .627 in R054/R056). A router keyed on
   question type would need non-benchmark evidence. Distilling more invoice thinking into the single pass (the R056
   recipe) avoids serve-time thinking entirely.
