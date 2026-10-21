# R056 results: thinking-teacher distillation into the single-pass student (R054 Phase 1, Track 1)

Plan: `docs/research/R054_PLAN_20261008.md` (Phase 1), cut to one seed and one arm. Runner: `scripts/r056_pod.sh`
(+ `r056_post.sh`). Data: `scripts/r053_generate_cases.py` (fixed filter, `--tag r056`), `scripts/r056_translate_twins.py`,
`scripts/r056_label.py`, `scripts/r056_build.py`. Status log: `.context/r056/STATUS.md`. Raw pod outputs:
`.context/r056/pod/`. Private backups: dataset `guychuk/typical-r056-data`, model `guychuk/typical-r056`.

**Verdict.** One fresh Gemma-4-12B-it LoRA, distilled at a real dose (42.9k presentations) from thinking-teacher
probabilities, scores **.649 on full WorkflowEvals** (705 cases, single forward pass per question): +12.8 over the frozen
base (.521), +12.1 over R044 (.528), +3.7 over its own Gemma-with-thinking teacher (.612), and 2.9 behind Jev 1.13 (.678).
Security goes from .333 to **.604** (Jev .617): the student's `is_true_positive` now sits in the .15–.75 band on 70% of
cases (gold 68%), where every earlier Typical was bimodal. Public JevBench stays at 196/231 (R044 196, Cygnet 203), so
the Phase-1 JevBench gate (≥199) is missed. Hebrew holds R055-level quality (external mean .803) and image retention
improves sharply (ScienceQA real 114/128 vs base 84, null recall 124/128).

## Results

Student: `typical:r056-sha256-4a727bab3a62` (step 1,052, adapter + `temperatures.json`; private model repo
`guychuk/typical-r056`). Served by `scripts/serve_typesafe_gemma_lora.py` (native option-letter readout, B=1) with
`--temperatures`; 2 replicas per GPU behind `scripts/r056_lb.py`. The id is `r056-sha256-<12 hex>` instead of
`r056-<sha8>` because the server requires a `-sha256-` suffix bound to the adapter tree (as R055).

### WorkflowEvals (full 705 cases; pinned WorkflowEvals 0ac3b8ad + typical/vllm patches, dataset revisions of `run_r050_official_eval.sh`; consensus labels; `typical:` provider, `--questions all`)

| model | security exact | trace primary | invoice exact | invoice primary | customer exact | **mean** |
|---|---|---|---|---|---|---|
| frozen Cygnet (Gemma single pass) | .333 | .523 | .531 | - | .698 | .521 |
| R044 | .338 | .550 | .518 | - | .708 | .528 |
| Gemma-4-12B-it + thinking (teacher, P0a) | .396 | .640 | .664 | .891 | .747 | .612 |
| Qwen3.8-27B + thinking (teacher, 235-case subset) | .487 | .724 | .740 | .780 | .784 | .684 (subset) |
| Jev 1.13 | .617 | .716 | .618 | - | .760 | .678 |
| **R056 student** | **.604** | **.635** | **.627** | .809 | **.729** | **.649** |

0 failed cases (240/111/150/204). R055-A/B were never run on WorkflowEvals.

Per question vs consensus (noul MAE / noul side flip / choice+score argmax flip):

| model | security | agent_trace | invoice | customer |
|---|---|---|---|---|
| frozen Cygnet | .253 / .292 / .492 | .150 / .178 / .357 | .177 / .066 / .178 | .049 / .049 / .163 |
| R044 | .274 / .287 / .479 | .149 / .172 / .297 | .089 / .060 / .174 | .048 / .041 / .205 |
| Gemma think (teacher) | .220 / .165 / .490 | .158 / .141 / .221 | .039 / .024 / .151 | .062 / .044 / .176 |
| **R056 student** | **.112 / .116** / .561 | **.128 / .150 / .185** | **.042 / .029** / .178 | .055 / .045 / .172 |

Security nouls are now closer to the consensus than either teacher measured in Phase 0 (Qwen-think .131 on its subset).
Security choice/score fields (attack_type, affected_scope, evidence_strength) did not improve (.561 flip); that and
agent_trace (.635 vs Jev .716) are the remaining gap.

### JevBench public 231 and Hebrew JevBench v0 (jevbench bb05a335, typesafe adapter, `scripts/hejev_score.py`)

| model | HE | MX | native | EN 231 | Belebele | HebNLI | G-MMLU | Sentiment | Winograd | HeQ | **ext. mean** | HE−EN | HE/EN agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R056 student** | .822 | .835 | .866 | .848 (196) | .911 | .825 | .613 | .814 | .727 | .930 | **.803** | −.026 | .896 |
| R055-A | .813 | .822 | .874 | .848 | .901 | .848 | .617 | .828 | .770 | .925 | .815 | −.035 | .883 |
| R055-B (twin JS .5) | .843 | .826 | .866 | .848 | .890 | .787 | .607 | .824 | .755 | .920 | .797 | −.004 | .926 |
| R044-R | .830 | .843 | .866 | .853 | .896 | .851 | .623 | .780 | .737 | .917 | .801 | −.022 | .913 |
| Cygnet (frozen Gemma, T=3.4) | .839 | .839 | .882 | .879 (203) | .899 | .743 | .647 | .760 | .745 | .910 | .784 | −.043 | .904 |
| Winnow-12B Q8_0 | .865 | .843 | .840 | .853 | .898 | .773 | .613 | .758 | .755 | .910 | .785 | +.013 | .939 |
| clef-flash | .774 | .752 | .782 | .823 | .888 | .846 | .728 | .640 | .806 | .895 | .801 | −.052 | .878 |
| StartLux-9B-Instant | .878 | .874 | .916 | .896 | .917 | .903 | .703 | .792 | .863 | .927 | .851 | −.017 | .935 |
| torchcast-12b | .865 | .865 | .899 | .879 | .900 | .791 | .630 | .786 | .773 | .920 | .800 | −.013 | .939 |

Gemma-think (teacher) scored 225/231 on EN. The student is level with R055/R044 on Hebrew and the external panel and
still 4–5 points behind StartLux on every column; EN 231 = 196, 7 below Cygnet. Twin consistency (JS .5) gave a smaller
HE−EN gap than R055-A (−.026 vs −.035) but not R055-B's −.004; this mix has ~25% Hebrew vs R055's 58%.

### Retention (`scripts/eval_gemma_mm_he.py`, 128 per suite; base recomputed in the same run)

| model | MASSIVE-he | MASSIVE-en | SQA real | shuffled | no-image | mismatch null-recall | gold-dropped null-recall | answerability AUROC |
|---|---|---|---|---|---|---|---|---|
| base Gemma-4-12B-it | 118 | 118 | 84 | 25 | 47 | 100 | 85 | .850 |
| R044-R | 120 | 120 | 103 | 27 | 65 | 98 | 94 | .914 |
| R055-A | 119 | 123 | 109 | 29 | 58 | 94 | 91 | .922 |
| **R056 student** | 116 | 119 | **114** | 2 | 54 | **124** | **113** | **.981** |

5% image replay with the R044 null-option recipe fixed R055's image regression: real-image accuracy +30 over base and
+11 over R044-R, null recall far above every earlier model; real − shuffled = .875 (images drive the answers).
MASSIVE-he is −2 vs base and −4 vs R044-R (HE/EN JSD +.004 vs base); the script's own gate (he errors lost ≤ 1)
fails by one item, all other gates pass.

### Phase-1 gate (plan, one seed)

| gate | result |
|---|---|
| ≥ +6 WF points over the frozen base | **+12.8** (.649 vs .521) — pass (one seed, not two) |
| security ≥ .50 | **.604** — pass |
| public JevBench ≥ 199/231 | 196 — **fail** |
| Hebrew/image within 2/128 of base | images far better; MASSIVE-he −2, MASSIVE-en +1 — pass (borderline on HE) |

### Checkpoint selection and calibration (dev only)

Dev = 115 held-out generated cases (2,855 node questions incl. HE twins), teacher targets. `scripts/r056_devsel.py`
(serving forward) on the last 500 steps; teacher cross-entropy / accuracy vs teacher argmax:

| step | 600 | 700 | 800 | 900 | 1000 | **1052** |
|---|---|---|---|---|---|---|
| dev teacher-NLL | .4446 | .4282 | .4231 | .4213 | .4180 | **.4178** |
| dev acc | .856 | .865 | .875 | .871 | .876 | .874 |
| security NLL | .571 | .549 | .536 | .540 | .533 | .534 |

Still falling at the end, so the last step was selected (a longer schedule would likely help a little). In-run step-0
(frozen base) dev teacher-CE was 1.56 on a 96-question sample. Per-question temperatures (`scripts/r056_calibrate.py`,
46 question ids with ≥30 dev rows; T in [0.60, 1.92]; `is_true_positive` T=1.04) moved dev CE only .4178 → .4122:
the distilled student is already teacher-calibrated, and post-hoc T is no longer the lever.

Dev teacher-NLL by generator (step 1052; style-leakage check requested by the coordinator):

| generator | all | security | agent_trace | customer |
|---|---|---|---|---|
| claude-haiku-5-5 (n=2,142) | .422 | .537 | .329 | .334 |
| gpt-5.6-terra (n=461) | .426 | .518 | .457 | .354 |
| Qwen3.8-27B (n=162) | .379 | - | .453 | .324 |
| Gemma-4-12B-it (n=90) | .354 | - | .354 | - |

No large generator gap; agent_trace from the minority generators is ~.12 worse than Haiku's (small n). HE twins .439 vs
EN .408.

## Cost and time

| item | |
|---|---|
| pod `jg0ump660t1swe`, 4×H100 80GB HBM3 SXM secure, $15.96/h | 07:34:23 → 17:07:09Z = 9.55 h ≈ **$152** (EXITED) |
| labeling (Gemma + Qwen think) | 07:58 → 10:34Z (~2.6 h, incl. ~1 h at low throughput before the shim fix) |
| training (1,052 steps) | 10:37 → 15:33Z (4.9 h, ~17 s/step) |
| dev selection + calibration | 15:33 → 16:00Z |
| eval (WF 705 + JevBench + hejev + external + retention) | 16:03 → 17:01Z |
| Claude CLI notional | generation $10.6 (Haiku) + twins $5.5 ≈ $16; GPT via codex subscription ($0) |

## Deviations

- One seed, one arm, no threshold loss (task spec). Plan's LoRA r64/lr 1e-4 replaced by the spec's r32, 5e-5 → 5e-6.
- Generator mix is 82% Haiku / 11% GPT / 7% open models, not the requested ~55/15/30 (open-model generation was starved by
  labeling on the same servers and cut at 09:40Z).
- Security targets are Qwen-only and customer/invoice targets Gemma-only (coordinator rules), so a Gemma-saturated
  teacher (97–100% exact 0/1 nouls) shapes customer and invoice; trace is the teacher mean. HE twins inherit the EN target.
- Served id format `r056-sha256-<12>` (server rule), not `r056-<sha8>`.
- The 3 h labeling box was used in full; the first hour ran at ~1/8 of Qwen's throughput because of the shim stall.
- 13:40–16:55Z the local session lost network; the unattended pod chain (`scripts/r056_post.sh`) completed selection,
  calibration, eval and the HF model backup without intervention.


## Data

**Contamination filter (fixed before generation).** `contamination_index` now uses 13-grams (was 8). N-grams, ID tokens
and name strings that occur in ≥5% of a workflow's benchmark cases are treated as template boilerplate. Generic field
names are no longer IDs: every JSON key in the benchmark, `key:` labels inside record strings, snake_case identifiers
(`card_last4`, `address_line1`), dotted event names without a digit or TLD (`user.session.start`), cloud regions
(`us-east-1`), stock binaries (`rundll32.exe`) and tokens shorter than 8 characters. R053's runs lost most cases to
exactly these hits; this run rejected 86 of ~1,560 parseable cases (5.5%; 83 ID hits, 2 13-gram hits, 1 name).
Invoice packets also get stray keys stripped before pydantic validation.

**Cases.** One structural skeleton + format notes per workflow, the R053 prompt plus a size hint (benchmark median
JSON length) and a "no conclusion-stating annotations" rule; 60% of seeds are ambiguous or mixed-signal by design.

| generator | security | agent_trace | customer | invoice | cases | share |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 (CLI, effort medium; notional $10.6) | 610 | 290 | 280 | 29 | 1,209 | 82% |
| gpt-5.6-terra (codex exec, effort low) | 59 | 67 | 37 | 5 | 168 | 11% |
| Qwen3.8-27B-FP8 (pod vLLM, no thinking) | - | 23 | 40 | - | 63 | 4% |
| Gemma-4-12B-it (pod vLLM, no thinking) | - | 30 | 4 | - | 34 | 2% |

1,474 cases → 27,222 node questions through WorkflowEvals' own adapters (5 stub-answer variants per case to reach
every branch). Security cases come only from Haiku/GPT (security labels are Qwen-only, so Qwen never labels its own
cases there). The coordinator's late request for ~30% open-model cases was not met: open-model generation shared the
vLLM servers with labeling and produced 97 cases before it was cut at 09:40Z.

**Hebrew twins.** 25% of security/agent_trace/customer cases (sha256(case_id) % 4 == 0): every node's state leaves,
question instructions and option texts translated by claude-haiku-5-5 with the R055 D1 prompt and deterministic
digit/option-ID QA (one fix retry): 1,339/1,341 nodes accepted (notional $5.5). Twins inherit the English node's teacher
target; `meta.twin_id` = English request id.

**Split.** dev = sha256(case_id) % 10 == 0, by case (all of a case's nodes and twins): 115 dev cases, 1,220 train.

## Teacher labels

Protocol = Phase 0's `vllm:` path (system_one probability prompt + schema, one question per call, xgrammar), thinking
on, called in-process (`scripts/r056_label.py`) instead of through the HTTP shim. 4×H100, one vLLM per GPU,
07:58–10:34Z.

| | Gemma-4-12B-it think | Qwen3.8-27B-FP8 think |
|---|---|---|
| questions labeled | 19,879 (all non-security + 2.1k security as reference) | 14,771 (all security + all agent_trace) |
| noul middle band [.15,.75] / exact 0 or 1, security | .004 / .974 (n=1,741) | .224 / .148 (n=7,357) |
| same, agent_trace | .002 / .997 | .047 / .783 |
| same, customer / invoice | .002 / .994, .000 / 1.000 | - |
| teacher-vs-teacher (both labeled) | security argmax flip .208, noul abs dp .209; agent_trace .168 / .108 | |

Soft target (coordinator rules): security = Qwen only (never Gemma); agent_trace = mean(Gemma, Qwen) (9 questions
Gemma-only); customer and invoice = Gemma, and any exact 0/1 Gemma distribution is smoothed to [.02, .98]
(p ← .96p + .04/K). **Final security noul targets: middle band 22.3%, exact 0/1 15.0% (n=6,752 train).**

Throughput note: the stdlib HTTP shim (`r054_systemone_server.py`) let each Qwen vLLM run only 3–9 requests while ~180
connections queued (the stall R055 also hit). Calling WorkflowEvals' client in-process lifted it to 64–68 running
requests and ~1.8k generated tok/s per GPU (Qwen averages ~3.4k output tokens per security call).

## Mix and training

`.context/r056/mix/manifest.json` (built on the pod; backed up to the private dataset). 42,883 presentations in
33,670 optimizer slots (twin groups share a slot):

| part | presentations | share |
|---|---|---|
| workflow nodes (security 10,872 / agent_trace 6,278 / customer 6,342 / invoice 2,327; EN + HE twins) | 25,819 | .602 |
| R055 Hebrew mix (D1 EN/HE/MX twins, open HE sets, Qwen-labeled native) | ~10.7k | ~.25 |
| R055 English replay (D1 source pools, R044 text) | ~4.3k | ~.10 |
| ScienceQA image replay (R044 recipe: 1,000 real / 500 mismatched image / 500 gold dropped, null option) | 2,000 | .047 |

Contamination: 13-gram check of every train/dev row against WorkflowEvals inputs (per-workflow ≥5% boilerplate
ignored; workflow rows are checked on their state, since the questions are WorkflowEvals' public schema, reused on
purpose), JevBench public 231, hejev-v0 and the external Hebrew panel: 0 rows dropped, assert passed. Train prompts
were verified byte-identical to the serving prompt (`render_typesafe_decision_prompt`) on 2,528/2,528 questions.

Training: fresh `google/gemma-4-12B-it`@707f0a3, LoRA r32/α64 on the language model, lr 5e-5 → 5e-6 cosine (3% warmup),
1,052 steps × 32 slots (torchrun ×4, microbatch 1 × accum 8 per GPU; LoRA gradients all-reduced), max length 16,384
(0 over-length slots skipped on rank 0), twin-consistency JS weight 0.5 (final twin JS .0012), seed 56, peak VRAM 48 GB/GPU,
4.9 h at ~16.8 s/step.

In-run eval (rank 0, 96 samples): wf_dev teacher-CE 1.564 → .432 (step 200) → .402 (step 400) → .373 (step 1052), acc vs
teacher argmax .854 → .948; he_dev (R055 dev) NLL 1.266 → .373, acc .813 → .865.
