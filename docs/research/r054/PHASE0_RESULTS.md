# R054 Phase 0 — teacher measurements (no training)

Plan: `docs/research/R054_PLAN_20261008.md`. Scoring: WorkflowEvals' own `scores.json` against `labels-consensus`
(headline = security exact / agent_trace primary / invoice exact / customer exact, equal-workflow mean);
per-question gaps reuse `.context/research-review/leaderboard_questions.py` logic (`scripts/r054_score.py`).
Raw runs: `<run>/<workflow>/{results.json,scores.json,cases/}`, `<run>/jevbench/*.jsonl`, logs alongside.

## Results (WorkflowEvals consensus; JevBench public 231)

Full 705 cases unless marked. Jev/231 = argmax hits on the public easy+original+hard files.

| run | security exact | trace primary | invoice exact | invoice primary | customer exact | **WF mean** | Jev/231 |
|---|---|---|---|---|---|---|---|
| frozen Cygnet (ref) | .333 | .523 | .531 | - | .698 | .521 | 203 |
| R044 (ref) | .338 | .550 | .518 | - | .708 | .528 | 196 |
| Jev 1.13 (ref) | .617 | .716 | .618 | - | .760 | .678 | - |
| **P0a** Gemma-4-12B-it think (`vllm:google/gemma-4-12B-it`) | .396 | .640 | .664 | .891 | .747 | **.612** | **225** (2 failed) |
| **P0b** Qwen3.8-27B-FP8 think xhigh, **235-case subset** | .487 | .724 | .740* | .780* | .784 | **.684** (subset) | **229** |
| **P0c** LiquidAI d1-3B native (`typical:d1-3B-051bcc4`) | .550 | .536 | .316 | .669 | .484 | **.471** | 174 |

\* 9/50 Qwen invoice cases failed because a question hit the 16,384-token output cap. They count as wrong, so invoice is a lower bound.

**Paired, on the same 235-case subset as P0b** (first ceil(n/3) cases per workflow in case-ID order: 80/38/50/68):

| run | security | trace | invoice | customer | mean |
|---|---|---|---|---|---|
| Qwen3.8-27B think | .487 | .724 | .740 | .784 | **.684** |
| Jev 1.13 (published run, rescored) | .562 | .684 | .647 | .848 | **.685** |
| Gemma-4-12B think | .463 | .671 | .747 | .775 | **.664** |
| frozen Cygnet | .325 | .632 | .553 | .784 | .574 |
| R044 | .275 | .605 | .580 | .770 | .557 |
| d1-3B | .562 | .513 | .320 | .539 | .484 |

The subset is easier than the full set for most models (subset minus full: Cygnet +.053, Gemma +.052, R044 +.029, Jev +.007).
Shifting Qwen by that range gives a full-set estimate of **~.63-.68**.

**Per question vs consensus** (noul MAE / noul side flip / choice+score argmax flip; security, trace, invoice, customer):

| run | security | trace | invoice | customer |
|---|---|---|---|---|
| Gemma think | .220/.165/.490 | .158/.141/.221 | .039/.024/.151 | .062/.044/.176 |
| Qwen think (subset) | .131/.166/.533 | .096/.083/.186 | .027/.013/.088 | .041/.028/.126 |
| d1-3B | .234/.331/.488 | .199/.214/.444 | .299/.249/.295 | .086/.087/.289 |
| frozen Cygnet | .253/.292/.492 | .150/.178/.357 | .177/.066/.178 | .049/.049/.163 |
| R044 | .274/.287/.479 | .149/.172/.297 | .089/.060/.174 | .048/.041/.205 |

**Cost and time** (one H100 80GB SXM, $3.49/h, pod up 15:40-23:10 UTC = **7.5 GPU-h, ~$26**; P0d notional Claude $19.82):

| run | calls | input tok | output tok (out/call) | JevBench out tok | wall |
|---|---|---|---|---|---|
| Gemma think | 13,377 | 64.5M | 13.06M (976) | 0.29M | ~2.1 h incl. the FD-crash restart |
| Qwen think (subset) | 4,271 | 16.6M | 4.34M (1,015) | 0.21M | ~4.0 h incl. 2 restarts; ~1.5k+0.6k tok/s |
| d1-3B | 1,263 joint calls | 5.8M | 0 | 0 | 3 min |

Per-question answers: `<run>/<workflow>/results.json` (every call's response, including probabilities) and
`<run>/jevbench/*.jsonl`. Machine-readable summary: `scoreboard.json` (`scripts/r054_score.py`).

## Verdicts (plan thresholds)

| ID | threshold | result | verdict |
|---|---|---|---|
| P0a | WF >= .58 go / < .56 drop (H2 also wants Jev/231 >= 207) | **.612**, Jev/231 **225** | **GO.** Gemma-4-12B-it with thinking is a usable self-teacher: +9.1 over its own single pass (Cygnet .521), and invoice (.664) is above Jev. Security (.396) remains the calibration hole. |
| P0b | WF >= .70 open teacher / < .64 buy GPT labels | subset **.684**, paired tie with Jev (.685), +2.0 over Gemma-think on the same cases; full-set estimate ~.63-.68 | **Inconclusive: no "go" at the .70 bar, not killed.** Qwen is the best open teacher measured and the clear calibration winner (security noul MAE .131 vs Gemma .220). It is too slow on this stack for mass labelling (~4x Gemma per question). Suggested follow-up: run the remaining 470 cases or switch serving stacks before deciding on GPT spend. |
| P0c | reference only | **.471**, Jev/231 174 | Below the frozen Gemma single pass, except security (.550, it hedges like the constant baseline). Invoice collapses (.316). No case for d1 as base or teacher. |
| P0d | noul mean gap <= .06 and flips <= 5% | noul MAE **.024**; flips **5.3%** (8/150; 4.1% excluding exact ties) | **Probability fidelity PASS; argmax gate missed by 1 item**, all on choice questions. Use this exact CLI path (structured, one question per call, `--effort high`); R053's low-effort labels are superseded. |
| P0e | rename | all new runs are `vllm:` / `typical:` | **Done.** Harness guard added. See below. |

## P0e — model identity (fixed)
Every saved Typical WorkflowEvals run was made with `--model typesafe:jev-1.13.0 --base-url <our shim>`, so it is
recorded as **`typesafe:jev-1.13.0`, lab `typesafe`, effort `None`, question_mode `all`**
(`cygnet-g0`, `intern-4b-g0`, `intern-4b-g0c`, `winnow-q8-g0b`, `typical-r044-b`).
Replaying WorkflowEvals' own `plot.py` grouping (`scripts/r054_p0e_merge_check.py`, key = (name, effort, question_mode)):

- **What would have merged:** `plot.py average` collapses all five local runs (Cygnet, Intern x2, Winnow, R044) into ONE
  point `('typesafe:jev-1.13.0', None, 'all')`, i.e. four different models averaged and plotted as "jev-1.13.0" in TypeSafe pink.
- The published Jev runs key as `('typesafe:jev-1.13.0', 'off', 'all')`, so they stayed separate only because our runs have
  effort `None` rather than `off`. In per-workflow plots with `--hf` our runs and the real Jev carry the same label and colour.
- **Fix** (`patches/workflowevals-r054-vllm-provider.patch`, applies on top of `workflowevals-typical-provider-0ac3b8ad.patch`):
  the harness now refuses `typesafe:` with any endpoint other than the official one ("A custom endpoint is not Jev; run it as
  typical:<model> or vllm:<model>"). Self-hosted decision models run as `typical:<immutable-id>` (native protocol, existing
  patch); self-hosted LLMs run through the new `vllm:<hf-id>` provider (system_one_adapter LLM path, OpenAI-compatible base_url).
  All R054 runs are recorded as `vllm:google/gemma-4-12B-it`, `vllm:Qwen/Qwen3.8-27B-FP8`, `typical:d1-3B-051bcc4`.
  Saved runs are left untouched (evidence). `.context/run_g0*.sh`, `run_intern*`, `run_winnow*` still pass
  `typesafe:jev-1.13.0` against unpatched upstream clones; do not reuse them.

## P0d — Claude-CLI teacher fidelity vs published claude-fable-5-1 labels
`scripts/r054_p0d_fidelity.py` -> `p0d/{answers.jsonl,summary.json}`. 150 benchmark questions (37-38 per workflow, equal
per kind within a workflow; only rows whose published anthropic label is fable-5-1, not the opus fallback), one question
per `claude -p --model claude-fable-5-1 --effort high` call, system_one_adapter's structured probability messages
(same system prompt as the gold run; schema passed with `--json-schema`). Diagnostic only: nothing is trained on these items.

| metric | value | gate |
|---|---|---|
| noul mean abs dp (n=58) | **0.024** | <= .06 PASS |
| all-kind mean abs dp (max over options) | 0.049 | - |
| argmax flips | **8/150 = 5.3%** | <= 5% FAIL (marginal) |
| flips by kind | noul 1/58, choice 7/58, score 0/34 | - |
| reasoning tokens / call | 259 (output 539) | R053 CLI labels had ~17-24 |
| notional cost | $19.82 (146 new calls) | cap $40 |

Of the 8 flips, 2 are exact ties on our side (noul p=.50; choice 2-way tie -> first-key argmax) and 3 more are
published near-ties (top-2 margin <= .06); excluding exact ties the flip rate is 6/148 = 4.1%. Flips sit on choice
questions (security 2/13, agent_trace 3/13, invoice 2/19, customer 0/13).
**Verdict: probability-level fidelity passes clearly (noul MAE .024, a third of the gate); argmax fidelity misses the 5%
gate by one item, entirely on choice questions.** The R053 effort problem is fixed (259 vs ~20 reasoning tokens). Before mass
labelling: use this exact path (structured, one question per call, `--effort high`), and treat choice-question labels as
~12% argmax-noisy against the published teacher (same order as run-to-run teacher noise is unknown; not measured here).

## Protocol notes (read before comparing numbers)
- **Harness path (P0a/P0b).** Pinned WorkflowEvals `0ac3b8ad` + `patches/workflowevals-typical-provider-0ac3b8ad.patch` +
  `patches/workflowevals-r054-vllm-provider.patch`, dataset revisions as in `scripts/run_r050_official_eval.sh`.
  `run.py --model vllm:<hf-id> --questions one --thinking <effort> --max-output-tokens 16384`: system_one_adapter's
  probability system prompt, one request per question, JSON-schema-constrained decoding (vLLM xgrammar,
  `disable_any_whitespace`), `--reasoning-parser gemma4|qwen3`, the model's own generation_config sampling (T=1).
- **One protocol deviation, needed for vLLM.** Hosted APIs (the gold teachers) show the native JSON schema to the
  model, and the questions exist only in that schema. vLLM only constrains decoding and never shows the schema.
  Without a fix, Gemma reasoned "no questions were provided" and answered blind. So the `vllm` provider runs
  system_one_adapter in its prompted mode: the same system prompt plus the schema appended (the adapter's own
  `_OUTPUT_SCHEMA_INSTRUCTION_TEMPLATE`), while decoding is still schema-constrained. The questions are seen exactly once.
  A side effect is that the schema sits before the document, so prefix caching across questions of one state is ~0%.
- **Thinking switch.** Gemma: `chat_template_kwargs.enable_thinking=true`, with no depth knob (recorded as effort `default`).
  Qwen3.8: thinking on at its default `reasoning_effort=xhigh`, recorded as `default`. Qwen only offers
  low/medium/xhigh, so "high" is not available. On the smoke test xhigh used 1,227 output tokens per call, against
  Gemma's 1,106.
- **JevBench/231 in the same mode.** `scripts/r054_systemone_server.py llm` exposes `/v1/systemone` over the same
  WorkflowEvals client (`vllm:` identity). jevbench `typesafe` adapter at commit bb05a335, client timeout raised from 120 s
  to 3600 s, the 231 public items in parallel chunks of 12. "Correct" = argmax hit, as in the G0 numbers (Cygnet 203).
- **d1-3B (P0c).** `LiquidAI/d1-3B@051bcc4`, its shipped `model.system_one` (trust_remote_code, bf16, transformers>=5.14)
  behind the same server (`d1` backend). Run as `typical:d1-3B-051bcc4` with `--questions all` (native joint readout, as Jev).
- **Wall times** are per case under heavy contention: all 4 workflows plus JevBench share one GPU, with 16 cases x 8
  questions in flight per workflow. They are throughput artifacts, not API latency.
- **Incidents.** (1) The first Gemma attempt crashed all four `run.py` processes on `Too many open files` (ulimit 1024).
  It was resumed from checkpoints with `--retry-errors`, and no case results were lost. (2) The JevBench runner is
  sequential and refuses to append, so a resumable chunked runner was added. One Gemma JevBench item failed on
  `RemoteDisconnected` during that restart and counts as wrong.
  (3) Qwen3.8-27B-FP8 ran at ~1.5-2k prompt tok/s plus ~0.6k gen tok/s on the H100 (vLLM 0.31; DeepGEMM disabled because
  NVCC<12.9; the triton GDN prefill made no difference). That projected to ~10 h for all 705 cases, so P0b was cut to a
  **budget-limited subset: the first ceil(n/3) cases per workflow in case-ID order** (80/37/50/68 = 235 cases), and is
  reported next to every other run rescored on exactly those cases (paired). The case-ID-order subset is biased:
  Jev scores .5625 on the first 80 security cases against .617 on all 240. Use the paired table, not the full-set
  references, for the P0b verdict.
