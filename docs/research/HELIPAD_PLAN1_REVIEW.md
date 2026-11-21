# Helipad Plan 1: pre-launch review (2026-11-21)

Three independent Fable reviews of Plan 1's code, data design and architecture, done before any GPU spend.

Full reports (local, untracked): `.context/helipads/review/{code,ml,arch}.md`.
- **code**: correctness, fidelity to `lz_check`, Decision Factory integration. It fuzzed 1,100 inputs through the real `lz_check.evaluate()`.
- **ml**: training and data design, shortcuts, the synthetic-to-real gap, dose, gates.
- **arch**: our design against Perplexity Decider, Jeff-1, torchcast-decision 4B/31B, MetaDecision-31B, Decision-2.0-Nox-4B, StartLux-27B and five more.

**Verdict: do not launch as built.** The run would train and probably pass Plan 1's gates, but the result would prove little:
- the gates can be passed by shortcuts (a line-count decision tree reaches 91.7% item agreement against a 90% gate);
- real cases are out of distribution and longer than `max_state`;
- the real-site eval was nearly empty.

**Confirmed correct:**
- The rule mirror matches `lz_check` on 1,100 fuzz cases except I1 and I2 below. That covers the shift search, in-pad obstacles, `levels`/IEC heights, minefields, corridors, GovMap shapes, and FAIL before missing data.
- The 7 fetched sites agree, and S4 reproduces the calibrated 22 m north-west touchdown shift.
- The job layout, row schema and scorer prediction format are compatible with Decision Factory.

## Findings and fixes

Workstreams: **A** = lz_check integration (`from_lz`, `helipad_sites`); **B** = case, rules, rows; **C** = generator; **D** = build, eval, scoring.

| ID | Sev | Finding | Fix | WS |
|---|---|---|---|---|
| C1 | critical | Rejection sampling makes "approved" mean "sparse": mean 0.83 obstacles, 55% empty. Real approved sites render 25 lines. No case has more than 12 obstacles or 620 tokens | Constructive, verdict-first sampling with 0–40 obstacles, real-like runs (pole rows, building clusters, tree rows) and dense approved cases | C |
| ML-2 | critical | Shortcuts pass the gates: a depth-8 tree on line counts reaches 91.7% items / 86% verdict | Dense approved cases (C1), minimal-pair contrast set, baseline table, stricter gates | C, D |
| ML-1 / I7 | critical | The real-site eval was nearly empty; incomplete fetches (no OSM) were included; "not checked" counted as agreeing with "not approved" | Wait for all 16 sites; exclude incomplete fetches; report expert "not checked" as abstention | D |
| I1 | important | `verdict.json` caps obstacles at 60; on 4 of 5 dense real sites a verdict-relevant obstacle is dropped (a reproduced fails → probably_fails) | Recompute the full obstacle list from `data.json` with lz_check's own functions, unedited, at fetch time | A |
| I2 | important | Canopy in pad: the rules compare the rounded 1.0 m; `lz_check` fails on 1.04 m | Carry `lz_check`'s `rotor_obstacle_in_pad` flag | A, B |
| I3 | important | Labels use all obstacles; the text shows the nearest 25 | Relevance-first obstacle selection; label the visible case | B |
| I4 | important | `task: "helipad"` collapses the pod's base-vs-tuned reveal into one decision | `task = "import_<family>"` | B |
| I5 | important | Real states exceed 1,024 tokens (S2: 1,125); right-truncation cuts slope, canopy, dust and missing-data lines | Fixed lines before obstacles, compact obstacle lines, token-budgeted cap; test in tokens, not characters | B |
| I6 | important | Rows are ordered, so `--eval_limit 200` picks best.pt on 16 val cases and `--limit-cases` is biased | Shuffle every split; random case sample in the scorer | D |
| I8 | important | Every real fetch is degraded (GovMap 403; TAMA 10 corridors "Token Required"); the case cannot say so | Add a `corridors_ok` line; the GovMap line already exists | A, B |
| I9 | important | Policy text misses rules the labels apply (`levels`/`iec` are certain; "not checked if OSM missing"; canopy doesn't count for rings) | Complete the policy sentences | B |
| I10 | important | Rendering 49.6 m as "50 m" contradicts the label (32 of 6,000 cases) | One-decimal distances and heights | B |
| I11 | important | Rare labels starve (`minefield_500=fails` 13 of 6,000) | A rare-status floor of ≥1% per allowed status | C |
| ARCH-1 | important | `Site: <name>` leaks synthetic vs real and makes the dedupe a no-op | Drop the name from the state; keep it in meta | B |
| ARCH-2 | important | Fixed candidate order ("passes" is option A on 9 of 11 items) | Seeded per-row candidate shuffle | B |
| ARCH-3 / M11 | important | No ECE, not-checked recall or unsafe (false-clear) rate | Add them to the report | D |
| ARCH-4 | important | `expert_reason` (item-level evidence) unused | Expert item rows: rejected sites → `high_300` / `low_50` ∈ {fails, probably fails} | D |
| ML-V | important | Verdict query lacks item thresholds | The verdict of record is derived in code from item answers; the model's own verdict is kept as a consistency cross-check | D |
| M1 | minor | The dedupe compares whole states (name line) | Compare bodies; also dedupe val vs holdout | D |
| M2 | minor | GovMap hits not deduped across tolerances | Dedupe on (layer, name), keeping the tightest | A |
| M3 | minor | `kind_from_tags` gaps against `est_height` | Align the branch order | A |
| M4 | minor | `ultra_1km` uses edge, not centre, distance | Add `dist_center_m` to Obstacle | A, B |
| M5–M7 | minor | `--out` outside repo crashes; empty eval files break the pod; unknown slug KeyError; no fetch timeout | Fix each | D, A |
| M8 | minor | `_derived` in the scorer duplicates `rules.verdict` | One shared function | B, D |
| M9 | minor | Generator contradictions | Fixed with C1 | C |
| M10 | minor | "Touchdown point inside it" when the pad only touches a zone | Reword | B |

## Rulings (decided, not deferred)

- **Training size stays at 6,000 cases** (spec said 20k+). Case coverage is fine per the ml review, and at 800 steps the model sees about 25.6k rows anyway. Cost if wrong: less diversity; revisit after the probe.
- **No resampled soft targets in Plan 1** (arch #5). Uncertainty is already expressed in the `probably_fails` / `not_checked` classes, which mirror the calibrated rules. Resampling would change what the labels mean. Calibration is measured with ECE on the categorical labels.
- **The Plan 1 checkpoint is a baseline.** The findings format may change when video findings arrive in Plan 2 (arch #6).
- **The frozen base is the zero-shot baseline.** `reveal.json` and `base_eval_co.json` come from the pod job (arch #7). The rule engine (100% by construction), majority class and a line-count tree complete the baseline table.
- **Gates** (ml review), replacing Plan 1's ≥90% items / ≥95% consistency:
  - ≥97% item agreement on pad, the three rings and the derived verdict;
  - ≥90% on minimal-pair contrast sets;
  - unsafe errors ≤1% (predicting passes/approved where the label is fails / probably fails / not checked);
  - the expert result reported as counts, not a gate.
- **Launch sequence:** a 200-step probe, then 800 steps if the probe is healthy.

## Outcome (all fixes landed, 2026-11-21)

- Fix workstreams: B and A (Opus), D (Sonnet), C (Fable). One commit each, plus a baselines fix (its obstacle feature was always 0 under the new format).
- **Tests:** 84 passed, 1 skipped in the base env. In the geo env, `tests/test_helipad_lz_full.py` runs 150 seeded fuzz cases through the real `lz_check.evaluate()`: 4 passed.
- **Real sites:**
  - 16/16 verdicts match `lz_check` with uncapped obstacle lists (e.g. S3: 921 obstacles, not the 60 in verdict.json).
  - OSM failed twice for O10, so it is excluded.
  - GovMap (403) and TAMA 10 corridors ("Token Required") are unavailable at every site, and the case text says so.
- **Expert direction on map data alone:** the rules agree with the expert on 6/6 September sites. S1, S2 and S3 come out probably_fails; S4, S5 and S6 come out approved.
- **Dataset:**

  | File | Rows |
  |---|---|
  | train | 72,000 |
  | val | 7,176 |
  | holdout | 7,188 |
  | contrast | 5,400 |
  | no-rule probe | 2,400 |
  | real (15 sites) | 180 |
  | expert (6 verdict + 5 item rows) | 11 |

  The longest state plus query is 843 tokens, under the 1,024 limit.
- **Shortcut baselines (overall items / verdict):** the model must clear these.

  | Eval set | Majority | Line-count tree |
  |---|---|---|
  | Holdout | .745 / .351 | .816 / .477 |
  | Contrast | .716 / .464 | .777 / .371 |
  | Real | .861 / .267 | .891 / .667 |

  Before the generator rewrite, the tree reached .86 on the verdict.
