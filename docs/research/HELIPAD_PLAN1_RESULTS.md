# Helipad Plan 1 results: text decision model on rule-generated cases (2026-11-21)

Plan: `docs/superpowers/plans/2026-11-21-helipad-plan1-decision-foundation.md`. Spec: `docs/superpowers/specs/2026-11-21-helipad-lz-decision-model-design.md`. Pre-launch review: `docs/research/HELIPAD_PLAN1_REVIEW.md`.

**Model.** typical-medium (Qwen3.5-4B, tap 23, LoRA r16, native readout), fine-tuned through the unchanged Decision Factory job:
- command: `scripts/typically_job.py helipad medium <steps>`, one H100;
- mixture: 50% our data, 50% replay of the existing corpora;
- training data: 6,000 rule-labelled synthetic sites × 12 questions.

**Runs.** W&B project `decision-factory`:
- 200-step probe: `helipad-5e8165f3`;
- 800-step run: `helipad-e407d5ca`.

The rows below come from the 800-step checkpoint (`best.pt` = step 800).

## Headline

| | Result | Bar | |
|---|---|---|---|
| Expert direction (6 September sites, verdict of record) | **6/6** (the model's own verdict question: 5/6) | report as counts | the rules were tuned on these sites, so this shows faithful distillation, not generalisation |
| Real sites (15): verdict of record vs `lz_check` | **14/15**; **0/6 unsafe approvals** | — | |
| Pad item (held-out) | **99.5%** | ≥97% | ✅ |
| 300 m ring | 96.2% | ≥97% | ❌ (close) |
| 10 m ring | 93.8% | ≥97% | ❌ |
| 50 m ring | 84.5% | ≥97% | ❌ |
| Near-threshold pairs: rows whose label flips within the pair | **72.6%** (both pair members right: 69.4%) | ≥90% | ❌ (the earlier 94.3% pooled unflipped rows, so the gate could not fail) |
| Verdict of record (held-out) | 85.5% | ≥97% | ❌ |
| **Unsafe approvals** (verdict of record says approved, gold fails / probably fails / not checked) | **14.9%** held-out, 14.6% near-threshold | ≤1% | ❌ **the blocking failure** |
| **Unsafe item clears** (predicted passes where gold fails / probably fails, only on items where passes is possible) | **20.7%** held-out (low_50 33.3%, very_low_10 28.9%, high_300 8.0%, pad 4.2%), 24.0% near-threshold | — | the item-level view of the same failure |
| Calibration (ECE, held-out) | 0.009 | — | well calibrated |

## Against baselines (item agreement / the model's own verdict question)

| Eval set | Frozen base | Line-count tree | Probe (200 steps) | **800 steps** |
|---|---|---|---|---|
| Held-out | .478 / .301 | .816 / .477 | .896 / .479 | **.939 / .628** |
| Near-threshold pairs | .492 / .307 | .777 / .371 | .868 / .511 | **.910 / .556** |
| Real (15 sites) | .533 / .133 | .891 / .667 | .928 / .533 | **.956 / .667** |
| No-rule probe | .350 | — | .528 | .500 |

These are the pod's `eval_wf` numbers, over all rows. The local scorer's item agreement, which excludes the verdict row, is held-out .967, near-threshold .943 and real .982.

## What it means

1. **It learns far more than the shortcut, and it is calibrated.** It gains +46 points over the frozen base and +12 over the line-count tree on held-out sites, with ECE about 0.01. The simple items (dust, belly, zones, ultra-high, minefield, wires) are at 97–100%. Validation accuracy was still rising at step 800 (.92 → .925).
2. **The failure is the distance rings, and it fails unsafe.** 103 of the 2,500 held-out item rows whose gold is fails / probably fails / not checked are predicted "passes":
   - 51 on the 50 m ring;
   - 26 on the 10 m ring;
   - 14 on slope;
   - 10 on the 300 m ring.

   One missed ring makes the derived verdict "approved". The task needs the model to compare a printed distance with a printed threshold, and the numbers tokenise digit by digit.
3. **The model's own verdict question is weak (.628), and it is inconsistent with its item answers** (consistency .65). This confirms the ruling that the verdict of record is derived in code. Example: S2, where the model's verdict says "approved" while its own `high_300` answer is "probably fails" at .96.
4. **It relies on the rule text.** With the rule removed, item agreement drops from .94 to .50. Acceptable while every query carries its rule; it is a risk for paraphrased deployments.
5. **On the real sites it behaves well**: 14/15 verdicts and no unsafe approvals. The one miss is O7, where the model returns "probably fails" instead of "fails" on a pad obstacle; that is conservative, not unsafe. But n is 15.

## Next steps, ranked by expected value

1. **A safety decision rule. No retraining, about an hour.** Calibration is good overall (per-ring ECE still to measure), so flag an item as at least probably_fails whenever P(fails) + P(probably fails) ≥ τ, rather than taking the argmax. Fit τ on val for ≤1% unsafe approvals and report the false-alarm cost. The scorer now keeps full probabilities, but `run800.json` predates that, so it needs one re-scoring run on the scoring pod (`scripts/helipad_score_pod.py`).
2. **Dose and mixture.** The rings were still improving at 800 steps, and Decision Factory caps steps at 800. Run research training directly with `pcdm/train.py`: 2–4k steps, a higher share of our data than 0.5, and more near-threshold cases.
3. **Ring-relative distances in the case text.** The map step knows each ring, so it can print "inside the 50 m ring by 3.2 m". This removes numeric comparison from the model; what remains is height certainty, sources and aggregation, which is where the expert's judgement lives. It is a deliberate design change to decide with the user.
4. **Rebuild the dataset before the next run.** The final review capped zone lines and changed the obstacle cap from 10 to 9. This checkpoint was trained and scored on the old format.

## Operations notes

- `runpodctl` 2.1.9 called the retired REST v1 host `rest.runpod.io` (NXDOMAIN). Updating to 3.1.1, as the official RunPod plugin's skill prescribes, fixed Decision Factory jobs.
- Scoring on MPS is impractical: the Qwen3.5 linear-attention kernels fall back to a reference path. `scripts/helipad_score_pod.sh` scores on a rented H100 in about 15 minutes.
- Decision Factory jobs now log to W&B when a key exists (project `decision-factory`, one run per job).

## Final review (whole branch, Fable) and fixes

The verdict was "with fixes": 0 critical, 8 important. All 8 are fixed, each with a test that failed first:
- **I1–I4, scorer.**
  - The unsafe rate is computed only on items where "passes" is possible, plus per family.
  - Verdict-of-record safety and expert metrics come from code: expert 6/6 by record, 5/6 by the model's own verdict question, 4/5 on expert item rows.
  - The contrast set is scored only on rows whose label flips within the pair.
  - not_checked recall and ECE are reported per family.
  - `--preds-in` re-scores saved predictions without a GPU. The numbers above come from re-scoring `run800.json` this way.
- **I5, scoring pod.** A Python launcher with name-based, retried deletion on every exit path.
- **I6, W&B.** A W&B failure no longer kills a paid Decision Factory job; an empty `WANDB_API_KEY` opts out.
- **I7, zones.** Zone lines are capped at 4 and the obstacle cap is now 9. The worst case is 961 tokens, and a character-budget guard never skips.
- **I8, rounding.** Printed numbers never round across a protocol limit. `check` compares the visible case too: 16/16 sites.

Suite: 165 passed, 1 skipped (geo; 4/4 pass in the geo env). Deferred minors are listed in the execution ledger.

## Follow-up: a cautious threshold does not rescue the 800-step model

`scripts/helipad_threshold.py` reports an item as "probably fails" when P(fails) + P(probably fails) ≥ τ. The probabilities are unchanged; every option is still shown with its probability. τ was fitted on half the held-out sites (split by case) and is reported on the other half.

| τ (held-out test half) | Unsafe approvals | False alarms on truly approved sites | Verdict of record |
|---|---|---|---|
| none (argmax) | 13.8% | 11.2% | 85.0% |
| 0.46 | 12.7% | 12.2% | 85.4% |
| 0.21 | 6.9% | 30.6% | 81.9% |
| 0.11 | 1.6% | 50.0% | 78.7% |
| **0.03 (fitted for ≤1%)** | **0.5%** | **75.5%** | 70.7% |

At τ = 0.03:
- **near-threshold pairs:** 1.1% unsafe, 74.7% false alarms;
- **real sites:** 0 unsafe, 4 of 9 truly approved sites flagged.

**Reading.** Many ring violations get almost no probability on "fails". The model is confidently wrong, not uncertain, so no threshold separates hazards from clear sites. The fix has to come from the model: a larger dose (the 3,200-step run is next) or ring-relative distances in the case text. Re-scoring used the eval files in the format the 800-step model was trained on, rebuilt deterministically from the pre-cap-change commit.

## Follow-up: 3,200 steps

The 3,200-step run trained through Decision Factory (research step count) on the rebuilt case format (obstacle cap 9, zone cap 4). **Dose and format are confounded**: the 800-step model was trained and scored on the old format. Both checkpoints are scored on the held-out files in their own format. Numbers come from `helipad_score.py` `report()` (full-probability pass, `--limit-cases 0`), except ECE per ring, which is from the Decision Factory `eval_co.json` (same checkpoints, native mode).

| Metric (800 -> 3,200) | 800 steps | 3,200 steps |
|---|---|---|
| Item agreement, contrast, overall | .943 | .970 |
| Item agreement, contrast: very_low_10 / low_50 / high_300 / pad | .924 / .776 / .836 / .984 | .987 / .904 / .838 / .987 |
| `unsafe_item_rate`, contrast | 24.0% | 8.4% |
| `unsafe_item_rate`, rule holdout (import one-liner) | 20.7% | 5.3% |
| `unsafe_record_rate`, contrast | 14.6% | 2.5% |
| `unsafe_record_rate`, rule holdout | 14.9% | 2.3% |
| `verdict_of_record_agreement`, contrast | .764 | .896 |
| `verdict_of_record_agreement`, rule holdout | .855 | .957 |
| Flipped-pair agreement (contrast) | .726 | .865 |
| Real sites (S1-S6 and O1-O10): record agreement / unsafe records | .933 / 0 | 1.000 / 0 |
| Real sites: unsafe item rate | 18.2% | 0% |
| Expert direction by record | 6/6 | 6/6 |
| Expert direction, model's own verdict question | 5/6 | 6/6 |
| Expert item rows | 4/5 | 5/5 |
| ECE very_low_10 / low_50 / high_300 / pad | .032 / .045 / .086 / .015 | .010 / .047 / .127 / .013 |

Base (frozen) model item agreement on contrast was .491. Two weak spots remain at 3,200: `high_300` did not improve (.836 -> .838) and its ECE got worse; and on `helipad_norule` (no rule text in the prompt) `ultra_1km` collapsed to .09 agreement and overall agreement is only .553, so the model leans on the rule text. The 11-item expert file is too small to read per family.

### Cautious threshold

Same method as above, fitted on half the holdout (split by case), reported on the other half. Fitted tau = 0.44 (largest value with <=1% unsafe record rate on the fit half).

| tau | Unsafe records | False alarms |
|---|---|---|
| none (argmax), test half | 3.2% | 5.1% |
| 0.44 (fitted), fit half | 1.0% | 8.0% |
| 0.44 (fitted), test half | 2.6% | 5.1% |
| 0.44 (fitted), contrast | 2.3% | 33.7% |
| 0.21, full rule holdout | 0.5% | 10.2% |
| 0.11, full rule holdout | 0.5% | 17.3% |
| 0.06, full rule holdout | 0.5% | 20.4% |
| 0.01, full rule holdout | 0.5% | 32.7% |

Real sites: 0 unsafe, 0 false alarms at any tau.

### Verdict

- **Does the dose fix the rings?** Mostly. very_low_10 and low_50 improved clearly, unsafe records fell about six-fold (14.6% -> 2.5% on contrast), and real sites are clean. `high_300` and pad are unchanged, and the no-rule prompt still fails.
- **At <=1% unsafe approvals with an acceptable false-alarm rate?** Not yet. Unsafe records are 2.3-3.2% at argmax and 2.3-2.6% with the fitted tau, which fell short of the 1% target on the held-out half. The fitted tau is too loose; tau <= 0.21 reaches 0.5% on the full holdout but costs 10-33% false alarms, and on the contrast near-threshold pairs the false-alarm rate is already ~32% at argmax. The threshold now helps, because errors are less confidently wrong than at 800 steps, but it does not close the gap. Next candidates: ring-relative distances in the case text (option 3 above) and more near-threshold training data for `high_300`.
