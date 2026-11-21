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
