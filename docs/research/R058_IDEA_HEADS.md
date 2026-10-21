# R058: three idea heads on frozen features of the R056 student

Date: 2026-10-21. Ideas: `.context/novel-ideas/{SYNTHESIS,forecasting,neuro_bio,econ_law_pl}.md`.
Code: `scripts/r058_heads.py` (rows / cache / train / replay / serve / diag) and `scripts/r058_pod.sh`.
Artifacts: `.context/r058/` (`results/replay/replay.json`, `results/heads/{heads.pt,pred.jsonl,train_log.json}`,
`results/wf-c_cert/`, `diag_clamp_r056_official_security.json`, pod logs).

## Verdict

None of the three heads beats the R056 student it sits on. On held-out synthetic replay the plain student scores
**.709**. The best head, burden certificates (c), scores .691. The criterion-spread head (a) and the replay gate (b)
fail their own kill rules. Certificates (c) pass their literal synthetic gate, which compares them with their own
uncertified distribution, so (c) got its one official run: officially it scores **.635**, which is .013 below R056 alone (.649) and .043 below Jev (.678). Its security score rises by .004, but customer service falls by .062..
R056 has already closed the calibration hole these ideas were aimed at. Its security answers score .604 officially,
and the fixed [.15,.75] projection now *lowers* that to .600 (it gave R044 +.22).

## Setup

- **Student.** R056 step 1052, `r056-sha256-4a727bab3a62` (LoRA r32 on Gemma-4-12B-it `707f0a3`), served with its
  per-question temperatures (`temperatures.json`, fitted by R056 on its dev split). Every head starts from the
  *served* option logits (logits / T).
- **Features** (1×H100, ~2.5 h): the answer-slot hidden state is the lm_head input at the last prompt token, taken
  from the exact serving forward. Before caching, the cached logits were asserted equal to `score_options`. Also
  cached: the option logits and the metadata (workflow, node, qid, type, lang), plus both teacher probabilities
  (Qwen3.8-27B think, Gemma-4-12B think) where they exist. There were 0 over-length skips.
  - **train:** 20,000 labeled English questions. Nodes come from R056's decontaminated `mix/train.jsonl` and are
    taken in sha256 order (a random sample). By workflow: security 7,800, trace 4,922, invoice 2,207, customer 5,071.
    A 10% case-salted internal split (1,890) chose each head's epoch.
  - **dev:** R056's held-out split (`sha256(case_id) % 10 == 0`). It holds every question of every node (2,841;
    939 English nodes' questions plus 902 Hebrew-twin rows), so head-driven branches always have answers. 16 dev
    cases whose labeled nodes R056's 13-gram filter had dropped were removed. Replayable cases: security 54,
    trace 36, customer 23, invoice 1.
  - **Targets** = R056's own `teacher_target`: security uses Qwen only; elsewhere it is mean(Gemma, Qwen), and a
    lone Gemma answer is smoothed off 0/1. The "teacher mean" in every comparison below is this target.
- **Heads** (CPU, minutes; every hyperparameter is in `HP` and was fixed before any dev score). All heads are
  residual to the served logits, z-scored state, AdamW 1e-3, wd .01, ≤30 epochs, best internal-val epoch.
  - (0) `base0`: a linear head on the state, KL to the target.
  - (a) `a_spread`: on binaries, z = student evidence + w·h, criterion c = e_qid + w·h, spread s = softplus(·), and
    p = E_C~N(c,s²) σ(z−C) by 5-point Gauss–Hermite. The loss is Brier(p, m) + 1·(Var_C − (p1−p2)²/2)² + .01 s².
    Choice and score rows use base0.
  - (b) `b_gate`: p = softmax(l0 + g·W1h). The gate g = σ(w·h + a·entropy + b·margin + e_type) is trained with
    BCE toward min(1, JS(p0, m)/τ), with τ = the 70th percentile on train. The band hinge sits at the security
    .15/.75 (ρ=1), with β=.1·(1−g)·KL(p0‖p). The comparators are constant blends g ∈ {.25, .5, .75, 1}.
  - (c) `c_cert`: base0 plus two certificate logits on the compiled slice. The compiler reads `P_CLOSE`/`P_ACT`
    from `security_incidents/workflow.py`. `is_true_positive` is the only noul whose two thresholds bracket a
    middle action; every other noul in the four programs is a single floor. The certificate targets come from
    teacher unanimity, min(teachers) > .75 and max(teachers) < .15; security has Qwen alone, so unanimity
    degenerates to Qwen. λ=1, pass threshold .5. An uncertified crossing is projected 1e-6 inside [.15,.75].
- **Replay**: the WorkflowEvals programs (pinned `0ac3b8a`) run on each arm's answers and on the target's answers
  for every dev case (`r053_calibration_headroom.ReplaySession`). The metric is agreement with the program run on
  the target, using the headline metric per workflow. A branch with no cached answer counts as wrong; there were
  none. The mean is the preregistered equal mean over workflows with ≥10 replayable cases, so invoice (1 case) is
  excluded.

## Held-out synthetic replay (agreement with program(teacher target))

n decisions: security 54, trace 72, customer 69. MAE and middle band are over the 1,391 English dev nouls. The
"tp" columns cover the 54 `is_true_positive` answers; the teacher's middle band is .12 overall and .28 on tp.

| arm | security | trace | customer | **mean** | noul MAE | mid band | tp MAE | tp mid |
|---|---|---|---|---|---|---|---|---|
| **R056 student (served)** | .611 | .833 | .681 | **.709** | .114 | .29 | .150 | .44 |
| (0) base0 | .537 | .833 | .667 | .679 | .102 | .21 | .149 | .43 |
| (a) a_spread | .593 | .806 | .696 | .698 | .102 | .19 | .151 | .39 |
| (b) b_gate | .593 | .833 | .638 | .688 | .103 | .24 | .138 | .43 |
| (c) c_cert | .574 | .833 | .667 | .691 | .102 | .21 | .148 | .50 |
| best constant blend (g=.5) | .630 | .861 | .710 | .734 | .105 | .25 | .144 | .43 |
| *diag: student, tp clamped to [.15,.75]* | .315 | .833 | .681 | .610 | .116 | .31 | .217 | 1.00 |

## Gates (each idea's own rule, fixed before scoring)

| head | rule | measured | verdict |
|---|---|---|---|
| (a) criterion spread | binary MAE cut ≥ .025 vs base0 **and** agreement ≥ +.020 **and** HE argmax flips ≤ 2/128 | MAE −.0003; agreement +.019; HE flips 5.7% (51/902) | **kill** |
| (b) replay gate | gate AUPRC ≥ .50 **and** ≥ +.020 over the student and the best constant blend **and** security not worse | AUPRC .82 (base rate .33); −.021 vs student; −.046 vs blend g=.5; security −.019 | **kill** |
| (c) certificates | security ≥ .50 **and** mean ≥ .57 (econ_law_pl.md) **and**, on synthetic, mean ≥ base0 | .574 / .691 / ≥ .679 | **pass → one official run** |

The gate itself works: it does predict where thinking-distilled targets disagree with the student (AUPRC .82). But
the learned residual it opens does not help, and an untrained half-blend of the same residual does better than the
learned gate. Every residual head, base0 included, lands below the student. The features come from the student's
own training split, where it already fits the teacher, so the heads learn little correction that transfers to
held-out cases.

## R059 variant (coordinator request): security nouls retargeted to `p_corrected`

`.context/r059/security_targets_corrected.jsonl` maps Qwen p toward gold style, i.e. the mean of gpt-6-astra and
claude-fable-5-1, logit-linear and fitted on R059's 400-question anchor. That anchor lies entirely in R056's train
split, so there are no anchor gold labels on dev. "Gold-style" scoring of held-out security therefore uses the
corrected map applied to dev's Qwen targets (n = 594 English security nouls, 54 cases).

| arm | mean (vs R056 target) | security vs Qwen target | **security vs corrected** | security noul MAE vs corrected |
|---|---|---|---|---|
| R056 student | .709 | .611 | **.648** | .139 |
| base0 / base0+corr | .679 / .714 | .537 / .611 | .556 / .630 | .126 / .130 |
| a_spread / +corr | .698 / .686 | .593 / .556 | .611 / .574 | .141 / .132 |
| b_gate / +corr | .688 / .658 | .593 / .574 | .611 / .611 | .126 / .131 |
| c_cert / +corr | .691 / .683 | .574 / .593 | .593 / .630 | .126 / .130 |

Gates for the +corr variants: a_spread+corr **kill** (agreement −.027 vs base0+corr), b_gate+corr **kill**
(AUPRC .83 but −.050 vs the student), c_cert+corr **kill** (.683 < base0+corr .714). Against the gold-style
reference the student is still the best security arm (.648). The heads reach a lower noul MAE (.126–.131 vs .139)
without better actions.

## Official WorkflowEvals (705 cases, pinned revisions, `typical:` provider, questions all)

| model | security | trace | invoice | customer | **mean** |
|---|---|---|---|---|---|
| Jev 1.13 | .617 | .716 | .618 | .760 | **.678** |
| R056 student alone (`r056-sha256-4a727bab3a62`) | .604 | .635 | .627 | .729 | **.649** |
| R056 + (c) c_cert (`typical-r058-c_cert-sha256-4a727bab3a62`) | .608 | .640 | .627 | .667 | **.635** |
| *NON-SHIPPABLE DIAGNOSTIC: R056 security answers with tp clamped to [.15,.75]* | *.600* | | | | |

(c) passes econ_law_pl.md's literal official gate (security .608 ≥ .50, mean .635 ≥ .57; that gate was set against R044's .528). It still loses to the student it was added to. Most of the loss is customer service, where (c) is just the base0 linear residual: no certificate slice touches customer, and base0 was already .014 below the student on synthetic customer service. Certificates do not ship.

The diagnostic replays R056's saved official security answers with the fixed projection
(`r058_heads.py diag`). It hard-codes the evaluator's thresholds into the output, so it is not a model and not
eligible for release. It is reported only because the synthesis cited it (+.22 on R044). On R056 it is −.004.

## Caveats

- Dev is small: 54/36/23 cases, where one security decision is .019. Every gap above is within a few decisions.
- R056 chose its checkpoint and fitted its temperatures on the same dev split, which favours the student arm
  slightly. That bias works against the heads.
- Hebrew rows inherit English targets (R056's twin rule). They feed only the HE argmax-drift check. No image rows.
- JevBench, Hebrew and image retention were not re-run for (c), because a head that does not beat the student is
  not a release candidate.
