# R060-pre: blending R056 toward the base, adapter merges, and a dev teacher-NLL proxy

Eval only, no training. Runner `scripts/r060pre_pod.sh` (+ `scripts/r060pre_proxy.py` for scale / merge / sample /
score / bench / gate / corr, `scripts/contested_eval.py`). Status log and raw outputs: `.context/r060pre/` (STATUS.md,
`pod/`). Pod `typical-r060pre` = q1a9q3lvk2shft, 2×H100 80GB SXM secure ($7.98/h; 4× H100/H200/A100 were sold out).

**Verdict.**
- **B (blend).** Scaling R056's LoRA delta trades WorkflowEvals for public JevBench and Hebrew almost linearly. λ=.5 keeps
  +5.1 WF points over the frozen base on the same serving path (.601 vs .550) and gives back exactly the base's EN 231
  (202/231) and Hebrew HE (.852). It also scores the best native Hebrew of any Typical (.899) and an external panel
  above R056's (.809 vs .803; R055-A .815), with every retention gate passing. λ=.75 sits close to λ=.5 on everything (WF .608, EN 199). R056 itself
  stays the WorkflowEvals leader (.649), with all of the security gain (.604 vs .471 at λ=.5). No λ gets both WF ≥ .64
  and EN ≥ 200.
- **Merges** (rank concatenation, exact delta sums, fixed arms). Adding half of R055-B on top of R056 does not beat
  either blend. The better arm, M2 = .75·R056 + .5·R055-B, has proxy NLL .427 vs λ.75 .421 and Hebrew mean .845 vs
  λ.5 .865 / λ.75 .857, with EN 198. The coordinator's gate failed, so no merge WorkflowEvals was run. The merges do
  reach the best MASSIVE-he of any model (121 vs base 118) and the best external panel (.812), but their image
  null-recall is lower than R056's.
- **D (proxy).** The ~2k-question dev teacher-NLL proxy ranks models by their mean WorkflowEvals score well (Spearman
  .90, n=5) and ranks models within security and agent_trace (.90–1.0). It does **not** reach ρ ≥ .7 across all
  (model × workflow) points: pooled Spearman is .43 (n=20), .70 after centering each workflow. Invoice (49 proxy
  questions, argmax agreement saturated at .939) and customer (the proxy prefers R056, WorkflowEvals prefers the blends)
  break it. Use it to reject bad checkpoints and to rank within security/trace. Do not use it to pick between close
  candidates or to tune customer/invoice heads.
- **Contested-fast-decisions** (eval only): λ.75 59.5%, R056 59.4%, λ.5 58.3%, base 55.0%. That is 3 points under JEV
  1.13 (62.6%) and level with StartLux 9B / Liquid d1 (60.1%). On the 788 decisions our server can take, the numbers
  are 61.8 / 61.7 / 60.5 / 57.1.

## Setup

- **λ blends.** `r060pre_proxy.py scale`: every `lora_B` tensor of `guychuk/typical-r056` (328 tensors) × λ, same
  config, plus R056's `temperatures.json`. This scales the LoRA delta by λ. λ=0 is exactly the base, and λ=1
  re-serializes byte-identically to R056 (same tree hash). Served by `serve_typesafe_gemma_lora.py` (native
  option-letter readout, B=1, `--temperatures` = R056's), with replicas behind `r056_lb.py`. Served ids:
  `typical-r060-lam000-sha256-cdf3e36830d4`, `…-lam050-sha256-e239bf81110f`, `…-lam075-sha256-bc691c0fb3a5`
  (λ1 = `…-lam100-sha256-4a727bab3a62`, R056's tree); merges `typical-r060-m1-sha256-3f5e2f4da57b`,
  `typical-r060-m2-sha256-361698cc8c7f`.
- **Merges.** `r060pre_proxy.py merge` concatenates ranks: A = [A1; A2] and B = [w1·B1, w2·B2] at r=64, α=128, so α/r
  stays 2 and the delta is exactly w1·Δ1 + w2·Δ2 (checked in `scripts/test_r060pre_proxy.py`). R056's temperatures
  apply. M1 = 1.0·R056 + 0.5·R055-B and M2 = 0.75·R056 + 0.5·R055-B were fixed in advance, with no tuning.
- **WorkflowEvals.** All 705 cases, pinned WorkflowEvals 0ac3b8ad + typical patch, the dataset revisions of
  `run_r050_official_eval.sh`, `typical:` provider, `--questions all`, consensus labels: R056's `wf` settings.
- **JevBench / Hebrew.** JevBench public 231 + hejev v0 (translated/mixed/native) + the external Hebrew panel, via
  jevbench bb05a335 `typesafe` adapter. Scored with `r060pre_proxy.py bench`, which reproduces R056's published row
  exactly. Retention: `scripts/eval_gemma_mm_he.py`, 128 per suite.
- **λ=0 rows.** The JevBench/Hebrew/external values are R057's frozen base, zero LoRA, through the same native server.
  R056's temperatures are keyed by WorkflowEvals question ids, so they do not touch JevBench. λ=0 WorkflowEvals was
  rerun here through the R056 path (zero LoRA + R056 temperatures) for an apples-to-apples curve.

## B: trade-off curve

| model | **WF mean** | security | trace | invoice | customer | **EN 231** | **HE** | MX | native | ext. mean | HE−EN | MASSIVE he/en | SQA real | null recall mism./gold-dropped | contested |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 (frozen base, R056 path) | .550 | .392 | .550 | .547 | .711 | .874 (202) | .852 | .835 | .815 | .790 | −.022 | 118/118 | 84 | 100/85 | 55.0 |
| **λ=.5** | .601 | .471 | .577 | .609 | **.748** | **.874 (202)** | **.852** | **.843** | **.899** | .809 | −.022 | 118/120 | 110 | 117/103 | 58.3 |
| λ=.75 | .608 | .517 | .590 | .582 | .743 | .861 (199) | **.852** | **.843** | .874 | **.810** | −.009 | 117/119 | **115** | 119/110 | **59.5** |
| λ=1 (R056) | **.649** | **.604** | **.635** | **.627** | .729 | .848 (196) | .822 | .835 | .866 | .803 | −.026 | 116/119 | 114 | **124/113** | 59.4 |
| M1 = R056 + .5·R055-B (merge) | not run (gate) | | | | | .853 (197) | .830 | .830 | .840 | **.812** | −.022 | **121/122** | 108 | 112/103 | - |
| M2 = .75·R056 + .5·R055-B (merge) | not run (gate) | | | | | .857 (198) | .839 | .830 | .866 | **.812** | −.017 | **121/123** | 109 | 104/98 | - |
| ref: Cygnet (frozen Gemma, its T=3.4 letter readout) | .521 | .333 | .523 | .531 | .698 | .879 (203) | .839 | .839 | .882 | .784 | −.043 | - | - | - | - |
| ref: R044 | .528 | .338 | .550 | .518 | .708 | - | - | - | - | - | - | - | - | - | - |
| ref: Jev 1.13 | .678 | .617 | .716 | .618 | .760 | - | - | - | - | - | - | - | - | - | 62.6 |

WF = WorkflowEvals headline per workflow (security exact, trace primary, invoice exact, customer exact; consensus
labels). Every run had 0 failed cases. External panel per split, λ.5 / λ.75: belebele .906/.908, hebnli .829/.836,
gmmlu .623/.623, sentiment .798/.808, winograd .773/.755, heq .927/.927. Retention gates (script's own) pass for both
blends. MASSIVE-he loses 0 (λ.5) and 1 (λ.75) errors vs base; R056 lost 2.

Per question vs consensus (noul MAE / noul side flip / choice+score argmax flip):

| model | security | agent_trace | invoice | customer |
|---|---|---|---|---|
| λ=.5 | .133 / .143 / .516 | .142 / .160 / .217 | .043 / .032 / .167 | .049 / .041 / .162 |
| λ=.75 | .115 / .128 / .532 | .132 / .158 / .211 | .042 / .031 / .173 | .047 / .041 / .162 |
| R056 | .112 / .116 / .561 | .128 / .150 / .185 | .042 / .029 / .178 | .055 / .045 / .172 |

Reading the curve:
- WF is roughly linear in λ from .5 to 1 on security (.471 → .517 → .604) and trace, and saturates early on invoice.
  On customer it is non-monotone: the blends beat R056 (.748/.743 vs .729), and R056's customer nouls were the
  least consensus-like (MAE .055).
- Security choice/score fields get *better* as λ falls (argmax flip .561 → .516). R056's remaining security gap is in
  those fields, not in the nouls.
- Public JevBench and the main Hebrew columns come back to the base at λ=.5 exactly (202/231, HE .852). Native Hebrew
  (.899) and the external panel (.809) go *above* both the base and R056. Half of the R056 delta keeps the Hebrew and
  native gains while removing most of what costs JevBench.

## Contested-fast-decisions (zoom-ai/contested-fast-decisions, eval only)

`scripts/contested_eval.py`, README protocol: one `choice` per single-label decision, one yes/no `noul` per label
for multi-label decisions (label kept iff noul > .5), only the task name and candidate labels. Exact set match over
all 818 decisions. One request per text, R056 serving path and temperatures (the question ids here are not WF ids, so
T = 1). The 30 `support_intent` decisions have 28 candidate labels, beyond the native server's 26-letter readout
(HTTP 422). They count as wrong in the comparable column.

| model | accuracy | single-label | multi-label | on the 788 servable |
|---|---|---|---|---|
| Typical λ=0 (frozen base) | 55.0 | 63.3 | 26.2 | 57.1 |
| Typical λ=.5 | 58.3 | 65.4 | 33.9 | 60.5 |
| Typical λ=.75 | **59.5** | **67.4** | 32.2 | 61.8 |
| Typical R056 (λ=1) | 59.4 | 66.5 | **35.0** | 61.7 |
CONTESTED_| M1 = R056 + .5·R055-B (merge) | not run (gate) | | | | | .853 (197) | .830 | .830 | .840 | **.812** | −.022 | **121/122** | 108 | 112/103 | - |
| M2 = .75·R056 + .5·R055-B (merge) | not run (gate) | | | | | .857 (198) | .839 | .830 | .866 | **.812** | −.017 | **121/123** | 109 | 104/98 | - |
| Zoom Decisions Pro | 73.7 | 73.7 | 73.8 | |
| JEV 1.14 | 66.0 | 68.5 | 57.4 | |
| JEV 1.13 | 62.6 | 65.5 | 52.5 | |
| StartLux Decision 9B Instant | 60.1 | 63.1 | 49.7 | |
| LiquidAI d1 | 60.1 | 60.3 | 59.6 | |
| Cloudflare Clef-Flash | 46.2 | 51.3 | 28.4 | |

Single-label accuracy is in JEV 1.13's range. The gap is multi-label (one noul per label with no label description;
we over- or under-select), which every Typical has around 26–35%.

## D: the dev teacher-NLL proxy

Proxy set: `.context/r060pre/proxy_ids.txt` has 2,033 of the 2,855 R056 dev questions (dev = by case). It was sampled
with `r060pre_proxy.py sample`, stratified by workflow × question type, with EN/HE twins drawn together; seed 60.
Composition: security 794, agent_trace 477, customer 713, invoice 49; 664 HE twins. Targets are R056's teacher soft
targets. Forward = serving forward (`r056_devsel.py`, now with `--ids` and `--bidirectional-prompt`), raw
probabilities with no temperatures.

| model | all (NLL / Brier / agree) | security NLL | agent_trace NLL | invoice NLL | customer NLL | EN NLL | HE NLL |
|---|---|---|---|---|---|---|---|
| base (λ=0) | 1.339 / 0.227 / 0.835 | 1.567 | 1.355 | 0.418 | 1.139 | 1.347 | 1.324 |
| λ=.5 | 0.490 / 0.122 / 0.866 | 0.624 | 0.459 | 0.308 | 0.375 | 0.482 | 0.508 |
| λ=.75 | 0.421 / 0.096 / 0.873 | 0.552 | 0.365 | 0.289 | 0.321 | 0.416 | 0.431 |
| R056 (λ=1) | 0.411 / 0.092 / 0.879 | 0.533 | 0.359 | 0.279 | 0.319 | 0.405 | 0.424 |
| M1 = R056 + .5·R055-B | 0.434 / 0.104 / 0.871 | 0.552 | 0.380 | 0.211 | 0.354 | 0.426 | 0.452 |
| M2 = .75·R056 + .5·R055-B | 0.427 / 0.101 / 0.871 | 0.546 | 0.378 | 0.208 | 0.341 | 0.417 | 0.446 |
| R044-R | 0.632 / 0.179 / 0.827 | 0.807 | 0.650 | 0.259 | 0.451 | 0.622 | 0.652 |
| R055-A | 0.582 / 0.178 / 0.818 | 0.716 | 0.573 | 0.523 | 0.443 | 0.566 | 0.616 |
| R055-B | 0.550 / 0.170 / 0.819 | 0.693 | 0.513 | 0.315 | 0.433 | 0.545 | 0.562 |
| R057h (bidirectional mask) | 0.664 / 0.253 / 0.727 | 0.783 | 0.608 | 0.768 | 0.562 | 0.644 | 0.705 |

(Brier = Σ over options of (p − teacher)², vs soft targets. Agree = argmax agreement with the teacher.)

Correlation with WorkflowEvals headline per (model × workflow). Models with WF: base (λ0, R056 path), λ.5, λ.75,
R056, and R044 (WF .528 from `.context/r044-workflows`). The R044 proxy row is R044-R, a reconstruction (adapter
3b5535de…), not the historical R044-B, so that point is a stand-in. R055-A/B and R057h have no WF run. Signs are
flipped so that positive = proxy agrees with WF.

| set | −NLL (Spearman / Pearson) | −Brier | agreement |
|---|---|---|---|
| pooled, 5 models × 4 workflows (n=20) | .43 / .43 | .49 / .59 | .47 / .65 |
| pooled, each workflow centered (n=20) | .70 / .49 | .69 / .64 | .74 / .61 |
| model mean over workflows (n=5) | .90 / .61 | .90 / .85 | 1.00 / .98 |
| within security (n=5) | .90 / .60 | .90 / .82 | .67 / .80 |
| within agent_trace (n=5) | .97 / .67 | .97 / .87 | .97 / .94 |
| within invoice (n=5) | −.10 / .18 | −.70 / −.78 | - (constant) |
| within customer (n=5) | .50 / .60 | .50 / .77 | .60 / .68 |
| λ family only, pooled (n=16) | .54 / .48 | .65 / .61 | .49 / .64 |
| λ family only, model mean (n=4) | 1.00 / .88 | 1.00 / .92 | 1.00 / .96 |

**Is it good enough (ρ ≥ .7)?** Not as one number over (model × workflow): pooled Spearman .43. The workflow offsets
differ between the proxy and WorkflowEvals; for example, invoice has the lowest NLL but not the highest WF score.
- It is good enough to **rank whole models by mean WF** (.90; the only swap is base vs the R044 stand-in) and to rank
  within **security and agent_trace** (.90–.97).
- It is not good enough on **invoice**: 49 questions from ~4 dev cases, every model at .939 agreement.
- It is not good enough on **customer**: the proxy ranks R056 ≥ λ.75 > λ.5, while WF ranks λ.5 > λ.75 > R056. The
  teacher there is saturated Gemma-think (customer targets are Gemma-only and smoothed), so lower teacher-NLL means
  closer to a teacher that is not the consensus.
- **For checkpoint and head selection:** use proxy NLL (or Brier, which tracks WF better within workflows) to drop
  clear losers and to choose security/trace heads. Grow the invoice dev slice (≥ 300 questions) and re-anchor
  customer targets (gold-style or Qwen) before trusting it there. Confirm any close call on WorkflowEvals.

## Cost and time

| item | |
|---|---|
| pod q1a9q3lvk2shft, 2×H100 SXM secure, $7.98/h | 19:40:48 → 02:48:59Z = 7.14 h ≈ **$57** (EXITED; cap $72) |
| proxy (8 adapters) + retention | 19:43 → 20:29Z |
| λ.5 + λ.75 WF 705 + JevBench/hejev/external | 20:29 → 00:06Z (invoice is 89% of WF input tokens; ~6k prefill tok/s per H100) |
| λ0 WF 705 | 00:13 → 01:59Z |
| contested (4 models, dedicated replicas) + merges (proxy, retention, JevBench/hejev/external, gate) | 01:59 → 02:29Z |

## Deviations

- 2×H100 instead of the requested 4× (none of 4× H100/H200/A100, secure or community, was available). The same GPU-hours
  took about twice the wall time.
- The adapters (R044-R, R055-A/B, fp32, no README) went once from the laptop to private `guychuk/typical-r060pre-adapters`.
- evalpair's bare `wait` also waited on the server processes and hung ~7 min after the WF finished, until the
  fleets were killed by hand. Fixed with pid-scoped waits.
- The first contested runs rode the WF fleets and queued behind invoice calls. They were killed and rerun on dedicated
  replicas after the main chain.
- The 30 `support_intent` contested decisions (28 labels) cannot be served by the 26-letter native readout.
- Merge WorkflowEvals: not run. The coordinator's gate failed (`.context/r060pre/pod/merge-gate.json`). A
  `COST_DEADLINE` guard would also have skipped it, because a ~1.9 h WF would have ended past the $72 cap.
- The pod idled ~20 min after MERGE_DONE (02:29Z) before the results were copied and the pod stopped (02:49Z).
