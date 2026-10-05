# Experiment Tracker — Jev parity

| Run ID | Block | Purpose | System / protocol | Primary metrics | Status | Notes |
|---|---|---|---|---|---|---|
| R001 | prior | Restore artifacts | `tm2` + E/K/W/U/V/H | schema audit | DONE | 1,054,473 training rows checked |
| R001a | prior | Native reference | `tm2`, row-wise Jev public | accuracy/calibration/latency | DONE | standard/easy/hard `.861/1.000/.486`; total 164/231 |
| R001b | prior | Same-backbone direct letters | frozen Qwen3.5-4B | accuracy/calibration/latency | DONE | `.847/.979/.468`; did not close capability gap |
| R002 | prior | MM/HE smoke | frozen LoRA, shared head | image/HE/text | DONE | signal positive; insufficient retention proof |
| R003 | prior | Visual baseline | direct Qwen3.5 letters | visual controls | DONE | real/no-image/shuffled `.898/.727/.617` |
| R005 | prior | E1 gate | frozen LoRA, E/K/W/U/V/H | all prior gates | FAILED | Jev 164→159/231; typed `.373→.340`; mismatch null `.105`; MM/HE improved |
| R010 | B0 | Correct state length | all `eval_val_loss` entry points | propagated `max_state` | DONE | fixed 256-token default leak; targeted tests pass |
| R011 | B0 | Official workflow harness audit | WorkflowEvals 705 cases | exact actions, request shape | DONE | Jev target mean 67.8; benchmark quarantined |
| R012 | B0/B1 | Cygnet identity | pinned Gemma-4-12B-it + exact letter readout | public Jev 203/231 ±1 | DONE | 203/231 exactly: 48 easy, 70 standard, 85 hard |
| R013 | B1 | Cygnet workflow baseline | same untouched model, official WorkflowEvals | 61.7/71.6/61.8/76.0 and mean 67.8 targets | DONE | security/trace/invoice/customer `.333/.523/.531/.698`; mean `.521` |
| R014 | B1 | Winnow baseline matrix | Winnow-12B Q8 reference runtime | Jev/workflow | DONE | 199/231; workflow `.392/.626/.478/.735`; best baseline mean `.558` |
| R015 | B1 | Intern fast/MM matrix | Intern-Decision-4B | Jev/workflow/latency | DONE | 200/231; workflow `.521/.468/.447/.559`; mean `.499`; trainable native-MM base selected |
| R016 | B1 | Existing 9B evaluation | `tl2` | corrected context/grouping | QUEUED | evaluate before any new training |
| R017 | B0 | Grouped typed baseline | full five-question cases | accuracy/KL/Brier/ECE | QUEUED | replaces non-comparable first-N row proxy |
| R018 | B2 | Readout ablation | final symbol vs pointer vs native | held-out accuracy/NLL/Brier | BLOCKED | run only after baseline matrix |
| R019 | B3 | 250-step clean competence gate | Intern-Decision-4B + rank-16 language LoRA | hard + workflow dev + retention | FAILED | Internal NLL improved `.775/.949/.598`→`.359/.180/.423`, but public Jev fell 200→197/231; WorkflowEvals security/trace/invoice `.479/.500/.478`, customer exceeded the unoptimized 96 GB runtime ceiling |
| R020 | B3 | Soft public-distillation gate | base-reset Intern LoRA; filtered Tasksource/Open-Jev/Sarge + 30% replay | locked dev gates | FAILED | Public `199/231`; workflow security/trace/invoice `.588/.495/.469`; no transferable workflow gain |
| R021 | B3 | Fresh-base sensitivity control | pinned `google/gemma-4-12B-it` + own option-logit rank-32 LoRA | internal option NLL/accuracy, retention | FAILED | Public diagnostic 203/231, but workflows `.300/.536/.298/.716`, mean `.462`; invoice primary `.573` vs Cygnet `.813` |
| R022 | B0/B3 | Scorer + action-threshold forensics | cached probabilities; no training | scorer parity, threshold/action failure decomposition | DONE | S0 trained on six aliases/letter while structured serving allowed two exact-uppercase aliases; scale `.25/.5/.75/1` did not expose a retention-preserving interior optimum |
| R023 | B3 | Exact-serving scorer control | fresh base; otherwise identical to R021 | corrected internal option NLL/accuracy, retention | FAILED | Public-source/replay/workflow/hard-rule `.652/.797/.832/.836`; forbidden aliases held <`.005%` mass; no aggregate rescue |
| R024 | B3 | Output-retention ablation | own fresh base; corrected scorer + base-logit KL | grouped/source-balanced internal dev | BLOCKED | interface-coverage factor now precedes KL; localization remains separate |
| R025 | B4 | Hebrew/MM residual | own text winner + residual/KL retention | MM/HE plus all retention gates | BLOCKED | must use pixels; cannot share the old unconstrained continuation path |
| R026 | B5 | Frozen final public eval | promoted checkpoint | diagnostic only | BLOCKED | one run after internal freeze; not a promotion gate |
| R027 | B5 | Official JevBench submission | promoted checkpoint | score ≥72.1, Intelligence ≥72.0 | BLOCKED | sealed result required for completion |
| R028 | B6 | Serving optimization | Q8/prefix/concurrency | equivalence, p50/p95, cost | BLOCKED | quality first |
| R029 | B0/B3 | Noul interface canonicalization | S1; generic semantic `False/True` fallback for `criteria=null` | crossed prompt probe; invoice actions reporting-only | DONE | Same-weight generic renderer exact/primary `.396/.649`; canonical `.504/.811` (`+.109/+.162`), matching the predeclared counterfactual replay |
| R030 | B3 | Matched interface-coverage screen | fresh base; 1,488 shared + 512 matched Noul presentations per arm | independent 2,048-view crossed panel plus retention | DONE | A `.836`, B `.911`; B repaired negated declarative/generic `4.7→95.3%` but reduced negated explicit to `78–79%`; paired-group B−A `+7.42` points, 95% CI `[+4.69,+10.06]` |
| R031 | B3 | Exact LoRA-delta interpolation | exact rank-64 `(1−λ)ΔA+λΔB`, frozen `λ=.25/.5/.75` | all 1,024 compact crossed views | FAILED | Best `λ=.75`: overall `.943`, worst cell `.875`, target `.914`; missed locked `.90/.93` gates, so no grid refinement or full eval |
| R032 | B3 | Confound-resistant mixed-interface arm | fresh base; 1,488 shared + balanced 256 A-format / 256 B-format views | full 2,048 crossed panel plus retention | FAILED | Overall `.931`, but negated declarative/generic was `.793` and fell to `.680` on full context (`.578` no/yes order, `.781` yes/no); WH/replay `.820/.859` also missed retention floors |
| R033 | B3 | Complete paired-view CE | fresh base; 1,488 shared slots + 512 aligned A/B donor pairs, `0.5 CE(A)+0.5 CE(B)` per donor slot | fixed-final 250 steps; full crossed panel plus retention | QUEUED | Coverage-only screen: 2,000 logical slots but 2,512 physical views; same optimizer, schedule, seed, and donor/shared gradient weights as R032 |

## Active resource

- Pod: `ujiq3ud4vnhf80` (`typical-jev-parity-g1`), one RTX PRO 6000 (96 GB, $2.09/hour), currently stopped while R033 is audited.
- Next job: R033 Arm D, followed automatically by its fixed full crossed evaluation.
- Stop condition: fixed-final checkpoint only; persist checkpoint/logs/results, then advance only on the locked worst-cell, order, recall, and retention gates. The cost guard checks utilization every 30 minutes and stops the pod only after artifacts are persisted and no job remains.
