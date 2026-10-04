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
| R020 | B3 | Soft public-distillation gate | base-reset Intern LoRA; filtered Tasksource/Open-Jev/Sarge + 30% replay | locked dev gates | RUNNING | 40k rows, 4,812 soft targets; exact public overlap 0; LR `2e-5`; pod `ujiq3ud4vnhf80` |
| R021 | B3 | Hard refinement | long docs + staged hard negatives | hard families + workflow | BLOCKED | attention still excluded |
| R022 | B4 | Hebrew/MM residual | text winner + residual/KL retention | MM/HE plus all retention gates | BLOCKED | cannot share the old unconstrained continuation path |
| R023 | B5 | Frozen final public eval | promoted checkpoint | all declared gates | BLOCKED | one run after freeze |
| R024 | B5 | Official JevBench submission | promoted checkpoint | score ≥72.1, Intelligence ≥72.0 | BLOCKED | sealed result required for completion |
| R025 | B6 | Serving optimization | Q8/prefix/concurrency | equivalence, p50/p95, cost | BLOCKED | quality first |

## Active resource

- Pod: `ujiq3ud4vnhf80` (`typical-jev-parity-g1`), one RTX PRO 6000 (96 GB, $2.09/hour).
- Job: R020 fixed 250-step Intern LoRA gate; 32,464,896 trainable parameters; base reset after R019 failed transfer.
- Stop condition: pull adapter, optimizer state, and metrics after step 250; stop immediately on failure or after collection.
