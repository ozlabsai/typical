# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Restore artifacts | tm2 checkpoint + E/K/W/U/V/H data | all | file/schema audit | MUST | DONE | tm2 restored from public release; pinned E/K/W/U plus 128-row V/H pilots; 1,054,473 train rows schema-checked |
| R001a | M0.5 | Native reference | current tm2 native head | JevBench public | accuracy, Brier, ECE, latency | MUST | DONE | standard/easy/hard acc .861/1.000/.486; p50 .122/.123/.197 s |
| R001b | M0.5 | Prompt/readout control | frozen Qwen3.5-4B direct option letters | same JevBench rows | accuracy, Brier, ECE, latency | MUST | DONE | acc .847/.979/.468; faster (.095/.062/.125 s) and better hard calibration, but lower accuracy |
| R001c | M0.5 | Pooling control | endpoint instead of mean-span option pooling | same held-out text | accuracy, NLL, Brier, ECE | MUST | DEFERRED | tm2's head was trained on mean-span features, so a frozen endpoint swap is invalid; direct readout showed no accuracy gap justifying a retrain |
| R002 | M1 | 200-step overfit/sanity | tm2 init, frozen LoRA, V/H/text mini mix | 128-row V/H pilots + capped text | loss, image sensitivity, retention | MUST | DONE | image/no-image/shuffled acc .906/.641/.617; MASSIVE-HE .797 (+.070); 64-suite text macro acc +.007; LoRA max diff 0; see `docs/research/MM_HE_SMOKE_20261004.md` |
| R003 | M2 | Visual upper baseline | direct Qwen3.5 letter scoring | ScienceQA test | acc, NLL, Brier, ECE | MUST | TODO | same render/options as Typical |
| R004 | M2 | Pre-training baselines | tm2, real/shuffled/no-image, Hebrew | V/H tests | all primary metrics | MUST | TODO | establishes deltas |
| R005 | M3 | E1 seed-0 gate | frozen LoRA, E/K/W/U/V/H mix | all | visual gap, null AUROC, text retention | MUST | TODO | stop if gate fails |
| R006 | M3 | E1 confirmation | R005 seed 1 | all | seed interval | MUST | TODO | conditional on R005 |
| R007 | M3 | E1 confirmation | R005 seed 2 | all | seed interval | MUST | TODO | conditional on R005 |
| R008 | M4 | Hebrew isolation | R005 with H=0, weight to E | H + English | paired deltas | MUST | TODO | matched seed/budget |
| R009 | M5 | Conditional capacity test | top-eight LoRA trainable | all | B2 metrics, cost | NICE | TODO | only if direct VLM passes and E1 does not |
