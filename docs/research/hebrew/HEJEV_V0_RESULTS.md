# Hebrew JevBench v0: results (2026-10-20)

> **EVAL-ONLY.** Nothing here may be used for training, distillation or training-data selection.

Scores come from `scripts/hejev_score.py` (argmax accuracy with invalid
counted as wrong; jevbench Brier; top-label ECE over 10 bins). EN is the public 231. HE is `translated` (230), MX is
`mixed` (230) and native is `native` (119, gold = teacher-consensus argmax, so its accuracy is accuracy against the
soft gold). Twin columns give Δacc (HE−EN or MX−EN, on the 230 pairs) / prediction agreement / mean JSD (log2).
The full reports are in `<model>/score.json`.

## Decision models, 9–14B (apples-to-apples)

Only single-pass decision models of 9-14B, each served through its own released readout, on the same jevbench runner,
adapter and pod type. Accuracy columns are EN / HE translated / MX mixed / native. Brier / ECE are given for EN and
HE. Gap, agreement and JSD are HE vs EN on the 230 twin pairs. Sorted by HE accuracy.

| model (run id) | params | base | EN | HE | MX | native | EN Brier / ECE | HE Brier / ECE | HE−EN | agree | JSD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| StartLux-Decision-v1.1-9B-Instant, bidirectional letter readout (`StartLux-Decision-v1.1-9B-Instant-18a17f7`) | 9.7B | Qwen3.5-9B | **.896** | **.878** | **.874** | **.916** | **.134 / .035** | **.163 / .036** | −.017 | .935 | **.016** |
| Torchcast Decision 12B, BF16, letter readout (`torchcast-decision-12b`, rev 49107b5) | 12B | Gemma-4-12B-it (LoRA, merged) | .879 | .865 | .865 | .899 | .165 / .051 | .216 / .064 | **−.013** | **.939** | .036 |
| Winnow-12B Q8_0 (`Winnow-12B`, label `hejev-winnow-12b-*`) | 12B | Gemma-4-12B-it (LoRA) | .853 | .865 | .843 | .840 | .206 / .065 | .213 / .076 | +.013 | .939 | .034 |
| Cygnet: frozen Gemma-4-12B-it, letter readout, T=3.4 (`cygnet-gemma-4-12B-it-707f0a3-T3.4`) | 12B | Gemma-4-12B-it (frozen) | .879 | .839 | .839 | .882 | .173 / .026 | .216 / .049 | −.043 | .904 | .027 |
| Typical R044-R, native readout (`typical-r044r-sha256-198185cfb418`) | 12B | Gemma-4-12B-it (LoRA) | .853 | .830 | .843 | .866 | .195 / .062 | .241 / .090 | −.022 | .913 | .038 |
| Cloudflare Clef-Flash, joint schema head (`clef-flash-f0b7d3c`) | 9.4B | Qwen3.5-9B | .823 | .774 | .752 | .782 | .235 / .062 | .297 / .072 | −.052 | .878 | .036 |

Native NLL vs soft gold: StartLux .584, Cygnet .658, Clef .767, R044-R .870, Torchcast .900, Winnow .911.
MX−EN Δacc / agreement: StartLux −.022 / .939, Torchcast −.013 / .948, Winnow −.009 / .926, Cygnet −.043 / .913,
R044-R −.009 / .913, Clef −.074 / .887.

Peer readout:
- StartLux 9B Instant leads every accuracy and calibration column. Its HE Brier (.163) is lower than every other
  peer's EN Brier. Its Hebrew penalty is small (−1.7, hard −0.9) and its EN/HE probability vectors are the closest (JSD .016).
- Torchcast 12B reproduces its card's 203/231 exactly (.879) and loses only 1.3 points on HE. It matches Cygnet on EN
  with the same base model and readout family, but holds up much better in Hebrew (Cygnet −4.3).
- Clef-Flash is the weakest peer here and degrades the most: −5.2 on HE and −7.4 on MX (original −6.9, hard −6.4,
  easy 0). Its card lists English benchmarks only.
- Typical R044-R is fifth of six on HE accuracy and last on HE Brier/ECE (.241 / .090).

## Reference (not peers)

General LLMs with thinking, and a 3B decision model. Same files, runner and scorer, but not comparable as peers.
Each cell is accuracy / Brier / ECE.

| model (run id) | EN | HE translated | MX mixed | native | native NLL vs soft gold | HE−EN Δacc / agree / JSD | MX−EN Δacc / agree / JSD |
|---|---|---|---|---|---|---|---|
| Gemma-4-12B-it thinking (`vllm:google/gemma-4-12B-it`) | .974 / .039 / .029 ᴾ | .952 / .099 / .055 | .987 / .034 / .031 | .933 / .132 / .063 | 1.626 | −.022 / .952 / .044 | +.013 / .974 / .027 |
| Qwen3.8-27B-FP8 thinking (`vllm:Qwen/Qwen3.8-27B-FP8`) | .991 / .026 / .019 ᴾ | .996 / .018 / .016 | .996 / .018 / .016 | .966 / .071 / .050 | .832 | +.004 / .996 / .008 | +.004 / .996 / .007 |
| LiquidAI d1-3B@051bcc4 native (`d1-3B-051bcc4`) | .753 / .349 / .108 ᴾ | .683 / .425 / .084 | .661 / .454 / .103 | .605 / .492 / .121 | 1.015 | −.074 / .822 / .078 | −.096 / .791 / .080 |

ᴾ marks an English run reused from R054 Phase 0 (`.context/r054-phase0/<run>/jevbench/`). It has the same jevbench
commit bb05a335, the same `r054_systemone_server.py` path, the same vLLM 0.31.0 and the same model id, and the
scorer reproduces its counts (225/229/174). The other EN runs were rerun in this session through the same server as
the Hebrew runs.

Readout:
- Qwen-think shows no Hebrew penalty (0 EN→HE flips, agreement .996).
- d1-3B degrades the most: −7.4 points on HE and −9.6 on MX. Native is its worst split (.605). Hebrew is not
  among its card languages.
- The single-pass Gemma-12B models lose 2-4 points on HE (Cygnet −4.3, R044-R −2.2), most of it on hard items
  (Cygnet hard −6.4).
- Winnow is flat to slightly positive (+1.3, 8 EN-wrong→HE-right vs 5 the other way).
- Gemma-think gains on MX (+1.3), and Cygnet loses as much on MX as on HE. So the Hebrew state hurts as much as a
  Hebrew question does.
- Native-item calibration is worst for the thinking-Gemma run. Its NLL against the soft gold is 1.63 because it puts
  near-zero mass on options the teachers keep at 1-10%; the NLL floors probabilities at 1e-6. Cygnet's tempered
  letter readout has the best native NLL (.658).

## Protocol notes and deviations

**Decision peers (StartLux, Torchcast, Clef), second pod.** One H100 80GB HBM3, pod `798zfnc7mwo41i`, $3.99/h, up
19:16:15 → 20:12:28 UTC = **0.94 GPU-h, about $3.7**. It is now EXITED. Runner, adapter, chunking and scorer are the
same as above (`psetup`, `clef`, `startlux`, `torchcast` in `scripts/hejev_pod_run.sh`). Results are in
`<model>/` (`clef-flash`, `startlux-9b-instant`, `torchcast-12b`); pod logs are in `peers/`. Each model was served
the way its own card or code says:
- **StartLux-Decision-v1.1-9B-Instant** @18a17f7 (CC BY-NC 4.0). `startlux-decide serve --launch-vllm` from
  StartLuxLabs/StartLux-Decision@83fe6c7 on vLLM 0.31.0, with the launcher's own flags (prefix caching and chunked
  prefill off, 131k context). The vLLM log confirms "prefix-bidirectional prompt attention ON". The card does not
  name AutoThink as the recommended mode, so the AutoThink variant was not run.
  - Incident: the first start aborted after 3 s. The launcher's default vLLM port 8001 is taken by RunPod's nginx,
    which answered `/health`, and the launcher's plugin check then refused to serve. It was rerun with
    `--vllm-base-port 8101`. No item was answered by the failed start.
- **Torchcast Decision 12B** (CC BY-NC 4.0). The task named `mradermacher/torchcast-decision-12b-GGUF` Q8 via
  llama.cpp. That repo is a third-party quant with no inference spec. The original card says that generic examples do
  not reproduce the evaluated interface, and its shim needs vLLM (`structured_outputs` letter constraint and
  `/v1/models` `max_model_len`). So the original BF16 weights were run exactly as its serving README says:
  `torchcast-ai/torchcast-decision-12b@49107b5` (weight sha256 f8553c9e… verified), vLLM 0.30.0, and
  `serving/torchcast_shim.py` from the GitHub tag v1.0.0, with no `SHIM_*` overrides beyond those shown. The HF tag
  v1.0.0 points at a47d9f4, but the serving README pins 49107b5. The served name is fixed (`torchcast-decision-12b`),
  so the run is told apart by its directory and label. EN 203/231 matches the card's 203.
  - First pass: 11 `ConnectionReset` errors in 10 chunks, the same listen-backlog issue as Cygnet. Each chunk was
    rerun once and all of them passed. The readout is a deterministic logprob read, so the rerun does not resample.
- **Clef-Flash** @f0b7d3c (Apache-2.0). Its release code `joint_schema_model.systemone()` (joint schema head, bf16)
  with transformers 5.10.2 as on the card, behind `r054_systemone_server.py clef`. The card's SGLang image needs
  Docker, which the pod does not have. flash-linear-attention was installed, causal-conv1d was not (torch conv
  fallback), and one forward runs at a time, so it took 33 min. Speed does not change the answers. `encode_record`
  truncates the state at 16,384 tokens, but no item came near that.

- **Single pod.** One H100 80GB HBM3, pod `0kajvwon98ar1n`, $3.99/h, up 17:29:37 → 19:08:32 UTC = **1.65 GPU-h,
  about $6.6**. It is now EXITED.
- **Runner.** Stock `jevbench.cli run` with the `typesafe` adapter at commit bb05a335, and the client timeout raised
  to 3600 s (as in Phase 0). Each file was run in parallel chunks of 12 by `scripts/hejev_pod_run.sh`, and every
  chunk was sent unchanged.
- **Retries (deviation).** The first pass had transient transport failures:
  - Cygnet: 2 `ConnectionReset`, caused by a 70-connection burst against a ThreadingHTTPServer backlog of 5.
  - Gemma-think: 9 failures (connection errors plus 1 hit the 16,384-token output cap).
  - Qwen-think: 4 connection errors.
  Every chunk that had a failed row was rerun whole, at most twice. That re-samples all 12 items of the chunk at
  T=1, not only the failed one.
  - One Gemma-think item still fails, `mx-hard-opus-c-temporal_numeric-04`, which hits the output cap. It counts
    as wrong. Several `temporal_numeric` items hit the cap at least once on HE and MX.
  - First-pass numbers (snapshot in `first-pass/`):
    - Gemma-think: HE .948 / Brier .064, MX .978, native .933.
    - Qwen: MX .983, native .958.
    - Cygnet: HE .830.
  - The swing in Gemma-think's HE Brier (.064 → .099, from re-sampling 5 chunks) shows that run-to-run noise for
    the T=1 thinking models is on the order of the HE−EN gaps.
- **Cygnet.** vLLM 0.31.0 instead of the 0.30.0 used in G0, with cygnet-recipe pinned at 10b4099, the HEAD when G0
  ran. The EN rerun gives 203/231, Brier .173, an exact accuracy match with G0 (`.context/g0-cygnet`, Brier .171).
  The run is served under a distinct name via `CYGNET_MODEL_NAME`.
- **R044-R.** The checkpoint is `.context/r044-rebuild/b/step-000100`, and the adapter_model sha256 3b5535de…
  matches `checkpoint.sha256`. It was shipped as a 3-file tree (README, config, weights; no trainer.pt), tree
  sha 198185cf…, and served through `serve_typesafe_gemma_lora.py` with base gemma-4-12B-it@707f0a3, max context
  16384 and bf16/sdpa. The startup receipt is in `r044r/`.
  - The existing EN run `.context/official-public-jevbench-v2` is **not comparable**. It is the original R044
    checkpoint (tree 1493c114…), run through `jevbench_gemma_lora.py` rather than the server. That run scored 196;
    the R044-R rerun here scores 197.
- **Winnow.** winnow-inference at 6c2b3c0 with the Q8_0 GGUF (sha verified) and the same flags as G0. The EN rerun
  gives 197/231 (.853), against G0's 199 (.861). The llama.cpp server with `--decision-parallel 4` is not bitwise
  deterministic. Paired gaps use the same-session EN run. Its served model name is fixed (`Winnow-12B`), so the run
  is told apart by its label and directory.
- **Qwen3.8-27B-think.** All three Hebrew files ran in full, so no cut was needed. That took about 20 minutes at
  roughly 1k prompt + 1k generation tok/s.
- **Native gold.** It is Claude-teacher consensus (fable-5-1 + opus-5-5) and not human-verified (see
  `hejev-v0/README.md`). "Accuracy vs soft gold" means argmax agreement with the consensus. NLL is
  −Σ gold·log p with p floored at 1e-6, over valid rows.
- **Context.** All runs used the existing context limits (16k for the single-pass models, 64k for thinking). No
  Hebrew item was rejected for length; the valid rate is 1.000 everywhere except the one Gemma-think output-cap
  item.

Files:
- `<model>/<split>/NNN.jsonl`: per-item rows, one file per chunk.
- `<model>/<split>/{raw,ledger,manifest,logs}/`
- `<model>/score.json`
- `logs/`, `chain.log`, `retry.log`
- Runner: `scripts/hejev_pod_run.sh`.

---
<sup>†</sup> Reference context from `../SOURCES_BENCHMARKS.md`. These are **different tasks and are not directly
comparable**:
- On AlephBench (11 Hebrew tasks: MC, NLI, sentiment, QA; letter-parsed accuracy, no calibration), the 2026-05-11
  averages are gemma-4-31b-it 88.1, gemini-2.5-flash 88.8, DictaLM-3.0-24B-Thinking 85.2 and gemma-3-27b-it 83.5.
  There is no Gemma-4-12B or Qwen entry.
- On the Hebrew LLM Chat Leaderboard (Winograd / Trivia, 0-shot):
  - gemma-4-31B-it: 86.0 / 58.5
  - gemma-4-26B-A4B-it: 78.1 / 44.5
  - Qwen3.5-122B-A10B: 93.2 / 70.1
- No decision model (Jev, Clef, d1, Winnow, Cygnet) has a published Hebrew result. This table is the first
  Hebrew decision measurement with Brier/ECE that we know of.
