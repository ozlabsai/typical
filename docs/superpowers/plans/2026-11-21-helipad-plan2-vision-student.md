# Helipad Plan 2: Vision Student v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Gemma-4-12B vision Typical that answers the fixed frame questions about video keyframes of a landing site, with a real "can't tell". It is distilled from a Claude vision teacher, and its answers become video findings that feed Plan 1's decision model. v0 proves the whole chain on the 15 real videos; data volume comes in Plan 3.

**Architecture:**
- **Frames:** keyframes are extracted from each video with ffmpeg.
- **Teacher:** Claude vision through OpenRouter (CLI fallback) answers all frame questions per keyframe with a probability for every option. Answers are cached.
- **Training rows:** rows follow the R044/R056 grouped image format: one row per keyframe, all frame questions inside it, plus swapped-image and dropped-answer rows built from real ones.
- **Training:** the student is trained with the existing `scripts/train_gemma_decision_lora.py` on one rented H100. A generic pod runner, factored out of `scripts/helipad_score_pod.py`, handles upload, run, fetch and guaranteed deletion.
- **Merge:** frame answers are merged into video findings in the Plan 1 case text, and the text decision model scores real sites with and without video.

**Tech stack:**
- Python via `uv`, plus ffmpeg (installed locally) and Pillow.
- Gemma-4-12B-it at the revision pinned in `scripts/r055_pod.sh` / `scripts/hejev_pod_run.sh`.
- `scripts/train_gemma_decision_lora.py` and `scripts/gemma_decision_runtime.py`.
- `runpodctl` ≥ 3 through `scripts/typically_job.py` helpers.

**Spec:** `docs/superpowers/specs/2026-11-21-helipad-lz-decision-model-design.md` (§5 approach B, §6 frame questions and merge, §7 T-frame, §8 R2/R3). Plan 1 and its results: `docs/superpowers/plans/2026-11-21-helipad-plan1-decision-foundation.md`, `docs/research/HELIPAD_PLAN1_RESULTS.md`.

## Global Constraints

- **Privacy.** Videos, frames, teacher answers and the video-to-site mapping are private. They live only under `.context/` (gitignored), with paths from `helipad/paths.py` (`$HELIPAD_PRIVATE`). Code, tests and docs use the public site IDs (S1–S6, O1–O10) and synthetic fixture images. The repo `ozlabsai/typical` is public.
- **Always show every option with its probability.** Teacher answers, student outputs, eval records and reports keep the full distribution per question, never just the argmax.
- **"Can't tell" is a real option on every question.** Its text is R044's `NULL_OPTION`, "None of the options is supported by the image.". "Not seen" never means "absent". Low wires stay "not checked" in the decision model unless a wire is seen.
- **Split by video.** No frame from a test video, and no frame from the same video as a test frame, enters training. `wa_41.mp4` is byte-identical to `v01.mp4` and is dropped.
- **Spend.** One H100 per run; every pod is deleted on every exit path through the name-based, retried cleanup in `typically_job`. Teacher calls are cached by (image sha256, question-set version, teacher id), so reruns are free.
- **Teacher.** OpenRouter (user-provided `OPENROUTER_API_KEY` in the repo `.env`; read it as `scripts/r053_teacher_label.py` does) with a Claude vision model, sending the image as an OpenAI-style `image_url` data URI. `claude -p` is the fallback runner.
- **Success bars (spec R3, held-out videos):**
  - per-question agreement with the teacher's argmax ≥ 85%;
  - "can't tell" recall on swapped-image rows ≥ 0.8;
  - real − shuffled ≥ 10 points;
  - real − no-image ≥ 10 points.
  
  With only 15 videos these are reported as counts with n. They are feasibility signals, not claims.
- **Commits.** Commit after each task, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A video shorter than the keyframe count, or a still or black video, still gives distinct, valid keyframes and no crash. → `test_select_keyframes_short_or_static_video` (Task 1).
2. A teacher reply that is not JSON, misses a question, invents an option, or has probabilities that don't sum to 1 is rejected, retried once, then recorded as failed; it is never silently used. → `test_teacher_reply_validation` (Task 3).
3. A swapped-image row must never come from the same video as its text. → `test_mismatch_rows_never_reuse_the_same_video` (Task 5).
4. A killed training launcher leaves no pod. Covered by the generic runner's cleanup tests. → `test_runner_deletes_pod_on_error_and_signal` (Task 6).
5. Video findings without a compass get `bearing=None` and never a guessed direction; a height or distance class becomes a range, never a point value. → `test_video_findings_have_no_invented_direction_or_point_height` (Task 7).

---

### Task 1: Keyframes

**Files:** Create `scripts/helipad_frames.py`, `tests/test_helipad_frames.py`.

**Interfaces.** Produces:
- `select_keyframes(hists: list[list[float]], k: int) -> list[int]`: greedy farthest-point selection on normalised colour histograms. It starts from the frame nearest the median, returns sorted unique indices, and returns at most `min(k, len(hists))` of them.
- CLI `uv run scripts/helipad_frames.py --videos <dir> --out <frames_dir> [--fps 2 --k 16]`. It writes `<frames_dir>/<video_stem>/f<NN>.jpg` (long side 896 px) and `<frames_dir>/frames.json` = `[{"video": stem, "t": seconds, "path": rel, "sha256": hex}]`. It skips any video whose file sha256 equals an earlier one and prints the skip.

- [ ] **Step 1: tests.**
  - `test_select_keyframes_spreads_over_distinct_scenes`: three clusters of 5 identical histograms with k=3 gives one index per cluster.
  - `test_select_keyframes_short_or_static_video`: 2 frames with k=16 gives `[0, 1]`; 10 identical frames with k=4 gives 4 distinct indices without crashing.
  - `test_duplicate_videos_are_skipped`: two byte-identical tiny mp4 files, made with ffmpeg `testsrc` in tmp_path and skipped if ffmpeg is missing, yield frames for one video only.
- [ ] **Step 2: run them and watch them fail.** `uv run pytest tests/test_helipad_frames.py -q`.
- [ ] **Step 3: implement.**
  - ffmpeg decodes at `--fps` into a temp dir.
  - Histograms are 8×8×8 RGB via Pillow, L1-normalised.
  - Keep the selected frames and resize them with Pillow.
- [ ] **Step 4: tests pass.** Then run it on the real footage:

  `uv run scripts/helipad_frames.py --videos .context/helipads/raw --out .context/helipads/frames`

  Expected: 15 videos, wa_41 skipped, about 16 keyframes each.
- [ ] **Step 5: commit.**

---

### Task 2: Frame question set and grouped rows

**Files:** Create `helipad/frames.py`, `tests/test_helipad_frames_rows.py`.

**Interfaces.** Produces:
- `QSET_VERSION = "v0"`.
- `NULL_OPTION = "None of the options is supported by the image."`.
- `FRAME_QUESTIONS: dict[str, tuple[str, tuple[str, ...]]]`, mapping id → (question, options without null), exactly these:
  - `wires`: "Are overhead wires or cables visible?" — yes / no
  - `pole`: "Is a power or lighting pole visible?" — yes / no
  - `tall_structure`: "Is a tall mast, tower or crane visible on the horizon?" — yes / no
  - `tallest_height`: "Height of the tallest object in view?" — below 1.2 m / 1.2–4.5 m / 4.5–24 m / above 24 m
  - `tallest_distance`: "Distance from the camera to that object?" — below 10 m / 10–50 m / 50–300 m / above 300 m
  - `tallest_type`: "What is the tallest object in view?" — tree / building / wall or fence / pole / vehicle / none
  - `ground`: "Ground surface in the foreground?" — paved / hard earth / stony / ploughed / vegetation / powdery
  - `vegetation_belly`: "Is foreground vegetation taller than 10 cm?" — yes / no
  - `dust`: "Likely dust if a helicopter lands here?" — light / medium / heavy
- `grouped_row(frame_id: str, image_rel: str, targets: dict[str, dict[str, float]] | None) -> dict`. This is the R044 schema: `{id, state, questions: {qid: {type: "choice", instructions, criteria: {option_00…, null}}}, targets: {qid: {...}}, image, meta: {qset: QSET_VERSION}}`.
  - The state is one short fixed context line: "A ground photo taken at a candidate helicopter landing site."
  - Targets use the soft-target field that `scripts/r056_build.py` writes for teacher probabilities. Read it there, and use the same field name and shape.
  - The null option's target key is `null`.
- [ ] **Step 1: tests.**
  - `test_questions_match_spec_and_every_row_has_null`.
  - `test_grouped_row_targets_sum_to_one_and_use_option_keys`.
  - `test_grouped_row_loads_in_the_gemma_trainer`. Import the trainer's row loader or validator from `scripts/train_gemma_decision_lora.py` (find the function that reads rows, near line 135–145) and run it on a row pointing at a 32×32 PNG fixture written in tmp_path.
- [ ] **Steps 2–5:** watch the tests fail, implement, watch them pass, commit.

---

### Task 3: Claude vision teacher

**Files:** Create `scripts/helipad_frame_teacher.py`, `tests/test_helipad_frame_teacher.py`.

**Interfaces.**
- Consumes Task 1's `frames.json` and Task 2's `FRAME_QUESTIONS` / `NULL_OPTION`.
- Produces:
  - `prompt(questions) -> str`. It asks for JSON `{qid: {option: probability}}` covering every option including `NULL_OPTION`, with probabilities summing to 1, and the instruction "use the null option when the image does not show it".
  - `validate(reply: str, questions) -> dict | None`. It parses the first JSON object; it requires every qid and only known options, renormalises when the total is within 0.95–1.05, and returns None otherwise.
  - `cache_key(sha256, qset, teacher) -> str`.
  - `label(frames, cache_dir, runner) -> list[dict]`. `runner(image_path, prompt) -> str` is injectable. A reply that fails validation is retried once, then recorded as `{"failed": true}`.
- Runners:
  - **OpenRouter (default):** POST to `r053_teacher_label.OPENROUTER` with content `[{type:"image_url", image_url:{url:"data:image/jpeg;base64,…"}}, {type:"text", text: prompt}]`, the current best Claude vision model on OpenRouter, and the key loader from r053. Log the cost from the response's `usage`.
  - **CLI (fallback):** `claude -p --model <teacher> --allowedTools Read --output-format text` with the prompt plus "Read the image at <absolute path>". If `claude -p` supports image input through `--input-format stream-json`, use that instead and say so.
- CLI: `uv run scripts/helipad_frame_teacher.py --frames .context/helipads/frames/frames.json --out .context/helipads/teacher/frames_v0.jsonl [--limit N --teacher <id>]`. It prints the cost or time per frame.

- [ ] **Step 1: tests (fake runner).**
  - `test_prompt_lists_every_option_and_null`.
  - `test_teacher_reply_validation`: non-JSON gives None; a missing qid gives None; an unknown option gives None; a sum of 0.97 gets renormalised; a sum of 0.5 gives None.
  - `test_cache_makes_reruns_free`: the second `label()` call makes zero runner calls.
  - `test_failed_after_one_retry_is_recorded`.
- [ ] **Steps 2–4:** watch them fail, implement, watch them pass.
- [ ] **Step 5: pilot.** Run `--limit 5` on real frames with the CLI runner. Expected: 5 valid records. Eyeball one frame against its answers: does `tallest_type` / `wires` match the picture? Record the result in the ledger.
- [ ] **Step 6: full run** on all keyframes (about 240). Expected: at most 2% failed.
- [ ] **Step 7: commit** (code only).

---

### Task 4: Video-to-site mapping (private)

**Files:** Create `scripts/helipad_match_videos.py`, `tests/test_helipad_match_videos.py`.

**Interfaces.**
- Produces `contact_sheet(frames: list[Path], out: Path, cols=6)`, a Pillow tile.
- CLI that writes `$HELIPAD_PRIVATE/videos.json` = `{video_stem: {"site": public_id | null, "confidence": "high"|"low"|"none", "evidence": str}}`.
  - For each video it asks the teacher runner (Task 3), given the contact sheet plus the private site descriptions, which site this is. The descriptions are the deck's per-site lines in the private brief (`.context/helipads/BRIEF.md`) and `sites.json` names.
  - It prints a table for the user to confirm, and edits nothing automatically after the first run.
- [ ] **Step 1: tests.**
  - `test_contact_sheet_tiles_all_frames`: 7 fixture images with cols=6 give a 6×2 grid.
  - `test_mapping_schema`.
- [ ] **Steps 2–4:** watch them fail, implement, watch them pass. Run it, show the table to the user, and record their corrections in `videos.json`.
- [ ] **Step 5: commit.**

---

### Task 5: Vision training rows

**Files:** Create `scripts/helipad_vision_build.py`, `tests/test_helipad_vision_build.py`.

**Interfaces.**
- Consumes Tasks 1–3, plus Task 4 for split stratification only.
- Produces the dir `.context/helipads/vision_v0/` containing `train.jsonl`, `dev.jsonl`, `test_real.jsonl`, `test_shuffled.jsonl`, `test_noimage.jsonl`, `test_mismatch.jsonl` and `images/`.
- Splits are by video and seeded: test is 3 videos, dev 2, train the rest. Prefer one test video per round when Task 4 knows the rounds.
- **train / dev:** the R044 thirds, built per frame:
  - **real:** teacher soft targets;
  - **mismatch:** an image from a *different video*, every question's target is null;
  - **gold-dropped:** one question's teacher-argmax option removed, so that question's target is null and the others keep the teacher's.
- **test_* (from test videos only):**
  - `real` and `mismatch` built the same way;
  - `shuffled`: an image from another test video with the original targets, the same idea as `eval_gemma_mm_he.py`'s suite;
  - `noimage`: the row with its image field removed.
- `split_by_video(frames, seed) -> dict[str, list]`.

- [ ] **Step 1: tests.**
  - `test_no_video_in_two_splits`.
  - `test_mismatch_rows_never_reuse_the_same_video`.
  - `test_gold_dropped_targets_null_only_for_the_dropped_question`.
  - `test_noimage_rows_have_no_image_field`.
- [ ] **Steps 2–5:** watch them fail, implement, watch them pass, build, commit.

---

### Task 6: Generic pod runner, and training on one H100

**Files:**
- Create `scripts/helipad_pod.py` (generic) and `scripts/helipad_vision_train.py`.
- Modify `scripts/helipad_score_pod.py` to use `helipad_pod.run`.
- Tests: `tests/test_helipad_pod.py`; keep `tests/test_helipad_score_pod.py` green.

**Interfaces.** Produces `helipad_pod.run(name_tag, uploads: dict[Path, str], remote_script: str, fetch: dict[str, Path], gpu="NVIDIA H100 80GB HBM3", disk_gb=150, timeout_s=...) -> None`. This is the logic of `helipad_score_pod.py`, moved here:
- name-based, retried deletion via `typically_job.Job`;
- SIGINT, SIGTERM and SIGHUP raise into a `finally`;
- `pod get` ssh polling;
- one tarball up;
- the remote script runs with `set -eo pipefail`;
- scp back.

`helipad_score_pod.py` becomes a thin caller.

`helipad_vision_train.py` uploads the repo plus `vision_v0/` and runs remotely:
- the pinned setup from `scripts/r055_pod.sh` `rsetup`, narrowed to the pinned venv plus the Gemma download at the pinned revision;
- `torchrun --standalone --nproc_per_node 1 -m scripts.train_gemma_decision_lora --model <gemma path> --train vision_v0/train.jsonl --val dev=vision_v0/dev.jsonl --out out/vision_v0 --steps <S> --microbatch 1 --grad-accum 8 --rank 32 --alpha 64 --lr 5e-5 --min-lr 5e-6 --max-length 4096 --eval-every 100 --save-every 100 --seed 56`.

  Flags follow R056 (`scripts/r056_pod.sh:67-77`), minus the text-twin options. The step count is about 3 epochs of train, printed by the build.
- Then evaluation (Task 7) on the pod, and fetching `out/vision_v0/{adapter,eval}`.

- [ ] **Step 1: tests.**
  - `test_runner_deletes_pod_on_error_and_signal` (port the existing scoring-pod tests to `helipad_pod.run`).
  - `test_score_pod_uses_the_generic_runner`.
- [ ] **Steps 2–4:** watch them fail, implement, watch them pass; `uv run pytest tests/test_helipad_pod.py tests/test_helipad_score_pod.py tests/test_typically_job.py -q` must be green.
- [ ] **Step 5: train.** The user approved this run, a separate pod from Plan 1's 3,200-step run: about 1–2 h on one H100. Watch it on W&B if the trainer logs there; otherwise tail the log over ssh.
- [ ] **Step 6: commit.**

---

### Task 7: Evaluation, video findings, and end-to-end on real sites

**Files:** Create `scripts/helipad_vision_eval.py`; modify `helipad/frames.py` (`video_findings`). Tests: `tests/test_helipad_vision_eval.py`, `tests/test_helipad_frames_rows.py`.

**Interfaces.**
- `vision_report(rows, preds) -> dict`, where `preds[i] = {qid: {option: p}}`. It returns:
  - per-question agreement with the teacher's argmax;
  - `null_recall` on mismatch rows;
  - `real_minus_shuffled`, `real_minus_noimage`;
  - answerability AUROC on P(null);
  - ECE per question;
  - n for every figure.

  It runs on the pod: the base model and the adapter, through `gemma_decision_runtime.py` with the image. Every option's probability is saved.
- `video_findings(frame_preds: list[dict], video_stem) -> tuple[list[Obstacle], dict]`. This is spec §6's merge rule:
  - presence (wires, pole, tall structure) = the maximum P(yes) over frames, with the frame timestamps;
  - the tallest object's height and distance classes become `Obstacle` entries with `source="video"`, `bearing=None` and `height_src="estimated"`. Height and distance are the midpoint of the confidence-weighted class range, and the label states the range ("height 4.5–24 m (video)").
  - ground, dust and vegetation become `dust` / `belly` values.
- `helipad_vision_eval.py --e2e`: for each real site in `videos.json`, add the video findings to its `case_from_lz` case and score it with the Plan 1 decision checkpoint (via `helipad_score_pod.py`), with and without video. It reports the verdict-of-record changes, expert direction on S1–S6 where a video exists, and every option's probability.

- [ ] **Step 1: tests.**
  - `vision_report` on hand-built preds: each metric, including n = 0 giving None.
  - `test_video_findings_have_no_invented_direction_or_point_height`.
  - `test_wire_seen_in_video_becomes_a_wire_obstacle`.
- [ ] **Steps 2–4:** watch them fail, implement, watch them pass.
- [ ] **Step 5: run.** The eval runs as part of Task 6's pod run. Then the end-to-end scoring (one scoring pod).
- [ ] **Step 6: write the results** in `docs/research/HELIPAD_PLAN2_RESULTS.md` (public IDs only, no images): the bars versus results, base versus tuned, and the end-to-end table. Commit.
