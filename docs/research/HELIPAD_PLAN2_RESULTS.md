# Helipad Plan 2 results: vision student v0

Plan: `docs/superpowers/plans/2026-11-21-helipad-plan2-vision-student.md`. Spec: `docs/superpowers/specs/2026-11-21-helipad-lz-decision-model-design.md` (§6, §8 R3).
Date: 2026-11-22. Public site IDs only; frames, teacher answers and the video-to-site mapping stay private.

## What ran

- **Frames.** 15 unique videos (one byte-identical duplicate skipped), 16 keyframes each, so 240 frames. Selection is farthest-point on colour histograms, at 2 fps.
- **Teacher.** `anthropic/claude-fable-5.2` on OpenRouter, reasoning effort low. It answered the 9 frame questions with every option's probability, null ("None of the options is supported by the image.") included. All 240 frames were valid with 0 failures, at $0.031 per frame. On real frames the teacher never put its argmax on null.
- **Split by video (seed 0).** 10 videos for train, 2 for dev, 3 for test, with one test video from each site round. Train has 480 rows (4,320 question examples): for each frame a real row, a mismatch row and a gold-dropped row. Each test suite has 48 rows (432 question examples).
- **Student.** Gemma-4-12B-it (pinned revision) with a LoRA, R056 flags: rank 32, alpha 64, lr 5e-5 to 5e-6, grad-accum 8, seed 56. It trained for 1,620 steps (3 epochs) on one H100 in 95 min. The checkpoint was the saved step with the lowest dev NLL, which was step 600 (dev NLL 0.678). Base and tuned were scored through the trainer's own prompt and image encoding.

**Deviation: mismatch rows.** The plan's mismatch row pairs a frame from another video with all-null targets. The frame questions are generic ("Are overhead wires visible?"), so a frame from another video still answers them. That row would be a second real image with a null target, which trains P(null) ≈ .5 on real frames. Instead, a mismatch row here shows a heavily blurred frame (Gaussian radius 6% of the long side) from a different video, with all targets null. This is the "can't tell" a phone or drone video really produces. As a result, the can't-tell bar below measures recall on blurred frames. That is an easier test than R044's text/image mismatch.

## Bars vs results (held-out videos, n = 48 frames × 9 questions)

| Bar (spec R3) | Base Gemma | Tuned | Pass? |
|---|---|---|---|
| Per-question agreement with the teacher's argmax ≥ 85% | overall .789 (341/432) | overall **.900** (389/432); 7 of 9 questions ≥ .85 | overall yes; tallest_distance (.812) and dust (.750) miss |
| Can't-tell recall on mismatch (blurred) rows ≥ 0.8 | .757 | **1.000** (432/432) | yes (easier test, see above) |
| Real − shuffled ≥ 10 points | +35.0 | **+38.4** (.900 vs .516) | yes |
| Real − no-image ≥ 10 points | +78.9 | **+90.0** (no image: null argmax on 432/432) | yes |
| Answerability AUROC of P(null), mismatch vs real | .971 | 1.000 | — |

These are feasibility signals from 3 test videos, not claims.

### Per question, base vs tuned (test_real, n = 48 each)

| question | majority baseline* | base | tuned | tuned shuffled | tuned null recall (blurred) | base ECE | tuned ECE |
|---|---|---|---|---|---|---|---|
| wires | .812 | .979 | .979 | .604 | 1.000 | .028 | .068 |
| pole | .812 | .833 | .958 | .583 | 1.000 | .152 | .063 |
| tall_structure | .917 | .833 | .958 | .875 | 1.000 | .137 | .061 |
| tallest_height | .771 | .771 | .896 | .708 | 1.000 | .166 | .122 |
| tallest_distance | .792 | .667 | .812 | .667 | 1.000 | .267 | .137 |
| tallest_type | .708 | .833 | .896 | .542 | 1.000 | .148 | .137 |
| ground | .333 | 1.000 | .938 | .000 | 1.000 | .008 | .275 |
| vegetation_belly | .667 | .896 | .917 | .292 | 1.000 | .094 | .101 |
| dust | .500 | .292 | .750 | .375 | 1.000 | .647 | .206 |
| **overall** | .701 | .789 | **.900** | .516 | 1.000 | | |

\*The majority baseline is the test set's own most common teacher answer, so it is optimistic.

What tuning bought:
- Calibration to the teacher's soft answers: dust went from .292 to .750, and its ECE from .647 to .206.
- Distance and height classes improved.
- Pole and tall-structure presence improved.

What it cost: ground dropped from 1.000 to .938, and ground ECE rose from .008 to .275. Wires were already at ceiling. Base Gemma already says null without an image.

### One test frame, every option's probability

| question | source | options |
|---|---|---|
| wires | teacher | yes .02 · no .95 · null .03 |
| | base | yes .00 · no 1.00 · null .00 |
| | tuned | yes .02 · no .96 · null .02 |
| tallest_distance | teacher | below 10 m .00 · 10–50 m .01 · 50–300 m .32 · above 300 m .65 · null .02 |
| | base | below 10 m .00 · 10–50 m .00 · 50–300 m .83 · above 300 m .02 · null .14 |
| | tuned | below 10 m .01 · 10–50 m .02 · 50–300 m .26 · above 300 m .70 · null .02 |
| dust | teacher | light .03 · medium .30 · heavy .66 · null .01 |
| | base | light .00 · medium .01 · heavy .99 · null .00 |
| | tuned | light .06 · medium .30 · heavy .63 · null .01 |
| dust (blurred frame) | tuned | light .00 · medium .00 · heavy .00 · null 1.00 |
| wires (no image) | tuned | yes .00 · no .00 · null 1.00 |

## End to end: video findings into the Plan 1 decision model

**Setup.** Each real site with a matched video was scored by the Plan 1 decision checkpoint (800 steps) on one pod, three times:
- *none*: map findings only;
- *student*: plus the student's video findings;
- *teacher*: plus the teacher's video findings, for comparison.

**Merge.** `helipad.frames.video_findings` turns frame answers into findings:
- A wire seen in any frame (max P(yes) ≥ .5) becomes "overhead wire, height and distance unknown, treated as within 50 m".
- Each distinct tallest-object (type, height class, distance class) becomes an estimated-height obstacle. Its direction is unknown, and its label states the class ranges.
- Dust is the worse of map and video. Belly comes from video only.

**Mapping caveat.** The video-to-site mapping is the teacher's proposal (10 high and 5 low confidence) and the user has not confirmed it.

**Columns.**
- "Split" is the split of the site's video(s) in vision training. Student findings on train and dev videos come from frames the student was fitted on.
- "Rules" is the rule engine's verdict on that case.
- "Record" is the verdict derived from the model's own item answers.
- The full verdict distribution is from the model's verdict question.

| Site | Split | Video findings | Rules | Record | Verdict question: every option | Record changed vs none | Expert |
|---|---|---|---|---|---|---|---|
| O1 | train | none | approved | approved | approved .71 · probably fails .25 · fails .03 · not checked .00 | | — |
| O1 | train | student | probably fails | probably fails | approved .27 · probably fails .60 · fails .14 · not checked .00 | yes | — |
| O1 | train | teacher | probably fails | probably fails | approved .22 · probably fails .61 · fails .17 · not checked .00 | yes | — |
| O2 | train | none | approved | approved | approved .73 · probably fails .24 · fails .03 · not checked .00 | | — |
| O2 | train | student | probably fails | approved | approved .34 · probably fails .57 · fails .10 · not checked .00 | no | — |
| O2 | train | teacher | probably fails | approved | approved .45 · probably fails .46 · fails .09 · not checked .00 | no | — |
| O3 | test | none | approved | approved | approved .83 · probably fails .14 · fails .03 · not checked .00 | | — |
| O3 | test | student | approved | approved | approved .20 · probably fails .35 · fails .45 · not checked .00 | no | — |
| O3 | test | teacher | probably fails | probably fails | approved .32 · probably fails .30 · fails .38 · not checked .00 | yes | — |
| O4 | test | none | approved | approved | approved .60 · probably fails .34 · fails .06 · not checked .00 | | — |
| O4 | test | student | probably fails | approved | approved .47 · probably fails .49 · fails .04 · not checked .00 | no | — |
| O4 | test | teacher | probably fails | probably fails | approved .27 · probably fails .64 · fails .10 · not checked .00 | yes | — |
| O5 | dev | none | fails | fails | approved .40 · probably fails .42 · fails .17 · not checked .00 | | — |
| O5 | dev | student | fails | fails | approved .35 · probably fails .42 · fails .24 · not checked .00 | no | — |
| O5 | dev | teacher | fails | fails | approved .19 · probably fails .43 · fails .37 · not checked .00 | no | — |
| O6 | train | none | approved | approved | approved .32 · probably fails .48 · fails .21 · not checked .00 | | — |
| O6 | train | student | probably fails | approved | approved .25 · probably fails .65 · fails .09 · not checked .00 | no | — |
| O6 | train | teacher | probably fails | approved | approved .20 · probably fails .66 · fails .13 · not checked .00 | no | — |
| O7 | train | none | fails | probably fails | approved .24 · probably fails .30 · fails .45 · not checked .00 | | — |
| O7 | train | student | fails | fails | approved .17 · probably fails .18 · fails .65 · not checked .00 | yes | — |
| O7 | train | teacher | fails | fails | approved .17 · probably fails .18 · fails .65 · not checked .00 | yes | — |
| O9 | train | none | probably fails | probably fails | approved .32 · probably fails .40 · fails .28 · not checked .00 | | — |
| O9 | train | student | probably fails | probably fails | approved .22 · probably fails .68 · fails .10 · not checked .00 | no | — |
| O9 | train | teacher | probably fails | probably fails | approved .26 · probably fails .59 · fails .14 · not checked .00 | no | — |
| O10 | train | none | not checked | not checked | approved .00 · probably fails .00 · fails .20 · not checked .80 | | — |
| O10 | train | student | not checked | not checked | approved .00 · probably fails .01 · fails .34 · not checked .65 | no | — |
| O10 | train | teacher | not checked | not checked | approved .01 · probably fails .02 · fails .14 · not checked .84 | no | — |
| S1 | train | none | probably fails | probably fails | approved .37 · probably fails .55 · fails .08 · not checked .00 | | not approved ✓ |
| S1 | train | student | probably fails | probably fails | approved .27 · probably fails .61 · fails .13 · not checked .00 | no | not approved ✓ |
| S1 | train | teacher | probably fails | probably fails | approved .28 · probably fails .61 · fails .11 · not checked .00 | no | not approved ✓ |
| S2 | dev | none | probably fails | probably fails | approved .58 · probably fails .35 · fails .07 · not checked .00 | | not approved ✓ |
| S2 | dev | student | probably fails | probably fails | approved .30 · probably fails .55 · fails .15 · not checked .00 | no | not approved ✓ |
| S2 | dev | teacher | probably fails | probably fails | approved .30 · probably fails .57 · fails .13 · not checked .00 | no | not approved ✓ |
| S3 | train | none | probably fails | probably fails | approved .29 · probably fails .49 · fails .22 · not checked .00 | | not approved ✓ |
| S3 | train | student | probably fails | probably fails | approved .24 · probably fails .50 · fails .26 · not checked .00 | no | not approved ✓ |
| S3 | train | teacher | probably fails | probably fails | approved .19 · probably fails .54 · fails .27 · not checked .00 | no | not approved ✓ |
| S5 | test | none | approved | approved | approved .42 · probably fails .39 · fails .20 · not checked .00 | | approved ✓ |
| S5 | test | student | approved | approved | approved .52 · probably fails .33 · fails .15 · not checked .00 | no | approved ✓ |
| S5 | test | teacher | approved | approved | approved .43 · probably fails .37 · fails .20 · not checked .00 | no | approved ✓ |
| S6 | train | none | approved | approved | approved .39 · probably fails .37 · fails .24 · not checked .00 | | approved ✓ |
| S6 | train | student | probably fails | probably fails | approved .32 · probably fails .36 · fails .32 · not checked .00 | yes | approved ✗ |
| S6 | train | teacher | approved | approved | approved .25 · probably fails .34 · fails .41 · not checked .00 | no | approved ✓ |

S4 and O8 have no matched video. Per-item distributions (low_wires, rings, dust, belly) are in the private `e2e_table.json`.

**Reading.**
- **Student vs teacher findings.** The rule verdicts agree on 12 of 14 sites, and the verdicts of record on 11 of 14. The disagreements are O3 (a test video), O4 (test, record only) and S6 (train). At S6, the student saw a 1.2–4.5 m object at 10–50 m, which made the rules say probably fails.
- **Expert direction (S1, S2, S3, S5, S6).** It matched on 5/5 without video, 4/5 with the student's findings (S6 flipped) and 5/5 with the teacher's. Only S5's video is held out from vision training.
- **Video changes the verdict of record.** Student findings changed it at 3 of 14 sites (O1, O7, S6); teacher findings at 4 of 14 (O1, O3, O4, O7).
- **The decision model under-reads video findings.** At O2, O4 and O6, the rules say probably fails from a video wire or a video obstacle in the 50 m ring, yet the model's items keep "approved". Its low_wires item puts at most .56 on probably fails for a seen wire. This is the Plan 1 checkpoint's known weakness: 800 steps, and video findings are rare in its training. The vision stage is not the bottleneck here.

## Limits

- 15 videos and 3 test videos. The bars are counts with n, not claims.
- The teacher is one model with no second-teacher average, and its answers were spot-checked on 3 frames only. They were plausible.
- The can't-tell bar uses blurred frames, so it does not test a text/image mismatch.
- The video-to-site mapping is unconfirmed. S1 has two matched videos (low confidence).
- E2E student findings for 11 of 14 sites come from training or dev videos.
