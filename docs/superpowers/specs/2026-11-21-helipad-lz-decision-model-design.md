# Helipad LZ decision model: research design

Date: 2026-11-21. Status: draft for approval.
Source material: the source brief (private). Local copy: `.context/helipads/` (gitignored: BRIEF.md, raw files, the unpacked pipeline).

## 1. Intent

The expert inspector approves helicopter landing zones for civilian evacuation and rescue under the *shortened site-inspection protocol*. Per site the inspector looks at a location and imagery, a ground video and the map layers, then rules: **approved / probably fails / fails**.

The partner's Claude-skills pipeline (heli-lz-check, private) already does this well. Its rules were calibrated against the expert and agree with the expert on 6/6 sites. But it is **costly**, and it **cannot run in an isolated environment**.

We want a **Typical decision model** that does the same job:
- runs isolated on one GPU server;
- is driven by the **video frames and imagery** plus the map data;
- returns the site verdict **and** a status for every protocol item, each with a confidence, so every call can be explained.

The work is research first: find the best recipe. That recipe is then fitted into Decision Factory, so customers can build models like this themselves. The first public use is a demo showing that a decision model can do this job.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. Calibrated, explainable per-site and per-item decisions that run offline.
- G2. Images and video actually drive the answers, shown by real-vs-shuffled and real-vs-no-image controls.
- G3. On the real expert-labelled sites, match the expert's direction at least as well as the Claude skill does.
- G4. A recipe Decision Factory can reproduce: data in → teacher labels → train → per-question API.

**Non-goals (for now)**
- Measuring metric distances or heights from phone video. No model fixes that: there is no compass and no scale. The map supplies distances; the video confirms what is there and gives rough heights.
- Replacing the site walk or the expert. Low wires and minefields stay "not checked".
- Producing the slide deck. The demo verdict page replaces it.
- Finding a data source for distribution lines (22/33 kV). The model is handed "power line, height unknown" and must apply the calibrated rule.

**Anti-goals (would be failure even if the metrics look good)**
- Saying "clear" where the honest answer is "not checked". "Not found on map" never means clear.
- A model whose answers barely move when the images are shuffled. That is text-only reasoning in disguise.
- Ground condition (vegetation, stones, dust) changing the verdict. Ground condition is always **prep**.
- Hebrew-only case text. Hebrew is an untested risk for Typical (see `hebrew-jevbench-v0`), so case text is English and only place names stay Hebrew.

## 3. Constraints

- **Isolated runtime.** The models (vision and decision) run with no network. The isolated environment has its own GIS layers.
- **Maps, for now.** The research uses the **online** sources `lz_check.py` already calls. GovMap only answers Israeli IPs, so the map fetch runs on the user's machine in Israel, not on RunPod, and the results are cached per site. Porting to local GIS sources is deferred.
- **Distribution lines.** The isolated environment may or may not have a distribution-line layer. The case format carries height certainty (measured / estimated / unknown), and R-rule mixes lines of all three kinds, so the same model works either way.
- **Hardware.** The isolated GPU can serve a **12B** model (user). Primary base: the **Gemma-4-12B** image line (R056 is proven on images, null recall and calibration). Comparison: **Qwen3.5-4B**, the Decision Factory size, to learn what the refit loses.
- **Existing code.** `pcdm/encode.py` `Backbone(multimodal=True)` runs Qwen3.5 as a vision-language model with **one image plus text per case** (`visual_prefix`). `pcdm/native.py` trains and evaluates image rows. Teacher labelling, case generation, distillation and calibration follow R053/R056 (`scripts/r053_*`, `scripts/r056_*`).
- **Teacher.** The Claude skill and Claude vision may see site data at training time (confirmed).
- **Labels are scarce.** Only the 6 September verdicts exist. The October 10 sites have system output only, and no other expert labels exist. There are 15 unique videos.

## 4. Protocol encoding

**Site verdict** (4-way choice): `approved` / `probably_fails` / `fails` / `not_checked`. The fourth class means core map data (OSM or elevation) is missing, so the site must not be read as clear. It mirrors `lz_check.py`'s existing 'not checked' verdict (*my call*, following anti-goal 1).

**Protocol items** (one Typical question each; each item allows only the statuses listed):

| id | Item | Allowed statuses |
|---|---|---|
| pad | 35×35 m pad clear of rotor obstacles (above 1 m / 2 m) | passes, fails, probably_fails, not_checked |
| very_low_10 | nothing 1.2–4.5 m within 10 m of the pad edge | passes, fails, probably_fails, not_checked |
| low_50 | nothing 4.5–24 m within 50 m | passes, fails, probably_fails, not_checked |
| high_300 | nothing 24–76 m within 300 m | passes, fails, probably_fails, not_checked |
| ultra_1km | no 76 m+ obstacle or forbidden sector within 1 km | passes, coordination, not_checked |
| slope | slope ≤15° (aircraft 1) / ≤10° (aircraft 2) | passes, fails, not_checked |
| zones | firing zone, nature reserve or park within 1 km | passes, coordination, not_checked |
| belly | belly rectangle clear (10 / 20 cm) | passes, prep, not_checked |
| dust | dust light | passes, prep, not_checked |
| low_wires | low wires (no data source) | not_checked, fails, probably_fails |
| minefield_500 | no minefield within 500 m | not_checked, fails |

**Verdict logic the model must learn.** This comes from the protocol and the 9 calibrated rules in BRIEF.md §2:
- Any `fails` → `fails`.
- Otherwise, if core map data (OSM or DEM) is missing → `not_checked`.
- Otherwise any `probably_fails` → `probably_fails`.
- Otherwise → `approved`.

This precedence mirrors `lz_check.py` `evaluate()`. The conditions are every `not_checked`, `prep` and `coordination` item.

**Always show every option with its probability.** Every protocol item and the verdict are presented with all their candidate statuses and the model's probability for each, never just the top answer. For example, `low_50: passes 0.62 · probably fails 0.35 · fails 0.02 · not checked 0.01`. This applies to reports, the scorer's case records, the demo page and the API (user requirement, 2026-11-21). A decision rule, such as a safety threshold, may set the reported status, but the full distribution is always shown next to it.

The verdict is still its own question, so we can measure whether the model stays consistent with its item answers. Aircraft type is part of the case text.

**Case text** (English, one line per finding, with the source and certainty stated):

```
Aircraft: 1
Touchdown: <lat>, <lon> (pin; not moved)
Finding: power line on poles · height unknown · 120–160 m from pad edge · north · source map+video
Finding: lighting poles · height est. 8–10 m · 30–50 m · direction unknown · source video
Finding: eucalyptus row · height 15–20 m (canopy layer) · 40 m · west · source map
Slope in pad: 2° (±6°)
Zone: firing zone within 500 m · source map
Ground: ploughed clods; vegetation 10–20 cm · source video
Dust: medium · source video
Not available: low wires, minefields
```

Policy text = the protocol table plus the calibrated rules, in English (see `airdef-finops-demo`: the policy must be in the model's text).

## 5. Approaches considered

| | Approach | Verdict |
|---|---|---|
| A | Frontier model reads the frames → text findings → text Typical decides | Not isolated: the vision step needs Claude at runtime. **Kept as the teacher and the baseline.** |
| B | **Two stages, both Typical.** Vision Typical answers fixed frame questions per keyframe; plain code merges them with map findings into case text; text Typical decides the items and the verdict. | Explainable chain (frame → answer → finding → item → verdict). The decision stage can use unlimited rule-generated cases. |
| B′ | **One multimodal Typical.** Satellite crop + keyframes (as one mosaic image, so it fits one-image-per-case) + map findings text → items and verdict directly. | May catch things the frame questions never asked about. Can only train on teacher-labelled real sites, not rule-generated ones. Explanation is weaker. |
| C | Classical CV (SAM, Grounding DINO, monocular depth) as the measurement source | Rejected. Monocular depth is unreliable at 50–300 m, thin wires are the classic detector miss (and wires are the #1 risk), and heights still need a scale reference. **Optional later** for drawing evidence boxes on frames. |

**Chosen: research B and B′ side by side** (track R5 decides between them). Both are distilled from the Claude skill as teacher and fall back to A while the vision student is not ready. The user chose research first, best model second, Decision Factory refit third.

## 6. Runtime design (isolated)

```
location + video (+ aircraft type)
 ├─ Map step: lz_check.py fetch+evaluate against local GIS sources
 │    → map findings (distance, compass bearing to nearest point, height if known), slope, zones,
 │      satellite crop; touchdown may move ≤60 m (existing logic), and any move is reported
 ├─ Frame step: ffmpeg 1 fps → keyframe selection (~15: horizon sectors + ground close-ups)
 │    → vision Typical, fixed frame questions per keyframe (one encode per frame)
 ├─ Merge (plain code): frame answers → video findings; a video sighting that matches a map object
 │    becomes "map+video" and takes the map's distance and bearing; otherwise direction is "unknown"
 └─ Decision Typical → item statuses + verdict, with confidences
     (B′: one model replaces the last three steps)
```

**Frame questions** (each has a `can't tell` option, which uses Typical's factored null):
- wires visible: yes / no
- pole (power or lighting) visible: yes / no
- tall mast, tower or crane on the horizon: yes / no
- tallest object's height class: below 1.2 / 1.2–4.5 / 4.5–24 / above 24 m
- its distance class from the camera: below 10 / 10–50 / 50–300 / above 300 m
- tallest object type: tree / building / wall-fence / pole / vehicle / none
- ground surface: paved / hard / stony / ploughed / vegetation / powdery
- vegetation above the belly threshold in the foreground: yes / no
- dust class: light / medium / heavy

**Merge rules** (code, unit-tested):
- Presence = max confidence over frames, with the frame timestamps kept.
- Height and distance = the confidence-weighted class distribution, reported as a range.
- The final question set is left to the builder (§10); the list above is the starting point.

## 7. Data

| Set | What | Size target | Use |
|---|---|---|---|
| E-expert | The 6 September sites | 6 | **Test only for the students** (never trained on). Caveat: the teacher's rules were tuned on these, so the teacher's 6/6 is optimistic. A student matching them shows faithful distillation, not independent generalisation. Report counts |
| T-holdout | Teacher-labelled sites held out by site, including the October 10 and the Nov 16 videos | ~50+ | **Main fidelity test**: student vs teacher on items and verdict |
| E-pilot | The October 10, if the expert inspector labels them blind (*recommended, not a blocker*) | 10 | The only independent expert evidence. The October deck itself asks for this pilot |
| T-site | Sites run through the full Claude skill: findings, item statuses, verdict | ~300 map-only + all available video sites (*my call; adjust to budget*) | Decision-stage and B′ training labels |
| T-frame | Keyframes labelled by Claude vision on the frame questions | ~5–10k frames | Vision-stage training |
| R-rule | Rule-generated findings cases labelled by a findings-level rule engine | 20k+ | Decision-stage training volume; also teaches unknown vs estimated vs measured |

**Sources for scale**
- *Map-only sites*: random points in Israel, stratified by land cover (fields, desert, near villages, near roads) and oversampled near power lines, roads and buildings so the verdicts are balanced.
- *Video and frames*: the 15 videos, the expert inspector's archive (*to request*), new shoots following the deck's 7 filming rules, and street-level imagery near the sampled points (e.g. Mapillary; check the licence).

**Rule engine.** A findings-level function implementing the protocol plus the 9 calibrated rules. Its validity check: it must reproduce `lz_check.py`'s verdicts on the real sites.

**Real-site data.** The saved `data.json` files are not in the zip. R1 re-runs `lz_check.py` on all real sites (*my call*):
- The September coordinates come from the saved Maps list, which needs a browser.
- If a fresh fetch doesn't reproduce a calibrated verdict, record that as a finding. Don't tune the rules to fix it.

**Teacher.**
- *Site level*: the Claude skill steps (`lz_check` + the video skill), run in batch through an automated harness with cached responses, as in `r053_teacher_label.py`.
- *Frame level*: a direct Claude vision call on the fixed questions, asking for probabilities.
- If budget allows, average two teachers (see `workflowevals-label-teachers`).

**Leakage.** Splits are by site. E-expert, T-holdout and E-pilot sites never enter training.

## 8. Research tracks and success criteria

Thresholds are *my calls* and can be changed.

| Track | Work | Pass |
|---|---|---|
| R1 Eval | Assemble E-expert and T-holdout; re-run `lz_check.py` on every real site. Map videos to sites by comparing frames with imagery. Run the Claude skill as the reference | Eval harness reports verdict agreement, item agreement, calibration (ECE) and can't-tell recall for the teacher, A, B and B′ |
| R2 Teacher corpus | Batch harness for T-site and T-frame; cost log per site | Corpus built within budget; cache makes reruns free |
| R3 Vision student | Gemma-4-12B multimodal on T-frame (Qwen3.5-4B as the Decision Factory-size comparison) | Agreement with the teacher ≥ 85% on held-out sites; can't-tell recall ≥ .8; real − shuffled ≥ 10 points |
| R4 Decision student | Text Typical on R-rule + T-site | Item agreement with the teacher ≥ 90%; verdict consistent with its own items ≥ 95% |
| R5 B vs B′ | T-holdout, then E-expert (and E-pilot if available) | Pick the winner on expert-direction agreement, then item agreement; explanation quality breaks ties toward B |
| R6 Local inference | Both models on one GPU with no network (map step stays online for now) | One site in ≤ 2 min, excluding the map fetch; zero outbound calls from the models |

**Headline claim for the demo:** a local Typical reproduces the Claude skill's item-by-item calls on held-out sites, and the expert's direction on the expert-labelled sites, at a fraction of the cost and with no network for the models. With E-pilot labels, the claim extends to independent expert agreement.

## 9. Decision Factory refit (after research)

Bring the winning recipe in as features:
- image and video cases (upload video plus location, or images plus text);
- frame questions as a question type;
- an optional "teacher labels your data" step (online only);
- training on a multimodal base (Qwen3.5-4B or Gemma-4-12B, whichever R3 and R5 support);
- a decide API that accepts images.

The verdict page from the demo becomes the model's playground view. Design details are deferred until R5 is settled.

## 10. Left to the builder

- Keyframe selection method and count.
- The final frame-question wording and classes (§6 is the starting point).
- Mosaic layout for B′.
- The teacher model choice and prompts.
- Training hyperparameters, following R056 conventions.
- Harness and eval code structure, reusing `pcdm/native.py` and `scripts/r05x_*`.

## 11. Open questions (owner: user)

1. What is the teacher budget for T-site and T-frame?
2. Is there an archive of past site videos?
3. Will the expert inspector blind-label the October 10 (E-pilot)?
4. Who is the demo audience, and will they bring their own sites (decides live vs precomputed)?

Builder's job: matching the Nov 16 videos to sites by comparing frames with imagery.
