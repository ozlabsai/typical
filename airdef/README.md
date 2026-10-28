# Air Defence Decision Desk (Typical demo)

A live, simulated air-defence battle in which Typical makes the calls at two levels:

- **Global (battle manager).** For each threat: which interceptor takes it? Options are the named ground interceptors
  that can fire now (Arrow 3, Arrow 2, David's Sling, Iron Dome, Iron Beam, THAAD, SM-3, SM-6, EW jamming),
  **Fighter jets**, **Helicopter**, **Wait for cheaper**, **Wait**, or **Let it fall**.
- **Local (fire unit).** The assigned battery or aircraft decides how many interceptors to fire now: 0, 1, 2 or 3,
  a Typical *Score*. It weighs free launch channels, rounds on the launchers vs. reloading, a radar that has to turn,
  own aircraft near the intercept point, debris over a city, the last chance, and time for a second look.

The teacher is a planner that prices every option through three lenses:
- **This threat:** a small DP over its remaining engagement windows (fire, look, fire again).
- **Tonight's raid:** shadow prices for scarce interceptors, plus a congestion term for layers other threats have already
  booked.
- **The campaign:** replacement cost × scarcity × magazine posture.

Typical is fine-tuned to reproduce the planner's calls from plain-language situation text. The design is in
`DECISIONS.md`.

All data is simulated. Parameters (costs, ranges, kill probabilities, channels, reload times) are notional, battery
positions are not real deployments, and the scenarios are loosely modelled on public events.

## Run

```bash
uv run --no-sync uvicorn --app-dir airdef server:app --port 8790     # open http://127.0.0.1:8790
AIRDEF_MODEL=none uv run --no-sync uvicorn --app-dir airdef server:app --port 8790   # planner only, no model
```

- The server loads `.context/typically/results/co_airdef_v3/best.pt` (the v3 fine-tune) when it exists, otherwise
  `typical-small`.
- Scoring goes through the training repo's decider (`scorer.py`). The public `inference/typical` port used to disagree
  with it on semif checkpoints (an extra eos); fixed, with a parity test on the release.
- The map tiles need internet. maplibre and deck.gl are vendored.
- The clock never waits for the model. Typical runs in a worker thread over a queue: the threat on screen first,
  then by urgency. An answer that misses its 1 s budget (or a fire unit's last chance) is replaced by the planner's
  call. The screen does not annotate this; the ledger counts it (`late` in `/api/state`). On a Mac (MPS, ~150 ms per
  call) nearly every call is the model's at 1-5x, far fewer in the 20x bursts; on CUDA the worker scores 16 at once.
  Low-confidence calls likewise take the planner's call (`conf_threshold`, `escalated`), unannotated on screen.
- Quiet stretches are skipped automatically; a plain *Wait* on a drone hours out is re-asked at ~10% of its remaining
  flight. Whole scenarios at 20x: Red Sea ~50 s, northern ~70 s, October ~70 s, June ~100 s, April ~2 min.

## Five-minute demo script

The panel has two modes (header). **Focus** (default) tells the raid as stories: when a missile is engaged, aircraft
are sent, or Typical waits for a cheaper interceptor, Focus follows that threat to its outcome. The clock plays the
rest of its flight in about 25 s, so every re-ask happens on screen: the live call, why, the fire unit's call, the
option bars moving, and underneath how the call developed (each call with its probability as it firms up, each shot,
the outcome). A salvo let fall, a unit holding fire and a hit are short moments. The clock slows, the camera frames the decision (the threat, where it lands, who takes it),
the rest of the map dims, and the panel tells it: the call, why, the fire unit's call, every option with its
probability. Between moments it fast-forwards. *Next moment* skips; the markers on the timeline replay any moment; at
the end, *Raid over* lists every moment. **General** is the whole raid: salvos, aircraft and fire units with their
status; click one, or its map object, to expand it. Map: circle = threat (filled engaged, hollow waiting, ring with a
dashed path = let it fall), square = fire unit (filled = firing, grey = reloading or empty), triangle = aircraft,
× = an intercept, red = a hit.

1. **0:00 Northern front, Focus, 20x, paused.** The panel reads "Every threat, one question: which interceptor, and
   when." Press Play.
2. **0:30 The moments arrive.** Hezbollah drone, *Fighter jets*; the wait for Iron Beam pays off; a salvo let fall
   (no interceptor spent); a unit holding fire. *Next moment* between them.
3. **1:30 13 April, Focus, 20x.** Helicopters and jets over Jordan; *Wait for cheaper* and its payoff; holding fire for
   own aircraft. **Tonight** shows the magazines and the held-back line (let fall, waits, $ kept).
4. **2:30 Drills, Wait for cheaper.** Every option played out to the end 40 times: Typical's pick is the best.
   *Next case*, then *Helicopter*.
5. **3:30 Settings.** Magazine posture to *Critical*, a priority site; watch the calls change. Press עברית for a
   few seconds.
6. **4:15 General** on the finished raid: salvos, fire units, aircraft; close on **Tonight**.

Keep October for questions (the stress case). Notional positions: A2-C and DS-C were moved west on 2026-10-28, after
the measurements below.

## What was measured

Planner vs doctrine (layered, cheapest effective layer first, fires when in its window). Mean over 8 seeds per
scenario. Total cost = interceptor spend + damage from hits, $M-equivalent.

| scenario | doctrine: total · hits | planner: total · hits |
|---|---|---|
| 13 April | 13,950 · 10.3 | **4,331 · 2.9** |
| June attrition | 5,665 · 8.0 | **959 · 1.4** |
| 1 October (saturation) | 40,710 · 22.0 | **33,534 · 17.5** |
| Northern front | 238 · 3.1 | **148 · 1.8** |
| Red Sea | 805 · 1.3 | **11 · 0** |

Typical (fine-tuned `typical-small`, v3) on held-out raids, scored as agreement with the planner:

| test | base | v3 |
|---|---|---|
| Global call | 5% | **85%** |
| Global, commit moments only | 13% | **70%** |
| Local fire count (0–3) | 34% | **72%** |

Whole raids with Typical playing every call (mean total · hits; scored on H100s so the model is never late;
`remote_harness.py --ckpts v3=... v4=...`, compared seed by seed with `compare_runs.py`; engine as of 2026-10-27
evening, before the A2-C/DS-C move):

| scenario | seeds | doctrine | planner | Typical v3 (demo) | Typical v4 | v4 - v3 per raid (se) |
|---|---|---|---|---|---|---|
| 13 April | 6 | 14,506 · 10.5 | 5,209 · 3.7 | **4,191 · 2.7** | 5,566 · 3.7 | +1,375 (1,418) |
| June attrition | 4 | 6,660 · 8.8 | **1,206 · 1.8** | 2,320 · 3.5 | 1,805 · 2.5 | -515 (539) |
| 1 October (saturation) | 8 | 42,210 · 22.9 | **34,097 · 17.8** | 35,476 · 18.1 | 42,056 · 20.5 | +6,579 (1,557), v4 worse on 8/8 |
| Northern front | 8 | 240 · 3.4 | **149 · 1.8** | 154 · 1.6 | 194 · 2.4 | +41 (23), v4 worse on 8/8 |
| Red Sea | 8 | 618 · 1.1 | **10 · 0** | **10 · 0** | **10 · 0** | 0 |

v3 is at or near the planner everywhere and keeps most of its gain over doctrine on the October saturation. April and
June are within noise between v3 and v4 (fewer seeds: those raids are slow).

**Why the demo runs v3, not v4.** v4 is trained on the fixed teacher (aircraft stay on their target, fire counts
capped at what the unit can launch) and is better on every per-decision test: commit moments 78 → 81%, fire count
73 → 80%, situation drill 75 → 84%, drill regret $0.97M → $0.44M per case. On whole raids it is worse: on October it
loses to doctrine on all 3 seeds. It is more confident (sends 5–21 calls to the operator per October raid vs v3's
43–46) and spends less, so its mistakes in a saturation raid are not caught. Per-decision accuracy did not predict
whole-raid cost (v2 showed the same). Next lever: fewer steps or a calibrated escalation bar, gated on whole raids.

Situation drill: 56 curated cases where every option was played out to the end (`situations.py`). Mean extra cost
vs the best option, $M-equivalent (`drill_regret.py`): **planner 0.14, Typical v4 0.44, Typical v3 0.97, doctrine
1.52.** The planner is optimal on 5 of 6 families; on an unclear early track it waits too often (0.80 per case).
v3's gap is the helicopter family (4.79; v4 1.84).

## Files

| file | what |
|---|---|
| `sim.py` | threats, scenarios, batteries, aircraft, notional parameters |
| `air.py` | moving air units: scramble, intercept along the threat's path, return to rearm |
| `planner.py` | the teacher: global options with the three lenses, the local fire-count rule |
| `engine.py` | the world loop, both levels, policies (doctrine / planner / typical), the text Typical reads |
| `situations.py` | curated situations, optimum by rollout; writes the drill eval set |
| `dataset.py` | distillation data for both levels from randomised raids |
| `scorer.py` | loads a checkpoint through the training repo's decider |
| `harness.py`, `remote_harness.py` | whole raids, locally or on a rented H100 |
| `eval_model.py`, `drill_regret.py` | held-out accuracy, regret on the drill |
| `server.py`, `web/` | the live demo (EN/HE), built on the `brand/` taste index |
