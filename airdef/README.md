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

0. **The panel** lists every object: threats (status and countdown), aircraft and fire units (rounds ready / left,
   channels, reload). Click a row, or the object on the map, to expand it; click it again to go back to following the
   most urgent threat. On the map: filled = engaged, hollow = waiting, faint = let it fall, blue = in focus.
1. **Red Sea, 5×.** Three ballistic missiles launch at once. Follow one in the right panel: the threat, its countdown,
   Typical's call in blue (e.g. *Arrow 3*, or *Wait for cheaper*) and one plain-language reason. When it is assigned,
   the line underneath is the fire unit's own call, e.g. "A3-C fires 1 · time to look before a second shot". The
   flight line at the bottom shows each call where it was made, the shots (✓/✕) and *now*.
2. **The drones that follow, 20×.** The jets on patrol over the Red Sea are marked on the map. About 40 s in, the
   Shaheds reach them: click one to see *Fighter jets* and the unit flying out to meet it.
3. **Northern front, 20×.** Iron Dome calls from the first minute, a fighter for the Hezbollah drone (~20 s), then
   the barrage (~50 s): rockets heading for open ground get *Let it fall*. Watch the fire units hold when all channels
   are busy, or when own aircraft are near the intercept point. **Tonight** compares spend and hits with doctrine.
4. **13 April, 20×.** Drones and cruise missiles first; about a minute in, *Fighter jets* and *Helicopter* take the
   drones over Jordan, then the ballistic wave: *Wait for cheaper* keeps the Arrow 3 stock for the threats that need
   it, and the congestion lens stops everyone waiting for the same layer.
5. **Drills** (bottom of the scenario menu). One curated situation frozen at its decision moment: Typical's call in
   the panel, and below it every option played out to the end 40 times, with the best marked and who picked what
   (Typical / planner / doctrine). *Next case* steps through instances; *Play* continues the case. The panel's reason
   is the planner's *estimate*; the drill card is the simulated outcome, and they can differ (coverage gap: the
   estimate prefers jets, the outcomes tie). *Unclear track* is the honest weak spot: everyone fires early.
   Cases: `situations.py --instances 3 --rollouts 40 --out .context/airdef/drill_ui.jsonl` (~30 s).
6. **Settings.** Switch *Magazine posture* to *Critical* or change a site's damage weight, and watch the live calls
   change.

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

Whole raids with Typical playing every call (3 seeds, mean total · hits; scored on an H100 so the model is never
late; `remote_harness.py --ckpts v3=... v4=...`, engine as of 2026-10-27 evening):

| scenario | doctrine | planner | Typical v3 (demo) | Typical v4 |
|---|---|---|---|---|
| 13 April | 13,090 · 9.7 | 4,368 · 3.3 | **1,422 · 0.3** | 3,146 · 2.0 |
| June attrition | 6,837 · 9.0 | **1,490 · 2.3** | 1,832 · 2.3 | 1,812 · 2.3 |
| 1 October (saturation) | 40,527 · 21.7 | **30,523 · 15.7** | 34,413 · 17.7 | 46,359 · 21.3 |
| Northern front | 210 · 3.3 | **117 · 1.3** | 129 · 1.3 | 166 · 1.7 |
| Red Sea | 340 · 0.7 | **9 · 0** | **9 · 0** | **9 · 0** |

v3 is near the planner everywhere and keeps about 60% of its gain over doctrine on the October saturation. Beating
the planner on 13 April is 3 seeds and a handful of hits: do not quote it.

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
