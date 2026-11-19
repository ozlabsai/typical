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

## Step by step (levels) and the policy

From customer feedback (2026-11): the menu opens on four levels that build up one decision at a time, each told from
the threat's first plots (detected, classified hostile, contact lost and regained), in nautical miles and knots:

1. **One drone, one battery.** A Shahed 80 NM east, 40 kts, heading for Jerusalem; contact lost 5 NM before the border.
   Watch the call change as the track firms up.
2. **One drone, two Iron Domes.** Which battery takes it (`Q_BATTERY`, a third decision level). Four cases:
   a) the sooner one; b) the near one is at its 12-round floor; c) the near one has other plans (committed);
   d) only the battery at its floor stops it before the border, so it fires anyway.
3. **Three threats** at once, told as one group story.
4. **A swarm:** 15 drones after an 8-hour warning, the closest 120 NM out.

On screen, a level opens on its brief (the set-up and what to watch) and ends on its outcome (stopped before or after the
border, what it took) with the next level as the one button. From Level 2 the picked battery is in the headline, the
reason names the trade-off, and the batteries are compared side by side as they stood at the call (rounds against the
floor, other tasks, time to intercept, before or after the border). In Levels 3-4 the threats are numbered on the map
and in the list; click a row for that threat's story.

The policy (Settings, live and hosted) adds three rules the planner prices and Typical reads (as sentences: "Policy:
stop it before the border if possible.", "Policy: going below a battery's ammo floor is a reason to think twice.", and
"it crosses in about 4 minutes unless stopped"):
- **Ammo floor** per Iron Dome battery (None / 12 / 20): not a ban, a cost (*below the floor*, $M per round).
- **Border:** we prefer no target to cross. An option that lets it cross before it is stopped pays the border price;
  every crossing counts in the night's total.
- **Plans:** a battery with other things to do costs more to commit.

All data is simulated. Parameters (costs, ranges, kill probabilities, channels, reload times) are notional, battery
positions are not real deployments, and the scenarios are loosely modelled on public events.

## Run

```bash
uv run --no-sync uvicorn --app-dir airdef server:app --port 8790     # open http://127.0.0.1:8790
AIRDEF_MODEL=none uv run --no-sync uvicorn --app-dir airdef server:app --port 8790   # planner only, no model
```

- The server loads `.context/typically/results/co_airdef_v8/best.pt` (the v8 fine-tune: reads the policy, level-2
  trade-offs) when it exists, otherwise `typical-small`. Escalation bar `conf_threshold` 0.5 (see v8 below).
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

## Hosted demo (typical-airdefence.ozlabs.ai; typical-ad redirects)

**Live mode (2026-11-19).** When the live server is up, the hosted page runs live instead of playing tapes: every
setting, any combination of priority areas, one world per viewer, Typical v8 answering (its recorded answers first, so
default settings are instant). The page asks `/api/meta` at load; the Worker forwards `/api/*` to `LIVE_URL` with a
token only it holds (`LIVE_TOKEN`); without an answer the page plays its recordings, as below.

```bash
uv run --no-project python airdef/deploy/live_pod.py up       # cheapest free GPU (RTX 2000 Ada .. RTX 6000 Ada), ~15 min
uv run --no-project python airdef/deploy/live_pod.py status
uv run --no-project python airdef/deploy/live_pod.py down     # delete the pod; the site plays its recordings again
```

Cost: the pod runs until `down` (RTX 3090 $0.50/h, about $360 a month left on).

A static site behind a shared-password Worker (Cloudflare free plan; no server, no model). `record.py` recorded
Typical's real answers once; the sim is deterministic (fixed DT steps, stable text), so `bake.py` plays every scenario
and drill case through the live server's own clock and director and writes a tape per scenario; `web/tape.js` plays
it in the browser through the same API `app.js` uses. Settings work too: every level is recorded and baked for each
policy in `sim.POLICY_GRID` (magazine posture × ammo floor × below-floor price × border price, 81 per level; posture
only for the full raids), and each scenario's priority areas (`sim.PRIORITY`) at the default policy. Changing a setting
loads that policy's tape at the same moment (`tape.js` `setPolicy`); a combination never recorded falls back to the
closest one with the change asked for, and Settings shows what plays.

Recording has two traps, both handled: (1) record through the build's own loop (`record.py` calls `bake.bake` with the
real model): a Focus story turns off the quiet-time skip, which moves when calls are re-asked; recorded on any other
clock, the build asks questions nobody answered. (2) The GPU answers the same question slightly differently from one
job to the next (up to 0.03); a merged answer flips near-ties on replay. `record.merge` keeps each job apart and
`Recorded(tape=...)` replays a tape from its own job. Check: `.context/airdef/verify_rec.py` (zero misses on the build
path). Pods: `RUNPOD_COUNTRY=CA` steers off a flaky host.

```bash
# record on an H100 (needs runpodctl >= 3: the 2.x CLI calls rest.runpod.io, which no longer resolves)
uv run --no-project python airdef/remote_harness.py --record --ckpts v8=.context/typically/results/co_airdef_v8/best.pt
AIRDEF_MODEL=recorded uv run --no-sync python airdef/bake.py --procs 8   # -> airdef/static
cd airdef/deploy && npx wrangler deploy                            # password: npx wrangler secret put DEMO_PASSWORD
```

The zone's bot protection shows non-browser clients a "Just a moment..." check; browsers pass it.

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

0. **Step by step first** (customers asked for it): Level 1, then Level 2 a–d, then the swarm. In Settings, set the
   border to *Indifferent* on 2d: the call moves to the battery above its floor.
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

### v5 (2026-11-18): levels, battery choice, floor and border

Trained on the levels engine (battery rows, randomised floors, plans and border prices). Held out:

| test | base | v3 | v5 |
|---|---|---|---|
| Battery choice | 41% | 43% | **74%** (94% within the planner's acceptable set) |
| Global call | 7% | 24% | **83%** |
| Global, commit moments | 8% | — | **86%** |
| Local fire count | 39% | — | **80%** |
| Situation drill | 45% | — | **93%** (regret 0.03) |

Level 2 under every recorded policy: v5 matches the planner on 297 of 297 battery picks. Whole raids, v5 vs v3, 6
seeds: October **33,174 vs 39,852** (v5 better on 5 of 6), Red Sea tied, northern 859 vs 757 (+102 ± 61: v5 worse;
the follow-up). Totals now include crossings × border price, so they are not comparable with the tables above.
v5 is less confident (median top choice 0.49; 31% of global calls go to the operator).

### v6-v8 (2026-11-19): Typical reads the policy

Fable (a UI review agent) found Level 2d letting the drone cross under the default policy. Cause: the border and floor
settings only moved the planner's prices; Typical's text was identical under every setting, so it learned their
average. v6 adds the policy and the time to the border to the text; v7 adds level-2 routes that vary (origin, target,
when classified) with their own balance quota (~2,000 floor-vs-border rows, from ~20); v8 fixes "below its floor of 0"
(a battery short for its other engagements). P(fire now) on the 2d text: v5 0.11, v6 0.33, v7 0.51, v8 fires.

v8 on Level 2d: default, *Strongly*, *Allowed*, any posture -> ID-N before the border; *Indifferent* -> ID-H after it
(as the planner). Off on one setting: *Last resort* + *Prefer* lets it fall first (the planner fires ID-N).

| held out | v5 | v8 |
|---|---|---|
| Battery | 74% | 75% (95% acceptable) |
| Global | 83% | 74% (now 86%, cheaper 71%) |
| Local fire count | 80% | 79% |
| Situation drill | 93% | 93% |

Whole raids, 6 seeds, vs v5: at the old escalation bar (0.42) v8 escalated 0.8 calls per northern raid (v5 8.5) and
took 1.2 more hits (northern 1,014 vs 859, worse on 6/6). At **0.5**: northern 881 vs 859 (+22 ± 25), October 34,151
vs 33,174 (+977 ± 2,255), Red Sea tied -- level with v5, and the levels tell the right story. Shipped: v8 at 0.5.

### v9-v10 (2026-11-19): weights on what a hit is worth -- not shipped

Settings briefly offered Low/Normal/High for what a hit on a city, military site or infrastructure is worth. v9 read
it as dollars, v10 as words (negligible..catastrophic); neither used it (P(let it fall) on 60 held-out rows: 0.046
either way) and both regressed Level 2d. The reason is upstream: x0.5..x2 on hit values barely moves the planner
either (0-1 of 15 swarm first calls, 0-24 of 174 northern) -- an Iron Dome round is $50k against hits in the $M.
`sim.WEIGHTS = []` keeps the machinery; the text is v8's. Trade-offs that do move calls (time vs certainty, a battery's
other tasks) are the candidates for the next steerable setting.

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
