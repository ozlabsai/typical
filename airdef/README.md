# Air Defence Decision Desk (Typical demo)

A live, simulated air-defence battle. Three policies fight the same raid with the same dice:

- **Fire at everything**: the most reliable plan on every threat, no doctrine.
- **Cost-aware optimizer**: minimises interceptor cost + magazine value + expected leak damage.
- **Typical** (`typical-small`, zero-shot): reads the written rules of engagement and the commander's orders and
  chooses the *doctrine action* (hold fire / sure-kill plan / cheapest reliable plan). The solver turns that
  action into a concrete plan. When Typical isn't confident enough, the track goes to the operator.

All data is simulated from open-source figures. Battery positions are notional, not real deployments, and
Pk values are notional knobs.

## Run

```bash
uv run --no-sync uvicorn --app-dir airdef server:app --port 8790     # open http://127.0.0.1:8790
AIRDEF_MODEL=none uv run --no-sync uvicorn --app-dir airdef server:app --port 8790   # no model: optimizer only
```

The map tiles (CARTO dark) need internet. maplibre and deck.gl are vendored in `web/vendor/`.

Checks:

```bash
uv run --no-sync python airdef/engine.py        # sim self-check: optimizer vs naive, stock never negative
uv run --no-sync python airdef/harness.py --seed 1   # gate: the three policies + Typical's doctrine-following
uv run --no-sync python airdef/probe.py         # wording probe: state/label variants vs doctrine
```

## Five-minute demo script

1. **Red Sea** (small raid). Play at 60×, then press *Skip transit*. Click a Palestine-2 track. The right panel shows
   what Typical reads: three rules plus the situation. It also shows Typical's probability over the three
   actions, which concrete plan each action resolves to, and where the doctrine and the optimizer land.
2. **Northern front** at 20×. Most rockets fall in open ground, so Typical holds fire on them and the
   fire-at-everything ledger burns Iron Dome and David's Sling. Look at the spend columns.
3. **The reactiveness moment.** Pause. Select a drone heading for Haifa: "cheapest reliable plan" (about 60% when tested).
   Open *Commander's console*, click the preset "When the threat is a drone, fire the sure-kill plan.", then
   *Issue order*. Play at 1× and select the drone again: it is now on "sure-kill" (about 67% when tested). Every live track
   that hasn't launched is re-decided.
4. **13 April / 1 October** at 20×. Ballistic arcs out of Iran, Arrow 3 and THAAD intercepts over Jordan.
   Toggle *Magazine posture → Critical* and watch the resolved plans move to shoot-look-shoot and lower tiers.
5. **Escalation.** Raise *Escalate if top below* to 0.55. More tracks go to the operator, shown in amber.
   This is the human-on-the-loop knob.

## What was measured (typical-small, MPS, 2 seeds × 4 scenarios)

- Follows the written doctrine on **94–100%** of decisions. Each model call takes ~100–140 ms, and there is one
  call per situation change, not per tick.
- Agrees with the optimizer on 61–90% of decisions. Total cost (spend + damage) is at or below the optimizer's
  in 6 of 8 runs. Against fire-at-everything it spends 22–68% less, except in the saturation raid,
  where everyone empties the magazine.
- Commander's orders written in the rule grammar: 15/16 on a paired order/control probe. Orders lift Typical's
  ∅ mass to ~0.5, so ∅ has its own escalation bar (0.9).
- Wording mattered more than anything else. Choosing among plan labels got 58% doctrine-following. Choosing
  among the doctrine's own actions got 99%. Adding a fourth standing rule (class-based) dropped it to ~50%.
- Not solved: 1 October saturation is a coin flip for every policy, because the magazines run dry. Typical's
  confidence is modest (median top probability 0.52 over 3 actions). Fine-tuning on doctrine data via
  Decision Factory is the next step.
