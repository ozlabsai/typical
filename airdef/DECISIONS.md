# Decision model: two levels, operative options

Agreed 2026-10-27. Fallback demo before this rebuild: git tag `airdef-v2-demo`.

```
 ALL threats ─┐   ┌──────────── GLOBAL · battle manager (whole picture) ────────────┐
 ALL assets   ├──►│ For each threat, most urgent first: "Which interceptor takes T017?"   ← Typical CHOICE
 stock, waves │   │ [Arrow 3] [Arrow 2] [David's Sling] [Iron Dome] … [Fighter jets] [Helicopter]
 campaign  ───┘   │ [Wait for cheaper] [Wait] [Let it fall]                                          │
                  └───────────────────────────────┬────────────────────────────────────────────┘
                                                  │ assignment: T017 → Arrow 2 (A2-C) | Fighter pair "Viper N"
                  ┌───────────────────────────────▼──── LOCAL · the assigned unit ─────────────┐
                  │ "Fire at T017 now?"  [Fire one] [Fire two] [Hold]                ← Typical CHOICE
                  │ in window/range? channels free? track firm? someone else engaging?
                  │ debris over a city? last chance? time for a second look?
                  └───────────────────────────────┬────────────────────────────────────────────┘
                                 fire → look → hit: done · miss: back to GLOBAL · hold: re-asked next tick
```

## Global (battle manager)

Options for one threat at time t, only the feasible ones:

| option | meaning |
|---|---|
| a ground interceptor by name (Arrow 3, Arrow 2, David's Sling, Iron Dome, Iron Beam, THAAD, SM-3, SM-6, EW jamming) | it can launch now; assign the threat to its best battery |
| Fighter jets / Helicopter | scramble or vector the best free unit; it intercepts later along the path |
| Wait for cheaper | commit to a cheaper ground layer that reaches it later (named in the reason) |
| Wait | no commitment yet: the track is not firm (it may land in open ground) or it is not urgent |
| Let it fall | it will land in open ground, or the damage is worth less than any interceptor |

Teacher: the multi-lens planner (this threat · tonight's raid · the campaign). Wait is priced as value of information:
deciding after the track firms up, with whatever windows remain.

## Local (fire unit / aircraft)

Asked every tick while a unit holds an assignment and can engage: Fire one / Fire two / Hold.
Facts: window or weapon range, free launch channels, track quality, another unit already engaging, debris over a
populated site, last chance, time for a second look. Teacher: rules + the DP over the unit's own window.

## Assets

- Ground batteries with launch channels (simultaneous engagements) per type.
- Air units, mobile: fighters (fast, 4 AIM-9X + APKWS, vs drones and cruise missiles) and attack helicopters
  (slow, cannon/Hellfire, drones only). Each has a base, scramble delay, speed, endurance, weapon load, hourly cost.
  States: ground alert, airborne on a CAP station, tasked, returning.

## Baselines and evaluation

- Doctrine: layered, cheapest effective first, fire when in window. Planner: all lenses. Typical: both levels.
- Curated situation families with an exact optimum: wait for cheaper, last chance, shoot-look-shoot, let it fall,
  save for the next wave, coverage gap (air unit), saturation (channels), debris.
- Full raids on a rented GPU.
