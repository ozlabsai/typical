# Helipad demo: data contract

The app reads **recorded** results. One JSON file per site, plus images, from a data directory.
- Development uses `helipad-demo/fixtures/`: committed, synthetic, public IDs only.
- Real recordings are written by `scripts/helipad_demo_bake.py` to `$HELIPAD_PRIVATE/demo/` and never committed.
- The app reads `HELIPAD_DEMO_DATA` (a path) at build time and at runtime, defaulting to `fixtures/`.

**Live mode later** is a backend that returns the same JSON for a new pin and video. The UI does not change.

## `index.json`

```json
{ "model": {"name": "typical-medium · helipad", "version": "plan1-800"},
  "sites": [ {"id": "S2", "round": "2026-09", "verdict": "probably_fails"} ] }
```

## `sites/<id>.json`

```json
{
  "id": "S2",
  "name": {"en": "S2", "he": "S2"},
  "round": "2026-09",
  "aircraft": 1,
  "touchdown": {"lat": 31.0, "lon": 34.0, "moved": null},
  "expert": {"call": "not_approved", "reason": {"en": "…", "he": "…"}},
  "verdict": {
    "status": "probably_fails",
    "probs": {"approved": 0.02, "probably_fails": 0.95, "fails": 0.02, "not_checked": 0.01},
    "deciding": ["high_300"]
  },
  "items": [
    {"id": "high_300", "status": "probably_fails",
     "probs": {"passes": 0.03, "probably_fails": 0.96, "fails": 0.01, "not_checked": 0.0},
     "findings": [0]}
  ],
  "conditions": ["minefield_check", "low_wire_walk", "approach_heading", "notam"],
  "prep": ["wetting"],
  "findings": {
    "obstacles": [
      {"kind": "power_line", "label": "power line (voltage unknown)", "height_m": 24, "height_src": "unknown",
       "dist_m": 142.0, "bearing": 8, "in_pad": false, "source": "map",
       "xy": [0.52, 0.31]}
    ],
    "zones": [{"kind": "firing_zone", "dist_m": 500, "inside": false, "bearing": null}],
    "slope": {"deg": 2.0, "unc": 6.0},
    "canopy": {"pad_max_m": 0.4, "max_10_m": 1.1, "max_50_m": 7.9},
    "dust": "medium",
    "belly": null,
    "layers": {"osm": true, "dem": true, "govmap": false, "corridors": false},
    "counts": {"obstacles_total": 47, "within_300": 3}
  },
  "evidence": {
    "satellite": {"image": "img/S2_sat.jpg", "half_width_m": 400, "pad_xy": [0.5, 0.5]},
    "frames": {
      "source": "model",
      "items": [
        {"image": "img/S2_f03.jpg", "t": 6.5,
         "answers": {"wires": {"yes": 0.88, "no": 0.10, "cant_tell": 0.02}}}
      ]
    }
  }
}
```

## Rules the UI must follow

- **`verdict.probs` are derived from the item probabilities by the verdict rule** (any fails → fails; else missing core data → not checked; else any probably fails → probably fails; else approved), treating the items as independent. The bar therefore agrees with how the verdict is decided. `verdict.model_question_probs`, which may be absent, is the model's direct answer to the verdict question: a diagnostic only, not shown as the verdict.

- **Every probability shown with every option.** Each item, the verdict and each frame answer shows all options with their probabilities, never only the top one. `status` is the reported status, set by a decision rule; the distribution is always shown next to it.
- **Item ids:** `pad, very_low_10, low_50, high_300, ultra_1km, slope, zones, belly, dust, low_wires, minefield_500`.
- **Statuses:** `passes, fails, probably_fails, not_checked, prep, coordination`.
- **Verdicts:** `approved, probably_fails, fails, not_checked`.
- **Frame questions:** `wires, pole, tall_structure, tallest_height, tallest_distance, tallest_type, ground, vegetation_belly, dust`, plus the null option `cant_tell`. All UI text for ids, statuses, kinds, conditions and prep comes from the EN/HE dictionary. The data carries ids and numbers, not display strings. The exceptions are free-text `label`, `name` and `reason`.
- **Positions and directions.**
  - `xy` is the obstacle's position as a 0–1 fraction of the satellite image, with x to the right and y downward.
  - `half_width_m` gives the scale, so the 35 m pad and the 10/50/300 m rings around it can be drawn.
  - `bearing: null` means the direction is unknown (a video sighting): never draw it on the map.
- **Optional parts.** `evidence.frames` may be absent (no video yet), and `expert` may be null. The UI hides those sections rather than showing empty ones.
- **No hedging copy.** Show calls plainly. Uncertainty lives in the probabilities.
