# Helipad Plan 1: Decision Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a text decision Typical that gives the protocol item statuses and the site verdict for a helicopter landing zone from an English findings case. Train it through the existing Decision Factory job on rule-generated cases, and measure it on held-out rule cases, the real sites and the 6 expert verdicts.

**Architecture:**
- A `helipad/` package holds four parts:
  - the case model and English renderer;
  - a rule engine that mirrors the calibrated `lz_check.py` `evaluate()` item by item;
  - a converter from `lz_check` output to a case;
  - a balanced synthetic case generator.
- Scripts in three places:
  - one fetches the real sites with the vendored `lz_check.py` (online maps, Israeli IP);
  - one writes a Decision Factory dataset directory, trained unchanged with `scripts/typically_job.py helipad medium 800`;
  - one scores the trained model locally and reports item agreement, verdict consistency and expert direction.

**Tech Stack:**
- Python ≥3.11 via `uv`, pytest.
- The existing `pcdm` / `inference/typical` / `scripts/typically_*` stack.
- `lz_check.py` runs with `uv run --with shapely --with pyproj --with rasterio --with mercantile --with requests --with numpy --with pillow`. That keeps these geo dependencies out of `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-11-21-helipad-lz-decision-model-design.md` (tracks R1 and R4; R-rule data). Plans 2+ will cover the teacher corpus (R2), the vision student (R3), B vs B′ (R5) and local inference (R6), and are written after this plan's results.

## Global Constraints

- Python via `uv` only (`uv run …`); never bare `python`/`pip`.
- Case text is **English**. Only the site name may be Hebrew.
- Status vocabulary is fixed:
  - items: `passes`, `fails`, `probably_fails`, `not_checked`, `prep`, `coordination`;
  - verdict: `approved`, `probably_fails`, `fails`, `not_checked`;
  - human labels replace `_` with a space.
- Verdict precedence, mirroring `lz_check.py`: any `fails` → `fails`; else core map data missing (OSM or DEM) → `not_checked`; else any `probably_fails` → `probably_fails`; else `approved`.
- Height thresholds: very low 1.2–4.5 m within 10 m of the pad edge; low 4.5–24 m within 50 m; high 24–76 m within 300 m; ultra ≥76 m → sector within 1 km. Minefield 500 m from touchdown. Pad 35×35 m.
- Aircraft: aircraft 1 has a rotor obstacle above 1.0 m and max slope 15°; aircraft 2 has a rotor obstacle above 2.0 m and max slope 10°.
- "Certain" height sources are `tagged`, `levels`, `iec` and `measured`. Every other source can at most give `probably_fails` (calibrated rules 1 and 3).
- "Not found on the map" never yields `passes` for low wires or minefields. Ground condition (belly, dust) is only ever `prep`, `passes` or `not_checked`, and never changes the verdict.
- Never tune rules to the real sites. A mismatch with `lz_check` is recorded, not patched per site.
- Decision Factory job slug `helipad`; base `medium`; steps ∈ {200, 400, 800}.
- GovMap answers Israeli IPs only: run the map fetch on the user's machine, never on RunPod.
- Commit after each task on the current branch, ending every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A site where every map layer failed must give `not_checked`, never `approved`. → `test_core_missing_is_not_checked_not_approved` (Task 3).
2. An obstacle exactly at a ring distance (e.g. 50.0 m) is outside the ring (`lz_check` uses `<`). → `test_ring_boundary_is_exclusive` (Task 3).
3. Aircraft 2 differs: a 1.8 m fence in the pad fails aircraft 1 and passes aircraft 2; 12° slope passes aircraft 1 and fails aircraft 2. → `test_aircraft2_thresholds` (Task 3).
4. OSM names and labels are Hebrew; none may leak into case text. → `test_render_has_no_hebrew_outside_name` (Task 2) and `test_from_lz_labels_are_english` (Task 4).
5. A real site has up to 60 obstacles; the case must fit `max_state` 1024 tokens. → `test_render_caps_obstacles_and_length` (Task 2).

---

### Task 1: Vendor the existing pipeline, register the real sites, fix the spec's verdict order

**Files:**
- Create: the private lz_check pipeline (`$HELIPAD_PRIVATE/heli_lz`): a copy of the partner's Claude-skills pipeline (heli-lz-check, private): its scripts, references and evals, plus its calibration notes and README
- Create: `helipad/__init__.py` (empty), `helipad/sites.json` (now private: `$HELIPAD_PRIVATE/sites.json`)
- Modify: `docs/superpowers/specs/2026-11-21-helipad-lz-decision-model-design.md` (§4 verdict logic)
- Test: `tests/test_helipad_sites.py`

**Interfaces:**
- Produces: `helipad/sites.json`, a list of `{"slug": str, "public_id": str, "name": str, "round": "2026-09"|"2026-10", "link": str|null, "lat": float|null, "lon": float|null, "expert": "approved"|"not_approved"|null, "expert_reason": str|null}`
  - September: the 6 sites with `expert` set and `expert_reason` copied from the source brief (private) §3.
  - October: the 10 sites with `expert: null`. Use the links and coordinates from the source brief (private).
  - Slugs: ASCII, `^[a-z0-9_]+$`, kept in the private registry. Public IDs, in registry order: `S1`–`S6` (September), `O1`–`O10` (October).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_helipad_sites.py
import json, re
from pathlib import Path

def test_sites_registry():
    sites = json.loads(Path("helipad/sites.json").read_text())
    assert len(sites) == 16
    assert len({s["slug"] for s in sites}) == 16
    assert all(re.fullmatch(r"[a-z0-9_]+", s["slug"]) for s in sites)
    sept = [s for s in sites if s["round"] == "2026-09"]
    assert sorted(s["expert"] for s in sept) == ["approved"] * 3 + ["not_approved"] * 3
    assert all(s["expert"] is None for s in sites if s["round"] == "2026-10")
    assert all(s["link"] or (s["lat"] and s["lon"]) for s in sites if s["round"] == "2026-10")
```

- [ ] **Step 2: Run it, expect FAIL** — `uv run pytest tests/test_helipad_sites.py -v` → FileNotFoundError.

- [ ] **Step 3: Recover the September coordinates.**
  - Open the saved Maps list (link in the source brief (private)) in a browser (Claude in Chrome or the built-in browser) and read each pin's `lat,lon`.
  - If the list can't be read, set those six `lat`/`lon` to `null` and tell the user. Task 5 then blocks on the user supplying them.

- [ ] **Step 4: Copy the pipeline files and write `helipad/sites.json`.** Add a one-line README header in the private lz_check pipeline (`$HELIPAD_PRIVATE/heli_lz`): source = the source brief (private), scripts dated 4.11.2026, do not edit, re-copy instead.

- [ ] **Step 5: Fix the spec.** Make two changes in §4.
  - In "Verdict logic", replace the four bullets with the precedence from Global Constraints (`fails` > missing core data → `not_checked` > `probably_fails` > `approved`), and add the note "mirrors `lz_check.py` `evaluate()`".
  - Make the item table's "Allowed statuses" column match `ALLOWED` in Task 2: `ultra_1km` = passes / coordination / not checked; `slope` = passes / fails / not checked; `minefield_500` = not checked / fails.

- [ ] **Step 6: Run the test, expect PASS.**

- [ ] **Step 7: Commit** — `git add helipad tests/test_helipad_sites.py docs/superpowers/specs && git commit -m "helipad: register the 16 real sites"` (the pipeline and registry stay private)

---

### Task 2: Case model and English renderer

**Files:**
- Create: `helipad/case.py`
- Test: `tests/test_helipad_case.py`

**Interfaces:**
- Produces:

```python
ITEMS = ("pad", "very_low_10", "low_50", "high_300", "ultra_1km", "slope", "zones",
         "belly", "dust", "low_wires", "minefield_500")
ALLOWED: dict[str, tuple[str, ...]] = {
    "pad": ("passes", "fails", "probably_fails", "not_checked"),
    "very_low_10": ("passes", "fails", "probably_fails", "not_checked"),
    "low_50": ("passes", "fails", "probably_fails", "not_checked"),
    "high_300": ("passes", "fails", "probably_fails", "not_checked"),
    "ultra_1km": ("passes", "coordination", "not_checked"),
    "slope": ("passes", "fails", "not_checked"),
    "zones": ("passes", "coordination", "not_checked"),
    "belly": ("passes", "prep", "not_checked"),
    "dust": ("passes", "prep", "not_checked"),
    "low_wires": ("not_checked", "probably_fails", "fails"),
    "minefield_500": ("not_checked", "fails"),
}
VERDICTS = ("approved", "probably_fails", "fails", "not_checked")
CERTAIN_SRC = frozenset({"tagged", "levels", "iec", "measured"})
def label(status: str) -> str            # "probably_fails" -> "probably fails"

@dataclass(frozen=True)
class Obstacle:
    kind: str            # power_line, power_pole, power_tower, lighting_pole, lit_road, building, wall, fence,
                         # tree, tree_row, woodland, orchard, vineyard, scrub, mast, crane, wind_turbine, wire, vehicle, other
    label: str           # English, e.g. "power line (voltage unknown)"
    height_m: float
    height_src: str      # tagged | levels | iec | measured | estimated | unknown
    dist_edge_m: float   # from the 35x35 pad edge; 0 when in_pad
    in_pad: bool
    bearing: int | None  # compass deg to the nearest point; None = unknown (video)
    source: str          # map | video | map+video

@dataclass(frozen=True)
class Zone:
    kind: str            # firing_zone | military_area | nature_reserve | minefield | hv_corridor
    dist_center_m: float
    inside_pad: bool
    bearing: int | None

@dataclass
class SiteCase:
    name: str
    aircraft: int                        # 1 | 2
    obstacles: list[Obstacle]
    zones: list[Zone]
    slope_deg: float | None              # None = DEM missing
    slope_unc_deg: float                 # 6 for 30 m DEM
    canopy: dict | None                  # {"pad_max_m", "max_10_m", "max_50_m"} or None
    dust: str | None                     # light | medium | heavy | None
    belly: str | None                    # clear | obstacles | None
    govmap_ok: bool
    core_missing: tuple[str, ...]        # subset of ("osm", "dem")
    shift: dict | None = None            # {"m": int, "dir": str} when the touchdown point was moved

def render_case(c: SiteCase, max_obstacles: int = 25) -> str
```

`render_case` output shape: lines in this order.
1. `Site: …`
2. `Aircraft: …`
3. `Touchdown: pin` or `Touchdown: moved 22 m north-west of the pin`
4. One `Obstacle:` line per obstacle (sorted by `dist_edge_m`, at most `max_obstacles`, in-pad first), e.g. `Obstacle: power line (voltage unknown) · height unknown (treated as 24 m) · 140 m from pad edge · north (8°) · source map`. Height phrasing: `height 9 m (tagged)` for certain sources, `height est. ~6 m` for estimated, `height unknown (treated as N m)` for unknown.
5. `Zone:` lines.
6. `Slope in pad: 2.0° (±6°)` or `Slope: not available`.
7. `Tree canopy: pad max 0.4 m; within 10 m 1.1 m; within 50 m 7.9 m (2009–2020 layer)` or `Tree canopy: not available`.
8. `Dust: medium`, or `Dust: not assessed`.
9. `Belly area: clear`, `has obstacles`, or `not assessed`.
10. `Zones layer (GovMap): available` or `not available`.
11. `Missing core data: OSM`, only if any is missing.
12. `Not available from any source: low wires, minefields.`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_helipad_case.py
import re
from helipad.case import Obstacle, Zone, SiteCase, render_case, label, ALLOWED, ITEMS

def _case(**kw):
    base = dict(name="site_x", aircraft=1, obstacles=[], zones=[], slope_deg=2.0, slope_unc_deg=6,
                canopy=None, dust=None, belly=None, govmap_ok=True, core_missing=())
    return SiteCase(**{**base, **kw})

def test_label():
    assert label("probably_fails") == "probably fails"

def test_allowed_covers_items():
    assert set(ALLOWED) == set(ITEMS)

def test_render_unknown_height_line():
    o = Obstacle("power_line", "power line (voltage unknown)", 24, "unknown", 140, False, 8, "map")
    t = render_case(_case(obstacles=[o]))
    assert "height unknown (treated as 24 m)" in t and "140 m from pad edge" in t and "north" in t
    assert t.rstrip().endswith("Not available from any source: low wires, minefields.")

def test_render_has_no_hebrew_outside_name():
    o = Obstacle("building", "building", 6, "estimated", 30, False, 200, "map")
    t = render_case(_case(name="\u05e9\u05dd", obstacles=[o], zones=[Zone("firing_zone", 400, False, None)]))
    body = t.split("\n", 1)[1]
    assert not re.search(r"[\u0590-\u05ff]", body)

def test_render_caps_obstacles_and_length():
    obs = [Obstacle("tree", "tree", 8, "estimated", float(i), False, i % 360, "map") for i in range(60)]
    t = render_case(_case(obstacles=obs))
    assert t.count("Obstacle:") == 25 and len(t) < 4000

def test_render_missing_core():
    t = render_case(_case(slope_deg=None, core_missing=("osm", "dem")))
    assert "Missing core data: OSM, DEM" in t and "Slope: not available" in t
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_helipad_case.py -v` → ImportError.
- [ ] **Step 3: Implement `helipad/case.py`.** Compass words come from 8 sectors of 45° centred on N (N, NE, …), written out in full ("north-east").
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "helipad: case model and English renderer"` (plus trailer).

---

### Task 3: Rule engine (protocol + calibrated rules, mirroring `lz_check.evaluate`)

**Files:**
- Create: `helipad/rules.py`
- Test: `tests/test_helipad_rules.py`

**Interfaces:**
- Consumes: Task 2's `SiteCase`, `Obstacle`, `Zone`, `ALLOWED`, `VERDICTS`, `CERTAIN_SRC`.
- Produces: `judge(c: SiteCase) -> dict[str, str]` (every item in `ITEMS`, each value in `ALLOWED[item]`), `verdict(c: SiteCase, items: dict[str, str]) -> str`.

Item rules. Worst status wins within an item, ordered `fails` > `probably_fails` > `coordination` > `prep` > `not_checked` > `passes`.

| Item | Rule (source: `lz_check.py` `evaluate` in the private lz_check pipeline (`$HELIPAD_PRIVATE/heli_lz`)) |
|---|---|
| pad | `fails` if any obstacle `in_pad` and `height_m > rotor`, or canopy `pad_max_m > rotor`. Else `not_checked` if `"osm" in core_missing`. Else `passes` |
| very_low_10 / low_50 / high_300 | For obstacles not in the pad with `hmin <= height_m < hmax` and `dist_edge_m < ring`: `fails` if `height_src in CERTAIN_SRC`, else `probably_fails`. `high_300` is also `probably_fails` for an `hv_corridor` zone with `dist_center_m - 17.5 < 300`. Otherwise `not_checked` if OSM is missing, else `passes`. Canopy rings never change these items (`lz_check` only warns on them): `# ponytail: mirrors lz_check WARN; revisit with the expert` |
| ultra_1km | `coordination` if any obstacle has `height_m >= 76` and `dist_edge_m <= 1000`. Else `not_checked` if OSM is missing, else `passes` |
| slope | `not_checked` if `slope_deg is None`. `fails` if `slope_deg - unc > max_slope`. `not_checked` if `slope_deg + unc > max_slope` (borderline: field measurement needed). Else `passes` |
| zones | `coordination` if any zone of kind firing_zone / military_area / nature_reserve, or a minefield with `dist_center_m > 500`. Else `not_checked` if `not govmap_ok`. Else `passes` |
| minefield_500 | `fails` if a minefield zone has `dist_center_m <= 500`, else `not_checked` (never `passes`) |
| belly | `clear` → `passes`, `obstacles` → `prep`, `None` → `not_checked` |
| dust | `light` → `passes`, `medium`/`heavy` → `prep`, `None` → `not_checked` |
| low_wires | `fails` if a `wire` obstacle is `in_pad`. `probably_fails` if a `wire` obstacle has `dist_edge_m < 50`. Else `not_checked` |

`verdict` follows the Global Constraints precedence.

- [ ] **Step 1: Write the failing tests** (use `_case` from Task 2's test as a local helper)

```python
# tests/test_helipad_rules.py
from helipad.case import Obstacle, Zone, SiteCase, ALLOWED, ITEMS
from helipad.rules import judge, verdict

from tests.test_helipad_case import _case
def ob(kind="building", h=6.0, src="estimated", d=100.0, in_pad=False, source="map"):
    return Obstacle(kind, kind, h, src, d, in_pad, 0, source)

def test_every_item_judged_within_allowed():
    s = judge(_case())
    assert set(s) == set(ITEMS) and all(s[i] in ALLOWED[i] for i in ITEMS)

def test_unknown_power_line_in_300_is_probably_fails():          # calibrated rule 1
    c = _case(obstacles=[ob("power_line", 24, "unknown", 140)])
    s = judge(c); assert s["high_300"] == "probably_fails" and verdict(c, s) == "probably_fails"

def test_tagged_height_in_ring_fails():
    c = _case(obstacles=[ob("building", 9, "levels", 30)])
    s = judge(c); assert s["low_50"] == "fails" and verdict(c, s) == "fails"

def test_estimated_building_in_ring_is_probably():                # calibrated rule 3
    assert judge(_case(obstacles=[ob("building", 6, "estimated", 30)]))["low_50"] == "probably_fails"

def test_ring_boundary_is_exclusive():
    assert judge(_case(obstacles=[ob("tree", 8, "measured", 50.0)]))["low_50"] == "passes"
    assert judge(_case(obstacles=[ob("tree", 8, "measured", 49.9)]))["low_50"] == "fails"

def test_unknown_height_pole_in_pad_fails_pad():
    assert judge(_case(obstacles=[ob("power_pole", 24, "unknown", 0, True)]))["pad"] == "fails"

def test_aircraft2_thresholds():
    fence = [ob("fence", 1.8, "estimated", 0, True)]
    assert judge(_case(obstacles=fence))["pad"] == "fails"
    assert judge(_case(aircraft=2, obstacles=fence))["pad"] == "passes"
    assert judge(_case(slope_deg=12, slope_unc_deg=0))["slope"] == "passes"
    assert judge(_case(aircraft=2, slope_deg=12, slope_unc_deg=0))["slope"] == "fails"

def test_core_missing_is_not_checked_not_approved():
    c = _case(slope_deg=None, core_missing=("osm", "dem"), govmap_ok=False)
    assert verdict(c, judge(c)) == "not_checked"

def test_fail_beats_missing_core():
    c = _case(core_missing=("dem",), slope_deg=None, obstacles=[ob("building", 6, "estimated", 0, True)])
    assert verdict(c, judge(c)) == "fails"

def test_ground_never_changes_verdict():                          # calibrated rule 4
    c = _case(dust="heavy", belly="obstacles")
    s = judge(c); assert (s["dust"], s["belly"]) == ("prep", "prep") and verdict(c, s) == "approved"

def test_firing_zone_is_coordination_not_failure():               # calibrated rule 7
    c = _case(zones=[Zone("firing_zone", 0, True, None)])
    s = judge(c); assert s["zones"] == "coordination" and verdict(c, s) == "approved"

def test_minefield_never_passes():
    assert judge(_case())["minefield_500"] == "not_checked"
    assert judge(_case(zones=[Zone("minefield", 400, False, 90)]))["minefield_500"] == "fails"

def test_hv_corridor_within_300_probably():
    assert judge(_case(zones=[Zone("hv_corridor", 250, False, 0)]))["high_300"] == "probably_fails"

def test_wire_over_pad_fails():
    assert judge(_case(obstacles=[ob("wire", 6, "estimated", 0, True, "video")]))["low_wires"] == "fails"
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_helipad_rules.py -v`.
- [ ] **Step 3: Implement `helipad/rules.py`** as specified in the table.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "helipad: protocol rule engine mirroring lz_check evaluate"`.

---

### Task 4: `lz_check` output → case

**Files:**
- Create: `helipad/from_lz.py`
- Create: `tests/fixtures/helipad/lz_min.json` (a hand-written minimal `lz.json`: 3 obstacles (an OSM `power=line` with no height, a building with `building:levels=2`, an `iec/…` line), one `military` zone, one GovMap hit at tolerance `"500"` with a layer containing `שטחי אש`, slope available, canopy available, dust `בינוני`, verdict `לא עומד`)
- Test: `tests/test_helipad_from_lz.py`

**Interfaces:**
- Consumes: the `lz.json` written by Task 5: `{"verdict": <lz_check verdict.json dict>, "govmap": <data.json["govmap"]>, "osm_ok": bool, "dem_ok": bool}`. Used `verdict.json` fields:
  - `aircraft`, `verdict`, `shift`;
  - `obstacles_sorted[]` with `label`, `height_m`, `height_src`, `dist_pad_edge_m`, `in_pad`, `near_bearing`, `osm`, `tags`;
  - `zones[]` with `kind`, `dist_center_m`, `inside_pad`, `near_bearing`;
  - `slope{available, pad_slope_plane_deg, pad_slope_pixel_max_deg, uncertainty_deg}`;
  - `canopy{available, pad_max_m, max_within_10m_of_pad, max_within_50m_of_pad}`;
  - `dust{estimated_level}`.
- Produces: `case_from_lz(lz: dict, name: str) -> SiteCase`, `lz_verdict(lz: dict) -> str` (in `VERDICTS`), `kind_from_tags(tags: dict, osm_id: str) -> tuple[str, str]` (kind, English label).

Mapping rules:
- Kind, following `lz_check.est_height`'s branches:
  - `osm` id starting `iec/` → `power_line`, labelled with "kV" when the voltage is known;
  - `power` line/minor_line → `power_line`; tower/portal → `power_tower`; pole → `power_pole`;
  - lit or motorway/trunk `highway` → `lit_road`;
  - `street_lamp` → `lighting_pole`;
  - `building` → `building`;
  - `barrier` fence/hedge/guard_rail → `fence`; wall/retaining_wall → `wall`;
  - `natural` tree / tree_row / wood → `tree` / `tree_row` / `woodland`;
  - `landuse` orchard/vineyard → `orchard`/`vineyard`;
  - `man_made` mast/tower → `mast`; crane → `crane`;
  - wind generator → `wind_turbine`;
  - else `other`.

  Labels are built from the kind (plus kV for power) only. **Never** copy `name` or Hebrew `label` text.
- Zones:
  - `military` → `military_area`; `protected` → `nature_reserve`; `minefield/danger` → `minefield`; `hv_corridor` → `hv_corridor`; `aeroway` is dropped.
  - GovMap hits: tolerance `"5"` → `inside_pad=True, dist_center_m=0`; `"500"`/`"1000"` → `dist_center_m=500/1000`.
  - A hit's kind is `firing_zone` if `"שטחי אש"` is in its layer, else `nature_reserve`.
- `govmap_ok = bool(lz["govmap"].get("hits"))`. This mirrors `lz_check`, which treats an empty hits dict as unavailable.
- Slope: `pad_slope_plane_deg`, falling back to `pad_slope_pixel_max_deg`; `None` when unavailable.
- Dust: `קל→light`, `בינוני→medium`, `כבד→heavy`, else `None`. `belly=None`.
- `core_missing`: `"osm"` if `not osm_ok`, `"dem"` if `not dem_ok`.
- `lz_verdict`: `לא עומד→fails`, `כנראה לא עומד→probably_fails`, `מאושר→approved`, starts with `לא נבדק→not_checked`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_helipad_from_lz.py
import json, re
from pathlib import Path
from helipad.from_lz import case_from_lz, lz_verdict, kind_from_tags
from helipad.case import render_case
from helipad.rules import judge, verdict

LZ = json.loads(Path("tests/fixtures/helipad/lz_min.json").read_text())

def test_kind_from_tags():
    assert kind_from_tags({"power": "line"}, "way/1")[0] == "power_line"
    assert kind_from_tags({"highway": "secondary", "lit": "yes"}, "way/2")[0] == "lit_road"
    assert kind_from_tags({}, "iec/7")[0] == "power_line"

def test_from_lz_labels_are_english():
    c = case_from_lz(LZ, "\u05e9\u05dd")
    assert not re.search(r"[\u0590-\u05ff]", render_case(c).split("\n", 1)[1])

def test_from_lz_fields():
    c = case_from_lz(LZ, "x")
    assert c.dust == "medium" and c.govmap_ok and any(z.kind == "firing_zone" for z in c.zones)

def test_rules_reproduce_lz_verdict_on_fixture():
    c = case_from_lz(LZ, "x")
    assert verdict(c, judge(c)) == lz_verdict(LZ)
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Write the fixture, then implement `helipad/from_lz.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "helipad: lz_check output to findings case"`.

---

### Task 5: Fetch the real sites (online maps) and check rule-engine agreement

**Files:**
- Create: `scripts/helipad_sites.py`
- Test: `tests/test_helipad_sites_script.py`

**Interfaces:**
- Consumes: `helipad/sites.json` (Task 1), `case_from_lz`/`lz_verdict` (Task 4), `judge`/`verdict` (Task 3), `lz_check.py` in the private lz_check pipeline (`$HELIPAD_PRIVATE/heli_lz`) (CLI: `locate <input>`, `fetch --lat --lon --out`, `evaluate --data <dir>/data.json --aircraft 1`).
- Produces: `.context/helipads/sites/<slug>/{data.json, verdict.json, lz.json}`, plus the function `agreement(sites_dir: Path) -> list[dict]` (rows `{"slug", "lz": str, "rules": str, "match": bool}`).

CLI:
- `uv run scripts/helipad_sites.py fetch [--only slug,…]`:
  - resolves `link` with `lz_check.py locate` when `lat` is null;
  - skips any site whose `lz.json` exists (the cache);
  - runs fetch + evaluate through `uv run --with shapely --with pyproj --with rasterio --with mercantile --with requests --with numpy --with pillow python $HELIPAD_PRIVATE/heli_lz/scripts/lz_check.py …`;
  - writes `lz.json`, with `osm_ok = bool((data["osm"] or {}).get("elements") is not None)` and `dem_ok = verdict["slope"].get("available", False)`.
- `uv run scripts/helipad_sites.py check` prints the agreement table and exits 1 on any mismatch.

- [ ] **Step 1: Write the failing test** (offline: it builds a sites dir from the Task 4 fixture)

```python
# tests/test_helipad_sites_script.py
import json, shutil
from pathlib import Path
from scripts.helipad_sites import agreement

def test_agreement_reads_lz_json(tmp_path):
    d = tmp_path / "s1"; d.mkdir()
    shutil.copy("tests/fixtures/helipad/lz_min.json", d / "lz.json")
    rows = agreement(tmp_path)
    assert rows == [{"slug": "s1", "lz": "fails", "rules": "fails", "match": True}]
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `scripts/helipad_sites.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Fetch for real (user's machine, Israeli IP, ~3 min per site)** — `uv run scripts/helipad_sites.py fetch`. Expected: 16 `lz.json` files, or a printed list of sites that couldn't be located. If the September coordinates are null, stop and ask the user for them.
- [ ] **Step 6: Check agreement** — `uv run scripts/helipad_sites.py check`.
  - Expected: every row matches.
  - On a mismatch, first look for a bug in `from_lz`/`rules`; fix it only if it violates a rule in Task 3's table.
  - Record any remaining mismatch, with its cause, in `.context/helipads/AGREEMENT.md`. Do not special-case sites.
- [ ] **Step 7: Commit** (script and test only; `.context` is gitignored) — `git commit -m "helipad: fetch real sites with lz_check, rule agreement check"`.

---

### Task 6: Balanced synthetic case generator

**Files:**
- Create: `helipad/generate.py`
- Test: `tests/test_helipad_generate.py`

**Interfaces:**
- Consumes: Task 2 types, Task 3 `judge`/`verdict`.
- Produces: `sample_case(rng: random.Random) -> SiteCase`; `sample_balanced(n: int, seed: int, mix: dict[str, float] = VERDICT_MIX) -> list[tuple[SiteCase, dict[str, str], str]]` (case, item statuses, verdict); `VERDICT_MIX = {"approved": .35, "probably_fails": .35, "fails": .20, "not_checked": .10}`.

Sampling requirements (the builder picks the remaining distributions):
- `aircraft` is 1 with probability .7, else 2.
- 0–12 obstacles. Kinds and default heights follow `lz_check.est_height`.
- Power kinds: `height_src` is `unknown` .7, `tagged` .15, `iec` .15. Others: `estimated` .7, `levels`/`tagged`/`measured` .3.
- Distances: 40% are drawn within ±40% of a ring threshold (10/50/300); `in_pad` .05; the rest log-uniform up to 1100 m.
- `source`: `map` .6, `video` .2, `map+video` .2. Video-only obstacles get `bearing=None` and `estimated`/`unknown` heights. `wire` obstacles are video-only.
- Zones: firing zone .15 (a third of them inside the pad), reserve .10, minefield .02, hv_corridor .05. `govmap_ok` .85.
- Slope: `abs(gauss(3, 4))`, with 8% of cases at 10–25°. `core_missing` is `("osm",)` in 8% of cases, `("dem",)` in 3%.
- Canopy is present 80% of the time. `dust`/`belly` are set in 50% of cases (video), else `dust` alone in 30%.
- `sample_balanced` uses rejection sampling to hit the verdict mix. Names are `syn_<seed>_<i>`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_helipad_generate.py
from collections import Counter
from helipad.generate import sample_balanced, VERDICT_MIX
from helipad.case import ALLOWED
from helipad.rules import judge, verdict

def test_deterministic():
    a = sample_balanced(50, seed=3); b = sample_balanced(50, seed=3)
    assert [x[2] for x in a] == [x[2] for x in b]

def test_labels_are_rule_labels():
    for c, items, v in sample_balanced(200, seed=1):
        assert items == judge(c) and v == verdict(c, items)

def test_mix_and_coverage():
    data = sample_balanced(4000, seed=0)
    mix = Counter(v for _, _, v in data)
    for k, p in VERDICT_MIX.items():
        assert abs(mix[k] / 4000 - p) < 0.05
    seen = {(i, s) for _, items, _ in data for i, s in items.items()}
    assert seen == {(i, s) for i, ss in ALLOWED.items() for s in ss}   # every allowed status occurs
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `helipad/generate.py`.** If `test_mix_and_coverage` can't reach a rare status, adjust the sampling probabilities. Never adjust the rules.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "helipad: balanced rule-labelled case generator"`.

---

### Task 7: Decision Factory rows and dataset build

**Files:**
- Create: `helipad/rows.py`, `scripts/helipad_build.py`
- Test: `tests/test_helipad_rows.py`

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces:
  - `QUESTIONS: dict[str, str]` and `POLICY: dict[str, str]`, keyed by item id plus `"verdict"`. Each policy is one or two sentences stating that item's rule from Task 3's table in plain English. The `"verdict"` policy states the precedence.
  - `case_rows(c: SiteCase, items: dict[str, str] | None, v: str | None, expert: str | None = None) -> list[dict]`. Returns 12 rows (11 items + verdict) when `items` is given; returns only the verdict row when `items is None` (expert rows).
  - Row schema, identical to `scripts/typically_spike.row`: `{"state": render_case(c), "query": QUESTIONS[k] + "\nRule: " + POLICY[k], "candidates": [label(s) for s in ALLOWED[k] or VERDICTS], "target": [...], "p_null": 0.0, "task": "helipad", "label": argmax(target), "meta": {"qtype": "choice", "family": k, "case": c.name, "core_missing": list(c.core_missing), **({"expert": expert} if expert else {})}}`.
  - Expert target: `approved` → one-hot `approved`. `not_approved` → `0.5` on `probably fails` and `0.5` on `fails`, so either counts as the expert's direction.

`scripts/helipad_build.py --n-train 6000 --n-val 600 --n-holdout 600 --seed 0 --sites .context/helipads/sites --out .context/typically/jobs/helipad` writes:
- `train.jsonl`, from `sample_balanced(n_train, seed)`;
- `val.jsonl`, from seed + 1;
- `eval/import_oneliner.jsonl`: the rule holdout, from seed + 2. The Decision Factory pod requires this exact name; it drives the base vs tuned comparison in `reveal.json`.
- `eval/helipad_real.jsonl`: every fetched real site, labelled by the rules on its `lz.json` case;
- `eval/helipad_expert.jsonl`: verdict rows for the September sites, with the expert target.

Any val or holdout case whose `state` also appears in train is dropped. The script prints the row count per file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_helipad_rows.py
from helipad.generate import sample_balanced
from helipad.rows import case_rows, QUESTIONS, POLICY
from helipad.case import ITEMS

def test_rows_shape():
    c, items, v = sample_balanced(1, seed=5)[0]
    rows = case_rows(c, items, v)
    assert len(rows) == 12 and {r["meta"]["family"] for r in rows} == set(ITEMS) | {"verdict"}
    for r in rows:
        assert abs(sum(r["target"]) - 1) < 1e-9 and r["candidates"][r["label"]] in r["candidates"]
        assert "Rule:" in r["query"] and r["task"] == "helipad"

def test_expert_row_soft_target():
    c, _, _ = sample_balanced(1, seed=6)[0]
    (r,) = case_rows(c, None, None, expert="not_approved")
    t = dict(zip(r["candidates"], r["target"]))
    assert t["probably fails"] == 0.5 and t["fails"] == 0.5 and r["meta"]["expert"] == "not_approved"

def test_policy_covers_all():
    assert set(QUESTIONS) == set(POLICY) == set(ITEMS) | {"verdict"}
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `helipad/rows.py` and `scripts/helipad_build.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Build the dataset** — `uv run scripts/helipad_build.py`. Expected rows: train ≈72,000; val ≈7,200; `import_oneliner` ≈7,200; `helipad_real` = 12 × the number of fetched sites; `helipad_expert` = 6.
- [ ] **Step 6: Token check** — `uv run python -c "from transformers import AutoTokenizer as A; import json; t=A.from_pretrained('Qwen/Qwen3.5-4B-Base'); m=max(len(t(json.loads(l)['state'])['input_ids']) for f in ['train','eval/helipad_real'] for l in open(f'.context/typically/jobs/helipad/{f}.jsonl')); print(m); assert m < 1024"`. Expected: prints the maximum, below 1024.
- [ ] **Step 7: Commit** — `git commit -m "helipad: Decision Factory rows and dataset build"`.

---

### Task 8: Train through Decision Factory, score, report

**Files:**
- Create: `scripts/helipad_score.py`
- Test: `tests/test_helipad_score.py`

**Interfaces:**
- Consumes:
  - The job result directory `.context/typically/results/co_helipad/` (`best.pt`, `eval_co.json`, `base_eval_co.json`, `reveal.json`).
  - The eval files from Task 7.
  - `typically_reveal.score(m, state, qs)` and `typically_reveal.queries(rows)`.
  - `typical.Typical.from_pretrained(path, backbones={})` plus `.activate()`.
- Produces:
  - `report(rows: list[dict], preds: list[dict]) -> dict`. A pure function: `preds[i]` is the `{"probs": {label: p}, "argmax": label}` for `rows[i]`. Returns:
    - `item_agreement` (per family, and `overall` over items);
    - `verdict_agreement`;
    - `consistency` (the share of cases whose verdict argmax equals the precedence applied to that case's item argmaxes and `meta.core_missing`);
    - `expert` (`{"n", "direction_match"}`, where direction = approved vs not approved).
  - CLI: `uv run scripts/helipad_score.py --ckpt <best.pt> --files <jsonl…> --out .context/helipads/eval/<name>.json`. It groups rows by `state`, so there is one KV encode per case, and writes `report` per file plus per-case explanation records `{case, verdict, items: {k: [argmax, p]}}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_helipad_score.py
from scripts.helipad_score import report

def _row(fam, cands, gold, case="c1", core=(), expert=None):
    meta = {"family": fam, "case": case, "core_missing": list(core), **({"expert": expert} if expert else {})}
    return {"candidates": cands, "label": cands.index(gold), "meta": meta}

def _pred(ans, cands): return {"argmax": ans, "probs": {c: float(c == ans) for c in cands}}

V = ["approved", "probably fails", "fails", "not checked"]
H = ["passes", "fails", "probably fails", "not checked"]

def test_consistency_and_agreement():
    rows = [_row("high_300", H, "probably fails"), _row("verdict", V, "probably fails")]
    preds = [_pred("probably fails", H), _pred("approved", V)]   # item says probably, verdict says approved
    r = report(rows, preds)
    assert r["item_agreement"]["overall"] == 1.0 and r["verdict_agreement"] == 0.0 and r["consistency"] == 0.0

def test_expert_direction():
    rows = [_row("verdict", V, "probably fails", case="a", expert="not_approved"),
            _row("verdict", V, "approved", case="b", expert="approved")]
    preds = [_pred("fails", V), _pred("probably fails", V)]
    assert report(rows, preds)["expert"] == {"n": 2, "direction_match": 1}
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `scripts/helipad_score.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "helipad: scoring and report"`.
- [ ] **Step 6: Train** — `uv run --no-project python scripts/typically_job.py helipad medium 800`. Expected: `.context/typically/jobs/helipad/status.json` reaches `done`, and `.context/typically/results/co_helipad/` holds `best.pt`, `eval_co.json`, `base_eval_co.json` and `reveal.json`. RunPod costs money: confirm with the user before this step.
- [ ] **Step 7: Score** — `uv run scripts/helipad_score.py --ckpt .context/typically/results/co_helipad/best.pt --files .context/typically/jobs/helipad/eval/helipad_real.jsonl .context/typically/jobs/helipad/eval/helipad_expert.jsonl .context/typically/jobs/helipad/eval/import_oneliner.jsonl --out .context/helipads/eval/plan1.json`.
  - Runs locally, the same way the Decision Factory server loads medium models.
  - If MPS is too slow or hangs, run it on the training pod instead (memory: `gpu-pod-over-local`).
- [ ] **Step 8: Write the results** to `docs/research/HELIPAD_PLAN1_RESULTS.md`:
  - the per-file `report`;
  - base vs tuned from `reveal.json` and `eval_co.json`;
  - the agreement table from Task 5;
  - the spec's R4 gates: item agreement ≥ 90%, consistency ≥ 95%, and the expert direction count.

  State failures plainly. Then commit: `git commit -m "helipad: plan 1 results"`.
