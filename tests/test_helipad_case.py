import re

import pytest

from helipad.case import (ALLOWED, DEFAULT_CAP, ITEMS, PAD_HALF, RINGS, ZONE_CAP, Obstacle, SiteCase, Zone, label,
                          render_case, select_obstacles, visible)
from helipad.rules import AIRCRAFT, judge


def _case(**kw):
    base = dict(name="site_x", aircraft=1, obstacles=[], zones=[], slope_deg=2.0, slope_unc_deg=6,
                canopy=None, dust=None, belly=None, govmap_ok=True, core_missing=())
    return SiteCase(**{**base, **kw})


def _ob(kind="building", h=3.0, src="estimated", d=30.0, in_pad=False, bearing=0, center=None):
    return Obstacle(kind, kind, h, src, d, in_pad, bearing, "map", center)


def _worst():
    """Every line present and at its longest: shift, both layers down, both core sources missing, 30 distinct zones of
    the longest kind, 60 long obstacle lines, numbers next to a limit (printed with two decimals)."""
    obs = [Obstacle("lit_road", "lit road (row of lighting poles)", 76.04, "unknown", 982.46, False, 315,
                    "map+video", 999.96) for i in range(60)]
    zones = [Zone("hv_corridor", 1000.04, False, 300 + i) for i in range(30)]
    canopy = {"pad_max_m": 1.04, "max_10_m": 23.96, "max_50_m": 23.96, "pad_over_rotor": True}
    return _case(aircraft=2, obstacles=obs, zones=zones, slope_deg=4.04, slope_unc_deg=6, canopy=canopy, dust="medium",
                 belly="obstacles", govmap_ok=False, corridors_ok=False, core_missing=("osm", "dem"),
                 shift={"m": 22, "dir": "north-west"})


def test_label():
    assert label("probably_fails") == "probably fails"


def test_allowed_covers_items():
    assert set(ALLOWED) == set(ITEMS)


def test_new_fields_default():
    assert _ob().dist_center_m is None and _case().corridors_ok is True


def test_render_compact_obstacle_line():
    o = Obstacle("power_line", "power line (voltage unknown)", 24, "unknown", 140, False, 8, "map")
    t = render_case(_case(obstacles=[o]))
    assert "power line (voltage unknown) | height unknown, treated as 24.0 m | 140.0 m from pad edge | north 8° | map" in t


def test_render_one_decimal():                                    # I10: 49.6 m must not print as 50 m
    t = render_case(_case(obstacles=[_ob("tree", 23.6, "measured", 49.6)]))
    assert "height 23.6 m (measured)" in t and "49.6 m from pad edge" in t


def test_visible_never_rounds_onto_a_limit():                     # I8: 49.96 m is inside the 50 m ring
    c = visible(_case(obstacles=[_ob("tree", 8, "measured", 49.96)]))
    assert c.obstacles[0].dist_edge_m == 49.96 and judge(c)["low_50"] == "fails"
    assert "49.96 m from pad edge" in render_case(c)
    assert "49.6 m from pad edge" in render_case(_case(obstacles=[_ob("tree", 8, "measured", 49.64)]))   # one decimal


ROTORS = tuple(a["rotor"] for a in AIRCRAFT.values())
HEIGHTS = ROTORS + tuple(x for _, lo, hi, _ in RINGS for x in (lo, hi))
DISTS = tuple(r for *_, r in RINGS) + (300 + PAD_HALF, 500, 1000 - PAD_HALF, 1000)
NEAR = (-0.06, -0.05, -0.04, -0.01, -0.001, 0, 0.001, 0.01, 0.04, 0.05, 0.06)
SRC = ("measured", "estimated", "unknown")


def _num_after(text: str, before: str, after: str = " m") -> float:
    return float(re.search(re.escape(before) + r"(-?[\d.]+)" + re.escape(after), text).group(1))


def test_heights_near_every_limit_keep_labels():                  # I8: 1.04 m in the pad fails, 75.96 m is not ultra
    for b in HEIGHTS:
        for dx in NEAR:
            h = round(b + dx, 3)
            for src in SRC:
                for ac in AIRCRAFT:
                    o_pad, o_ring = _ob("fence", h, src, 0, in_pad=True), _ob("mast", h, src, 5, center=500)
                    for o in (o_pad, o_ring):
                        c = _case(aircraft=ac, obstacles=[o])
                        v = visible(c)
                        assert judge(v) == judge(c), (h, src, ac, o.in_pad)
                        assert v.obstacles[0].height_m == _num_after(render_case(c), "height " if src == "measured" else
                                                                     "treated as " if src == "unknown" else "est. ")
                    cn = {"pad_max_m": h, "max_10_m": h, "max_50_m": h}   # no flag: the rule falls back to pad_max_m
                    c = _case(aircraft=ac, canopy=cn)
                    assert judge(visible(c)) == judge(c), (h, ac)
                    assert visible(c).canopy["pad_max_m"] == _num_after(render_case(c), "pad max ")


def test_distances_near_every_limit_keep_labels():
    hs = (1.5, 6, 30, 80)                                         # one height per ring, and an ultra-tall one
    for b in DISTS:
        for dx in NEAR:
            d = round(b + dx, 3)
            for h in hs:
                for center in (None, d):
                    c = _case(obstacles=[_ob("wire", h, "measured", d, center=center)])
                    v = visible(c)
                    assert judge(v) == judge(c), (d, h, center)
                    t = render_case(c)
                    assert v.obstacles[0].dist_edge_m == _num_after(t, "| ", " m from pad edge")
                    if h >= 76:
                        assert v.obstacles[0].dist_center_m == _num_after(t, "| ", " m from touchdown")
            for kind in ("minefield", "hv_corridor", "firing_zone"):
                c = _case(zones=[Zone(kind, d, False, 0)])
                v = visible(c)
                assert judge(v) == judge(c), (kind, d)
                assert v.zones[0].dist_center_m == _num_after(render_case(c), "within ", " m of touchdown")


def test_slope_near_every_limit_keeps_label():
    for ac, a in AIRCRAFT.items():
        for unc in (0, 2.5, 6, 6.04):
            for b in (a["max_slope"] - unc, a["max_slope"] + unc):
                for dx in NEAR:
                    c = _case(aircraft=ac, slope_deg=round(b + dx, 3), slope_unc_deg=unc)
                    v = visible(c)
                    assert judge(v) == judge(c), (ac, unc, c.slope_deg)
                    t = render_case(c)
                    assert (v.slope_deg, v.slope_unc_deg) == (_num_after(t, "Slope in pad: ", "°"), _num_after(t, "±", "°"))


def test_visible_is_idempotent():
    for c in (_worst(), _case(obstacles=[_ob("tree", 8, "measured", 49.96)], slope_deg=4.04)):
        assert render_case(visible(c)) == render_case(c)


def test_zones_capped_deduped_deciding_first():                  # I7: 30 zones must not push obstacles out
    far = [Zone(k, 1000, False, None) for k in ("firing_zone", "military_area", "nature_reserve") for _ in range(5)]
    hv_far = [Zone("hv_corridor", 400 + i, False, 90) for i in range(10)]
    deciding = [Zone("minefield", 499.96, False, 180), Zone("hv_corridor", 317.46, False, 270), Zone("minefield", 900, False, 0)]
    c = _case(zones=hv_far + far + deciding, obstacles=[_ob("power_line", 30, "unknown", 250)])
    assert len(c.zones) == 28 and ZONE_CAP >= 3
    v = visible(c)
    assert judge(v) == judge(c) and judge(c)["minefield_500"] == "fails" and judge(c)["zones"] == "coordination"
    assert len(v.zones) == ZONE_CAP and set(deciding[:2]) <= set(v.zones)
    t = render_case(c)
    zl = [ln for ln in t.split("\n") if ln.startswith("Zone:")]
    assert len(zl) == len(set(zl)) == ZONE_CAP and "250.0 m from pad edge" in t
    assert sum("direction unknown" in ln for ln in zl) <= 3    # 15 far zones are 3 distinct lines


def test_layout_fixed_lines_then_zones_then_obstacles():          # I5: right-truncation only cuts far obstacles
    lines = render_case(_worst()).split("\n")
    zone = min(i for i, ln in enumerate(lines) if ln.startswith("Zone:"))
    obs = lines.index(next(ln for ln in lines if ln.startswith("Obstacles")))
    for fixed in ("Aircraft:", "Touchdown:", "Slope", "Tree canopy:", "Dust:", "Belly area:", "Zones layer (GovMap):",
                  "Planned HV corridors layer (TAMA 10): not available", "Missing core data:", "Not available from any"):
        assert max(i for i, ln in enumerate(lines) if ln.startswith(fixed)) < zone, fixed
    assert zone < obs and all("|" in ln for ln in lines[obs + 1:])


def test_corridors_line_only_when_missing():
    assert "Planned HV corridors layer (TAMA 10): not available" in render_case(_case(corridors_ok=False))
    assert "TAMA 10)" not in render_case(_case())


def test_no_site_line_and_no_hebrew():                            # ARCH-1
    o = _ob("building", 6, "estimated", 30, bearing=200)
    t = render_case(_case(name="\u05e9\u05dd", obstacles=[o], zones=[Zone("firing_zone", 400, False, None)]))
    assert not re.search(r"[\u0590-\u05ff]", t) and "Site:" not in t and "site_x" not in render_case(_case())


def test_zone_touching_pad_wording():                             # M10
    assert "firing zone | pad touches or is inside it" in render_case(_case(zones=[Zone("firing_zone", 0, True, None)]))


def test_canopy_flag_is_rendered():                               # I2: "1.0 m" must not read as within the limit
    t = render_case(_case(canopy={"pad_max_m": 1.04, "max_10_m": 2, "max_50_m": 3, "pad_over_rotor": True}))
    assert "pad max 1.04 m (above the rotor limit)" in t


def test_ultra_tall_shows_touchdown_distance():                   # M4: the rule uses the centre distance
    t = render_case(_case(obstacles=[_ob("mast", 90, "measured", 990, center=1010)]))
    assert "1010.0 m from touchdown" in t


def test_render_missing_core():
    t = render_case(_case(slope_deg=None, core_missing=("osm", "dem")))
    assert "Missing core data: OSM, DEM" in t and "Slope: not available" in t


def test_select_keeps_relevant_beyond_nearest():                  # I3
    near = [_ob("building", 3, "estimated", 20 + i) for i in range(30)]   # 1.2-4.5 m outside 10 m: changes nothing
    far = _ob("power_line", 24, "unknown", 200)
    c = _case(obstacles=near + [far])
    shown = select_obstacles(c, 25)
    assert len(shown) == 25 and far in shown and shown[0] == far
    assert shown[1:] == near[:24]
    assert judge(visible(c)) == judge(c) and len(visible(c).obstacles) == min(DEFAULT_CAP, 31)


def test_select_relevance_rules():
    rel = [_ob("fence", 0.5, d=0, in_pad=True), _ob("tree", 3, d=9.9), _ob("wire", 0.5, d=49),
           _ob("mast", 90, d=950, center=990), _ob("mast", 90, d=980)]   # 980 + 17.5 <= 1000
    irr = [_ob("tree", 3, d=10), _ob("mast", 90, d=990, center=1010), _ob("tree", 30, d=300)]
    shown = select_obstacles(_case(obstacles=irr + rel), len(rel))
    assert set(shown) == set(rel)


def test_select_keeps_each_items_worst_obstacle():               # with cap >= 6, statuses survive any number of hits
    near = [_ob("building", 6, "estimated", 20 + i) for i in range(20)]          # low_50 probably fails, nearest
    certain = _ob("building", 9, "levels", 49)                                   # low_50 fails, farthest in the ring
    pad = [_ob("fence", 0.5, d=0, in_pad=True)] * 5 + [_ob("fence", 1.5, d=0, in_pad=True)]
    others = [_ob("wire", 6, d=45), _ob("mast", 90, d=900), _ob("tree", 3, d=5), _ob("power_line", 24, "unknown", 250)]
    c = _case(obstacles=near + [certain] + pad + others)
    assert judge(_case(obstacles=select_obstacles(c, 6))) == judge(c)
    assert judge(c)["low_50"] == "fails" and judge(c)["pad"] == "fails"


def test_render_caps_obstacles():
    t = render_case(_worst())
    assert t.count("m from pad edge") == DEFAULT_CAP == len(visible(_worst()).obstacles)


def _longest_prompt() -> str:
    from helipad.rows import case_rows
    return max((r["state"] + "\n" + r["query"] for r in case_rows(_worst())), key=len)


# Measured with the Qwen3.5-4B tokenizer: _worst's longest prompt is 961 tokens for 2,664 chars (2.77 chars/token; long
# synthetic prompts run 2.75, digits cost a token each). 2,700 chars at that ratio is ~975 tokens, so this guard trips
# on any longer worst case (ZONE_CAP 5 or DEFAULT_CAP 10 both exceed it) even where the tokenizer is not cached.
MAX_CHARS = 2700


def test_char_budget():                                           # I7: never skips
    assert len(_longest_prompt()) < MAX_CHARS, len(_longest_prompt())


def test_token_budget():                                          # I5: state + longest query < 1000 Qwen tokens
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-4B tokenizer not cached")
    from helipad.rows import case_rows
    n = max(len(tok(r["state"] + "\n" + r["query"])["input_ids"]) for r in case_rows(_worst()))
    assert n < 1000, n
