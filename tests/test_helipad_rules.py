from helipad.case import ALLOWED, ITEMS, Obstacle, Zone
from helipad.rules import derive_verdict, judge, verdict
from tests.test_helipad_case import _case


def ob(kind="building", h=6.0, src="estimated", d=100.0, in_pad=False, source="map"):
    return Obstacle(kind, kind, h, src, d, in_pad, 0, source)


def test_every_item_judged_within_allowed():
    s = judge(_case())
    assert set(s) == set(ITEMS) and all(s[i] in ALLOWED[i] for i in ITEMS)


def test_unknown_power_line_in_300_is_probably_fails():          # calibrated rule 1
    c = _case(obstacles=[ob("power_line", 24, "unknown", 140)])
    s = judge(c)
    assert s["high_300"] == "probably_fails" and verdict(c, s) == "probably_fails"


def test_tagged_height_in_ring_fails():
    c = _case(obstacles=[ob("building", 9, "levels", 30)])
    s = judge(c)
    assert s["low_50"] == "fails" and verdict(c, s) == "fails"


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
    s = judge(c)
    assert (s["dust"], s["belly"]) == ("prep", "prep") and verdict(c, s) == "approved"


def test_firing_zone_is_coordination_not_failure():               # calibrated rule 7
    c = _case(zones=[Zone("firing_zone", 0, True, None)])
    s = judge(c)
    assert s["zones"] == "coordination" and verdict(c, s) == "approved"


def test_minefield_never_passes():
    assert judge(_case())["minefield_500"] == "not_checked"
    assert judge(_case(zones=[Zone("minefield", 400, False, 90)]))["minefield_500"] == "fails"


def test_hv_corridor_within_300_probably():
    assert judge(_case(zones=[Zone("hv_corridor", 250, False, 0)]))["high_300"] == "probably_fails"


def test_wire_over_pad_fails():
    assert judge(_case(obstacles=[ob("wire", 6, "estimated", 0, True, "video")]))["low_wires"] == "fails"


def test_canopy_flag_overrides_rounded_value():                   # I2
    cn = {"pad_max_m": 1.0, "max_10_m": 2, "max_50_m": 3}
    assert judge(_case(canopy=cn))["pad"] == "passes"
    assert judge(_case(canopy={**cn, "pad_over_rotor": True}))["pad"] == "fails"
    assert judge(_case(canopy={**cn, "pad_max_m": 1.5, "pad_over_rotor": False}))["pad"] == "passes"
    assert judge(_case(canopy={**cn, "pad_max_m": 1.5}))["pad"] == "fails"


def test_ultra_uses_centre_distance():                            # M4
    def mast(d, center=None):
        return Obstacle("mast", "mast", 90, "measured", d, False, 0, "map", center)
    assert judge(_case(obstacles=[mast(990, 1010)]))["ultra_1km"] == "passes"
    assert judge(_case(obstacles=[mast(970, 1000)]))["ultra_1km"] == "coordination"
    assert judge(_case(obstacles=[mast(982.5)]))["ultra_1km"] == "coordination"
    assert judge(_case(obstacles=[mast(982.6)]))["ultra_1km"] == "passes"


def test_derive_verdict():                                        # M8: the scorer imports this
    assert derive_verdict(["passes", "fails"], True) == "fails"
    assert derive_verdict(["passes", "probably_fails"], True) == "not_checked"
    assert derive_verdict(["passes", "probably_fails"], False) == "probably_fails"
    assert derive_verdict(("passes", "prep", "coordination", "not_checked"), False) == "approved"
    c = _case(obstacles=[ob("power_line", 24, "unknown", 140)])
    assert verdict(c, judge(c)) == derive_verdict(judge(c).values(), False)
