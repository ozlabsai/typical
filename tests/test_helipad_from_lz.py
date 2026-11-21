import json
import re
from pathlib import Path

from helipad.case import render_case
from helipad.from_lz import case_from_lz, kind_from_tags, lz_verdict
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


def _lz(**extra):
    lz = json.loads(json.dumps(LZ))
    lz.update(extra)
    return lz


def test_kind_follows_est_height_branch_order():
    k = lambda tags: kind_from_tags(tags, "way/1")[0]  # noqa: E731
    assert k({"building": "yes", "power": "substation"}) == "substation"
    assert k({"building": "yes", "power": "line"}) == "power_line"
    assert k({"building": "yes", "aerialway": "cable_car"}) == "aerialway"
    assert k({"building": "yes", "telecom": "line"}) == "telecom_line"
    assert k({"man_made": "tower", "natural": "tree"}) == "mast"
    assert k({"railway": "rail", "highway": "residential", "lit": "yes"}) == "railway"
    assert k({"barrier": "fence", "highway": "secondary", "lit": "yes"}) == "fence"
    assert [k({"power": "cable"}), k({"highway": "primary"}), k({"man_made": "pipeline"}), k({"man_made": "flagpole"}),
            k({"man_made": "storage_tank"}), k({"man_made": "chimney"}), k({"man_made": "windmill"})] == \
        ["cable", "road", "pipeline", "flagpole", "tank", "chimney", "windmill"]


def test_full_obstacle_list_beats_capped_verdict_list():
    line = {**LZ["verdict"]["obstacles_sorted"][2], "dist_pad_edge_m": 250.0, "dist_center_m": 268, "osm": "iec/9"}
    lz = _lz(obstacles_all=LZ["verdict"]["obstacles_sorted"] + [line], zones_all=[])
    c = case_from_lz(lz, "x")
    assert len(c.obstacles) == 4 and c.obstacles[3].dist_center_m == 268
    assert [z.kind for z in c.zones] == ["firing_zone"]   # zones_all (empty) replaces verdict zones; GovMap still added
    assert [o.dist_center_m for o in case_from_lz(LZ, "x").obstacles] == [52, 160, 420]


def test_canopy_carries_lz_check_rotor_flag():
    lz = _lz()
    lz["verdict"]["canopy"].update(pad_max_m=1.0, rotor_obstacle_in_pad=True)
    lz["verdict"]["obstacles_sorted"] = []
    c = case_from_lz(lz, "x")
    assert c.canopy["pad_over_rotor"] and judge(c)["pad"] == "fails"
    assert not case_from_lz(LZ, "x").canopy["pad_over_rotor"]


def test_corridors_ok_from_lz_json():
    assert not case_from_lz(LZ, "x").corridors_ok   # never confirmed = not ok
    assert case_from_lz(_lz(corridors_ok=True), "x").corridors_ok


def test_govmap_hits_dedupe_on_layer_and_name_keeping_tightest():
    fire, res = {"layer": "שטחי אש - ציבורי", "name": "x"}, {"layer": "שמורות יער", "name": "x"}
    lz = _lz(govmap={"hits": {"1000": [fire, res], "500": [fire], "5": [fire]}}, zones_all=[])
    assert sorted((z.kind, z.dist_center_m, z.inside_pad) for z in case_from_lz(lz, "x").zones) == \
        [("firing_zone", 0, True), ("nature_reserve", 1000, False)]
