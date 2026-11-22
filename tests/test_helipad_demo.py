import math

from helipad.case import ITEMS, Obstacle
from helipad.demo import conditions, prep, site_record, xy
from tests.test_helipad_case import _case

H = ["passes", "fails", "probably fails", "not checked"]


def test_xy_places_north_up_and_drops_unknown_or_off_image():
    assert xy(0, 0, 400) == [0.5, 0.5]
    x, y = xy(200, 90, 400)            # 200 m east -> right of centre
    assert math.isclose(x, 0.75) and math.isclose(y, 0.5)
    x, y = xy(200, 0, 400)             # north -> up (smaller y)
    assert math.isclose(y, 0.25)
    assert xy(200, None, 400) is None and xy(500, 0, 400) is None


def test_conditions_and_prep_follow_item_statuses():
    items = {i: "passes" for i in ITEMS} | {"minefield_500": "not_checked", "low_wires": "not_checked", "dust": "prep"}
    assert conditions(items) == ["minefield_check", "low_wire_walk", "approach_heading", "notam"]
    assert prep(items) == ["wetting"]


def _probs(top, labels=H):
    return {"argmax": top, "probs": {l: (0.9 if l == top else 0.1 / (len(labels) - 1)) for l in labels}}


def test_site_record_uses_contract_ids_record_verdict_and_deciding_findings():
    c = _case(obstacles=[Obstacle("power_line", "power line (voltage unknown)", 24, "unknown", 140, False, 8, "map")])
    items = {i: _probs("passes") for i in ITEMS} | {"high_300": _probs("probably fails")}
    verdict = _probs("approved", ["approved", "probably fails", "fails", "not checked"])
    r = site_record("S2", {"en": "S2", "he": "S2"}, "2026-09", None, c, items, verdict, "img/S2_sat.jpg", 400)
    assert r["verdict"]["status"] == "probably_fails"            # verdict of record, from the items
    assert set(r["verdict"]["probs"]) == {"approved", "probably_fails", "fails", "not_checked"}
    assert r["verdict"]["deciding"] == ["high_300"]
    item = next(i for i in r["items"] if i["id"] == "high_300")
    assert item["status"] == "probably_fails" and item["findings"] == [0] and abs(sum(item["probs"].values()) - 1) < 1e-3   # 4-decimal display rounding
    ob = r["findings"]["obstacles"][0]
    assert ob["dist_m"] == 140.0 and ob["bearing"] == 8 and ob["xy"] is not None
    assert r["evidence"]["satellite"]["half_width_m"] == 400 and "frames" not in r["evidence"]


def test_bbox_is_square_in_metres_around_the_point():
    from helipad.demo import bbox
    lon0, lat0, lon1, lat1 = bbox(31.5, 34.6, 400)
    assert math.isclose((lat1 - lat0) * 111_320, 800, rel_tol=1e-3)
    assert math.isclose((lon1 - lon0) * 111_320 * math.cos(math.radians(31.5)), 800, rel_tol=1e-3)
    assert lat0 < 31.5 < lat1 and lon0 < 34.6 < lon1


def test_verdict_probs_follow_the_verdict_rule_over_item_probabilities():
    from helipad.demo import verdict_probs
    items = {i: {"passes": 1.0} for i in ITEMS}
    items["high_300"] = {"passes": 0.2, "probably_fails": 0.8}
    items["pad"] = {"passes": 0.9, "fails": 0.1}
    p = verdict_probs(items, core_missing=False)
    assert math.isclose(p["fails"], 0.1) and math.isclose(p["probably_fails"], 0.9 * 0.8)
    assert math.isclose(p["approved"], 0.9 * 0.2) and p["not_checked"] == 0.0
    assert math.isclose(sum(p.values()), 1.0)
    q = verdict_probs(items, core_missing=True)          # missing core data: anything not failing is "not checked"
    assert math.isclose(q["fails"], 0.1) and math.isclose(q["not_checked"], 0.9) and q["approved"] == 0.0


UI_OPTIONS = {  # helipad-demo/lib/protocol.ts QUESTION_OPTIONS
    "tallest_height": ["below_1_2", "1_2_to_4_5", "4_5_to_24", "above_24"],
    "tallest_distance": ["below_10", "10_to_50", "50_to_300", "above_300"],
    "tallest_type": ["tree", "building", "wall_or_fence", "pole", "vehicle", "none"],
    "ground": ["paved", "hard_earth", "stony", "ploughed", "vegetation", "powdery"],
}


def test_frame_option_ids_match_the_demo_contract():
    from helipad.demo import frame_option_id
    from helipad.frames import FRAME_QUESTIONS, NULL_OPTION
    for q, ids in UI_OPTIONS.items():
        assert [frame_option_id(o) for o in FRAME_QUESTIONS[q][1]] == ids
    assert frame_option_id(NULL_OPTION) == "cant_tell" and frame_option_id("yes") == "yes"


def _frame(t, wires):
    from helipad.frames import NULL_OPTION
    return {"t": t, "answers": {"wires": {"yes": wires, "no": 1 - wires - 0.01, NULL_OPTION: 0.01}}}


def test_pick_frames_spreads_over_time_and_keeps_the_most_telling_wire_frame():
    from helipad.demo import pick_frames
    rows = [_frame(float(t), 0.9 if t == 7 else 0.1) for t in range(16)]
    picked = pick_frames(rows, k=4)
    assert len(picked) == 4 and any(r["t"] == 7.0 for r in picked)
    assert [r["t"] for r in picked] == sorted(r["t"] for r in picked)
    assert len(pick_frames(rows[:2], k=4)) == 2


def test_frame_evidence_converts_every_option_and_keeps_all_probabilities():
    from helipad.demo import frame_evidence
    ev = frame_evidence([("img/a.jpg", _frame(2.0, 0.88))])
    item = ev["items"][0]
    assert ev["source"] == "model" and item["image"] == "img/a.jpg" and item["t"] == 2.0
    assert set(item["answers"]["wires"]) == {"yes", "no", "cant_tell"}


def test_bake_finds_predictions_by_public_id_or_name():
    from scripts.helipad_demo_bake import pred_key
    s = {"public_id": "S2", "name": "x"}
    assert pred_key({"S2": {}}, s) == "S2" and pred_key({"x": {}}, s) == "x" and pred_key({}, s) is None
