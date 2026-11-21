"""The rules mirror against the private lz_check.evaluate() ($HELIPAD_PRIVATE/heli_lz), once lz.json carries the full obstacle list.
Geo env: uv run --with shapely --with pyproj --with rasterio --with mercantile --with requests --with pillow pytest tests/test_helipad_lz_full.py"""
import contextlib
import io
import json
import math
import random

import pytest

pytest.importorskip("shapely")
pytest.importorskip("pyproj")
pytest.importorskip("rasterio")
from helipad.paths import LZ_CHECK  # noqa: E402

if not LZ_CHECK.exists():
    pytest.skip(f"private lz_check not found at {LZ_CHECK}; set HELIPAD_PRIVATE", allow_module_level=True)
import numpy as np  # noqa: E402

from helipad.from_lz import case_from_lz, lz_verdict  # noqa: E402
from helipad.rules import judge, verdict  # noqa: E402
from scripts.helipad_lz_full import L, lz_full  # noqa: E402
from scripts.helipad_sites import lz_record, refresh  # noqa: E402

LAT, LON = 31.5, 34.8
CX, CY = L.to_utm(LON, LAT)


def ll(dx, dy):
    lon, lat = L.to_wgs(CX + dx, CY + dy)
    return {"lon": lon, "lat": lat}


def iec_line(dy, kv=400):
    p, q = ll(-800, dy), ll(800, dy)
    return {"attributes": {"VOLTAGE": kv, "FID": 1}, "geometry": {"paths": [[[p["lon"], p["lat"]], [q["lon"], q["lat"]]]]}}


def evaluate(data, d, aircraft=1):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        L.evaluate(json.loads(json.dumps(data)), d, aircraft)
    (d / "data.json").write_text(json.dumps(data))
    v = json.loads((d / "verdict.json").read_text())
    return v, lz_record(data, v)


def mirror(lz):
    c = case_from_lz(lz, "x")
    return verdict(c, judge(c))


def hidden_line_site(d, iec_error=None):
    """70 trees nearer than an IEC 400 kV line 250 m north: the line is not in verdict.json's nearest 60."""
    trees = [{"type": "node", "id": i, **ll((90 + 1.5 * i) * math.sin(i * .9), (90 + 1.5 * i) * math.cos(i * .9)),
              "tags": {"natural": "tree"}} for i in range(70)]
    iec = {"circuits": [iec_line(250)], "corridors": [], **({"error": iec_error} if iec_error else {})}
    return evaluate({"lat": LAT, "lon": LON, "osm": {"elements": trees}, "iec": iec}, d)


def test_full_list_restores_obstacle_hidden_by_the_cap(tmp_path):
    v, lz = hidden_line_site(tmp_path)
    assert len(v["obstacles_sorted"]) == 60 and lz_verdict(lz) == "fails" and mirror(lz) != "fails"
    lz.update(lz_full(json.loads((tmp_path / "data.json").read_text()), v))
    assert len(lz["obstacles_all"]) == 71 and lz["corridors_ok"] and mirror(lz) == "fails"


def test_corridors_ok_false_on_iec_error(tmp_path):
    v, _ = hidden_line_site(tmp_path, {"corridors": "Token Required"})
    assert not lz_full(json.loads((tmp_path / "data.json").read_text()), v)["corridors_ok"]
    data = {"lat": LAT, "lon": LON, "osm": {"elements": []}, "iec": None}
    v, _ = evaluate(data, tmp_path)
    assert not lz_full(data, v)["corridors_ok"]
    with pytest.raises(ValueError):   # a pad that does not reproduce verdict.json is refused
        lz_full({**data, "lat": LAT + 0.01}, hidden_line_site(tmp_path)[0])


def test_refresh_writes_full_lists_into_lz_json(tmp_path):
    d = tmp_path / "site"
    d.mkdir()
    _, lz = hidden_line_site(d)
    (d / "lz.json").write_text(json.dumps(lz))
    refresh(tmp_path)
    lz = json.loads((d / "lz.json").read_text())
    assert len(lz["obstacles_all"]) == 71 and lz["corridors_ok"] is True and mirror(lz) == "fails"


# Adapted from the review's fuzz harness (/tmp/helipad_review_fuzz/fuzz.py): random data dicts through evaluate().
TAGS = [{"power": "line"}, {"power": "minor_line", "voltage": "22000"}, {"power": "line", "voltage": "161000"}, {"power": "line", "height": "12"},
        {"power": "tower"}, {"power": "pole"}, {"power": "cable"}, {"power": "substation", "building": "yes"}, {"power": "generator", "generator:source": "wind"},
        {"building": "yes"}, {"building": "house", "building:levels": "2"}, {"building": "yes", "building:levels": "G+1"}, {"building": "yes", "height": "30"},
        {"building": "yes", "height": "80"}, {"man_made": "mast"}, {"man_made": "chimney"}, {"man_made": "crane"}, {"man_made": "storage_tank"},
        {"highway": "street_lamp"}, {"natural": "tree"}, {"natural": "tree_row"}, {"natural": "wood"}, {"landuse": "orchard"}, {"landuse": "vineyard"},
        {"natural": "scrub"}, {"barrier": "fence"}, {"barrier": "wall", "height": "5"}, {"highway": "primary"}, {"highway": "residential", "lit": "yes"},
        {"railway": "rail"}, {"hazard": "minefield"}, {"military": "danger_area"}, {"boundary": "protected_area"}, {"aeroway": "helipad"},
        {"aeroway": "hangar", "building": "yes"}, {"aerialway": "cable_car"}, {"telecom": "line"}, {"man_made": "pipeline"}, {"landuse": "farmland"}]
FIRE = {"layer": "שטחי אש - ציבורי", "name": "x"}


def element(rng, i):
    tags = dict(rng.choice(TAGS))
    r = rng.choice((0, 5, 15, 26, 28, 40, 60, 66, 68, 80, 150, 310, 318, 330, 500, 900, 1010, 1040)) * rng.uniform(0.9, 1.1)
    a = rng.uniform(0, 2 * math.pi)
    x, y = r * math.sin(a), r * math.cos(a)
    t = rng.random()
    if t < .4:
        return {"type": "node", "id": i, **ll(x, y), "tags": tags}
    if t < .7:
        b, s = rng.uniform(0, math.pi), rng.uniform(5, 400)
        return {"type": "way", "id": i, "tags": tags, "geometry": [ll(x - s * math.sin(b), y - s * math.cos(b)), ll(x + s * math.sin(b), y + s * math.cos(b))]}
    s = rng.uniform(3, 200)
    return {"type": "way", "id": i, "tags": tags, "geometry": [ll(x - s, y - s), ll(x + s, y - s), ll(x + s, y + s), ll(x - s, y + s), ll(x - s, y - s)]}


def random_site(rng, d):
    n = rng.choice((0, 1, 2, 3, 5, 8, 15, 70))
    els = [element(rng, i) for i in range(n)] if n != 70 else \
        [{"type": "node", "id": 1000 + i, **ll(*(lambda r, a: (r * math.sin(a), r * math.cos(a)))(rng.uniform(90, 280), rng.uniform(0, 6.28))),
          "tags": {"natural": "tree"}} for i in range(70)] + [element(rng, i) for i in range(2)]
    osm = rng.choice(["ok"] * 8 + ["none", "empty"])
    data = {"lat": LAT, "lon": LON, "osm": None if osm == "none" else {"elements": [] if osm == "empty" else els}}
    if rng.random() < .85:
        yy, xx = np.mgrid[0:41, 0:41]
        np.save(d / "dem.npy", (xx * 30 * rng.uniform(0, 0.6)).astype(float))
        data["dem"] = {"file": "dem.npy", "res": [30, 30]}
    if rng.random() < .8:
        chm = np.zeros((801, 801))
        if rng.random() < .3:
            chm[400 + rng.randint(-30, 30), 400 + rng.randint(-30, 30)] = rng.choice((0.9, 1.0, 1.04, 1.06, 1.5, 2.0, 2.04, 5))
        np.save(d / "chm.npy", chm)
        data["canopy"] = {"file": "chm.npy", "res": [1, 1]}
    gm = rng.random()
    data["govmap"] = (None if gm < .1 else {"hits": {}, "error": "HTTP 403"} if gm < .3 else {"hits": {"5": [], "500": [], "1000": []}} if gm < .6 else
                      {"hits": {"5": [FIRE] if rng.random() < .5 else [], "500": [FIRE], "1000": [FIRE, {"layer": None, "name": None}]}})
    iec = {"circuits": [iec_line(rng.choice((100, 290, 310, 330, 600)), rng.choice((161, 400, None)))] if rng.random() < .25 else [], "corridors": []}
    if rng.random() < .2:
        r = rng.choice((200, 310, 318, 325, 700))
        iec["corridors"].append({"attributes": {"FID": 2}, "geometry": {"rings": [[[p["lon"], p["lat"]] for p in
                                                                                    (ll(-500, r), ll(500, r), ll(500, r + 100), ll(-500, r + 100), ll(-500, r))]]}})
    if rng.random() < .15:
        iec["error"] = {"corridors": "Token Required"}
    data["iec"] = iec if rng.random() < .9 else None
    return data


def test_rules_mirror_lz_check_on_random_sites(tmp_path):
    rng, bad, shifts = random.Random(7), [], 0
    for k in range(150):
        d = tmp_path / str(k)
        d.mkdir()
        data = random_site(rng, d)
        v, lz = evaluate(data, d, rng.choice((1, 1, 2)))
        lz.update(lz_full(data, v))   # also asserts the rebuilt pad reproduces verdict.json's nearest 60
        shifts += bool(v["shift"])
        if mirror(lz) != lz_verdict(lz):
            bad.append((k, lz_verdict(lz), mirror(lz)))
    assert not bad and shifts > 5
