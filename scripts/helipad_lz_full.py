"""Uncapped obstacle and zone lists for a fetched site: verdict.json keeps only the nearest 60 obstacles, which drops
verdict-relevant ones on dense sites. Rebuilds the pad exactly as lz_check.evaluate did, with lz_check's own functions,
and writes obstacles_all, zones_all and corridors_ok into <site_dir>/lz.json. Needs the geo env:

uv run --no-project --with shapely --with pyproj --with rasterio --with mercantile --with requests --with numpy --with pillow \
    python scripts/helipad_lz_full.py <site_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helipad.paths import LZ_CHECK, LZ_SCRIPTS, require  # noqa: E402

require(LZ_CHECK)
sys.path.insert(0, str(LZ_SCRIPTS))
import lz_check as L  # noqa: E402
from shapely.geometry import Point  # noqa: E402

STEP = 10   # evaluate's shift-search grid, m


def _plain(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def lz_full(data: dict, v: dict) -> dict:
    """{"obstacles_all", "zones_all", "corridors_ok"} for a site's data.json and verdict.json."""
    pin = Point(*L.to_utm(data["lon"], data["lat"]))
    x, y = L.to_utm(v["center_used"][1], v["center_used"][0])
    # center_used is rounded to 1e-6 deg (~0.1 m); evaluate's centre is the pin or a STEP-grid point from it
    center = Point(pin.x + round((x - pin.x) / STEP) * STEP, pin.y + round((y - pin.y) / STEP) * STEP)
    pad = L._make_pad(center, L.PROTOCOL["pad"] / 2, v["heading"], None)
    osm = data.get("osm")
    items = L._prepare_osm(osm) if (osm and osm.get("elements")) else []
    items += L._prepare_iec(data.get("iec"))
    obstacles, zones = L._collect(items, pad, center)
    obstacles, zones = _plain(sorted(obstacles, key=lambda o: o["dist_pad_edge_m"])), _plain(zones)
    if obstacles[:60] != v["obstacles_sorted"] or zones != v["zones"]:
        raise ValueError("rebuilt pad does not reproduce verdict.json (heading or pad polygon not from evaluate's defaults?)")
    tags = {oid: t for t, _, oid, *_ in items}   # _collect keeps 6 tags (alphabetical): `voltage` falls off a tagged line
    for o in obstacles:
        o["tags"] = _plain(tags[o["osm"]])
    iec = data.get("iec") or {}
    return {"obstacles_all": obstacles, "zones_all": zones,
            "corridors_ok": bool(iec) and "corridors" not in (iec.get("error") or {})}   # e.g. {"corridors": "Token Required"}


def main(site_dir: str) -> None:
    d = Path(site_dir)
    lz = json.loads((d / "lz.json").read_text())
    lz.update(lz_full(json.loads((d / "data.json").read_text()), json.loads((d / "verdict.json").read_text())))
    (d / "lz.json").write_text(json.dumps(lz, ensure_ascii=False, indent=1))
    print(f"{d.name}: {len(lz['obstacles_all'])} obstacles, corridors_ok={lz['corridors_ok']}")


if __name__ == "__main__":
    main(sys.argv[1])
