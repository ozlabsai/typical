"""lz_check output (the private lz_check pipeline, `$HELIPAD_PRIVATE/heli_lz`) (lz.json = verdict.json + GovMap hits + core-data flags) -> SiteCase.
Labels are built from the object's kind only, so no Hebrew OSM name or label reaches the case text."""
from __future__ import annotations

import re

from helipad.case import Obstacle, SiteCase, Zone, compass

KIND_LABEL = {"power_line": "power line", "power_tower": "high-voltage tower", "power_pole": "power pole",
              "lit_road": "lit road (row of lighting poles)", "lighting_pole": "lighting pole", "building": "building",
              "wall": "wall", "fence": "fence", "tree": "tree", "tree_row": "tree row", "woodland": "woodland",
              "orchard": "orchard", "vineyard": "vineyard (trellis wires)", "scrub": "scrub", "mast": "mast or tower",
              "crane": "crane", "wind_turbine": "wind turbine", "other": "other obstacle",
              "cable": "underground cable", "substation": "substation", "aerialway": "cable car or lift",
              "telecom_line": "telecom line", "chimney": "chimney", "tank": "water tower, silo or tank",
              "flagpole": "flagpole", "windmill": "windmill", "pipeline": "pipeline (if above ground)",
              "railway": "railway (overhead catenary)", "road": "road (traffic, signs)"}
ZONE_KIND = {"military": "military_area", "protected": "nature_reserve", "minefield/danger": "minefield", "hv_corridor": "hv_corridor"}
DUST = {"קל": "light", "בינוני": "medium", "כבד": "heavy"}
VERDICT = {"לא עומד": "fails", "כנראה לא עומד": "probably_fails", "מאושר": "approved"}
GOVMAP_DIST = {"5": 0, "500": 500, "1000": 1000}


def _kv(tags: dict) -> str | None:
    v = str(tags.get("voltage") or tags.get("VOLTAGE") or "")
    kv = [float(x) / (1000 if float(x) > 1000 else 1) for x in re.findall(r"\d+", v)]
    return f"{max(kv):.0f} kV" if kv else None


def kind_from_tags(tags: dict, osm_id: str) -> tuple[str, str]:
    """(kind, English label): the first lz_check.est_height branch that matches decides, in est_height's order."""
    p, mm, nat, lu, b, hw = (tags.get(k) for k in ("power", "man_made", "natural", "landuse", "barrier", "highway"))
    kind = next(k for k, hit in (
        ("power_line", osm_id.startswith("iec/") or p in ("line", "minor_line")), ("cable", p == "cable"),
        ("power_tower", p in ("tower", "portal")), ("power_pole", p == "pole"), ("substation", p == "substation"),
        ("wind_turbine", p == "generator" and tags.get("generator:source") == "wind"),
        ("aerialway", bool(tags.get("aerialway"))), ("telecom_line", bool(tags.get("telecom") or tags.get("communication:line"))),
        ("building", "building" in tags or "building:part" in tags),
        ("mast", mm in ("mast", "tower", "communications_tower", "antenna")), ("chimney", mm == "chimney"),
        ("tank", mm in ("water_tower", "silo", "storage_tank")), ("crane", mm == "crane"),
        ("lighting_pole", mm == "street_lamp" or hw == "street_lamp"), ("flagpole", mm == "flagpole"),
        ("windmill", mm == "windmill"), ("pipeline", mm == "pipeline"),
        ("tree", nat == "tree"), ("tree_row", nat == "tree_row"), ("woodland", nat == "wood" or lu == "forest"),
        ("orchard", lu == "orchard"), ("vineyard", lu == "vineyard"), ("scrub", nat == "scrub"),
        ("fence", b in ("fence", "hedge", "guard_rail")), ("wall", b in ("wall", "city_wall", "retaining_wall")),
        ("railway", tags.get("railway") in ("rail", "light_rail", "tram")),
        ("lit_road", hw in ("motorway", "trunk", "motorway_link", "trunk_link") or bool(hw and tags.get("lit") == "yes")),
        ("road", hw in ("primary", "secondary")), ("other", True)) if hit)
    label = KIND_LABEL[kind]
    if kind == "power_line":
        kv = _kv(tags)
        label = f"{'high-voltage line' if osm_id.startswith('iec/') else 'power line'} {kv}" if kv else "power line (voltage unknown)"
    return kind, label


def _obstacle(o: dict) -> Obstacle:
    kind, label = kind_from_tags(o.get("tags") or {}, str(o.get("osm", "")))
    return Obstacle(kind, label, float(o["height_m"]), o["height_src"], float(o["dist_pad_edge_m"]), bool(o["in_pad"]),
                    o.get("near_bearing"), "map", float(o["dist_center_m"]))


def case_from_lz(lz: dict, name: str) -> SiteCase:
    """Uses lz.json's uncapped obstacles_all / zones_all (scripts/helipad_lz_full.py) when present; verdict.json keeps
    only the nearest 60 obstacles."""
    v = lz["verdict"]
    zones = [Zone(ZONE_KIND[z["kind"]], float(z["dist_center_m"]), bool(z["inside_pad"]), z.get("near_bearing"))
             for z in lz.get("zones_all", v.get("zones")) or [] if z["kind"] in ZONE_KIND]
    hits = (lz.get("govmap") or {}).get("hits") or {}
    seen = set()
    for tol, dist in GOVMAP_DIST.items():   # tightest tolerance first; lz_check dedupes on (layer, name)
        for h in hits.get(tol) or []:
            if (h.get("layer"), h.get("name")) in seen:
                continue
            seen.add((h.get("layer"), h.get("name")))
            kind = "firing_zone" if "שטחי אש" in (h.get("layer") or "") else "nature_reserve"
            zones.append(Zone(kind, dist, tol == "5", None))
    sl, cn = v.get("slope") or {}, v.get("canopy") or {}
    slope = None
    if sl.get("available"):
        slope = sl["pad_slope_plane_deg"] if sl.get("pad_slope_plane_deg") is not None else sl["pad_slope_pixel_max_deg"]
    canopy = ({"pad_max_m": cn["pad_max_m"], "max_10_m": cn["max_within_10m_of_pad"], "max_50_m": cn["max_within_50m_of_pad"],
               "pad_over_rotor": bool(cn.get("rotor_obstacle_in_pad"))} if cn.get("available") else None)
    shift = v.get("shift")
    return SiteCase(
        name=name, aircraft=int(v.get("aircraft") or 1),
        obstacles=[_obstacle(o) for o in lz.get("obstacles_all", v.get("obstacles_sorted")) or []], zones=zones,
        slope_deg=slope, slope_unc_deg=float(sl.get("uncertainty_deg") or 6), canopy=canopy,
        dust=DUST.get((v.get("dust") or {}).get("estimated_level")), belly=None,
        govmap_ok=bool(hits),
        core_missing=tuple(m for m, ok in (("osm", lz.get("osm_ok")), ("dem", lz.get("dem_ok"))) if not ok),
        shift={"m": shift["m"], "dir": compass(shift["bearing"])} if shift else None,
        corridors_ok=bool(lz.get("corridors_ok")))   # absent = never confirmed


def lz_verdict(lz: dict) -> str:
    v = lz["verdict"]["verdict"]
    return "not_checked" if v.startswith("לא נבדק") else VERDICT[v]
