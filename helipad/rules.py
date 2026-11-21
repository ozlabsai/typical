"""The protocol plus the calibrated rules, per item, on a findings case. This mirrors
lz_check.py `evaluate()` in the private lz_check pipeline (`$HELIPAD_PRIVATE/heli_lz`): it labels the synthetic cases,
and it must reproduce that script's verdict on every real site (scripts/helipad_sites.py check)."""
from __future__ import annotations

from collections.abc import Iterable

from helipad.case import CERTAIN_SRC, ITEMS, PAD_HALF, RINGS, SiteCase, center_dist

AIRCRAFT = {1: {"rotor": 1.0, "max_slope": 15}, 2: {"rotor": 2.0, "max_slope": 10}}
SEVERITY = ("passes", "not_checked", "prep", "coordination", "probably_fails", "fails")   # worst last
COORD_ZONES = {"firing_zone", "military_area", "nature_reserve"}


def _worst(*statuses: str) -> str:
    return max(statuses, key=SEVERITY.index)


def judge(c: SiteCase) -> dict[str, str]:
    ac = AIRCRAFT[c.aircraft]
    osm_missing = "osm" in c.core_missing
    map_default = "not_checked" if osm_missing else "passes"
    out = {}

    pad = [o for o in c.obstacles if o.in_pad and o.height_m > ac["rotor"]]
    cn = c.canopy or {}
    # lz_check's own flag is set from the unrounded raster maximum; the rounded pad_max_m is only a fallback
    canopy_pad = bool(cn) and (cn["pad_over_rotor"] if "pad_over_rotor" in cn else cn["pad_max_m"] > ac["rotor"])
    out["pad"] = "fails" if pad or canopy_pad else map_default

    for item, lo, hi, ring in RINGS:
        hits = ["fails" if o.height_src in CERTAIN_SRC else "probably_fails"
                for o in c.obstacles if not o.in_pad and lo <= o.height_m < hi and o.dist_edge_m < ring]
        if item == "high_300":
            hits += ["probably_fails" for z in c.zones if z.kind == "hv_corridor" and z.dist_center_m - PAD_HALF < 300]
        # ponytail: canopy rings only WARN in lz_check, so they never change these items; revisit with the expert
        out[item] = _worst(*hits) if hits else map_default

    ultra = any(o.height_m >= 76 and center_dist(o) <= 1000 for o in c.obstacles)
    out["ultra_1km"] = "coordination" if ultra else map_default

    if c.slope_deg is None or c.slope_deg + c.slope_unc_deg > ac["max_slope"]:
        out["slope"] = "fails" if c.slope_deg is not None and c.slope_deg - c.slope_unc_deg > ac["max_slope"] else "not_checked"
    else:
        out["slope"] = "passes"

    coord = any(z.kind in COORD_ZONES or (z.kind == "minefield" and z.dist_center_m > 500) for z in c.zones)
    out["zones"] = "coordination" if coord else ("passes" if c.govmap_ok else "not_checked")
    out["minefield_500"] = "fails" if any(z.kind == "minefield" and z.dist_center_m <= 500 for z in c.zones) else "not_checked"

    out["belly"] = {"clear": "passes", "obstacles": "prep"}.get(c.belly, "not_checked")
    out["dust"] = {"light": "passes", "medium": "prep", "heavy": "prep"}.get(c.dust, "not_checked")

    wires = [o for o in c.obstacles if o.kind == "wire"]
    out["low_wires"] = ("fails" if any(o.in_pad for o in wires) else
                        "probably_fails" if any(o.dist_edge_m < 50 for o in wires) else "not_checked")
    return {k: out[k] for k in ITEMS}


def derive_verdict(statuses: Iterable[str], core_missing: bool) -> str:
    s = set(statuses)
    if "fails" in s:
        return "fails"
    if core_missing:
        return "not_checked"
    return "probably_fails" if "probably_fails" in s else "approved"


def verdict(c: SiteCase, items: dict[str, str]) -> str:
    return derive_verdict(items.values(), bool(c.core_missing))
