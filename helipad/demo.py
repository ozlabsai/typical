"""Recorded demo results in the format of helipad-demo/DATA.md: one site's case, the model's per-option probabilities,
and the evidence. Pure functions; scripts/helipad_demo_bake.py does the I/O."""
from __future__ import annotations

import math

from helipad.case import ITEMS, PAD_HALF, SiteCase, visible
from helipad.rules import derive_verdict, triggers

FAILING = ("fails", "probably_fails")


def _ids(probs: dict[str, float]) -> dict[str, float]:
    """Model labels ("probably fails") -> contract ids ("probably_fails")."""
    return {k.replace(" ", "_"): round(float(v), 4) for k, v in probs.items()}


def xy(dist_center_m: float, bearing: int | None, half_width_m: float) -> list[float] | None:
    """Position on a north-up satellite crop centred on touchdown, as 0-1 fractions (x right, y down).
    None when the direction is unknown (a video sighting) or the point is off the image."""
    if bearing is None or dist_center_m > half_width_m:
        return None
    b = math.radians(bearing)
    scale = 0.5 / half_width_m
    return [round(0.5 + dist_center_m * math.sin(b) * scale, 4), round(0.5 - dist_center_m * math.cos(b) * scale, 4)]


def conditions(items: dict[str, str]) -> list[str]:
    out = [k for k, item in (("minefield_check", "minefield_500"), ("low_wire_walk", "low_wires")) if items[item] == "not_checked"]
    return out + ["approach_heading", "notam"]   # always, as in the existing pipeline's report


def prep(items: dict[str, str]) -> list[str]:
    return [k for k, item in (("wetting", "dust"), ("clear_belly_area", "belly")) if items[item] == "prep"]


def verdict_probs(item_probs: dict[str, dict[str, float]], core_missing: bool) -> dict[str, float]:
    """The verdict rule (derive_verdict) applied to item probabilities, items taken as independent:
    fails if any item fails; else not checked if core data is missing; else probably fails if any item probably fails;
    else approved. The verdict's bar then agrees with how the verdict is decided."""
    no_fail = math.prod(1.0 - p.get("fails", 0.0) for p in item_probs.values())
    clear = math.prod(1.0 - p.get("fails", 0.0) - p.get("probably_fails", 0.0) for p in item_probs.values())
    if core_missing:
        return {"approved": 0.0, "probably_fails": 0.0, "fails": round(1 - no_fail, 4), "not_checked": round(no_fail, 4)}
    return {"approved": round(clear, 4), "probably_fails": round(no_fail - clear, 4), "fails": round(1 - no_fail, 4),
            "not_checked": 0.0}


def _center(o) -> float:
    return o.dist_center_m if o.dist_center_m is not None else o.dist_edge_m + PAD_HALF


def site_record(sid: str, name: dict, round_: str, expert: dict | None, case: SiteCase, item_preds: dict[str, dict],
                verdict_pred: dict, sat_image: str, half_width_m: float, frames: dict | None = None) -> dict:
    c = visible(case)
    trig = triggers(c)
    items = []
    for i in ITEMS:
        probs = _ids(item_preds[i]["probs"])
        items.append({"id": i, "status": max(probs, key=probs.get), "probs": probs,
                      "findings": trig[i] if max(probs, key=probs.get) in FAILING else []})
    statuses = {it["id"]: it["status"] for it in items}
    status = derive_verdict(statuses.values(), bool(c.core_missing))
    obstacles = [{"kind": o.kind, "label": o.label, "height_m": o.height_m, "height_src": o.height_src,
                  "dist_m": o.dist_edge_m, "bearing": o.bearing, "in_pad": o.in_pad, "source": o.source,
                  "xy": [0.5, 0.5] if o.in_pad else xy(_center(o), o.bearing, half_width_m)} for o in c.obstacles]
    rec = {
        "id": sid, "name": name, "round": round_, "aircraft": c.aircraft,
        "touchdown": {"moved": case.shift},
        "expert": expert,
        "verdict": {"status": status, "probs": verdict_probs({it["id"]: it["probs"] for it in items}, bool(c.core_missing)),
                    "model_question_probs": _ids(verdict_pred["probs"]),
                    "deciding": [it["id"] for it in items if it["status"] in FAILING]},
        "items": items, "conditions": conditions(statuses), "prep": prep(statuses),
        "findings": {
            "obstacles": obstacles,
            "zones": [{"kind": z.kind, "dist_m": z.dist_center_m, "inside": z.inside_pad, "bearing": z.bearing} for z in c.zones],
            "slope": None if c.slope_deg is None else {"deg": c.slope_deg, "unc": c.slope_unc_deg},
            "canopy": c.canopy, "dust": c.dust, "belly": c.belly,
            "layers": {"osm": "osm" not in c.core_missing, "dem": "dem" not in c.core_missing,
                       "govmap": c.govmap_ok, "corridors": c.corridors_ok},
            "counts": {"obstacles_total": len(case.obstacles),
                       "within_300": sum(o.dist_edge_m < 300 for o in case.obstacles)}},
        "evidence": {"satellite": {"image": sat_image, "half_width_m": half_width_m, "pad_xy": [0.5, 0.5]}},
    }
    if frames:
        rec["evidence"]["frames"] = frames
    return rec


def bbox(lat: float, lon: float, half_m: float) -> tuple[float, float, float, float]:
    """(lon0, lat0, lon1, lat1) of a square of side 2*half_m metres around a point (small-area approximation)."""
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def frame_option_id(label: str) -> str:
    """Vision option label -> demo option id (helipad-demo/lib/protocol.ts): "1.2–4.5 m" -> "1_2_to_4_5"."""
    from helipad.frames import NULL_OPTION
    if label == NULL_OPTION:
        return "cant_tell"
    s = label.removesuffix(" m").replace("–", "_to_").replace(".", "_").replace(" ", "_")
    return s


def pick_frames(rows: list[dict], k: int = 6) -> list[dict]:
    """Up to k keyframes spread evenly over the video, always including the frame most likely to show wires
    (the finding a viewer most needs to see). Returned in time order."""
    rows = sorted(rows, key=lambda r: r["t"])
    if len(rows) <= k:
        return rows
    step = (len(rows) - 1) / (k - 1)
    picked = {round(i * step) for i in range(k)}
    wires = max(range(len(rows)), key=lambda i: rows[i]["answers"].get("wires", {}).get("yes", 0.0))
    if wires not in picked:
        picked.discard(min(picked, key=lambda i: abs(i - wires)))
        picked.add(wires)
    return [rows[i] for i in sorted(picked)]


def frame_evidence(frames: list[tuple[str, dict]]) -> dict:
    """[(image path, {"t", "answers"})] -> DATA.md evidence.frames, every option with its probability."""
    return {"source": "model", "items": [
        {"image": img, "t": r["t"],
         "answers": {q: {frame_option_id(o): round(p, 4) for o, p in dist.items()} for q, dist in r["answers"].items()}}
        for img, r in frames]}
