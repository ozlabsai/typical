"""A landing-zone site as English findings text: the state a helipad decision Typical reads.

Vocabulary (items, statuses, verdicts) follows the shortened site-inspection protocol and the calibrated
rules in the private lz_check pipeline (`$HELIPAD_PRIVATE/heli_lz`; see docs/superpowers/specs/2026-11-21-helipad-lz-decision-model-design.md §4).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import zip_longest

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
COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
RINGS = (("very_low_10", 1.2, 4.5, 10), ("low_50", 4.5, 24, 50), ("high_300", 24, 76, 300))
RING_ITEMS = frozenset(k for k, *_ in RINGS)
PAD_HALF = 17.5
DEFAULT_CAP = 9    # with ZONE_CAP, the worst case (tests/test_helipad_case.py::_worst) is 961 Qwen3.5 tokens with the
ZONE_CAP = 4       # longest query (< 1,000); >= 3 keeps one deciding zone per item (minefield_500, high_300, zones)
DISTS = tuple(r for *_, r in RINGS) + (300 + PAD_HALF, 500, 1000 - PAD_HALF, 1000)   # rings, HV corridor, minefield, ultra
ZONE_NAMES = {"firing_zone": "firing zone", "military_area": "military area", "nature_reserve": "nature reserve or park",
              "minefield": "minefield", "hv_corridor": "planned high-voltage corridor (TAMA 10)"}


def label(status: str) -> str:
    return status.replace("_", " ")


def compass(deg: float) -> str:
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


@dataclass(frozen=True)
class Obstacle:
    kind: str
    label: str
    height_m: float
    height_src: str      # tagged | levels | iec | measured | estimated | unknown
    dist_edge_m: float   # from the 35x35 pad edge; 0 when in_pad
    in_pad: bool
    bearing: int | None  # compass deg to the nearest point; None = unknown
    source: str          # map | video | map+video
    dist_center_m: float | None = None   # from the touchdown point; None = pad edge + PAD_HALF


@dataclass(frozen=True)
class Zone:
    kind: str            # firing_zone | military_area | nature_reserve | minefield | hv_corridor
    dist_center_m: float
    inside_pad: bool
    bearing: int | None


@dataclass
class SiteCase:
    name: str
    aircraft: int
    obstacles: list[Obstacle]
    zones: list[Zone]
    slope_deg: float | None
    slope_unc_deg: float
    canopy: dict | None          # {"pad_max_m", "max_10_m", "max_50_m"[, "pad_over_rotor"]}
    dust: str | None             # light | medium | heavy
    belly: str | None            # clear | obstacles
    govmap_ok: bool
    core_missing: tuple[str, ...]
    shift: dict | None = None    # {"m": int, "dir": str}
    corridors_ok: bool = True    # planned HV corridors layer (TAMA 10) fetched


def center_dist(o: Obstacle) -> float:
    return o.dist_edge_m + PAD_HALF if o.dist_center_m is None else o.dist_center_m


def _items_hit(o: Obstacle) -> list[str]:
    """The items whose status this obstacle can change."""
    hit = ["pad"] if o.in_pad else []
    if o.kind == "wire" and o.dist_edge_m < 50:
        hit.append("low_wires")
    if o.height_m >= 76 and center_dist(o) <= 1000:
        hit.append("ultra_1km")
    return hit + [k for k, lo, hi, ring in RINGS if not o.in_pad and lo <= o.height_m < hi and o.dist_edge_m < ring]


def _worst_first(item: str, o: Obstacle):
    if item == "pad":
        return -o.height_m
    return o.height_src not in CERTAIN_SRC if item in RING_ITEMS else 0


def select_obstacles(c: SiteCase, cap: int) -> list[Obstacle]:
    """Every obstacle that can change a status, then the nearest others, up to `cap`. Relevant ones go round-robin over
    the items they hit, each item's worst first (tallest in the pad, certain heights in a ring), so with cap >= 6 the
    statuses of the shown obstacles equal those of all of them."""
    near = sorted(c.obstacles, key=lambda o: (not o.in_pad, o.dist_edge_m))
    queues: dict[str, list[Obstacle]] = {}
    for o in near:
        for k in _items_hit(o):
            queues.setdefault(k, []).append(o)
    for k, q in queues.items():
        q.sort(key=lambda o: _worst_first(k, o))   # stable: nearest first within a tie
    picked, seen = [], set()
    for o in [o for row in zip_longest(*queues.values()) for o in row if o is not None] + near:
        if id(o) not in seen:
            seen.add(id(o))
            picked.append(o)
    return picked[:cap]


def _zone_item(z: Zone) -> str | None:
    """The item this zone decides, as helipad.rules.judge reads it."""
    from helipad.rules import COORD_ZONES   # ponytail: rules imports this module
    if z.kind == "minefield" and z.dist_center_m <= 500:
        return "minefield_500"
    if z.kind == "hv_corridor" and z.dist_center_m - PAD_HALF < 300:
        return "high_300"
    return "zones" if z.kind in COORD_ZONES or z.kind == "minefield" else None


def select_zones(zones: list[Zone], cap: int) -> list[Zone]:
    """Distinct zone lines, the nearest zone that decides each item first, then the nearest others, up to `cap`; with
    cap >= 3 the statuses of the shown zones equal those of all of them."""
    lines: dict[str, Zone] = {}
    for z in sorted(zones, key=lambda z: z.dist_center_m):
        lines.setdefault(_zone_line(z), z)
    first: dict[str, Zone] = {}
    for z in lines.values():
        if (k := _zone_item(z)) and k not in first:
            first[k] = z
    deciding = list(first.values())
    return (deciding + [z for z in lines.values() if z not in deciding])[:cap]


def _side(x: float, limit: float) -> int:
    return (x > limit) - (x < limit)


def _fewest(sides, *xs: float) -> tuple[float, ...]:
    """xs at one decimal, or at the fewest more that leave sides(*xs) unchanged: rounding never moves a value onto or
    across a protocol limit (1.04 m in the pad stays 1.04, not 1.0; 75.96 m stays below 76)."""
    want = sides(*xs)
    for d in range(1, 10):
        r = tuple(round(x, d) for x in xs)
        if sides(*r) == want:
            return r
    return xs


def _safe(x: float | None, limits) -> float | None:
    return None if x is None else _fewest(lambda v: [_side(v, b) for b in limits], x)[0]


def visible(c: SiteCase, cap: int = DEFAULT_CAP, zone_cap: int = ZONE_CAP) -> SiteCase:
    """The case as rendered: the selected obstacles and zones, every number at its printed precision, so labels match
    the text. A number gets more than one decimal only where one would put it on or across a limit."""
    from helipad.rules import AIRCRAFT
    heights = tuple(a["rotor"] for a in AIRCRAFT.values()) + tuple(x for _, lo, hi, _ in RINGS for x in (lo, hi))
    obs = [replace(o, height_m=_safe(o.height_m, heights), dist_edge_m=_safe(o.dist_edge_m, DISTS),
                   dist_center_m=_safe(center_dist(o), DISTS)) for o in select_obstacles(c, cap)]
    zones = select_zones([replace(z, dist_center_m=_safe(z.dist_center_m, DISTS)) for z in c.zones], zone_cap)
    canopy = c.canopy and {k: v if isinstance(v, bool) else _safe(v, heights) for k, v in c.canopy.items()}
    m = AIRCRAFT[c.aircraft]["max_slope"]
    slope, unc = ((None, round(c.slope_unc_deg, 1)) if c.slope_deg is None else
                  _fewest(lambda s, u: (_side(s + u, m), _side(s - u, m)), c.slope_deg, c.slope_unc_deg))
    return replace(c, obstacles=obs, zones=zones, slope_deg=slope, slope_unc_deg=unc, canopy=canopy)


def _num(x: float) -> str:
    """A visible() number exactly: one decimal, or the shortest repr when visible() kept more."""
    s = f"{x:.1f}"
    return s if float(s) == x else repr(x)


def _height(o: Obstacle) -> str:
    if o.height_src in CERTAIN_SRC:
        return f"height {_num(o.height_m)} m ({o.height_src})"
    if o.height_src == "unknown":
        return f"height unknown, treated as {_num(o.height_m)} m"
    return f"height est. {_num(o.height_m)} m"


def _where(bearing: int | None) -> str:
    return "direction unknown" if bearing is None else f"{compass(bearing)} {bearing}°"


def _obstacle_line(o: Obstacle) -> str:
    dist = ["inside the pad" if o.in_pad else f"{_num(o.dist_edge_m)} m from pad edge"]
    if o.height_m >= 76:
        dist.append(f"{_num(center_dist(o))} m from touchdown")
    return " | ".join([o.label, _height(o), *dist, _where(o.bearing), o.source])


def _zone_line(z: Zone) -> str:
    where = "pad touches or is inside it" if z.inside_pad else f"within {_num(z.dist_center_m)} m of touchdown | {_where(z.bearing)}"
    return f"Zone: {ZONE_NAMES.get(z.kind, z.kind)} | {where}"


def render_case(c: SiteCase, cap: int = DEFAULT_CAP) -> str:
    """Fixed lines, then zones, then obstacles (relevant first), so right-truncation can only cut far obstacles."""
    c = visible(c, cap)
    lines = [f"Aircraft: {c.aircraft}",
             f"Touchdown: moved {c.shift['m']} m {c.shift['dir']} of the pin" if c.shift else "Touchdown: pin",
             "Slope: not available" if c.slope_deg is None else f"Slope in pad: {_num(c.slope_deg)}° (±{_num(c.slope_unc_deg)}°)"]
    cn = c.canopy
    lines.append("Tree canopy: not available" if not cn else
                 f"Tree canopy: pad max {_num(cn['pad_max_m'])} m{' (above the rotor limit)' if cn.get('pad_over_rotor') else ''}; "
                 f"within 10 m {_num(cn['max_10_m'])} m; within 50 m {_num(cn['max_50_m'])} m (2009–2020 layer)")
    lines.append(f"Dust: {c.dust}" if c.dust else "Dust: not assessed")
    lines.append({"clear": "Belly area: clear", "obstacles": "Belly area: has obstacles"}.get(c.belly, "Belly area: not assessed"))
    lines.append(f"Zones layer (GovMap): {'available' if c.govmap_ok else 'not available'}")
    if not c.corridors_ok:
        lines.append("Planned HV corridors layer (TAMA 10): not available")
    if c.core_missing:
        lines.append("Missing core data: " + ", ".join(m.upper() for m in c.core_missing))
    lines.append("Not available from any map layer: low wires, minefields.")
    lines += [_zone_line(z) for z in c.zones]
    if c.obstacles:
        lines.append("Obstacles (what | height | distance | direction | source):")
        lines += [_obstacle_line(o) for o in c.obstacles]
    return "\n".join(lines)
