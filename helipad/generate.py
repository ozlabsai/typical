"""Synthetic landing-zone cases, labelled by helipad.rules, built verdict-first.

A scene (obstacle runs, zones, layers) is drawn the same way for every verdict, with every obstacle outside its own ring
and many just outside it. The target verdict then only moves things across a threshold: one or a few obstacles are pulled
inside a ring, the in-pad object is made taller than the rotor limit, the slope or a minefield crosses its limit. So the
number and the mix of lines say almost nothing about the verdict; the heights against the distances do.
Scene statistics follow the real sites in .context/helipads/sites (see tests/test_helipad_generate.py)."""
from __future__ import annotations

import math
import random
from dataclasses import replace

from helipad.case import CERTAIN_SRC, COMPASS, RINGS, Obstacle, SiteCase, Zone, visible
from helipad.from_lz import KIND_LABEL
from helipad.rules import AIRCRAFT, judge, verdict

VERDICT_MIX = {"approved": .35, "probably_fails": .35, "fails": .20, "not_checked": .10}
# lz_check.est_height defaults: (height m, height source) for an untagged object
DEFAULT = {"building": (6, "estimated"), "road": (.3, "estimated"), "woodland": (10, "estimated"), "tree": (8, "estimated"),
           "tree_row": (8, "estimated"), "orchard": (5, "estimated"), "vineyard": (2.5, "estimated"), "scrub": (1.5, "estimated"),
           "fence": (1.8, "estimated"), "wall": (2.5, "estimated"), "mast": (30, "estimated"), "crane": (40, "estimated"),
           "chimney": (40, "estimated"), "tank": (20, "estimated"), "lighting_pole": (8, "estimated"),
           "substation": (8, "estimated"), "telecom_line": (8, "estimated"), "flagpole": (8, "estimated"),
           "cable": (0, "estimated"), "pipeline": (.5, "estimated"), "wind_turbine": (100, "estimated"),
           "power_pole": (24, "unknown"), "power_tower": (35, "unknown"), "lit_road": (10, "unknown")}
# (label, height, source): OSM lines have unknown heights by voltage; the IEC layer is 161/400 kV only and certain
LINES = (("power line (voltage unknown)", 24, "unknown"), ("power line 22 kV", 24, "unknown"),
         ("power line 161 kV", 30, "unknown"), ("power line 400 kV", 45, "unknown"),
         ("high-voltage line 161 kV", 30, "iec"), ("high-voltage line 400 kV", 45, "iec"))
CLASS_KINDS = {"very_low_10": ("fence", "wall", "scrub", "vineyard"),
               "low_50": ("building", "building", "tree", "woodland", "orchard", "lighting_pole", "tank"),
               "high_300": ("mast", "crane", "chimney", "power_pole", "power_tower")}
SINGLES = ("fence", "wall", "vineyard", "orchard", "scrub", "mast", "crane", "chimney", "tank", "lighting_pole", "substation",
           "telecom_line", "flagpole", "cable", "pipeline", "power_tower")
NEAR_KINDS = ("road", "road", "fence", "wall", "scrub", "vineyard", "cable", "pipeline")
IN_PAD = ("road", "cable", "pipeline", "scrub", "fence", "wall", "tree")
EDGES = (1.2, 4.5, 24, 76)
NEAR = .15   # "near a threshold" = within 15% of it
FAIL_W = {"very_low_10": 15, "low_50": 25, "high_300": 22, "slope": 13, "canopy": 8, "minefield": 17}
PROB_W = {"very_low_10": 20, "low_50": 35, "high_300": 25, "corridor": 10, "wire": 15}


def _ob(kind: str, h: float | None = None, src: str | None = None, label: str | None = None, d: float = 0.0,
        bearing: int | None = None, source: str = "map", in_pad: bool = False) -> Obstacle:
    h0, s0 = DEFAULT.get(kind, (0, "estimated"))
    return Obstacle(kind, label or KIND_LABEL.get(kind, kind), round(float(h0 if h is None else h), 1), src or s0,
                    round(d, 1), in_pad, bearing, source)


def _ring(h: float) -> float:
    return next((r for _, lo, hi, r in RINGS if lo <= h < hi), 0)


def _outside(rng: random.Random, ring: float) -> float:
    return round(ring * (1 + (rng.uniform(0, NEAR) if rng.random() < .5 else rng.uniform(NEAR, 1.5))), 1)


def _inside(rng: random.Random, ring: float) -> float:
    return min(round(ring * (rng.uniform(1 - NEAR, 1) if rng.random() < .5 else rng.uniform(.05, 1 - NEAR)), 1), ring - .1)


def _varied(rng: random.Random, base: float) -> float:
    """A tagged or measured height: often within 15% of a class boundary, on either side."""
    edges = [e for e in EDGES if base / 4 <= e <= base * 4]
    if edges and rng.random() < .45:
        return round(rng.choice(edges) * rng.uniform(1 - NEAR, 1 + NEAR), 1)
    return round(base * rng.uniform(.5, 2), 1)


def _building(rng: random.Random) -> Obstacle:
    r = rng.random()
    if r < .88:
        return _ob("building")
    return _ob("building", rng.randint(1, 8) * 3 + 2, "levels") if r < .96 else _ob("building", _varied(rng, 6), "tagged")


def _run(rng: random.Random) -> tuple[list[Obstacle], float, float]:
    """One real-like group, where it starts (m from the pad edge) and the spacing between its members along the run."""
    kind = rng.choices(("buildings", "roads", "poles", "trees", "lit", "single", "near"), (50, 12, 10, 12, 6, 8, 8))[0]
    far = math.exp(rng.uniform(math.log(8), math.log(900)))
    if kind == "near":   # ground clutter by the pad: short things that are harmless a few metres further out
        return [_ob(rng.choice(NEAR_KINDS)) for _ in range(rng.randint(1, 4))], rng.uniform(1, 40), rng.uniform(2, 15)
    if kind == "buildings":
        return [_building(rng) for _ in range(rng.randint(2, 25))], far, rng.uniform(4, 25)
    if kind == "roads":
        return [_ob("road") for _ in range(rng.randint(1, 4))], far, rng.uniform(20, 150)
    if kind == "poles":   # a line, then its poles (or towers) at increasing distances, heights unknown
        label, h, src = rng.choices(LINES, (30, 30, 12, 6, 14, 8))[0]
        support = "power_tower" if h > 24 else "power_pole"
        line = [_ob("power_line", h, src, label)] if rng.random() < .6 else []
        return line + [_ob(support) for _ in range(rng.randint(3, 12))], far, rng.uniform(25, 60)
    if kind == "trees":
        k = rng.choice(("woodland", "woodland", "tree_row", "tree", "scrub"))
        return [_ob(k, _varied(rng, 8), "tagged") if k == "tree" and rng.random() < .2 else _ob(k)
                for _ in range(rng.randint(1, 6))], far, rng.uniform(10, 60)
    if kind == "lit":
        return [_ob("lit_road") for _ in range(rng.randint(1, 4))] + [_ob("road")], far, rng.uniform(30, 120)
    k = rng.choice(SINGLES)
    return [_ob(k, _varied(rng, DEFAULT[k][0]), "tagged") if k == "mast" and rng.random() < .4 else _ob(k)], far, 0


def _clear(rng: random.Random, o: Obstacle, d: float) -> Obstacle:
    """Place at about `d` m, pushed outside the obstacle's own ring."""
    d, ring = min(d, rng.uniform(900, 1100)), _ring(o.height_m)
    return replace(o, dist_edge_m=round(d, 1) if d >= ring else _outside(rng, ring))


def _map_scene(rng: random.Random) -> list[Obstacle]:
    """0-60 map obstacles in runs. Real complete fetches hold 42-60 (verdict.json keeps the nearest 60)."""
    r = rng.random()
    n = rng.randint(0, 8) if r < .10 else rng.randint(9, 35) if r < .30 else rng.randint(36, 60) if r < .70 else 60
    out: list[Obstacle] = []
    while len(out) < n:
        members, d, step = _run(rng)
        bearing = rng.randrange(360)
        for o in members:
            out.append(replace(_clear(rng, o, d), bearing=(bearing + rng.randint(-25, 25)) % 360))
            d += step * rng.uniform(.6, 1.4)
    out = out[:n]
    if out and rng.random() < .08:   # an ultra-high object around the 1 km line, or one just under 76 m
        tall = rng.choice((_ob("wind_turbine"), _ob("mast", rng.uniform(76, 150), "tagged"), _ob("mast", rng.uniform(65, 75.9), "tagged")))
        d = rng.uniform(850, 982.5) if rng.random() < .6 else rng.uniform(982.6, 1100)   # 1,000 m from touchdown = 982.5
        out[0] = replace(tall, dist_edge_m=round(max(d, _ring(tall.height_m)), 1), bearing=rng.randrange(360))
    return out


def _of_class(rng: random.Random, item: str, certain: bool, map_ok: bool) -> Obstacle:
    """A new obstacle in `item`'s height class; without the map it is a video finding."""
    lo, hi = next((lo, hi) for k, lo, hi, _ in RINGS if k == item)
    kind = rng.choice(CLASS_KINDS[item])
    seen = {} if map_ok else {"source": "video"}
    if not certain:
        return _ob(kind, bearing=rng.randrange(360) if map_ok else None, **seen)
    if map_ok and item == "high_300" and rng.random() < .3:
        label, h, src = rng.choice(LINES[4:])
        return _ob("power_line", h, src, label, bearing=rng.randrange(360))
    if map_ok and kind == "building" and rng.random() < .6:
        return _ob(kind, rng.randint(math.ceil((lo - 2) / 3), int((hi - 2.1) // 3)) * 3 + 2, "levels", bearing=rng.randrange(360))
    r = rng.random()
    h = lo * rng.uniform(1, 1 + NEAR) if r < .25 else hi * rng.uniform(1 - NEAR, 1) if r < .5 else rng.uniform(lo, hi)
    return _ob(kind, min(max(h, lo), hi - .1), "tagged" if map_ok else "measured", bearing=rng.randrange(360) if map_ok else None, **seen)


def _hit(rng: random.Random, obs: list[Obstacle], used: set[int], item: str, certain: bool, map_ok: bool) -> None:
    """Pull one obstacle of `item`'s class (a few, for a run) inside its ring; make one if the scene has none."""
    lo, hi, ring = next((lo, hi, r) for k, lo, hi, r in RINGS if k == item)
    pool = [i for i, o in enumerate(obs) if i not in used and lo <= o.height_m < hi and not o.in_pad and o.kind != "wire"
            and (o.height_src in CERTAIN_SRC) == certain]
    if not pool:
        free = [i for i in range(len(obs)) if i not in used]
        if free:
            pool = [rng.choice(free)]
        else:
            obs.append(None)
            pool = [len(obs) - 1]
        obs[pool[0]] = _of_class(rng, item, certain, map_ok)
    for i in rng.sample(pool, min(len(pool), rng.choice((1, 1, 1, 2, 3, 6)))):
        obs[i] = replace(obs[i], dist_edge_m=_inside(rng, ring))
        used.add(i)


def _in_pad(rng: random.Random, rotor: float, tall: bool, map_ok: bool) -> Obstacle:
    """An object inside the pad, over the rotor limit or (as common) under it."""
    kind = rng.choice([k for k in IN_PAD if (DEFAULT[k][0] > rotor) == tall])
    h, src = DEFAULT[kind]
    if rng.random() < .4:   # a known height close to the limit
        h, src = (rotor * rng.uniform(1, 1 + NEAR) + .1 if tall else rotor * rng.uniform(1 - NEAR, 1)), "tagged" if map_ok else "measured"
    return _ob(kind, h, src, bearing=rng.randrange(360) if map_ok else None, source="map" if map_ok else "video", in_pad=True)


def _osm_zone(rng: random.Random, kind: str) -> Zone:
    return Zone(kind, float(rng.randint(20, 1100)), False, rng.randrange(360))


def _near_int(rng: random.Random, edge: int, below: bool, lo: int = 20, hi: int = 1100) -> int:
    """An integer distance on one side of `edge` (<= edge when below), half the time within 15% of it."""
    if below:
        return rng.randint(round(edge * (1 - NEAR)), edge) if rng.random() < .5 else rng.randint(lo, edge)
    return rng.randint(edge + 1, round(edge * (1 + NEAR))) if rng.random() < .5 else rng.randint(edge + 1, hi)


def _zones(rng: random.Random, osm: bool, govmap: bool, corridors: bool, mine: bool, corridor: bool) -> list[Zone]:
    z = []
    if osm and rng.random() < .35:   # OSM form: a distance and a bearing, or the pad inside it
        z += [Zone("military_area", 0.0, True, None) if rng.random() < .2 else _osm_zone(rng, "military_area")
              for _ in range(rng.choice((1, 1, 1, 2, 3, 6, 9)))]
    if osm and rng.random() < .12:
        z.append(_osm_zone(rng, "nature_reserve"))
    for kind, p in (("firing_zone", .2), ("nature_reserve", .12)):   # GovMap form: the tolerance that hit, no bearing
        if govmap and rng.random() < p:
            d = rng.choice((0, 500, 1000))
            z.append(Zone(kind, float(d), d == 0, None))
    if mine:   # known within 500 m of touchdown
        z.append(Zone("minefield", 0.0, True, None) if rng.random() < .1 else
                 Zone("minefield", float(_near_int(rng, 500, True)), False, rng.randrange(360)))
    elif osm and rng.random() < .06:
        z.append(Zone("minefield", float(_near_int(rng, 500, False)), False, rng.randrange(360)))
    if corridor:   # 300 m of the pad edge = 317.5 m of touchdown
        z.append(Zone("hv_corridor", 0.0, True, None) if rng.random() < .1 else
                 Zone("hv_corridor", float(_near_int(rng, 317, True)), False, rng.randrange(360)))
    elif corridors and rng.random() < .1:
        z.append(Zone("hv_corridor", float(_near_int(rng, 317, False)), False, rng.randrange(360)))
    rng.shuffle(z)
    return z


def _pick(rng: random.Random, weights: dict[str, float], ok: set[str], k: int) -> set[str]:
    opts = [o for o in weights if o in ok]
    out = set()
    while opts and len(out) < k:
        out.add(rng.choices(opts, [weights[o] for o in opts])[0])
    return out


def sample_case(rng: random.Random, name: str = "syn", target: str | None = None) -> SiteCase:
    """A case whose rule verdict (on `visible()`) is `target`; drawn from VERDICT_MIX when not given."""
    target = target or rng.choices(list(VERDICT_MIX), list(VERDICT_MIX.values()))[0]
    aircraft = 1 if rng.random() < .7 else 2
    rotor, max_slope = AIRCRAFT[aircraft]["rotor"], AIRCRAFT[aircraft]["max_slope"]
    core: tuple[str, ...] = ()
    if target == "not_checked" or (target == "fails" and rng.random() < .15):
        core = rng.choices((("osm",), ("dem",), ("osm", "dem")), (55, 30, 15))[0]
    osm, dem = "osm" not in core, "dem" not in core
    video = rng.random() < .5                       # video findings: the only source of wires, belly and non-map obstacles
    govmap = rng.random() < .4                      # every real fetch so far has GovMap and the corridors layer down
    corridors = govmap if rng.random() < .7 else rng.random() < .4
    wire = video and rng.random() < .7
    slot = rng.random() < .4 and (osm or video)     # something lies inside the pad

    seen = osm or video
    fail: set[str] = set()
    if target == "fails":
        if slot:
            fail = {rng.choices(("wire_pad", "pad", ""), (70, 20, 10))[0] if wire else rng.choices(("pad", ""), (60, 40))[0]} - {""}
        if not fail or rng.random() < .3:
            ok = {"canopy"} | ({"slope"} if dem else set()) | ({"minefield"} if osm else set()) | (set(CLASS_KINDS) if seen else set())
            fail |= _pick(rng, FAIL_W, ok, 1)
    ok = (set(CLASS_KINDS) if seen else set()) | ({"corridor"} if corridors else set()) | ({"wire"} if wire else set())
    if target == "probably_fails":
        prob = _pick(rng, PROB_W, ok, rng.choice((1, 1, 2)))
    else:
        prob = {k for k in PROB_W if k in ok and target != "approved" and rng.random() < .12}

    obs = _map_scene(rng) if osm else []
    if video:
        if osm:
            obs = [replace(o, source="map+video") if rng.random() < .1 else o for o in obs]
        for _ in range(rng.randint(0, 2 if osm else 6)):
            o = _of_class(rng, rng.choice(list(CLASS_KINDS)), rng.random() < .15, False)
            obs.append(_clear(rng, o, math.exp(rng.uniform(math.log(8), math.log(400)))))
    used: set[int] = set()
    for item in CLASS_KINDS:
        if item in fail or item in prob:
            _hit(rng, obs, used, item, item in fail, osm)
    if wire:
        d = 0.0 if "wire_pad" in fail else _inside(rng, 50) if "wire" in prob else _outside(rng, 50)
        obs.append(_ob("wire", rng.uniform(4, 12), "estimated", "overhead wire", d, source="video", in_pad="wire_pad" in fail))
    if slot and "wire_pad" not in fail:
        obs += [_in_pad(rng, rotor, "pad" in fail and i == 0, osm) for i in range(rng.choice((1, 1, 2)))]
    rng.shuffle(obs)

    unc = 6.0 if rng.random() < .85 else 2.0
    if not dem:
        slope = None
    elif "slope" in fail:
        slope = max_slope + unc + (rng.uniform(.1, 1.5) if rng.random() < .5 else rng.uniform(1.5, 8))
    elif rng.random() < .1:   # the uncertainty band straddles the limit: not checked
        slope = rng.uniform(max_slope - unc + .1, max_slope + unc)
    else:
        slope = (max_slope - unc) * (rng.uniform(1 - NEAR, 1) if rng.random() < .25 else rng.uniform(0, 1 - NEAR))
    canopy = None
    if "canopy" in fail or rng.random() < .85:   # the layer is in whole metres
        over = "canopy" in fail
        quirk = over and rng.random() < .1       # lz_check flags the unrounded maximum: 1.04 m prints as 1.0 m and fails
        pad = float(rotor if quirk else rng.randint(int(rotor) + 1, 12) if over else rng.choice((0, 0, 0, 0, rng.randint(0, int(rotor)))))
        m10 = pad + rng.choice((0, 0, 0, 1, 2, 5))
        canopy = {"pad_max_m": pad, "max_10_m": m10, "max_50_m": m10 + rng.choice((0, 0, 1, 3, 7, 10, 15)), "pad_over_rotor": over}
    return SiteCase(
        name=name, aircraft=aircraft, obstacles=obs,
        zones=_zones(rng, osm, govmap, corridors, "minefield" in fail, "corridor" in prob),
        slope_deg=None if slope is None else round(slope, 1), slope_unc_deg=unc, canopy=canopy,
        dust=rng.choices((None, "light", "medium", "heavy"), (10, 55, 27, 8))[0],
        belly=rng.choices(("clear", "obstacles"), (60, 40))[0] if video else None,
        govmap_ok=govmap, core_missing=core, corridors_ok=corridors,
        shift={"m": rng.randint(10, 60), "dir": rng.choice(COMPASS)} if rng.random() < .05 else None)


def sample_balanced(n: int, seed: int, mix: dict[str, float] = VERDICT_MIX) -> list[tuple[SiteCase, dict[str, str], str]]:
    """`n` cases in the exact `mix`, shuffled, each labelled by the rules on the case as rendered."""
    rng = random.Random(seed)
    want = {k: round(n * p) for k, p in mix.items()}
    want[max(want, key=want.get)] += n - sum(want.values())
    targets = [v for v, k in want.items() for _ in range(k)]
    rng.shuffle(targets)
    out = []
    for i, target in enumerate(targets):
        c = sample_case(rng, f"syn_{seed}_{i}", target)
        v = visible(c)
        items = judge(v)
        got = verdict(v, items)
        assert got == target, (c.name, target, got)   # the construction, not rejection, sets the verdict
        out.append((c, items, got))
    return out
