"""Minimal pairs: a base case plus variants that change exactly one thing across a decision boundary. A model that
reads the rule gets both members right; one leaning on line-count shortcuts does not."""
from __future__ import annotations

import random
from dataclasses import replace

from helipad.case import CERTAIN_SRC, Obstacle, SiteCase, Zone, render_case, visible
from helipad.rules import RINGS, judge

HEIGHT_EDGES = (4.4, 4.6, 23.9, 24.1, 75.9, 76.1)   # just under / over the ring classes at 4.5, 24 and 76 m


def _obstacle_edits(o: Obstacle):
    ring = next((r for _, lo, hi, r in RINGS if lo <= o.height_m < hi), None)
    if not o.in_pad and ring:
        yield from (replace(o, dist_edge_m=ring + d) for d in (-0.5, 0.5))
    yield replace(o, height_src="estimated" if o.height_src in CERTAIN_SRC else "measured")
    yield from (replace(o, height_m=h) for h in HEIGHT_EDGES)


def _candidates(c: SiteCase):
    for i, o in enumerate(c.obstacles):
        for new in _obstacle_edits(o):
            yield replace(c, obstacles=[*c.obstacles[:i], new, *c.obstacles[i + 1:]])
    yield replace(c, core_missing=() if c.core_missing else ("osm",))
    for kind in ("firing_zone", "minefield"):
        yield replace(c, zones=[z for z in c.zones if z.kind not in ("firing_zone", "minefield")] + [Zone(kind, 400.0, False, None)])


def _body(c: SiteCase) -> str:
    return render_case(visible(c)).split("\n", 1)[1]   # the Site: name line is not part of the state's content


def minimal_pairs(cases: list[SiteCase], rng: random.Random, per_case: int = 2) -> list[SiteCase]:
    """Variants named `<base>~<k>` whose labels differ from the base on at least one item."""
    out = []
    for c in cases:
        base_items, base_body = judge(visible(c)), _body(c)
        flips = [v for v in _candidates(c) if judge(visible(v)) != base_items and _body(v) != base_body]
        out += [replace(v, name=f"{c.name}~{k}") for k, v in enumerate(rng.sample(flips, min(per_case, len(flips))))]
    return out
