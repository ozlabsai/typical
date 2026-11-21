"""The generator against the review's findings (C1, ML-2, I11, M9): verdict-first construction, real-like density in every
verdict class, a floor for rare statuses, mass near the thresholds, and no line-count shortcut to the labels."""
import random
import statistics
from collections import Counter, defaultdict
from functools import cache

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

from helipad.case import ALLOWED, DEFAULT_CAP, RINGS, render_case, visible
from helipad.generate import VERDICT_MIX, sample_balanced, sample_case
from helipad.rows import case_rows
from helipad.rules import judge, verdict
from scripts.helipad_baselines import features

N = 6000
OSM_ZONES = {"military_area", "minefield"}


@cache
def _data():
    return sample_balanced(N, seed=0)


def _share(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def test_deterministic():
    a, b = sample_balanced(50, seed=3), sample_balanced(50, seed=3)
    assert [render_case(c) for c, _, _ in a] == [render_case(c) for c, _, _ in b]
    assert [(i, v) for _, i, v in a] == [(i, v) for _, i, v in b]
    assert [render_case(c) for c, _, _ in a] != [render_case(c) for c, _, _ in sample_balanced(50, seed=4)]


def test_labels_are_rule_labels_on_the_visible_case():
    for c, items, v in _data()[:1500]:
        shown = visible(c)
        assert items == judge(shown) and v == verdict(shown, items)
        assert items == judge(c)   # the cap and the printed precision hide nothing the rules use


def test_verdict_mix_is_constructed():
    mix = Counter(v for _, _, v in _data())
    for k, p in VERDICT_MIX.items():
        assert abs(mix[k] / N - p) < 0.03
    rng = random.Random(0)   # no target given: drawn from the mix, and still built to it
    free = Counter(verdict(v, judge(v)) for v in (visible(sample_case(rng, "x")) for _ in range(1500)))
    for k, p in VERDICT_MIX.items():
        assert abs(free[k] / 1500 - p) < 0.04


def test_every_reachable_status_has_a_floor():
    seen = Counter((i, s) for _, items, _ in _data() for i, s in items.items())
    # pad/probably_fails stays a candidate for teacher labels (Plan 2) but the rules, like lz_check, never emit it
    reachable = {(i, s) for i, ss in ALLOWED.items() for s in ss} - {("pad", "probably_fails")}
    assert set(seen) == reachable
    assert min(seen.values()) >= 0.01 * N, min(seen.items(), key=lambda kv: kv[1])


def test_approved_cases_are_as_dense_as_failing_ones():
    raw, shown = defaultdict(list), defaultdict(list)
    for c, _, v in _data():
        raw[v].append(len(c.obstacles))
        shown[v].append(len(visible(c).obstacles))
    bad = raw["probably_fails"] + raw["fails"]
    assert abs(statistics.mean(raw["approved"]) / statistics.mean(bad) - 1) < 0.2
    assert abs(statistics.mean(shown["approved"]) / statistics.mean(shown["probably_fails"] + shown["fails"]) - 1) < 0.1
    assert _share(n == DEFAULT_CAP for n in shown["approved"]) > 0.8      # real approved sites render the full list
    assert statistics.median(raw["approved"]) >= 40                        # real complete fetches: 42-60
    assert min(raw["approved"]) == 0 and 60 <= max(len(c.obstacles) for c, _, _ in _data()) <= 66


def test_real_forms_without_contradictions():
    zone_forms, labels = set(), set()
    for c, _, _ in _data():
        osm = "osm" not in c.core_missing
        video = c.belly is not None
        for o in c.obstacles:
            labels.add((o.label, o.height_src))
            assert osm or o.source == "video"
            assert video or o.source == "map"
            assert (o.kind == "wire") <= (o.source == "video")
            assert (o.height_src == "iec") == o.label.startswith("high-voltage line")
            assert (o.bearing is None) == (o.source == "video")   # only the map gives a direction
        for z in c.zones:
            zone_forms.add((z.kind, z.bearing is not None, z.inside_pad))
            assert osm or z.kind not in OSM_ZONES
            assert c.govmap_ok or z.kind != "firing_zone"
            assert c.corridors_ok or z.kind != "hv_corridor"
            if z.kind == "firing_zone":   # GovMap form: the tolerance that hit, no bearing
                assert z.dist_center_m in (0, 500, 1000) and z.bearing is None and z.inside_pad == (z.dist_center_m == 0)
    assert {("military_area", True, False), ("military_area", False, True), ("nature_reserve", True, False),
            ("nature_reserve", False, False), ("firing_zone", False, True), ("firing_zone", False, False),
            ("hv_corridor", True, False), ("minefield", True, False)} <= zone_forms
    assert {("high-voltage line 161 kV", "iec"), ("power line (voltage unknown)", "unknown"), ("power pole", "unknown"),
            ("lit road (row of lighting poles)", "unknown"), ("building", "levels")} <= labels
    by_verdict = defaultdict(list)
    for c, _, v in _data():
        by_verdict[v].append(c)
    for v, cs in by_verdict.items():   # every real fetch so far has both layers down
        assert _share(not c.govmap_ok for c in cs) > 0.5 and _share(not c.corridors_ok for c in cs) > 0.5, v


def test_runs_of_unknown_height_poles_at_increasing_distances():
    rows = [sorted(o.dist_edge_m for o in c.obstacles if o.kind == "power_pole" and o.height_src == "unknown") for c, _, _ in _data()]
    long = [r for r in rows if len(r) >= 5]
    assert len(long) > 0.1 * N and all(len(set(r)) > 1 for r in long)


def test_deciding_obstacles_sit_near_the_thresholds_on_both_sides():
    hit, miss = [], []
    for c, _, _ in _data():
        for _, lo, hi, ring in RINGS:
            d = [o.dist_edge_m for o in visible(c).obstacles if lo <= o.height_m < hi and not o.in_pad]
            if d and min(d) < ring:
                hit.append(any(0.85 * ring <= x < ring for x in d))
            elif d:
                miss.append(min(d) <= 1.15 * ring)
    assert _share(hit) > 0.4 and _share(miss) > 0.4
    heights = [o.height_m for c, _, _ in _data() for o in c.obstacles if o.height_src in ("tagged", "measured")]
    near = [h for h in heights if any(0.85 * e <= h <= 1.15 * e for e in (1.2, 4.5, 24, 76))]
    assert len(near) > 0.25 * len(heights) and any(h < 4.5 for h in near) and any(h >= 4.5 for h in near)


def _shortcut_features(state: str) -> dict:
    """scripts/helipad_baselines.features plus what the review's count tree also saw: obstacle lines, certain heights,
    counts by kind and inside each ring, the nearest distance. It never pairs a height with its own distance."""
    f = features(state)
    lines = [ln for ln in state.split("\n") if " | " in ln and not ln.startswith(("Zone:", "Obstacles"))]
    dist = [float(ln.split(" m from pad edge")[0].rsplit(" ", 1)[1]) for ln in lines if " m from pad edge" in ln]
    f |= {"obstacle_lines": len(lines), "certain": sum(" m (" in ln for ln in lines), "nearest": min(dist, default=9999)}
    f |= {f"closer_{r}": sum(d < r for d in dist) for r in (10, 50, 300)}
    f |= {w: sum(w in ln for ln in lines) for w in ("power", "lit road", "overhead wire", "building")}
    return f


def test_line_count_tree_cannot_shortcut_the_labels():
    """ML-2: the old generator gave this tree 0.86 verdict / 0.81 items; majority class is 0.35 / 0.70."""
    def by_family(data):
        out = defaultdict(lambda: ([], []))
        for c, _, _ in data:
            for r in case_rows(c):
                X, y = out[r["meta"]["family"]]
                X.append(_shortcut_features(r["state"]))
                y.append(r["candidates"][r["label"]])
        return out

    train, held = by_family(_data()[:3000]), by_family(sample_balanced(1000, seed=1))
    acc, item_hits = {}, []
    for fam, (X, y) in train.items():
        vec = DictVectorizer()
        tree = DecisionTreeClassifier(max_depth=8, random_state=0).fit(vec.fit_transform(X), y)
        ok = [p == t for p, t in zip(tree.predict(vec.transform(held[fam][0])), held[fam][1])]
        acc[fam] = _share(ok)
        if fam != "verdict":
            item_hits += ok
    assert acc["verdict"] <= 0.70, acc
    assert _share(item_hits) <= 0.88, acc
