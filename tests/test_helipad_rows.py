from helipad.case import ALLOWED, ITEMS, Obstacle, Zone, label, render_case, visible
from helipad.rows import POLICY, QUESTIONS, case_rows
from helipad.rules import judge, verdict
from tests.test_helipad_case import _case


def _site(**kw):
    obs = [Obstacle("power_line", "power line (voltage unknown)", 24, "unknown", 140, False, 8, "map"),
           Obstacle("building", "building", 9, "levels", 30, False, 200, "map")]
    return _case(obstacles=obs, zones=[Zone("firing_zone", 500, False, None)], dust="medium", **kw)


def _gold(r):
    return r["candidates"][r["label"]]


def test_rows_shape():
    c = _site()
    rows = case_rows(c)
    assert len(rows) == 12 and {r["meta"]["family"] for r in rows} == set(ITEMS) | {"verdict"}
    for r in rows:
        assert abs(sum(r["target"]) - 1) < 1e-9 and r["target"][r["label"]] == 1.0
        assert "Rule:" in r["query"] and r["task"] == f"import_{r['meta']['family']}"     # I4
        assert set(r["meta"]) == {"qtype", "family", "case", "core_missing"} and r["meta"]["case"] == "site_x"


def test_rows_label_the_visible_case():                           # I3: labels computed by case_rows itself
    c = _case(obstacles=[Obstacle("tree", "tree", 3, "estimated", 20.0 + i, False, 0, "map") for i in range(40)] +
              [Obstacle("power_line", "power line", 24, "unknown", 200, False, 0, "map")])
    rows = {r["meta"]["family"]: r for r in case_rows(c)}
    items = judge(visible(c))
    assert all(_gold(rows[k]) == label(items[k]) for k in ITEMS)
    assert _gold(rows["verdict"]) == label(verdict(visible(c), items)) == "probably fails"
    assert all(r["state"] == render_case(c) for r in rows.values())


def test_candidate_shuffle_deterministic():                       # ARCH-2
    c = _site()
    a, b = case_rows(c), case_rows(c)
    assert [r["candidates"] for r in a] == [r["candidates"] for r in b]
    orders = {tuple(r["candidates"]) for r in a if r["meta"]["family"] in ("pad", "very_low_10", "low_50", "high_300")}
    assert len(orders) > 1                                        # same option set, not the same order everywhere
    other = case_rows(c, rng_seed=123)
    assert [r["candidates"] for r in other] != [r["candidates"] for r in a]
    for r in a + other:
        assert sorted(r["candidates"]) == sorted(label(s) for s in ALLOWED.get(r["meta"]["family"],
                                                                                 ("approved", "probably_fails", "fails", "not_checked")))


def test_expert_row_soft_target():
    (r,) = case_rows(_site(), expert="not_approved")
    t = dict(zip(r["candidates"], r["target"]))
    assert t["probably fails"] == 0.5 and t["fails"] == 0.5 and r["meta"]["expert"] == "not_approved"
    (r,) = case_rows(_site(), expert="approved")
    assert dict(zip(r["candidates"], r["target"]))["approved"] == 1.0 and _gold(r) == "approved"


def test_expert_item_rows():                                      # ARCH-4
    rows = case_rows(_site(), expert="not_approved", expert_items={"high_300": ("fails", "probably_fails")})
    assert [r["meta"]["family"] for r in rows] == ["verdict", "high_300"]
    t = dict(zip(rows[1]["candidates"], rows[1]["target"]))
    assert t == {"passes": 0, "fails": 0.5, "probably fails": 0.5, "not checked": 0}
    assert rows[1]["meta"]["expert"] == "not_approved" and rows[1]["task"] == "import_high_300"


def test_policy_covers_all():
    assert set(QUESTIONS) == set(POLICY) == set(ITEMS) | {"verdict"}


def test_policy_states_label_rules():                             # I9
    for k in ("very_low_10", "low_50", "high_300"):
        assert all(s in POLICY[k] for s in ("tagged", "levels", "IEC", "measured", "OSM", "canopy")), k
    for k in ("pad", "ultra_1km"):
        assert "OSM" in POLICY[k]
    assert "corridor" in POLICY["high_300"] and "minefield" in POLICY["zones"]
    assert all(POLICY[k].count(". ") <= 2 for k in POLICY)        # two or three sentences each
