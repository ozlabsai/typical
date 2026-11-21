from scripts.helipad_baselines import baselines, features

S1 = ("Aircraft: 1\nSlope: not available\nObstacles (what | height | distance | direction | source):\n"
      "pole | height unknown, treated as 12.0 m | inside the pad | direction unknown | map")
S2 = "Aircraft: 1\nSlope in pad: 3.0° (±1°)\nMissing core data: OSM"


def _row(state, gold, fam="pad", cands=("passes", "fails")):
    return {"state": state, "candidates": list(cands), "label": list(cands).index(gold),
            "meta": {"family": fam, "case": state[:3], "expert": None}}


def test_features_count_lines_and_phrases():
    f = features(S1)
    assert f["height unknown"] == 1 and f["inside the pad"] == 1 and f["obstacles"] == 1
    assert features(S2)["Missing core data"] == 1


def test_tree_learns_a_line_count_shortcut_and_majority_does_not():
    train = [_row(S1, "fails"), _row(S2, "passes"), _row(S2, "passes")] * 3
    out = baselines(train, [_row(S1, "fails"), _row(S2, "passes"), _row(S2, "passes")])
    assert out["pad"] == {"majority": 2 / 3, "tree": 1.0, "n": 3}
    assert out["overall"]["tree"] == 1.0 and "by construction" in out["rule_engine"]


def test_expert_rows_and_unseen_families_are_skipped():
    out = baselines([_row(S1, "fails")], [{**_row(S1, "fails"), "meta": {"family": "pad", "case": "a", "expert": "approved"}},
                                          _row(S1, "fails", fam="dust")])
    assert "pad" not in out and "dust" not in out


def test_features_count_obstacle_lines_in_rendered_format():
    from helipad.case import render_case, visible
    from helipad.generate import sample_balanced
    from scripts.helipad_baselines import features
    c = next(c for c, _, _ in sample_balanced(20, seed=11) if 3 <= len(visible(c).obstacles))
    assert features(render_case(visible(c)))["obstacles"] == len(visible(c).obstacles)
