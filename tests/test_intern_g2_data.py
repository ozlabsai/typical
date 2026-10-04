from scripts.intern_g2_data import canonical, valid


def test_canonical_preserves_soft_noul_distribution_order():
    row = {"id": "x", "kind": "noul", "state": "s", "question": "true?",
           "options": ["true", "false"], "target": [0.8, 0.2]}

    record = canonical("sarge", row)

    assert record["targets"]["decision"] == {"label": "yes"}
    assert record["soft_targets"]["decision"] == [0.2, 0.8]
    assert not valid(row | {"target": [0.5, 0.5]})
