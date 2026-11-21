import random
from dataclasses import fields

from helipad.case import render_case, visible
from helipad.contrast import minimal_pairs
from helipad.generate import sample_balanced
from helipad.rules import judge


def _cases():
    return [c for c, _, _ in sample_balanced(40, 3)]


def test_every_variant_flips_a_label_and_changes_the_text():
    cases = {c.name: c for c in _cases()}
    variants = minimal_pairs(list(cases.values()), random.Random(0))
    assert variants
    for v in variants:
        base = cases[v.name.split("~")[0]]
        assert judge(visible(v)) != judge(visible(base))
        assert render_case(visible(v)).split("\n", 1)[1] != render_case(visible(base)).split("\n", 1)[1]


def test_variants_change_one_thing():
    cases = {c.name: c for c in _cases()}
    for v in minimal_pairs(list(cases.values()), random.Random(1)):
        base = cases[v.name.split("~")[0]]
        diff = [f.name for f in fields(v) if f.name != "name" and getattr(v, f.name) != getattr(base, f.name)]
        assert len(diff) == 1 and diff[0] in {"obstacles", "core_missing", "zones"}


def test_deterministic_and_capped():
    cs = _cases()
    a, b = minimal_pairs(cs, random.Random(5)), minimal_pairs(cs, random.Random(5))
    assert [v.name for v in a] == [v.name for v in b]
    assert all(sum(v.name.startswith(c.name + "~") for v in a) <= 2 for c in cs)
