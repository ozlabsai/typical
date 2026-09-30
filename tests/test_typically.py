"""build_import (scripts/typically_spike.py): schema, 70/10/20 split, yes/no mapping, train-only balancing. No model."""
import json
import random
import re
from collections import Counter

import pytest

from scripts.typically_spike import build_import

RECORDS = [{"t": f"ticket {i}", "team": "A" if i % 2 else "B", "esc": "Yes" if i % 10 == 0 else "no", "urg": str(i % 4)}
           for i in range(100)]
DECISIONS = [{"column": "team", "question": "Which team?", "type": "choice"},
             {"column": "esc", "question": "Escalate?", "type": "noul"},
             {"column": "urg", "question": "How urgent?", "type": "score"}]


def test_build_import():
    s = build_import(RECORDS, "t", DECISIONS, random.Random(0), balance=False, flips=0)
    assert set(s) == {"train", "val", "import_oneliner"}
    assert {k: len(v) // 3 for k, v in s.items()} == {"train": 70, "val": 10, "import_oneliner": 20}
    r = next(r for r in s["train"] if r["task"] == "import_esc")
    assert set(r) == {"state", "query", "candidates", "target", "p_null", "task", "label", "meta"}
    assert r["candidates"] == ["no", "yes"] and r["target"][r["label"]] == 1.0
    assert {x["state"] for v in s.values() for x in v} == {x["t"] for x in RECORDS}
    assert len({x["state"] for k in ("train", "val") for x in s[k]} & {x["state"] for x in s["import_oneliner"]}) == 0

    b = build_import(RECORDS, "t", DECISIONS, random.Random(0), flips=0)
    c = Counter(x["label"] for x in b["train"] if x["task"] == "import_esc")
    assert c[1] >= c[0] // 3 - 1 and c[1] > Counter(x["label"] for x in s["train"] if x["task"] == "import_esc")[1]
    assert len(b["val"]) == len(s["val"]) and len(b["import_oneliner"]) == len(s["import_oneliner"])

    with pytest.raises(ValueError):
        build_import([{"t": "x", "e": "maybe"}], "t", [{"column": "e", "question": "?", "type": "noul"}], random.Random(0))


def test_import_flips():
    words = ["alpha", "bravo", "cargo", "delta", "eagle"]   # each in 20% of the cases
    recs = [{**r, "t": f"ticket {i} {words[i % 5]}"} for i, r in enumerate(RECORDS)]
    build = lambda **kw: build_import(recs, "t", DECISIONS, random.Random(0), balance=False, **kw)
    s, off = build(), build(flips=0)
    held = lambda sp: {k: sorted(json.dumps(x) for x in sp[k]) for k in ("val", "import_oneliner")}
    assert held(s) == held(off)   # no flip or ms_group row outside train
    assert not any("rflip" in x["task"] or "ms_group" in x["meta"] for k in ("val", "import_oneliner") for x in s[k])
    assert not any("rflip" in x["task"] or "ms_group" in x["meta"] for x in off["train"])
    flips = [x for x in s["train"] if x["task"] == "import_rflip"]
    assert 0.03 < len(flips) / (70 * 3) < 0.20   # ~10% of train (case, decision) pairs
    plain = {x["meta"]["ms_group"]: x for x in s["train"] if x["task"] != "import_rflip" and "ms_group" in x["meta"]}
    for f in flips:
        p = plain[f["meta"]["ms_group"]]
        assert p["state"] == f["state"] and p["candidates"] == f["candidates"] and f["query"].startswith(p["query"])
        word = re.search(r'mentions "(\w+)"', f["query"]).group(1)
        hit = word in f["state"].split()
        if p["candidates"] == ["no", "yes"] and "answer yes only" in f["query"]:
            assert f["candidates"][f["label"]] == ("yes" if hit else "no")
        else:
            x, y = re.search(r"pick (.+) when .*, otherwise (.+)\.$", f["query"]).groups()
            assert f["candidates"][f["label"]] == (x if hit else y)
