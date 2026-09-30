"""build_import (scripts/typically_spike.py): schema, 70/10/20 split, yes/no mapping, train-only balancing. No model."""
import random
from collections import Counter

import pytest

from scripts.typically_spike import build_import

RECORDS = [{"t": f"ticket {i}", "team": "A" if i % 2 else "B", "esc": "Yes" if i % 10 == 0 else "no", "urg": str(i % 4)}
           for i in range(100)]
DECISIONS = [{"column": "team", "question": "Which team?", "type": "choice"},
             {"column": "esc", "question": "Escalate?", "type": "noul"},
             {"column": "urg", "question": "How urgent?", "type": "score"}]


def test_build_import():
    s = build_import(RECORDS, "t", DECISIONS, random.Random(0), balance=False)
    assert set(s) == {"train", "val", "import_oneliner"}
    assert {k: len(v) // 3 for k, v in s.items()} == {"train": 70, "val": 10, "import_oneliner": 20}
    r = next(r for r in s["train"] if r["task"] == "import_esc")
    assert set(r) == {"state", "query", "candidates", "target", "p_null", "task", "label", "meta"}
    assert r["candidates"] == ["no", "yes"] and r["target"][r["label"]] == 1.0
    assert {x["state"] for v in s.values() for x in v} == {x["t"] for x in RECORDS}
    assert len({x["state"] for k in ("train", "val") for x in s[k]} & {x["state"] for x in s["import_oneliner"]}) == 0

    b = build_import(RECORDS, "t", DECISIONS, random.Random(0))
    c = Counter(x["label"] for x in b["train"] if x["task"] == "import_esc")
    assert c[1] >= c[0] // 3 - 1 and c[1] > Counter(x["label"] for x in s["train"] if x["task"] == "import_esc")[1]
    assert len(b["val"]) == len(s["val"]) and len(b["import_oneliner"]) == len(s["import_oneliner"])

    with pytest.raises(ValueError):
        build_import([{"t": "x", "e": "maybe"}], "t", [{"column": "e", "question": "?", "type": "noul"}], random.Random(0))
