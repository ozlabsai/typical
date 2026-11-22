"""scripts/helipad_vision_eval.py: the vision report on hand-built predictions. uv run pytest tests/test_helipad_vision_eval.py -q"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import helipad_vision_eval as ve  # noqa: E402
from helipad import frames as fr  # noqa: E402

NULL = fr.NULL_OPTION
Q2 = {"wires": fr.FRAME_QUESTIONS["wires"], "pole": fr.FRAME_QUESTIONS["pole"]}


@pytest.fixture(autouse=True)
def two_questions(monkeypatch):
    monkeypatch.setattr(fr, "FRAME_QUESTIONS", Q2)


def row(variant: str, wires: str = "yes", pole: str = "no") -> dict:
    t = {"wires": {wires: 1.0}, "pole": {pole: 1.0}} if variant != "mismatch" else {q: {NULL: 1.0} for q in Q2}
    return fr.grouped_row("v/f00", None if variant == "noimage" else "x.jpg", t, variant=variant)


def pred(wires: str, pole: str, p: float = 0.9) -> dict:
    out = {}
    for q, top in (("wires", wires), ("pole", pole)):
        rest = (1 - p) / 2
        out[q] = {o: (p if o == top else rest) for o in fr.options(q)}
    return out


def test_vision_report_metrics():
    rows = [row("real"), row("real"), row("shuffled"), row("shuffled"), row("noimage"), row("noimage"),
            row("mismatch"), row("mismatch")]
    preds = [pred("yes", "no"), pred("no", "no"),            # real: wires 1/2, pole 2/2
             pred("no", "yes"), pred("no", "no"),            # shuffled: wires 0/2, pole 1/2
             pred(NULL, NULL), pred(NULL, NULL),             # noimage: 0/4
             pred(NULL, NULL, 0.95), pred(NULL, "no", 0.6)]  # mismatch: null 3/4
    r = ve.vision_report(rows, preds)
    assert r["agreement"]["real"]["wires"] == {"value": 0.5, "n": 2}
    assert r["agreement"]["real"]["pole"] == {"value": 1.0, "n": 2}
    assert r["agreement"]["real"]["overall"] == {"value": 0.75, "n": 4}
    assert r["agreement"]["shuffled"]["overall"] == {"value": 0.25, "n": 4}
    assert r["real_minus_shuffled"] == {"value": 0.5, "n": 4}
    assert r["real_minus_noimage"] == {"value": 0.75, "n": 4}
    assert r["null_recall"]["overall"] == {"value": 0.75, "n": 4} and r["null_recall"]["pole"]["value"] == 0.5
    # P(null): real rows .05 each (4 negatives), mismatch .95, .95, .6, .2 (positives) -> every positive outranks
    assert r["answerability_auroc"] == {"value": 1.0, "n_pos": 4, "n_neg": 4}
    assert r["ece"]["wires"]["n"] == 2 and r["ece"]["wires"]["value"] == pytest.approx(abs(0.9 - 0.5))
    assert r["majority_baseline"]["wires"] == {"value": 1.0, "n": 2}


def test_empty_suites_give_none():
    r = ve.vision_report([row("real")], [pred("yes", "no")])
    assert r["null_recall"]["overall"] == {"value": None, "n": 0}
    assert r["real_minus_shuffled"] == {"value": None, "n": 0}
    assert r["answerability_auroc"]["value"] is None
