import json
import math
import random

import pytest
import torch
import torch.nn.functional as F

from scripts.train_gemma_decision_lora import (
    group_twins, load_examples, logical_slot_views, loss_for_examples, twin_js_term)
from tests.test_train_gemma_pair_consistency import TinyModel, TinyTokenizer


def _row(identifier, twin_id, lang, criteria):
    return {"id": identifier, "state": f"state {lang}",
            "questions": {"q": {"type": "noul", "instructions": f"choose {lang}",
                                "criteria": criteria}},
            "targets": {"q": {"label": "yes"}},
            "meta": {"twin_id": twin_id, "lang": lang}}


def test_js_aligns_by_option_key_not_letter_position():
    # Same distribution, shuffled presentation: JS must be zero.
    same = twin_js_term([(torch.tensor([2.0, 0.0]), ["yes", "no"]),
                         (torch.tensor([0.0, 2.0]), ["no", "yes"])])
    assert same.item() == pytest.approx(0.0, abs=1e-7)
    # Positional comparison would call these equal; key-aligned JS must not.
    p, q = torch.tensor([2.0, 0.0]), torch.tensor([2.0, 0.0])
    js = twin_js_term([(p, ["yes", "no"]), (q, ["no", "yes"])])
    a, b = F.softmax(p, 0), F.softmax(q, 0).flip(0)
    m = (a + b) / 2
    expected = 0.5 * ((a * (a / m).log()).sum() + (b * (b / m).log()).sum())
    assert js.item() == pytest.approx(expected.item(), rel=1e-6)
    assert 0 < js.item() <= math.log(2)
    with pytest.raises(ValueError, match="identical option keys"):
        twin_js_term([(p, ["yes", "no"]), (q, ["no", "maybe"])])


def test_js_gradient_flows_through_both_twins():
    p = torch.tensor([1.0, 0.0], requires_grad=True)
    q = torch.tensor([0.0, 1.0], requires_grad=True)
    twin_js_term([(p, ["no", "yes"]), (q, ["no", "yes"])]).backward()
    assert p.grad.abs().sum() > 0 and q.grad.abs().sum() > 0


def test_twins_group_into_one_slot_and_loss_adds_weighted_js(tmp_path):
    criteria = {"no": "No", "yes": "Yes"}
    path = tmp_path / "rows.jsonl"
    rows = [_row("en-1", "t1", "en", criteria), _row("solo", "t9", "en", criteria),
            {**_row("plain", None, "en", criteria), "meta": {}},
            _row("he-1", "t1", "he", {"yes": "כן", "no": "לא"}),
            _row("mx-1", "t1", "mx", criteria)]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    flat = load_examples([path])
    assert "twin" not in flat[2]
    slots = group_twins(flat)
    assert [slot["id"] for slot in slots] == ["twin:t1/q", "solo/q", "plain/q"]
    assert [view["id"] for view in logical_slot_views(slots[0])] == ["en-1/q", "he-1/q", "mx-1/q"]

    model, groups = TinyModel(), [torch.tensor([1]), torch.tensor([2])]
    state = random.Random(5).getstate()
    random.setstate(state)
    base, _, stats0 = loss_for_examples(model, slots[:1], TinyTokenizer(), groups, 100,
                                        torch.device("cpu"), shuffle=True)
    random.setstate(state)
    weighted, _, stats = loss_for_examples(model, slots[:1], TinyTokenizer(), groups, 100,
                                           torch.device("cpu"), shuffle=True,
                                           twin_consistency_weight=0.5)
    assert stats["twin_seen"] == 1 and stats["canonical_views_seen"] == 3
    assert weighted.item() == pytest.approx(base.item() + 0.5 * stats["twin_js"])

    he = _row("he-2", "t2", "he", {"no": "לא", "yes": "כן", "maybe": "אולי"})
    path.write_text(json.dumps(_row("en-2", "t2", "en", criteria)) + "\n" + json.dumps(he) + "\n")
    with pytest.raises(ValueError, match="identical option keys"):
        group_twins(load_examples([path]))
