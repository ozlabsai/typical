import pytest

from scripts.helipad_score import pooled_records, preds_from_cases, report

V = ["approved", "probably fails", "fails", "not checked"]
H = ["passes", "fails", "probably fails", "not checked"]
W = ["not checked", "probably fails", "fails"]   # low_wires: no "passes" candidate


def _row(fam, cands, gold, case="c1", core=False, expert=None, target=None):
    meta = {"family": fam, "case": case, "core_missing": core, "expert": expert}
    return {"candidates": cands, "label": cands.index(gold), "target": target, "meta": meta}


def _pred(ans, cands, conf=1.0):
    rest = (1 - conf) / (len(cands) - 1)
    return {"argmax": ans, "probs": {c: conf if c == ans else rest for c in cands}}


def test_verdict_of_record_and_consistency():
    rows = [_row("high_300", H, "probably fails"), _row("verdict", V, "probably fails")]
    preds = [_pred("probably fails", H), _pred("approved", V)]   # item says probably, own verdict says approved
    r = report(rows, preds)
    assert r["item_agreement"]["overall"] == 1.0 and r["verdict_agreement"] == 0.0
    assert r["verdict_of_record_agreement"] == 1.0 and r["consistency"] == 0.0


def test_verdict_of_record_uses_core_missing():
    rows = [_row("high_300", H, "passes", core=True), _row("verdict", V, "not checked", core=True)]
    r = report(rows, [_pred("passes", H), _pred("not checked", V)])
    assert r["verdict_of_record_agreement"] == 1.0 and r["consistency"] == 1.0


def test_expert_model_verdict_without_items():
    rows = [_row("verdict", V, "probably fails", case="a", expert="not_approved"),
            _row("verdict", V, "approved", case="b", expert="approved"),
            _row("verdict", V, "probably fails", case="c", expert="not_approved")]
    preds = [_pred("fails", V), _pred("probably fails", V), _pred("not checked", V)]
    r = report(rows, preds)
    want = {"n": 3, "direction_match": 1, "abstained": 1}
    assert r["expert"] == r["expert_model_verdict"] == want   # no item rows: falls back to the model's verdict
    assert r["expert_record"] == {"n": 0, "direction_match": 0, "abstained": 0}


def test_expert_uses_record_when_items_present():
    rows = [_row("verdict", V, "probably fails", case="a", expert="not_approved"),   # has items below
            _row("verdict", V, "approved", case="b", expert="approved"),             # no items
            _row("low_50", H, "probably fails", case="a"), _row("verdict", V, "probably fails", case="a")]
    preds = [_pred("approved", V), _pred("approved", V), _pred("probably fails", H), _pred("approved", V)]
    r = report(rows, preds)
    assert r["expert_model_verdict"] == {"n": 2, "direction_match": 1, "abstained": 0}
    assert r["expert_record"] == {"n": 1, "direction_match": 1, "abstained": 0}   # record of a = probably fails
    assert r["expert"] == {"n": 2, "direction_match": 2, "abstained": 0}


def test_expert_joins_records_from_other_files():
    site = ([_row("pad", H, "fails", case="a"), _row("verdict", V, "fails", case="a")],
            [_pred("fails", H), _pred("approved", V)])
    expert = ([_row("verdict", V, "fails", case="a", expert="not_approved")], [_pred("approved", V)])
    r = report(*expert, records=pooled_records([site, expert]))
    assert r["expert_record"] == {"n": 1, "direction_match": 1, "abstained": 0}
    assert r["expert_model_verdict"]["direction_match"] == 0


def test_expert_items_match_any_acceptable_status():
    rows = [_row("high_300", H, "fails", case="a", expert="not_approved", target=[0, .5, .5, 0]),
            _row("low_50", H, "fails", case="a", expert="not_approved", target=[0, .5, .5, 0])]
    r = report(rows, [_pred("probably fails", H), _pred("passes", H)])
    assert r["expert_items"] == {"n": 2, "match": 1}
    assert r["item_agreement"]["overall"] is None   # expert rows stay out of the rule metrics


def test_unsafe_item_rate_only_where_passes_is_a_candidate():
    rows = [_row("high_300", H, "fails", case="a"), _row("low_50", H, "probably fails", case="a"),
            _row("pad", H, "not checked", case="a"), _row("slope", H, "passes", case="a"),
            _row("low_wires", W, "fails", case="a"), _row("verdict", V, "fails", case="a")]
    preds = [_pred("passes", H), _pred("probably fails", H), _pred("passes", H), _pred("passes", H),
             _pred("not checked", W), _pred("approved", V)]
    r = report(rows, preds)
    assert r["unsafe_item_rate"] == {"high_300": 1.0, "low_50": 0.0, "overall": 0.5}
    assert r["false_clear_on_not_checked"] == {"pad": 1.0, "overall": 1.0}


def test_no_risky_rows_give_none():
    rows = [_row("pad", H, "passes", case="a"), _row("verdict", V, "approved", case="a")]
    r = report(rows, [_pred("passes", H), _pred("approved", V)])
    assert r["unsafe_item_rate"] == {"overall": None}
    assert r["false_clear_on_not_checked"] == {"overall": None}
    assert r["not_checked_recall"] == {"overall": None}
    assert r["unsafe_record_rate"] is None
    assert r["contrast_flipped_agreement"] is None and r["contrast_flipped_n"] == 0


def test_unsafe_record_rate():
    rows = [_row("low_50", H, "probably fails", case="a"), _row("verdict", V, "probably fails", case="a"),
            _row("pad", H, "fails", case="b"), _row("verdict", V, "fails", case="b"),
            _row("pad", H, "passes", case="c"), _row("verdict", V, "approved", case="c")]
    preds = [_pred("passes", H), _pred("fails", V), _pred("fails", H), _pred("fails", V),
             _pred("passes", H), _pred("approved", V)]
    assert report(rows, preds)["unsafe_record_rate"] == 0.5   # a cleared by its record, b blocked; c is not risky


def test_not_checked_recall_per_family_and_overall_without_always_not_checked():
    rows = [_row("pad", H, "not checked", case="a"), _row("slope", H, "not checked", case="a"),
            _row("low_wires", W, "not checked", case="a")]
    r = report(rows, [_pred("not checked", H), _pred("passes", H), _pred("not checked", W)])
    assert r["not_checked_recall"] == {"low_wires": 1.0, "pad": 1.0, "slope": 0.0, "overall": 0.5}


def test_ece_per_family_and_rings():
    rows = [_row("pad", H, "passes", case="a"), _row("slope", H, "passes", case="a"),
            _row("very_low_10", H, "passes", case="a")]
    r = report(rows, [_pred("passes", H, conf=0.9), _pred("fails", H, conf=0.9), _pred("passes", H, conf=0.8)])
    assert r["ece"]["overall"] == pytest.approx((0.4 * 2 + 0.2) / 3)   # bins .9 (acc .5) and .8 (acc 1)
    assert r["ece"]["slope"] == pytest.approx(0.9) and r["ece"]["pad"] == pytest.approx(0.1)
    assert r["ece_rings"] == pytest.approx((0.1 + 0.2) / 2)


def test_contrast_scores_only_flipped_rows_and_pairs():
    rows = [_row("low_50", H, "passes", case="s"), _row("low_50", H, "fails", case="s~0"),
            _row("pad", H, "passes", case="s"), _row("pad", H, "passes", case="s~0"),          # not flipped
            _row("high_300", H, "fails", case="s"), _row("high_300", H, "passes", case="s~1"),
            _row("low_50", H, "fails", case="gone~0")]                                          # base missing
    preds = [_pred("passes", H), _pred("fails", H), _pred("passes", H), _pred("passes", H),
             _pred("passes", H), _pred("passes", H), _pred("fails", H)]
    r = report(rows, preds)
    assert r["contrast_flipped_n"] == 2
    assert r["contrast_flipped_agreement"] == 1.0   # both variants right
    assert r["contrast_pair_accuracy"] == 0.5       # the high_300 base is wrong


def test_preds_from_cases_new_and_old_format():
    rows = [_row("pad", H, "passes", case="a"), _row("verdict", V, "approved", case="a"),
            _row("pad", H, "passes", case="b"), _row("pad", H, "passes", case="dropped")]
    cases = {"a": {"items": {"pad": {"argmax": "passes", "probs": {"passes": 0.9, "fails": 0.1}}},
                   "verdict": {"argmax": "approved", "probs": {"approved": 0.7, "fails": 0.3}}},
             "b": {"items": {"pad": ["passes", 0.9]}}}   # old format: argmax and its probability only
    kept, preds = preds_from_cases(rows, cases)
    assert [r["meta"]["case"] for r in kept] == ["a", "a", "b"]
    assert preds[0]["probs"]["passes"] == 0.9 and preds[2] == {"argmax": "passes", "probs": None}
    r = report(kept, preds)
    assert r["item_agreement"]["overall"] == 1.0 and r["ece"]["overall"] is None   # old rows: no calibration
