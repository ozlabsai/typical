import copy

import pytest

from scripts.build_r045_compositional_transfer import _pair
from scripts.eval_r045_transfer import audit


def _panel():
    return (_pair("test", ("decimal", "ledger"), 80, 45)
            + _pair("test", ("datetime", "policy"), 81, 45))


def _predictions(rows, flip=()):
    result = []
    for row in rows:
        for view in row["paired_views"]:
            gold = view["targets"]["decision"]["label"]
            if view["id"] in flip:
                gold = "no" if gold == "yes" else "yes"
            result.append({"id": view["id"], "probabilities":
                           {"no": .9 if gold == "no" else .1, "yes": .9 if gold == "yes" else .1}})
    return result


def test_audit_reports_full_pair_language_and_source_metrics():
    rows = _panel()
    report = audit(rows, _predictions(rows))
    assert report["presentations"] == 8 and report["accuracy"] == 1
    assert report["pairs"] == 4 and report["pair_both_correct"] == 1
    assert report["logical_sources"] == 2 and report["source_all_four_correct"] == 1
    assert set(report["language"]) == {"en", "he"}
    assert set(report["composition"]) == {"datetime+policy", "decimal+ledger"}
    assert set(report["option_order"]) == {"False", "True"}


def test_one_error_lowers_the_right_full_denominators():
    rows = _panel()
    bad_id = rows[0]["paired_views"][0]["id"]
    report = audit(rows, _predictions(rows, {bad_id}))
    assert report["accuracy"] == 7 / 8
    assert report["pair_both_correct"] == 3 / 4
    assert report["source_all_four_correct"] == 1 / 2


@pytest.mark.parametrize("mutation,match", [
    (lambda rows, predictions: predictions.pop(), "cover every presentation"),
    (lambda rows, predictions: predictions[0]["probabilities"].update(yes=.8), "sum to one"),
    (lambda rows, predictions: rows[0]["paired_views"][0]["targets"]["decision"].update(label="maybe"),
     "executable gold"),
])
def test_audit_fails_closed(mutation, match):
    rows = _panel()
    predictions = _predictions(rows)
    mutation(rows, predictions)
    with pytest.raises(ValueError, match=match):
        audit(rows, predictions)


def test_duplicate_prediction_ids_are_rejected():
    rows = _panel()
    predictions = _predictions(rows)
    predictions[1] = copy.deepcopy(predictions[0])
    with pytest.raises(ValueError, match="unique"):
        audit(rows, predictions)
