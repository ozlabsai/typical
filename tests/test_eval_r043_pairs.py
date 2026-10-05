import pytest

from scripts.build_r043_causal_pairs import FAMILIES, _source
from scripts.eval_r043_pairs import audit


def panel():
    return [_source("eval", family, index, 104772, 1_000_000, index)
            for family in FAMILIES for index in range(32)]


def predictions(rows, failed_sources=()):
    failed_sources = set(failed_sources)
    result = []
    for pair in rows:
        for view in pair["paired_views"]:
            yes = view["targets"]["decision"]["label"] == "yes"
            if view["meta"]["source_id"] in failed_sources:
                yes = not yes
            result.append({"id": view["id"], "probabilities": {
                "yes": .9 if yes else .1, "no": .1 if yes else .9}})
    return result


def test_r043_pair_audit_scores_semantics_actions_and_diagnostics():
    rows = panel()
    report = audit(rows, predictions(rows))
    assert report["pairs"] == 256
    assert report["presentations"] == 512
    assert report["accuracy"] == report["exact_action_accuracy"] == 1
    assert report["twin_both_correct"] == 1
    assert report["option_order_accuracy_delta"] == 0
    assert set(report["families"]) == set(FAMILIES)
    assert all(values["pairs"] == 32 for values in report["families"].values())


def test_r043_pair_audit_rejects_any_missing_prediction():
    rows = panel()
    with pytest.raises(ValueError, match="cover every presentation"):
        audit(rows, predictions(rows)[:-1])
