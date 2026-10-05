import copy

from scripts.build_r045_compositional_transfer import _pair
from scripts.compare_r045_transfer import compare


def _rows():
    return (_pair("test", ("decimal", "ledger"), 80, 45)
            + _pair("test", ("decimal", "ledger"), 81, 45))


def _predictions(rows):
    return [{"id": view["id"], "probabilities": {
        "yes": .9 if view["targets"]["decision"]["label"] == "yes" else .1,
        "no": .9 if view["targets"]["decision"]["label"] == "no" else .1}}
        for row in rows for view in row["paired_views"]]


def test_compare_reports_paired_scenario_and_secondary_deltas():
    rows = _rows()
    right = _predictions(rows)
    left = copy.deepcopy(right)
    left[0]["probabilities"] = {"yes": left[0]["probabilities"]["no"],
                                  "no": left[0]["probabilities"]["yes"]}
    result = compare(rows, left, right, "base", "candidate")
    assert result["paired_scenarios"] == {"gains": 1, "losses": 0, "ties": 0,
                                           "one_sided_p": .5}
    assert result["deltas"] == {"successful_scenarios": 1, "complete_sources": 1,
                                 "correct_pairs": 1, "correct_presentations": 1,
                                 "english_correct": 1, "hebrew_correct": 0}
    assert result["composition_scenarios"]["decimal+ledger"]["delta"] == 1


def test_one_sided_exact_sign_tail():
    rows = []
    for scenario in range(5):
        index = 80 + scenario * 2
        rows.extend(_pair("test", ("decimal", "ledger"), index, 45, scenario * 2))
        rows.extend(_pair("test", ("decimal", "ledger"), index + 1, 45, scenario * 2 + 1))
    right = _predictions(rows)
    left = copy.deepcopy(right)
    for item in left[::8]:
        item["probabilities"] = {"yes": item["probabilities"]["no"],
                                  "no": item["probabilities"]["yes"]}
    result = compare(rows, left, right)
    assert result["paired_scenarios"]["gains"] == 5
    assert result["paired_scenarios"]["one_sided_p"] == 1 / 32
