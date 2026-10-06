import copy

import pytest

from scripts.build_r045_compositional_transfer import _pair
from scripts.eval_r047_executor import (
    audit,
    clause_view,
    execute_clause,
    execute_program,
    m0_gate,
    row_clause_views,
)


KINDS = ("decimal", "ledger")


def _rows():
    return (_pair("test", KINDS, 80, 47, domain_offset=2)
            + _pair("test", KINDS, 81, 47, domain_offset=2))


def _predictions(rows, flip=()):
    result = []
    for row in rows:
        projected_by_parent = row_clause_views(row)
        for view in row["paired_views"]:
            projected = [view] + projected_by_parent[view["id"]]
            for item in projected:
                gold = item["targets"]["decision"]["label"]
                if item["id"] in flip:
                    gold = "yes" if gold == "no" else "no"
                result.append({"id": item["id"], "probabilities":
                               {"no": .9 if gold == "no" else .1,
                                "yes": .9 if gold == "yes" else .1}})
    return result


def test_audit_reports_executable_clause_vector_coherence_and_direction():
    rows = _rows()
    report = audit(rows, _predictions(rows), min_clauses=2)
    assert report["rows"] == 4
    assert report["final"]["accuracy"] == 1
    assert report["final_pairs"]["direction_rate"] == 1
    assert report["clauses"]["n"] == 16
    assert report["clauses"]["balanced_accuracy"] == 1
    assert report["clauses"]["vector_accuracy"] == 1
    assert report["coherence"]["accuracy"] == 1
    assert set(report["clauses"]["kind"]) == set(KINDS)
    assert set(report["clauses"]["language"]) == {"en", "he"}


def test_independent_executor_reproduces_all_six_clause_kinds():
    row = _pair("test", ("decimal", "datetime", "policy"), 80, 47, domain_offset=2)[0]
    for view in row["paired_views"]:
        assert execute_program(view["meta"]["program"], view["meta"]["clauses"]) is view["meta"]["gold"]
        assert all(isinstance(execute_clause(clause), bool) for clause in view["meta"]["clauses"])
    row = _pair("test", ("ledger", "fraction", "conditional"), 96, 47, domain_offset=2)[0]
    for view in row["paired_views"]:
        assert execute_program(view["meta"]["program"], view["meta"]["clauses"]) is view["meta"]["gold"]


def test_clause_error_lowers_vector_balanced_accuracy_and_coherence():
    rows = _rows()
    bad = clause_view(rows[0]["paired_views"][0], 0)["id"]
    report = audit(rows, _predictions(rows, {bad}), min_clauses=2)
    assert report["clauses"]["accuracy"] == 15 / 16
    assert report["clauses"]["vector_accuracy"] == 7 / 8
    assert report["coherence"]["accuracy"] == 7 / 8


def test_audit_scores_exact_training_auxiliary_views_when_present():
    rows = _rows()
    for row in rows:
        row["auxiliary_views"] = []
        for view in row["paired_views"]:
            for index, projected in enumerate(
                    [clause_view(view, i) for i in range(len(view["meta"]["clauses"]))]):
                projected["id"] = f"{view['id']}::training-aux-{index}"
                projected["meta"]["r047_auxiliary"] = {
                    "parent_view_id": view["id"], "clause_index": index,
                    "clause_kind": view["meta"]["clauses"][index]["kind"],
                    "role": "auxiliary"}
                row["auxiliary_views"].append(projected)
    assert all("training-aux" in item["id"] for row in rows
               for items in row_clause_views(row).values() for item in items)
    report = audit(rows, _predictions(rows), min_clauses=2)
    assert report["clauses"]["balanced_accuracy"] == 1


def test_m0_gate_checks_every_required_operator():
    report = audit(_rows(), _predictions(_rows()), min_clauses=2)
    assert m0_gate(report, KINDS)["passed"]
    broken = copy.deepcopy(report)
    broken["final_pairs"]["decisive_kind"]["ledger"]["direction_rate"] = .5
    gate = m0_gate(broken, KINDS)
    assert not gate["passed"] and "ledger final direction" in gate["failures"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows, predictions: predictions.pop(), "cover every final and clause"),
    (lambda rows, predictions: predictions[0]["probabilities"].update(yes=.8), "sum to one"),
    (lambda rows, predictions: rows[0]["paired_views"][0]["targets"]["decision"].update(
        label="maybe"), "executable gold"),
])
def test_audit_fails_closed(mutation, match):
    rows = _rows()
    predictions = _predictions(rows)
    mutation(rows, predictions)
    with pytest.raises(ValueError, match=match):
        audit(rows, predictions, min_clauses=2)
