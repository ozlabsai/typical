import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.build_r045_compositional_transfer import _pair
from scripts.eval_r047_executor import (
    audit,
    causal_coverage_report,
    clause_view,
    execute_clause,
    execute_program,
    m0_gate,
    row_clause_views,
    score_causal_rows_batched,
    seal_causal_predictions,
    validate_sealed_causal_predictions,
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


def _causal_row():
    view = copy.deepcopy(_rows()[0]["paired_views"][0])
    meta = view["meta"]
    by_kind = {clause["kind"]: clause for clause in meta["clauses"]}
    meta["clauses"] = [by_kind[kind] for kind in meta["composition"]]
    pattern = [execute_clause(by_kind[kind]) for kind in meta["composition"]]
    meta.update({
        "semantic_cluster_id": "cluster-0", "semantic_view_id": "view-0",
        "arity": len(meta["composition"]),
        "option_order": ("yes_no" if next(iter(view["questions"]["decision"]["criteria"]))
                         == "yes" else "no_yes"),
        "prompt_family": "final", "prediction_role": "final", "primitive": None,
        "decisive_role": "final", "truth_hamming_weight": sum(pattern),
        "truth_pattern": pattern, "seen_edge_count": 0,
    })
    return view


def test_causal_seal_binds_metadata_and_detects_tampering():
    row = _causal_row()
    predictions = [{"id": row["id"], "probabilities": {"no": .1, "yes": .9}}]
    sealed = seal_causal_predictions([row], predictions)
    assert sealed[0]["semantic_cluster_id"] == "cluster-0"
    assert causal_coverage_report(sealed)["dimensions"]["arity"] == {"2": 1}
    validate_sealed_causal_predictions(sealed)
    sealed[0]["language"] = "he"
    with pytest.raises(ValueError, match="seal mismatch"):
        validate_sealed_causal_predictions(sealed)


def test_causal_seal_rejects_missing_semantic_binding():
    row = _causal_row()
    del row["meta"]["seen_edge_count"]
    with pytest.raises(ValueError, match="missing seen_edge_count"):
        seal_causal_predictions(
            [row], [{"id": row["id"], "probabilities": {"no": .1, "yes": .9}}])


def test_causal_batched_scorer_is_stable_across_batch_boundaries(monkeypatch):
    import scripts.train_gemma_decision_lora as trainer

    rows = [{"id": "row-a", "token": 1, "length": 2},
            {"id": "row-b", "token": 2, "length": 3},
            {"id": "row-c", "token": 1, "length": 2}]

    def fake_encode(row, tokenizer, max_length, shuffle, include_labels):
        ids = torch.tensor([row["token"]] + [3] * (row["length"] - 1))
        return ids, torch.tensor([0.0, 1.0]), ["no", "yes"]

    monkeypatch.setattr(trainer, "flatten_row", lambda row: [row])
    monkeypatch.setattr(trainer, "encode", fake_encode)

    class Model:
        def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
            logits = torch.zeros((len(input_ids), len(logits_to_keep), 4))
            for index in range(len(input_ids)):
                position = int(attention_mask[index].sum()) - 1
                offset = int((logits_to_keep == position).nonzero()[0])
                token = int(input_ids[index, 0])
                logits[index, offset, 1 if token == 1 else 0] = 2
            return SimpleNamespace(logits=logits)

    class Tokenizer:
        pad_token_id = 0

    groups = [torch.tensor([0]), torch.tensor([1])]
    together = score_causal_rows_batched(
        Model(), Tokenizer(), groups, rows, torch.device("cpu"), 10,
        max_batch_tokens=6, max_batch_rows=2)
    singly = score_causal_rows_batched(
        Model(), Tokenizer(), groups, rows, torch.device("cpu"), 10,
        max_batch_tokens=10, max_batch_rows=1)
    assert together == singly
    assert [item["id"] for item in together] == [row["id"] for row in rows]
