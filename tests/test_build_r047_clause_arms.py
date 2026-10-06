import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_r045_compositional_transfer import _evaluate, _pair, execute
from scripts.build_r047_clause_arms import build


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fixture(tmp_path, composition=("decimal", "policy")):
    source = tmp_path / "source"
    source.mkdir()
    rows = _pair("train", composition, 72, seed=45, case_index=0)
    rows.extend(_pair("train", ("fraction",), 12, seed=45, case_index=0))
    payload = _jsonl(rows)
    path = source / "arm-b.jsonl"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest(), rows


def _without_auxiliary(row):
    result = json.loads(json.dumps(row))
    result.pop("auxiliary_views", None)
    return result


def test_builds_deterministic_matched_arms_with_executable_auxiliary_gold(tmp_path):
    source, digest, source_rows = _fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build(source, first, expected_input_sha256=digest)
    build(source, second, expected_input_sha256=digest)

    for filename in ("control.jsonl", "clause.jsonl", "m0-treatment.jsonl"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    control, clause = _rows(first / "control.jsonl"), _rows(first / "clause.jsonl")
    assert [_without_auxiliary(row) for row in control] == source_rows
    assert [_without_auxiliary(row) for row in clause] == source_rows
    assert [row["id"] for row in control] == [row["id"] for row in clause]

    # Only the two language twins of the K=2 composition are augmented.
    assert manifest["arms"]["control"]["augmented_rows"] == 2
    assert manifest["arms"]["control"]["auxiliary_views"] == 8
    assert manifest["arms"]["control"]["auxiliary_labels"] == {"no": 4, "yes": 4}
    assert manifest["arms"]["clause"]["auxiliary_labels"] == {"no": 2, "yes": 6}
    assert manifest["arms"]["clause"]["auxiliary_languages"] == {"en": 4, "he": 4}
    assert manifest["arms"]["clause"]["auxiliary_clause_kinds"] == {
        "decimal": 4, "policy": 4}
    assert manifest["contract"]["token_budget_requires_preflight"]
    assert manifest["outputs"]["control"]["sha256"]
    m0 = _rows(first / "m0-treatment.jsonl")
    assert m0 == [row for row in clause if row.get("auxiliary_views")]
    assert manifest["m0_treatment"]["rows"] == 2
    assert manifest["m0_treatment"]["canonical_views"] == 4
    assert manifest["m0_treatment"]["auxiliary_views"] == 8
    assert manifest["m0_treatment"]["effective_presentations"] == 12
    assert manifest["m0_treatment"]["source_identifier_field"] == "id"
    assert manifest["m0_treatment"]["sorted_source_ids_sha256"]

    for control_row, clause_row in zip(control, clause):
        control_aux = control_row.get("auxiliary_views", [])
        clause_aux = clause_row.get("auxiliary_views", [])
        assert len(control_aux) == len(clause_aux)
        if not control_aux:
            continue
        cursor = 0
        for control_view, clause_view in zip(
                control_row.get("paired_views", []), clause_row.get("paired_views", [])):
            for position, executable_clause in enumerate(clause_view["meta"]["clauses"]):
                control_item, clause_item = control_aux[cursor], clause_aux[cursor]
                cursor += 1
                assert control_item["id"] == clause_item["id"]
                assert control_item["state"] == clause_item["state"] == control_view["state"]
                cq = next(iter(control_item["questions"].values()))
                tq = next(iter(clause_item["questions"].values()))
                source_q = next(iter(control_view["questions"].values()))
                assert cq["criteria"] == tq["criteria"] == source_q["criteria"]
                final = execute(control_view["meta"]["program"], control_view["meta"]["clauses"])
                assert next(iter(control_item["targets"].values()))["label"] == (
                    "yes" if final else "no")
                clause_gold = _evaluate(executable_clause)
                assert next(iter(clause_item["targets"].values()))["label"] == (
                    "yes" if clause_gold else "no")
                aux_meta = clause_item["meta"]["r047_auxiliary"]
                assert aux_meta["parent_view_id"] == clause_view["id"]
                assert aux_meta["clause_index"] == position
                assert aux_meta["clause_kind"] == executable_clause["kind"]
                assert len(aux_meta["clause_sha256"]) == 64
                assert aux_meta["role"] == "auxiliary"
                assert aux_meta["arm"] == "treatment"
        assert cursor == len(control_aux)


def test_auxiliary_generation_is_generic_over_composition_arity(tmp_path):
    source, digest, _ = _fixture(tmp_path, ("decimal", "datetime", "policy"))
    manifest = build(source, tmp_path / "out", expected_input_sha256=digest)
    assert manifest["arms"]["clause"]["augmented_arities"] == {"3": 2}
    assert manifest["arms"]["clause"]["auxiliary_views"] == 12
    assert manifest["arms"]["clause"]["auxiliary_clause_kinds"] == {
        "datetime": 4, "decimal": 4, "policy": 4}


def test_rejects_wrong_frozen_hash_and_existing_auxiliary_data(tmp_path):
    source, digest, rows = _fixture(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        build(source, tmp_path / "wrong", expected_input_sha256="0" * 64)

    rows[0]["auxiliary_views"] = []
    payload = _jsonl(rows)
    source.write_bytes(payload)
    with pytest.raises(ValueError, match="already contain"):
        build(source, tmp_path / "duplicate",
              expected_input_sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("filename,state_text", [
    ("JevBench-arm.jsonl", None),
    ("arm-b.jsonl", "copied WorkflowEvals prompt"),
])
def test_rejects_benchmark_source_path_or_content(tmp_path, filename, state_text):
    source, _, rows = _fixture(tmp_path)
    target = source.with_name(filename)
    if state_text:
        rows[0]["paired_views"][0]["state"] = state_text
    payload = _jsonl(rows)
    target.write_bytes(payload)
    with pytest.raises(ValueError, match="prohibited benchmark source"):
        build(target, tmp_path / "out",
              expected_input_sha256=hashlib.sha256(payload).hexdigest())
