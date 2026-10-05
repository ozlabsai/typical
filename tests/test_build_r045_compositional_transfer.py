import json
from collections import Counter
from decimal import Decimal

import pytest

from scripts.build_r045_compositional_transfer import (
    COMPOSITIONS,
    SPLITS,
    _evaluate,
    build,
    execute,
)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_build_is_deterministic_and_emits_trainer_rows_with_executable_gold(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build(first, seed=45)
    build(second, seed=45)
    assert manifest["experiment"] == "r045_compositional_transfer"
    assert manifest["source_inputs"] == []
    assert len(manifest["builder_sha256"]) == 64
    for split in SPLITS:
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
        rows = _rows(first / f"{split}.jsonl")
        assert len(rows) == manifest["splits"][split]["rows"]
        assert manifest["splits"][split]["sha256"]
        for row in rows:
            assert row["pair_mode"] == "counterfactual"
            assert len(row["paired_views"]) == 2
            left, right = row["paired_views"]
            assert row["source_ids"] == [left["meta"]["source_id"], right["meta"]["source_id"]]
            assert left["questions"] == right["questions"]
            assert {left["targets"]["decision"]["label"], right["targets"]["decision"]["label"]} == {"yes", "no"}
            for view in (left, right):
                meta = view["meta"]
                assert view["questions"]["decision"]["type"] == "choice"
                assert set(view["questions"]["decision"]["criteria"]) == {"yes", "no"}
                assert execute(meta["program"], meta["clauses"]) is meta["gold"]


def test_splits_hold_out_compositions_values_thresholds_entities_and_renderers(tmp_path):
    manifest = build(tmp_path, seed=45)
    observed = {}
    for split in SPLITS:
        rows = _rows(tmp_path / f"{split}.jsonl")
        seen = {"compositions": set(), "entities": set(), "renderers": set(), "decimal": [],
                "fraction": [], "years": set(), "priorities": set()}
        for row in rows:
            meta = row["meta"]
            seen["compositions"].add(tuple(meta["composition"]))
            seen["entities"].add(meta["entity_vocab"])
            seen["renderers"].add(meta["renderer"])
            for view in row["paired_views"]:
                for clause in view["meta"]["clauses"]:
                    if clause["kind"] in ("decimal", "ledger"):
                        seen["decimal"].append(Decimal(clause["threshold"]))
                    elif clause["kind"] in ("fraction", "conditional"):
                        seen["fraction"].append(tuple(clause["threshold"]))
                    elif clause["kind"] == "datetime":
                        seen["years"].add(int(clause["local_time"][:4]))
                    elif clause["kind"] == "policy":
                        seen["priorities"].add(clause["rules"][1]["priority"])
        assert seen["compositions"] == set(COMPOSITIONS[split])
        assert seen["entities"] == set(SPLITS[split]["entities"])
        assert seen["renderers"] == set(SPLITS[split]["renderers"])
        assert manifest["leakage_contract"]["disjoint_compositions"]
        observed[split] = seen

    for left, right in (("train", "cal"), ("train", "test"), ("cal", "test")):
        assert observed[left]["compositions"].isdisjoint(observed[right]["compositions"])
        assert observed[left]["entities"].isdisjoint(observed[right]["entities"])
        assert observed[left]["renderers"].isdisjoint(observed[right]["renderers"])
        assert max(observed[left]["decimal"]) < min(observed[right]["decimal"])
        assert observed[left]["years"].isdisjoint(observed[right]["years"])
        assert observed[left]["priorities"].isdisjoint(observed[right]["priorities"])
    assert {observed[split]["fraction"][0] for split in SPLITS} == {
        (1, 4), (1, 2), (3, 4)}
    for left, right in (("train", "cal"), ("train", "test"), ("cal", "test")):
        assert set(observed[left]["fraction"]).isdisjoint(observed[right]["fraction"])


def test_computational_counterfactual_and_language_twins_stay_in_one_split(tmp_path):
    build(tmp_path, seed=12)
    all_rows = {split: _rows(tmp_path / f"{split}.jsonl") for split in SPLITS}
    owner = {}
    for split, rows in all_rows.items():
        for row in rows:
            source = row["meta"]["source_id"]
            owner.setdefault(source, set()).add(split)
            assert row["meta"]["language"] in ("en", "he")
            left, right = row["paired_views"]
            a, b = left["meta"]["clauses"], right["meta"]["clauses"]
            changed = [i for i, (one, two) in enumerate(zip(a, b)) if one != two]
            assert len(changed) == 1
            assert changed[0] == row["meta"]["composition"].index(left["meta"]["decisive_kind"])
    assert all(len(splits) == 1 for splits in owner.values())
    twins = Counter((split, row["meta"]["twin_id"]) for split, rows in all_rows.items() for row in rows)
    assert set(twins.values()) == {2}
    assert (set(_evaluate(c) for rows in all_rows.values() for row in rows
                for view in row["paired_views"] for c in view["meta"]["clauses"])
            == {True, False})


@pytest.mark.parametrize("path,content", [
    ("/tmp/JevBench/train.jsonl", None),
    (None, "copied WorkflowEvals prompt and answer"),
    ("/tmp/workflow_evals.jsonl", None),
])
def test_rejects_benchmark_source_references(tmp_path, path, content):
    with pytest.raises(ValueError, match="prohibited benchmark source"):
        build(tmp_path / "out", source_paths=[path] if path else (),
              source_contents=[content] if content else ())
