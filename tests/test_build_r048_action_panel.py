import copy
import hashlib
import json
from collections import defaultdict

import pytest

from scripts.build_r048_action_panel import (
    DEFAULT_ROWS,
    FIELDS,
    PRIMITIVES,
    build,
    validate_corpus,
    validate_split,
    write_corpus,
)
from scripts.exact_policy_executor import execute_policy
from scripts.train_gemma_decision_lora import load_examples


def test_default_corpus_is_deterministic_and_valid():
    first = {split: build(split, count, seed=48)
             for split, count in DEFAULT_ROWS.items()}
    second = {split: build(split, count, seed=48)
              for split, count in DEFAULT_ROWS.items()}
    assert first == second
    audit = validate_corpus(first, DEFAULT_ROWS)
    assert set(audit["splits"]) == set(DEFAULT_ROWS)
    assert audit["all_protected_fingerprints_disjoint"] is True
    for split, report in audit["splits"].items():
        assert report["rows"] == DEFAULT_ROWS[split]
        assert set(report["mode_counts"]) == {"priority", "collect"}
        assert set(report["field_size_counts"]) == {str(size) for size in range(2, 9)}
        assert report["independent_latent_clusters"] == DEFAULT_ROWS[split] // 8
        assert report["semantic_endpoints"] == DEFAULT_ROWS[split] // 4
    for overlap in audit["cross_split_overlaps"].values():
        assert overlap["model_input_exact"] == 0
        assert overlap["latent_signature"] == 0
    assert all(report["comparisons"] > 0
               for report in audit["fuzzy_model_input"].values())


def test_rows_satisfy_exact_policy_contract_and_cover_seed_outcomes():
    rows = build("train", 160, seed=3)
    shapes = set()
    for row in rows:
        meta = row["meta"]
        assert meta["panel"] == "exact_policy_v1"
        assert meta["field_order"] == list(meta["policy"]["fields"])
        assert meta["qid_to_field"] == {
            f"q{index:02d}": field
            for index, field in enumerate(meta["field_order"])
        }
        assert execute_policy(meta["policy"], meta["action_trace"]).as_dict() == meta["gold_outcome"]
        shapes.add((meta["policy"]["mode"], len(meta["gold_outcome"]["actions"]),
                    meta["gold_outcome"]["primary"] is None))
    assert ("collect", 0, True) in shapes
    assert ("collect", 1, True) in shapes
    assert ("collect", 2, False) in shapes


def test_languages_and_option_orders_are_semantic_not_positional():
    rows = build("eval", 64, seed=7)
    assert {(row["meta"]["language"], row["meta"]["option_order"])
            for row in rows} == {
                ("en", "no_yes"), ("en", "yes_no"),
                ("he", "no_yes"), ("he", "yes_no"),
            }
    for row in rows:
        expected = (["no", "yes"] if row["meta"]["option_order"] == "no_yes"
                    else ["yes", "no"])
        assert all(list(question["criteria"]) == expected
                   for question in row["questions"].values())


def test_clusters_are_one_fact_causal_twins_with_all_eight_views():
    groups = defaultdict(list)
    for row in build("train", 224, seed=9):
        groups[row["meta"]["counterfactual_cluster_id"]].append(row)
    assert len(groups) == 28
    for group in groups.values():
        assert len(group) == 8
        assert len({row["meta"]["latent_signature"] for row in group}) == 1
        variants = {(row["meta"]["counterfactual_side"], row["meta"]["language"],
                     row["meta"]["option_order"]) for row in group}
        assert len(variants) == 8
        for language in ("en", "he"):
            for order in ("no_yes", "yes_no"):
                pair = sorted((row for row in group
                               if row["meta"]["language"] == language
                               and row["meta"]["option_order"] == order),
                              key=lambda row: row["meta"]["counterfactual_side"])
                left, right = (row["meta"] for row in pair)
                changed = {field for field in left["facts"]
                           if left["facts"][field] != right["facts"][field]}
                assert changed == {left["flipped_field"]}
                assert left["gold_outcome"] != right["gold_outcome"]


def test_primitives_roles_and_external_policy_topologies_are_real_holdouts():
    development = build("train", 224, seed=48)
    evaluation = build("eval", 224, seed=48)
    assert {row["meta"]["external_policy_topology"] for row in development} == {"direct"}
    assert {row["meta"]["external_policy_topology_slice"] for row in evaluation} == {
        "same_topology", "heldout_topology",
    }
    assert {row["meta"]["external_policy_topology"] for row in evaluation
            if row["meta"]["external_policy_topology_slice"] == "heldout_topology"} == {
                "guarded_any", "demorgan",
            }
    assert {row["meta"]["flipped_field"] for row in development} == set(FIELDS)
    assert {primitive for row in development
            for primitive in row["meta"]["primitive_by_field"].values()} == set(PRIMITIVES.values())
    hebrew = next(row for row in development if row["meta"]["language"] == "he")
    assert any("א" <= character <= "ת" for character in hebrew["state"])


def test_generated_rows_load_as_grouped_trainer_examples(tmp_path):
    path = tmp_path / "panel.jsonl"
    rows = build("train", 16, seed=11)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    examples = load_examples([path])
    assert len(examples) == len(rows)
    assert all(example["action_kind"] == "exact_policy_v1" for example in examples)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["meta"].__setitem__("panel", "action_v1"), "panel metadata"),
        (lambda row: row["meta"]["action_trace"].__setitem__("risk", False),
         "action trace mismatch"),
        (lambda row: row["meta"]["gold_outcome"].__setitem__(
            "primary", {"name": "impossible", "payload": {}}),
         "gold outcome mismatch"),
        (lambda row: row["meta"]["qid_to_field"].__setitem__("q00", "unknown"),
         "qid_to_field"),
    ],
)
def test_validation_fails_closed_on_tampering(mutation, message):
    rows = build("train", 16, seed=5)
    mutation(rows[0])
    with pytest.raises(ValueError, match=message):
        validate_split(rows, "train", 16)


def test_write_corpus_emits_bound_hashes_and_refuses_overwrite(tmp_path):
    counts = {"train": 224, "cal": 224, "eval": 224}
    manifest = write_corpus(tmp_path, seed=17, row_counts=counts)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    for split, count in counts.items():
        payload = (tmp_path / f"{split}.jsonl").read_bytes()
        assert manifest["outputs"][split] == {
            "rows": count,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_corpus(tmp_path, seed=17, row_counts=counts)


def test_validation_detects_inconsistent_split_label():
    rows = build("cal", 16, seed=4)
    broken = copy.deepcopy(rows)
    broken[0]["meta"]["split"] = "eval"
    with pytest.raises(ValueError, match="incomplete panel metadata"):
        validate_split(broken, "cal", 16)
