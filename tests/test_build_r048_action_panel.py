import copy
import hashlib
import json
from collections import defaultdict

import pytest

import scripts.build_r048_action_panel as panel_builder
from scripts.build_r048_action_panel import (
    DEFAULT_ROWS,
    FIELDS,
    PANEL_VERSION,
    PRIMITIVES,
    SPLIT_FAMILIES,
    _fact_truth,
    _model_input,
    _normalize_text,
    _partition_signature,
    _policy,
    _semantic_signature,
    build,
    validate_corpus,
    validate_split,
    write_corpus,
)
from scripts.exact_policy_executor import execute_policy
from scripts.score_exact_policy_panel import summarize
from scripts.train_gemma_decision_lora import load_examples


def test_default_corpus_is_deterministic_and_valid():
    first = {split: build(split, count, seed=48)
             for split, count in DEFAULT_ROWS.items()}
    assert build("train", 224, seed=48) == first["train"][:224]
    audit = validate_corpus(first, DEFAULT_ROWS)
    assert set(audit["splits"]) == set(DEFAULT_ROWS)
    assert audit["all_protected_fingerprints_disjoint"] is True
    for split, report in audit["splits"].items():
        assert report["rows"] == DEFAULT_ROWS[split]
        assert set(report["mode_counts"]) == {"priority", "collect"}
        assert set(report["field_size_counts"]) == {str(size) for size in range(2, 9)}
        assert report["generated_counterfactual_clusters"] == DEFAULT_ROWS[split] // 8
        assert report["unique_exact_semantic_case_clusters"] == DEFAULT_ROWS[split] // 8
        assert report["unique_complete_cluster_serializations"] == DEFAULT_ROWS[split] // 8
        if DEFAULT_ROWS[split] >= 1600:
            assert (report["unique_split_neutral_semantic_signatures"]
                    < report["generated_counterfactual_clusters"])
        assert report["semantic_endpoints"] == DEFAULT_ROWS[split] // 4
    for overlap in audit["cross_split_overlaps"].values():
        assert overlap["model_input_exact"] == 0
        assert overlap["model_input_normalized"] == 0
        assert overlap["semantic_signature"] == 0
    assert all(report["pairs_at_or_above_threshold"] == 0
               and report["maximum_similarity"] < report["preregistered_max_similarity"]
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


def test_field_primitives_have_native_operands_and_evaluators():
    rows = build("train", 224, seed=13)
    expected_operand_keys = {
        "identity_verified": {"registry_record", "submitted_record"},
        "policy_eligible": {"policy_tier", "eligible_tiers"},
        "amount_within_limit": {"requested_amount", "limit_amount", "currency"},
        "submission_timely": {"submitted_on", "deadline_on"},
        "evidence_complete": {"required_items", "present_items"},
        "region_covered": {"recorded_region", "covered_regions"},
        "consent_valid": {"review_on", "expires_on"},
        "account_active": {"account_ref", "status_by_account"},
    }
    common = {"primitive", "value_regime", "operator_composition"}
    seen = {}
    for row in rows:
        for field, fact in row["meta"]["facts"].items():
            assert set(fact) == common | expected_operand_keys[field]
            assert _fact_truth(field, fact) == row["meta"]["action_trace"][field]
            seen[field] = frozenset(set(fact) - common)
    assert set(seen) == set(FIELDS)
    assert len(set(seen.values())) == len(FIELDS)


def test_program_and_action_families_are_executable_not_nominal():
    rows = build("eval", 224, seed=18)
    families = {(row["meta"]["policy_topology"], row["meta"]["program_family"],
                 row["meta"]["action_family"]) for row in rows}
    assert families == {
        ("priority_coarse_gate", "coarse_disjunction", "single_review_gate"),
        ("collect_coarse_gate", "coarse_disjunction", "aggregate_review_set"),
        ("priority_fieldwise", "ordered_field_conditions", "first_remediation"),
        ("collect_fieldwise", "independent_field_conditions", "remediation_set"),
    }
    assert all(row["meta"]["panel_version"] == PANEL_VERSION == "1.4" for row in rows)


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
    assert {row["meta"]["topology_slice"] for row in development} == {"development"}
    assert {row["meta"]["topology_slice"] for row in evaluation} == {
        "same", "heldout",
    }
    assert {row["meta"]["policy_topology"] for row in evaluation
            if row["meta"]["topology_slice"] == "heldout"} == {
                "priority_fieldwise", "collect_fieldwise",
            }
    for row in evaluation:
        meta = row["meta"]
        if meta["topology_slice"] == "heldout":
            development_policy = _policy(meta["policy_mode"], meta["field_order"],
                                         f"{meta['policy_mode']}_coarse_gate")
            assert (_partition_signature(meta["policy"])
                    != _partition_signature(development_policy))
            assert meta["topology_transfer_scope"] == "external_policy_outcome_mapping_only"
            witness = meta["topology_separation_witness"]
            assert execute_policy(development_policy, witness["trace_a"]) == execute_policy(
                development_policy, witness["trace_b"])
            assert execute_policy(meta["policy"], witness["trace_a"]) != execute_policy(
                meta["policy"], witness["trace_b"])
            assert sum(witness["trace_a"][field] != witness["trace_b"][field]
                       for field in meta["field_order"]) == 1
    assert {row["meta"]["flipped_field"] for row in development} == set(FIELDS)
    assert {primitive for row in development
            for primitive in row["meta"]["primitive_by_field"].values()} == set(PRIMITIVES.values())
    hebrew = next(row for row in development if row["meta"]["language"] == "he")
    assert any("א" <= character <= "ת" for character in hebrew["state"])

    # This reconstruction catches the exact v1.2 defect: syntactic wrapping
    # with OR false changes the AST but not the truth partition.
    direct = _policy("priority", ["identity_verified", "policy_eligible"],
                     "priority_coarse_gate")
    identity_wrapper = copy.deepcopy(direct)
    for rule in identity_wrapper["rules"]:
        rule["when"] = {"any": [rule["when"], {"const": False}]}
    assert _partition_signature(identity_wrapper) == _partition_signature(direct)


def test_semantic_signatures_are_reconstructed_without_split_ids_or_raw_values():
    signatures = {}
    for split in DEFAULT_ROWS:
        rows = build(split, 224, seed=21)
        groups = defaultdict(list)
        for row in rows:
            groups[row["meta"]["semantic_group_id"]].append(row)
        signatures[split] = set()
        for group in groups.values():
            meta = group[0]["meta"]
            truths = {
                side: next(row["meta"]["action_trace"] for row in group
                           if row["meta"]["counterfactual_side"] == side)
                for side in (0, 1)
            }
            reconstructed = _semantic_signature(
                meta["field_order"], meta["policy_mode"], meta["flipped_field"], truths,
                meta["renderer_family"], meta["source_family"], meta["value_regime"],
                meta["operator_composition"], meta["policy"],
            )
            assert reconstructed == meta["semantic_signature"]
            signatures[split].add(reconstructed)
        allowed = SPLIT_FAMILIES[split]
        assert {row["meta"]["renderer_family"] for row in rows} == set(allowed["renderer"])
        assert {row["meta"]["source_family"] for row in rows} == set(allowed["source"])
        assert {row["meta"]["value_regime"] for row in rows} == set(allowed["value_regime"])
        assert {row["meta"]["operator_composition"] for row in rows} == set(allowed["operator"])
    assert signatures["train"].isdisjoint(signatures["cal"])
    assert signatures["train"].isdisjoint(signatures["eval"])
    assert signatures["cal"].isdisjoint(signatures["eval"])


def test_normalized_model_inputs_are_split_disjoint_without_record_ids():
    normalized = {
        split: {_normalize_text(_model_input(row)) for row in build(split, 224, seed=29)}
        for split in DEFAULT_ROWS
    }
    assert normalized["train"].isdisjoint(normalized["cal"])
    assert normalized["train"].isdisjoint(normalized["eval"])
    assert normalized["cal"].isdisjoint(normalized["eval"])


def test_corpus_validation_enforces_normalized_and_fuzzy_caps(monkeypatch):
    corpus = {split: build(split, 224, seed=31) for split in DEFAULT_ROWS}
    original_fingerprints = panel_builder._fingerprints

    def overlapping_fingerprints(rows):
        result = original_fingerprints(rows)
        result["model_input_normalized"] = {"forced-overlap"}
        return result

    monkeypatch.setattr(panel_builder, "_fingerprints", overlapping_fingerprints)
    with pytest.raises(ValueError, match="protected fingerprints"):
        validate_corpus(corpus, {split: 224 for split in DEFAULT_ROWS})
    monkeypatch.undo()

    def excessive_fuzzy(_left, _right):
        return {"pair_rate_at_or_above_threshold": 0.5, "maximum_similarity": 1.0}

    monkeypatch.setattr(panel_builder, "_fuzzy_report", excessive_fuzzy)
    with pytest.raises(ValueError, match="fuzzy model-input overlap"):
        validate_corpus(corpus, {split: 224 for split in DEFAULT_ROWS})


def test_validation_rejects_exact_complete_cluster_duplicates(monkeypatch):
    monkeypatch.setattr(panel_builder, "_exact_cluster_signature",
                        lambda *args, **kwargs: "same")
    rows = build("train", 112, seed=31)
    with pytest.raises(ValueError, match="duplicate exact semantic case clusters"):
        validate_split(rows, "train", 112)


def test_validation_rejects_duplicate_complete_cluster_serializations(monkeypatch):
    rows = build("train", 112, seed=32)
    monkeypatch.setattr(panel_builder, "_complete_cluster_serialization_signature",
                        lambda _group: "same")
    with pytest.raises(ValueError, match="duplicate complete cluster serializations"):
        validate_split(rows, "train", 112)


def test_scorer_integration_populates_endpoints_clusters_and_topology_slices():
    rows = build("eval", 32, seed=4)
    predictions = []
    for row in rows:
        probabilities = {
            qid: ({"no": 0.0, "yes": 1.0} if target["label"] == "yes"
                  else {"no": 1.0, "yes": 0.0})
            for qid, target in row["targets"].items()
        }
        predictions.append({"id": row["id"], "probabilities": probabilities})
    report = summarize(rows, predictions)
    assert report["semantic_endpoints"]["n"] == 8
    assert report["independent_clusters"]["n"] == 4
    assert set(report["slices"]["topology_slice"]) == {"same", "heldout"}
    assert set(report["slices"]["policy_topology"]) == {
        "priority_coarse_gate", "priority_fieldwise",
    }
    assert report["rendered"]["exact_outcome_accuracy"] == 1.0


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
        (lambda row: row["meta"].__setitem__("program_family", "nominal_only"),
         "program/action family"),
        (lambda row: row["questions"]["q00"].__setitem__(
            "criteria", dict(reversed(row["questions"]["q00"]["criteria"].items()))),
         "semantic question"),
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
        persisted = [json.loads(line) for line in payload.decode().splitlines()]
        assert persisted == build(split, count, seed=17)
        for row in persisted:
            expected = (["no", "yes"] if row["meta"]["option_order"] == "no_yes"
                        else ["yes", "no"])
            assert all(list(question["criteria"]) == expected
                       for question in row["questions"].values())
        assert (manifest["audit"]["splits"][split]
                ["unique_exact_semantic_case_clusters"] == count // 8)
        assert (manifest["audit"]["splits"][split]
                ["unique_complete_cluster_serializations"] == count // 8)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_corpus(tmp_path, seed=17, row_counts=counts)


def test_validation_detects_inconsistent_split_label():
    rows = build("cal", 16, seed=4)
    broken = copy.deepcopy(rows)
    broken[0]["meta"]["split"] = "eval"
    with pytest.raises(ValueError, match="incomplete panel metadata"):
        validate_split(broken, "cal", 16)
