import itertools
import json
from collections import Counter, defaultdict

from scripts.build_r047_panel import (
    FROZEN_COUNTS,
    K3,
    PROMPT_FAMILIES,
    SEEN_K2,
    UNSEEN_K2,
    _cluster_plan,
    build,
)


MINI_COUNTS = {"unseen_k2": 1, "k3": 1, "seen_k2": 1}


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_frozen_cluster_plan_has_preregistered_counts_and_compositions():
    plan = _cluster_plan()
    assert Counter(item[0] for item in plan) == {
        "unseen_k2": 312, "k3": 320, "seen_k2": 48}
    assert len(UNSEEN_K2) == 13
    assert len(K3) == 20
    assert len(SEEN_K2) == 2
    assert set(UNSEEN_K2).isdisjoint(SEEN_K2)
    assert set(UNSEEN_K2) | set(SEEN_K2) == set(itertools.combinations(
        ("decimal", "fraction", "datetime", "ledger", "conditional", "policy"), 2))
    assert FROZEN_COUNTS == {"unseen_k2": 24, "k3": 16, "seen_k2": 24}


def test_mini_panel_is_deterministic_flat_and_has_complete_crossings(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build(first, seed=4701, counts=MINI_COUNTS)
    build(second, seed=4701, counts=MINI_COUNTS)
    assert manifest["clusters"] == {"unseen_k2": 13, "k3": 20,
                                    "seen_k2_near_transfer": 2,
                                    "causal_primary": 33}
    assert not manifest["contract"]["frozen_production"]
    for name in ("primary", "near-transfer", "legacy-smoke"):
        assert (first / f"{name}.jsonl").read_bytes() == (
            second / f"{name}.jsonl").read_bytes()

    rows = _rows(first / "primary.jsonl") + _rows(first / "near-transfer.jsonl")
    clusters, views = defaultdict(list), defaultdict(list)
    for row in rows:
        assert set(row["questions"]) == set(row["targets"]) == {"decision"}
        meta = row["meta"]
        for key in ("semantic_cluster_id", "semantic_view_id", "seen_edge_count",
                    "composition", "arity", "language", "option_order", "prompt_family",
                    "primitive", "decisive_role", "truth_hamming_weight", "truth_pattern",
                    "prediction_role"):
            assert key in meta
        clusters[meta["semantic_cluster_id"]].append(row)
        views[meta["semantic_view_id"]].append(row)
    for items in clusters.values():
        arity = items[0]["meta"]["arity"]
        assert len(items) == 2 ** arity * 2 * 2 * (arity + 1)
        assert {tuple(row["meta"]["truth_pattern"]) for row in items} == set(
            itertools.product((False, True), repeat=arity))
        assert {row["meta"]["language"] for row in items} == {"en", "he"}
        assert {row["meta"]["option_order"] for row in items} == {"yes_no", "no_yes"}
        assert len({row["meta"]["clause_prompt_family"] for row in items}) == 1
    for items in views.values():
        arity = items[0]["meta"]["arity"]
        assert Counter(row["meta"]["prediction_role"] for row in items) == {
            "final": 1, "clause": arity}
        assert len({row["state"] for row in items}) == 1
    assert {row["meta"]["clause_prompt_family"] for row in rows} == set(PROMPT_FAMILIES)


def test_legacy_slice_is_separate_and_repairs_the_policy_specific_question(tmp_path):
    manifest = build(tmp_path, counts=MINI_COUNTS)
    rows = _rows(tmp_path / "legacy-smoke.jsonl")
    assert manifest["legacy_smoke"] == {
        "seed": 47, "domain_offset": 2, "split": "test",
        "question_repair": "composition-neutral-v1", "causal_gate_eligible": False}
    assert len(rows) == 144
    assert len({row["meta"]["scenario_id"] for row in rows}) == 32
    for row in rows:
        assert row["meta"]["panel_stratum"] == "legacy_near_transfer_smoke"
        for view in row["paired_views"]:
            instruction = view["questions"]["decision"]["instructions"]
            assert "policy authorize" not in instruction
            assert view["meta"]["question_repair"] == "composition-neutral-v1"
