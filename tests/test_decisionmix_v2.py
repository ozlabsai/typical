"""PLAN7 Track D: DecisionMix v2 schema/invariant tests. Runs the real generators at --limit scale
(small, fast, no model needed) and checks the memo's required-metadata schema, counterfactual-group
invariants, holdout absence from train, the JevBench leak check, and soft-target normalisation.

data_u needs network (HF hub) for UNLI/AmbiEnt/chaos-mnli-ambiguity; skipped if unavailable (offline CI).
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import decisionmix_v2 as dm  # noqa: E402

JEVBENCH = "/tmp/jevbench/datasets/public"
REQUIRED_META = ["decision_type", "task_family", "workflow_family", "rubric_family", "rubric_style",
                  "rule_depth", "exception_depth", "candidate_count", "gold_present", "catch_all_present",
                  "requires_temporal", "requires_numeric", "requires_probability", "requires_tradeoff",
                  "gold_source", "soft_target", "source_dataset", "fam_bucket"]


def wh_args(tmp_path, limit=25):
    return argparse.Namespace(out_wh=str(tmp_path / "data_wh"), limit=limit, seed=0, jevbench=JEVBENCH)


def load(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wh")
    dm.build_wh(wh_args(tmp))
    out = tmp / "data_wh"
    rows = {"train": load(out / "train.jsonl"), "val": load(out / "val.jsonl")}
    for f in (out / "eval").glob("*.jsonl"):
        rows[f.stem] = load(f)
    rows["manifest"] = json.load(open(out / "manifest.json"))
    return rows


def all_wh_rows(wh):
    return [r for k, rs in wh.items() if k != "manifest" for r in rs]


# wh_rubric_flip/wh_rubric_shuffled are DERIVED views over heldout_pool (workflow_corpus.flip_pairs /
# shuffled_rubric copy a source row's meta, including its rubric_group id, onto a new probe row) --
# they intentionally duplicate/relabel rows rather than forming fresh counterfactual units, so the
# "one group = 2-3 rows, same state+candidates" invariant is checked on the raw categories only.
RAW_KEYS = {"train", "val", "wh_heldout_family", "wh_heldout_style", "wh_heldout_grammar", "wh_level7"}


def raw_wh_rows(wh):
    return [r for k, rs in wh.items() if k in RAW_KEYS for r in rs]


def test_wh_schema(wh):
    rows = all_wh_rows(wh)
    assert rows
    for r in rows:
        assert set(["state", "query", "candidates", "target", "p_null", "task", "label", "meta"]) <= r.keys()
        assert len(r["candidates"]) == len(r["target"]) == r["meta"]["candidate_count"]
        for key in REQUIRED_META:
            assert key in r["meta"], key
        assert r["meta"]["fam_bucket"] == "W"
        assert sum(r["target"]) == pytest.approx(1.0, abs=1e-6)


def test_wh_rubric_groups_share_state_and_candidates_but_not_always_gold(wh):
    by_group = defaultdict(list)
    for r in raw_wh_rows(wh):
        gid = r["meta"].get("rubric_group")
        if gid:
            by_group[gid].append(r)
    assert by_group, "no rubric groups found"
    n_differ = 0
    for gid, members in by_group.items():
        assert len(members) >= 2
        states = {m["state"] for m in members}
        # candidate ORDER may vary per row (position bias mitigation); the memo's invariant is the same
        # candidate SET, and each row's own target still aligns positionally with its own candidates list.
        cand_sets = {frozenset(m["candidates"]) for m in members}
        assert len(states) == 1, f"group {gid} has divergent states"
        assert len(cand_sets) == 1, f"group {gid} has divergent candidate sets"
        if len({m["label"] for m in members}) > 1:
            n_differ += 1
    assert n_differ > 0, "no group had differing golds across rubric variants"


def test_wh_every_train_row_is_in_a_rubric_group(wh):
    assert wh["train"]
    for r in wh["train"]:
        assert r["meta"].get("rubric_group"), "every WH training unit must be a counterfactual rubric group"


def test_wh_at_least_one_row_per_level_1_to_7(wh):
    levels = {r["meta"]["rule_depth"] for r in all_wh_rows(wh)}
    assert levels >= set(range(1, 8)), f"missing levels: {set(range(1, 8)) - levels}"
    # level 7 is eval-only, never in train/val
    assert not any(r["meta"]["rule_depth"] == 7 for r in wh["train"] + wh["val"])


def test_wh_holdouts_absent_from_train(wh):
    m = wh["manifest"]
    train_families = {r["meta"]["workflow_family"] for r in wh["train"]}
    assert not (train_families & set(m["held_out_families"])), "held-out family leaked into train"

    train_styles = {r["meta"]["rubric_style"] for r in wh["train"]}
    assert not (train_styles & set(m["held_out_styles"])), "held-out rubric style leaked into train"

    for level_str, domain in m["held_out_grammar"].items():
        level = int(level_str)
        hit = [r for r in wh["train"] if r["meta"]["rule_depth"] == level and r["meta"]["workflow_family"] == domain]
        assert not hit, f"held-out grammar (L{level}, {domain}) leaked into train"


def test_wh_null_rows_have_uniform_target_and_no_gold(wh):
    null_rows = [r for r in wh["train"] if not r["meta"]["gold_present"]]
    assert null_rows, "expected some true-null (~10%) rows in train"
    for r in null_rows:
        assert r["label"] == -1 and r["p_null"] == pytest.approx(1.0)
        k = len(r["candidates"])
        assert all(t == pytest.approx(1.0 / k) for t in r["target"])


def test_wh_jevbench_leak_check_ran(wh):
    # leak_check() itself asserts 0 hits internally; a nonzero states-checked count here confirms
    # the real JevBench corpus (not the "file missing" skip path) was actually used.
    assert wh["manifest"]["jevbench_states_checked"] and wh["manifest"]["jevbench_states_checked"] > 0


def test_score_case_renders_selected_fields_order_and_complete_mapping():
    rows = dm.score_case("procurement_request", random.Random(4), False, 3)
    assert {row["meta"]["score_rule"] for row in rows} == {"count", "worst_first"}
    for row in rows:
        fields = row["meta"]["score_fields"]
        assert len(fields) == 4 and len(set(fields)) == 4
        for position, field in enumerate(fields, 1):
            assert f"{position}) {dm.DOMAINS['procurement_request']['fields'][field][0]}" in row["query"]
        assert all(level in row["query"] for level in row["candidates"])


def test_l4_precedence_survives_candidate_reordering(monkeypatch):
    monkeypatch.setattr(dm, "SCORE_FRAC", 0)
    rows = dm.l4_builder("procurement_request", random.Random(8))()
    assert rows
    for row in rows:
        criteria = row["question"]["criteria"]
        ranked = [description for label, description in criteria.items()
                  if label != dm.DOMAINS["procurement_request"]["routes"][3]]
        assert len(ranked) == 3
        assert all(f"priority {rank}" in " ".join(ranked) for rank in (1, 2, 3))


def test_temporal_rendering_matches_elapsed_gold_across_timezones():
    for seed in range(40):
        rows = dm.l7_temporal_builder(random.Random(seed))()
        for row in rows:
            times = re.findall(r"(\d\d):00 \(UTC([+-]\d+)\) on day (\d+)", row["state"])
            assert len(times) == 2
            utc = sorted((int(day) - 1) * 24 + int(hour) - int(offset)
                         for hour, offset, day in times)
            assert utc[1] - utc[0] == row["meta"]["elapsed_h"]
            grace = 2 if row["meta"]["rubric_family"].endswith("grace") else 0
            expected = row["meta"]["elapsed_h"] <= row["meta"]["deadline_h"] + grace
            assert row["candidates"][row["label"]] == ("true" if expected else "false")


def _hf_reachable():
    try:
        from huggingface_hub import HfApi
        HfApi().dataset_info("Zhengping/UNLI")
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def u(tmp_path_factory):
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")
    tmp = tmp_path_factory.mktemp("u")
    args = argparse.Namespace(out_u=str(tmp / "data_u"), limit=20, seed=0, jevbench=JEVBENCH)
    dm.build_u(args)
    out = tmp / "data_u"
    rows = {"train": load(out / "train.jsonl"), "val": load(out / "val.jsonl")}
    for f in (out / "eval").glob("*.jsonl"):
        rows[f.stem] = load(f)
    rows["manifest"] = json.load(open(out / "manifest.json"))
    return rows


def test_u_schema_and_soft_targets_sum_to_one(u):
    rows = [r for k, rs in u.items() if k != "manifest" for r in rs]
    assert rows
    for r in rows:
        for key in REQUIRED_META:
            assert key in r["meta"], key
        assert r["meta"]["fam_bucket"] == "U"
        assert r["meta"]["soft_target"] is True
        assert sum(r["target"]) == pytest.approx(1.0, abs=1e-6)
        assert r["meta"]["gold_source"] in ("human", "programmatic", "teacher")


def test_u_has_real_and_synthetic_sources(u):
    sources = {r["meta"]["source_dataset"] for r in u["train"]}
    assert any(s.startswith("synthetic:") for s in sources)
    assert any(not s.startswith("synthetic:") for s in sources), "expected a real annotator-grounded source too"


def test_u_never_trains_on_the_datasets_already_used_as_base_evals(u):
    # pcdm/data.py's "chaos_mnli" eval consumes the entire metaeval/chaos-mnli-ambiguity file; data_u must
    # only use it in its own eval file, never in train/val.
    train_val = u["train"] + u["val"]
    assert not any(r["meta"]["source_dataset"] == "metaeval/chaos-mnli-ambiguity" for r in train_val)
