import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts import build_r048_action_panel as r048
from scripts import build_r050_headroom as headroom
from scripts.score_exact_policy_panel import summarize
from scripts.train_gemma_decision_lora import load_examples


EXPECTED = headroom.FROZEN_CORE_MANIFEST_SHA256


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    path = tmp_path_factory.mktemp("r050-core")
    r048.write_corpus(path, 48)
    assert headroom._sha((path / "manifest.json").read_bytes()) == EXPECTED
    return path


@pytest.fixture(scope="module")
def prepared(core):
    corpus, payloads, _, _ = headroom._load_core(core, EXPECTED)
    expanded = r048.build("eval", headroom.EXPANDED_EVAL_ROWS, 48)
    with pytest.MonkeyPatch.context() as patch:
        # The real deterministic expansion above is reused, not approximated.
        patch.setattr(r048, "build", lambda split, count, seed: expanded)
        rows, manifest = headroom.build_headroom(core, EXPECTED)
    return {"core": core, "corpus": corpus, "payloads": payloads, "expanded": expanded,
            "candidates": expanded[r048.DEFAULT_ROWS["eval"]:], "rows": rows, "manifest": manifest}


def _selection(prepared):
    return prepared["manifest"]["construction"]["selection"]


def _mock_build(monkeypatch, prepared):
    monkeypatch.setattr(headroom, "build_headroom",
                        lambda core, expected: (prepared["rows"], prepared["manifest"]))


def test_actual_rebuild_is_balanced_deterministic_and_bound(prepared):
    rows, manifest = headroom.build_headroom(prepared["core"], EXPECTED)
    assert rows == prepared["rows"]
    assert r048._jsonl_bytes(rows) == r048._jsonl_bytes(prepared["rows"])
    assert manifest == prepared["manifest"]
    assert manifest["schema"] == "r050_headroom_v2"
    assert manifest["panel_version"] == "1.6"
    assert manifest["role"] == "headroom"
    assert {row["meta"]["split"] for row in rows} == {"eval"}
    assert manifest["bindings"]["core_manifest"]["sha256"] == EXPECTED
    assert manifest["bindings"]["r048_builder_sha256"] == headroom.FROZEN_BUILDER_SHA256
    for split, entry in manifest["bindings"]["core_files"].items():
        assert entry["sha256"] == headroom.FROZEN_FILE_SHA256[split]
        assert entry["rows"] == r048.DEFAULT_ROWS[split]
    construction = manifest["construction"]
    assert construction["prefix_objects_and_bytes_identical"] is True
    assert construction["expanded_eval_rows"] == 5376
    assert construction["expanded_eval_sha256"] == headroom._sha(r048._jsonl_bytes(prepared["expanded"]))
    assert construction["candidate_pool"] == {
        "cluster_index_start": 336, "cluster_index_stop": 672, "clusters": 336, "rows": 2688,
        "sha256": headroom._sha(r048._jsonl_bytes(prepared["candidates"])),
    }
    groups = headroom._groups(rows)
    indices = list(groups)
    assert len(indices) == 84 and indices != list(range(336, 420))
    assert indices == sorted(indices)
    assert _selection(prepared)["selected_cluster_indices"] == indices
    assert _selection(prepared)["selected_cluster_indices_sha256"] == hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    metas = [group[0]["meta"] for group in groups.values()]
    assert Counter(headroom._base_cell(meta) for meta in metas) == {cell: 1 for cell in headroom._base_cells()}
    assert Counter(headroom._pair(meta) for meta in metas) == {pair: 21 for pair in headroom._pairs()}
    assert Counter(meta["value_regime"] for meta in metas) == {"derived_residuals": 42, "raw_operands": 42}
    assert Counter(meta["policy_mode"] for meta in metas) == {"priority": 42, "collect": 42}
    assert manifest["factor_counts"]["clusters_with_at_least_five_fields"] == 48
    assert manifest["audit"]["canonical_eval_validation"]["unique_encoder_endpoints"] == 168
    assert manifest["audit"]["expanded_corpus"]["splits"]["eval"]["generated_counterfactual_clusters"] == 672
    for overlaps in manifest["audit"]["cross_core_overlaps"].values():
        assert set(overlaps) == {*headroom.PROTECTED_KEYS, "row_ids"}
        assert set(overlaps.values()) == {0}


@pytest.mark.parametrize("expected", [None, "0" * 64, "9A477C" + "0" * 58])
def test_external_core_authentication_is_mandatory(tmp_path, expected):
    with pytest.raises(ValueError, match="externally expected"):
        headroom.build_headroom(tmp_path, expected)
    with pytest.raises(TypeError):
        headroom.build_headroom(tmp_path)


def test_rewriting_core_file_and_its_manifest_hash_cannot_reauthenticate(core, tmp_path):
    manifest = json.loads((core / "manifest.json").read_bytes())
    manifest["outputs"]["eval"]["sha256"] = "0" * 64
    altered = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (tmp_path / "manifest.json").write_bytes(altered)
    with pytest.raises(ValueError, match="manifest sha256"):
        headroom.build_headroom(tmp_path, EXPECTED)
    with pytest.raises(ValueError, match="externally expected"):
        headroom.build_headroom(tmp_path, headroom._sha(altered))


@pytest.mark.parametrize("split", r048.SPLITS)
def test_each_core_file_hash_is_authenticated_before_parsing(core, tmp_path, split):
    (tmp_path / "manifest.json").write_bytes((core / "manifest.json").read_bytes())
    for name in r048.SPLITS:
        path = tmp_path / f"{name}.jsonl"
        if name == split:
            path.write_bytes((core / f"{name}.jsonl").read_bytes() + b"\n")
        else:
            path.symlink_to(core / f"{name}.jsonl")
    with pytest.raises(ValueError, match=f"{split}: frozen file sha256"):
        headroom.build_headroom(tmp_path, EXPECTED)


def test_builder_binding_is_literal_not_self_rewritten(core, monkeypatch, tmp_path):
    path = tmp_path / "wrong_builder.py"
    path.write_text("# changed generator\n")
    monkeypatch.setattr(r048, "__file__", str(path))
    with pytest.raises(ValueError, match="bind the frozen"):
        headroom.build_headroom(core, EXPECTED)


@pytest.mark.parametrize("drift", ["object", "bytes_only", "truncated", "tail_id"])
def test_expansion_rejects_prefix_objects_bytes_and_count_drift(prepared, drift):
    expanded = list(prepared["expanded"])
    if drift in {"object", "bytes_only"}:
        expanded[0] = copy.deepcopy(expanded[0])
        if drift == "object":
            expanded[0]["state"] += " drift"
        else:
            criteria = expanded[0]["questions"]["q00"]["criteria"]
            expanded[0]["questions"]["q00"]["criteria"] = dict(reversed(criteria.items()))
            assert expanded[0] == prepared["expanded"][0]
    elif drift == "truncated":
        expanded.pop()
    else:
        expanded[-1] = copy.deepcopy(expanded[-1])
        expanded[-1]["id"] = expanded[-2]["id"]
    with pytest.raises(ValueError, match="prefix|IDs/order/count"):
        headroom._validate_expansion(expanded, prepared["corpus"], prepared["payloads"])


def test_biased_first_eighty_four_clusters_are_rejected(prepared):
    biased = prepared["candidates"][:headroom.HEADROOM_ROWS]
    indices = list(headroom._groups(biased))
    assert indices == list(range(336, 420))
    with pytest.raises(ValueError, match="rotation"):
        headroom.validate_selection(biased, headroom._selection_receipt(indices), headroom._factor_counts(biased))


@pytest.mark.parametrize("mutation", ["missing_row", "missing_cluster", "duplicate_cluster", "missing_pair"])
def test_candidate_pool_proves_every_complete_cell_and_all_four_pairs(prepared, mutation):
    candidates = list(prepared["candidates"])
    if mutation == "missing_row":
        candidates.pop()
    elif mutation == "missing_cluster":
        candidates = candidates[:-8]
    elif mutation == "duplicate_cluster":
        candidates[-8:] = candidates[:8]
    else:
        candidates[:8] = copy.deepcopy(candidates[:8])
        for row in candidates[:8]:
            meta = row["meta"]
            pairs = headroom._pairs()
            same_regime = next(pair for pair in pairs if pair[0] == meta["value_regime"]
                               and pair[1] != meta["renderer_family"])
            meta["renderer_family"], meta["source_family"] = same_regime[1:]
    with pytest.raises(ValueError, match="orbit|complete clusters|duplicate|repeats|all four"):
        headroom._select_candidates(candidates)


@pytest.mark.parametrize("mutation", ["method", "indices", "digest", "pair_order", "factor_counts"])
def test_selection_and_factor_receipts_are_reconstructed_not_trusted(prepared, mutation):
    selection = copy.deepcopy(_selection(prepared))
    counts = copy.deepcopy(prepared["manifest"]["factor_counts"])
    if mutation == "method":
        selection["method"] = "first84"
    elif mutation == "indices":
        selection["selected_cluster_indices"][0] += 1
    elif mutation == "digest":
        selection["selected_cluster_indices_sha256"] = "0" * 64
    elif mutation == "pair_order":
        selection["pair_order"].reverse()
    else:
        counts["regime_surface"][0]["clusters"] += 1
    with pytest.raises(ValueError, match="receipt"):
        headroom.validate_selection(prepared["rows"], selection, counts)


def test_actual_core_overlap_including_row_ids_fails_closed(prepared):
    with pytest.raises(ValueError, match="overlap frozen eval"):
        headroom._audit_headroom(prepared["rows"], {"eval": prepared["rows"]})


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "reordered"])
def test_selected_units_preserve_unique_complete_ordered_orbits(prepared, mutation):
    rows = list(prepared["rows"])
    if mutation == "duplicate":
        rows[-8:] = rows[:8]
    elif mutation == "missing":
        rows.pop()
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError):
        headroom.validate_selection(rows, _selection(prepared), prepared["manifest"]["factor_counts"])


def test_atomic_publication_is_reproducible_manifest_last_and_trainer_scorer_compatible(
        prepared, monkeypatch, tmp_path):
    _mock_build(monkeypatch, prepared)
    before = {name: (prepared["core"] / name).read_bytes()
              for name in ("train.jsonl", "cal.jsonl", "eval.jsonl", "manifest.json")}
    original_write, original_publish = headroom._write_synced, headroom._atomic_publish
    write_order = []

    def write(path, payload):
        write_order.append(path.name)
        original_write(path, payload)

    def publish(stage, out):
        assert not out.exists()
        assert {path.name for path in stage.iterdir()} == {"headroom.jsonl", "manifest.json"}
        assert json.loads((stage / "manifest.json").read_bytes()) == prepared["manifest"]
        original_publish(stage, out)

    monkeypatch.setattr(headroom, "_write_synced", write)
    monkeypatch.setattr(headroom, "_atomic_publish", publish)
    first, second = tmp_path / "first", tmp_path / "second"
    assert headroom.write_headroom(prepared["core"], first, EXPECTED) == prepared["manifest"]
    assert headroom.write_headroom(prepared["core"], second, EXPECTED) == prepared["manifest"]
    assert write_order == ["headroom.jsonl", "manifest.json"] * 2
    for name in ("headroom.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert {name: (prepared["core"] / name).read_bytes() for name in before} == before
    assert not list(tmp_path.glob(".*.stage-*"))
    rows = prepared["rows"]
    assert len(load_examples([first / "headroom.jsonl"])) == 672
    predictions = [{"id": row["id"], "probabilities": {
        qid: {"no": float(target["label"] == "no"), "yes": float(target["label"] == "yes")}
        for qid, target in row["targets"].items()}} for row in rows]
    report = summarize(rows, predictions)
    assert report["rendered"]["exact_outcome_accuracy"] == 1.0
    assert report["semantic_endpoints"]["n"] == 168
    assert report["independent_clusters"]["n"] == 84


@pytest.mark.parametrize("kind", ["empty", "sentinel", "dangling_symlink"])
def test_existing_outputs_are_never_replaced(prepared, monkeypatch, tmp_path, kind):
    out = tmp_path / "out"
    if kind == "dangling_symlink":
        out.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        out.mkdir()
        if kind == "sentinel":
            (out / "sentinel").write_bytes(b"winner")
    monkeypatch.setattr(headroom, "build_headroom", lambda *args: pytest.fail("must fail before building"))
    with pytest.raises(FileExistsError):
        headroom.write_headroom(prepared["core"], out, EXPECTED)
    if kind == "sentinel":
        assert (out / "sentinel").read_bytes() == b"winner"
    if kind == "dangling_symlink":
        assert out.is_symlink()


@pytest.mark.parametrize("sentinel", [False, True])
def test_atomic_racing_destination_wins_without_partial_output(prepared, monkeypatch, tmp_path, sentinel):
    _mock_build(monkeypatch, prepared)
    original = headroom._atomic_publish
    out = tmp_path / "race"

    def racing_publish(stage, target):
        assert (stage / "manifest.json").exists()
        target.mkdir()
        if sentinel:
            (target / "sentinel").write_bytes(b"winner")
        original(stage, target)

    monkeypatch.setattr(headroom, "_atomic_publish", racing_publish)
    with pytest.raises(FileExistsError):
        headroom.write_headroom(prepared["core"], out, EXPECTED)
    assert {path.name for path in out.iterdir()} == ({"sentinel"} if sentinel else set())
    assert not list(tmp_path.glob(".*.stage-*"))


def test_failed_readback_never_publishes_output_or_marker(prepared, monkeypatch, tmp_path):
    _mock_build(monkeypatch, prepared)
    monkeypatch.setattr(r048, "_read_jsonl", lambda path: [])
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="round-trip"):
        headroom.write_headroom(prepared["core"], out, EXPECTED)
    assert not out.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_cannot_publish_inside_the_frozen_core(core):
    with pytest.raises(ValueError, match="separate"):
        headroom.write_headroom(core, core / "headroom", EXPECTED)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_json_authentication_rejects_ambiguous_or_nonfinite_receipts(payload):
    with pytest.raises(ValueError):
        headroom._json(payload, "test")
