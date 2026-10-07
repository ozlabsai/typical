import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from scripts import build_r050_replay_pools as pools
from scripts.build_r050_mix import _slot_info, _tree
from scripts.train_gemma_decision_lora import load_examples


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _flat(index, source, choices=12):
    return {"state": f"רשומה {index}" if source == "MASSIVE" else f"science state {index}",
            "query": f"question {index}", "candidates": [f"choice {index}-{i}" for i in range(choices)],
            "target": [float(i == index % choices) for i in range(choices)],
            "label": index % choices, "p_null": 0.0,
            "meta": {"source": source, "source_id": f"{source}-{index}",
                     **({"lang": "he"} if source == "MASSIVE" else {})}}


def _image(path, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (index % 256, index // 256 % 256, 10)).save(path, format="PNG")
    return path


def _fixture(tmp_path, he_count=15, mm_count=6):
    he = [_flat(i, "MASSIVE") for i in range(he_count)]
    mm = []
    for i in range(mm_count):
        row = _flat(i, "ScienceQA", 4)
        row["image"] = f"images/{i}.png"
        _image(tmp_path / "science" / row["image"], i)
        mm.append(row)
    protected = _flat(999, "ScienceQA", 4)
    protected["image"] = "images/protected.png"
    _image(tmp_path / "science" / protected["image"], 999)
    return {"he": _write(tmp_path / "hebrew.jsonl", he),
            "mm": _write(tmp_path / "science" / "train.jsonl", mm),
            "protect": _write(tmp_path / "science" / "val.jsonl", [protected])}


def _build(paths, out, **kwargs):
    return pools.build(paths["he"], paths["mm"], [paths["protect"]], out,
                       hebrew_count=kwargs.pop("hebrew_count", 10),
                       multimodal_count=kwargs.pop("multimodal_count", 4), **kwargs)


def test_deterministic_exact_balanced_real_pools_bind_receipts_and_work_with_mix(tmp_path):
    paths = _fixture(tmp_path)
    first = _build(paths, tmp_path / "first")
    second = _build(paths, tmp_path / "second")
    assert first == second
    for name in ("hebrew.jsonl", "multimodal.jsonl", "manifest.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert first["schema"] == "r050_replay_pools_v1"
    he, mm = (_read(tmp_path / "first" / f"{name}.jsonl") for name in ("hebrew", "multimodal"))
    assert Counter(row["meta"]["r050_gold_position"] for row in he) == {i: 1 for i in range(10)}
    assert len({row["meta"]["source_id"] for row in he}) == 10
    originals = {row["meta"]["source_id"]: row for path in (paths["he"], paths["mm"]) for row in _read(path)}
    for row in he + mm:
        original = originals[row["meta"]["source_id"]]
        question = row["questions"]["decision"]
        assert row["id"].startswith("r050-")
        assert row["state"] == original["state"]
        assert question["instructions"] == original["query"]
        assert question["criteria"][row["targets"]["decision"]["label"]] == original["candidates"][original["label"]]
        assert not any(key.startswith("r044_") for key in row["meta"])
    assert all(len(row["questions"]["decision"]["criteria"]) == 10 for row in he)
    for row in mm:
        assert row["meta"]["variant"] == "real"
        assert row["targets"]["decision"]["label"] != "null"
        assert list(row["questions"]["decision"]["criteria"].values())[-1] == pools.NULL_OPTION
        copied = tmp_path / "first" / row["image"]
        assert copied.name == row["meta"]["r050_image_sha256"] + ".png"
        assert _sha(copied.read_bytes()) == row["meta"]["r050_image_sha256"]
    for name, rows in (("hebrew", he), ("multimodal", mm)):
        path = tmp_path / "first" / f"{name}.jsonl"
        assert len(load_examples([path])) == len(rows)
        assert len(_slot_info(rows, path, name)) == len(rows)
        assert first["outputs"][name]["sha256"] == _sha(path.read_bytes())
        audit = first["audit"][name]
        assert audit["input_rows"] == audit["selected"] + audit["eligible_not_selected"] + sum(audit["rejections"].values())
    assert first["images"] == _tree({row["image"]: row["meta"]["r050_image_sha256"] for row in mm})
    assert first["inputs"]["scienceqa_train"]["images"]["files"] == 6
    assert first["inputs"]["scienceqa_protection"][0]["images"]["files"] == 1
    assert first["code_sha256"]["build_r050_replay_pools.py"] == _sha(Path(pools.__file__).read_bytes())
    assert {item["license"] for item in first["sources_and_licenses"]} == {"CC-BY-4.0", "CC-BY-NC-SA-4.0"}
    assert first["leakage_checks"]["protected_image_overlap"] == first["leakage_checks"]["protected_decision_overlap"] == 0


def test_defaults_freeze_400_each_with_40_per_hebrew_gold_position(tmp_path):
    paths = _fixture(tmp_path, 400, 400)
    manifest = pools.build(paths["he"], paths["mm"], [paths["protect"]], tmp_path / "out")
    assert manifest["counts"] == {"hebrew": 400, "multimodal": 400}
    assert manifest["gold_positions"] == {str(i): 40 for i in range(10)}
    assert len(manifest["selection"]["multimodal"]) == manifest["images"]["files"] == 400


def test_protected_pixels_and_decisions_excluded_and_all_rows_audited(tmp_path):
    paths = _fixture(tmp_path)
    protected = _read(paths["protect"])[0]
    same_pixels = _flat(200, "ScienceQA", 4)
    same_pixels["image"] = protected["image"]
    same_decision = copy.deepcopy(protected)
    same_decision["meta"]["source_id"] = "copied-eval-decision"
    same_decision["image"] = "images/decision.png"
    _image(paths["mm"].parent / same_decision["image"], 200)
    rows = _read(paths["mm"]) + [same_pixels, same_decision]
    _write(paths["mm"], rows)
    # Protection supports image-free controls and ignores gold/targets in the decision tuple.
    no_image = copy.deepcopy(protected)
    no_image.pop("image")
    no_image["label"] = 2
    second = _write(paths["protect"].parent / "eval.jsonl", [no_image])
    manifest = pools.build(paths["he"], paths["mm"], [paths["protect"], second], tmp_path / "out",
                           hebrew_count=10, multimodal_count=4)
    assert manifest["audit"]["multimodal"]["rejections"] == {"protected_image": 1, "protected_decision": 1}
    assert manifest["protection"]["decision_tuple_count"] == 1
    assert len(manifest["inputs"]["scienceqa_protection"]) == 2
    assert {item["source_id"] for item in manifest["selection"]["multimodal"]}.isdisjoint(
        {"ScienceQA-200", "copied-eval-decision"})


def test_deduplicates_upstream_sources_and_pixel_hashes_without_replacement(tmp_path):
    paths = _fixture(tmp_path)
    he = _read(paths["he"])
    copied_he = copy.deepcopy(he[0])
    copied_he["id"] = "different-rendering-id"
    _write(paths["he"], he + [copied_he])
    mm = _read(paths["mm"])
    copied_pixels = copy.deepcopy(mm[0])
    copied_pixels["meta"]["source_id"] = "same-pixels-new-source"
    copied_source = copy.deepcopy(mm[1])
    copied_source["image"] = "images/source-copy.png"
    copied_source["id"] = "different-rendering-id"
    _image(paths["mm"].parent / copied_source["image"], 100)
    _write(paths["mm"], mm + [copied_pixels, copied_source])
    manifest = _build(paths, tmp_path / "out")
    assert manifest["audit"]["hebrew"]["rejections"]["duplicate_source"] == 1
    assert sum(manifest["audit"]["multimodal"]["rejections"].values()) == 2
    assert len({item["source_id"] for item in manifest["selection"]["multimodal"]}) == 4
    assert len({item["image_sha256"] for item in manifest["selection"]["multimodal"]}) == 4


def test_protection_accepts_one_choice_gold_dropped_evaluation_controls(tmp_path):
    paths = _fixture(tmp_path)
    protected = _read(paths["protect"])[0]
    protected.update(candidates=[protected["candidates"][0]], target=[1.0], label=-1, p_null=1.0)
    protected["meta"]["variant"] = "remove_gold"
    _write(paths["protect"], [protected])
    manifest = _build(paths, tmp_path / "out")
    assert manifest["protection"]["decision_tuple_count"] == 1
    assert manifest["protection"]["image_hash_count"] == 1


@pytest.mark.parametrize("change", ["wrong_source", "english", "few_options", "boolean_label", "negative_target", "image"])
def test_hebrew_invalid_rows_rejected_with_exact_audit_accounting(tmp_path, change):
    paths = _fixture(tmp_path)
    rows = _read(paths["he"])
    row = copy.deepcopy(rows[0])
    row["meta"]["source_id"] = "bad"
    if change == "wrong_source": row["meta"]["source"] = "Other"
    if change == "english": row["meta"]["lang"] = "en"
    if change == "few_options": row = _flat(100, "MASSIVE", 9)
    if change == "boolean_label": row["label"] = False
    if change == "negative_target": row["target"][1:3] = [-1.0, 1.0]
    if change == "image": row["image"] = "unused.png"
    _write(paths["he"], rows + [row])
    manifest = _build(paths, tmp_path / "out")
    assert sum(manifest["audit"]["hebrew"]["rejections"].values()) == 1


@pytest.mark.parametrize("change", ["wrong_source", "no_image", "mismatch", "gold_dropped", "null_present", "too_many_options"])
def test_multimodal_is_scienceqa_real_only_with_trainer_option_limit(tmp_path, change):
    paths = _fixture(tmp_path)
    rows = _read(paths["mm"])
    row = _flat(100, "ScienceQA", 26 if change == "too_many_options" else 4)
    row["image"] = "images/bad-row.png"
    _image(paths["mm"].parent / row["image"], 100)
    if change == "wrong_source": row["meta"]["source"] = "Other"
    if change == "no_image": row.pop("image")
    if change in {"mismatch", "gold_dropped"}: row["meta"]["variant"] = change
    if change == "null_present": row["candidates"][-1] = pools.NULL_OPTION
    _write(paths["mm"], rows + [row])
    manifest = _build(paths, tmp_path / "out")
    assert sum(manifest["audit"]["multimodal"]["rejections"].values()) == 1


@pytest.mark.parametrize("failure", ["hebrew_shortage", "mm_shortage", "fake_image", "missing_image", "invalid_protected", "no_protection"])
def test_failure_leaves_no_output_or_private_staging(tmp_path, failure):
    paths = _fixture(tmp_path)
    if failure == "hebrew_shortage": _write(paths["he"], _read(paths["he"])[:9])
    if failure == "mm_shortage": _write(paths["mm"], _read(paths["mm"])[:3])
    if failure == "fake_image": (paths["mm"].parent / "images/0.png").write_bytes(b"not pixels")
    if failure == "missing_image": (paths["mm"].parent / "images/0.png").unlink()
    if failure == "invalid_protected": _write(paths["protect"], [{"query": "incomplete"}])
    out = tmp_path / "out"
    with pytest.raises((ValueError, FileNotFoundError)):
        if failure == "no_protection":
            pools.build(paths["he"], paths["mm"], [], out)
        else:
            _build(paths, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".out-*"))


def test_never_overwrites_even_empty_dir_or_destination_created_at_publish(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError): _build(paths, out)
    publish = pools._atomic_publish
    def racing_publish(stage, target):
        target.mkdir()
        publish(stage, target)
    monkeypatch.setattr(pools, "_atomic_publish", racing_publish)
    race = tmp_path / "race"
    with pytest.raises(FileExistsError): _build(paths, race)
    assert list(race.iterdir()) == []
    assert not list(tmp_path.glob(".race-*"))


def test_staging_cleanup_on_loader_failure_and_changing_inputs(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    import scripts.train_gemma_decision_lora as trainer
    def failed_loader(paths):
        raise ValueError("injected trainer failure")
    monkeypatch.setattr(trainer, "load_examples", failed_loader)
    with pytest.raises(ValueError, match="injected trainer"):
        _build(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))


def test_image_or_jsonl_change_during_selection_fails_closed(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    original = pools._multimodal
    def changed(*args):
        result = original(*args)
        paths["he"].write_text(paths["he"].read_text() + "\n")
        return result
    monkeypatch.setattr(pools, "_multimodal", changed)
    with pytest.raises(ValueError, match="changed during"):
        _build(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kwargs", [{"hebrew_count": 11}, {"hebrew_count": True}, {"multimodal_count": 0}, {"seed": True}])
def test_invalid_count_or_seed_fails_closed(tmp_path, kwargs):
    paths = _fixture(tmp_path)
    with pytest.raises(ValueError): _build(paths, tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("payload", [b'{"meta":{},"meta":{}}\n', b'{"state":NaN}\n', b'[]\n', b'\xff\n'])
def test_strict_json_inputs(tmp_path, payload):
    paths = _fixture(tmp_path)
    paths["he"].write_bytes(payload)
    with pytest.raises(ValueError): _build(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_upstream_ids_have_manifest_bound_dataset_line_identity(tmp_path):
    paths = _fixture(tmp_path)
    for key in ("he", "mm"):
        rows = _read(paths[key])
        for row in rows: row["meta"].pop("source_id")
        _write(paths[key], rows)
    manifest = _build(paths, tmp_path / "out")
    assert all(item["source_id"] == f"massive-line-{item['source_index'] + 1}" for item in manifest["selection"]["hebrew"])
    assert all(item["source_id"] == f"scienceqa-line-{item['source_index'] + 1}" for item in manifest["selection"]["multimodal"])


def test_direct_cli_help_outside_repo(tmp_path):
    result = subprocess.run([sys.executable, str(Path(pools.__file__).resolve()), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--scienceqa-protect" in result.stdout
