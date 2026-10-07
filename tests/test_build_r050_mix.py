import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_r050_mix import DEFAULT_WEIGHTS, build_mix, row_hashes
from scripts.exact_policy_executor import execute_policy
from scripts.train_gemma_decision_lora import load_examples, logical_slot_views


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def _single(identifier, *, hebrew=False):
    return {
        "id": identifier,
        "state": f"רשומה {identifier}" if hebrew else f"record {identifier}",
        "questions": {"decision": {
            "type": "choice", "instructions": "Choose.",
            "criteria": {"yes": "Yes", "no": "No"},
        }},
        "targets": {"decision": {"label": "yes"}},
        "meta": {"source_id": f"source-{identifier}", **({"lang": "he"} if hebrew else {})},
    }


def _executable(group, side, language, order):
    policy = {
        "fields": ["risk", "evidence"], "mode": "priority",
        "rules": [{"when": {"field": "risk"},
                   "action": {"name": "review", "payload": {"queue": "review"}}},
                  {"when": {"field": "evidence"},
                   "action": {"name": "retain", "payload": {"queue": "retain"}}}],
        "fallback": {"name": "approve", "payload": {"queue": "approve"}},
    }
    trace = {"risk": bool(side), "evidence": False}
    labels = ("no", "yes") if order == "no_yes" else ("yes", "no")
    return {
        "id": f"exec-{group}-{side}-{language}-{order}", "state": f"case {group}/{side}/{language}",
        "questions": {
            f"q{index:02d}": {"type": "noul", "instructions": field,
                             "criteria": {label: label.upper() for label in labels}}
            for index, field in enumerate(policy["fields"])
        },
        "targets": {"q00": {"label": "yes" if side else "no"}, "q01": {"label": "no"}},
        "meta": {
            "panel": "exact_policy_v1", "panel_version": "1.4", "split": "train",
            "semantic_group_id": f"exec-group-{group}",
            "semantic_signature": _sha(f"semantics-{group}".encode()),
            "counterfactual_side": side, "language": language, "option_order": order,
            "field_order": policy["fields"], "qid_to_field": {"q00": "risk", "q01": "evidence"},
            "decision_threshold": 0.5, "policy_mode": "priority", "policy": policy,
            "action_trace": trace, "gold_outcome": execute_policy(policy, trace).as_dict(),
        },
    }


def _fixture(tmp_path):
    executable = [_executable(group, side, lang, order) for group in range(2)
                  for side in (0, 1) for lang in ("en", "he") for order in ("no_yes", "yes_no")]
    panel = _write(tmp_path / "panel" / "train.jsonl", executable)
    manifest = tmp_path / "panel" / "manifest.json"
    manifest.write_text(json.dumps({
        "panel": "exact_policy_v1", "panel_version": "1.4",
        "generator_sha256": _sha(b"frozen accepted builder"),
        "outputs": {"train": {"rows": len(executable), "sha256": _sha(panel.read_bytes())}},
    }))
    broad = [_single(f"broad-{index}") for index in range(6)]
    broad.append({"id": "broad-pair", "paired_views": [_single("pair-a"), _single("pair-b")]})
    fields = {"safe": True, "approved": True, "within_limit": False}
    action = {"id": "broad-action", "state": "Read the recorded fields.",
        "questions": {f"q{i:02d}": {"type": "noul", "instructions": field,
                                   "criteria": {"no": "No", "yes": "Yes"}}
                      for i, field in enumerate(fields)},
        "targets": {f"q{i:02d}": {"label": "yes" if truth else "no"}
                    for i, truth in enumerate(fields.values())},
        "meta": {"panel": "action_v1", "field_order": list(fields),
                 "action_trace": fields, "action_gold": "escalate", "decision_threshold": 0.6}}
    broad.append(action)
    hebrew = [_single(f"hebrew-{index}", hebrew=True) for index in range(4)]
    multimodal = []
    for index in range(4):
        row = _single(f"mm-{index}")
        row["image"] = f"images/pixels-{index}.png"
        row["meta"]["r044_variant"] = "real"
        image = tmp_path / "mm" / row["image"]
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), (index * 50, 10, 20)).save(image)
        multimodal.append(row)
    return {
        "panel": panel, "manifest": manifest,
        "broad": _write(tmp_path / "broad.jsonl", broad),
        "hebrew": _write(tmp_path / "hebrew.jsonl", hebrew),
        "multimodal": _write(tmp_path / "mm" / "train.jsonl", multimodal),
        "rows": {"executable": executable, "broad": broad,
                 "hebrew": hebrew, "multimodal": multimodal},
    }


def _build(paths, out, **kwargs):
    return build_mix(paths["panel"], paths["manifest"], [paths["broad"]],
                     [paths["hebrew"]], [paths["multimodal"]], out,
                     total_slots=16, **kwargs)


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _update_panel_manifest(paths, rows):
    _write(paths["panel"], rows)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["outputs"]["train"] = {"rows": len(rows), "sha256": _sha(paths["panel"].read_bytes())}
    paths["manifest"].write_text(json.dumps(manifest))


def test_deterministic_raw_mix_preserves_orbits_replay_options_and_pixel_bytes(tmp_path):
    paths = _fixture(tmp_path)
    first = _build(paths, tmp_path / "first", seed=50)
    second = _build(paths, tmp_path / "second", seed=50)
    assert first == second
    for name in ("train.jsonl", "manifest.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    rows = _read(tmp_path / "first" / "train.jsonl")
    originals = {row["id"]: row for pool in paths["rows"].values() for row in pool}
    assert len(rows) == len({row["id"] for row in rows}) == 16
    assert all(row == originals[row["id"]] for row in rows)
    assert {name: item["optimizer_slots"] for name, item in first["strata"].items()} == {
        "executable": 8, "broad": 4, "hebrew": 2, "multimodal": 2}
    executable_indices = [i for i, row in enumerate(rows) if row["id"].startswith("exec-")]
    assert executable_indices == list(range(executable_indices[0], executable_indices[0] + 8))
    executable = [rows[i] for i in executable_indices]
    assert len({row["meta"]["semantic_group_id"] for row in executable}) == 1
    assert first["strata"]["executable"]["semantic_groups"] == 1
    assert first["strata"]["executable"]["semantic_signatures"] == 1
    for row in executable:
        expected = row["meta"]["option_order"].split("_")
        assert all(list(question["criteria"]) == expected for question in row["questions"].values())
    loaded = load_examples([tmp_path / "first" / "train.jsonl"])
    assert len(loaded) == first["optimizer_slots"] == 16
    assert sum(len(logical_slot_views(row)) for row in loaded) == first["rendered_views"]
    assert first["images"]["files"] == 2
    for row in rows:
        if row.get("image"):
            assert (tmp_path / "first" / row["image"]).read_bytes() == (
                paths["multimodal"].parent / row["image"]).read_bytes()
    assert first["outputs"]["train"]["sha256"] == _sha((tmp_path / "first" / "train.jsonl").read_bytes())
    assert first["code_sha256"]["build_r050_mix.py"] == _sha(Path("scripts/build_r050_mix.py").read_bytes())
    assert first["inputs"]["multimodal:0"]["referenced_images"]["files"] == 4
    assert "eight optimizer slots" in first["protocol"]["executable"]
    assert "singleton real-image" in first["protocol"]["multimodal"]


def test_ids_hashes_and_eval_files_exclude_whole_executable_orbits_and_replay(tmp_path):
    paths = _fixture(tmp_path)
    locked = copy.deepcopy(paths["rows"]["broad"][0])
    locked["id"] = "renamed-locked-case"
    locked_path = _write(tmp_path / "locked.jsonl", [locked])
    locked_image = paths["multimodal"].parent / paths["rows"]["multimodal"][0]["image"]
    hashes = {_sha(locked_image.read_bytes()), *row_hashes(paths["rows"]["hebrew"][0])}
    receipt = _build(paths, tmp_path / "mix", locked_ids=[paths["rows"]["executable"][0]["id"]],
                     locked_hashes=hashes, locked_eval_paths=[locked_path])
    rows = _read(tmp_path / "mix" / "train.jsonl")
    assert all(not row["id"].startswith("exec-0-") for row in rows)
    assert not {"broad-0", "hebrew-0", "mm-0"} & {row["id"] for row in rows}
    assert receipt["quarantine"]["excluded"]["executable"] == {"units": 1, "optimizer_slots": 8}
    assert receipt["quarantine"]["selected_id_overlap"] == 0
    assert receipt["quarantine"]["selected_hash_overlap"] == 0


def test_quarantine_uses_semantic_identity_not_shared_grammar_stratum(tmp_path):
    paths = _fixture(tmp_path)
    locked = copy.deepcopy(paths["rows"]["executable"][0])
    locked["id"] = "locked-eval"
    locked["state"] = "independent locked evaluation case"
    locked["meta"]["semantic_group_id"] = "locked-eval-group"
    locked["meta"]["semantic_signature"] = _sha(b"locked-eval-semantics")
    locked["meta"]["grammar_signature"] = "shared-structural-stratum"
    for row in paths["rows"]["executable"]:
        row["meta"]["grammar_signature"] = "shared-structural-stratum"
    _update_panel_manifest(paths, paths["rows"]["executable"])
    locked_path = _write(tmp_path / "locked-grammar.jsonl", [locked])

    receipt = _build(paths, tmp_path / "mix", locked_eval_paths=[locked_path])

    assert receipt["strata"]["executable"]["optimizer_slots"] == 8
    assert receipt["quarantine"]["excluded"]["executable"] == {
        "units": 0,
        "optimizer_slots": 0,
    }
    assert "shared-structural-stratum" not in row_hashes(locked)
    assert locked["meta"]["semantic_signature"] in row_hashes(locked)


def test_existing_paired_schema_remains_one_optimizer_slot(tmp_path):
    paths = _fixture(tmp_path)
    paired = paths["rows"]["broad"][-2]
    _write(paths["broad"], [paired])
    receipt = build_mix(paths["panel"], paths["manifest"], [paths["broad"]], [], [],
        tmp_path / "mix", total_slots=9,
        weights={"executable": "8/9", "broad": "1/9", "hebrew": 0, "multimodal": 0})
    rows = _read(tmp_path / "mix" / "train.jsonl")
    assert next(row for row in rows if row["id"] == paired["id"]) == paired
    assert receipt["strata"]["broad"]["optimizer_slots"] == 1
    assert receipt["strata"]["broad"]["rendered_views"] == 2
    assert receipt["optimizer_slots"] == len(load_examples([tmp_path / "mix" / "train.jsonl"]))


@pytest.mark.parametrize("case", ["duplicate", "flat", "multi_question", "unknown_options"])
def test_ambiguous_or_invalid_replay_fails_before_publication(tmp_path, case):
    paths = _fixture(tmp_path)
    rows = paths["rows"]["broad"]
    if case == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif case == "flat":
        rows[0] = {"id": "flat", "state": "record", "query": "choose", "candidates": ["a", "b"], "label": 0}
    elif case == "multi_question":
        rows[0]["questions"]["second"] = copy.deepcopy(rows[0]["questions"]["decision"])
        rows[0]["targets"]["second"] = {"label": "yes"}
    else:
        rows[0]["questions"]["decision"]["criteria"] = {str(i): str(i) for i in range(27)}
        rows[0]["targets"]["decision"]["label"] = "0"
    _write(paths["broad"], rows)
    with pytest.raises(ValueError):
        _build(paths, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("case", ["missing", "fake", "paired", "mismatch", "absolute", "traversal"])
def test_multimodal_protocol_fails_closed(tmp_path, case):
    paths = _fixture(tmp_path)
    rows = paths["rows"]["multimodal"]
    image = paths["multimodal"].parent / rows[0]["image"]
    if case == "missing":
        rows[0]["image"] = "images/missing.png"
    elif case == "fake":
        image.write_bytes(b"not pixels")
    elif case == "paired":
        rows[0] = {"id": "paired-mm", "paired_views": [rows[0], _single("second-mm")]}
    elif case == "mismatch":
        rows[0]["meta"]["r044_variant"] = "mismatch"
    elif case == "absolute":
        rows[0]["image"] = str(image.resolve())
    else:
        rows[0]["image"] = "../outside.png"
    _write(paths["multimodal"], rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        _build(paths, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("case", ["incomplete", "wrong_order", "eval", "manifest_tamper"])
def test_frozen_executable_contract_fails_closed(tmp_path, case):
    paths = _fixture(tmp_path)
    rows = paths["rows"]["executable"]
    if case == "incomplete":
        rows.pop()
        _update_panel_manifest(paths, rows)
    elif case == "wrong_order":
        rows[0]["meta"]["option_order"] = "yes_no"
        _update_panel_manifest(paths, rows)
    elif case == "eval":
        rows[0]["meta"]["split"] = "eval"
        _update_panel_manifest(paths, rows)
    else:
        rows[0]["state"] = "tampered"
        _write(paths["panel"], rows)
    with pytest.raises(ValueError):
        _build(paths, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_shortage_bad_hash_and_locked_source_fail_closed(tmp_path):
    paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="needs 1 eligible units"):
        _build(paths, tmp_path / "short", locked_ids=["exec-group-0", "exec-group-1"])
    with pytest.raises(ValueError, match="input sha256 mismatch"):
        _build(paths, tmp_path / "hash", expected_hashes={"broad:0": "0" * 64})
    with pytest.raises(ValueError, match="locked evaluation material"):
        _build(paths, tmp_path / "locked", locked_hashes=[_sha(paths["broad"].read_bytes())])
    with pytest.raises(ValueError, match="unknown input labels"):
        _build(paths, tmp_path / "unknown", expected_hashes={"typo": "0" * 64})
    assert not any((tmp_path / name).exists() for name in ("short", "hash", "locked", "unknown"))


@pytest.mark.parametrize("kwargs", [
    {"total_slots": 17},
    {"total_slots": True},
    {"total_slots": 8, "weights": DEFAULT_WEIGHTS},
    {"total_slots": 16, "weights": {"executable": "1/2", "broad": "1/2"}},
])
def test_fractional_slots_or_partial_executable_orbits_are_rejected(tmp_path, kwargs):
    paths = _fixture(tmp_path)
    with pytest.raises(ValueError):
        build_mix(paths["panel"], paths["manifest"], [paths["broad"]], [paths["hebrew"]],
                  [paths["multimodal"]], tmp_path / "bad", **kwargs)
    assert not (tmp_path / "bad").exists()


def test_refuses_overwrite_even_empty_existing_directory(tmp_path):
    paths = _fixture(tmp_path)
    out = tmp_path / "mix"
    _build(paths, out)
    before = {file.relative_to(out): file.read_bytes() for file in out.rglob("*") if file.is_file()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(paths, out)
    assert before == {file.relative_to(out): file.read_bytes() for file in out.rglob("*") if file.is_file()}
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        _build(paths, empty)


def test_direct_script_help_works_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, str(Path("scripts/build_r050_mix.py").resolve()), "--help"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--locked-eval" in result.stdout
    assert "--multimodal" in result.stdout


def test_native_r048_train_rows_remain_trainer_consumable(tmp_path):
    from scripts.build_r048_action_panel import PANEL_VERSION, build
    paths = _fixture(tmp_path)
    rows = build("train", 16, seed=48)
    _update_panel_manifest(paths, rows)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["panel_version"] = PANEL_VERSION
    paths["manifest"].write_text(json.dumps(manifest))
    receipt = _build(paths, tmp_path / "native")
    assert receipt["optimizer_slots"] == len(load_examples([tmp_path / "native" / "train.jsonl"])) == 16


def test_trainer_id_collision_is_rejected_across_distinct_raw_ids(tmp_path):
    paths = _fixture(tmp_path)
    single = _single("collision")
    pair = {"id": "collision/decision", "paired_views": [_single("nested-a"), _single("nested-b")]}
    _write(paths["broad"], [single, pair, *paths["rows"]["broad"]])
    with pytest.raises(ValueError, match="trainer.*slot ids"):
        _build(paths, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
