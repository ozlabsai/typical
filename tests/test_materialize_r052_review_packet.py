from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_r052_review_packet.py"
SPEC = importlib.util.spec_from_file_location("materialize_r052_review_packet", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_build_packet_is_deterministic_and_fail_closed() -> None:
    first = module.build_packet(ROOT)
    second = module.build_packet(ROOT)
    assert first == second
    effective_raw, index_raw, blockers_raw = first
    assert len(effective_raw) == 45_640
    assert module._sha(effective_raw) == module.HEBREW_EFFECTIVE_SHA256

    effective = json.loads(effective_raw)
    index = json.loads(index_raw)
    blockers = json.loads(blockers_raw)
    assert effective["schema_version"] == module.HEBREW_EFFECTIVE_SCHEMA
    assert index["status"] == "development_review_only"
    assert index["eligible_for_evaluation"] is False
    assert index["is_audit_pass"] is False
    assert index["is_production_builder_spec"] is False
    assert index["is_promotion_receipt"] is False
    assert index["promotion_allowed"] is False
    assert set(index["human_review_status"].values()) == {False}
    assert blockers["all_required_human_reviews_complete"] is False
    assert blockers["promotion_allowed"] is False
    assert blockers["production_panel_allowed"] is False
    assert set(blockers["human_review_status"].values()) == {False}
    assert all(blocker["resolved"] is False for blocker in blockers["blockers"])
    assert index["summary"] == {
        "hebrew_group_count": 10,
        "mm_case_count": 10,
        "mm_decision_type_distribution": {"Choice": 10, "Noul": 0, "Score": 0},
        # Includes both candidate artifacts, all source bindings, and the complete
        # MM memo/generator/receipt/sidecar/image/control inventory.
        "verified_input_file_count": 59,
    }


def test_cli_writer_refuses_overwrite(tmp_path: Path) -> None:
    effective, index, blockers = module.build_packet(ROOT)
    output = tmp_path / "packet"
    files = {
        "blockers.json": blockers,
        "hebrew-effective-development-candidate.json": effective,
        "review-index.json": index,
    }
    module._write_new_directory(output, files)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        module._write_new_directory(output, files)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_writer_rejects_escape_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    escaped = tmp_path / "escaped.json"
    with pytest.raises(ValueError, match="path-free leaf"):
        module._write_new_directory(output, {"../escaped.json": b"outside"})
    assert not output.exists()
    assert not escaped.exists()


def test_writer_rejects_indirect_symlink_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "nested" / "packet"
    with pytest.raises(ValueError, match="may not traverse a symlink"):
        module._write_new_directory(output, {"index.json": b"{}"})
    assert not (real / "nested" / "packet").exists()


def test_writer_rolls_back_after_mid_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "packet"
    real_open = module.os.open
    created_file_count = 0

    def fail_second_create(
        path: Path, flags: int, mode: int = 0o777, **kwargs: object
    ) -> int:
        nonlocal created_file_count
        if flags & module.os.O_CREAT:
            created_file_count += 1
            if created_file_count == 2:
                raise OSError("injected second-file failure")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(module.os, "open", fail_second_create)
    with pytest.raises(OSError, match="injected second-file failure"):
        module._write_new_directory(output, {"first.json": b"1", "second.json": b"2"})
    assert not output.exists()


@pytest.mark.parametrize(
    "path",
    ["../escape", "/absolute", "a/./b", "a//b", "a\\b"],
)
def test_declared_paths_reject_traversal_and_noncanonical_forms(path: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        module._declared_path(path, "adversarial path")


def test_bound_reader_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.write_bytes(b"bound")
    (tmp_path / "alias").symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        module._read_bound(tmp_path, "alias", module._sha(b"bound"), "symlink input")


def test_json_parser_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module._parse_json(b'{"a":1,"a":2}', "duplicate")
    with pytest.raises(ValueError, match="non-finite"):
        module._parse_json(b'{"a":NaN}', "nonfinite")


def test_materializer_rejects_order_drift_unknown_operation_and_duplicate_group() -> None:
    base_raw = (ROOT / module.HEBREW_BASE_PATH).read_bytes()
    overlay_raw = (ROOT / module.HEBREW_OVERLAY_PATH).read_bytes()
    base = module._parse_json(base_raw, "base")
    overlay = module._parse_json(overlay_raw, "overlay")

    reordered = json.loads(json.dumps(overlay))
    reordered["operations"][0]["preconditions"]["known_distractor_chunk_refs"].reverse()
    with pytest.raises(ValueError, match="ordered preconditions"):
        module._materialize_hebrew(base, reordered, module._sha(base_raw))

    unsupported = json.loads(json.dumps(overlay))
    unsupported["operations"][0]["operation"] = "invented_operation"
    with pytest.raises(ValueError, match="unsupported Hebrew overlay operation"):
        module._materialize_hebrew(base, unsupported, module._sha(base_raw))

    duplicated = json.loads(json.dumps(base))
    duplicated["groups"].append(json.loads(json.dumps(duplicated["groups"][8])))
    with pytest.raises(ValueError, match="exactly one"):
        module._materialize_hebrew(duplicated, overlay, module._sha(base_raw))


def test_inventory_rejects_duplicate_role_and_conflicting_binding() -> None:
    inventory: dict[str, dict[str, object]] = {}
    item = {"path": "a", "bytes": 1, "sha256": "0" * 64}
    module._inventory_add(inventory, item, "role")
    with pytest.raises(ValueError, match="duplicate inventory role"):
        module._inventory_add(inventory, item, "role")
    with pytest.raises(ValueError, match="conflicting inventory binding"):
        module._inventory_add(
            inventory,
            {"path": "a", "bytes": 2, "sha256": "1" * 64},
            "other",
        )


def test_unsupported_versions_are_rejected() -> None:
    base_raw = (ROOT / module.HEBREW_BASE_PATH).read_bytes()
    overlay_raw = (ROOT / module.HEBREW_OVERLAY_PATH).read_bytes()
    base = module._parse_json(base_raw, "base")
    overlay = module._parse_json(overlay_raw, "overlay")
    base["schema_version"] = "r052-future"
    with pytest.raises(ValueError, match="unsupported Hebrew base"):
        module._materialize_hebrew(base, overlay, module._sha(base_raw))
