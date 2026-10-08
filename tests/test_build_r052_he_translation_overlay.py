from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_r052_he_translation_overlay as builder
from scripts import score_r052_evidence_retrieval as retrieval


def _effective(*, duplicate: bool = False) -> dict[str, Any]:
    refs = ["d1", "d2", "d2"] if duplicate else ["d1", "d2"]
    return {
        "schema_version": "effective",
        "input_bindings": {
            "ntsb_candidate_dossier_sha256": "n" * 64,
            "gao_candidate_dossier_sha256": "g" * 64,
        },
        "groups": [
            {
                "group_id": "group-1",
                "source_id": "SOURCE",
                "decisive_evidence": [
                    {
                        "candidate_chunk_refs": [refs[0]],
                        "he_translation": "ראיה מכרעת קיימת",
                    }
                ],
                "known_distractor_chunk_refs": refs[1:],
                "expected_typed_outcome": {"secret": "DO_NOT_READ_GOLD"},
                "options": [{"secret": "DO_NOT_READ_OPTIONS"}],
            }
        ],
    }


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, effective: dict[str, Any]) -> None:
    monkeypatch.setattr(
        builder,
        "_load_inputs",
        lambda *_args, **_kwargs: (
            effective,
            {"SOURCE": {"d1": "decisive evidence", "d2": "neutral distractor"}},
            {
                "hebrew_v2_raw_sha256": "1" * 64,
                "hebrew_v4_raw_sha256": "2" * 64,
                "hebrew_v4_effective_canonical_sha256": "3" * 64,
                "ntsb_candidate_sha256": "n" * 64,
                "gao_candidate_sha256": "g" * 64,
            },
        ),
    )


def _kwargs(tmp_path: Path) -> dict[str, Path]:
    return {
        "he_base": tmp_path / "base.json",
        "he_overlay": tmp_path / "overlay.json",
        "ntsb_candidates": tmp_path / "ntsb.json",
        "gao_candidates": tmp_path / "gao.json",
    }


def test_exact_rows_hashes_uniform_catalog_no_gold_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch, _effective())
    catalog = {
        ("group-1", "d1"): "מקטע מכריע",
        ("group-1", "d2"): "מסיח ניטרלי",
    }
    first = builder.build_overlay(**_kwargs(tmp_path), translation_catalog=catalog)
    second = builder.build_overlay(**_kwargs(tmp_path), translation_catalog=catalog)
    assert first == second
    assert first["schema_version"] == builder.SCHEMA_VERSION
    assert first["status"] == builder.STATUS
    assert first["human_reviewed"] is False
    assert first["eligible_for_b1_or_benchmark_claims"] is False
    assert first["counts"] == {
        "groups": 1,
        "rows": 2,
        "catalog_translations": 2,
    }
    rows = first["groups"]["group-1"]
    assert [row["chunk_ref"] for row in rows] == ["d1", "d2"]
    assert all(
        set(row) == {"chunk_ref", "en_text_sha256", "he_text", "he_text_sha256"}
        for row in rows
    )
    assert rows[0]["he_text"] == "מקטע מכריע"
    assert rows[0]["he_text"] != "ראיה מכרעת קיימת"
    assert rows[1]["he_text"] == "מסיח ניטרלי"
    assert rows[1]["en_text_sha256"] == retrieval.sha256_bytes(b"neutral distractor")
    assert rows[1]["he_text_sha256"] == retrieval.sha256_bytes("מסיח ניטרלי".encode())
    serialized = retrieval.canonical_json(first).decode()
    assert "DO_NOT_READ_GOLD" not in serialized
    assert "DO_NOT_READ_OPTIONS" not in serialized
    body = {key: value for key, value in first.items() if key != "overlay_sha256"}
    assert first["overlay_sha256"] == retrieval.sha256_json(body)


@pytest.mark.parametrize(
    ("effective", "catalog", "message"),
    [
        (
            _effective(),
            {("group-1", "d1"): "מכריע"},
            "missing catalog translation",
        ),
        (
            _effective(),
            {
                ("group-1", "d1"): "מכריע",
                ("group-1", "d2"): "מסיח",
                ("group-1", "stale"): "מיושן",
            },
            "stale entries",
        ),
        (
            _effective(duplicate=True),
            {("group-1", "d1"): "מכריע", ("group-1", "d2"): "מסיח"},
            "incomplete or duplicate",
        ),
    ],
)
def test_rejects_incomplete_stale_and_duplicate_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    effective: dict[str, Any],
    catalog: dict[tuple[str, str], str],
    message: str,
) -> None:
    _patch_inputs(monkeypatch, effective)
    with pytest.raises(ValueError, match=message):
        builder.build_overlay(**_kwargs(tmp_path), translation_catalog=catalog)


def test_exact_candidate_bindings_reject_stale_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {"input_bindings": {"ntsb_candidate_dossier_sha256": "x" * 64}}
    monkeypatch.setattr(
        builder.diagnostic,
        "_load_json",
        lambda path, _label: (
            (base, "1" * 64) if path.name == "base.json" else ({}, "2" * 64)
        ),
    )
    monkeypatch.setattr(
        builder.diagnostic, "materialize_hebrew_v4", lambda *_args: base
    )
    monkeypatch.setattr(
        builder.diagnostic,
        "_candidate_maps",
        lambda *_args: (
            {},
            {"ntsb_candidate_sha256": "n" * 64, "gao_candidate_sha256": "g" * 64},
        ),
    )
    with pytest.raises(ValueError, match="does not bind"):
        builder._load_inputs(**_kwargs(tmp_path))


def test_atomic_canonical_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch, _effective())
    output = tmp_path / "built.json"
    overlay = builder.build_and_publish(
        **_kwargs(tmp_path),
        output=output,
        translation_catalog={
            ("group-1", "d1"): "מקטע מכריע",
            ("group-1", "d2"): "מסיח ניטרלי",
        },
    )
    assert output.read_bytes() == retrieval.canonical_json(overlay)
    assert json.loads(output.read_bytes()) == overlay
    with pytest.raises(FileExistsError, match="overwrite"):
        builder.build_and_publish(
            **_kwargs(tmp_path),
            output=output,
            translation_catalog={
                ("group-1", "d1"): "מקטע מכריע",
                ("group-1", "d2"): "מסיח ניטרלי",
            },
        )


def test_frozen_catalog_shape_matches_current_missing_translation_count() -> None:
    assert len(builder.TRANSLATIONS) == 82
    assert all(
        isinstance(key, tuple)
        and len(key) == 2
        and all(isinstance(part, str) and part for part in key)
        and isinstance(value, str)
        and value.strip() == value
        and value
        for key, value in builder.TRANSLATIONS.items()
    )
    assert (
        builder.TRANSLATIONS[("he_en_04_par_fireball_visibility", "par_ignition")]
        == "כדור אש שנשמע ונראה ממרחק של מיילים."
    )


def test_publish_rejects_broken_symlink_and_symlink_ancestor(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.symlink_to(tmp_path / "absent.json")
    with pytest.raises(FileExistsError, match="symlink"):
        builder._exclusive_publish(broken, b"{}\n")

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestors"):
        builder._exclusive_publish(linked_parent / "escaped.json", b"{}\n")
    assert not (real_parent / "escaped.json").exists()
