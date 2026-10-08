from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts import run_r052_he_mm_dev_diag as diag
from scripts import score_r052_evidence_retrieval as retrieval


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[list[diag.EmbeddingInput]] = []

    def embed(self, inputs: list[diag.EmbeddingInput]) -> list[list[float]]:
        self.calls.append(list(inputs))
        result = []
        for item in inputs:
            media_sha = "" if item.media is None else item.media["sha256"]
            raw = hashlib.shake_256((item.text + media_sha).encode()).digest(768)
            result.append(
                (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 127.5).tolist()
            )
        return result

    def identity(self) -> dict[str, Any]:
        return {"kind": "fake_deterministic_test_provider", "dimension": 768}

    def runtime_identity(self) -> dict[str, Any]:
        return {"fake_device": "cpu"}


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return path


def _questions(kind: str) -> list[dict[str, Any]]:
    result = []
    for decision in ("Choice", "Noul", "Score"):
        question: dict[str, Any] = {
            "question_ref": f"q-{decision.lower()}",
            "instructions": f"Answer the {decision} question.",
            "criteria": {"a": "Alpha", "b": "Beta"}
            if kind == "ntsb"
            else ["Alpha", "Beta"],
            "decisive_chunk_refs": ["d1"],
            "known_distractor_chunk_refs": ["d2"],
            "type": decision,
            "policy_role": decision.lower(),
        }
        if kind == "ntsb":
            question.update({"primitive": decision, "options": ["a", "b"]})
        result.append(question)
    return result


def _candidate(kind: str, source_id: str) -> dict[str, Any]:
    if kind == "ntsb":
        chunks = [
            {
                "chunk_ref": ref,
                "source_anchor": {"verbatim_short_phrase": text},
            }
            for ref, text in (
                ("d1", "decisive English evidence"),
                ("d2", "irrelevant English evidence"),
            )
        ]
        case = {
            "case_ref": "ntsb-case",
            "source": {"snapshot_id": source_id},
            "retrievable_chunks": chunks,
            "questions": _questions(kind),
        }
    else:
        chunks = [
            {"chunk_ref": ref, "anchor": {"verbatim_phrase": text}}
            for ref, text in (("g1", "GAO decisive"), ("g2", "GAO distractor"))
        ]
        questions = _questions(kind)
        for question in questions:
            question["decisive_chunk_refs"] = ["g1"]
            question["known_distractor_chunk_refs"] = ["g2"]
        case = {
            "case_ref": "gao-case",
            "source": {"id": "GAO-X"},
            "retrieval_plan": {"chunks": chunks},
            "questions": questions,
        }
    return {
        "schema_version": diag.text_pilot.EXPECTED_CANDIDATE_SCHEMAS[kind],
        "cases": [case],
    }


def _hebrew_files(
    tmp_path: Path,
    ntsb_sha: str,
    gao_sha: str,
    acquisition_sha: str,
    *,
    reverse_options: bool = False,
) -> tuple[Path, Path]:
    group = {
        "group_id": "he_en_09_gao_positive_assurance",
        "primitive": "Choice",
        "policy_role": "select_supported_account",
        "source_id": "SRC",
        "question_ref": "he-q",
        "views": {
            "en": {"question": "Which account is supported?"},
            "he": {"question": "איזה תיאור נתמך?"},
        },
        "options": [
            {"option_id": "a", "en": "Alpha", "he": "אלפא"},
            {"option_id": "b", "en": "Beta", "he": "בטא"},
        ],
        "expected_typed_outcome": {"status": "SECRET_OUTCOME"},
        "decisive_evidence": [{"evidence_ref": "ev", "candidate_chunk_refs": ["d1"]}],
        "excluded_answer_support_chunk_refs": [],
        "known_distractor_chunk_refs": ["d2"],
        "queries": {
            "he_to_en": "איזה תיאור נתמך?\nאפשרויות:\na\tאלפא\nb\tבטא",
            "he_to_he": "איזה תיאור נתמך?\nאפשרויות:\na\tאלפא\nb\tבטא",
        },
    }
    if reverse_options:
        group["options"].reverse()
    base = {
        "schema_version": diag.HE_V2_SCHEMA,
        "input_bindings": {
            "acquisition_manifest_sha256": acquisition_sha,
            "ntsb_candidate_dossier_sha256": ntsb_sha,
            "gao_candidate_dossier_sha256": gao_sha,
        },
        "groups": [group],
    }
    base_path = _write_json(tmp_path / "he-v2.json", base)
    base_sha = retrieval.file_sha256(base_path)
    operation = {
        "operation": "replace_group09_removal_partition",
        "group_id": group["group_id"],
        "preconditions": {
            "decisive_candidate_chunk_refs": ["d1"],
            "excluded_answer_support_chunk_refs": [],
            "known_distractor_chunk_refs": ["d2"],
        },
        "replacement": {
            "excluded_answer_support_chunk_refs": [],
            "known_distractor_chunk_refs": ["d2"],
            "added_genuine_distractor_chunks": [],
        },
    }
    operations_sha = retrieval.sha256_bytes(diag._canonical_bytes([operation]))
    effective = deepcopy(base)
    effective["groups"][0]["added_genuine_distractor_chunks"] = []
    effective["schema_version"] = diag.HE_EFFECTIVE_SCHEMA
    effective["materialization_receipt_ref"] = {
        "overlay_schema_version": diag.HE_V4_SCHEMA,
        "base_raw_sha256": base_sha,
        "operations_canonical_sha256": operations_sha,
    }
    effective_bytes = diag._canonical_bytes(effective)
    overlay = {
        "schema_version": diag.HE_V4_SCHEMA,
        "base": {"sha256": base_sha},
        "operations": [operation],
        "deterministic_materialization": {
            "materialization_receipt": {
                "operations_canonical_sha256": operations_sha,
                "effective_canonical_byte_length": len(effective_bytes),
                "effective_canonical_sha256": retrieval.sha256_bytes(effective_bytes),
            }
        },
    }
    return base_path, _write_json(tmp_path / "he-v4.json", overlay)


def _png(path: Path, color: tuple[int, int, int]) -> dict[str, Any]:
    Image.new("RGB", (2, 2), color).save(path, format="PNG")
    raw = path.read_bytes()
    return {"path": path.name, "sha256": retrieval.sha256_bytes(raw), "bytes": len(raw)}


def _mm_file(tmp_path: Path, generator: Path, acquisition: Path) -> tuple[Path, Path]:
    root = tmp_path / "mm"
    root.mkdir()
    real = _png(root / "real.png", (255, 0, 0))
    shuffled = _png(root / "shuffled.png", (0, 255, 0))
    missing = _png(root / "missing.png", (255, 255, 255))
    distractors = {}
    ids = []
    for index in range(5):
        key = f"D{index}"
        text = f"unrelated exact source passage {index}"
        ids.append(key)
        distractors[key] = {
            "content": text,
            "content_sha256": retrieval.sha256_bytes(text.encode()),
        }
    neutral = "The image is the only admissible evidence."
    options = ["z", "a", "decision.insufficient_visual_evidence"]
    cases = []
    for index in range(10):
        case_id = f"M{index + 1:02d}"
        display = ["z", "decision.insufficient_visual_evidence", "a"]
        cases.append(
            {
                "id": case_id,
                "question": f"Which visual option is supported in panel {index}?",
                "options": list(options),
                "option_permutation": {
                    "display_order": display,
                    "display_order_sha256": retrieval.sha256_json(display),
                    "semantic_to_display_index": {
                        option: display.index(option) for option in sorted(options)
                    },
                },
                "real_image": real,
                "shuffled_image": shuffled,
                "missing_image": missing,
                "distractors": ids,
                "real_gold": "SECRET_GOLD_LABEL",
            }
        )
    receipt = {
        "schema_version": diag.MM_SCHEMA,
        "status": "ai_authored_unreviewed_development_only",
        "eligible_for_sealed_or_public_eval": False,
        "pixel_causality_review_complete": False,
        "rights_review_complete": False,
        "semantic_review_complete": False,
        "generator": {
            "path": str(generator),
            "sha256": retrieval.file_sha256(generator),
        },
        "acquisition_manifest": {
            "path": str(acquisition),
            "sha256": retrieval.file_sha256(acquisition),
        },
        "neutral_context": neutral,
        "neutral_context_sha256": retrieval.sha256_bytes(neutral.encode()),
        "distractors": distractors,
        "cases": cases,
    }
    return _write_json(root / "receipt.json", receipt), root


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    generator = tmp_path / "build.py"
    generator.write_text("# frozen test generator\n")
    acquisition = _write_json(tmp_path / "acquisition.json", {"frozen": True})
    ntsb = _write_json(tmp_path / "ntsb.json", _candidate("ntsb", "SRC"))
    gao = _write_json(tmp_path / "gao.json", _candidate("gao", "GAO-X"))
    he_base, he_overlay = _hebrew_files(
        tmp_path,
        retrieval.file_sha256(ntsb),
        retrieval.file_sha256(gao),
        retrieval.file_sha256(acquisition),
    )
    mm_receipt, mm_root = _mm_file(tmp_path, generator, acquisition)
    return {
        "he_base": he_base,
        "he_overlay": he_overlay,
        "ntsb_candidates": ntsb,
        "gao_candidates": gao,
        "mm_receipt": mm_receipt,
        "mm_artifact_root": mm_root,
    }


def _build(
    inputs: dict[str, Path], provider: FakeProvider | None = None
) -> tuple[dict[str, Any], FakeProvider]:
    actual = provider or FakeProvider()
    receipt = diag.build_receipt(
        **inputs,
        provider=actual,
        artifact_bindings={"runtime_lock_sha256": diag.text_pilot.RUNTIME_LOCK_SHA256},
    )
    return receipt, actual


def test_schema_gold_isolation_missing_media_and_exact_bm25_tie(
    inputs: dict[str, Path],
) -> None:
    receipt, provider = _build(inputs)
    assert receipt["claim_status"] == diag.CLAIM_STATUS
    assert receipt["frozen_evaluation_panel"] is False
    assert receipt["eligible_for_b1_or_benchmark_claims"] is False
    assert receipt["hebrew"]["summary"]["he_to_he"] == {
        "available": False,
        "reason": "complete_translation_overlay_not_supplied",
    }
    row = receipt["multimodal"]["rows"][0]
    assert row["bm25"]["matched_control_exact_tie"] is True
    assert len(set(row["bm25"]["matched_control_scores"].values())) == 1
    payloads = provider.calls[0]
    assert all(not hasattr(payload, "role") for payload in payloads)
    embedded = "\n".join(payload.text for payload in payloads)
    assert "SECRET_GOLD_LABEL" not in embedded
    assert "SECRET_OUTCOME" not in embedded
    neutral = receipt["contracts"]["multimodal_matched_text"]
    neutral_payloads = [payload for payload in payloads if payload.text == neutral]
    assert len(neutral_payloads) == 3
    assert sum(payload.media is None for payload in neutral_payloads) == 1
    assert (
        len(receipt["multimodal"]["rows"][0]["dimensions"]["768"]["candidate_ranking"])
        == 8
    )


def test_option_order_permutation_does_not_change_query_or_rankings(
    inputs: dict[str, Path],
) -> None:
    first, _ = _build(inputs)
    value = json.loads(inputs["mm_receipt"].read_text())
    for case in value["cases"]:
        case["options"].reverse()
        case["distractors"].reverse()
        display = case["option_permutation"]["display_order"]
        display.reverse()
        case["option_permutation"]["display_order_sha256"] = retrieval.sha256_json(
            display
        )
        case["option_permutation"]["semantic_to_display_index"] = {
            option: display.index(option) for option in sorted(case["options"])
        }
    _write_json(inputs["mm_receipt"], value)
    second, _ = _build(inputs)
    assert (
        first["multimodal"]["rows"][0]["dimensions"]
        == second["multimodal"]["rows"][0]["dimensions"]
    )


def test_duplicate_distractors_are_rejected(inputs: dict[str, Path]) -> None:
    value = json.loads(inputs["mm_receipt"].read_text())
    value["cases"][0]["distractors"] = ["D0"] * 5
    _write_json(inputs["mm_receipt"], value)
    with pytest.raises(ValueError, match="exactly five text distractors"):
        _build(inputs)


def test_hebrew_option_and_candidate_order_do_not_change_scores(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    first, _ = _build(inputs)
    ntsb = json.loads(inputs["ntsb_candidates"].read_text())
    ntsb["cases"][0]["retrievable_chunks"].reverse()
    for question in ntsb["cases"][0]["questions"]:
        question["options"].reverse()
    _write_json(inputs["ntsb_candidates"], ntsb)
    mm = json.loads(inputs["mm_receipt"].read_text())
    he_base, he_overlay = _hebrew_files(
        tmp_path,
        retrieval.file_sha256(inputs["ntsb_candidates"]),
        retrieval.file_sha256(inputs["gao_candidates"]),
        mm["acquisition_manifest"]["sha256"],
        reverse_options=True,
    )
    inputs["he_base"] = he_base
    inputs["he_overlay"] = he_overlay
    second, _ = _build(inputs)
    first_row = first["hebrew"]["rows"][0]
    second_row = second["hebrew"]["rows"][0]
    assert first_row["query_sha256"] == second_row["query_sha256"]
    assert first_row["he_to_en"] == second_row["he_to_en"]
    assert first_row["en_to_en"] == second_row["en_to_en"]


def test_hebrew_translation_overlay_rejects_missing_coverage(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    overlay = _write_json(
        tmp_path / "translations.json",
        {
            "schema_version": diag.TRANSLATION_SCHEMA,
            "groups": {"he_en_09_gao_positive_assurance": []},
        },
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        diag.build_receipt(
            **inputs,
            he_translation_overlay=overlay,
            provider=FakeProvider(),
            artifact_bindings={
                "runtime_lock_sha256": diag.text_pilot.RUNTIME_LOCK_SHA256
            },
        )


def test_atomic_no_overwrite_and_runtime_separation(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "result.json"
    runtime = tmp_path / "runtime.json"
    receipt = diag.run_and_publish(
        **inputs,
        output=output,
        runtime_sidecar=runtime,
        provider=FakeProvider(),
        artifact_bindings={"runtime_lock_sha256": diag.text_pilot.RUNTIME_LOCK_SHA256},
    )
    assert output.read_bytes() == retrieval.canonical_json(receipt)
    runtime_value = json.loads(runtime.read_text())
    assert runtime_value["deterministic_receipt_file_sha256"] == retrieval.file_sha256(
        output
    )
    assert "elapsed_seconds" not in receipt
    with pytest.raises(FileExistsError):
        diag.run_and_publish(
            **inputs,
            output=output,
            runtime_sidecar=runtime,
            provider=FakeProvider(),
            artifact_bindings={
                "runtime_lock_sha256": diag.text_pilot.RUNTIME_LOCK_SHA256
            },
        )


def test_publish_rejects_broken_symlink_escape(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    target_parent = tmp_path / "elsewhere"
    output_parent.mkdir()
    target_parent.mkdir()
    target = target_parent / "escaped.json"
    output = output_parent / "result.json"
    output.symlink_to(target)
    with pytest.raises(FileExistsError, match="symlink output path"):
        diag._exclusive_publish(output, b"must not escape")
    assert output.is_symlink()
    assert not target.exists()


def test_runtime_failure_does_not_remove_same_content_replacement(
    inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    runtime = tmp_path / "runtime.json"
    displaced = tmp_path / "original-result.json"
    real_publish = diag._exclusive_publish
    calls = 0

    def replace_before_runtime_failure(path: Path, data: bytes) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_publish(path, data)
        output.replace(displaced)
        output.write_bytes(displaced.read_bytes())
        raise OSError("injected runtime publication failure")

    monkeypatch.setattr(diag, "_exclusive_publish", replace_before_runtime_failure)
    with pytest.raises(OSError, match="injected runtime publication failure"):
        diag.run_and_publish(
            **inputs,
            output=output,
            runtime_sidecar=runtime,
            provider=FakeProvider(),
            artifact_bindings={
                "runtime_lock_sha256": diag.text_pilot.RUNTIME_LOCK_SHA256
            },
        )
    assert output.is_file()
    assert displaced.is_file()
    assert output.read_bytes() == displaced.read_bytes()
