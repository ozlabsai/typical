import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".context/r052-image-candidates/build_v3.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r052_mm_v3_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder():
    return _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_build_is_deterministic_development_only_and_no_overwrite(tmp_path, builder):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = builder.build(first)
    second_result = builder.build(second)

    assert first_result["receipt_sha256"] == second_result["receipt_sha256"]
    assert _tree(first) == _tree(second)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build(first)

    raw = (first / "development-receipt.json").read_bytes()
    receipt = json.loads(raw)
    assert raw == builder.canonical(receipt)
    assert first_result["receipt_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["schema_version"] == "r052-mm-development-candidate-v3"
    assert receipt["status"] == {
        "stage": "development_candidate_only",
        "ai_authored": True,
        "human_reviewed": False,
        "adjudicated": False,
        "rights_cleared": False,
        "semantic_ground_truth_ready": False,
        "pixel_causality_verified": False,
        "promotion_ready": False,
        "eligible_for_training": False,
        "eligible_for_sealed_or_public_eval": False,
        "benchmark_derived": False,
        "sealed_benchmark_content_included": False,
    }


def test_images_are_declared_once_and_controls_reference_artifacts(tmp_path, builder):
    output = tmp_path / "candidate"
    builder.build(output)
    receipt = json.loads((output / "development-receipt.json").read_bytes())

    artifacts = receipt["image_artifacts"]
    assert len(artifacts) == 10
    assert len({value["sha256"] for value in artifacts.values()}) == 10
    assert not (output / "shuffled").exists()
    assert not (output / "missing").exists()
    assert (
        receipt["canonical_blank_provenance"]["declared_as_builder_image_artifact"]
        is False
    )
    assert (
        receipt["canonical_blank_provenance"]["missing_controls_embed_media"] is False
    )

    for case in receipt["cases"]:
        controls = case["controls"]
        real = controls["real"]
        shuffled = controls["shuffled"]
        missing = controls["missing"]
        assert real["sha256"] == artifacts[real["image_artifact"]]["sha256"]
        assert shuffled["sha256"] == artifacts[shuffled["image_artifact"]]["sha256"]
        assert shuffled["source_case"] != case["id"]
        assert missing["image_artifact"] is None
        assert missing["media"] is None
        assert "path" not in missing


def test_distractors_are_exact_same_source_slices_and_options_are_labeled(
    tmp_path, builder
):
    output = tmp_path / "candidate"
    builder.build(output)
    receipt = json.loads((output / "development-receipt.json").read_bytes())

    for case in receipt["cases"]:
        distractors = case["same_source_text_distractors"]
        assert len(distractors) == 5
        assert len({item["content_sha256"] for item in distractors}) == 5
        for item in distractors:
            assert item["snapshot_id"] == case["snapshot"]
            source = ROOT / item["source_path"]
            raw = source.read_bytes()
            text = raw.decode("utf-8")
            start, end = item["char_range_utf8_decoded"]
            byte_start, byte_end = item["byte_range"]
            assert _sha(source) == item["source_sha256"]
            assert text[start:end] == item["content"]
            assert raw[byte_start:byte_end].decode("utf-8") == item["content"]
            assert (
                hashlib.sha256(item["content"].encode()).hexdigest()
                == item["content_sha256"]
            )
            assert item["same_source_verified_mechanically"] is True
            assert item["semantic_non_answer_review_complete"] is False
            assert item["reviewed_source_offset_remap_complete"] is False

        options = case["options"]
        assert len(options) == 5
        assert len({item["option_id"] for item in options}) == 5
        assert all(item["label"].strip() for item in options)
        displayed = case["option_permutation"]["display_options"]
        assert [item["option_id"] for item in displayed] == case["option_permutation"][
            "display_order"
        ]


def test_rejects_v2_hash_drift_before_output(tmp_path, builder, monkeypatch):
    tampered = tmp_path / "tampered-v2.json"
    tampered.write_bytes(builder.V2_RECEIPT.read_bytes() + b" ")
    monkeypatch.setattr(builder, "V2_RECEIPT", tampered)

    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="v2 receipt hash drift"):
        builder.build(output)
    assert not output.exists()


def test_rejects_unknown_or_cross_plan_distractor_reference(
    tmp_path, builder, monkeypatch
):
    selections = dict(builder.CASE_DISTRACTOR_REFS)
    selections["M01"] = (
        "har_fatalities",
        *selections["M01"][1:],
    )
    monkeypatch.setattr(builder, "CASE_DISTRACTOR_REFS", selections)

    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="unknown same-source chunk"):
        builder.build(output)
    assert not output.exists()


def test_rejects_missing_human_readable_option_label(tmp_path, builder, monkeypatch):
    labels = dict(builder.OPTION_LABELS)
    labels.pop("parameter.vertical_acceleration")
    monkeypatch.setattr(builder, "OPTION_LABELS", labels)

    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="missing option labels"):
        builder.build(output)
    assert not output.exists()


def test_rejects_full_source_drift_outside_selected_anchor(
    tmp_path, builder, monkeypatch
):
    plan = builder._candidate_cases()["AAR0002"]
    chunk = next(
        item
        for item in builder._chunks(plan)
        if item["chunk_ref"] == builder.CASE_DISTRACTOR_REFS["M01"][0]
    )
    source = ROOT / ".context/r052-sources/text/AAR0002.txt"
    tampered = tmp_path / ".context/r052-sources/text/AAR0002.txt"
    tampered.parent.mkdir(parents=True)
    tampered.write_bytes(source.read_bytes() + b" ")
    monkeypatch.setattr(builder, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="full extracted source hash drift"):
        builder._exact_anchor_span(
            "AAR0002", chunk, builder._candidate_source_sha256(plan)
        )


def test_concurrent_output_creation_is_never_replaced(tmp_path, builder, monkeypatch):
    output = tmp_path / "candidate"
    original_publish = builder._atomic_install_no_replace

    def racing_publish(staging, destination):
        destination.mkdir()
        (destination / "race-winner.txt").write_text("preserve me", encoding="utf-8")
        original_publish(staging, destination)

    monkeypatch.setattr(builder, "_atomic_install_no_replace", racing_publish)
    with pytest.raises(FileExistsError, match="concurrently created"):
        builder.build(output)

    assert _tree(output) == {"race-winner.txt": b"preserve me"}
