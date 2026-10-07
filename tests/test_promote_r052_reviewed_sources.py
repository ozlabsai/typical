import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import build_r052_evidence_panel as builder
from scripts import promote_r052_reviewed_sources as promoter
from scripts import score_r052_evidence_retrieval as scorer

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "sources" / "r052" / "ntsb_gao_v1.json"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _review_policy(policy: dict) -> dict:
    return {
        "source_family": policy["source_family"],
        "acquisition_license_id": policy["license_id"],
        "manifest_license_id": promoter.MANIFEST_LICENSE_ID,
        "license_url": policy["policy_url"],
        "scope": policy["scope"],
        "attribution": policy["attribution"],
        "license_cleared": True,
        "government_staff_authored_only": True,
        "third_party_material_excluded": True,
        "evaluation_use_cleared": True,
        "redistribution_cleared": True,
    }


def _source_review(snapshot: dict, text: str) -> dict:
    return {
        "snapshot_id": snapshot["id"],
        "license_policy": snapshot["license_policy"],
        "retained_ranges": [{"char_start": 0, "char_end": len(text)}],
        "government_staff_authored_only": True,
        "third_party_material_excluded": True,
        "evaluation_use_cleared": True,
        "redistribution_cleared": True,
        "unresolved_concerns": [],
    }


def _write_acquisition(fixture: dict, *, bind_review: bool = True) -> None:
    payload = promoter.canonical_json(fixture["manifest"])
    fixture["manifest_path"].write_bytes(payload)
    fixture["manifest_sha"] = _sha(payload)
    if bind_review:
        fixture["review"]["acquisition_manifest_sha256"] = fixture["manifest_sha"]
        _write_review(fixture)


def _write_review(fixture: dict) -> None:
    payload = promoter.canonical_json(fixture["review"])
    fixture["review_path"].write_bytes(payload)
    fixture["review_sha"] = _sha(payload)


def _replace_source_payload(fixture: dict, index: int, payload: bytes) -> None:
    snapshot = fixture["manifest"]["snapshots"][index]
    path = fixture["text_dir"] / snapshot["extracted_text"]["filename"]
    path.write_bytes(payload)
    text = payload.decode("utf-8")
    snapshot["extracted_text"].update(
        {"sha256": _sha(payload), "bytes": len(payload), "words": len(text.split())}
    )
    fixture["texts"][snapshot["id"]] = text
    fixture["review"]["sources"][index]["retained_ranges"] = [
        {"char_start": 0, "char_end": len(text)}
    ]
    _write_acquisition(fixture)


def _fixture(tmp_path: Path) -> dict:
    manifest = deepcopy(json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8")))
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    texts = {}
    for index, snapshot in enumerate(manifest["snapshots"]):
        text = (
            f"Official government source {snapshot['id']} record {index}. "
            f"This retained passage contains verified operational details {index}. "
        ) * 8
        payload = text.encode("utf-8")
        texts[snapshot["id"]] = text
        snapshot["extracted_text"].update(
            {
                "sha256": _sha(payload),
                "bytes": len(payload),
                "words": len(text.split()),
            }
        )
        (text_dir / snapshot["extracted_text"]["filename"]).write_bytes(payload)

    manifest_path = tmp_path / "acquisition.json"
    review_path = tmp_path / "review.json"
    fixture = {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "text_dir": text_dir,
        "texts": texts,
        "review_path": review_path,
        "output": tmp_path / "promoted",
    }
    _write_acquisition(fixture, bind_review=False)
    fixture["review"] = {
        "schema_version": promoter.REVIEW_SCHEMA_VERSION,
        "acquisition_manifest_sha256": fixture["manifest_sha"],
        "review_date": "2026-10-07",
        "reviewers": ["reviewer-01", "reviewer-02"],
        "adjudicator_id": "adjudicator-01",
        "license_policies": {
            name: _review_policy(policy)
            for name, policy in manifest["license_policies"].items()
        },
        "sources": [
            _source_review(snapshot, texts[snapshot["id"]])
            for snapshot in manifest["snapshots"]
        ],
    }
    _write_review(fixture)
    return fixture


def _promote(fixture: dict) -> dict:
    return promoter.promote(
        fixture["manifest_path"],
        fixture["manifest_sha"],
        fixture["text_dir"],
        fixture["review_path"],
        fixture["review_sha"],
        fixture["output"],
    )


def test_happy_path_emits_deterministic_v3_accepted_by_builder_and_scorer(tmp_path):
    fixture = _fixture(tmp_path)

    receipt = _promote(fixture)

    output = fixture["output"]
    manifest_raw = (output / "source-manifest-v3.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == promoter.canonical_json(manifest)
    assert receipt["semantic_review_verified_by_tool"] is False
    assert receipt["source_manifest_sha256"] == _sha(manifest_raw)
    assert len(manifest["snapshots"]) == 10
    assert set(manifest["license_policies"]) == {
        "ntsb_staff_text",
        "gao_government_work",
    }
    assert {
        policy["license_id"] for policy in manifest["license_policies"].values()
    } == {promoter.MANIFEST_LICENSE_ID}

    artifact_paths = {
        name: output / relative for name, relative in receipt["artifact_paths"].items()
    }
    assert set(builder._validate_source_manifest(manifest, artifact_paths)) == set(
        artifact_paths
    )
    panel = {
        "cases": [
            {
                "source_snapshot_sha256": manifest["artifacts"][
                    snapshot["reviewed_text_artifact"]
                ]["sha256"],
                "source_family": snapshot["source_family"],
            }
            for snapshot in manifest["snapshots"].values()
        ]
    }
    digest = scorer.sha256_json(manifest)
    validated_digest, by_hash = scorer.validate_source_snapshot_manifest(
        panel, manifest, digest, artifact_paths
    )
    assert validated_digest == digest
    assert len(by_hash) == 20

    first_id = next(iter(manifest["snapshots"]))
    source_record = receipt["sources"][first_id]
    reviewed = artifact_paths[source_record["reviewed_text_artifact"]].read_text()
    expected = fixture["texts"][first_id]
    assert reviewed == expected
    acquisition_record = json.loads(
        artifact_paths[source_record["acquisition_receipt_artifact"]].read_bytes()
    )
    assert acquisition_record["review_receipt_sha256"] == fixture["review_sha"]
    assert acquisition_record["reviewed_text_sha256"] == _sha(reviewed.encode())


@pytest.mark.parametrize("which", ["acquisition", "review"])
def test_external_hashes_are_mandatory_and_exact(tmp_path, which):
    fixture = _fixture(tmp_path)
    fixture[f"{which}_sha" if which == "review" else "manifest_sha"] = "0" * 64

    with pytest.raises(promoter.PromotionError, match="external SHA-256"):
        _promote(fixture)
    assert not fixture["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda receipt, _text: receipt.__setitem__(
                "reviewers", ["same-reviewer", "same-reviewer"]
            ),
            "IDs must be distinct",
        ),
        (
            lambda receipt, _text: receipt["license_policies"][
                "ntsb_staff_text"
            ].__setitem__("redistribution_cleared", False),
            "rights assertions",
        ),
        (
            lambda receipt, _text: receipt["sources"][0].__setitem__(
                "government_staff_authored_only", False
            ),
            "must assert government_staff_authored_only",
        ),
        (
            lambda receipt, _text: receipt["sources"][0].__setitem__(
                "unresolved_concerns", ["possible third-party table"]
            ),
            "unresolved concerns",
        ),
        (
            lambda receipt, _text: receipt["sources"].pop(),
            "cover all snapshot IDs exactly",
        ),
        (
            lambda receipt, text: receipt["sources"][0].__setitem__(
                "retained_ranges",
                [
                    {"char_start": 0, "char_end": len(text) - 1},
                    {"char_start": 10, "char_end": len(text)},
                ],
            ),
            "ordered and non-overlapping",
        ),
        (
            lambda receipt, text: receipt["sources"][0].__setitem__(
                "retained_ranges", [{"char_start": 0, "char_end": len(text) + 1}]
            ),
            "outside the source text",
        ),
    ],
)
def test_review_claims_coverage_and_ranges_fail_closed(tmp_path, mutation, message):
    fixture = _fixture(tmp_path)
    first_id = fixture["manifest"]["snapshots"][0]["id"]
    mutation(fixture["review"], fixture["texts"][first_id])
    _write_review(fixture)

    with pytest.raises(promoter.PromotionError, match=message):
        _promote(fixture)
    assert not fixture["output"].exists()


def test_review_receipt_must_bind_acquisition_lineage(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["review"]["acquisition_manifest_sha256"] = "1" * 64
    _write_review(fixture)

    with pytest.raises(promoter.PromotionError, match="acquisition-manifest hash"):
        _promote(fixture)


def test_source_tamper_after_receipts_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    first = fixture["manifest"]["snapshots"][0]
    path = fixture["text_dir"] / first["extracted_text"]["filename"]
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(promoter.PromotionError, match="hash/size mismatch"):
        _promote(fixture)


def test_symlink_source_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    first = fixture["manifest"]["snapshots"][0]
    path = fixture["text_dir"] / first["extracted_text"]["filename"]
    target = tmp_path / "outside.txt"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(promoter.PromotionError, match="symlink"):
        _promote(fixture)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"%PDF-1.7\n" + b"government record " * 30, "not PDF/binary"),
        (b"government\x00record " * 30, "not PDF/binary"),
        (b"government\x7frecord " * 30, "binary control"),
        (b"copied from JevBench " * 30, "benchmark lineage"),
    ],
)
def test_pdf_binary_and_benchmark_lineage_are_rejected(tmp_path, payload, message):
    fixture = _fixture(tmp_path)
    _replace_source_payload(fixture, 0, payload)

    with pytest.raises(promoter.PromotionError, match=message):
        _promote(fixture)


def test_acquisition_filename_path_traversal_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["manifest"]["snapshots"][0]["extracted_text"]["filename"] = "../AAR0002.txt"
    _write_acquisition(fixture)

    with pytest.raises(promoter.PromotionError, match="traversal-free"):
        _promote(fixture)


def test_too_little_retained_content_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["review"]["sources"][0]["retained_ranges"] = [
        {"char_start": 0, "char_end": 20}
    ]
    _write_review(fixture)

    with pytest.raises(promoter.PromotionError, match="fewer than 256"):
        _promote(fixture)


def test_noncanonical_duplicate_and_nonfinite_review_json_are_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    canonical = fixture["review_path"].read_bytes()
    hostile_payloads = [
        json.dumps(fixture["review"], indent=2).encode(),
        canonical.replace(b"{", b'{"schema_version":"duplicate",', 1),
        canonical.replace(b"{", b'{"unexpected":NaN,', 1),
    ]
    messages = ["canonical JSON", "duplicate JSON key", "non-finite JSON"]
    for payload, message in zip(hostile_payloads, messages, strict=True):
        fixture["review_path"].write_bytes(payload)
        fixture["review_sha"] = _sha(payload)
        with pytest.raises(promoter.PromotionError, match=message):
            _promote(fixture)
        assert not fixture["output"].exists()


def test_existing_output_directory_is_never_overwritten(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["output"].mkdir()
    sentinel = fixture["output"] / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(promoter.PromotionError, match="never overwrites"):
        _promote(fixture)
    assert sentinel.read_text() == "keep"
