import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import build_r052_evidence_panel as builder
from scripts import promote_r052_reviewed_sources as promoter
from scripts import score_r052_evidence_retrieval as scorer

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "sources" / "r052" / "ntsb_gao_v1.json"


@pytest.fixture(autouse=True)
def _compact_test_token_floor(monkeypatch):
    monkeypatch.setattr(promoter, "MIN_REVIEWED_WORD_TOKENS", 64)
    monkeypatch.setattr(scorer, "MIN_REVIEWED_WORD_TOKENS", 64)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _line_end(text: str, line_number: int = 1) -> int:
    offset = 0
    for _ in range(line_number):
        offset = text.index("\n", offset) + 1
    return offset


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


def _genuine_acquisition_receipt(manifest: dict, manifest_sha: str) -> dict:
    return {
        "schema_version": promoter.ACQUISITION_RECEIPT_VERSION,
        "mode": "verify-only",
        "manifest_sha256": manifest_sha,
        "verified": [
            {
                "id": snapshot["id"],
                "raw_sha256": snapshot["raw_pdf"]["sha256"],
                "text_sha256": snapshot["extracted_text"]["sha256"],
            }
            for snapshot in manifest["snapshots"]
        ],
    }


def _write_acquisition_receipt(fixture: dict, *, bind_review: bool = True) -> None:
    payload = promoter.canonical_json(fixture["acquisition_receipt"])
    fixture["acquisition_receipt_path"].write_bytes(payload)
    fixture["acquisition_receipt_sha"] = _sha(payload)
    if bind_review and "review" in fixture:
        fixture["review"]["acquisition_receipt_sha256"] = fixture[
            "acquisition_receipt_sha"
        ]
        _write_review(fixture)


def _write_acquisition(fixture: dict, *, bind_review: bool = True) -> None:
    payload = promoter.canonical_json(fixture["manifest"])
    fixture["manifest_path"].write_bytes(payload)
    fixture["manifest_sha"] = _sha(payload)
    if "acquisition_receipt" in fixture:
        fixture["acquisition_receipt"] = _genuine_acquisition_receipt(
            fixture["manifest"], fixture["manifest_sha"]
        )
        _write_acquisition_receipt(fixture, bind_review=False)
    if bind_review and "review" in fixture:
        fixture["review"]["acquisition_manifest_sha256"] = fixture["manifest_sha"]
        fixture["review"]["acquisition_receipt_sha256"] = fixture[
            "acquisition_receipt_sha"
        ]
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
    text_dir.mkdir(parents=True)
    texts = {}
    for index, snapshot in enumerate(manifest["snapshots"]):
        line = (
            f"Official government source {snapshot['id']} record {index}. "
            f"This retained passage contains verified operational details {index}.\n"
        )
        text = line * 12
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
    acquisition_receipt_path = tmp_path / "acquisition-receipt.json"
    review_path = tmp_path / "review.json"
    fixture = {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "acquisition_receipt_path": acquisition_receipt_path,
        "text_dir": text_dir,
        "texts": texts,
        "review_path": review_path,
        "output": tmp_path / "promoted",
    }
    _write_acquisition(fixture, bind_review=False)
    fixture["acquisition_receipt"] = _genuine_acquisition_receipt(
        manifest, fixture["manifest_sha"]
    )
    _write_acquisition_receipt(fixture, bind_review=False)
    fixture["review"] = {
        "schema_version": promoter.REVIEW_SCHEMA_VERSION,
        "acquisition_manifest_sha256": fixture["manifest_sha"],
        "acquisition_receipt_sha256": fixture["acquisition_receipt_sha"],
        "review_date": datetime.now(tz=UTC).date().isoformat(),
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
        fixture["acquisition_receipt_path"],
        fixture["acquisition_receipt_sha"],
        fixture["text_dir"],
        fixture["review_path"],
        fixture["review_sha"],
        fixture["output"],
    )


def test_happy_path_emits_deterministic_v5_accepted_by_builder_and_scorer(tmp_path):
    fixture = _fixture(tmp_path)

    receipt = _promote(fixture)

    output = fixture["output"]
    manifest_raw = (output / "source-manifest-v5.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == promoter.canonical_json(manifest)
    assert receipt["semantic_review_verified_by_tool"] is False
    assert receipt["acquisition_receipt_sha256"] == fixture["acquisition_receipt_sha"]
    assert receipt["source_manifest_sha256"] == _sha(manifest_raw)
    assert manifest["schema_version"] == "r052-source-provenance-manifest-v5"
    assert (
        manifest["acquisition_manifest_artifact"]
        == promoter.ACQUISITION_MANIFEST_ARTIFACT_NAME
    )
    assert (
        manifest["acquisition_run_receipt_artifact"]
        == promoter.ACQUISITION_RUN_ARTIFACT_NAME
    )
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
    run_name = manifest["acquisition_run_receipt_artifact"]
    acquisition_name = manifest["acquisition_manifest_artifact"]
    assert manifest["artifacts"][acquisition_name] == {
        "role": promoter.ACQUISITION_MANIFEST_ARTIFACT_ROLE,
        "sha256": fixture["manifest_sha"],
    }
    assert (
        artifact_paths[acquisition_name].read_bytes()
        == fixture["manifest_path"].read_bytes()
    )
    assert manifest["artifacts"][run_name] == {
        "role": promoter.ACQUISITION_RUN_ARTIFACT_ROLE,
        "sha256": fixture["acquisition_receipt_sha"],
    }
    assert (
        artifact_paths[run_name].read_bytes()
        == fixture["acquisition_receipt_path"].read_bytes()
    )
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
    assert len(by_hash) == 22

    first_id = next(iter(manifest["snapshots"]))
    source_record = receipt["sources"][first_id]
    reviewed = artifact_paths[source_record["reviewed_text_artifact"]].read_text()
    expected = fixture["texts"][first_id]
    assert reviewed == expected
    acquisition_record = json.loads(
        artifact_paths[source_record["acquisition_binding_artifact"]].read_bytes()
    )
    binding_name = source_record["acquisition_binding_artifact"]
    assert (
        manifest["artifacts"][binding_name]["role"]
        == promoter.ACQUISITION_BINDING_ARTIFACT_ROLE
    )
    assert acquisition_record["schema_version"] == promoter.ACQUISITION_BINDING_VERSION
    assert acquisition_record["schema_version"] != promoter.ACQUISITION_RECEIPT_VERSION
    assert (
        acquisition_record["acquisition_receipt_sha256"]
        == fixture["acquisition_receipt_sha"]
    )
    assert acquisition_record["review_receipt_sha256"] == fixture["review_sha"]
    assert acquisition_record["reviewed_text_sha256"] == _sha(reviewed.encode())
    assert acquisition_record["reviewed_character_count"] == len(reviewed)
    assert acquisition_record["reviewed_unicode_word_token_count"] == len(
        promoter.UNICODE_WORD_RE.findall(reviewed)
    )
    assert (
        acquisition_record["word_token_count_method"]
        == promoter.WORD_TOKEN_COUNT_METHOD
    )
    completion = json.loads((output / "PROMOTION_COMPLETE.json").read_bytes())
    assert completion["source_manifest_sha256"] == receipt["source_manifest_sha256"]


def test_noncanonical_genuine_manifest_bytes_are_copied_exactly(tmp_path):
    fixture = _fixture(tmp_path)
    payload = (
        json.dumps(fixture["manifest"], ensure_ascii=False, indent=2) + "\n"
    ).encode()
    assert payload != promoter.canonical_json(fixture["manifest"])
    fixture["manifest_path"].write_bytes(payload)
    fixture["manifest_sha"] = _sha(payload)
    fixture["acquisition_receipt"] = _genuine_acquisition_receipt(
        fixture["manifest"], fixture["manifest_sha"]
    )
    _write_acquisition_receipt(fixture, bind_review=False)
    fixture["review"]["acquisition_manifest_sha256"] = fixture["manifest_sha"]
    fixture["review"]["acquisition_receipt_sha256"] = fixture["acquisition_receipt_sha"]
    _write_review(fixture)

    receipt = _promote(fixture)
    copied = (
        fixture["output"]
        / receipt["artifact_paths"][promoter.ACQUISITION_MANIFEST_ARTIFACT_NAME]
    )
    assert copied.read_bytes() == payload
    manifest = json.loads((fixture["output"] / "source-manifest-v5.json").read_bytes())
    assert manifest["artifacts"][promoter.ACQUISITION_MANIFEST_ARTIFACT_NAME][
        "sha256"
    ] == _sha(payload)


def test_generated_manifest_v4_requires_migration_in_current_consumers(tmp_path):
    fixture = _fixture(tmp_path)
    receipt = _promote(fixture)
    output = fixture["output"]
    manifest = json.loads((output / "source-manifest-v5.json").read_bytes())
    artifact_paths = {
        name: output / relative for name, relative in receipt["artifact_paths"].items()
    }
    manifest["schema_version"] = "r052-source-provenance-manifest-v4"

    with pytest.raises(ValueError, match="migration to v5"):
        builder._validate_source_manifest(manifest, artifact_paths)
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
    with pytest.raises(ValueError, match="migration to v5"):
        scorer.validate_source_snapshot_manifest(
            panel, manifest, scorer.sha256_json(manifest), artifact_paths
        )


@pytest.mark.parametrize(
    "hash_field", ["manifest_sha", "acquisition_receipt_sha", "review_sha"]
)
def test_external_hashes_are_mandatory_and_exact(tmp_path, hash_field):
    fixture = _fixture(tmp_path)
    fixture[hash_field] = "0" * 64

    with pytest.raises(promoter.PromotionError, match="external SHA-256"):
        _promote(fixture)
    assert not fixture["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt["verified"].pop(), "all ten snapshots"),
        (
            lambda receipt: receipt["verified"][0].__setitem__("id", "HAR0302"),
            "exactly match acquisition manifest hashes",
        ),
        (
            lambda receipt: receipt["verified"][0].__setitem__("raw_sha256", "0" * 64),
            "exactly match acquisition manifest hashes",
        ),
        (
            lambda receipt: receipt["verified"][0].__setitem__("text_sha256", "0" * 64),
            "exactly match acquisition manifest hashes",
        ),
        (lambda receipt: receipt.__setitem__("mode", "unknown"), "mode is invalid"),
        (
            lambda receipt: receipt.__setitem__("unexpected", True),
            "keys must be exactly",
        ),
        (
            lambda receipt: receipt.__setitem__("manifest_sha256", "0" * 64),
            "manifest hash mismatch",
        ),
    ],
)
def test_genuine_whole_run_acquisition_receipt_is_required(tmp_path, mutation, message):
    fixture = _fixture(tmp_path)
    mutation(fixture["acquisition_receipt"])
    _write_acquisition_receipt(fixture)

    with pytest.raises(promoter.PromotionError, match=message):
        _promote(fixture)
    assert not fixture["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda receipt, _text: receipt.__setitem__(
                "reviewers", ["Same-Reviewer", "same-reviewer"]
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
                    {"char_start": 0, "char_end": _line_end(text, 2)},
                    {"char_start": _line_end(text), "char_end": len(text)},
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


@pytest.mark.parametrize(
    "reviewer_id", [" reviewer-01", "reviewer-01 ", "bad id", "bad\x7fid"]
)
def test_reviewer_ids_are_strict_normalized_identifiers(tmp_path, reviewer_id):
    fixture = _fixture(tmp_path)
    fixture["review"]["reviewers"][0] = reviewer_id
    _write_review(fixture)

    with pytest.raises(promoter.PromotionError, match="normalized .* ASCII identifier"):
        _promote(fixture)


def test_future_review_date_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["review"]["review_date"] = (
        datetime.now(tz=UTC).date() + timedelta(days=1)
    ).isoformat()
    _write_review(fixture)

    with pytest.raises(promoter.PromotionError, match="future"):
        _promote(fixture)


def test_review_receipt_must_bind_acquisition_lineage(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["review"]["acquisition_manifest_sha256"] = "1" * 64
    _write_review(fixture)

    with pytest.raises(promoter.PromotionError, match="acquisition-manifest hash"):
        _promote(fixture)

    fixture = _fixture(tmp_path / "receipt-hash")
    fixture["review"]["acquisition_receipt_sha256"] = "2" * 64
    _write_review(fixture)
    with pytest.raises(promoter.PromotionError, match="acquisition-receipt hash"):
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
        (b" \n%PDF-1.7\n" + b"government record " * 30, "not PDF/binary"),
        (b"\xef\xbb\xbfgovernment record " * 30, "not PDF/binary"),
        (b"PK\x03\x04government record " * 30, "not PDF/binary"),
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
    text = fixture["texts"][fixture["manifest"]["snapshots"][0]["id"]]
    fixture["review"]["sources"][0]["retained_ranges"] = [
        {"char_start": 0, "char_end": _line_end(text)}
    ]
    _write_review(fixture)

    with pytest.raises(promoter.PromotionError, match="fewer than 64"):
        _promote(fixture)


def test_production_word_floor_is_8192_and_is_not_a_model_token_claim():
    assert promoter.PRODUCTION_MIN_REVIEWED_WORD_TOKENS == 8_192
    assert promoter.WORD_TOKEN_COUNT_METHOD.endswith("not-model-tokenizer")


def test_ranges_must_be_line_aligned_and_mapping_prevents_semantic_joins(tmp_path):
    fixture = _fixture(tmp_path)
    first = fixture["manifest"]["snapshots"][0]
    text = fixture["texts"][first["id"]]
    fixture["review"]["sources"][0]["retained_ranges"] = [
        {"char_start": 1, "char_end": len(text)}
    ]
    _write_review(fixture)
    with pytest.raises(promoter.PromotionError, match="complete LF-delimited lines"):
        _promote(fixture)

    fixture = _fixture(tmp_path / "partial-end")
    fixture["review"]["sources"][0]["retained_ranges"] = [
        {"char_start": 0, "char_end": 20}
    ]
    _write_review(fixture)
    with pytest.raises(promoter.PromotionError, match="complete LF-delimited lines"):
        _promote(fixture)

    fixture = _fixture(tmp_path / "mapped")
    first = fixture["manifest"]["snapshots"][0]
    text = fixture["texts"][first["id"]]
    first_end = _line_end(text, 4)
    second_start = _line_end(text, 8)
    ranges = [
        {"char_start": 0, "char_end": first_end},
        {"char_start": second_start, "char_end": len(text)},
    ]
    fixture["review"]["sources"][0]["retained_ranges"] = ranges
    _write_review(fixture)
    receipt = _promote(fixture)
    record = receipt["sources"][first["id"]]
    expected = text[:first_end] + text[second_start:]
    reviewed_path = (
        fixture["output"] / receipt["artifact_paths"][record["reviewed_text_artifact"]]
    )
    assert reviewed_path.read_text() == expected
    assert record["range_mappings"] == [
        {
            "source_char_start": 0,
            "source_char_end": first_end,
            "output_char_start": 0,
            "output_char_end": first_end,
        },
        {
            "source_char_start": second_start,
            "source_char_end": len(text),
            "output_char_start": first_end,
            "output_char_end": len(expected),
        },
    ]


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


def test_existing_empty_output_directory_is_not_race_overwritten(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["output"].mkdir()

    with pytest.raises(promoter.PromotionError, match="never overwrites"):
        _promote(fixture)
    assert list(fixture["output"].iterdir()) == []


def test_concurrent_empty_output_reservation_is_not_overwritten(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    output = fixture["output"]
    real_mkdir = Path.mkdir

    def reserve_before_publish(path, *args, **kwargs):
        if path == output and not path.exists():
            real_mkdir(path, *args, **kwargs)
            (path / "owned-by-racer.txt").write_text("keep")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", reserve_before_publish)
    with pytest.raises(promoter.PromotionError, match="never overwrites"):
        _promote(fixture)
    assert (fixture["output"] / "owned-by-racer.txt").read_text() == "keep"


def test_completion_failure_leaves_owned_incomplete_output(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    real_write = promoter._write

    def fail_completion(path, payload):
        if path.name == "PROMOTION_COMPLETE.json":
            raise OSError("injected completion failure")
        return real_write(path, payload)

    monkeypatch.setattr(promoter, "_write", fail_completion)
    with pytest.raises(OSError, match="injected completion failure"):
        _promote(fixture)
    assert fixture["output"].is_dir()
    assert not (fixture["output"] / "PROMOTION_COMPLETE.json").exists()


def test_replaced_output_cannot_be_reported_as_success(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    real_write = promoter._write

    def replace_before_completion(path, payload):
        if path.name == "PROMOTION_COMPLETE.json":
            promoter.shutil.rmtree(path.parent)
            path.parent.mkdir()
            (path.parent / "owned-by-racer.txt").write_text("keep")
        return real_write(path, payload)

    monkeypatch.setattr(promoter, "_write", replace_before_completion)
    with pytest.raises(promoter.PromotionError, match="identity changed"):
        _promote(fixture)
    assert (fixture["output"] / "owned-by-racer.txt").read_text() == "keep"
    assert not (fixture["output"] / "source-manifest-v5.json").exists()


def test_copy_failure_never_deletes_replacement_directory(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)

    def replace_then_fail(_source, destination, *, dirs_exist_ok):
        assert dirs_exist_ok is True
        promoter.shutil.rmtree(destination)
        destination.mkdir()
        (destination / "owned-by-racer.txt").write_text("keep")
        raise OSError("injected copy failure after replacement")

    monkeypatch.setattr(promoter.shutil, "copytree", replace_then_fail)
    with pytest.raises(OSError, match="injected copy failure"):
        _promote(fixture)
    assert (fixture["output"] / "owned-by-racer.txt").read_text() == "keep"
