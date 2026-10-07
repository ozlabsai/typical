#!/usr/bin/env python3
r"""Promote externally reviewed R052 acquisitions into source manifest v4.

This bridge verifies bytes and enforces a closed review-receipt contract.  It
does not determine whether the retained text is actually government-authored
or whether a human review was competently performed; those are externally
authored assertions bound by the supplied review receipt.

The production size floor counts deterministic Unicode ``\w+`` spans.  It is
only a readiness guard and is not a model-tokenizer or context-window check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from scripts import fetch_r052_source_snapshot as acquisition
    from scripts import score_r052_evidence_retrieval as scorer
except ModuleNotFoundError:  # Supports ``uv run python scripts/promote_...py``.
    import fetch_r052_source_snapshot as acquisition
    import score_r052_evidence_retrieval as scorer

REVIEW_SCHEMA_VERSION = "r052-reviewed-source-review-v1"
ACQUISITION_RECEIPT_VERSION = scorer.ACQUISITION_RUN_RECEIPT_VERSION
ACQUISITION_BINDING_VERSION = scorer.ACQUISITION_BINDING_VERSION
PROMOTION_RECEIPT_VERSION = "r052-reviewed-source-promotion-receipt-v1"
COMPLETION_MARKER_VERSION = "r052-reviewed-source-promotion-complete-v1"
SOURCE_MANIFEST_VERSION = scorer.SOURCE_MANIFEST_VERSION
SOURCE_LICENSE_POLICY = scorer.SOURCE_LICENSE_POLICY
MANIFEST_LICENSE_ID = "LicenseRef-USGov-Public-Domain"
ACQUISITION_RUN_ARTIFACT_NAME = "acquisition-run"
ACQUISITION_RUN_ARTIFACT_ROLE = "source_acquisition_run_receipt"
ACQUISITION_BINDING_ARTIFACT_ROLE = "reviewed_source_acquisition_binding"
PRODUCTION_MIN_REVIEWED_WORD_TOKENS = scorer.MIN_REVIEWED_WORD_TOKENS
MIN_REVIEWED_WORD_TOKENS = PRODUCTION_MIN_REVIEWED_WORD_TOKENS
WORD_TOKEN_COUNT_METHOD = scorer.WORD_TOKEN_COUNT_METHOD
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVIEWER_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@+-]{0,126}[A-Za-z0-9])?$")
UNICODE_WORD_RE = re.compile(r"\w+", re.UNICODE)
FORBIDDEN_MARKERS = tuple(
    marker.encode() for marker in scorer.FORBIDDEN_LINEAGE_MARKERS
)
BOMS = (
    b"\xef\xbb\xbf",
    b"\xff\xfe\x00\x00",
    b"\x00\x00\xfe\xff",
    b"\xff\xfe",
    b"\xfe\xff",
    b"+/v8",
    b"+/v9",
    b"+/v+",
    b"+/v/",
    b"\xf7\x64\x4c",
    b"\xdd\x73\x66\x73",
    b"\x0e\xfe\xff",
    b"\xfb\xee\x28",
    b"\x84\x31\x95\x33",
)
BINARY_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
    b"\x7fELF",
    b"\xff\xd8\xff",
    b"\x1f\x8b",
    b"MZ",
    b"SQLite format 3\x00",
    b"Rar!\x1a\x07",
    b"7z\xbc\xaf\x27\x1c",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
)

REVIEW_KEYS = {
    "schema_version",
    "acquisition_manifest_sha256",
    "acquisition_receipt_sha256",
    "review_date",
    "reviewers",
    "adjudicator_id",
    "license_policies",
    "sources",
}
REVIEW_POLICY_KEYS = {
    "source_family",
    "acquisition_license_id",
    "manifest_license_id",
    "license_url",
    "scope",
    "attribution",
    "license_cleared",
    "government_staff_authored_only",
    "third_party_material_excluded",
    "evaluation_use_cleared",
    "redistribution_cleared",
}
REVIEW_SOURCE_KEYS = {
    "snapshot_id",
    "license_policy",
    "retained_ranges",
    "government_staff_authored_only",
    "third_party_material_excluded",
    "evaluation_use_cleared",
    "redistribution_cleared",
    "unresolved_concerns",
}
RANGE_KEYS = {"char_start", "char_end"}
ACQUISITION_RECEIPT_KEYS = {
    "schema_version",
    "mode",
    "manifest_sha256",
    "verified",
}
ACQUISITION_VERIFIED_KEYS = {"id", "raw_sha256", "text_sha256"}
ACQUISITION_MODES = {"fetch", "offline", "verify-only"}


class PromotionError(ValueError):
    """An input cannot be safely promoted under the closed contract."""


def canonical_json(value: Any) -> bytes:
    """Encode deterministic UTF-8 JSON used for all generated records."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise PromotionError(f"{label} keys must be exactly {sorted(expected)}")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PromotionError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionError(f"{label} must be non-empty text")
    return value


def _reviewer_id(value: Any, label: str) -> str:
    value = _text(value, label)
    if (
        value != value.strip()
        or unicodedata.normalize("NFKC", value) != value
        or REVIEWER_ID_RE.fullmatch(value) is None
        or any(unicodedata.category(char).startswith("C") for char in value)
    ):
        raise PromotionError(
            f"{label} must be a normalized 1-128 character ASCII identifier"
        )
    return value


def _reject_markers(payload: bytes, label: str) -> None:
    folded = payload.lower()
    marker = next((item.decode() for item in FORBIDDEN_MARKERS if item in folded), None)
    if marker is not None:
        raise PromotionError(
            f"{label} contains forbidden benchmark lineage marker {marker!r}"
        )


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise PromotionError(f"{label} may not contain symlink components")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _parse_unique_finite_json(raw: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PromotionError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PromotionError(f"{label} contains non-finite JSON value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"{label} must be valid UTF-8 JSON") from error


def _read_hashed_json(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    require_canonical: bool,
) -> tuple[Any, bytes, str]:
    _reject_symlink_components(path, label)
    if not path.is_file():
        raise PromotionError(f"{label} must be a regular file")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != _sha(expected_sha256, f"expected {label} hash"):
        raise PromotionError(f"{label} differs from its external SHA-256")
    _reject_markers(raw, label)
    value = _parse_unique_finite_json(raw, label)
    if require_canonical and raw != canonical_json(value):
        raise PromotionError(f"{label} must use canonical JSON bytes")
    return value, raw, digest


def _validate_text_payload(payload: bytes, label: str) -> str:
    prefix = payload[:1024].lstrip(b" \t\r\n\f\v")
    if (
        payload.startswith(BOMS)
        or prefix.startswith(BOMS)
        or b"%PDF-" in payload[:1024]
        or prefix.startswith(BINARY_SIGNATURES)
        or b"\x00" in payload
    ):
        raise PromotionError(f"{label} must be extracted text, not PDF/binary data")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PromotionError(f"{label} must be valid UTF-8 text") from error
    if any(
        unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in text
    ):
        raise PromotionError(f"{label} contains binary control characters")
    if not text.strip():
        raise PromotionError(f"{label} must contain non-empty text")
    _reject_markers(payload, label)
    return text


def _validate_acquisition_receipt(
    receipt: Any, manifest: Mapping[str, Any], manifest_sha256: str
) -> None:
    if not isinstance(receipt, dict):
        raise PromotionError("acquisition receipt must be an object")
    _exact_keys(receipt, ACQUISITION_RECEIPT_KEYS, "acquisition receipt")
    if receipt["schema_version"] != ACQUISITION_RECEIPT_VERSION:
        raise PromotionError(
            f"acquisition receipt schema must be {ACQUISITION_RECEIPT_VERSION}"
        )
    if not isinstance(receipt["mode"], str) or receipt["mode"] not in ACQUISITION_MODES:
        raise PromotionError("acquisition receipt mode is invalid")
    if receipt["manifest_sha256"] != manifest_sha256:
        raise PromotionError("acquisition receipt manifest hash mismatch")
    verified = receipt["verified"]
    snapshots = manifest["snapshots"]
    if not isinstance(verified, list) or len(verified) != len(snapshots):
        raise PromotionError("acquisition receipt must verify all ten snapshots")
    for index, (record, snapshot) in enumerate(zip(verified, snapshots, strict=True)):
        label = f"acquisition receipt verified[{index}]"
        if not isinstance(record, dict):
            raise PromotionError(f"{label} must be an object")
        _exact_keys(record, ACQUISITION_VERIFIED_KEYS, label)
        expected = {
            "id": snapshot["id"],
            "raw_sha256": snapshot["raw_pdf"]["sha256"],
            "text_sha256": snapshot["extracted_text"]["sha256"],
        }
        if record != expected:
            raise PromotionError(
                f"{label} does not exactly match acquisition manifest hashes"
            )


def _validate_review_policy(
    name: str, review_policy: Any, acquired_policy: Mapping[str, Any]
) -> dict[str, Any]:
    label = f"review license policy {name}"
    if not isinstance(review_policy, dict):
        raise PromotionError(f"{label} must be an object")
    _exact_keys(review_policy, REVIEW_POLICY_KEYS, label)
    expected_values = {
        "source_family": acquired_policy["source_family"],
        "acquisition_license_id": acquired_policy["license_id"],
        "manifest_license_id": MANIFEST_LICENSE_ID,
        "license_url": acquired_policy["policy_url"],
        "scope": acquired_policy["scope"],
        "attribution": acquired_policy["attribution"],
    }
    if any(review_policy[field] != value for field, value in expected_values.items()):
        raise PromotionError(f"{label} does not exactly match acquisition scope")
    required_true = REVIEW_POLICY_KEYS - set(expected_values)
    if any(review_policy[field] is not True for field in required_true):
        raise PromotionError(f"{label} rights assertions must all be true")
    return {
        "source_family": review_policy["source_family"],
        "license_id": review_policy["manifest_license_id"],
        "license_url": review_policy["license_url"],
        "attribution": review_policy["attribution"],
        "license_cleared": True,
        "evaluation_use_cleared": True,
        "redistribution_cleared": True,
    }


def _validate_review_receipt(
    receipt: Any,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    acquisition_receipt_sha256: str,
    source_texts: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(receipt, dict):
        raise PromotionError("review receipt must be an object")
    _exact_keys(receipt, REVIEW_KEYS, "review receipt")
    if receipt["schema_version"] != REVIEW_SCHEMA_VERSION:
        raise PromotionError(f"review receipt schema must be {REVIEW_SCHEMA_VERSION}")
    if receipt["acquisition_manifest_sha256"] != manifest_sha256:
        raise PromotionError("review receipt acquisition-manifest hash mismatch")
    if receipt["acquisition_receipt_sha256"] != acquisition_receipt_sha256:
        raise PromotionError("review receipt acquisition-receipt hash mismatch")
    try:
        parsed_date = date.fromisoformat(receipt["review_date"])
    except (TypeError, ValueError) as error:
        raise PromotionError("review_date must be an ISO YYYY-MM-DD date") from error
    if parsed_date.isoformat() != receipt["review_date"]:
        raise PromotionError("review_date must use canonical YYYY-MM-DD form")
    if parsed_date > datetime.now(tz=UTC).date():
        raise PromotionError("review_date may not be in the future")

    reviewers = receipt["reviewers"]
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise PromotionError("review receipt must identify exactly two reviewers")
    reviewer_ids = [_reviewer_id(value, "reviewer id") for value in reviewers]
    adjudicator_id = _reviewer_id(receipt["adjudicator_id"], "adjudicator id")
    normalized_ids = {value.casefold() for value in [*reviewer_ids, adjudicator_id]}
    if len(normalized_ids) != 3:
        raise PromotionError(
            "reviewer and adjudicator IDs must be distinct after casefolding"
        )

    policies = receipt["license_policies"]
    acquired_policies = manifest["license_policies"]
    if not isinstance(policies, dict) or set(policies) != set(acquired_policies):
        raise PromotionError("review receipt must cover every license policy exactly")
    output_policies = {
        name: _validate_review_policy(name, policies[name], acquired_policies[name])
        for name in acquired_policies
    }

    sources = receipt["sources"]
    snapshots = manifest["snapshots"]
    expected_ids = [snapshot["id"] for snapshot in snapshots]
    if not isinstance(sources, list):
        raise PromotionError("review receipt sources must be a list")
    observed_ids = [
        source.get("snapshot_id") if isinstance(source, dict) else None
        for source in sources
    ]
    if observed_ids != expected_ids:
        raise PromotionError(
            "review receipt must cover all snapshot IDs exactly in order"
        )

    by_id: dict[str, dict[str, Any]] = {}
    for source, snapshot in zip(sources, snapshots, strict=True):
        source_id = snapshot["id"]
        label = f"review source {source_id}"
        if not isinstance(source, dict):
            raise PromotionError(f"{label} must be an object")
        _exact_keys(source, REVIEW_SOURCE_KEYS, label)
        if source["license_policy"] != snapshot["license_policy"]:
            raise PromotionError(f"{label} license policy mismatch")
        for field in (
            "government_staff_authored_only",
            "third_party_material_excluded",
            "evaluation_use_cleared",
            "redistribution_cleared",
        ):
            if source[field] is not True:
                raise PromotionError(f"{label} must assert {field}")
        if source["unresolved_concerns"] != []:
            raise PromotionError(f"{label} has unresolved concerns")

        ranges = source["retained_ranges"]
        if not isinstance(ranges, list) or not ranges:
            raise PromotionError(f"{label} must retain at least one character range")
        normalized: list[dict[str, int]] = []
        previous_end = 0
        text_length = len(source_texts[source_id])
        for index, retained_range in enumerate(ranges):
            range_label = f"{label} retained_ranges[{index}]"
            if not isinstance(retained_range, dict):
                raise PromotionError(f"{range_label} must be an object")
            _exact_keys(retained_range, RANGE_KEYS, range_label)
            start = retained_range["char_start"]
            end = retained_range["char_end"]
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or start >= end
                or end > text_length
            ):
                raise PromotionError(f"{range_label} is outside the source text")
            if index and start < previous_end:
                raise PromotionError(
                    f"{label} ranges must be ordered and non-overlapping"
                )
            source_text = source_texts[source_id]
            if (start != 0 and source_text[start - 1] != "\n") or (
                end != text_length and source_text[end - 1] != "\n"
            ):
                raise PromotionError(
                    f"{label} ranges must align to complete LF-delimited lines"
                )
            normalized.append({"char_start": start, "char_end": end})
            previous_end = end
        by_id[source_id] = {**source, "retained_ranges": normalized}
    return by_id, output_policies


def _safe_input_text(
    text_root: Path, snapshot: Mapping[str, Any]
) -> tuple[Path, str, bytes]:
    source_id = snapshot["id"]
    filename = snapshot["extracted_text"]["filename"]
    pure = PurePosixPath(filename)
    if pure.is_absolute() or pure.name != filename or pure.suffix != ".txt":
        raise PromotionError(f"{source_id} text filename contains path traversal")
    path = text_root / filename
    _reject_symlink_components(path, f"{source_id} extracted text")
    if not path.is_file():
        raise PromotionError(f"{source_id} extracted text must be a regular file")
    resolved_root = text_root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path.parent != resolved_root:
        raise PromotionError(f"{source_id} extracted text escapes the input directory")
    payload = path.read_bytes()
    declaration = snapshot["extracted_text"]
    if (
        sha256_bytes(payload) != declaration["sha256"]
        or len(payload) != declaration["bytes"]
    ):
        raise PromotionError(f"{source_id} extracted text hash/size mismatch")
    text = _validate_text_payload(payload, f"{source_id} extracted text")
    if len(text.split()) != declaration["words"]:
        raise PromotionError(f"{source_id} extracted text word-count mismatch")
    return resolved_path, text, payload


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def promote(
    acquisition_manifest_path: Path,
    acquisition_manifest_sha256: str,
    acquisition_receipt_path: Path,
    acquisition_receipt_sha256: str,
    extracted_text_dir: Path,
    review_receipt_path: Path,
    review_receipt_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate inputs and publish behind an exclusive completion marker."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise PromotionError(
            "output directory already exists; promotion never overwrites"
        )

    manifest, _, manifest_digest = _read_hashed_json(
        Path(acquisition_manifest_path),
        acquisition_manifest_sha256,
        "acquisition manifest",
        require_canonical=False,
    )
    if not isinstance(manifest, dict):
        raise PromotionError("acquisition manifest must be an object")
    try:
        snapshots = acquisition.validate_manifest(manifest)
    except acquisition.SnapshotError as error:
        raise PromotionError(str(error)) from error

    (
        acquisition_receipt,
        acquisition_receipt_payload,
        acquisition_receipt_digest,
    ) = _read_hashed_json(
        Path(acquisition_receipt_path),
        acquisition_receipt_sha256,
        "acquisition receipt",
        require_canonical=True,
    )
    _validate_acquisition_receipt(acquisition_receipt, manifest, manifest_digest)

    text_root = Path(extracted_text_dir)
    _reject_symlink_components(text_root, "extracted text directory")
    if not text_root.is_dir():
        raise PromotionError("extracted text directory must be a regular directory")
    source_texts: dict[str, str] = {}
    source_payloads: dict[str, bytes] = {}
    for snapshot in snapshots:
        _, text, payload = _safe_input_text(text_root, snapshot)
        source_texts[snapshot["id"]] = text
        source_payloads[snapshot["id"]] = payload

    review, _, review_digest = _read_hashed_json(
        Path(review_receipt_path),
        review_receipt_sha256,
        "review receipt",
        require_canonical=True,
    )
    source_reviews, output_policies = _validate_review_receipt(
        review,
        manifest,
        manifest_digest,
        acquisition_receipt_digest,
        source_texts,
    )

    reviewed_payloads: dict[str, bytes] = {}
    reviewed_metadata: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        source_id = snapshot["id"]
        segments: list[str] = []
        mappings: list[dict[str, int]] = []
        output_offset = 0
        for retained_range in source_reviews[source_id]["retained_ranges"]:
            segment = source_texts[source_id][
                retained_range["char_start"] : retained_range["char_end"]
            ]
            output_end = output_offset + len(segment)
            mappings.append(
                {
                    "source_char_start": retained_range["char_start"],
                    "source_char_end": retained_range["char_end"],
                    "output_char_start": output_offset,
                    "output_char_end": output_end,
                }
            )
            segments.append(segment)
            output_offset = output_end
        reviewed = "".join(segments)
        word_count = len(UNICODE_WORD_RE.findall(reviewed))
        if word_count < MIN_REVIEWED_WORD_TOKENS:
            raise PromotionError(
                f"{source_id} retained text has fewer than "
                f"{MIN_REVIEWED_WORD_TOKENS} deterministic Unicode-word tokens"
            )
        payload = reviewed.encode("utf-8")
        _validate_text_payload(payload, f"{source_id} retained text")
        reviewed_payloads[source_id] = payload
        reviewed_metadata[source_id] = {
            "range_mappings": mappings,
            "reviewed_character_count": len(reviewed),
            "reviewed_unicode_word_token_count": word_count,
            "word_token_count_method": WORD_TOKEN_COUNT_METHOD,
        }
    reviewed_hashes = [sha256_bytes(value) for value in reviewed_payloads.values()]
    if len(reviewed_hashes) != len(set(reviewed_hashes)):
        raise PromotionError("reviewed source artifacts must have unique hashes")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output_dir.parent, "output parent")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        run_receipt_relative = "acquisition/acquisition-run-receipt.json"
        copied_run_digest = _write(
            temporary / run_receipt_relative, acquisition_receipt_payload
        )
        if copied_run_digest != acquisition_receipt_digest:
            raise PromotionError("copied acquisition run receipt hash mismatch")
        artifact_declarations: dict[str, dict[str, str]] = {
            ACQUISITION_RUN_ARTIFACT_NAME: {
                "role": ACQUISITION_RUN_ARTIFACT_ROLE,
                "sha256": acquisition_receipt_digest,
            }
        }
        artifact_paths: dict[str, str] = {
            ACQUISITION_RUN_ARTIFACT_NAME: run_receipt_relative
        }
        output_snapshots: dict[str, dict[str, Any]] = {}
        source_records: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            source_id = snapshot["id"]
            reviewed_name = f"reviewed-{source_id}"
            acquisition_name = f"binding-{source_id}"
            reviewed_relative = f"reviewed/{source_id}.txt"
            acquisition_relative = f"bindings/{source_id}.json"
            reviewed_hash = _write(
                temporary / reviewed_relative, reviewed_payloads[source_id]
            )
            acquisition_record = {
                "schema_version": ACQUISITION_BINDING_VERSION,
                "snapshot_id": source_id,
                "acquisition_manifest_sha256": manifest_digest,
                "acquisition_receipt_sha256": acquisition_receipt_digest,
                "extracted_text_sha256": snapshot["extracted_text"]["sha256"],
                "review_receipt_sha256": review_digest,
                "retained_ranges": source_reviews[source_id]["retained_ranges"],
                **reviewed_metadata[source_id],
                "reviewed_text_sha256": reviewed_hash,
                "upstream_url": snapshot["upstream_url"],
                "upstream_revision": snapshot["upstream_revision"],
                "license_policy": snapshot["license_policy"],
                "attribution": manifest["license_policies"][snapshot["license_policy"]][
                    "attribution"
                ],
            }
            acquisition_hash = _write(
                temporary / acquisition_relative, canonical_json(acquisition_record)
            )
            artifact_declarations[reviewed_name] = {
                "role": "reviewed_source_text",
                "sha256": reviewed_hash,
            }
            artifact_declarations[acquisition_name] = {
                "role": ACQUISITION_BINDING_ARTIFACT_ROLE,
                "sha256": acquisition_hash,
            }
            artifact_paths[reviewed_name] = reviewed_relative
            artifact_paths[acquisition_name] = acquisition_relative
            policy_name = snapshot["license_policy"]
            output_snapshots[source_id] = {
                "reviewed_text_artifact": reviewed_name,
                "acquisition_binding_artifact": acquisition_name,
                "source_family": "public_domain",
                "upstream_url": snapshot["upstream_url"],
                "upstream_revision": snapshot["upstream_revision"],
                "upstream_date": snapshot["public_release_date"]
                or snapshot["adoption_or_issue_date"],
                "untouched_split": snapshot["untouched_split"],
                "license_policy": policy_name,
                "attribution": output_policies[policy_name]["attribution"],
                "third_party_images_excluded": True,
                "natural_language_authored": True,
                "benchmark_derived": False,
            }
            source_records[source_id] = {
                "input_extracted_text_sha256": sha256_bytes(source_payloads[source_id]),
                "retained_ranges": source_reviews[source_id]["retained_ranges"],
                **reviewed_metadata[source_id],
                "reviewed_text_artifact": reviewed_name,
                "reviewed_text_sha256": reviewed_hash,
                "acquisition_binding_artifact": acquisition_name,
                "acquisition_binding_sha256": acquisition_hash,
            }

        output_manifest = {
            "schema_version": SOURCE_MANIFEST_VERSION,
            "license_policy": SOURCE_LICENSE_POLICY,
            "benchmark_derived": False,
            "license_policies": output_policies,
            "artifacts": artifact_declarations,
            "acquisition_run_receipt_artifact": ACQUISITION_RUN_ARTIFACT_NAME,
            "snapshots": output_snapshots,
        }
        output_manifest_payload = canonical_json(output_manifest)
        output_manifest_hash = _write(
            temporary / "source-manifest-v4.json", output_manifest_payload
        )
        promotion_receipt = {
            "schema_version": PROMOTION_RECEIPT_VERSION,
            "acquisition_manifest_sha256": manifest_digest,
            "acquisition_receipt_sha256": acquisition_receipt_digest,
            "acquisition_run_receipt_artifact": ACQUISITION_RUN_ARTIFACT_NAME,
            "review_receipt_sha256": review_digest,
            "source_manifest_sha256": output_manifest_hash,
            "semantic_review_verified_by_tool": False,
            "artifact_paths": artifact_paths,
            "sources": source_records,
        }
        promotion_payload = canonical_json(promotion_receipt)
        _reject_markers(output_manifest_payload, "generated source manifest")
        _reject_markers(promotion_payload, "generated promotion receipt")
        promotion_hash = _write(temporary / "promotion-receipt.json", promotion_payload)

        try:
            shutil.copytree(temporary, output_dir, dirs_exist_ok=False)
        except FileExistsError as error:
            raise PromotionError(
                "output directory already exists; promotion never overwrites"
            ) from error
        except Exception:
            if output_dir.exists() and not output_dir.is_symlink():
                shutil.rmtree(output_dir, ignore_errors=True)
            raise
        completion_marker = {
            "schema_version": COMPLETION_MARKER_VERSION,
            "source_manifest_sha256": output_manifest_hash,
            "promotion_receipt_sha256": promotion_hash,
        }
        try:
            _write(
                output_dir / "PROMOTION_COMPLETE.json",
                canonical_json(completion_marker),
            )
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        return promotion_receipt
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-manifest-sha256", required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--acquisition-receipt-sha256", required=True)
    parser.add_argument("--extracted-text-dir", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--review-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = promote(
        args.acquisition_manifest,
        args.acquisition_manifest_sha256,
        args.acquisition_receipt,
        args.acquisition_receipt_sha256,
        args.extracted_text_dir,
        args.review_receipt,
        args.review_receipt_sha256,
        args.output_dir,
    )
    print(canonical_json(receipt).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
