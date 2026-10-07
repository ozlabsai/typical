#!/usr/bin/env python3
"""Promote externally reviewed R052 acquisitions into source manifest v3.

This bridge verifies bytes and enforces a closed review-receipt contract.  It
does not determine whether the retained text is actually government-authored
or whether a human review was competently performed; those are externally
authored assertions bound by the supplied review receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from scripts import fetch_r052_source_snapshot as acquisition
    from scripts import score_r052_evidence_retrieval as scorer
except ModuleNotFoundError:  # Supports ``uv run python scripts/promote_...py``.
    import fetch_r052_source_snapshot as acquisition
    import score_r052_evidence_retrieval as scorer

REVIEW_SCHEMA_VERSION = "r052-reviewed-source-review-v1"
ACQUISITION_RECEIPT_VERSION = "r052-source-acquisition-receipt-v1"
PROMOTION_RECEIPT_VERSION = "r052-reviewed-source-promotion-receipt-v1"
SOURCE_MANIFEST_VERSION = "r052-source-provenance-manifest-v3"
SOURCE_LICENSE_POLICY = "approved-evaluation-and-redistribution-v1"
MANIFEST_LICENSE_ID = "LicenseRef-USGov-Public-Domain"
MIN_NON_WHITESPACE_CHARS = 256
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_MARKERS = tuple(
    marker.encode() for marker in scorer.FORBIDDEN_LINEAGE_MARKERS
)

REVIEW_KEYS = {
    "schema_version",
    "acquisition_manifest_sha256",
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
    if payload.lstrip().startswith(b"%PDF-") or b"\x00" in payload:
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
    source_texts: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(receipt, dict):
        raise PromotionError("review receipt must be an object")
    _exact_keys(receipt, REVIEW_KEYS, "review receipt")
    if receipt["schema_version"] != REVIEW_SCHEMA_VERSION:
        raise PromotionError(f"review receipt schema must be {REVIEW_SCHEMA_VERSION}")
    if receipt["acquisition_manifest_sha256"] != manifest_sha256:
        raise PromotionError("review receipt acquisition-manifest hash mismatch")
    try:
        parsed_date = date.fromisoformat(receipt["review_date"])
    except (TypeError, ValueError) as error:
        raise PromotionError("review_date must be an ISO YYYY-MM-DD date") from error
    if parsed_date.isoformat() != receipt["review_date"]:
        raise PromotionError("review_date must use canonical YYYY-MM-DD form")

    reviewers = receipt["reviewers"]
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise PromotionError("review receipt must identify exactly two reviewers")
    reviewer_ids = [_text(value, "reviewer id") for value in reviewers]
    adjudicator_id = _text(receipt["adjudicator_id"], "adjudicator id")
    if len({*reviewer_ids, adjudicator_id}) != 3:
        raise PromotionError("reviewer and adjudicator IDs must be distinct")

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
    extracted_text_dir: Path,
    review_receipt_path: Path,
    review_receipt_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate inputs and atomically publish a new reviewed-source directory."""
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
        review, manifest, manifest_digest, source_texts
    )

    reviewed_payloads: dict[str, bytes] = {}
    for snapshot in snapshots:
        source_id = snapshot["id"]
        reviewed = "".join(
            source_texts[source_id][item["char_start"] : item["char_end"]]
            for item in source_reviews[source_id]["retained_ranges"]
        )
        if len("".join(reviewed.split())) < MIN_NON_WHITESPACE_CHARS:
            raise PromotionError(
                f"{source_id} retained text has fewer than "
                f"{MIN_NON_WHITESPACE_CHARS} non-whitespace characters"
            )
        payload = reviewed.encode("utf-8")
        _validate_text_payload(payload, f"{source_id} retained text")
        reviewed_payloads[source_id] = payload
    reviewed_hashes = [sha256_bytes(value) for value in reviewed_payloads.values()]
    if len(reviewed_hashes) != len(set(reviewed_hashes)):
        raise PromotionError("reviewed source artifacts must have unique hashes")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output_dir.parent, "output parent")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        artifact_declarations: dict[str, dict[str, str]] = {}
        artifact_paths: dict[str, str] = {}
        output_snapshots: dict[str, dict[str, Any]] = {}
        source_records: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            source_id = snapshot["id"]
            reviewed_name = f"reviewed-{source_id}"
            acquisition_name = f"acquisition-{source_id}"
            reviewed_relative = f"reviewed/{source_id}.txt"
            acquisition_relative = f"acquisition/{source_id}.json"
            reviewed_hash = _write(
                temporary / reviewed_relative, reviewed_payloads[source_id]
            )
            acquisition_record = {
                "schema_version": ACQUISITION_RECEIPT_VERSION,
                "snapshot_id": source_id,
                "acquisition_manifest_sha256": manifest_digest,
                "extracted_text_sha256": snapshot["extracted_text"]["sha256"],
                "review_receipt_sha256": review_digest,
                "retained_ranges": source_reviews[source_id]["retained_ranges"],
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
                "role": "source_acquisition_receipt",
                "sha256": acquisition_hash,
            }
            artifact_paths[reviewed_name] = reviewed_relative
            artifact_paths[acquisition_name] = acquisition_relative
            policy_name = snapshot["license_policy"]
            output_snapshots[source_id] = {
                "reviewed_text_artifact": reviewed_name,
                "acquisition_receipt_artifact": acquisition_name,
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
                "reviewed_text_artifact": reviewed_name,
                "reviewed_text_sha256": reviewed_hash,
                "acquisition_receipt_artifact": acquisition_name,
                "acquisition_receipt_sha256": acquisition_hash,
            }

        output_manifest = {
            "schema_version": SOURCE_MANIFEST_VERSION,
            "license_policy": SOURCE_LICENSE_POLICY,
            "benchmark_derived": False,
            "license_policies": output_policies,
            "artifacts": artifact_declarations,
            "snapshots": output_snapshots,
        }
        output_manifest_payload = canonical_json(output_manifest)
        output_manifest_hash = _write(
            temporary / "source-manifest-v3.json", output_manifest_payload
        )
        promotion_receipt = {
            "schema_version": PROMOTION_RECEIPT_VERSION,
            "acquisition_manifest_sha256": manifest_digest,
            "review_receipt_sha256": review_digest,
            "source_manifest_sha256": output_manifest_hash,
            "semantic_review_verified_by_tool": False,
            "artifact_paths": artifact_paths,
            "sources": source_records,
        }
        promotion_payload = canonical_json(promotion_receipt)
        _reject_markers(output_manifest_payload, "generated source manifest")
        _reject_markers(promotion_payload, "generated promotion receipt")
        _write(temporary / "promotion-receipt.json", promotion_payload)

        if output_dir.exists() or output_dir.is_symlink():
            raise PromotionError(
                "output directory already exists; promotion never overwrites"
            )
        os.rename(temporary, output_dir)
        published = True
        return promotion_receipt
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-manifest-sha256", required=True)
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
        args.extracted_text_dir,
        args.review_receipt,
        args.review_receipt_sha256,
        args.output_dir,
    )
    print(canonical_json(receipt).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
