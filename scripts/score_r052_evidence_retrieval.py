#!/usr/bin/env python3
"""Score the frozen R052 evidence-retrieval diagnostic.

The input is deliberately self-contained: media is inline, identifiers are
panel-local content-independent hashes, and benchmark provenance/paths are not
accepted.  BM25 is always available; dense providers can be supplied as
precomputed vectors or produced by a locally cached EmbeddingGemma 2 checkout.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

SCHEMA_VERSION = "r052-evidence-panel-v4"
RECEIPT_VERSION = "r052-evidence-retrieval-receipt-v3"
SOURCE_MANIFEST_VERSION = "r052-source-provenance-manifest-v4"
SOURCE_LICENSE_POLICY = "approved-evaluation-and-redistribution-v1"
CHUNKING_RECEIPT_VERSION = "r052-case-chunking-receipt-v2"
ACQUISITION_RUN_RECEIPT_VERSION = "r052-source-acquisition-receipt-v1"
ACQUISITION_BINDING_VERSION = "r052-reviewed-source-acquisition-binding-v1"
WORD_TOKEN_COUNT_METHOD = "python-re-unicode-word-v1-not-model-tokenizer"
MIN_REVIEWED_WORD_TOKENS = 8_192
MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
TRANSFORMERS_VERSION = "5.19.0"
TRANSFORMERS_WHEEL_SHA256 = (
    "afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518"
)
TRANSFORMERS_SOURCE_COMMIT = "c4c4605f961877852b89b97478536bb3a26b6fd1"
FROZEN_MODEL_FILES = {
    ".gitattributes": "bcdccfaa66bc1240c7192dbd374cfb39b695bb1d8ff6b7ce0113aa5e9ac21a44",
    "README.md": "b677a0ee2818c36f091b65fd3d9b4baea9f04d53b02b49cae916c74bf9682037",
    "chat_template.jinja": "4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e",
    "config.json": "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
    "config_sentence_transformers.json": "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5",
    "modules.json": "3d02572a0455b832de67fb8e63a54981bc7e8b46e337c95e917bd8122a533bfd",
    "1_Pooling/config.json": "8759bdf7c77efc7df7723f64856a593c8943b71ee38baf2a88771fbaf78438f9",
    "2_Normalize/config.json": "cdb09dfca347a56aa2d691744e38d5ad3c7cbc2834e7181272b9a15328b82524",
    "processor_config.json": "168f6a08522f3ce5dea596d94d003af2fd691742d4f41fe1f9d8cce76bfbf69c",
    "preprocessor_config.json": "ea2ae257e901064abdd98dceb19f2b0da06af600bed15e0f99f5c85c37ee9d78",
    "sentence_bert_config.json": "b1bcd9f2dce3ae863b359e87d0710b5dbc3314a59ecb4e2f97c7778fc8e4b228",
    "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    "tokenizer.model": "e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03",
    "tokenizer_config.json": "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874",
    "model.safetensors": "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
}
SUPPORTED_MRL_DIMS = frozenset({128, 256, 512, 768})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
FORBIDDEN_LINEAGE_MARKERS = (
    "jevbench",
    "workflowevals",
    "typed-decisions",
    "open-jev",
    "jev-distill",
)
SOURCE_FAMILIES = frozenset({"synthetic", "public_domain", "permissive", "first_party"})
SOURCE_DOMAINS = frozenset(
    {"legal", "finance", "operations", "safety", "support", "technical", "general"}
)

TOP_LEVEL_KEYS = {"schema_version", "provenance", "cases", "providers"}
PROVENANCE_KEYS = {"kind", "benchmark_derived"}
CASE_KEYS = {
    "case_key",
    "language",
    "twin_group_key",
    "source_family",
    "source_snapshot_sha256",
    "source_domain",
    "context_tokens",
    "chunking_receipt",
    "contamination_audit_sha256",
    "chunks",
    "questions",
}
SOURCE_MANIFEST_KEYS = {
    "schema_version",
    "license_policy",
    "benchmark_derived",
    "license_policies",
    "artifacts",
    "snapshots",
    "acquisition_run_receipt_artifact",
}
SOURCE_LICENSE_KEYS = {
    "source_family",
    "license_id",
    "license_url",
    "attribution",
    "license_cleared",
    "evaluation_use_cleared",
    "redistribution_cleared",
}
SOURCE_ARTIFACT_KEYS = {"role", "sha256"}
SOURCE_SNAPSHOT_KEYS = {
    "reviewed_text_artifact",
    "acquisition_binding_artifact",
    "source_family",
    "upstream_url",
    "upstream_revision",
    "upstream_date",
    "untouched_split",
    "license_policy",
    "attribution",
    "third_party_images_excluded",
    "natural_language_authored",
    "benchmark_derived",
}
SOURCE_ARTIFACT_ROLES = frozenset(
    {
        "reviewed_source_text",
        "source_acquisition_run_receipt",
        "reviewed_source_acquisition_binding",
        "image",
        "translation_alignment_receipt",
        "image_control_receipt",
        "chunk_boundary_receipt",
    }
)
ACQUISITION_RUN_RECEIPT_KEYS = {
    "schema_version",
    "mode",
    "manifest_sha256",
    "verified",
}
ACQUISITION_VERIFIED_KEYS = {"id", "raw_sha256", "text_sha256"}
ACQUISITION_MODES = {"fetch", "offline", "verify-only"}
ACQUISITION_BINDING_KEYS = {
    "schema_version",
    "snapshot_id",
    "acquisition_manifest_sha256",
    "acquisition_receipt_sha256",
    "extracted_text_sha256",
    "review_receipt_sha256",
    "retained_ranges",
    "range_mappings",
    "reviewed_character_count",
    "reviewed_unicode_word_token_count",
    "word_token_count_method",
    "reviewed_text_sha256",
    "upstream_url",
    "upstream_revision",
    "license_policy",
    "attribution",
}
RETAINED_RANGE_KEYS = {"char_start", "char_end"}
RANGE_MAPPING_KEYS = {
    "source_char_start",
    "source_char_end",
    "output_char_start",
    "output_char_end",
}
CHUNKING_RECEIPT_KEYS = {
    "schema_version",
    "mode",
    "max_chunk_tokens",
    "fallback_overlap_tokens",
    "tokenizer_artifact_sha256",
    "chunker_artifact_sha256",
    "boundary_receipt_artifact_sha256",
    "semantic_verification",
    "ordered_spans",
    "ordered_derivations",
    "receipt_sha256",
}
CHUNK_SPAN_KEYS = {"chunk_key", "token_start", "token_end"}
CHUNK_DERIVATION_KEYS = {"chunk_key", "derivation"}
CHUNK_KEYS = {
    "chunk_key",
    "kind",
    "modality",
    "text",
    "media",
    "immutable",
    "twin_anchor",
    "image_control",
    "content_sha256",
    "token_start",
    "token_end",
    "derivation",
}
MEDIA_KEYS = {"mime_type", "data_base64", "sha256"}
IMAGE_CONTROL_KEYS = {"group_key", "variant"}
SOURCE_SLICE_DERIVATION_KEYS = {
    "kind",
    "source_artifact_sha256",
    "char_start",
    "char_end",
    "slice_sha256",
}
NON_SOURCE_DERIVATION_KEYS = {"kind"}
TRANSLATED_DERIVATION_KEYS = {
    "kind",
    "source_twin_chunk_key",
    "source_twin_content_sha256",
    "alignment_receipt_artifact_sha256",
}
IMAGE_DERIVATION_KEYS = {
    "kind",
    "image_artifact_sha256",
    "control_receipt_artifact_sha256",
}
QUESTION_KEYS = {
    "question_key",
    "text",
    "options",
    "decision_type",
    "policy_role",
    "twin_anchor",
    "decisive_chunk_keys",
    "known_distractor_chunk_keys",
    "causal_proof",
    "content_sha256",
}
PROOF_KEYS = {
    "kind",
    "removed_chunk_keys",
    "full_context_sha256",
    "removed_context_sha256",
    "full_outcome_sha256",
    "removed_outcome_sha256",
    "full_outcome",
    "removed_outcome",
    "changed_fields",
}
PROVIDER_KEYS = {
    "kind",
    "origin",
    "hosted_api",
    "benchmark_derived",
    "model",
    "revision",
    "generator_code_sha256",
    "artifact_sha256",
    "dimension",
    "queries",
    "documents",
}
CAUSAL_MANIFEST_KEYS = {
    "schema_version",
    "annotation_builder_code_sha256",
    "entries",
}
CAUSAL_MANIFEST_VERSION = "r052-causal-proof-manifest-v1"
CONTAMINATION_MANIFEST_KEYS = {
    "schema_version",
    "auditor_code_sha256",
    "entries",
    "providers",
}
CONTAMINATION_MANIFEST_VERSION = "r052-contamination-audit-manifest-v2"
AUDIT_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "algorithm",
    "auditor_code_sha256",
    "audited_corpus_roots_sha256",
    "exact_overlap_count",
    "near_overlap_max_jaccard",
    "tokenizer_receipt_sha256",
    "chunker_receipt_sha256",
    "receipt_sha256",
}
AUDIT_RECEIPT_VERSION = "r052-contamination-audit-receipt-v1"


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def parse_unique_finite_json(raw: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    """Hash one frozen file or a directory's complete relative-path inventory."""
    path = path.resolve(strict=True)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise ValueError(f"artifact is neither a regular file nor directory: {path}")
    records = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": file_sha256(item),
        }
        for item in sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
    ]
    if not records:
        raise ValueError(f"artifact directory is empty: {path}")
    return sha256_json(records)


def verify_artifact(path: Path, expected_sha256: str, label: str) -> str:
    expected = _require_sha(expected_sha256, f"{label} sha256")
    observed = artifact_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} artifact hash mismatch")
    return observed


def _safe_source_artifact(path: Path, label: str) -> Path:
    """Resolve a regular source snapshot without following hidden symlinks."""
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_dir():
        for child in resolved.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"{label} contains a symlink")
            if not child.is_file() and not child.is_dir():
                raise ValueError(f"{label} contains a non-regular entry")
    elif not resolved.is_file():
        raise ValueError(f"{label} must be a regular file or directory")
    return resolved


def validate_reviewed_text_artifact(path: Path, label: str) -> None:
    """Require a reviewed text snapshot rather than an upstream binary document."""
    if not path.is_file():
        raise ValueError(f"{label} must be one reviewed UTF-8 text file")
    raw = path.read_bytes()
    if raw.startswith(b"%PDF-") or b"\x00" in raw:
        raise ValueError(
            f"{label} must be a reviewed UTF-8 text snapshot, not a PDF or binary"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8 text") from exc
    if not text.strip():
        raise ValueError(f"{label} must contain non-empty reviewed text")


def _canonical_artifact_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} must be one canonical JSON file")
    raw = path.read_bytes()
    value = parse_unique_finite_json(raw, label)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if raw != canonical_json(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value


def validate_source_acquisition_provenance(
    manifest: Mapping[str, Any], artifact_paths: Mapping[str, Path]
) -> None:
    """Validate the whole-run receipt and every reviewed-source binding."""
    artifacts = manifest["artifacts"]
    snapshots = manifest["snapshots"]
    run_name = _require_text(
        manifest["acquisition_run_receipt_artifact"],
        "acquisition_run_receipt_artifact",
    )
    run_artifacts = {
        name
        for name, declaration in artifacts.items()
        if declaration["role"] == "source_acquisition_run_receipt"
    }
    if run_artifacts != {run_name}:
        raise ValueError(
            "source manifest must declare exactly one acquisition run receipt"
        )
    run_receipt = _canonical_artifact_json(
        artifact_paths[run_name], "source acquisition run receipt"
    )
    _exact_keys(
        run_receipt,
        ACQUISITION_RUN_RECEIPT_KEYS,
        "source acquisition run receipt",
    )
    if run_receipt["schema_version"] != ACQUISITION_RUN_RECEIPT_VERSION:
        raise ValueError(
            "source acquisition run receipt schema must be "
            f"{ACQUISITION_RUN_RECEIPT_VERSION}"
        )
    if (
        not isinstance(run_receipt["mode"], str)
        or run_receipt["mode"] not in ACQUISITION_MODES
    ):
        raise ValueError("source acquisition run receipt mode is invalid")
    run_manifest_sha256 = _require_sha(
        run_receipt["manifest_sha256"], "acquisition manifest sha256"
    )
    verified = run_receipt["verified"]
    if not isinstance(verified, list) or len(verified) != len(snapshots):
        raise ValueError(
            "source acquisition run receipt must cover every snapshot exactly"
        )
    verified_by_id: dict[str, dict[str, str]] = {}
    for index, record in enumerate(verified):
        label = f"source acquisition verified[{index}]"
        if not isinstance(record, dict):
            raise TypeError(f"{label} must be an object")
        _exact_keys(record, ACQUISITION_VERIFIED_KEYS, label)
        source_id = _require_text(record["id"], f"{label}.id")
        if source_id in verified_by_id:
            raise ValueError("source acquisition verified IDs must be unique")
        _require_sha(record["raw_sha256"], f"{label}.raw_sha256")
        _require_sha(record["text_sha256"], f"{label}.text_sha256")
        verified_by_id[source_id] = record
    if set(verified_by_id) != set(snapshots):
        raise ValueError(
            "source acquisition run receipt must cover every snapshot exactly"
        )
    if len({record["raw_sha256"] for record in verified}) != len(verified) or len(
        {record["text_sha256"] for record in verified}
    ) != len(verified):
        raise ValueError("source acquisition raw/text hashes must be unique")

    binding_names: set[str] = set()
    run_receipt_sha256 = artifacts[run_name]["sha256"]
    review_receipt_hashes: set[str] = set()
    for source_id, snapshot in snapshots.items():
        binding_name = _require_text(
            snapshot["acquisition_binding_artifact"],
            f"source snapshot {source_id} acquisition binding",
        )
        if binding_name in binding_names:
            raise ValueError("source snapshots must have unique acquisition bindings")
        binding_names.add(binding_name)
        if (
            binding_name not in artifacts
            or artifacts[binding_name]["role"] != "reviewed_source_acquisition_binding"
        ):
            raise ValueError(
                "source snapshot must bind a reviewed-source acquisition binding"
            )
        binding = _canonical_artifact_json(
            artifact_paths[binding_name],
            f"source snapshot {source_id} acquisition binding",
        )
        _exact_keys(
            binding,
            ACQUISITION_BINDING_KEYS,
            f"source snapshot {source_id} acquisition binding",
        )
        if binding["schema_version"] != ACQUISITION_BINDING_VERSION:
            raise ValueError(
                "reviewed-source acquisition binding schema must be "
                f"{ACQUISITION_BINDING_VERSION}"
            )
        reviewed_name = snapshot["reviewed_text_artifact"]
        expected_values = {
            "snapshot_id": source_id,
            "acquisition_manifest_sha256": run_manifest_sha256,
            "acquisition_receipt_sha256": run_receipt_sha256,
            "extracted_text_sha256": verified_by_id[source_id]["text_sha256"],
            "reviewed_text_sha256": artifacts[reviewed_name]["sha256"],
            "upstream_url": snapshot["upstream_url"],
            "upstream_revision": snapshot["upstream_revision"],
            "license_policy": snapshot["license_policy"],
            "attribution": snapshot["attribution"],
        }
        if any(binding[key] != value for key, value in expected_values.items()):
            raise ValueError(
                f"source snapshot {source_id} acquisition binding mismatch"
            )
        review_receipt_hashes.add(
            _require_sha(
                binding["review_receipt_sha256"],
                f"source snapshot {source_id} review receipt sha256",
            )
        )

        ranges = binding["retained_ranges"]
        mappings = binding["range_mappings"]
        if (
            not isinstance(ranges, list)
            or not ranges
            or not isinstance(mappings, list)
            or len(mappings) != len(ranges)
        ):
            raise ValueError(
                f"source snapshot {source_id} ranges/mappings must align exactly"
            )
        previous_source_end = 0
        previous_output_end = 0
        for index, (retained_range, mapping) in enumerate(
            zip(ranges, mappings, strict=True)
        ):
            label = f"source snapshot {source_id} range[{index}]"
            if not isinstance(retained_range, dict) or not isinstance(mapping, dict):
                raise TypeError(f"{label} and mapping must be objects")
            _exact_keys(retained_range, RETAINED_RANGE_KEYS, label)
            _exact_keys(mapping, RANGE_MAPPING_KEYS, f"{label} mapping")
            start = retained_range["char_start"]
            end = retained_range["char_end"]
            source_start = mapping["source_char_start"]
            source_end = mapping["source_char_end"]
            output_start = mapping["output_char_start"]
            output_end = mapping["output_char_end"]
            if (
                type(start) is not int
                or type(end) is not int
                or type(source_start) is not int
                or type(source_end) is not int
                or type(output_start) is not int
                or type(output_end) is not int
                or start < 0
                or start >= end
                or (index and start < previous_source_end)
                or output_start != previous_output_end
                or output_end <= output_start
                or output_end - output_start != end - start
                or source_start != start
                or source_end != end
            ):
                raise ValueError(
                    f"source snapshot {source_id} range mapping is invalid"
                )
            previous_source_end = end
            previous_output_end = output_end

        reviewed_text = artifact_paths[reviewed_name].read_text(encoding="utf-8")
        character_count = binding["reviewed_character_count"]
        if (
            type(character_count) is not int
            or character_count != len(reviewed_text)
            or previous_output_end != character_count
        ):
            raise ValueError(
                f"source snapshot {source_id} reviewed character count mismatch"
            )
        word_count = len(TOKEN_RE.findall(reviewed_text))
        if (
            binding["word_token_count_method"] != WORD_TOKEN_COUNT_METHOD
            or type(binding["reviewed_unicode_word_token_count"]) is not int
            or binding["reviewed_unicode_word_token_count"] != word_count
            or word_count < MIN_REVIEWED_WORD_TOKENS
        ):
            raise ValueError(
                f"source snapshot {source_id} Unicode-word token count/method mismatch"
            )

    declared_bindings = {
        name
        for name, declaration in artifacts.items()
        if declaration["role"] == "reviewed_source_acquisition_binding"
    }
    if binding_names != declared_bindings:
        raise ValueError(
            "source snapshots must consume every acquisition binding exactly"
        )
    if len(review_receipt_hashes) != 1:
        raise ValueError(
            "source acquisition bindings must share one review receipt hash"
        )


def _artifact_contains_forbidden_lineage(path: Path) -> bool:
    files = (
        [path]
        if path.is_file()
        else sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    )
    markers = tuple(marker.encode() for marker in FORBIDDEN_LINEAGE_MARKERS)
    overlap = max(map(len, markers)) - 1
    for file_path in files:
        relative = (
            file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        )
        if any(marker in relative.casefold().encode() for marker in markers):
            return True
        tail = b""
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                folded = (tail + block).lower()
                if any(marker in folded for marker in markers):
                    return True
                tail = folded[-overlap:]
    return False


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _require_https_url(value: Any, label: str) -> str:
    value = _require_text(value, label)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise ValueError(f"{label} must be an HTTPS URL")
    return value


def validate_source_snapshot_manifest(
    panel: dict[str, Any],
    manifest: Any,
    expected_sha256: str,
    artifact_paths: dict[str, Path],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Resolve every case to exact externally supplied source bytes/tree."""
    _require_sha(expected_sha256, "trusted source snapshot manifest sha256")
    if not isinstance(manifest, dict):
        raise TypeError("trusted source snapshot manifest must be an object")
    if manifest.get("schema_version") in {
        "r052-source-provenance-manifest-v2",
        "r052-source-provenance-manifest-v3",
    }:
        raise ValueError(
            "source manifest v2/v3 requires migration to v4 acquisition provenance"
        )
    _exact_keys(manifest, SOURCE_MANIFEST_KEYS, "trusted source snapshot manifest")
    if (
        manifest["schema_version"] != SOURCE_MANIFEST_VERSION
        or manifest["license_policy"] != SOURCE_LICENSE_POLICY
        or manifest["benchmark_derived"] is not False
    ):
        raise ValueError("source snapshot manifest policy/provenance is invalid")
    if sha256_json(manifest) != expected_sha256:
        raise ValueError(
            "trusted source snapshot manifest does not match its external sha256 anchor"
        )
    _reject_benchmark_lineage(manifest)
    license_policies = manifest["license_policies"]
    if not isinstance(license_policies, dict) or not license_policies:
        raise ValueError("source license_policies must be a non-empty object")
    for name, declaration in license_policies.items():
        _require_text(name, "source license policy name")
        if not isinstance(declaration, dict):
            raise TypeError(f"source license policy {name} must be an object")
        _exact_keys(declaration, SOURCE_LICENSE_KEYS, f"source license policy {name}")
        if declaration["source_family"] not in SOURCE_FAMILIES:
            raise ValueError(f"source license policy {name} family is invalid")
        _require_text(declaration["license_id"], f"source license policy {name} id")
        _require_https_url(
            declaration["license_url"], f"source license policy {name} URL"
        )
        _require_text(
            declaration["attribution"], f"source license policy {name} attribution"
        )
        if any(
            declaration[field] is not True
            for field in (
                "license_cleared",
                "evaluation_use_cleared",
                "redistribution_cleared",
            )
        ):
            raise ValueError(f"source license policy {name} is not cleared")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("source snapshot manifest artifacts must be non-empty")
    if set(artifact_paths) != set(artifacts):
        raise ValueError("source snapshot artifact paths must cover manifest exactly")

    by_hash: dict[str, dict[str, Any]] = {}
    for name, declaration in artifacts.items():
        _require_text(name, "source snapshot artifact name")
        if not isinstance(declaration, dict):
            raise TypeError(f"source snapshot artifact {name} must be an object")
        _exact_keys(
            declaration,
            SOURCE_ARTIFACT_KEYS,
            f"source snapshot artifact {name}",
        )
        role = declaration["role"]
        if role not in SOURCE_ARTIFACT_ROLES:
            raise ValueError(f"source snapshot artifact {name} role is invalid")
        digest = _require_sha(declaration["sha256"], f"source artifact {name} sha256")
        if digest in by_hash:
            raise ValueError("source snapshot manifest has duplicate artifact hashes")
        path = _safe_source_artifact(
            artifact_paths[name], f"source snapshot artifact {name}"
        )
        if artifact_sha256(path) != digest:
            raise ValueError(f"source snapshot artifact {name} hash mismatch")
        if role == "reviewed_source_text":
            validate_reviewed_text_artifact(path, f"source snapshot artifact {name}")
        if _artifact_contains_forbidden_lineage(path):
            raise ValueError(
                f"source snapshot artifact {name} contains benchmark lineage"
            )
        by_hash[digest] = {**declaration, "name": name, "path": path}

    snapshots = manifest["snapshots"]
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("source snapshots must be a non-empty object")
    snapshot_by_reviewed_hash: dict[str, dict[str, Any]] = {}
    used_reviewed: set[str] = set()
    used_acquisition_bindings: set[str] = set()
    used_policies: set[str] = set()
    for name, snapshot in snapshots.items():
        _require_text(name, "source snapshot name")
        if not isinstance(snapshot, dict):
            raise TypeError(f"source snapshot {name} must be an object")
        _exact_keys(snapshot, SOURCE_SNAPSHOT_KEYS, f"source snapshot {name}")
        reviewed_name = _require_text(
            snapshot["reviewed_text_artifact"],
            f"source snapshot {name} reviewed_text_artifact",
        )
        acquisition_name = _require_text(
            snapshot["acquisition_binding_artifact"],
            f"source snapshot {name} acquisition_binding_artifact",
        )
        if (
            reviewed_name in used_reviewed
            or acquisition_name in used_acquisition_bindings
        ):
            raise ValueError(
                "source snapshots have duplicate reviewed/acquisition bindings"
            )
        if (
            reviewed_name not in artifacts
            or artifacts[reviewed_name]["role"] != "reviewed_source_text"
            or acquisition_name not in artifacts
            or artifacts[acquisition_name]["role"]
            != "reviewed_source_acquisition_binding"
        ):
            raise ValueError(
                "source snapshot reviewed/acquisition artifact roles mismatch"
            )
        used_reviewed.add(reviewed_name)
        used_acquisition_bindings.add(acquisition_name)
        family = snapshot["source_family"]
        policy_name = snapshot["license_policy"]
        if (
            family not in SOURCE_FAMILIES
            or policy_name not in license_policies
            or license_policies[policy_name]["source_family"] != family
        ):
            raise ValueError("source snapshot license policy/family mismatch")
        used_policies.add(policy_name)
        _require_https_url(snapshot["upstream_url"], f"source snapshot {name} URL")
        revision = _require_text(
            snapshot["upstream_revision"],
            f"source snapshot {name} upstream_revision",
        )
        if revision.casefold() in {"head", "latest", "main", "master", "unknown"}:
            raise ValueError("source snapshot revision must be immutable and exact")
        upstream_date = _require_text(
            snapshot["upstream_date"], f"source snapshot {name} upstream_date"
        )
        try:
            date.fromisoformat(upstream_date)
        except ValueError as exc:
            raise ValueError(
                "source snapshot upstream_date must be ISO YYYY-MM-DD"
            ) from exc
        untouched_split = _require_text(
            snapshot["untouched_split"], f"source snapshot {name} untouched_split"
        )
        if untouched_split.casefold() in {"unknown", "unspecified", "none"}:
            raise ValueError("source snapshot untouched_split must be exact")
        _require_text(snapshot["attribution"], f"source snapshot {name} attribution")
        if (
            snapshot["third_party_images_excluded"] is not True
            or snapshot["natural_language_authored"] is not True
            or snapshot["benchmark_derived"] is not False
        ):
            raise ValueError(
                "source snapshot must be authored, independent, and exclude third-party images"
            )
        reviewed_hash = artifacts[reviewed_name]["sha256"]
        snapshot_by_reviewed_hash[reviewed_hash] = snapshot

    declared_reviewed = {
        name
        for name, item in artifacts.items()
        if item["role"] == "reviewed_source_text"
    }
    declared_acquisition_bindings = {
        name
        for name, item in artifacts.items()
        if item["role"] == "reviewed_source_acquisition_binding"
    }
    if (
        used_reviewed != declared_reviewed
        or used_acquisition_bindings != declared_acquisition_bindings
    ):
        raise ValueError(
            "source snapshots must bind every reviewed text/acquisition binding exactly"
        )
    if used_policies != set(license_policies):
        raise ValueError("source snapshots must use every declared license policy")

    case_hashes = [case["source_snapshot_sha256"] for case in panel["cases"]]
    for value in case_hashes:
        _require_sha(value, "case source_snapshot_sha256")
    if set(case_hashes) != set(snapshot_by_reviewed_hash):
        raise ValueError(
            "case source snapshot bindings must cover source snapshots exactly"
        )
    for case in panel["cases"]:
        snapshot = snapshot_by_reviewed_hash[case["source_snapshot_sha256"]]
        if snapshot["source_family"] != case["source_family"]:
            raise ValueError("case source family mismatches its source snapshot")
    validate_source_acquisition_provenance(manifest, artifact_paths)
    return expected_sha256, by_hash


def chunk_hash_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    media = chunk["media"]
    return {
        "kind": chunk["kind"],
        "modality": chunk["modality"],
        "text": chunk["text"],
        "token_start": chunk["token_start"],
        "token_end": chunk["token_end"],
        "media": None
        if media is None
        else {"mime_type": media["mime_type"], "sha256": media["sha256"]},
        "derivation": chunk["derivation"],
    }


def _validate_chunk_derivation(
    case: dict[str, Any],
    chunk: dict[str, Any],
    source_artifacts_by_hash: dict[str, dict[str, Any]],
    chunks_by_key: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    derivation = chunk["derivation"]
    if not isinstance(derivation, dict):
        raise TypeError("chunk derivation must be an object")
    kind = derivation.get("kind")
    if chunk["kind"] == "global_policy":
        _exact_keys(derivation, NON_SOURCE_DERIVATION_KEYS, "policy derivation")
        if kind != "non_source_global_policy":
            raise ValueError("global policy requires explicit non-source derivation")
        return

    if chunk["modality"] == "image":
        _exact_keys(derivation, IMAGE_DERIVATION_KEYS, "image derivation")
        if kind != "image_control":
            raise ValueError("image chunks require image-control derivation")
        image_hash = derivation["image_artifact_sha256"]
        if chunk["media"] is None:
            if image_hash is not None:
                raise ValueError("missing image control cannot bind image bytes")
        else:
            image_hash = _require_sha(image_hash, "image derivation artifact sha256")
            image_artifact = source_artifacts_by_hash.get(image_hash)
            if (
                image_artifact is None
                or image_artifact["role"] != "image"
                or chunk["media"]["sha256"] != image_hash
            ):
                raise ValueError(
                    "image derivation does not bind supplied image artifact"
                )
        control_hash = _require_sha(
            derivation["control_receipt_artifact_sha256"],
            "image control receipt artifact sha256",
        )
        control_artifact = source_artifacts_by_hash.get(control_hash)
        if (
            control_artifact is None
            or control_artifact["role"] != "image_control_receipt"
        ):
            raise ValueError("image derivation does not bind a control receipt")
        return

    if case["language"] == "he":
        _exact_keys(derivation, TRANSLATED_DERIVATION_KEYS, "translated derivation")
        if kind != "translated_twin":
            raise ValueError("Hebrew text chunks require translated-twin derivation")
        source_key = _require_text(
            derivation["source_twin_chunk_key"], "translated source twin chunk key"
        )
        source = chunks_by_key.get(source_key)
        if source is None:
            raise ValueError(
                "translated derivation references unknown source twin chunk"
            )
        source_case, source_chunk = source
        if (
            source_case["language"] != "en"
            or source_case["twin_group_key"] != case["twin_group_key"]
            or source_chunk["twin_anchor"] != chunk["twin_anchor"]
            or source_chunk["kind"] != chunk["kind"]
            or source_chunk["modality"] != "text"
            or not isinstance(source_chunk["derivation"], dict)
            or source_chunk["derivation"].get("kind") != "source_text_slice"
            or derivation["source_twin_content_sha256"]
            != source_chunk["content_sha256"]
        ):
            raise ValueError(
                "translated derivation does not bind its English source twin"
            )
        alignment_hash = _require_sha(
            derivation["alignment_receipt_artifact_sha256"],
            "translation alignment receipt artifact sha256",
        )
        alignment = source_artifacts_by_hash.get(alignment_hash)
        if alignment is None or alignment["role"] != "translation_alignment_receipt":
            raise ValueError("translated derivation does not bind an alignment receipt")
        return

    _exact_keys(derivation, SOURCE_SLICE_DERIVATION_KEYS, "source slice derivation")
    if kind != "source_text_slice":
        raise ValueError("English evidence text requires source-slice derivation")
    source_hash = _require_sha(
        derivation["source_artifact_sha256"], "source slice artifact sha256"
    )
    if source_hash != case["source_snapshot_sha256"]:
        raise ValueError("source slice does not bind the case reviewed source")
    artifact = source_artifacts_by_hash.get(source_hash)
    if artifact is None or artifact["role"] != "reviewed_source_text":
        raise ValueError("source slice must bind a reviewed-source-text artifact")
    start, end = derivation["char_start"], derivation["char_end"]
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise ValueError("source slice character offsets are invalid")
    text = artifact["path"].read_text(encoding="utf-8")
    if end > len(text):
        raise ValueError("source slice character offsets are out of range")
    exact_slice = text[start:end]
    if chunk["text"] != exact_slice:
        raise ValueError("chunk text does not exactly equal its reviewed source slice")
    slice_hash = _require_sha(derivation["slice_sha256"], "source slice sha256")
    if slice_hash != sha256_bytes(exact_slice.encode("utf-8")):
        raise ValueError("source slice sha256 mismatch")


def question_hash_payload(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": question["text"],
        "options": question["options"],
        "decision_type": question["decision_type"],
        "policy_role": question["policy_role"],
    }


def causal_context_sha256(
    case: dict[str, Any], question: dict[str, Any], removed_chunk_keys: set[str]
) -> str:
    """Hash the ordered question context, optionally filtering decisive chunks."""
    return sha256_json(
        {
            "question_content_sha256": question["content_sha256"],
            "ordered_chunk_content_sha256": [
                chunk["content_sha256"]
                for chunk in case["chunks"]
                if chunk["chunk_key"] not in removed_chunk_keys
            ],
        }
    )


def causal_entry_key(case: dict[str, Any], question: dict[str, Any]) -> str:
    return sha256_json(
        {
            "case_key": case["case_key"],
            "question_key": question["question_key"],
            "question_content_sha256": question["content_sha256"],
        }
    )


def validate_causal_manifest(
    panel: dict[str, Any],
    manifest: Any,
    expected_sha256: str,
    annotation_builder_artifact: Path,
) -> str:
    """Validate externally anchored causal annotations, not executor semantics."""
    _require_sha(expected_sha256, "trusted causal manifest sha256")
    if not isinstance(manifest, dict):
        raise TypeError("trusted causal manifest must be an object")
    _exact_keys(manifest, CAUSAL_MANIFEST_KEYS, "trusted causal manifest")
    if manifest["schema_version"] != CAUSAL_MANIFEST_VERSION:
        raise ValueError(f"causal manifest schema must be {CAUSAL_MANIFEST_VERSION}")
    verify_artifact(
        annotation_builder_artifact,
        manifest["annotation_builder_code_sha256"],
        "causal annotation builder",
    )
    if sha256_json(manifest) != expected_sha256:
        raise ValueError(
            "trusted causal manifest does not match its external sha256 anchor"
        )
    _reject_benchmark_lineage(manifest)
    expected = {
        causal_entry_key(case, question): question["causal_proof"]
        for case in panel["cases"]
        for question in case["questions"]
    }
    question_count = sum(len(case["questions"]) for case in panel["cases"])
    if len(expected) != question_count:
        raise ValueError(
            "causal manifest entry keys must be unique per case and question"
        )
    if manifest["entries"] != expected:
        raise ValueError(
            "trusted causal manifest entries do not exactly match panel proofs"
        )
    return expected_sha256


def case_content_sha256(case: dict[str, Any]) -> str:
    return sha256_json(
        {
            "language": case["language"],
            "source_family": case["source_family"],
            "source_snapshot_sha256": case["source_snapshot_sha256"],
            "source_domain": case["source_domain"],
            "context_tokens": case["context_tokens"],
            "chunking_receipt": case["chunking_receipt"],
            "chunks": [chunk["content_sha256"] for chunk in case["chunks"]],
            "questions": [question["content_sha256"] for question in case["questions"]],
        }
    )


def validate_case_chunking_receipt(
    case: dict[str, Any],
    source_artifacts_by_hash: dict[str, dict[str, Any]],
    tokenizer_artifact_sha256: str,
    chunker_artifact_sha256: str,
    chunks_by_key: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> None:
    """Replay source derivations; token boundaries remain externally computed."""
    receipt = case["chunking_receipt"]
    if not isinstance(receipt, dict):
        raise TypeError("case chunking_receipt must be an object")
    _exact_keys(receipt, CHUNKING_RECEIPT_KEYS, "case chunking_receipt")
    if receipt["schema_version"] != CHUNKING_RECEIPT_VERSION:
        raise ValueError(f"chunking receipt schema must be {CHUNKING_RECEIPT_VERSION}")
    if (
        receipt["max_chunk_tokens"] != 512
        or type(receipt["max_chunk_tokens"]) is not int
        or receipt["fallback_overlap_tokens"] != 64
        or type(receipt["fallback_overlap_tokens"]) is not int
    ):
        raise ValueError("chunking receipt must freeze 512-token/64-overlap contract")
    if (
        receipt["semantic_verification"]
        != "external_tokenizer_and_chunker_not_recomputed"
    ):
        raise ValueError(
            "chunking receipt must disclose the external tokenizer/chunker boundary"
        )
    if (
        receipt["tokenizer_artifact_sha256"] != tokenizer_artifact_sha256
        or receipt["chunker_artifact_sha256"] != chunker_artifact_sha256
    ):
        raise ValueError("chunking receipt tokenizer/chunker artifact hash mismatch")

    spans = receipt["ordered_spans"]
    expected_spans = [
        {
            "chunk_key": chunk["chunk_key"],
            "token_start": chunk["token_start"],
            "token_end": chunk["token_end"],
        }
        for chunk in case["chunks"]
    ]
    if not isinstance(spans, list) or len(spans) != len(expected_spans):
        raise ValueError("chunking receipt must cover every chunk exactly once")
    for span in spans:
        if not isinstance(span, dict):
            raise TypeError("chunking receipt spans must be objects")
        _exact_keys(span, CHUNK_SPAN_KEYS, "chunking receipt span")
    if spans != expected_spans:
        raise ValueError("chunking receipt ordered spans do not match panel chunks")
    if any(span["token_end"] - span["token_start"] > 512 for span in spans):
        raise ValueError("chunking receipt contains a chunk exceeding 512 tokens")

    derivations = receipt["ordered_derivations"]
    expected_derivations = [
        {"chunk_key": chunk["chunk_key"], "derivation": chunk["derivation"]}
        for chunk in case["chunks"]
    ]
    if not isinstance(derivations, list) or len(derivations) != len(
        expected_derivations
    ):
        raise ValueError("chunking receipt must cover every derivation exactly once")
    for record in derivations:
        if not isinstance(record, dict):
            raise TypeError("chunking receipt derivations must be objects")
        _exact_keys(record, CHUNK_DERIVATION_KEYS, "chunking receipt derivation")
    if derivations != expected_derivations:
        raise ValueError("chunking receipt derivations do not match panel chunks")

    if chunks_by_key is None:
        chunks_by_key = {chunk["chunk_key"]: (case, chunk) for chunk in case["chunks"]}
    for chunk in case["chunks"]:
        _validate_chunk_derivation(case, chunk, source_artifacts_by_hash, chunks_by_key)
    source_slices = [
        chunk["derivation"]
        for chunk in case["chunks"]
        if chunk["derivation"]["kind"] == "source_text_slice"
    ]
    if case["language"] == "en":
        if not source_slices:
            raise ValueError("English case must derive text from its reviewed source")
        artifact = source_artifacts_by_hash[case["source_snapshot_sha256"]]
        reviewed_text = artifact["path"].read_text(encoding="utf-8")
        if source_slices[0]["char_start"] != 0 or source_slices[-1]["char_end"] != len(
            reviewed_text
        ):
            raise ValueError(
                "reviewed source slices must cover the artifact without gaps"
            )
        if any(
            previous["char_end"] != current["char_start"]
            for previous, current in pairwise(source_slices)
        ):
            raise ValueError("reviewed source slices must be contiguous without gaps")

    mode = receipt["mode"]
    boundary_digest = receipt["boundary_receipt_artifact_sha256"]
    if mode == "fixed_512_overlap_64":
        if boundary_digest is not None:
            raise ValueError("fixed chunking cannot claim a semantic boundary receipt")
        raise ValueError(
            "fixed 512/64 chunking fails closed because the external tokenizer "
            "boundary cannot be deterministically recomputed"
        )
    elif mode == "semantic_sections":
        digest = _require_sha(
            boundary_digest, "semantic boundary receipt artifact sha256"
        )
        declaration = source_artifacts_by_hash.get(digest)
        if declaration is None or declaration["role"] != "chunk_boundary_receipt":
            raise ValueError(
                "semantic chunking requires a resolved chunk-boundary receipt artifact"
            )
    else:
        raise ValueError(
            "chunking mode must be semantic_sections or fixed_512_overlap_64"
        )

    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != sha256_json(body):
        raise ValueError("chunking receipt hash mismatch")


def validate_contamination_manifest(
    panel: dict[str, Any],
    manifest: Any,
    expected_sha256: str,
    auditor_artifact: Path,
    corpus_artifacts: list[Path],
    tokenizer_artifact: Path,
    chunker_artifact: Path,
) -> str:
    _require_sha(expected_sha256, "trusted contamination manifest sha256")
    if not isinstance(manifest, dict):
        raise TypeError("trusted contamination manifest must be an object")
    _exact_keys(manifest, CONTAMINATION_MANIFEST_KEYS, "trusted contamination manifest")
    if manifest["schema_version"] != CONTAMINATION_MANIFEST_VERSION:
        raise ValueError(
            f"contamination manifest schema must be {CONTAMINATION_MANIFEST_VERSION}"
        )
    verify_artifact(
        auditor_artifact,
        manifest["auditor_code_sha256"],
        "contamination auditor",
    )
    if sha256_json(manifest) != expected_sha256:
        raise ValueError(
            "trusted contamination manifest does not match its external sha256 anchor"
        )
    _reject_benchmark_lineage(manifest)
    entries = manifest["entries"]
    if not isinstance(entries, dict) or set(entries) != {
        case["case_key"] for case in panel["cases"]
    }:
        raise ValueError("contamination manifest must cover every case exactly")
    corpus_hashes = [artifact_sha256(path) for path in corpus_artifacts]
    if not corpus_hashes or len(corpus_hashes) != len(set(corpus_hashes)):
        raise ValueError("contamination corpus artifacts must be non-empty and unique")
    tokenizer_hash = artifact_sha256(tokenizer_artifact)
    chunker_hash = artifact_sha256(chunker_artifact)
    expected_entries = {}
    for case in panel["cases"]:
        entry = entries[case["case_key"]]
        if not isinstance(entry, dict) or set(entry) != {
            "case_content_sha256",
            "audit_receipt",
        }:
            raise ValueError("contamination entry schema is invalid")
        receipt = entry["audit_receipt"]
        if not isinstance(receipt, dict):
            raise TypeError("contamination audit receipt must be an object")
        _exact_keys(receipt, AUDIT_RECEIPT_KEYS, "contamination audit receipt")
        if (
            receipt["schema_version"] != AUDIT_RECEIPT_VERSION
            or receipt["status"] != "pass"
            or receipt["algorithm"] != "exact-sha256+minhash-jaccard-v1"
            or receipt["auditor_code_sha256"] != manifest["auditor_code_sha256"]
            or type(receipt["exact_overlap_count"]) is not int
            or receipt["exact_overlap_count"] != 0
            or not isinstance(receipt["near_overlap_max_jaccard"], (int, float))
            or isinstance(receipt["near_overlap_max_jaccard"], bool)
            or not math.isfinite(receipt["near_overlap_max_jaccard"])
            or not 0 <= receipt["near_overlap_max_jaccard"] < 0.60
        ):
            raise ValueError("contamination audit receipt did not pass frozen gates")
        roots = receipt["audited_corpus_roots_sha256"]
        if not isinstance(roots, list) or not roots or len(roots) != len(set(roots)):
            raise ValueError(
                "contamination receipt requires unique audited corpus roots"
            )
        for value in [
            *roots,
            receipt["tokenizer_receipt_sha256"],
            receipt["chunker_receipt_sha256"],
        ]:
            _require_sha(value, "contamination receipt binding")
        if roots != corpus_hashes:
            raise ValueError(
                "contamination corpus roots do not match supplied artifacts"
            )
        if receipt["tokenizer_receipt_sha256"] != tokenizer_hash:
            raise ValueError("contamination tokenizer artifact hash mismatch")
        if receipt["chunker_receipt_sha256"] != chunker_hash:
            raise ValueError("contamination chunker artifact hash mismatch")
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt["receipt_sha256"] != sha256_json(body):
            raise ValueError("contamination audit receipt hash mismatch")
        if case["contamination_audit_sha256"] != receipt["receipt_sha256"]:
            raise ValueError("case does not bind its contamination audit receipt")
        expected_entries[case["case_key"]] = {
            "case_content_sha256": case_content_sha256(case),
            "audit_receipt": receipt,
        }
    if entries != expected_entries:
        raise ValueError(
            "trusted contamination manifest entries do not exactly match panel content"
        )
    expected_providers = {
        name: {
            key: provider[key]
            for key in (
                "origin",
                "hosted_api",
                "benchmark_derived",
                "model",
                "revision",
                "generator_code_sha256",
                "artifact_sha256",
            )
        }
        for name, provider in panel["providers"].items()
    }
    if manifest["providers"] != expected_providers:
        raise ValueError("trusted provider provenance does not match panel providers")
    return expected_sha256


def _validate_media(media: Any, label: str) -> None:
    if media is None:
        return
    if not isinstance(media, dict):
        raise TypeError(f"{label} must be null or an object")
    _exact_keys(media, MEDIA_KEYS, label)
    mime = _require_text(media["mime_type"], f"{label}.mime_type")
    if not mime.startswith("image/"):
        raise ValueError(f"{label}.mime_type must be image/*")
    expected = _require_sha(media["sha256"], f"{label}.sha256")
    try:
        raw = base64.b64decode(media["data_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.data_base64 is invalid") from exc
    if not raw or sha256_bytes(raw) != expected:
        raise ValueError(f"{label} bytes do not match sha256")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except Exception as exc:
        raise ValueError(f"{label} is not a valid decodable image") from exc
    mime_formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
    if mime not in mime_formats or image_format != mime_formats[mime]:
        raise ValueError(f"{label} MIME type does not match decoded image")
    if width < 1 or height < 1 or width * height > 25_000_000:
        raise ValueError(f"{label} image dimensions are unsafe")


def _reject_benchmark_lineage(value: Any) -> None:
    def visit(node: Any) -> None:
        if isinstance(node, dict):
            forbidden_keys = {
                "benchmark_id",
                "benchmark_path",
                "source_id",
                "source_path",
                "dataset_id",
                "dataset_path",
                "upstream_id",
                "upstream_path",
                "lineage",
            }
            overlap = forbidden_keys.intersection(str(key).casefold() for key in node)
            if overlap:
                raise ValueError(
                    f"benchmark lineage/path/id fields are forbidden: {sorted(overlap)}"
                )
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, str):
            folded = node.casefold()
            if any(marker in folded for marker in FORBIDDEN_LINEAGE_MARKERS):
                raise ValueError("benchmark lineage marker is forbidden")

    visit(value)


def validate_panel(
    panel: Any,
    source_snapshot_manifest: Any,
    source_snapshot_manifest_sha256: str,
    causal_manifest: Any,
    causal_manifest_sha256: str,
    contamination_manifest: Any,
    contamination_manifest_sha256: str,
    *,
    annotation_builder_artifact: Path,
    contamination_auditor_artifact: Path,
    contamination_corpus_artifacts: list[Path],
    contamination_tokenizer_artifact: Path,
    contamination_chunker_artifact: Path,
    source_snapshot_artifacts: dict[str, Path],
    provider_generator_artifacts: dict[str, Path],
    selected_precomputed_providers: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(panel, dict):
        raise TypeError("panel must be an object")
    _exact_keys(panel, TOP_LEVEL_KEYS, "panel")
    if panel["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    provenance = panel["provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be an object")
    _exact_keys(provenance, PROVENANCE_KEYS, "provenance")
    if provenance != {"kind": "independent_internal", "benchmark_derived": False}:
        raise ValueError(
            "panel must attest independent_internal and benchmark_derived=false"
        )
    if not isinstance(panel["cases"], list) or not panel["cases"]:
        raise ValueError("cases must be a non-empty list")
    if not isinstance(panel["providers"], dict):
        raise TypeError("providers must be an object")

    _reject_benchmark_lineage(panel)
    for case_index, case in enumerate(panel["cases"]):
        if not isinstance(case, dict):
            raise TypeError(f"cases[{case_index}] must be an object")
        _exact_keys(case, CASE_KEYS, f"cases[{case_index}]")
    _, source_artifacts_by_hash = validate_source_snapshot_manifest(
        panel,
        source_snapshot_manifest,
        source_snapshot_manifest_sha256,
        source_snapshot_artifacts,
    )
    tokenizer_artifact_sha256 = artifact_sha256(contamination_tokenizer_artifact)
    chunker_artifact_sha256 = artifact_sha256(contamination_chunker_artifact)
    chunks_by_key: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for candidate_case in panel["cases"]:
        for candidate_chunk in candidate_case["chunks"]:
            candidate_key = candidate_chunk.get("chunk_key")
            if isinstance(candidate_key, str):
                if candidate_key in chunks_by_key:
                    raise ValueError(
                        "chunk_key values must be globally unique for derivation references"
                    )
                chunks_by_key[candidate_key] = (candidate_case, candidate_chunk)

    case_keys: set[str] = set()
    twin_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_controls = 0
    for case_index, case in enumerate(panel["cases"]):
        label = f"cases[{case_index}]"
        if not isinstance(case, dict):
            raise TypeError(f"{label} must be an object")
        _exact_keys(case, CASE_KEYS, label)
        case_key = _require_sha(case["case_key"], f"{label}.case_key")
        if case_key in case_keys:
            raise ValueError("case_key values must be unique")
        case_keys.add(case_key)
        if case["language"] not in {"en", "he"}:
            raise ValueError(f"{label}.language must be en or he")
        if case["source_family"] not in SOURCE_FAMILIES:
            raise ValueError(f"{label}.source_family is not allowlisted")
        if case["source_domain"] not in SOURCE_DOMAINS:
            raise ValueError(f"{label}.source_domain is not allowlisted")
        if (
            not isinstance(case["context_tokens"], int)
            or isinstance(case["context_tokens"], bool)
            or case["context_tokens"] < 1
        ):
            raise ValueError(f"{label}.context_tokens must be a positive integer")
        _require_sha(
            case["contamination_audit_sha256"],
            f"{label}.contamination_audit_sha256",
        )
        twin_group = case["twin_group_key"]
        if twin_group is not None:
            _require_sha(twin_group, f"{label}.twin_group_key")
            twin_groups[twin_group].append(case)
        elif case["language"] == "he":
            raise ValueError("every Hebrew case must have an English twin")
        chunks = case["chunks"]
        questions = case["questions"]
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"{label}.chunks must be non-empty")
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"{label}.questions must be non-empty")
        chunk_map: dict[str, dict[str, Any]] = {}
        controls: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        previous_start = -1
        for chunk_index, chunk in enumerate(chunks):
            chunk_label = f"{label}.chunks[{chunk_index}]"
            if not isinstance(chunk, dict):
                raise TypeError(f"{chunk_label} must be an object")
            _exact_keys(chunk, CHUNK_KEYS, chunk_label)
            chunk_key = _require_text(chunk["chunk_key"], f"{chunk_label}.chunk_key")
            if chunk_key in chunk_map:
                raise ValueError(f"duplicate chunk_key {chunk_key}")
            chunk_map[chunk_key] = chunk
            if chunk["kind"] not in {"global_policy", "evidence", "distractor"}:
                raise ValueError(f"{chunk_label}.kind is invalid")
            if chunk["modality"] not in {"text", "image"}:
                raise ValueError(f"{chunk_label}.modality is invalid")
            _require_text(chunk["text"], f"{chunk_label}.text")
            start, end = chunk["token_start"], chunk["token_end"]
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > case["context_tokens"]
                or start < previous_start
            ):
                raise ValueError(f"{chunk_label} has invalid ordered token offsets")
            previous_start = start
            _validate_media(chunk["media"], f"{chunk_label}.media")
            if chunk["modality"] == "text" and chunk["media"] is not None:
                raise ValueError("text chunks cannot contain media")
            if chunk["modality"] == "image" and chunk["media"] is None:
                control = chunk["image_control"]
                if not isinstance(control, dict) or control.get("variant") != "missing":
                    raise ValueError("only missing image controls may omit image media")
            if chunk["kind"] == "global_policy":
                if chunk["immutable"] is not True or chunk["image_control"] is not None:
                    raise ValueError(
                        "global policy chunks must be immutable and not image controls"
                    )
            elif chunk["immutable"] is not False:
                raise ValueError("only global policy chunks may be immutable")
            if chunk["twin_anchor"] is not None:
                _require_text(chunk["twin_anchor"], f"{chunk_label}.twin_anchor")
            control = chunk["image_control"]
            if control is not None:
                if not isinstance(control, dict):
                    raise ValueError(f"{chunk_label}.image_control must be an object")
                _exact_keys(control, IMAGE_CONTROL_KEYS, f"{chunk_label}.image_control")
                group = _require_sha(control["group_key"], "image control group_key")
                variant = control["variant"]
                if (
                    variant not in {"real", "shuffled", "missing"}
                    or variant in controls[group]
                ):
                    raise ValueError(
                        "image control variants must be unique real/shuffled/missing"
                    )
                controls[group][variant] = chunk
            expected_hash = sha256_json(chunk_hash_payload(chunk))
            if chunk["content_sha256"] != expected_hash:
                raise ValueError(f"{chunk_label}.content_sha256 mismatch")
        if not any(chunk["kind"] == "global_policy" for chunk in chunks):
            raise ValueError(
                "every case requires at least one immutable global policy chunk"
            )
        if max(chunk["token_end"] for chunk in chunks) != case["context_tokens"]:
            raise ValueError("context_tokens must equal the final covered token offset")
        validate_case_chunking_receipt(
            case,
            source_artifacts_by_hash,
            tokenizer_artifact_sha256,
            chunker_artifact_sha256,
            chunks_by_key,
        )

        question_keys: set[str] = set()
        question_map: dict[str, dict[str, Any]] = {}
        for question_index, question in enumerate(questions):
            question_label = f"{label}.questions[{question_index}]"
            if not isinstance(question, dict):
                raise TypeError(f"{question_label} must be an object")
            _exact_keys(question, QUESTION_KEYS, question_label)
            question_key = _require_text(
                question["question_key"], f"{question_label}.question_key"
            )
            if question_key in question_keys:
                raise ValueError(f"duplicate question_key {question_key}")
            question_keys.add(question_key)
            question_map[question_key] = question
            _require_text(question["text"], f"{question_label}.text")
            _require_text(question["policy_role"], f"{question_label}.policy_role")
            if question["decision_type"] not in {"choice", "noul", "score"}:
                raise ValueError(f"{question_label}.decision_type is invalid")
            if (
                not isinstance(question["options"], list)
                or len(question["options"]) < 2
            ):
                raise ValueError(
                    f"{question_label}.options requires at least two options"
                )
            if any(
                not isinstance(option, str) or not option.strip()
                for option in question["options"]
            ):
                raise ValueError(f"{question_label}.options must be non-empty strings")
            if len(set(question["options"])) != len(question["options"]):
                raise ValueError(f"{question_label}.options must be unique")
            if question["twin_anchor"] is not None:
                _require_text(question["twin_anchor"], f"{question_label}.twin_anchor")
            decisive = question["decisive_chunk_keys"]
            distractors = question["known_distractor_chunk_keys"]
            if (
                not isinstance(decisive, list)
                or not decisive
                or len(decisive) != len(set(decisive))
            ):
                raise ValueError("decisive_chunk_keys must be a non-empty unique list")
            if (
                not isinstance(distractors, list)
                or not distractors
                or len(distractors) != len(set(distractors))
            ):
                raise ValueError(
                    "known_distractor_chunk_keys must be a non-empty unique list"
                )
            if set(decisive) & set(distractors):
                raise ValueError("decisive and distractor chunks must be disjoint")
            candidate_keys = {
                key
                for key, value in chunk_map.items()
                if value["kind"] != "global_policy"
            }
            if set(decisive) | set(distractors) != candidate_keys:
                raise ValueError(
                    "decisive and distractor labels must exhaust every retrievable chunk"
                )
            for chunk_key in decisive + distractors:
                if (
                    chunk_key not in chunk_map
                    or chunk_map[chunk_key]["kind"] == "global_policy"
                ):
                    raise ValueError(
                        "question references unknown or global-policy evidence"
                    )
            if any(
                chunk_map[chunk_key]["kind"] != "evidence" for chunk_key in decisive
            ):
                raise ValueError("decisive_chunk_keys must reference evidence chunks")
            proof = question["causal_proof"]
            if not isinstance(proof, dict):
                raise TypeError("causal_proof must be an object")
            _exact_keys(proof, PROOF_KEYS, "causal_proof")
            if proof["kind"] != "causal_annotation":
                raise ValueError("causal_proof.kind must be causal_annotation")
            removed = proof["removed_chunk_keys"]
            if not isinstance(removed, list) or removed != decisive:
                raise ValueError(
                    "causal_proof.removed_chunk_keys must exactly bind decisive_chunk_keys"
                )
            for field in PROOF_KEYS - {
                "kind",
                "changed_fields",
                "removed_chunk_keys",
                "full_outcome",
                "removed_outcome",
            }:
                _require_sha(proof[field], f"causal_proof.{field}")
            if proof["full_context_sha256"] != causal_context_sha256(
                case, question, set()
            ) or proof["removed_context_sha256"] != causal_context_sha256(
                case, question, set(decisive)
            ):
                raise ValueError(
                    "causal proof context hashes do not match ordered case content"
                )
            if proof["full_outcome_sha256"] == proof["removed_outcome_sha256"]:
                raise ValueError("decisive evidence removal must change the outcome")
            full_outcome = proof["full_outcome"]
            removed_outcome = proof["removed_outcome"]
            if (
                not isinstance(full_outcome, dict)
                or not full_outcome
                or not isinstance(removed_outcome, dict)
                or set(full_outcome) != set(removed_outcome)
            ):
                raise ValueError(
                    "causal outcomes must be non-empty objects with identical fields"
                )
            try:
                full_outcome_sha256 = sha256_bytes(
                    (
                        json.dumps(
                            full_outcome,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode()
                )
                removed_outcome_sha256 = sha256_bytes(
                    (
                        json.dumps(
                            removed_outcome,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode()
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "causal outcomes must be finite canonical JSON"
                ) from exc
            if (
                proof["full_outcome_sha256"] != full_outcome_sha256
                or proof["removed_outcome_sha256"] != removed_outcome_sha256
            ):
                raise ValueError("causal outcome hashes do not match inline outcomes")
            changed = proof["changed_fields"]
            if (
                not isinstance(changed, list)
                or not changed
                or any(not isinstance(field, str) or not field for field in changed)
            ):
                raise ValueError("causal_proof.changed_fields must be non-empty")
            actual_changed = sorted(
                field
                for field in full_outcome
                if full_outcome[field] != removed_outcome[field]
            )
            if changed != actual_changed:
                raise ValueError(
                    "causal_proof.changed_fields does not match outcome diff"
                )
            if question["content_sha256"] != sha256_json(
                question_hash_payload(question)
            ):
                raise ValueError(f"{question_label}.content_sha256 mismatch")

        for group, variants in controls.items():
            if set(variants) != {"real", "shuffled", "missing"}:
                raise ValueError(
                    f"image control {group} requires real, shuffled, and missing variants"
                )
            real, shuffled, missing = (
                variants[name] for name in ("real", "shuffled", "missing")
            )
            if real["modality"] != "image" or shuffled["modality"] != "image":
                raise ValueError("real and shuffled controls must be images")
            if (
                missing["media"] is not None
                or real["text"] != shuffled["text"]
                or real["text"] != missing["text"]
            ):
                raise ValueError(
                    "image controls require matched text and a media-free missing variant"
                )
            if real["media"]["sha256"] == shuffled["media"]["sha256"]:
                raise ValueError("real and shuffled image bytes must differ")
            matched = [
                question
                for question in questions
                if real["chunk_key"] in question["decisive_chunk_keys"]
                and {shuffled["chunk_key"], missing["chunk_key"]}.issubset(
                    question["known_distractor_chunk_keys"]
                )
            ]
            if len(matched) != 1:
                raise ValueError(
                    "each image control must bind exactly one causal question"
                )
            image_controls += 1

    if not twin_groups:
        raise ValueError("panel requires at least one paired Hebrew/English twin group")
    for group, cases in twin_groups.items():
        if len(cases) != 2 or {case["language"] for case in cases} != {"en", "he"}:
            raise ValueError(
                f"twin group {group} must contain exactly one English and one Hebrew case"
            )
        left, right = cases
        if (left["source_family"], left["source_domain"]) != (
            right["source_family"],
            right["source_domain"],
        ):
            raise ValueError("paired cases must share source family and domain")
        for key in ("questions", "chunks"):
            anchors_left = [item["twin_anchor"] for item in left[key]]
            anchors_right = [item["twin_anchor"] for item in right[key]]
            if any(anchor is None for anchor in anchors_left + anchors_right):
                raise ValueError("all paired questions/chunks require twin_anchor")
            if len(set(anchors_left)) != len(anchors_left) or sorted(
                anchors_left
            ) != sorted(anchors_right):
                raise ValueError(
                    f"paired {key} must have a one-to-one twin_anchor mapping"
                )
        left_chunks = {chunk["twin_anchor"]: chunk for chunk in left["chunks"]}
        right_chunks = {chunk["twin_anchor"]: chunk for chunk in right["chunks"]}
        for anchor, left_chunk in left_chunks.items():
            right_chunk = right_chunks[anchor]
            left_variant = (
                None
                if left_chunk["image_control"] is None
                else left_chunk["image_control"]["variant"]
            )
            right_variant = (
                None
                if right_chunk["image_control"] is None
                else right_chunk["image_control"]["variant"]
            )
            if (
                left_chunk["kind"],
                left_chunk["modality"],
                left_chunk["immutable"],
                left_variant,
            ) != (
                right_chunk["kind"],
                right_chunk["modality"],
                right_chunk["immutable"],
                right_variant,
            ):
                raise ValueError("paired chunks must preserve structural roles")
        left_questions = {q["twin_anchor"]: q for q in left["questions"]}
        right_questions = {q["twin_anchor"]: q for q in right["questions"]}
        for anchor, left_question in left_questions.items():
            right_question = right_questions[anchor]

            def aligned_chunk_anchors(
                question: dict[str, Any],
                field: str,
                case_chunks: dict[str, dict[str, Any]],
            ) -> list[str]:
                by_key = {
                    chunk["chunk_key"]: chunk["twin_anchor"]
                    for chunk in case_chunks.values()
                }
                return [by_key[key] for key in question[field]]

            if (
                len(left_question["options"]) != len(right_question["options"])
                or left_question["decision_type"] != right_question["decision_type"]
                or left_question["policy_role"] != right_question["policy_role"]
                or left_question["causal_proof"]["full_outcome"]
                != right_question["causal_proof"]["full_outcome"]
                or left_question["causal_proof"]["removed_outcome"]
                != right_question["causal_proof"]["removed_outcome"]
                or left_question["causal_proof"]["changed_fields"]
                != right_question["causal_proof"]["changed_fields"]
                or aligned_chunk_anchors(
                    left_question, "decisive_chunk_keys", left_chunks
                )
                != aligned_chunk_anchors(
                    right_question, "decisive_chunk_keys", right_chunks
                )
                or aligned_chunk_anchors(
                    left_question, "known_distractor_chunk_keys", left_chunks
                )
                != aligned_chunk_anchors(
                    right_question, "known_distractor_chunk_keys", right_chunks
                )
            ):
                raise ValueError(
                    "paired questions must preserve typed and evidence roles"
                )
    if image_controls == 0:
        raise ValueError(
            "panel requires at least one real/shuffled/missing image causal control"
        )

    validate_providers(
        panel,
        provider_generator_artifacts,
        (
            set(provider_generator_artifacts)
            if selected_precomputed_providers is None
            else selected_precomputed_providers
        ),
    )
    validate_causal_manifest(
        panel,
        causal_manifest,
        causal_manifest_sha256,
        annotation_builder_artifact,
    )
    validate_contamination_manifest(
        panel,
        contamination_manifest,
        contamination_manifest_sha256,
        contamination_auditor_artifact,
        contamination_corpus_artifacts,
        contamination_tokenizer_artifact,
        contamination_chunker_artifact,
    )
    return panel


def validate_providers(
    panel: dict[str, Any],
    generator_artifacts: dict[str, Path],
    selected_precomputed_providers: set[str],
) -> None:
    query_hashes = {
        q["content_sha256"] for case in panel["cases"] for q in case["questions"]
    }
    document_hashes = {
        c["content_sha256"] for case in panel["cases"] for c in case["chunks"]
    }
    unknown = selected_precomputed_providers - set(panel["providers"])
    if unknown:
        raise ValueError(f"unknown selected precomputed providers: {sorted(unknown)}")
    if set(generator_artifacts) != selected_precomputed_providers:
        raise ValueError(
            "provider generator artifacts must cover selected providers exactly"
        )
    for name, provider in panel["providers"].items():
        _require_text(name, "provider name")
        if not isinstance(provider, dict):
            raise TypeError(f"provider {name} must be an object")
        _exact_keys(provider, PROVIDER_KEYS, f"provider {name}")
        if provider["kind"] != "precomputed":
            raise ValueError(f"provider {name}.kind must be precomputed")
        if provider["origin"] not in {
            "local_open_model",
            "local_typical",
            "local_baseline",
        }:
            raise ValueError(f"provider {name}.origin is not an allowed local origin")
        if (
            provider["hosted_api"] is not False
            or provider["benchmark_derived"] is not False
        ):
            raise ValueError(f"provider {name} must be local and benchmark-independent")
        _require_text(provider["model"], f"provider {name}.model")
        _require_text(provider["revision"], f"provider {name}.revision")
        _require_sha(
            provider["generator_code_sha256"], f"provider {name}.generator_code_sha256"
        )
        if name in selected_precomputed_providers:
            verify_artifact(
                generator_artifacts[name],
                provider["generator_code_sha256"],
                f"provider {name} generator",
            )
        _require_sha(provider["artifact_sha256"], f"provider {name}.artifact_sha256")
        dimension = provider["dimension"]
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise ValueError(f"provider {name}.dimension must be a positive integer")
        for field, required in (
            ("queries", query_hashes),
            ("documents", document_hashes),
        ):
            vectors = provider[field]
            if not isinstance(vectors, dict) or set(vectors) != required:
                raise ValueError(
                    f"provider {name}.{field} must exactly cover panel content hashes"
                )
            for content_hash, vector in vectors.items():
                _validate_vector(
                    vector, dimension, f"provider {name}.{field}[{content_hash}]"
                )
        vector_hash = sha256_json(
            {"queries": provider["queries"], "documents": provider["documents"]}
        )
        if provider["artifact_sha256"] != vector_hash:
            raise ValueError(f"provider {name}.artifact_sha256 does not bind vectors")


def _validate_vector(vector: Any, dimension: int, label: str) -> np.ndarray:
    if not isinstance(vector, list) or len(vector) != dimension:
        raise ValueError(f"{label} must have dimension {dimension}")
    result = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(result).all() or float(np.linalg.norm(result)) <= 0.0:
        raise ValueError(f"{label} must be finite and non-zero")
    return result


def normalize_mrl(vector: Any, truncate_dim: int | None = None) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or result.size == 0:
        raise ValueError("embedding must be a finite non-empty vector")
    if truncate_dim is not None:
        if truncate_dim not in SUPPORTED_MRL_DIMS:
            raise ValueError(
                f"truncate_dim must be one of {sorted(SUPPORTED_MRL_DIMS)}"
            )
        if truncate_dim > result.size:
            raise ValueError("truncate_dim exceeds provider dimension")
        result = result[:truncate_dim]
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding norm must be finite and positive")
    normalized = result / norm
    if not np.isfinite(normalized).all() or not math.isclose(
        float(np.linalg.norm(normalized)), 1.0, abs_tol=1e-9
    ):
        raise ValueError("embedding normalization failed")
    return normalized


class Provider(Protocol):
    name: str

    def query(self, question: dict[str, Any]) -> np.ndarray: ...

    def document(self, chunk: dict[str, Any]) -> np.ndarray: ...

    def identity(self) -> dict[str, Any]: ...


class PrecomputedProvider:
    def __init__(self, name: str, spec: dict[str, Any], truncate_dim: int | None):
        self.name = name
        self.spec = spec
        self.truncate_dim = truncate_dim

    def query(self, question: dict[str, Any]) -> np.ndarray:
        return normalize_mrl(
            self.spec["queries"][question["content_sha256"]], self.truncate_dim
        )

    def document(self, chunk: dict[str, Any]) -> np.ndarray:
        return normalize_mrl(
            self.spec["documents"][chunk["content_sha256"]], self.truncate_dim
        )

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "precomputed",
            "origin": self.spec["origin"],
            "hosted_api": False,
            "benchmark_derived": False,
            "model": self.spec["model"],
            "revision": self.spec["revision"],
            "generator_code_sha256": self.spec["generator_code_sha256"],
            "artifact_sha256": self.spec["artifact_sha256"],
            "source_dimension": self.spec["dimension"],
            "scored_dimension": self.truncate_dim or self.spec["dimension"],
            "l2_normalized_after_truncation": True,
            "vectors_sha256": sha256_json(
                {"queries": self.spec["queries"], "documents": self.spec["documents"]}
            ),
        }


def _tree_hash(root: Path) -> tuple[str, int]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
        )
    return sha256_json(records), len(records)


def verify_frozen_snapshot(
    root: Path, expected_files: dict[str, str] = FROZEN_MODEL_FILES
) -> dict[str, str]:
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != set(expected_files):
        raise ValueError("pinned model snapshot file inventory mismatch")
    observed = {}
    for relative, expected in expected_files.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned model file is missing: {relative}")
        observed[relative] = file_sha256(path)
        if observed[relative] != expected:
            raise ValueError(f"pinned model file hash mismatch: {relative}")
    config = json.loads((root / "config.json").read_text())
    pooling = json.loads((root / "1_Pooling/config.json").read_text())
    if (
        config.get("model_type") != "embedding_gemma2"
        or config.get("text_config", {}).get("embedding_dim") != 768
        or pooling
        != {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True}
    ):
        raise ValueError("pinned model architecture/pooling contract mismatch")
    return observed


class LocalEmbeddingGemma2Provider:
    """Local-cache-only EmbeddingGemma 2 provider; never calls a hosted API."""

    name = "embeddinggemma2-local"

    def __init__(
        self,
        dtype: str,
        device: str,
        truncate_dim: int | None,
        runtime_lock: Path | None = None,
        runtime_lock_sha256: str | None = None,
        transformers_wheel: Path | None = None,
    ):
        if dtype not in {"bf16", "fp32"}:
            raise ValueError(
                "EmbeddingGemma 2 dtype must be bf16 or fp32; fp16 is unsafe"
            )
        if truncate_dim is not None and truncate_dim not in SUPPORTED_MRL_DIMS:
            raise ValueError(
                f"truncate_dim must be one of {sorted(SUPPORTED_MRL_DIMS)}"
            )
        if (
            runtime_lock is None
            or runtime_lock_sha256 is None
            or transformers_wheel is None
        ):
            raise ValueError(
                "local provider requires a frozen runtime lock, its external sha256, "
                "and the pinned Transformers wheel artifact"
            )
        self.runtime_lock_sha256 = verify_artifact(
            runtime_lock, runtime_lock_sha256, "local provider runtime lock"
        )
        self.transformers_wheel_sha256 = verify_artifact(
            transformers_wheel,
            TRANSFORMERS_WHEEL_SHA256,
            "Transformers wheel",
        )
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        try:
            import torch
            import transformers
            from huggingface_hub import snapshot_download
            from transformers import AutoModel, AutoProcessor
            from transformers.models.embedding_gemma2 import (
                EmbeddingGemma2Model,  # noqa: F401
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "the locked Transformers runtime lacks EmbeddingGemma 2 support; "
                "install a released version containing transformers.models.embedding_gemma2"
            ) from exc
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"EmbeddingGemma 2 provider requires Transformers {TRANSFORMERS_VERSION}; "
                f"found {transformers.__version__}"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True
            )
        ).resolve()
        self.frozen_files = verify_frozen_snapshot(snapshot)
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch_dtype
        ).to(device)
        self.model.eval()
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.truncate_dim = truncate_dim
        self.transformers_version = transformers.__version__
        self.snapshot_hash, self.snapshot_files = _tree_hash(snapshot)
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _query_text(question: dict[str, Any]) -> str:
        options = " | ".join(question["options"])
        return (
            "task: question answering | query: "
            f"{question['text']}\noptions: {options}\npolicy role: {question['policy_role']}"
            f"\ndecision type: {question['decision_type']}"
        )

    @staticmethod
    def _document_text(chunk: dict[str, Any]) -> str:
        return f"title: evidence | text: {chunk['text']}"

    def _embed(
        self, text: str, media: dict[str, Any] | None, cache_key: str
    ) -> np.ndarray:
        if cache_key in self._cache:
            return self._cache[cache_key]
        kwargs: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if media is not None:
            from PIL import Image

            image = Image.open(
                io.BytesIO(base64.b64decode(media["data_base64"]))
            ).convert("RGB")
            kwargs["text"] = [f"{text} <|image|>"]
            kwargs["images"] = [[image]]
        inputs = self.processor(**kwargs)
        attention_mask = inputs["attention_mask"]
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.model(**inputs, return_dict=True).last_hidden_state.float()
            mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        result = normalize_mrl(pooled[0].cpu().numpy(), self.truncate_dim)
        self._cache[cache_key] = result
        return result

    def query(self, question: dict[str, Any]) -> np.ndarray:
        return self._embed(
            self._query_text(question), None, "q:" + question["content_sha256"]
        )

    def document(self, chunk: dict[str, Any]) -> np.ndarray:
        return self._embed(
            self._document_text(chunk), chunk["media"], "d:" + chunk["content_sha256"]
        )

    def identity(self) -> dict[str, Any]:
        runtime = {
            "torch_version": self.torch.__version__,
            "cuda_runtime_version": self.torch.version.cuda,
            "cudnn_version": self.torch.backends.cudnn.version(),
        }
        if self.device == "cuda":
            properties = self.torch.cuda.get_device_properties(
                self.torch.cuda.current_device()
            )
            runtime["cuda_device"] = {
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            }
        return {
            "kind": "local_embeddinggemma2",
            "origin": "local_open_model",
            "hosted_api": False,
            "benchmark_derived": False,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": self.dtype,
            "device": self.device,
            "source_dimension": 768,
            "scored_dimension": self.truncate_dim or 768,
            "l2_normalized_after_truncation": True,
            "local_files_only": True,
            "transformers_version": self.transformers_version,
            "transformers_wheel_sha256": self.transformers_wheel_sha256,
            "transformers_source_commit": TRANSFORMERS_SOURCE_COMMIT,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "snapshot_tree_sha256": self.snapshot_hash,
            "snapshot_files": self.snapshot_files,
            "frozen_file_sha256": self.frozen_files,
            "deterministic_algorithms": True,
            "runtime_identity": runtime,
        }


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def bm25_rank(
    question: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[tuple[str, float]]:
    query = tokenize(
        " ".join(
            [
                question["text"],
                question["decision_type"],
                question["policy_role"],
                *question["options"],
            ]
        )
    )
    documents = [tokenize(chunk["text"]) for chunk in candidates]
    document_frequency = Counter(token for doc in documents for token in set(doc))
    average_length = sum(map(len, documents)) / len(documents) if documents else 0.0
    scores = []
    for chunk, document in zip(candidates, documents, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for token in query:
            frequency = frequencies[token]
            if frequency == 0:
                continue
            inverse = math.log(
                1.0
                + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * len(document) / max(average_length, 1.0)
            )
            score += inverse * frequency * 2.2 / denominator
        scores.append((chunk["chunk_key"], score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def dense_rank(
    question: dict[str, Any], candidates: list[dict[str, Any]], provider: Provider
) -> list[tuple[str, float]]:
    query = provider.query(question)
    scored = []
    for chunk in candidates:
        document = provider.document(chunk)
        if query.shape != document.shape:
            raise ValueError("query and document embeddings must share a dimension")
        scored.append((chunk["chunk_key"], float(np.dot(query, document))))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def _metric_summary(
    records: list[dict[str, Any]], ks: tuple[int, ...]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in ks:
        complete = []
        item_recall = []
        false_positives = 0
        total_negatives = 0
        precisions = []
        for record in records:
            selected = set(record["ranking"][:k])
            decisive = set(record["decisive"])
            distractors = set(record["distractors"])
            complete.append(float(decisive.issubset(selected)))
            item_recall.append(len(selected & decisive) / len(decisive))
            false_positives += len(selected & distractors)
            total_negatives += len(distractors)
            precisions.append(len(selected & decisive) / max(len(selected), 1))
        output[f"decisive_recall@{k}"] = sum(complete) / len(complete)
        output[f"decisive_item_recall@{k}"] = sum(item_recall) / len(item_recall)
        output[f"distractor_fpr@{k}"] = false_positives / max(total_negatives, 1)
        output[f"precision@{k}"] = sum(precisions) / len(precisions)
    reciprocal_ranks = []
    for record in records:
        decisive = set(record["decisive"])
        first = next(
            (
                rank
                for rank, chunk_key in enumerate(record["ranking"], start=1)
                if chunk_key in decisive
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
    output["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    output["questions"] = len(records)
    return output


def _slice_summaries(
    records: list[dict[str, Any]], ks: tuple[int, ...]
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        for dimension, values in record["slices"].items():
            for value in values:
                grouped[dimension][value].append(record)
    return {
        dimension: {
            value: _metric_summary(items, ks)
            for value, items in sorted(categories.items())
        }
        for dimension, categories in sorted(grouped.items())
    }


def _twin_agreement(
    panel: dict[str, Any],
    rankings: dict[tuple[str, str], list[str]],
    ks: tuple[int, ...],
) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in panel["cases"]:
        if case["twin_group_key"] is not None:
            groups[case["twin_group_key"]].append(case)
    values: dict[int, list[float]] = defaultdict(list)
    position_values: dict[int, list[float]] = defaultdict(list)
    decisive_rank_deltas = []
    for cases in groups.values():
        english = next(case for case in cases if case["language"] == "en")
        hebrew = next(case for case in cases if case["language"] == "he")
        q_he = {q["twin_anchor"]: q for q in hebrew["questions"]}
        chunks_en = {c["chunk_key"]: c["twin_anchor"] for c in english["chunks"]}
        chunks_he = {c["chunk_key"]: c["twin_anchor"] for c in hebrew["chunks"]}
        for question_en in english["questions"]:
            question_he = q_he[question_en["twin_anchor"]]
            left = rankings[(english["case_key"], question_en["question_key"])]
            right = rankings[(hebrew["case_key"], question_he["question_key"])]
            for k in ks:
                left_order = [chunks_en[key] for key in left[:k]]
                right_order = [chunks_he[key] for key in right[:k]]
                left_anchors = set(left_order)
                right_anchors = set(right_order)
                union = left_anchors | right_anchors
                values[k].append(len(left_anchors & right_anchors) / max(len(union), 1))
                compared = max(len(left_order), len(right_order), 1)
                position_values[k].append(
                    sum(a == b for a, b in zip(left_order, right_order, strict=False))
                    / compared
                )
            decisive_en = {chunks_en[key] for key in question_en["decisive_chunk_keys"]}
            decisive_he = {chunks_he[key] for key in question_he["decisive_chunk_keys"]}
            rank_en = min(
                i
                for i, key in enumerate(left, start=1)
                if chunks_en[key] in decisive_en
            )
            rank_he = min(
                i
                for i, key in enumerate(right, start=1)
                if chunks_he[key] in decisive_he
            )
            decisive_rank_deltas.append(abs(rank_en - rank_he))
    output = {
        f"top{k}_set_jaccard": sum(items) / len(items) for k, items in values.items()
    }
    output.update(
        {
            f"top{k}_position_agreement": sum(items) / len(items)
            for k, items in position_values.items()
        }
    )
    output["mean_decisive_rank_delta"] = sum(decisive_rank_deltas) / len(
        decisive_rank_deltas
    )
    return output


def _image_margins(
    panel: dict[str, Any], score_maps: dict[tuple[str, str], dict[str, float]]
) -> dict[str, Any]:
    margins = []
    for case in panel["cases"]:
        chunks = {chunk["chunk_key"]: chunk for chunk in case["chunks"]}
        controls: dict[str, dict[str, str]] = defaultdict(dict)
        for chunk in case["chunks"]:
            if chunk["image_control"] is not None:
                controls[chunk["image_control"]["group_key"]][
                    chunk["image_control"]["variant"]
                ] = chunk["chunk_key"]
        for group, variants in controls.items():
            question = next(
                q
                for q in case["questions"]
                if variants["real"] in q["decisive_chunk_keys"]
            )
            scores = score_maps[(case["case_key"], question["question_key"])]
            margins.append(
                {
                    "case_key": case["case_key"],
                    "question_key": question["question_key"],
                    "control_group_key": group,
                    "real_minus_shuffled": scores[variants["real"]]
                    - scores[variants["shuffled"]],
                    "real_minus_missing": scores[variants["real"]]
                    - scores[variants["missing"]],
                    "real_content_sha256": chunks[variants["real"]]["content_sha256"],
                }
            )
    return {
        "controls": len(margins),
        "mean_real_minus_shuffled": sum(item["real_minus_shuffled"] for item in margins)
        / len(margins),
        "mean_real_minus_missing": sum(item["real_minus_missing"] for item in margins)
        / len(margins),
        "positive_shuffled_fraction": sum(
            item["real_minus_shuffled"] > 0 for item in margins
        )
        / len(margins),
        "positive_missing_fraction": sum(
            item["real_minus_missing"] > 0 for item in margins
        )
        / len(margins),
        "items": margins,
    }


def score_method(
    panel: dict[str, Any],
    name: str,
    ks: tuple[int, ...],
    provider: Provider | None = None,
) -> dict[str, Any]:
    records = []
    rankings: dict[tuple[str, str], list[str]] = {}
    score_maps: dict[tuple[str, str], dict[str, float]] = {}
    unions: dict[str, list[str]] = {}
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_k = max(ks)
    for case in panel["cases"]:
        candidates = [
            chunk for chunk in case["chunks"] if chunk["kind"] != "global_policy"
        ]
        selected = {
            chunk["chunk_key"]
            for chunk in case["chunks"]
            if chunk["kind"] == "global_policy"
        }
        chunk_by_key = {chunk["chunk_key"]: chunk for chunk in case["chunks"]}
        for question in case["questions"]:
            ranked = (
                bm25_rank(question, candidates)
                if provider is None
                else dense_rank(question, candidates, provider)
            )
            key = (case["case_key"], question["question_key"])
            ranking_keys = [chunk_key for chunk_key, _ in ranked]
            rankings[key] = ranking_keys
            score_maps[key] = dict(ranked)
            for chunk_key in ranking_keys[:max_k]:
                selected.add(chunk_key)
            decisive_positions = [
                (
                    chunk_by_key[chunk_key]["token_start"]
                    + chunk_by_key[chunk_key]["token_end"]
                )
                / 2
                for chunk_key in question["decisive_chunk_keys"]
            ]
            position_slices = sorted(
                {
                    ("early", "middle", "late")[
                        min(2, int(position * 3 // case["context_tokens"]))
                    ]
                    for position in decisive_positions
                }
            )
            record = {
                "case_key": case["case_key"],
                "question_key": question["question_key"],
                "decision_type": question["decision_type"],
                "ranking": ranking_keys,
                "decisive": question["decisive_chunk_keys"],
                "distractors": question["known_distractor_chunk_keys"],
                "slices": {
                    "language": [case["language"]],
                    "decision_type": [question["decision_type"]],
                    "context_length": [
                        "long" if case["context_tokens"] >= 8192 else "short"
                    ],
                    "source_domain": [case["source_domain"]],
                    "source_family": [case["source_family"]],
                    "evidence_position": position_slices,
                    "decisive_modality": sorted(
                        {
                            chunk_by_key[chunk_key]["modality"]
                            for chunk_key in question["decisive_chunk_keys"]
                        }
                    ),
                },
            }
            records.append(record)
            by_language[case["language"]].append(record)
        unions[case["case_key"]] = [
            chunk["chunk_key"]
            for chunk in case["chunks"]
            if chunk["chunk_key"] in selected
        ]
    return {
        "method": name,
        "provider": {
            "kind": "bm25",
            "tokenization": "unicode-word-casefold",
            "k1": 1.2,
            "b": 0.75,
        }
        if provider is None
        else provider.identity(),
        "overall": _metric_summary(records, ks),
        "by_language": {
            language: _metric_summary(items, ks)
            for language, items in sorted(by_language.items())
        },
        "slices": _slice_summaries(records, ks),
        "he_en_twin_agreement": _twin_agreement(panel, rankings, ks),
        "image_margin": _image_margins(panel, score_maps),
        "per_question": records,
        "per_case_evidence_union_at_max_k": unions,
        "union_contract": {
            "max_k": max_k,
            "per_question_top_k": True,
            "all_immutable_global_policy_chunks_included": True,
            "reassembled_in_original_case_order": True,
        },
    }


def build_receipt(
    panel: dict[str, Any],
    raw_input_sha256: str,
    source_snapshot_manifest_sha256: str,
    causal_manifest_sha256: str,
    contamination_manifest_sha256: str,
    ks: tuple[int, ...],
    methods: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "input_raw_sha256": raw_input_sha256,
        "input_canonical_sha256": sha256_json(panel),
        "trusted_source_snapshot_manifest_sha256": source_snapshot_manifest_sha256,
        "trusted_causal_manifest_sha256": causal_manifest_sha256,
        "trusted_contamination_manifest_sha256": contamination_manifest_sha256,
        "code_sha256": file_sha256(Path(__file__)),
        "configuration": {"ks": list(ks)},
        "panel": {
            "cases": len(panel["cases"]),
            "questions": sum(len(case["questions"]) for case in panel["cases"]),
            "languages": dict(
                sorted(Counter(case["language"] for case in panel["cases"]).items())
            ),
            "benchmark_derived": False,
        },
        "methods": methods,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different receipt: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_replace_json(path: Path, value: Any) -> None:
    """Atomically replace non-deterministic runtime telemetry, never a score receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ks must be comma-separated positive integers"
        ) from exc
    if not ks or any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("ks must be comma-separated positive integers")
    return ks


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def validate_io_paths(
    panel: Path,
    causal_manifest: Path,
    contamination_manifest: Path,
    source_snapshot_manifest: Path,
    output: Path,
    runtime_output: Path,
    artifact_inputs: tuple[Path, ...] = (),
) -> None:
    inputs = {
        "panel": panel.resolve(strict=True),
        "causal manifest": causal_manifest.resolve(strict=True),
        "contamination manifest": contamination_manifest.resolve(strict=True),
        "source snapshot manifest": source_snapshot_manifest.resolve(strict=True),
    }
    if len(set(inputs.values())) != len(inputs):
        raise ValueError("panel and manifest input paths must not alias")
    outputs = {
        "score output": output.resolve(strict=False),
        "runtime output": runtime_output.resolve(strict=False),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("score and runtime output paths must not alias")
    protected = [
        *inputs.values(),
        *(path.resolve(strict=True) for path in artifact_inputs),
    ]
    for output_path in outputs.values():
        for input_path in protected:
            same_path = output_path == input_path
            same_inode = (
                output_path.exists()
                and input_path.exists()
                and os.path.samefile(output_path, input_path)
            )
            inside_directory = input_path.is_dir() and output_path.is_relative_to(
                input_path
            )
            if same_path or same_inode or inside_directory:
                raise ValueError(
                    "output paths must not alias or mutate any supplied input artifact"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--causal-manifest", type=Path, required=True)
    parser.add_argument("--causal-manifest-sha256", required=True)
    parser.add_argument("--contamination-manifest", type=Path, required=True)
    parser.add_argument("--contamination-manifest-sha256", required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--source-snapshot-manifest-sha256", required=True)
    parser.add_argument(
        "--source-snapshot-artifact",
        type=parse_named_path,
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--causal-annotation-builder-artifact", type=Path, required=True
    )
    parser.add_argument("--contamination-auditor-artifact", type=Path, required=True)
    parser.add_argument(
        "--contamination-corpus-artifact", type=Path, action="append", required=True
    )
    parser.add_argument("--contamination-tokenizer-artifact", type=Path, required=True)
    parser.add_argument("--contamination-chunker-artifact", type=Path, required=True)
    parser.add_argument(
        "--provider-generator-artifact",
        type=parse_named_path,
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--ks", type=parse_ks, default=(1, 2, 4))
    parser.add_argument(
        "--provider", action="append", default=[], help="precomputed provider name"
    )
    parser.add_argument("--truncate-dim", type=int)
    parser.add_argument("--embeddinggemma2-local", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--runtime-lock-sha256")
    parser.add_argument("--transformers-wheel", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    runtime_output = args.runtime_output or args.output.with_suffix(".runtime.json")
    generator_artifacts = dict(args.provider_generator_artifact)
    if len(generator_artifacts) != len(args.provider_generator_artifact):
        raise ValueError("provider generator artifact names must be unique")
    if len(set(args.provider)) != len(args.provider):
        raise ValueError("selected provider names must be unique")
    source_snapshot_artifacts = dict(args.source_snapshot_artifact)
    if len(source_snapshot_artifacts) != len(args.source_snapshot_artifact):
        raise ValueError("source snapshot artifact names must be unique")
    artifact_inputs = [
        args.causal_annotation_builder_artifact,
        args.contamination_auditor_artifact,
        *args.contamination_corpus_artifact,
        args.contamination_tokenizer_artifact,
        args.contamination_chunker_artifact,
        *source_snapshot_artifacts.values(),
        *generator_artifacts.values(),
    ]
    artifact_inputs.extend(
        path
        for path in (args.runtime_lock, args.transformers_wheel)
        if path is not None
    )
    validate_io_paths(
        args.panel,
        args.causal_manifest,
        args.contamination_manifest,
        args.source_snapshot_manifest,
        args.output,
        runtime_output,
        tuple(artifact_inputs),
    )
    raw = args.panel.read_bytes()
    causal_raw = args.causal_manifest.read_bytes()
    contamination_raw = args.contamination_manifest.read_bytes()
    source_snapshot_raw = args.source_snapshot_manifest.read_bytes()
    panel = validate_panel(
        parse_unique_finite_json(raw, "panel"),
        parse_unique_finite_json(source_snapshot_raw, "source snapshot manifest"),
        args.source_snapshot_manifest_sha256,
        parse_unique_finite_json(causal_raw, "causal manifest"),
        args.causal_manifest_sha256,
        parse_unique_finite_json(contamination_raw, "contamination manifest"),
        args.contamination_manifest_sha256,
        annotation_builder_artifact=args.causal_annotation_builder_artifact,
        contamination_auditor_artifact=args.contamination_auditor_artifact,
        contamination_corpus_artifacts=args.contamination_corpus_artifact,
        contamination_tokenizer_artifact=args.contamination_tokenizer_artifact,
        contamination_chunker_artifact=args.contamination_chunker_artifact,
        source_snapshot_artifacts=source_snapshot_artifacts,
        provider_generator_artifacts=generator_artifacts,
        selected_precomputed_providers=set(args.provider),
    )
    method_times = []
    started = time.perf_counter()
    methods = [score_method(panel, "bm25", args.ks)]
    method_times.append(
        {"method": "bm25", "wall_seconds": time.perf_counter() - started}
    )
    selected = args.provider
    unknown = sorted(set(selected) - set(panel["providers"]))
    if unknown:
        raise ValueError(f"unknown providers: {unknown}")
    for name in selected:
        provider = PrecomputedProvider(
            name, panel["providers"][name], args.truncate_dim
        )
        started = time.perf_counter()
        methods.append(score_method(panel, name, args.ks, provider))
        method_times.append(
            {"method": name, "wall_seconds": time.perf_counter() - started}
        )
    if args.embeddinggemma2_local:
        if args.device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        provider = LocalEmbeddingGemma2Provider(
            args.dtype,
            args.device,
            args.truncate_dim,
            args.runtime_lock,
            args.runtime_lock_sha256,
            args.transformers_wheel,
        )
        model_load_wall_seconds = time.perf_counter() - load_started
        started = time.perf_counter()
        methods.append(score_method(panel, provider.name, args.ks, provider))
        method_times.append(
            {"method": provider.name, "wall_seconds": time.perf_counter() - started}
        )
    receipt = build_receipt(
        panel,
        sha256_bytes(raw),
        args.source_snapshot_manifest_sha256,
        args.causal_manifest_sha256,
        args.contamination_manifest_sha256,
        args.ks,
        methods,
    )
    atomic_write_json(args.output, receipt)
    peak_cuda_bytes = 0
    if args.embeddinggemma2_local and args.device == "cuda":
        peak_cuda_bytes = int(provider.torch.cuda.max_memory_allocated())
    atomic_replace_json(
        runtime_output,
        {
            "schema_version": "r052-runtime-observation-v1",
            "score_receipt_sha256": receipt["receipt_sha256"],
            "method_wall_time": method_times,
            "method_wall_time_scope": "warm_retrieval_scoring_excludes_model_load",
            "model_load_wall_seconds": (
                model_load_wall_seconds if args.embeddinggemma2_local else 0.0
            ),
            "cold_total_wall_seconds": time.perf_counter() - process_started,
            "peak_cuda_memory_bytes": peak_cuda_bytes,
            "non_deterministic_observation": True,
        },
    )
    print(
        json.dumps(
            {"output": str(args.output), "receipt_sha256": receipt["receipt_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
