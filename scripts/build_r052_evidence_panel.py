#!/usr/bin/env python3
"""Build a provenance-bound R052 evidence panel without inventing evidence.

The builder has two deliberately separate phases::

    prepare-audit  -> immutable, content-addressed audit input
    finalize       -> panel + causal/contamination manifests + build receipt

Authors supply natural dossier text, evidence labels, translated twins, image
controls, and causal outcomes.  An external auditor supplies overlap results.
This module only validates, binds, and packages those claims; it does not
translate text, shuffle images, execute policy semantics, or perform a
contamination audit.  ``test_fixture`` inputs require an explicit CLI opt-in
and are labelled as such in the receipt, so tiny tests cannot masquerade as a
production panel.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from scripts import replay_r052_typed_policy as policy_replay
    from scripts import score_r052_evidence_retrieval as scoring
except ModuleNotFoundError:  # Supports ``uv run python scripts/build_...py``.
    import replay_r052_typed_policy as policy_replay
    import score_r052_evidence_retrieval as scoring

SPEC_VERSION = "r052-authored-dossier-spec-v4"
SOURCE_MANIFEST_VERSION = "r052-source-provenance-manifest-v5"
ANNOTATION_VERSION = "r052-authored-causal-annotations-v2"
AUDIT_INPUT_VERSION = "r052-contamination-audit-input-v2"
AUDIT_RESULTS_VERSION = "r052-external-contamination-results-v1"
PROVIDER_BUNDLE_VERSION = "r052-precomputed-provider-bundle-v2"
BUILD_RECEIPT_VERSION = "r052-evidence-panel-build-receipt-v2"
LICENSE_POLICY = "approved-evaluation-and-redistribution-v1"
AUDIT_ALGORITHM = "exact-sha256+minhash-jaccard-v1"
LONG_CONTEXT_TOKENS = 8_192
PRODUCTION_MINIMUMS = {
    "cases": 40,
    "questions": 60,
    "long_english_cases": 10,
    "twin_groups": 10,
    "hebrew_questions": 10,
    "image_questions": 10,
    "choice_questions": 10,
    "noul_questions": 10,
    "score_questions": 10,
    "position_questions": 10,
    "long_source_domains": 5,
    "retrievable_chunks_per_long_case": 15,
}

SPEC_KEYS = {"schema_version", "panel_profile", "benchmark_derived", "cases"}
CASE_SPEC_KEYS = {
    "case_ref",
    "language",
    "twin_ref",
    "source_snapshot",
    "alignment_receipt_artifact",
    "source_domain",
    "context_tokens",
    "chunking",
    "chunks",
    "questions",
}
CHUNKING_KEYS = {"mode", "boundary_receipt_artifact"}
CHUNK_SPEC_KEYS = {
    "chunk_ref",
    "kind",
    "modality",
    "text",
    "token_start",
    "token_end",
    "twin_anchor",
    "media_artifact",
    "image_control",
    "derivation",
}
IMAGE_SPEC_KEYS = {"group_ref", "variant", "mapping_receipt_artifact"}
QUESTION_SPEC_KEYS = {
    "question_ref",
    "text",
    "options",
    "option_ids",
    "decision_type",
    "policy_role",
    "twin_anchor",
    "decisive_chunk_refs",
    "known_distractor_chunk_refs",
}
SOURCE_MANIFEST_KEYS = {
    "schema_version",
    "license_policy",
    "benchmark_derived",
    "license_policies",
    "artifacts",
    "snapshots",
    "acquisition_manifest_artifact",
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
ANNOTATION_KEYS = {"schema_version", "benchmark_derived", "entries"}
ANNOTATION_ENTRY_KEYS = {
    "case_ref",
    "question_ref",
    "removed_chunk_refs",
    "full_outcome",
    "removed_outcome",
    "changed_fields",
}
AUDIT_RESULTS_KEYS = {
    "schema_version",
    "audit_input_sha256",
    "sealed_benchmark_text_included",
    "algorithm",
    "auditor_code_sha256",
    "audited_corpus_roots_sha256",
    "tokenizer_receipt_sha256",
    "chunker_receipt_sha256",
    "entries",
}
AUDIT_RESULT_ENTRY_KEYS = {
    "case_ref",
    "case_content_sha256",
    "status",
    "exact_overlap_count",
    "near_overlap_max_jaccard",
}
PROVIDER_BUNDLE_KEYS = {
    "schema_version",
    "origin",
    "hosted_api",
    "benchmark_derived",
    "model",
    "revision",
    "generator_code_sha256",
    "dimension",
    "queries",
    "documents",
}
ALLOWED_LICENSES = {
    "synthetic": {"LicenseRef-Synthetic"},
    "public_domain": {
        "CC0-1.0",
        "PDM-1.0",
        "LicenseRef-USGov-Public-Domain",
    },
    "permissive": {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
    },
    "first_party": {"LicenseRef-First-Party"},
}
ARTIFACT_ROLES = {
    "reviewed_source_text",
    "source_acquisition_run_receipt",
    "source_acquisition_manifest",
    "reviewed_source_acquisition_binding",
    "image",
    "translation_alignment_receipt",
    "image_control_receipt",
    "chunk_boundary_receipt",
}
FORBIDDEN_MARKERS = tuple(
    marker.encode() for marker in scoring.FORBIDDEN_LINEAGE_MARKERS
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    return scoring.canonical_json(value)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, expected_sha256: str, label: str) -> tuple[Any, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} must be a non-empty regular file")
    raw = path.read_bytes()
    digest = _sha_bytes(raw)
    if digest != _sha(expected_sha256, f"expected {label} hash"):
        raise ValueError(f"{label} differs from its external SHA-256 expectation")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_json,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be valid finite UTF-8 JSON") from exc
    if raw != _canonical(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value, digest


def _check_no_symlinks(path: Path, label: str) -> Path:
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


def _contains_forbidden_bytes(path: Path) -> bool:
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    for file_path in files:
        relative = (
            file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        )
        if any(marker in relative.casefold().encode() for marker in FORBIDDEN_MARKERS):
            return True
        overlap = max(map(len, FORBIDDEN_MARKERS)) - 1
        tail = b""
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                folded = (tail + block).lower()
                if any(marker in folded for marker in FORBIDDEN_MARKERS):
                    return True
                tail = folded[-overlap:]
    return False


def _validate_source_manifest(
    manifest: Any, artifact_paths: Mapping[str, Path]
) -> dict[str, Path]:
    if not isinstance(manifest, dict):
        raise TypeError("source manifest must be an object")
    if manifest.get("schema_version") in {
        "r052-source-provenance-manifest-v2",
        "r052-source-provenance-manifest-v3",
        "r052-source-provenance-manifest-v4",
    }:
        raise ValueError(
            "source manifest v2/v3/v4 requires migration to v5 acquisition provenance"
        )
    _exact_keys(manifest, SOURCE_MANIFEST_KEYS, "source manifest")
    if (
        manifest["schema_version"] != SOURCE_MANIFEST_VERSION
        or manifest["license_policy"] != LICENSE_POLICY
        or manifest["benchmark_derived"] is not False
    ):
        raise ValueError("source manifest policy/provenance is invalid")
    license_policies = manifest["license_policies"]
    if not isinstance(license_policies, dict) or not license_policies:
        raise ValueError("source license_policies must be a non-empty object")
    for name, policy in license_policies.items():
        _text(name, "source license policy name")
        if not isinstance(policy, dict):
            raise TypeError(f"source license policy {name} must be an object")
        _exact_keys(policy, SOURCE_LICENSE_KEYS, f"source license policy {name}")
        family = policy["source_family"]
        if family not in scoring.SOURCE_FAMILIES:
            raise ValueError(f"source license policy {name} family is not allowed")
        if policy["license_id"] not in ALLOWED_LICENSES[family]:
            raise ValueError(
                f"source license policy {name} license/family declaration conflicts"
            )
        license_url = _text(policy["license_url"], f"license policy {name} URL")
        parsed = urlparse(license_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
            raise ValueError("license URLs must be HTTPS")
        _text(policy["attribution"], f"source license policy {name} attribution")
        if any(
            policy[field] is not True
            for field in (
                "license_cleared",
                "evaluation_use_cleared",
                "redistribution_cleared",
            )
        ):
            raise ValueError(f"source license policy {name} is not cleared")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("source manifest artifacts must be a non-empty object")
    if set(artifacts) != set(artifact_paths):
        raise ValueError("source artifact paths must cover the manifest exactly")
    resolved: dict[str, Path] = {}
    declared_hashes: set[str] = set()
    for name, declaration in artifacts.items():
        _text(name, "source artifact name")
        if not isinstance(declaration, dict):
            raise TypeError(f"source artifact {name} declaration must be an object")
        _exact_keys(declaration, SOURCE_ARTIFACT_KEYS, f"source artifact {name}")
        if declaration["role"] not in ARTIFACT_ROLES:
            raise ValueError(f"source artifact {name} role is not allowed")
        path = _check_no_symlinks(artifact_paths[name], f"source artifact {name}")
        declared_hash = _sha(declaration["sha256"], f"source artifact {name} hash")
        if declared_hash in declared_hashes:
            raise ValueError("source artifact hash declarations must be unique")
        declared_hashes.add(declared_hash)
        if scoring.artifact_sha256(path) != declared_hash:
            raise ValueError(f"source artifact {name} hash mismatch")
        if declaration["role"] == "reviewed_source_text":
            scoring.validate_reviewed_text_artifact(path, f"source artifact {name}")
        if _contains_forbidden_bytes(path):
            raise ValueError(
                f"source artifact {name} contains benchmark lineage markers"
            )
        resolved[name] = path

    snapshots = manifest["snapshots"]
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("source snapshots must be a non-empty object")
    used_snapshot_artifacts: list[str] = []
    for name, snapshot in snapshots.items():
        _text(name, "source snapshot name")
        if not isinstance(snapshot, dict):
            raise TypeError(f"source snapshot {name} must be an object")
        _exact_keys(snapshot, SOURCE_SNAPSHOT_KEYS, f"source snapshot {name}")
        reviewed_name = snapshot["reviewed_text_artifact"]
        acquisition_name = snapshot["acquisition_binding_artifact"]
        if (
            reviewed_name not in artifacts
            or artifacts[reviewed_name]["role"] != "reviewed_source_text"
            or acquisition_name not in artifacts
            or artifacts[acquisition_name]["role"]
            != "reviewed_source_acquisition_binding"
            or reviewed_name == acquisition_name
        ):
            raise ValueError(
                "source snapshot must bind reviewed text and acquisition binding"
            )
        used_snapshot_artifacts.extend((reviewed_name, acquisition_name))
        policy_name = snapshot["license_policy"]
        if policy_name not in license_policies:
            raise ValueError("source snapshot references an unknown license policy")
        if snapshot["source_family"] != license_policies[policy_name]["source_family"]:
            raise ValueError("source snapshot family conflicts with its license policy")
        upstream_url = _text(snapshot["upstream_url"], f"source snapshot {name} URL")
        parsed = urlparse(upstream_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
            raise ValueError("upstream URLs must be HTTPS")
        revision = _text(
            snapshot["upstream_revision"], f"source snapshot {name} revision"
        )
        if revision.casefold() in {"head", "latest", "main", "master", "unknown"}:
            raise ValueError("source snapshot revision must be immutable and exact")
        try:
            parsed_date = date.fromisoformat(snapshot["upstream_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError("upstream_date must be an ISO YYYY-MM-DD date") from exc
        if parsed_date.isoformat() != snapshot["upstream_date"]:
            raise ValueError("upstream_date must use canonical YYYY-MM-DD form")
        untouched = _text(
            snapshot["untouched_split"], "source snapshot untouched split"
        )
        if untouched.casefold() in {"unknown", "unspecified", "none"}:
            raise ValueError("source snapshot untouched split must be exact")
        _text(snapshot["attribution"], f"source snapshot {name} attribution")
        if (
            snapshot["third_party_images_excluded"] is not True
            or snapshot["natural_language_authored"] is not True
            or snapshot["benchmark_derived"] is not False
        ):
            raise ValueError("source snapshot is not eligible and independent")
    expected_snapshot_artifacts = {
        name
        for name, declaration in artifacts.items()
        if declaration["role"]
        in {"reviewed_source_text", "reviewed_source_acquisition_binding"}
    }
    if (
        len(used_snapshot_artifacts) != len(set(used_snapshot_artifacts))
        or set(used_snapshot_artifacts) != expected_snapshot_artifacts
    ):
        raise ValueError(
            "reviewed text/acquisition bindings must occur exactly once across snapshots"
        )
    if {snapshot["license_policy"] for snapshot in snapshots.values()} != set(
        license_policies
    ):
        raise ValueError("source snapshots must use every declared license policy")
    scoring._reject_benchmark_lineage(manifest)
    scoring.validate_source_acquisition_provenance(manifest, resolved)
    return resolved


def _validate_spec(
    spec: Any, source_manifest: Mapping[str, Any], allow_test_profile: bool
) -> None:
    if not isinstance(spec, dict):
        raise TypeError("dossier spec must be an object")
    _exact_keys(spec, SPEC_KEYS, "dossier spec")
    if spec["schema_version"] != SPEC_VERSION:
        raise ValueError(
            f"prior dossier specs require migration; schema_version must be {SPEC_VERSION}"
        )
    if spec["benchmark_derived"] is not False:
        raise ValueError("dossier spec provenance is invalid")
    profile = spec["panel_profile"]
    if profile not in {"production", "test_fixture"}:
        raise ValueError("panel_profile must be production or test_fixture")
    if profile == "test_fixture" and not allow_test_profile:
        raise ValueError("test_fixture profile requires explicit opt-in")
    cases = spec["cases"]
    sources = source_manifest["artifacts"]
    snapshots = source_manifest["snapshots"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("dossier spec cases must be a non-empty list")
    seen_cases: set[str] = set()
    used_artifacts: set[str] = {
        source_manifest["acquisition_manifest_artifact"],
        source_manifest["acquisition_run_receipt_artifact"],
    }
    has_long = False
    for case_index, case in enumerate(cases):
        label = f"cases[{case_index}]"
        if not isinstance(case, dict):
            raise TypeError(f"{label} must be an object")
        _exact_keys(case, CASE_SPEC_KEYS, label)
        case_ref = _text(case["case_ref"], f"{label}.case_ref")
        if case_ref in seen_cases:
            raise ValueError("case_ref values must be unique")
        seen_cases.add(case_ref)
        if case["language"] not in {"en", "he"}:
            raise ValueError(f"{label}.language must be en or he")
        if case["source_domain"] not in scoring.SOURCE_DOMAINS:
            raise ValueError(f"{label}.source_domain is not allowed")
        context_tokens = case["context_tokens"]
        if type(context_tokens) is not int or context_tokens < 1:
            raise ValueError(f"{label}.context_tokens must be positive")
        has_long |= context_tokens >= LONG_CONTEXT_TOKENS
        snapshot_name = case["source_snapshot"]
        if snapshot_name not in snapshots:
            raise ValueError(f"{label} must reference a declared source snapshot")
        snapshot = snapshots[snapshot_name]
        used_artifacts.update(
            (
                snapshot["reviewed_text_artifact"],
                snapshot["acquisition_binding_artifact"],
            )
        )
        if profile == "production" and snapshot["source_family"] not in {
            "public_domain",
            "first_party",
        }:
            raise ValueError(
                "production dossiers must be human-authored public-domain or first-party sources"
            )
        twin_ref = case["twin_ref"]
        alignment = case["alignment_receipt_artifact"]
        if twin_ref is None:
            if case["language"] == "he" or alignment is not None:
                raise ValueError("Hebrew/alignment cases require a twin")
        else:
            _text(twin_ref, f"{label}.twin_ref")
            if (
                alignment not in sources
                or sources[alignment]["role"] != "translation_alignment_receipt"
            ):
                raise ValueError("twin cases require a translation/alignment receipt")
            used_artifacts.add(alignment)
        chunks = case["chunks"]
        questions = case["questions"]
        chunking = case["chunking"]
        if not isinstance(chunking, dict):
            raise TypeError(f"{label}.chunking must be an object")
        _exact_keys(chunking, CHUNKING_KEYS, f"{label}.chunking")
        if chunking["mode"] == "semantic_sections":
            boundary_receipt = chunking["boundary_receipt_artifact"]
            if (
                boundary_receipt not in sources
                or sources[boundary_receipt]["role"] != "chunk_boundary_receipt"
            ):
                raise ValueError(
                    "semantic-section chunking requires a boundary receipt artifact"
                )
            used_artifacts.add(boundary_receipt)
        elif chunking["mode"] == "fixed_512_overlap_64":
            if chunking["boundary_receipt_artifact"] is not None:
                raise ValueError(
                    "fixed chunking cannot claim a semantic-boundary receipt"
                )
            raise ValueError(
                "fixed 512/64 chunking fails closed because the external tokenizer "
                "boundary cannot be deterministically recomputed"
            )
        else:
            raise ValueError(
                "chunking mode must freeze semantic sections or 512/64 fallback"
            )
        if (
            not isinstance(chunks, list)
            or not chunks
            or not isinstance(questions, list)
            or not questions
        ):
            raise ValueError(f"{label} needs non-empty chunks and questions")
        chunk_refs: set[str] = set()
        previous_start = -1
        for chunk_index, chunk in enumerate(chunks):
            chunk_label = f"{label}.chunks[{chunk_index}]"
            if not isinstance(chunk, dict):
                raise TypeError(f"{chunk_label} must be an object")
            _exact_keys(chunk, CHUNK_SPEC_KEYS, chunk_label)
            ref = _text(chunk["chunk_ref"], f"{chunk_label}.chunk_ref")
            if ref in chunk_refs:
                raise ValueError("chunk_ref values must be unique within a case")
            chunk_refs.add(ref)
            if chunk["kind"] not in {"global_policy", "evidence", "distractor"}:
                raise ValueError(f"{chunk_label}.kind is invalid")
            if chunk["modality"] not in {"text", "image"}:
                raise ValueError(f"{chunk_label}.modality is invalid")
            _text(chunk["text"], f"{chunk_label}.text")
            start, end = chunk["token_start"], chunk["token_end"]
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or start < previous_start
                or end > context_tokens
            ):
                raise ValueError(f"{chunk_label} has invalid ordered token offsets")
            previous_start = start
            if chunk["twin_anchor"] is not None:
                _text(chunk["twin_anchor"], f"{chunk_label}.twin_anchor")
            media_name = chunk["media_artifact"]
            control = chunk["image_control"]
            derivation = chunk["derivation"]
            if not isinstance(derivation, dict):
                raise TypeError(f"{chunk_label}.derivation must be an object")
            if chunk["modality"] == "text":
                if media_name is not None or control is not None:
                    raise ValueError("text chunks cannot carry media or image controls")
            else:
                if not isinstance(control, dict):
                    raise ValueError(
                        "image chunks require an explicit control declaration"
                    )
                _exact_keys(control, IMAGE_SPEC_KEYS, f"{chunk_label}.image_control")
                _text(control["group_ref"], "image control group_ref")
                if control["variant"] not in {"real", "shuffled", "missing"}:
                    raise ValueError("image control variant is invalid")
                mapping_name = control["mapping_receipt_artifact"]
                if (
                    mapping_name not in sources
                    or sources[mapping_name]["role"] != "image_control_receipt"
                ):
                    raise ValueError(
                        "image controls require a mapping receipt artifact"
                    )
                used_artifacts.add(mapping_name)
                if control["variant"] == "missing":
                    if media_name is not None:
                        raise ValueError("missing image controls cannot carry media")
                elif (
                    media_name not in sources or sources[media_name]["role"] != "image"
                ):
                    raise ValueError(
                        "real/shuffled controls require declared image artifacts"
                    )
                if media_name is not None:
                    used_artifacts.add(media_name)
            if chunk["kind"] == "global_policy":
                _exact_keys(derivation, {"kind"}, f"{chunk_label}.derivation")
                if derivation["kind"] != "non_source_global_policy":
                    raise ValueError(
                        "global policy requires explicit non-source derivation"
                    )
            elif chunk["modality"] == "image":
                _exact_keys(
                    derivation,
                    {"kind", "image_artifact", "control_receipt_artifact"},
                    f"{chunk_label}.derivation",
                )
                if (
                    derivation["kind"] != "image_control"
                    or derivation["image_artifact"] != media_name
                    or derivation["control_receipt_artifact"]
                    != control["mapping_receipt_artifact"]
                ):
                    raise ValueError(
                        "image derivation must exactly bind image/control artifacts"
                    )
            elif case["language"] == "en":
                _exact_keys(
                    derivation,
                    {"kind", "source_artifact", "char_start", "char_end"},
                    f"{chunk_label}.derivation",
                )
                if (
                    derivation["kind"] != "source_text_slice"
                    or derivation["source_artifact"]
                    != snapshot["reviewed_text_artifact"]
                    or type(derivation["char_start"]) is not int
                    or type(derivation["char_end"]) is not int
                    or derivation["char_start"] < 0
                    or derivation["char_end"] <= derivation["char_start"]
                ):
                    raise ValueError(
                        "English text derivation must bind a valid reviewed-source slice"
                    )
            else:
                _exact_keys(
                    derivation,
                    {
                        "kind",
                        "source_twin_case_ref",
                        "source_twin_chunk_ref",
                        "alignment_receipt_artifact",
                    },
                    f"{chunk_label}.derivation",
                )
                if (
                    derivation["kind"] != "translated_twin"
                    or derivation["alignment_receipt_artifact"] != alignment
                ):
                    raise ValueError(
                        "Hebrew text derivation must bind its twin/alignment receipt"
                    )
                _text(
                    derivation["source_twin_case_ref"],
                    f"{chunk_label}.derivation.source_twin_case_ref",
                )
                _text(
                    derivation["source_twin_chunk_ref"],
                    f"{chunk_label}.derivation.source_twin_chunk_ref",
                )
        if max(chunk["token_end"] for chunk in chunks) != context_tokens:
            raise ValueError("context_tokens must equal the final chunk token_end")
        if not any(chunk["kind"] == "global_policy" for chunk in chunks):
            raise ValueError(
                "each dossier must include immutable global policy context"
            )
        if any(chunk["token_end"] - chunk["token_start"] > 512 for chunk in chunks):
            raise ValueError(
                "all semantic or fallback chunks may contain at most 512 tokens"
            )
        spans = [(chunk["token_start"], chunk["token_end"]) for chunk in chunks]
        if chunking["mode"] == "semantic_sections":
            if spans[0][0] != 0 or any(
                start != spans[index - 1][1]
                for index, (start, _) in enumerate(spans[1:], start=1)
            ):
                raise ValueError(
                    "semantic-section chunks must cover the entire dossier contiguously"
                )
        else:
            for index, (start, end) in enumerate(spans):
                if end - start > 512:
                    raise ValueError(
                        "fixed fallback chunks may contain at most 512 tokens"
                    )
                if index and start != spans[index - 1][1] - 64:
                    raise ValueError(
                        "fixed fallback chunks must overlap by exactly 64 tokens"
                    )
        question_refs: set[str] = set()
        for question_index, question in enumerate(questions):
            question_label = f"{label}.questions[{question_index}]"
            if not isinstance(question, dict):
                raise TypeError(f"{question_label} must be an object")
            _exact_keys(question, QUESTION_SPEC_KEYS, question_label)
            ref = _text(question["question_ref"], f"{question_label}.question_ref")
            if ref in question_refs:
                raise ValueError("question_ref values must be unique within a case")
            question_refs.add(ref)
            for field in ("text", "policy_role"):
                _text(question[field], f"{question_label}.{field}")
            if question["decision_type"] not in {"choice", "noul", "score"}:
                raise ValueError(f"{question_label}.decision_type is invalid")
            if question["twin_anchor"] is not None:
                _text(question["twin_anchor"], f"{question_label}.twin_anchor")
            options = question["options"]
            if (
                not isinstance(options, list)
                or len(options) < 2
                or any(
                    not isinstance(value, str) or not value.strip() for value in options
                )
                or len(set(options)) != len(options)
            ):
                raise ValueError(f"{question_label}.options is invalid")
            option_ids = question["option_ids"]
            if (
                not isinstance(option_ids, list)
                or len(option_ids) != len(options)
                or len(set(option_ids)) != len(option_ids)
                or any(
                    not isinstance(value, str)
                    or policy_replay.IDENTIFIER_RE.fullmatch(value) is None
                    for value in option_ids
                )
            ):
                raise ValueError(
                    f"{question_label}.option_ids must be parallel unique identifiers"
                )
            decisive = question["decisive_chunk_refs"]
            distractors = question["known_distractor_chunk_refs"]
            if (
                not isinstance(decisive, list)
                or not decisive
                or len(decisive) != len(set(decisive))
            ):
                raise ValueError("decisive_chunk_refs must be a non-empty unique list")
            if (
                not isinstance(distractors, list)
                or not distractors
                or len(distractors) != len(set(distractors))
            ):
                raise ValueError(
                    "known_distractor_chunk_refs must be a non-empty unique list"
                )
            if set(decisive) & set(distractors) or not (
                set(decisive) | set(distractors)
            ).issubset(chunk_refs):
                raise ValueError("question chunk references are invalid")
    if {case["source_snapshot"] for case in cases} != set(snapshots):
        raise ValueError("authored cases must cover every declared source snapshot")
    cases_by_ref = {case["case_ref"]: case for case in cases}
    for case in cases:
        for chunk in case["chunks"]:
            derivation = chunk["derivation"]
            if derivation["kind"] != "translated_twin":
                continue
            source_case = cases_by_ref.get(derivation["source_twin_case_ref"])
            if (
                source_case is None
                or source_case["language"] != "en"
                or source_case["twin_ref"] != case["twin_ref"]
            ):
                raise ValueError("translated chunk source case is not its English twin")
            source_chunks = {item["chunk_ref"]: item for item in source_case["chunks"]}
            source_chunk = source_chunks.get(derivation["source_twin_chunk_ref"])
            if (
                source_chunk is None
                or source_chunk["twin_anchor"] != chunk["twin_anchor"]
                or source_chunk["kind"] != chunk["kind"]
                or source_chunk["modality"] != "text"
                or source_chunk["derivation"]["kind"] != "source_text_slice"
            ):
                raise ValueError("translated chunk source twin binding is invalid")
    if used_artifacts != set(sources):
        raise ValueError("authored cases must consume every declared source artifact")
    if profile == "production":
        _validate_production_contract(cases)
    elif not has_long:
        # This is informational only for explicitly opted-in tiny fixtures.
        pass
    scoring._reject_benchmark_lineage(spec)


def _validate_production_contract(cases: Sequence[Mapping[str, Any]]) -> None:
    """Enforce the preregistered minimum resolution of the R052 falsifier."""
    questions = sum(len(case["questions"]) for case in cases)
    twins = {case["twin_ref"] for case in cases if case["twin_ref"] is not None}
    long_english = [
        case
        for case in cases
        if case["language"] == "en" and case["context_tokens"] >= LONG_CONTEXT_TOKENS
    ]
    hebrew_questions = sum(
        len(case["questions"]) for case in cases if case["language"] == "he"
    )
    image_questions = 0
    typed_role_counts = {role: 0 for role in ("choice", "noul", "score")}
    position_counts = {"early": 0, "middle": 0, "late": 0}
    for case in cases:
        chunks = {chunk["chunk_ref"]: chunk for chunk in case["chunks"]}
        real_refs = {
            chunk["chunk_ref"]
            for chunk in case["chunks"]
            if chunk["image_control"] is not None
            and chunk["image_control"]["variant"] == "real"
        }
        for question in case["questions"]:
            typed_role_counts[question["decision_type"]] += 1
            if real_refs.intersection(question["decisive_chunk_refs"]):
                image_questions += 1
            if case["context_tokens"] >= LONG_CONTEXT_TOKENS:
                positions = {
                    ("early", "middle", "late")[
                        min(
                            2,
                            int(
                                (chunks[ref]["token_start"] + chunks[ref]["token_end"])
                                / 2
                                * 3
                                // case["context_tokens"]
                            ),
                        )
                    ]
                    for ref in question["decisive_chunk_refs"]
                }
                for position in positions:
                    position_counts[position] += 1
    failures = []
    observed = {
        "cases": len(cases),
        "questions": questions,
        "long_english_cases": len(long_english),
        "twin_groups": len(twins),
        "hebrew_questions": hebrew_questions,
        "image_questions": image_questions,
        **{f"{role}_questions": count for role, count in typed_role_counts.items()},
        "long_source_domains": len({case["source_domain"] for case in long_english}),
    }
    for key in (
        "cases",
        "questions",
        "long_english_cases",
        "twin_groups",
        "hebrew_questions",
        "image_questions",
        "choice_questions",
        "noul_questions",
        "score_questions",
        "long_source_domains",
    ):
        if observed[key] < PRODUCTION_MINIMUMS[key]:
            failures.append(f"{key}={observed[key]}<{PRODUCTION_MINIMUMS[key]}")
    for position, count in position_counts.items():
        if count < PRODUCTION_MINIMUMS["position_questions"]:
            failures.append(
                f"{position}_position_questions={count}<"
                f"{PRODUCTION_MINIMUMS['position_questions']}"
            )
    for case in long_english:
        retrievable = sum(chunk["kind"] != "global_policy" for chunk in case["chunks"])
        if retrievable < PRODUCTION_MINIMUMS["retrievable_chunks_per_long_case"]:
            failures.append(
                f"long_case[{case['case_ref']}].retrievable_chunks={retrievable}<"
                f"{PRODUCTION_MINIMUMS['retrievable_chunks_per_long_case']}"
            )
    if failures:
        raise ValueError(
            "production panel minimum contract failed: " + "; ".join(failures)
        )


def _load_media(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError("image artifacts must be individual regular files")
    raw = path.read_bytes()
    suffix = path.suffix.casefold()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix)
    if mime is None:
        raise ValueError("image artifacts must be PNG, JPEG, or WEBP")
    media = {
        "mime_type": mime,
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": _sha_bytes(raw),
    }
    scoring._validate_media(media, "authored media")
    return media


def _validate_annotations(value: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(value, dict):
        raise TypeError("causal annotations must be an object")
    _exact_keys(value, ANNOTATION_KEYS, "causal annotations")
    if (
        value["schema_version"] != ANNOTATION_VERSION
        or value["benchmark_derived"] is not False
    ):
        raise ValueError("causal annotation schema/provenance is invalid")
    if not isinstance(value["entries"], list) or not value["entries"]:
        raise ValueError("causal annotations require entries")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(value["entries"]):
        if not isinstance(entry, dict):
            raise TypeError("causal annotation entries must be objects")
        _exact_keys(entry, ANNOTATION_ENTRY_KEYS, f"causal annotations[{index}]")
        key = (
            _text(entry["case_ref"], "causal case_ref"),
            _text(entry["question_ref"], "causal question_ref"),
        )
        if key in result:
            raise ValueError("causal annotation keys must be unique")
        if (
            not isinstance(entry["removed_chunk_refs"], list)
            or not entry["removed_chunk_refs"]
        ):
            raise ValueError("causal removed_chunk_refs must be non-empty")
        full = entry["full_outcome"]
        removed = entry["removed_outcome"]
        if (
            not isinstance(full, dict)
            or set(full) != {"status", "option_index"}
            or full["status"] != "resolved"
            or type(full["option_index"]) is not int
            or full["option_index"] < 0
            or removed != {"status": "unresolved"}
        ):
            raise ValueError(
                "causal outcomes must be resolved option_index versus unresolved"
            )
        if entry["changed_fields"] != ["status", "option_index"]:
            raise ValueError("causal changed_fields must be status then option_index")
        # Reject non-finite/non-JSON values now rather than when emitting hashes.
        _canonical(entry["full_outcome"])
        _canonical(entry["removed_outcome"])
        result[key] = entry
    scoring._reject_benchmark_lineage(value)
    return result


def _build_cases(
    spec: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    annotations: Mapping[tuple[str, str], Mapping[str, Any]],
    source_artifact_paths: Mapping[str, Path],
    tokenizer_artifact_sha256: str,
    chunker_artifact_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cases_by_index: dict[int, dict[str, Any]] = {}
    case_ref_to_key: dict[str, str] = {}
    built_chunks: dict[tuple[str, str], dict[str, Any]] = {}
    used_annotations: set[tuple[str, str]] = set()
    sources = source_manifest["artifacts"]
    snapshots = source_manifest["snapshots"]
    authored_cases = list(enumerate(spec["cases"]))
    for case_index, authored in sorted(
        authored_cases, key=lambda item: item[1]["language"] == "he"
    ):
        source_declaration = snapshots[authored["source_snapshot"]]
        reviewed_name = source_declaration["reviewed_text_artifact"]
        reviewed_declaration = sources[reviewed_name]
        reviewed_text = source_artifact_paths[reviewed_name].read_text(encoding="utf-8")
        twin_key = (
            None
            if authored["twin_ref"] is None
            else _sha_json(
                {
                    "twin_ref": authored["twin_ref"],
                    "source_family": source_declaration["source_family"],
                    "source_domain": authored["source_domain"],
                }
            )
        )
        chunks: list[dict[str, Any]] = []
        chunk_ref_to_key: dict[str, str] = {}
        source_slices: list[tuple[int, int]] = []
        for chunk_index, item in enumerate(authored["chunks"]):
            media = (
                None
                if item["media_artifact"] is None
                else _load_media(source_artifact_paths[item["media_artifact"]])
            )
            control = item["image_control"]
            output_control = None
            if control is not None:
                output_control = {
                    "group_key": _sha_json(
                        {
                            "case_ref": authored["case_ref"],
                            "group_ref": control["group_ref"],
                        }
                    ),
                    "variant": control["variant"],
                }
            authored_derivation = item["derivation"]
            derivation_kind = authored_derivation["kind"]
            if derivation_kind == "source_text_slice":
                start = authored_derivation["char_start"]
                end = authored_derivation["char_end"]
                if end > len(reviewed_text):
                    raise ValueError("reviewed source slice is out of range")
                exact_text = reviewed_text[start:end]
                if item["text"] != exact_text:
                    raise ValueError(
                        "authored chunk text does not exactly match reviewed source slice"
                    )
                source_slices.append((start, end))
                output_derivation = {
                    "kind": "source_text_slice",
                    "source_artifact_sha256": reviewed_declaration["sha256"],
                    "char_start": start,
                    "char_end": end,
                    "slice_sha256": _sha_bytes(exact_text.encode("utf-8")),
                }
                output_text = exact_text
            elif derivation_kind == "non_source_global_policy":
                output_derivation = {"kind": "non_source_global_policy"}
                output_text = item["text"]
            elif derivation_kind == "translated_twin":
                source_identity = (
                    authored_derivation["source_twin_case_ref"],
                    authored_derivation["source_twin_chunk_ref"],
                )
                source_twin = built_chunks.get(source_identity)
                if source_twin is None:
                    raise ValueError(
                        "translated chunk source twin must be a built English chunk"
                    )
                alignment_name = authored_derivation["alignment_receipt_artifact"]
                output_derivation = {
                    "kind": "translated_twin",
                    "source_twin_chunk_key": source_twin["chunk_key"],
                    "source_twin_content_sha256": source_twin["content_sha256"],
                    "alignment_receipt_artifact_sha256": sources[alignment_name][
                        "sha256"
                    ],
                }
                output_text = item["text"]
            elif derivation_kind == "image_control":
                image_name = authored_derivation["image_artifact"]
                control_name = authored_derivation["control_receipt_artifact"]
                output_derivation = {
                    "kind": "image_control",
                    "image_artifact_sha256": None
                    if image_name is None
                    else sources[image_name]["sha256"],
                    "control_receipt_artifact_sha256": sources[control_name]["sha256"],
                }
                output_text = item["text"]
            else:  # Defensive: _validate_spec rejects this before building.
                raise ValueError("unknown chunk derivation kind")
            partial = {
                "kind": item["kind"],
                "modality": item["modality"],
                "text": output_text,
                "media": media,
                "immutable": item["kind"] == "global_policy",
                "twin_anchor": item["twin_anchor"],
                "image_control": output_control,
                "token_start": item["token_start"],
                "token_end": item["token_end"],
                "derivation": output_derivation,
            }
            content_hash = scoring.sha256_json(scoring.chunk_hash_payload(partial))
            chunk_key = _sha_json(
                {
                    "case_ref": authored["case_ref"],
                    "ordinal": chunk_index,
                    "content_sha256": content_hash,
                }
            )
            chunk_ref_to_key[item["chunk_ref"]] = chunk_key
            chunks.append(
                {"chunk_key": chunk_key, **partial, "content_sha256": content_hash}
            )
            built_chunks[(authored["case_ref"], item["chunk_ref"])] = chunks[-1]
        if authored["language"] == "en" and (
            not source_slices
            or source_slices[0][0] != 0
            or source_slices[-1][1] != len(reviewed_text)
            or any(
                previous[1] != current[0]
                for previous, current in pairwise(source_slices)
            )
        ):
            raise ValueError(
                "English reviewed source slices must cover the artifact without gaps"
            )
        questions: list[dict[str, Any]] = []
        for question_index, item in enumerate(authored["questions"]):
            annotation_key = (authored["case_ref"], item["question_ref"])
            if annotation_key not in annotations:
                raise ValueError(f"missing causal annotation for {annotation_key}")
            annotation = annotations[annotation_key]
            used_annotations.add(annotation_key)
            if annotation["removed_chunk_refs"] != item["decisive_chunk_refs"]:
                raise ValueError(
                    "causal removed chunks must exactly match decisive chunks in source order"
                )
            if annotation["full_outcome"]["option_index"] >= len(item["option_ids"]):
                raise ValueError("causal resolved option_index is out of range")
            question_base = {
                "text": item["text"],
                "options": item["options"],
                "option_ids": item["option_ids"],
                "decision_type": item["decision_type"],
                "policy_role": item["policy_role"],
            }
            content_hash = scoring.sha256_json(question_base)
            question_key = _sha_json(
                {
                    "case_ref": authored["case_ref"],
                    "ordinal": question_index,
                    "content_sha256": content_hash,
                }
            )
            questions.append(
                {
                    "question_key": question_key,
                    **question_base,
                    "twin_anchor": item["twin_anchor"],
                    "decisive_chunk_keys": [
                        chunk_ref_to_key[value] for value in item["decisive_chunk_refs"]
                    ],
                    "known_distractor_chunk_keys": [
                        chunk_ref_to_key[value]
                        for value in item["known_distractor_chunk_refs"]
                    ],
                    "causal_proof": {
                        "kind": "causal_annotation",
                        "removed_chunk_keys": [
                            chunk_ref_to_key[value]
                            for value in annotation["removed_chunk_refs"]
                        ],
                        "full_context_sha256": "0" * 64,
                        "removed_context_sha256": "0" * 64,
                        "full_outcome_sha256": _sha_json(annotation["full_outcome"]),
                        "removed_outcome_sha256": _sha_json(
                            annotation["removed_outcome"]
                        ),
                        "full_outcome": annotation["full_outcome"],
                        "removed_outcome": annotation["removed_outcome"],
                        "changed_fields": annotation["changed_fields"],
                    },
                    "content_sha256": content_hash,
                }
            )
        boundary_name = authored["chunking"]["boundary_receipt_artifact"]
        chunking_body = {
            "schema_version": scoring.CHUNKING_RECEIPT_VERSION,
            "mode": authored["chunking"]["mode"],
            "max_chunk_tokens": 512,
            "fallback_overlap_tokens": 64,
            "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
            "chunker_artifact_sha256": chunker_artifact_sha256,
            "boundary_receipt_artifact_sha256": None
            if boundary_name is None
            else sources[boundary_name]["sha256"],
            "semantic_verification": "external_tokenizer_and_chunker_not_recomputed",
            "ordered_spans": [
                {
                    "chunk_key": chunk["chunk_key"],
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],
                }
                for chunk in chunks
            ],
            "ordered_derivations": [
                {
                    "chunk_key": chunk["chunk_key"],
                    "derivation": chunk["derivation"],
                }
                for chunk in chunks
            ],
        }
        case = {
            "case_key": "0" * 64,
            "language": authored["language"],
            "twin_group_key": twin_key,
            "source_family": source_declaration["source_family"],
            "source_snapshot_sha256": reviewed_declaration["sha256"],
            "source_domain": authored["source_domain"],
            "context_tokens": authored["context_tokens"],
            "chunking_receipt": {
                **chunking_body,
                "receipt_sha256": scoring.sha256_json(chunking_body),
            },
            "contamination_audit_sha256": "0" * 64,
            "chunks": chunks,
            "questions": questions,
        }
        for question in questions:
            proof = question["causal_proof"]
            proof["full_context_sha256"] = scoring.causal_context_sha256(
                case, question, set()
            )
            proof["removed_context_sha256"] = scoring.causal_context_sha256(
                case, question, set(proof["removed_chunk_keys"])
            )
        content_hash = scoring.case_content_sha256(case)
        case["case_key"] = _sha_json(
            {
                "case_content_sha256": content_hash,
                "twin_group_key": twin_key,
                "source_ordinal": case_index,
            }
        )
        case_ref_to_key[authored["case_ref"]] = case["case_key"]
        cases_by_index[case_index] = case
    if used_annotations != set(annotations):
        raise ValueError("causal annotations must cover authored questions exactly")
    return [
        cases_by_index[index] for index in range(len(spec["cases"]))
    ], case_ref_to_key


def build_audit_input(
    spec: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    annotations: Mapping[tuple[str, str], Mapping[str, Any]],
    source_artifact_paths: Mapping[str, Path],
    tokenizer_artifact_sha256: str,
    chunker_artifact_sha256: str,
    *,
    spec_sha256: str,
    source_manifest_sha256: str,
    annotations_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    cases, case_ref_to_key = _build_cases(
        spec,
        source_manifest,
        annotations,
        source_artifact_paths,
        tokenizer_artifact_sha256,
        chunker_artifact_sha256,
    )
    audit_cases = []
    for authored, case in zip(spec["cases"], cases, strict=True):
        audit_cases.append(
            {
                "case_ref": authored["case_ref"],
                "case_key": case["case_key"],
                "case_content_sha256": scoring.case_content_sha256(case),
                "language": case["language"],
                "source_family": case["source_family"],
                "source_domain": case["source_domain"],
                "ordered_chunks": [
                    {
                        "chunk_key": chunk["chunk_key"],
                        "content_sha256": chunk["content_sha256"],
                        "text": chunk["text"],
                        "media_sha256": None
                        if chunk["media"] is None
                        else chunk["media"]["sha256"],
                    }
                    for chunk in case["chunks"]
                ],
                "ordered_questions": [
                    {
                        "question_key": question["question_key"],
                        "content_sha256": question["content_sha256"],
                        "text": question["text"],
                        "options": question["options"],
                        "option_ids": question["option_ids"],
                        "policy_role": question["policy_role"],
                    }
                    for question in case["questions"]
                ],
            }
        )
    value = {
        "schema_version": AUDIT_INPUT_VERSION,
        "panel_profile": spec["panel_profile"],
        "benchmark_derived": False,
        "dossier_spec_sha256": spec_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "causal_annotations_sha256": annotations_sha256,
        "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
        "chunker_artifact_sha256": chunker_artifact_sha256,
        # This exact projection is frozen before audit/provider receipts exist. Policy
        # authors bind it in their ledger/rubric, avoiding a receipt/panel hash cycle.
        "policy_replay_draft": scoring.policy_replay_draft({"cases": cases}),
        "cases": audit_cases,
    }
    value["policy_replay_draft_sha256"] = _sha_json(value["policy_replay_draft"])
    return value, cases, case_ref_to_key


def _validate_audit_results(
    results: Any,
    audit_input_sha256: str,
    audit_input: Mapping[str, Any],
    *,
    auditor_artifact: Path,
    corpus_artifacts: Sequence[Path],
    tokenizer_artifact: Path,
    chunker_artifact: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(results, dict):
        raise TypeError("audit results must be an object")
    _exact_keys(results, AUDIT_RESULTS_KEYS, "audit results")
    corpus_hashes = [
        scoring.artifact_sha256(_check_no_symlinks(path, "audit corpus"))
        for path in corpus_artifacts
    ]
    expected = {
        "schema_version": AUDIT_RESULTS_VERSION,
        "audit_input_sha256": audit_input_sha256,
        "sealed_benchmark_text_included": False,
        "algorithm": AUDIT_ALGORITHM,
        "auditor_code_sha256": scoring.artifact_sha256(
            _check_no_symlinks(auditor_artifact, "auditor artifact")
        ),
        "audited_corpus_roots_sha256": corpus_hashes,
        "tokenizer_receipt_sha256": scoring.artifact_sha256(
            _check_no_symlinks(tokenizer_artifact, "tokenizer artifact")
        ),
        "chunker_receipt_sha256": scoring.artifact_sha256(
            _check_no_symlinks(chunker_artifact, "chunker artifact")
        ),
    }
    for key, value in expected.items():
        if results[key] != value:
            raise ValueError(f"audit results {key} does not match supplied artifacts")
    if not corpus_hashes or len(corpus_hashes) != len(set(corpus_hashes)):
        raise ValueError("audit corpus artifacts must be non-empty and unique")
    input_cases = {case["case_ref"]: case for case in audit_input["cases"]}
    entries = results["entries"]
    if not isinstance(entries, list) or len(entries) != len(input_cases):
        raise ValueError("audit results must cover every case exactly")
    mapped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("audit result entries must be objects")
        _exact_keys(entry, AUDIT_RESULT_ENTRY_KEYS, "audit result entry")
        case_ref = entry["case_ref"]
        if case_ref in mapped or case_ref not in input_cases:
            raise ValueError("audit result case references must be exact and unique")
        if entry["case_content_sha256"] != input_cases[case_ref]["case_content_sha256"]:
            raise ValueError("audit result does not bind exact case content")
        if (
            entry["status"] != "pass"
            or type(entry["exact_overlap_count"]) is not int
            or entry["exact_overlap_count"] != 0
            or not isinstance(entry["near_overlap_max_jaccard"], (int, float))
            or isinstance(entry["near_overlap_max_jaccard"], bool)
            or not 0 <= entry["near_overlap_max_jaccard"] < 0.60
        ):
            raise ValueError("audit result did not pass frozen overlap gates")
        mapped[case_ref] = entry
    return mapped


def _build_providers(
    bundles: Mapping[str, Mapping[str, Any]],
    generator_artifacts: Mapping[str, Path],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(bundles) != set(generator_artifacts):
        raise ValueError(
            "provider bundles and generator artifacts must cover one another"
        )
    query_hashes = {
        scoring.retrieval_query_sha256(question)
        for case in cases
        for question in case["questions"]
    }
    document_hashes = {
        chunk["content_sha256"] for case in cases for chunk in case["chunks"]
    }
    providers: dict[str, dict[str, Any]] = {}
    for name in sorted(bundles):
        value = bundles[name]
        if not isinstance(value, dict):
            raise TypeError(f"provider bundle {name} must be an object")
        _exact_keys(value, PROVIDER_BUNDLE_KEYS, f"provider bundle {name}")
        if (
            value["schema_version"] != PROVIDER_BUNDLE_VERSION
            or value["origin"]
            not in {"local_open_model", "local_typical", "local_baseline"}
            or value["hosted_api"] is not False
            or value["benchmark_derived"] is not False
        ):
            raise ValueError(f"provider bundle {name} provenance is invalid")
        generator_hash = scoring.artifact_sha256(
            _check_no_symlinks(generator_artifacts[name], f"provider {name} generator")
        )
        if value["generator_code_sha256"] != generator_hash:
            raise ValueError(f"provider bundle {name} generator hash mismatch")
        if (
            set(value["queries"]) != query_hashes
            or set(value["documents"]) != document_hashes
        ):
            raise ValueError(
                f"provider bundle {name} vectors do not exactly cover panel content"
            )
        provider = {
            "kind": "precomputed",
            "origin": value["origin"],
            "hosted_api": False,
            "benchmark_derived": False,
            "model": _text(value["model"], f"provider {name} model"),
            "revision": _text(value["revision"], f"provider {name} revision"),
            "generator_code_sha256": generator_hash,
            "artifact_sha256": scoring.sha256_json(
                {"queries": value["queries"], "documents": value["documents"]}
            ),
            "dimension": value["dimension"],
            "queries": value["queries"],
            "documents": value["documents"],
        }
        providers[name] = provider
    return providers


def _write_exclusive_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_audit_input(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical(value)
    _write_exclusive_file(path, payload)
    return _sha_bytes(payload)


def finalize_bundle(
    *,
    output_dir: Path,
    spec: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    annotations: Mapping[tuple[str, str], Mapping[str, Any]],
    audit_input: Mapping[str, Any],
    audit_input_sha256: str,
    audit_results: Mapping[str, Any],
    provider_bundles: Mapping[str, Mapping[str, Any]],
    provider_bundle_sha256: Mapping[str, str],
    provider_generator_artifacts: Mapping[str, Path],
    source_artifact_paths: Mapping[str, Path],
    auditor_artifact: Path,
    corpus_artifacts: Sequence[Path],
    tokenizer_artifact: Path,
    chunker_artifact: Path,
    bindings: Mapping[str, str],
    policy_fact_ledger: Mapping[str, Any],
    policy_fact_ledger_sha256: str,
    policy_rubric: Mapping[str, Any],
    policy_rubric_sha256: str,
    policy_replay_schema_artifact: Path,
    policy_replay_schema_sha256: str,
    policy_evaluator_artifact: Path,
    policy_evaluator_sha256: str,
    policy_adjudication_artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    tokenizer_hash = scoring.artifact_sha256(
        _check_no_symlinks(tokenizer_artifact, "tokenizer artifact")
    )
    chunker_hash = scoring.artifact_sha256(
        _check_no_symlinks(chunker_artifact, "chunker artifact")
    )
    regenerated, cases, case_ref_to_key = build_audit_input(
        spec,
        source_manifest,
        annotations,
        source_artifact_paths,
        tokenizer_hash,
        chunker_hash,
        spec_sha256=bindings["dossier_spec_sha256"],
        source_manifest_sha256=bindings["source_manifest_sha256"],
        annotations_sha256=bindings["causal_annotations_sha256"],
    )
    if audit_input != regenerated or _sha_json(audit_input) != audit_input_sha256:
        raise ValueError("audit input is not the exact deterministic builder output")
    if _sha_json(policy_fact_ledger) != policy_fact_ledger_sha256:
        raise ValueError("fact ledger does not match its external SHA-256 binding")
    if _sha_json(policy_rubric) != policy_rubric_sha256:
        raise ValueError("policy rubric does not match its external SHA-256 binding")
    scoring.verify_artifact(
        policy_replay_schema_artifact,
        policy_replay_schema_sha256,
        "policy replay schema",
    )
    if policy_replay_schema_sha256 != policy_replay.EXPECTED_SCHEMA_SHA256:
        raise ValueError("policy replay schema is not the frozen evaluator schema")
    scoring.verify_artifact(
        policy_evaluator_artifact,
        policy_evaluator_sha256,
        "policy replay evaluator",
    )
    adjudication_bytes: dict[str, bytes] = {}
    for digest, path in policy_adjudication_artifacts.items():
        _sha(digest, "adjudication receipt hash")
        scoring.verify_artifact(path, digest, "adjudication receipt")
        adjudication_bytes[digest] = path.read_bytes()
    replay_draft = audit_input["policy_replay_draft"]
    if replay_draft != scoring.policy_replay_draft({"cases": cases}) or audit_input[
        "policy_replay_draft_sha256"
    ] != _sha_json(replay_draft):
        raise ValueError("audit input policy replay draft/hash is not deterministic")
    try:
        policy_replay_receipt = policy_replay.replay(
            replay_draft,
            policy_fact_ledger,
            policy_rubric,
            panel_sha256=audit_input["policy_replay_draft_sha256"],
            ledger_sha256=policy_fact_ledger_sha256,
            rubric_sha256=policy_rubric_sha256,
            evaluator_sha256=policy_evaluator_sha256,
            schema_sha256=policy_replay_schema_sha256,
            adjudication_receipts=adjudication_bytes,
        )
    except policy_replay.PolicyError as exc:
        raise ValueError(f"typed policy replay failed: {exc}") from exc
    policy_replay_receipt_sha256 = _sha_json(policy_replay_receipt)
    if _sha_json(audit_results) != bindings["audit_results_sha256"]:
        raise ValueError("audit results do not match their external SHA-256 binding")
    if set(provider_bundles) != set(provider_bundle_sha256):
        raise ValueError("provider bundles and external hashes must cover one another")
    for name, bundle in provider_bundles.items():
        if _sha_json(bundle) != provider_bundle_sha256[name]:
            raise ValueError(f"provider bundle {name} external hash mismatch")
    audit_by_ref = _validate_audit_results(
        audit_results,
        audit_input_sha256,
        audit_input,
        auditor_artifact=auditor_artifact,
        corpus_artifacts=corpus_artifacts,
        tokenizer_artifact=tokenizer_artifact,
        chunker_artifact=chunker_artifact,
    )
    corpus_hashes = [scoring.artifact_sha256(path) for path in corpus_artifacts]
    auditor_hash = scoring.artifact_sha256(auditor_artifact)
    for authored, case in zip(spec["cases"], cases, strict=True):
        result = audit_by_ref[authored["case_ref"]]
        body = {
            "schema_version": scoring.AUDIT_RECEIPT_VERSION,
            "status": "pass",
            "algorithm": AUDIT_ALGORITHM,
            "auditor_code_sha256": auditor_hash,
            "audited_corpus_roots_sha256": corpus_hashes,
            "exact_overlap_count": result["exact_overlap_count"],
            "near_overlap_max_jaccard": result["near_overlap_max_jaccard"],
            "tokenizer_receipt_sha256": tokenizer_hash,
            "chunker_receipt_sha256": chunker_hash,
        }
        case["contamination_audit_sha256"] = scoring.sha256_json(body)
    providers = _build_providers(provider_bundles, provider_generator_artifacts, cases)
    panel = {
        "schema_version": scoring.SCHEMA_VERSION,
        "provenance": {"kind": "independent_internal", "benchmark_derived": False},
        "cases": cases,
        "providers": providers,
    }
    causal_manifest = {
        "schema_version": scoring.CAUSAL_MANIFEST_VERSION,
        "annotation_builder_code_sha256": scoring.file_sha256(Path(__file__)),
        "entries": {
            scoring.causal_entry_key(case, question): question["causal_proof"]
            for case in cases
            for question in case["questions"]
        },
    }
    contamination_entries = {}
    for authored, case in zip(spec["cases"], cases, strict=True):
        result = audit_by_ref[authored["case_ref"]]
        receipt_body = {
            "schema_version": scoring.AUDIT_RECEIPT_VERSION,
            "status": "pass",
            "algorithm": AUDIT_ALGORITHM,
            "auditor_code_sha256": auditor_hash,
            "audited_corpus_roots_sha256": corpus_hashes,
            "exact_overlap_count": result["exact_overlap_count"],
            "near_overlap_max_jaccard": result["near_overlap_max_jaccard"],
            "tokenizer_receipt_sha256": tokenizer_hash,
            "chunker_receipt_sha256": chunker_hash,
        }
        receipt = {**receipt_body, "receipt_sha256": scoring.sha256_json(receipt_body)}
        contamination_entries[case["case_key"]] = {
            "case_content_sha256": scoring.case_content_sha256(case),
            "audit_receipt": receipt,
        }
    contamination_manifest = {
        "schema_version": scoring.CONTAMINATION_MANIFEST_VERSION,
        "auditor_code_sha256": auditor_hash,
        "entries": contamination_entries,
        "providers": {
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
            for name, provider in providers.items()
        },
    }
    causal_sha = scoring.sha256_json(causal_manifest)
    contamination_sha = scoring.sha256_json(contamination_manifest)
    scoring.validate_panel(
        panel,
        source_manifest,
        bindings["source_manifest_sha256"],
        causal_manifest,
        causal_sha,
        contamination_manifest,
        contamination_sha,
        annotation_builder_artifact=Path(__file__),
        contamination_auditor_artifact=auditor_artifact,
        contamination_corpus_artifacts=list(corpus_artifacts),
        contamination_tokenizer_artifact=tokenizer_artifact,
        contamination_chunker_artifact=chunker_artifact,
        source_snapshot_artifacts=dict(source_artifact_paths),
        provider_generator_artifacts=dict(provider_generator_artifacts),
        policy_fact_ledger=policy_fact_ledger,
        policy_fact_ledger_sha256=policy_fact_ledger_sha256,
        policy_rubric=policy_rubric,
        policy_rubric_sha256=policy_rubric_sha256,
        policy_replay_receipt=policy_replay_receipt,
        policy_replay_receipt_sha256=policy_replay_receipt_sha256,
        policy_replay_schema_artifact=policy_replay_schema_artifact,
        policy_replay_schema_sha256=policy_replay_schema_sha256,
        policy_evaluator_artifact=policy_evaluator_artifact,
        policy_evaluator_sha256=policy_evaluator_sha256,
        policy_adjudication_artifacts=policy_adjudication_artifacts,
    )
    files = {
        "panel.json": panel,
        "source_manifest.json": source_manifest,
        "causal_manifest.json": causal_manifest,
        "contamination_manifest.json": contamination_manifest,
        "audit_input.json": audit_input,
        "policy_replay_receipt.json": policy_replay_receipt,
    }
    file_hashes = {name: _sha_bytes(_canonical(value)) for name, value in files.items()}
    receipt = {
        "schema_version": BUILD_RECEIPT_VERSION,
        "panel_profile": spec["panel_profile"],
        "benchmark_derived": False,
        "semantic_ground_truth": {
            "origin": "deterministic_typed_policy_replay",
            "annotations_role": "preregistered_draft_inputs_only",
            "builder_executed_domain_semantics": True,
            "builder_claims_semantic_validation": True,
        },
        "bindings": {
            **bindings,
            "audit_input_sha256": audit_input_sha256,
            "audit_results_sha256": bindings["audit_results_sha256"],
            "source_artifacts": {
                name: source_manifest["artifacts"][name]["sha256"]
                for name in sorted(source_artifact_paths)
            },
            "provider_bundles": dict(sorted(provider_bundle_sha256.items())),
            "provider_generators": {
                name: scoring.artifact_sha256(path)
                for name, path in sorted(provider_generator_artifacts.items())
            },
            "contamination_auditor": auditor_hash,
            "contamination_corpora": corpus_hashes,
            "contamination_tokenizer": tokenizer_hash,
            "contamination_chunker": chunker_hash,
            "policy_replay_draft_sha256": audit_input["policy_replay_draft_sha256"],
            "policy_fact_ledger_sha256": policy_fact_ledger_sha256,
            "policy_rubric_sha256": policy_rubric_sha256,
            "policy_replay_schema_sha256": policy_replay_schema_sha256,
            "policy_evaluator_sha256": policy_evaluator_sha256,
            "policy_adjudication_receipts": sorted(policy_adjudication_artifacts),
        },
        "outputs": {
            **file_hashes,
            "causal_manifest_external_sha256": causal_sha,
            "contamination_manifest_external_sha256": contamination_sha,
            "policy_replay_receipt_external_sha256": policy_replay_receipt_sha256,
            "policy_replay_receipt_sha256": policy_replay_receipt["receipt_sha256"],
        },
        "counts": {
            "cases": len(cases),
            "questions": sum(len(case["questions"]) for case in cases),
            "providers": len(providers),
            "hebrew_cases": sum(case["language"] == "he" for case in cases),
            "long_cases": sum(
                case["context_tokens"] >= LONG_CONTEXT_TOKENS for case in cases
            ),
        },
        "case_ref_to_key": case_ref_to_key,
    }
    receipt["receipt_sha256"] = scoring.sha256_json(receipt)
    files["build_receipt.json"] = receipt
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, value in files.items():
            _write_exclusive_file(stage / name, _canonical(value))
        os.replace(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return receipt


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw = value.partition("=")
    if not separator or not name or not raw:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw)


def _named_sha(value: str) -> tuple[str, str]:
    name, separator, digest = value.partition("=")
    if not separator or not name or SHA256_RE.fullmatch(digest) is None:
        raise argparse.ArgumentTypeError("expected NAME=SHA256")
    return name, digest


def _unique_named(items: Sequence[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in items:
        if name in result:
            raise ValueError(f"duplicate {label} name: {name}")
        result[name] = value
    return result


def _common_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Path],
    dict[str, str],
]:
    spec, spec_sha = _load_json(args.spec, args.expected_spec_sha256, "dossier spec")
    source_manifest, source_sha = _load_json(
        args.source_manifest, args.expected_source_manifest_sha256, "source manifest"
    )
    annotation_raw, annotation_sha = _load_json(
        args.causal_annotations,
        args.expected_causal_annotations_sha256,
        "causal annotations",
    )
    artifact_paths = _unique_named(args.source_artifact, "source artifact")
    resolved_paths = _validate_source_manifest(source_manifest, artifact_paths)
    _validate_spec(spec, source_manifest, args.allow_test_profile)
    annotations = _validate_annotations(annotation_raw)
    bindings = {
        "dossier_spec_sha256": spec_sha,
        "source_manifest_sha256": source_sha,
        "causal_annotations_sha256": annotation_sha,
    }
    return spec, source_manifest, annotations, resolved_paths, bindings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-audit", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--spec", type=Path, required=True)
        child.add_argument("--expected-spec-sha256", required=True)
        child.add_argument("--source-manifest", type=Path, required=True)
        child.add_argument("--expected-source-manifest-sha256", required=True)
        child.add_argument("--causal-annotations", type=Path, required=True)
        child.add_argument("--expected-causal-annotations-sha256", required=True)
        child.add_argument(
            "--source-artifact", type=_named_path, action="append", required=True
        )
        child.add_argument(
            "--contamination-tokenizer-artifact", type=Path, required=True
        )
        child.add_argument("--contamination-chunker-artifact", type=Path, required=True)
        child.add_argument(
            "--allow-test-profile", action="store_true", help=argparse.SUPPRESS
        )
    prepare = subparsers.choices["prepare-audit"]
    prepare.add_argument("--output", type=Path, required=True)
    final = subparsers.choices["finalize"]
    final.add_argument("--audit-input", type=Path, required=True)
    final.add_argument("--expected-audit-input-sha256", required=True)
    final.add_argument("--audit-results", type=Path, required=True)
    final.add_argument("--expected-audit-results-sha256", required=True)
    final.add_argument("--contamination-auditor-artifact", type=Path, required=True)
    final.add_argument(
        "--contamination-corpus-artifact", type=Path, action="append", required=True
    )
    final.add_argument(
        "--provider-bundle", type=_named_path, action="append", default=[]
    )
    final.add_argument(
        "--expected-provider-bundle-sha256",
        type=_named_sha,
        action="append",
        default=[],
    )
    final.add_argument(
        "--provider-generator-artifact", type=_named_path, action="append", default=[]
    )
    final.add_argument("--policy-fact-ledger", type=Path, required=True)
    final.add_argument("--expected-policy-fact-ledger-sha256", required=True)
    final.add_argument("--policy-rubric", type=Path, required=True)
    final.add_argument("--expected-policy-rubric-sha256", required=True)
    final.add_argument("--policy-replay-schema-artifact", type=Path, required=True)
    final.add_argument("--expected-policy-replay-schema-sha256", required=True)
    final.add_argument("--policy-evaluator-artifact", type=Path, required=True)
    final.add_argument("--expected-policy-evaluator-sha256", required=True)
    final.add_argument(
        "--policy-adjudication-artifact",
        type=_named_path,
        action="append",
        required=True,
        metavar="SHA256=PATH",
    )
    final.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    spec, source_manifest, annotations, source_paths, bindings = _common_inputs(args)
    tokenizer_hash = scoring.artifact_sha256(
        _check_no_symlinks(args.contamination_tokenizer_artifact, "tokenizer artifact")
    )
    chunker_hash = scoring.artifact_sha256(
        _check_no_symlinks(args.contamination_chunker_artifact, "chunker artifact")
    )
    audit_input, _, _ = build_audit_input(
        spec,
        source_manifest,
        annotations,
        source_paths,
        tokenizer_hash,
        chunker_hash,
        spec_sha256=bindings["dossier_spec_sha256"],
        source_manifest_sha256=bindings["source_manifest_sha256"],
        annotations_sha256=bindings["causal_annotations_sha256"],
    )
    if args.command == "prepare-audit":
        print(write_audit_input(args.output, audit_input))
        return
    stored_audit, audit_sha = _load_json(
        args.audit_input, args.expected_audit_input_sha256, "audit input"
    )
    audit_results, results_sha = _load_json(
        args.audit_results, args.expected_audit_results_sha256, "audit results"
    )
    raw_provider_paths = _unique_named(args.provider_bundle, "provider bundle")
    expected_provider_hashes = _unique_named(
        args.expected_provider_bundle_sha256, "provider bundle hash"
    )
    if set(raw_provider_paths) != set(expected_provider_hashes):
        raise ValueError("provider paths and expected hashes must cover one another")
    provider_bundles: dict[str, Mapping[str, Any]] = {}
    for name, path in raw_provider_paths.items():
        provider, _ = _load_json(
            path, expected_provider_hashes[name], f"provider bundle {name}"
        )
        provider_bundles[name] = provider
    generators = _unique_named(args.provider_generator_artifact, "provider generator")
    fact_ledger, fact_ledger_sha = _load_json(
        args.policy_fact_ledger,
        args.expected_policy_fact_ledger_sha256,
        "policy fact ledger",
    )
    rubric, rubric_sha = _load_json(
        args.policy_rubric,
        args.expected_policy_rubric_sha256,
        "policy rubric",
    )
    adjudications = _unique_named(
        args.policy_adjudication_artifact, "policy adjudication artifact"
    )
    bindings = {**bindings, "audit_results_sha256": results_sha}
    receipt = finalize_bundle(
        output_dir=args.output_dir,
        spec=spec,
        source_manifest=source_manifest,
        annotations=annotations,
        audit_input=stored_audit,
        audit_input_sha256=audit_sha,
        audit_results=audit_results,
        provider_bundles=provider_bundles,
        provider_bundle_sha256=expected_provider_hashes,
        provider_generator_artifacts=generators,
        source_artifact_paths=source_paths,
        auditor_artifact=args.contamination_auditor_artifact,
        corpus_artifacts=args.contamination_corpus_artifact,
        tokenizer_artifact=args.contamination_tokenizer_artifact,
        chunker_artifact=args.contamination_chunker_artifact,
        bindings=bindings,
        policy_fact_ledger=fact_ledger,
        policy_fact_ledger_sha256=fact_ledger_sha,
        policy_rubric=rubric,
        policy_rubric_sha256=rubric_sha,
        policy_replay_schema_artifact=args.policy_replay_schema_artifact,
        policy_replay_schema_sha256=args.expected_policy_replay_schema_sha256,
        policy_evaluator_artifact=args.policy_evaluator_artifact,
        policy_evaluator_sha256=args.expected_policy_evaluator_sha256,
        policy_adjudication_artifacts=adjudications,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
