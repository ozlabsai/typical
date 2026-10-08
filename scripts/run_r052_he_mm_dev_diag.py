#!/usr/bin/env python3
"""Run development-only Hebrew and image EmbeddingGemma 2 diagnostics.

The inputs are unreviewed R052 authoring drafts.  This tool therefore emits
directional development evidence, never B1 or benchmark evidence.  Gold labels,
source identifiers, filenames, and display positions are never sent to the
embedding provider.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import platform
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    from scripts import run_r052_embedding_dev_pilot as text_pilot
    from scripts import score_r052_evidence_retrieval as retrieval
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import run_r052_embedding_dev_pilot as text_pilot
    import score_r052_evidence_retrieval as retrieval

SCHEMA_VERSION = "r052-he-mm-development-diagnostic-receipt-v1"
RUNTIME_SCHEMA_VERSION = "r052-he-mm-development-diagnostic-runtime-v1"
CLAIM_STATUS = "development_only_unreviewed_not_b1_not_benchmark"
HE_V2_SCHEMA = "r052-he-en-twin-panel-development-draft-v2"
HE_V4_SCHEMA = "r052-he-en-twin-panel-development-draft-v4-overlay"
HE_EFFECTIVE_SCHEMA = "r052-he-en-twin-panel-development-draft-v4-effective"
MM_SCHEMA = "r052-mm-development-receipt-v2"
TRANSLATION_SCHEMA = "r052-hebrew-complete-candidate-translation-overlay-v1"
DIMENSIONS = (768, 256)
TOP_KS = (1, 2, 4)


@dataclass(frozen=True)
class EmbeddingInput:
    text: str
    media: Mapping[str, Any] | None


class EmbeddingProvider(Protocol):
    def embed(self, inputs: Sequence[EmbeddingInput]) -> Sequence[Sequence[float]]: ...

    def identity(self) -> Mapping[str, Any]: ...

    def runtime_identity(self) -> Mapping[str, Any]: ...


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    value = retrieval.parse_unique_finite_json(raw, label)
    if not isinstance(value, dict):
        raise TypeError(f"{label} root must be an object")
    return value, retrieval.sha256_bytes(raw)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-empty trimmed text")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def materialize_hebrew_v4(
    base: Mapping[str, Any], base_raw_sha256: str, overlay: Mapping[str, Any]
) -> dict[str, Any]:
    if base.get("schema_version") != HE_V2_SCHEMA:
        raise ValueError("Hebrew base is not the exact v2 schema")
    if overlay.get("schema_version") != HE_V4_SCHEMA:
        raise ValueError("Hebrew overlay is not the exact v4 schema")
    declared_base = overlay.get("base")
    if (
        not isinstance(declared_base, dict)
        or declared_base.get("sha256") != base_raw_sha256
    ):
        raise ValueError("Hebrew v4 overlay does not bind the supplied v2 bytes")
    contract = overlay.get("deterministic_materialization")
    if not isinstance(contract, dict) or not isinstance(
        contract.get("materialization_receipt"), dict
    ):
        raise TypeError("Hebrew v4 materialization receipt is missing")
    receipt = contract["materialization_receipt"]
    operations = overlay.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("Hebrew v4 operations must be non-empty")
    operations_sha = retrieval.sha256_bytes(_canonical_bytes(operations))
    if operations_sha != receipt.get("operations_canonical_sha256"):
        raise ValueError("Hebrew v4 operations hash mismatch")

    effective = deepcopy(base)
    groups = effective.get("groups")
    if not isinstance(groups, list):
        raise TypeError("Hebrew v2 groups are missing")
    for operation in operations:
        if operation.get("operation") != "replace_group09_removal_partition":
            raise ValueError("unknown Hebrew v4 operation")
        matches = [g for g in groups if g.get("group_id") == operation.get("group_id")]
        if len(matches) != 1:
            raise ValueError("Hebrew v4 operation must match exactly one group")
        group = matches[0]
        decisive = [
            ref
            for evidence in group.get("decisive_evidence", [])
            for ref in evidence.get("candidate_chunk_refs", [])
        ]
        excluded = [
            item["chunk_ref"]
            for item in group.get("excluded_answer_support_chunk_refs", [])
        ]
        pre = operation.get("preconditions", {})
        if decisive != pre.get("decisive_candidate_chunk_refs"):
            raise ValueError("Hebrew v4 decisive precondition mismatch")
        if excluded != pre.get("excluded_answer_support_chunk_refs"):
            raise ValueError("Hebrew v4 excluded-support precondition mismatch")
        if group.get("known_distractor_chunk_refs") != pre.get(
            "known_distractor_chunk_refs"
        ):
            raise ValueError("Hebrew v4 distractor precondition mismatch")
        replacement = operation.get("replacement", {})
        group["excluded_answer_support_chunk_refs"] = deepcopy(
            replacement.get("excluded_answer_support_chunk_refs")
        )
        group["known_distractor_chunk_refs"] = deepcopy(
            replacement.get("known_distractor_chunk_refs")
        )
        group["added_genuine_distractor_chunks"] = deepcopy(
            replacement.get("added_genuine_distractor_chunks")
        )
    effective["schema_version"] = HE_EFFECTIVE_SCHEMA
    effective["materialization_receipt_ref"] = {
        "overlay_schema_version": HE_V4_SCHEMA,
        "base_raw_sha256": base_raw_sha256,
        "operations_canonical_sha256": operations_sha,
    }
    encoded = _canonical_bytes(effective)
    if len(encoded) != receipt.get("effective_canonical_byte_length"):
        raise ValueError("Hebrew effective byte length mismatch")
    if retrieval.sha256_bytes(encoded) != receipt.get("effective_canonical_sha256"):
        raise ValueError("Hebrew effective content hash mismatch")
    return effective


def _source_id(case: text_pilot.Case, raw_cases: Sequence[Mapping[str, Any]]) -> str:
    raw = next(item for item in raw_cases if item.get("case_ref") == case.case_ref)
    source = raw.get("source", {})
    value = source.get("snapshot_id", source.get("id"))
    return _text(value, f"{case.case_ref}.source id")


def _he_query(group: Mapping[str, Any], language: str) -> dict[str, Any]:
    view = group.get("views", {}).get(language)
    if not isinstance(view, dict):
        raise TypeError(f"{group.get('group_id')}: missing {language} view")
    options = group.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError("Hebrew group options must be non-empty")
    pairs = sorted(
        (
            _text(option.get("option_id"), "option id"),
            _text(option.get(language), f"option {language}"),
        )
        for option in options
    )
    return {
        "text": _text(view.get("question"), f"question {language}"),
        "options": [pair[1] for pair in pairs],
        "option_ids": [pair[0] for pair in pairs],
        "decision_type": _text(group.get("primitive"), "primitive"),
        "policy_role": _text(group.get("policy_role"), "policy role"),
    }


def _candidate_maps(
    ntsb_path: Path, gao_path: Path
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    cases, bindings = text_pilot.load_candidates(ntsb_path, gao_path)
    raw_cases: list[Mapping[str, Any]] = []
    for path in (ntsb_path, gao_path):
        value, _ = _load_json(path, "candidate dossier")
        raw_cases.extend(value["cases"])
    by_source: dict[str, dict[str, str]] = {}
    for case in cases:
        source_id = _source_id(case, raw_cases)
        if source_id in by_source:
            raise ValueError(f"duplicate source id {source_id}")
        by_source[source_id] = {doc.chunk_ref: doc.text for doc in case.documents}
    return by_source, bindings


def _translation_overlay(
    path: Path | None, expected_refs: Mapping[str, Sequence[str]]
) -> tuple[dict[str, dict[str, str]] | None, dict[str, Any]]:
    if path is None:
        return None, {
            "available": False,
            "reason": "complete_translation_overlay_not_supplied",
        }
    value, raw_sha = _load_json(path, "Hebrew translation overlay")
    if value.get("schema_version") != TRANSLATION_SCHEMA:
        raise ValueError("unexpected Hebrew translation overlay schema")
    groups = value.get("groups")
    if not isinstance(groups, dict):
        raise TypeError("translation overlay groups must be an object")
    result: dict[str, dict[str, str]] = {}
    if set(groups) != set(expected_refs):
        raise ValueError("translation overlay must cover every Hebrew group exactly")
    for group_id, refs in expected_refs.items():
        rows = groups[group_id]
        if not isinstance(rows, list):
            raise TypeError(f"translation overlay rows missing for {group_id}")
        mapping: dict[str, str] = {}
        if {row.get("chunk_ref") for row in rows if isinstance(row, dict)} != set(refs):
            raise ValueError(f"translation overlay coverage mismatch for {group_id}")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "chunk_ref",
                "en_text_sha256",
                "he_text",
                "he_text_sha256",
            }:
                raise ValueError("translation overlay row schema mismatch")
            he_text = _text(row["he_text"], f"translation {group_id}")
            if retrieval.sha256_bytes(he_text.encode()) != row["he_text_sha256"]:
                raise ValueError("Hebrew translation text hash mismatch")
            mapping[row["chunk_ref"]] = he_text
        result[group_id] = mapping
    return result, {"available": True, "input_sha256": raw_sha}


def _media(
    path: Path, expected_sha: str, expected_bytes: int, root: Path
) -> dict[str, str]:
    root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("image artifact escapes the bound artifact root") from exc
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("image artifact path must not contain symlinks")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("image artifact escapes the bound artifact root")
    raw = resolved.read_bytes()
    if retrieval.sha256_bytes(raw) != expected_sha:
        raise ValueError("image artifact hash mismatch")
    if type(expected_bytes) is not int or len(raw) != expected_bytes:
        raise ValueError("image artifact byte count mismatch")
    media = {
        "mime_type": "image/png",
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": expected_sha,
    }
    retrieval._validate_media(media, "development image")
    return media


def _vectors(
    entries: Sequence[tuple[EmbeddingInput, str]], provider: EmbeddingProvider
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    unique: dict[str, EmbeddingInput] = {}
    roles: dict[str, set[str]] = {}
    for item, role in entries:
        media_sha = None if item.media is None else item.media["sha256"]
        key = retrieval.sha256_json({"text": item.text, "media_sha256": media_sha})
        prior = unique.get(key)
        if prior is not None and (prior.text != item.text or prior.media != item.media):
            raise ValueError("embedding input hash collision")
        unique[key] = item
        roles.setdefault(key, set()).add(role)
    keys = sorted(unique)
    ordered = [unique[key] for key in keys]
    raw = provider.embed(ordered)
    if len(raw) != len(ordered):
        raise ValueError("embedding provider returned the wrong vector count")
    vectors: dict[str, np.ndarray] = {}
    manifest = []
    for key, item, vector in zip(keys, ordered, raw, strict=True):
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (768,) or not np.isfinite(array).all():
            raise ValueError("embedding provider must return finite 768d vectors")
        vectors[key] = array
        manifest.append(
            {
                "input_sha256": key,
                "text_sha256": retrieval.sha256_bytes(item.text.encode()),
                "media_sha256": None if item.media is None else item.media["sha256"],
                "roles": sorted(roles[key]),
                "scoring_vector_sha256": {
                    str(dim): retrieval.sha256_json(
                        retrieval.normalize_mrl(array, dim).tolist()
                    )
                    for dim in DIMENSIONS
                },
            }
        )
    return vectors, manifest


def _input_key(item: EmbeddingInput) -> str:
    return retrieval.sha256_json(
        {
            "text": item.text,
            "media_sha256": None if item.media is None else item.media["sha256"],
        }
    )


def _rank(
    query: EmbeddingInput,
    documents: Sequence[tuple[str, EmbeddingInput]],
    vectors: Mapping[str, np.ndarray],
    dim: int,
) -> tuple[list[str], dict[str, float]]:
    q = retrieval.normalize_mrl(vectors[_input_key(query)], dim)
    scores = {
        key: float(
            np.dot(q, retrieval.normalize_mrl(vectors[_input_key(document)], dim))
        )
        for key, document in documents
    }
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("dense similarity must be finite")
    ranking = [
        key for key, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    return ranking, dict(sorted(scores.items()))


def _retrieval_metrics(
    rows: Sequence[Mapping[str, Any]], ranking_key: str
) -> dict[str, Any]:
    decisive_items = sum(len(row["decisive_refs"]) for row in rows)
    result: dict[str, Any] = {"questions": len(rows), "decisive_items": decisive_items}
    for k in TOP_KS:
        result[f"complete_recall_at_{k}"] = sum(
            set(row["decisive_refs"]).issubset(row[ranking_key][:k]) for row in rows
        ) / len(rows)
        result[f"item_recall_at_{k}"] = (
            sum(
                len(set(row["decisive_refs"]) & set(row[ranking_key][:k]))
                for row in rows
            )
            / decisive_items
        )
    result["mrr"] = math.fsum(
        1.0 / min(row[ranking_key].index(ref) + 1 for ref in row["decisive_refs"])
        for row in rows
    ) / len(rows)
    return result


def _production_bm25(
    question: Mapping[str, Any], documents: Sequence[tuple[str, str]]
) -> tuple[list[str], dict[str, float]]:
    ranked = retrieval.bm25_rank(
        dict(question), [{"chunk_key": key, "text": text} for key, text in documents]
    )
    return [key for key, _ in ranked], dict(sorted(ranked))


def _verify_declared_artifact(spec: Any, label: str) -> None:
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise TypeError(f"{label} artifact binding is invalid")
    path = Path(spec["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} artifact must be a regular non-symlink file")
    if retrieval.file_sha256(path) != spec.get("sha256"):
        raise ValueError(f"{label} artifact hash mismatch")


def build_receipt(
    *,
    he_base: Path,
    he_overlay: Path,
    ntsb_candidates: Path,
    gao_candidates: Path,
    mm_receipt: Path,
    mm_artifact_root: Path,
    provider: EmbeddingProvider,
    artifact_bindings: Mapping[str, str],
    he_translation_overlay: Path | None = None,
) -> dict[str, Any]:
    if artifact_bindings.get("runtime_lock_sha256") != text_pilot.RUNTIME_LOCK_SHA256:
        raise ValueError("diagnostic requires the dedicated R052 runtime lock")
    base, base_sha = _load_json(he_base, "Hebrew v2 base")
    overlay, overlay_sha = _load_json(he_overlay, "Hebrew v4 overlay")
    effective = materialize_hebrew_v4(base, base_sha, overlay)
    by_source, candidate_bindings = _candidate_maps(ntsb_candidates, gao_candidates)
    declared_inputs = effective.get("input_bindings", {})
    if candidate_bindings.get("ntsb_candidate_sha256") != declared_inputs.get(
        "ntsb_candidate_dossier_sha256"
    ) or candidate_bindings.get("gao_candidate_sha256") != declared_inputs.get(
        "gao_candidate_dossier_sha256"
    ):
        raise ValueError("Hebrew draft does not bind the supplied candidate dossiers")

    he_specs = []
    expected_translations: dict[str, list[str]] = {}
    expected_en_hashes: dict[str, dict[str, str]] = {}
    for group in effective["groups"]:
        group_id = _text(group.get("group_id"), "group id")
        source_id = _text(group.get("source_id"), f"{group_id}.source_id")
        corpus = dict(by_source.get(source_id, {}))
        for added in group.get("added_genuine_distractor_chunks", []):
            corpus[_text(added.get("chunk_ref"), "added chunk ref")] = _text(
                added.get("en_complete_proposition"), "added English proposition"
            )
        decisive = [
            ref
            for evidence in group["decisive_evidence"]
            for ref in evidence["candidate_chunk_refs"]
        ]
        distractors = list(group["known_distractor_chunk_refs"])
        refs = decisive + distractors
        if not decisive or len(refs) != len(set(refs)) or not set(refs) <= set(corpus):
            raise ValueError(f"{group_id}: invalid complete English corpus partition")
        expected_translations[group_id] = sorted(refs)
        expected_en_hashes[group_id] = {
            ref: retrieval.sha256_bytes(corpus[ref].encode()) for ref in refs
        }
        he_specs.append(
            (group, sorted(decisive), [(ref, corpus[ref]) for ref in sorted(refs)])
        )
    translations, he_he_status = _translation_overlay(
        he_translation_overlay, expected_translations
    )
    if translations is not None:
        overlay_value, _ = _load_json(
            he_translation_overlay, "Hebrew translation overlay"
        )  # type: ignore[arg-type]
        for group_id, rows in overlay_value["groups"].items():
            for row in rows:
                if (
                    row["en_text_sha256"]
                    != expected_en_hashes[group_id][row["chunk_ref"]]
                ):
                    raise ValueError("translation overlay English binding mismatch")

    mm, mm_sha = _load_json(mm_receipt, "multimodal v2 receipt")
    if mm.get("schema_version") != MM_SCHEMA:
        raise ValueError("unexpected multimodal development receipt schema")
    if (
        mm.get("status") != "ai_authored_unreviewed_development_only"
        or mm.get("eligible_for_sealed_or_public_eval") is not False
        or mm.get("pixel_causality_review_complete") is not False
        or mm.get("rights_review_complete") is not False
        or mm.get("semantic_review_complete") is not False
    ):
        raise ValueError("multimodal receipt must retain development-only false flags")
    _verify_declared_artifact(mm.get("generator"), "multimodal generator")
    _verify_declared_artifact(mm.get("acquisition_manifest"), "acquisition manifest")
    if mm["acquisition_manifest"]["sha256"] != declared_inputs.get(
        "acquisition_manifest_sha256"
    ):
        raise ValueError("Hebrew and multimodal acquisition bindings differ")
    neutral = _text(mm.get("neutral_context"), "multimodal neutral context")
    if retrieval.sha256_bytes(neutral.encode()) != mm.get("neutral_context_sha256"):
        raise ValueError("multimodal neutral context hash mismatch")
    if mm_artifact_root.is_symlink():
        raise ValueError("multimodal artifact root must not be a symlink")
    mm_root = mm_artifact_root.resolve(strict=True)
    distractor_specs = mm.get("distractors")
    if not isinstance(distractor_specs, dict):
        raise TypeError("multimodal distractor corpus must be an object")

    inputs: list[tuple[EmbeddingInput, str]] = []
    he_queries: dict[str, EmbeddingInput] = {}
    en_queries: dict[str, EmbeddingInput] = {}
    he_en_docs: dict[str, list[tuple[str, EmbeddingInput]]] = {}
    he_he_docs: dict[str, list[tuple[str, EmbeddingInput]]] = {}
    for group, _, docs in he_specs:
        group_id = group["group_id"]
        question = _he_query(group, "he")
        query_text = retrieval.retrieval_query_text(question)
        declared = group.get("queries", {}).get("he_to_en")
        expected_declared = (
            question["text"]
            + "\nאפשרויות:\n"
            + "\n".join(
                f"{option_id}\t{option}"
                for option_id, option in zip(
                    question["option_ids"], question["options"], strict=True
                )
            )
        )
        if declared != expected_declared:
            raise ValueError(f"{group_id}: declared Hebrew query is not canonical")
        query = EmbeddingInput(query_text, None)
        he_queries[group_id] = query
        inputs.append((query, f"he_query:{group_id}"))
        en_query = EmbeddingInput(
            retrieval.retrieval_query_text(_he_query(group, "en")), None
        )
        en_queries[group_id] = en_query
        inputs.append((en_query, f"en_query:{group_id}"))
        he_en_docs[group_id] = []
        for ref, text in docs:
            item = EmbeddingInput(f"title: evidence | text: {text}", None)
            he_en_docs[group_id].append((ref, item))
            inputs.append((item, f"he_en_document:{group_id}:{ref}"))
        if translations is not None:
            he_he_docs[group_id] = []
            for ref, text in sorted(translations[group_id].items()):
                item = EmbeddingInput(f"title: evidence | text: {text}", None)
                he_he_docs[group_id].append((ref, item))
                inputs.append((item, f"he_he_document:{group_id}:{ref}"))

    mm_queries: dict[str, EmbeddingInput] = {}
    mm_variant_docs: dict[str, dict[str, EmbeddingInput]] = {}
    mm_candidate_docs: dict[str, list[tuple[str, EmbeddingInput]]] = {}
    mm_candidate_text: dict[str, list[tuple[str, str]]] = {}
    mm_cases = mm.get("cases")
    if not isinstance(mm_cases, list) or len(mm_cases) != 10:
        raise ValueError("multimodal receipt must contain exactly ten cases")
    case_ids = [case.get("id") for case in mm_cases if isinstance(case, dict)]
    if len(case_ids) != 10 or len(set(case_ids)) != 10:
        raise ValueError("multimodal case ids must be ten unique values")
    mm_cases = sorted(mm_cases, key=lambda case: case["id"])
    for case in mm_cases:
        case_id = _text(case.get("id"), "multimodal case id")
        options = case.get("options")
        if not isinstance(options, list) or len(options) != len(set(options)):
            raise ValueError(f"{case_id}: invalid options")
        canonical = sorted(_text(option, f"{case_id} option") for option in options)
        display = case.get("option_permutation", {}).get("display_order")
        if not isinstance(display, list) or sorted(display) != canonical:
            raise ValueError(f"{case_id}: display permutation is not option-invariant")
        permutation = case["option_permutation"]
        if permutation.get("display_order_sha256") != retrieval.sha256_json(display):
            raise ValueError(f"{case_id}: display order hash mismatch")
        if permutation.get("semantic_to_display_index") != {
            option_id: display.index(option_id) for option_id in canonical
        }:
            raise ValueError(f"{case_id}: display index mapping mismatch")
        question = {
            "text": _text(case.get("question"), f"{case_id}.question"),
            "options": canonical,
            "option_ids": canonical,
            "decision_type": "Choice",
            "policy_role": "select_visual_evidence_or_abstain",
        }
        query = EmbeddingInput(retrieval.retrieval_query_text(question), None)
        mm_queries[case_id] = query
        inputs.append((query, f"mm_query:{case_id}"))
        variants: dict[str, EmbeddingInput] = {}
        for variant, field in (
            ("real", "real_image"),
            ("shuffled", "shuffled_image"),
            ("missing", "missing_image"),
        ):
            spec = case.get(field)
            if not isinstance(spec, dict):
                raise TypeError(f"{case_id}: missing {field}")
            relative = Path(_text(spec.get("path"), f"{case_id}.{field}.path"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "multimodal artifact paths must be safe relative paths"
                )
            verified_media = _media(
                mm_root / relative,
                _text(spec.get("sha256"), "image sha256"),
                spec.get("bytes"),
                mm_root,
            )
            # Missing means no media in production.  The canonical blank file is
            # verified above only as control provenance and is never embedded.
            media = None if variant == "missing" else verified_media
            item = EmbeddingInput(neutral, media)
            variants[variant] = item
            inputs.append((item, f"mm_document:{case_id}:{variant}"))
        mm_variant_docs[case_id] = variants
        candidates = [
            (f"{case_id}:real", variants["real"]),
            (f"{case_id}:shuffled", variants["shuffled"]),
            (f"{case_id}:missing", variants["missing"]),
        ]
        candidate_text = [(key, neutral) for key, _ in candidates]
        raw_distractor_ids = case.get("distractors")
        if not isinstance(raw_distractor_ids, list):
            raise TypeError(f"{case_id}: exactly five text distractors are required")
        distractor_ids = [
            _text(value, f"{case_id} distractor id") for value in raw_distractor_ids
        ]
        if len(distractor_ids) != 5 or len(set(distractor_ids)) != 5:
            raise ValueError(f"{case_id}: exactly five text distractors are required")
        for distractor_id in sorted(distractor_ids):
            spec = distractor_specs.get(distractor_id)
            if not isinstance(spec, dict):
                raise TypeError(f"{case_id}: unknown distractor {distractor_id}")
            text = _text(spec.get("content"), f"distractor {distractor_id}")
            if retrieval.sha256_bytes(text.encode()) != spec.get("content_sha256"):
                raise ValueError("multimodal distractor content hash mismatch")
            item = EmbeddingInput(f"title: evidence | text: {text}", None)
            key = f"{case_id}:text:{distractor_id}"
            candidates.append((key, item))
            candidate_text.append((key, text))
            inputs.append((item, f"mm_document:{case_id}:text:{distractor_id}"))
        mm_candidate_docs[case_id] = candidates
        mm_candidate_text[case_id] = candidate_text

    vectors, vector_manifest = _vectors(inputs, provider)
    he_rows = []
    for group, decisive, _ in he_specs:
        group_id = group["group_id"]
        row: dict[str, Any] = {
            "group_id": group_id,
            "decision_type": group["primitive"],
            "query_sha256": retrieval.sha256_bytes(he_queries[group_id].text.encode()),
            "decisive_refs": decisive,
            "candidate_refs": [ref for ref, _ in he_en_docs[group_id]],
            "he_to_en": {},
            "en_to_en": {},
        }
        for dim in DIMENSIONS:
            ranking, scores = _rank(
                he_queries[group_id], he_en_docs[group_id], vectors, dim
            )
            row["he_to_en"][str(dim)] = {"ranking": ranking, "scores": scores}
            ranking, scores = _rank(
                en_queries[group_id], he_en_docs[group_id], vectors, dim
            )
            row["en_to_en"][str(dim)] = {"ranking": ranking, "scores": scores}
        en_question = _he_query(group, "en")
        he_question = _he_query(group, "he")
        source_docs = [
            (ref, item.text.removeprefix("title: evidence | text: "))
            for ref, item in he_en_docs[group_id]
        ]
        for label, question in (("he_to_en", he_question), ("en_to_en", en_question)):
            ranking, scores = _production_bm25(question, source_docs)
            row[label]["bm25"] = {"ranking": ranking, "scores": scores}
        if translations is not None:
            row["he_to_he"] = {}
            for dim in DIMENSIONS:
                ranking, scores = _rank(
                    he_queries[group_id], he_he_docs[group_id], vectors, dim
                )
                row["he_to_he"][str(dim)] = {"ranking": ranking, "scores": scores}
        he_rows.append(row)

    mm_rows = []
    for case in mm_cases:
        case_id = case["id"]
        decisive_key = f"{case_id}:real"
        row = {"case_id": case_id, "decisive_refs": [decisive_key], "dimensions": {}}
        for dim in DIMENSIONS:
            ranking, scores = _rank(
                mm_queries[case_id], mm_candidate_docs[case_id], vectors, dim
            )
            query_vector = retrieval.normalize_mrl(
                vectors[_input_key(mm_queries[case_id])], dim
            )
            variant_scores = {
                variant: float(
                    np.dot(
                        query_vector,
                        retrieval.normalize_mrl(vectors[_input_key(item)], dim),
                    )
                )
                for variant, item in sorted(mm_variant_docs[case_id].items())
            }
            row["dimensions"][str(dim)] = {
                "candidate_ranking": ranking,
                "candidate_scores": scores,
                "own_variant_scores": variant_scores,
                "real_minus_shuffled": variant_scores["real"]
                - variant_scores["shuffled"],
                "real_minus_missing": variant_scores["real"]
                - variant_scores["missing"],
            }
        mm_question = {
            "text": case["question"],
            "options": sorted(case["options"]),
            "option_ids": sorted(case["options"]),
            "decision_type": "Choice",
            "policy_role": "select_visual_evidence_or_abstain",
        }
        bm25_ranking, bm25_scores = _production_bm25(
            mm_question, mm_candidate_text[case_id]
        )
        variant_bm25 = {
            key.rsplit(":", 1)[-1]: bm25_scores[key]
            for key in (f"{case_id}:real", f"{case_id}:shuffled", f"{case_id}:missing")
        }
        if len(set(variant_bm25.values())) != 1:
            raise ValueError(
                "matched multimodal control text must produce an exact BM25 tie"
            )
        row["bm25"] = {
            "ranking": bm25_ranking,
            "scores": bm25_scores,
            "matched_control_scores": variant_bm25,
            "matched_control_exact_tie": True,
        }
        mm_rows.append(row)

    he_summary = {
        "he_to_en": {
            str(dim): _retrieval_metrics(
                [
                    {**row, "ranking": row["he_to_en"][str(dim)]["ranking"]}
                    for row in he_rows
                ],
                "ranking",
            )
            for dim in DIMENSIONS
        },
        "en_to_en": {
            str(dim): _retrieval_metrics(
                [
                    {**row, "ranking": row["en_to_en"][str(dim)]["ranking"]}
                    for row in he_rows
                ],
                "ranking",
            )
            for dim in DIMENSIONS
        },
        "bm25_he_to_en": _retrieval_metrics(
            [{**row, "ranking": row["he_to_en"]["bm25"]["ranking"]} for row in he_rows],
            "ranking",
        ),
        "bm25_en_to_en": _retrieval_metrics(
            [{**row, "ranking": row["en_to_en"]["bm25"]["ranking"]} for row in he_rows],
            "ranking",
        ),
        "he_to_he": he_he_status,
    }
    if translations is not None:
        he_summary["he_to_he"]["metrics"] = {
            str(dim): _retrieval_metrics(
                [
                    {**row, "ranking": row["he_to_he"][str(dim)]["ranking"]}
                    for row in he_rows
                ],
                "ranking",
            )
            for dim in DIMENSIONS
        }
    mm_summary = {}
    for dim in DIMENSIONS:
        metrics = _retrieval_metrics(
            [
                {**row, "ranking": row["dimensions"][str(dim)]["candidate_ranking"]}
                for row in mm_rows
            ],
            "ranking",
        )
        metrics.update(
            {
                "real_beats_shuffled": sum(
                    row["dimensions"][str(dim)]["real_minus_shuffled"] > 0
                    for row in mm_rows
                )
                / len(mm_rows),
                "real_beats_missing": sum(
                    row["dimensions"][str(dim)]["real_minus_missing"] > 0
                    for row in mm_rows
                )
                / len(mm_rows),
                "mean_real_minus_shuffled": math.fsum(
                    row["dimensions"][str(dim)]["real_minus_shuffled"]
                    for row in mm_rows
                )
                / len(mm_rows),
                "mean_real_minus_missing": math.fsum(
                    row["dimensions"][str(dim)]["real_minus_missing"] for row in mm_rows
                )
                / len(mm_rows),
            }
        )
        mm_summary[str(dim)] = metrics
    mm_summary["bm25"] = _retrieval_metrics(
        [{**row, "ranking": row["bm25"]["ranking"]} for row in mm_rows], "ranking"
    )
    mm_summary["bm25"]["matched_control_exact_ties"] = sum(
        row["bm25"]["matched_control_exact_tie"] for row in mm_rows
    )

    body = {
        "schema_version": SCHEMA_VERSION,
        "claim_status": CLAIM_STATUS,
        "frozen_evaluation_panel": False,
        "eligible_for_b1_or_benchmark_claims": False,
        "limitations": [
            "Hebrew translations and multimodal cases are AI-authored and not human-reviewed.",
            "Hebrew-to-English is a ten-case within-document candidate retrieval diagnostic.",
            "Hebrew-to-Hebrew is unavailable without an exact full-candidate translation overlay.",
            "The image slice is Choice-only and cannot establish typed-workflow parity.",
        ],
        "input_sha256": {
            "hebrew_v2_raw": base_sha,
            "hebrew_v4_raw": overlay_sha,
            "hebrew_v4_effective_canonical": retrieval.sha256_bytes(
                _canonical_bytes(effective)
            ),
            "multimodal_v2_raw": mm_sha,
            **candidate_bindings,
            **(
                {}
                if he_translation_overlay is None
                else {"hebrew_translation_overlay_raw": he_he_status["input_sha256"]}
            ),
        },
        "artifact_bindings": dict(sorted(artifact_bindings.items())),
        "code_bindings": {
            "diagnostic_runner_sha256": retrieval.file_sha256(Path(__file__).resolve()),
            "production_scorer_sha256": retrieval.file_sha256(
                Path(retrieval.__file__).resolve()
            ),
        },
        "provider": {
            key: value
            for key, value in sorted(provider.identity().items())
            if key != "runtime_identity"
        },
        "contracts": {
            "gold_isolation": "provider receives only canonical question/options/policy/type or neutral text plus pixels",
            "option_invariance": "semantic option_id bytewise sort; display order validated but never embedded",
            "candidate_invariance": "candidate references sort only tie-breaks; identifiers and filenames are never embedded",
            "multimodal_matched_text": neutral,
            "scoring": "cosine at 768d and truncated-renormalized 256d",
        },
        "embedding_input_vectors": vector_manifest,
        "embedding_input_vectors_sha256": retrieval.sha256_json(vector_manifest),
        "hebrew": {"rows": he_rows, "summary": he_summary},
        "multimodal": {"rows": mm_rows, "summary": mm_summary},
    }
    return {**body, "receipt_sha256": retrieval.sha256_json(body)}


def _publication_path(path: Path, label: str) -> Path:
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError(f"{label} must be a normalized path")
    cursor = Path(path.anchor)
    for component in path.parent.parts[1:]:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink: {cursor}")
    if path.is_symlink():
        raise FileExistsError(f"refusing symlink output path: {path}")
    return path


def _exclusive_publish(path: Path, data: bytes) -> tuple[int, int]:
    path = _publication_path(path, "output path")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        installed = path.lstat()
        if not stat.S_ISREG(installed.st_mode):
            raise ValueError("published output is not a regular file")
        return installed.st_dev, installed.st_ino
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_and_publish(
    *, output: Path, runtime_sidecar: Path, provider: EmbeddingProvider, **kwargs: Any
) -> dict[str, Any]:
    output = _publication_path(output, "receipt output")
    runtime_sidecar = _publication_path(runtime_sidecar, "runtime output")
    if (
        output == runtime_sidecar
        or os.path.lexists(output)
        or os.path.lexists(runtime_sidecar)
    ):
        raise FileExistsError("receipt and runtime outputs must differ and be new")
    started = time.time()
    monotonic = time.perf_counter()
    receipt = build_receipt(provider=provider, **kwargs)
    receipt_bytes = retrieval.canonical_json(receipt)
    runtime = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "deterministic_receipt_file_sha256": retrieval.sha256_bytes(receipt_bytes),
        "started_unix_seconds": started,
        "elapsed_seconds": time.perf_counter() - monotonic,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **dict(provider.runtime_identity()),
        },
    }
    output_identity = _exclusive_publish(output, receipt_bytes)
    try:
        _exclusive_publish(runtime_sidecar, retrieval.canonical_json(runtime))
    except Exception:
        if not output.is_symlink():
            installed = output.lstat() if output.exists() else None
            if (
                installed is not None
                and stat.S_ISREG(installed.st_mode)
                and (installed.st_dev, installed.st_ino) == output_identity
                and retrieval.file_sha256(output)
                == retrieval.sha256_bytes(receipt_bytes)
            ):
                output.unlink()
        raise
    return receipt


class LocalProvider:
    def __init__(
        self,
        *,
        model_snapshot: Path,
        model_snapshot_tree_sha256: str,
        runtime_lock: Path,
        transformers_wheel: Path,
        dtype: str,
        device: str,
    ) -> None:
        observed = retrieval.artifact_sha256(model_snapshot.resolve())
        if observed != model_snapshot_tree_sha256:
            raise ValueError("model snapshot does not match external tree sha256")
        retrieval.verify_frozen_snapshot(model_snapshot.resolve())
        text_pilot.verify_runtime_lock(runtime_lock)
        text_pilot.verify_runtime_environment(runtime_lock)
        self._provider = retrieval.LocalEmbeddingGemma2Provider(
            dtype=dtype,
            device=device,
            truncate_dim=None,
            runtime_lock=runtime_lock,
            runtime_lock_sha256=text_pilot.RUNTIME_LOCK_SHA256,
            transformers_wheel=transformers_wheel,
        )
        if self._provider.snapshot_hash != observed:
            raise ValueError("locally loaded model differs from bound snapshot")

    def embed(self, inputs: Sequence[EmbeddingInput]) -> Sequence[Sequence[float]]:
        return [
            self._provider._embed(
                item.text, item.media, "r052-he-mm-dev:" + _input_key(item)
            )
            for item in inputs
        ]

    def identity(self) -> Mapping[str, Any]:
        return self._provider.identity()

    def runtime_identity(self) -> Mapping[str, Any]:
        return {
            **self._provider.identity()["runtime_identity"],
            "interpreter_prefix": str(Path(sys.prefix).resolve()),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "he-base",
        "he-overlay",
        "ntsb-candidates",
        "gao-candidates",
        "mm-receipt",
        "mm-artifact-root",
        "output",
        "runtime-sidecar",
        "model-snapshot",
        "runtime-lock",
        "transformers-wheel",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--he-translation-overlay", type=Path)
    parser.add_argument("--model-snapshot-tree-sha256", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    tree_sha = text_pilot._sha(
        args.model_snapshot_tree_sha256, "model snapshot tree sha256"
    )
    provider = LocalProvider(
        model_snapshot=args.model_snapshot,
        model_snapshot_tree_sha256=tree_sha,
        runtime_lock=args.runtime_lock,
        transformers_wheel=args.transformers_wheel,
        dtype=args.dtype,
        device=args.device,
    )
    bindings = {
        "model_snapshot_tree_sha256": tree_sha,
        "runtime_lock_sha256": text_pilot.verify_runtime_lock(args.runtime_lock),
        "transformers_wheel_sha256": retrieval.verify_artifact(
            args.transformers_wheel,
            retrieval.TRANSFORMERS_WHEEL_SHA256,
            "Transformers wheel",
        ),
    }
    run_and_publish(
        he_base=args.he_base,
        he_overlay=args.he_overlay,
        ntsb_candidates=args.ntsb_candidates,
        gao_candidates=args.gao_candidates,
        mm_receipt=args.mm_receipt,
        mm_artifact_root=args.mm_artifact_root,
        he_translation_overlay=args.he_translation_overlay,
        output=args.output,
        runtime_sidecar=args.runtime_sidecar,
        provider=provider,
        artifact_bindings=bindings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
