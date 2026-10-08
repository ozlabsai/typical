#!/usr/bin/env python3
"""Materialize the R052 Hebrew/MM development review packet.

This is intentionally not an evidence-panel builder.  It verifies and indexes
the frozen, AI-authored Hebrew and multimodal candidate bytes so human reviewers
can work from one deterministic packet.  It never asserts a review pass and it
never emits a production dossier spec, audit result, or promotion receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "r052-development-review-packet-v1"
BLOCKER_SCHEMA_VERSION = "r052-development-review-blockers-v1"
HEBREW_BASE_SCHEMA = "r052-he-en-twin-panel-development-draft-v2"
HEBREW_OVERLAY_SCHEMA = "r052-he-en-twin-panel-development-draft-v4-overlay"
HEBREW_EFFECTIVE_SCHEMA = "r052-he-en-twin-panel-development-draft-v4-effective"
MM_SCHEMA = "r052-mm-development-receipt-v2"

HEBREW_BASE_PATH = ".context/r052-hebrew-candidates/he_en_twin_panel_draft_v2.json"
HEBREW_BASE_SHA256 = "1530ff53615516b240c55aaea7a12ac98780663a2ad1d079528980a0c7b083c0"
HEBREW_OVERLAY_PATH = ".context/r052-hebrew-candidates/he_en_twin_panel_draft_v4.json"
HEBREW_OVERLAY_SHA256 = "7e08ed31ba0ab64f628fe1e4807b81c4a55993cdb65d77c3f5250b787ff303e8"
HEBREW_EFFECTIVE_SHA256 = "0f485e3d3528dda689e43fb4b10b552d87a1fb966a9d879f181cdb09482cbf76"
HEBREW_EFFECTIVE_BYTES = 45_640
HEBREW_NTSB_DOSSIER = ".context/r052-authoring/ntsb_candidates.json"
HEBREW_GAO_DOSSIER = ".context/r052-authoring/gao_candidates.json"
ACQUISITION_MANIFEST = "sources/r052/ntsb_gao_v1.json"

MM_MEMO_PATH = ".context/r052-image-candidates/R052_MM_10_CASE_DRAFT_V2.md"
MM_MEMO_SHA256 = "f2cabb6bd1e1f95e63c6551c6ebe5556bea68652fad742f11c3eb94345b30597"
MM_GENERATOR_PATH = ".context/r052-image-candidates/build_v2.py"
MM_GENERATOR_SHA256 = "5618e2d954ce70f269c3322c96c162d48d79a952c03d4b74dbb5bebcf9dc8895"
MM_RECEIPT_PATH = ".context/r052-image-candidates/v2/development-receipt.json"
MM_RECEIPT_SHA256 = "053789622b5267e20e98f722f3c016a79df801058865ccd7610c1b4b9a7600a5"
MM_RECEIPT_SIDECAR_PATH = f"{MM_RECEIPT_PATH}.sha256"
MM_TREE = ".context/r052-image-candidates/v2"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
HUMAN_REVIEW_STATUS = {
    "alignment_review_complete": False,
    "bilingual_human_review_complete": False,
    "contamination_review_complete": False,
    "hebrew_semantic_review_complete": False,
    "multimodal_semantic_review_complete": False,
    "pixel_causality_review_complete": False,
    "rights_review_complete": False,
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error


def _declared_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path")
    if path.as_posix() != value:
        raise ValueError(f"{label} must be a normalized relative path")
    return value


def _sha_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink: {current}")


def _output_leaf(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise ValueError("output file names must be non-empty path-free leaf names")
    return value


def _safe_file(root: Path, relative: str, label: str) -> Path:
    relative = _declared_path(relative, label)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink: {relative}")
    if not current.is_file() or current.stat().st_size == 0:
        raise ValueError(f"{label} must be a non-empty regular file: {relative}")
    return current


def _read_bound(
    root: Path, relative: str, expected_sha256: str, label: str
) -> tuple[bytes, dict[str, Any]]:
    expected_sha256 = _sha_field(expected_sha256, f"{label} expected hash")
    path = _safe_file(root, relative, label)
    raw = path.read_bytes()
    digest = _sha(raw)
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 drift: expected {expected_sha256}, got {digest}")
    return raw, {"bytes": len(raw), "path": relative, "sha256": digest}


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}")
    return value


def _materialize_hebrew(base: Any, overlay: Any, base_raw_sha256: str) -> bytes:
    if not isinstance(base, dict) or base.get("schema_version") != HEBREW_BASE_SCHEMA:
        raise ValueError("unsupported Hebrew base schema_version")
    if not isinstance(overlay, dict) or overlay.get("schema_version") != HEBREW_OVERLAY_SCHEMA:
        raise ValueError("unsupported Hebrew overlay schema_version")
    base_binding = _exact_keys(overlay.get("base"), {"path", "sha256", "reason"}, "overlay.base")
    if base_binding["path"] != HEBREW_BASE_PATH or base_binding["sha256"] != base_raw_sha256:
        raise ValueError("Hebrew overlay does not bind the exact supplied v2 base")

    operations = overlay.get("operations")
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError("Hebrew overlay must contain exactly one supported operation")
    operation = _exact_keys(
        operations[0], {"operation", "group_id", "preconditions", "replacement"}, "operation"
    )
    if operation["operation"] != "replace_group09_removal_partition":
        raise ValueError("unsupported Hebrew overlay operation")
    preconditions = _exact_keys(
        operation["preconditions"],
        {
            "decisive_candidate_chunk_refs",
            "excluded_answer_support_chunk_refs",
            "known_distractor_chunk_refs",
        },
        "operation.preconditions",
    )
    replacement = _exact_keys(
        operation["replacement"],
        {
            "excluded_answer_support_chunk_refs",
            "known_distractor_chunk_refs",
            "added_genuine_distractor_chunks",
            "removal_partition_claim",
        },
        "operation.replacement",
    )
    result = copy.deepcopy(base)
    groups = result.get("groups")
    if not isinstance(groups, list):
        raise TypeError("Hebrew base groups must be an array")
    matches = [group for group in groups if isinstance(group, dict) and group.get("group_id") == operation["group_id"]]
    if len(matches) != 1:
        raise ValueError("Hebrew overlay group_id must match exactly one base group")
    group = matches[0]
    decisive = group.get("decisive_evidence")
    if not isinstance(decisive, list):
        raise TypeError("target decisive_evidence must be an array")
    decisive_refs: list[Any] = []
    for entry in decisive:
        if not isinstance(entry, dict) or not isinstance(entry.get("candidate_chunk_refs"), list):
            raise TypeError("decisive evidence candidate_chunk_refs must be arrays")
        decisive_refs.extend(entry["candidate_chunk_refs"])
    actual = {
        "decisive_candidate_chunk_refs": decisive_refs,
        "excluded_answer_support_chunk_refs": [
            entry.get("chunk_ref") if isinstance(entry, dict) else None
            for entry in group.get("excluded_answer_support_chunk_refs", [])
        ],
        "known_distractor_chunk_refs": group.get("known_distractor_chunk_refs"),
    }
    if actual != dict(preconditions):
        raise ValueError("Hebrew overlay ordered preconditions do not match the base")
    group["excluded_answer_support_chunk_refs"] = copy.deepcopy(
        replacement["excluded_answer_support_chunk_refs"]
    )
    group["known_distractor_chunk_refs"] = copy.deepcopy(
        replacement["known_distractor_chunk_refs"]
    )
    group["added_genuine_distractor_chunks"] = copy.deepcopy(
        replacement["added_genuine_distractor_chunks"]
    )

    contract = overlay.get("deterministic_materialization")
    if not isinstance(contract, dict):
        raise TypeError("missing deterministic_materialization contract")
    receipt = contract.get("materialization_receipt")
    if not isinstance(receipt, dict):
        raise TypeError("missing Hebrew materialization receipt")
    operations_hash = _sha(_canonical(operations))
    if receipt.get("base_raw_sha256") != base_raw_sha256:
        raise ValueError("Hebrew receipt base hash mismatch")
    if receipt.get("operations_canonical_sha256") != operations_hash:
        raise ValueError("Hebrew operations canonical hash mismatch")
    if receipt.get("effective_schema_version") != HEBREW_EFFECTIVE_SCHEMA:
        raise ValueError("unsupported effective Hebrew schema version")
    result["schema_version"] = HEBREW_EFFECTIVE_SCHEMA
    result["materialization_receipt_ref"] = {
        "overlay_schema_version": HEBREW_OVERLAY_SCHEMA,
        "base_raw_sha256": base_raw_sha256,
        "operations_canonical_sha256": operations_hash,
    }
    raw = _canonical(result)
    expected_bytes = receipt.get("effective_canonical_byte_length")
    expected_hash = receipt.get("effective_canonical_sha256")
    if expected_bytes != HEBREW_EFFECTIVE_BYTES or expected_hash != HEBREW_EFFECTIVE_SHA256:
        raise ValueError("Hebrew overlay declares an unsupported effective output")
    if len(raw) != expected_bytes or _sha(raw) != expected_hash:
        raise ValueError("materialized Hebrew output differs from its exact receipt")
    return raw


def _inventory_add(
    inventory: dict[str, dict[str, Any]], item: Mapping[str, Any], role: str
) -> None:
    path = str(item["path"])
    prior = inventory.get(path)
    if prior is None:
        inventory[path] = {
            "bytes": item["bytes"],
            "path": path,
            "roles": [role],
            "sha256": item["sha256"],
        }
        return
    if prior["bytes"] != item["bytes"] or prior["sha256"] != item["sha256"]:
        raise ValueError(f"conflicting inventory binding for {path}")
    if role in prior["roles"]:
        raise ValueError(f"duplicate inventory role {role} for {path}")
    prior["roles"].append(role)
    prior["roles"].sort()


def _verify_hebrew_sources(
    root: Path, effective: Mapping[str, Any], inventory: dict[str, dict[str, Any]]
) -> None:
    bindings = _exact_keys(
        effective.get("input_bindings"),
        {
            "acquisition_manifest_sha256",
            "ntsb_candidate_dossier_sha256",
            "gao_candidate_dossier_sha256",
        },
        "Hebrew input_bindings",
    )
    for path, key, role in (
        (ACQUISITION_MANIFEST, "acquisition_manifest_sha256", "hebrew_acquisition_manifest"),
        (HEBREW_NTSB_DOSSIER, "ntsb_candidate_dossier_sha256", "hebrew_ntsb_candidate_dossier"),
        (HEBREW_GAO_DOSSIER, "gao_candidate_dossier_sha256", "hebrew_gao_candidate_dossier"),
    ):
        _, item = _read_bound(root, path, bindings[key], role)
        _inventory_add(inventory, item, role)

    source_texts = effective.get("source_texts")
    if not isinstance(source_texts, dict) or not source_texts:
        raise ValueError("Hebrew source_texts must be a non-empty object")
    source_values: dict[str, str] = {}
    for source_id in sorted(source_texts):
        binding = _exact_keys(source_texts[source_id], {"path", "sha256"}, f"source {source_id}")
        raw, item = _read_bound(root, binding["path"], binding["sha256"], f"source {source_id}")
        try:
            source_values[source_id] = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"source {source_id} is not UTF-8") from error
        _inventory_add(inventory, item, "hebrew_source_text")

    groups = effective.get("groups")
    if not isinstance(groups, list) or len(groups) != 10:
        raise ValueError("effective Hebrew candidate must contain exactly ten groups")
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("group_id"), str):
            raise TypeError("malformed Hebrew group")
        if group["group_id"] in group_ids:
            raise ValueError(f"duplicate Hebrew group_id: {group['group_id']}")
        group_ids.add(group["group_id"])
        source_id = group.get("source_id")
        if source_id not in source_values:
            raise ValueError(f"unknown Hebrew group source_id: {source_id}")
        candidates = list(group.get("decisive_evidence", [])) + list(
            group.get("added_genuine_distractor_chunks", [])
        )
        for candidate in candidates:
            candidate_source_id = candidate.get("source_id", source_id)
            if candidate_source_id not in source_values:
                raise ValueError(f"unknown Hebrew candidate source_id: {candidate_source_id}")
            for span in candidate.get("source_spans", []):
                span = _exact_keys(
                    span,
                    {"char_start_0based", "char_end_exclusive", "en_exact", "sha256"},
                    "Hebrew source span",
                )
                start, end, exact = span["char_start_0based"], span["char_end_exclusive"], span["en_exact"]
                if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
                    raise TypeError("Hebrew source span offsets must be integers")
                if start < 0 or end <= start or not isinstance(exact, str):
                    raise ValueError("invalid Hebrew source span")
                if source_values[candidate_source_id][start:end] != exact:
                    raise ValueError("Hebrew exact source span drift")
                if _sha(exact.encode("utf-8")) != _sha_field(span["sha256"], "span hash"):
                    raise ValueError("Hebrew source span SHA-256 drift")


def _verify_mm(root: Path, inventory: dict[str, dict[str, Any]]) -> Mapping[str, Any]:
    _, memo_item = _read_bound(root, MM_MEMO_PATH, MM_MEMO_SHA256, "MM v2 memo")
    _inventory_add(inventory, memo_item, "mm_review_memo")
    _, generator_item = _read_bound(root, MM_GENERATOR_PATH, MM_GENERATOR_SHA256, "MM v2 generator")
    _inventory_add(inventory, generator_item, "mm_generator")
    receipt_raw, receipt_item = _read_bound(root, MM_RECEIPT_PATH, MM_RECEIPT_SHA256, "MM v2 receipt")
    _inventory_add(inventory, receipt_item, "mm_development_receipt")
    receipt = _parse_json(receipt_raw, "MM receipt")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != MM_SCHEMA:
        raise ValueError("unsupported MM receipt schema_version")
    if receipt.get("status") != "ai_authored_unreviewed_development_only":
        raise ValueError("MM receipt status is not development-only")
    if receipt.get("eligible_for_sealed_or_public_eval") is not False:
        raise ValueError("MM receipt must be ineligible for sealed/public evaluation")
    for key in ("pixel_causality_review_complete", "rights_review_complete", "semantic_review_complete"):
        if receipt.get(key) is not False:
            raise ValueError(f"MM human gate must remain false: {key}")

    sidecar_raw, sidecar_item = _read_bound(
        root,
        MM_RECEIPT_SIDECAR_PATH,
        # The sidecar's own bytes are pinned by its exact expected content below.
        _sha(f"{MM_RECEIPT_SHA256}  development-receipt.json\n".encode()),
        "MM receipt sidecar",
    )
    if sidecar_raw != f"{MM_RECEIPT_SHA256}  development-receipt.json\n".encode():
        raise ValueError("MM receipt sidecar content drift")
    _inventory_add(inventory, sidecar_item, "mm_receipt_sidecar")

    generator = _exact_keys(receipt.get("generator"), {"normalization", "path", "pillow", "poppler", "python", "sha256"}, "MM generator binding")
    if generator["path"] != MM_GENERATOR_PATH or generator["sha256"] != MM_GENERATOR_SHA256:
        raise ValueError("MM receipt generator binding drift")
    acquisition = _exact_keys(receipt.get("acquisition_manifest"), {"path", "sha256"}, "MM acquisition binding")
    _, acquisition_item = _read_bound(root, acquisition["path"], acquisition["sha256"], "MM acquisition manifest")
    _inventory_add(inventory, acquisition_item, "mm_acquisition_manifest")

    cases = receipt.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("MM receipt must contain exactly ten cases")
    case_ids: set[str] = set()
    expected_tree_files = {"development-receipt.json", "development-receipt.json.sha256"}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise TypeError("malformed MM case")
        case_id = case["id"]
        if case_id in case_ids:
            raise ValueError(f"duplicate MM case id: {case_id}")
        case_ids.add(case_id)
        if case_id not in {f"M{number:02d}" for number in range(1, 11)}:
            raise ValueError(f"unsupported MM case id: {case_id}")
        for field, role in (
            ("real_image", "mm_real_image"),
            ("shuffled_image", "mm_shuffled_control"),
            ("missing_image", "mm_missing_control"),
        ):
            binding = case.get(field)
            if not isinstance(binding, dict):
                raise TypeError(f"missing {field} for {case_id}")
            relative_inside_tree = _declared_path(binding.get("path"), f"{case_id} {field} path")
            expected_tree_files.add(relative_inside_tree)
            full_path = f"{MM_TREE}/{relative_inside_tree}"
            raw, item = _read_bound(root, full_path, binding.get("sha256"), f"{case_id} {field}")
            if binding.get("bytes") != len(raw):
                raise ValueError(f"{case_id} {field} byte-length drift")
            _inventory_add(inventory, item, role)
        full_render = case.get("full_render")
        if not isinstance(full_render, dict) or not isinstance(full_render.get("command"), list):
            raise TypeError(f"malformed full_render for {case_id}")
        raw_paths = [arg for arg in full_render["command"] if isinstance(arg, str) and arg.endswith(".pdf")]
        if len(raw_paths) != 1:
            raise ValueError(f"{case_id} full_render must bind exactly one PDF")
        _, raw_item = _read_bound(root, raw_paths[0], case.get("raw_sha256"), f"{case_id} raw PDF")
        _inventory_add(inventory, raw_item, f"mm_raw_source:{case_id}")

    blank = _exact_keys(receipt.get("canonical_blank"), {"bytes", "path", "sha256"}, "MM canonical blank")
    blank_inside_tree = _declared_path(blank["path"], "MM blank path")
    expected_tree_files.add(blank_inside_tree)
    blank_raw, blank_item = _read_bound(root, f"{MM_TREE}/{blank_inside_tree}", blank["sha256"], "MM blank")
    if blank["bytes"] != len(blank_raw):
        raise ValueError("MM canonical blank byte-length drift")
    _inventory_add(inventory, blank_item, "mm_canonical_blank_control")

    distractors = receipt.get("distractors")
    if not isinstance(distractors, dict) or len(distractors) != 9:
        raise ValueError("MM receipt must bind exactly nine distractors")
    for distractor_id in sorted(distractors):
        binding = distractors[distractor_id]
        if not isinstance(binding, dict):
            raise TypeError(f"malformed MM distractor {distractor_id}")
        raw, item = _read_bound(root, binding.get("source_path"), binding.get("source_sha256"), f"MM distractor {distractor_id} source")
        _inventory_add(inventory, item, f"mm_distractor_source:{distractor_id}")
        byte_range = binding.get("byte_range")
        content = binding.get("content")
        if not isinstance(byte_range, list) or len(byte_range) != 2 or not all(isinstance(value, int) and not isinstance(value, bool) for value in byte_range):
            raise ValueError(f"invalid MM distractor byte range: {distractor_id}")
        if not isinstance(content, str) or raw[byte_range[0] : byte_range[1]] != content.encode("utf-8"):
            raise ValueError(f"MM distractor exact source bytes drift: {distractor_id}")
        if _sha(content.encode("utf-8")) != _sha_field(binding.get("content_sha256"), "distractor content hash"):
            raise ValueError(f"MM distractor content hash drift: {distractor_id}")

    tree = root / MM_TREE
    actual_tree_files: set[str] = set()
    for path in tree.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"MM v2 tree may not contain symlinks: {path.relative_to(tree)}")
        if path.is_file():
            actual_tree_files.add(path.relative_to(tree).as_posix())
    if actual_tree_files != expected_tree_files:
        missing = sorted(expected_tree_files - actual_tree_files)
        extra = sorted(actual_tree_files - expected_tree_files)
        raise ValueError(f"MM v2 file inventory drift; missing={missing}, extra={extra}")
    return receipt


def build_packet(workspace_root: Path) -> tuple[bytes, bytes, bytes]:
    root = workspace_root.absolute()
    _reject_symlink_ancestors(root, "workspace root")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("workspace root must be a non-symlink directory")
    inventory: dict[str, dict[str, Any]] = {}

    base_raw, base_item = _read_bound(root, HEBREW_BASE_PATH, HEBREW_BASE_SHA256, "Hebrew v2 base")
    overlay_raw, overlay_item = _read_bound(root, HEBREW_OVERLAY_PATH, HEBREW_OVERLAY_SHA256, "Hebrew v4 overlay")
    _inventory_add(inventory, base_item, "hebrew_v2_base")
    _inventory_add(inventory, overlay_item, "hebrew_v4_overlay")
    base = _parse_json(base_raw, "Hebrew v2 base")
    overlay = _parse_json(overlay_raw, "Hebrew v4 overlay")
    effective_raw = _materialize_hebrew(base, overlay, _sha(base_raw))
    effective = _parse_json(effective_raw, "effective Hebrew candidate")
    if effective.get("status", {}).get("promotion_ready") is not False:
        raise ValueError("effective Hebrew candidate promotion gate must remain false")
    _verify_hebrew_sources(root, effective, inventory)
    mm_receipt = _verify_mm(root, inventory)

    blockers = {
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "status": "human_review_required",
        "all_required_human_reviews_complete": False,
        "human_review_status": dict(HUMAN_REVIEW_STATUS),
        "promotion_allowed": False,
        "production_panel_allowed": False,
        "blockers": [
            {
                "area": "hebrew_alignment_and_semantics",
                "resolved": False,
                "requirements": overlay["remaining_human_only_risks"][:6],
            },
            {
                "area": "source_rights_and_contamination",
                "resolved": False,
                "requirements": [overlay["remaining_human_only_risks"][6]],
            },
            {
                "area": "multimodal_semantics_and_pixel_causality",
                "resolved": False,
                "requirements": [
                    "Two independent reviewers plus adjudication must validate every real/shuffled/missing image decision and pixel-causality claim.",
                    "Reviewers must resolve the recorded M01/M10 ambiguity and every case-level semantic claim.",
                ],
            },
            {
                "area": "multimodal_rights",
                "resolved": False,
                "requirements": [
                    "A qualified reviewer must complete source and image rights review, including the recorded M03 concern."
                ],
            },
            {
                "area": "typed_panel_coverage",
                "resolved": False,
                "requirements": [
                    "The MM slice is Choice-only (10 Choice, 0 Noul, 0 Score) and cannot satisfy the production typed-balance contract."
                ],
            },
        ],
    }
    blocker_raw = _canonical(blockers)
    index = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_review_only",
        "benchmark_derived": False,
        "eligible_for_evaluation": False,
        "is_audit_pass": False,
        "is_production_builder_spec": False,
        "is_promotion_receipt": False,
        "promotion_allowed": False,
        "human_review_status": dict(HUMAN_REVIEW_STATUS),
        "artifacts": {
            "blocker_report": {
                "bytes": len(blocker_raw),
                "path": "blockers.json",
                "sha256": _sha(blocker_raw),
            },
            "effective_hebrew_candidate": {
                "bytes": len(effective_raw),
                "path": "hebrew-effective-development-candidate.json",
                "sha256": _sha(effective_raw),
            },
        },
        "verified_input_inventory": [inventory[path] for path in sorted(inventory)],
        "summary": {
            "hebrew_group_count": len(effective["groups"]),
            "mm_case_count": len(mm_receipt["cases"]),
            "mm_decision_type_distribution": mm_receipt["decision_type_distribution"],
            "verified_input_file_count": len(inventory),
        },
    }
    return effective_raw, _canonical(index), blocker_raw


def _write_new_directory(output_dir: Path, files: Mapping[str, bytes]) -> None:
    output_dir = output_dir.absolute()
    parent = output_dir.parent
    safe_files = [(_output_leaf(name), raw) for name, raw in files.items()]
    _reject_symlink_ancestors(parent, "output parent")
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("output parent must be an existing non-symlink directory")
    try:
        output_dir.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite existing output path: {output_dir}") from error
    try:
        for name, raw in safe_files:
            path = output_dir / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(output_dir)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    effective, index, blockers = build_packet(args.workspace_root)
    _write_new_directory(
        args.output_dir,
        {
            "blockers.json": blockers,
            "hebrew-effective-development-candidate.json": effective,
            "review-index.json": index,
        },
    )
    print(f"wrote development-only R052 review packet: {args.output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
