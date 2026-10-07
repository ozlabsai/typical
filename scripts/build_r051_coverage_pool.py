"""Build the source-bound R051 typed-policy coverage pool.

The builder deliberately does not manufacture examples.  It authenticates a
candidate pool, natural-source snapshot, quarantine corpus, and frozen-parent
score receipt; selects roots deterministically; renders only the reversed
option-order sibling; and publishes a self-auditing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.exact_policy_executor import compile_policy
from scripts.typed_policy_executor import (
    validate_typed_policy_row,
    validate_typed_policy_siblings,
)

SCHEMA = "r051-coverage-pool-v1"
CANDIDATE_SCHEMA = "r051-candidate-pool-manifest-v1"
SOURCE_SCHEMA = "r051-natural-source-manifest-v1"
QUARANTINE_SCHEMA = "r051-quarantine-manifest-v1"
SCORE_SCHEMA = "r051-hard-root-score-v1"
ROOTS = 800
HARD_ROOTS = 200
STRATIFIED_ROOTS = 600
LONG_ROOTS = 40
FOUR_FIELD_ROOTS = 8
MAX_LENGTH = 16_384
LONG_TOKEN_MIN = 8_000
LONG_TOKEN_MAX = 12_000
MAX_PER_CELL = 20
MAX_PER_SOURCE_FAMILY = 80
TYPE_TOTALS = {"choice": 1_331, "noul": 1_331, "score": 1_330}
RENDERED_TYPE_TOTALS = {key: value * 2 for key, value in TYPE_TOTALS.items()}
QUARANTINE_TARGETS = {
    "jevbench": {
        "commit": "bb05a335bc809e61b20c0f745d25499a82b326fc",
        "coverage": "public-open-text-plus-sealed-lineage-denylist",
    },
    "workflowevals": {
        "commit": "0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb",
        "dataset_revisions": {
            "agent_trace_observability": "8635540973910a92465fe2bc53e195375aa6e1a8",
            "customer_service": "b1342f5a704587dbc465867c38c2694348ff86e4",
            "invoice_processing": "6beeb2d2acd65c086c835022f5f4d7434114cafc",
            "security_incidents": "fbe1ea5c69cf494157fd23f2002a0d9d9a418443",
        },
    },
}
MIN_SUBSTRING_CHARS = 200
TOKEN_SHINGLE_SIZE = 5
TOKEN_WINDOW_SIZE = 20
MINHASH_PERMUTATIONS = 64
MINHASH_THRESHOLD = 0.80
LICENSE_POLICY = "approved-training-and-redistribution-v1"
HEX64 = re.compile(r"[0-9a-f]{64}")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    # Criteria insertion order is the intervention. Sorting JSON object keys
    # would silently collapse canonical and reversed siblings on disk.
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode() + b"\n"
        for row in rows
    )


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse(payload: bytes, label: str) -> Any:
    def reject_constant(value):
        raise ValueError(f"{label}: nonfinite JSON constant {value}")

    try:
        return json.loads(
            payload, object_pairs_hook=_unique_object, parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label}: invalid JSON") from error


def _rows(payload: bytes, label: str) -> list[dict[str, Any]]:
    parsed = [_parse(line, label) for line in payload.splitlines() if line.strip()]
    if not parsed or any(not isinstance(row, dict) for row in parsed):
        raise ValueError(f"{label}: JSONL must contain nonempty objects")
    if _jsonl(parsed) != payload:
        raise ValueError(f"{label}: JSONL must use canonical persisted bytes")
    return parsed


def _checked_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _load_manifest(path: Path, expected_sha: str, schema: str, label: str) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    payload = path.read_bytes()
    digest = _sha(payload)
    if digest != _checked_sha(expected_sha, f"expected {label} hash"):
        raise ValueError(f"{label} differs from its external SHA-256 expectation")
    value = _parse(payload, label)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{label} has the wrong schema")
    return value, digest


def _load_bound_jsonl(
    path: Path, manifest: Mapping[str, Any], binding_name: str, label: str
) -> tuple[list[dict], bytes, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = path.read_bytes()
    binding = manifest.get(binding_name)
    rows = _rows(payload, label)
    digest = _sha(payload)
    if (
        not isinstance(binding, Mapping)
        or binding.get("sha256") != digest
        or type(binding.get("rows")) is not int
        or binding["rows"] != len(rows)
    ):
        raise ValueError(f"{label} differs from its manifest binding")
    return rows, payload, digest


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _text_fields(row: Mapping[str, Any]) -> list[str]:
    fields = [row["state"]]
    for question in row["questions"].values():
        fields.append(question["instructions"])
        fields.extend(question["criteria"].values())
    return fields


def _shingles(tokens: Sequence[str], size: int = TOKEN_SHINGLE_SIZE) -> frozenset[str]:
    if len(tokens) < size:
        return frozenset((" ".join(tokens),)) if tokens else frozenset()
    return frozenset(" ".join(tokens[index:index + size])
                     for index in range(len(tokens) - size + 1))


@lru_cache(maxsize=32_768)
def _minhash(shingles: frozenset[str]) -> tuple[int, ...]:
    if not shingles:
        return ()
    signatures = []
    for permutation in range(MINHASH_PERMUTATIONS):
        prefix = permutation.to_bytes(2, "big")
        signatures.append(min(
            int.from_bytes(hashlib.blake2b(prefix + item.encode(), digest_size=8).digest(), "big")
            for item in shingles
        ))
    return tuple(signatures)


def _windows(tokens: Sequence[str]) -> frozenset[str]:
    if len(tokens) < TOKEN_WINDOW_SIZE:
        return frozenset()
    return frozenset(" ".join(tokens[index:index + TOKEN_WINDOW_SIZE])
                     for index in range(len(tokens) - TOKEN_WINDOW_SIZE + 1))


@lru_cache(maxsize=32_768)
def _field_features(text: str) -> tuple[str, tuple[str, ...], frozenset[str], frozenset[str]]:
    normalized = _normalize(text)
    tokens = tuple(normalized.split())
    return normalized, tokens, _shingles(tokens), _windows(tokens)


def _substantive(normalized: str, tokens: Sequence[str]) -> bool:
    """Exclude ubiquitous option glosses while retaining real questions/facts."""
    return len(normalized) >= 32 or len(tokens) >= 4


def _contamination_audit(
    candidates: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    reference_raw: set[str] = set()
    reference_normalized: set[str] = set()
    reference_values: list[
        tuple[str, tuple[str, ...], frozenset[str], tuple[int, ...], frozenset[str]]
    ] = []
    reference_lineage: set[str] = set()
    validation_source_families: set[str] = set()
    validation_domains: set[str] = set()
    validation_generators: set[str] = set()
    validation_templates: set[str] = set()
    validation_documents: set[tuple[str, str]] = set()
    reference_ids: set[str] = set()
    for row in references:
        if (not isinstance(row.get("id"), str) or not row["id"]
                or row["id"] in reference_ids):
            raise ValueError("quarantine rows need unique nonempty IDs")
        reference_ids.add(row["id"])
        texts = row.get("texts")
        lineage = row.get("lineage")
        if (not isinstance(texts, list) or not texts
                or any(not isinstance(text, str) or not text for text in texts)
                or not isinstance(lineage, list)
                or any(not isinstance(value, str) or not value for value in lineage)):
            raise ValueError(f"{row['id']}: malformed quarantine texts or lineage")
        reference_lineage.update(lineage)
        if row.get("role") in {"validation", "benchmark"}:
            dimensions = (
                "source_family", "domain", "generator_family", "policy_template",
                "source_id", "document_id",
            )
            if any(not isinstance(row.get(key), str) or not row[key] for key in dimensions):
                raise ValueError(f"{row['id']}: held-out provenance dimensions are incomplete")
            validation_source_families.add(row["source_family"])
            validation_domains.add(row["domain"])
            validation_generators.add(row["generator_family"])
            validation_templates.add(row["policy_template"])
            validation_documents.add((row["source_id"], row["document_id"]))
        for text in texts:
            normalized, tokens, shingles, windows = _field_features(text)
            reference_raw.add(_sha(text.encode()))
            reference_normalized.add(_sha(normalized.encode()))
            if _substantive(normalized, tokens):
                reference_values.append((
                    normalized, tokens, shingles, _minhash(shingles), windows,
                ))

    seen_candidate_raw: set[str] = set()
    seen_candidate_normalized: set[str] = set()
    comparisons = 0
    for row in candidates:
        source = row["meta"]["source"]
        lineage = set(source["lineage"])
        if lineage & reference_lineage:
            raise ValueError(f"{row['id']}: upstream lineage overlaps quarantine")
        if source["source_family"] in validation_source_families:
            raise ValueError(f"{row['id']}: source family overlaps held-out validation")
        if source["domain"] in validation_domains:
            raise ValueError(f"{row['id']}: domain overlaps held-out validation")
        if source["generator_family"] in validation_generators:
            raise ValueError(f"{row['id']}: generator family overlaps held-out validation")
        if source["policy_template"] in validation_templates:
            raise ValueError(f"{row['id']}: policy template overlaps held-out validation")
        if (source["source_id"], source["document_id"]) in validation_documents:
            raise ValueError(f"{row['id']}: source document overlaps held-out validation")
        for text in _text_fields(row):
            raw_hash = _sha(text.encode())
            normalized, candidate_tokens, candidate_shingles, windows = _field_features(text)
            normalized_hash = _sha(normalized.encode())
            if (_substantive(normalized, candidate_tokens)
                    and (raw_hash in reference_raw or normalized_hash in reference_normalized)):
                raise ValueError(f"{row['id']}: exact text fingerprint overlaps quarantine")
            seen_candidate_raw.add(raw_hash)
            seen_candidate_normalized.add(normalized_hash)
            if not _substantive(normalized, candidate_tokens):
                continue
            signature = None
            for (reference_text, _, reference_shingles, reference_signature,
                 reference_windows) in reference_values:
                comparisons += 1
                shorter, longer = sorted((normalized, reference_text), key=len)
                if len(shorter) >= MIN_SUBSTRING_CHARS and shorter in longer:
                    raise ValueError(f"{row['id']}: substantial substring overlaps quarantine")
                if windows and reference_windows and windows & reference_windows:
                    raise ValueError(f"{row['id']}: sliding token window overlaps quarantine")
                smaller = min(len(candidate_shingles), len(reference_shingles))
                larger = max(len(candidate_shingles), len(reference_shingles))
                # Jaccard cannot reach the threshold when the cardinality ratio
                # is already below it, so avoid hashing obviously unlike lengths.
                if (larger and smaller / larger >= MINHASH_THRESHOLD
                        and candidate_shingles and reference_signature):
                    if signature is None:
                        signature = _minhash(candidate_shingles)
                    estimate = sum(left == right for left, right in zip(
                        signature, reference_signature)) / MINHASH_PERMUTATIONS
                    if estimate >= MINHASH_THRESHOLD:
                        raise ValueError(f"{row['id']}: MinHash overlap exceeds threshold")
    return {
        "status": "pass",
        "candidate_raw_field_hashes": len(seen_candidate_raw),
        "candidate_raw_field_inventory_sha256": _sha(_canonical(sorted(seen_candidate_raw))),
        "candidate_normalized_field_hashes": len(seen_candidate_normalized),
        "candidate_normalized_field_inventory_sha256": _sha(
            _canonical(sorted(seen_candidate_normalized))
        ),
        "reference_raw_field_hashes": len(reference_raw),
        "reference_raw_field_inventory_sha256": _sha(_canonical(sorted(reference_raw))),
        "reference_normalized_field_hashes": len(reference_normalized),
        "reference_normalized_field_inventory_sha256": _sha(
            _canonical(sorted(reference_normalized))
        ),
        "pairwise_field_comparisons": comparisons,
        "substantial_substring_chars": MIN_SUBSTRING_CHARS,
        "token_shingle_size": TOKEN_SHINGLE_SIZE,
        "sliding_window_tokens": TOKEN_WINDOW_SIZE,
        "minhash_permutations": MINHASH_PERMUTATIONS,
        "minhash_threshold": MINHASH_THRESHOLD,
        "lineage_disjoint": True,
        "validation_source_domain_generator_policy_document_disjoint": True,
    }


def _validate_sources(
    documents: Sequence[Mapping[str, Any]], source_manifest: Mapping[str, Any]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if (source_manifest.get("license_policy") != LICENSE_POLICY
            or source_manifest.get("license_cleared") is not True):
        raise ValueError("source snapshot lacks the approved license contract")
    tokenizer = source_manifest.get("tokenizer")
    if (not isinstance(tokenizer, Mapping)
            or not isinstance(tokenizer.get("id"), str) or not tokenizer["id"]
            or not isinstance(tokenizer.get("revision"), str) or not tokenizer["revision"]):
        raise ValueError("source manifest must pin tokenizer ID and revision")
    indexed = {}
    lineage_splits: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        required = {
            "source_id", "document_id", "upstream_id", "source_family", "domain",
            "revision", "license", "license_url", "split", "content_kind", "text",
            "lineage",
        }
        if set(document) != required:
            raise ValueError("natural-source documents must use the exact source schema")
        if any(not isinstance(document[key], str) or not document[key]
               for key in required - {"lineage"}):
            raise ValueError("natural-source document fields must be nonempty strings")
        if (document["content_kind"] != "natural"
                or document["split"] != "train"
                or not isinstance(document["lineage"], list)
                or not document["lineage"]
                or any(not isinstance(value, str) or not value for value in document["lineage"])):
            raise ValueError("source documents must be natural, training-only, and lineaged")
        key = (document["source_id"], document["document_id"])
        if key in indexed:
            raise ValueError(f"duplicate natural-source document: {key}")
        indexed[key] = document
        for lineage in document["lineage"]:
            lineage_splits[lineage].add(document["split"])
    if any(len(splits) != 1 for splits in lineage_splits.values()):
        raise ValueError("an upstream lineage crosses source splits")
    return indexed


def _validate_candidate(
    row: Mapping[str, Any], documents: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, Any]:
    contract = validate_typed_policy_row(row)
    if contract["option_order"] != "canonical":
        raise ValueError(f"{row['id']}: candidates must be canonical roots")
    meta = row["meta"]
    source = meta.get("source")
    if (meta.get("split") != "train" or not isinstance(source, Mapping)):
        raise ValueError(f"{row['id']}: candidate lacks training source provenance")
    required = {
        "source_id", "document_id", "upstream_id", "source_family", "domain",
        "revision", "license", "license_url", "generator_family", "policy_template",
        "lineage", "raw_state_sha256", "normalized_state_sha256",
    }
    if set(source) != required:
        raise ValueError(f"{row['id']}: source provenance fields differ from the contract")
    key = (source["source_id"], source["document_id"])
    document = documents.get(key)
    if document is None:
        raise ValueError(f"{row['id']}: source document is absent from the snapshot")
    copied = {key: document[key] for key in (
        "upstream_id", "source_family", "domain", "revision", "license", "license_url",
        "lineage",
    )}
    if any(source[key] != value for key, value in copied.items()):
        raise ValueError(f"{row['id']}: source provenance differs from the snapshot")
    if row["state"] != document["text"]:
        raise ValueError(f"{row['id']}: state is not byte-identical natural source text")
    raw_hash = _sha(row["state"].encode())
    normalized_hash = _sha(_normalize(row["state"]).encode())
    if (source["raw_state_sha256"] != raw_hash
            or source["normalized_state_sha256"] != normalized_hash):
        raise ValueError(f"{row['id']}: raw or normalized state hash mismatch")
    if not isinstance(source["generator_family"], str) or not source["generator_family"]:
        raise ValueError(f"{row['id']}: generator family is missing")
    if not isinstance(source["policy_template"], str) or not source["policy_template"]:
        raise ValueError(f"{row['id']}: policy template is missing")
    if not isinstance(source["lineage"], list) or not source["lineage"]:
        raise ValueError(f"{row['id']}: source lineage is missing")
    forbidden_identity = " ".join(
        str(source[key]) for key in ("source_id", "document_id", "upstream_id")
    ).casefold()
    if any(name in forbidden_identity for name in ("jevbench", "workfloweval", "typesafe")):
        raise ValueError(f"{row['id']}: benchmark-like source identity is forbidden")
    if row.get("soft_targets"):
        teacher = meta.get("teacher_validation")
        if (not isinstance(teacher, Mapping)
                or set(teacher) != {"method", "receipt_sha256", "top1_matches_gold"}
                or teacher.get("method") not in {"human_gold", "independently_executed"}
                or teacher.get("top1_matches_gold") is not True):
            raise ValueError(f"{row['id']}: teacher probabilities lack independent validation")
        _checked_sha(teacher.get("receipt_sha256"), f"{row['id']} teacher receipt")
    return contract


def _score_map(
    receipt: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], bindings: Mapping[str, str]
) -> dict[str, Mapping[str, Any]]:
    if receipt.get("schema") != SCORE_SCHEMA or receipt.get("status") != "complete":
        raise ValueError("hard-root score receipt is incomplete or has the wrong schema")
    required_bindings = {
        "candidate_pool_sha256", "candidate_manifest_sha256", "parent_model_tree_sha256",
        "parent_adapter_tree_sha256", "scorer_code_sha256", "runtime_lock_sha256",
    }
    actual_bindings = receipt.get("bindings")
    if not isinstance(actual_bindings, Mapping) or set(actual_bindings) != required_bindings:
        raise ValueError("hard-root receipt has incomplete bindings")
    for key in required_bindings:
        _checked_sha(actual_bindings[key], f"hard-root {key}")
    if any(actual_bindings[key] != value for key, value in bindings.items()):
        raise ValueError("hard-root receipt input binding mismatch")
    ids = sorted(row["meta"]["semantic_group_id"] for row in candidates)
    inventory = receipt.get("root_inventory")
    if (inventory != ids
            or receipt.get("root_inventory_sha256") != _sha(_canonical(ids))):
        raise ValueError("hard-root receipt does not bind the exact candidate inventory")
    scores = receipt.get("scores")
    if not isinstance(scores, list) or len(scores) != len(ids):
        raise ValueError("hard-root receipt must score every candidate exactly once")
    indexed = {}
    for score in scores:
        if not isinstance(score, Mapping) or set(score) != {
            "semantic_group_id", "wrong_complete_outcome", "complete_outcome_nll",
            "mean_gold_label_nll",
        }:
            raise ValueError("malformed hard-root score")
        identifier = score["semantic_group_id"]
        values = (score["complete_outcome_nll"], score["mean_gold_label_nll"])
        if (identifier in indexed or identifier not in ids
                or type(score["wrong_complete_outcome"]) is not bool
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in values)):
            raise ValueError("invalid or duplicate hard-root score")
        indexed[identifier] = score
    if set(indexed) != set(ids):
        raise ValueError("hard-root score coverage differs from candidates")
    return indexed


def _cell(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = row["meta"]["source"]
    return source["source_family"], source["domain"], source["policy_template"]


def _rank(row: Mapping[str, Any], scores: Mapping[str, Mapping[str, Any]]) -> tuple:
    identifier = row["meta"]["semantic_group_id"]
    score = scores[identifier]
    return (
        -int(score["wrong_complete_outcome"]),
        -float(score["complete_outcome_nll"]),
        -float(score["mean_gold_label_nll"]),
        identifier,
    )


def _select(
    candidates: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], set[str]]:
    ranked = sorted(candidates, key=lambda row: _rank(row, scores))
    selected: list[Mapping[str, Any]] = []
    hard_ids: set[str] = set()
    cell_counts: Counter = Counter()
    family_counts: Counter = Counter()

    def eligible(row):
        family = row["meta"]["source"]["source_family"]
        return cell_counts[_cell(row)] < MAX_PER_CELL and family_counts[family] < MAX_PER_SOURCE_FAMILY

    def add(row, hard=False):
        selected.append(row)
        cell_counts[_cell(row)] += 1
        family = row["meta"]["source"]["source_family"]
        family_counts[family] += 1
        if hard:
            hard_ids.add(row["meta"]["semantic_group_id"])

    for row in ranked:
        if eligible(row):
            add(row, hard=True)
            if len(selected) == HARD_ROOTS:
                break
    if len(selected) != HARD_ROOTS:
        raise ValueError("candidate pool cannot supply 200 capped hard roots")

    queues: dict[tuple[str, str, str], deque] = defaultdict(deque)
    for row in ranked:
        if row["meta"]["semantic_group_id"] not in hard_ids:
            queues[_cell(row)].append(row)
    cells = sorted(queues)
    while len(selected) < ROOTS:
        progressed = False
        for cell in cells:
            queue = queues[cell]
            while queue and not eligible(queue[0]):
                queue.popleft()
            if queue and len(selected) < ROOTS:
                add(queue.popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != ROOTS:
        raise ValueError("candidate pool cannot supply 600 capped stratified roots")
    return sorted(selected, key=lambda row: row["meta"]["semantic_group_id"]), hard_ids


def _long_contract(
    row: Mapping[str, Any], token_offsets: Callable[[str], Sequence[tuple[int, int]]]
) -> str | None:
    is_long = row["meta"].get("is_long_context")
    evidence = row["meta"].get("decisive_evidence")
    if type(is_long) is not bool:
        raise ValueError(f"{row['id']}: is_long_context must be Boolean")
    offsets = list(token_offsets(row["state"]))
    if any(not isinstance(item, Sequence) or len(item) != 2 for item in offsets):
        raise ValueError(f"{row['id']}: tokenizer returned malformed offsets")
    if len(offsets) > MAX_LENGTH:
        raise ValueError(f"{row['id']}: state alone exceeds max_length={MAX_LENGTH}")
    if not is_long:
        if evidence is not None:
            raise ValueError(f"{row['id']}: short root cannot declare decisive evidence")
        return None
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "text", "start_char", "end_char", "field", "removed_label", "removed_outcome",
        "removed_state_sha256", "validation",
    }:
        raise ValueError(f"{row['id']}: long root lacks a decisive-evidence counterfactual")
    if len(offsets) < LONG_TOKEN_MIN or len(offsets) > LONG_TOKEN_MAX:
        raise ValueError(f"{row['id']}: long root is outside the 8k-12k token bucket")
    start, end = evidence["start_char"], evidence["end_char"]
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start
            or row["state"][start:end] != evidence["text"]):
        raise ValueError(f"{row['id']}: decisive exhibit span is invalid")
    removed_state = row["state"][:start] + row["state"][end:]
    if evidence["removed_state_sha256"] != _sha(removed_state.encode()):
        raise ValueError(f"{row['id']}: evidence-removed state hash mismatch")
    validation = evidence["validation"]
    expected_validation_keys = {
        "method", "receipt_sha256", "verifier_code_sha256",
        "original_state_sha256", "removed_state_sha256",
        "original_label", "removed_label", "original_outcome", "removed_outcome",
    }
    if (not isinstance(validation, Mapping)
            or set(validation) != expected_validation_keys
            or validation.get("method") not in {"human_gold", "independently_executed"}):
        raise ValueError(f"{row['id']}: decisive counterfactual lacks independent validation")
    _checked_sha(validation.get("receipt_sha256"), f"{row['id']} evidence receipt")
    _checked_sha(validation.get("verifier_code_sha256"), f"{row['id']} evidence verifier")
    field = evidence["field"]
    meta = row["meta"]
    if field not in meta["field_order"]:
        raise ValueError(f"{row['id']}: decisive field is unknown")
    qid = next(qid for qid, value in meta["qid_to_field"].items() if value == field)
    original_label = row["targets"][qid]["label"]
    removed_label = evidence["removed_label"]
    if removed_label not in row["questions"][qid]["criteria"]:
        raise ValueError(f"{row['id']}: removed-evidence label is invalid")
    removed_trace = dict(meta["action_trace"])
    positives = set(meta["field_events"][field]["positive_labels"])
    removed_trace[field] = removed_label in positives
    policy = compile_policy(meta["policy"])
    removed_outcome = policy.truth_outcomes[sum(
        int(removed_trace[name]) << index for index, name in enumerate(policy.fields)
    )].as_dict()
    if (evidence["removed_outcome"] != removed_outcome
            or (removed_label == original_label and removed_outcome == meta["gold_outcome"])):
        raise ValueError(f"{row['id']}: exhibit removal does not flip field or policy outcome")
    expected_validation = {
        "original_state_sha256": _sha(row["state"].encode()),
        "removed_state_sha256": _sha(removed_state.encode()),
        "original_label": original_label,
        "removed_label": removed_label,
        "original_outcome": meta["gold_outcome"],
        "removed_outcome": removed_outcome,
    }
    if any(validation[key] != value for key, value in expected_validation.items()):
        raise ValueError(f"{row['id']}: decisive validation binding mismatch")
    midpoint = next((index for index, (left, right) in enumerate(offsets)
                     if left <= (start + end) / 2 <= right), None)
    if midpoint is None:
        raise ValueError(f"{row['id']}: decisive exhibit is not token-aligned")
    quantile = (midpoint + 0.5) / len(offsets)
    return "early" if quantile < 1 / 3 else "middle" if quantile < 2 / 3 else "late"


def _validate_distribution(
    selected: Sequence[Mapping[str, Any]],
    token_offsets: Callable[[str], Sequence[tuple[int, int]]],
    input_tokens: Callable[[Mapping[str, Any], str, bool], int],
) -> dict[str, Any]:
    arities = Counter(len(row["questions"]) for row in selected)
    if arities != {5: ROOTS - FOUR_FIELD_ROOTS, 4: FOUR_FIELD_ROOTS}:
        raise ValueError("selected roots must contain 792 five-field and 8 four-field groups")
    type_counts: Counter = Counter()
    noul_truths: Counter = Counter()
    score_ranks: Counter = Counter()
    score_ranks_by_arity: dict[int, Counter] = defaultdict(Counter)
    quantiles: Counter = Counter()
    long_count = 0
    max_observed_input_tokens = 0
    for row in selected:
        for qid, question in row["questions"].items():
            for reversed_options in (False, True):
                length = input_tokens(row, qid, reversed_options)
                if type(length) is not int or length <= 0:
                    raise ValueError(f"{row['id']}/{qid}: invalid rendered token count")
                if length > MAX_LENGTH:
                    raise ValueError(
                        f"{row['id']}/{qid}: {length} rendered tokens exceeds {MAX_LENGTH}"
                    )
                max_observed_input_tokens = max(max_observed_input_tokens, length)
            kind = question["type"]
            type_counts[kind] += 1
            label = row["targets"][qid]["label"]
            if kind == "noul":
                noul_truths[label] += 1
            if kind == "score":
                order = row["meta"]["score_orders"][qid]
                rank = order.index(label)
                score_ranks[rank] += 1
                score_ranks_by_arity[len(order)][rank] += 1
        quantile = _long_contract(row, token_offsets)
        if quantile is not None:
            long_count += 1
            quantiles[quantile] += 1
    if dict(type_counts) != TYPE_TOTALS:
        raise ValueError("selected canonical type totals differ from the frozen contract")
    if set(noul_truths) != {"no", "yes"} or max(noul_truths.values()) - min(noul_truths.values()) > 1:
        raise ValueError("Noul truth values are not maximally balanced")
    if not score_ranks:
        raise ValueError("ordinal Score target ranks are not maximally balanced")
    for arity, counts in score_ranks_by_arity.items():
        if (set(counts) != set(range(arity))
                or max(counts.values()) - min(counts.values()) > 1):
            raise ValueError(
                f"ordinal Score target ranks are not balanced for arity {arity}"
            )
    if long_count != LONG_ROOTS or set(quantiles) != {"early", "middle", "late"} \
            or max(quantiles.values()) - min(quantiles.values()) > 1:
        raise ValueError("long roots or decisive-evidence quantiles differ from the contract")
    return {
        "semantic_roots": ROOTS,
        "canonical_decisions": sum(type_counts.values()),
        "rendered_decisions": 2 * sum(type_counts.values()),
        "arities": {str(key): value for key, value in sorted(arities.items())},
        "canonical_types": dict(sorted(type_counts.items())),
        "rendered_types": RENDERED_TYPE_TOTALS,
        "noul_truths": dict(sorted(noul_truths.items())),
        "score_target_ranks": {str(key): value for key, value in sorted(score_ranks.items())},
        "score_target_ranks_by_arity": {
            str(arity): {str(rank): count for rank, count in sorted(counts.items())}
            for arity, counts in sorted(score_ranks_by_arity.items())
        },
        "long_roots": long_count,
        "decisive_evidence_quantiles": dict(sorted(quantiles.items())),
        "max_length": MAX_LENGTH,
        "max_observed_rendered_input_tokens": max_observed_input_tokens,
    }


def _render(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for root in selected:
        canonical = deepcopy(root)
        group_id = canonical["meta"]["semantic_group_id"]
        canonical["id"] = f"{group_id}--canonical"
        canonical["meta"]["option_order"] = "canonical"
        reversed_row = deepcopy(canonical)
        reversed_row["id"] = f"{group_id}--reversed"
        reversed_row["meta"]["option_order"] = "reversed"
        for question in reversed_row["questions"].values():
            question["criteria"] = dict(reversed(tuple(question["criteria"].items())))
        rows.extend((canonical, reversed_row))
    validate_typed_policy_siblings(rows)
    return rows


def build_pool(
    *,
    candidates: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    quarantine: Sequence[Mapping[str, Any]],
    score_receipt: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    quarantine_manifest: Mapping[str, Any],
    bindings: Mapping[str, str],
    token_offsets: Callable[[str], Sequence[tuple[int, int]]],
    input_tokens: Callable[[Mapping[str, Any], str, bool], int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate, select, and render the exact R051 executable pool."""
    if (candidate_manifest.get("schema") != CANDIDATE_SCHEMA
            or source_manifest.get("schema") != SOURCE_SCHEMA
            or quarantine_manifest.get("schema") != QUARANTINE_SCHEMA):
        raise ValueError("R051 input manifest schema mismatch")
    if candidate_manifest.get("max_length") != MAX_LENGTH:
        raise ValueError("candidate manifest must freeze max_length=16384")
    indexed_documents = _validate_sources(documents, source_manifest)
    contracts = [_validate_candidate(row, indexed_documents) for row in candidates]
    ids = [contract["semantic_group_id"] for contract in contracts]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate semantic group IDs must be unique")
    if candidate_manifest.get("source_manifest_sha256") != bindings["source_manifest_sha256"]:
        raise ValueError("candidate manifest does not bind the source manifest")
    if (quarantine_manifest.get("scope") != "jevbench-and-typesafe-heldout-v1"
            or quarantine_manifest.get("targets") != QUARANTINE_TARGETS):
        raise ValueError("quarantine manifest does not cover frozen external targets")
    contamination = _contamination_audit(candidates, quarantine)
    scores = _score_map(score_receipt, candidates, {
        "candidate_pool_sha256": bindings["candidate_pool_sha256"],
        "candidate_manifest_sha256": bindings["candidate_manifest_sha256"],
    })
    selected, hard_ids = _select(candidates, scores)
    distribution = _validate_distribution(selected, token_offsets, input_tokens)
    rendered = _render(selected)
    selected_ids = [row["meta"]["semantic_group_id"] for row in selected]
    cell_counts = Counter(_cell(row) for row in selected)
    family_counts = Counter(row["meta"]["source"]["source_family"] for row in selected)
    selection = {
        "method": "ranked-hard-then-cell-round-robin-v1",
        "rank": ["wrong_complete_outcome_desc", "complete_outcome_nll_desc",
                 "mean_gold_label_nll_desc", "semantic_group_id_asc"],
        "hard_roots": HARD_ROOTS,
        "stratified_roots": STRATIFIED_ROOTS,
        "max_per_source_domain_policy_cell": MAX_PER_CELL,
        "max_per_source_family": MAX_PER_SOURCE_FAMILY,
        "selected_root_ids": selected_ids,
        "selected_root_ids_sha256": _sha(_canonical(selected_ids)),
        "hard_root_ids": sorted(hard_ids),
        "hard_root_ids_sha256": _sha(_canonical(sorted(hard_ids))),
        "selected_cells": len(cell_counts),
        "observed_max_per_cell": max(cell_counts.values()),
        "selected_source_families": len(family_counts),
        "observed_max_per_source_family": max(family_counts.values()),
    }
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "panel": "typed_policy_v1",
        "builder_sha256": _sha(Path(__file__).read_bytes()),
        "max_length": MAX_LENGTH,
        "outputs": {"coverage_pool": {
            "rows": len(rendered), "sha256": _sha(_jsonl(rendered)),
        }},
        "bindings": dict(sorted(bindings.items())),
        "source_contract": {
            "natural_source_only": True,
            "license_policy": LICENSE_POLICY,
            "tokenizer": source_manifest["tokenizer"],
            "source_documents": len(documents),
        },
        "selection": selection,
        "distribution": distribution,
        "contamination_audit": contamination,
    }
    return rendered, manifest


def _hf_token_tools(model_id: str, revision: str):
    from transformers import AutoTokenizer

    from scripts.train_gemma_decision_lora import SYSTEM, render

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def offsets(text: str):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        mapping = encoded.get("offset_mapping")
        if mapping is None:
            raise ValueError("the pinned tokenizer does not expose offset mappings")
        return [tuple(item) for item in mapping if tuple(item) != (0, 0)]

    def input_length(row: Mapping[str, Any], qid: str, reversed_options: bool) -> int:
        question = row["questions"][qid]
        items = list(question["criteria"].items())
        if reversed_options:
            items.reverse()
        prompt, _ = render({
            "state": row["state"],
            "instructions": question["instructions"],
            "items": items,
            "probabilities": [0.0] * len(items),
        })
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return offsets, input_length


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_pool(
    *, candidates_path: Path, candidate_manifest_path: Path,
    sources_path: Path, source_manifest_path: Path,
    quarantine_path: Path, quarantine_manifest_path: Path,
    score_receipt_path: Path, out: Path,
    expected_candidate_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    expected_quarantine_manifest_sha256: str,
    expected_score_receipt_sha256: str,
) -> dict[str, Any]:
    """Authenticate path inputs and atomically publish a complete pool."""
    out = out.absolute()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite R051 pool: {out}")
    candidate_manifest, candidate_manifest_sha = _load_manifest(
        candidate_manifest_path, expected_candidate_manifest_sha256,
        CANDIDATE_SCHEMA, "candidate manifest")
    source_manifest, source_manifest_sha = _load_manifest(
        source_manifest_path, expected_source_manifest_sha256,
        SOURCE_SCHEMA, "source manifest")
    quarantine_manifest, quarantine_manifest_sha = _load_manifest(
        quarantine_manifest_path, expected_quarantine_manifest_sha256,
        QUARANTINE_SCHEMA, "quarantine manifest")
    if (score_receipt_path.is_symlink() or not score_receipt_path.is_file()
            or score_receipt_path.stat().st_size <= 0):
        raise ValueError("score receipt must be a nonempty regular file")
    score_payload = score_receipt_path.read_bytes()
    if _sha(score_payload) != _checked_sha(expected_score_receipt_sha256, "expected score receipt hash"):
        raise ValueError("score receipt differs from its external SHA-256 expectation")
    score_receipt = _parse(score_payload, "score receipt")
    candidates, _, candidate_sha = _load_bound_jsonl(
        candidates_path, candidate_manifest, "output", "candidate pool")
    documents, _, sources_sha = _load_bound_jsonl(
        sources_path, source_manifest, "output", "source snapshot")
    quarantine, _, quarantine_sha = _load_bound_jsonl(
        quarantine_path, quarantine_manifest, "output", "quarantine corpus")
    tokenizer = source_manifest.get("tokenizer", {})
    bindings = {
        "candidate_pool_sha256": candidate_sha,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "source_snapshot_sha256": sources_sha,
        "source_manifest_sha256": source_manifest_sha,
        "quarantine_sha256": quarantine_sha,
        "quarantine_manifest_sha256": quarantine_manifest_sha,
        "score_receipt_sha256": _sha(score_payload),
    }
    token_offsets, input_tokens = _hf_token_tools(tokenizer["id"], tokenizer["revision"])
    rows, manifest = build_pool(
        candidates=candidates, documents=documents, quarantine=quarantine,
        score_receipt=score_receipt, candidate_manifest=candidate_manifest,
        source_manifest=source_manifest, quarantine_manifest=quarantine_manifest,
        bindings=bindings, token_offsets=token_offsets, input_tokens=input_tokens,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=out.parent))
    try:
        pool_payload = _jsonl(rows)
        _write_exclusive(stage / "coverage-pool.jsonl", pool_payload)
        persisted = _rows((stage / "coverage-pool.jsonl").read_bytes(), "persisted coverage pool")
        validate_typed_policy_siblings(persisted)
        if persisted != rows or (stage / "coverage-pool.jsonl").read_bytes() != pool_payload:
            raise ValueError("persisted coverage pool failed exact byte/object round-trip validation")
        _write_exclusive(stage / "manifest.json", json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode() + b"\n")
        os.rename(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--quarantine-manifest", type=Path, required=True)
    parser.add_argument("--score-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-quarantine-manifest-sha256", required=True)
    parser.add_argument("--expected-score-receipt-sha256", required=True)
    args = parser.parse_args()
    manifest = write_pool(
        candidates_path=args.candidates, candidate_manifest_path=args.candidate_manifest,
        sources_path=args.sources, source_manifest_path=args.source_manifest,
        quarantine_path=args.quarantine, quarantine_manifest_path=args.quarantine_manifest,
        score_receipt_path=args.score_receipt, out=args.out,
        expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_quarantine_manifest_sha256=args.expected_quarantine_manifest_sha256,
        expected_score_receipt_sha256=args.expected_score_receipt_sha256,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
