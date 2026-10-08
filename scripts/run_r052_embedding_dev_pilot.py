#!/usr/bin/env python3
"""Reproduce the English-only R052 EmbeddingGemma 2 development pilot.

This diagnostic intentionally consumes the unreviewed NTSB/GAO candidate plans,
not the frozen R052 panel.  Its output is therefore development evidence only:
it is neither B1 nor benchmark evidence.  The deterministic receipt contains
the exact query/document serialization and rankings; timings and host details
are written to a separate runtime sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    from scripts import score_r052_evidence_retrieval as retrieval
except ModuleNotFoundError:  # Supports ``uv run python scripts/run_...py``.
    import score_r052_evidence_retrieval as retrieval

SCHEMA_VERSION = "r052-embedding-development-pilot-receipt-v2"
RUNTIME_SCHEMA_VERSION = "r052-embedding-development-pilot-runtime-v1"
CLAIM_STATUS = "development_only_not_human_reviewed_not_b1_not_benchmark"
MODES = ("question_only", "all", "per_option", "fact_all")
DIMENSIONS = (768, 256)
TOP_KS = (1, 2, 4)
EXPECTED_CANDIDATE_SCHEMAS = {
    "ntsb": "r052-ntsb-candidate-dossier-plan-v1",
    "gao": "r052-gao-candidate-dossier-plan-v1",
}
DECISION_TYPES = frozenset({"Choice", "Noul", "Score"})


class TextEmbeddingProvider(Protocol):
    """Minimal interface used by the pure pilot implementation."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def identity(self) -> Mapping[str, Any]: ...

    def runtime_identity(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Document:
    case_ref: str
    chunk_ref: str
    text: str
    source_anchor: Mapping[str, Any]

    @property
    def embedding_text(self) -> str:
        return f"title: evidence | text: {self.text}"


@dataclass(frozen=True)
class Question:
    case_ref: str
    question_ref: str
    decision_type: str
    policy_role: str
    instructions: str
    options: tuple[tuple[str, str], ...]
    decisive: tuple[str, ...]
    distractors: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    case_ref: str
    source_kind: str
    documents: tuple[Document, ...]
    questions: tuple[Question, ...]


def _json_no_duplicates(path: Path) -> tuple[dict[str, Any], str]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidate JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path}: candidate root must be an object")
    return value, retrieval.sha256_bytes(raw)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-empty, trimmed text")
    return value


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    items = tuple(_text(item, f"{label} item") for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{label} contains duplicates")
    return items


def _canonical_options(
    question: Mapping[str, Any], kind: str, label: str
) -> tuple[tuple[str, str], ...]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        if not criteria:
            raise ValueError(f"{label}.criteria must not be empty")
        pairs = tuple(
            sorted(
                (
                    _text(option_id, f"{label} option id"),
                    _text(description, f"{label}.criteria[{option_id!r}]"),
                )
                for option_id, description in criteria.items()
            )
        )
        if kind == "ntsb":
            declared = _string_list(question.get("options"), f"{label}.options")
            if set(declared) != {option_id for option_id, _ in pairs}:
                raise ValueError(f"{label}.options must exactly match criteria keys")
        return pairs
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{label}.criteria must be a non-empty object or list")
    descriptions = tuple(_text(item, f"{label}.criteria item") for item in criteria)
    if len(set(descriptions)) != len(descriptions):
        raise ValueError(f"{label}.criteria contains duplicates")
    if kind == "ntsb":
        option_ids = _string_list(question.get("options"), f"{label}.options")
        if len(option_ids) != len(descriptions):
            raise ValueError(f"{label}.options and criteria length differ")
        return tuple(sorted(zip(option_ids, descriptions, strict=True)))
    # GAO score-band strings are themselves the stable semantic option IDs.
    return tuple(sorted((description, description) for description in descriptions))


def _parse_case(raw: Any, kind: str, candidate_label: str) -> Case:
    if not isinstance(raw, dict):
        raise TypeError(f"{candidate_label} case must be an object")
    case_ref = _text(raw.get("case_ref"), f"{candidate_label}.case_ref")
    chunks = (
        raw.get("retrievable_chunks")
        if kind == "ntsb"
        else raw.get("retrieval_plan", {}).get("chunks")
    )
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{case_ref}: retrievable chunks must be a non-empty list")
    documents: list[Document] = []
    for index, chunk in enumerate(chunks):
        label = f"{case_ref}.chunks[{index}]"
        if not isinstance(chunk, dict):
            raise TypeError(f"{label} must be an object")
        chunk_ref = _text(chunk.get("chunk_ref"), f"{label}.chunk_ref")
        anchor = chunk.get("source_anchor") if kind == "ntsb" else chunk.get("anchor")
        if not isinstance(anchor, dict):
            raise TypeError(f"{label} source anchor must be an object")
        phrase_key = "verbatim_short_phrase" if kind == "ntsb" else "verbatim_phrase"
        phrase = _text(anchor.get(phrase_key), f"{label}.{phrase_key}")
        documents.append(
            Document(
                case_ref=case_ref,
                chunk_ref=chunk_ref,
                text=phrase,
                source_anchor=dict(sorted(anchor.items())),
            )
        )
    chunk_refs = tuple(document.chunk_ref for document in documents)
    if len(set(chunk_refs)) != len(chunk_refs):
        raise ValueError(f"{case_ref}: duplicate chunk_ref")

    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(f"{case_ref}: questions must be a non-empty list")
    questions: list[Question] = []
    for index, question in enumerate(raw_questions):
        label = f"{case_ref}.questions[{index}]"
        if not isinstance(question, dict):
            raise TypeError(f"{label} must be an object")
        question_ref = _text(question.get("question_ref"), f"{label}.question_ref")
        raw_type = question.get("primitive") if kind == "ntsb" else question.get("type")
        decision_type = _text(raw_type, f"{label}.decision_type")
        decision_type = decision_type[0].upper() + decision_type[1:].lower()
        if decision_type not in DECISION_TYPES:
            raise ValueError(f"{label}: unsupported decision type {decision_type!r}")
        decisive = _string_list(
            question.get("decisive_chunk_refs"), f"{label}.decisive_chunk_refs"
        )
        distractors = _string_list(
            question.get("known_distractor_chunk_refs"),
            f"{label}.known_distractor_chunk_refs",
        )
        if set(decisive) & set(distractors):
            raise ValueError(f"{label}: decisive and distractor chunks overlap")
        if set(decisive) | set(distractors) != set(chunk_refs):
            raise ValueError(
                f"{label}: decisive and distractor chunks must exactly partition the corpus"
            )
        policy_role = question.get("policy_role", decision_type.casefold())
        questions.append(
            Question(
                case_ref=case_ref,
                question_ref=question_ref,
                decision_type=decision_type,
                policy_role=_text(policy_role, f"{label}.policy_role"),
                instructions=_text(
                    question.get("instructions"), f"{label}.instructions"
                ),
                options=_canonical_options(question, kind, label),
                decisive=tuple(sorted(decisive)),
                distractors=tuple(sorted(distractors)),
            )
        )
    question_refs = tuple(question.question_ref for question in questions)
    if len(set(question_refs)) != len(question_refs):
        raise ValueError(f"{case_ref}: duplicate question_ref")
    if {question.decision_type for question in questions} != DECISION_TYPES:
        raise ValueError(
            f"{case_ref}: candidate case must cover Choice, Noul, and Score"
        )
    return Case(
        case_ref=case_ref,
        source_kind=kind,
        documents=tuple(sorted(documents, key=lambda item: item.chunk_ref)),
        questions=tuple(sorted(questions, key=lambda item: item.question_ref)),
    )


def load_candidates(
    ntsb_path: Path, gao_path: Path
) -> tuple[tuple[Case, ...], dict[str, str]]:
    cases: list[Case] = []
    bindings: dict[str, str] = {}
    for kind, path in (("ntsb", ntsb_path), ("gao", gao_path)):
        value, input_sha256 = _json_no_duplicates(path)
        if value.get("schema_version") != EXPECTED_CANDIDATE_SCHEMAS[kind]:
            raise ValueError(f"{path}: unexpected {kind.upper()} candidate schema")
        raw_cases = value.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"{path}: cases must be a non-empty list")
        cases.extend(
            _parse_case(case, kind, f"{path.name}.cases[{index}]")
            for index, case in enumerate(raw_cases)
        )
        bindings[f"{kind}_candidate_sha256"] = input_sha256
    case_refs = tuple(case.case_ref for case in cases)
    if len(set(case_refs)) != len(case_refs):
        raise ValueError("candidate files contain duplicate case_ref values")
    return tuple(sorted(cases, key=lambda item: item.case_ref)), bindings


def query_variants(question: Question) -> dict[str, tuple[dict[str, str], ...]]:
    option_text = " | ".join(
        f"{option_id}: {description}" for option_id, description in question.options
    )
    base = question.instructions

    def item(text: str, option_id: str | None = None) -> dict[str, str]:
        result = {"text": text, "sha256": retrieval.sha256_bytes(text.encode("utf-8"))}
        if option_id is not None:
            result["option_id"] = option_id
        return result

    return {
        "question_only": (item(f"task: question answering | query: {base}"),),
        "all": (
            item(f"task: question answering | query: {base}\noptions: {option_text}"),
        ),
        "per_option": tuple(
            item(
                f"task: question answering | query: {base}\noption: {option_id}: {description}",
                option_id,
            )
            for option_id, description in question.options
        ),
        "fact_all": (
            item(f"task: fact checking | query: {base}\noptions: {option_text}"),
        ),
    }


def _rank(scores: Mapping[str, float]) -> list[str]:
    return [
        key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _fused_dense_scores(
    queries: Sequence[Mapping[str, str]],
    documents: Sequence[Document],
    vectors: Mapping[str, np.ndarray],
    dimension: int,
) -> dict[str, float]:
    document_vectors = {
        document.chunk_ref: retrieval.normalize_mrl(
            vectors[document.embedding_text], dimension
        )
        for document in documents
    }
    scores = {document.chunk_ref: -math.inf for document in documents}
    for query in queries:
        query_vector = retrieval.normalize_mrl(vectors[query["text"]], dimension)
        for chunk_ref, document_vector in document_vectors.items():
            score = float(np.dot(query_vector, document_vector))
            if not math.isfinite(score):
                raise ValueError("dense similarity must be finite")
            scores[chunk_ref] = max(scores[chunk_ref], score)
    return scores


def _production_bm25(
    question: Question, documents: Sequence[Document]
) -> tuple[dict[str, Any], list[str], dict[str, float]]:
    production_question = {
        "text": question.instructions,
        "decision_type": question.decision_type,
        "policy_role": question.policy_role,
        "option_ids": [option_id for option_id, _ in question.options],
        "options": [description for _, description in question.options],
    }
    candidates = [
        {"chunk_key": document.chunk_ref, "text": document.text}
        for document in documents
    ]
    ranked = retrieval.bm25_rank(production_question, candidates)
    return production_question, [item[0] for item in ranked], dict(sorted(ranked))


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("metrics require at least one question")
    decisive_items = sum(len(record["decisive"]) for record in records)
    result: dict[str, Any] = {
        "questions": len(records),
        "decisive_items": decisive_items,
    }
    for k in TOP_KS:
        complete = 0
        recalled = 0
        for record in records:
            selected = set(record["ranking"][:k])
            decisive = set(record["decisive"])
            complete += decisive.issubset(selected)
            recalled += len(decisive & selected)
        result[f"complete_recall_at_{k}"] = complete / len(records)
        result[f"item_recall_at_{k}"] = recalled / decisive_items
    reciprocal_ranks = []
    for record in records:
        decisive = set(record["decisive"])
        rank = next(
            (
                index
                for index, chunk_ref in enumerate(record["ranking"], start=1)
                if chunk_ref in decisive
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    result["mrr_first_decisive"] = sum(reciprocal_ranks) / len(records)
    return result


def _paired_at_4(
    dense_records: Sequence[Mapping[str, Any]],
    bm25_records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if len(dense_records) != len(bm25_records):
        raise ValueError("paired comparison inputs differ in length")
    counts = {
        "bm25_only_complete_at_4": 0,
        "both_complete_at_4": 0,
        "embedding_only_complete_at_4": 0,
        "neither_complete_at_4": 0,
    }
    for dense, bm25 in zip(dense_records, bm25_records, strict=True):
        decisive = set(dense["decisive"])
        if decisive != set(bm25["decisive"]):
            raise ValueError("paired comparison decisive sets differ")
        dense_hit = decisive.issubset(set(dense["ranking"][:4]))
        bm25_hit = decisive.issubset(set(bm25["ranking"][:4]))
        key = {
            (False, False): "neither_complete_at_4",
            (False, True): "bm25_only_complete_at_4",
            (True, False): "embedding_only_complete_at_4",
            (True, True): "both_complete_at_4",
        }[(dense_hit, bm25_hit)]
        counts[key] += 1
    return counts


def build_receipt(
    cases: Sequence[Case],
    candidate_bindings: Mapping[str, str],
    provider: TextEmbeddingProvider,
    artifact_bindings: Mapping[str, str],
) -> dict[str, Any]:
    query_metadata: dict[tuple[str, str], dict[str, tuple[dict[str, str], ...]]] = {}
    embedding_inputs: set[str] = set()
    input_roles: dict[str, set[str]] = {}
    for case in cases:
        for document in case.documents:
            embedding_inputs.add(document.embedding_text)
            input_roles.setdefault(document.embedding_text, set()).add(
                f"document:{case.case_ref}:{document.chunk_ref}"
            )
        for question in case.questions:
            variants = query_variants(question)
            query_metadata[(case.case_ref, question.question_ref)] = variants
            for mode in MODES:
                for item in variants[mode]:
                    text = item["text"]
                    embedding_inputs.add(text)
                    role = f"query:{case.case_ref}:{question.question_ref}:{mode}"
                    if "option_id" in item:
                        role += f":{item['option_id']}"
                    input_roles.setdefault(text, set()).add(role)
    ordered_inputs = sorted(embedding_inputs)
    raw_vectors = provider.embed(ordered_inputs)
    if len(raw_vectors) != len(ordered_inputs):
        raise ValueError("embedding provider returned the wrong vector count")
    vectors: dict[str, np.ndarray] = {}
    for text, vector in zip(ordered_inputs, raw_vectors, strict=True):
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (768,) or not np.isfinite(array).all():
            raise ValueError("embedding provider must return finite 768d vectors")
        vectors[text] = array

    vector_manifest = []
    for text in ordered_inputs:
        text_sha256 = retrieval.sha256_bytes(text.encode("utf-8"))
        vector_manifest.append(
            {
                "cache_key": f"r052-dev:{text_sha256}",
                "input_sha256": text_sha256,
                "roles": sorted(input_roles[text]),
                "scoring_vector_sha256": {
                    str(dimension): retrieval.sha256_json(
                        retrieval.normalize_mrl(vectors[text], dimension).tolist()
                    )
                    for dimension in DIMENSIONS
                },
            }
        )

    rows: list[dict[str, Any]] = []
    metric_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(dimension): {mode: [] for mode in MODES} for dimension in DIMENSIONS
    }
    bm25_records: list[dict[str, Any]] = []
    for case in cases:
        for question in case.questions:
            variants = query_metadata[(case.case_ref, question.question_ref)]
            dense_rankings: dict[str, dict[str, list[str]]] = {}
            dense_scores: dict[str, dict[str, dict[str, float]]] = {}
            for mode in MODES:
                dense_rankings[mode] = {}
                dense_scores[mode] = {}
                for dimension in DIMENSIONS:
                    scores = _fused_dense_scores(
                        variants[mode], case.documents, vectors, dimension
                    )
                    ranking = _rank(scores)
                    dense_rankings[mode][str(dimension)] = ranking
                    dense_scores[mode][str(dimension)] = dict(sorted(scores.items()))
                    metric_records[str(dimension)][mode].append(
                        {"decisive": question.decisive, "ranking": ranking}
                    )
            bm25_question, bm25_ranking, bm25_scores = _production_bm25(
                question, case.documents
            )
            bm25_records.append(
                {"decisive": question.decisive, "ranking": bm25_ranking}
            )
            rows.append(
                {
                    "case_ref": case.case_ref,
                    "source_kind": case.source_kind,
                    "question_ref": question.question_ref,
                    "decision_type": question.decision_type,
                    "policy_role": question.policy_role,
                    "option_ids": [option_id for option_id, _ in question.options],
                    "decisive": list(question.decisive),
                    "distractors": list(question.distractors),
                    "queries": {mode: list(variants[mode]) for mode in MODES},
                    "embedding": {
                        "rankings": dense_rankings,
                        "scores": dense_scores,
                    },
                    "bm25": {
                        "production_question": bm25_question,
                        "query_tokens": retrieval.tokenize(
                            " ".join(
                                [
                                    bm25_question["text"],
                                    bm25_question["decision_type"],
                                    bm25_question["policy_role"],
                                    *[
                                        value
                                        for pair in sorted(
                                            zip(
                                                bm25_question["option_ids"],
                                                bm25_question["options"],
                                                strict=True,
                                            )
                                        )
                                        for value in pair
                                    ],
                                ]
                            )
                        ),
                        "ranking": bm25_ranking,
                        "scores": bm25_scores,
                    },
                }
            )

    embedding_summary = {
        dimension: {mode: _metrics(metric_records[dimension][mode]) for mode in MODES}
        for dimension in sorted(metric_records)
    }
    bm25_summary = _metrics(bm25_records)
    paired = {
        dimension: {
            mode: _paired_at_4(metric_records[dimension][mode], bm25_records)
            for mode in MODES
        }
        for dimension in sorted(metric_records)
    }
    documents = [
        {
            "case_ref": case.case_ref,
            "source_kind": case.source_kind,
            "chunk_ref": document.chunk_ref,
            "text": document.text,
            "text_sha256": retrieval.sha256_bytes(document.text.encode("utf-8")),
            "embedding_text": document.embedding_text,
            "embedding_text_sha256": retrieval.sha256_bytes(
                document.embedding_text.encode("utf-8")
            ),
            "source_anchor": document.source_anchor,
        }
        for case in cases
        for document in case.documents
    ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "claim_status": CLAIM_STATUS,
        "frozen_evaluation_panel": False,
        "limitations": [
            "Candidate anchors are not human-reviewed source-derived full chunks.",
            "English only; there are no Hebrew twins or image controls.",
            "This development evidence cannot select or tune the frozen B1 panel.",
        ],
        "input_candidate_sha256": dict(sorted(candidate_bindings.items())),
        "artifact_bindings": dict(sorted(artifact_bindings.items())),
        "code_bindings": {
            "pilot_runner_sha256": retrieval.file_sha256(Path(__file__).resolve()),
            "production_scorer_sha256": retrieval.file_sha256(
                Path(retrieval.__file__).resolve()
            ),
        },
        "provider": dict(sorted(provider.identity().items())),
        "embedding_input_vectors": vector_manifest,
        "embedding_input_vectors_sha256": retrieval.sha256_json(vector_manifest),
        "corpus_contract": "exact short source-anchor phrases only; no summaries",
        "query_contract": (
            "canonical semantic option_id ordering; exact serialized query text; "
            "per-option maximum-score fusion"
        ),
        "ranking_contract": (
            "four embedding modes use cosine at 768d and truncated-renormalized "
            "256d; all are paired at identical final top-k against one fixed "
            "production bm25_rank baseline; chunk_ref tie break"
        ),
        "config": {
            "modes": list(MODES),
            "dimensions": list(DIMENSIONS),
            "top_ks": list(TOP_KS),
            "local_files_only": True,
        },
        "cases": len(cases),
        "questions": sum(len(case.questions) for case in cases),
        "documents": documents,
        "rows": rows,
        "summary": {"embedding": embedding_summary, "bm25": bm25_summary},
        "paired_complete_recall_at_4": paired,
    }
    return {**body, "receipt_sha256": retrieval.sha256_json(body)}


def _exclusive_publish(path: Path, data: bytes) -> None:
    path = path.resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
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
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_and_publish(
    *,
    ntsb_candidates: Path,
    gao_candidates: Path,
    output: Path,
    runtime_sidecar: Path,
    provider: TextEmbeddingProvider,
    artifact_bindings: Mapping[str, str],
) -> dict[str, Any]:
    if output.resolve() == runtime_sidecar.resolve():
        raise ValueError("receipt and runtime sidecar paths must differ")
    if output.exists() or runtime_sidecar.exists():
        raise FileExistsError("receipt and runtime outputs are no-overwrite artifacts")
    started = time.time()
    started_monotonic = time.perf_counter()
    cases, candidate_bindings = load_candidates(ntsb_candidates, gao_candidates)
    receipt = build_receipt(cases, candidate_bindings, provider, artifact_bindings)
    receipt_bytes = retrieval.canonical_json(receipt)
    elapsed = time.perf_counter() - started_monotonic
    runtime = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "deterministic_receipt_file_sha256": retrieval.sha256_bytes(receipt_bytes),
        "started_unix_seconds": started,
        "elapsed_seconds": elapsed,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **dict(provider.runtime_identity()),
        },
    }
    runtime_bytes = retrieval.canonical_json(runtime)
    _exclusive_publish(output, receipt_bytes)
    try:
        _exclusive_publish(runtime_sidecar, runtime_bytes)
    except Exception:
        # Roll back only the receipt bytes created by this call.  The initial
        # no-overwrite check and exact hash prevent deletion of unrelated data.
        if output.is_file() and retrieval.file_sha256(output) == retrieval.sha256_bytes(
            receipt_bytes
        ):
            output.unlink()
        raise
    return receipt


class LocalEmbeddingGemma2TextProvider:
    """Adapter around the production, cache-only EmbeddingGemma 2 provider."""

    def __init__(
        self,
        *,
        model_snapshot: Path,
        model_snapshot_tree_sha256: str,
        runtime_lock: Path,
        runtime_lock_sha256: str,
        transformers_wheel: Path,
        dtype: str,
        device: str,
    ) -> None:
        observed_tree = retrieval.artifact_sha256(model_snapshot.resolve())
        if observed_tree != model_snapshot_tree_sha256:
            raise ValueError("model snapshot does not match external tree sha256")
        retrieval.verify_frozen_snapshot(model_snapshot.resolve())
        self._provider = retrieval.LocalEmbeddingGemma2Provider(
            dtype=dtype,
            device=device,
            truncate_dim=None,
            runtime_lock=runtime_lock,
            runtime_lock_sha256=runtime_lock_sha256,
            transformers_wheel=transformers_wheel,
        )
        if self._provider.snapshot_hash != observed_tree:
            raise ValueError("locally loaded model differs from bound snapshot")
        self._identity = {
            key: value
            for key, value in self._provider.identity().items()
            if key != "runtime_identity"
        }

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [
            self._provider._embed(
                text,
                None,
                "r052-dev:" + retrieval.sha256_bytes(text.encode("utf-8")),
            )
            for text in texts
        ]

    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def runtime_identity(self) -> Mapping[str, Any]:
        return dict(self._provider.identity()["runtime_identity"])


def _sha(value: str, label: str) -> str:
    if not retrieval.SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"{label} must be a lowercase sha256")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntsb-candidates", type=Path, required=True)
    parser.add_argument("--gao-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-sidecar", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-snapshot-tree-sha256", required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--transformers-wheel", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    model_tree_sha256 = _sha(
        args.model_snapshot_tree_sha256, "model snapshot tree sha256"
    )
    runtime_lock_sha256 = _sha(args.runtime_lock_sha256, "runtime lock sha256")
    if args.output.exists() or args.runtime_sidecar.exists():
        parser.error("output artifacts already exist; overwrite is forbidden")
    provider = LocalEmbeddingGemma2TextProvider(
        model_snapshot=args.model_snapshot,
        model_snapshot_tree_sha256=model_tree_sha256,
        runtime_lock=args.runtime_lock,
        runtime_lock_sha256=runtime_lock_sha256,
        transformers_wheel=args.transformers_wheel,
        dtype=args.dtype,
        device=args.device,
    )
    artifact_bindings = {
        "model_snapshot_tree_sha256": model_tree_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "transformers_wheel_sha256": retrieval.verify_artifact(
            args.transformers_wheel,
            retrieval.TRANSFORMERS_WHEEL_SHA256,
            "Transformers wheel",
        ),
    }
    run_and_publish(
        ntsb_candidates=args.ntsb_candidates,
        gao_candidates=args.gao_candidates,
        output=args.output,
        runtime_sidecar=args.runtime_sidecar,
        provider=provider,
        artifact_bindings=artifact_bindings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
