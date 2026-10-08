from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_r052_embedding_dev_pilot as pilot
from scripts import score_r052_evidence_retrieval as retrieval


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            raw = hashlib.shake_256(text.encode()).digest(768)
            vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 127.5
            vectors.append(vector.tolist())
        return vectors

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "fake_deterministic_test_provider",
            "dimension": 768,
            "local_files_only": True,
        }

    def runtime_identity(self) -> dict[str, Any]:
        return {"fake_device": "cpu"}


def _questions(kind: str) -> list[dict[str, Any]]:
    base = [
        {
            "question_ref": "q-choice",
            "instructions": "Which statement is supported?",
            "criteria": {"option_b": "Beta claim", "option_a": "Alpha claim"},
            "decisive_chunk_refs": ["chunk-b"],
            "known_distractor_chunk_refs": ["chunk-a", "chunk-c", "chunk-d"],
        },
        {
            "question_ref": "q-noul",
            "instructions": "Is the statement established?",
            "criteria": {"yes": "Established", "no": "Not established"},
            "decisive_chunk_refs": ["chunk-a", "chunk-c"],
            "known_distractor_chunk_refs": ["chunk-b", "chunk-d"],
        },
        {
            "question_ref": "q-score",
            "instructions": "Select the supported severity band.",
            "criteria": ["Low", "High"],
            "decisive_chunk_refs": ["chunk-d"],
            "known_distractor_chunk_refs": ["chunk-a", "chunk-b", "chunk-c"],
        },
    ]
    if kind == "ntsb":
        for question, primitive, options in zip(
            base,
            ("Choice", "Noul", "Score"),
            (("option_b", "option_a"), ("yes", "no"), ("0", "1")),
            strict=True,
        ):
            question["primitive"] = primitive
            question["type"] = primitive.casefold()
            question["options"] = list(options)
    else:
        for question, decision_type in zip(
            base, ("Choice", "Noul", "Score"), strict=True
        ):
            question["type"] = decision_type
            question["policy_role"] = f"role-{decision_type.casefold()}"
    return base


def _candidate(kind: str) -> dict[str, Any]:
    phrases = {
        "chunk-a": "alpha evidence",
        "chunk-b": "beta decisive evidence",
        "chunk-c": "gamma corroborating evidence",
        "chunk-d": "delta severity evidence",
    }
    if kind == "ntsb":
        chunks = [
            {
                "chunk_ref": chunk_ref,
                "source_anchor": {
                    "verbatim_short_phrase": phrase,
                    "char_start_0based": index * 100,
                },
            }
            for index, (chunk_ref, phrase) in enumerate(phrases.items())
        ]
        case = {
            "case_ref": "ntsb-case",
            "retrievable_chunks": chunks,
            "questions": _questions(kind),
        }
    else:
        chunks = [
            {
                "chunk_ref": chunk_ref,
                "anchor": {
                    "verbatim_phrase": phrase,
                    "char_start": index * 100,
                },
            }
            for index, (chunk_ref, phrase) in enumerate(phrases.items())
        ]
        case = {
            "case_ref": "gao-case",
            "retrieval_plan": {"chunks": chunks},
            "questions": _questions(kind),
        }
    return {
        "schema_version": pilot.EXPECTED_CANDIDATE_SCHEMAS[kind],
        "cases": [case],
    }


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return path


@pytest.fixture
def candidate_paths(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_json(tmp_path / "ntsb.json", _candidate("ntsb")),
        _write_json(tmp_path / "gao.json", _candidate("gao")),
    )


def _run(
    tmp_path: Path,
    candidate_paths: tuple[Path, Path],
    *,
    stem: str = "pilot",
    provider: FakeProvider | None = None,
) -> tuple[dict[str, Any], Path, Path, FakeProvider]:
    output = tmp_path / f"{stem}.json"
    runtime = tmp_path / f"{stem}.runtime.json"
    actual_provider = provider or FakeProvider()
    receipt = pilot.run_and_publish(
        ntsb_candidates=candidate_paths[0],
        gao_candidates=candidate_paths[1],
        output=output,
        runtime_sidecar=runtime,
        provider=actual_provider,
        artifact_bindings={
            "model_snapshot_tree_sha256": "1" * 64,
            "runtime_lock_sha256": "2" * 64,
            "transformers_wheel_sha256": "3" * 64,
        },
    )
    return receipt, output, runtime, actual_provider


def test_reproducible_receipt_hashes_counts_metrics_and_runtime_separation(
    tmp_path: Path, candidate_paths: tuple[Path, Path]
) -> None:
    receipt, output, runtime_path, provider = _run(tmp_path, candidate_paths)

    raw = output.read_bytes()
    assert raw == retrieval.canonical_json(receipt)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == retrieval.sha256_json(body)
    assert receipt["schema_version"] == pilot.SCHEMA_VERSION
    assert receipt["claim_status"] == pilot.CLAIM_STATUS
    assert receipt["frozen_evaluation_panel"] is False
    assert receipt["cases"] == 2
    assert receipt["questions"] == 6
    assert len(receipt["documents"]) == 8
    assert len(receipt["rows"]) == 6
    vector_manifest = receipt["embedding_input_vectors"]
    assert receipt["embedding_input_vectors_sha256"] == retrieval.sha256_json(
        vector_manifest
    )
    assert len(vector_manifest) == len(
        {item["input_sha256"] for item in vector_manifest}
    )
    assert all(item["cache_key"].startswith("r052-dev:") for item in vector_manifest)
    assert all(
        set(item["scoring_vector_sha256"]) == {"256", "768"} for item in vector_manifest
    )
    assert receipt["input_candidate_sha256"] == {
        "gao_candidate_sha256": retrieval.file_sha256(candidate_paths[1]),
        "ntsb_candidate_sha256": retrieval.file_sha256(candidate_paths[0]),
    }
    assert list(receipt["summary"]["embedding"]) == ["256", "768"]
    for dimension in ("256", "768"):
        for mode in pilot.MODES:
            metrics = receipt["summary"]["embedding"][dimension][mode]
            assert metrics["questions"] == 6
            assert metrics["decisive_items"] == 8
            assert 0 <= metrics["complete_recall_at_4"] <= 1
            assert (
                sum(receipt["paired_complete_recall_at_4"][dimension][mode].values())
                == 6
            )
    assert len(provider.calls) == 1
    assert provider.calls[0] == sorted(set(provider.calls[0]))

    runtime = json.loads(runtime_path.read_bytes())
    assert (
        runtime["deterministic_receipt_file_sha256"] == hashlib.sha256(raw).hexdigest()
    )
    assert "elapsed_seconds" in runtime
    assert "started_unix_seconds" in runtime
    assert "elapsed_seconds" not in receipt
    assert "started_unix_seconds" not in receipt


def test_query_metadata_uses_canonical_option_ids_and_per_option_fusion(
    tmp_path: Path, candidate_paths: tuple[Path, Path]
) -> None:
    receipt, _, _, _ = _run(tmp_path, candidate_paths)
    ntsb_choice = next(
        row
        for row in receipt["rows"]
        if row["case_ref"] == "ntsb-case" and row["question_ref"] == "q-choice"
    )
    assert ntsb_choice["option_ids"] == ["option_a", "option_b"]
    assert [query["option_id"] for query in ntsb_choice["queries"]["per_option"]] == [
        "option_a",
        "option_b",
    ]
    assert (
        "options: option_a: Alpha claim | option_b: Beta claim"
        in ntsb_choice["queries"]["all"][0]["text"]
    )
    assert ntsb_choice["queries"]["fact_all"][0]["text"].startswith(
        "task: fact checking | query:"
    )
    for query in (
        item for mode in pilot.MODES for item in ntsb_choice["queries"][mode]
    ):
        assert query["sha256"] == hashlib.sha256(query["text"].encode()).hexdigest()
    for mode in pilot.MODES:
        assert len(ntsb_choice["embedding"]["rankings"][mode]["768"]) == 4
    assert len(ntsb_choice["bm25"]["ranking"]) == 4


def test_repeated_fake_provider_runs_have_identical_deterministic_receipts(
    tmp_path: Path, candidate_paths: tuple[Path, Path]
) -> None:
    first, first_path, _, _ = _run(tmp_path, candidate_paths, stem="first-repeat")
    second, second_path, _, _ = _run(tmp_path, candidate_paths, stem="second-repeat")
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()


def test_bm25_is_the_production_formula_on_the_exact_query_and_anchor_corpus(
    tmp_path: Path, candidate_paths: tuple[Path, Path]
) -> None:
    receipt, _, _, _ = _run(tmp_path, candidate_paths)
    row = receipt["rows"][0]
    documents = [
        document
        for document in receipt["documents"]
        if document["case_ref"] == row["case_ref"]
    ]
    question = row["bm25"]["production_question"]
    expected = retrieval.bm25_rank(
        question,
        [
            {"chunk_key": document["chunk_ref"], "text": document["text"]}
            for document in documents
        ],
    )
    assert row["bm25"]["ranking"] == [item[0] for item in expected]
    assert row["bm25"]["scores"] == dict(sorted(expected))


def test_semantic_result_is_invariant_to_candidate_and_option_order(
    tmp_path: Path, candidate_paths: tuple[Path, Path]
) -> None:
    first, _, _, _ = _run(tmp_path, candidate_paths, stem="first")
    shuffled_paths = []
    for kind, original in zip(("ntsb", "gao"), candidate_paths, strict=True):
        candidate = json.loads(original.read_text())
        case = candidate["cases"][0]
        chunks_key = "retrievable_chunks" if kind == "ntsb" else "chunks"
        container = case if kind == "ntsb" else case["retrieval_plan"]
        container[chunks_key].reverse()
        case["questions"].reverse()
        for question in case["questions"]:
            if isinstance(question["criteria"], dict):
                question["criteria"] = dict(
                    reversed(list(question["criteria"].items()))
                )
            elif "options" in question:
                question["criteria"].reverse()
            if "options" in question:
                question["options"].reverse()
        shuffled_paths.append(
            _write_json(tmp_path / f"{kind}-shuffled.json", candidate)
        )
    second, _, _, _ = _run(
        tmp_path,
        (shuffled_paths[0], shuffled_paths[1]),
        stem="second",
    )

    def semantic(receipt: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(receipt)
        result.pop("receipt_sha256")
        result.pop("input_candidate_sha256")
        return result

    assert semantic(first) == semantic(second)


def test_outputs_are_no_overwrite_and_provider_is_not_called(
    tmp_path: Path, candidate_paths: tuple[Path, Path]
) -> None:
    output = tmp_path / "exists.json"
    output.write_text("owned")
    provider = FakeProvider()
    with pytest.raises(FileExistsError, match="no-overwrite"):
        pilot.run_and_publish(
            ntsb_candidates=candidate_paths[0],
            gao_candidates=candidate_paths[1],
            output=output,
            runtime_sidecar=tmp_path / "runtime.json",
            provider=provider,
            artifact_bindings={},
        )
    assert output.read_text() == "owned"
    assert provider.calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(schema_version="wrong"),
            "unexpected NTSB candidate schema",
        ),
        (
            lambda value: value["cases"][0]["retrievable_chunks"][0][
                "source_anchor"
            ].update(verbatim_short_phrase=""),
            "must be non-empty",
        ),
        (
            lambda value: value["cases"][0]["questions"][0].update(
                options=["option_a", "unknown"]
            ),
            "must exactly match criteria keys",
        ),
        (
            lambda value: value["cases"][0]["questions"][0].update(
                decisive_chunk_refs=["chunk-b", "missing"]
            ),
            "exactly partition",
        ),
        (
            lambda value: value["cases"][0]["questions"].pop(),
            "cover Choice, Noul, and Score",
        ),
    ],
)
def test_rejects_malformed_candidates(
    tmp_path: Path,
    candidate_paths: tuple[Path, Path],
    mutation: Any,
    message: str,
) -> None:
    malformed = _candidate("ntsb")
    mutation(malformed)
    malformed_path = _write_json(tmp_path / "malformed.json", malformed)
    with pytest.raises((TypeError, ValueError), match=message):
        pilot.load_candidates(malformed_path, candidate_paths[1])


def test_rejects_bad_provider_dimension_without_publishing(
    tmp_path: Path, candidate_paths: tuple[Path, Path]
) -> None:
    class BadProvider(FakeProvider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0] for _ in texts]

    output = tmp_path / "bad.json"
    runtime = tmp_path / "bad.runtime.json"
    with pytest.raises(ValueError, match="finite 768d"):
        pilot.run_and_publish(
            ntsb_candidates=candidate_paths[0],
            gao_candidates=candidate_paths[1],
            output=output,
            runtime_sidecar=runtime,
            provider=BadProvider(),
            artifact_bindings={},
        )
    assert not output.exists()
    assert not runtime.exists()
