import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import score_r052_evidence_retrieval as scorer

TEST_ARTIFACT = Path(__file__).resolve()
TEST_ARTIFACT_SHA256 = scorer.file_sha256(TEST_ARTIFACT)


def _sha(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return scorer.sha256_bytes(value)


def _media(value: bytes) -> dict:
    import base64
    import io

    from PIL import Image

    stream = io.BytesIO()
    color = tuple((value + b"\x00\x00\x00")[:3])
    Image.new("RGB", (2, 2), color).save(stream, format="PNG")
    value = stream.getvalue()

    return {
        "mime_type": "image/png",
        "data_base64": base64.b64encode(value).decode(),
        "sha256": _sha(value),
    }


def _chunk(
    key: str,
    kind: str,
    text: str,
    anchor: str,
    *,
    media: dict | None = None,
    control: dict | None = None,
) -> dict:
    chunk = {
        "chunk_key": key,
        "kind": kind,
        "modality": "image" if control is not None else "text",
        "text": text,
        "media": media,
        "immutable": kind == "global_policy",
        "twin_anchor": anchor,
        "image_control": control,
        "content_sha256": "",
        "token_start": 0,
        "token_end": 1,
    }
    chunk["content_sha256"] = scorer.sha256_json(scorer.chunk_hash_payload(chunk))
    return chunk


def _question(
    key: str,
    text: str,
    anchor: str,
    decisive: list[str],
    distractors: list[str],
) -> dict:
    full_outcome = {"authorization": "allow"}
    removed_outcome = {"authorization": "deny"}
    question = {
        "question_key": key,
        "text": text,
        "options": ["allow", "deny"],
        "policy_role": "authorization",
        "twin_anchor": anchor,
        "decisive_chunk_keys": decisive,
        "known_distractor_chunk_keys": distractors,
        "causal_proof": {
            "kind": "causal_annotation",
            "removed_chunk_keys": list(decisive),
            "full_context_sha256": _sha(key + "-full-context"),
            "removed_context_sha256": _sha(key + "-removed-context"),
            "full_outcome_sha256": scorer.sha256_json(full_outcome),
            "removed_outcome_sha256": scorer.sha256_json(removed_outcome),
            "full_outcome": full_outcome,
            "removed_outcome": removed_outcome,
            "changed_fields": ["authorization"],
        },
        "content_sha256": "",
    }
    question["content_sha256"] = scorer.sha256_json(
        scorer.question_hash_payload(question)
    )
    return question


def _case(language: str, twin_group: str) -> dict:
    suffix = language
    control_group = _sha("image-control-" + language)
    control = lambda variant: {"group_key": control_group, "variant": variant}
    translations = {
        "en": {
            "policy": "Always apply the authorization policy.",
            "evidence": "The signed approval code is cobalt.",
            "distractor": "The unsigned memo discusses lunch.",
            "image": "Inspect the pictured approval seal.",
            "q_image": "Does the image contain the valid approval seal?",
            "q_text": "Is the signed approval present?",
        },
        "he": {
            "policy": "יש להחיל תמיד את מדיניות ההרשאה.",
            "evidence": "קוד האישור החתום הוא קובלט.",
            "distractor": "המזכר הלא חתום דן בארוחת צהריים.",
            "image": "בדקו את חותמת האישור שבתמונה.",
            "q_image": "האם בתמונה מופיעה חותמת האישור התקפה?",
            "q_text": "האם קיים אישור חתום?",
        },
    }[language]
    chunks = [
        _chunk(f"policy-{suffix}", "global_policy", translations["policy"], "policy"),
        _chunk(
            f"evidence-{suffix}", "evidence", translations["evidence"], "text-evidence"
        ),
        _chunk(
            f"distractor-{suffix}",
            "distractor",
            translations["distractor"],
            "text-distractor",
        ),
        _chunk(
            f"real-{suffix}",
            "evidence",
            translations["image"],
            "image-real",
            media=_media(b"real-image"),
            control=control("real"),
        ),
        _chunk(
            f"shuffled-{suffix}",
            "distractor",
            translations["image"],
            "image-shuffled",
            media=_media(b"shuffled-image"),
            control=control("shuffled"),
        ),
        _chunk(
            f"missing-{suffix}",
            "distractor",
            translations["image"],
            "image-missing",
            control=control("missing"),
        ),
    ]
    boundaries = [0, 1365, 2730, 4096, 5461, 6826, 8192]
    for index, chunk in enumerate(chunks):
        chunk["token_start"] = boundaries[index]
        chunk["token_end"] = boundaries[index + 1]
        chunk["content_sha256"] = scorer.sha256_json(scorer.chunk_hash_payload(chunk))
    result = {
        "case_key": _sha("case-" + language),
        "language": language,
        "twin_group_key": twin_group,
        "source_family": "synthetic",
        "source_domain": "general",
        "context_tokens": 8192,
        "contamination_audit_sha256": _sha("contamination-audit-" + language),
        "chunks": chunks,
        "questions": [
            _question(
                f"q-image-{suffix}",
                translations["q_image"],
                "q-image",
                [f"real-{suffix}"],
                [
                    f"evidence-{suffix}",
                    f"distractor-{suffix}",
                    f"shuffled-{suffix}",
                    f"missing-{suffix}",
                ],
            ),
            _question(
                f"q-text-{suffix}",
                translations["q_text"],
                "q-text",
                [f"evidence-{suffix}"],
                [
                    f"distractor-{suffix}",
                    f"real-{suffix}",
                    f"shuffled-{suffix}",
                    f"missing-{suffix}",
                ],
            ),
        ],
    }
    for question in result["questions"]:
        question["causal_proof"]["full_context_sha256"] = scorer.causal_context_sha256(
            result, question, set()
        )
        question["causal_proof"]["removed_context_sha256"] = (
            scorer.causal_context_sha256(
                result, question, set(question["decisive_chunk_keys"])
            )
        )
    return result


def make_panel() -> dict:
    cases = [_case("en", _sha("twins")), _case("he", _sha("twins"))]
    query_vectors = {}
    document_vectors = {}
    for case in cases:
        for question in case["questions"]:
            query_vectors[question["content_sha256"]] = (
                [1.0, 0.0, 0.0, 0.0]
                if question["twin_anchor"] == "q-image"
                else [0.0, 1.0, 0.0, 0.0]
            )
        for chunk in case["chunks"]:
            vector = {
                "policy": [0.1, 0.1, 0.9, 0.0],
                "text-evidence": [0.0, 1.0, 0.0, 0.0],
                "text-distractor": [0.0, -1.0, 0.0, 0.0],
                "image-real": [1.0, 0.0, 0.0, 0.0],
                "image-shuffled": [-1.0, 0.0, 0.0, 0.0],
                "image-missing": [-0.5, 0.1, 0.0, 0.0],
            }[chunk["twin_anchor"]]
            document_vectors[chunk["content_sha256"]] = vector
    vectors = {"queries": query_vectors, "documents": document_vectors}
    provider = {
        "kind": "precomputed",
        "origin": "local_baseline",
        "hosted_api": False,
        "benchmark_derived": False,
        "model": "fixture-model",
        "revision": _sha("fixture-revision"),
        "generator_code_sha256": TEST_ARTIFACT_SHA256,
        "artifact_sha256": scorer.sha256_json(vectors),
        "dimension": 4,
        **vectors,
    }
    return {
        "schema_version": scorer.SCHEMA_VERSION,
        "provenance": {"kind": "independent_internal", "benchmark_derived": False},
        "cases": cases,
        "providers": {"fixture-dense": provider},
    }


def _causal_manifest(panel: dict) -> tuple[dict, str]:
    manifest = {
        "schema_version": scorer.CAUSAL_MANIFEST_VERSION,
        "annotation_builder_code_sha256": TEST_ARTIFACT_SHA256,
        "entries": {
            scorer.causal_entry_key(case, question): question["causal_proof"]
            for case in panel["cases"]
            for question in case["questions"]
        },
    }
    return manifest, scorer.sha256_json(manifest)


def _contamination_manifest(panel: dict) -> tuple[dict, str]:
    auditor_code = TEST_ARTIFACT_SHA256
    entries = {}
    for case in panel["cases"]:
        receipt = {
            "schema_version": scorer.AUDIT_RECEIPT_VERSION,
            "status": "pass",
            "algorithm": "exact-sha256+minhash-jaccard-v1",
            "auditor_code_sha256": auditor_code,
            "audited_corpus_roots_sha256": [TEST_ARTIFACT_SHA256],
            "exact_overlap_count": 0,
            "near_overlap_max_jaccard": 0.0,
            "tokenizer_receipt_sha256": TEST_ARTIFACT_SHA256,
            "chunker_receipt_sha256": TEST_ARTIFACT_SHA256,
        }
        receipt["receipt_sha256"] = scorer.sha256_json(receipt)
        case["contamination_audit_sha256"] = receipt["receipt_sha256"]
        entries[case["case_key"]] = {
            "case_content_sha256": scorer.case_content_sha256(case),
            "audit_receipt": receipt,
        }
    manifest = {
        "schema_version": scorer.CONTAMINATION_MANIFEST_VERSION,
        "auditor_code_sha256": auditor_code,
        "entries": entries,
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
            for name, provider in panel["providers"].items()
        },
    }
    return manifest, scorer.sha256_json(manifest)


def _validate(panel: dict) -> dict:
    causal, causal_digest = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    return scorer.validate_panel(
        panel,
        causal,
        causal_digest,
        contamination,
        contamination_digest,
        annotation_builder_artifact=TEST_ARTIFACT,
        contamination_auditor_artifact=TEST_ARTIFACT,
        contamination_corpus_artifacts=[TEST_ARTIFACT],
        contamination_tokenizer_artifact=TEST_ARTIFACT,
        contamination_chunker_artifact=TEST_ARTIFACT,
        provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
    )


@pytest.fixture
def panel() -> dict:
    return make_panel()


def test_dense_scoring_is_exact_deterministic_and_unions_every_question(panel):
    _validate(panel)
    provider = scorer.PrecomputedProvider(
        "fixture-dense", panel["providers"]["fixture-dense"], None
    )
    first = scorer.score_method(panel, "fixture-dense", (1, 2, 4), provider)
    second = scorer.score_method(panel, "fixture-dense", (1, 2, 4), provider)

    assert first == second
    assert first["overall"]["decisive_recall@1"] == 1.0
    assert first["overall"]["mrr"] == 1.0
    assert first["overall"]["distractor_fpr@1"] == 0.0
    assert first["by_language"]["en"]["decisive_recall@1"] == 1.0
    assert first["he_en_twin_agreement"]["top1_set_jaccard"] == 1.0
    assert first["he_en_twin_agreement"]["top1_position_agreement"] == 1.0
    assert first["image_margin"]["mean_real_minus_shuffled"] == pytest.approx(2.0)
    assert first["image_margin"]["positive_missing_fraction"] == 1.0
    for case in panel["cases"]:
        union = first["per_case_evidence_union_at_max_k"][case["case_key"]]
        assert f"policy-{case['language']}" in union
        assert f"real-{case['language']}" in union
        assert f"evidence-{case['language']}" in union


def test_evidence_union_is_selected_by_rank_but_reassembled_in_source_order(panel):
    _validate(panel)
    provider = scorer.PrecomputedProvider(
        "fixture-dense", panel["providers"]["fixture-dense"], None
    )
    result = scorer.score_method(panel, "fixture-dense", (1,), provider)
    for case in panel["cases"]:
        assert result["per_case_evidence_union_at_max_k"][case["case_key"]] == [
            f"policy-{case['language']}",
            f"evidence-{case['language']}",
            f"real-{case['language']}",
        ]
    assert result["union_contract"]["reassembled_in_original_case_order"] is True
    assert result["slices"]["context_length"]["long"]["decisive_recall@1"] == 1.0
    assert result["slices"]["decisive_modality"]["image"]["decisive_recall@1"] == 1.0


def test_bm25_is_unicode_deterministic_and_reports_controls(panel):
    _validate(panel)
    result = scorer.score_method(panel, "bm25", (1, 2, 4))
    assert result == scorer.score_method(panel, "bm25", (1, 2, 4))
    assert result["overall"]["questions"] == 4
    assert result["image_margin"]["controls"] == 2
    assert scorer.tokenize("אישור חתום") == ["אישור", "חתום"]


def test_mrl_truncation_always_renormalizes_and_rejects_bad_values():
    vector = np.arange(1, 769, dtype=float)
    result = scorer.normalize_mrl(vector, 256)
    assert result.shape == (256,)
    assert np.linalg.norm(result) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one of"):
        scorer.normalize_mrl(vector, 255)
    with pytest.raises(ValueError, match="finite"):
        scorer.normalize_mrl([1.0, float("nan")])
    with pytest.raises(ValueError, match="positive"):
        scorer.normalize_mrl([0.0, 0.0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["cases"][0]["chunks"][0].update(immutable=False),
            "global policy",
        ),
        (
            lambda value: value["cases"][0]["questions"][0][
                "decisive_chunk_keys"
            ].clear(),
            "decisive",
        ),
        (
            lambda value: value["cases"][0]["questions"][0]["causal_proof"][
                "removed_chunk_keys"
            ].clear(),
            "exactly bind",
        ),
        (
            lambda value: value["cases"][0]["questions"][0]["causal_proof"].update(
                removed_outcome_sha256=value["cases"][0]["questions"][0][
                    "causal_proof"
                ]["full_outcome_sha256"]
            ),
            "change the outcome",
        ),
        (
            lambda value: value["cases"][0]["chunks"][1].update(
                content_sha256="0" * 64
            ),
            "mismatch",
        ),
        (lambda value: value["cases"][1].update(twin_group_key=None), "Hebrew"),
        (
            lambda value: value["cases"][0]["chunks"][3]["image_control"].update(
                variant="missing"
            ),
            "variants",
        ),
    ],
)
def test_panel_guards_fail_closed(panel, mutation, message):
    changed = copy.deepcopy(panel)
    mutation(changed)
    with pytest.raises(ValueError, match=message):
        _validate(changed)


def test_panel_rejects_benchmark_lineage_paths_and_markers(panel):
    changed = copy.deepcopy(panel)
    changed["cases"][0]["questions"][0]["causal_proof"]["changed_fields"] = [
        "copied from JevBench"
    ]
    with pytest.raises(ValueError, match="lineage marker"):
        _validate(changed)


def test_external_causal_manifest_is_required_but_only_validates_annotations(panel):
    manifest, _ = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    changed_manifest = copy.deepcopy(manifest)
    first = next(iter(changed_manifest["entries"].values()))
    first["changed_fields"] = ["authorization", "unbound-field"]
    with pytest.raises(ValueError, match="manifest entries"):
        scorer.validate_panel(
            panel,
            changed_manifest,
            scorer.sha256_json(changed_manifest),
            contamination,
            contamination_digest,
            annotation_builder_artifact=TEST_ARTIFACT,
            contamination_auditor_artifact=TEST_ARTIFACT,
            contamination_corpus_artifacts=[TEST_ARTIFACT],
            contamination_tokenizer_artifact=TEST_ARTIFACT,
            contamination_chunker_artifact=TEST_ARTIFACT,
            provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
        )
    with pytest.raises(ValueError, match="external sha256 anchor"):
        scorer.validate_panel(
            panel,
            manifest,
            "0" * 64,
            contamination,
            contamination_digest,
            annotation_builder_artifact=TEST_ARTIFACT,
            contamination_auditor_artifact=TEST_ARTIFACT,
            contamination_corpus_artifacts=[TEST_ARTIFACT],
            contamination_tokenizer_artifact=TEST_ARTIFACT,
            contamination_chunker_artifact=TEST_ARTIFACT,
            provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
        )


def test_causal_context_hashes_are_recomputed_from_ordered_content(panel):
    changed = copy.deepcopy(panel)
    chunk = changed["cases"][0]["chunks"][1]
    chunk["text"] += " changed"
    chunk["content_sha256"] = scorer.sha256_json(scorer.chunk_hash_payload(chunk))
    with pytest.raises(ValueError, match="context hashes"):
        _validate(changed)


def test_causal_manifest_keys_do_not_collapse_identical_questions(panel):
    case = panel["cases"][0]
    first = case["questions"][0]
    duplicate = copy.deepcopy(first)
    duplicate["question_key"] = "distinct-local-question-key"
    assert duplicate["content_sha256"] == first["content_sha256"]
    assert scorer.causal_entry_key(case, duplicate) != scorer.causal_entry_key(
        case, first
    )

    changed = copy.deepcopy(panel)
    changed["cases"][0]["benchmark_path"] = "/secret/eval"
    with pytest.raises(ValueError, match="lineage/path/id"):
        _validate(changed)


def test_precomputed_provider_requires_exact_hash_coverage_and_finite_vectors(panel):
    changed = copy.deepcopy(panel)
    changed["providers"]["fixture-dense"]["queries"].pop(
        next(iter(changed["providers"]["fixture-dense"]["queries"]))
    )
    with pytest.raises(ValueError, match="exactly cover"):
        _validate(changed)

    changed = copy.deepcopy(panel)
    key = next(iter(changed["providers"]["fixture-dense"]["documents"]))
    changed["providers"]["fixture-dense"]["documents"][key][0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        _validate(changed)

    changed = copy.deepcopy(panel)
    changed["providers"]["fixture-dense"]["hosted_api"] = True
    with pytest.raises(ValueError, match="local and benchmark-independent"):
        _validate(changed)


def test_receipt_is_content_bound_timestamp_free_and_atomic(panel, tmp_path):
    _validate(panel)
    method = scorer.score_method(panel, "bm25", (1, 2, 4))
    _, causal_digest = _causal_manifest(panel)
    _, contamination_digest = _contamination_manifest(panel)
    receipt = scorer.build_receipt(
        panel,
        _sha("raw"),
        causal_digest,
        contamination_digest,
        (1, 2, 4),
        [method],
    )
    assert "timestamp" not in json.dumps(receipt)
    expected = scorer.sha256_json(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    assert receipt["receipt_sha256"] == expected
    output = tmp_path / "receipt.json"
    scorer.atomic_write_json(output, receipt)
    assert output.read_bytes() == scorer.canonical_json(receipt)
    scorer.atomic_write_json(output, receipt)
    assert output.read_bytes() == scorer.canonical_json(receipt)
    changed = copy.deepcopy(receipt)
    changed["configuration"]["ks"] = [1]
    with pytest.raises(FileExistsError, match="refusing to replace"):
        scorer.atomic_write_json(output, changed)


def test_cli_scores_bm25_and_precomputed_without_model_download(
    panel, tmp_path, monkeypatch
):
    source = tmp_path / "panel.json"
    causal_source = tmp_path / "causal.json"
    contamination_source = tmp_path / "contamination.json"
    output = tmp_path / "receipt.json"
    causal_manifest, causal_digest = _causal_manifest(panel)
    causal_source.write_bytes(scorer.canonical_json(causal_manifest))
    contamination_manifest, contamination_digest = _contamination_manifest(panel)
    contamination_source.write_bytes(scorer.canonical_json(contamination_manifest))
    source.write_bytes(scorer.canonical_json(panel))
    monkeypatch.setattr(
        "sys.argv",
        [
            "score_r052_evidence_retrieval.py",
            "--panel",
            str(source),
            "--causal-manifest",
            str(causal_source),
            "--causal-manifest-sha256",
            causal_digest,
            "--contamination-manifest",
            str(contamination_source),
            "--contamination-manifest-sha256",
            contamination_digest,
            "--causal-annotation-builder-artifact",
            str(TEST_ARTIFACT),
            "--contamination-auditor-artifact",
            str(TEST_ARTIFACT),
            "--contamination-corpus-artifact",
            str(TEST_ARTIFACT),
            "--contamination-tokenizer-artifact",
            str(TEST_ARTIFACT),
            "--contamination-chunker-artifact",
            str(TEST_ARTIFACT),
            "--provider-generator-artifact",
            f"fixture-dense={TEST_ARTIFACT}",
            "--output",
            str(output),
            "--provider",
            "fixture-dense",
        ],
    )
    scorer.main()
    receipt = json.loads(output.read_text())
    assert [method["method"] for method in receipt["methods"]] == [
        "bm25",
        "fixture-dense",
    ]
    assert receipt["input_raw_sha256"] == scorer.file_sha256(source)


def test_claimed_provenance_must_resolve_to_supplied_artifacts(panel, tmp_path):
    causal, causal_digest = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    impostor = tmp_path / "impostor.py"
    impostor.write_text("pass\n")
    with pytest.raises(ValueError, match="annotation builder artifact hash mismatch"):
        scorer.validate_panel(
            panel,
            causal,
            causal_digest,
            contamination,
            contamination_digest,
            annotation_builder_artifact=impostor,
            contamination_auditor_artifact=TEST_ARTIFACT,
            contamination_corpus_artifacts=[TEST_ARTIFACT],
            contamination_tokenizer_artifact=TEST_ARTIFACT,
            contamination_chunker_artifact=TEST_ARTIFACT,
            provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
        )


def test_io_paths_reject_all_primary_input_and_output_aliases(tmp_path):
    panel = tmp_path / "panel.json"
    causal = tmp_path / "causal.json"
    contamination = tmp_path / "contamination.json"
    for path in (panel, causal, contamination):
        path.write_text("{}")
    with pytest.raises(ValueError, match="supplied input artifact"):
        scorer.validate_io_paths(
            panel, causal, contamination, panel, tmp_path / "runtime.json"
        )
    with pytest.raises(ValueError, match="score and runtime"):
        scorer.validate_io_paths(
            panel,
            causal,
            contamination,
            tmp_path / "receipt.json",
            tmp_path / "receipt.json",
        )
    with pytest.raises(ValueError, match="input paths must not alias"):
        scorer.validate_io_paths(
            panel, panel, contamination, tmp_path / "out.json", tmp_path / "run.json"
        )
    artifact = tmp_path / "generator.py"
    artifact.write_text("pass\n")
    with pytest.raises(ValueError, match="supplied input artifact"):
        scorer.validate_io_paths(
            panel,
            causal,
            contamination,
            tmp_path / "out.json",
            artifact,
            (artifact,),
        )
    frozen_directory = tmp_path / "frozen-runtime"
    frozen_directory.mkdir()
    (frozen_directory / "lock.json").write_text("{}")
    with pytest.raises(ValueError, match="supplied input artifact"):
        scorer.validate_io_paths(
            panel,
            causal,
            contamination,
            frozen_directory / "out.json",
            tmp_path / "run.json",
            (frozen_directory,),
        )


def test_local_provider_is_exactly_pinned_offline_and_rejects_fp16():
    assert scorer.MODEL_REVISION == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert scorer.TRANSFORMERS_VERSION == "5.19.0"
    assert scorer.TRANSFORMERS_WHEEL_SHA256 == (
        "afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518"
    )
    source = Path(scorer.__file__).read_text()
    assert "local_files_only=True" in source
    assert "snapshot_download(" in source
    assert "revision=MODEL_REVISION" in source
    with pytest.raises(ValueError, match="fp16 is unsafe"):
        scorer.LocalEmbeddingGemma2Provider("fp16", "cpu", None)


def test_frozen_snapshot_contract_checks_hashes_and_architecture(tmp_path):
    files = {
        "config.json": json.dumps(
            {"model_type": "embedding_gemma2", "text_config": {"embedding_dim": 768}}
        ).encode(),
        "1_Pooling/config.json": json.dumps(
            {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True}
        ).encode(),
    }
    for relative, payload in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    expected = {
        relative: scorer.sha256_bytes(payload) for relative, payload in files.items()
    }
    assert scorer.verify_frozen_snapshot(tmp_path, expected) == expected
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        scorer.verify_frozen_snapshot(tmp_path, expected)


def test_local_provider_contract_if_compatible_runtime_and_weights_are_cached(panel):
    import transformers

    if transformers.__version__ != scorer.TRANSFORMERS_VERSION:
        pytest.skip("exact pinned Transformers runtime is absent")
    pytest.importorskip("transformers.models.embedding_gemma2")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(
            repo_id=scorer.MODEL_ID,
            revision=scorer.MODEL_REVISION,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        pytest.skip("exact pinned EmbeddingGemma 2 snapshot is not cached")
    runtime_lock = os.environ.get("R052_RUNTIME_LOCK")
    runtime_lock_sha256 = os.environ.get("R052_RUNTIME_LOCK_SHA256")
    transformers_wheel = os.environ.get("R052_TRANSFORMERS_WHEEL")
    if not runtime_lock or not runtime_lock_sha256 or not transformers_wheel:
        pytest.skip("frozen R052 runtime lock and Transformers wheel are absent")
    provider = scorer.LocalEmbeddingGemma2Provider(
        "fp32",
        "cpu",
        None,
        Path(runtime_lock),
        runtime_lock_sha256,
        Path(transformers_wheel),
    )
    question = panel["cases"][0]["questions"][0]
    image_chunk = next(
        chunk
        for chunk in panel["cases"][0]["chunks"]
        if chunk["image_control"] is not None
        and chunk["image_control"]["variant"] == "real"
    )
    query_one = provider.query(question)
    query_two = provider.query(question)
    image_vector = provider.document(image_chunk)
    assert query_one.shape == image_vector.shape == (768,)
    assert np.array_equal(query_one, query_two)
    assert np.linalg.norm(query_one) == pytest.approx(1.0)
    assert np.isfinite(image_vector).all()
