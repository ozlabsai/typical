import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from scripts import build_r052_evidence_panel as builder
from scripts import score_r052_evidence_retrieval as scorer


def _write_json(path: Path, value) -> str:
    payload = scorer.canonical_json(value)
    path.write_bytes(payload)
    return scorer.sha256_bytes(payload)


def _artifact_declaration(path: Path, role: str) -> dict:
    return {
        "role": role,
        "sha256": scorer.artifact_sha256(path),
    }


def _license_policy() -> dict:
    return {
        "source_family": "first_party",
        "license_id": "LicenseRef-First-Party",
        "license_url": "https://example.invalid/license/first-party",
        "attribution": "R052 test fixture author",
        "license_cleared": True,
        "evaluation_use_cleared": True,
        "redistribution_cleared": True,
    }


def _snapshot(raw: str, extracted: str) -> dict:
    return {
        "raw_artifact": raw,
        "extraction_artifact": extracted,
        "source_family": "first_party",
        "upstream_url": f"https://example.invalid/snapshots/{raw}",
        "upstream_revision": f"immutable-{raw}-v1",
        "upstream_date": "2026-10-08",
        "untouched_split": "authored-test-eval-v1",
        "license_policy": "first-party",
        "attribution": "R052 test fixture author",
        "third_party_images_excluded": True,
        "natural_language_authored": True,
        "benchmark_derived": False,
    }


def _case(language: str, suffix: str, source: str, image_a: str, image_b: str) -> dict:
    hebrew = language == "he"
    chunks = [
        {
            "chunk_ref": "policy",
            "kind": "global_policy",
            "modality": "text",
            "text": "פעל לפי הרשומה." if hebrew else "Follow the authored record.",
            "token_start": 0,
            "token_end": 5,
            "twin_anchor": "policy",
            "media_artifact": None,
            "image_control": None,
        },
        {
            "chunk_ref": "decisive-text",
            "kind": "evidence",
            "modality": "text",
            "text": "האישור בתוקף." if hebrew else "The approval remains valid.",
            "token_start": 5,
            "token_end": 10,
            "twin_anchor": "decisive-text",
            "media_artifact": None,
            "image_control": None,
        },
    ]
    for index, (variant, media) in enumerate(
        (("real", image_a), ("shuffled", image_b), ("missing", None)), start=10
    ):
        chunks.append(
            {
                "chunk_ref": f"image-{variant}",
                "kind": "evidence" if variant == "real" else "distractor",
                "modality": "image",
                "text": "תצלום החותמת." if hebrew else "Photograph of the seal.",
                "token_start": index,
                "token_end": index + 1,
                "twin_anchor": f"image-{variant}",
                "media_artifact": media,
                "image_control": {
                    "group_ref": f"seal-{suffix}",
                    "variant": variant,
                    "mapping_receipt_artifact": "control-receipt",
                },
            }
        )
    return {
        "case_ref": f"case-{suffix}",
        "language": language,
        "twin_ref": "approval-twin",
        "source_snapshot": source,
        "alignment_receipt_artifact": "alignment-receipt",
        "source_domain": "operations",
        "context_tokens": 13,
        "chunking": {
            "mode": "semantic_sections",
            "boundary_receipt_artifact": "boundary-receipt",
        },
        "chunks": chunks,
        "questions": [
            {
                "question_ref": "approve",
                "text": "האם לאשר?" if hebrew else "Should this be approved?",
                "options": ["כן", "לא"] if hebrew else ["yes", "no"],
                "policy_role": "choice",
                "twin_anchor": "approve",
                "decisive_chunk_refs": ["decisive-text", "image-real"],
                "known_distractor_chunk_refs": ["image-shuffled", "image-missing"],
            }
        ],
    }


def _fixture(tmp_path: Path) -> dict:
    artifact_paths = {}
    for name, text in {
        "source-en": "An independently authored English operations dossier.",
        "source-he": "תיק תפעולי עברי שנכתב באופן עצמאי.",
        "alignment-receipt": "Human reviewer aligned the two authored cases.",
        "control-receipt": "Human reviewer designated real, shuffled, and absent controls.",
        "boundary-receipt": "Human author marked semantic section boundaries.",
    }.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(text)
        artifact_paths[name] = path
    for name, content in {
        "raw-en": b"independently authored raw English source\n",
        "raw-he": b"independently authored raw Hebrew source\n",
    }.items():
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(content)
        artifact_paths[name] = path
    for name, color in (("real-image", "red"), ("shuffled-image", "blue")):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (2, 2), color=color).save(path)
        artifact_paths[name] = path

    roles = {
        "raw-en": "raw_source",
        "raw-he": "raw_source",
        "source-en": "extracted_text",
        "source-he": "extracted_text",
        "alignment-receipt": "translation_alignment_receipt",
        "control-receipt": "image_control_receipt",
        "boundary-receipt": "chunk_boundary_receipt",
        "real-image": "image",
        "shuffled-image": "image",
    }
    source_manifest = {
        "schema_version": builder.SOURCE_MANIFEST_VERSION,
        "license_policy": builder.LICENSE_POLICY,
        "benchmark_derived": False,
        "license_policies": {"first-party": _license_policy()},
        "artifacts": {
            name: _artifact_declaration(artifact_paths[name], role)
            for name, role in roles.items()
        },
        "snapshots": {
            "snapshot-en": _snapshot("raw-en", "source-en"),
            "snapshot-he": _snapshot("raw-he", "source-he"),
        },
    }
    spec = {
        "schema_version": builder.SPEC_VERSION,
        "panel_profile": "test_fixture",
        "benchmark_derived": False,
        "cases": [
            _case("en", "en", "snapshot-en", "real-image", "shuffled-image"),
            _case("he", "he", "snapshot-he", "real-image", "shuffled-image"),
        ],
    }
    annotations = {
        "schema_version": builder.ANNOTATION_VERSION,
        "benchmark_derived": False,
        "entries": [
            {
                "case_ref": case["case_ref"],
                "question_ref": "approve",
                "removed_chunk_refs": ["decisive-text", "image-real"],
                "full_outcome": {"decision": "approve", "score": 1},
                "removed_outcome": {"decision": "review", "score": 0},
                "changed_fields": ["decision", "score"],
            }
            for case in spec["cases"]
        ],
    }
    spec_path = tmp_path / "spec.json"
    source_path = tmp_path / "sources.json"
    annotation_path = tmp_path / "annotations.json"
    hashes = {
        "dossier_spec_sha256": _write_json(spec_path, spec),
        "source_manifest_sha256": _write_json(source_path, source_manifest),
        "causal_annotations_sha256": _write_json(annotation_path, annotations),
    }
    resolved = builder._validate_source_manifest(source_manifest, artifact_paths)
    builder._validate_spec(spec, source_manifest, allow_test_profile=True)
    annotation_map = builder._validate_annotations(annotations)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"tokenizer":"frozen"}\n')
    chunker = tmp_path / "chunker.json"
    chunker.write_text('{"chunker":"frozen"}\n')
    audit_input, cases, _ = builder.build_audit_input(
        spec,
        source_manifest,
        annotation_map,
        resolved,
        scorer.artifact_sha256(tokenizer),
        scorer.artifact_sha256(chunker),
        spec_sha256=hashes["dossier_spec_sha256"],
        source_manifest_sha256=hashes["source_manifest_sha256"],
        annotations_sha256=hashes["causal_annotations_sha256"],
    )
    audit_input_path = tmp_path / "audit-input.json"
    audit_input_sha = _write_json(audit_input_path, audit_input)

    auditor = tmp_path / "auditor.py"
    auditor.write_text("# independently run audit implementation\n")
    corpus = tmp_path / "public-corpus.txt"
    corpus.write_text(
        "Public open reference material used only by the external auditor."
    )
    audit_results = {
        "schema_version": builder.AUDIT_RESULTS_VERSION,
        "audit_input_sha256": audit_input_sha,
        "sealed_benchmark_text_included": False,
        "algorithm": builder.AUDIT_ALGORITHM,
        "auditor_code_sha256": scorer.artifact_sha256(auditor),
        "audited_corpus_roots_sha256": [scorer.artifact_sha256(corpus)],
        "tokenizer_receipt_sha256": scorer.artifact_sha256(tokenizer),
        "chunker_receipt_sha256": scorer.artifact_sha256(chunker),
        "entries": [
            {
                "case_ref": authored["case_ref"],
                "case_content_sha256": scorer.case_content_sha256(case),
                "status": "pass",
                "exact_overlap_count": 0,
                "near_overlap_max_jaccard": 0.12,
            }
            for authored, case in zip(spec["cases"], cases, strict=True)
        ],
    }
    audit_results_path = tmp_path / "audit-results.json"
    hashes["audit_results_sha256"] = _write_json(audit_results_path, audit_results)

    generator = tmp_path / "generator.py"
    generator.write_text("# local deterministic embedding generator\n")
    queries = {
        question["content_sha256"]: [1.0, 0.0]
        for case in cases
        for question in case["questions"]
    }
    documents = {
        chunk["content_sha256"]: [0.0, 1.0]
        for case in cases
        for chunk in case["chunks"]
    }
    provider = {
        "schema_version": builder.PROVIDER_BUNDLE_VERSION,
        "origin": "local_baseline",
        "hosted_api": False,
        "benchmark_derived": False,
        "model": "fixture-embedder",
        "revision": "fixture-v1",
        "generator_code_sha256": scorer.artifact_sha256(generator),
        "dimension": 2,
        "queries": queries,
        "documents": documents,
    }
    provider_path = tmp_path / "provider.json"
    provider_sha = _write_json(provider_path, provider)
    return locals()


def _finalize(fixture: dict, output_name: str = "bundle", **overrides):
    arguments = {
        "output_dir": fixture["tmp_path"] / output_name,
        "spec": fixture["spec"],
        "source_manifest": fixture["source_manifest"],
        "annotations": fixture["annotation_map"],
        "audit_input": fixture["audit_input"],
        "audit_input_sha256": fixture["audit_input_sha"],
        "audit_results": fixture["audit_results"],
        "provider_bundles": {"fixture": fixture["provider"]},
        "provider_bundle_sha256": {"fixture": fixture["provider_sha"]},
        "provider_generator_artifacts": {"fixture": fixture["generator"]},
        "source_artifact_paths": fixture["resolved"],
        "auditor_artifact": fixture["auditor"],
        "corpus_artifacts": [fixture["corpus"]],
        "tokenizer_artifact": fixture["tokenizer"],
        "chunker_artifact": fixture["chunker"],
        "bindings": fixture["hashes"],
    }
    arguments.update(overrides)
    return builder.finalize_bundle(**arguments)


def test_builds_deterministic_scorer_valid_bundle_and_preserves_order(tmp_path):
    fixture = _fixture(tmp_path)
    receipt = _finalize(fixture)
    output = tmp_path / "bundle"
    panel = json.loads((output / "panel.json").read_bytes())
    causal = json.loads((output / "causal_manifest.json").read_bytes())
    contamination = json.loads((output / "contamination_manifest.json").read_bytes())

    assert [case["language"] for case in panel["cases"]] == ["en", "he"]
    assert panel["cases"][0]["source_snapshot_sha256"] == scorer.artifact_sha256(
        fixture["artifact_paths"]["raw-en"]
    )
    assert panel["cases"][0]["chunking_receipt"][
        "tokenizer_artifact_sha256"
    ] == scorer.artifact_sha256(fixture["tokenizer"])
    assert [chunk["twin_anchor"] for chunk in panel["cases"][0]["chunks"]] == [
        "policy",
        "decisive-text",
        "image-real",
        "image-shuffled",
        "image-missing",
    ]
    assert panel["cases"][0]["chunks"][2]["media"]["sha256"] == scorer.file_sha256(
        fixture["artifact_paths"]["real-image"]
    )
    assert receipt["semantic_ground_truth"] == {
        "origin": "externally_authored_causal_annotations",
        "builder_executed_domain_semantics": False,
        "builder_claims_semantic_validation": False,
    }
    scorer.validate_panel(
        panel,
        fixture["source_manifest"],
        fixture["hashes"]["source_manifest_sha256"],
        causal,
        receipt["outputs"]["causal_manifest_external_sha256"],
        contamination,
        receipt["outputs"]["contamination_manifest_external_sha256"],
        annotation_builder_artifact=Path(builder.__file__),
        contamination_auditor_artifact=fixture["auditor"],
        contamination_corpus_artifacts=[fixture["corpus"]],
        contamination_tokenizer_artifact=fixture["tokenizer"],
        contamination_chunker_artifact=fixture["chunker"],
        source_snapshot_artifacts=fixture["resolved"],
        provider_generator_artifacts={"fixture": fixture["generator"]},
    )

    second = _finalize(fixture, "bundle-second")
    assert second == receipt
    for name in (
        "panel.json",
        "source_manifest.json",
        "causal_manifest.json",
        "contamination_manifest.json",
        "audit_input.json",
        "build_receipt.json",
    ):
        assert (output / name).read_bytes() == (
            tmp_path / "bundle-second" / name
        ).read_bytes()


def test_can_finalize_provider_free_panel_for_bm25_preflight(tmp_path):
    fixture = _fixture(tmp_path)
    receipt = _finalize(
        fixture,
        provider_bundles={},
        provider_bundle_sha256={},
        provider_generator_artifacts={},
    )
    panel = json.loads((tmp_path / "bundle" / "panel.json").read_bytes())
    assert panel["providers"] == {}
    assert receipt["counts"]["providers"] == 0


def test_rejects_tampered_audit_binding_before_publication(tmp_path):
    fixture = _fixture(tmp_path)
    tampered = deepcopy(fixture["audit_results"])
    tampered["entries"][0]["case_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact case content"):
        _finalize(
            fixture,
            audit_results=tampered,
            bindings={
                **fixture["hashes"],
                "audit_results_sha256": scorer.sha256_json(tampered),
            },
        )
    assert not (tmp_path / "bundle").exists()


def test_rejects_audit_results_not_matching_external_hash(tmp_path):
    fixture = _fixture(tmp_path)
    tampered = deepcopy(fixture["audit_results"])
    tampered["entries"][0]["near_overlap_max_jaccard"] = 0.13
    with pytest.raises(ValueError, match="external SHA-256 binding"):
        _finalize(fixture, audit_results=tampered)


def test_rejects_provider_bundle_not_matching_external_hash(tmp_path):
    fixture = _fixture(tmp_path)
    tampered = deepcopy(fixture["provider"])
    tampered["revision"] = "fixture-v2"
    with pytest.raises(ValueError, match="external hash mismatch"):
        _finalize(fixture, provider_bundles={"fixture": tampered})


def test_rejects_causal_claim_not_equal_to_authored_outcome_diff(tmp_path):
    fixture = _fixture(tmp_path)
    annotations = deepcopy(fixture["annotations"])
    annotations["entries"][0]["changed_fields"] = ["decision"]
    with pytest.raises(ValueError, match="exactly match"):
        builder._validate_annotations(annotations)


def test_rejects_uncleared_or_family_incompatible_license(tmp_path):
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["source_manifest"])
    manifest["license_policies"]["first-party"]["source_family"] = "permissive"
    manifest["license_policies"]["first-party"]["license_id"] = "CC-BY-NC-SA-4.0"
    with pytest.raises(ValueError, match="license/family"):
        builder._validate_source_manifest(manifest, fixture["artifact_paths"])


def test_rejects_benchmark_lineage_marker_in_source_bytes(tmp_path):
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["source_manifest"])
    source = fixture["artifact_paths"]["source-en"]
    source.write_text("copied from open-jev")
    manifest["artifacts"]["source-en"]["sha256"] = scorer.artifact_sha256(source)
    with pytest.raises(ValueError, match="benchmark lineage markers"):
        builder._validate_source_manifest(manifest, fixture["artifact_paths"])


def test_rejects_duplicate_source_snapshot_hash_declarations(tmp_path):
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["source_manifest"])
    duplicate = fixture["artifact_paths"]["source-en"].read_bytes()
    fixture["artifact_paths"]["source-he"].write_bytes(duplicate)
    manifest["artifacts"]["source-he"]["sha256"] = manifest["artifacts"]["source-en"][
        "sha256"
    ]
    with pytest.raises(ValueError, match="hash declarations must be unique"):
        builder._validate_source_manifest(manifest, fixture["artifact_paths"])


def test_rejects_declared_source_artifact_not_consumed_by_authored_cases(tmp_path):
    fixture = _fixture(tmp_path)
    extra = tmp_path / "unused-control.txt"
    extra.write_text("independent unused control receipt")
    fixture["source_manifest"]["artifacts"]["unused-control"] = {
        "role": "image_control_receipt",
        "sha256": scorer.artifact_sha256(extra),
    }
    with pytest.raises(ValueError, match="consume every declared"):
        builder._validate_spec(fixture["spec"], fixture["source_manifest"], True)


def test_test_fixture_requires_explicit_opt_in_and_production_rejects_small_panel(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit opt-in"):
        builder._validate_spec(fixture["spec"], fixture["source_manifest"], False)
    production = deepcopy(fixture["spec"])
    production["panel_profile"] = "production"
    with pytest.raises(ValueError, match="production panel minimum contract"):
        builder._validate_spec(production, fixture["source_manifest"], False)


def test_fixed_fallback_enforces_512_token_64_overlap(tmp_path):
    fixture = _fixture(tmp_path)
    spec = deepcopy(fixture["spec"])
    case = spec["cases"][0]
    case["chunking"] = {
        "mode": "fixed_512_overlap_64",
        "boundary_receipt_artifact": None,
    }
    with pytest.raises(ValueError, match="overlap by exactly 64"):
        builder._validate_spec(spec, fixture["source_manifest"], True)


def test_refuses_to_replace_existing_bundle(tmp_path):
    fixture = _fixture(tmp_path)
    _finalize(fixture)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _finalize(fixture)
