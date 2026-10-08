import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from scripts import build_r052_evidence_panel as builder
from scripts import replay_r052_typed_policy as replay
from scripts import score_r052_evidence_retrieval as scorer

ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_SOURCE = ROOT / "sources" / "r052" / "ntsb_gao_v1.json"
TEST_SOURCE_IDS = ("AAR0002", "GAO-01-203")


@pytest.fixture(autouse=True)
def _compact_acquisition_contract(monkeypatch):
    monkeypatch.setattr(scorer.acquisition, "EXPECTED_IDS", TEST_SOURCE_IDS)


def _write_json(path: Path, value) -> str:
    payload = scorer.canonical_json(value)
    path.write_bytes(payload)
    return scorer.sha256_bytes(payload)


def _artifact_declaration(path: Path, role: str) -> dict:
    return {
        "role": role,
        "sha256": scorer.artifact_sha256(path),
    }


def _license_policy(acquired_policy: dict) -> dict:
    return {
        "source_family": acquired_policy["source_family"],
        "license_id": "LicenseRef-USGov-Public-Domain",
        "license_url": acquired_policy["policy_url"],
        "attribution": acquired_policy["attribution"],
        "license_cleared": True,
        "evaluation_use_cleared": True,
        "redistribution_cleared": True,
    }


def _snapshot(
    reviewed: str, acquisition: str, acquired: dict, acquired_policy: dict
) -> dict:
    return {
        "reviewed_text_artifact": reviewed,
        "acquisition_binding_artifact": acquisition,
        "source_family": acquired_policy["source_family"],
        "upstream_url": acquired["upstream_url"],
        "upstream_revision": acquired["upstream_revision"],
        "upstream_date": acquired["public_release_date"]
        or acquired["adoption_or_issue_date"],
        "untouched_split": acquired["untouched_split"],
        "license_policy": acquired["license_policy"],
        "attribution": acquired_policy["attribution"],
        "third_party_images_excluded": True,
        "natural_language_authored": True,
        "benchmark_derived": False,
    }


def _case(
    language: str,
    suffix: str,
    source: str,
    image_a: str,
    image_b: str,
    reviewed_text: str,
) -> dict:
    hebrew = language == "he"
    split_points = [len(reviewed_text) * index // 6 for index in range(7)]
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
            "derivation": {"kind": "non_source_global_policy"},
        },
        {
            "chunk_ref": "decisive-text",
            "kind": "evidence",
            "modality": "text",
            "text": (
                "האישור בתוקף."
                if hebrew
                else reviewed_text[split_points[0] : split_points[1]]
            ),
            "token_start": 5,
            "token_end": 10,
            "twin_anchor": "decisive-text",
            "media_artifact": None,
            "image_control": None,
            "derivation": (
                {
                    "kind": "translated_twin",
                    "source_twin_case_ref": "case-en",
                    "source_twin_chunk_ref": "decisive-text",
                    "alignment_receipt_artifact": "alignment-receipt",
                }
                if hebrew
                else {
                    "kind": "source_text_slice",
                    "source_artifact": "source-en",
                    "char_start": 0,
                    "char_end": split_points[1],
                }
            ),
        },
    ]
    for index in range(1, 6):
        chunks.append(
            {
                "chunk_ref": f"text-distractor-{index}",
                "kind": "distractor",
                "modality": "text",
                "text": (
                    f"מידע לא מכריע {index}."
                    if hebrew
                    else reviewed_text[split_points[index] : split_points[index + 1]]
                ),
                "token_start": 9 + index,
                "token_end": 10 + index,
                "twin_anchor": f"text-distractor-{index}",
                "media_artifact": None,
                "image_control": None,
                "derivation": (
                    {
                        "kind": "translated_twin",
                        "source_twin_case_ref": "case-en",
                        "source_twin_chunk_ref": f"text-distractor-{index}",
                        "alignment_receipt_artifact": "alignment-receipt",
                    }
                    if hebrew
                    else {
                        "kind": "source_text_slice",
                        "source_artifact": "source-en",
                        "char_start": split_points[index],
                        "char_end": split_points[index + 1],
                    }
                ),
            }
        )
    for index, (variant, media) in enumerate(
        (("real", image_a), ("shuffled", image_b), ("missing", None)), start=15
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
                "derivation": {
                    "kind": "image_control",
                    "image_artifact": media,
                    "control_receipt_artifact": "control-receipt",
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
        "context_tokens": 18,
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
                "option_ids": ["yes", "no"],
                "decision_type": "choice",
                "policy_role": "choice",
                "twin_anchor": "approve",
                "decisive_chunk_refs": ["decisive-text", "image-real"],
                "known_distractor_chunk_refs": [
                    *[f"text-distractor-{index}" for index in range(1, 6)],
                    "image-shuffled",
                    "image-missing",
                ],
            }
        ],
    }


def _policy_artifacts(tmp_path: Path, audit_input: dict) -> dict:
    draft = audit_input["policy_replay_draft"]
    panel_digest = audit_input["policy_replay_draft_sha256"]
    facts = {
        "f0001": {"type": "boolean", "unit": None},
        "f0002": {"type": "boolean", "unit": None},
    }
    adjudication = tmp_path / "policy-adjudication.txt"
    adjudication.write_text("Independent pre-question fact adjudication.\n")
    adjudication_digest = scorer.artifact_sha256(adjudication)
    bindings = []
    presentations = []
    decision_id = "d.approve"
    for case in draft["cases"]:
        chunks = {chunk["twin_anchor"]: chunk for chunk in case["chunks"]}
        for fact, anchor in (("f0001", "decisive-text"), ("f0002", "image-real")):
            chunk = chunks[anchor]
            locator = (
                {
                    "kind": "image_region",
                    "region_sha256": chunk["media"]["sha256"],
                }
                if chunk["media"] is not None
                else {
                    "kind": "text_span",
                    "char_start": 0,
                    "char_end": len(chunk["text"]),
                    "span_sha256": scorer.sha256_bytes(chunk["text"].encode()),
                }
            )
            bindings.append(
                {
                    "case_key": case["case_key"],
                    "chunk_key": chunk["chunk_key"],
                    "chunk_content_sha256": chunk["content_sha256"],
                    "observations": [{"fact": fact, "value": True, "locator": locator}],
                    "adjudication_receipt_sha256": adjudication_digest,
                }
            )
        question = case["questions"][0]
        full = [chunk["chunk_key"] for chunk in case["chunks"]]
        immutable = {
            chunk["chunk_key"] for chunk in case["chunks"] if chunk["immutable"]
        }
        decisive = question["decisive_chunk_keys"]
        distractors = question["known_distractor_chunk_keys"]
        presentations.append(
            {
                "presentation_id": f"p.approve.{case['language']}",
                "case_key": case["case_key"],
                "question_bindings": {
                    decision_id: {
                        "question_key": question["question_key"],
                        "question_content_sha256": question["content_sha256"],
                    }
                },
                "views": {
                    "full": full,
                    "removed": [key for key in full if key not in decisive],
                    "decisive_only": [
                        key for key in full if key in immutable | set(decisive)
                    ],
                    "distractor_only": [
                        key for key in full if key in immutable | set(distractors)
                    ],
                },
                "ablations": [
                    {
                        "decision_id": decision_id,
                        "remove_chunk_keys": decisive,
                        "required_transition": "resolution_loss",
                        "assert_full": {"status": "resolved", "output": "yes"},
                        "assert_removed": {"status": "unresolved"},
                    }
                ],
            }
        )
    ledger = {
        "schema_version": replay.LEDGER_VERSION,
        "panel_sha256": panel_digest,
        "facts": facts,
        "bindings": bindings,
    }
    ledger_digest = scorer.sha256_json(ledger)
    any_fact = {
        "op": "any",
        "args": [
            {"op": "fact", "id": "f0001"},
            {"op": "fact", "id": "f0002"},
        ],
    }
    rubric = {
        "schema_version": replay.RUBRIC_VERSION,
        "panel_sha256": panel_digest,
        "fact_ledger_sha256": ledger_digest,
        "semantic_groups": [
            {
                "group_id": "g.approve",
                "facts": facts,
                "decisions": [
                    {
                        "decision_id": decision_id,
                        "type": "choice",
                        "outputs": [
                            {"id": "yes", "label": "yes"},
                            {"id": "no", "label": "no"},
                        ],
                        "order": ["yes", "no"],
                        "rules": [
                            {"output": "yes", "when": any_fact},
                            {"output": "no", "when": {"op": "not", "arg": any_fact}},
                        ],
                    }
                ],
                "presentations": presentations,
            }
        ],
    }
    rubric_digest = scorer.sha256_json(rubric)
    schema_artifact = Path("sources/r052/policy_replay.schema.json").resolve()
    schema_digest = scorer.artifact_sha256(schema_artifact)
    evaluator_artifact = Path(replay.__file__).resolve()
    evaluator_digest = scorer.artifact_sha256(evaluator_artifact)
    return {
        "policy_fact_ledger": ledger,
        "policy_fact_ledger_sha256": ledger_digest,
        "policy_rubric": rubric,
        "policy_rubric_sha256": rubric_digest,
        "policy_replay_schema_artifact": schema_artifact,
        "policy_replay_schema_sha256": schema_digest,
        "policy_evaluator_artifact": evaluator_artifact,
        "policy_evaluator_sha256": evaluator_digest,
        "policy_adjudication_artifacts": {adjudication_digest: adjudication},
    }


def _fixture(tmp_path: Path) -> dict:
    artifact_paths = {}
    reviewed_en = " ".join(f"approval{i}" for i in range(8_192))
    reviewed_he = " ".join(f"אישור{i}" for i in range(8_192))
    for name, text in {
        "source-en": reviewed_en,
        "source-he": reviewed_he,
        "alignment-receipt": "Human reviewer aligned the two authored cases.",
        "control-receipt": "Human reviewer designated real, shuffled, and absent controls.",
        "boundary-receipt": "Human author marked semantic section boundaries.",
    }.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(text)
        artifact_paths[name] = path
    for name, color in (("real-image", "red"), ("shuffled-image", "blue")):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (2, 2), color=color).save(path)
        artifact_paths[name] = path

    acquisition_manifest = json.loads(ACQUISITION_SOURCE.read_text(encoding="utf-8"))
    acquisition_manifest["snapshots"] = [
        snapshot
        for snapshot in acquisition_manifest["snapshots"]
        if snapshot["id"] in TEST_SOURCE_IDS
    ]
    acquired_by_id = {
        snapshot["id"]: snapshot for snapshot in acquisition_manifest["snapshots"]
    }
    language_to_id = dict(zip(("en", "he"), TEST_SOURCE_IDS, strict=True))
    extracted_hashes = {}
    for language, source_id in language_to_id.items():
        digest = scorer.sha256_bytes(f"extracted-{language}".encode())
        acquired_by_id[source_id]["extracted_text"].update(
            {"sha256": digest, "bytes": 100, "words": 10}
        )
        extracted_hashes[language] = digest
    acquisition_manifest_path = tmp_path / "acquisition-manifest.json"
    acquisition_manifest_sha = _write_json(
        acquisition_manifest_path, acquisition_manifest
    )
    artifact_paths["acquisition-manifest"] = acquisition_manifest_path
    acquisition_run = {
        "schema_version": scorer.ACQUISITION_RUN_RECEIPT_VERSION,
        "mode": "offline",
        "manifest_sha256": acquisition_manifest_sha,
        "verified": [
            {
                "id": language_to_id[language],
                "raw_sha256": acquired_by_id[language_to_id[language]]["raw_pdf"][
                    "sha256"
                ],
                "text_sha256": extracted_hashes[language],
            }
            for language in ("en", "he")
        ],
    }
    run_path = tmp_path / "acquisition-run.json"
    _write_json(run_path, acquisition_run)
    artifact_paths["acquisition-run"] = run_path
    run_sha = scorer.artifact_sha256(run_path)
    for language, reviewed in (("en", reviewed_en), ("he", reviewed_he)):
        source_id = language_to_id[language]
        acquired = acquired_by_id[source_id]
        acquired_policy = acquisition_manifest["license_policies"][
            acquired["license_policy"]
        ]
        reviewed_name = f"source-{language}"
        binding = {
            "schema_version": scorer.ACQUISITION_BINDING_VERSION,
            "snapshot_id": source_id,
            "acquisition_manifest_sha256": acquisition_manifest_sha,
            "acquisition_receipt_sha256": run_sha,
            "extracted_text_sha256": extracted_hashes[language],
            "review_receipt_sha256": scorer.sha256_bytes(b"review-receipt"),
            "retained_ranges": [{"char_start": 0, "char_end": len(reviewed)}],
            "range_mappings": [
                {
                    "source_char_start": 0,
                    "source_char_end": len(reviewed),
                    "output_char_start": 0,
                    "output_char_end": len(reviewed),
                }
            ],
            "reviewed_character_count": len(reviewed),
            "reviewed_unicode_word_token_count": len(scorer.TOKEN_RE.findall(reviewed)),
            "word_token_count_method": scorer.WORD_TOKEN_COUNT_METHOD,
            "reviewed_text_sha256": scorer.artifact_sha256(
                artifact_paths[reviewed_name]
            ),
            "upstream_url": acquired["upstream_url"],
            "upstream_revision": acquired["upstream_revision"],
            "license_policy": acquired["license_policy"],
            "attribution": acquired_policy["attribution"],
        }
        path = tmp_path / f"binding-{language}.json"
        _write_json(path, binding)
        artifact_paths[f"binding-{language}"] = path

    roles = {
        "source-en": "reviewed_source_text",
        "source-he": "reviewed_source_text",
        "acquisition-manifest": "source_acquisition_manifest",
        "acquisition-run": "source_acquisition_run_receipt",
        "binding-en": "reviewed_source_acquisition_binding",
        "binding-he": "reviewed_source_acquisition_binding",
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
        "license_policies": {
            name: _license_policy(policy)
            for name, policy in acquisition_manifest["license_policies"].items()
        },
        "artifacts": {
            name: _artifact_declaration(artifact_paths[name], role)
            for name, role in roles.items()
        },
        "acquisition_manifest_artifact": "acquisition-manifest",
        "acquisition_run_receipt_artifact": "acquisition-run",
        "snapshots": {
            source_id: _snapshot(
                f"source-{language}",
                f"binding-{language}",
                acquired_by_id[source_id],
                acquisition_manifest["license_policies"][
                    acquired_by_id[source_id]["license_policy"]
                ],
            )
            for language, source_id in language_to_id.items()
        },
    }
    spec = {
        "schema_version": builder.SPEC_VERSION,
        "panel_profile": "test_fixture",
        "benchmark_derived": False,
        "cases": [
            _case(
                "en",
                "en",
                language_to_id["en"],
                "real-image",
                "shuffled-image",
                reviewed_en,
            ),
            _case(
                "he",
                "he",
                language_to_id["he"],
                "real-image",
                "shuffled-image",
                reviewed_he,
            ),
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
                "full_outcome": {"status": "resolved", "option_index": 0},
                "removed_outcome": {"status": "unresolved"},
                "changed_fields": ["status", "option_index"],
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
        scorer.retrieval_query_sha256(question): [1.0, 0.0]
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
    policy = _policy_artifacts(tmp_path, audit_input)
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
        **fixture["policy"],
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
        fixture["artifact_paths"]["source-en"]
    )
    assert panel["cases"][0]["chunking_receipt"][
        "tokenizer_artifact_sha256"
    ] == scorer.artifact_sha256(fixture["tokenizer"])
    assert [chunk["twin_anchor"] for chunk in panel["cases"][0]["chunks"]] == [
        "policy",
        "decisive-text",
        "text-distractor-1",
        "text-distractor-2",
        "text-distractor-3",
        "text-distractor-4",
        "text-distractor-5",
        "image-real",
        "image-shuffled",
        "image-missing",
    ]
    assert panel["cases"][0]["chunks"][7]["media"]["sha256"] == scorer.file_sha256(
        fixture["artifact_paths"]["real-image"]
    )
    assert receipt["semantic_ground_truth"] == {
        "origin": "deterministic_typed_policy_replay",
        "annotations_role": "preregistered_draft_inputs_only",
        "builder_executed_domain_semantics": True,
        "builder_claims_semantic_validation": True,
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
        policy_replay_receipt=json.loads(
            (output / "policy_replay_receipt.json").read_bytes()
        ),
        policy_replay_receipt_sha256=receipt["outputs"][
            "policy_replay_receipt_external_sha256"
        ],
        **fixture["policy"],
    )

    second = _finalize(fixture, "bundle-second")
    assert second == receipt
    for name in (
        "panel.json",
        "source_manifest.json",
        "causal_manifest.json",
        "contamination_manifest.json",
        "audit_input.json",
        "policy_replay_receipt.json",
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


def test_prepare_audit_exposes_exact_acyclic_policy_draft(tmp_path):
    fixture = _fixture(tmp_path)
    draft = fixture["audit_input"]["policy_replay_draft"]
    assert fixture["audit_input"]["policy_replay_draft_sha256"] == scorer.sha256_json(
        draft
    )
    assert fixture["policy"]["policy_fact_ledger"][
        "panel_sha256"
    ] == scorer.sha256_json(draft)
    assert draft["providers"] == {}
    assert {case["contamination_audit_sha256"] for case in draft["cases"]} == {"0" * 64}


def test_prior_dossier_spec_requires_explicit_v4_migration(tmp_path):
    fixture = _fixture(tmp_path)
    old = deepcopy(fixture["spec"])
    old["schema_version"] = "r052-authored-dossier-spec-v3"
    with pytest.raises(ValueError, match="prior dossier specs require migration"):
        builder._validate_spec(old, fixture["source_manifest"], allow_test_profile=True)


def test_finalize_rejects_policy_artifact_hash_and_coverage_tampering(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="fact ledger does not match"):
        _finalize(fixture, policy_fact_ledger_sha256="0" * 64)

    rubric = deepcopy(fixture["policy"]["policy_rubric"])
    rubric["semantic_groups"][0]["presentations"] = rubric["semantic_groups"][0][
        "presentations"
    ][:1]
    with pytest.raises(ValueError, match="every panel question|Hebrew presentation"):
        _finalize(
            fixture,
            policy_rubric=rubric,
            policy_rubric_sha256=scorer.sha256_json(rubric),
        )

    wrong = tmp_path / "wrong-adjudication.txt"
    wrong.write_text("tampered")
    digest = next(iter(fixture["policy"]["policy_adjudication_artifacts"]))
    with pytest.raises(ValueError, match="adjudication receipt artifact hash mismatch"):
        _finalize(fixture, policy_adjudication_artifacts={digest: wrong})


def test_rejects_binary_pdf_as_reviewed_source_text(tmp_path):
    fixture = _fixture(tmp_path)
    upstream_pdf = tmp_path / "upstream.pdf"
    upstream_pdf.write_bytes(b"%PDF-1.7\nnot a reviewed text snapshot\n")
    manifest = deepcopy(fixture["source_manifest"])
    manifest["artifacts"]["source-en"] = _artifact_declaration(
        upstream_pdf, "reviewed_source_text"
    )
    paths = dict(fixture["artifact_paths"])
    paths["source-en"] = upstream_pdf

    with pytest.raises(ValueError, match="reviewed UTF-8 text snapshot"):
        builder._validate_source_manifest(manifest, paths)


def test_rejects_opaque_text_as_acquisition_binding(tmp_path):
    fixture = _fixture(tmp_path)
    opaque = tmp_path / "not-a-receipt.txt"
    opaque.write_text("not a receipt")
    manifest = deepcopy(fixture["source_manifest"])
    manifest["artifacts"]["binding-en"] = _artifact_declaration(
        opaque, "reviewed_source_acquisition_binding"
    )
    paths = dict(fixture["artifact_paths"])
    paths["binding-en"] = opaque

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        builder._validate_source_manifest(manifest, paths)


def test_rejects_semantic_chunk_gaps_that_fake_long_context(tmp_path):
    fixture = _fixture(tmp_path)
    spec = deepcopy(fixture["spec"])
    spec["cases"][0]["chunks"][1]["token_start"] = 6

    with pytest.raises(ValueError, match="cover the entire dossier contiguously"):
        builder._validate_spec(
            spec, fixture["source_manifest"], allow_test_profile=True
        )


def test_builder_rejects_invented_text_and_reviewed_source_slice_gaps(tmp_path):
    fixture = _fixture(tmp_path)
    invented = deepcopy(fixture["spec"])
    invented["cases"][0]["chunks"][1]["text"] = "invented evidence"
    with pytest.raises(ValueError, match="exactly match reviewed source slice"):
        builder.build_audit_input(
            invented,
            fixture["source_manifest"],
            fixture["annotation_map"],
            fixture["resolved"],
            scorer.artifact_sha256(fixture["tokenizer"]),
            scorer.artifact_sha256(fixture["chunker"]),
            spec_sha256="0" * 64,
            source_manifest_sha256=fixture["hashes"]["source_manifest_sha256"],
            annotations_sha256=fixture["hashes"]["causal_annotations_sha256"],
        )

    gap = deepcopy(fixture["spec"])
    gap["cases"][0]["chunks"][1]["derivation"]["char_start"] = 1
    gap["cases"][0]["chunks"][1]["text"] = fixture["reviewed_en"][
        1 : len(fixture["reviewed_en"]) // 6
    ]
    with pytest.raises(ValueError, match="without gaps"):
        builder.build_audit_input(
            gap,
            fixture["source_manifest"],
            fixture["annotation_map"],
            fixture["resolved"],
            scorer.artifact_sha256(fixture["tokenizer"]),
            scorer.artifact_sha256(fixture["chunker"]),
            spec_sha256="0" * 64,
            source_manifest_sha256=fixture["hashes"]["source_manifest_sha256"],
            annotations_sha256=fixture["hashes"]["causal_annotations_sha256"],
        )


def test_rejects_dossier_without_global_policy_context(tmp_path):
    fixture = _fixture(tmp_path)
    spec = deepcopy(fixture["spec"])
    spec["cases"][0]["chunks"][0]["kind"] = "distractor"
    spec["cases"][0]["chunks"][0]["derivation"] = {
        "kind": "source_text_slice",
        "source_artifact": "source-en",
        "char_start": 0,
        "char_end": 5,
    }

    with pytest.raises(ValueError, match="global policy context"):
        builder._validate_spec(
            spec, fixture["source_manifest"], allow_test_profile=True
        )


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
    with pytest.raises(ValueError, match="status then option_index"):
        builder._validate_annotations(annotations)


def test_rejects_uncleared_or_family_incompatible_license(tmp_path):
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["source_manifest"])
    policy_name = next(iter(manifest["license_policies"]))
    manifest["license_policies"][policy_name]["source_family"] = "permissive"
    manifest["license_policies"][policy_name]["license_id"] = "CC-BY-NC-SA-4.0"
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


def test_fixed_fallback_fails_closed_at_external_tokenizer_boundary(tmp_path):
    fixture = _fixture(tmp_path)
    spec = deepcopy(fixture["spec"])
    case = spec["cases"][0]
    case["chunking"] = {
        "mode": "fixed_512_overlap_64",
        "boundary_receipt_artifact": None,
    }
    with pytest.raises(ValueError, match="external tokenizer boundary"):
        builder._validate_spec(spec, fixture["source_manifest"], True)


def test_refuses_to_replace_existing_bundle(tmp_path):
    fixture = _fixture(tmp_path)
    _finalize(fixture)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _finalize(fixture)
