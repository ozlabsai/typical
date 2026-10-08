import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts import replay_r052_typed_policy as replay
from scripts import score_r052_evidence_retrieval as scorer

TEST_ARTIFACT = Path(__file__).resolve()
TEST_ARTIFACT_SHA256 = scorer.file_sha256(TEST_ARTIFACT)
ACQUISITION_SOURCE = Path("sources/r052/ntsb_gao_v1.json").resolve()
TEST_SOURCE_IDS = ("AAR0002", "GAO-01-203")


@pytest.fixture(autouse=True)
def _compact_acquisition_contract(monkeypatch):
    monkeypatch.setattr(scorer.acquisition, "EXPECTED_IDS", TEST_SOURCE_IDS)


SOURCE_ARTIFACT_PATHS = {
    "reviewed-en": Path("uv.lock").resolve(),
    "reviewed-he": Path("site/style.css").resolve(),
    "boundary-receipt": Path("site/package.json").resolve(),
    "alignment-receipt": Path("site/js/vendor/ScrollTrigger.min.js").resolve(),
    "control-receipt": Path("site/js/vendor/gsap.min.js").resolve(),
    "real-image": Path("site/favicon.png").resolve(),
    "shuffled-image": Path("site/og.png").resolve(),
}

_SOURCE_PROVENANCE_TEMP = tempfile.TemporaryDirectory()
_SOURCE_PROVENANCE_ROOT = Path(_SOURCE_PROVENANCE_TEMP.name)
_EXTRACTED_TEXT_HASHES = {
    language: scorer.sha256_bytes(f"extracted-{language}".encode())
    for language in ("en", "he")
}
_ACQUISITION_MANIFEST = json.loads(ACQUISITION_SOURCE.read_text(encoding="utf-8"))
_ACQUISITION_MANIFEST["snapshots"] = [
    snapshot
    for snapshot in _ACQUISITION_MANIFEST["snapshots"]
    if snapshot["id"] in TEST_SOURCE_IDS
]
_ACQUIRED_BY_ID = {
    snapshot["id"]: snapshot for snapshot in _ACQUISITION_MANIFEST["snapshots"]
}
_LANGUAGE_TO_SOURCE_ID = dict(zip(("en", "he"), TEST_SOURCE_IDS, strict=True))
for _language, _source_id in _LANGUAGE_TO_SOURCE_ID.items():
    _ACQUIRED_BY_ID[_source_id]["extracted_text"].update(
        {
            "sha256": _EXTRACTED_TEXT_HASHES[_language],
            "bytes": 100,
            "words": 10,
        }
    )
_ACQUISITION_MANIFEST_PATH = _SOURCE_PROVENANCE_ROOT / "acquisition-manifest.json"
_ACQUISITION_MANIFEST_PATH.write_bytes(scorer.canonical_json(_ACQUISITION_MANIFEST))
SOURCE_ARTIFACT_PATHS["acquisition-manifest"] = _ACQUISITION_MANIFEST_PATH
_ACQUISITION_MANIFEST_SHA256 = scorer.artifact_sha256(_ACQUISITION_MANIFEST_PATH)
_RUN_RECEIPT = {
    "schema_version": scorer.ACQUISITION_RUN_RECEIPT_VERSION,
    "mode": "offline",
    "manifest_sha256": _ACQUISITION_MANIFEST_SHA256,
    "verified": [
        {
            "id": _LANGUAGE_TO_SOURCE_ID[language],
            "raw_sha256": _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]]["raw_pdf"][
                "sha256"
            ],
            "text_sha256": _EXTRACTED_TEXT_HASHES[language],
        }
        for language in ("en", "he")
    ],
}
_RUN_RECEIPT_PATH = _SOURCE_PROVENANCE_ROOT / "acquisition-run.json"
_RUN_RECEIPT_PATH.write_bytes(scorer.canonical_json(_RUN_RECEIPT))
SOURCE_ARTIFACT_PATHS["acquisition-run"] = _RUN_RECEIPT_PATH
_RUN_RECEIPT_SHA256 = scorer.artifact_sha256(_RUN_RECEIPT_PATH)
for _language in ("en", "he"):
    _source_id = _LANGUAGE_TO_SOURCE_ID[_language]
    _acquired = _ACQUIRED_BY_ID[_source_id]
    _acquired_policy = _ACQUISITION_MANIFEST["license_policies"][
        _acquired["license_policy"]
    ]
    _reviewed_path = SOURCE_ARTIFACT_PATHS[f"reviewed-{_language}"]
    _reviewed_text = _reviewed_path.read_text(encoding="utf-8")
    _binding = {
        "schema_version": scorer.ACQUISITION_BINDING_VERSION,
        "snapshot_id": _source_id,
        "acquisition_manifest_sha256": _ACQUISITION_MANIFEST_SHA256,
        "acquisition_receipt_sha256": _RUN_RECEIPT_SHA256,
        "extracted_text_sha256": _EXTRACTED_TEXT_HASHES[_language],
        "review_receipt_sha256": scorer.sha256_bytes(b"review-receipt"),
        "retained_ranges": [{"char_start": 0, "char_end": len(_reviewed_text)}],
        "range_mappings": [
            {
                "source_char_start": 0,
                "source_char_end": len(_reviewed_text),
                "output_char_start": 0,
                "output_char_end": len(_reviewed_text),
            }
        ],
        "reviewed_character_count": len(_reviewed_text),
        "reviewed_unicode_word_token_count": len(
            scorer.TOKEN_RE.findall(_reviewed_text)
        ),
        "word_token_count_method": scorer.WORD_TOKEN_COUNT_METHOD,
        "reviewed_text_sha256": scorer.artifact_sha256(_reviewed_path),
        "upstream_url": _acquired["upstream_url"],
        "upstream_revision": _acquired["upstream_revision"],
        "license_policy": _acquired["license_policy"],
        "attribution": _acquired_policy["attribution"],
    }
    _binding_path = _SOURCE_PROVENANCE_ROOT / f"binding-{_language}.json"
    _binding_path.write_bytes(scorer.canonical_json(_binding))
    SOURCE_ARTIFACT_PATHS[f"binding-{_language}"] = _binding_path
SOURCE_ARTIFACT_HASHES = {
    name: scorer.artifact_sha256(path) for name, path in SOURCE_ARTIFACT_PATHS.items()
}


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


def _artifact_media(name: str) -> dict:
    import base64
    import mimetypes

    path = SOURCE_ARTIFACT_PATHS[name]
    raw = path.read_bytes()
    return {
        "mime_type": mimetypes.guess_type(path.name)[0],
        "data_base64": base64.b64encode(raw).decode(),
        "sha256": scorer.sha256_bytes(raw),
    }


def _chunk(
    key: str,
    kind: str,
    text: str,
    anchor: str,
    *,
    media: dict | None = None,
    control: dict | None = None,
    derivation: dict | None = None,
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
        "derivation": derivation,
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
    full_outcome = {"status": "resolved", "option_index": 0}
    removed_outcome = {"status": "unresolved"}
    question = {
        "question_key": key,
        "text": text,
        "options": ["allow", "deny"],
        "option_ids": ["allow", "deny"],
        "decision_type": "choice",
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
            "changed_fields": ["status", "option_index"],
        },
        "content_sha256": "",
    }
    question["content_sha256"] = scorer.sha256_json(
        scorer.question_hash_payload(question)
    )
    return question


def _case(
    language: str,
    twin_group: str,
    english_chunks: dict[str, dict] | None = None,
) -> dict:
    suffix = language
    control_group = _sha("image-control-" + language)
    control = lambda variant: {"group_key": control_group, "variant": variant}
    translations = {
        "en": {
            "policy": "Always apply the authorization policy.",
            "evidence": "The signed approval code is cobalt.",
            "distractors": [
                "The unsigned memo discusses lunch.",
                "The archive is closed.",
                "The courier arrives tomorrow.",
                "The meeting room is empty.",
            ],
            "image": "Inspect the pictured approval seal.",
            "q_image": "Does the image contain the valid approval seal?",
            "q_text": "Is the signed approval present?",
        },
        "he": {
            "policy": "יש להחיל תמיד את מדיניות ההרשאה.",
            "evidence": "קוד האישור החתום הוא קובלט.",
            "distractors": [
                "המזכר הלא חתום דן בארוחת צהריים.",
                "הארכיון סגור.",
                "השליח מגיע מחר.",
                "חדר הישיבות ריק.",
            ],
            "image": "בדקו את חותמת האישור שבתמונה.",
            "q_image": "האם בתמונה מופיעה חותמת האישור התקפה?",
            "q_text": "האם קיים אישור חתום?",
        },
    }[language]
    reviewed_en = SOURCE_ARTIFACT_PATHS["reviewed-en"].read_text(encoding="utf-8")
    split_points = [len(reviewed_en) * index // 5 for index in range(6)]
    if language == "en":
        translations["evidence"] = reviewed_en[split_points[0] : split_points[1]]
        translations["distractors"] = [
            reviewed_en[split_points[index] : split_points[index + 1]]
            for index in range(1, 5)
        ]

    def text_derivation(anchor: str, start: int, end: int) -> dict:
        if language == "en":
            exact = reviewed_en[start:end]
            return {
                "kind": "source_text_slice",
                "source_artifact_sha256": SOURCE_ARTIFACT_HASHES["reviewed-en"],
                "char_start": start,
                "char_end": end,
                "slice_sha256": scorer.sha256_bytes(exact.encode("utf-8")),
            }
        assert english_chunks is not None
        source = english_chunks[anchor]
        return {
            "kind": "translated_twin",
            "source_twin_chunk_key": source["chunk_key"],
            "source_twin_content_sha256": source["content_sha256"],
            "alignment_receipt_artifact_sha256": SOURCE_ARTIFACT_HASHES[
                "alignment-receipt"
            ],
        }

    def image_derivation(name: str | None) -> dict:
        return {
            "kind": "image_control",
            "image_artifact_sha256": None
            if name is None
            else SOURCE_ARTIFACT_HASHES[name],
            "control_receipt_artifact_sha256": SOURCE_ARTIFACT_HASHES[
                "control-receipt"
            ],
        }

    chunks = [
        _chunk(
            f"policy-{suffix}",
            "global_policy",
            translations["policy"],
            "policy",
            derivation={"kind": "non_source_global_policy"},
        ),
        _chunk(
            f"evidence-{suffix}",
            "evidence",
            translations["evidence"],
            "text-evidence",
            derivation=text_derivation(
                "text-evidence", split_points[0], split_points[1]
            ),
        ),
        *[
            _chunk(
                f"distractor-{index}-{suffix}",
                "distractor",
                translations["distractors"][index - 1],
                f"text-distractor-{index}",
                derivation=text_derivation(
                    f"text-distractor-{index}",
                    split_points[index],
                    split_points[index + 1],
                ),
            )
            for index in range(1, 5)
        ],
        _chunk(
            f"real-{suffix}",
            "evidence",
            translations["image"],
            "image-real",
            media=_artifact_media("real-image"),
            control=control("real"),
            derivation=image_derivation("real-image"),
        ),
        _chunk(
            f"shuffled-{suffix}",
            "distractor",
            translations["image"],
            "image-shuffled",
            media=_artifact_media("shuffled-image"),
            control=control("shuffled"),
            derivation=image_derivation("shuffled-image"),
        ),
        _chunk(
            f"missing-{suffix}",
            "distractor",
            translations["image"],
            "image-missing",
            control=control("missing"),
            derivation=image_derivation(None),
        ),
    ]
    boundaries = [0, 960, 1920, 2880, 3840, 4800, 5760, 6720, 7680]
    for index, chunk in enumerate(chunks):
        chunk["token_start"] = boundaries[index]
        chunk["token_end"] = (
            8192 if index == len(chunks) - 1 else boundaries[index] + 512
        )
        chunk["content_sha256"] = scorer.sha256_json(scorer.chunk_hash_payload(chunk))
    chunking_receipt = {
        "schema_version": scorer.CHUNKING_RECEIPT_VERSION,
        "mode": "semantic_sections",
        "max_chunk_tokens": 512,
        "fallback_overlap_tokens": 64,
        "tokenizer_artifact_sha256": TEST_ARTIFACT_SHA256,
        "chunker_artifact_sha256": TEST_ARTIFACT_SHA256,
        "boundary_receipt_artifact_sha256": SOURCE_ARTIFACT_HASHES["boundary-receipt"],
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
            {"chunk_key": chunk["chunk_key"], "derivation": chunk["derivation"]}
            for chunk in chunks
        ],
    }
    chunking_receipt["receipt_sha256"] = scorer.sha256_json(chunking_receipt)
    result = {
        "case_key": _sha("case-" + language),
        "language": language,
        "twin_group_key": twin_group,
        "source_family": "public_domain",
        "source_snapshot_sha256": SOURCE_ARTIFACT_HASHES[f"reviewed-{language}"],
        "source_domain": "general",
        "context_tokens": 8192,
        "chunking_receipt": chunking_receipt,
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
                    *[f"distractor-{index}-{suffix}" for index in range(1, 5)],
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
                    *[f"distractor-{index}-{suffix}" for index in range(1, 5)],
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
    twin_group = _sha("twins")
    english = _case("en", twin_group)
    english_chunks = {chunk["twin_anchor"]: chunk for chunk in english["chunks"]}
    cases = [english, _case("he", twin_group, english_chunks)]
    query_vectors = {}
    document_vectors = {}
    for case in cases:
        for question in case["questions"]:
            query_vectors[scorer.retrieval_query_sha256(question)] = (
                [1.0, 0.0, 0.0, 0.0]
                if question["twin_anchor"] == "q-image"
                else [0.0, 1.0, 0.0, 0.0]
            )
        for chunk in case["chunks"]:
            vector = {
                "policy": [0.1, 0.1, 0.9, 0.0],
                "text-evidence": [0.0, 1.0, 0.0, 0.0],
                "text-distractor-1": [0.0, -1.0, 0.0, 0.0],
                "text-distractor-2": [0.0, -1.0, 0.0, 0.0],
                "text-distractor-3": [0.0, -1.0, 0.0, 0.0],
                "text-distractor-4": [0.0, -1.0, 0.0, 0.0],
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


def _source_manifest() -> tuple[dict, str]:
    artifacts = {
        name: {
            "role": (
                "reviewed_source_text"
                if name.startswith("reviewed-")
                else "source_acquisition_manifest"
                if name == "acquisition-manifest"
                else "source_acquisition_run_receipt"
                if name == "acquisition-run"
                else "reviewed_source_acquisition_binding"
                if name.startswith("binding-")
                else "translation_alignment_receipt"
                if name == "alignment-receipt"
                else "image_control_receipt"
                if name == "control-receipt"
                else "image"
                if name.endswith("-image")
                else "chunk_boundary_receipt"
            ),
            "sha256": SOURCE_ARTIFACT_HASHES[name],
        }
        for name in SOURCE_ARTIFACT_PATHS
    }
    license_policies = {
        name: {
            "source_family": policy["source_family"],
            "license_id": "LicenseRef-USGov-Public-Domain",
            "license_url": policy["policy_url"],
            "attribution": policy["attribution"],
            "license_cleared": True,
            "evaluation_use_cleared": True,
            "redistribution_cleared": True,
        }
        for name, policy in _ACQUISITION_MANIFEST["license_policies"].items()
    }
    snapshots = {
        _LANGUAGE_TO_SOURCE_ID[language]: {
            "reviewed_text_artifact": f"reviewed-{language}",
            "acquisition_binding_artifact": f"binding-{language}",
            "source_family": _ACQUISITION_MANIFEST["license_policies"][
                _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]]["license_policy"]
            ]["source_family"],
            "upstream_url": _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]][
                "upstream_url"
            ],
            "upstream_revision": _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]][
                "upstream_revision"
            ],
            "upstream_date": _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]][
                "public_release_date"
            ]
            or _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]][
                "adoption_or_issue_date"
            ],
            "untouched_split": _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]][
                "untouched_split"
            ],
            "license_policy": _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]][
                "license_policy"
            ],
            "attribution": _ACQUISITION_MANIFEST["license_policies"][
                _ACQUIRED_BY_ID[_LANGUAGE_TO_SOURCE_ID[language]]["license_policy"]
            ]["attribution"],
            "third_party_images_excluded": True,
            "natural_language_authored": True,
            "benchmark_derived": False,
        }
        for language in ("en", "he")
    }
    manifest = {
        "schema_version": scorer.SOURCE_MANIFEST_VERSION,
        "license_policy": scorer.SOURCE_LICENSE_POLICY,
        "benchmark_derived": False,
        "license_policies": license_policies,
        "artifacts": artifacts,
        "acquisition_manifest_artifact": "acquisition-manifest",
        "acquisition_run_receipt_artifact": "acquisition-run",
        "snapshots": snapshots,
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


def _policy_artifacts(panel: dict) -> dict:
    draft = scorer.policy_replay_draft(panel)
    panel_digest = scorer.sha256_json(draft)
    facts = {
        "f0001": {"type": "boolean", "unit": None},
        "f0002": {"type": "boolean", "unit": None},
    }
    bindings = []
    for case in panel["cases"]:
        by_anchor = {chunk["twin_anchor"]: chunk for chunk in case["chunks"]}
        for fact, anchor in (("f0001", "image-real"), ("f0002", "text-evidence")):
            chunk = by_anchor[anchor]
            locator = (
                {
                    "kind": "image_region",
                    "region_sha256": chunk["media"]["sha256"],
                }
                if chunk["modality"] == "image"
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
                    "adjudication_receipt_sha256": TEST_ARTIFACT_SHA256,
                }
            )
    ledger = {
        "schema_version": replay.LEDGER_VERSION,
        "panel_sha256": panel_digest,
        "facts": facts,
        "bindings": bindings,
    }
    ledger_digest = scorer.sha256_json(ledger)
    groups = []
    for anchor, fact in (("q-image", "f0001"), ("q-text", "f0002")):
        decision_id = f"d.{anchor}"
        presentations = []
        for case in panel["cases"]:
            question = next(q for q in case["questions"] if q["twin_anchor"] == anchor)
            full = [chunk["chunk_key"] for chunk in case["chunks"]]
            immutable = {
                chunk["chunk_key"] for chunk in case["chunks"] if chunk["immutable"]
            }
            decisive = question["decisive_chunk_keys"]
            distractors = question["known_distractor_chunk_keys"]
            presentations.append(
                {
                    "presentation_id": f"p.{anchor}.{case['language']}",
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
                            "assert_full": {"status": "resolved", "output": "allow"},
                            "assert_removed": {"status": "unresolved"},
                        }
                    ],
                }
            )
        groups.append(
            {
                "group_id": f"g.{anchor}",
                "facts": {fact: facts[fact]},
                "decisions": [
                    {
                        "decision_id": decision_id,
                        "type": "choice",
                        "outputs": [
                            {"id": "allow", "label": "allow"},
                            {"id": "deny", "label": "deny"},
                        ],
                        "order": ["allow", "deny"],
                        "rules": [
                            {
                                "output": "allow",
                                "when": {"op": "fact", "id": fact},
                            },
                            {
                                "output": "deny",
                                "when": {
                                    "op": "not",
                                    "arg": {"op": "fact", "id": fact},
                                },
                            },
                        ],
                    }
                ],
                "presentations": presentations,
            }
        )
    rubric = {
        "schema_version": replay.RUBRIC_VERSION,
        "panel_sha256": panel_digest,
        "fact_ledger_sha256": ledger_digest,
        "semantic_groups": groups,
    }
    rubric_digest = scorer.sha256_json(rubric)
    evaluator_digest = scorer.artifact_sha256(TEST_ARTIFACT)
    schema_path = Path("sources/r052/policy_replay.schema.json").resolve()
    schema_digest = scorer.artifact_sha256(schema_path)
    receipt = replay.replay(
        draft,
        ledger,
        rubric,
        panel_sha256=panel_digest,
        ledger_sha256=ledger_digest,
        rubric_sha256=rubric_digest,
        evaluator_sha256=evaluator_digest,
        schema_sha256=schema_digest,
        adjudication_receipts={TEST_ARTIFACT_SHA256: TEST_ARTIFACT.read_bytes()},
    )
    return {
        "policy_fact_ledger": ledger,
        "policy_fact_ledger_sha256": ledger_digest,
        "policy_rubric": rubric,
        "policy_rubric_sha256": rubric_digest,
        "policy_replay_receipt": receipt,
        "policy_replay_receipt_sha256": scorer.sha256_json(receipt),
        "policy_replay_schema_artifact": schema_path,
        "policy_replay_schema_sha256": schema_digest,
        "policy_evaluator_artifact": TEST_ARTIFACT,
        "policy_evaluator_sha256": evaluator_digest,
        "policy_adjudication_artifacts": {TEST_ARTIFACT_SHA256: TEST_ARTIFACT},
    }


def _validate(panel: dict) -> dict:
    source, source_digest = _source_manifest()
    causal, causal_digest = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    try:
        policy = _policy_artifacts(panel)
    except replay.PolicyError:
        # Structural mutation tests must reach the scorer's earlier schema checks.
        policy = _policy_artifacts(make_panel())
    return scorer.validate_panel(
        panel,
        source,
        source_digest,
        causal,
        causal_digest,
        contamination,
        contamination_digest,
        annotation_builder_artifact=TEST_ARTIFACT,
        contamination_auditor_artifact=TEST_ARTIFACT,
        contamination_corpus_artifacts=[TEST_ARTIFACT],
        contamination_tokenizer_artifact=TEST_ARTIFACT,
        contamination_chunker_artifact=TEST_ARTIFACT,
        source_snapshot_artifacts=SOURCE_ARTIFACT_PATHS,
        provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
        **policy,
    )


def _validate_selection(
    panel: dict, selected: set[str], artifacts: dict[str, Path]
) -> dict:
    source, source_digest = _source_manifest()
    causal, causal_digest = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    return scorer.validate_panel(
        panel,
        source,
        source_digest,
        causal,
        causal_digest,
        contamination,
        contamination_digest,
        annotation_builder_artifact=TEST_ARTIFACT,
        contamination_auditor_artifact=TEST_ARTIFACT,
        contamination_corpus_artifacts=[TEST_ARTIFACT],
        contamination_tokenizer_artifact=TEST_ARTIFACT,
        contamination_chunker_artifact=TEST_ARTIFACT,
        source_snapshot_artifacts=SOURCE_ARTIFACT_PATHS,
        provider_generator_artifacts=artifacts,
        selected_precomputed_providers=selected,
        **_policy_artifacts(panel),
    )


def _validate_replay_only(panel: dict, **overrides) -> str:
    values = overrides.pop("_base_policy", None) or _policy_artifacts(panel)
    values.update(overrides)
    return scorer.validate_policy_replay(
        panel,
        values["policy_fact_ledger"],
        values["policy_fact_ledger_sha256"],
        values["policy_rubric"],
        values["policy_rubric_sha256"],
        values["policy_replay_receipt"],
        values["policy_replay_receipt_sha256"],
        replay_schema_artifact=values["policy_replay_schema_artifact"],
        replay_schema_sha256=values["policy_replay_schema_sha256"],
        evaluator_artifact=values["policy_evaluator_artifact"],
        evaluator_sha256=values["policy_evaluator_sha256"],
        adjudication_artifacts=values["policy_adjudication_artifacts"],
    )


@pytest.fixture
def panel() -> dict:
    return make_panel()


def test_scorer_requires_all_typed_replay_artifacts(panel):
    source, source_digest = _source_manifest()
    causal, causal_digest = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    with pytest.raises(ValueError, match="all typed policy replay artifacts"):
        scorer.validate_panel(
            panel,
            source,
            source_digest,
            causal,
            causal_digest,
            contamination,
            contamination_digest,
            annotation_builder_artifact=TEST_ARTIFACT,
            contamination_auditor_artifact=TEST_ARTIFACT,
            contamination_corpus_artifacts=[TEST_ARTIFACT],
            contamination_tokenizer_artifact=TEST_ARTIFACT,
            contamination_chunker_artifact=TEST_ARTIFACT,
            source_snapshot_artifacts=SOURCE_ARTIFACT_PATHS,
            provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
        )


def test_replay_rejects_option_id_order_tamper_and_authored_proof_drift(panel):
    policy = _policy_artifacts(panel)
    rubric = copy.deepcopy(policy["policy_rubric"])
    decision = rubric["semantic_groups"][0]["decisions"][0]
    decision["outputs"].reverse()
    decision["order"].reverse()
    rubric_digest = scorer.sha256_json(rubric)
    with pytest.raises(ValueError, match="output ids/order"):
        _validate_replay_only(
            panel,
            policy_rubric=rubric,
            policy_rubric_sha256=rubric_digest,
        )

    changed = copy.deepcopy(panel)
    proof = changed["cases"][0]["questions"][0]["causal_proof"]
    proof["full_outcome"] = {"status": "resolved", "option_index": 1}
    proof["full_outcome_sha256"] = scorer.sha256_json(proof["full_outcome"])
    resealed = copy.deepcopy(policy)
    changed_panel_digest = scorer.sha256_json(scorer.policy_replay_draft(changed))
    resealed["policy_fact_ledger"]["panel_sha256"] = changed_panel_digest
    resealed["policy_fact_ledger_sha256"] = scorer.sha256_json(
        resealed["policy_fact_ledger"]
    )
    resealed["policy_rubric"]["panel_sha256"] = changed_panel_digest
    resealed["policy_rubric"]["fact_ledger_sha256"] = resealed[
        "policy_fact_ledger_sha256"
    ]
    resealed["policy_rubric_sha256"] = scorer.sha256_json(resealed["policy_rubric"])
    with pytest.raises(ValueError, match="contradicts authoritative replay"):
        _validate_replay_only(changed, _base_policy=resealed)


def test_replay_rejects_omitted_and_duplicate_question_coverage(panel):
    policy = _policy_artifacts(panel)
    omitted = copy.deepcopy(policy["policy_rubric"])
    omitted["semantic_groups"][0]["presentations"] = omitted["semantic_groups"][0][
        "presentations"
    ][:1]
    with pytest.raises(ValueError, match="every panel question|Hebrew presentation"):
        _validate_replay_only(
            panel,
            policy_rubric=omitted,
            policy_rubric_sha256=scorer.sha256_json(omitted),
        )

    duplicate = copy.deepcopy(policy["policy_rubric"])
    copied = copy.deepcopy(duplicate["semantic_groups"][0]["presentations"][0])
    copied["presentation_id"] = "p.duplicate"
    duplicate["semantic_groups"][0]["presentations"].append(copied)
    with pytest.raises(ValueError, match="reused by multiple decisions"):
        _validate_replay_only(
            panel,
            policy_rubric=duplicate,
            policy_rubric_sha256=scorer.sha256_json(duplicate),
        )


def test_replay_rejects_bound_artifact_and_receipt_tampering(panel, tmp_path):
    policy = _policy_artifacts(panel)
    with pytest.raises(ValueError, match="fact ledger does not match"):
        _validate_replay_only(panel, policy_fact_ledger_sha256="0" * 64)
    with pytest.raises(ValueError, match="policy rubric does not match"):
        _validate_replay_only(panel, policy_rubric_sha256="0" * 64)
    with pytest.raises(ValueError, match="schema artifact hash mismatch"):
        _validate_replay_only(panel, policy_replay_schema_sha256="0" * 64)
    with pytest.raises(ValueError, match="evaluator artifact hash mismatch"):
        _validate_replay_only(panel, policy_evaluator_sha256="0" * 64)
    wrong_adjudication = tmp_path / "wrong-adjudication.txt"
    wrong_adjudication.write_text("tampered")
    with pytest.raises(ValueError, match="adjudication receipt artifact hash mismatch"):
        _validate_replay_only(
            panel,
            policy_adjudication_artifacts={
                next(iter(policy["policy_adjudication_artifacts"])): wrong_adjudication
            },
        )
    receipt = copy.deepcopy(policy["policy_replay_receipt"])
    receipt["schema_version"] = "tampered"
    with pytest.raises(ValueError, match="does not match deterministic replay"):
        _validate_replay_only(
            panel,
            policy_replay_receipt=receipt,
            policy_replay_receipt_sha256=scorer.sha256_json(receipt),
        )


def test_hebrew_option_ids_must_equal_english_twin(panel):
    changed = copy.deepcopy(panel)
    question = changed["cases"][1]["questions"][0]
    question["option_ids"].reverse()
    question["content_sha256"] = scorer.sha256_json(
        scorer.question_hash_payload(question)
    )
    proof = question["causal_proof"]
    proof["full_context_sha256"] = scorer.causal_context_sha256(
        changed["cases"][1], question, set()
    )
    proof["removed_context_sha256"] = scorer.causal_context_sha256(
        changed["cases"][1], question, set(question["decisive_chunk_keys"])
    )
    with pytest.raises(ValueError, match="paired questions"):
        _validate(changed)


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
    assert result["slices"]["decision_type"]["choice"]["decisive_recall@1"] == 1.0
    assert result["slices"]["decisive_modality"]["image"]["decisive_recall@1"] == 1.0


def test_bm25_is_unicode_deterministic_and_reports_controls(panel):
    _validate(panel)
    result = scorer.score_method(panel, "bm25", (1, 2, 4))
    assert result == scorer.score_method(panel, "bm25", (1, 2, 4))
    assert result["overall"]["questions"] == 4
    assert result["image_margin"]["controls"] == 2
    assert scorer.tokenize("אישור חתום") == ["אישור", "חתום"]


def test_retrieval_query_is_invariant_to_parallel_option_permutation(panel):
    question = panel["cases"][0]["questions"][0]
    permuted = copy.deepcopy(question)
    permuted["options"].reverse()
    permuted["option_ids"].reverse()
    permuted["content_sha256"] = scorer.sha256_json(
        scorer.question_hash_payload(permuted)
    )
    assert question["content_sha256"] != permuted["content_sha256"]
    assert scorer.LocalEmbeddingGemma2Provider._query_text(
        question
    ) == scorer.LocalEmbeddingGemma2Provider._query_text(permuted)
    assert scorer.LocalEmbeddingGemma2Provider._query_cache_key(
        question
    ) == scorer.LocalEmbeddingGemma2Provider._query_cache_key(permuted)
    assert scorer.retrieval_query_sha256(question) == scorer.retrieval_query_sha256(
        permuted
    )
    provider = scorer.PrecomputedProvider(
        "fixture-dense", panel["providers"]["fixture-dense"], None
    )
    assert np.array_equal(provider.query(question), provider.query(permuted))
    candidates = [
        chunk for chunk in panel["cases"][0]["chunks"] if not chunk["immutable"]
    ]
    assert scorer.bm25_rank(question, candidates) == scorer.bm25_rank(
        permuted, candidates
    )


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
            lambda value: value["cases"][0]["chunks"][6]["image_control"].update(
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
    source, source_digest = _source_manifest()
    manifest, _ = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    changed_manifest = copy.deepcopy(manifest)
    first = next(iter(changed_manifest["entries"].values()))
    first["changed_fields"] = ["authorization", "unbound-field"]
    with pytest.raises(ValueError, match="manifest entries"):
        scorer.validate_panel(
            panel,
            source,
            source_digest,
            changed_manifest,
            scorer.sha256_json(changed_manifest),
            contamination,
            contamination_digest,
            annotation_builder_artifact=TEST_ARTIFACT,
            contamination_auditor_artifact=TEST_ARTIFACT,
            contamination_corpus_artifacts=[TEST_ARTIFACT],
            contamination_tokenizer_artifact=TEST_ARTIFACT,
            contamination_chunker_artifact=TEST_ARTIFACT,
            source_snapshot_artifacts=SOURCE_ARTIFACT_PATHS,
            provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
            **_policy_artifacts(panel),
        )
    with pytest.raises(ValueError, match="external sha256 anchor"):
        scorer.validate_panel(
            panel,
            source,
            source_digest,
            manifest,
            "0" * 64,
            contamination,
            contamination_digest,
            annotation_builder_artifact=TEST_ARTIFACT,
            contamination_auditor_artifact=TEST_ARTIFACT,
            contamination_corpus_artifacts=[TEST_ARTIFACT],
            contamination_tokenizer_artifact=TEST_ARTIFACT,
            contamination_chunker_artifact=TEST_ARTIFACT,
            source_snapshot_artifacts=SOURCE_ARTIFACT_PATHS,
            provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
            **_policy_artifacts(panel),
        )


def test_causal_context_hashes_are_recomputed_from_ordered_content(panel):
    changed = copy.deepcopy(panel)
    chunk = changed["cases"][0]["chunks"][1]
    chunk["text"] += " changed"
    chunk["content_sha256"] = scorer.sha256_json(scorer.chunk_hash_payload(chunk))
    with pytest.raises(ValueError, match="exactly equal its reviewed source slice"):
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


def test_provider_artifacts_are_optional_for_bm25_and_exact_for_selected(panel):
    _validate_selection(panel, set(), {})
    bm25_only = copy.deepcopy(panel)
    bm25_only["providers"] = {}
    _validate_selection(bm25_only, set(), {})
    with pytest.raises(ValueError, match="cover selected providers exactly"):
        _validate_selection(panel, {"fixture-dense"}, {})
    with pytest.raises(ValueError, match="cover selected providers exactly"):
        _validate_selection(panel, set(), {"fixture-dense": TEST_ARTIFACT})


def test_source_snapshots_bind_reviewed_acquisition_metadata_and_case_family(
    panel, tmp_path
):
    manifest, digest = _source_manifest()
    _, by_hash = scorer.validate_source_snapshot_manifest(
        panel, manifest, digest, SOURCE_ARTIFACT_PATHS
    )
    assert (
        by_hash[panel["cases"][0]["source_snapshot_sha256"]]["role"]
        == "reviewed_source_text"
    )

    missing_path = dict(SOURCE_ARTIFACT_PATHS)
    missing_path.pop("reviewed-en")
    with pytest.raises(ValueError, match="cover manifest exactly"):
        scorer.validate_source_snapshot_manifest(panel, manifest, digest, missing_path)

    duplicate = copy.deepcopy(manifest)
    duplicate["artifacts"]["reviewed-he"]["sha256"] = duplicate["artifacts"][
        "reviewed-en"
    ]["sha256"]
    with pytest.raises(ValueError, match="duplicate artifact hashes"):
        scorer.validate_source_snapshot_manifest(
            panel,
            duplicate,
            scorer.sha256_json(duplicate),
            SOURCE_ARTIFACT_PATHS,
        )

    duplicate_binding = copy.deepcopy(manifest)
    duplicate_binding["snapshots"][TEST_SOURCE_IDS[1]]["reviewed_text_artifact"] = (
        "reviewed-en"
    )
    with pytest.raises(ValueError, match="duplicate reviewed/acquisition bindings"):
        scorer.validate_source_snapshot_manifest(
            panel,
            duplicate_binding,
            scorer.sha256_json(duplicate_binding),
            SOURCE_ARTIFACT_PATHS,
        )

    upstream_pdf = tmp_path / "upstream.pdf"
    upstream_pdf.write_bytes(b"%PDF-1.7\nnot a reviewed text snapshot\n")
    binary_paths = dict(SOURCE_ARTIFACT_PATHS)
    binary_paths["reviewed-en"] = upstream_pdf
    binary_manifest = copy.deepcopy(manifest)
    binary_manifest["artifacts"]["reviewed-en"]["sha256"] = scorer.file_sha256(
        upstream_pdf
    )
    with pytest.raises(ValueError, match="reviewed UTF-8 text snapshot"):
        scorer.validate_source_snapshot_manifest(
            panel,
            binary_manifest,
            scorer.sha256_json(binary_manifest),
            binary_paths,
        )

    mismatched = copy.deepcopy(panel)
    mismatched["cases"][0]["source_snapshot_sha256"] = SOURCE_ARTIFACT_HASHES[
        "boundary-receipt"
    ]
    with pytest.raises(ValueError, match="cover source snapshots exactly"):
        scorer.validate_source_snapshot_manifest(
            mismatched, manifest, digest, SOURCE_ARTIFACT_PATHS
        )

    wrong_family = copy.deepcopy(panel)
    wrong_family["cases"][0]["source_family"] = "permissive"
    with pytest.raises(ValueError, match="family mismatches"):
        scorer.validate_source_snapshot_manifest(
            wrong_family, manifest, digest, SOURCE_ARTIFACT_PATHS
        )

    hostile = tmp_path / "extracted.txt"
    hostile.write_text("copied from a JevBench example")
    hostile_manifest = copy.deepcopy(manifest)
    hostile_manifest["artifacts"]["reviewed-en"]["sha256"] = scorer.artifact_sha256(
        hostile
    )
    hostile_paths = dict(SOURCE_ARTIFACT_PATHS)
    hostile_paths["reviewed-en"] = hostile
    with pytest.raises(ValueError, match="contains benchmark lineage"):
        scorer.validate_source_snapshot_manifest(
            panel,
            hostile_manifest,
            scorer.sha256_json(hostile_manifest),
            hostile_paths,
        )


def _rewrite_source_json(
    tmp_path: Path,
    manifest: dict,
    paths: dict[str, Path],
    name: str,
    mutate,
) -> tuple[dict, dict[str, Path]]:
    changed_manifest = copy.deepcopy(manifest)
    changed_paths = dict(paths)
    value = json.loads(changed_paths[name].read_bytes())
    mutate(value)
    path = tmp_path / f"changed-{name}.json"
    path.write_bytes(scorer.canonical_json(value))
    changed_paths[name] = path
    changed_manifest["artifacts"][name]["sha256"] = scorer.artifact_sha256(path)
    return changed_manifest, changed_paths


def test_acquisition_manifest_rejects_self_consistent_wrapper_rewrites(panel, tmp_path):
    manifest, _ = _source_manifest()
    changed = copy.deepcopy(manifest)
    changed_paths = dict(SOURCE_ARTIFACT_PATHS)
    source_id = TEST_SOURCE_IDS[0]
    changed["snapshots"][source_id]["upstream_url"] = (
        "https://example.invalid/investigations/AccidentReports/Reports/AAR0002.pdf"
    )
    changed["snapshots"][source_id]["upstream_revision"] = "immutable-rewrite-v2"
    binding = json.loads(changed_paths["binding-en"].read_bytes())
    binding["upstream_url"] = changed["snapshots"][source_id]["upstream_url"]
    binding["upstream_revision"] = changed["snapshots"][source_id]["upstream_revision"]
    binding_path = tmp_path / "rewritten-binding.json"
    binding_path.write_bytes(scorer.canonical_json(binding))
    changed_paths["binding-en"] = binding_path
    changed["artifacts"]["binding-en"]["sha256"] = scorer.artifact_sha256(binding_path)

    with pytest.raises(ValueError, match="metadata mismatches acquisition manifest"):
        scorer.validate_source_snapshot_manifest(
            panel, changed, scorer.sha256_json(changed), changed_paths
        )


def test_acquisition_manifest_bytes_and_exact_record_coverage_are_authoritative(
    panel, tmp_path
):
    manifest, _ = _source_manifest()
    substituted = copy.deepcopy(_ACQUISITION_MANIFEST)
    substituted["purpose"] += " rewritten"
    substituted_path = tmp_path / "substituted-acquisition.json"
    substituted_path.write_bytes(scorer.canonical_json(substituted))
    changed = copy.deepcopy(manifest)
    changed_paths = dict(SOURCE_ARTIFACT_PATHS)
    changed_paths["acquisition-manifest"] = substituted_path
    changed["artifacts"]["acquisition-manifest"]["sha256"] = scorer.artifact_sha256(
        substituted_path
    )
    with pytest.raises(ValueError, match="does not bind exact manifest bytes"):
        scorer.validate_source_snapshot_manifest(
            panel, changed, scorer.sha256_json(changed), changed_paths
        )

    for suffix, mutate in (
        ("missing", lambda value: value["snapshots"].pop()),
        (
            "extra",
            lambda value: value["snapshots"].append(
                copy.deepcopy(value["snapshots"][0])
            ),
        ),
    ):
        value = copy.deepcopy(_ACQUISITION_MANIFEST)
        mutate(value)
        path = tmp_path / f"{suffix}-acquisition.json"
        path.write_bytes(scorer.canonical_json(value))
        changed = copy.deepcopy(manifest)
        changed_paths = dict(SOURCE_ARTIFACT_PATHS)
        changed_paths["acquisition-manifest"] = path
        changed["artifacts"]["acquisition-manifest"]["sha256"] = scorer.artifact_sha256(
            path
        )
        with pytest.raises(ValueError, match="source acquisition manifest is invalid"):
            scorer.validate_source_snapshot_manifest(
                panel, changed, scorer.sha256_json(changed), changed_paths
            )


def test_acquisition_license_attribution_rewrite_is_rejected(panel, tmp_path):
    manifest, _ = _source_manifest()
    changed = copy.deepcopy(manifest)
    changed_paths = dict(SOURCE_ARTIFACT_PATHS)
    policy_name = "ntsb_staff_text"
    rewritten = "Self-consistent but unauthorized attribution"
    changed["license_policies"][policy_name]["attribution"] = rewritten
    changed["snapshots"][TEST_SOURCE_IDS[0]]["attribution"] = rewritten
    binding = json.loads(changed_paths["binding-en"].read_bytes())
    binding["attribution"] = rewritten
    binding_path = tmp_path / "rewritten-attribution-binding.json"
    binding_path.write_bytes(scorer.canonical_json(binding))
    changed_paths["binding-en"] = binding_path
    changed["artifacts"]["binding-en"]["sha256"] = scorer.artifact_sha256(binding_path)
    with pytest.raises(ValueError, match="license policy mismatches acquisition"):
        scorer.validate_source_snapshot_manifest(
            panel, changed, scorer.sha256_json(changed), changed_paths
        )


def test_acquisition_run_receipt_rejects_non_receipt_and_bad_entries(panel, tmp_path):
    manifest, _ = _source_manifest()
    paths = dict(SOURCE_ARTIFACT_PATHS)
    opaque = tmp_path / "not-a-receipt.txt"
    opaque.write_text("not a receipt")
    paths["acquisition-run"] = opaque
    manifest["artifacts"]["acquisition-run"]["sha256"] = scorer.artifact_sha256(opaque)
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        scorer.validate_source_snapshot_manifest(
            panel, manifest, scorer.sha256_json(manifest), paths
        )

    for mutate, message in (
        (
            lambda value: value["verified"][0].update(raw_sha256="0" * 63),
            "raw_sha256",
        ),
        (
            lambda value: value["verified"].pop(),
            "cover every snapshot",
        ),
        (
            lambda value: value["verified"][1].update(id=TEST_SOURCE_IDS[0]),
            "IDs must be unique",
        ),
        (
            lambda value: value["verified"][1].update(
                raw_sha256=value["verified"][0]["raw_sha256"]
            ),
            "raw/text hashes must be unique|run entries",
        ),
        (
            lambda value: value["verified"][0].update(text_sha256="0" * 64),
            "run entries",
        ),
    ):
        base, _ = _source_manifest()
        changed, changed_paths = _rewrite_source_json(
            tmp_path, base, SOURCE_ARTIFACT_PATHS, "acquisition-run", mutate
        )
        with pytest.raises(ValueError, match=message):
            scorer.validate_source_snapshot_manifest(
                panel,
                changed,
                scorer.sha256_json(changed),
                changed_paths,
            )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(upstream_revision="wrong-revision"),
            "binding mismatch",
        ),
        (
            lambda value: value["range_mappings"][0].update(output_char_start=1),
            "range mapping is invalid",
        ),
        (
            lambda value: value["range_mappings"][0].update(source_char_start=False),
            "range mapping is invalid",
        ),
        (
            lambda value: value.update(reviewed_character_count=1),
            "character count mismatch",
        ),
        (
            lambda value: value.update(reviewed_unicode_word_token_count=8_191),
            "token count/method mismatch",
        ),
        (
            lambda value: value.update(word_token_count_method="caller-claimed"),
            "token count/method mismatch",
        ),
        (
            lambda value: value.update(extracted_text_sha256="0" * 64),
            "binding mismatch",
        ),
        (
            lambda value: value.update(review_receipt_sha256="0" * 64),
            "share one review receipt hash",
        ),
    ],
)
def test_acquisition_binding_semantics_fail_closed(panel, tmp_path, mutate, message):
    manifest, _ = _source_manifest()
    changed, paths = _rewrite_source_json(
        tmp_path, manifest, SOURCE_ARTIFACT_PATHS, "binding-en", mutate
    )
    with pytest.raises(ValueError, match=message):
        scorer.validate_source_snapshot_manifest(
            panel, changed, scorer.sha256_json(changed), paths
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["snapshots"][TEST_SOURCE_IDS[0]].update(
                upstream_revision="latest"
            ),
            "immutable and exact",
        ),
        (
            lambda value: value["snapshots"][TEST_SOURCE_IDS[0]].update(
                upstream_date="October 1"
            ),
            "ISO YYYY-MM-DD",
        ),
        (
            lambda value: value["snapshots"][TEST_SOURCE_IDS[0]].update(
                untouched_split="unknown"
            ),
            "untouched_split must be exact",
        ),
        (
            lambda value: value["snapshots"][TEST_SOURCE_IDS[0]].update(
                third_party_images_excluded=False
            ),
            "exclude third-party images",
        ),
        (
            lambda value: value["license_policies"]["ntsb_staff_text"].update(
                license_url="http://example.invalid/license"
            ),
            "HTTPS URL",
        ),
    ],
)
def test_source_snapshot_metadata_fails_closed(panel, mutate, message):
    manifest, _ = _source_manifest()
    mutate(manifest)
    with pytest.raises(ValueError, match=message):
        scorer.validate_source_snapshot_manifest(
            panel,
            manifest,
            scorer.sha256_json(manifest),
            SOURCE_ARTIFACT_PATHS,
        )


def _rehash_chunking_receipt(receipt: dict) -> None:
    receipt["receipt_sha256"] = scorer.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def _replace_chunk_derivation(case: dict, index: int, derivation: dict) -> None:
    chunk = case["chunks"][index]
    chunk["derivation"] = derivation
    chunk["content_sha256"] = scorer.sha256_json(scorer.chunk_hash_payload(chunk))
    case["chunking_receipt"]["ordered_derivations"][index]["derivation"] = derivation
    _rehash_chunking_receipt(case["chunking_receipt"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda case, derivation: derivation.update(
                char_end=len(
                    SOURCE_ARTIFACT_PATHS["reviewed-en"].read_text(encoding="utf-8")
                )
                + 1
            ),
            "out of range",
        ),
        (
            lambda case, derivation: derivation.update(slice_sha256="0" * 64),
            "slice sha256 mismatch",
        ),
        (
            lambda case, derivation: derivation.update(
                source_artifact_sha256=SOURCE_ARTIFACT_HASHES["reviewed-he"]
            ),
            "does not bind the case reviewed source",
        ),
    ],
)
def test_source_slice_derivation_rejects_wrong_bindings(panel, mutate, message):
    changed = copy.deepcopy(panel)
    case = changed["cases"][0]
    derivation = copy.deepcopy(case["chunks"][1]["derivation"])
    mutate(case, derivation)
    _replace_chunk_derivation(case, 1, derivation)
    with pytest.raises(ValueError, match=message):
        _validate(changed)


def test_rejects_invented_chunk_text_even_when_content_hash_is_recomputed(panel):
    changed = copy.deepcopy(panel)
    chunk = changed["cases"][0]["chunks"][1]
    chunk["text"] = "invented evidence not present in the reviewed source"
    chunk["content_sha256"] = scorer.sha256_json(scorer.chunk_hash_payload(chunk))
    with pytest.raises(ValueError, match="exactly equal its reviewed source slice"):
        _validate(changed)


def test_rejects_hebrew_source_slice_masquerading_as_translation(panel):
    changed = copy.deepcopy(panel)
    case = changed["cases"][1]
    source_text = SOURCE_ARTIFACT_PATHS["reviewed-he"].read_text(encoding="utf-8")
    derivation = {
        "kind": "source_text_slice",
        "source_artifact_sha256": SOURCE_ARTIFACT_HASHES["reviewed-he"],
        "char_start": 0,
        "char_end": 1,
        "slice_sha256": scorer.sha256_bytes(source_text[:1].encode("utf-8")),
    }
    _replace_chunk_derivation(case, 1, derivation)
    with pytest.raises(ValueError, match="translated derivation keys"):
        _validate(changed)


def test_cross_case_derivation_keys_are_globally_unambiguous(panel):
    changed = copy.deepcopy(panel)
    changed["cases"][1]["chunks"][0]["chunk_key"] = changed["cases"][0]["chunks"][0][
        "chunk_key"
    ]
    with pytest.raises(ValueError, match="globally unique"):
        _validate(changed)


def test_image_derivation_binds_image_and_control_receipts(panel):
    changed = copy.deepcopy(panel)
    case = changed["cases"][0]
    derivation = copy.deepcopy(case["chunks"][6]["derivation"])
    derivation["image_artifact_sha256"] = SOURCE_ARTIFACT_HASHES["shuffled-image"]
    _replace_chunk_derivation(case, 6, derivation)
    with pytest.raises(ValueError, match="supplied image artifact"):
        _validate(changed)


def test_rejects_chunking_receipt_derivation_tampering(panel):
    changed = copy.deepcopy(panel)
    receipt = changed["cases"][0]["chunking_receipt"]
    tampered = copy.deepcopy(receipt["ordered_derivations"][1]["derivation"])
    tampered["slice_sha256"] = "0" * 64
    receipt["ordered_derivations"][1]["derivation"] = tampered
    _rehash_chunking_receipt(receipt)
    with pytest.raises(ValueError, match="derivations do not match panel chunks"):
        _validate(changed)


def test_chunking_receipt_enforces_frozen_span_and_semantic_disclosure(panel):
    source, digest = _source_manifest()
    _, by_hash = scorer.validate_source_snapshot_manifest(
        panel, source, digest, SOURCE_ARTIFACT_PATHS
    )
    case = panel["cases"][0]
    scorer.validate_case_chunking_receipt(
        case, by_hash, TEST_ARTIFACT_SHA256, TEST_ARTIFACT_SHA256
    )

    too_long = copy.deepcopy(case)
    too_long["chunks"][0]["token_end"] = 513
    too_long["chunking_receipt"]["ordered_spans"][0]["token_end"] = 513
    _rehash_chunking_receipt(too_long["chunking_receipt"])
    with pytest.raises(ValueError, match="exceeding 512"):
        scorer.validate_case_chunking_receipt(
            too_long, by_hash, TEST_ARTIFACT_SHA256, TEST_ARTIFACT_SHA256
        )

    unbound = copy.deepcopy(case)
    unbound["chunking_receipt"]["ordered_spans"].reverse()
    _rehash_chunking_receipt(unbound["chunking_receipt"])
    with pytest.raises(ValueError, match="ordered spans"):
        scorer.validate_case_chunking_receipt(
            unbound, by_hash, TEST_ARTIFACT_SHA256, TEST_ARTIFACT_SHA256
        )

    overclaim = copy.deepcopy(case)
    overclaim["chunking_receipt"]["semantic_verification"] = "verified"
    _rehash_chunking_receipt(overclaim["chunking_receipt"])
    with pytest.raises(ValueError, match="external tokenizer/chunker boundary"):
        scorer.validate_case_chunking_receipt(
            overclaim, by_hash, TEST_ARTIFACT_SHA256, TEST_ARTIFACT_SHA256
        )

    with pytest.raises(ValueError, match="tokenizer/chunker"):
        scorer.validate_case_chunking_receipt(
            case, by_hash, "0" * 64, TEST_ARTIFACT_SHA256
        )


def test_fixed_chunking_fails_closed_without_tokenizer_recomputation(panel):
    source, digest = _source_manifest()
    _, by_hash = scorer.validate_source_snapshot_manifest(
        panel, source, digest, SOURCE_ARTIFACT_PATHS
    )
    case = copy.deepcopy(panel["cases"][0])
    starts = [index * 448 for index in range(len(case["chunks"]))]
    for chunk, start in zip(case["chunks"], starts, strict=True):
        chunk["token_start"] = start
        chunk["token_end"] = start + 512
    receipt = case["chunking_receipt"]
    receipt["mode"] = "fixed_512_overlap_64"
    receipt["boundary_receipt_artifact_sha256"] = None
    receipt["ordered_spans"] = [
        {
            "chunk_key": chunk["chunk_key"],
            "token_start": chunk["token_start"],
            "token_end": chunk["token_end"],
        }
        for chunk in case["chunks"]
    ]
    _rehash_chunking_receipt(receipt)
    with pytest.raises(ValueError, match="external tokenizer boundary"):
        scorer.validate_case_chunking_receipt(
            case, by_hash, TEST_ARTIFACT_SHA256, TEST_ARTIFACT_SHA256
        )


def test_prior_panel_and_source_manifest_require_explicit_migration(panel):
    old_panel = copy.deepcopy(panel)
    old_panel["schema_version"] = "r052-evidence-panel-v1"
    with pytest.raises(ValueError, match=scorer.SCHEMA_VERSION):
        _validate(old_panel)

    old_panel["schema_version"] = "r052-evidence-panel-v2"
    with pytest.raises(ValueError, match=scorer.SCHEMA_VERSION):
        _validate(old_panel)

    source, _ = _source_manifest()
    source["schema_version"] = "r052-source-provenance-manifest-v2"
    source.pop("acquisition_run_receipt_artifact")
    with pytest.raises(ValueError, match="v2/v3/v4 requires migration to v5"):
        scorer.validate_source_snapshot_manifest(
            panel,
            source,
            scorer.sha256_json(source),
            SOURCE_ARTIFACT_PATHS,
        )

    source, _ = _source_manifest()
    source["schema_version"] = "r052-source-provenance-manifest-v3"
    source.pop("acquisition_run_receipt_artifact")
    with pytest.raises(ValueError, match="v2/v3/v4 requires migration to v5"):
        scorer.validate_source_snapshot_manifest(
            panel,
            source,
            scorer.sha256_json(source),
            SOURCE_ARTIFACT_PATHS,
        )

    source, _ = _source_manifest()
    source["schema_version"] = "r052-source-provenance-manifest-v1"
    with pytest.raises(ValueError, match="policy/provenance"):
        scorer.validate_source_snapshot_manifest(
            panel,
            source,
            scorer.sha256_json(source),
            SOURCE_ARTIFACT_PATHS,
        )


def test_json_parser_rejects_duplicate_bindings_and_non_finite_values():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        scorer.parse_unique_finite_json(b'{"artifacts":{},"artifacts":{}}', "source")
    with pytest.raises(ValueError, match="non-finite"):
        scorer.parse_unique_finite_json(b'{"value":NaN}', "source")


def test_receipt_is_content_bound_timestamp_free_and_atomic(panel, tmp_path):
    _validate(panel)
    method = scorer.score_method(panel, "bm25", (1, 2, 4))
    _, causal_digest = _causal_manifest(panel)
    _, contamination_digest = _contamination_manifest(panel)
    source, source_digest = _source_manifest()
    provenance_artifact_digests = {
        "acquisition_manifest": source["artifacts"][
            source["acquisition_manifest_artifact"]
        ]["sha256"],
        "acquisition_run_receipt": source["artifacts"][
            source["acquisition_run_receipt_artifact"]
        ]["sha256"],
    }
    receipt = scorer.build_receipt(
        panel,
        _sha("raw"),
        source_digest,
        provenance_artifact_digests,
        causal_digest,
        contamination_digest,
        _policy_artifacts(panel)["policy_replay_receipt_sha256"],
        (1, 2, 4),
        [method],
    )
    assert "timestamp" not in json.dumps(receipt)
    assert (
        receipt["trusted_source_provenance_artifact_sha256"]
        == provenance_artifact_digests
    )
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
    source_snapshot_path = tmp_path / "source-snapshots.json"
    causal_source = tmp_path / "causal.json"
    contamination_source = tmp_path / "contamination.json"
    output = tmp_path / "receipt.json"
    causal_manifest, causal_digest = _causal_manifest(panel)
    causal_source.write_bytes(scorer.canonical_json(causal_manifest))
    contamination_manifest, contamination_digest = _contamination_manifest(panel)
    contamination_source.write_bytes(scorer.canonical_json(contamination_manifest))
    source_snapshot_manifest, source_snapshot_digest = _source_manifest()
    source_snapshot_path.write_bytes(scorer.canonical_json(source_snapshot_manifest))
    source.write_bytes(scorer.canonical_json(panel))
    policy = _policy_artifacts(panel)
    ledger_path = tmp_path / "fact-ledger.json"
    rubric_path = tmp_path / "rubric.json"
    replay_receipt_path = tmp_path / "policy-replay-receipt.json"
    ledger_path.write_bytes(scorer.canonical_json(policy["policy_fact_ledger"]))
    rubric_path.write_bytes(scorer.canonical_json(policy["policy_rubric"]))
    replay_receipt_path.write_bytes(
        scorer.canonical_json(policy["policy_replay_receipt"])
    )
    cli_args = [
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
        "--source-snapshot-manifest",
        str(source_snapshot_path),
        "--source-snapshot-manifest-sha256",
        source_snapshot_digest,
        *[
            item
            for name, path in SOURCE_ARTIFACT_PATHS.items()
            for item in ("--source-snapshot-artifact", f"{name}={path}")
        ],
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
        "--policy-fact-ledger",
        str(ledger_path),
        "--policy-fact-ledger-sha256",
        policy["policy_fact_ledger_sha256"],
        "--policy-rubric",
        str(rubric_path),
        "--policy-rubric-sha256",
        policy["policy_rubric_sha256"],
        "--policy-replay-receipt",
        str(replay_receipt_path),
        "--policy-replay-receipt-sha256",
        policy["policy_replay_receipt_sha256"],
        "--policy-replay-schema-artifact",
        str(policy["policy_replay_schema_artifact"]),
        "--policy-replay-schema-sha256",
        policy["policy_replay_schema_sha256"],
        "--policy-evaluator-artifact",
        str(policy["policy_evaluator_artifact"]),
        "--policy-evaluator-sha256",
        policy["policy_evaluator_sha256"],
        "--policy-adjudication-artifact",
        f"{TEST_ARTIFACT_SHA256}={TEST_ARTIFACT}",
        "--output",
        str(output),
        "--provider",
        "fixture-dense",
    ]
    monkeypatch.setattr("sys.argv", cli_args)
    scorer.main()
    receipt = json.loads(output.read_text())
    assert [method["method"] for method in receipt["methods"]] == [
        "bm25",
        "fixture-dense",
    ]
    assert receipt["input_raw_sha256"] == scorer.file_sha256(source)

    bm25_output = tmp_path / "bm25-receipt.json"
    bm25_args = list(cli_args)
    for flag in ("--provider-generator-artifact", "--provider"):
        index = bm25_args.index(flag)
        del bm25_args[index : index + 2]
    bm25_args[bm25_args.index("--output") + 1] = str(bm25_output)
    monkeypatch.setattr("sys.argv", bm25_args)
    scorer.main()
    bm25_receipt = json.loads(bm25_output.read_text())
    assert [method["method"] for method in bm25_receipt["methods"]] == ["bm25"]


def test_claimed_provenance_must_resolve_to_supplied_artifacts(panel, tmp_path):
    source, source_digest = _source_manifest()
    causal, causal_digest = _causal_manifest(panel)
    contamination, contamination_digest = _contamination_manifest(panel)
    impostor = tmp_path / "impostor.py"
    impostor.write_text("pass\n")
    with pytest.raises(ValueError, match="annotation builder artifact hash mismatch"):
        scorer.validate_panel(
            panel,
            source,
            source_digest,
            causal,
            causal_digest,
            contamination,
            contamination_digest,
            annotation_builder_artifact=impostor,
            contamination_auditor_artifact=TEST_ARTIFACT,
            contamination_corpus_artifacts=[TEST_ARTIFACT],
            contamination_tokenizer_artifact=TEST_ARTIFACT,
            contamination_chunker_artifact=TEST_ARTIFACT,
            source_snapshot_artifacts=SOURCE_ARTIFACT_PATHS,
            provider_generator_artifacts={"fixture-dense": TEST_ARTIFACT},
            **_policy_artifacts(panel),
        )


def test_io_paths_reject_all_primary_input_and_output_aliases(tmp_path):
    panel = tmp_path / "panel.json"
    causal = tmp_path / "causal.json"
    contamination = tmp_path / "contamination.json"
    source = tmp_path / "source.json"
    for path in (panel, causal, contamination, source):
        path.write_text("{}")
    with pytest.raises(ValueError, match="supplied input artifact"):
        scorer.validate_io_paths(
            panel, causal, contamination, source, panel, tmp_path / "runtime.json"
        )
    with pytest.raises(ValueError, match="score and runtime"):
        scorer.validate_io_paths(
            panel,
            causal,
            contamination,
            source,
            tmp_path / "receipt.json",
            tmp_path / "receipt.json",
        )
    with pytest.raises(ValueError, match="input paths must not alias"):
        scorer.validate_io_paths(
            panel,
            panel,
            contamination,
            source,
            tmp_path / "out.json",
            tmp_path / "run.json",
        )
    artifact = tmp_path / "generator.py"
    artifact.write_text("pass\n")
    with pytest.raises(ValueError, match="supplied input artifact"):
        scorer.validate_io_paths(
            panel,
            causal,
            contamination,
            source,
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
            source,
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
