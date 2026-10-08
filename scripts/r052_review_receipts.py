"""Strict human-review receipts for R052 translation, image, and chunk claims.

These receipts attest coverage and reviewer/adjudicator decisions.  Validation
does not independently establish that a translation, image control, or semantic
boundary is correct; the fixed claim scope makes that limitation explicit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TRANSLATION_SCHEMA = "r052-translation-alignment-human-review-receipt-v2"
IMAGE_SCHEMA = "r052-image-control-human-review-receipt-v2"
BOUNDARY_SCHEMA = "r052-chunk-boundary-human-review-receipt-v2"
TEAM_MANIFEST_SCHEMA = "r052-trusted-review-team-manifest-v1"
TEAM_MANIFEST_ROLE = "trusted_review_team_manifest"
CLAIM_SCOPE = "human_attestation_only_not_automated_semantic_verification"
TRUST_SCOPE = "identity_allowlist_attestation_not_proof_of_humanness"

ROLE_TO_SCHEMA = {
    "translation_alignment_receipt": TRANSLATION_SCHEMA,
    "image_control_receipt": IMAGE_SCHEMA,
    "chunk_boundary_receipt": BOUNDARY_SCHEMA,
}
REVIEW_ROLES = ("reviewer_1", "reviewer_2")
ADJUDICATOR_ROLE = "adjudicator"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

RECEIPT_KEYS = {
    "schema_version",
    "status",
    "claim_scope",
    "review_team_manifest_sha256",
    "review_team",
    "coverage",
    "receipt_sha256",
}
TEAM_KEYS = {"reviewer_id", "role", "status"}
COVERAGE_KEYS = {"item", "reviewer_roles", "adjudicator_role", "status"}
TEAM_MANIFEST_KEYS = {
    "schema_version",
    "trust_scope",
    "members",
    "manifest_sha256",
}
TRUSTED_MEMBER_KEYS = {"reviewer_id", "allowed_roles", "status"}


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-empty trimmed text")
    return value


def _content_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _parse_canonical_receipt(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be strict finite UTF-8 JSON") from exc
    if not isinstance(value, dict) or raw != canonical_json(value):
        raise ValueError(f"{label} must use canonical JSON object bytes")
    return value


def _surface_sha(value: Mapping[str, Any]) -> str:
    return sha256_json(value)


def _chunk_surface(chunk: Mapping[str, Any]) -> dict[str, Any]:
    control = chunk.get("image_control")
    return {
        "kind": chunk.get("kind"),
        "modality": chunk.get("modality"),
        "text": chunk.get("text"),
        "immutable": chunk.get("immutable"),
        "twin_anchor": chunk.get("twin_anchor"),
        "image_control_variant": (
            None if not isinstance(control, Mapping) else control.get("variant")
        ),
    }


def _question_surface(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": question.get("text"),
        "options": question.get("options"),
        "option_ids": question.get("option_ids"),
        "decision_type": question.get("decision_type"),
        "policy_role": question.get("policy_role"),
        "twin_anchor": question.get("twin_anchor"),
    }


def _boundary_item(case: Mapping[str, Any]) -> dict[str, Any]:
    spans = []
    for chunk in case.get("chunks", []):
        derivation = chunk.get("derivation")
        if not isinstance(derivation, Mapping):
            raise TypeError("chunk derivation must be an object")
        derivation_kind = _text(derivation.get("kind"), "chunk derivation kind")
        boundary_derivation = {"kind": derivation_kind}
        if derivation_kind == "source_text_slice":
            boundary_derivation = dict(derivation)
        spans.append(
            {
                "chunk_key": _text(chunk.get("chunk_key"), "chunk key"),
                "twin_anchor": chunk.get("twin_anchor"),
                "kind": chunk.get("kind"),
                "modality": chunk.get("modality"),
                "text_sha256": hashlib.sha256(
                    _content_text(chunk.get("text"), "chunk text").encode("utf-8")
                ).hexdigest(),
                "token_start": chunk.get("token_start"),
                "token_end": chunk.get("token_end"),
                "derivation": boundary_derivation,
            }
        )
    identity = {
        "language": case.get("language"),
        "source_snapshot_sha256": case.get("source_snapshot_sha256"),
        "source_domain": case.get("source_domain"),
        "context_tokens": case.get("context_tokens"),
        "ordered_spans": spans,
    }
    return {"case_surface_sha256": _surface_sha(identity), **identity}


def expected_receipt_items(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Derive exact, non-circular coverage records grouped by receipt hash."""
    expected: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: defaultdict(list) for role in ROLE_TO_SCHEMA
    }
    twins: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        twin = case.get("twin_group_key")
        if isinstance(twin, str):
            twins[twin].append(case)

        chunking = case.get("chunking_receipt")
        if not isinstance(chunking, Mapping):
            raise TypeError("case chunking_receipt must be an object")
        if chunking.get("mode") == "semantic_sections":
            digest = _sha(
                chunking.get("boundary_receipt_artifact_sha256"),
                "chunk-boundary receipt hash",
            )
            expected["chunk_boundary_receipt"][digest].append(_boundary_item(case))

        controls: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for chunk in case.get("chunks", []):
            control = chunk.get("image_control")
            if isinstance(control, Mapping):
                group_key = _sha(control.get("group_key"), "image group key")
                variant = _text(control.get("variant"), "image variant")
                if variant in controls[group_key]:
                    raise ValueError("image control variants must be unique per group")
                controls[group_key][variant] = chunk
        for group_key, variants in controls.items():
            if set(variants) != {"real", "shuffled", "missing"}:
                raise ValueError("image receipt coverage requires all three controls")
            receipt_hashes = {
                chunk.get("derivation", {}).get("control_receipt_artifact_sha256")
                for chunk in variants.values()
            }
            if len(receipt_hashes) != 1:
                raise ValueError("image controls must share one receipt hash")
            digest = _sha(receipt_hashes.pop(), "image-control receipt hash")
            real, shuffled, missing = (
                variants[name] for name in ("real", "shuffled", "missing")
            )
            if any(
                chunk.get("modality") != "image" for chunk in (real, shuffled, missing)
            ):
                raise ValueError("image controls must use image-modality chunks")
            real_text = _content_text(real.get("text"), "real image control text")
            if any(
                _content_text(chunk.get("text"), "image control text") != real_text
                for chunk in (shuffled, missing)
            ):
                raise ValueError("image controls must preserve exactly matched text")
            real_media = real.get("media")
            shuffled_media = shuffled.get("media")
            if not isinstance(real_media, Mapping) or not isinstance(
                shuffled_media, Mapping
            ):
                raise TypeError("real/shuffled image controls require media objects")
            if missing.get("media") is not None:
                raise ValueError("missing image controls must have null media")
            real_image_sha256 = _sha(real_media.get("sha256"), "real image hash")
            shuffled_image_sha256 = _sha(
                shuffled_media.get("sha256"), "shuffled image hash"
            )
            if real_image_sha256 == shuffled_image_sha256:
                raise ValueError("real and shuffled image controls must differ")
            matching = [
                question
                for question in case.get("questions", [])
                if real.get("chunk_key") in question.get("decisive_chunk_keys", [])
                and {
                    shuffled.get("chunk_key"),
                    missing.get("chunk_key"),
                }.issubset(question.get("known_distractor_chunk_keys", []))
            ]
            if len(matching) != 1:
                raise ValueError("image receipt requires one causal question")
            expected["image_control_receipt"][digest].append(
                {
                    "group_key": group_key,
                    "real_image_sha256": real_image_sha256,
                    "shuffled_image_sha256": shuffled_image_sha256,
                    "missing_media": True,
                    "matched_text_sha256": hashlib.sha256(
                        real_text.encode("utf-8")
                    ).hexdigest(),
                    "causal_question_surface_sha256": _surface_sha(
                        _question_surface(matching[0])
                    ),
                }
            )

    for twin_group, paired in twins.items():
        if len(paired) != 2 or {case.get("language") for case in paired} != {
            "en",
            "he",
        }:
            raise ValueError("translation receipt requires one EN/HE case pair")
        en = next(case for case in paired if case.get("language") == "en")
        he = next(case for case in paired if case.get("language") == "he")
        alignment_hashes = {
            chunk.get("derivation", {}).get("alignment_receipt_artifact_sha256")
            for chunk in he.get("chunks", [])
            if chunk.get("derivation", {}).get("kind") == "translated_twin"
        }
        if len(alignment_hashes) != 1:
            raise ValueError("Hebrew twin must bind exactly one alignment receipt")
        digest = _sha(alignment_hashes.pop(), "translation-alignment receipt hash")
        for item_kind, key in (("chunk", "chunks"), ("question", "questions")):
            en_items = {item.get("twin_anchor"): item for item in en.get(key, [])}
            he_items = {item.get("twin_anchor"): item for item in he.get(key, [])}
            if set(en_items) != set(he_items) or None in en_items:
                raise ValueError("translation receipt twin anchors must match exactly")
            for anchor in sorted(en_items):
                surface = _chunk_surface if item_kind == "chunk" else _question_surface
                expected["translation_alignment_receipt"][digest].append(
                    {
                        "kind": item_kind,
                        "twin_group_key": twin_group,
                        "twin_anchor": anchor,
                        "source_language": "en",
                        "target_language": "he",
                        "source_surface_sha256": _surface_sha(
                            surface(en_items[anchor])
                        ),
                        "target_surface_sha256": _surface_sha(
                            surface(he_items[anchor])
                        ),
                    }
                )

    return {
        role: {
            digest: sorted(items, key=canonical_json)
            for digest, items in sorted(grouped.items())
        }
        for role, grouped in expected.items()
    }


def validate_trusted_review_team_manifest(
    path: Path, expected_artifact_sha256: str
) -> dict[str, frozenset[str]]:
    """Validate the externally anchored identity allowlist.

    This attests only that reviewer identifiers are allowlisted for declared
    roles. It is intentionally not evidence that an identifier is human.
    """
    expected_digest = _sha(
        expected_artifact_sha256, "trusted review-team manifest artifact hash"
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("trusted review-team manifest artifact hash mismatch")
    manifest = _parse_canonical_receipt(path, TEAM_MANIFEST_ROLE)
    _exact_keys(manifest, TEAM_MANIFEST_KEYS, TEAM_MANIFEST_ROLE)
    if (
        manifest["schema_version"] != TEAM_MANIFEST_SCHEMA
        or manifest["trust_scope"] != TRUST_SCOPE
    ):
        raise ValueError("trusted review-team manifest trust contract is invalid")
    members = manifest["members"]
    if not isinstance(members, list) or not members:
        raise ValueError("trusted review-team manifest members must be non-empty")
    trusted: dict[str, frozenset[str]] = {}
    canonical_members: list[dict[str, Any]] = []
    valid_roles = {*REVIEW_ROLES, ADJUDICATOR_ROLE}
    for raw_member in members:
        member = _exact_keys(
            raw_member, TRUSTED_MEMBER_KEYS, "trusted review-team member"
        )
        reviewer_id = _text(member["reviewer_id"], "trusted reviewer id")
        if reviewer_id in trusted:
            raise ValueError("trusted reviewer identities must be unique")
        allowed_roles = member["allowed_roles"]
        if (
            not isinstance(allowed_roles, list)
            or not allowed_roles
            or allowed_roles != sorted(set(allowed_roles))
            or not set(allowed_roles).issubset(valid_roles)
        ):
            raise ValueError("trusted reviewer allowed_roles are invalid")
        if member["status"] != "trusted":
            raise ValueError("trusted review-team members must be active")
        trusted[reviewer_id] = frozenset(allowed_roles)
        canonical_members.append(dict(member))
    if canonical_members != sorted(canonical_members, key=canonical_json):
        raise ValueError("trusted review-team members must use canonical order")
    if not all(
        any(role in roles for roles in trusted.values()) for role in valid_roles
    ):
        raise ValueError("trusted review-team manifest lacks required role coverage")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != sha256_json(body):
        raise ValueError("trusted review-team manifest self-hash mismatch")
    return trusted


def validate_review_receipt(
    path: Path,
    role: str,
    expected_items: Sequence[Mapping[str, Any]],
    trusted_team_manifest_sha256: str,
    trusted_reviewers: Mapping[str, frozenset[str]],
) -> None:
    schema = ROLE_TO_SCHEMA.get(role)
    if schema is None:
        raise ValueError(f"unsupported review receipt role: {role}")
    receipt = _parse_canonical_receipt(path, role)
    _exact_keys(receipt, RECEIPT_KEYS, role)
    if (
        receipt["schema_version"] != schema
        or receipt["status"] != "pass"
        or receipt["claim_scope"] != CLAIM_SCOPE
    ):
        raise ValueError(f"{role} must be a canonical completed human-review receipt")
    if receipt["review_team_manifest_sha256"] != _sha(
        trusted_team_manifest_sha256, "trusted review-team manifest hash"
    ):
        raise ValueError(f"{role} trusted review-team manifest binding mismatch")

    team = receipt["review_team"]
    if not isinstance(team, list) or len(team) != 3:
        raise ValueError(f"{role} requires two reviewers and one adjudicator")
    team_by_role: dict[str, Mapping[str, Any]] = {}
    reviewer_ids: set[str] = set()
    for member in team:
        member = _exact_keys(member, TEAM_KEYS, f"{role} review team member")
        member_role = _text(member["role"], "review role")
        reviewer_id = _text(member["reviewer_id"], "reviewer id")
        if member_role in team_by_role:
            raise ValueError(f"{role} contains duplicate reviewer roles")
        if reviewer_id in reviewer_ids:
            raise ValueError(f"{role} reviewer identities must be unique")
        if member_role not in trusted_reviewers.get(reviewer_id, frozenset()):
            raise ValueError(
                f"{role} reviewer identity/role is not in the trusted allowlist"
            )
        if member["status"] != "pass":
            raise ValueError(f"{role} review team must unanimously pass")
        team_by_role[member_role] = member
        reviewer_ids.add(reviewer_id)
    if set(team_by_role) != {*REVIEW_ROLES, ADJUDICATOR_ROLE}:
        raise ValueError(f"{role} review team roles are incomplete")

    coverage = receipt["coverage"]
    expected = sorted((dict(item) for item in expected_items), key=canonical_json)
    if not expected or not isinstance(coverage, list) or len(coverage) != len(expected):
        raise ValueError(f"{role} coverage must be complete and non-empty")
    actual_items = []
    for entry in coverage:
        entry = _exact_keys(entry, COVERAGE_KEYS, f"{role} coverage entry")
        if (
            entry["reviewer_roles"] != list(REVIEW_ROLES)
            or entry["adjudicator_role"] != ADJUDICATOR_ROLE
            or entry["status"] != "pass"
            or not isinstance(entry["item"], Mapping)
        ):
            raise ValueError(f"{role} coverage lacks reviewer/adjudicator passes")
        actual_items.append(dict(entry["item"]))
    if actual_items != expected:
        raise ValueError(f"{role} exact coverage/hash mismatch")

    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != sha256_json(body):
        raise ValueError(f"{role} receipt self-hash mismatch")


def validate_panel_review_receipts(
    cases: Sequence[Mapping[str, Any]],
    artifacts_by_hash: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = expected_receipt_items(cases)
    team_manifests = {
        digest: artifact
        for digest, artifact in artifacts_by_hash.items()
        if artifact.get("role") == TEAM_MANIFEST_ROLE
    }
    if len(team_manifests) != 1:
        raise ValueError("panel requires exactly one trusted review-team manifest")
    team_manifest_sha256, team_manifest_artifact = next(iter(team_manifests.items()))
    team_manifest_path = team_manifest_artifact.get("path")
    if not isinstance(team_manifest_path, Path):
        raise TypeError("trusted review-team manifest artifact path is missing")
    trusted_reviewers = validate_trusted_review_team_manifest(
        team_manifest_path, team_manifest_sha256
    )
    declared = {
        role: {
            digest: artifact
            for digest, artifact in artifacts_by_hash.items()
            if artifact.get("role") == role
        }
        for role in ROLE_TO_SCHEMA
    }
    for role in ROLE_TO_SCHEMA:
        if set(declared[role]) != set(expected[role]):
            raise ValueError(f"{role} declarations must exactly cover panel usage")
        for digest, items in expected[role].items():
            artifact = declared[role][digest]
            path = artifact.get("path")
            if not isinstance(path, Path):
                raise TypeError(f"{role} artifact path is missing")
            validate_review_receipt(
                path,
                role,
                items,
                team_manifest_sha256,
                trusted_reviewers,
            )
