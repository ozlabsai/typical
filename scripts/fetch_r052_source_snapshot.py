#!/usr/bin/env python3
"""Fetch and verify the immutable R052 NTSB/GAO acquisition snapshot.

The committed manifest describes acquisition only.  It intentionally is not the
``r052-source-provenance-manifest-v2`` consumed by the panel scorer: the panel
builder must separately bind any chunks, translations, images, and review
receipts derived from these inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SCHEMA_VERSION = "r052-source-acquisition-manifest-v1"
EXPECTED_IDS = (
    "AAR0002",
    "HAR0302",
    "MAR0905",
    "PAR0901",
    "RAR1202",
    "GAO-01-203",
    "GAO-01-463",
    "GAO-06-184",
    "GAO-06-970",
    "GAO-08-1075R",
)
PDFTOTEXT_VERSION = "26.04.0"
PDFTOTEXT_FLAGS = ("-layout", "-enc", "UTF-8", "-eol", "unix")
ALLOWED_HOSTS = frozenset({"www.ntsb.gov", "www.gao.gov", "www.govinfo.gov"})
ALLOWED_DOMAINS = frozenset(
    {"legal", "finance", "operations", "safety", "support", "technical", "general"}
)
FORBIDDEN_BENCHMARK_MARKERS = (
    b"jevbench",
    b"workflowevals",
    b"typed-decisions",
    b"open-jev",
    b"jev-distill",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MANIFEST_KEYS = {
    "schema_version",
    "purpose",
    "benchmark_derived",
    "sealed_benchmark_data_included",
    "extraction",
    "license_policies",
    "snapshots",
}
EXTRACTION_KEYS = {"program", "version", "flags", "encoding"}
LICENSE_KEYS = {
    "source_family",
    "license_id",
    "policy_url",
    "scope",
    "attribution",
    "evaluation_use_cleared",
    "staff_text_redistribution_permitted",
    "raw_pdf_redistribution_permitted",
    "third_party_material_excluded",
    "downstream_review_required",
}
SNAPSHOT_KEYS = {
    "id",
    "agency",
    "title",
    "report_id",
    "package_id",
    "upstream_revision",
    "upstream_url",
    "adoption_or_issue_date",
    "public_release_date",
    "intended_domain",
    "untouched_split",
    "license_policy",
    "natural_language_authored",
    "benchmark_derived",
    "raw_pdf",
    "extracted_text",
}
RAW_KEYS = {"filename", "sha256", "bytes", "redistribute"}
TEXT_KEYS = {
    "filename",
    "sha256",
    "bytes",
    "words",
    "encoding",
    "redistribute_after_review_only",
}


class SnapshotError(ValueError):
    """The acquisition manifest or a local artifact failed closed."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise SnapshotError(f"{label} keys must be exactly {sorted(expected)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError(f"{label} must be non-empty text")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SnapshotError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise SnapshotError(f"{label} must be a positive integer")
    return value


def _safe_filename(value: Any, suffix: str, label: str) -> str:
    filename = _text(value, label)
    pure = PurePosixPath(filename)
    if (
        pure.is_absolute()
        or pure.name != filename
        or filename in {".", ".."}
        or not filename.endswith(suffix)
    ):
        raise SnapshotError(f"{label} must be a traversal-free {suffix} basename")
    return filename


def _https_url(value: Any, label: str) -> str:
    url = _text(value, label)
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SnapshotError(f"{label} must be a direct HTTPS allowlisted federal URL")
    return url


def _reject_markers(payload: bytes, label: str) -> None:
    folded = payload.lower()
    marker = next(
        (item.decode() for item in FORBIDDEN_BENCHMARK_MARKERS if item in folded),
        None,
    )
    if marker is not None:
        raise SnapshotError(f"{label} contains forbidden benchmark marker {marker!r}")


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise SnapshotError(f"{label} may not contain symlink components")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _read_manifest(path: Path) -> dict[str, Any]:
    _reject_symlink_components(path, "manifest path")
    if not path.is_file():
        raise SnapshotError("manifest must be a regular file")
    raw = path.read_bytes()
    _reject_markers(raw, "manifest")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotError(f"manifest contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotError("manifest must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SnapshotError("manifest must be an object")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the closed acquisition schema and return its snapshots."""
    _reject_markers(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode(), "manifest"
    )
    _exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise SnapshotError(f"schema_version must be {SCHEMA_VERSION}")
    _text(manifest["purpose"], "manifest purpose")
    if manifest["benchmark_derived"] is not False:
        raise SnapshotError("acquisition inputs must not be benchmark-derived")
    if manifest["sealed_benchmark_data_included"] is not False:
        raise SnapshotError("sealed benchmark data must never be acquired")

    extraction = manifest["extraction"]
    if not isinstance(extraction, dict):
        raise SnapshotError("extraction must be an object")
    _exact_keys(extraction, EXTRACTION_KEYS, "extraction")
    if extraction != {
        "program": "pdftotext",
        "version": PDFTOTEXT_VERSION,
        "flags": list(PDFTOTEXT_FLAGS),
        "encoding": "UTF-8",
    }:
        raise SnapshotError("extraction recipe does not match the frozen recipe")

    policies = manifest["license_policies"]
    if not isinstance(policies, dict) or set(policies) != {
        "ntsb_staff_text",
        "gao_government_work",
    }:
        raise SnapshotError("license_policies must contain the two frozen policies")
    for name, policy in policies.items():
        if not isinstance(policy, dict):
            raise SnapshotError(f"license policy {name} must be an object")
        _exact_keys(policy, LICENSE_KEYS, f"license policy {name}")
        if policy["source_family"] != "public_domain":
            raise SnapshotError(f"license policy {name} must be public_domain")
        for field in ("license_id", "scope", "attribution"):
            _text(policy[field], f"license policy {name} {field}")
        _https_url(policy["policy_url"], f"license policy {name} URL")
        if policy["evaluation_use_cleared"] is not True:
            raise SnapshotError(f"license policy {name} is not cleared for evaluation")
        if policy["staff_text_redistribution_permitted"] is not True:
            raise SnapshotError(f"license policy {name} must clear staff text")
        if policy["raw_pdf_redistribution_permitted"] is not False:
            raise SnapshotError(
                f"license policy {name} must prohibit raw PDF publication"
            )
        if policy["third_party_material_excluded"] is not True:
            raise SnapshotError(
                f"license policy {name} must exclude third-party material"
            )
        if policy["downstream_review_required"] is not True:
            raise SnapshotError(f"license policy {name} must require downstream review")

    snapshots = manifest["snapshots"]
    if not isinstance(snapshots, list):
        raise SnapshotError("snapshots must be a list")
    ids = [item.get("id") if isinstance(item, dict) else None for item in snapshots]
    if len(ids) != len(set(ids)):
        raise SnapshotError("snapshot ids must be unique")
    if tuple(ids) != EXPECTED_IDS:
        raise SnapshotError(f"snapshots must be exactly {list(EXPECTED_IDS)} in order")

    seen_urls: set[str] = set()
    seen_revisions: set[str] = set()
    seen_filenames: set[str] = set()
    seen_hashes: set[str] = set()
    for index, snapshot in enumerate(snapshots):
        label = f"snapshots[{index}]"
        if not isinstance(snapshot, dict):
            raise SnapshotError(f"{label} must be an object")
        _exact_keys(snapshot, SNAPSHOT_KEYS, label)
        for field in ("id", "agency", "title", "report_id", "upstream_revision"):
            _text(snapshot[field], f"{label}.{field}")
        package_id = snapshot["package_id"]
        if package_id is not None:
            _text(package_id, f"{label}.package_id")
        url = _https_url(snapshot["upstream_url"], f"{label}.upstream_url")
        if url in seen_urls:
            raise SnapshotError("upstream URLs must be unique")
        seen_urls.add(url)
        revision = snapshot["upstream_revision"]
        if revision in seen_revisions:
            raise SnapshotError("upstream revisions must be unique")
        seen_revisions.add(revision)
        parsed_dates: dict[str, date | None] = {}
        for field in ("adoption_or_issue_date", "public_release_date"):
            value = snapshot[field]
            if value is not None and (
                not isinstance(value, str) or ISO_DATE_RE.fullmatch(value) is None
            ):
                raise SnapshotError(f"{label}.{field} must be an ISO date or null")
            try:
                parsed_dates[field] = date.fromisoformat(value) if value else None
            except ValueError as error:
                raise SnapshotError(
                    f"{label}.{field} must be a valid ISO date"
                ) from error
        if parsed_dates["adoption_or_issue_date"] is None:
            raise SnapshotError(f"{label}.adoption_or_issue_date is required")
        if (
            parsed_dates["public_release_date"] is not None
            and parsed_dates["public_release_date"]
            < parsed_dates["adoption_or_issue_date"]
        ):
            raise SnapshotError(f"{label}.public_release_date predates issue")
        if snapshot["intended_domain"] not in ALLOWED_DOMAINS:
            raise SnapshotError(f"{label}.intended_domain is invalid")
        _text(snapshot["untouched_split"], f"{label}.untouched_split")
        if snapshot["license_policy"] not in policies:
            raise SnapshotError(f"{label}.license_policy is not declared")
        if snapshot["natural_language_authored"] is not True:
            raise SnapshotError(f"{label} must be naturally authored")
        if snapshot["benchmark_derived"] is not False:
            raise SnapshotError(f"{label} must not be benchmark-derived")

        raw = snapshot["raw_pdf"]
        text = snapshot["extracted_text"]
        if not isinstance(raw, dict) or not isinstance(text, dict):
            raise SnapshotError(f"{label} artifacts must be objects")
        _exact_keys(raw, RAW_KEYS, f"{label}.raw_pdf")
        _exact_keys(text, TEXT_KEYS, f"{label}.extracted_text")
        raw_name = _safe_filename(raw["filename"], ".pdf", f"{label} raw filename")
        text_name = _safe_filename(text["filename"], ".txt", f"{label} text filename")
        for filename in (raw_name, text_name):
            if filename in seen_filenames:
                raise SnapshotError("artifact filenames must be unique")
            seen_filenames.add(filename)
        for artifact, artifact_label in ((raw, "raw"), (text, "text")):
            digest = _sha256(artifact["sha256"], f"{label} {artifact_label} sha256")
            if digest in seen_hashes:
                raise SnapshotError("artifact hashes must be unique")
            seen_hashes.add(digest)
        _positive_int(raw["bytes"], f"{label} raw bytes")
        _positive_int(text["bytes"], f"{label} text bytes")
        _positive_int(text["words"], f"{label} text words")
        if raw["redistribute"] is not False:
            raise SnapshotError("raw PDFs may not be redistributed")
        if text["encoding"] != "UTF-8":
            raise SnapshotError("extracted text encoding must be UTF-8")
        if text["redistribute_after_review_only"] is not True:
            raise SnapshotError("extracted text requires downstream material review")

        identifier = snapshot["id"]
        parsed_url = urlparse(url)
        if raw_name != f"{identifier}.pdf" or text_name != f"{identifier}.txt":
            raise SnapshotError(f"{label} artifact names must match its id")
        if identifier.startswith("GAO-"):
            package_id = f"GAOREPORTS-{identifier}"
            expected = {
                "agency": "U.S. Government Accountability Office",
                "report_id": identifier,
                "package_id": package_id,
                "upstream_revision": package_id,
                "license_policy": "gao_government_work",
            }
            expected_url_path = f"/content/pkg/{package_id}/pdf/{package_id}.pdf"
        else:
            report_id = f"NTSB/{identifier[:3]}-{identifier[3:5]}/{identifier[5:]}"
            expected = {
                "agency": "National Transportation Safety Board",
                "report_id": report_id,
                "package_id": package_id,
                "upstream_revision": f"{report_id}+{package_id}",
                "license_policy": "ntsb_staff_text",
            }
            expected_url_path = (
                f"/investigations/AccidentReports/Reports/{identifier}.pdf"
            )
            if not isinstance(package_id, str) or not package_id.startswith("PB"):
                raise SnapshotError(f"{label}.package_id must bind its PB package")
        if any(
            snapshot[field] != expected_value
            for field, expected_value in expected.items()
        ):
            raise SnapshotError(
                f"{label} agency/report/package revision binding is invalid"
            )
        if parsed_url.path != expected_url_path:
            raise SnapshotError(f"{label}.upstream_url does not bind its report id")
    return snapshots


def _digest(path: Path) -> tuple[str, int, bytes]:
    hasher = hashlib.sha256()
    payload = bytearray()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
            payload.extend(block)
            size += len(block)
    return hasher.hexdigest(), size, bytes(payload)


def _verify_file(path: Path, declaration: Mapping[str, Any], label: str) -> bytes:
    _reject_symlink_components(path, label)
    if not path.is_file():
        raise SnapshotError(f"{label} is missing or not a regular file")
    digest, size, payload = _digest(path)
    if digest != declaration["sha256"] or size != declaration["bytes"]:
        raise SnapshotError(f"{label} hash/size mismatch")
    _reject_markers(payload, label)
    return payload


def _check_pdftotext_version(binary: str = "pdftotext") -> None:
    result = subprocess.run([binary, "-v"], capture_output=True, text=True, check=False)
    output = f"{result.stdout}\n{result.stderr}"
    expected = rf"(?m)^pdftotext version {re.escape(PDFTOTEXT_VERSION)}\s*$"
    if result.returncode != 0 or re.search(expected, output) is None:
        raise SnapshotError(f"pdftotext must be exactly version {PDFTOTEXT_VERSION}")


def _ensure_untracked_pdf_destination(path: Path) -> None:
    """Refuse PDF publication into a non-ignored location inside a Git worktree."""
    probe = path.parent
    while not probe.exists():
        probe = probe.parent
    result = subprocess.run(
        ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    repo = Path(result.stdout.strip()).resolve(strict=True)
    try:
        relative = path.absolute().relative_to(repo)
    except ValueError:
        return
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            str(relative),
        ],
        check=False,
    )
    if ignored.returncode != 0:
        raise SnapshotError(
            f"raw PDF destination {relative} is not ignored; PDFs must never be committed"
        )


def _open_https(url: str) -> BinaryIO:
    request = Request(url, headers={"User-Agent": "Typical-R052-source-fetcher/1.0"})
    response = urlopen(request, timeout=60)
    final = urlparse(response.geturl())
    if final.scheme != "https" or final.hostname not in ALLOWED_HOSTS:
        response.close()
        raise SnapshotError("download redirected outside the HTTPS official hosts")
    return response


def _download_atomic(
    url: str, destination: Path, declaration: Mapping[str, Any]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination, "raw PDF destination")
    _ensure_untracked_pdf_destination(destination)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            with _open_https(url) as response:
                while block := response.read(1024 * 1024):
                    temporary.write(block)
                    if temporary.tell() > declaration["bytes"]:
                        raise SnapshotError("download exceeds its declared byte size")
            temporary.flush()
            os.fsync(temporary.fileno())
        _verify_file(temp_path, declaration, "downloaded raw PDF")
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _extract_atomic(
    raw_path: Path,
    text_path: Path,
    declaration: Mapping[str, Any],
    binary: str = "pdftotext",
) -> None:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(text_path, "extracted text destination")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=text_path.parent, prefix=f".{text_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temp_path = Path(temporary_name)
    try:
        result = subprocess.run(
            [binary, *PDFTOTEXT_FLAGS, str(raw_path), str(temp_path)], check=False
        )
        if result.returncode != 0:
            raise SnapshotError("pdftotext extraction failed")
        payload = _verify_file(temp_path, declaration, "new extracted text")
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SnapshotError("new extracted text is not UTF-8") from error
        if len(decoded.split()) != declaration["words"]:
            raise SnapshotError("new extracted text word-count mismatch")
        os.replace(temp_path, text_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def acquire(
    manifest_path: Path,
    output_root: Path,
    *,
    offline: bool = False,
    verify_only: bool = False,
    pdftotext_binary: str = "pdftotext",
) -> dict[str, Any]:
    """Acquire or verify every declared artifact, returning a compact receipt."""
    if offline and verify_only:
        raise SnapshotError("offline and verify-only are mutually exclusive")
    manifest = _read_manifest(manifest_path)
    snapshots = validate_manifest(manifest)
    _reject_symlink_components(output_root, "output root")
    if verify_only and not output_root.is_dir():
        raise SnapshotError("verify-only output root does not exist")
    if not verify_only:
        output_root.mkdir(parents=True, exist_ok=True)
    _check_pdftotext_version(pdftotext_binary)

    verified: list[dict[str, Any]] = []
    for snapshot in snapshots:
        raw_path = output_root / "raw" / snapshot["raw_pdf"]["filename"]
        text_path = output_root / "text" / snapshot["extracted_text"]["filename"]
        if raw_path.exists() or raw_path.is_symlink():
            _verify_file(raw_path, snapshot["raw_pdf"], f"{snapshot['id']} raw PDF")
        elif offline or verify_only:
            raise SnapshotError(f"{snapshot['id']} raw PDF is unavailable offline")
        else:
            _download_atomic(snapshot["upstream_url"], raw_path, snapshot["raw_pdf"])

        if text_path.exists() or text_path.is_symlink():
            text_payload = _verify_file(
                text_path,
                snapshot["extracted_text"],
                f"{snapshot['id']} extracted text",
            )
        elif verify_only:
            raise SnapshotError(f"{snapshot['id']} extracted text is missing")
        else:
            _extract_atomic(
                raw_path,
                text_path,
                snapshot["extracted_text"],
                binary=pdftotext_binary,
            )
            text_payload = _verify_file(
                text_path,
                snapshot["extracted_text"],
                f"{snapshot['id']} extracted text",
            )
        try:
            decoded = text_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SnapshotError(
                f"{snapshot['id']} extracted text is not UTF-8"
            ) from error
        if len(decoded.split()) != snapshot["extracted_text"]["words"]:
            raise SnapshotError(f"{snapshot['id']} extracted text word-count mismatch")
        verified.append(
            {
                "id": snapshot["id"],
                "raw_sha256": snapshot["raw_pdf"]["sha256"],
                "text_sha256": snapshot["extracted_text"]["sha256"],
            }
        )
    return {
        "schema_version": "r052-source-acquisition-receipt-v1",
        "mode": "verify-only" if verify_only else "offline" if offline else "fetch",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "verified": verified,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("sources/r052/ntsb_gao_v1.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(".context/r052-sources")
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    parser.add_argument("--pdftotext", default="pdftotext")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = acquire(
        args.manifest,
        args.output_root,
        offline=args.offline,
        verify_only=args.verify_only,
        pdftotext_binary=args.pdftotext,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
