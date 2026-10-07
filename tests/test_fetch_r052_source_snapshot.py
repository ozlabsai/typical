import hashlib
import io
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import fetch_r052_source_snapshot as fetcher

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "sources" / "r052" / "ntsb_gao_v1.json"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _fixture_manifest(
    tmp_path: Path, *, write_text: bool = True
) -> tuple[Path, Path, dict]:
    manifest = deepcopy(_manifest())
    output = tmp_path / "artifacts"
    (output / "raw").mkdir(parents=True)
    (output / "text").mkdir()
    for index, snapshot in enumerate(manifest["snapshots"]):
        raw = f"%PDF-1.4\nfixture {index} {snapshot['id']}\n".encode()
        text = f"Natural source fixture {index} for {snapshot['id']}.\n".encode()
        snapshot["raw_pdf"]["sha256"] = _sha(raw)
        snapshot["raw_pdf"]["bytes"] = len(raw)
        snapshot["extracted_text"]["sha256"] = _sha(text)
        snapshot["extracted_text"]["bytes"] = len(text)
        snapshot["extracted_text"]["words"] = len(text.decode().split())
        (output / "raw" / snapshot["raw_pdf"]["filename"]).write_bytes(raw)
        if write_text:
            (output / "text" / snapshot["extracted_text"]["filename"]).write_bytes(text)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, output, manifest


def test_committed_manifest_binds_exact_selected_snapshot_and_license_scope():
    manifest = _manifest()
    snapshots = fetcher.validate_manifest(manifest)

    assert tuple(item["id"] for item in snapshots) == fetcher.EXPECTED_IDS
    assert len(snapshots) == 10
    assert manifest["extraction"] == {
        "program": "pdftotext",
        "version": "26.04.0",
        "flags": ["-layout", "-enc", "UTF-8", "-eol", "unix"],
        "encoding": "UTF-8",
    }
    assert {item["untouched_split"] for item in snapshots} == {
        "r052-natural-eval-untouched-v1"
    }
    assert {item["intended_domain"] for item in snapshots} >= {
        "safety",
        "operations",
        "support",
        "technical",
        "finance",
        "legal",
    }
    assert all(item["upstream_url"].startswith("https://") for item in snapshots)
    for item in snapshots[5:]:
        package = f"GAOREPORTS-{item['id']}"
        assert item["package_id"] == package
        assert item["upstream_revision"] == package
        assert item["upstream_url"] == (
            f"https://www.govinfo.gov/content/pkg/{package}/pdf/{package}.pdf"
        )
    assert all(item["raw_pdf"]["redistribute"] is False for item in snapshots)
    assert all(
        item["extracted_text"]["redistribute_after_review_only"] is True
        for item in snapshots
    )
    for policy in manifest["license_policies"].values():
        assert policy["staff_text_redistribution_permitted"] is True
        assert policy["raw_pdf_redistribution_permitted"] is False
        assert policy["third_party_material_excluded"] is True
        assert policy["downstream_review_required"] is True
    assert (
        "Courtesy: National Transportation Safety Board"
        == manifest["license_policies"]["ntsb_staff_text"]["attribution"]
    )
    assert manifest["schema_version"] != "r052-source-provenance-manifest-v2"


def test_verify_only_checks_all_artifacts_without_network_or_writes(
    tmp_path, monkeypatch
):
    manifest_path, output, _ = _fixture_manifest(tmp_path)
    monkeypatch.setattr(fetcher, "_check_pdftotext_version", lambda _binary: None)
    monkeypatch.setattr(
        fetcher,
        "_open_https",
        lambda _url: pytest.fail("verify-only must not access the network"),
    )

    receipt = fetcher.acquire(manifest_path, output, verify_only=True)

    assert receipt["mode"] == "verify-only"
    assert [item["id"] for item in receipt["verified"]] == list(fetcher.EXPECTED_IDS)


def test_offline_extracts_missing_text_but_never_accesses_network(
    tmp_path, monkeypatch
):
    manifest_path, output, manifest = _fixture_manifest(tmp_path, write_text=False)
    expected = {
        item["id"]: f"Natural source fixture {index} for {item['id']}.\n".encode()
        for index, item in enumerate(manifest["snapshots"])
    }
    monkeypatch.setattr(fetcher, "_check_pdftotext_version", lambda _binary: None)
    monkeypatch.setattr(
        fetcher,
        "_open_https",
        lambda _url: pytest.fail("offline mode must not access the network"),
    )

    def fake_extract(raw_path, text_path, declaration, binary="pdftotext"):
        del raw_path, declaration, binary
        source_id = text_path.stem
        text_path.write_bytes(expected[source_id])

    monkeypatch.setattr(fetcher, "_extract_atomic", fake_extract)
    receipt = fetcher.acquire(manifest_path, output, offline=True)

    assert receipt["mode"] == "offline"
    assert len(receipt["verified"]) == 10


def test_extract_invokes_frozen_flags_and_publishes_only_verified_text(
    tmp_path, monkeypatch
):
    raw_path = tmp_path / "input.pdf"
    text_path = tmp_path / "output.txt"
    raw_path.write_bytes(b"%PDF-1.4\nfixture\n")
    expected = b"alpha beta gamma\n"
    declaration = {
        "sha256": _sha(expected),
        "bytes": len(expected),
        "words": 3,
    }
    observed = []

    def fake_run(command, check):
        observed.append(command)
        Path(command[-1]).write_bytes(expected)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)
    fetcher._extract_atomic(raw_path, text_path, declaration, binary="pdftotext-x")

    assert observed == [
        [
            "pdftotext-x",
            "-layout",
            "-enc",
            "UTF-8",
            "-eol",
            "unix",
            str(raw_path),
            observed[0][-1],
        ]
    ]
    assert text_path.read_bytes() == expected
    assert not list(tmp_path.glob(".output.txt.*.tmp"))


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str):
        super().__init__(payload)
        self._url = url

    def geturl(self) -> str:
        return self._url


def test_download_is_atomic_and_hash_gated(tmp_path, monkeypatch):
    payload = b"%PDF-1.4\ntrusted fixture\n"
    url = "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR0002.pdf"
    declaration = {"sha256": _sha(payload), "bytes": len(payload)}
    destination = tmp_path / "AAR0002.pdf"
    monkeypatch.setattr(fetcher, "_open_https", lambda _url: _Response(payload, url))
    monkeypatch.setattr(
        fetcher, "_ensure_untracked_pdf_destination", lambda _path: None
    )

    fetcher._download_atomic(url, destination, declaration)
    assert destination.read_bytes() == payload

    bad_destination = tmp_path / "bad.pdf"
    monkeypatch.setattr(fetcher, "_open_https", lambda _url: _Response(b"wrong", url))
    with pytest.raises(fetcher.SnapshotError, match="hash/size mismatch"):
        fetcher._download_atomic(url, bad_destination, declaration)
    assert not bad_destination.exists()
    assert not list(tmp_path.glob(".bad.pdf.*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["snapshots"].__setitem__(
                1, deepcopy(value["snapshots"][0])
            ),
            "ids must be unique",
        ),
        (
            lambda value: value["snapshots"][0]["raw_pdf"].__setitem__(
                "filename", "../escape.pdf"
            ),
            "traversal-free",
        ),
        (
            lambda value: value["snapshots"][0].__setitem__(
                "upstream_url", "http://www.ntsb.gov/AAR0002.pdf"
            ),
            "direct HTTPS",
        ),
        (
            lambda value: value.__setitem__("sealed_benchmark_data_included", True),
            "sealed benchmark data",
        ),
    ],
)
def test_manifest_validation_fails_closed(mutation, message):
    manifest = _manifest()
    mutation(manifest)
    with pytest.raises(fetcher.SnapshotError, match=message):
        fetcher.validate_manifest(manifest)


def test_artifact_benchmark_marker_and_symlink_are_rejected(tmp_path):
    marker_path = tmp_path / "marker.txt"
    payload = b"copied from JevBench"
    marker_path.write_bytes(payload)
    declaration = {"sha256": _sha(payload), "bytes": len(payload)}
    with pytest.raises(fetcher.SnapshotError, match="forbidden benchmark marker"):
        fetcher._verify_file(marker_path, declaration, "fixture")

    target = tmp_path / "target.txt"
    target.write_bytes(b"safe")
    symlink = tmp_path / "link.txt"
    symlink.symlink_to(target)
    safe_declaration = {"sha256": _sha(b"safe"), "bytes": 4}
    with pytest.raises(fetcher.SnapshotError, match="symlink"):
        fetcher._verify_file(symlink, safe_declaration, "fixture")


def test_pdftotext_version_is_exact(monkeypatch):
    monkeypatch.setattr(
        fetcher.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="", stderr="pdftotext version 26.03.0\n"
        ),
    )
    with pytest.raises(fetcher.SnapshotError, match="exactly version 26.04.0"):
        fetcher._check_pdftotext_version()


def test_pdf_destination_inside_git_must_be_ignored(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(fetcher.SnapshotError, match="PDFs must never be committed"):
        fetcher._ensure_untracked_pdf_destination(tmp_path / "raw" / "source.pdf")

    (tmp_path / ".gitignore").write_text("raw/\n")
    fetcher._ensure_untracked_pdf_destination(tmp_path / "raw" / "source.pdf")
