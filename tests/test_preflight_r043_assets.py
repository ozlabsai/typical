import json

import pytest

from scripts.preflight_r043_assets import KINDS, file_hash, preflight, tree_hash


def test_r043_asset_preflight_binds_paths_tree_contents_and_exact_entries(tmp_path):
    paths = {}
    for name, kind in KINDS.items():
        path = tmp_path / name
        if kind == "tree":
            path.mkdir()
            (path / "nested").mkdir()
            (path / "nested" / "asset.bin").write_bytes(name.encode())
        else:
            path.write_text(name)
        paths[name] = str(path)
    assets = {}
    for name, kind in KINDS.items():
        digest, files = ((file_hash(paths[name]), 1) if kind == "file" else tree_hash(paths[name]))
        assets[name] = {"path": name, "kind": kind, "sha256": digest, "files": files}
    manifest = tmp_path / "assets-manifest.json"
    manifest.write_text(json.dumps({"version": 1, "assets": assets}))

    report = preflight(manifest, paths)
    assert report["promotion_pass"]
    assert set(report["assets"]) == set(KINDS)
    (tmp_path / "model" / "nested" / "asset.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model: asset hash mismatch"):
        preflight(manifest, paths)


def test_r043_asset_preflight_rejects_extra_entries(tmp_path):
    manifest = tmp_path / "assets-manifest.json"
    manifest.write_text(json.dumps({"version": 1, "assets": {"extra": {}}}))
    with pytest.raises(ValueError, match="do not match"):
        preflight(manifest, {name: str(tmp_path / name) for name in KINDS})
