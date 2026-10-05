"""Verify every staged R043 external asset against one hash-locked manifest."""

import argparse
import hashlib
import json
from pathlib import Path


KINDS = {
    "model": "tree",
    "val-wf": "file",
    "val-wh": "file",
    "val-replay": "file",
    "action-cal": "file",
    "action-eval": "file",
    "he-left": "file",
    "he-right": "file",
    "mm": "tree",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path):
    """Hash sorted relative paths and contents, independent of the tree's root path."""
    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"asset tree is empty: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0" + file_hash(item).encode() + b"\n")
    return digest.hexdigest(), len(files)


def preflight(manifest_path, paths):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("assets manifest version must be 1")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(KINDS):
        raise ValueError("assets manifest entries do not match the frozen R043 asset set")
    if set(paths) != set(KINDS):
        raise ValueError("staged paths do not match the frozen R043 asset set")
    result = {}
    for name, kind in KINDS.items():
        entry = assets[name]
        if not isinstance(entry, dict) or entry.get("kind") != kind:
            raise ValueError(f"{name}: expected asset kind {kind}")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"{name}: asset path must be a nonempty string")
        recorded = Path(entry["path"])
        recorded = recorded if recorded.is_absolute() else manifest_path.parent / recorded
        staged = Path(paths[name])
        if staged.resolve() != recorded.resolve():
            raise ValueError(f"{name}: staged path does not match assets manifest")
        if kind == "file":
            if not staged.is_file():
                raise ValueError(f"{name}: asset file is missing")
            digest, count = file_hash(staged), 1
        else:
            if not staged.is_dir():
                raise ValueError(f"{name}: asset tree is missing")
            digest, count = tree_hash(staged)
        if entry.get("sha256") != digest:
            raise ValueError(f"{name}: asset hash mismatch")
        if type(entry.get("files")) is not int or entry["files"] != count:
            raise ValueError(f"{name}: asset file count mismatch")
        result[name] = {"kind": kind, "path": str(staged.resolve()),
                        "sha256": digest, "files": count}
    return {"manifest_sha256": file_hash(manifest_path),
            "assets": result, "promotion_pass": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    for name in KINDS:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = preflight(args.manifest, {name: getattr(args, name.replace("-", "_")) for name in KINDS})
    output = Path(args.out)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
