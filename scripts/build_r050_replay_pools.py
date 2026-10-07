"""Freeze balanced Hebrew and leakage-free singleton real-image R050 replay pools."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_r044_mm_replay import (
    NULL_OPTION, _decision_key, _flat_valid, _grouped, _licenses,
)


SCHEMA = "r050_replay_pools_v1"
DEFAULT_COUNTS = {"hebrew": 400, "multimodal": 400}
DEFAULT_LICENSES = {"MASSIVE": "CC-BY-4.0", "ScienceQA": "CC-BY-NC-SA-4.0"}


def _json(value, *, sort_keys=False):
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    try:
        rows = [json.loads(line, object_pairs_hook=_object)
                for line in payload.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"input must contain nonempty object rows: {path}")
    for row in rows:
        _json(row)
        if "meta" in row and not isinstance(row["meta"], dict):
            raise ValueError(f"metadata must be an object: {path}")
    return rows, {"path": str(path), "rows": len(rows), "bytes": len(payload),
                  "sha256": _sha(payload)}


def _rank(seed, name, value):
    return _sha(_json(["r050-replay", seed, name, value], sort_keys=True))


def _identity(row, position, prefix):
    # Prefer the upstream identity, not a possibly re-rendered row's own ID.
    meta = row.get("meta") or {}
    value = meta.get("source_id") or meta.get("pair") or row.get("id")
    if value is None:
        return f"{prefix}-line-{position + 1}"
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("source IDs must be nonempty strings or integers")
    return str(value)


def _valid(row):
    candidates, target, label = row.get("candidates"), row.get("target"), row.get("label")
    return (type(label) is int and isinstance(candidates, list) and isinstance(target, list)
            and all(isinstance(choice, str) and choice.strip() for choice in candidates)
            and all(type(value) in (int, float) and value == int(index == label)
                    for index, value in enumerate(target))
            and _flat_valid(row) and bool(row["query"].strip())
            and not any(key in row for key in ("questions", "paired_views", "auxiliary_views"))
            and not row.get("images"))


def _tree(entries):
    payload = "".join(f"{name}\0{digest}\n" for name, digest in sorted(entries.items()))
    return {"files": len(entries), "tree_sha256": _sha(payload.encode()),
            "files_by_path": dict(sorted(entries.items()))}


def _image(reference, source):
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"image reference must be a nonempty string: {source}")
    path = (source.parent / reference).resolve()
    payload = path.read_bytes()
    from PIL import Image
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            image_format = image.format
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError(f"image is not decodable real pixels: {path}") from error
    extensions = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif",
                  "BMP": ".bmp", "TIFF": ".tiff"}
    if image_format not in extensions:
        raise ValueError(f"unsupported image format: {image_format}")
    return path, _sha(payload), extensions[image_format]


def _images(rows, source):
    images, entries = {}, {}
    for position, row in enumerate(rows):
        if row.get("image"):
            images[position] = _image(row["image"], source)
            entries[row["image"]] = images[position][1]
    return images, _tree(entries)


def _meta(row, category, source_id, position):
    original = row.get("meta") or {}
    source = original["source"]
    license_name = original.get("license") or DEFAULT_LICENSES[source]
    if not isinstance(license_name, str) or not license_name.strip():
        raise ValueError("source license must be a nonempty string")
    return {**{key: value for key, value in original.items() if not key.startswith("r044_")},
            "panel": "r050_replay_v1", "source_id": source_id, "license": license_name,
            "r050_license_provenance": "source_metadata" if original.get("license") else "dataset_default",
            "r050_category": category, "r050_source_index": position,
            "r050_original_id": row.get("id"), "r050_original_label": row["label"]}


def _hebrew(rows, seed, count):
    safe, rejected, seen = [], Counter(), set()
    for position in sorted(range(len(rows)), key=lambda i: _rank(seed, "hebrew", i)):
        row, meta = rows[position], rows[position].get("meta") or {}
        if meta.get("source") != "MASSIVE" or meta.get("lang") != "he":
            rejected["wrong_source_or_language"] += 1
        elif not _valid(row) or len(row["candidates"]) < 10 or row.get("image"):
            rejected["invalid"] += 1
        else:
            identifier = _identity(row, position, "massive")
            if identifier in seen:
                rejected["duplicate_source"] += 1
            else:
                seen.add(identifier)
                safe.append((position, row, identifier))
    if len(safe) < count:
        raise ValueError(f"needs {count} unique valid Hebrew rows; found {len(safe)}")
    positions = sorted(range(count), key=lambda i: _rank(seed, "gold-position", i))
    result, selection = [], []
    for index, ((position, row, identifier), schedule) in enumerate(zip(safe[:count], positions)):
        gold_position = schedule % 10
        distractors = sorted((i for i in range(len(row["candidates"])) if i != row["label"]),
                             key=lambda i: _rank(seed, "distractor", [identifier, position, i]))[:9]
        candidate_indices = list(distractors)
        candidate_indices.insert(gold_position, row["label"])
        meta = {**_meta(row, "hebrew", identifier, position), "r050_gold_position": gold_position,
                "r050_selected_candidate_indices": candidate_indices}
        result.append(_grouped(row, f"r050-hebrew-{index:04d}",
                               [row["candidates"][i] for i in candidate_indices], gold_position, meta))
        selection.append({"id": result[-1]["id"], "source_id": identifier, "source_index": position,
                          "source_row_sha256": _sha(_json(row, sort_keys=True)),
                          "gold_position": gold_position, "candidate_indices": candidate_indices})
    return result, selection, dict(rejected), len(safe)


def _multimodal(rows, images, protected_hashes, protected_decisions, seed, count):
    safe, rejected, seen_images, seen_sources = [], Counter(), set(), set()
    for position in sorted(range(len(rows)), key=lambda i: _rank(seed, "multimodal", i)):
        row, meta = rows[position], rows[position].get("meta") or {}
        if meta.get("source") != "ScienceQA":
            rejected["wrong_source"] += 1
            continue
        if (not _valid(row) or position not in images
                or meta.get("variant", "real") not in {"real", "base"}
                or meta.get("r044_variant", "real") != "real"):
            rejected["invalid_or_nonreal"] += 1
            continue
        identifier = _identity(row, position, "scienceqa")
        path, digest, extension = images[position]
        if digest in protected_hashes:
            rejected["protected_image"] += 1
        elif _decision_key(row) in protected_decisions:
            rejected["protected_decision"] += 1
        elif digest in seen_images:
            rejected["duplicate_image"] += 1
        elif identifier in seen_sources:
            rejected["duplicate_source"] += 1
        elif len(row["candidates"]) >= 26 or NULL_OPTION in row["candidates"]:
            rejected["unsupported_options"] += 1
        else:
            safe.append((position, row, identifier, path, digest, extension))
            seen_images.add(digest)
            seen_sources.add(identifier)
    if len(safe) < count:
        raise ValueError(f"needs {count} unique leakage-free real-image rows; found {len(safe)}")
    result, selection, files = [], [], {}
    for index, (position, row, identifier, path, digest, extension) in enumerate(safe[:count]):
        reference = f"images/{digest}{extension}"
        meta = {**_meta(row, "multimodal", identifier, position), "variant": "real",
                "r050_image_sha256": digest}
        result.append(_grouped(row, f"r050-multimodal-{index:04d}",
                               [*row["candidates"], NULL_OPTION], row["label"], meta,
                               reference, null_last=True))
        selection.append({"id": result[-1]["id"], "source_id": identifier, "source_index": position,
                          "source_row_sha256": _sha(_json(row, sort_keys=True)), "variant": "real",
                          "image_sha256": digest, "image": reference,
                          "decision_sha256": _sha(_decision_key(row).encode())})
        files[reference] = (path, digest)
    return result, selection, dict(rejected), len(safe), files


def _atomic_publish(stage, out):
    """Atomic no-replace directory rename, including a racing empty destination."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(stage), os.fsencode(out), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = (-100, os.fsencode(stage), -100, os.fsencode(out), 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError("atomic no-replace publication requires macOS or Linux")
    function.restype = ctypes.c_int
    if function(*arguments):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(out))


def build(hebrew_path, scienceqa_train, protection_paths, out, seed=50, *,
          hebrew_count=DEFAULT_COUNTS["hebrew"], multimodal_count=DEFAULT_COUNTS["multimodal"]):
    if not protection_paths:
        raise ValueError("at least one protected ScienceQA validation/evaluation path is required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (type(hebrew_count) is not int or hebrew_count <= 0 or hebrew_count % 10
            or type(multimodal_count) is not int or multimodal_count <= 0):
        raise ValueError("counts must be positive integers; Hebrew count must be a multiple of ten")
    out = Path(out).absolute()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite output: {out}")
    hebrew, hebrew_info = _read(hebrew_path)
    science, science_info = _read(scienceqa_train)
    train_images, science_info["images"] = _images(science, Path(science_info["path"]))
    inputs = {"hebrew": hebrew_info, "scienceqa_train": science_info, "scienceqa_protection": []}
    protected_hashes, protected_decisions, all_images = set(), set(), list(train_images.values())
    for path in protection_paths:
        rows, info = _read(path)
        for row in rows:
            if ((row.get("meta") or {}).get("source") != "ScienceQA"
                    or not isinstance(row.get("state"), (str, dict, list))
                    or not isinstance(row.get("query"), str) or not row["query"].strip()
                    or not isinstance(row.get("candidates"), list) or not row["candidates"]
                    or any(not isinstance(value, str) or not value.strip() for value in row["candidates"])
                    or row.get("images")):
                raise ValueError("protected inputs must contain flat ScienceQA decision tuples")
            protected_decisions.add(_decision_key(row))
        images, info["images"] = _images(rows, Path(info["path"]))
        protected_hashes.update(item[1] for item in images.values())
        all_images.extend(images.values())
        inputs["scienceqa_protection"].append(info)
    he_rows, he_selection, he_rejected, he_eligible = _hebrew(hebrew, seed, hebrew_count)
    mm_rows, mm_selection, mm_rejected, mm_eligible, files = _multimodal(
        science, train_images, protected_hashes, protected_decisions, seed, multimodal_count)
    named_rows = {"hebrew": he_rows, "multimodal": mm_rows}
    payloads = {name: b"".join(_json(row) + b"\n" for row in rows) for name, rows in named_rows.items()}
    counts = {"hebrew": hebrew_count, "multimodal": multimodal_count}
    code_paths = [Path(__file__), Path(__file__).with_name("build_r044_mm_replay.py"),
                  Path(__file__).with_name("build_r043_causal_pairs.py"),
                  Path(__file__).with_name("train_gemma_decision_lora.py")]
    selected_hashes = {item["image_sha256"] for item in mm_selection}
    leakage_checks = {
        "protected_image_overlap": len(selected_hashes & protected_hashes),
        "protected_decision_overlap": sum(_decision_key(science[item["source_index"]]) in protected_decisions
                                          for item in mm_selection),
        "selected_image_hashes_unique": len(selected_hashes) == multimodal_count,
        "selected_source_ids_unique": all(len({item["source_id"] for item in selection}) == count
                                          for selection, count in ((he_selection, hebrew_count),
                                                                   (mm_selection, multimodal_count))),
    }
    if (leakage_checks["protected_image_overlap"] or leakage_checks["protected_decision_overlap"]
            or not leakage_checks["selected_image_hashes_unique"] or not leakage_checks["selected_source_ids_unique"]):
        raise AssertionError("selected pools failed the independent leakage/uniqueness audit")
    manifest = {
        "schema": SCHEMA, "seed": seed, "counts": counts,
        "protocol": {"optimizer_slot": "one raw grouped row / one choice question",
                     "multimodal": "singleton real-image replay; no grouped image controls",
                     "selection": "seeded SHA256 ranking without replacement; full input audit",
                     "fallback_source_id": "dataset plus one-based input line when upstream ID is absent"},
        "inputs": inputs, "code_sha256": {path.name: _sha(path.read_bytes()) for path in code_paths},
        "outputs": {name: {"path": f"{name}.jsonl", "rows": counts[name],
                           "optimizer_slots": counts[name], "rendered_views": counts[name],
                           "sha256": _sha(payloads[name]), "bytes": len(payloads[name])} for name in named_rows},
        "images": _tree({reference: digest for reference, (_, digest) in files.items()}),
        "sources_and_licenses": _licenses(named_rows),
        "selection": {"hebrew": he_selection, "multimodal": mm_selection},
        "gold_positions": {str(i): hebrew_count // 10 for i in range(10)},
        "audit": {name: {"input_rows": len(rows), "selected": counts[name], "eligible": eligible,
                         "eligible_not_selected": eligible - counts[name], "rejections": rejected}
                  for name, rows, eligible, rejected in (("hebrew", hebrew, he_eligible, he_rejected),
                                                        ("multimodal", science, mm_eligible, mm_rejected))},
        "protection": {"image_hash_count": len(protected_hashes), "decision_tuple_count": len(protected_decisions),
                       "image_hashes_sha256": _sha(_json(sorted(protected_hashes))),
                       "decision_tuples_sha256": _sha(_json(sorted(protected_decisions)))},
        "leakage_checks": leakage_checks,
    }
    if (Counter(row["meta"]["r050_gold_position"] for row in he_rows)
            != {i: hebrew_count // 10 for i in range(10)}):
        raise AssertionError("Hebrew gold positions must be exactly balanced")
    # Recheck every bound file, including rejected/unselected images, before publication.
    for info in [hebrew_info, science_info, *inputs["scienceqa_protection"]]:
        if _sha(Path(info["path"]).read_bytes()) != info["sha256"]:
            raise ValueError("input JSONL changed during construction")
    for path, digest, _ in all_images:
        if _sha(path.read_bytes()) != digest:
            raise ValueError("input image changed during construction")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        for reference, (path, digest) in files.items():
            target = stage / reference
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = path.read_bytes()
            if _sha(payload) != digest:
                raise ValueError("selected image changed during copying")
            with target.open("xb") as stream:
                stream.write(payload)
            if _sha(target.read_bytes()) != digest:
                raise IOError("copied image hash mismatch")
        from scripts.train_gemma_decision_lora import load_examples, logical_slot_views
        for name, payload in payloads.items():
            target = stage / f"{name}.jsonl"
            with target.open("xb") as stream:
                stream.write(payload)
            examples = load_examples([target])
            if (len(examples) != counts[name] or len({example["id"] for example in examples}) != counts[name]
                    or any(len(logical_slot_views(example)) != 1 for example in examples)):
                raise ValueError("output does not produce exactly one trainer slot/view per row")
            if _sha(target.read_bytes()) != manifest["outputs"][name]["sha256"]:
                raise IOError("output JSONL hash mismatch")
        with (stage / "manifest.json").open("xb") as stream:
            stream.write(_json(manifest, sort_keys=True) + b"\n")
        _atomic_publish(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # Only this call's private, explicit staging directory.
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hebrew", required=True, type=Path)
    parser.add_argument("--scienceqa-train", required=True, type=Path)
    parser.add_argument("--scienceqa-protect", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--hebrew-count", type=int, default=DEFAULT_COUNTS["hebrew"])
    parser.add_argument("--multimodal-count", type=int, default=DEFAULT_COUNTS["multimodal"])
    args = parser.parse_args()
    manifest = build(args.hebrew, args.scienceqa_train, args.scienceqa_protect, args.out, args.seed,
                     hebrew_count=args.hebrew_count, multimodal_count=args.multimodal_count)
    print(json.dumps({"schema": manifest["schema"], "counts": manifest["counts"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
