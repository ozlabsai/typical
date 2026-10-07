"""Freeze one ordered, trainer-consumable R050 continuation mixture.

An executable eight-row semantic orbit stays contiguous in the JSONL, but each
row is one optimizer slot whose fields remain grouped by the existing trainer.
Replay rows retain their schema. Multimodal replay is singleton real-pixel CE;
this artifact does not claim grouped image controls or joint orbit optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_exact_policy_panel import _row_contract


SCHEMA = "r050_training_mix_v1"
STRATA = ("executable", "broad", "hebrew", "multimodal")
DEFAULT_WEIGHTS = dict(zip(STRATA, ("1/2", "1/4", "1/8", "1/8")))
ORBIT_SIZE = 8
PROTECTED_METADATA_HASH_KEYS = frozenset({
    "semantic_signature",
    "latent_signature",
    "exact_cluster_signature",
    "encoder_semantic_signature",
    "encoder_endpoint_signature",
})


def _json(value: Any, *, sort_keys: bool = False) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checked_hash(value: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)):
        raise ValueError("hashes must be 64-character hexadecimal sha256 values")
    return value.lower()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _children(row):
    if "meta" in row and not isinstance(row["meta"], dict):
        raise ValueError(f"{row.get('id')}: metadata must be an object")
    yield row
    for key in ("paired_views", "auxiliary_views"):
        values = row.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{row.get('id')}: {key} must be a list")
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"{row.get('id')}: nested views must be objects")
            yield from _children(value)


def row_hashes(row: dict[str, Any]) -> set[str]:
    """Hashes accepted by --locked-hash: rows, decisions, semantic identities.

    ID-free decision hashes include targets and preserve option insertion order.
    Image byte hashes and complete source-file hashes are also quarantined.
    Structural grammar labels are deliberately excluded: they are experimental
    strata expected to recur across otherwise disjoint train/evaluation cases.
    """
    hashes = set()
    for view in _children(row):
        hashes.add(_sha(_json(view, sort_keys=True)))
        content = {key: view[key] for key in (
            "state", "questions", "targets", "soft_targets", "query", "candidates",
            "label", "target",
        ) if key in view}
        if content:
            hashes.add(_sha(_json(content)))
        for key, value in (view.get("meta") or {}).items():
            if key in PROTECTED_METADATA_HASH_KEYS and isinstance(value, str):
                try:
                    hashes.add(_checked_hash(value))
                except ValueError:
                    pass
    return hashes


def _ids(row, *, include_sources=False):
    identifiers = []
    for view in _children(row):
        identifier = view.get("id")
        if isinstance(identifier, str) and identifier:
            identifiers.append(identifier)
        if include_sources:
            meta = view.get("meta") or {}
            identifiers.extend(value for key in ("source_id", "semantic_group_id")
                               if isinstance(value := meta.get(key), str) and value)
            identifiers.extend(value for value in view.get("source_ids", [])
                               if isinstance(value, str) and value)
    return identifiers


def _read(path, label, inputs, expected):
    path = Path(path).resolve()
    payload = path.read_bytes()
    digest = _sha(payload)
    if label in expected and digest != _checked_hash(expected[label]):
        raise ValueError(f"{label}: input sha256 mismatch")
    try:
        rows = [json.loads(line, object_pairs_hook=_unique_object)
                for line in payload.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: invalid JSONL") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label}: JSONL must contain nonempty object rows")
    # This also rejects NaN/Infinity before schema validation or publication.
    for row in rows:
        _json(row)
        if "meta" in row and not isinstance(row["meta"], dict):
            raise ValueError(f"{label}: metadata must be an object")
    inputs[label] = {"path": str(path), "sha256": digest, "rows": len(rows),
                     "bytes": len(payload), "expected_sha256_verified": label in expected}
    return rows, path


def _tree(entries):
    payload = "".join(f"{name}\0{digest}\n" for name, digest in sorted(entries.items()))
    return {"files": len(entries), "tree_sha256": _sha(payload.encode()),
            "files_by_path": dict(sorted(entries.items()))}


def _image(view, source, *, portable):
    reference = view.get("image")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{view.get('id')}: a real image reference is required")
    relative = PurePosixPath(reference)
    if portable and (not relative.parts or relative.is_absolute() or ".." in relative.parts
                     or relative.parts[0] in {"train.jsonl", "manifest.json"}):
        raise ValueError("image references must be safe output-relative paths")
    path = (source.parent / reference).resolve()
    payload = path.read_bytes()
    from PIL import Image
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except (OSError, ValueError) as error:
        raise ValueError(f"{view.get('id')}: image is not a decodable real image") from error
    return relative.as_posix(), path, _sha(payload)


def _slot_info(rows, path, stratum):
    """Reuse the trainer loader so a serialized row is exactly one actual slot."""
    from scripts.train_gemma_decision_lora import load_examples, logical_slot_views

    for row in rows:
        if stratum != "executable" and (row.get("meta") or {}).get("panel") == "exact_policy_v1":
            raise ValueError("exact-policy rows belong in the executable stratum")
        if stratum != "executable" and "paired_views" not in row:
            panel = (row.get("meta") or {}).get("panel")
            if panel not in {"action_v1", "exact_policy_v1"}:
                if not isinstance(row.get("questions"), dict) or len(row["questions"]) != 1:
                    raise ValueError(f"{row.get('id')}: replay needs one question per optimizer slot; "
                                     "convert flat or multi-question sources first")
        for view in _children(row):
            if view.get("images"):
                raise ValueError("multi-image replay is unsupported")
            if stratum != "multimodal" and view.get("image"):
                raise ValueError(f"{stratum}: image rows belong in multimodal replay")
            if stratum == "multimodal":
                variant = (view.get("meta") or {}).get("r044_variant",
                          (view.get("meta") or {}).get("variant", "real"))
                if variant != "real" or "paired_views" in row or "auxiliary_views" in row:
                    raise ValueError("multimodal replay must be singleton real-image rows")
                if not isinstance(row.get("questions"), dict) or len(row["questions"]) != 1:
                    raise ValueError("multimodal replay must have one question")
    try:
        examples = load_examples([path])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{stratum}: source is not trainer-consumable: {error}") from error
    by_id = {example["id"]: example for example in examples}
    if len(by_id) != len(examples):
        raise ValueError(f"{stratum}: duplicate trainer slot ids")
    result = {}
    for row in rows:
        grouped = ((row.get("meta") or {}).get("panel") in {"action_v1", "exact_policy_v1"}
                   or "paired_views" in row)
        identifier = row["id"] if grouped else f"{row['id']}/{next(iter(row['questions']))}"
        if identifier not in by_id:
            raise ValueError(f"{row['id']}: source row did not produce one optimizer slot")
        views = logical_slot_views(by_id[identifier])
        for view in views:
            if any(not math.isfinite(float(value)) for value in view["probabilities"]):
                raise ValueError(f"{row['id']}: nonfinite training target")
        result[row["id"]] = {"optimizer_slots": 1, "rendered_views": len(views),
                             "optimizer_slot_id": identifier}
    if len(result) != len(examples):
        raise ValueError(f"{stratum}: source row/optimizer slot counts differ")
    return result


def _executable_units(rows, manifest):
    if (manifest.get("panel") != "exact_policy_v1"
            or not isinstance(manifest.get("panel_version"), str)
            or not manifest["panel_version"]):
        raise ValueError("executable manifest must freeze an exact-policy panel version")
    groups = defaultdict(list)
    for row in rows:
        meta = row.get("meta") or {}
        if meta.get("split") != "train" or meta.get("panel_version") != manifest["panel_version"]:
            raise ValueError("executable source must be the frozen train split")
        _row_contract(row)
        expected_order = ("no", "yes") if meta.get("option_order") == "no_yes" else ("yes", "no")
        if any(tuple(question["criteria"]) != expected_order
               for question in row["questions"].values()):
            raise ValueError(f"{row['id']}: serialized option order disagrees with metadata")
        for key in ("semantic_group_id", "semantic_signature"):
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f"{row['id']}: missing {key}")
        _checked_hash(meta["semantic_signature"])
        if type(meta.get("counterfactual_side")) is not int:
            raise ValueError(f"{row['id']}: counterfactual sides must be integers")
        groups[meta["semantic_group_id"]].append(row)
    variants = {(side, language, order) for side in (0, 1)
                for language in ("en", "he") for order in ("no_yes", "yes_no")}
    units = []
    for identifier, group in sorted(groups.items()):
        keys = [(r["meta"].get("counterfactual_side"), r["meta"].get("language"),
                 r["meta"].get("option_order")) for r in group]
        if len(group) != ORBIT_SIZE or set(keys) != variants:
            raise ValueError(f"{identifier}: executable semantic orbit is incomplete")
        for key in ("semantic_signature", "policy", "field_order", "qid_to_field"):
            if len({_json(r["meta"].get(key), sort_keys=True) for r in group}) != 1:
                raise ValueError(f"{identifier}: inconsistent executable {key}")
        for side in (0, 1):
            for key in ("action_trace", "gold_outcome", "facts"):
                if len({_json(r["meta"].get(key), sort_keys=True) for r in group
                        if r["meta"]["counterfactual_side"] == side}) != 1:
                    raise ValueError(f"{identifier}: inconsistent endpoint {key}")
        endpoints = {r["meta"]["counterfactual_side"]: r["meta"] for r in group}
        if (sum(endpoints[0]["action_trace"][field] != endpoints[1]["action_trace"][field]
                for field in endpoints[0]["field_order"]) != 1
                or endpoints[0]["gold_outcome"] == endpoints[1]["gold_outcome"]):
            raise ValueError(f"{identifier}: executable twins must flip one field and the outcome")
        ordered = sorted(group, key=lambda r: (r["meta"]["counterfactual_side"],
                         r["meta"]["language"], r["meta"]["option_order"]))
        units.append((identifier, ordered))
    return units


def _counts(total_slots, weights):
    if isinstance(total_slots, bool) or not isinstance(total_slots, int) or total_slots <= 0:
        raise ValueError("total_slots must be a positive integer")
    weights = DEFAULT_WEIGHTS if weights is None else weights
    if set(weights) != set(STRATA) or any(isinstance(value, bool) for value in weights.values()):
        raise ValueError("weights must name exactly executable, broad, hebrew, multimodal")
    fractions = {name: Fraction(str(value)) for name, value in weights.items()}
    if any(value < 0 for value in fractions.values()) or sum(fractions.values()) != 1:
        raise ValueError("mixture weights must be nonnegative and sum to one")
    counts = {name: total_slots * value for name, value in fractions.items()}
    if any(value.denominator != 1 for value in counts.values()):
        raise ValueError("mixture weights must yield exact integer optimizer-slot counts")
    counts = {name: int(value) for name, value in counts.items()}
    if counts["executable"] % ORBIT_SIZE:
        raise ValueError("executable optimizer slots must be a multiple of eight")
    return counts, {name: str(value) for name, value in fractions.items()}


def build_mix(
    executable_path: str | Path,
    panel_manifest_path: str | Path,
    broad_paths: Iterable[str | Path],
    hebrew_paths: Iterable[str | Path],
    multimodal_paths: Iterable[str | Path],
    out: str | Path,
    *, seed: int = 50, total_slots: int = 3200, weights=None,
    locked_ids: Iterable[str] = (), locked_hashes: Iterable[str] = (),
    locked_eval_paths: Iterable[str | Path] = (), expected_hashes=None,
) -> dict[str, Any]:
    """Publish train.jsonl, unchanged image references, and a deterministic receipt."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite mix directory: {out}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    counts, fractions = _counts(total_slots, weights)
    expected = dict(expected_hashes or {})
    inputs = {}
    blocked_ids = set(locked_ids)
    if any(not isinstance(value, str) or not value for value in blocked_ids):
        raise ValueError("locked ids must be nonempty strings")
    blocked_hashes = {_checked_hash(value) for value in locked_hashes}
    panel_manifest_path = Path(panel_manifest_path).resolve()
    manifest_bytes = panel_manifest_path.read_bytes()
    if "panel_manifest" in expected and _sha(manifest_bytes) != _checked_hash(expected["panel_manifest"]):
        raise ValueError("panel_manifest: input sha256 mismatch")
    panel_manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_object)
    if not isinstance(panel_manifest, dict) or not isinstance(panel_manifest.get("outputs"), dict):
        raise ValueError("panel manifest must be an object with frozen outputs")
    inputs["panel_manifest"] = {"path": str(panel_manifest_path), "sha256": _sha(manifest_bytes),
                                "bytes": len(manifest_bytes),
                                "expected_sha256_verified": "panel_manifest" in expected}
    for index, path in enumerate(locked_eval_paths):
        label = f"locked_eval:{index}"
        rows, source = _read(path, label, inputs, expected)
        blocked_hashes.add(inputs[label]["sha256"])
        for row in rows:
            blocked_ids.update(_ids(row, include_sources=True))
            blocked_hashes.update(row_hashes(row))
            for view in _children(row):
                if view.get("image"):
                    blocked_hashes.add(_image(view, source, portable=False)[2])
    paths = {"executable": [executable_path], "broad": list(broad_paths),
             "hebrew": list(hebrew_paths), "multimodal": list(multimodal_paths)}
    pools = {name: [] for name in STRATA}
    slot_info, images, seen_ids = {}, {}, set()
    for stratum in STRATA:
        for index, path in enumerate(paths[stratum]):
            label = "executable" if stratum == "executable" else f"{stratum}:{index}"
            rows, source = _read(path, label, inputs, expected)
            if inputs[label]["sha256"] in blocked_hashes:
                raise ValueError(f"{label}: source file is locked evaluation material")
            for row in rows:
                actual_ids = _ids(row)
                if any(not isinstance(view.get("id"), str) or not view["id"]
                       for view in _children(row)):
                    raise ValueError(f"{label}: every row/view needs a nonempty id")
                if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) & seen_ids:
                    raise ValueError(f"{label}: overlapping row/view ids")
                seen_ids.update(actual_ids)
            slot_info.update(_slot_info(rows, source, stratum))
            if _sha(source.read_bytes()) != inputs[label]["sha256"]:
                raise ValueError(f"{label}: source changed during validation")
            image_entries = {}
            for position, row in enumerate(rows):
                if stratum == "multimodal":
                    reference, image_path, digest = _image(row, source, portable=True)
                    images[row["id"]] = (reference, image_path, digest)
                    image_entries[reference] = digest
                pools[stratum].append((row, label, position))
            if image_entries:
                inputs[label]["referenced_images"] = _tree(image_entries)
    if set(expected) - set(inputs):
        raise ValueError("expected_hashes contains unknown input labels")
    slot_ids = [info["optimizer_slot_id"] for info in slot_info.values()]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("overlapping trainer optimizer-slot ids")
    executable_rows = [item[0] for item in pools["executable"]]
    frozen = panel_manifest.get("outputs", {}).get("train")
    if (not isinstance(frozen, dict) or frozen.get("rows") != len(executable_rows)
            or _checked_hash(frozen.get("sha256")) != inputs["executable"]["sha256"]):
        raise ValueError("executable train bytes/count differ from frozen panel manifest")
    panel_generator = _checked_hash(panel_manifest.get("generator_sha256"))
    executable_units = _executable_units(executable_rows, panel_manifest)
    positions = {row["id"]: (label, position) for pool in pools.values()
                 for row, label, position in pool}
    def blocked(row):
        return (bool(set(_ids(row, include_sources=True)) & blocked_ids)
                or bool(row_hashes(row) & blocked_hashes)
                or (row["id"] in images and images[row["id"]][2] in blocked_hashes))
    def rank(name, identifier):
        return _sha(_json(["r050", seed, name, identifier]))
    selected_units, exclusions = [], {}
    for stratum in STRATA:
        units = (executable_units if stratum == "executable"
                 else [(row["id"], [row]) for row, _, _ in pools[stratum]])
        eligible = [(identifier, rows) for identifier, rows in units if not any(blocked(r) for r in rows)]
        exclusions[stratum] = {"units": len(units) - len(eligible),
                               "optimizer_slots": sum(len(rows) for _, rows in units)
                                                  - sum(len(rows) for _, rows in eligible)}
        needed = counts[stratum] // ORBIT_SIZE if stratum == "executable" else counts[stratum]
        if len(eligible) < needed:
            raise ValueError(f"{stratum}: needs {needed} eligible units, found {len(eligible)}")
        eligible.sort(key=lambda unit: (rank(stratum, unit[0]), unit[0]))
        selected_units.extend((stratum, identifier, rows) for identifier, rows in eligible[:needed])
    selected_units.sort(key=lambda unit: (rank("order", f"{unit[0]}:{unit[1]}"), unit[0], unit[1]))
    output_rows, selection, summaries, copied_images = [], [], {}, {}
    for stratum in STRATA:
        summaries[stratum] = {"optimizer_slots": 0, "rendered_rows": 0,
                              "rendered_views": 0, "semantic_groups": 0,
                              "semantic_signatures": 0}
    signatures = defaultdict(set)
    for stratum, identifier, rows in selected_units:
        start = len(output_rows)
        for row in rows:
            info = slot_info[row["id"]]
            output_rows.append(row)
            summaries[stratum]["optimizer_slots"] += info["optimizer_slots"]
            summaries[stratum]["rendered_rows"] += 1
            summaries[stratum]["rendered_views"] += info["rendered_views"]
            signature = (row.get("meta") or {}).get("semantic_signature")
            if signature:
                signatures[stratum].add(signature)
            if row["id"] in images:
                reference, source, digest = images[row["id"]]
                payload = source.read_bytes()
                if _sha(payload) != digest:
                    raise ValueError("source image changed during mix construction")
                if reference in copied_images and copied_images[reference] != payload:
                    raise ValueError(f"conflicting preserved image path: {reference}")
                copied_images[reference] = payload
        summaries[stratum]["semantic_groups"] += int(stratum == "executable")
        selection.append({"stratum": stratum, "unit_id": identifier, "slot_start": start,
                          "slot_count": len(rows), "rows": [
                              {"id": r["id"], "input": positions[r["id"]][0],
                               "source_position": positions[r["id"]][1]} for r in rows]})
    for stratum in STRATA:
        summaries[stratum]["semantic_signatures"] = len(signatures[stratum])
    payload = b"".join(_json(row) + b"\n" for row in output_rows)
    image_tree = _tree({name: _sha(data) for name, data in copied_images.items()})
    output_names = {PurePosixPath(name) for name in copied_images} | {
        PurePosixPath("train.jsonl"), PurePosixPath("manifest.json")}
    if any(parent in output_names for name in output_names for parent in name.parents):
        raise ValueError("preserved image paths conflict with output files/directories")
    code_paths = [Path(__file__), Path(sys.modules[_row_contract.__module__].__file__),
                  Path(sys.modules["scripts.exact_policy_executor"].__file__),
                  Path(sys.modules["scripts.train_gemma_decision_lora"].__file__)]
    receipt = {
        "schema": SCHEMA, "seed": seed, "weights": fractions,
        "protocol": {"serialization": "raw trainer-consumable rows; option order preserved",
            "executable": "complete eight-row semantic orbits, contiguous; eight optimizer slots",
            "optimizer_slot": "one exact/action group, existing paired wrapper, or single-question row",
            "multimodal": "singleton real-image replay; no grouped image-control objective",
            "amendment": "semantic twins share an input block, not one optimizer slot",
            "sampling": "without replacement; SHA256 seed/stratum/id ranking and block ordering"},
        "inputs": inputs, "panel_generator_sha256": panel_generator,
        "code_sha256": {path.name: _sha(path.read_bytes()) for path in code_paths},
        "strata": summaries, "optimizer_slots": len(output_rows),
        "rendered_views": sum(info["rendered_views"] for info in summaries.values()),
        "selection": selection, "images": image_tree,
        "quarantine": {"ids": len(blocked_ids), "hashes": len(blocked_hashes),
            "ids_sha256": _sha(_json(sorted(blocked_ids))),
            "hashes_sha256": _sha(_json(sorted(blocked_hashes))), "excluded": exclusions,
            "selected_id_overlap": 0, "selected_hash_overlap": 0},
        "outputs": {"train": {"rows": len(output_rows), "optimizer_slots": len(output_rows),
                               "sha256": _sha(payload), "bytes": len(payload)}},
    }
    manifest_payload = _json(receipt, sort_keys=True) + b"\n"
    out.mkdir(parents=True, exist_ok=False)
    for name, data in [("train.jsonl", payload), *sorted(copied_images.items()),
                       ("manifest.json", manifest_payload)]:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        if target.read_bytes() != data:
            raise ValueError(f"persisted artifact failed byte verification: {name}")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable-panel", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    for name in STRATA[1:]:
        parser.add_argument(f"--{name}", action="append", default=[], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--slots", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--weight", action="append", default=[], help="STRATUM=FRACTION (all four)")
    parser.add_argument("--locked-eval", action="append", default=[], type=Path)
    parser.add_argument("--locked-id", action="append", default=[])
    parser.add_argument("--locked-hash", action="append", default=[])
    parser.add_argument("--expected-sha256", action="append", default=[],
                        help="INPUT_LABEL=SHA256; e.g. executable, panel_manifest, broad:0")
    args = parser.parse_args(argv)
    def mapping(values):
        result = {}
        for value in values:
            key, item = value.split("=", 1)
            if key in result:
                raise ValueError(f"duplicate setting: {key}")
            result[key] = item
        return result
    receipt = build_mix(args.executable_panel, args.panel_manifest, args.broad,
        args.hebrew, args.multimodal, args.out, seed=args.seed, total_slots=args.slots,
        weights=mapping(args.weight) if args.weight else None, locked_eval_paths=args.locked_eval,
        locked_ids=args.locked_id, locked_hashes=args.locked_hash,
        expected_hashes=mapping(args.expected_sha256))
    print(json.dumps({"schema": SCHEMA, "optimizer_slots": receipt["optimizer_slots"],
                      "train_sha256": receipt["outputs"]["train"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
