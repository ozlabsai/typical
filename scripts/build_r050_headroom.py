"""Freeze a balanced, non-contiguous R050 headroom selection from R048 v1.6.

The authenticated core is never rewritten. A full new Cartesian candidate
block supplies four schema/surface choices per base cell; sorted-cell rotation
selects one complete causal orbit per cell, retaining canonical evaluation IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_r048_action_panel as r048
from scripts.build_r050_replay_pools import _atomic_publish


SCHEMA = "r050_headroom_v2"
HEADROOM_CLUSTERS = 84
HEADROOM_ROWS = HEADROOM_CLUSTERS * r048.CLUSTER_SIZE
CANDIDATE_CLUSTERS = 336
CANDIDATE_ROWS = CANDIDATE_CLUSTERS * r048.CLUSTER_SIZE
EXPANDED_EVAL_ROWS = r048.DEFAULT_ROWS["eval"] + CANDIDATE_ROWS
FROZEN_CORE_MANIFEST_SHA256 = "9a477c41999abf2dab8c5e00f156ed96c01d318b9fd54364802be0c3e6cd95bb"
FROZEN_BUILDER_SHA256 = "a9945518f134d5b5a62195c8ab727373743a7194e54a96f688a817120f671f5e"
FROZEN_FILE_SHA256 = {
    "train": "086e461c1342ca05a9ddcc64bae2b35853f7bac1468598f861eb2df6817e6234",
    "cal": "1cbb44c4ea92b9502c5274599dd6bde83e07cc41f3f55ed59774f3841581cf81",
    "eval": "b764082343c68ec29bd94d9ca041c3e922b5aa421d40c06519683b804fd3a5e7",
}
PROTECTED_KEYS = (
    "model_input_exact", "model_input_normalized", "encoder_input_normalized",
    "encoder_endpoint_signature", "encoder_semantic_signature", "semantic_signature",
)
BASE_CELL_FIELDS = ["field_size", "policy_mode", "operator_composition", "topology_slice"]
SELECTION_METHOD = "sorted_base_cell_rotation_v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(payload: bytes, label: str) -> Any:
    def reject_constant(value):
        raise ValueError(f"{label}: nonfinite JSON value {value}")

    try:
        return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: invalid JSON") from error


def _load_core(core_dir: Path, expected_manifest_sha256: str):
    if expected_manifest_sha256 != FROZEN_CORE_MANIFEST_SHA256:
        raise ValueError("an externally expected audited v1.6 core manifest sha256 is required")
    manifest_payload = (core_dir / "manifest.json").read_bytes()
    if _sha(manifest_payload) != expected_manifest_sha256:
        raise ValueError("core manifest sha256 differs from the frozen expectation")
    manifest = _json(manifest_payload, "core manifest")
    generator_hash = _sha(Path(r048.__file__).read_bytes())
    if (not isinstance(manifest, dict) or manifest.get("panel") != "exact_policy_v1"
            or manifest.get("panel_version") != "1.6" or r048.PANEL_VERSION != "1.6"
            or generator_hash != FROZEN_BUILDER_SHA256
            or manifest.get("generator_sha256") != generator_hash):
        raise ValueError("core manifest does not bind the frozen R048 v1.6 builder")
    if (manifest.get("row_counts") != r048.DEFAULT_ROWS
            or manifest.get("cluster_size") != r048.CLUSTER_SIZE or manifest.get("seed") != 48):
        raise ValueError("core manifest must contain the complete audited default v1.6 corpus")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(r048.SPLITS):
        raise ValueError("core manifest must bind train, cal, and eval outputs")
    corpus, payloads, files = {}, {}, {}
    # Authenticate every file before parsing any of their rows.
    for split, count in r048.DEFAULT_ROWS.items():
        expectation = outputs[split]
        if (not isinstance(expectation, dict) or type(expectation.get("rows")) is not int
                or expectation["rows"] != count):
            raise ValueError(f"{split}: frozen row count mismatch")
        payload = (core_dir / f"{split}.jsonl").read_bytes()
        digest = _sha(payload)
        if digest != FROZEN_FILE_SHA256[split] or expectation.get("sha256") != digest:
            raise ValueError(f"{split}: frozen file sha256 mismatch")
        payloads[split] = payload
        files[split] = {"sha256": digest, "bytes": len(payload), "rows": count}
    for split, payload in payloads.items():
        rows = [_json(line, f"{split} row") for line in payload.splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"{split}: frozen JSONL rows must be objects")
        r048.validate_split(rows, split, r048.DEFAULT_ROWS[split])
        if r048._jsonl_bytes(rows) != payload:
            raise ValueError(f"{split}: frozen JSONL is not canonical persisted bytes")
        corpus[split] = rows
    return corpus, payloads, manifest, {
        "core_manifest": {"sha256": _sha(manifest_payload), "bytes": len(manifest_payload)},
        "core_files": files, "r048_builder_sha256": generator_hash,
    }


def _validate_expansion(expanded, corpus, payloads):
    prefix_count = r048.DEFAULT_ROWS["eval"]
    prefix = expanded[:prefix_count]
    if prefix != corpus["eval"] or r048._jsonl_bytes(prefix) != payloads["eval"]:
        raise ValueError("expanded evaluation prefix differs from frozen core objects/bytes")
    expected_ids = [f"r048-eval-0048-{index:06d}" for index in range(EXPANDED_EVAL_ROWS)]
    if [row["id"] for row in expanded] != expected_ids:
        raise ValueError("expanded evaluation IDs/order/count differ from the canonical build")
    return r048.validate_corpus({**corpus, "eval": expanded},
                                {**r048.DEFAULT_ROWS, "eval": EXPANDED_EVAL_ROWS})


def _groups(rows):
    groups, identifiers = defaultdict(list), set()
    variants = set(product((0, 1), r048.LANGUAGES, r048.OPTION_ORDERS))
    for row in rows:
        try:
            meta = row["meta"]
            index = int(meta["semantic_group_id"].rsplit("c", 1)[1])
            row_index = int(row["id"].rsplit("-", 1)[1])
            if row["id"] in identifiers or row_index // r048.CLUSTER_SIZE != index:
                raise ValueError("duplicate or inconsistent cluster/row identity")
            identifiers.add(row["id"])
            groups[index].append(row)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("invalid headroom cluster identity") from error
    for items in groups.values():
        observed = {(row["meta"]["counterfactual_side"], row["meta"]["language"],
                     row["meta"]["option_order"]) for row in items}
        if len(items) != r048.CLUSTER_SIZE or observed != variants:
            raise ValueError("each cluster must be a complete unique eight-row orbit")
    return dict(sorted(groups.items()))


def _base_cell(meta):
    return (len(meta["field_order"]), meta["policy_mode"],
            meta["operator_composition"], meta["topology_slice"])


def _pair(meta):
    return meta["value_regime"], meta["renderer_family"], meta["source_family"]


def _base_cells():
    return sorted(product(range(2, 9), ("priority", "collect"),
                          r048.SPLIT_FAMILIES["eval"]["operator"], ("same", "heldout")))


def _pairs():
    families = r048.SPLIT_FAMILIES["eval"]
    return sorted((regime, renderer, source) for regime in families["value_regime"]
                  for renderer, source in zip(families["renderer"], families["source"]))


def _factor_counts(rows):
    metas = [group[0]["meta"] for group in _groups(rows).values()]
    return {
        "field_size": {str(size): sum(len(meta["field_order"]) == size for meta in metas)
                       for size in range(2, 9)},
        **{key: dict(sorted(Counter(meta[key] for meta in metas).items()))
           for key in ("policy_mode", "operator_composition", "topology_slice", "value_regime")},
        "surface": [{"renderer": renderer, "source": source,
                     "clusters": sum((meta["renderer_family"], meta["source_family"]) == (renderer, source)
                                     for meta in metas)}
                    for renderer, source in zip(r048.SPLIT_FAMILIES["eval"]["renderer"],
                                                r048.SPLIT_FAMILIES["eval"]["source"])],
        "regime_surface": [{"value_regime": regime, "renderer": renderer, "source": source,
                            "clusters": sum(_pair(meta) == (regime, renderer, source) for meta in metas)}
                           for regime, renderer, source in _pairs()],
        "base_cells": [{"cell": list(cell), "clusters": sum(_base_cell(meta) == cell for meta in metas)}
                       for cell in _base_cells()],
        "clusters_with_at_least_five_fields": sum(len(meta["field_order"]) >= 5 for meta in metas),
    }


def _selection_receipt(indices):
    return {
        "method": SELECTION_METHOD, "base_cell_fields": BASE_CELL_FIELDS,
        "selected_cluster_indices": indices,
        "selected_cluster_indices_sha256": _sha(_canonical_json(indices)),
        "pair_order": [list(pair) for pair in _pairs()],
        "offered_pairs_per_base_cell": 4, "base_cells": HEADROOM_CLUSTERS,
        "clusters_per_pair": 21,
        "interaction_scope": "base factors fully crossed; deepest schema/surface interactions diagnostic",
    }


def validate_selection(rows, selection, factor_counts):
    """Reconstruct original units, balance, rotation, and receipts from rows."""
    groups = _groups(rows)
    indices = list(groups)
    if len(rows) != HEADROOM_ROWS or len(groups) != HEADROOM_CLUSTERS:
        raise ValueError("headroom must contain exactly 84 complete clusters / 672 rows")
    if not set(indices) <= set(range(336, 672)):
        raise ValueError("selected cluster indices must belong to the unconsumed candidate block")
    expected_order = [row["id"] for group in groups.values()
                      for row in sorted(group, key=lambda item: item["id"])]
    if [row["id"] for row in rows] != expected_order:
        raise ValueError("selected rows must retain original cluster/variant order")
    cells = {_base_cell(group[0]["meta"]): group[0]["meta"] for group in groups.values()}
    if len(cells) != HEADROOM_CLUSTERS or sorted(cells) != _base_cells():
        raise ValueError("headroom needs exactly one cluster per size/mode/operator/topology base cell")
    for rank, cell in enumerate(_base_cells()):
        if _pair(cells[cell]) != _pairs()[rank % 4]:
            raise ValueError("headroom schema/surface pairs violate sorted-base-cell rotation")
    counts = _factor_counts(rows)
    if any(item["clusters"] != 21 for item in counts["regime_surface"]):
        raise ValueError("headroom must have exactly 21 clusters per regime/surface pair")
    if selection != _selection_receipt(indices) or factor_counts != counts:
        raise ValueError("selection/factor receipt differs from reconstructed headroom rows")
    return counts


def _select_candidates(candidates):
    groups = _groups(candidates)
    if len(candidates) != CANDIDATE_ROWS or set(groups) != set(range(336, 672)):
        raise ValueError("candidate block must contain exactly the complete clusters 336 through 671")
    offered = defaultdict(dict)
    for index, group in groups.items():
        meta = group[0]["meta"]
        cell, pair = _base_cell(meta), _pair(meta)
        if pair in offered[cell]:
            raise ValueError("candidate block repeats a base-cell schema/surface pair")
        offered[cell][pair] = index
    if sorted(offered) != _base_cells() or any(sorted(pairs) != _pairs() for pairs in offered.values()):
        raise ValueError("every candidate base cell must offer all four schema/surface pairs")
    indices = sorted(offered[cell][_pairs()[rank % 4]] for rank, cell in enumerate(_base_cells()))
    rows = [row for index in indices for row in sorted(groups[index], key=lambda item: item["id"])]
    selection, counts = _selection_receipt(indices), _factor_counts(rows)
    validate_selection(rows, selection, counts)
    return rows, selection, counts


def _audit_headroom(rows, corpus):
    canonical = r048.validate_split(rows, "eval", HEADROOM_ROWS)
    fingerprints, ids = r048._fingerprints(rows), {row["id"] for row in rows}
    overlaps = {}
    for split, core_rows in corpus.items():
        core_keys = r048._fingerprints(core_rows)
        overlaps[split] = {key: len(fingerprints[key] & core_keys[key]) for key in PROTECTED_KEYS}
        overlaps[split]["row_ids"] = len(ids & {row["id"] for row in core_rows})
        if any(overlaps[split].values()):
            raise ValueError(f"headroom protected fingerprints/row IDs overlap frozen {split}")
    return {"canonical_eval_validation": canonical, "cross_core_overlaps": overlaps,
            "all_protected_fingerprints_disjoint": True, "row_ids_disjoint": True}


def build_headroom(core_dir: Path, expected_manifest_sha256: str):
    corpus, payloads, core_manifest, bindings = _load_core(Path(core_dir), expected_manifest_sha256)
    expanded = r048.build("eval", EXPANDED_EVAL_ROWS, core_manifest["seed"])
    expanded_audit = _validate_expansion(expanded, corpus, payloads)
    candidates = expanded[r048.DEFAULT_ROWS["eval"]:]
    rows, selection, counts = _select_candidates(candidates)
    audit = _audit_headroom(rows, corpus)
    manifest = {
        "schema": SCHEMA, "role": "headroom", "panel": "exact_policy_v1",
        "panel_version": r048.PANEL_VERSION, "generator_sha256": bindings["r048_builder_sha256"],
        "headroom_builder_sha256": _sha(Path(__file__).read_bytes()),
        "atomic_publisher_sha256": _sha(Path(sys.modules[_atomic_publish.__module__].__file__).read_bytes()),
        "seed": core_manifest["seed"], "cluster_size": r048.CLUSTER_SIZE,
        "row_counts": {"headroom": HEADROOM_ROWS},
        "outputs": {"headroom": {"rows": HEADROOM_ROWS, "sha256": _sha(r048._jsonl_bytes(rows))}},
        "bindings": bindings,
        "construction": {
            "canonical_row_split": "eval", "frozen_prefix_rows": r048.DEFAULT_ROWS["eval"],
            "expanded_eval_rows": len(expanded), "expanded_eval_sha256": _sha(r048._jsonl_bytes(expanded)),
            "prefix_objects_and_bytes_identical": True,
            "candidate_pool": {"cluster_index_start": 336, "cluster_index_stop": 672,
                               "clusters": CANDIDATE_CLUSTERS, "rows": CANDIDATE_ROWS,
                               "sha256": _sha(r048._jsonl_bytes(candidates))},
            "selection": selection,
        },
        "factor_counts": counts, "source_contract": core_manifest["source_contract"],
        "audit": {**audit, "expanded_corpus": expanded_audit},
    }
    return rows, manifest


def _write_synced(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_headroom(core_dir: Path, out: Path, expected_manifest_sha256: str):
    core_dir, out = Path(core_dir).resolve(), Path(out).absolute()
    if out.resolve() == core_dir or core_dir in out.resolve().parents:
        raise ValueError("headroom output must be separate from the frozen core directory")
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite headroom artifact: {out}")
    rows, manifest = build_headroom(core_dir, expected_manifest_sha256)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=out.parent))
    try:
        payload = r048._jsonl_bytes(rows)
        _write_synced(stage / "headroom.jsonl", payload)
        if ((stage / "headroom.jsonl").read_bytes() != payload
                or r048._read_jsonl(stage / "headroom.jsonl") != rows):
            raise ValueError("persisted headroom failed exact byte/object round-trip validation")
        validate_selection(rows, manifest["construction"]["selection"], manifest["factor_counts"])
        # The completion marker is last; only the complete staged directory is
        # atomically published. Exclusive rename protects even an empty racer.
        _write_synced(stage / "manifest.json", (json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        _atomic_publish(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-core-manifest-sha256", required=True)
    args = parser.parse_args()
    manifest = write_headroom(args.core_dir, args.out, args.expected_core_manifest_sha256)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
