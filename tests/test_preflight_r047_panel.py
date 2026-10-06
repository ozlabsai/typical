import hashlib
import json

import pytest

from scripts.build_r047_panel import build
from scripts.preflight_r047_panel import FINGERPRINT_FIELDS, audit


MINI_COUNTS = {"unseen_k2": 1, "k3": 1, "seen_k2": 1}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return _sha(path.read_bytes())


def _plain_row(row_id, state, source):
    return {"id": row_id, "state": state,
            "questions": {"q": {"type": "choice", "instructions": f"Unique {source} prompt?",
                                   "criteria": {"a": "A", "b": "B"}}},
            "targets": {"q": {"label": "a"}},
            "meta": {"source": source, "scenario_id": f"{source}-scenario",
                     "renderer": f"{source}-renderer", "generator": f"{source}-generator"}}


def _inventory(tmp_path, benchmark_sha_override=None, training_rows=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = []
    for lineage in ("r044", "r045", "r047"):
        path = tmp_path / f"{lineage}.jsonl"
        rows = (training_rows if lineage == "r047" and training_rows is not None
                else [_plain_row(lineage, f"isolated training state {lineage}", lineage)])
        entries.append({"lineage": lineage, "path": path.name,
                        "sha256": _write_jsonl(path, rows)})
    benchmark = tmp_path / "sealed.jsonl"
    benchmark_hash = _write_jsonl(benchmark, [{
        "id": "sealed", "state": "SEALED_CONTENT_MUST_NOT_ESCAPE",
        "question": {"type": "noul", "instructions": "Sealed singular-format prompt?",
                     "criteria": {"false": "No", "true": "Yes"}},
        "expected": "false",
    }])
    manifest = {"version": 1, "training": entries,
                "benchmarks": [{"path": benchmark.name,
                                "sha256": benchmark_sha_override or benchmark_hash}]}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return path, _sha(path.read_bytes())


def test_preflight_reexecutes_cube_and_emits_content_free_hash_receipts(tmp_path):
    panel = tmp_path / "panel"
    build(panel, counts=MINI_COUNTS)
    inventory, inventory_hash = _inventory(tmp_path)
    report = audit(panel, inventory, inventory_hash, require_frozen=False)
    assert report["status"] == "pass"
    assert report["causal"]["primary"]["clusters"] == 33
    assert report["causal"]["near_transfer"]["clusters"] == 2
    assert report["legacy_smoke"] == {
        "rows": 144, "scenarios": 32, "causal_gate_eligible": False}
    assert report["benchmark_content_emitted"] is False
    assert all(value == 0 for scope in report["overlap"].values() for value in scope.values())
    assert set(report["overlap"]["training"]) == set(FINGERPRINT_FIELDS)
    assert "SEALED_CONTENT_MUST_NOT_ESCAPE" not in json.dumps(report)
    assert report["benchmark_inventories"][0]["views"] == 0
    assert set(report["benchmark_inventories"][0]) == {"sha256", "files", "rows", "views"}


def test_preflight_rejects_a_training_state_collision(tmp_path):
    panel = tmp_path / "panel"
    build(panel, counts=MINI_COUNTS)
    first = json.loads((panel / "primary.jsonl").read_text().splitlines()[0])
    collision = _plain_row("collision", first["state"], "isolated-r047")
    inventory, inventory_hash = _inventory(tmp_path, training_rows=[collision])
    with pytest.raises(ValueError, match="training leakage: raw_state overlap"):
        audit(panel, inventory, inventory_hash, require_frozen=False)


def test_preflight_fails_closed_on_manifest_and_benchmark_hashes(tmp_path):
    panel = tmp_path / "panel"
    build(panel, counts=MINI_COUNTS)
    inventory, inventory_hash = _inventory(tmp_path)
    with pytest.raises(ValueError, match="inventory-manifest hash mismatch"):
        audit(panel, inventory, "0" * 64, require_frozen=False)

    bad_inventory, bad_hash = _inventory(tmp_path / "bad", benchmark_sha_override="0" * 64)
    with pytest.raises(ValueError, match="benchmark inventory hash mismatch"):
        audit(panel, bad_inventory, bad_hash, require_frozen=False)


def test_preflight_rejects_tampered_executable_gold_even_with_updated_file_hash(tmp_path):
    panel = tmp_path / "panel"
    build(panel, counts=MINI_COUNTS)
    lines = (panel / "primary.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["meta"]["gold"] = not row["meta"]["gold"]
    lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    (panel / "primary.jsonl").write_text("\n".join(lines) + "\n")
    manifest = json.loads((panel / "manifest.json").read_text())
    manifest["outputs"]["primary"]["sha256"] = _sha((panel / "primary.jsonl").read_bytes())
    (panel / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    inventory, inventory_hash = _inventory(tmp_path)
    with pytest.raises(ValueError, match="target disagrees with independently executed gold"):
        audit(panel, inventory, inventory_hash, require_frozen=False)
