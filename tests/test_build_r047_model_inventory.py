import hashlib
import json

import pytest

from scripts.build_r047_model_inventory import (
    FROZEN_BASE_REVISION,
    build_inventory,
    main,
    verify_inventory,
)


def _model(tmp_path):
    model = (tmp_path / "models--google--gemma-4-12B-it" / "snapshots"
             / FROZEN_BASE_REVISION)
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n")
    (model / "weights.safetensors").write_bytes(b"weights")
    return model


def test_inventory_builds_and_reverifies_full_tree(tmp_path):
    model = _model(tmp_path)
    output = tmp_path / "inventory.json"
    assert main(["--model-path", str(model), "--out", str(output)]) == 0
    inventory, payload_hash = verify_inventory(output, model)
    assert inventory == build_inventory(model)
    assert inventory["files"] == 2
    assert payload_hash == hashlib.sha256(output.read_bytes()).hexdigest()


def test_inventory_rejects_model_tree_drift(tmp_path):
    model = _model(tmp_path)
    output = tmp_path / "inventory.json"
    output.write_text(json.dumps(build_inventory(model)) + "\n")
    (model / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from model inventory"):
        verify_inventory(output, model)


def test_inventory_rejects_wrong_revision_or_extra_schema_field(tmp_path):
    model = _model(tmp_path)
    with pytest.raises(ValueError, match="frozen R047 revision"):
        build_inventory(model, "wrong")
    inventory = build_inventory(model)
    inventory["unbound"] = True
    output = tmp_path / "inventory.json"
    output.write_text(json.dumps(inventory) + "\n")
    with pytest.raises(ValueError, match="schema is invalid"):
        verify_inventory(output, model)
