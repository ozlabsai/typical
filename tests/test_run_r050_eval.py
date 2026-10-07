import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_r050_eval as evaluator
from scripts.exact_policy_executor import execute_policy
from scripts.prepare_r050_arms import SCHEMA as RECIPE_SCHEMA, TRAINING, _canonical


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _full_panel_fixture(monkeypatch):
    rows = [
        {"id": f"r048-eval-0048-{index:06d}",
         "meta": {"semantic_group_id": "group", "semantic_signature": "signature"}}
        for index in range(2)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    outputs = {
        "train": {"rows": 2, "sha256": "1" * 64},
        "cal": {"rows": 2, "sha256": "2" * 64},
        "eval": {"rows": 2, "sha256": _sha(payload)},
    }
    audit = {
        "all_protected_fingerprints_disjoint": True,
        "cross_split_overlaps": {
            pair: {name: 0 for name in evaluator.PROTECTED_FINGERPRINTS}
            for pair in ("train__cal", "train__eval", "cal__eval")
        },
        "splits": {"eval": {"rows": 2, "generated_counterfactual_clusters": 1,
                             "unique_exact_semantic_case_clusters": 1,
                             "unique_split_neutral_semantic_signatures": 1,
                             "expected_design_cells": 1, "design_cells": 1}},
    }
    manifest = {
        "panel": evaluator.scorer.PANEL, "panel_version": "1.6", "seed": 48,
        "row_counts": {"train": 2, "cal": 2, "eval": 2}, "cluster_size": 2,
        "source_contract": {}, "headroom": {},
        "generator_sha256": evaluator.file_hash(evaluator.r048.__file__),
        "outputs": outputs, "audit": audit,
    }
    manifest_payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    monkeypatch.setattr(evaluator, "EVAL_ROWS", 2)
    monkeypatch.setattr(evaluator, "EVAL_CLUSTERS", 1)
    monkeypatch.setattr(evaluator, "FROZEN_EVAL_SHA256", _sha(payload))
    monkeypatch.setattr(evaluator, "FROZEN_MANIFEST_SHA256", _sha(manifest_payload))
    monkeypatch.setattr(evaluator, "FROZEN_OUTPUTS", outputs)
    monkeypatch.setattr(evaluator.r048, "DEFAULT_ROWS", manifest["row_counts"])
    monkeypatch.setattr(evaluator.r048, "CLUSTER_SIZE", 2)
    monkeypatch.setattr(evaluator.r048, "validate_split", lambda rows, split, count: None)
    monkeypatch.setattr(evaluator.scorer, "_row_contract", lambda row: None)
    return rows, payload, manifest, manifest_payload


def _action(name):
    return {"name": name, "payload": {"queue": name}}


def _row(identifier="row"):
    policy = {"fields": ["risk", "evidence"], "mode": "priority",
              "rules": [{"when": {"field": "risk"}, "action": _action("review")},
                        {"when": {"field": "evidence"}, "action": _action("retain")}],
              "fallback": _action("approve")}
    trace = {"risk": True, "evidence": False}
    return {
        "id": identifier, "state": "record",
        "questions": {
            "q00": {"type": "noul", "instructions": "risk?",
                    "criteria": {"no": "No", "yes": "Yes"}},
            "q01": {"type": "noul", "instructions": "evidence?",
                    "criteria": {"no": "No", "yes": "Yes"}},
        },
        "targets": {"q00": {"label": "yes"}, "q01": {"label": "no"}},
        "meta": {
            "panel": "exact_policy_v1", "panel_version": "1.6", "split": "eval",
            "semantic_group_id": "cluster", "semantic_signature": "signature",
            "counterfactual_side": 0, "language": "en", "option_order": "no_yes",
            "policy_mode": "priority", "policy_topology": "priority_gate",
            "topology_slice": "heldout", "renderer_family": "renderer",
            "source_family": "source", "program_family": "program",
            "action_family": "action", "field_order": ["risk", "evidence"],
            "qid_to_field": {"q00": "risk", "q01": "evidence"},
            "decision_threshold": 0.5, "action_trace": trace, "policy": policy,
            "gold_outcome": execute_policy(policy, trace).as_dict(),
        },
    }


def _prediction(row):
    return {"id": row["id"], "probabilities": {
        "q00": {"no": 0.1, "yes": 0.9},
        "q01": {"no": 0.9, "yes": 0.1},
    }}


def test_full_panel_validator_requires_literal_anchors_and_complete_manifest(monkeypatch):
    rows, payload, manifest, manifest_payload = _full_panel_fixture(monkeypatch)
    checked, value = evaluator.validate_full_panel(
        payload, manifest_payload, _sha(payload), _sha(manifest_payload))
    assert checked == rows
    assert value["outputs"]["eval"]["rows"] == 2

    subset = payload.splitlines(keepends=True)[0]
    rewritten = dict(manifest)
    rewritten["outputs"] = {**manifest["outputs"],
                            "eval": {"rows": 1, "sha256": _sha(subset)}}
    rewritten_payload = (json.dumps(rewritten, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="literal anchor"):
        evaluator.validate_full_panel(
            subset, rewritten_payload, _sha(subset), _sha(rewritten_payload))

    forged = json.loads(manifest_payload)
    forged["audit"]["cross_split_overlaps"]["train__eval"][
        "semantic_signature"] = 1
    forged_payload = (json.dumps(forged, sort_keys=True) + "\n").encode()
    monkeypatch.setattr(evaluator, "FROZEN_MANIFEST_SHA256", _sha(forged_payload))
    with pytest.raises(ValueError, match="protected overlap"):
        evaluator.validate_full_panel(payload, forged_payload, _sha(payload),
                                      _sha(forged_payload))


def test_prediction_validator_rejects_missing_invalid_and_false_claims():
    row = _row()
    with pytest.raises(ValueError, match="cover every panel row"):
        evaluator.validate_predictions([row], [])
    invalid = _prediction(row)
    invalid["probabilities"]["q00"] = {"no": 0.4, "yes": 0.4}
    with pytest.raises(ValueError, match="invalid prediction"):
        evaluator.validate_predictions([row], [invalid])
    false_claim = _prediction(row)
    false_claim["error_category"] = "ranking"
    with pytest.raises(ValueError, match="differs from recomputation"):
        evaluator.validate_predictions([row], [false_claim])
    report = evaluator.validate_predictions([row], [_prediction(row)])
    assert report["rendered"]["coverage"] == 1.0


def _completion_fixture(tmp_path):
    model = tmp_path / "model"
    inventory = tmp_path / "inventory.json"
    adapter = tmp_path / "step-000400"
    model.mkdir(parents=True)
    inventory.write_text("{}\n")
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}\n")
    (adapter / "trainer.pt").write_bytes(b"state")
    assets = {
        "adapter_tree_sha256": "a" * 64, "adapter_tree_files": 3,
        "adapter_weights_sha256": evaluator.file_hash(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": evaluator.file_hash(adapter / "adapter_config.json"),
        "base_model_tree_sha256": "b" * 64, "base_model_tree_files": 2,
        "model_inventory_sha256": evaluator.file_hash(inventory),
    }
    recipe = {
        "schema": RECIPE_SCHEMA, "arm_id": "slot_0123456789ab", "run_order": 1,
        "objective": {"objective": "field_ce", "action_loss_weight": 0.0,
                      "exact_policy_outcome_loss": "nll"},
        "training": dict(TRAINING),
        "epoch_contract": {"optimizer_slots": 3200, "slots_per_update": 8,
                           "updates": 400, "passes": 1,
                           "slot_order": "trainer seeded global shuffle"},
        "bindings": {"model": {"path": str(model.resolve()),
                                "revision": evaluator.FROZEN_BASE_REVISION,
                                "tree_sha256": assets["base_model_tree_sha256"],
                                "files": assets["base_model_tree_files"],
                                "inventory_path": str(inventory.resolve()),
                                "inventory_sha256": assets["model_inventory_sha256"]}},
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_bytes(_canonical(recipe))
    completion = {
        "schema": evaluator.ARM_COMPLETION_SCHEMA, "status": "complete",
        "arm_id": recipe["arm_id"], "recipe_path": str(recipe_path.resolve()),
        "recipe_sha256": evaluator.file_hash(recipe_path),
        "artifacts": {"checkpoint": {
            "path": str(adapter.resolve()), "tree_sha256": assets["adapter_tree_sha256"],
            "files": 3, "adapter_sha256": assets["adapter_weights_sha256"],
            "adapter_config_sha256": assets["adapter_config_sha256"]}},
    }
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(completion) + "\n")
    return model, inventory, adapter, assets, recipe_path, completion, completion_path


def test_completion_binds_exact_checkpoint_and_base_inventory(tmp_path):
    model, inventory, adapter, assets, recipe_path, completion, path = _completion_fixture(tmp_path)
    result = evaluator.validate_arm_completion(
        path, evaluator.file_hash(path), adapter_path=adapter, assets=assets,
        model_path=model, inventory_path=inventory)
    assert result["arm_id"] == "slot_0123456789ab"
    with pytest.raises(ValueError, match="supplied together"):
        evaluator.validate_arm_completion(path, None, adapter_path=adapter, assets=assets,
                                          model_path=model, inventory_path=inventory)
    completion["artifacts"]["checkpoint"]["tree_sha256"] = "c" * 64
    path.write_text(json.dumps(completion) + "\n")
    with pytest.raises(ValueError, match="exact candidate checkpoint"):
        evaluator.validate_arm_completion(
            path, evaluator.file_hash(path), adapter_path=adapter, assets=assets,
            model_path=model, inventory_path=inventory)

    _, _, adapter, assets, recipe_path, completion, path = _completion_fixture(tmp_path / "base")
    recipe = json.loads(recipe_path.read_text())
    recipe["bindings"]["model"]["tree_sha256"] = "d" * 64
    recipe_path.write_bytes(_canonical(recipe))
    completion["recipe_sha256"] = evaluator.file_hash(recipe_path)
    path.write_text(json.dumps(completion) + "\n")
    with pytest.raises(ValueError, match="fixed base/inventory"):
        evaluator.validate_arm_completion(
            path, evaluator.file_hash(path), adapter_path=adapter, assets=assets,
            model_path=tmp_path / "base/model", inventory_path=tmp_path / "base/inventory.json")


def test_publish_is_atomic_no_overwrite_and_exposes_prediction_hash(tmp_path, monkeypatch):
    output = tmp_path / "eval"
    sealed = evaluator.publish(output, [{"id": "row", "probabilities": {}}],
                               {"summary": True}, {"schema": evaluator.SCHEMA})
    assert sealed["prediction_sha256"] == evaluator.file_hash(output / "predictions.jsonl")
    assert json.loads((output / "complete.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluator.publish(output, [], {}, {})

    raced = tmp_path / "raced"
    def race(stage, out):
        Path(out).mkdir()
        (Path(out) / "sentinel").write_text("winner")
        raise FileExistsError("race")
    monkeypatch.setattr(evaluator, "_atomic_publish", race)
    with pytest.raises(FileExistsError, match="race"):
        evaluator.publish(raced, [], {}, {})
    assert (raced / "sentinel").read_text() == "winner"
    assert not list(tmp_path.glob(".raced-*"))


def test_run_detects_asset_change_after_cpu_mocked_inference(tmp_path, monkeypatch):
    row = _row()
    data = tmp_path / "eval.jsonl"
    manifest = tmp_path / "manifest.json"
    adapter = tmp_path / "adapter.bin"
    inventory = tmp_path / "inventory.json"
    data.write_text("panel\n")
    manifest.write_text("manifest\n")
    adapter.write_bytes(b"before")
    inventory.write_text("inventory\n")
    monkeypatch.setattr(evaluator, "validate_full_panel",
                        lambda *args: ([row], {"outputs": {}}))
    expected_adapter = evaluator.file_hash(adapter)
    assets = {"adapter": expected_adapter}
    assets.update({"base_model_tree_sha256": evaluator.FROZEN_BASE_TREE_SHA256,
                   "base_model_tree_files": evaluator.FROZEN_BASE_TREE_FILES})
    def verify(*args, **kwargs):
        if evaluator.file_hash(adapter) != expected_adapter:
            raise ValueError("asset changed")
        return assets
    monkeypatch.setattr(evaluator.trusted_runtime, "verify_assets", verify)
    monkeypatch.setattr(evaluator, "validate_arm_completion", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "_code_hashes", lambda: {"code": "frozen"})
    def infer(*args):
        adapter.write_bytes(b"after")
        return [_prediction(row)], {"runtime": True}, {"1": 2}
    monkeypatch.setattr(evaluator.trusted_runtime, "infer_rows", infer)
    monkeypatch.setattr(evaluator, "validate_runtime_fingerprint", lambda value: value)
    with pytest.raises(ValueError, match="asset changed"):
        evaluator.run(
            data, manifest, tmp_path / "model", inventory, adapter, tmp_path / "out",
            expected_data_sha256="0" * 64, expected_manifest_sha256="0" * 64,
            expected_inventory_sha256=evaluator.FROZEN_INVENTORY_SHA256,
            expected_adapter_tree_sha256="0" * 64, expected_adapter_tree_files=1)
    assert not (tmp_path / "out").exists()


def test_run_rejects_rewritten_inventory_anchor_before_asset_validation(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    manifest = tmp_path / "manifest.json"
    data.write_bytes(b"panel\n")
    manifest.write_bytes(b"manifest\n")
    monkeypatch.setattr(evaluator, "validate_full_panel", lambda *args: ([], {}))
    monkeypatch.setattr(
        evaluator.trusted_runtime, "verify_assets",
        lambda *args, **kwargs: pytest.fail("must reject before asset validation"),
    )
    with pytest.raises(ValueError, match="frozen literal anchor"):
        evaluator.run(
            data, manifest, tmp_path / "model", tmp_path / "inventory.json",
            tmp_path / "adapter", tmp_path / "out",
            expected_data_sha256="0" * 64, expected_manifest_sha256="0" * 64,
            expected_inventory_sha256="0" * 64,
            expected_adapter_tree_sha256="0" * 64, expected_adapter_tree_files=1)
