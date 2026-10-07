import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_r050_retention as retention


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _row(pair, suite, image=None):
    candidates = ["left", "right"]
    label, target, p_null = 0, [1.0, 0.0], 0.0
    variant = {"real": "base", "shuffled": "shuffled_forced_answer",
               "noimage": "base", "mismatch": "image_mismatch",
               "gold_dropped": "remove_gold"}.get(suite)
    if suite in {"mismatch", "gold_dropped"}:
        label, p_null = -1, 1.0
    if suite == "gold_dropped":
        candidates, target = ["right"], [1.0]
    row = {"state": "state", "query": "query", "candidates": candidates,
           "target": target, "p_null": p_null,
           "task": "massive" if suite in {"he", "en"} else "scienceqa",
           "label": label, "meta": {"pair": pair}}
    if variant:
        row["meta"]["variant"] = variant
    if image:
        row["image"] = image
    return row


def _source_fixture(tmp_path, monkeypatch):
    eval_dir = tmp_path / "data" / "eval"
    image_dir = tmp_path / "data" / "images"
    eval_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    images = {}
    for index in range(129):
        path = image_dir / f"{index}.png"
        path.write_bytes(f"image-{index}".encode())
        images[f"../images/{index}.png"] = retention.file_hash(path)
    paths, hashes = {}, {}
    for suite in retention.SUITES:
        rows = []
        for index in range(128):
            reference = None
            if suite in {"real", "gold_dropped"}:
                reference = f"../images/{index}.png"
            elif suite in {"shuffled", "mismatch"}:
                reference = f"../images/{index + 1}.png"
            rows.append(_row(f"pair-{index}", suite, reference))
        path = eval_dir / f"{suite}.jsonl"
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(payload)
        paths[suite], hashes[suite] = path, _sha(payload)
    monkeypatch.setattr(retention, "SOURCE_SHA256", hashes)
    monkeypatch.setattr(retention, "SCIENCEQA_INVENTORY_SOURCE_SHA256",
                        {suite: hashes[suite] for suite in retention.MM_SUITES})
    monkeypatch.setattr(retention, "FROZEN_SCIENCEQA_IMAGE_TREE_FILES", 129)
    image_tree = _sha("".join(
        f"{name}\0{digest}\n" for name, digest in sorted(images.items())).encode())
    monkeypatch.setattr(retention, "FROZEN_SCIENCEQA_IMAGE_TREE_SHA256", image_tree)
    protection = []
    for suite in retention.MM_SUITES:
        entries = {} if suite == "noimage" else images
        tree = _sha("".join(f"{name}\0{digest}\n"
                            for name, digest in sorted(entries.items())).encode())
        protection.append({"sha256": hashes[suite], "images": {
            "files": len(entries), "tree_sha256": tree, "files_by_path": entries}})
    inventory = {"schema": retention.ASSET_INVENTORY_SCHEMA,
                 "inputs": {"scienceqa_protection": protection}}
    inventory_path = tmp_path / "assets.json"
    inventory_path.write_text(json.dumps(inventory) + "\n")
    inventory_sha = retention.file_hash(inventory_path)
    monkeypatch.setattr(retention, "FROZEN_ASSET_INVENTORY_SHA256", inventory_sha)
    return paths, hashes, inventory_path, inventory_sha, image_dir


def _predictions(suites):
    result = {}
    for suite, cases in suites.items():
        values = []
        for case in cases:
            if suite in {"shuffled", "noimage"}:
                vector = [0.1, 0.9, 0.0]
            elif suite in {"mismatch", "gold_dropped"}:
                vector = [0.005] * (len(case["candidates"]) - 1) + [0.995]
                vector[0] += 1.0 - sum(vector)
            elif suite == "real":
                vector = [0.98, 0.01, 0.01]
            else:
                vector = [0.99, 0.01]
            predicted = max(range(len(vector)), key=vector.__getitem__)
            values.append({"id": case["id"], "probabilities": vector,
                           "predicted_index": predicted, "label": case["label"],
                           "null_probability": vector[-1]})
        result[suite] = values
    return result


def test_source_validator_uses_literal_anchors_alignment_and_authenticated_pixels(
        tmp_path, monkeypatch):
    paths, hashes, inventory, inventory_sha, image_dir = _source_fixture(tmp_path, monkeypatch)
    suites, binding = retention.validate_sources(paths, hashes, inventory, inventory_sha)
    assert all(len(rows) == 128 for rows in suites.values())
    assert binding["selected_images"]["files"] == 129
    wrong = dict(hashes, he="0" * 64)
    with pytest.raises(ValueError, match="literal frozen anchor"):
        retention.validate_sources(paths, wrong, inventory, inventory_sha)
    forged = (tmp_path / "forged-assets.json")
    forged.write_text(inventory.read_text() + " ")
    with pytest.raises(ValueError, match="literal frozen anchor"):
        retention.validate_sources(paths, hashes, forged, retention.file_hash(forged))
    (image_dir / "0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="authenticated inventory"):
        retention.validate_sources(paths, hashes, inventory, inventory_sha)


def test_source_validator_rejects_a_nonhistorical_prefix(tmp_path, monkeypatch):
    paths, hashes, inventory, inventory_sha, _ = _source_fixture(tmp_path, monkeypatch)
    he = paths["he"]
    he.write_bytes(he.read_bytes() + he.read_bytes().splitlines(keepends=True)[0])
    hashes["he"] = retention.file_hash(he)
    monkeypatch.setattr(retention, "SOURCE_SHA256", hashes)
    with pytest.raises(ValueError, match="exactly 128 rows"):
        retention.validate_sources(paths, hashes, inventory, inventory_sha)


def test_comparison_is_parent_relative_and_applies_every_gate(tmp_path, monkeypatch):
    paths, hashes, inventory, inventory_sha, _ = _source_fixture(tmp_path, monkeypatch)
    suites, _ = retention.validate_sources(paths, hashes, inventory, inventory_sha)
    parent = _predictions(suites)
    candidate = json.loads(json.dumps(parent))
    report = retention.compare(suites, parent, candidate)
    assert report["promotion_pass"] is True
    assert all(report["gates"].values())
    assert report["candidate"]["real"]["accuracy"] == 1.0
    assert report["candidate"]["answerability_auroc"] == 1.0
    assert retention._relative(0.0, 0.0)["passed"] is True
    assert retention._relative(0.001, 0.0) == {
        "parent": 0.0, "candidate": 0.001, "relative_regression": None,
        "passed": False, "zero_parent_behavior": "candidate must also equal zero"}
    missing = {name: list(values) for name, values in candidate.items()}
    missing["he"] = missing["he"][:-1]
    with pytest.raises(ValueError, match="cardinality"):
        retention.compare(suites, parent, missing)
    candidate["he"][0]["probabilities"] = [float("nan"), float("nan")]
    with pytest.raises(ValueError, match="invalid probabilities"):
        retention.compare(suites, parent, candidate)


def test_model_and_adapter_validation_are_literal_and_fail_closed(tmp_path, monkeypatch):
    revision = retention.FROZEN_BASE_REVISION
    model = tmp_path / "models--google--gemma-4-12B-it" / "snapshots" / revision
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"model")
    from scripts.build_r047_model_inventory import build_inventory
    inventory_value = build_inventory(model)
    inventory = tmp_path / "model-inventory.json"
    inventory.write_text(json.dumps(inventory_value, sort_keys=True) + "\n")
    monkeypatch.setattr(retention, "FROZEN_MODEL_INVENTORY_SHA256", retention.file_hash(inventory))
    monkeypatch.setattr(retention, "FROZEN_MODEL_TREE_SHA256", inventory_value["tree_sha256"])
    monkeypatch.setattr(retention, "FROZEN_MODEL_TREE_FILES", inventory_value["files"])
    result = retention.verify_model(model, inventory, retention.file_hash(inventory))
    assert result["model_revision"] == revision
    with pytest.raises(ValueError, match="literal frozen anchor"):
        retention.verify_model(model, inventory, "0" * 64)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": retention.MODEL_ID,
        "revision": revision}) + "\n")
    digest, files = retention.tree_hash(adapter)
    assert retention.verify_adapter(adapter, digest, files, "candidate")[
        "adapter_tree_sha256"] == digest
    (adapter / "extra.bin").write_bytes(b"drift")
    with pytest.raises(ValueError, match="externally expected tree"):
        retention.verify_adapter(adapter, digest, files, "candidate")


def test_candidate_completion_is_required_before_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "validate_sources", lambda *args: ({}, {}))
    model = {"model_inventory_sha256": "i", "base_model_tree_sha256": "m",
             "base_model_tree_files": 9}
    candidate = {"adapter_tree_sha256": "c", "adapter_tree_files": 3,
                 "adapter_weights_sha256": "w", "adapter_config_sha256": "a"}
    monkeypatch.setattr(retention, "verify_model", lambda *args: model)
    monkeypatch.setattr(retention, "verify_adapter",
                        lambda path, *args: candidate if "candidate" in str(path) else {})
    monkeypatch.setattr(retention, "validate_arm_completion", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="completion receipt is required"):
        retention.run(
            paths={}, expected_hashes={}, asset_inventory="assets",
            expected_asset_inventory_sha256="0" * 64, model="model",
            model_inventory="inventory", expected_model_inventory_sha256="0" * 64,
            parent_adapter="parent", candidate_adapter="candidate",
            expected_candidate_tree_sha256="0" * 64, expected_candidate_tree_files=3,
            candidate_completion="completion", expected_candidate_completion_sha256="0" * 64,
            out=tmp_path / "out")


def test_atomic_publication_preserves_race_winner(tmp_path, monkeypatch):
    predictions = {suite: [] for suite in retention.SUITES}
    output = tmp_path / "retention"
    sealed = retention.publish(output, predictions, predictions,
                               {"promotion_pass": True}, {"schema": retention.SCHEMA})
    assert set(sealed["outputs"]) == {
        "parent_predictions.jsonl", "candidate_predictions.jsonl", "report.json"}
    assert json.loads((output / "complete.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError, match="overwrite"):
        retention.publish(output, predictions, predictions, {"promotion_pass": True}, {})

    raced = tmp_path / "race"
    def race(stage, out):
        Path(out).mkdir()
        (Path(out) / "winner").write_text("safe")
        raise FileExistsError("publication race")
    monkeypatch.setattr(retention, "_atomic_publish", race)
    with pytest.raises(FileExistsError, match="publication race"):
        retention.publish(raced, predictions, predictions, {"promotion_pass": False}, {})
    assert (raced / "winner").read_text() == "safe"
    assert not list(tmp_path.glob(".race-*"))
