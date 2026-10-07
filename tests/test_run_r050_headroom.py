import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_r050_headroom as runner
from scripts.build_r047_model_inventory import build_inventory
from scripts.exact_policy_executor import execute_policy
from scripts.singleton_option_scorer import RUNTIME_BOOLS, RUNTIME_FIELDS


def _rows(specs=(("priority", 5), ("collect", 2))):
    result = []
    for group, (mode, size) in enumerate(specs):
        fields = [f"field{i}" for i in range(size)]
        policy = {"fields": fields, "mode": mode,
                  "rules": [{"when": {"field": field}, "action": {"name": f"action{i}", "payload": {}},
                             **({"primary_rank": None} if mode == "collect" else {})}
                            for i, field in enumerate(fields)], "fallback": {"name": "default", "payload": {}}}
        for side in (0, 1):
            trace = dict.fromkeys(fields, False)
            trace[fields[0]] = bool(side)
            for lang in ("en", "he"):
                for order in ("no_yes", "yes_no"):
                    qids = [f"q{i:02d}" for i in range(size)]
                    result.append({"id": f"row-{group}-{side}-{lang}-{order}", "state": f"case {group}",
                        "questions": {qid: {"type": "noul", "instructions": field,
                                             "criteria": {label: label for label in order.split("_")}}
                                      for qid, field in zip(qids, fields)},
                        "targets": {qid: {"label": "yes" if trace[field] else "no"} for qid, field in zip(qids, fields)},
                        "meta": {"panel": "exact_policy_v1", "policy": policy, "policy_mode": mode,
                                 "field_order": fields, "qid_to_field": dict(zip(qids, fields)),
                                 "decision_threshold": 0.5, "action_trace": trace,
                                 "gold_outcome": execute_policy(policy, trace).as_dict(),
                                 "semantic_group_id": f"group-{group}", "semantic_signature": runner._sha(str(group).encode()),
                                 "counterfactual_side": side, "language": lang, "option_order": order}})
    return result


def _predictions(rows, failed=()):
    failed = set(failed)
    result = []
    for index, row in enumerate(rows):
        probabilities = {}
        for qid, field in row["meta"]["qid_to_field"].items():
            truth = row["meta"]["action_trace"][field]
            if index in failed and qid == "q00": truth = not truth
            probabilities[qid] = {"no": float(not truth), "yes": float(truth)}
        result.append({"id": row["id"], "probabilities": probabilities})
    return result


@pytest.fixture(scope="module")
def canonical_payloads():
    builder = runner.r048
    prefix = builder.DEFAULT_ROWS["eval"]
    count = runner.headroom.HEADROOM_ROWS
    rows = builder.build("eval", prefix + count, seed=48)[prefix:]
    payload = builder._jsonl_bytes(rows)
    builder_hash = runner.file_hash(builder.__file__)
    manifest = {"schema": runner.headroom.SCHEMA, "role": "headroom", "panel": "exact_policy_v1",
                "panel_version": builder.PANEL_VERSION, "generator_sha256": builder_hash,
                "headroom_builder_sha256": runner.file_hash(runner.headroom.__file__), "seed": 48,
                "cluster_size": 8, "row_counts": {"headroom": count},
                "outputs": {"headroom": {"rows": count, "sha256": runner._sha(payload)}},
                "bindings": {"r048_builder_sha256": builder_hash,
                             "core_manifest": {"sha256": "a" * 64},
                             "core_files": {split: {"rows": n, "sha256": "b" * 64}
                                            for split, n in builder.DEFAULT_ROWS.items()}},
                "construction": {"canonical_row_split": "eval", "frozen_prefix_rows": prefix,
                                 "expanded_eval_rows": prefix + count, "tail_clusters": count // 8,
                                 "prefix_objects_and_bytes_identical": True}}
    return rows, payload, manifest


def _validate(payload, manifest):
    encoded = runner._json(manifest)
    return runner.validate_headroom(payload, encoded, runner._sha(payload), runner._sha(encoded))


def _assets(tmp_path):
    model = tmp_path / "models--google--gemma-4-12B-it" / "snapshots" / runner.FROZEN_BASE_REVISION
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"gemma4"}')
    (model / "model.safetensors").write_bytes(b"synthetic model fixture, never loaded")
    inventory = tmp_path / "inventory.json"
    inventory.write_bytes(runner._json(build_inventory(model)))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_bytes(runner._json({"peft_type": "LORA", "base_model_name_or_path": runner.MODEL_ID}))
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter fixture, never loaded")
    digest, count = runner.tree_hash(adapter)
    return model, inventory, adapter, {"expected_inventory_sha256": runner.file_hash(inventory),
                                      "expected_adapter_tree_sha256": digest, "expected_adapter_tree_files": count}


def _runtime():
    value = dict.fromkeys(RUNTIME_FIELDS, True)
    value.update({key: False for key in RUNTIME_BOOLS})
    value.update(torch_version="fixture", transformers_version="fixture", peft_version="fixture",
                 cuda_runtime_version="fixture", cudnn_version=1, gpu_name="synthetic-not-a-GPU-run",
                 gpu_capability=[8, 0], driver_version="fixture", base_parameter_dtype_counts={"bfloat16": 1},
                 adapter_parameter_dtype_counts={"float32": 1}, observed_logits_dtype="bfloat16",
                 text_attention_implementations=["sdpa"], sdpa_profiler_observed=True)
    return value


def test_canonical_full_tail_external_hashes_and_current_builder_validation(canonical_payloads):
    rows, payload, manifest = canonical_payloads
    assert _validate(payload, manifest)[0] == rows
    assert len(rows) == 672
    encoded = runner._json(manifest)
    for data_hash, manifest_hash in ((None, runner._sha(encoded)), (runner._sha(payload), None),
                                     ("0" * 64, runner._sha(encoded)), (runner._sha(payload), "0" * 64)):
        with pytest.raises(ValueError, match="externally expected"):
            runner.validate_headroom(payload, encoded, data_hash, manifest_hash)


@pytest.mark.parametrize("key,value", [("role", "eval"), ("schema", "untrusted"), ("panel_version", "1.4"),
                                        ("generator_sha256", "0" * 64), ("headroom_builder_sha256", "0" * 64),
                                        ("cluster_size", True), ("seed", True)])
def test_authenticated_but_invalid_manifest_fails(canonical_payloads, key, value):
    _, payload, manifest = canonical_payloads
    changed = copy.deepcopy(manifest)
    changed[key] = value
    with pytest.raises(ValueError): _validate(payload, changed)


@pytest.mark.parametrize("change", ["row_count", "output_hash", "core_binding", "prefix", "tail_size", "no_tail_claim"])
def test_tampered_bindings_and_core_relabeling_fail(canonical_payloads, change):
    _, payload, manifest = canonical_payloads
    changed = copy.deepcopy(manifest)
    if change == "row_count": changed["outputs"]["headroom"]["rows"] -= 1
    if change == "output_hash": changed["outputs"]["headroom"]["sha256"] = "0" * 64
    if change == "core_binding": changed["bindings"]["r048_builder_sha256"] = "0" * 64
    if change == "prefix": changed["construction"]["frozen_prefix_rows"] = 0
    if change == "tail_size": changed["construction"]["tail_clusters"] -= 1
    if change == "no_tail_claim": changed["construction"]["prefix_objects_and_bytes_identical"] = False
    with pytest.raises(ValueError): _validate(payload, changed)


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_variant", "bad_gold", "signature", "row_order", "version_stripped"])
def test_forged_rows_still_fail_when_hashes_and_manifest_rebound(canonical_payloads, change):
    rows, _, manifest = canonical_payloads
    changed = copy.deepcopy(rows)
    if change == "duplicate_id": changed[1]["id"] = changed[0]["id"]
    if change == "duplicate_variant": changed[1]["meta"]["counterfactual_side"] = changed[0]["meta"]["counterfactual_side"]
    if change == "bad_gold": changed[0]["targets"]["q00"]["label"] = "yes" if changed[0]["targets"]["q00"]["label"] == "no" else "no"
    if change == "signature": changed[0]["meta"]["semantic_signature"] = "0" * 64
    if change == "row_order": changed[0], changed[1] = changed[1], changed[0]
    if change == "version_stripped":
        for row in changed: row["meta"].pop("panel_version")
    payload = runner.r048._jsonl_bytes(changed)
    rebound = copy.deepcopy(manifest)
    rebound["outputs"]["headroom"]["sha256"] = runner._sha(payload)
    with pytest.raises(ValueError): _validate(payload, rebound)


def test_error_gate_recomputes_outcomes_and_reports_rows_and_independent_clusters():
    rows = _rows()
    report = runner.evaluate_headroom(rows, _predictions(rows, failed=(0, 8)))
    assert report["headroom_sufficient"] is True
    assert report["overall"] == {"rows": 16, "row_errors": 2, "independent_clusters": 2,
                                  "failed_independent_clusters": 2, "complete_outcome_error_rate": 0.125}
    assert report["fields_at_least_five"]["failed_independent_clusters"] == 1
    assert report["valid_predictions"] == 16 and report["invalid"] == 0 and report["coverage"] == 1
    assert report == runner.evaluate_headroom(rows, list(reversed(_predictions(rows, failed=(0, 8)))))


def test_ten_percent_boundary_is_exact_integer_gate_not_rounding():
    rows = _rows((("priority", 5), ("collect", 5)) * 5)
    passing = runner.evaluate_headroom(rows, _predictions(rows, failed=(0, 1, 2, 3, 8, 9, 10, 11)))
    failing = runner.evaluate_headroom(rows, _predictions(rows, failed=(0, 1, 2, 8, 9, 10, 11)))
    assert passing["headroom_sufficient"] is True and passing["overall"]["complete_outcome_error_rate"] == 0.1
    assert failing["headroom_sufficient"] is False
    assert failing["failures"] == ["complete_outcome_error_at_least_ten_percent"]


def test_saturation_modes_and_large_field_errors_are_separate_required_gates():
    rows = _rows((("priority", 5), ("collect", 5), ("priority", 2), ("collect", 2)))
    saturated = runner.evaluate_headroom(rows, _predictions(rows))
    assert not saturated["headroom_sufficient"] and saturated["overall"]["row_errors"] == 0
    priority_only = runner.evaluate_headroom(rows, _predictions(rows, range(8)))
    assert priority_only["failures"] == ["collect_has_failed_independent_cluster"]
    small_only = runner.evaluate_headroom(rows, _predictions(rows, (16, 17, 24, 25)))
    assert small_only["failures"] == ["five_or_more_fields_has_failed_independent_cluster"]


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "nonobject", "missing_qid", "nan", "infinity",
                                     "huge_integer", "boolean", "bad_sum", "claimed_outcome", "claimed_trace_type", "nonfinite_extra"])
def test_invalid_predictions_cannot_create_headroom_errors_or_pass(change):
    rows = _rows()
    predictions = _predictions(rows, (0, 8))
    if change == "missing": predictions.pop()
    if change == "extra": predictions.append({"id": "unknown", "probabilities": {}})
    if change == "duplicate": predictions[-1] = predictions[0]
    if change == "nonobject": predictions[-1] = None
    if change == "missing_qid": predictions[0]["probabilities"].pop("q00")
    if change in {"nan", "infinity", "huge_integer", "boolean", "bad_sum"}:
        predictions[0]["probabilities"]["q00"]["yes"] = {"nan": float("nan"), "infinity": float("inf"),
                                                           "huge_integer": 10 ** 400, "boolean": True, "bad_sum": 0.4}[change]
    if change == "claimed_outcome": predictions[0]["predicted_outcome"] = rows[0]["meta"]["gold_outcome"]
    if change == "claimed_trace_type": predictions[1]["predicted_trace"] = {key: int(value) for key, value in rows[1]["meta"]["action_trace"].items()}
    if change == "nonfinite_extra": predictions[0]["ignored"] = float("nan")
    with pytest.raises(ValueError): runner.evaluate_headroom(rows, predictions)


@pytest.mark.parametrize("change", ["duplicate_id", "incomplete", "duplicate_variant", "signature_split", "signature_merge",
                                     "mode_relabel", "field_relabel", "order_relabel"])
def test_bad_cluster_metadata_cannot_inflate_independent_failures(change):
    rows = _rows()
    if change == "duplicate_id": rows[1]["id"] = rows[0]["id"]
    if change == "incomplete": rows.pop()
    if change == "duplicate_variant": rows[1]["meta"]["option_order"] = rows[0]["meta"]["option_order"]
    if change == "signature_split": rows[0]["meta"]["semantic_signature"] = "f" * 64
    if change == "signature_merge":
        for row in rows[8:]: row["meta"]["semantic_signature"] = rows[0]["meta"]["semantic_signature"]
    if change == "mode_relabel": rows[0]["meta"]["policy_mode"] = "collect"
    if change == "field_relabel": rows[0]["meta"]["field_order"] = rows[0]["meta"]["field_order"][:2]
    if change == "order_relabel": rows[0]["meta"]["option_order"] = "yes_no"
    with pytest.raises(ValueError): runner.evaluate_headroom(rows, _predictions(rows))


def test_actual_inventory_and_adapter_trees_require_external_anchors(tmp_path):
    model, inventory, adapter, expected = _assets(tmp_path)
    receipt = runner.verify_assets(model, inventory, adapter, **expected)
    assert receipt["model_revision"] == runner.FROZEN_BASE_REVISION
    assert receipt["adapter_tree_files"] == 2
    for key, value in (("expected_inventory_sha256", "0" * 64), ("expected_adapter_tree_sha256", "0" * 64),
                       ("expected_adapter_tree_files", 3), ("expected_adapter_tree_files", True)):
        changed = {**expected, key: value}
        with pytest.raises(ValueError): runner.verify_assets(model, inventory, adapter, **changed)
    (model / "model.safetensors").write_bytes(b"tampered base")
    with pytest.raises(ValueError, match="model tree"):
        runner.verify_assets(model, inventory, adapter, **expected)


def test_adapter_payload_symlink_and_base_revision_tampering_fail(tmp_path):
    model, inventory, adapter, expected = _assets(tmp_path)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"different parent")
    with pytest.raises(ValueError, match="parent adapter tree"):
        runner.verify_assets(model, inventory, adapter, **expected)
    weights.unlink()
    weights.symlink_to(model / "model.safetensors")
    with pytest.raises(ValueError, match="symbolic"):
        runner.verify_assets(model, inventory, adapter, **expected)
    weights.unlink()
    weights.write_bytes(b"fixture")
    config = adapter / "adapter_config.json"
    config.write_bytes(runner._json({"peft_type": "LORA", "base_model_name_or_path": runner.MODEL_ID, "revision": "wrong"}))
    digest, count = runner.tree_hash(adapter)
    expected.update(expected_adapter_tree_sha256=digest, expected_adapter_tree_files=count)
    with pytest.raises(ValueError, match="different base revision"):
        runner.verify_assets(model, inventory, adapter, **expected)


def test_atomic_receipt_predictions_report_and_completion_marker_bind_all_bytes(tmp_path):
    rows = _rows()
    predictions = _predictions(rows, (0, 8))
    report = runner.evaluate_headroom(rows, predictions)
    receipt = {"schema": runner.RECEIPT_SCHEMA, "fixture_only": True}
    one = runner.publish(tmp_path / "one", predictions, report, receipt)
    two = runner.publish(tmp_path / "two", predictions, report, receipt)
    assert one == two
    for name in ("predictions.jsonl", "report.json", "receipt.json", "complete.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    marker = json.loads((tmp_path / "one" / "complete.json").read_bytes())
    assert marker["receipt_sha256"] == runner.file_hash(tmp_path / "one" / "receipt.json")
    for name, item in one["outputs"].items():
        assert item["sha256"] == runner.file_hash(tmp_path / "one" / name)
    with pytest.raises(FileExistsError): runner.publish(tmp_path / "one", predictions, report, receipt)


def test_publication_race_and_write_failure_leave_no_partial_artifact(tmp_path, monkeypatch):
    publish = runner._atomic_publish
    def race(stage, out):
        out.mkdir()
        publish(stage, out)
    monkeypatch.setattr(runner, "_atomic_publish", race)
    with pytest.raises(FileExistsError): runner.publish(tmp_path / "race", [], {"headroom_sufficient": False}, {})
    assert list((tmp_path / "race").iterdir()) == []
    assert not list(tmp_path.glob(".race-*"))
    def failed_write(path, payload): raise IOError("injected write failure")
    monkeypatch.setattr(runner, "_write", failed_write)
    with pytest.raises(IOError): runner.publish(tmp_path / "failed", [], {"headroom_sufficient": False}, {})
    assert not (tmp_path / "failed").exists() and not list(tmp_path.glob(".failed-*"))


def test_cpu_mocked_run_binds_assets_code_runtime_and_complete_predictions(tmp_path, canonical_payloads, monkeypatch):
    rows, payload, manifest = canonical_payloads
    data, manifest_path = tmp_path / "headroom.jsonl", tmp_path / "manifest.json"
    data.write_bytes(payload)
    manifest_path.write_bytes(runner._json(manifest))
    model, inventory, adapter, asset_expectations = _assets(tmp_path)
    def fake_infer(rows, model, adapter):
        return _predictions(rows, range(len(rows))), _runtime(), {"1": sum(len(row["questions"]) for row in rows)}
    monkeypatch.setattr(runner, "infer_rows", fake_infer)
    report, receipt = runner.run(data, manifest_path, model, inventory, adapter, tmp_path / "out",
                                expected_data_sha256=runner._sha(payload), expected_manifest_sha256=runner.file_hash(manifest_path),
                                **asset_expectations)
    assert report["headroom_sufficient"] is True
    assert receipt["headroom_data_sha256"] == runner.file_hash(data)
    assert receipt["code_sha256"]["run_r050_headroom.py"] == runner.file_hash(runner.__file__)
    assert receipt["runtime_fingerprint_sha256"] == runner._sha(runner._json(_runtime()))
    assert (tmp_path / "out" / "complete.json").is_file()


@pytest.mark.parametrize("failure", ["inventory", "adapter", "missing_prediction", "runtime", "batch_shape", "changed_data"])
def test_run_invalid_assets_execution_or_predictions_never_publish(tmp_path, canonical_payloads, monkeypatch, failure):
    _, payload, manifest = canonical_payloads
    data, manifest_path = tmp_path / "headroom.jsonl", tmp_path / "manifest.json"
    data.write_bytes(payload)
    manifest_path.write_bytes(runner._json(manifest))
    model, inventory, adapter, expectations = _assets(tmp_path)
    if failure == "inventory": expectations["expected_inventory_sha256"] = "0" * 64
    if failure == "adapter": expectations["expected_adapter_tree_sha256"] = "0" * 64
    def fake_infer(rows, model, adapter):
        predictions, runtime, histogram = _predictions(rows), _runtime(), {"1": sum(len(row["questions"]) for row in rows)}
        if failure == "missing_prediction": predictions.pop()
        if failure == "runtime": runtime["sdpa_profiler_observed"] = False
        if failure == "batch_shape": histogram = {"2": 1}
        if failure == "changed_data": data.write_bytes(payload + b"\n")
        return predictions, runtime, histogram
    monkeypatch.setattr(runner, "infer_rows", fake_infer)
    with pytest.raises(ValueError):
        runner.run(data, manifest_path, model, inventory, adapter, tmp_path / "out",
                   expected_data_sha256=runner._sha(payload), expected_manifest_sha256=runner.file_hash(manifest_path), **expectations)
    assert not (tmp_path / "out").exists()


def test_strict_json_and_cli_external_expected_values_required():
    for payload in (b'{"key":0,"key":1}', b'{"key":NaN}', b'\xff'):
        with pytest.raises(ValueError): runner._parse(payload)
    with pytest.raises(SystemExit) as error:
        runner.main(["--data", "data", "--manifest", "manifest", "--model", "model", "--inventory", "inventory",
                     "--adapter", "adapter", "--out", "out"])
    assert error.value.code == 2


def test_gpu_path_contract_is_cpu_mocked_not_actual_inference(monkeypatch):
    import torch
    import peft
    import transformers
    import scripts.singleton_option_scorer as singleton
    import scripts.train_gemma_decision_lora as trainer
    observed = {}
    rows = _rows()
    class FakeModel:
        config = SimpleNamespace(use_cache=True)
        def parameters(self): return iter([SimpleNamespace(dtype=torch.bfloat16)])
        def to(self, device):
            observed["device"] = str(device)
            return self
        def eval(self):
            observed["eval"] = True
            return self
    model = FakeModel()
    def load_model(path, **kwargs):
        observed["model_load"] = (path, kwargs)
        return model
    def load_parent(base, path, **kwargs):
        observed["adapter_load"] = (base, path, kwargs)
        return base
    def score(base, tokenizer, groups, selected, device, length, observations):
        observed["scoring"] = (base, groups, selected, length, observations.require_sdpa)
        observations.batch_histogram[1] = sum(len(row["questions"]) for row in selected)
        return _predictions(selected)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda value: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda value: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", load_model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", load_parent)
    monkeypatch.setattr(trainer, "letter_token_groups", lambda tokenizer: ["synthetic-groups"])
    monkeypatch.setattr(runner.scorer, "score_rows_scalar", score)
    monkeypatch.setattr(singleton, "collect_runtime_fingerprint", lambda *args: _runtime())
    predictions, _, histogram = runner.infer_rows(rows, "local-model", "local-parent")
    assert observed["model_load"] == ("local-model", {"dtype": torch.bfloat16, "local_files_only": True, "attn_implementation": "sdpa"})
    assert observed["adapter_load"] == (model, "local-parent", {"local_files_only": True, "is_trainable": False})
    assert observed["scoring"] == (model, ["synthetic-groups"], rows, 8192, True)
    assert observed["device"] == "cuda:0" and observed["eval"] is True
    assert model.config.use_cache is False
    assert len(predictions) == len(rows) and histogram == {"1": sum(len(row["questions"]) for row in rows)}


def test_direct_cli_help_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and "--expected-inventory-sha256" in result.stdout
