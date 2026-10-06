import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.build_r045_compositional_transfer import _pair
from scripts.eval_r047_executor import (
    audit,
    audit_causal_scalar_repeatability,
    causal_parity_subset,
    causal_coverage_report,
    clause_view,
    execute_clause,
    execute_program,
    m0_gate,
    row_clause_views,
    score_causal_rows_scalar,
    seal_causal_predictions,
    validate_sealed_causal_predictions,
    validate_causal_panel_payload,
    validate_causal_model_inventory,
    validate_causal_run_receipt,
    validate_causal_supporting_provenance,
    validate_loaded_adapter_tree,
    validate_loaded_candidate_adapter,
    validate_loaded_reference_adapter,
    validate_reference_provenance,
)
from scripts.build_r047_panel import build as build_r047_panel


KINDS = ("decimal", "ledger")


def _rows():
    return (_pair("test", KINDS, 80, 47, domain_offset=2)
            + _pair("test", KINDS, 81, 47, domain_offset=2))


def _predictions(rows, flip=()):
    result = []
    for row in rows:
        projected_by_parent = row_clause_views(row)
        for view in row["paired_views"]:
            projected = [view] + projected_by_parent[view["id"]]
            for item in projected:
                gold = item["targets"]["decision"]["label"]
                if item["id"] in flip:
                    gold = "yes" if gold == "no" else "no"
                result.append({"id": item["id"], "probabilities":
                               {"no": .9 if gold == "no" else .1,
                                "yes": .9 if gold == "yes" else .1}})
    return result


def test_audit_reports_executable_clause_vector_coherence_and_direction():
    rows = _rows()
    report = audit(rows, _predictions(rows), min_clauses=2)
    assert report["rows"] == 4
    assert report["final"]["accuracy"] == 1
    assert report["final_pairs"]["direction_rate"] == 1
    assert report["clauses"]["n"] == 16
    assert report["clauses"]["balanced_accuracy"] == 1
    assert report["clauses"]["vector_accuracy"] == 1
    assert report["coherence"]["accuracy"] == 1
    assert set(report["clauses"]["kind"]) == set(KINDS)
    assert set(report["clauses"]["language"]) == {"en", "he"}


def test_independent_executor_reproduces_all_six_clause_kinds():
    row = _pair("test", ("decimal", "datetime", "policy"), 80, 47, domain_offset=2)[0]
    for view in row["paired_views"]:
        assert execute_program(view["meta"]["program"], view["meta"]["clauses"]) is view["meta"]["gold"]
        assert all(isinstance(execute_clause(clause), bool) for clause in view["meta"]["clauses"])
    row = _pair("test", ("ledger", "fraction", "conditional"), 96, 47, domain_offset=2)[0]
    for view in row["paired_views"]:
        assert execute_program(view["meta"]["program"], view["meta"]["clauses"]) is view["meta"]["gold"]


def test_clause_error_lowers_vector_balanced_accuracy_and_coherence():
    rows = _rows()
    bad = clause_view(rows[0]["paired_views"][0], 0)["id"]
    report = audit(rows, _predictions(rows, {bad}), min_clauses=2)
    assert report["clauses"]["accuracy"] == 15 / 16
    assert report["clauses"]["vector_accuracy"] == 7 / 8
    assert report["coherence"]["accuracy"] == 7 / 8


def test_audit_scores_exact_training_auxiliary_views_when_present():
    rows = _rows()
    for row in rows:
        row["auxiliary_views"] = []
        for view in row["paired_views"]:
            for index, projected in enumerate(
                    [clause_view(view, i) for i in range(len(view["meta"]["clauses"]))]):
                projected["id"] = f"{view['id']}::training-aux-{index}"
                projected["meta"]["r047_auxiliary"] = {
                    "parent_view_id": view["id"], "clause_index": index,
                    "clause_kind": view["meta"]["clauses"][index]["kind"],
                    "role": "auxiliary"}
                row["auxiliary_views"].append(projected)
    assert all("training-aux" in item["id"] for row in rows
               for items in row_clause_views(row).values() for item in items)
    report = audit(rows, _predictions(rows), min_clauses=2)
    assert report["clauses"]["balanced_accuracy"] == 1


def test_m0_gate_checks_every_required_operator():
    report = audit(_rows(), _predictions(_rows()), min_clauses=2)
    assert m0_gate(report, KINDS)["passed"]
    broken = copy.deepcopy(report)
    broken["final_pairs"]["decisive_kind"]["ledger"]["direction_rate"] = .5
    gate = m0_gate(broken, KINDS)
    assert not gate["passed"] and "ledger final direction" in gate["failures"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows, predictions: predictions.pop(), "cover every final and clause"),
    (lambda rows, predictions: predictions[0]["probabilities"].update(yes=.8), "sum to one"),
    (lambda rows, predictions: rows[0]["paired_views"][0]["targets"]["decision"].update(
        label="maybe"), "executable gold"),
])
def test_audit_fails_closed(mutation, match):
    rows = _rows()
    predictions = _predictions(rows)
    mutation(rows, predictions)
    with pytest.raises(ValueError, match=match):
        audit(rows, predictions, min_clauses=2)


def _causal_row():
    view = copy.deepcopy(_rows()[0]["paired_views"][0])
    meta = view["meta"]
    by_kind = {clause["kind"]: clause for clause in meta["clauses"]}
    meta["clauses"] = [by_kind[kind] for kind in meta["composition"]]
    pattern = [execute_clause(by_kind[kind]) for kind in meta["composition"]]
    meta.update({
        "semantic_cluster_id": "cluster-0", "semantic_view_id": "view-0",
        "arity": len(meta["composition"]),
        "option_order": ("yes_no" if next(iter(view["questions"]["decision"]["criteria"]))
                         == "yes" else "no_yes"),
        "prompt_family": "final", "prediction_role": "final", "primitive": None,
        "decisive_role": "final", "truth_hamming_weight": sum(pattern),
        "truth_pattern": pattern, "seen_edge_count": 0,
    })
    return view


def test_causal_seal_binds_metadata_and_detects_tampering():
    row = _causal_row()
    predictions = [{"id": row["id"], "probabilities": {"no": .1, "yes": .9}}]
    sealed = seal_causal_predictions([row], predictions)
    assert sealed[0]["semantic_cluster_id"] == "cluster-0"
    assert causal_coverage_report(sealed)["dimensions"]["arity"] == {"2": 1}
    validate_sealed_causal_predictions(sealed)
    sealed[0]["language"] = "he"
    with pytest.raises(ValueError, match="seal mismatch"):
        validate_sealed_causal_predictions(sealed)


def test_causal_seal_rejects_missing_semantic_binding():
    row = _causal_row()
    del row["meta"]["seen_edge_count"]
    with pytest.raises(ValueError, match="missing seen_edge_count"):
        seal_causal_predictions(
            [row], [{"id": row["id"], "probabilities": {"no": .1, "yes": .9}}])


@pytest.mark.parametrize("shape", [(2, 3), (0, 3), (1, 0), (3,)])
def test_shared_causal_scorer_rejects_non_singleton_inputs(shape):
    from scripts.singleton_option_scorer import singleton_option_probabilities

    with pytest.raises(ValueError, match="B=1"):
        singleton_option_probabilities(
            object(), torch.zeros(shape, dtype=torch.long), [torch.tensor([0])], ["no"])


def test_causal_production_scorer_uses_singleton_forwards_and_is_order_invariant(monkeypatch):
    import scripts.train_gemma_decision_lora as trainer
    from scripts.singleton_option_scorer import ForwardObservations

    rows = [{"id": "row-a", "token": 1, "length": 2},
            {"id": "row-b", "token": 2, "length": 3},
            {"id": "row-c", "token": 1, "length": 2}]

    def fake_encode(row, tokenizer, max_length, shuffle, include_labels):
        ids = torch.tensor([row["token"]] + [3] * (row["length"] - 1))
        return ids, torch.tensor([0.0, 1.0]), ["no", "yes"]

    monkeypatch.setattr(trainer, "flatten_row", lambda row: [row])
    monkeypatch.setattr(trainer, "encode", fake_encode)

    class Model:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, input_ids, use_cache, logits_to_keep):
            self.batch_sizes.append(input_ids.shape[0])
            assert input_ids.shape[0] == 1
            assert use_cache is False and logits_to_keep == 1
            logits = torch.zeros((1, 1, 4))
            token = int(input_ids[0, 0])
            logits[0, 0, 1 if token == 1 else 0] = 2
            # A batch-size-sensitive model would alter the logits outside singleton calls.
            logits[0, 0, 2] = float(input_ids.shape[0] - 1)
            return SimpleNamespace(logits=logits)

    class Tokenizer:
        pad_token_id = 0

    groups = [torch.tensor([0]), torch.tensor([1])]
    model = Model()
    observations = ForwardObservations()
    all_rows = score_causal_rows_scalar(
        model, Tokenizer(), groups, rows, torch.device("cpu"), 10, observations)
    left = score_causal_rows_scalar(
        model, Tokenizer(), groups, rows[:1], torch.device("cpu"), 10)
    right = score_causal_rows_scalar(
        model, Tokenizer(), groups, rows[1:], torch.device("cpu"), 10)
    reversed_rows = score_causal_rows_scalar(
        model, Tokenizer(), groups, list(reversed(rows)), torch.device("cpu"), 10)
    assert all_rows == left + right
    assert {item["id"]: item for item in all_rows} == {
        item["id"]: item for item in reversed_rows}
    repeatability = audit_causal_scalar_repeatability(rows, all_rows, reversed_rows)
    assert repeatability["status"] == "pass" and repeatability["max_probability_delta"] == 0.0
    with pytest.raises(ValueError, match="canonical and reverse order"):
        audit_causal_scalar_repeatability(rows, all_rows, all_rows)
    assert model.batch_sizes == [1] * 9
    assert observations.report() == {"1": 3}
    assert observations.logits_dtypes == {"float32"}


def test_profiler_canary_fails_when_sdpa_dispatch_is_absent(monkeypatch):
    from contextlib import nullcontext
    from scripts.singleton_option_scorer import ForwardObservations, singleton_option_probabilities

    profiler = SimpleNamespace(key_averages=lambda: [SimpleNamespace(key="aten::matmul")])
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: nullcontext(profiler))
    ids = SimpleNamespace(ndim=2, shape=(1, 3), device=SimpleNamespace(type="cuda"))
    observations = ForwardObservations(require_sdpa=True)
    with pytest.raises(ValueError, match="did not observe scaled_dot_product_attention"):
        singleton_option_probabilities(
            lambda **kwargs: SimpleNamespace(logits=torch.zeros(1, 1, 2)), ids,
            [torch.tensor([0]), torch.tensor([1])], ["no", "yes"], observations)
    assert observations.report() == {"1": 1}
    assert not observations.sdpa_observed


def test_runtime_fingerprint_collects_real_parameter_dtypes_without_gpu(monkeypatch):
    import scripts.singleton_option_scorer as scorer

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
            self.lora_A = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
            self.config = SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(_attn_implementation="sdpa"))

    monkeypatch.setattr(scorer, "version", lambda package: "1.2.3")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91002)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(scorer.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="560.1\n"))
    observations = scorer.ForwardObservations(require_sdpa=True)
    observations.sdpa_observed = True
    observations.logits_dtypes = {"bfloat16"}
    result = scorer.collect_runtime_fingerprint(Model(), observations, "cuda")
    assert result["base_parameter_dtype_counts"] == {"bfloat16": 6}
    assert result["adapter_parameter_dtype_counts"] == {"float32": 4}
    assert result["driver_version"] == "560.1"
    assert result["gpu_capability"] == [9, 0]


def test_causal_panel_manifest_binding_and_frozen_contract(tmp_path):
    panel = tmp_path / "panel"
    build_r047_panel(panel, counts={"unseen_k2": 1, "k3": 1, "seen_k2": 1})
    payload = (panel / "primary.jsonl").read_bytes()
    manifest = (panel / "manifest.json").read_bytes()
    rows, _ = validate_causal_panel_payload(payload, manifest, require_frozen=False)
    assert len(rows) > 0
    subset = causal_parity_subset(rows)
    assert len(subset) == len({
        (row["meta"]["arity"], row["meta"]["language"], row["meta"]["option_order"],
         row["meta"]["prediction_role"], row["meta"]["prompt_family"],
         row["meta"]["primitive"]) for row in rows})
    with pytest.raises(ValueError, match="exact frozen production contract"):
        validate_causal_panel_payload(payload, manifest, require_frozen=True)
    changed = json.loads(manifest)
    changed["outputs"]["primary"]["rows"] -= 1
    with pytest.raises(ValueError, match="disagrees with manifest"):
        validate_causal_panel_payload(
            payload, json.dumps(changed).encode(), require_frozen=False)


def test_run_receipt_and_scalar_repeatability_contracts():
    sha = hashlib.sha256(b"x").hexdigest()
    receipt = {"schema_version": "r047-causal-run-receipt-v1", "slot": "slot_a",
               "status": "complete", "checkpoint_step": 400, "train_sha256": sha,
               "recipe_sha256": sha, "parent_adapter_sha256": sha,
               "model_revision": "revision", "model_inventory_sha256": sha,
               "adapter_sha256": sha, "checkpoint_payload_sha256": sha,
               "config_sha256": sha, "metrics_sha256": sha, "log_sha256": sha}
    assert validate_causal_run_receipt(json.dumps(receipt).encode(), sha)["slot"] == "slot_a"
    receipt["slot"] = "reference"
    with pytest.raises(ValueError, match="schema/status"):
        validate_causal_run_receipt(json.dumps(receipt).encode(), sha)
    receipt["slot"] = "slot_a"
    receipt["checkpoint_payload_sha256"] = "bad"
    with pytest.raises(ValueError, match="invalid SHA-256"):
        validate_causal_run_receipt(json.dumps(receipt).encode(), sha)

    rows = [{"id": "a"}, {"id": "b"}]
    same = [{"id": "a", "probabilities": {"no": .1, "yes": .9}},
            {"id": "b", "probabilities": {"no": .8, "yes": .2}}]
    report = audit_causal_scalar_repeatability(rows, same, list(reversed(copy.deepcopy(same))))
    assert report["exact_discrete_match"] and report["max_probability_delta"] == 0
    assert report["execution_mode"] == "scalar"
    assert report["effective_batch_rows"] == 1
    assert report["base_weight_dtype"] == "bfloat16"
    assert report["attention_interface"] == "sdpa" and report["max_length"] == 8192
    changed = copy.deepcopy(same)
    changed[0]["probabilities"] = {"no": .10002, "yes": .89998}
    with pytest.raises(ValueError, match="probability delta"):
        audit_causal_scalar_repeatability(rows, same, list(reversed(changed)))
    changed = copy.deepcopy(same)
    changed[0]["probabilities"] = {"no": .9, "yes": .1}
    with pytest.raises(ValueError, match="discrete prediction differs"):
        audit_causal_scalar_repeatability(rows, same, list(reversed(changed)))


def test_causal_cli_rejects_multirow_batches():
    from scripts.eval_r047_executor import main

    with pytest.raises(SystemExit) as error:
        main(["--model", "model", "--model-revision", "revision",
              "--data", "panel.jsonl", "--out", "out", "--causal-standalone",
              "--max-batch-rows", "2"])
    assert error.value.code == 2


def test_causal_cli_rejects_nonfrozen_max_length():
    from scripts.eval_r047_executor import main

    with pytest.raises(SystemExit) as error:
        main(["--model", "model", "--model-revision", "revision",
              "--data", "panel.jsonl", "--out", "out", "--causal-standalone",
              "--max-length", "4096"])
    assert error.value.code == 2


def test_model_inventory_verifies_actual_tree_and_revision(tmp_path):
    from scripts.preflight_r043_assets import tree_hash
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "weights.bin").write_bytes(b"weights")
    digest, files = tree_hash(model)
    inventory = {"schema_version": "r047-model-inventory-v1",
                 "model_id": "google/gemma-4-12B-it", "revision": "revision",
                 "tree_sha256": digest, "files": files}
    payload = json.dumps(inventory).encode()
    assert validate_causal_model_inventory(model, payload, "revision") == inventory
    (model / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from sealed inventory"):
        validate_causal_model_inventory(model, payload, "revision")
    with pytest.raises(ValueError, match="schema/revision"):
        validate_causal_model_inventory(model, payload, "other")


def test_loaded_adapter_tree_and_reference_provenance_are_exact(tmp_path):
    from scripts.preflight_r043_assets import file_hash, tree_hash
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    weights.write_bytes(b"weights")
    config.write_text("{}\n")
    tree_sha, files = tree_hash(adapter)
    source = b"archived checkpoint provenance\n"
    inventory_sha = hashlib.sha256(b"inventory").hexdigest()
    provenance = {
        "schema_version": "r047-reference-provenance-v1",
        "adapter_weights_sha256": file_hash(weights),
        "adapter_config_sha256": file_hash(config),
        "adapter_tree_sha256": tree_sha, "adapter_tree_files": files,
        "model_revision": "revision", "model_inventory_sha256": inventory_sha,
        "source_checkpoint_provenance_sha256": hashlib.sha256(source).hexdigest(),
    }
    payload = json.dumps(provenance).encode()
    assert validate_reference_provenance(payload) == provenance
    assert validate_loaded_adapter_tree(adapter, tree_sha, files) == (tree_sha, files)
    verified = validate_loaded_reference_adapter(
        adapter, payload, source, "revision", inventory_sha)
    assert verified[1:3] == (file_hash(weights), file_hash(config))
    with pytest.raises(ValueError, match="source-checkpoint"):
        validate_loaded_reference_adapter(
            adapter, payload, b"tampered", "revision", inventory_sha)
    config.write_text('{"tampered": true}\n')
    with pytest.raises(ValueError, match="tree differs"):
        validate_loaded_adapter_tree(adapter, tree_sha, files)
    provenance["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        validate_reference_provenance(json.dumps(provenance).encode())


def test_candidate_adapter_verifies_complete_checkpoint_tree(tmp_path):
    from scripts.preflight_r043_assets import file_hash, tree_hash
    adapter = tmp_path / "step-000400"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}\n")
    (adapter / "trainer.pt").write_bytes(b"trainer")
    tree_sha, _ = tree_hash(adapter)
    inventory_sha = hashlib.sha256(b"inventory").hexdigest()
    filler = hashlib.sha256(b"filler").hexdigest()
    receipt = {
        "schema_version": "r047-causal-run-receipt-v1", "slot": "slot_a",
        "status": "complete", "checkpoint_step": 400,
        "train_sha256": filler, "recipe_sha256": filler,
        "parent_adapter_sha256": filler, "model_revision": "revision",
        "model_inventory_sha256": inventory_sha,
        "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
        "checkpoint_payload_sha256": tree_sha, "config_sha256": filler,
        "metrics_sha256": filler, "log_sha256": filler,
    }
    payload = json.dumps(receipt).encode()
    assert validate_loaded_candidate_adapter(
        adapter, payload, "revision", inventory_sha)[3] == tree_sha
    (adapter / "trainer.pt").write_bytes(b"tampered trainer")
    with pytest.raises(ValueError, match="tree differs"):
        validate_loaded_candidate_adapter(adapter, payload, "revision", inventory_sha)


def test_panel_preflight_and_inventory_payloads_are_bound():
    panel = b'{"id":"row"}\n'
    manifest = b'{"manifest":1}\n'
    inventory = json.dumps({
        "version": 1,
        "training": [{"path": "train", "sha256": hashlib.sha256(b"t").hexdigest()}],
        "benchmarks": [{"path": "bench", "sha256": hashlib.sha256(b"b").hexdigest()}],
    }).encode()
    preflight = json.dumps({
        "experiment": "r047_primary_panel_preflight", "status": "pass",
        "panel_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "inventory_manifest_sha256": hashlib.sha256(inventory).hexdigest(),
        "outputs": {"primary": {"sha256": hashlib.sha256(panel).hexdigest(), "rows": 1}},
        "causal": {"primary": {"rows": 1}},
    }).encode()
    validate_causal_supporting_provenance(panel, manifest, preflight, inventory)
    changed = json.loads(preflight)
    changed["status"] = "fail"
    with pytest.raises(ValueError, match="does not bind"):
        validate_causal_supporting_provenance(
            panel, manifest, json.dumps(changed).encode(), inventory)
