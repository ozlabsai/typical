import copy
import hashlib
import json

import pytest
import torch

import scripts.probe_r050_gradient_scale as probe


def _signature(index):
    return f"{index:064x}"


def _row(index, *, mode="priority", regime="raw_operands",
         operator="all_checks", language="en", option_order="no_yes",
         split="train"):
    fields = ["field_a", "field_b"]
    return {
        "id": f"row-{index}-{mode}-{language}-{option_order}",
        "questions": {f"q{position:02d}": {} for position in range(len(fields))},
        "meta": {
            "panel": "exact_policy_v1",
            "split": split,
            "semantic_signature": _signature(index),
            "policy": {"fields": fields},
            "policy_mode": mode,
            "value_regime": regime,
            "operator_composition": operator,
            "language": language,
            "option_order": option_order,
        },
    }


def test_stratified_selection_is_deterministic_and_uses_canonical_views():
    rows = []
    for index in range(1, 7):
        mode = "priority" if index <= 3 else "collect"
        rows.extend([
            _row(index, mode=mode, language="he", option_order="yes_no"),
            _row(index, mode=mode, language="en", option_order="no_yes"),
        ])

    selected_a, receipt_a = probe.select_probe_rows(rows, 4, seed=50050)
    selected_b, receipt_b = probe.select_probe_rows(
        list(reversed(copy.deepcopy(rows))), 4, seed=50050)

    assert [row["id"] for row in selected_a] == [row["id"] for row in selected_b]
    assert receipt_a == receipt_b
    assert all(row["meta"]["language"] == "en" for row in selected_a)
    assert all(row["meta"]["option_order"] == "no_yes" for row in selected_a)
    assert sorted(receipt_a["stratum_counts"].values()) == [2, 2]
    assert len({row["meta"]["semantic_signature"] for row in selected_a}) == 4


@pytest.mark.parametrize(
    ("rows", "requested", "message"),
    [
        ([_row(1, split="eval")], 1, "training split"),
        ([_row(1)], 2, "only 1 are available"),
        ([_row(1), _row(1, mode="collect")], 1, "multiple factor cells"),
    ],
)
def test_stratified_selection_fails_closed(rows, requested, message):
    with pytest.raises(ValueError, match=message):
        probe.select_probe_rows(rows, requested, seed=1)


def test_panel_manifest_binds_panel_and_builder(tmp_path):
    panel = tmp_path / "train.jsonl"
    row = _row(1)
    panel.write_text(json.dumps(row) + "\n")
    builder = tmp_path / "builder.py"
    builder.write_text("# frozen\n")
    payload, rows = probe.read_jsonl(panel)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "panel_version": "test",
        "generator_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "outputs": {"train": {
            "sha256": hashlib.sha256(payload).hexdigest(), "rows": 1,
        }},
    }))

    binding = probe.verify_panel_manifest(manifest, payload, rows, builder)

    assert binding["panel_version"] == "test"
    builder.write_text("# changed\n")
    with pytest.raises(ValueError, match="builder differs"):
        probe.verify_panel_manifest(manifest, payload, rows, builder)


def test_gradient_statistics_computes_target_norm_coefficients_and_geometry():
    report = probe.gradient_statistics({
        "field_ce": [torch.tensor([3.0, 4.0])],
        "outcome_nll": [torch.tensor([0.0, 10.0])],
        "outcome_brier": [torch.tensor([5.0, 0.0])],
    })

    assert report["norms"] == {
        "field_ce": 5.0, "outcome_nll": 10.0, "outcome_brier": 5.0,
    }
    assert report["coefficients"]["lambda_nll"] == pytest.approx(0.125)
    assert report["coefficients"]["lambda_brier"] == pytest.approx(0.25)
    assert report["dots"]["field_ce__outcome_nll"] == 40.0
    assert report["cosines"]["field_ce__outcome_nll"] == pytest.approx(0.8)


@pytest.mark.parametrize("bad", [torch.tensor([0.0]), torch.tensor([float("nan")])])
def test_gradient_statistics_rejects_zero_or_nonfinite_norms(bad):
    with pytest.raises(ValueError, match="nonzero|finite"):
        probe.gradient_statistics({
            "field_ce": [torch.tensor([1.0])],
            "outcome_nll": [bad],
            "outcome_brier": [torch.tensor([1.0])],
        })


class TinyProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


def test_gradient_probe_uses_shared_components_without_mutating_parameters(monkeypatch):
    model = TinyProbeModel()
    before = model.theta.detach().clone()

    def fake_loss(current, batch, *_args, **kwargs):
        assert kwargs["return_exact_policy_components"] is True
        assert kwargs["action_loss_weight"] == 0.0
        scale = torch.tensor(float(len(batch))) / len(batch)
        components = {
            "field_ce": scale * (current.theta.square().sum()),
            "outcome_nll": scale * (3.0 * current.theta.sum()),
            "outcome_brier": scale * (2.0 * current.theta[0] - current.theta[1]),
        }
        return components["field_ce"], 0.0, {}, components

    monkeypatch.setattr(probe.trainer, "loss_for_examples", fake_loss)
    report, digest = probe.probe_gradient_norms(
        model, [object(), object(), object()], object(), object(),
        torch.device("cpu"), 100, microbatch=2)

    assert torch.equal(model.theta, before)
    assert model.theta.grad is None
    assert digest == probe.parameter_digest(["theta"], [model.theta])
    assert report["norms"]["field_ce"] == pytest.approx((20.0) ** 0.5)
    assert report["norms"]["outcome_nll"] == pytest.approx((18.0) ** 0.5)
    assert report["norms"]["outcome_brier"] == pytest.approx((5.0) ** 0.5)


def test_gradient_probe_detects_parameter_mutation(monkeypatch):
    model = TinyProbeModel()

    def mutating_loss(current, *_args, **_kwargs):
        with torch.no_grad():
            current.theta.add_(1.0)
        components = {
            "field_ce": current.theta.sum(),
            "outcome_nll": (2.0 * current.theta).sum(),
            "outcome_brier": (3.0 * current.theta).sum(),
        }
        return components["field_ce"], 0.0, {}, components

    monkeypatch.setattr(probe.trainer, "loss_for_examples", mutating_loss)
    with pytest.raises(RuntimeError, match="mutated"):
        probe.probe_gradient_norms(
            model, [object()], object(), object(), torch.device("cpu"), 100)


def test_receipt_is_deterministic_and_contains_no_timestamp():
    kwargs = {
        "selection": {"records": []},
        "gradient_report": {"coefficients": {"lambda_nll": 1.0}},
        "parameter_sha256": "a" * 64,
        "bindings": {"panel": "b" * 64},
        "configuration": {"target_ratio": 0.25},
        "environment": {"torch_version": "test"},
    }

    assert probe.build_receipt(**kwargs) == probe.build_receipt(**copy.deepcopy(kwargs))
    assert "timestamp" not in probe.build_receipt(**kwargs)
