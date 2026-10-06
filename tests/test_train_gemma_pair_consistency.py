import hashlib
import json
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from scripts.train_gemma_decision_lora import (
    accumulate_dense_gradients_,
    backward_into_dense_gradients_,
    pair_consistency_term,
    pair_consistency_stats,
    pair_slot_loss,
    pair_flip_term,
    pair_flip_weight_at_step,
    parse_save_steps,
    encode,
    loss_for_examples,
    load_examples,
    mean_logical_slot_loss,
    guard_logical_slot_loss,
    project_conflicting_gradient,
    projection_early_stop,
    semantic_margin,
    semantic_gold_margin,
    staged_counterfactual_order,
    validate_pair_objectives,
)


def test_explicit_checkpoint_steps_are_strict_and_deduplicated():
    assert parse_save_steps("20,50,100,200", 200) == {20, 50, 100, 200}
    assert parse_save_steps("", 200) == set()
    with pytest.raises(ValueError, match="within"):
        parse_save_steps("0,20", 200)
    with pytest.raises(ValueError, match="integers"):
        parse_save_steps("twenty", 200)


def test_semantic_margin_is_option_order_invariant_and_polarity_signed():
    scores_no_yes = torch.tensor([1.0, 3.0])
    scores_yes_no = torch.tensor([3.0, 1.0])
    assert semantic_margin(scores_no_yes, ["no", "yes"], "positive").item() == 2.0
    assert semantic_margin(scores_yes_no, ["yes", "no"], "positive").item() == 2.0
    assert semantic_margin(scores_no_yes, ["no", "yes"], "negative").item() == -2.0


def counterfactual_pair(mode="counterfactual"):
    source_ids = ["source-a", "source-a" if mode == "counterfactual" else "source-b"]

    def view(identifier, source_id, criteria, target):
        return {
            "id": identifier,
            "state": identifier,
            "questions": {"q": {"type": "noul", "instructions": "Choose.",
                                  "criteria": criteria}},
            "targets": {"q": {"label": target}},
            "meta": {"source_id": source_id, "family": "approval"},
        }

    return {"id": f"pair-{mode}", "pair_mode": mode, "source_ids": source_ids,
            "paired_views": [
                view("left", source_ids[0], {"no": "No", "yes": "Yes"}, "yes"),
                view("right", source_ids[1], {"yes": "Yes", "no": "No"}, "no"),
            ]}


def auxiliary_view(identifier, target="yes", image=None, questions=1):
    row = {
        "id": identifier,
        "state": f"auxiliary state {identifier}",
        "questions": {
            f"q{index}": {"type": "noul", "instructions": "Evaluate the clause.",
                           "criteria": {"no": "No", "yes": "Yes"}}
            for index in range(questions)
        },
        "targets": {f"q{index}": {"label": target} for index in range(questions)},
        "meta": {"source_id": "source-a", "family": "approval"},
    }
    if image is not None:
        row["image"] = image
    return row


def test_counterfactual_loader_distinguishes_true_and_shuffled_pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in (
        counterfactual_pair(), counterfactual_pair("shuffled_counterfactual"))))
    true_pair, shuffled = load_examples([path])
    assert true_pair["pair_mode"] == "counterfactual"
    assert true_pair["pair_source_ids"] == ["source-a", "source-a"]
    assert "auxiliary_views" not in true_pair
    assert shuffled["pair_mode"] == "shuffled_counterfactual"
    assert shuffled["pair_source_ids"] == ["source-a", "source-b"]


def test_paired_loader_adds_text_auxiliary_views_to_the_same_slot(tmp_path):
    row = counterfactual_pair()
    row["auxiliary_views"] = [auxiliary_view("clause-0"), auxiliary_view("clause-1", "no")]
    path = tmp_path / "auxiliary.jsonl"
    path.write_text(json.dumps(row) + "\n")

    pair, = load_examples([path])
    assert [view["id"] for view in pair["paired_views"]] == ["left/q", "right/q"]
    assert [view["id"] for view in pair["auxiliary_views"]] == [
        "clause-0/q0", "clause-1/q0",
    ]
    assert all(view["image"] is None for view in pair["auxiliary_views"])


def test_paired_loader_fails_closed_on_malformed_auxiliary_schema(tmp_path):
    invalid = []

    row = auxiliary_view("singleton")
    row["auxiliary_views"] = []
    invalid.append((row, "only valid on a paired slot"))

    row = counterfactual_pair()
    row["auxiliary_views"] = {"not": "a list"}
    invalid.append((row, "must be a list"))

    row = counterfactual_pair()
    row["paired_views"][0]["auxiliary_views"] = []
    invalid.append((row, "belong on the paired row"))

    row = counterfactual_pair()
    row["auxiliary_views"] = [auxiliary_view("too-many", questions=2)]
    invalid.append((row, "each auxiliary view must flatten to one example"))

    row = counterfactual_pair()
    row["auxiliary_views"] = [auxiliary_view("with-image", image="image.png")]
    invalid.append((row, "auxiliary views must be text-only"))

    row = counterfactual_pair()
    row["auxiliary_views"] = [auxiliary_view("left")]
    row["auxiliary_views"][0]["questions"] = {
        "q": row["auxiliary_views"][0]["questions"].pop("q0")}
    row["auxiliary_views"][0]["targets"] = {"q": {"label": "yes"}}
    invalid.append((row, "view ids must be unique"))

    for index, (row, message) in enumerate(invalid):
        path = tmp_path / f"invalid-aux-{index}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match=message):
            load_examples([path])


def test_counterfactual_loader_fails_closed_on_invalid_pair_metadata(tmp_path):
    invalid = []
    row = counterfactual_pair()
    row["source_ids"][1] = "source-b"
    row["paired_views"][1]["meta"]["source_id"] = "source-b"
    invalid.append((row, "must share a source_id"))
    row = counterfactual_pair("shuffled_counterfactual")
    row["paired_views"][1]["meta"]["family"] = "other"
    invalid.append((row, "one shared family"))
    row = counterfactual_pair("shuffled_counterfactual")
    row["source_ids"][1] = "source-a"
    row["paired_views"][1]["meta"]["source_id"] = "source-a"
    invalid.append((row, "distinct source_ids"))
    row = counterfactual_pair()
    row["paired_views"][1]["targets"]["q"]["label"] = "yes"
    invalid.append((row, "opposite gold labels"))
    row = counterfactual_pair()
    row["paired_views"][1]["questions"]["q"]["criteria"] = {"off": "No", "on": "Yes"}
    row["paired_views"][1]["targets"]["q"]["label"] = "off"
    invalid.append((row, "identical option labels"))
    row = counterfactual_pair()
    row["paired_views"][0]["questions"]["q"]["criteria"]["maybe"] = "Maybe"
    invalid.append((row, "exactly two options"))
    for index, (row, message) in enumerate(invalid):
        path = tmp_path / f"invalid-{index}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match=message):
            load_examples([path])


def test_flip_margin_is_semantic_order_invariant_and_uses_both_gold_labels():
    assert semantic_gold_margin(torch.tensor([1.0, 3.0]), ["no", "yes"], "yes").item() == 2.0
    assert semantic_gold_margin(torch.tensor([3.0, 1.0]), ["yes", "no"], "yes").item() == 2.0
    margin_a = torch.tensor(0.25, requires_grad=True)
    margin_b = torch.tensor(0.50, requires_grad=True)
    flip = pair_flip_term(margin_a, margin_b, margin=1.0)
    combined = pair_slot_loss([torch.tensor(2.0), torch.tensor(4.0)], flip, weight=0.5)
    assert combined.item() == 3.625
    combined.backward()
    assert margin_a.grad.item() == margin_b.grad.item() == -0.5


def test_pair_flip_configuration_rejects_invalid_or_mixed_objectives():
    validate_pair_objectives(0.0, False, 0.0, 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        validate_pair_objectives(0.0, False, -0.1, 1.0)
    with pytest.raises(ValueError, match="positive"):
        validate_pair_objectives(0.0, False, 0.0, 0.0)
    with pytest.raises(ValueError, match="incompatible"):
        validate_pair_objectives(0.1, False, 0.1, 1.0)
    with pytest.raises(ValueError, match="incompatible"):
        validate_pair_objectives(0.0, True, 0.1, 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        validate_pair_objectives(0.0, False, 0.0, 1.0, pair_flip_start_step=-1)
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_pair_objectives(0.0, False, 0.1, 1.0, pair_flip_start_step=11, steps=10)
    validate_pair_objectives(0.0, False, 0.0, 1.0, pair_flip_start_step=11, steps=10)


def test_pair_flip_weight_starts_at_configured_step():
    assert pair_flip_weight_at_step(0.25, start_step=10, step=9) == 0.0
    assert pair_flip_weight_at_step(0.25, start_step=10, step=10) == 0.25


def test_staged_counterfactual_order_is_exact_and_deterministic():
    examples = ([{"id": f"single-{index}"} for index in range(1600)] +
                [{"id": f"pair-{index}", "paired_views": [1, 2],
                  "pair_mode": "counterfactual"} for index in range(1200)])
    random.seed(43)
    first = staged_counterfactual_order(
        examples, steps=350, microbatch=1, grad_accum=8, start_step=201)
    random.seed(43)
    second = staged_counterfactual_order(
        examples, steps=350, microbatch=1, grad_accum=8, start_step=201)
    assert first == second
    assert len(first) == 2800 and len(set(first)) == 2800
    assert all("paired_views" not in examples[index] for index in first[:1600])
    assert all(examples[index].get("pair_mode") for index in first[1600:])


def test_staged_counterfactual_order_rejects_invalid_arithmetic_and_rows():
    singles = [{"id": f"single-{index}"} for index in range(8)]
    pairs = [{"id": f"pair-{index}", "paired_views": [1, 2],
              "pair_mode": "counterfactual"} for index in range(8)]
    with pytest.raises(ValueError, match="got singles=8, pairs=7"):
        staged_counterfactual_order(singles + pairs[:-1], 2, 1, 8, 2)
    with pytest.raises(ValueError, match="only supports"):
        staged_counterfactual_order(singles + pairs[:-1] + [{"paired_views": [1, 2]}],
                                    2, 1, 8, 2)
    with pytest.raises(ValueError, match="valid start step"):
        staged_counterfactual_order(singles + pairs, 2, 1, 8, 0)


def test_consistency_gate_inclusive_threshold_and_anchor_stop_gradient():
    anchor = torch.tensor(8.0, requires_grad=True)
    view_b = torch.tensor(1.0, requires_grad=True)
    term, eligible = pair_consistency_term(anchor, view_b, torch.tensor(0.80), clip=4.0)
    assert eligible
    # Clamped target is 4, so beta=1 smooth-L1 is 2.5.
    assert term.item() == 2.5
    term.backward()
    assert anchor.grad is None
    assert view_b.grad is not None and view_b.grad.item() == -1.0

    rejected, eligible = pair_consistency_term(anchor.detach(), view_b.detach(),
                                                torch.tensor(0.799999))
    assert not eligible and rejected.item() == 0.0
    assert pair_consistency_stats(active=1, seen=2)["pair_consistency_rate"] == 0.5


def test_zero_weight_is_exact_paired_ce_value_and_gradient():
    ce_a = torch.tensor(2.0, requires_grad=True)
    ce_b = torch.tensor(6.0, requires_grad=True)
    consistency = torch.tensor(100.0, requires_grad=True)
    expected = torch.stack([ce_a, ce_b]).mean()
    actual = pair_slot_loss([ce_a, ce_b], consistency, weight=0.0)
    assert torch.equal(actual, expected)
    expected.backward(retain_graph=True)
    expected_grads = (ce_a.grad.clone(), ce_b.grad.clone())
    ce_a.grad = ce_b.grad = None
    actual.backward()
    assert torch.equal(ce_a.grad, expected_grads[0])
    assert torch.equal(ce_b.grad, expected_grads[1])
    assert consistency.grad is None


def test_loader_preserves_pair_order_and_polarity(tmp_path):
    view = lambda arm: {
        "id": arm,
        "state": "state",
        "questions": {"q": {"type": "noul", "criteria": {"no": "No", "yes": "Yes"}}},
        "targets": {"q": {"label": "yes"}},
        "meta": {"s2_arm": arm, "polarity": "negative"},
    }
    path = tmp_path / "paired.jsonl"
    path.write_text(json.dumps({"id": "pair", "paired_views": [view("A"), view("B")]}) + "\n")
    pair, = load_examples([path])
    assert pair["paired_view_arms"] == ["A", "B"]
    assert pair["paired_polarity"] == "negative"


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **_):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        return prompt

    def __call__(self, text, **_):
        token = int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) % 100 + 1
        return {"input_ids": torch.tensor([[token]], dtype=torch.long)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 8))

    def forward(self, input_ids, **_):
        token = input_ids[:, 0].float()
        vocab = torch.arange(8, device=input_ids.device).float()
        base = token[:, None] * vocab[None, :] * 0.001
        logits = base + self.theta[None, :] * (token[:, None] % 13) / 13
        return SimpleNamespace(logits=logits[:, None, :])


def legacy_ce(model, examples, tokenizer, groups, max_length, device, shuffle):
    """Pre-consistency implementation of CE aggregation for a batch of slots."""
    logical_views = [example.get("paired_views", [example]) for example in examples]
    flat_examples = [view for views in logical_views for view in views]
    encoded = [encode(example, tokenizer, max_length, shuffle) for example in flat_examples]
    ids = pad_sequence([item[0] for item in encoded], batch_first=True,
                       padding_value=tokenizer.pad_token_id)
    mask = ids.ne(tokenizer.pad_token_id)
    positions = torch.tensor([len(item[0]) - 1 for item in encoded])
    union = torch.unique(positions, sorted=True)
    output = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False,
                   logits_to_keep=union.to(device))
    offsets = {position.item(): index for index, position in enumerate(union)}
    losses, correct = [], []
    for batch_index, ((_, probabilities), position) in enumerate(zip(encoded, positions)):
        logits = output.logits[batch_index, offsets[position.item()]].float()
        scores = torch.stack([torch.logsumexp(logits[token_ids.to(device)], dim=0)
                              for token_ids in groups[:len(probabilities)]])
        losses.append(-(probabilities.to(device) * F.log_softmax(scores, dim=0)).sum())
        correct.append(float(scores.argmax().item() == probabilities.argmax().item()))
    slot_losses, slot_correct = [], []
    cursor = 0
    for views in logical_views:
        end = cursor + len(views)
        slot_losses.append(losses[cursor:end])
        slot_correct.append(sum(correct[cursor:end]) / len(views))
        cursor = end
    return mean_logical_slot_loss(slot_losses), sum(slot_correct)


def test_counterfactual_zero_weight_matches_ce_and_positive_weight_adds_flip(tmp_path):
    path = tmp_path / "pair.jsonl"
    path.write_text(json.dumps(counterfactual_pair()) + "\n")
    examples = load_examples([path])
    groups = [torch.tensor([1]), torch.tensor([2])]
    model = TinyModel()
    rng = random.Random(307).getstate()

    random.setstate(rng)
    zero, zero_correct, zero_stats = loss_for_examples(
        model, examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True,
        pair_flip_weight=0.0)
    zero.backward()
    zero_grad = model.theta.grad.detach().clone()

    model.theta.grad = None
    random.setstate(rng)
    legacy, legacy_correct = legacy_ce(
        model, examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True)
    legacy.backward()
    assert torch.equal(zero, legacy)
    assert torch.equal(zero_grad, model.theta.grad)
    assert zero_correct == legacy_correct
    assert zero_stats["pair_flip_seen"] == 1

    model.theta.grad = None
    random.setstate(rng)
    weighted, _, weighted_stats = loss_for_examples(
        model, examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True,
        pair_flip_weight=0.25, pair_flip_margin=1.0)
    assert weighted_stats["pair_flip_seen"] == 1
    assert torch.allclose(
        weighted, legacy.detach() + 0.25 * weighted.new_tensor(weighted_stats["pair_flip_loss"]))


def test_auxiliary_ce_is_averaged_in_one_slot_and_flip_uses_only_final_pair(tmp_path):
    row = counterfactual_pair()
    row["auxiliary_views"] = [
        auxiliary_view("clause-0"), auxiliary_view("clause-1", "no"),
        auxiliary_view("clause-2"), auxiliary_view("clause-3", "no"),
    ]
    path = tmp_path / "pair-with-aux.jsonl"
    path.write_text(json.dumps(row) + "\n")
    examples = load_examples([path])
    groups = [torch.tensor([1]), torch.tensor([2])]
    model = TinyModel()

    rng = random.Random(1407).getstate()
    random.setstate(rng)
    zero, zero_correct, zero_stats = loss_for_examples(
        model, examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True,
        pair_flip_weight=0.0)

    expected_examples = [{"id": examples[0]["id"], "paired_views": [
        *examples[0]["paired_views"], *examples[0]["auxiliary_views"],
    ]}]
    random.setstate(rng)
    expected, expected_correct = legacy_ce(
        model, expected_examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True)
    assert torch.equal(zero, expected)
    assert zero_correct == expected_correct
    assert zero_stats["canonical_views_seen"] == 2
    assert zero_stats["auxiliary_views_seen"] == 4
    assert (zero_stats["canonical_views_correct"] +
            zero_stats["auxiliary_views_correct"]) >= 0
    assert zero.item() == pytest.approx(
        (zero_stats["canonical_ce_sum"] + zero_stats["auxiliary_ce_sum"]) / 6)
    assert zero_stats["canonical_ce"] == pytest.approx(
        zero_stats["canonical_ce_sum"] / 2)
    assert zero_stats["auxiliary_ce"] == pytest.approx(
        zero_stats["auxiliary_ce_sum"] / 4)

    random.setstate(rng)
    weighted, _, weighted_stats = loss_for_examples(
        model, examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True,
        pair_flip_weight=0.25, pair_flip_margin=1.0)
    assert weighted_stats["pair_flip_seen"] == 1
    assert torch.allclose(
        weighted, expected.detach() + 0.25 * weighted.new_tensor(weighted_stats["pair_flip_loss"]))

    # Auxiliary correctness can change the CE/statistics, but never creates extra flip endpoints.
    changed = counterfactual_pair()
    changed["auxiliary_views"] = [
        auxiliary_view("clause-0", "no"), auxiliary_view("clause-1", "yes"),
        auxiliary_view("clause-2", "no"), auxiliary_view("clause-3", "yes"),
    ]
    changed_path = tmp_path / "changed-aux.jsonl"
    changed_path.write_text(json.dumps(changed) + "\n")
    random.setstate(rng)
    _, _, changed_stats = loss_for_examples(
        model, load_examples([changed_path]), TinyTokenizer(), groups, 100,
        torch.device("cpu"), shuffle=True, pair_flip_weight=0.25, pair_flip_margin=1.0)
    assert changed_stats["pair_flip_seen"] == 1
    assert changed_stats["pair_flip_loss"] == weighted_stats["pair_flip_loss"]


def test_zero_weight_matches_legacy_loss_gradients_rng_and_option_order():
    def view(identifier, arm, criteria, target):
        return {
            "id": identifier,
            "state": identifier,
            "questions": {"q": {"type": "noul", "instructions": "Choose.",
                                  "criteria": criteria}},
            "targets": {"q": {"label": target}},
            "meta": {"s2_arm": arm, "polarity": "positive"},
        }

    examples = [{"id": "slot", "paired_views": [
        view("anchor", "A", {"no": "No", "yes": "Yes"}, "yes"),
        view("alternate", "B", {"yes": "Yes", "no": "No"}, "yes"),
    ]}]
    examples = load_examples_from_rows(examples)
    groups = [torch.tensor([1]), torch.tensor([2])]
    model = TinyModel()
    tokenizer = TinyTokenizer()
    random.seed(741)
    rng_before = random.getstate()
    actual, actual_correct, _ = loss_for_examples(
        model, examples, tokenizer, groups, 100, torch.device("cpu"), shuffle=True,
        pair_consistency_weight=0.0)
    rng_after_actual = random.getstate()
    actual_prompts = tokenizer.prompts[:]
    actual.backward()
    actual_grad = model.theta.grad.detach().clone()

    model.theta.grad = None
    tokenizer = TinyTokenizer()
    random.setstate(rng_before)
    expected, expected_correct = legacy_ce(
        model, examples, tokenizer, groups, 100, torch.device("cpu"), shuffle=True)
    rng_after_legacy = random.getstate()
    expected.backward()
    expected_grad = model.theta.grad.detach().clone()

    assert torch.equal(actual, expected)
    assert actual_correct == expected_correct
    assert torch.equal(actual_grad, expected_grad)
    assert rng_after_actual == rng_after_legacy
    assert actual_prompts == tokenizer.prompts

    model.theta.grad = None
    random.seed(917)
    rng_before = random.getstate()
    combined, _, _ = loss_for_examples(
        model, examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True,
        pair_consistency_weight=0.25, pair_consistency_min_gold_probability=0.0)
    combined.backward()
    combined_grad = model.theta.grad.detach().clone()

    model.theta.grad = None
    random.setstate(rng_before)
    _, _, _, components = loss_for_examples(
        model, examples, TinyTokenizer(), groups, 100, torch.device("cpu"), shuffle=True,
        pair_consistency_weight=0.25, pair_consistency_min_gold_probability=0.0,
        return_components=True)
    decomposed = components["d_loss"] + 0.25 * components["consistency_loss"]
    decomposed.backward()
    assert torch.equal(combined, decomposed)
    assert torch.equal(combined_grad, model.theta.grad)


def test_global_projection_removes_only_conflicting_component():
    projected, stats = project_conflicting_gradient(
        [torch.tensor([1.0, 2.0])], [torch.tensor([-1.0, 0.0])])
    assert torch.equal(projected[0], torch.tensor([0.0, 2.0]))
    assert stats["projection_conflict"] is True
    assert stats["projection_guard_cons_dot"] == -1.0
    assert stats["projection_removed_energy"] == 1.0
    assert abs(stats["projection_removed_energy_fraction"] - 0.2) < 1e-7

    unchanged, stats = project_conflicting_gradient(
        [torch.tensor([1.0, 2.0])], [torch.tensor([1.0, 0.0])])
    assert torch.equal(unchanged[0], torch.tensor([1.0, 2.0]))
    assert stats["projection_conflict"] is False
    assert stats["projection_removed_energy"] == 0.0

    # The coefficient uses one dot/norm across both parameter tensors.
    projected, stats = project_conflicting_gradient(
        [torch.tensor([1.0]), torch.tensor([1.0])],
        [torch.tensor([-2.0]), torch.tensor([1.0])])
    assert torch.allclose(projected[0], torch.tensor([0.6]))
    assert torch.allclose(projected[1], torch.tensor([1.2]))
    assert stats["projection_guard_cons_dot"] == -1.0
    projected_dot = (projected[0] * torch.tensor([-2.0])).sum() + (
        projected[1] * torch.tensor([1.0])).sum()
    assert abs(projected_dot.item()) < 1e-6


def test_guard_objective_uses_half_anchor_and_stop_falsifier_boundary():
    singleton = torch.tensor(2.0)
    paired_a, paired_b = torch.tensor(4.0), torch.tensor(100.0)
    # Mean of singleton CE and half-weighted paired A; B is omitted.
    assert guard_logical_slot_loss([[singleton], [paired_a, paired_b]]).item() == 2.0
    assert projection_early_stop(50, 0.009, 1.0)
    assert not projection_early_stop(50, 0.01, 1.0)
    assert not projection_early_stop(49, 0.0, 1.0)


def test_projection_rejects_unused_trainable_parameters_instead_of_weight_decaying_them():
    buffers = [torch.zeros(1), torch.zeros(1)]
    try:
        accumulate_dense_gradients_(buffers, [torch.ones(1), None], "test")
    except RuntimeError as error:
        assert "did not reach 1 trainable parameters" in str(error)
    else:
        raise AssertionError("unused trainable parameter must fail closed")


def test_backward_extraction_matches_autograd_grad_and_clears_parameter_grads():
    theta = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    shared = (theta.square().sum() + theta.prod()).tanh()
    losses = (shared * 2, shared * -3, shared.square())
    expected = [torch.autograd.grad(loss, theta, retain_graph=True)[0] for loss in losses]

    theta = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    shared = (theta.square().sum() + theta.prod()).tanh()
    losses = (shared * 2, shared * -3, shared.square())
    actual = [torch.zeros_like(theta) for _ in losses]
    for index, (loss, buffer) in enumerate(zip(losses, actual)):
        backward_into_dense_gradients_(
            loss, [theta], [buffer], f"loss {index}", retain_graph=index < len(losses) - 1)
        assert theta.grad is None

    for actual_grad, expected_grad in zip(actual, expected):
        assert torch.equal(actual_grad, expected_grad)


def load_examples_from_rows(rows):
    """Round-trip synthetic rows through the same flattening path without disk I/O."""
    from scripts.train_gemma_decision_lora import flatten_row

    loaded = []
    for row in rows:
        views = [flatten_row(view) for view in row["paired_views"]]
        loaded.append({"id": row["id"], "paired_views": [view[0] for view in views],
                       "paired_view_arms": ["A", "B"], "paired_polarity": "positive"})
    return loaded
