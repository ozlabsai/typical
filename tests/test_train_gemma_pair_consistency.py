import hashlib
import json
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from scripts.train_gemma_decision_lora import (
    pair_consistency_term,
    pair_consistency_stats,
    pair_slot_loss,
    encode,
    loss_for_examples,
    load_examples,
    mean_logical_slot_loss,
    semantic_margin,
)


def test_semantic_margin_is_option_order_invariant_and_polarity_signed():
    scores_no_yes = torch.tensor([1.0, 3.0])
    scores_yes_no = torch.tensor([3.0, 1.0])
    assert semantic_margin(scores_no_yes, ["no", "yes"], "positive").item() == 2.0
    assert semantic_margin(scores_yes_no, ["yes", "no"], "positive").item() == 2.0
    assert semantic_margin(scores_no_yes, ["no", "yes"], "negative").item() == -2.0


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


def load_examples_from_rows(rows):
    """Round-trip synthetic rows through the same flattening path without disk I/O."""
    from scripts.train_gemma_decision_lora import flatten_row

    loaded = []
    for row in rows:
        views = [flatten_row(view) for view in row["paired_views"]]
        loaded.append({"id": row["id"], "paired_views": [view[0] for view in views],
                       "paired_view_arms": ["A", "B"], "paired_polarity": "positive"})
    return loaded
