import contextlib
import json
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import scripts.train_gemma_decision_lora as train
from tests.test_train_gemma_multimodal import _Model, _Processor
from tests.test_train_gemma_pair_consistency import TinyModel, TinyTokenizer

CPU = torch.device("cpu")
GROUPS = [torch.tensor([1]), torch.tensor([2])]


class AdapterModel(TinyModel):
    """TinyModel whose theta plays the LoRA: disable_adapter() zeroes it."""

    def __init__(self):
        super().__init__()
        self.off = False
        self.disabled_calls = 0

    @contextlib.contextmanager
    def disable_adapter(self):
        self.off, self.disabled_calls = True, self.disabled_calls + 1
        try:
            yield
        finally:
            self.off = False

    def forward(self, input_ids, **kwargs):
        if not self.off:
            return super().forward(input_ids, **kwargs)
        token = input_ids[:, 0].float()
        logits = token[:, None] * torch.arange(8).float()[None, :] * 0.001
        return SimpleNamespace(logits=logits[:, None, :])


def _rows(tmp_path, meta, name="rows.jsonl"):
    rows = [{"id": f"r{i}", "state": f"state {i}",
             "questions": {"q": {"type": "noul", "instructions": "choose",
                                 "criteria": {"no": "No", "yes": "Yes"}}},
             "targets": {"q": {"label": "yes"}}, "meta": meta} for i in range(3)]
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return train.load_examples([path])


def _loss(model, examples, **kwargs):
    random.seed(7)
    return train.loss_for_examples(model, examples, TinyTokenizer(), GROUPS, 100, CPU,
                                   shuffle=True, **kwargs)


def test_kl_is_forward_base_to_student():
    student = torch.tensor([2.0, -1.0, 0.0])
    base = torch.tensor([0.2, 0.5, 0.3])
    q = F.softmax(student, 0)
    expected = (base * (base / q).log()).sum()
    reverse = (q * (q / base).log()).sum()
    got = train.kl_to_base_term(student, base)
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)
    assert got.item() != pytest.approx(reverse.item(), rel=1e-3)
    assert train.kl_to_base_term(base.log(), base).item() == pytest.approx(0.0, abs=1e-7)
    # Zero base mass is fine (xlogy) and unnormalized cache values are renormalized.
    assert torch.isfinite(train.kl_to_base_term(student, torch.tensor([0.0, 2.0, 2.0])))


def test_flag_off_is_identical_and_never_runs_the_base(tmp_path):
    examples = _rows(tmp_path, {"anchor": True})
    plain = _rows(tmp_path, {}, "plain.jsonl")  # same rows without meta
    model = AdapterModel()
    loss_off, correct_off, stats_off = _loss(model, examples)
    loss_plain, correct_plain, stats_plain = _loss(model, plain)
    assert torch.equal(loss_off, loss_plain) and correct_off == correct_plain
    assert stats_off == stats_plain and "kl_base_seen" not in stats_off
    assert model.disabled_calls == 0


def test_live_base_forward_adds_weighted_kl_on_anchored_rows_only(tmp_path):
    anchored = _rows(tmp_path, {"anchor": True})
    model = AdapterModel()
    ce, _, _ = _loss(model, anchored[:2])
    batch = [anchored[0], {**anchored[1], "meta": {}}]
    loss, _, stats = _loss(model, batch, kl_to_base_weight=0.5)
    assert model.disabled_calls == 1 and stats["kl_base_seen"] == 1
    # Slot mean of (ce0 + 0.5 kl0, ce1): only row 0 is anchored.
    assert loss.item() == pytest.approx(ce.item() + 0.5 * stats["kl_base_mean"] / 2, rel=1e-6)
    # Canonical CE telemetry stays pure CE.
    assert stats["canonical_ce"] == pytest.approx(ce.item(), rel=1e-6)
    loss.backward()  # base forward was no-grad; the student path still trains
    assert model.theta.grad.abs().sum() > 0


def test_cached_base_probs_skip_the_forward_and_kl_only_drops_ce(tmp_path):
    cached = _rows(tmp_path, {"anchor": True, "base_probs": {"q": {"no": 0.3, "yes": 0.7}}})
    assert cached[0]["base_probs"] == {"no": 0.3, "yes": 0.7}
    model = AdapterModel()
    loss, _, stats = _loss(model, cached[:1], kl_to_base_weight=2.0, kl_to_base_only=True)
    assert model.disabled_calls == 0 and stats["kl_base_seen"] == 1
    assert loss.item() == pytest.approx(2.0 * stats["kl_base_mean"], rel=1e-6)
    # Aligned by option label, not letter position: recompute against the rendered order.
    random.seed(7)
    ids, _, labels = train.encode(cached[0], TinyTokenizer(), 100, True, include_labels=True)
    logits = model(ids[None]).logits[0, 0].detach()
    scores = train.option_scores_from_logits(logits, GROUPS, 2, CPU)
    base = torch.tensor([{"no": 0.3, "yes": 0.7}[label] for label in labels])
    assert stats["kl_base_mean"] == pytest.approx(train.kl_to_base_term(scores, base).item(), rel=1e-6)


def test_anchored_image_row_whose_base_path_breaks_is_skipped_and_counted(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (2, 2), "red").save(path)
    row = {"id": "img", "state": "s", "image": str(path), "meta": {"anchor": True},
           "questions": {"q": {"type": "choice", "instructions": "c",
                               "criteria": {"yes": "Yes", "no": "No"}}},
           "targets": {"q": {"label": "yes"}}}
    example = train.flatten_row(row)[0]
    processor = _Processor()

    class Broken(_Model):
        def disable_adapter(self):
            raise RuntimeError("adapter-off path cannot take pixel_values")

    loss, _, stats = train.loss_for_examples(
        Broken(), [example], processor.tokenizer, GROUPS, 32, CPU, shuffle=False,
        processor=processor, kl_to_base_weight=1.0)
    assert stats["kl_base_seen"] == 0 and stats["kl_base_skipped_image"] == 1
    assert torch.isfinite(loss)
