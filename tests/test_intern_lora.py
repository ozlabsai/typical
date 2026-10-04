from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scripts.train_intern_lora import loss_for_rows


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_):
        return messages[0]["content"]

    def __call__(self, text, **_):
        ids = [7, 99, 8] if text == "short" else [7, 8, 99]
        return {"input_ids": torch.tensor([ids])}

    def encode(self, symbol, **_):
        return [{"A": 1, "B": 2}[symbol]]


class FakeModel:
    def __call__(self, input_ids, logits_to_keep, **_):
        logits = torch.zeros(input_ids.shape[0], len(logits_to_keep), 4)
        offsets = {position.item(): index for index, position in enumerate(logits_to_keep)}
        logits[0, offsets[0], 1] = 2
        if input_ids.shape[0] > 1:
            logits[1, offsets[1], 2] = 2
        return SimpleNamespace(logits=logits)


def test_marker_logits_align_across_padded_batch():
    def compile_row(row):
        return SimpleNamespace(messages=[{"content": row["text"]}], fields=("q",),
                               symbols={"q": ("A", "B")}, targets={"q": row["symbol"]})

    loss = loss_for_rows(FakeModel(), [{"text": "short", "symbol": "A"},
                                       {"text": "long", "symbol": "B"}],
                         FakeTokenizer(), compile_row, marker_id=99, max_length=8,
                         device=torch.device("cpu"))
    expected = F.cross_entropy(torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]),
                               torch.tensor([1, 2]))
    assert torch.equal(loss, expected)


def test_soft_targets_use_full_vocabulary_cross_entropy():
    def compile_row(row):
        return SimpleNamespace(messages=[{"content": row["text"]}], fields=("q",),
                               symbols={"q": ("A", "B")}, targets={"q": "A"})

    row = {"text": "short", "symbol": "A", "soft_targets": {"q": [0.25, 0.75]}}
    loss = loss_for_rows(FakeModel(), [row], FakeTokenizer(), compile_row,
                         marker_id=99, max_length=8, device=torch.device("cpu"))
    expected = -(torch.tensor([0.25, 0.75]) * F.log_softmax(torch.tensor([0.0, 2.0, 0.0, 0.0]), 0)[[1, 2]]).sum()
    assert torch.equal(loss, expected)
