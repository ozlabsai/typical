import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import scripts.train_gemma_decision_lora as train


def _row(image=None):
    row = {
        "id": "case",
        "state": "Use the evidence.",
        "questions": {"q": {
            "type": "choice", "instructions": "Choose.",
            "criteria": {"yes": "Yes", "no": "No"},
        }},
        "targets": {"q": {"label": "yes"}},
    }
    if image is not None:
        row["image"] = image
    return row


class _Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.templates = []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return "templated"

    def __call__(self, text, **kwargs):
        assert text == "templated"
        assert kwargs == {"add_special_tokens": False, "return_tensors": "pt"}
        return {"input_ids": torch.tensor([[4, 5, 6]])}


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.messages = None
        self.image_modes = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                          "enable_thinking": False}
        return "multimodal-template"

    def __call__(self, *, text, images, return_tensors):
        assert text == ["multimodal-template"]
        assert return_tensors == "pt"
        self.image_modes.append(images[0].mode)
        red, _, blue = images[0].getpixel((0, 0))
        return {
            "input_ids": torch.tensor([[7, 8, 9]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.tensor([[float(red - blue) / 255]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        value = kwargs["pixel_values"][0, 0]
        logits = torch.zeros(1, len(kwargs["logits_to_keep"]), 4)
        logits[0, 0, 1] = value
        logits[0, 0, 2] = -value
        return SimpleNamespace(logits=logits)


def test_load_examples_resolves_each_relative_image_from_its_jsonl(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        (directory / "rows.jsonl").write_text(json.dumps(_row("image.png")) + "\n")

    examples = train.load_examples([left / "rows.jsonl", right / "rows.jsonl"])

    assert [example["image"] for example in examples] == [
        str((left / "image.png").resolve()), str((right / "image.png").resolve())]


def test_image_prompt_and_all_processor_tensors_reach_model_and_change_loss(tmp_path):
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    Image.new("RGB", (2, 2), "red").save(red)
    Image.new("RGB", (2, 2), "blue").save(blue)
    processor = _Processor()
    model = _Model()
    groups = [torch.tensor([1]), torch.tensor([2])]

    losses = []
    for path in (red, blue):
        example = train.flatten_row(_row(str(path)))[0]
        loss, _, _ = train.loss_for_examples(
            model, [example], processor.tokenizer, groups, 32, torch.device("cpu"),
            shuffle=False, processor=processor)
        losses.append(loss)

    prompt, _ = train.render(train.flatten_row(_row(str(red)))[0])
    assert processor.messages == [
        {"role": "system", "content": train.SYSTEM},
        {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]},
    ]
    assert processor.image_modes == ["RGB", "RGB"]
    assert all("pixel_values" in call and "image_grid_thw" in call for call in model.calls)
    assert model.calls[0]["pixel_values"].item() == 1
    assert model.calls[1]["pixel_values"].item() == -1
    assert losses[0] < losses[1]


def test_image_slots_fail_closed_outside_singleton_microbatches(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (1, 1)).save(image)
    example = train.flatten_row(_row(str(image)))[0]
    with pytest.raises(ValueError, match="microbatch 1"):
        train.validate_multimodal_examples([example], microbatch=2)
    with pytest.raises(ValueError, match="singleton logical slots"):
        train.loss_for_examples(
            _Model(), [example, example], _Tokenizer(), [torch.tensor([1]), torch.tensor([2])],
            32, torch.device("cpu"), shuffle=False, processor=_Processor())


def test_init_adapter_is_loaded_trainable(monkeypatch):
    seen = {}

    def load(model, path, **kwargs):
        seen.update(model=model, path=path, **kwargs)
        return "continued"

    monkeypatch.setattr(train.PeftModel, "from_pretrained", load)
    monkeypatch.setattr(train, "get_peft_model", lambda *_args, **_kwargs: pytest.fail("new LoRA"))

    assert train.attach_lora("base", "adapter", 32, 64) == "continued"
    assert seen == {"model": "base", "path": "adapter", "is_trainable": True,
                    "local_files_only": True}


def test_text_encode_contract_is_unchanged():
    tokenizer = _Tokenizer()
    example = train.flatten_row(_row())[0]

    ids, probabilities, labels = train.encode(
        example, tokenizer, 16, shuffle=False, include_labels=True)

    assert ids.tolist() == [4, 5, 6]
    assert probabilities.tolist() == [1.0, 0.0]
    assert labels == ["yes", "no"]
    assert tokenizer.templates == [([
        {"role": "system", "content": train.SYSTEM},
        {"role": "user", "content": train.render(example)[0]},
    ], {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})]


def test_processor_is_loaded_only_for_image_data(monkeypatch):
    tokenizer = object()
    processor = SimpleNamespace(tokenizer=tokenizer)
    calls = []
    monkeypatch.setattr(train.AutoProcessor, "from_pretrained",
                        lambda *_a, **_k: calls.append("processor") or processor)
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained",
                        lambda *_a, **_k: calls.append("tokenizer") or tokenizer)

    assert train.load_tokenizer_and_processor("model", False) == (tokenizer, None)
    assert train.load_tokenizer_and_processor("model", True) == (tokenizer, processor)
    assert calls == ["tokenizer", "processor"]


def test_trainable_adapter_must_stay_in_language_model():
    class Model:
        def named_parameters(self):
            return [("base.language_model.layer.lora_A", SimpleNamespace(requires_grad=True)),
                    ("base.vision_tower.layer", SimpleNamespace(requires_grad=False))]

    train.validate_trainable_path(Model())
    bad = Model()
    bad.named_parameters = lambda: [
        ("base.language_model.layer.lora_A", SimpleNamespace(requires_grad=True)),
        ("base.vision_tower.layer.lora_A", SimpleNamespace(requires_grad=True)),
    ]
    with pytest.raises(RuntimeError, match="multimodal path"):
        train.validate_trainable_path(bad)
