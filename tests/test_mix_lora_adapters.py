import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.mix_lora_adapters import mix_adapter


def write_adapter(path, a, b, config=None):
    path.mkdir()
    adapter_config = config or {
        "peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": ["proj"],
        "rank_pattern": {}, "alpha_pattern": {}, "use_rslora": False, "use_dora": False,
    }
    (path / "adapter_config.json").write_text(json.dumps(adapter_config))
    save_file({"layer.lora_A.weight": a, "layer.lora_B.weight": b},
              str(path / "adapter_model.safetensors"))


def test_mixes_exact_convex_delta_and_updates_rank_alpha(tmp_path):
    a1 = torch.tensor([[1., 2., 0.], [0., 1., 3.]])
    b1 = torch.tensor([[1., 0.], [0., 2.], [3., 1.], [-1., 2.]])
    a2 = torch.tensor([[2., 0., 1.], [1., -1., 2.]])
    b2 = torch.tensor([[0., 1.], [2., 0.], [1., -1.], [3., 2.]])
    write_adapter(tmp_path / "a", a1, b1)
    write_adapter(tmp_path / "b", a2, b2)

    out = tmp_path / "mixed"
    result = mix_adapter(tmp_path / "a", tmp_path / "b", .25, out)
    tensors = load_file(str(out / "adapter_model.safetensors"))
    config = json.loads((out / "adapter_config.json").read_text())
    expected = .25 * (4 / 2) * (b1 @ a1) + .75 * (4 / 2) * (b2 @ a2)
    actual = (config["lora_alpha"] / config["r"] *
              tensors["layer.lora_B.weight"] @ tensors["layer.lora_A.weight"])

    assert torch.allclose(actual, expected)
    assert config["r"] == 4 and config["lora_alpha"] == 8
    assert result["tensor_pairs"] == 1
    assert not (out / "trainer.pt").exists()


def test_rejects_invalid_weight_and_incompatible_config(tmp_path):
    a, b = torch.ones(2, 3), torch.ones(4, 2)
    write_adapter(tmp_path / "a", a, b)
    config = {"peft_type": "LORA", "r": 3, "lora_alpha": 4, "target_modules": ["proj"],
              "rank_pattern": {}, "alpha_pattern": {}, "use_rslora": False, "use_dora": False}
    write_adapter(tmp_path / "b", a, b, config)
    with pytest.raises(ValueError, match="weight"):
        mix_adapter(tmp_path / "a", tmp_path / "b", 1.1, tmp_path / "bad-weight")
    with pytest.raises(ValueError, match="r"):
        mix_adapter(tmp_path / "a", tmp_path / "b", .5, tmp_path / "bad-config")
