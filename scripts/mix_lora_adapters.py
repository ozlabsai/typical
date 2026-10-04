"""Create an exact convex LoRA delta mixture by concatenating ranks."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _weights_path(adapter_dir):
    path = Path(adapter_dir)
    for name in ("adapter_model.safetensors", "model.safetensors"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no PEFT safetensors weights in {path}")


def _read_config(adapter_dir):
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text())


def _validate_configs(config_a, config_b):
    if config_a.get("peft_type", "LORA") != "LORA" or config_b.get("peft_type", "LORA") != "LORA":
        raise ValueError("only standard PEFT LoRA adapters are supported")
    for key in ("r", "lora_alpha"):
        if config_a.get(key) != config_b.get(key):
            raise ValueError(f"adapter configs differ at {key}")
    if int(config_a.get("r", 0)) <= 0 or float(config_a.get("lora_alpha", 0)) <= 0:
        raise ValueError("adapter rank and alpha must be positive")
    if config_a.get("rank_pattern") or config_b.get("rank_pattern") or \
            config_a.get("alpha_pattern") or config_b.get("alpha_pattern"):
        raise ValueError("rank/alpha patterns are unsupported; uniform rank and alpha are required")
    if config_a.get("use_rslora") or config_b.get("use_rslora") or \
            config_a.get("use_dora") or config_b.get("use_dora"):
        raise ValueError("RS-LoRA and DoRA scaling are unsupported")
    ignored = {"r", "lora_alpha"}
    config_a_common = {key: value for key, value in config_a.items() if key not in ignored}
    config_b_common = {key: value for key, value in config_b.items() if key not in ignored}
    if config_a_common != config_b_common:
        raise ValueError("adapter configs differ beyond r/lora_alpha")


def _tensor_keys(path):
    with safe_open(str(path), framework="pt", device="cpu") as source:
        return set(source.keys())


def _pairs(keys):
    suffix_a, suffix_b = "lora_A.weight", "lora_B.weight"
    a_keys = {key for key in keys if key.endswith(suffix_a)}
    b_keys = {key for key in keys if key.endswith(suffix_b)}
    if not a_keys or len(a_keys) != len(b_keys) or len(a_keys) + len(b_keys) != len(keys):
        raise ValueError("weights must contain only paired lora_A.weight/lora_B.weight tensors")
    result = []
    for a_key in sorted(a_keys):
        b_key = a_key[:-len(suffix_a)] + suffix_b
        if b_key not in b_keys:
            raise ValueError(f"missing paired tensor {b_key}")
        result.append((a_key, b_key))
    return result


def mix_adapter(adapter_a, adapter_b, weight_a, out_dir):
    if not math.isfinite(weight_a) or not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight must be finite and in [0, 1]")
    config_a, config_b = _read_config(adapter_a), _read_config(adapter_b)
    _validate_configs(config_a, config_b)
    path_a, path_b = _weights_path(adapter_a), _weights_path(adapter_b)
    keys_a, keys_b = _tensor_keys(path_a), _tensor_keys(path_b)
    if keys_a != keys_b:
        raise ValueError("adapter tensor key sets differ")
    pairs = _pairs(keys_a)
    out = Path(out_dir)
    mixed = {}
    with safe_open(str(path_a), framework="pt", device="cpu") as tensors_a, \
            safe_open(str(path_b), framework="pt", device="cpu") as tensors_b:
        for a_key, b_key in pairs:
            a1, b1 = tensors_a.get_tensor(a_key), tensors_a.get_tensor(b_key)
            a2, b2 = tensors_b.get_tensor(a_key), tensors_b.get_tensor(b_key)
            if a1.ndim != 2 or a2.ndim != 2 or b1.ndim != 2 or b2.ndim != 2:
                raise ValueError(f"LoRA tensors must be matrices: {a_key}, {b_key}")
            if a1.shape != a2.shape or b1.shape != b2.shape:
                raise ValueError(f"adapter tensor shapes differ for {a_key} / {b_key}")
            if a1.shape[0] != config_a["r"] or b1.shape[1] != config_a["r"]:
                raise ValueError(f"tensor rank disagrees with config for {a_key} / {b_key}")
            if (a1.dtype, b1.dtype) != (a2.dtype, b2.dtype):
                raise ValueError(f"adapter tensor dtypes differ for {a_key} / {b_key}")
            mixed[a_key] = torch.cat((a1, a2), dim=0).contiguous()
            mixed[b_key] = torch.cat((weight_a * b1, (1.0 - weight_a) * b2), dim=1).contiguous()

    out.mkdir(parents=True, exist_ok=False)
    config = dict(config_a)
    config["r"] = 2 * int(config_a["r"])
    config["lora_alpha"] = 2 * config_a["lora_alpha"]
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(mixed, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    readme_a, readme_b = Path(adapter_a) / "README.md", Path(adapter_b) / "README.md"
    readme = readme_a if readme_a.is_file() else readme_b
    if readme.is_file():
        shutil.copyfile(readme, out / "README.md")
    return {"rank": config["r"], "lora_alpha": config["lora_alpha"],
            "tensor_pairs": len(pairs), "weight_a": weight_a, "weight_b": 1.0 - weight_a,
            "out": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--weight-a", type=float, required=True,
                        help="convex coefficient for adapter A; B receives 1-weight-a")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(mix_adapter(args.adapter_a, args.adapter_b, args.weight_a, args.out), indent=2))


if __name__ == "__main__":
    main()
