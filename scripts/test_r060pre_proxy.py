"""R060-pre: rank-concatenated merge is the exact weighted sum of LoRA deltas.  uv run python scripts/test_r060pre_proxy.py"""
import json, tempfile, torch, sys
from pathlib import Path
from safetensors.torch import save_file, load_file
sys.path.insert(0, str(Path(__file__).parent)); import r060pre_proxy as r
d = Path(tempfile.mkdtemp()); cfg = {"r": 4, "lora_alpha": 8, "target_modules": "x"}
ts = []
for n in ("a", "b"):
    (d / n).mkdir(); t = {"m.lora_A.weight": torch.randn(4, 6), "m.lora_B.weight": torch.randn(5, 4)}; ts.append(t)
    save_file(t, d / n / "adapter_model.safetensors"); (d / n / "adapter_config.json").write_text(json.dumps(cfg))
r.merge(d / "a", 1.0, d / "b", 0.5, d / "m")
m = load_file(d / "m" / "adapter_model.safetensors"); c = json.loads((d / "m" / "adapter_config.json").read_text())
lhs = c["lora_alpha"] / c["r"] * m["m.lora_B.weight"] @ m["m.lora_A.weight"]
rhs = 2 * (ts[0]["m.lora_B.weight"] @ ts[0]["m.lora_A.weight"] + .5 * ts[1]["m.lora_B.weight"] @ ts[1]["m.lora_A.weight"])
assert torch.allclose(lhs, rhs, atol=1e-5) and c["r"] == 8 and c["lora_alpha"] == 16; print("merge exact-sum OK")
