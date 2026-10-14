"""Shared fixtures for tests/test_pipeline.py: a tiny real Qwen3 backbone (random
weights, real tokenizer) so tests exercise the actual Backbone/DecisionModel code
paths without downloading or running a full-size model. CPU only, must stay fast."""
import pytest
from transformers import AutoModel, AutoTokenizer, Qwen3Config

from encode import Backbone, FeatureCache

TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"  # cached locally; tiny model reuses its real vocab


@pytest.fixture(autouse=True)
def _no_paid_llm(monkeypatch):
    """site/server.py loads the repo .env at import: no test may reach a paid LLM API (slow, flaky, costs money).
    A test that needs the LLM path stubs typically_llm or sets a fake key itself."""
    for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A tiny random-weight Qwen3 model + the real Qwen3-0.6B-Base tokenizer, saved to
    disk so Backbone(name=<path>) can AutoModel/AutoTokenizer.from_pretrained() it."""
    d = tmp_path_factory.mktemp("tiny_qwen3")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, padding_side="right")
    cfg = Qwen3Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, intermediate_size=128, vocab_size=len(tok), head_dim=16)
    AutoModel.from_config(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def tiny_backbone(tiny_model_dir):
    """Read-only across tests: forward-only usage. Tests that need backward+optimizer
    steps (checkpoint resume) or a second lora_r variant build their own instance from
    tiny_model_dir instead, so they can't mutate this shared one."""
    return Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu")


@pytest.fixture(scope="session")
def tiny_cache(tiny_backbone):
    cache = FeatureCache(device="cpu")
    cache.add(tiny_backbone, [f"cand{i}" for i in range(8)], max_len=16)
    return cache
