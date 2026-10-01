"""scripts/typically_llm.py: provider selection, OpenRouter request shape, JSON extraction, model auto-pick, the Anthropic path. No network.
uv run --no-sync --with pytest --with httpx pytest tests/test_typically_llm.py -q"""
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts")]
import typically_llm as tl  # noqa: E402

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
MODELS = {"data": [
    {"id": "anthropic/claude-opus-4.1", "created": 100, "pricing": {"prompt": "0.000015", "completion": "0.000075"}},
    {"id": "anthropic/claude-opus-4.5", "created": 300, "pricing": {"prompt": "0.000005", "completion": "0.000025"}},
    {"id": "anthropic/claude-sonnet-9", "created": 900, "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    {"id": "openai/gpt-x", "created": 999}]}


@pytest.fixture
def env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    tl._openrouter_model.cache_clear()
    yield monkeypatch
    tl._openrouter_model.cache_clear()


def fake_http(monkeypatch, content='{"a": 1}', models=MODELS):
    calls = []

    def http(method, url, *, timeout, headers=None, body=None):
        calls.append({"method": method, "url": url, "timeout": timeout, "headers": headers, "body": body})
        if method == "GET":
            return models
        return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
    monkeypatch.setattr(tl, "_http", http)
    return calls


def test_provider_selection(env):
    assert tl.provider() is None and not tl.available()
    env.setenv("OPENROUTER_API_KEY", "or")
    assert tl.provider() == "openrouter" and tl.available()
    env.setenv("ANTHROPIC_API_KEY", "an")
    assert tl.provider() == "anthropic"   # wins when both are set


def test_openrouter_request_shape_and_usage(env):
    env.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    calls = fake_http(env)
    out, usage = tl.complete_json("sys", "hi", SCHEMA, max_tokens=50)
    assert out == {"a": 1} and usage == {"input_tokens": 11, "output_tokens": 7}
    get, post = calls
    assert post["url"] == "https://openrouter.ai/api/v1/chat/completions" and post["timeout"] > 0 and get["timeout"] > 0
    assert post["headers"] == {"Authorization": "Bearer sk-or-secret", "HTTP-Referer": "https://typical.ozlabs.ai", "X-Title": "typically"}
    assert post["body"] == {"model": "anthropic/claude-opus-4.5", "max_tokens": 50,
                            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": SCHEMA}}}


def test_messages_list_passes_through(env):
    env.setenv("OPENROUTER_API_KEY", "k")
    calls = fake_http(env)
    turns = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "{}"}, {"role": "user", "content": "fix"}]
    tl.complete_json("s", turns, SCHEMA, max_tokens=5)
    assert calls[-1]["body"]["messages"][1:] == turns


@pytest.mark.parametrize("content", ['Sure! Here you go:\n```json\n{"a": 2}\n```', '{"a": 2} trailing words'])
def test_json_extraction_fallback(env, content):
    env.setenv("OPENROUTER_API_KEY", "k")
    fake_http(env, content=content)
    assert tl.complete_json("s", "u", SCHEMA, max_tokens=5)[0] == {"a": 2}


@pytest.mark.parametrize("content", ["no json at all", "{broken", "[1, 2]", None])
def test_unusable_json_raises_typed_error(env, content):
    env.setenv("OPENROUTER_API_KEY", "k")
    fake_http(env, content=content)
    with pytest.raises(tl.LLMError):
        tl.complete_json("s", "u", SCHEMA, max_tokens=5)


def test_model_autopick_pin_pricing_and_fallback(env):
    env.setenv("OPENROUTER_API_KEY", "k")
    newer = {"id": "anthropic/claude-opus-5", "created": 500, "pricing": {"prompt": "0.000004", "completion": "0.00002"}}
    calls = fake_http(env, models={"data": MODELS["data"] + [newer]})
    assert tl._model() == ("anthropic/claude-opus-5", (4.0, 20.0)) and tl._model()[0] == "anthropic/claude-opus-5"
    assert [c["method"] for c in calls] == ["GET"]   # looked up once
    assert tl.usd(1e6, 1e6) == pytest.approx(24.0)
    tl._openrouter_model.cache_clear()
    env.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.1")
    fake_http(env)
    assert tl._model() == ("anthropic/claude-opus-4.1", (15.0, 75.0)) and tl.usd(1e6, 0) == pytest.approx(15.0)
    tl._openrouter_model.cache_clear()
    env.delenv("OPENROUTER_MODEL")

    def down(*a, **k):
        raise tl.LLMError("down")
    env.setattr(tl, "_http", down)
    assert tl._model() == ("anthropic/claude-opus-4.5", (4.0, 20.0))   # lookup failed: the fallback id at opus prices


def test_http_errors_never_carry_the_key(env):
    env.setattr(tl.httpx, "request", lambda *a, **k: httpx.Response(401, request=httpx.Request("POST", "https://x"), text="bad key sk-or-secret"))
    with pytest.raises(tl.LLMError) as e:
        tl._http("POST", "https://x", timeout=1, headers={"Authorization": "Bearer sk-or-secret"})
    assert "401" in str(e.value) and "secret" not in str(e.value)


def test_anthropic_path_with_fake_client(env):
    env.setenv("ANTHROPIC_API_KEY", "k")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"a": 3}')], usage=SimpleNamespace(input_tokens=5, output_tokens=2))
    out, usage = tl.complete_json("sys", "hi", SCHEMA, max_tokens=9, effort="low", client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert out == {"a": 3} and usage == {"input_tokens": 5, "output_tokens": 2}
    assert seen["model"] == "claude-opus-5-5" and seen["system"] == "sys" and seen["messages"] == [{"role": "user", "content": "hi"}]
    assert seen["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"}
    assert tl.usd(1e6, 1e6) == 24.0   # anthropic: opus list price, no /models lookup


def test_no_key_is_a_typed_error(env):
    with pytest.raises(tl.LLMError):
        tl.complete_json("s", "u", SCHEMA, max_tokens=5)
