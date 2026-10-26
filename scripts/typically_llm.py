"""One LLM provider layer for every typically AI feature: an Anthropic key or an OpenRouter key (ANTHROPIC_API_KEY wins).

    if available(): out, usage = complete_json(system, user, schema, max_tokens=1000)   # out: dict; usage: {input_tokens, output_tokens}
    usd(tokens_in, tokens_out)                                                           # list price of the active provider, for budgets
    msg, usage = chat_tools(messages, tools, model=...)                                   # one tool-use turn (OpenRouter), usage["usd"] = billed cost

`user` is a string or a full messages list (user/assistant turns). Failures raise LLMError (type + status only, never a key).
"""
import functools
import json
import os
import re

import httpx

ANTHROPIC_MODEL, FALLBACK_OR_MODEL = "claude-opus-5-5", "anthropic/claude-opus-4.5"
USD_PER_MTOK = (4.0, 20.0)   # opus input / output; also the OpenRouter fallback when /models has no pricing
OR_URL, TIMEOUT_S, MODELS_TIMEOUT_S = "https://openrouter.ai/api/v1", 180, 15


class LLMError(RuntimeError):
    """Any provider failure (HTTP, network, unusable JSON); the message never holds a key."""


def provider() -> str | None:
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openrouter" if os.environ.get("OPENROUTER_API_KEY") else None


def available() -> bool:
    return provider() is not None


def _http(method: str, url: str, *, timeout: float, headers: dict | None = None, body: dict | None = None) -> dict:
    try:
        r = httpx.request(method, url, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:   # type + status only: the request headers hold the key
        raise LLMError(f"HTTPStatusError {e.response.status_code}") from None
    except (httpx.HTTPError, ValueError) as e:
        raise LLMError(type(e).__name__) from None


@functools.lru_cache(maxsize=None)
def _openrouter_model(pinned: str) -> tuple[str, tuple[float, float]]:
    """(model id, (input, output) USD per MTok): OPENROUTER_MODEL if set, else the newest anthropic/claude-opus*; looked up once."""
    try:
        models = _http("GET", f"{OR_URL}/models", timeout=MODELS_TIMEOUT_S)["data"]
    except (LLMError, KeyError, TypeError):
        models = []
    opus = [m for m in models if re.match(r"anthropic/claude-opus", m.get("id", ""))]
    mid = pinned or (max(opus, key=lambda m: m.get("created", 0))["id"] if opus else FALLBACK_OR_MODEL)
    try:
        p = next(m for m in models if m.get("id") == mid)["pricing"]
        return mid, (float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6)
    except (StopIteration, KeyError, TypeError, ValueError):
        return mid, USD_PER_MTOK


def _model(model: str | None = None) -> tuple[str, tuple[float, float]]:
    return _openrouter_model(model or os.environ.get("OPENROUTER_MODEL", ""))


def usd(tokens_in: float, tokens_out: float, model: str | None = None) -> float:
    pin, pout = _model(model)[1] if provider() == "openrouter" else USD_PER_MTOK
    return (tokens_in * pin + tokens_out * pout) / 1e6


def _parse(text: str) -> dict:
    """json.loads, else the first {...} block (a model that ignored response_format may wrap the JSON in prose or fences)."""
    try:
        out = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            out = json.loads(m.group(0)) if m else None
        except ValueError:
            out = None
    if not isinstance(out, dict):
        raise LLMError("the model did not return a JSON object")
    return out


def complete_json(system: str, user: str | list[dict], schema: dict, *, max_tokens: int, effort: str = "medium", client=None,
                  model: str | None = None):
    """One structured-output call -> (dict, {"input_tokens", "output_tokens"}). `client`: an anthropic-shaped client (tests) forcing that path.
    `model`: a per-call override (e.g. a cheap triage model); price it with usd(..., model=model)."""
    messages = [{"role": "user", "content": user}] if isinstance(user, str) else user
    if client is not None or provider() == "anthropic":
        import anthropic
        client = client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=TIMEOUT_S)
        try:
            resp = client.messages.create(
                model=model or ANTHROPIC_MODEL, max_tokens=max_tokens, system=system, messages=messages,
                output_config={"format": {"type": "json_schema", "schema": schema}, "effort": effort})
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise LLMError(f"{type(e).__name__}{getattr(e, 'status_code', '') and ' ' + str(e.status_code)}") from None
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return _parse(text), {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    if provider() != "openrouter":
        raise LLMError("no LLM key configured (ANTHROPIC_API_KEY or OPENROUTER_API_KEY)")
    data = _http("POST", f"{OR_URL}/chat/completions", timeout=TIMEOUT_S, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "HTTP-Referer": "https://typical.ozlabs.ai", "X-Title": "typically"},
        body={"model": _model(model)[0], "max_tokens": max_tokens, "messages": [{"role": "system", "content": system}, *messages],
              "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}}})
    try:
        text, u = data["choices"][0]["message"]["content"] or "", data.get("usage") or {}
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected OpenRouter response") from None
    return _parse(text), {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}


def chat_tools(messages: list[dict], tools: list[dict], *, model: str | None = None, max_tokens: int = 2000,
               tool_choice: str | dict = "auto"):
    """One OpenRouter chat/completions turn with OpenAI-format `tools` -> (assistant message dict, usage). `messages` start with the
    system message; it is marked for Anthropic prompt caching (OpenRouter passes cache_control through; below the model's minimum
    prompt size it is a no-op). usage: {input_tokens, output_tokens, cached_tokens, usd} (usd: OpenRouter's billed cost, else list price)."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise LLMError("chat_tools needs OPENROUTER_API_KEY")
    sys_, *rest = messages
    sys_ = {"role": "system", "content": [{"type": "text", "text": sys_["content"], "cache_control": {"type": "ephemeral"}}]}
    data = _http("POST", f"{OR_URL}/chat/completions", timeout=TIMEOUT_S, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "HTTP-Referer": "https://typical.ozlabs.ai", "X-Title": "typically"},
        body={"model": _model(model)[0], "max_tokens": max_tokens, "messages": [sys_, *rest], "tools": tools, "tool_choice": tool_choice,
              "usage": {"include": True}, **({"provider": {"order": ["Anthropic"]}} if _model(model)[0].startswith("anthropic/") else {})})   # one provider keeps the cache warm
    try:
        msg, u = data["choices"][0]["message"], data.get("usage") or {}
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected OpenRouter response") from None
    tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    cost = u.get("cost")
    return msg, {"input_tokens": tin, "output_tokens": tout, "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                 "usd": float(cost) if cost is not None else usd(tin, tout, model)}
