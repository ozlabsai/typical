"""Deploy step of typically (.context/typically/FLOW.md): API keys, the protected decide endpoint, code snippets, push to Hugging Face.

Included from site/server.py; reuses its `_run` / `get_model` / `_lock` / `TYPICALLY` (read at call time, so the tests can patch them).
ponytail: v1 serves from this machine; keys live in a local file and the rate limit in memory -- a real deploy needs a database + a gateway.
"""
import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import torch
from fastapi import APIRouter, Header, HTTPException, Request
from huggingface_hub import CommitOperationAdd, HfApi
from pydantic import BaseModel

import server as S
import typically_job

router = APIRouter()
ALIASES = {"northwind": "co_f"}   # the demo model id -> its run
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
LICENSE = {"small": "apache-2.0", "medium": "other"}   # medium: the Qwen3.5-4B-Base licence is not verified (releases/typical-medium.md)
RATE, WINDOW_S = 60, 60

# the Northwind demo, same as SAMPLE in typically-app/src/lib/project.ts (keep in sync)
NORTHWIND = [
    {"question": "Which team should handle this ticket?", "type": "choice", "labels": ["Claims", "Billing", "Dispatch", "Sales"]},
    {"question": "Should we escalate this to a manager?", "type": "noul", "labels": ["no", "yes"]},
    {"question": "How urgent is this ticket?", "type": "score", "labels": ["0", "1", "2", "3"]},
    {"question": "Can the agent refund this without approval?", "type": "noul", "labels": ["no", "yes"]},
]
NORTHWIND_CASE = ("Customer: Cedar Foods (enterprise plan, US). Tickets from this customer in the last 30 days: 1. Shipment value: $450, shipped 4 days ago.\n\n"
                  "Hi, two cases of glassware arrived cracked this morning. Photos attached. Can you sort out a replacement?")


def run_of(model_id: str) -> str:
    if not typically_job.SLUG_RE.fullmatch(model_id):
        raise HTTPException(404, "no such model")
    return ALIASES.get(model_id, f"co_{model_id}")


def model_id_of(run: str) -> str:
    return next((k for k, v in ALIASES.items() if v == run), run.removeprefix("co_"))


# -- API keys: only sha256(key) is stored
_keys_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)   # ponytail: in-memory sliding window per key; gone on restart, not shared between processes


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _keys_file():
    return S.TYPICALLY / "keys.json"


def _read_keys() -> dict:
    f = _keys_file()
    return json.loads(f.read_text()) if f.exists() else {}


class KeyRequest(BaseModel):
    model_id: str


@router.post("/api/typically/keys")
def create_key(req: KeyRequest):
    run_of(req.model_id)   # validates the id
    key = "tpk_" + secrets.token_urlsafe(24)
    with _keys_lock:
        keys = _read_keys() | {_hash(key): {"model_id": req.model_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
        f = _keys_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # 0600 from the first byte
        with os.fdopen(fd, "w") as out:
            out.write(json.dumps(keys))
        os.replace(tmp, f)
    return {"key": key, "prefix": key[:8]}   # the key is shown once: nothing here can show it again


def _authorize(model_id: str, authorization: str | None) -> None:
    scheme, _, key = (authorization or "").partition(" ")
    h = _hash(key)
    with _keys_lock:
        rec = _read_keys().get(h) if scheme.lower() == "bearer" and key else None
        if rec is None or rec["model_id"] != model_id:
            raise HTTPException(401, "missing or invalid API key for this model", headers={"WWW-Authenticate": "Bearer"})
        now, hits = time.monotonic(), _hits[h]
        while hits and now - hits[0] > WINDOW_S:
            hits.popleft()
        if len(hits) >= RATE:
            raise HTTPException(429, f"rate limit: {RATE} requests per minute per key")
        hits.append(now)


@router.post("/v1/models/{model_id}/decide")
def decide(model_id: str, req: S.DecideRequest, authorization: str | None = Header(None)):
    _authorize(model_id, authorization)
    run = run_of(model_id)
    if not (S.TYPICALLY / "results" / run / "best.pt").exists():
        raise HTTPException(404, "that model is not trained yet")
    with S._lock:
        return {**S._run(S.get_model(f"local:{run}"), req.state, req.queries), "model": model_id}


# -- snippets
def _questions(model_id: str) -> list[dict]:
    plan = S.TYPICALLY / "jobs" / model_id / "plan.json"
    if plan.exists():
        return [{k: d[k] for k in ("question", "type", "labels")} for d in json.loads(plan.read_text())["decisions"] if d.get("include", True)]
    return NORTHWIND if model_id == "northwind" else []


def _example_case(model_id: str) -> str:
    val = S.TYPICALLY / "jobs" / model_id / "val.jsonl"
    if model_id == "northwind":
        return NORTHWIND_CASE
    return json.loads(val.open().readline())["state"] if val.exists() else "Describe one case here: the facts first, then what the person wrote."


def _indent(text: str, n: int) -> str:
    return text.replace("\n", "\n" + " " * n)


def _sdk(repo: str, model_id: str) -> str:
    calls = {"choice": "m.choice(case, {q!r}, {l})", "noul": "m.noul(case, {q!r})", "score": "m.score(case, {q!r}, {l})"}
    qs = _questions(model_id) or [{"question": "Your question?", "type": "choice", "labels": ["a", "b"]}]
    lines = "\n".join("print(" + calls[q["type"]].format(q=q["question"], l=q["labels"]) + ")" for q in qs)
    return (f"# pip install typical-ai   (private repo: export HF_TOKEN=hf_... first)\nfrom typical_ai import Typical\n\n"
            f"m = Typical.from_pretrained({repo!r})\ncase = {_example_case(model_id)!r}\n{lines}")


def _hf_repo_file(run: str):
    return S.TYPICALLY / "results" / run / "hf_repo.txt"


@router.get("/api/typically/snippets/{model_id}")
def snippets(model_id: str, request: Request):
    run = run_of(model_id)
    url = f"{str(request.base_url).rstrip('/')}/v1/models/{model_id}/decide"
    qs = _questions(model_id) or [{"question": "Your question?", "type": "choice", "labels": ["a", "b"]}]
    body = json.dumps({"state": _example_case(model_id), "queries": qs}, indent=2, ensure_ascii=False)
    repo = f.read_text().strip() if (f := _hf_repo_file(run)).exists() else "<hf-user>/<model-name>"
    shell_body = body.replace("'", "'\\''")   # close the quote, an escaped quote, reopen
    return {
        "curl": (f"curl -s {url} \\\n  -H \"Authorization: Bearer $TYPICAL_API_KEY\" \\\n  -H \"Content-Type: application/json\" \\\n"
                 f"  -d '{shell_body}'"),
        "python": (f"import os\nimport requests\n\nr = requests.post(\n    \"{url}\",\n"
                   f"    headers={{\"Authorization\": f\"Bearer {{os.environ['TYPICAL_API_KEY']}}\"}},\n"
                   f"    json={_indent(body, 4)},\n)\nprint(r.json()[\"results\"])"),
        "javascript": (f"const res = await fetch(\"{url}\", {{\n  method: \"POST\",\n"
                       f"  headers: {{ Authorization: `Bearer ${{process.env.TYPICAL_API_KEY}}`, \"Content-Type\": \"application/json\" }},\n"
                       f"  body: JSON.stringify({_indent(body, 2)}),\n}});\nconsole.log((await res.json()).results);"),
        "sdk": _sdk(repo, model_id),
    }


# -- push to Hugging Face
class PushRequest(BaseModel):
    run: str
    repo: str
    token: str   # ponytail: used for this one call only -- never written to disk, never logged
    private: bool = True


def _card(base: str, repo: str, model_id: str, step: int, reveal: dict | None) -> str:
    qs = _questions(model_id)
    decides = "\n".join(f"- {q['question']} ({q['type']}: {', '.join(q['labels'])})" for q in qs) or "- (see the usage snippet)"
    results = ""
    if reveal:
        rows = "\n".join(f"| {d['question']} | {d['standard']:.0%} | {d['yours']:.0%} |" for d in reveal["decisions"])
        results = (f"\n## Held-out results\n\n{reveal['n_cases']} held-out cases the model never trained on "
                   f"(overall {reveal['score']['standard']:.0%} for the standard model, {reveal['score']['yours']:.0%} for this one).\n\n"
                   f"| Decision | Standard {base} | This model |\n|---|---|---|\n{rows}\n")
    return (f"---\nlicense: {LICENSE[base]}\nbase_model: OzLabs/typical-{base}\nlibrary_name: pytorch\ntags:\n  - typical\n  - decision-model\n---\n\n"
            f"# {repo.split('/')[1]}\n\nA Typical decision model fine-tuned from [OzLabs/typical-{base}](https://huggingface.co/OzLabs/typical-{base}) "
            f"(step {step}). It decides:\n\n{decides}\n{results}\n## Usage\n\n```python\n{_sdk(repo, model_id)}\n```\n"
            + ("\nThis model inherits the terms of `Qwen/Qwen3.5-4B-Base`, whose licence has not been verified.\n" if base == "medium" else ""))


@router.post("/api/typically/push")
def push(req: PushRequest):
    if not typically_job.SLUG_RE.fullmatch(req.run) or not REPO_RE.fullmatch(req.repo):
        raise HTTPException(400, "need a valid run and a repo like user/name")
    res = S.TYPICALLY / "results" / req.run
    best = res / "best.pt"
    if not best.exists():
        raise HTTPException(404, "no such model")
    ckpt = torch.load(best, map_location="cpu", weights_only=True)
    base = next((b for b, a in typically_job.RELEASED_ARGS.items() if a["backbone"] == ckpt["args"]["backbone"]), None)
    if base is None:
        raise HTTPException(400, f"unknown backbone {ckpt['args']['backbone']!r}")
    reveal = json.loads((res / "reveal.json").read_text()) if (res / "reveal.json").exists() else None
    api = HfApi(token=req.token)
    try:
        manifest = {"base_repo": f"OzLabs/typical-{base}", "base_sha": api.model_info(f"OzLabs/typical-{base}").sha, "run": req.run,
                    "step": ckpt["step"], "best_pt_sha256": hashlib.sha256(best.read_bytes()).hexdigest()}
        api.create_repo(req.repo, private=req.private, exist_ok=True)
        api.create_commit(req.repo, commit_message=f"Typical fine-tune {req.run}", operations=[
            CommitOperationAdd("best.pt", str(best)),
            CommitOperationAdd("README.md", _card(base, req.repo, model_id_of(req.run), ckpt["step"], reveal).encode()),
            CommitOperationAdd("MANIFEST.json", json.dumps(manifest, indent=2).encode())])
    except Exception as e:   # hub errors can echo request headers: never hand the token back
        raise HTTPException(502, f"Hugging Face said: {type(e).__name__}: {str(e).replace(req.token, '***')[:300]}")
    _hf_repo_file(req.run).write_text(req.repo)
    return {"url": f"https://huggingface.co/{req.repo}", "files": ["best.pt", "README.md", "MANIFEST.json"]}
