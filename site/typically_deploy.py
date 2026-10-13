"""Deploy step of typically (.context/typically/FLOW.md): API keys, the protected decide endpoint, code snippets, push to Hugging Face.

Included from site/server.py; reuses its `_run` / `get_model` / `_lock` / `TYPICALLY` (read at call time, so the tests can patch them).
ponytail: keys live in <data>/keys.json and the rate limit in memory: one server process only (deploy/README.md); several need a database + a gateway.
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
from huggingface_hub import CommitOperationAdd, HfApi, get_token
from pydantic import BaseModel

import server as S
import typically_auth as auth
import typically_job
import typically_llm
from typically_reveal import baselines

router = APIRouter()
ALIASES = {"northwind": "co_f"}   # the demo model id -> its run
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
LICENSE = "apache-2.0"   # both releases and their backbones (Qwen3-1.7B-Base, Qwen3.5-4B-Base): checked on the Hub 2026-10-13
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
        raise HTTPException(404, "We can't find that model. It may have been archived.")
    return ALIASES.get(model_id, f"co_{model_id}")


def model_id_of(run: str) -> str:
    return next((k for k, v in ALIASES.items() if v == run), run.removeprefix("co_"))


def model_of_run(run: str) -> str:
    """A run ("co_<slug>", "co_f") -> its model id; 404 for anything else. Not SLUG_RE on the run: a 40-char slug makes a 43-char run."""
    mid = model_id_of(run)
    if run_of(mid) != run:
        raise HTTPException(404, "We can't find that model. It may have been archived.")
    return mid


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


def _write_keys(keys: dict) -> None:   # caller holds _keys_lock
    f = _keys_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # 0600 from the first byte
    with os.fdopen(fd, "w") as out:
        out.write(json.dumps(keys))
    os.replace(tmp, f)


@router.post("/api/typically/keys")
def create_key(req: KeyRequest):
    run_of(req.model_id)   # validates the id
    auth.check(req.model_id)
    key = "tpk_" + secrets.token_urlsafe(24)
    with _keys_lock:
        _write_keys(_read_keys() | {_hash(key): {"model_id": req.model_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **auth.stamp()}})
    return {"key": key, "prefix": key[:8]}   # the key is shown once: nothing here can show it again


def revoke_keys(model_id: str, mine_only: bool = False) -> int:
    """Delete the model's keys (mine_only: those the caller may see, the has_key rule); returns how many."""
    with _keys_lock:
        keys = _read_keys()
        keep = {h: r for h, r in keys.items() if r["model_id"] != model_id or (mine_only and not auth.can_see(r.get("owner")))}
        _write_keys(keep)
    return len(keys) - len(keep)


@router.delete("/api/typically/keys/{model_id}")
def delete_keys(model_id: str):
    """Revoke every key of this model the caller can see (all of them for its owner or an admin). Keys are hashed: no list to show."""
    run_of(model_id)
    auth.check(model_id)
    return {"revoked": revoke_keys(model_id, mine_only=True)}


def _authorize(model_id: str, authorization: str | None) -> None:
    scheme, _, key = (authorization or "").partition(" ")
    h = _hash(key)
    with _keys_lock:
        rec = _read_keys().get(h) if scheme.lower() == "bearer" and key else None
        revoked = rec and auth.enabled() and rec.get("owner") and rec["owner"] not in {auth.uid_of(c) for c in auth.invites()}   # invite removed
        if rec is None or rec["model_id"] != model_id or revoked:
            raise HTTPException(401, "Missing or invalid API key for this model.", headers={"WWW-Authenticate": "Bearer"})
        now, hits = time.monotonic(), _hits[h]
        while hits and now - hits[0] > WINDOW_S:
            hits.popleft()
        if len(hits) >= RATE:
            raise HTTPException(429, f"Rate limit: {RATE} requests per minute per key.")
        hits.append(now)


@router.post("/v1/models/{model_id}/decide")
def decide(model_id: str, req: S.DecideRequest, authorization: str | None = Header(None)):
    _authorize(model_id, authorization)
    run = run_of(model_id)
    if not (S.TYPICALLY / "results" / run / "best.pt").exists():
        raise HTTPException(404, "That model isn't trained yet.")
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
    auth.check(model_id)
    url = f"{str(request.base_url).rstrip('/')}/v1/models/{model_id}/decide"
    qs = _questions(model_id) or [{"question": "Your question?", "type": "choice", "labels": ["a", "b"]}]
    body = json.dumps({"state": _example_case(model_id), "queries": qs}, indent=2, ensure_ascii=False)
    repo = f.read_text().strip() if (f := _hf_repo_file(run)).exists() else f"./{run}.pt"   # not pushed: the file "Download weights" saves
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


# -- Hugging Face: the operator's token only (HF_TOKEN or the cached `hf auth login`); users never paste one
ORG = "OzLabs"
_namespaces: dict[str, str] = {}   # ponytail: token -> default push namespace, per process; a failed whoami is not cached


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or get_token()


def _namespace(api: HfApi, token: str | None) -> str | None:
    """OzLabs when the server token's user belongs to that org, else the user's own name; None without a (working) token."""
    if not token:
        return None
    if token not in _namespaces:
        try:
            me = api.whoami()
        except Exception:
            return None
        _namespaces[token] = ORG if any(o.get("name") == ORG for o in me.get("orgs") or []) else me["name"]
    return _namespaces[token]


def _may_push() -> bool:
    """Hosted: admins only (the push uses the operator's token, into the operator's namespace)."""
    return not auth.enabled() or bool((u := auth.current()) and u.admin)


@router.get("/api/typically/capabilities")
def capabilities():
    """can_push: this user may push to Hugging Face here (a working server token, and an admin when hosted); hf_namespace is null otherwise."""
    token = _hf_token()
    ns = _namespace(HfApi(token=token), token) if _may_push() else None
    return {"ai": typically_llm.available(), "hf_namespace": ns, "can_push": ns is not None, "languages": ["en", "he"]}


# -- push to Hugging Face
class PushRequest(BaseModel):
    run: str
    repo: str | None = None   # default: <hf_namespace>/<the model's display name, slugged with ->; ignored when hosted
    private: bool = True   # forced when hosted


def baselines_of(model_id: str, rev: dict | None) -> tuple[dict, float | None]:
    """{decision key: {"majority", "baseline"}} and the pooled baseline (always giving each decision's most common answer):
    from reveal.json, else from the model's eval file (reveals from before reveal() recorded them); ({}, None) without either."""
    if rev and rev.get("baseline") is not None and all("baseline" in d for d in rev["decisions"]):
        return {d["key"]: {"majority": d["majority"], "baseline": d["baseline"]} for d in rev["decisions"]}, rev["baseline"]
    f = (S.REPO_ROOT / "data_co_a/eval/a_oneliner.jsonl" if model_id == "northwind"
         else S.TYPICALLY / "jobs" / model_id / "eval" / "import_oneliner.jsonl")
    try:
        with f.open() as fh:
            return baselines(json.loads(line) for line in fh)
    except (OSError, ValueError, KeyError, IndexError):   # display-only: a missing or odd eval file shows no baseline, never a 500
        return {}, None


ANSWERS = {"noul": lambda q: "yes/no", "choice": lambda q: "pick one: " + ", ".join(q["labels"]), "score": lambda q: "scale: " + ", ".join(q["labels"])}


def _card(base: str, repo: str, model_id: str, step: int, reveal: dict | None) -> str:
    qs = _questions(model_id)
    decides = "\n".join(f"- {q['question']} ({ANSWERS[q['type']](q)})" for q in qs) or "- (see the usage snippet)"
    results = ""
    if reveal:
        per, pooled = baselines_of(model_id, reveal)
        always = lambda b: "n/a" if b is None else f"{b:.0%}"
        rows = "\n".join(f"| {d['question']} | {always(per.get(d.get('key'), {}).get('baseline'))} | {d['standard']:.0%} | {d['yours']:.0%} |"
                         for d in reveal["decisions"])
        results = (f"\n## Results on test cases\n\nAgreement: how often a model gives the answer the team actually gave, on {reveal['n_cases']} "
                   f"test cases kept out of training. Overall: {reveal['score']['standard']:.0%} for the standard model, "
                   f"{reveal['score']['yours']:.0%} for this one" + ("." if pooled is None else
                   f", and {pooled:.0%} for always giving each question's most common answer.") + "\n\n"
                   f"| Question | Most common answer | Standard (OzLabs/typical-{base}) | This model |\n|---|---|---|---|\n{rows}\n")
    return (f"---\nlicense: {LICENSE}\nbase_model: OzLabs/typical-{base}\nlibrary_name: pytorch\ntags:\n  - typical\n  - decision-model\n---\n\n"
            f"# {repo.split('/')[1]}\n\nA decision model fine-tuned from [OzLabs/typical-{base}](https://huggingface.co/OzLabs/typical-{base}) "
            f"with Decision Factory (best checkpoint: training step {step}). It answers:\n\n{decides}\n{results}\n## Usage\n\n```python\n{_sdk(repo, model_id)}\n```\n")


def _display_name(model_id: str) -> str:
    """The model's name as typed in the app (job.json); "" when it has none."""
    job = S.TYPICALLY / "jobs" / model_id / "job.json"
    return (json.loads(job.read_text()).get("name") if job.exists() else None) or ("Northwind triage" if model_id == "northwind" else "")


def _repo_name(model_id: str) -> str:
    """The model's display name as a repo name: "Northwind triage" -> northwind-triage."""
    return re.sub(r"[^a-z0-9]+", "-", _display_name(model_id).lower()).strip("-")[:96] or model_id.replace("_", "-")   # a non-latin name: the id


def _storage_full(e: Exception) -> bool:
    return getattr(getattr(e, "response", None), "status_code", None) == 403 and "storage" in str(e).lower()


@router.post("/api/typically/push")
def push(req: PushRequest):
    """Hosted (TYPICALLY_AUTH=1): admins only, always private, the repo name chosen here (users never publish into the operator's org)."""
    if not _may_push():
        raise HTTPException(403, "Only an admin can upload to Hugging Face on this server.")
    hosted = auth.enabled()
    repo_asked, private = (None, True) if hosted else (req.repo, req.private)
    if repo_asked is not None and not REPO_RE.fullmatch(repo_asked):
        raise HTTPException(400, "Enter a repository name like account/model-name.")
    auth.check(model_of_run(req.run))
    res = S.TYPICALLY / "results" / req.run
    best = res / "best.pt"
    if not best.exists():
        raise HTTPException(404, "We can't find that model. It may have been archived.")
    ckpt = torch.load(best, map_location="cpu", weights_only=True)
    base = next((b for b, a in typically_job.RELEASED_ARGS.items() if a["backbone"] == ckpt["args"]["backbone"]), None)
    if base is None:
        raise HTTPException(400, "This model can't be uploaded because its standard model wasn't recognized.")
    reveal = json.loads((res / "reveal.json").read_text()) if (res / "reveal.json").exists() else None
    if reveal and reveal.get("base", "typical-small") != f"typical-{base}":   # scored against another base: the card must not claim it
        reveal = None
    token = _hf_token()
    if not token:
        raise HTTPException(503, "Uploading to Hugging Face isn't set up on this server.")
    api = HfApi(token=token)
    repo = repo_asked or (f"{ns}/{_repo_name(model_id_of(req.run))}" if (ns := _namespace(api, token)) else None)
    if not repo:
        raise HTTPException(502, "Uploading to Hugging Face isn't working right now. Ask your administrator to check the Hugging Face token.")
    prev = f.read_text().strip() if (f := _hf_repo_file(req.run)).exists() else None
    created = False
    try:
        manifest = {"base_repo": f"OzLabs/typical-{base}", "base_sha": api.model_info(f"OzLabs/typical-{base}").sha, "run": req.run,
                    "step": ckpt["step"], "best_pt_sha256": hashlib.sha256(best.read_bytes()).hexdigest()}
        existed = api.repo_exists(repo)
        if existed and repo != prev:   # only ever overwrite this model's own earlier push (a model named "Typical Small" must not replace the release)
            raise FileExistsError(repo)
        api.create_repo(repo, private=private, exist_ok=True)
        created = not existed
        api.create_commit(repo, commit_message=f"Upload {_display_name(model_id_of(req.run)) or req.run} from Decision Factory", operations=[
            CommitOperationAdd("best.pt", str(best)),
            CommitOperationAdd("README.md", _card(base, repo, model_id_of(req.run), ckpt["step"], reveal).encode()),
            CommitOperationAdd("MANIFEST.json", json.dumps(manifest, indent=2).encode())])
    except Exception as e:   # hub errors can echo request headers: never hand the token back
        if isinstance(e, FileExistsError):
            raise HTTPException(409, f"{repo} already exists on Hugging Face. Pick another name.")
        if created:   # this call made the repo and the upload failed: do not leave an empty repo behind
            try:
                if set(api.list_repo_files(repo)) <= {".gitattributes"}:
                    api.delete_repo(repo)
            except Exception:
                pass   # ponytail: best effort; the push error below is what the user needs
        if _storage_full(e) and repo.split("/")[0] == ORG:
            try:   # hosted: the repo is chosen here, so there is no other namespace to offer
                user = None if hosted else api.whoami()["name"]
            except Exception:
                user = None
            raise HTTPException(507, {"code": "org_storage_full", "repo": user and f"{user}/{repo.split('/')[1]}", "message":
                                      f"The {ORG} organization's private storage on Hugging Face is full."
                                      + (f" You can push to your own account ({user}) instead." if user else "")})
        S.log.warning("push %s to %s failed: %s: %s", req.run, repo, type(e).__name__, str(e).replace(token, "***")[:500])   # never the token
        raise HTTPException(502, {"code": "hf_failed", "message": "Hugging Face didn't accept the upload. Try again in a few minutes."})
    _hf_repo_file(req.run).write_text(repo)
    return {"url": f"https://huggingface.co/{repo}", "files": ["best.pt", "README.md", "MANIFEST.json"]}
