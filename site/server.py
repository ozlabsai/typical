"""Typical site API: POST /api/decide over inference/typical, single-KV-encode per request
(mirrors gpu-runpod-full-experiment:demo/app.py's Batch tab -- native_kv_decide, not per-query
.choice/.score/.noul). Serves site/ as static files at "/".

Run: uv run uvicorn --app-dir site server:app --port 8787

Bind to 127.0.0.1 only (the uvicorn default) unless TYPICALLY_AUTH=1: /api/typically/teach rents GPUs. With TYPICALLY_AUTH=1
(site/typically_auth.py: invite sign-in, ownership, quotas) it can be hosted for a small invited audience; see deploy/README.md.
"""
import contextvars
import json
import logging
import os
import re
import random
import shutil
import sys
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in site/
sys.path.insert(0, str(REPO_ROOT / "inference"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ponytail: HF_TOKEN lives in .env, not the shell env -- same three-line loader as demo/app.py,
# not worth a python-dotenv dependency for.
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from typical import Typical
from typical.native import native_kv_decide
import typically_analyze
import typically_auth as auth
import typically_job
import typically_llm
import typically_plan
from typically_reveal import company, queries, reveal, score
from typically_spike import build_from_plan, write

TYPICALLY = typically_job.TYPICALLY   # TYPICALLY_DATA, else .context/typically
REPOS = {"typical-small": "OzLabs/typical-small", "typical-medium": "OzLabs/typical-medium"}

log = logging.getLogger("uvicorn.error")   # raw errors go to the server log; users get a plain message + a stable code


@asynccontextmanager
async def lifespan(app):
    """Rented pods never outlive the server: reconcile at start and every RECONCILE_S (orphans past the heartbeat window: pods deleted,
    statuses failed); at shutdown stop every job of this process and delete its pod (bounded wait; uvicorn waits for this)."""
    threading.Thread(target=typically_job.reconcile_forever, args=(log.info,), daemon=True).start()
    yield
    typically_job.shutdown(log=log.warning)


app = FastAPI(lifespan=lifespan, **({"docs_url": None, "redoc_url": None, "openapi_url": None} if auth.enabled() else {}))   # hosted: no open API map
if auth.enabled():
    auth._secret()   # fail at start, not at the first sign-in
else:   # hosted = same-origin only: no CORS at all
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8787", "http://127.0.0.1:8787"], allow_methods=["*"], allow_headers=["*"])
app.middleware("http")(auth.middleware)


# Multi-LoRA: every model (the base too) keeps only its LoRA + head; checkpoints with the same architecture share ONE Backbone,
# and Typical.activate() (called by get_model, under _lock) copies the wanted LoRA into it in place.
# ponytail: adapters are never evicted (~50 MB each on the CPU); add an LRU if one process ever serves hundreds of tuned models.
# name -> (best.pt mtime_ns, model): a retrain writes a new best.pt, and the stale entry is replaced (get_model runs under _lock).
_cache: dict[str, tuple[int, Typical]] = {}
_backbones: dict[tuple, object] = {}
# ponytail: one lock, one GPU -- concurrent forward passes on MPS from FastAPI's threadpool crash
# Metal (MTLCommandBufferStatusCommitted assertion); serialize instead of queueing. A base's first load downloads it from the Hub
# (and its Qwen backbone, GBs) under this lock, stalling every request meanwhile; upgrade: prefetch with snapshot_download before taking it.
_lock = threading.Lock()


def _warm(m: Typical) -> None:
    """First MPS call on a new shape is slow (batch of 3: 866 ms cold -> 105 ms warm; see
    content-spec.md 'Notes for infra/site workers'). Run one 3-query batch now so the first
    real request isn't the one that pays for it."""
    native_kv_decide(
        m.head, m.model, "warmup state.",
        [("a?", ["x", "y"]), ("b?", ["x", "y"]), ("c?", ["x", "y"])],
        max_state=m.max_state, max_suffix=2048,
    )


def tuned_models() -> list[str]:
    return sorted(f"local:{p.parent.name}" for p in (TYPICALLY / "results").glob("*/best.pt"))


def get_model(name: str) -> Typical:
    name = name or "typical-small"
    if name in REPOS:
        source, stamp = REPOS[name], 0
    elif name in tuned_models():
        best = TYPICALLY / "results" / name.removeprefix("local:") / "best.pt"
        source, stamp = str(best), best.stat().st_mtime_ns
    else:
        raise HTTPException(400, f"We can't find the model {name}.")
    if _cache.get(name, (None,))[0] != stamp:   # new, or retrained since it was loaded: the old entry is dropped
        model = Typical.from_pretrained(source, device="auto", backbones=_backbones)
        _warm(model)
        _cache[name] = (stamp, model)
    _cache[name][1].activate()
    return _cache[name][1]


@lru_cache(maxsize=None)
def _base_of(path: str, mtime_ns: int) -> str:
    backbone = torch.load(path, map_location="cpu", weights_only=True)["args"]["backbone"]
    if (base := next((b for b, a in typically_job.RELEASED_ARGS.items() if a["backbone"] == backbone), None)) is None:
        raise HTTPException(400, f"unknown backbone {backbone!r}")
    return f"typical-{base}"


def base_of(tuned: str) -> str:
    """The released model a tuned model ("local:<run>") was fine-tuned from, e.g. "typical-medium": the baseline to compare it against."""
    if tuned not in tuned_models():
        raise HTTPException(400, f"We can't find the model {tuned}.")
    best = TYPICALLY / "results" / tuned.removeprefix("local:") / "best.pt"
    return _base_of(str(best), best.stat().st_mtime_ns)


class Query(BaseModel):
    type: str  # "choice" | "noul" | "score"
    question: str = Field(max_length=2000)
    labels: list[str] = Field([], max_length=100)


class DecideRequest(BaseModel):   # also the public /v1/models/{id}/decide body: bounded like /ask
    model: str = "typical-small"
    state: str = Field(max_length=20_000)
    queries: list[Query] = Field(max_length=20)


def _run(m: Typical, state: str, queries: list[Query]) -> dict:
    """One model, one state, all queries in a single KV encode (typically_reveal.score, which the GPU pod's reveal runs too). Caller must hold _lock."""
    t0 = time.perf_counter()
    try:
        results = score(m, state, [(q.type, q.question, q.labels) for q in queries])
    except ValueError as e:   # a query with < 2 labels
        raise HTTPException(400, str(e))
    return {"results": results, "ms": (time.perf_counter() - t0) * 1000, "device": m.device}


@app.post("/api/decide")
def decide(req: DecideRequest):
    auth.check(auth.model_of(req.model))
    with _lock:
        return {**_run(get_model(req.model), req.state, req.queries), "model": req.model}


class CompareRequest(BaseModel):
    state: str = Field(max_length=20_000)
    decisions: list[Query] = Field(max_length=20)
    models: list[str] = Field(max_length=3)


@app.post("/api/typically/compare")
def compare(req: CompareRequest):
    """models may hold the literal "base": the base of the tuned ("local:...") model in the same request. Results stay keyed as asked;
    each says which model ran, and `base` names the one used (null when "base" was not asked for)."""
    for n in req.models:
        if n != "base":
            auth.check(auth.model_of(n))
    base = None
    if "base" in req.models:
        tuned = next((n for n in req.models if n.startswith("local:")), None)
        if tuned is None:
            raise HTTPException(400, '"base" needs a tuned (local:...) model in the same request')
        base = base_of(tuned)
    ran = {n: base if n == "base" else n for n in req.models}
    with _lock:
        return {"models": {n: {**_run(get_model(m), req.state, req.decisions), "model": m} for n, m in ran.items()}, "base": base}


_jobs_lock = threading.Lock()
_jobs = typically_job.RUNNING   # slug -> owner of every job running in this process; /teach refuses these slugs (the job snapshots the
# dataset when it starts). ponytail: in-process; a second server process would not see them


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class Enrich(BaseModel):
    balance: bool = True
    dedupe_soft: bool = True
    policy: dict[str, str] = {}
    synthetic: bool = False   # experimental, needs the server's LLM key (/teach fails the build without it, or when every call failed)
    languages: list[str] = Field([], max_length=3)   # experimental, as synthetic; capped: each language is ~30% more LLM calls on the operator's key


class Settings(BaseModel):
    steps: Literal[200, 400, 800] = 400   # Quick / Balanced / Thorough
    holdout: int = Field(20, ge=5, le=50)   # % of the cases held out to test the model
    seed: int = 0


class BuildPlanRequest(BaseModel):   # /teach: a confirmed DatasetPlan over a table kept server-side (/analyze's records_token)
    records_token: str
    plan: dict
    name: str = Field(max_length=80)
    base: Literal["small", "medium"] = "small"
    enrich: Enrich = Enrich()
    settings: Settings = Settings()


def _check_name(slug: str) -> None:
    if slug in auth.SHARED_MODELS or f"co_{slug}" in typically_deploy.ALIASES.values():   # "Northwind" / "f" would write over the shared sample
        raise HTTPException(409, "That name is reserved. Pick another.")
    if slug in _jobs:   # ponytail: check-then-write race is a few ms wide; a per-slug lock if builds ever run concurrently
        raise HTTPException(409, "This model is training right now. Wait for it to finish before changing its data.")
    if slug and (TYPICALLY / "jobs" / slug).exists() and not auth.visible(slug):
        raise HTTPException(409, "That name is taken. Pick another.")


def _plan_records(req: BuildPlanRequest, slug: str) -> list[dict]:
    """The fast checks of a plan build (name, upload, plan vs data), before any enrichment call."""
    if not typically_job.SLUG_RE.fullmatch(slug):
        raise HTTPException(400, "Give the model a name: letters and digits, up to 40 characters.")
    records = typically_analyze.load_upload(req.records_token)
    try:
        if errs := typically_plan.validate_plan(req.plan, typically_plan.profile(records)):
            raise HTTPException(400, "These settings don't match your data (" + "; ".join(errs[:5]) + "). Check the questions in the Understand step.")
    except (KeyError, TypeError, AttributeError) as e:
        raise HTTPException(400, "Something went wrong reading these settings. Go back to the Understand step and try again.")
    return records


def _save_meta(req: BuildPlanRequest, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, body in (("plan", req.plan), ("enrich", req.enrich.model_dump()), ("settings", req.settings.model_dump()),
                       ("job", {"name": req.name, "base": req.base, "steps": req.settings.steps, "records_token": req.records_token, **auth.stamp()})):   # retrain defaults from job.json; the token links the model to its dataset
        (out / f"{name}.json").write_text(json.dumps(body, indent=1))


def _build_plan(req: BuildPlanRequest, slug: str, records: list[dict], progress=None) -> dict:
    """The dataset of jobs/<slug>/ from the plan (AI enrichment included); returns build_from_plan's stats."""
    plan = req.plan
    llm = typically_llm.complete_json if typically_llm.available() and (req.enrich.synthetic or req.enrich.languages) else None   # the operator's key only
    try:
        split, stats = build_from_plan(records, plan, req.enrich.model_dump(), req.settings.model_dump(), random.Random(req.settings.seed), llm, progress)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (KeyError, TypeError, AttributeError) as e:
        raise HTTPException(400, "Something went wrong reading these settings. Go back to the Understand step and try again.")
    out = TYPICALLY / "jobs" / slug
    shutil.rmtree(out / "eval", ignore_errors=True)   # a rebuild must not keep the previous build's eval files (e.g. an old language)
    write(out, split)
    _save_meta(req, out)
    return stats


def start_job(slug: str, base: str = "small", steps: int = 400, prepare=None) -> dict:
    """409/429 unless auth.admit lets it start (this slug not running, TYPICALLY_MAX_JOBS, the user's quotas), else start
    `typically_job.run` in a daemon thread and return the queued status. prepare(): builds the dataset first, in the same
    thread (its failure fails the job with code failed_build)."""
    with _jobs_lock:
        auth.admit(slug, _jobs)
        _jobs[slug] = auth.stamp().get("owner")
        typically_job._STOP.discard(slug)   # a Stop that raced the previous run's end must not abort this one

    def end():
        with _jobs_lock:
            _jobs.pop(slug, None)
        typically_job._STOP.discard(slug)

    def work():
        try:
            if prepare:
                try:
                    prepare()
                    typically_job.check_stop(slug)   # stopped while preparing: never rent the GPU
                except typically_job.Stopped:
                    typically_job.write_status(slug, "failed", "You stopped this training run before it reached a GPU. Nothing is left running.",
                                               "stopped", events=[typically_job.event("failed")])
                    return
                except Exception as e:   # HTTPException (bad plan / data) or anything else: the card says why, nothing was rented
                    if not isinstance(e, HTTPException):
                        log.exception("preparing %s failed", slug)   # raw text in the server log only
                    why = e.detail if isinstance(e, HTTPException) else "Something went wrong on the server."
                    typically_job.write_status(slug, "failed", why, "failed_build",
                                               events=[typically_job.event("failed")])
                    return
            typically_job.run(slug, typically_job.file_log(slug), base, steps, **({"prepared": True} if prepare else {}))
        finally:
            end()
    try:   # the thread keeps the request's context (auth.stamp() in prepare's job.json)
        status = typically_job.write_status(slug, "queued", "Preparing your cases." if prepare else "Waiting to start.",
                                            "preparing" if prepare else None, events=[typically_job.event("prepare")] if prepare else ())
        threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True).start()
    except BaseException:
        end()
        raise
    return status


def train_slug(slug: str, base: str | None = None, steps: int | None = None) -> dict:
    """Start training a built dataset (jobs/<slug>/); base/steps default to its job.json. /retrain goes through here."""
    if not typically_job.SLUG_RE.fullmatch(slug) or not (TYPICALLY / "jobs" / slug / "train.jsonl").exists() or not auth.visible(slug):
        raise HTTPException(404, "We can't find the data for this model. Start a new model from your data.")
    saved = json.loads(f.read_text()) if (f := TYPICALLY / "jobs" / slug / "job.json").exists() else {}
    return start_job(slug, base or saved.get("base", "small"), steps or saved.get("steps", 400))


@app.post("/api/typically/train/{slug}/stop")
def stop(slug: str):
    """Ask a running job to stop: it raises at its next GPU-provider call (seconds; up to a few minutes mid upload / download) or
    AI enrichment call, and deletes the GPU in its finally, as on any failure."""
    if not typically_job.SLUG_RE.fullmatch(slug) or not auth.visible(slug) or slug not in _jobs:
        raise HTTPException(404, "This model isn't training right now.")
    typically_job.stop(slug)
    return {"stopping": slug}


@app.post("/api/typically/teach")
def teach(req: BuildPlanRequest):
    """Build the dataset + train in one call that returns at once: the job's first milestone builds the dataset (AI enrichment
    included), with progress on the same live card. Every new model goes through here (and so through auth.admit)."""
    slug = _slug(req.name)
    _check_name(slug)
    records = _plan_records(req, slug)

    def progress(kind: str, done: int, total: int):
        typically_job.check_stop(slug)   # Stop during AI enrichment: build_from_plan drops the queued calls
        typically_job.add_events(slug, typically_job.event(kind, n=done, of=total))

    def prepare():
        _save_meta(req, TYPICALLY / "jobs" / slug)   # first: the model page has its name and decisions while the examples are prepared
        stats = _build_plan(req, slug, records, progress)
        if (req.enrich.synthetic or req.enrich.languages) and (not typically_llm.available() or stats["llm_failed"] == stats["llm_calls"] > 0):
            raise HTTPException(502, "AI enrichment failed for every case, so training didn't start. Turn it off or try again.")
        typically_job.add_events(slug, typically_job.event("prepare", n=stats["rows"]["train"], synthetic=stats["synthetic"],
                                                           translated=stats["translated"]["train"], failed=stats["llm_failed"]))
    return {**start_job(slug, req.base, req.settings.steps, prepare), "job": slug}


@app.get("/api/typically/train/{slug}")
def train_status(slug: str):
    st = typically_job.read_status(slug) if typically_job.SLUG_RE.fullmatch(slug) and auth.visible(slug) else None
    if st is None:
        raise HTTPException(404, "We can't find that training run.")
    if st["phase"] not in ("queued", "starting_gpu", "uploading"):   # the curves stay on the card once training is over
        st["series"] = typically_job.log_series(typically_job.job_dir(slug) / "pod.log")
    return {**st, "run": f"co_{slug}"} if st["phase"] == "done" else st


_reveal_lock = threading.Lock()
_reveals: dict[str, dict] = {}   # run -> {"done", "total", "error"} while its background pass runs
# ponytail: in-process progress only; a second server process would start its own pass


def _reveal_target(project: str) -> tuple[str, Path, str]:
    """project -> (run, eval file, company): "northwind" is the demo (co_f); anything else is a job slug (co_<slug>)."""
    if project == "northwind":
        return "co_f", REPO_ROOT / "data_co_a/eval/a_oneliner.jsonl", "Northwind Freight"
    if not typically_job.SLUG_RE.fullmatch(project) or not auth.visible(project):
        raise HTTPException(404, "We can't find that model.")
    return f"co_{project}", TYPICALLY / "jobs" / project / "eval" / "import_oneliner.jsonl", company(project)


@app.get("/api/typically/results/{project}")
def results(project: str):
    run, eval_file, company = _reveal_target(project)
    best = TYPICALLY / "results" / run / "best.pt"
    cache = best.with_name("reveal.json")
    if not best.exists() or not eval_file.exists():
        raise HTTPException(404, "That model isn't trained yet.")
    with _reveal_lock:
        if cache.exists() and cache.stat().st_mtime >= best.stat().st_mtime and (c := json.loads(cache.read_text())).get("base", "typical-small") == base_of(f"local:{run}"):   # caches from before "base" was recorded are small's
            return c
        job = _reveals.get(run)
        if job and job.get("error"):
            _reveals.pop(run)
            raise HTTPException(500, "We couldn't score this model. Reload the page to try again.")   # the cause is in the server log
        if job is None:
            rows = [json.loads(line) for line in eval_file.open()]
            job = _reveals[run] = {"done": 0, "total": len({r["state"] for r in rows})}
            threading.Thread(target=_reveal_work, args=(run, rows, company, cache), daemon=True).start()
        return JSONResponse({"status": "computing", "done": job["done"], "total": job["total"]}, status_code=202)


def _reveal_work(run: str, rows: list[dict], company: str, cache: Path) -> None:
    job, tuned = _reveals[run], f"local:{run}"

    def run_models(state, group):   # one held-out case, the base it started from then tuned, through the same score as /compare
        with _lock:
            out = [score(get_model(n), state, queries(group)) for n in (base, tuned)]
        job["done"] += 1
        return out
    try:
        base = base_of(tuned)
        res = {**reveal(rows, run_models, company, tuned), "base": base}
        cache.write_text(json.dumps(res))
    except BaseException as e:   # surfaced on the next poll instead of a poll that never finishes
        log.exception("scoring %s failed", run)
        job["error"] = f"{type(e).__name__}: {e}"
        return
    with _reveal_lock:
        _reveals.pop(run)


@app.get("/api/typically/download/{run}")
def download(run: str):
    mid = typically_deploy.model_of_run(run)   # 404 unless co_<slug> / co_f
    if not (f := TYPICALLY / "results" / run / "best.pt").exists() or not auth.visible(mid):
        raise HTTPException(404, "We can't find that model. It may have been archived.")
    return FileResponse(f, filename=typically_deploy.download_name(run))


@app.get("/api/health")
def health():
    return {"ok": True, "loaded": len(_cache)}   # no model ids: it is open without sign-in


@app.get("/app")
@app.get("/app/{path:path}")
def spa(path: str = ""):   # the product app (typically-app/, built to site/app/): real files are served, anything else is a client route
    root = (REPO_ROOT / "site" / "app").resolve()
    f = (root / path).resolve()
    if f.is_file() and f.is_relative_to(root):
        return FileResponse(f)
    if not (root / "index.html").exists():
        raise HTTPException(404, "The app isn't built. Run `npm run build` in typically-app/.")
    return FileResponse(root / "index.html")


import typically_deploy  # noqa: E402  (reads server._run / get_model / _lock at call time, so it goes after their definitions)
app.include_router(typically_deploy.router)
import typically_analyze  # noqa: E402
app.include_router(typically_analyze.router)
import typically_models  # noqa: E402
app.include_router(typically_models.router)
import typically_corrections  # noqa: E402
app.include_router(typically_corrections.router)
import typically_share  # noqa: E402
app.include_router(typically_share.router)
app.include_router(auth.router)

app.mount("/", StaticFiles(directory=REPO_ROOT / "site", html=True), name="site")
