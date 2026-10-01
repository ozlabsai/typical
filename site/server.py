"""Typical site API: POST /api/decide over inference/typical, single-KV-encode per request
(mirrors gpu-runpod-full-experiment:demo/app.py's Batch tab -- native_kv_decide, not per-query
.choice/.score/.noul). Serves site/ as static files at "/".

Run: uv run uvicorn --app-dir site server:app --port 8787

Bind to 127.0.0.1 only (the uvicorn default; never --host 0.0.0.0): /api/typically/train rents GPUs and has no auth.
ponytail: auth + per-user quotas are required before this is ever public.
"""
import csv
import io
import json
import os
import re
import random
import shutil
import sys
import threading
import time
from collections import Counter
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
from typical.core import to_labels
from typical.native import native_kv_decide
import typically_analyze
import typically_job
import typically_plan
from typically_reveal import reveal
from typically_spike import build_from_plan, build_import, write

TYPICALLY = REPO_ROOT / ".context" / "typically"
REPOS = {"typical-small": "OzLabs/typical-small", "typical-medium": "OzLabs/typical-medium"}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8787", "http://127.0.0.1:8787"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _reconcile_pods():   # a crashed server may have left a rented pod running; off-thread so startup is not blocked
    threading.Thread(target=typically_job.reconcile, daemon=True).start()

# Multi-LoRA: every model (the base too) keeps only its LoRA + head; checkpoints with the same architecture share ONE Backbone,
# and Typical.activate() (called by get_model, under _lock) copies the wanted LoRA into it in place.
# ponytail: adapters are never evicted (~50 MB each on the CPU); add an LRU if one process ever serves hundreds of tuned models.
# name -> (best.pt mtime_ns, model): a retrain writes a new best.pt, and the stale entry is replaced (get_model runs under _lock).
_cache: dict[str, tuple[int, Typical]] = {}
_backbones: dict[tuple, object] = {}
# ponytail: one lock, one GPU -- concurrent forward passes on MPS from FastAPI's threadpool crash
# Metal (MTLCommandBufferStatusCommitted assertion); serialize instead of queueing.
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
        raise HTTPException(400, f"unknown model {name!r}")
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
        raise HTTPException(400, f"unknown model {tuned!r}")
    best = TYPICALLY / "results" / tuned.removeprefix("local:") / "best.pt"
    return _base_of(str(best), best.stat().st_mtime_ns)


class Query(BaseModel):
    type: str  # "choice" | "noul" | "score"
    question: str
    labels: list[str] = []


class DecideRequest(BaseModel):
    model: str = "typical-small"
    state: str
    queries: list[Query]


def _run(m: Typical, state: str, queries: list[Query]) -> dict:
    """One model, one state, all queries in a single KV encode. Caller must hold _lock."""
    qs = [(q.type, q.question, ["no", "yes"] if q.type == "noul" else q.labels) for q in queries]
    for typ, question, labels in qs:
        if len(labels) < 2:
            raise HTTPException(400, f"'{typ}' query {question!r} needs >=2 labels")

    t0 = time.perf_counter()
    # the device->host copy in to_labels must stay inside the lock too, or it races the next forward
    raws = native_kv_decide(m.head, m.model, state, [(q, l) for _, q, l in qs], max_state=m.max_state, max_suffix=2048)
    labelled = [to_labels(raw, labels) for (_, _, labels), raw in zip(qs, raws)]
    ms = (time.perf_counter() - t0) * 1000

    results = []
    for (typ, _, labels), (probs, p_null) in zip(qs, labelled):
        argmax = max(probs, key=probs.get)
        entry = {"probs": probs, "p_null": p_null, "argmax": argmax}
        if typ == "score":
            entry["expected"] = sum(i * probs[lv] for i, lv in enumerate(labels))
        results.append(entry)
    return {"results": results, "ms": ms, "device": m.device}


@app.post("/api/decide")
def decide(req: DecideRequest):
    with _lock:
        return {**_run(get_model(req.model), req.state, req.queries), "model": req.model}


class CompareRequest(BaseModel):
    state: str
    decisions: list[Query]
    models: list[str]


@app.post("/api/typically/compare")
def compare(req: CompareRequest):
    """models may hold the literal "base": the base of the tuned ("local:...") model in the same request. Results stay keyed as asked;
    each says which model ran, and `base` names the one used (null when "base" was not asked for)."""
    base = None
    if "base" in req.models:
        tuned = next((n for n in req.models if n.startswith("local:")), None)
        if tuned is None:
            raise HTTPException(400, '"base" needs a tuned (local:...) model in the same request')
        base = base_of(tuned)
    ran = {n: base if n == "base" else n for n in req.models}
    with _lock:
        return {"models": {n: {**_run(get_model(m), req.state, req.decisions), "model": m} for n, m in ran.items()}, "base": base}


@app.get("/api/typically/models")
def models():
    return {"base": ["typical-small"], "tuned": tuned_models()}


class CsvRequest(BaseModel):
    csv_text: str


class Decision(BaseModel):
    column: str
    question: str
    type: Literal["choice", "noul", "score"]


class BuildRequest(CsvRequest):
    base: Literal["small", "medium"] = "small"
    steps: Literal[200, 400, 800] = 400
    text_col: str
    decisions: list[Decision]
    name: str


def read_csv(csv_text: str) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    records = [{k: (v or "") for k, v in r.items() if k is not None} for r in reader]
    if not reader.fieldnames or not records:
        raise HTTPException(400, "the file has no rows")
    return reader.fieldnames, records


@app.post("/api/typically/preview")
def preview(req: CsvRequest):
    columns, records = read_csv(req.csv_text)
    uniques = {c: sorted({r[c].strip() for r in records}) for c in columns}
    # values: the short lists only (decision columns) -- the UI detects each column's type from them
    return {"columns": columns, "rows": records[:5], "n": len(records),
            "uniques": {c: len(u) for c, u in uniques.items()},
            "values": {c: u for c, u in uniques.items() if len(u) <= 20}}


def _dist(rows: list[dict]) -> dict:
    d: dict[str, Counter] = {}
    for r in rows:
        d.setdefault(r["task"].removeprefix("import_"), Counter())[r["candidates"][r["label"]]] += 1
    return {k: dict(v) for k, v in d.items()}


_job_lock = threading.Lock()   # ponytail: one rented GPU job at a time, in-process; a second server process would not see it
_job_slug: str | None = None   # the slug being trained; /build refuses it (the job snapshots the dataset when it starts)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class Enrich(BaseModel):
    balance: bool = True
    dedupe_soft: bool = True
    policy: dict[str, str] = {}
    synthetic: bool = False   # experimental: ignored (with a warning) when the server has no ANTHROPIC_API_KEY
    languages: list[str] = []   # experimental: ignored (with a warning) when the server has no ANTHROPIC_API_KEY


class Settings(BaseModel):
    steps: Literal[200, 400, 800] = 400   # Quick / Balanced / Thorough
    holdout: int = Field(20, ge=5, le=50)   # % of the cases held out to test the model
    seed: int = 0


class BuildPlanRequest(BaseModel):   # /build v2: a confirmed DatasetPlan over a table kept server-side (/analyze's records_token)
    records_token: str
    plan: dict
    name: str
    base: Literal["small", "medium"] = "small"
    enrich: Enrich = Enrich()
    settings: Settings = Settings()


def _train_command(slug: str, steps: int, base: str) -> str:
    # same flag builder the pod runs (typically_job.train_flags); here fed from the release table, there from runs/base/best.pt's own args
    return "uv run --no-sync python pcdm/train.py " + " ".join(typically_job.train_flags(slug, steps, f"co_{slug}", typically_job.RELEASED_ARGS[base], base))


@app.post("/api/typically/build")
def build(req: BuildPlanRequest | BuildRequest):
    slug = _slug(req.name)
    if slug == _job_slug:   # ponytail: check-then-write race is a few ms wide; a per-slug lock if builds ever run concurrently
        raise HTTPException(409, "that model is being taught right now; wait for it to finish before rebuilding its data")
    return _build_plan(req, slug) if isinstance(req, BuildPlanRequest) else _build_csv(req, slug)


def _build_plan(req: BuildPlanRequest, slug: str) -> dict:
    if not typically_job.SLUG_RE.fullmatch(slug):
        raise HTTPException(400, "need a model name (letters and digits, up to 40 characters)")
    records, plan = typically_analyze.load_upload(req.records_token), req.plan
    key = os.environ.get("ANTHROPIC_API_KEY")   # the operator's key only
    llm = None
    if key and (req.enrich.synthetic or req.enrich.languages):
        import anthropic
        llm = anthropic.Anthropic(api_key=key)
    try:
        if errs := typically_plan.validate_plan(plan, typically_plan.profile(records)):
            raise HTTPException(400, "the plan does not match the data: " + "; ".join(errs[:5]))
        split, stats = build_from_plan(records, plan, req.enrich.model_dump(), req.settings.model_dump(), random.Random(req.settings.seed), llm)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (KeyError, TypeError, AttributeError) as e:
        raise HTTPException(400, f"the plan is malformed: {e!r}")
    out = TYPICALLY / "jobs" / slug
    shutil.rmtree(out / "eval", ignore_errors=True)   # a rebuild must not keep the previous build's eval files (e.g. an old language)
    write(out, split)
    for name, body in (("plan", plan), ("enrich", req.enrich.model_dump()), ("settings", req.settings.model_dump()),
                       ("job", {"name": req.name, "base": req.base, "steps": req.settings.steps})):   # /train defaults from job.json
        (out / f"{name}.json").write_text(json.dumps(body, indent=1))
    return {"job": slug, "data_dir": f"data_co_{slug}", "splits": stats["rows"], "balance": stats["labels"],
            "command": _train_command(slug, req.settings.steps, req.base), "plan_saved": str(out / "plan.json"), "stats": stats}


def _build_csv(req: BuildRequest, slug: str) -> dict:
    """v1 body: one text column + decisions straight from a csv (kept for older clients)."""
    columns, records = read_csv(req.csv_text)
    if not typically_job.SLUG_RE.fullmatch(slug) or not req.decisions or req.text_col not in columns or any(d.column not in columns for d in req.decisions):
        raise HTTPException(400, "need a name, a text column and at least one decision column from the file")
    decisions = [d.model_dump() for d in req.decisions]
    try:
        raw = build_import(records, req.text_col, decisions, random.Random(0), balance=False)
        split = build_import(records, req.text_col, decisions, random.Random(0))
    except ValueError as e:
        raise HTTPException(400, str(e))
    write(TYPICALLY / "jobs" / slug, split)
    return {"job": slug, "data_dir": f"data_co_{slug}", "splits": {k: len(v) for k, v in split.items()},   # data_co_<slug>: the dir name train.py keys its bucket map on
            "balance": {"before": _dist(raw["train"]), "after": _dist(split["train"])}, "command": _train_command(slug, req.steps, req.base)}


def start_job(slug: str, base: str = "small", steps: int = 400) -> dict:
    """409 if a job is already running, else start `typically_job.run` in a daemon thread and return the queued status."""
    global _job_slug
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "another model is being taught right now; wait for it to finish")
    _job_slug = slug

    def end():
        global _job_slug
        _job_slug = None
        _job_lock.release()

    def work():
        try:
            typically_job.run(slug, typically_job.file_log(slug), base, steps)
        finally:
            end()
    try:
        status = typically_job.write_status(slug, "queued", "Waiting to start.")
        threading.Thread(target=work, daemon=True).start()
    except BaseException:
        end()
        raise
    return status


class TrainRequest(BaseModel):
    name: str
    base: Literal["small", "medium"] | None = None   # None: what /build saved in jobs/<slug>/job.json, else small
    steps: Literal[200, 400, 800] | None = None   # Quick / Balanced / Thorough; None: as base


@app.post("/api/typically/train")
def train(req: TrainRequest):
    slug = _slug(req.name)
    if not typically_job.SLUG_RE.fullmatch(slug) or not (TYPICALLY / "jobs" / slug / "train.jsonl").exists():
        raise HTTPException(404, "no dataset with that name; import and build it first")
    saved = json.loads(f.read_text()) if (f := TYPICALLY / "jobs" / slug / "job.json").exists() else {}
    return start_job(slug, req.base or saved.get("base", "small"), req.steps or saved.get("steps", 400))


def _held_out(path: Path) -> tuple[float, int, dict]:
    """eval_wf.py output -> (overall acc pooled over eval sets, rows, {decision: (acc, rows)})."""
    ev = [s for k, s in json.loads(path.read_text())["eval"].items() if k == "import_oneliner"]   # not the translated import_<lang> sets
    rows = sum(s["n"] for s in ev)
    return (sum(s["raw"]["acc"] * s["n"] for s in ev) / rows, rows,
            {f.removeprefix("import_"): (m["acc"], m["n"]) for s in ev for f, m in s["by_family"].items()})


def agreement(slug: str) -> dict:
    res = TYPICALLY / "results" / f"co_{slug}"
    (b_acc, _, b_dec), (y_acc, n, y_dec) = _held_out(res / "base_eval_co.json"), _held_out(res / "eval_co.json")
    return {"n": n, "overall": {"base": b_acc, "yours": y_acc},
            "decisions": {t: {"base": b_dec[t][0], "yours": a, "n": k} for t, (a, k) in y_dec.items() if t in b_dec}}


@app.get("/api/typically/train/{slug}")
def train_status(slug: str):
    st = typically_job.read_status(slug) if typically_job.SLUG_RE.fullmatch(slug) else None
    if st is None:
        raise HTTPException(404, "no such job")
    return {**st, "run": f"co_{slug}", "agreement": agreement(slug)} if st["phase"] == "done" else st


_reveal_lock = threading.Lock()
_reveals: dict[str, dict] = {}   # run -> {"done", "total", "error"} while its background pass runs
# ponytail: in-process progress only; a second server process would start its own pass


def _reveal_target(project: str) -> tuple[str, Path, str]:
    """project -> (run, eval file, company): "northwind" is the demo (co_f); anything else is a job slug (co_<slug>)."""
    if project == "northwind":
        return "co_f", REPO_ROOT / "data_co_a/eval/a_oneliner.jsonl", "Northwind Freight"
    if not typically_job.SLUG_RE.fullmatch(project):
        raise HTTPException(404, "no such project")
    return f"co_{project}", TYPICALLY / "jobs" / project / "eval" / "import_oneliner.jsonl", project.replace("_", " ").title()


@app.get("/api/typically/results/{project}")
def results(project: str):
    run, eval_file, company = _reveal_target(project)
    best = TYPICALLY / "results" / run / "best.pt"
    cache = best.with_name("reveal.json")
    if not best.exists() or not eval_file.exists():
        raise HTTPException(404, "that model is not trained yet")
    with _reveal_lock:
        if cache.exists() and cache.stat().st_mtime >= best.stat().st_mtime and (c := json.loads(cache.read_text())).get("base", "typical-small") == base_of(f"local:{run}"):   # caches from before "base" was recorded are small's
            return c
        job = _reveals.get(run)
        if job and job.get("error"):
            _reveals.pop(run)
            raise HTTPException(500, f"scoring failed: {job['error']}")
        if job is None:
            rows = [json.loads(line) for line in eval_file.open()]
            job = _reveals[run] = {"done": 0, "total": len({r["state"] for r in rows})}
            threading.Thread(target=_reveal_work, args=(run, rows, company, cache), daemon=True).start()
        return JSONResponse({"status": "computing", "done": job["done"], "total": job["total"]}, status_code=202)


def _reveal_work(run: str, rows: list[dict], company: str, cache: Path) -> None:
    job, tuned = _reveals[run], f"local:{run}"

    def run_models(state, group):   # one held-out case, the base it started from then tuned, through the same _run as /compare
        qs = [Query(type=r["meta"]["qtype"], question=r["query"], labels=r["candidates"]) for r in group]
        with _lock:
            out = [_run(get_model(n), state, qs)["results"] for n in (base, tuned)]
        job["done"] += 1
        return out
    try:
        base = base_of(tuned)
        res = {**reveal(rows, run_models, company, tuned), "base": base}
        cache.write_text(json.dumps(res))
    except BaseException as e:   # surfaced on the next poll instead of a poll that never finishes
        job["error"] = f"{type(e).__name__}: {e}"
        return
    with _reveal_lock:
        _reveals.pop(run)


@app.get("/api/typically/download/{run}")
def download(run: str):
    f = TYPICALLY / "results" / run / "best.pt"
    if not typically_job.SLUG_RE.fullmatch(run) or not f.exists():
        raise HTTPException(404, "no such model")
    return FileResponse(f, filename=f"{run}.pt")


@app.get("/api/health")
def health():
    return {"models": {n: m.device for n, (_, m) in _cache.items()}}


@app.get("/app")
@app.get("/app/{path:path}")
def spa(path: str = ""):   # the product app (typically-app/, built to site/app/): real files are served, anything else is a client route
    root = (REPO_ROOT / "site" / "app").resolve()
    f = (root / path).resolve()
    if f.is_file() and f.is_relative_to(root):
        return FileResponse(f)
    if not (root / "index.html").exists():
        raise HTTPException(404, "the app is not built; run `npm run build` in typically-app/")
    return FileResponse(root / "index.html")


import typically_deploy  # noqa: E402  (reads server._run / get_model / _lock at call time, so it goes after their definitions)
app.include_router(typically_deploy.router)
import typically_analyze  # noqa: E402
app.include_router(typically_analyze.router)
import typically_models  # noqa: E402
app.include_router(typically_models.router)

app.mount("/", StaticFiles(directory=REPO_ROOT / "site", html=True), name="site")
