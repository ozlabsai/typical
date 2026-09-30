"""Typical site API: POST /api/decide over inference/typical, single-KV-encode per request
(mirrors gpu-runpod-full-experiment:demo/app.py's Batch tab -- native_kv_decide, not per-query
.choice/.score/.noul). Serves site/ as static files at "/".

Run: uv run uvicorn --app-dir site server:app --port 8787
"""
import csv
import io
import os
import re
import random
import sys
import threading
import time
from collections import Counter
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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from typical import Typical
from typical.core import to_labels
from typical.native import native_kv_decide
from typically_spike import build_import, write

TYPICALLY = REPO_ROOT / ".context" / "typically"
REPOS = {"typical-small": "OzLabs/typical-small", "typical-medium": "OzLabs/typical-medium"}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ponytail: at most two models in memory -- the base plus one tuned; loading another of the same kind
# evicts the old one. A real LRU only matters if the site ever serves many tuned models at once.
_cache: dict[str, Typical] = {}
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
    if name not in _cache:
        if name in REPOS:
            source = REPOS[name]
        elif name in tuned_models():
            source = str(TYPICALLY / "results" / name.removeprefix("local:") / "best.pt")
        else:
            raise HTTPException(400, f"unknown model {name!r}")
        model = Typical.from_pretrained(source, device="auto")
        _warm(model)
        for k in [k for k in _cache if k.startswith("local:") == name.startswith("local:")]:
            del _cache[k]
        _cache[name] = model
    return _cache[name]


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
    with _lock:
        return {"models": {n: {**_run(get_model(n), req.state, req.decisions), "model": n} for n in req.models}}


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


@app.post("/api/typically/build")
def build(req: BuildRequest):
    columns, records = read_csv(req.csv_text)
    slug = re.sub(r"[^a-z0-9]+", "_", req.name.lower()).strip("_")
    if not slug or not req.decisions or req.text_col not in columns or any(d.column not in columns for d in req.decisions):
        raise HTTPException(400, "need a name, a text column and at least one decision column from the file")
    decisions = [d.model_dump() for d in req.decisions]
    try:
        raw = build_import(records, req.text_col, decisions, random.Random(0), balance=False)
        split = build_import(records, req.text_col, decisions, random.Random(0))
    except ValueError as e:
        raise HTTPException(400, str(e))
    write(TYPICALLY / "jobs" / slug, split)
    d = f"data_co_{slug}"   # the dir name train.py keys its bucket map on: rename the job dir to this on the GPU box
    cmd = (f"uv run --no-sync python pcdm/train.py --name co_{slug} --init_from runs/base/best.pt "
           "--readout native --nc_head n3 --nc_render semif --null factored --noul_head bern --score_head choice "
           "--ordinal_smooth 0.7 --tap_layer 20 --zscore --lora_r 16 --lora_layers 8 --max_state 1024 "
           # ponytail: flag list copied from scripts/typically_spike_pod.sh train(); keep the two in sync
           f"--data data_v5 --extra_data \"data_wf,data_wh,data_u,{d}\" --bucket_map \"data_wh=W,data_u=U,{d}=C\" "
           "--family_weights C:0.5,W:0.3,E:0.15,U:0.05 --null_aug W:0.20 "
           "--steps 400 --bs 64 --grad_accum 8 --val_every 50 --ckpt_every 100 --eval_every 400 --eval_limit 200 --eval_bs 8 "
           f"--best_on \"{d}_val\"")
    return {"job": slug, "data_dir": d, "splits": {k: len(v) for k, v in split.items()},
            "balance": {"before": _dist(raw["train"]), "after": _dist(split["train"])}, "command": cmd}


@app.get("/api/health")
def health():
    return {"models": {n: m.device for n, m in _cache.items()}}


app.mount("/", StaticFiles(directory=REPO_ROOT / "site", html=True), name="site")
