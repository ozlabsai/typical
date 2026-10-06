"""The corrections loop: "Wrong?" in chat stores the right answer for a case; /retrain teaches a new version on the original data plus
those corrections. Store: .context/typically/corrections/<model id>.jsonl, one correction per line, oldest first (n = line index).
Reads server.TYPICALLY / _job_lock / train_slug at call time, so it is included at the end of server.py."""
import json
import random
import shutil
import threading
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import server as S
import typically_auth as auth
import typically_job
import typically_models as M
import typically_plan as tp
from typically_spike import build_from_plan, row, write

router = APIRouter()
_lock = threading.Lock()   # ponytail: one process-wide lock for every corrections file; per-file locks if this ever has real traffic
REPEAT = 3   # each correction goes into train 3x: a handful of fixes must outweigh ~1k original rows without drowning them


class Correction(BaseModel):
    case: str = Field(min_length=1, max_length=20_000)
    question: str = Field(min_length=1)
    type: Literal["choice", "noul", "score"]
    labels: list[str] = []
    answer: str   # the correct label
    model_answer: str | None = None
    p: float | None = None


def _check(model_id: str):
    if not typically_job.SLUG_RE.fullmatch(model_id) or not (model_id == "northwind" or (S.TYPICALLY / "jobs" / model_id).is_dir()):
        raise HTTPException(404, "no such model")
    auth.check(model_id)


def read(model_id: str) -> list[dict]:
    f = M.corrections_file(model_id)
    return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []


def _write(model_id: str, items: list[dict]):
    (f := M.corrections_file(model_id)).parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in items))


@router.post("/api/typically/models/{model_id}/corrections")
def add(model_id: str, c: Correction):
    _check(model_id)
    labels = ["no", "yes"] if c.type == "noul" else c.labels
    if len(labels) < 2 or c.answer not in labels:
        raise HTTPException(400, "the answer must be one of the question's options")
    with _lock:   # the same (case, question) again replaces the old correction: the latest answer wins
        items = [x for x in read(model_id) if (x["case"], x["question"]) != (c.case, c.question)]
        items.append({**c.model_dump(), "labels": labels, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        _write(model_id, items)
    return {"count": len(items), "n": len(items) - 1}


@router.get("/api/typically/models/{model_id}/corrections")
def list_(model_id: str):
    _check(model_id)
    items = read(model_id)
    return {"count": len(items), "items": [{**x, "n": i} for i, x in reversed(list(enumerate(items)))][:200]}


@router.delete("/api/typically/models/{model_id}/corrections/{n}")
def remove(model_id: str, n: int):
    _check(model_id)
    with _lock:
        items = read(model_id)
        if not 0 <= n < len(items):
            raise HTTPException(404, "no such correction")
        del items[n]
        _write(model_id, items)
    return {"count": len(items)}


# ---------------------------------------------------------------- retrain

def _job(model_id: str) -> dict:
    return M._json(S.TYPICALLY / "jobs" / model_id / "job.json") or {}


def _northwind(out):
    """The sample's dataset, built the way /build v2 builds an analyzed table (heuristic plan, default enrich and settings),
    asked with the questions chat uses for Northwind."""
    records = tp.load_records({"kind": "sample"})
    plan = tp.heuristic_plan(tp.profile(records), records)
    questions = {d["column"]: d["question"] for d in M.NORTHWIND_DECISIONS}
    for d in plan["decisions"]:
        d["question"] = questions.get(d["column"], d["question"])
    split, _ = build_from_plan(records, plan, {}, {}, random.Random(0))
    write(out, split)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))


class RetrainRequest(BaseModel):
    steps: Literal[200, 400, 800] | None = None   # None: the parent's


@router.post("/api/typically/models/{model_id}/retrain")
def retrain(model_id: str, req: RetrainRequest = RetrainRequest()):
    """A new version <root>_v<N>: the parent's dataset as it was built (train/val/eval files copied, so eval is byte-identical and the
    scores compare) plus every correction as extra train rows. ponytail: copies the built split instead of re-running /build, because
    the job dir does not keep the upload; a v3 inherits v2's corrections through v2's train.jsonl."""
    _check(model_id)   # quotas / busy GPUs: S.train_slug -> start_job refuses, and the half-built version is removed below
    corrections = read(model_id)
    if not corrections:
        raise HTTPException(400, "no corrections to learn from yet")
    parent = S.TYPICALLY / "jobs" / model_id
    if model_id != "northwind" and not (parent / "train.jsonl").exists():
        raise HTTPException(400, "this model's training data is missing")
    root = model_id
    while up := _job(root).get("parent"):
        root = up
    n = 2
    while (S.TYPICALLY / "jobs" / f"{root}_v{n}").exists():
        n += 1
    slug = f"{root}_v{n}"
    if not typically_job.SLUG_RE.fullmatch(slug):
        raise HTTPException(400, "the model name is too long for another version")
    meta = M.NORTHWIND if model_id == "northwind" else _job(model_id)
    root_name = M.NORTHWIND["name"] if root == "northwind" else _job(root).get("name") or root

    out = S.TYPICALLY / "jobs" / slug
    out.mkdir(parents=True)
    try:
        if model_id == "northwind":
            _northwind(out)
        else:
            shutil.copytree(parent / "eval", out / "eval")
            for name in ("train.jsonl", "val.jsonl", "plan.json", "enrich.json", "settings.json"):
                if (parent / name).exists():
                    shutil.copy2(parent / name, out / name)
        # a correction on a held-out case (chat's "Try an example" cases are eval cases) would leak it into train: skipped, counted
        held = {tp.norm(json.loads(line)["state"]) for f in [out / "val.jsonl", *(out / "eval").glob("*.jsonl")] for line in f.open()}
        used = [c for c in corrections if tp.norm(c["case"]) not in held]
        if not used:
            raise HTTPException(400, "every correction is on a held-out test case; correct some new cases first")
        with open(out / "train.jsonl", "a") as f:
            for c in used:
                r = row(c["case"], c["question"], c["labels"], c["answer"], c["type"], "import_correction", correction=True)
                f.write((json.dumps(r) + "\n") * REPEAT)
        steps = req.steps or meta.get("steps") or 400
        lineage = {"parent": model_id, "version": n, "corrections": len(used), "held_out_skipped": len(corrections) - len(used)}
        (out / "job.json").write_text(json.dumps({"name": f"{root_name} v{n}", "base": meta.get("base") or "small", "steps": steps, **lineage, **auth.stamp()}, indent=1))
        status = S.train_slug(slug, steps=steps)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)   # nothing half-built is left to show up in the library
        raise
    return {"id": slug, "name": f"{root_name} v{n}", **lineage, "status": status}
