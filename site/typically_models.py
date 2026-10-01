"""Models library (base + fine-tunes) and the chat endpoints: /ask (one case, free-form questions) and /parse_question.
Reads server.TYPICALLY / _lock / get_model / base_of / _run at call time, so it is included at the end of server.py."""
import json
import re
import shutil
import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import server as S
import typically_auth as auth
import typically_deploy as D
import typically_job
import typically_llm as tllm
import typically_plan as tp

router = APIRouter()

BASES = {
    "typical-small": {"base": "small", "params": "1.7B", "description": "Fast and light. The default base model: a good fit for most routing and triage decisions."},
    "typical-medium": {"base": "medium", "params": "4B", "description": "Higher accuracy on harder cases, a little slower. The base for demanding decisions."},
}
# the Northwind sample: mirrors SAMPLE in typically-app/src/lib/project.ts (questions + tryCase), co_f's trained decisions
NORTHWIND = {"id": "northwind", "name": "Northwind triage", "base": "small", "steps": 400}
NORTHWIND_DECISIONS = [
    {"column": "team", "question": "Which team should handle this ticket?", "type": "choice", "labels": ["Claims", "Billing", "Dispatch", "Sales"]},
    {"column": "escalate", "question": "Should we escalate this to a manager?", "type": "noul", "labels": ["no", "yes"]},
    {"column": "urgency", "question": "How urgent is this ticket?", "type": "score", "labels": ["0", "1", "2", "3"]},
    {"column": "refund", "question": "Can the agent refund this without approval?", "type": "noul", "labels": ["no", "yes"]},
]
NORTHWIND_CASE = (
    "Customer: Cedar Foods (enterprise plan, US). Tickets from this customer in the last 30 days: 1. Shipment value: $450, shipped 4 days ago.\n\n"
    "Hi, two cases of glassware arrived cracked this morning. Photos attached. Can you sort out a replacement?")


def _json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _states(path, n: int) -> list[str]:
    """First n distinct `state`s of a jsonl eval file (each state repeats once per question)."""
    seen: dict[str, None] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            seen.setdefault(json.loads(line)["state"])
            if len(seen) == n:
                break
    return list(seen)


def _acc(path) -> tuple[float, int] | None:
    d = _json(path)
    r = d and d["eval"].get("import_oneliner")
    return (r["raw"]["acc_k"], r["n"]) if r else None


def _metrics(run: str):
    res = S.TYPICALLY / "results" / run
    if rev := _json(res / "reveal.json"):
        return {"standard": rev["score"]["standard"], "yours": rev["score"]["yours"], "n_cases": rev["n_cases"]}
    if (yours := _acc(res / "eval_co.json")) and (std := _acc(res / "base_eval_co.json")):
        return {"standard": std[0], "yours": yours[0], "n_cases": yours[1]}
    return None


def corrections_file(mid: str):   # chat's "Wrong?" store (typically_corrections.py); a shared model keeps one file per user
    uid = auth.stamp().get("owner") if mid in auth.SHARED_MODELS else None
    return S.TYPICALLY / "corrections" / (f"{mid}.{uid}.jsonl" if uid else f"{mid}.jsonl")


def _entry(mid: str, name: str, base: str, steps, created_at, status: dict, decisions: list, sample: bool = False, job: dict | None = None) -> dict:
    run, job, cf = D.run_of(mid), job or {}, corrections_file(mid)
    hf = S.TYPICALLY / "results" / run / "hf_repo.txt"
    return {"id": mid, "name": name, "kind": "custom", "base": base, "steps": steps, "created_at": created_at, "run": run, "sample": sample,
            "version": job.get("version", 1), "parent": job.get("parent"), "corrections": len(cf.read_text().splitlines()) if cf.exists() else 0,
            **status, "decisions": decisions, "metrics": _metrics(run),
            "has_key": any(k["model_id"] == mid and auth.can_see(k.get("owner")) for k in D._read_keys().values()),
            "hf_repo": hf.read_text().strip() if hf.exists() else None}


def _job_entry(d) -> dict | None:
    st, plan = _json(d / "status.json"), _json(d / "plan.json")
    if st is None and plan is None:   # test / junk dirs
        return None
    job = _json(d / "job.json") or _json(d / "settings.json") or {}
    phase, run = (st or {}).get("phase", "queued"), f"co_{d.name}"
    if phase == "failed" or (phase == "done" and not (S.TYPICALLY / "results" / run / "best.pt").exists()):
        status = {"status": "failed", "message": (st or {}).get("message"), "code": (st or {}).get("code")}
    elif phase == "done":
        status = {"status": "ready"}
    else:
        status = {"status": "training", "phase": phase, "progress": (st or {}).get("progress")}
    decisions = [{k: x[k] for k in ("column", "question", "type", "labels")} for x in (plan or {}).get("decisions", []) if x.get("include")]
    created = (st or {}).get("started_at") or datetime.fromtimestamp(d.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    return _entry(d.name, job.get("name") or d.name, job.get("base") or (st or {}).get("base") or "small",
                  job.get("steps") or (st or {}).get("steps"), created, status, decisions, job=job)


def _custom() -> list[dict]:
    jobs = sorted((p for p in (S.TYPICALLY / "jobs").glob("*") if p.is_dir() and typically_job.SLUG_RE.fullmatch(p.name) and p.name not in D.ALIASES
                   and auth.visible(p.name)), key=lambda p: p.name)
    entries = sorted(filter(None, map(_job_entry, jobs)), key=lambda e: e["created_at"], reverse=True)
    sample = _entry(NORTHWIND["id"], NORTHWIND["name"], NORTHWIND["base"], NORTHWIND["steps"], None, {"status": "ready"}, NORTHWIND_DECISIONS, sample=True)
    return entries + [sample]   # newest first, the sample last


@router.get("/api/typically/models/library")
def library():
    base = [{"id": i, "name": i, "kind": "base", "status": "ready", **b} for i, b in BASES.items()]
    return {"base": base, "custom": _custom()}


@router.get("/api/typically/models/library/{model_id}")
def library_entry(model_id: str):
    if model_id in BASES:
        return next(b for b in library()["base"] if b["id"] == model_id) | {"examples": [NORTHWIND_CASE]}
    if model_id == "northwind":
        return next(e for e in _custom() if e["sample"]) | {
            "examples": [NORTHWIND_CASE, *_states(S.REPO_ROOT / "data_co_a" / "eval" / "a_oneliner.jsonl", 2)]}
    if (not typically_job.SLUG_RE.fullmatch(model_id) or not auth.visible(model_id) or not (d := S.TYPICALLY / "jobs" / model_id).is_dir()
            or (e := _job_entry(d)) is None):
        raise HTTPException(404, "no such model")
    return e | {"examples": _states(d / "eval" / "import_oneliner.jsonl", 3)}


@router.delete("/api/typically/models/library/{model_id}")
def archive(model_id: str):
    if model_id in D.ALIASES or model_id in BASES:
        raise HTTPException(400, "the built-in models cannot be deleted")
    if not typically_job.SLUG_RE.fullmatch(model_id) or not auth.visible(model_id) or not (d := S.TYPICALLY / "jobs" / model_id).is_dir():
        raise HTTPException(404, "no such model")
    if (_json(d / "status.json") or {}).get("phase") in typically_job.ACTIVE:
        raise HTTPException(409, "this model is still training")
    dest = d.parent / ".archive" / model_id
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():   # archived before under the same slug: keep both
        dest = dest.with_name(f"{model_id}_{int(time.time())}")
    shutil.move(d, dest)
    return {"archived": model_id}


# ---------------------------------------------------------------- chat

class Question(BaseModel):
    question: str = Field(min_length=1)
    type: Literal["choice", "noul", "score"]
    labels: list[str] = []


class AskRequest(BaseModel):
    model: str = "typical-small"
    case: str = Field(min_length=1, max_length=20_000)
    questions: list[Question] = Field(min_length=1, max_length=20)
    compare_with_base: bool = False


@router.post("/api/typically/ask")
def ask(req: AskRequest):
    for q in req.questions:   # before any model load; noul labels are fixed (no/yes) by S._run
        if q.type != "noul" and len(q.labels) < 2:
            raise HTTPException(400, f"'{q.type}' question {q.question!r} needs >=2 labels")
    auth.check(req.model)
    model = req.model if req.model in BASES else f"local:{D.run_of(req.model)}"
    queries = [S.Query(type=q.type, question=q.question, labels=q.labels) for q in req.questions]
    t0 = time.perf_counter()
    with S._lock:
        main = S._run(S.get_model(model), req.case, queries)
        out = {"model": req.model, "results": main["results"]}
        if req.compare_with_base and model.startswith("local:"):
            base = S.base_of(model)
            out["base"] = {"model": base, "results": S._run(S.get_model(base), req.case, queries)["results"]}
    return out | {"ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------- natural-language question parsing

class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str = "en"


YESNO = re.compile(r"(is|are|does|do|did|should|can|could|will|would|has|have|האם)\b", re.I)
SCORE = re.compile(r"\bhow\s+(?!should|do|does|did|can|could|will|would|many|to\b)\w+|\brate\b|on a scale", re.I)
SPLIT = re.compile(r"\s*(?:[,/|]|\bor\b|\bאו\b)\s*")


def _options(text: str) -> list[str]:
    """Labels from "... (a / b / c)", "...: a, b or c?" or a bare "a, b or c?"; [] when there are none."""
    t = text.strip().rstrip("?؟ ").strip()
    inner = re.search(r"\(([^()]+)\)$", t)
    whole = not inner and ":" not in t
    parts = [p.strip(" .") for p in SPLIT.split(inner.group(1) if inner else t.rpartition(":")[2])]
    # ponytail: bare option lists must be short phrases and not a yes/no opener; ceiling = no real parsing, upgrade = the LLM path
    if len(parts) < 2 or not all(parts) or any(len(p.split()) > 3 for p in parts) or (whole and YESNO.match(t)):
        return []
    return parts


def _heuristic_question(text: str) -> dict:
    text = text.strip()
    if labels := _options(text):
        return {"question": text, "type": "choice", "labels": labels}
    if YESNO.match(text):
        return {"question": text, "type": "noul", "labels": ["no", "yes"]}
    if SCORE.search(text):
        return {"question": text, "type": "score", "labels": ["1", "2", "3", "4", "5"]}
    return {"question": text, "type": "noul", "labels": ["no", "yes"]}


QUESTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["question", "type", "labels"],
    "properties": {"question": {"type": "string"}, "type": {"enum": ["choice", "noul", "score"]}, "labels": {"type": "array", "items": {"type": "string"}}}}
QUESTION_SYSTEM = (
    "Turn the user's text into ONE decision question about a case (a support ticket, a document, a record). Return type: "
    '"choice" = pick one of 2+ named options (labels = the options); "noul" = a yes/no question (labels = ["no", "yes"]); '
    '"score" = a rating on an ordered scale (labels = the scale, lowest first, default ["1","2","3","4","5"]). '
    "Keep the user's wording and language; only fix grammar and add a question mark.")


def _llm_question(text: str, lang: str) -> dict:
    system = QUESTION_SYSTEM + (f" Write the question in {tp.LANGS[lang]}." if lang in tp.LANGS and lang != "en" else "")
    q, _ = tllm.complete_json(system, text, QUESTION_SCHEMA, max_tokens=1000)
    if q["type"] == "noul":
        q["labels"] = ["no", "yes"]
    if not q["question"].strip() or (q["type"] != "noul" and len(q["labels"]) < 2):
        raise ValueError("unusable question")
    return q


@router.post("/api/typically/parse_question")
def parse_question(req: ParseRequest):
    if tllm.available():
        try:
            return {**_llm_question(req.text, req.lang), "source": "llm"}
        except Exception:   # any API / schema failure: the heuristic is always available (never echo the key)
            pass
    return {**_heuristic_question(req.text), "source": "heuristic"}
