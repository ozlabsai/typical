"""POST /api/typically/analyze: read a table (csv / xlsx / Hugging Face / Google Sheets / sample / an earlier upload), profile it and
return the DatasetPlan (.context/typically/FLOW.md). Parsed records are kept server-side under `records_token` so /build does not
need the file again: in memory and in .context/typically/uploads/<token>.jsonl (see load_upload), described by <token>.meta.json.
GET/DELETE /api/typically/datasets: those uploads, and the models built from them (jobs/<slug>/job.json "records_token")."""
import json
import random
import re
import sys
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import typically_auth as auth
import typically_job
import typically_llm as tllm
import typically_plan as tp

UPLOADS = typically_job.TYPICALLY / "uploads"
# ponytail: in-memory cache lost on restart, capped at 16 tables; load_upload falls back to the jsonl on disk
_records: OrderedDict[str, list[dict]] = OrderedDict()
router = APIRouter()


class Source(BaseModel):
    kind: Literal["csv", "xlsx", "hf", "sheets", "sample", "upload"]
    text: str | None = None
    data_base64: str | None = None   # xlsx bytes
    sheet: str | None = None   # xlsx sheet; None = the first
    token: str | None = None   # upload: a records_token from an earlier /analyze
    name: str | None = None   # csv / xlsx file name
    dataset: str | None = None
    config: str | None = None
    split: str | None = None
    limit: int | None = None
    url: str | None = None


class AnalyzeRequest(BaseModel):
    source: Source
    lang: Literal["en", "he"] = "en"   # language of the questions, reasons and warnings we write


def _upload_file(token: str, must_exist: bool = True) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise HTTPException(400, "bad records_token")
    f = UPLOADS / f"{token}.jsonl"
    if must_exist and not f.exists():
        raise HTTPException(404, "unknown records_token; analyze the data again")
    return f


def _check_owner(token: str) -> None:
    if auth.enabled() and not auth.can_see(_meta(token).get("owner")):
        raise HTTPException(404, "unknown records_token; analyze the data again")


def load_upload(token: str) -> list[dict]:
    _upload_file(token, must_exist=False)   # validates the token
    _check_owner(token)
    if token in _records:
        return _records[token]
    return [json.loads(line) for line in _upload_file(token).read_text().splitlines()]


def trim(prof: dict) -> dict:
    """Full value counts only where they are chips (candidate decisions); the rest keep their top 10."""
    cols = {c: {**p, "values": p["values"][: tp.MAX_VALUES if c in prof["decisions"] else 10]} for c, p in prof["columns"].items()}
    return {**prof, "columns": cols}


@router.post("/api/typically/analyze")
def analyze(req: AnalyzeRequest):
    src = req.source.model_dump(exclude_none=True)
    sheets, token = None, None
    try:
        if src["kind"] == "upload":   # start again from a table we already have: no new upload
            token = src.get("token") or ""
            name, records = _meta(token)["name"], load_upload(token)
        else:
            sheets, records = tp.read_xlsx(src.get("data_base64") or "", src.get("sheet")) if src["kind"] == "xlsx" else (None, tp.load_records(src))
            name = tp.name_hint(src)
        prof = tp.profile(records)
        warnings: list[str] = []
        if tllm.available():   # the operator's key only: users never bring one
            plan, source, warnings = tp.llm_plan(prof, tp.sample_rows(prof, records), records, lang=req.lang)
        else:
            plan, source = tp.heuristic_plan(prof, records, req.lang), "heuristic"
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not plan["decisions"]:
        warnings.append(tp.msg(req.lang, "w_nodecision"))
    if token is None:
        token = uuid.uuid4().hex
        _records[token] = records
        while len(_records) > 16:
            _records.popitem(last=False)
        UPLOADS.mkdir(parents=True, exist_ok=True)
        (UPLOADS / f"{token}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        sheet = src.get("sheet") or (sheets[0] if sheets else None)
        meta = {"name": name + (f" ({sheet})" if sheets and len(sheets) > 1 else ""), "kind": src["kind"], "rows": len(records),
                "columns": list(prof["columns"]), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **auth.stamp()}
        (UPLOADS / f"{token}.meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    pick = sorted(random.Random(0).sample(range(len(records)), min(3, len(records))))
    return {
        "name_hint": name, "n_rows": len(records), "columns": list(prof["columns"]),
        "profile": trim(prof), "plan": plan, "plan_source": source, "lang": req.lang,
        "preview_cases": [{"case": tp.render_case(records[i], plan), "answers": tp.answers(records[i], plan)} for i in pick],
        "records_token": token, "warnings": warnings,
        **({"sheets": sheets, "sheet": src.get("sheet") or sheets[0]} if sheets else {}),   # xlsx: the UI offers the other sheets
    }


# ---------------------------------------------------------------- datasets (the Data page)

def _meta(token: str) -> dict:
    """<token>.meta.json; uploads from before it existed get one from the jsonl itself."""
    f = _upload_file(token)
    if (m := f.with_suffix(".meta.json")).exists():
        return json.loads(m.read_text())
    with f.open() as fh:
        first, rows = fh.readline(), 1 + sum(1 for _ in fh)
    return {"name": f"Upload {token[:6]}", "kind": None, "rows": rows, "columns": list(json.loads(first)) if first else [],
            "created": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}


def _models_by_token() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for j in (UPLOADS.parent / "jobs").glob("*/job.json"):
        job = json.loads(j.read_text())
        if (t := job.get("records_token")) and auth.can_see(job.get("owner")):
            out.setdefault(t, []).append({"id": j.parent.name, "name": job.get("name") or j.parent.name})
    return out


@router.get("/api/typically/datasets")
def datasets():
    used = _models_by_token()
    up = [{"token": f.stem, **_meta(f.stem), "sample": False, "models": used.get(f.stem, [])} for f in UPLOADS.glob("*.jsonl")]
    up = [{k: v for k, v in d.items() if k != "owner"} for d in up if auth.can_see(d.get("owner"))]
    # ponytail: re-analyses of the sample are copies of the Northwind row below; they stay on disk, hidden here
    up = sorted((d for d in up if d["kind"] != "sample"), key=lambda d: d["created"], reverse=True)
    rows = tp.load_records({"kind": "sample"})
    sample = {"token": "sample", "name": "Northwind sample", "kind": "sample", "rows": len(rows), "columns": list(rows[0]), "created": None,
              "sample": True, "models": [{"id": "northwind", "name": "Northwind triage"}]}
    return {"datasets": up + [sample]}


@router.delete("/api/typically/datasets/{token}")
def delete_dataset(token: str):
    _meta(token)   # 400 / 404
    _check_owner(token)
    if names := [m["name"] for m in _models_by_token().get(token, [])]:
        raise HTTPException(409, "models were trained from this data: " + ", ".join(names))
    for f in (UPLOADS / f"{token}.jsonl", UPLOADS / f"{token}.meta.json"):
        f.unlink(missing_ok=True)
    _records.pop(token, None)
    return {"deleted": token}
