"""POST /api/typically/analyze: read a table (csv / xlsx / Hugging Face / Google Sheets / sample / an earlier upload, or a mailbox
turned into one by typically_mail), profile it and
return the DatasetPlan (.context/typically/FLOW.md). Parsed records are kept server-side under `records_token` so /build does not
need the file again: in memory and in .context/typically/uploads/<token>.jsonl (see load_upload), described by <token>.meta.json.
GET/DELETE /api/typically/datasets: those uploads, and the models built from them (jobs/<slug>/job.json "records_token")."""
import base64
import binascii
import json
import random
import re
import sys
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import typically_auth as auth
import typically_job
import typically_llm as tllm
import typically_mail as tmail
import typically_plan as tp

UPLOADS = typically_job.TYPICALLY / "uploads"
MAX_MAIL_MB = 50
MAX_MAIL_B64 = MAX_MAIL_MB * 1024 * 1024 * 4 // 3 + 4   # ponytail: decoded and parsed in memory; bigger mailboxes need a chunked upload
# ponytail: in-memory cache lost on restart, capped at 16 tables; load_upload falls back to the jsonl on disk
_records: OrderedDict[str, list[dict]] = OrderedDict()
router = APIRouter()


class Source(BaseModel):
    kind: Literal["csv", "xlsx", "hf", "sheets", "sample", "upload", "mail"]
    text: str | None = None
    data_base64: str | None = None   # xlsx / mail bytes (.mbox, .eml, .zip)
    owner: str | None = None   # mail: the mailbox owner's address(es), comma-separated; None = detected
    sheet: str | None = None   # xlsx sheet; None = the first
    token: str | None = None   # upload: a records_token from an earlier /analyze
    name: str | None = None   # csv / xlsx / mail file name; sample: "northwind" (default) | "enron" | "enron-decisions"
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
    sheets, token, mail, warnings = None, None, None, []
    if src["kind"] == "mail" and len(src.get("data_base64") or "") > MAX_MAIL_B64:
        raise HTTPException(413, f"that mailbox is over {MAX_MAIL_MB} MB; export one folder or a shorter date range and try again")
    try:
        if src["kind"] == "upload":   # start again from a table we already have: no new upload
            token = src.get("token") or ""
            name, records = _meta(token)["name"], load_upload(token)
        elif src["kind"] == "mail":
            try:
                data = base64.b64decode(src.get("data_base64") or "", validate=True)
            except binascii.Error:
                raise ValueError("could not read that file")
            records, mail, warnings = tmail.read_mail(data, src.get("owner"), req.lang)
        else:
            sheets, records = tp.read_xlsx(src.get("data_base64") or "", src.get("sheet")) if src["kind"] == "xlsx" else (None, tp.load_records(src))
        if src["kind"] != "upload":
            name = tp.name_hint(src)
        prof = tp.profile(records)
        if tllm.available():   # the operator's key only: users never bring one
            plan, source, llm_warnings = tp.llm_plan(prof, tp.sample_rows(prof, records), records, lang=req.lang)
            warnings += llm_warnings
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
    # preview: 3 rows that between them show the most different answers (a mailbox sample is mostly unanswered junk otherwise)
    pool, seen, pick = random.Random(0).sample(range(len(records)), min(200, len(records))), set(), []
    for _ in range(min(3, len(pool))):
        i = max((j for j in pool if j not in pick), key=lambda j: len({kv for kv in tp.answers(records[j], plan).items() if kv[1] is not None} - seen))
        pick.append(i)
        seen |= {kv for kv in tp.answers(records[i], plan).items() if kv[1] is not None}
    pick.sort()
    return {
        "name_hint": name, "n_rows": len(records), "columns": list(prof["columns"]),
        "profile": trim(prof), "plan": plan, "plan_source": source, "lang": req.lang,
        "preview_cases": [{"case": tp.render_case(records[i], plan), "answers": tp.answers(records[i], plan)} for i in pick],
        "records_token": token, "warnings": warnings,
        **({"sheets": sheets, "sheet": src.get("sheet") or sheets[0]} if sheets else {}),   # xlsx: the UI offers the other sheets
        **({"mail": mail} if mail else {}),   # {owner, messages, threads, inbound}
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
    # ponytail: re-analyses of a sample are copies of its row below; they stay on disk, hidden here, their models listed on the row
    hidden = [d for d in up if d["kind"] == "sample"]
    samples = [{"token": tok, "name": label, "kind": "sample", "rows": (shape := _sample_shape(name))[0], "columns": list(shape[1]),
                "created": None, "sample": True, "models": fixed + [m for d in hidden if d["name"] == tp.SAMPLES[name][0] for m in d["models"]]}
               for tok, name, label, fixed in SAMPLE_ROWS]
    up = sorted((d for d in up if d["kind"] != "sample"), key=lambda d: d["created"], reverse=True)
    return {"datasets": up + samples}


# Data page token -> sample name, label, the models that come with it
SAMPLE_ROWS = [("sample", "northwind", "Northwind sample", [{"id": "northwind", "name": "Northwind triage"}]),
               ("sample-enron", "enron", "Enron email sample", []), ("sample-enron-decisions", "enron-decisions", "Enron decisions", [])]


@cache
def _sample_shape(name: str) -> tuple[int, tuple[str, ...]]:
    rows = tp.load_records({"kind": "sample", "name": name})
    return len(rows), tuple(rows[0])



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
