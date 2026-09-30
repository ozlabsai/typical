"""POST /api/typically/analyze: read a table (csv / Hugging Face / Google Sheets / sample), profile it and return the
DatasetPlan (.context/typically/FLOW.md). Parsed records are kept server-side under `records_token` so /build does not
need the file again: in memory and in .context/typically/uploads/<token>.jsonl (see load_upload)."""
import json
import os
import random
import re
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import typically_plan as tp

UPLOADS = REPO_ROOT / ".context" / "typically" / "uploads"
# ponytail: in-memory cache lost on restart, capped at 16 tables; load_upload falls back to the jsonl on disk
_records: OrderedDict[str, list[dict]] = OrderedDict()
router = APIRouter()


class Source(BaseModel):
    kind: Literal["csv", "hf", "sheets", "sample"]
    text: str | None = None
    name: str | None = None   # csv file name
    dataset: str | None = None
    config: str | None = None
    split: str | None = None
    limit: int | None = None
    url: str | None = None


class AnalyzeRequest(BaseModel):
    source: Source
    anthropic_key: str | None = None   # BYOK: used for this request only, never stored or logged


def load_upload(token: str) -> list[dict]:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise HTTPException(400, "bad records_token")
    if token in _records:
        return _records[token]
    f = UPLOADS / f"{token}.jsonl"
    if not f.exists():
        raise HTTPException(404, "unknown records_token; analyze the data again")
    return [json.loads(line) for line in f.read_text().splitlines()]


def trim(prof: dict) -> dict:
    """Full value counts only where they are chips (candidate decisions); the rest keep their top 10."""
    cols = {c: {**p, "values": p["values"][: tp.MAX_VALUES if c in prof["decisions"] else 10]} for c, p in prof["columns"].items()}
    return {**prof, "columns": cols}


@router.post("/api/typically/analyze")
def analyze(req: AnalyzeRequest):
    src = req.source.model_dump(exclude_none=True)
    try:
        records = tp.load_records(src)
        prof = tp.profile(records)
        key = req.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        warnings: list[str] = []
        if key:
            plan, source, warnings = tp.llm_plan(prof, tp.sample_rows(prof, records), key, records)
        else:
            plan, source = tp.heuristic_plan(prof, records), "heuristic"
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not plan["decisions"]:
        warnings.append("No column looks like a decision (2-20 different answers); add one in the next step.")
    token = uuid.uuid4().hex
    _records[token] = records
    while len(_records) > 16:
        _records.popitem(last=False)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    (UPLOADS / f"{token}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    pick = sorted(random.Random(0).sample(range(len(records)), min(3, len(records))))
    return {
        "name_hint": tp.name_hint(src), "n_rows": len(records), "columns": list(prof["columns"]),
        "profile": trim(prof), "plan": plan, "plan_source": source,
        "preview_cases": [{"case": tp.render_case(records[i], plan), "answers": tp.answers(records[i], plan)} for i in pick],
        "records_token": token, "warnings": warnings,
    }
