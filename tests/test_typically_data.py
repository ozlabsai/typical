"""Excel upload, analyze-from-an-earlier-upload, the datasets list / delete, and the pod.log -> training series parser. No model, no network.
uv run --no-sync --with pytest --with httpx pytest tests/test_typically_data.py -q"""
import base64
import csv
import datetime as dt
import io
import json
import os
import sys
from pathlib import Path

import openpyxl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "site"), str(ROOT / "scripts")]
import server  # noqa: E402  (first: the routers read server at call time)
import typically_analyze as ta  # noqa: E402
import typically_job as tj  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SAMPLE = list(csv.DictReader((ROOT / "site" / "data" / "typically_sample.csv").open()))


def xlsx(sheets: dict[str, list[list]]) -> str:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(ta, "_records", ta.OrderedDict())
    monkeypatch.setattr(server, "TYPICALLY", tmp_path)
    monkeypatch.setattr(tj, "TYPICALLY", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return TestClient(server.app), tmp_path


def test_xlsx_sheets_header_and_cells(env):
    c, _ = env
    cols = list(SAMPLE[0])
    data = xlsx({"Tickets": [[], [None], cols, *[[r[k] for k in cols] for r in SAMPLE]],   # the header is the first non-empty row
                 "Typed": [["when", "n", "x", None], [dt.datetime(2026, 9, 1), 3.0, 2.5, "dropped: no header"], [dt.datetime(2026, 9, 1, 8, 30), 4, True, None]]})
    sheets, typed = ta.tp.read_xlsx(data, "Typed")
    assert sheets == ["Tickets", "Typed"]
    assert typed == [{"when": "2026-09-01", "n": "3", "x": "2.5"}, {"when": "2026-09-01T08:30:00", "n": "4", "x": "True"}]

    r = c.post("/api/typically/analyze", json={"source": {"kind": "xlsx", "data_base64": data, "name": "tickets.xlsx"}})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["sheets"] == sheets and a["sheet"] == "Tickets"   # the first sheet by default; the UI offers the others
    assert a["n_rows"] == len(SAMPLE) and a["columns"] == cols and a["plan"]["decisions"]
    assert ta.load_upload(a["records_token"]) == [{k: v.strip() for k, v in row.items()} for row in SAMPLE]   # same records as the csv
    assert json.loads((ta.UPLOADS / f"{a['records_token']}.meta.json").read_text())["name"] == "tickets.xlsx (Tickets)"

    assert c.post("/api/typically/analyze", json={"source": {"kind": "xlsx", "data_base64": data, "sheet": "Nope"}}).status_code == 400
    assert c.post("/api/typically/analyze", json={"source": {"kind": "xlsx", "data_base64": base64.b64encode(b"not a zip").decode()}}).status_code == 400


def test_datasets_upload_reuse_and_delete(env):
    c, root = env
    text = (ROOT / "site" / "data" / "typically_sample.csv").read_text()
    a = c.post("/api/typically/analyze", json={"source": {"kind": "csv", "text": text, "name": "northwind.csv"}}).json()
    tok = a["records_token"]
    c.post("/api/typically/analyze", json={"source": {"kind": "sample"}})   # hidden: the sample row stands for it
    legacy = "f" * 32   # an upload from before .meta.json
    (root / "uploads" / f"{legacy}.jsonl").write_text('{"a": "1"}\n{"a": "2"}\n')
    os.utime(root / "uploads" / f"{legacy}.jsonl", (1e9, 1e9))   # older: the list is newest first

    again = c.post("/api/typically/analyze", json={"source": {"kind": "upload", "token": tok}, "lang": "he"}).json()
    assert again["records_token"] == tok and again["name_hint"] == "northwind.csv" and again["n_rows"] == a["n_rows"]
    assert len(list((root / "uploads").glob("*.jsonl"))) == 3   # no new upload
    assert c.post("/api/typically/analyze", json={"source": {"kind": "upload", "token": "0" * 32}}).status_code == 404

    (root / "jobs" / "acme").mkdir(parents=True)
    (root / "jobs" / "acme" / "job.json").write_text(json.dumps({"name": "Acme", "records_token": tok}))
    ds = c.get("/api/typically/datasets").json()["datasets"]
    assert [d["token"] for d in ds] == [tok, legacy, "sample", "sample-enron", "sample-enron-decisions"]
    assert ds[0]["name"] == "northwind.csv" and ds[0]["rows"] == len(SAMPLE) and ds[0]["models"] == [{"id": "acme", "name": "Acme"}]
    assert ds[1]["rows"] == 2 and ds[1]["columns"] == ["a"] and ds[2]["sample"] and ds[2]["rows"] == len(SAMPLE)

    assert c.delete(f"/api/typically/datasets/{tok}").status_code == 409   # a model was trained from it
    assert c.delete("/api/typically/datasets/sample").status_code == 400
    assert c.delete(f"/api/typically/datasets/{legacy}").status_code == 200
    assert c.delete(f"/api/typically/datasets/{legacy}").status_code == 404
    assert [d["token"] for d in c.get("/api/typically/datasets").json()["datasets"]] == [tok, "sample", "sample-enron", "sample-enron-decisions"]


POD_LOG = """ALIVE
step 10 bucket W loss 9.9 step_time 0.10s
[train] loading data_co_acme
step 50 bucket W,E,C loss 0.3697 step_time 0.97s tok/s 12899 lr_tower 2.82e-04 lr_lora 9.40e-05
step 50 val_nll 0.3536
step 50 best_on_nll 0.8272 (data_co_acme_val=0.8272)
step 75 bucket C loss nan step_time 0.80s tok/s 1 lr_tower 1e-4 lr_lora 1e-5
step 100 bucket W,C loss 0.3429 step_time 0.85s tok/s 11452 lr_tower 1.88e-04 lr_lora 6.28e-05
step 100 val_nll 0.3735
step 100 best_on_nll 0.7992 (data_co_acme_val=0.7992)
step 100 best_on_acc 0.7500
runs/co_acme best step 100
"""


def test_log_series(tmp_path):
    f = tmp_path / "pod.log"
    f.write_text(POD_LOG)
    assert tj.log_series(f)["step"] == [10, 50, 100]
    s = tj.log_series(f, tail=len(POD_LOG) - 10)   # the read starts inside the "step 10" line: it is dropped, not misread
    assert s == {"step": [50, 100], "loss": [0.3697, 0.3429], "val": [0.3536, 0.3735], "best_on": [0.8272, 0.7992],
                 "acc": [None, 0.75], "step_time": pytest.approx((0.97 + 0.80 + 0.85) / 3)}   # step 75's nan loss is dropped, its step_time kept
    assert tj.log_series(f, limit=1)["step"] == [100]
    assert tj.log_series(tmp_path / "missing.log")["step"] == []


def test_train_status_series(env):
    c, root = env
    d = root / "jobs" / "acme"
    d.mkdir(parents=True)
    (d / "pod.log").write_text(POD_LOG)
    (d / "status.json").write_text(json.dumps({"phase": "training", "progress": 0.25, "steps": 400}))
    assert c.get("/api/typically/train/acme").json()["series"]["step"] == [10, 50, 100]
    (d / "status.json").write_text(json.dumps({"phase": "uploading"}))
    assert "series" not in c.get("/api/typically/train/acme").json()
