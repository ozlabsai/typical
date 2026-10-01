"""site/typically_corrections.py: corrections store + retrain into a new version. Temp dirs, no model, no GPU (start_job is faked).
uv run --no-sync --with pytest --with httpx pytest tests/test_typically_corrections.py -q"""
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "site"), str(ROOT / "scripts")]
import server  # noqa: E402  (first: the routers read server at call time)
from fastapi.testclient import TestClient  # noqa: E402

REAL_START_JOB = server.start_job

EVAL_CASE = "held out case"
Q = {"question": "Which team?", "type": "choice", "labels": ["A", "B"]}


def rows(states):
    return "".join(json.dumps({"state": s, "query": "Which team?", "candidates": ["A", "B"], "label": 0}) + "\n" for s in states)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "TYPICALLY", tmp_path)
    started = []

    def fake_start(slug, base="small", steps=400):
        started.append((slug, base, steps))
        return {"phase": "queued"}
    monkeypatch.setattr(server, "start_job", fake_start)
    d = tmp_path / "jobs" / "acme"
    (d / "eval").mkdir(parents=True)
    (d / "train.jsonl").write_text(rows(["t1", "t2"]))
    (d / "val.jsonl").write_text(rows(["v1"]))
    (d / "eval" / "import_oneliner.jsonl").write_text(rows([EVAL_CASE]))
    (d / "plan.json").write_text(json.dumps({"decisions": [{"column": "team", "include": True, **Q}]}))
    (d / "job.json").write_text(json.dumps({"name": "Acme", "base": "medium", "steps": 200}))
    (d / "status.json").write_text(json.dumps({"phase": "done"}))
    return TestClient(server.app), tmp_path, started


def fix(c, mid, case, answer="B", **kw):
    return c.post(f"/api/typically/models/{mid}/corrections", json={"case": case, **Q, "answer": answer, "model_answer": "A", "p": 0.7, **kw})


def test_store_dedupe_list_delete(env):
    c, root, _ = env
    assert fix(c, "acme", "case one").json() == {"count": 1, "n": 0}
    assert fix(c, "acme", "case two").json() == {"count": 2, "n": 1}
    assert fix(c, "acme", "case one", answer="A").json()["count"] == 2   # same (case, question): replaced, latest wins
    got = c.get("/api/typically/models/acme/corrections").json()
    assert got["count"] == 2 and [(x["case"], x["answer"], x["n"]) for x in got["items"]] == [("case one", "A", 1), ("case two", "B", 0)]
    assert c.delete("/api/typically/models/acme/corrections/0").json() == {"count": 1}
    assert c.delete("/api/typically/models/acme/corrections/5").status_code == 404
    assert [x["case"] for x in c.get("/api/typically/models/acme/corrections").json()["items"]] == ["case one"]
    assert (root / "corrections" / "acme.jsonl").exists()
    assert fix(c, "northwind", "a ticket", type="noul", labels=[], answer="yes").json()["count"] == 1   # the sample; noul labels are fixed


def test_validation(env):
    c, _, _ = env
    assert fix(c, "nope", "x").status_code == 404   # no such job
    assert fix(c, "Bad-Id", "x").status_code == 404   # slug regex
    assert fix(c, "acme", "x", answer="C").status_code == 400   # not one of the options
    assert c.get("/api/typically/models/nope/corrections").status_code == 404


def test_library_counts_and_lineage(env):
    c, _, _ = env
    fix(c, "acme", "case one")
    lib = {m["id"]: m for m in c.get("/api/typically/models/library").json()["custom"]}
    assert (lib["acme"]["corrections"], lib["acme"]["version"], lib["acme"]["parent"]) == (1, 1, None)
    assert lib["northwind"]["corrections"] == 0


def test_retrain_builds_a_version(env):
    c, root, started = env
    fix(c, "acme", "new case")
    fix(c, "acme", EVAL_CASE)   # a held-out case: must not leak into train
    eval_before = (root / "jobs/acme/eval/import_oneliner.jsonl").read_bytes()
    r = c.post("/api/typically/models/acme/retrain", json={}).json()
    assert (r["id"], r["name"], r["version"], r["parent"], r["corrections"], r["held_out_skipped"]) == ("acme_v2", "Acme v2", 2, "acme", 1, 1)
    assert started == [("acme_v2", "medium", 200)]   # the parent's base and steps, through server.train_slug
    v2 = root / "jobs" / "acme_v2"
    train = [json.loads(line) for line in (v2 / "train.jsonl").open()]
    assert [x["state"] for x in train] == ["t1", "t2", "new case", "new case", "new case"]   # original + 3x the correction
    assert train[-1]["label"] == 1 and train[-1]["task"] == "import_correction"
    assert (v2 / "eval/import_oneliner.jsonl").read_bytes() == eval_before and (v2 / "val.jsonl").exists() and (v2 / "plan.json").exists()
    assert json.loads((v2 / "job.json").read_text()) == {"name": "Acme v2", "base": "medium", "steps": 200, "parent": "acme",
                                                             "version": 2, "corrections": 1, "held_out_skipped": 1}
    lib = {m["id"]: m for m in c.get("/api/typically/models/library").json()["custom"]}
    assert (lib["acme_v2"]["version"], lib["acme_v2"]["parent"], lib["acme_v2"]["name"]) == (2, "acme", "Acme v2")

    # v2's own corrections -> v3 of the same root, carrying v2's train (and so v1's corrections)
    fix(c, "acme_v2", "another case")
    r = c.post("/api/typically/models/acme_v2/retrain", json={"steps": 400}).json()
    assert (r["id"], r["name"], r["parent"]) == ("acme_v3", "Acme v3", "acme_v2") and started[-1] == ("acme_v3", "medium", 400)
    assert sum(1 for _ in (root / "jobs/acme_v3/train.jsonl").open()) == 8


def test_retrain_refusals(env, monkeypatch):
    c, root, _ = env
    assert c.post("/api/typically/models/acme/retrain", json={}).status_code == 400   # nothing to learn from
    fix(c, "acme", EVAL_CASE)
    assert c.post("/api/typically/models/acme/retrain", json={}).status_code == 400   # only held-out cases
    fix(c, "acme", "new case")
    monkeypatch.setattr(server, "start_job", REAL_START_JOB)   # the real admission check; nothing starts: it refuses
    monkeypatch.setenv("TYPICALLY_MAX_JOBS", "1")
    monkeypatch.setitem(server._jobs, "other", None)   # every GPU busy
    assert c.post("/api/typically/models/acme/retrain", json={}).status_code == 429
    assert not (root / "jobs" / "acme_v2").exists()
    server._jobs.pop("other")

    def busy(*a, **k):
        raise HTTPException(409, "busy")
    monkeypatch.setattr(server, "start_job", busy)
    assert c.post("/api/typically/models/acme/retrain", json={}).status_code == 409
    assert not (root / "jobs" / "acme_v2").exists()   # a failed start leaves nothing half-built


def test_retrain_northwind_from_the_sample(env):
    c, root, started = env
    fix(c, "northwind", "Customer: Test Co (starter plan, EU).\n\nMy parcel never arrived.", type="noul", labels=[], answer="yes",
        question="Should we escalate this to a manager?")
    r = c.post("/api/typically/models/northwind/retrain", json={}).json()
    assert (r["id"], r["name"], r["version"]) == ("northwind_v2", "Northwind triage v2", 2) and started == [("northwind_v2", "small", 400)]
    v2 = root / "jobs" / "northwind_v2"
    train = [json.loads(line) for line in (v2 / "train.jsonl").open()]
    assert len(train) > 100 and sum(x["meta"].get("correction", False) for x in train) == 3
    assert any(x["query"] == "Which team should handle this ticket?" for x in train)   # chat's Northwind questions
    assert (v2 / "eval" / "import_oneliner.jsonl").exists()
