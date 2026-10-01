"""site/typically_models.py: models library, archive, /ask validation, parse_question. No model, no network.
uv run --no-sync --with pytest --with httpx pytest tests/test_typically_models.py -q"""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "site"), str(ROOT / "scripts")]
import server  # noqa: E402  (first: the routers read server at call time)
import typically_models as tm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def job(root, slug, status=None, plan=True, name=None, evals=None):
    d = root / "jobs" / slug
    (d / "eval").mkdir(parents=True)
    (d / "eval" / "import_oneliner.jsonl").write_text("".join(json.dumps({"state": f"case {i}", "query": q}) + "\n" for i in range(5) for q in "ab"))
    if name:
        (d / "job.json").write_text(json.dumps({"name": name, "base": "medium", "steps": 200}))
    if plan:
        (d / "plan.json").write_text(json.dumps({"decisions": [
            {"column": "team", "include": True, "question": "Which team?", "type": "choice", "labels": ["A", "B"], "mapping": {}},
            {"column": "skip", "include": False, "question": "x", "type": "noul", "labels": ["no", "yes"]}]}))
    if status:
        (d / "status.json").write_text(json.dumps(status))
    return d


def trained(root, run, acc=None):
    (root / "results" / run).mkdir(parents=True, exist_ok=True)
    torch.save({"args": {}}, root / "results" / run / "best.pt")
    for f, a in (acc or {}).items():
        (root / "results" / run / f).write_text(json.dumps({"eval": {"import_oneliner": {"n": 55, "raw": {"acc_k": a}}}}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "TYPICALLY", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    job(tmp_path, "old_done", {"phase": "done", "started_at": "2026-09-01T00:00:00+00:00"}, name="Old")
    trained(tmp_path, "co_old_done", {"eval_co.json": 0.8, "base_eval_co.json": 0.5})
    job(tmp_path, "new_run", {"phase": "evaluating", "started_at": "2026-09-30T00:00:00+00:00", "progress": 0.4})
    job(tmp_path, "bad", {"phase": "failed", "started_at": "2026-09-15T00:00:00+00:00", "message": "no gpu", "code": "failed"})
    job(tmp_path, "junk", plan=False)   # no plan, no status: skipped
    (tmp_path / "results" / "co_old_done" / "hf_repo.txt").write_text("me/old")
    (tmp_path / "keys.json").write_text(json.dumps({"h": {"model_id": "old_done", "created_at": "x"}}))
    return TestClient(server.app), tmp_path


def test_library(env):
    c, _ = env
    lib = c.get("/api/typically/models/library").json()
    assert [b["id"] for b in lib["base"]] == ["typical-small", "typical-medium"] and lib["base"][0]["params"] == "1.7B"
    assert [m["id"] for m in lib["custom"]] == ["new_run", "bad", "old_done", "northwind"]
    new, bad, old, nw = lib["custom"]
    assert (new["status"], new["phase"], new["progress"]) == ("training", "evaluating", 0.4)
    assert (bad["status"], bad["message"]) == ("failed", "no gpu")
    assert (old["status"], old["name"], old["base"], old["steps"], old["run"]) == ("ready", "Old", "medium", 200, "co_old_done")
    assert old["metrics"] == {"standard": 0.5, "yours": 0.8, "n_cases": 55}
    assert old["decisions"] == [{"column": "team", "question": "Which team?", "type": "choice", "labels": ["A", "B"]}]
    assert old["has_key"] and old["hf_repo"] == "me/old" and not new["has_key"] and new["metrics"] is None
    assert nw["sample"] and nw["run"] == "co_f" and [d["column"] for d in nw["decisions"]] == ["team", "escalate", "urgency", "refund"]


def test_reveal_metrics_win(env):
    c, tmp = env
    (tmp / "results" / "co_old_done" / "reveal.json").write_text(json.dumps({"score": {"standard": 0.4, "yours": 0.9}, "n_cases": 7}))
    old = c.get("/api/typically/models/library/old_done").json()
    assert old["metrics"] == {"standard": 0.4, "yours": 0.9, "n_cases": 7} and old["examples"] == ["case 0", "case 1", "case 2"]


def test_entry_northwind_and_404(env):
    c, _ = env
    nw = c.get("/api/typically/models/library/northwind").json()
    assert nw["examples"][0].startswith("Customer: Cedar Foods") and 1 <= len(nw["examples"]) <= 3
    assert c.get("/api/typically/models/library/nope").status_code == 404


def test_delete_archives(env):
    c, tmp = env
    assert c.delete("/api/typically/models/library/new_run").status_code == 409   # active
    assert c.delete("/api/typically/models/library/northwind").status_code == 400
    assert c.delete("/api/typically/models/library/nope").status_code == 404
    assert c.delete("/api/typically/models/library/old_done").status_code == 200
    assert (tmp / "jobs" / ".archive" / "old_done" / "plan.json").exists() and not (tmp / "jobs" / "old_done").exists()
    assert "old_done" not in [m["id"] for m in c.get("/api/typically/models/library").json()["custom"]]


class Fake:
    device = "cpu"


def test_ask(env, monkeypatch):
    c, tmp = env
    trained(tmp, "co_f")
    ran = []
    monkeypatch.setattr(server, "get_model", lambda name: ran.append(name) or Fake())
    monkeypatch.setattr(server, "base_of", lambda name: "typical-small")
    monkeypatch.setattr(server, "_run", lambda m, state, qs: {"results": [{"probs": {}, "p_null": 0, "argmax": "x"}] * len(qs), "ms": 1})
    q = {"question": "Escalate?", "type": "noul"}
    ok = c.post("/api/typically/ask", json={"model": "northwind", "case": "hi", "questions": [q], "compare_with_base": True}).json()
    assert ran == ["local:co_f", "typical-small"] and ok["model"] == "northwind" and ok["base"]["model"] == "typical-small" and len(ok["results"]) == 1 and "ms" in ok
    ran.clear()
    assert "base" not in c.post("/api/typically/ask", json={"model": "typical-medium", "case": "hi", "questions": [q], "compare_with_base": True}).json() and ran == ["typical-medium"]
    ran.clear()
    ask = lambda **kw: c.post("/api/typically/ask", json={"case": "hi", "questions": [q], **kw}).status_code
    assert ask(questions=[{"question": "t?", "type": "choice", "labels": ["a"]}]) == 400
    assert ask(questions=[{"question": "t?", "type": "score"}]) == 400
    assert ask(questions=[q] * 21) == 422 and ask(questions=[]) == 422
    assert ask(case="x" * 20_001) == 422 and ask(case="") == 422
    assert ask(questions=[{"question": "t?", "type": "bogus"}]) == 422
    assert ask(model="BAD ID") == 404
    assert ran == []   # nothing validated-away ever reached the model


@pytest.mark.parametrize("text, type_, labels", [
    ("Should we refund this?", "noul", ["no", "yes"]),
    ("Which team: billing, claims or sales?", "choice", ["billing", "claims", "sales"]),
    ("Which team? (billing / claims / sales)", "choice", ["billing", "claims", "sales"]),
    ("billing, claims or sales?", "choice", ["billing", "claims", "sales"]),
    ("How urgent is this?", "score", ["1", "2", "3", "4", "5"]),
    ("Rate the tone", "score", ["1", "2", "3", "4", "5"]),
    ("האם להחזיר כסף ללקוח?", "noul", ["no", "yes"]),
    ("איזו מחלקה: חיובים, תביעות או מכירות?", "choice", ["חיובים", "תביעות", "מכירות"]),
    ("Is this billing or claims?", "noul", ["no", "yes"]),
    ("What is the customer mood", "noul", ["no", "yes"]),
])
def test_parse_heuristic(env, text, type_, labels):
    r = env[0].post("/api/typically/parse_question", json={"text": text}).json()
    assert (r["type"], r["labels"], r["source"], r["question"]) == (type_, labels, "heuristic", text)


def test_parse_llm_and_fallback(env, monkeypatch):
    c, _ = env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(tm, "_llm_question", lambda text, lang: {"question": "Refund?", "type": "noul", "labels": ["no", "yes"]})
    assert c.post("/api/typically/parse_question", json={"text": "refund"}).json()["source"] == "llm"
    monkeypatch.setattr(tm, "_llm_question", lambda *a: 1 / 0)
    assert c.post("/api/typically/parse_question", json={"text": "Should we refund?"}).json()["source"] == "heuristic"
