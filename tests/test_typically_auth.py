"""site/typically_auth.py + site/typically_share.py: invite sign-in, signed cookie, the middleware's allow / deny lists, ownership,
quotas and public share cards. Temp data dir, no model, no GPU (typically_job.run is faked).
uv run --no-sync --with pytest --with httpx pytest tests/test_typically_auth.py -q"""
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "site"), str(ROOT / "scripts")]
import server  # noqa: E402  (first: the routers read server at call time)
import typically_analyze as ta  # noqa: E402
import typically_auth as auth  # noqa: E402
import typically_job as tj  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ALICE, BOB, ADMIN = "alice-invite-code-1", "bob-invite-code-2", "admin-invite-code-3"
SECRET_CASE = "Customer Jane Roe, card 4111 1111 1111 1111, wants a refund"


def model(root, slug, owner=None, name=None, scored=False):
    d = root / "jobs" / slug
    (d / "eval").mkdir(parents=True)
    (d / "eval" / "import_oneliner.jsonl").write_text(json.dumps({"state": SECRET_CASE, "query": "q"}) + "\n")
    (d / "train.jsonl").write_text(json.dumps({"state": SECRET_CASE}) + "\n")
    (d / "val.jsonl").write_text(json.dumps({"state": SECRET_CASE}) + "\n")
    (d / "plan.json").write_text(json.dumps({"decisions": [{"column": "team", "include": True, "question": "Which team?", "type": "choice", "labels": ["A", "B"]}]}))
    (d / "status.json").write_text(json.dumps({"phase": "done", "started_at": "2026-09-01T00:00:00+00:00"}))
    (d / "job.json").write_text(json.dumps({"name": name or slug, "base": "small", "steps": 200, **({"owner": auth.uid_of(owner)} if owner else {})}))
    if scored:
        res = root / "results" / f"co_{slug}"
        res.mkdir(parents=True)
        torch.save({"args": {}}, res / "best.pt")
        (res / "reveal.json").write_text(json.dumps({
            "score": {"standard": 0.55, "yours": 0.8}, "n_cases": 40,
            "decisions": [{"key": "team", "question": "Which team?", "type": "choice", "labels": ["A", "B"], "standard": 0.5, "yours": 0.85}],
            "disagreements": [{"case": SECRET_CASE, "key": "team"}]}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    for mod, attr, val in ((server, "TYPICALLY", tmp_path), (tj, "TYPICALLY", tmp_path), (ta, "UPLOADS", tmp_path / "uploads"),
                           (ta, "_records", ta.OrderedDict())):
        monkeypatch.setattr(mod, attr, val)
    monkeypatch.setenv("TYPICALLY_AUTH", "1")
    monkeypatch.setenv("TYPICALLY_SECRET", "test-secret")
    monkeypatch.setenv("TYPICALLY_INVITES", f"{ALICE}, {BOB}")
    monkeypatch.setenv("TYPICALLY_ADMINS", ADMIN)
    for k in ("TYPICALLY_DAILY_RUNS", "TYPICALLY_MAX_JOBS", "TYPICALLY_PUBLIC_URL", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    model(tmp_path, "alice_m", ALICE, name="Alice <script>alert(1)</script>", scored=True)
    model(tmp_path, "alice_two", ALICE)
    model(tmp_path, "bob_m", BOB, scored=True)
    model(tmp_path, "legacy")   # from before auth: no owner, admins only
    return tmp_path


def client(code=None, name=None):
    c = TestClient(server.app, base_url="http://localhost")
    if code:
        r = c.post("/api/auth/login", json={"code": code, "name": name})
        assert r.status_code == 200, r.text
    return c


def ids(c):
    return [m["id"] for m in c.get("/api/typically/models/library").json()["custom"]]


def test_login_me_logout(env):
    c = client()
    assert c.get("/api/auth/me").status_code == 401
    assert c.post("/api/auth/login", json={"code": "nope"}).status_code == 401
    r = c.post("/api/auth/login", json={"code": f" {ALICE} ", "name": "  Alice   A "})
    cookie = r.headers["set-cookie"].lower()
    assert r.status_code == 200 and "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie   # localhost: no Secure
    assert c.get("/api/auth/me").json() == {"auth": True, "user": {"id": auth.uid_of(ALICE), "name": "Alice A", "admin": False}}
    assert client(ADMIN).get("/api/auth/me").json()["user"]["admin"] is True
    c.post("/api/auth/logout")
    assert c.get("/api/auth/me").status_code == 401
    hosted = TestClient(server.app, base_url="http://typically.example").post("/api/auth/login", json={"code": BOB})
    assert "secure" in hosted.headers["set-cookie"].lower()


def test_cookie_tamper_expiry_and_revocation(env, monkeypatch):
    good = auth.make_cookie(auth.uid_of(ALICE), "A")
    assert auth.read_cookie(good).uid == auth.uid_of(ALICE)
    payload, sig = good.split(".")
    forged = auth.make_cookie(auth.uid_of(ADMIN), "A").split(".")[0] + "." + sig   # someone else's payload, Alice's signature
    assert auth.read_cookie(forged) is None and auth.read_cookie(payload + ".") is None and auth.read_cookie("junk") is None
    assert auth.read_cookie(auth.make_cookie(auth.uid_of(ALICE), "A", now=time.time() - auth.MAX_AGE - 1)) is None   # expired
    c = client()
    c.cookies.set(auth.COOKIE, forged)
    assert c.get("/api/typically/models/library").status_code == 401
    monkeypatch.setenv("TYPICALLY_SECRET", "rotated")
    assert auth.read_cookie(good) is None   # a new secret signs everyone out
    monkeypatch.setenv("TYPICALLY_SECRET", "test-secret")
    monkeypatch.setenv("TYPICALLY_INVITES", BOB)
    assert auth.read_cookie(good) is None   # Alice's invite was removed
    (env / "invites.json").write_text(json.dumps([ALICE]))
    assert auth.read_cookie(good).uid == auth.uid_of(ALICE)   # invites.json counts too


def test_middleware_allow_and_deny(env, monkeypatch):
    c = client()
    for method, path in (("get", "/api/typically/models/library"), ("get", "/api/typically/datasets"), ("post", "/api/decide"),
                         ("get", "/api/typically/capabilities"), ("get", "/api/typically/results/northwind")):
        assert getattr(c, method)(path).status_code == 401, path
    assert c.get("/api/health").status_code == 200
    r = c.post("/v1/models/alice_m/decide", json={"state": "x", "queries": []})
    assert r.status_code == 401 and "API key" in r.json()["detail"]   # its own bearer check, not the session
    assert c.get("/s/doesnotexist").status_code == 404   # public route, unknown link
    assert c.get("/app/").status_code != 401 and c.get("/").status_code == 200   # the bundle renders the sign-in screen itself
    monkeypatch.setenv("TYPICALLY_AUTH", "0")
    assert c.get("/api/typically/models/library").status_code == 200   # off: local dev unchanged
    assert c.get("/api/auth/me").json() == {"auth": False, "user": None}


def test_ownership(env):
    a, b, admin = client(ALICE), client(BOB), client(ADMIN)
    assert ids(a) == ["alice_m", "alice_two", "northwind"] and ids(b) == ["bob_m", "northwind"]
    assert set(ids(admin)) == {"alice_m", "alice_two", "bob_m", "legacy", "northwind"}
    for path in ("models/library/bob_m", "train/bob_m", "results/bob_m", "download/co_bob_m", "models/bob_m/corrections",
                 "snippets/bob_m", "models/bob_m/share"):
        assert a.get(f"/api/typically/{path}").status_code == 404, path
    assert b.get("/api/typically/models/library/bob_m").status_code == 200 and b.get("/api/typically/download/co_bob_m").status_code == 200
    assert a.get("/api/typically/models").json()["tuned"] == ["local:co_alice_m"]
    assert a.post("/api/typically/keys", json={"model_id": "bob_m"}).status_code == 404
    assert a.delete("/api/typically/models/library/bob_m").status_code == 404
    assert a.post("/api/typically/ask", json={"model": "bob_m", "case": "x", "questions": [{"question": "q?", "type": "noul"}]}).status_code == 404
    assert a.post("/api/typically/compare", json={"state": "x", "decisions": [], "models": ["local:co_bob_m"]}).status_code == 404
    assert a.post("/api/typically/train", json={"name": "bob_m"}).status_code == 404
    plan = {"records_token": "a" * 32, "plan": {}, "name": "Bob M"}
    assert a.post("/api/typically/build", json=plan).status_code == 409   # someone else's name
    # corrections: own models, and the shared sample keeps one list per user
    fix = {"case": "a ticket", "question": "Escalate?", "type": "noul", "answer": "yes"}
    assert a.post("/api/typically/models/northwind/corrections", json=fix).json()["count"] == 1
    assert b.get("/api/typically/models/northwind/corrections").json()["count"] == 0
    assert a.post("/api/typically/models/bob_m/corrections", json=fix).status_code == 404


def test_datasets_are_per_user(env):
    a, b = client(ALICE), client(BOB)
    csv_text = (ROOT / "site" / "data" / "typically_sample.csv").read_text()
    tok = a.post("/api/typically/analyze", json={"source": {"kind": "csv", "text": csv_text, "name": "mine.csv"}}).json()["records_token"]
    assert json.loads((env / "uploads" / f"{tok}.meta.json").read_text())["owner"] == auth.uid_of(ALICE)
    mine = a.get("/api/typically/datasets").json()["datasets"]
    assert [d["token"] for d in mine] == [tok, "sample"] and "owner" not in mine[0]
    assert [d["token"] for d in b.get("/api/typically/datasets").json()["datasets"]] == ["sample"]
    assert b.post("/api/typically/analyze", json={"source": {"kind": "upload", "token": tok}}).status_code == 404
    assert b.delete(f"/api/typically/datasets/{tok}").status_code == 404
    assert a.delete(f"/api/typically/datasets/{tok}").status_code == 200


@pytest.fixture
def gate(monkeypatch):
    g = threading.Event()
    monkeypatch.setattr(tj, "run", lambda slug, log, base, steps: g.wait(5))
    yield g
    g.set()
    wait_idle()


def wait_idle():
    for _ in range(200):
        if not server._jobs:
            return
        time.sleep(0.01)


def test_quotas(env, gate, monkeypatch):
    a, b, admin = client(ALICE), client(BOB), client(ADMIN)
    train = lambda c, name: c.post("/api/typically/train", json={"name": name})
    assert train(a, "alice_m").status_code == 200
    r = train(a, "alice_two")
    assert r.status_code == 429 and "already have a model training" in r.json()["detail"]   # one active job per user
    monkeypatch.setenv("TYPICALLY_MAX_JOBS", "1")
    r = train(b, "bob_m")
    assert r.status_code == 429 and "busy" in r.json()["detail"]   # global cap
    monkeypatch.setenv("TYPICALLY_MAX_JOBS", "2")
    gate.set()
    wait_idle()
    gate.clear()
    monkeypatch.setenv("TYPICALLY_DAILY_RUNS", "2")
    assert train(a, "alice_two").status_code == 200   # run 2 of 2
    gate.set()
    wait_idle()
    r = train(a, "alice_m")
    assert r.status_code == 429 and "today" in r.json()["detail"]   # daily runs used up
    assert train(admin, "legacy").status_code == 200   # admins: no daily cap
    wait_idle()
    # retrain goes through the same gate, and leaves nothing half-built when refused
    a.post("/api/typically/models/alice_m/corrections", json={"case": "new case", "question": "Which team?", "type": "choice", "labels": ["A", "B"], "answer": "B"})
    assert a.post("/api/typically/models/alice_m/retrain", json={}).status_code == 429
    assert not (env / "jobs" / "alice_m_v2").exists()


def test_share_card(env):
    a, b, anon = client(ALICE), client(BOB), client()
    assert a.get("/api/typically/models/alice_m/share").json() == {"share": None}
    assert a.post("/api/typically/models/alice_two/share").status_code == 400   # not scored yet
    assert b.post("/api/typically/models/alice_m/share").status_code == 404   # not Bob's
    s = a.post("/api/typically/models/alice_m/share").json()["share"]
    assert s["url"].startswith("http://localhost/s/") and s["image"] == s["url"] + "/og.png"
    assert a.post("/api/typically/models/alice_m/share").json()["share"]["id"] == s["id"]   # idempotent
    assert a.get("/api/typically/models/alice_m/share").json()["share"]["id"] == s["id"]

    page = anon.get(f"/s/{s['id']}")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    body = page.text
    assert "55%" in body and "80%" in body and "85%" in body and "40 held-out cases" in body and "Which team?" in body
    assert f'<meta property="og:image" content="{s["url"]}/og.png">' in body and 'property="og:title"' in body
    assert "<script>alert" not in body and "&lt;script&gt;" in body   # the model name is escaped
    assert SECRET_CASE not in body and "4111" not in body and auth.uid_of(ALICE) not in body   # aggregates only

    png = anon.get(f"/s/{s['id']}/og.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(png.content)).size == (1200, 630)

    assert b.delete("/api/typically/models/alice_m/share").status_code == 404
    assert a.delete("/api/typically/models/alice_m/share").json() == {"revoked": 1}
    assert anon.get(f"/s/{s['id']}").status_code == 404 and anon.get(f"/s/{s['id']}/og.png").status_code == 404


def test_review_fixes(env, monkeypatch):
    """decide checks ownership, health lists no ids, reserved and over-long names are refused, hf never reads a server dir, push never overwrites."""
    import typically_deploy as D
    import typically_plan as tp
    bob = client(BOB)
    assert bob.post("/api/decide", json={"model": "local:co_alice_m", "state": "x", "queries": []}).status_code == 404
    assert "models" not in client().get("/api/health").json()
    for name in ("Northwind", "f"):
        assert bob.post("/api/typically/build", json={"records_token": "0" * 32, "plan": {}, "name": name}).status_code == 409, name
    assert bob.post("/api/typically/build", json={"records_token": "0" * 32, "plan": {}, "name": "x" * 81}).status_code == 422
    with pytest.raises(ValueError, match="not a Hugging Face dataset id"):
        tp.load_records({"kind": "hf", "dataset": str(env)})

    class Hub:   # the repo exists and this model never pushed to it
        def __init__(self, token=None): pass
        def model_info(self, repo): return type("I", (), {"sha": "s"})()
        def repo_exists(self, repo): return True
        def create_repo(self, *a, **k): raise AssertionError("must not touch an existing repo")
    monkeypatch.setattr(D, "HfApi", Hub)
    monkeypatch.setattr(D, "_hf_token", lambda: "tok")
    monkeypatch.setattr(D.torch, "load", lambda *a, **k: {"args": {"backbone": tj.RELEASED_ARGS["small"]["backbone"]}, "step": 1})
    r = client(ALICE).post("/api/typically/push", json={"run": "co_alice_m", "repo": "OzLabs/typical-small"})
    assert r.status_code == 409 and "already exists" in r.json()["detail"]
