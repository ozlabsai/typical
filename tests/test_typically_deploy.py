"""site/typically_deploy.py (keys, protected decide, snippets, push to HF) + Typical.activate (LoRA swap). No model, no network.
uv run --no-sync --with pytest --with httpx pytest tests/test_typically_deploy.py -q"""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "site"), str(ROOT / "scripts")]
import server  # noqa: E402  (first: typically_deploy reads server at call time)
import typically_deploy as td  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from typical.core import Typical  # noqa: E402

TOKEN = "hf_SECRET_TOKEN"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "TYPICALLY", tmp_path)
    td._hits.clear()
    return TestClient(server.app)


def make_key(c, model_id="northwind"):
    return c.post("/api/typically/keys", json={"model_id": model_id}).json()["key"]


def trained(tmp_path, run="co_f", backbone="Qwen/Qwen3-1.7B-Base"):
    (tmp_path / "results" / run).mkdir(parents=True, exist_ok=True)
    torch.save({"step": 350, "args": {"backbone": backbone}}, tmp_path / "results" / run / "best.pt")


# -- keys
def test_key_is_shown_once_and_only_its_hash_is_stored(env, tmp_path):
    r = env.post("/api/typically/keys", json={"model_id": "northwind"}).json()
    assert r["key"].startswith("tpk_") and r["prefix"] == r["key"][:8] and len(r["key"]) > 30
    f = tmp_path / "keys.json"
    assert r["key"] not in f.read_text() and stat.S_IMODE(f.stat().st_mode) == 0o600
    rec = json.loads(f.read_text())[hashlib.sha256(r["key"].encode()).hexdigest()]
    assert rec["model_id"] == "northwind" and rec["created_at"]
    assert make_key(env) != r["key"] and len(json.loads(f.read_text())) == 2
    assert env.post("/api/typically/keys", json={"model_id": "../x"}).status_code == 404


# -- decide
def fake_model(monkeypatch):
    seen = []
    monkeypatch.setattr(server, "get_model", lambda name: seen.append(name) or name)
    monkeypatch.setattr(server, "_run", lambda m, state, queries: {"results": [{"argmax": "Claims"}], "ms": 1.5, "device": "mps"})
    return seen


BODY = {"state": "a case", "queries": [{"type": "choice", "question": "q?", "labels": ["a", "b"]}]}


def test_decide_401_without_a_valid_key_for_this_model(env, tmp_path, monkeypatch):
    fake_model(monkeypatch)
    trained(tmp_path, "co_f")
    trained(tmp_path, "co_acme")
    url = "/v1/models/northwind/decide"
    assert env.post(url, json=BODY).status_code == 401
    assert env.post(url, json=BODY, headers={"Authorization": "Bearer tpk_nope"}).status_code == 401
    assert env.post(url, json=BODY, headers={"Authorization": "tpk_nope"}).status_code == 401
    assert env.post(url, json=BODY, headers={"Authorization": "Bearer "}).status_code == 401
    other = make_key(env, "acme")   # a real key, but for another model
    r = env.post(url, json=BODY, headers={"Authorization": f"Bearer {other}"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_decide_runs_the_tuned_model_and_maps_northwind_to_co_f(env, tmp_path, monkeypatch):
    seen = fake_model(monkeypatch)
    trained(tmp_path, "co_f")
    trained(tmp_path, "co_acme")
    kn, ka = make_key(env), make_key(env, "acme")
    r = env.post("/v1/models/northwind/decide", json=BODY, headers={"Authorization": f"Bearer {kn}"})
    assert r.status_code == 200 and r.json() == {"model": "northwind", "results": [{"argmax": "Claims"}], "ms": 1.5, "device": "mps"}
    assert env.post("/v1/models/acme/decide", json=BODY, headers={"Authorization": f"Bearer {ka}"}).status_code == 200
    assert seen == ["local:co_f", "local:co_acme"]


def test_decide_404_when_not_trained_and_429_over_the_rate_limit(env, tmp_path, monkeypatch):
    fake_model(monkeypatch)
    k = make_key(env, "ghost")
    h = {"Authorization": f"Bearer {k}"}
    assert env.post("/v1/models/ghost/decide", json=BODY, headers=h).status_code == 404
    trained(tmp_path, "co_ghost")
    monkeypatch.setattr(td, "RATE", 3)
    codes = [env.post("/v1/models/ghost/decide", json=BODY, headers=h).status_code for _ in range(5)]
    assert codes == [200, 200, 429, 429, 429]   # every authorised request counts, including the 404 above


# -- snippets
def test_northwind_snippets_are_prefilled(env):
    r = env.get("/api/typically/snippets/northwind")
    assert r.status_code == 200
    s = r.json()
    assert set(s) == {"curl", "python", "javascript", "sdk"}
    for kind in ("curl", "python", "javascript"):
        assert "http://testserver/v1/models/northwind/decide" in s[kind] and "TYPICAL_API_KEY" in s[kind]
        assert "Which team should handle this ticket?" in s[kind] and "Claims" in s[kind] and "Cedar Foods" in s[kind]
        assert "Can the agent refund this without approval?" in s[kind]
    assert "tpk_" not in "".join(s.values())   # the key is never in a snippet
    assert "pip install typical-ai" in s["sdk"] and "from typical_ai import Typical" in s["sdk"] and "HF_TOKEN" in s["sdk"]
    assert "Typical.from_pretrained('<hf-user>/<model-name>')" in s["sdk"] and "m.noul(case, 'Should we escalate this to a manager?')" in s["sdk"]
    body = s["curl"].split("-d '", 1)[1].removesuffix("'")
    assert json.loads(body)["queries"][0]["labels"] == ["Claims", "Billing", "Dispatch", "Sales"]   # the curl payload is valid JSON
    assert json.loads(s["python"].split("json=", 1)[1].rsplit(",\n)", 1)[0])["state"].startswith("Customer: Cedar Foods")


def test_snippets_use_the_saved_plan_and_escape_shell_quotes(env, tmp_path):
    jd = tmp_path / "jobs" / "acme"
    jd.mkdir(parents=True)
    (jd / "plan.json").write_text(json.dumps({"decisions": [
        {"column": "a", "include": True, "question": "Is it Bob's?", "type": "noul", "labels": ["no", "yes"]},
        {"column": "b", "include": False, "question": "hidden?", "type": "choice", "labels": ["x", "y"]}]}))
    (jd / "val.jsonl").write_text(json.dumps({"state": "it's a case"}) + "\n")
    s = env.get("/api/typically/snippets/acme").json()
    assert "hidden?" not in s["curl"] and "Bob" in s["python"] and "it's a case" in s["python"]
    assert "'\\''" in s["curl"]   # ' -> '\'' inside the single-quoted -d body
    assert env.get("/api/typically/snippets/Bad-Id").status_code == 404


# -- push to Hugging Face
@pytest.fixture
def hub(monkeypatch):
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(sha="basesha123")
    api.whoami.return_value = {"name": "me", "orgs": [{"name": "OzLabs"}]}
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    td._namespaces.clear()
    cls = MagicMock(return_value=api)
    monkeypatch.setattr(td, "HfApi", cls)
    return cls, api


def push(c, **kw):
    return c.post("/api/typically/push", json={"run": "co_f", "repo": "me/northwind-triage", **kw})


def test_push_is_private_by_default_and_uploads_weights_card_and_manifest(env, tmp_path, hub):
    cls, api = hub
    trained(tmp_path, "co_f")
    (tmp_path / "results" / "co_f" / "reveal.json").write_text(json.dumps({
        "n_cases": 200, "score": {"standard": 0.57, "yours": 0.78},
        "decisions": [{"question": "Which team should handle this ticket?", "standard": 0.6, "yours": 0.86}]}))
    r = push(env)
    assert r.status_code == 200 and r.json() == {"url": "https://huggingface.co/me/northwind-triage", "files": ["best.pt", "README.md", "MANIFEST.json"]}
    cls.assert_called_once_with(token=TOKEN)   # the server's token, never one from the request
    api.create_repo.assert_called_once_with("me/northwind-triage", private=True, exist_ok=True)
    ops = {o.path_in_repo: o.path_or_fileobj for o in api.create_commit.call_args.kwargs["operations"]}
    assert set(ops) == {"best.pt", "README.md", "MANIFEST.json"} and ops["best.pt"] == str(tmp_path / "results/co_f/best.pt")
    card = ops["README.md"].decode()
    assert card.startswith("---\nlicense: apache-2.0\nbase_model: OzLabs/typical-small\nlibrary_name: pytorch\ntags:\n  - typical\n  - decision-model\n---")
    assert "Which team should handle this ticket?" in card and "| 60% | 86% |" in card and "200 held-out cases" in card
    assert "Typical.from_pretrained('me/northwind-triage')" in card
    m = json.loads(ops["MANIFEST.json"])
    assert m == {"base_repo": "OzLabs/typical-small", "base_sha": "basesha123", "run": "co_f", "step": 350,
                 "best_pt_sha256": hashlib.sha256((tmp_path / "results/co_f/best.pt").read_bytes()).hexdigest()}
    assert env.get("/api/typically/snippets/northwind").json()["sdk"].count("me/northwind-triage")   # the pushed repo feeds the SDK snippet


def test_push_public_medium_and_the_token_is_never_stored(env, tmp_path, hub):
    cls, api = hub
    trained(tmp_path, "co_acme", "Qwen/Qwen3.5-4B-Base")
    assert push(env, run="co_acme", private=False).status_code == 200
    assert api.create_repo.call_args.kwargs["private"] is False
    card = next(o for o in api.create_commit.call_args.kwargs["operations"] if o.path_in_repo == "README.md").path_or_fileobj.decode()
    assert "base_model: OzLabs/typical-medium" in card and "license: other" in card
    assert not any(TOKEN in p.read_text(errors="ignore") for p in tmp_path.rglob("*") if p.is_file() and p.name != "best.pt")


def test_push_errors_never_echo_the_token(env, tmp_path, hub):
    _, api = hub
    trained(tmp_path, "co_f")
    api.create_commit.side_effect = RuntimeError(f"401 for Authorization: Bearer {TOKEN}")
    r = push(env)
    assert r.status_code == 502 and TOKEN not in r.text and "***" in r.text
    assert push(env, repo="no-slash").status_code == 400 and push(env, run="co_missing").status_code == 404


def test_push_card_skips_results_scored_against_another_base(env, tmp_path, hub):
    _, api = hub
    trained(tmp_path, "co_acme", "Qwen/Qwen3.5-4B-Base")
    reveal = {"n_cases": 200, "score": {"standard": 0.57, "yours": 0.78}, "decisions": []}
    card = lambda: next(o for o in api.create_commit.call_args.kwargs["operations"] if o.path_in_repo == "README.md").path_or_fileobj.decode()
    for base, shown in (("typical-small", False), ("typical-medium", True)):
        (tmp_path / "results" / "co_acme" / "reveal.json").write_text(json.dumps({**reveal, "base": base}))
        assert push(env, run="co_acme").status_code == 200 and ("Held-out results" in card()) is shown


def test_compare_base_is_the_base_the_tuned_model_started_from(env, tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(server, "get_model", lambda n: n)
    monkeypatch.setattr(server, "_run", lambda m, state, qs: ran.append(m) or {"results": []})
    trained(tmp_path, "co_small")
    trained(tmp_path, "co_acme", "Qwen/Qwen3.5-4B-Base")
    body = lambda *models: {"state": "s", "decisions": [], "models": list(models)}
    r = env.post("/api/typically/compare", json=body("base", "local:co_acme")).json()
    assert r["base"] == "typical-medium" and r["models"]["base"]["model"] == "typical-medium" and ran == ["typical-medium", "local:co_acme"]
    r = env.post("/api/typically/compare", json=body("local:co_small", "base")).json()
    assert r["base"] == "typical-small" and r["models"]["base"]["model"] == "typical-small"
    r = env.post("/api/typically/compare", json=body("typical-small", "local:co_acme")).json()   # explicit names still work
    assert r["base"] is None and list(r["models"]) == ["typical-small", "local:co_acme"]
    assert env.post("/api/typically/compare", json=body("base")).status_code == 400
    assert env.post("/api/typically/compare", json=body("base", "local:nope")).status_code == 400


def test_get_model_reloads_after_the_checkpoint_changes(env, tmp_path, monkeypatch):
    loads = []

    def load(source, **kw):
        loads.append(source)
        return SimpleNamespace(activate=lambda: None, tag=len(loads))
    monkeypatch.setattr(server.Typical, "from_pretrained", load)
    monkeypatch.setattr(server, "_warm", lambda m: None)
    monkeypatch.setattr(server, "_cache", {})
    trained(tmp_path, "co_f")
    best = tmp_path / "results/co_f/best.pt"
    assert server.get_model("local:co_f").tag == 1 and server.get_model("local:co_f").tag == 1 and len(loads) == 1   # cached
    os.utime(best, ns=(1, best.stat().st_mtime_ns + 10**9))   # a retrain replaced best.pt
    assert server.get_model("local:co_f").tag == 2 and server.get_model("local:co_f").tag == 2 and len(loads) == 2
    assert list(server._cache) == ["local:co_f"]   # the stale entry is gone, not kept beside the new one


# -- adapter swap
def test_activate_swaps_the_lora_in_place_only_when_needed():
    loaded = []
    bb = SimpleNamespace(load_lora_state_dict=lambda sd: loaded.append(sd))
    a, b = (Typical.__new__(Typical) for _ in range(2))
    for m, tag in ((a, "A"), (b, "B")):
        m.head, m.lora = SimpleNamespace(backbone=bb), {"tag": tag}
    a.activate(); a.activate(); b.activate(); a.activate()
    assert [x["tag"] for x in loaded] == ["A", "B", "A"]   # the repeated call did nothing
    solo = Typical.__new__(Typical)
    solo.head, solo.lora = SimpleNamespace(backbone=bb), None   # not sharing a backbone: activate is a no-op
    solo.activate()
    assert len(loaded) == 3


def test_push_default_repo_is_namespace_slash_model_id_and_ignores_a_token_field(env, tmp_path, hub):
    _, api = hub
    trained(tmp_path, "co_acme_co")
    r = env.post("/api/typically/push", json={"run": "co_acme_co", "token": "hf_from_the_request"})   # extra field: ignored
    assert r.status_code == 200 and r.json()["url"] == "https://huggingface.co/OzLabs/acme-co"
    api.create_repo.assert_called_once_with("OzLabs/acme-co", private=True, exist_ok=True)
    assert (tmp_path / "results" / "co_acme_co" / "hf_repo.txt").read_text() == "OzLabs/acme-co"


def test_push_without_a_server_token_is_503(env, tmp_path, hub, monkeypatch):
    trained(tmp_path, "co_f")
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setattr(td, "get_token", lambda: None)
    assert push(env).status_code == 503


def test_capabilities(env, hub, monkeypatch):
    _, api = hub
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert env.get("/api/typically/capabilities").json() == {"ai": True, "hf_namespace": "OzLabs", "languages": ["en", "he"]}
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")   # either key turns the AI features on
    assert env.get("/api/typically/capabilities").json()["ai"] is True
    monkeypatch.delenv("OPENROUTER_API_KEY")
    td._namespaces.clear()
    api.whoami.return_value = {"name": "me", "orgs": [{"name": "Other"}]}   # not in the org: the token owner's own name
    assert env.get("/api/typically/capabilities").json() == {"ai": False, "hf_namespace": "me", "languages": ["en", "he"]}
    env.get("/api/typically/capabilities")
    assert api.whoami.call_count == 2   # cached per token
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setattr(td, "get_token", lambda: None)
    assert env.get("/api/typically/capabilities").json()["hf_namespace"] is None


def test_push_default_repo_is_the_display_name_slug(env, tmp_path, hub):
    _, api = hub
    trained(tmp_path, "co_f")
    trained(tmp_path, "co_acme")
    (tmp_path / "jobs" / "acme").mkdir(parents=True)
    (tmp_path / "jobs" / "acme" / "job.json").write_text(json.dumps({"name": "Acme  Support Triage!"}))
    assert env.post("/api/typically/push", json={"run": "co_f"}).json()["url"].endswith("/OzLabs/northwind-triage")
    assert env.post("/api/typically/push", json={"run": "co_acme"}).json()["url"].endswith("/OzLabs/acme-support-triage")
    (tmp_path / "jobs" / "acme" / "job.json").write_text(json.dumps({"name": "מיון פניות"}))   # no latin letters: the id
    assert env.post("/api/typically/push", json={"run": "co_acme"}).json()["url"].endswith("/OzLabs/acme")


def test_failed_upload_deletes_only_a_repo_this_push_created_and_left_empty(env, tmp_path, hub):
    _, api = hub
    trained(tmp_path, "co_f")
    api.create_commit.side_effect = RuntimeError("upload broke")
    api.list_repo_files.return_value = [".gitattributes"]
    for existed, files, deleted in ((False, [".gitattributes"], True), (True, [".gitattributes"], False), (False, ["best.pt"], False)):
        api.repo_exists.return_value, api.list_repo_files.return_value = existed, files
        api.delete_repo.reset_mock()
        assert push(env).status_code == 502
        assert api.delete_repo.called is deleted
    api.delete_repo.assert_not_called()
    api.repo_exists.return_value = False
    api.create_repo.side_effect = RuntimeError("could not create")   # nothing was created: nothing to delete
    assert push(env).status_code == 502 and not api.delete_repo.called


class StorageFull(Exception):
    response = SimpleNamespace(status_code=403)


def test_org_storage_full_is_507_and_offers_the_users_namespace(env, tmp_path, hub):
    _, api = hub
    trained(tmp_path, "co_f")
    api.repo_exists.return_value = False
    api.list_repo_files.return_value = [".gitattributes"]
    api.create_commit.side_effect = StorageFull(f"403 Forbidden: Private repository storage limit reached (Bearer {TOKEN})")
    r = push(env, repo="OzLabs/northwind-triage")
    d = r.json()["detail"]
    assert r.status_code == 507 and d["code"] == "org_storage_full" and d["repo"] == "me/northwind-triage"
    assert "storage" in d["message"] and "(me)" in d["message"] and TOKEN not in r.text
    api.delete_repo.assert_called_once_with("OzLabs/northwind-triage")   # the empty org repo it just made is gone
    assert push(env, repo="me/northwind-triage").status_code == 502   # the user's own space full: nothing better to offer
