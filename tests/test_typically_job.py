"""scripts/typically_job.py orchestration + the /api/typically/train endpoints, with the subprocess layer faked. No network, no pods.
uv run --with pytest --with fastapi --with httpx pytest tests/test_typically_job.py -q"""
import json
import re
import sys
import tarfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "site"), str(ROOT / "scripts")]
import server  # noqa: E402  (also puts inference/ + scripts/ on sys.path)
import typically_job as tj  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SLUG, TOKEN = "acme", "hf_SECRET_TOKEN"


class FakeCli:
    """Stands in for typically_job._run: records argv, simulates runpodctl / ssh / scp / git."""

    def __init__(self, polls=(), fail_on=None, pods=(), on_get=None):
        self.calls, self.polls, self.fail_on, self.pods, self.on_get = [], list(polls), fail_on, list(pods), on_get
        self.tar_names = []

    def __call__(self, argv, timeout, out=None):
        self.calls.append(argv)
        prog, sub = argv[0], (argv[2] if argv[0] == "runpodctl" else "")
        if self.fail_on and self.fail_on(argv):
            raise RuntimeError("boom")
        if prog == "runpodctl" and sub == "create":
            self.pods.append({"id": "pod1", "name": argv[argv.index("--name") + 1]})
            return json.dumps({"id": "pod1"})
        if prog == "runpodctl" and sub == "list":
            return json.dumps(self.pods)
        if prog == "runpodctl" and sub == "get":
            self.on_get and self.on_get()
            return json.dumps({"ssh": {"ip": "1.2.3.4", "port": 2222}})
        if prog == "runpodctl" and sub == "delete":
            self.pods = [p for p in self.pods if p["id"] != argv[3]]
            return ""
        if prog == "git":
            return "README.md\nsite/server.py\nscripts/typically_job.py\n"
        if prog == "ssh":
            return self.polls.pop(0) if "job.pid" in argv[-1] and "kill" in argv[-1] and self.polls else ""
        if prog == "scp":
            src = argv[-2]
            if src.endswith("repo.tar.gz"):
                self.tar_names = tarfile.open(src).getnames()
            if src.startswith("root@"):   # download: create the file the caller expects
                Path(argv[-1]).write_text("x")
            return ""
        raise AssertionError(argv)

    def deleted(self):
        return [a for a in self.calls if a[:3] == ["runpodctl", "pod", "delete"]]

    def deleted_ids(self):
        return {a[3] for a in self.deleted()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".cache/huggingface").mkdir(parents=True)
    (home / ".cache/huggingface/token").write_text(TOKEN)
    monkeypatch.setenv("HOME", str(home))
    (tmp_path / "key.pub").write_text("ssh-ed25519 AAAA test")
    monkeypatch.setattr(tj, "KEY", tmp_path / "key")
    monkeypatch.setattr(tj, "TYPICALLY", tmp_path / "typ")
    monkeypatch.setattr(server, "TYPICALLY", tmp_path / "typ")
    jd = tmp_path / "typ" / "jobs" / SLUG
    (jd / "eval").mkdir(parents=True)
    for f in ("train.jsonl", "val.jsonl", "eval/import_oneliner.jsonl"):
        (jd / f).write_text("{}\n")
    monkeypatch.setattr(tj, "_ACTIVE", set())
    monkeypatch.setattr(tj.time, "sleep", lambda s: None)
    monkeypatch.setattr(tj, "POLL_S", 0)
    history = []
    real = tj.write_status
    monkeypatch.setattr(tj, "write_status", lambda *a, **k: history.append(a[1]) or real(*a, **k))
    return history


def use(monkeypatch, cli):
    monkeypatch.setattr(tj, "_run", cli)
    return cli


ALIVE = "ALIVE\n"
POLLS = [ALIVE + "step 200 bucket C loss 0.3\n", ALIVE + "step 400 val_nll 0.3\nruns/co_acme best step 350\n",
         "EXIT 0\nJOB_DONE\n"]


def put_status(slug, **kw):
    d = tj.job_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps(kw))


def test_success_phases_download_and_delete(env, monkeypatch, tmp_path):
    cli = use(monkeypatch, FakeCli(POLLS))
    logs = []
    tj.run(SLUG, logs.append)
    phases = [p for i, p in enumerate(env) if i == 0 or p != env[i - 1]]
    assert phases == ["queued", "starting_gpu", "uploading", "training", "evaluating", "downloading", "done"]
    assert len(cli.deleted()) == 1 and cli.pods == []
    st = tj.read_status(SLUG)
    assert st["phase"] == "done" and st["pod_id"] == "pod1"
    created = next(a for a in cli.calls if a[:3] == ["runpodctl", "pod", "create"])
    assert re.fullmatch(rf"typically-job-{SLUG}-[0-9a-f]{{8}}", created[created.index("--name") + 1])
    assert created[created.index("--name") + 1].endswith(st["job_id"])
    res = tmp_path / "typ/results/co_acme"
    assert {p.name for p in res.iterdir()} == {"best.pt", "eval_co.json", "base_eval_co.json"}
    assert "data_co_acme/train.jsonl" in cli.tar_names and "data_co_acme/eval/import_oneliner.jsonl" in cli.tar_names
    assert "README.md" in cli.tar_names and not any(n.startswith("site/") for n in cli.tar_names)
    assert not any(TOKEN in " ".join(a) for a in cli.calls) and not any(TOKEN in m for m in logs)


def test_names_are_unique_and_prefix_pods_are_never_touched(env, monkeypatch):
    # another live job of a similarly named slug: same "typically-job-acme" prefix, active in this process
    put_status("acme2", phase="training", job_id="cafe1234", pod_id="podX", updated_at=tj._now())
    tj._ACTIVE.add("cafe1234")
    cli = use(monkeypatch, FakeCli(POLLS, pods=[{"id": "podX", "name": "typically-job-acme2-cafe1234"}]))
    tj.run(SLUG, lambda m: None)
    tj.run(SLUG, lambda m: None)
    names = [a[a.index("--name") + 1] for a in cli.calls if a[:3] == ["runpodctl", "pod", "create"]]
    assert len(set(names)) == 2   # two runs of one slug never share a pod name
    assert cli.deleted_ids() == {"pod1"} and [p["id"] for p in cli.pods] == ["podX"]


def test_unparseable_create_deletes_only_the_exact_name(env, monkeypatch):
    class Cli(FakeCli):
        def __call__(self, argv, timeout, out=None):
            if argv[:3] == ["runpodctl", "pod", "create"]:
                self.calls.append(argv)
                self.pods.append({"id": "pod9", "name": argv[argv.index("--name") + 1]})   # created, but the reply is garbage
                return "not json"
            return super().__call__(argv, timeout, out)
    put_status("acme2", phase="training", job_id="cafe1234", updated_at=tj._now())
    tj._ACTIVE.add("cafe1234")
    cli = use(monkeypatch, Cli(pods=[{"id": "podX", "name": "typically-job-acme2-cafe1234"}]))
    tj.run(SLUG, lambda m: None)
    assert tj.read_status(SLUG)["phase"] == "failed"
    assert cli.deleted_ids() == {"pod9"} and [p["id"] for p in cli.pods] == ["podX"]


def test_reconcile_deletes_stale_and_unknown_but_not_other_pods(env, monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(minutes=100)).isoformat(timespec="seconds")
    put_status("stale", phase="training", job_id="aaaa1111", pod_id="p_stale", updated_at=old)
    put_status("live", phase="training", job_id="bbbb2222", pod_id="p_live", updated_at=tj._now())
    put_status("orphan", phase="training", job_id="cccc3333", pod_id="p_orphan", updated_at=tj._now())   # active phase, another process
    put_status("finished", phase="done", job_id="eeee5555", pod_id="p_done", updated_at=tj._now())
    tj._ACTIVE.update({"aaaa1111", "bbbb2222", "eeee5555"})
    pod = lambda i, n: {"id": i, "name": n}
    cli = use(monkeypatch, FakeCli(pods=[pod("p_stale", "typically-job-stale-aaaa1111"), pod("p_live", "typically-job-live-bbbb2222"),
                                         pod("p_orphan", "typically-job-orphan-cccc3333"), pod("p_done", "typically-job-finished-eeee5555"),
                                         pod("p_ghost", "typically-job-ghost-dddd4444"), pod("p_other", "someone-elses-pod"),
                                         pod("p_prefix", "typically-spike-1")]))
    logs = []
    assert set(tj.reconcile(logs.append)) == {"p_stale", "p_orphan", "p_done", "p_ghost"}
    assert cli.deleted_ids() == {"p_stale", "p_orphan", "p_done", "p_ghost"} and len(logs) == 4
    assert {p["id"] for p in cli.pods} == {"p_live", "p_other", "p_prefix"}


def test_run_reconciles_before_creating_a_pod(env, monkeypatch):
    cli = use(monkeypatch, FakeCli(POLLS, pods=[{"id": "leak", "name": "typically-job-old-00000000"}]))
    tj.run(SLUG, lambda m: None)
    kinds = [a[2] for a in cli.calls if a[0] == "runpodctl"]
    assert kinds.index("delete") < kinds.index("create") and "leak" in cli.deleted_ids()


def test_snapshot_is_taken_before_the_pod_exists(env, monkeypatch):
    cli = use(monkeypatch, FakeCli(POLLS))
    tj.run(SLUG, lambda m: None)
    order = [a[0] if a[0] != "runpodctl" else a[2] for a in cli.calls]
    assert order.index("git") < order.index("create")


def test_bad_slug_rejected(env):
    for bad in ("../x", "Acme", "", "a" * 41, "a-b"):
        with pytest.raises(ValueError):
            tj.run(bad, lambda m: None)


def test_deadline_raises():
    j = tj.Job(SLUG, lambda m: None, Path("."))
    j.deadline = time.monotonic() - 1
    with pytest.raises(tj.Deadline):
        j.call(["true"], 10)


def test_calls_are_capped_by_the_remaining_budget(env, monkeypatch):
    seen = []
    monkeypatch.setattr(tj, "_run", lambda argv, timeout, out=None: seen.append(timeout) or "")
    j = tj.Job(SLUG, lambda m: None, Path("."))
    j.call(["true"], 10_000)
    j.deadline = time.monotonic() + 5
    j.call(["true"], 10_000)
    assert seen[0] <= tj.DEADLINE_S and seen[1] <= 5


def test_deadline_mid_run_fails_and_still_deletes_pod(env, monkeypatch):
    monkeypatch.setattr(tj, "DEADLINE_S", 0.5)
    cli = use(monkeypatch, FakeCli(POLLS, on_get=lambda: threading.Event().wait(0.6)))   # the budget runs out while the pod boots
    tj.run(SLUG, lambda m: None)
    assert tj.read_status(SLUG)["phase"] == "failed" and "75 minutes" in tj.read_status(SLUG)["message"]
    assert cli.deleted_ids() == {"pod1"} and cli.pods == []


def test_poll_timeout_deletes_pod(env, monkeypatch):
    monkeypatch.setattr(tj, "DEADLINE_S", 0.3)
    monkeypatch.setattr(tj.time, "sleep", lambda s: time.monotonic() and threading.Event().wait(0.01))
    cli = use(monkeypatch, FakeCli([ALIVE + "step 10 x\n"] * 10_000))
    tj.run(SLUG, lambda m: None)
    assert tj.read_status(SLUG)["phase"] == "failed" and "75 minutes" in tj.read_status(SLUG)["message"]
    assert len(cli.deleted()) == 1 and cli.pods == []


def test_exception_mid_upload_deletes_pod(env, monkeypatch):
    cli = use(monkeypatch, FakeCli(POLLS, fail_on=lambda a: a[0] == "scp" and a[-2].endswith("repo.tar.gz")))
    tj.run(SLUG, lambda m: None)
    assert tj.read_status(SLUG)["phase"] == "failed" and env[-1] == "failed"
    assert not any(a[0] == "ssh" and "job.sh" in a[-1] for a in cli.calls)   # never launched
    assert len(cli.deleted()) == 1 and cli.pods == []


def test_dead_chain_without_exit_code_is_a_failure(env, monkeypatch):
    cli = use(monkeypatch, FakeCli([ALIVE + "step 50 x\n", "DEAD\nstep 50 x\nJOB_DONE\n"]))
    tj.run(SLUG, lambda m: None)
    assert tj.read_status(SLUG)["phase"] == "failed" and len(cli.deleted()) == 1


def test_nonzero_exit_fails_even_if_the_log_says_done(env, monkeypatch):
    cli = use(monkeypatch, FakeCli([ALIVE + "step 50 x\n", "EXIT 1\nTraceback (most recent call last)\nJOB_DONE\n"]))
    tj.run(SLUG, lambda m: None)
    st = tj.read_status(SLUG)
    assert st["phase"] == "failed" and "exit 1" in st["message"] and "traceback" in st["message"] and len(cli.deleted()) == 1


def test_traceback_in_log_alone_does_not_fail_a_zero_exit(env, monkeypatch):
    use(monkeypatch, FakeCli([ALIVE + "Traceback (a warning)\n", "EXIT 0\nTraceback (a warning)\n"]))
    tj.run(SLUG, lambda m: None)
    assert tj.read_status(SLUG)["phase"] == "done"


def test_train_endpoint_409_and_404(env, monkeypatch):
    c = TestClient(server.app)
    assert c.post("/api/typically/train", json={"name": "nope"}).status_code == 404
    assert c.post("/api/typically/train", json={"name": "x" * 60}).status_code == 404   # slug over 40 chars
    gate = threading.Event()
    monkeypatch.setattr(tj, "run", lambda slug, log: gate.wait(5))
    r = c.post("/api/typically/train", json={"name": "Acme"})
    assert r.status_code == 200 and r.json()["phase"] == "queued"
    assert c.post("/api/typically/train", json={"name": "Acme"}).status_code == 409
    build = {"csv_text": "t,l\nx,y\n", "text_col": "t", "decisions": [{"column": "l", "question": "q", "type": "choice"}]}
    assert c.post("/api/typically/build", json={**build, "name": "Acme"}).status_code == 409   # its job is active
    assert c.post("/api/typically/build", json={**build, "name": "other"}).status_code != 409
    gate.set()
    for _ in range(100):   # the lock is released when the job thread ends
        if not server._job_lock.locked():
            break
        time.sleep(0.02)
    assert c.post("/api/typically/train", json={"name": "Acme"}).status_code == 200
    gate.set()


def test_status_agreement(env, tmp_path):
    ev = lambda acc_a, acc_b: {"eval": {"import_oneliner": {"n": 30, "raw": {"acc": (acc_a * 20 + acc_b * 10) / 30},
                                                            "by_family": {"import_team": {"acc": acc_a, "n": 20}, "import_esc": {"acc": acc_b, "n": 10}}}}}
    res = tmp_path / "typ/results/co_acme"
    res.mkdir(parents=True)
    (res / "base_eval_co.json").write_text(json.dumps(ev(0.5, 0.7)))
    (res / "eval_co.json").write_text(json.dumps(ev(0.9, 0.8)))
    c = TestClient(server.app)
    tj.write_status(SLUG, "training", "x")
    assert "agreement" not in c.get(f"/api/typically/train/{SLUG}").json()
    tj.write_status(SLUG, "done", "ok")
    a = c.get(f"/api/typically/train/{SLUG}").json()["agreement"]
    assert a["n"] == 30 and a["decisions"]["team"] == {"base": 0.5, "yours": 0.9, "n": 20}
    assert a["overall"] == {"base": pytest.approx(0.5667, abs=1e-3), "yours": pytest.approx(0.8667, abs=1e-3)}
    assert c.get("/api/typically/train/none").status_code == 404


def test_cors_is_local_only():
    c = TestClient(server.app)
    hdr = lambda o: c.get("/api/health", headers={"Origin": o}).headers.get("access-control-allow-origin")
    assert hdr("http://127.0.0.1:8787") == "http://127.0.0.1:8787" and hdr("https://evil.example") is None
