"""scripts/helipad_score_pod.py with runpodctl / ssh / scp / git faked at typically_job._run. No network, no pods.
uv run pytest tests/test_helipad_score_pod.py -q"""
import json
import os
import signal
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import helipad_score_pod as hs  # noqa: E402
import typically_job as tj  # noqa: E402

SECRETS = ("hf_SECRET_TOKEN", "wandb_SECRET", "PRIVATE KEY BODY")


class FakeCli:
    """Stands in for typically_job._run. `create`: what the create call does after the pod exists (return its stdout or raise)."""

    def __init__(self, create=lambda: json.dumps({"id": "pod1"}), fail_on=None, delete_fails=0):
        self.calls, self.pods, self.create, self.fail_on, self.delete_fails = [], [], create, fail_on, delete_fails
        self.uploaded = []

    def __call__(self, argv, timeout, out=None):
        self.calls.append(argv)
        if self.fail_on and self.fail_on(argv):
            raise RuntimeError("boom")
        prog, sub = argv[0], (argv[2] if argv[0] == "runpodctl" else "")
        if sub == "create":
            self.pods.append({"id": "pod1", "name": argv[argv.index("--name") + 1]})
            return self.create()
        if sub == "list":
            return json.dumps(self.pods)
        if sub == "get":
            return json.dumps({"ssh": {"ip": "1.2.3.4", "port": 2222}})
        if sub == "delete":
            if self.delete_fails:
                self.delete_fails -= 1
                raise RuntimeError("runpodctl pod delete failed (1): 502")
            self.pods = [p for p in self.pods if p["id"] != argv[3]]
            return ""
        if prog == "git":
            return "README.md\nscripts/typically_job.py\nsite/server.py\n"
        if prog == "ssh":
            return ""
        if prog == "scp":
            if argv[-2].startswith("root@"):   # download
                Path(argv[-1]).write_text('{"ok": 1}')
            else:
                self.uploaded.append(tarfile.open(argv[-2]).getnames())
            return ""
        raise AssertionError(argv)

    def deleted(self):
        return [a[3] for a in self.calls if a[:3] == ["runpodctl", "pod", "delete"]]


@pytest.fixture
def env(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("-----BEGIN-----\nPRIVATE KEY BODY\n")
    Path(f"{key}.pub").write_text("ssh-ed25519 AAAApublic me\n")
    monkeypatch.setattr(tj, "KEY", key)
    monkeypatch.setattr(tj, "TYPICALLY", tmp_path / "typ")
    monkeypatch.setattr(tj.time, "sleep", lambda s: None)
    monkeypatch.setenv("WANDB_API_KEY", "wandb_SECRET")
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET_TOKEN")
    ev = tmp_path / "eval"
    ev.mkdir()
    for f in (*hs.FILES, "helipad_norule.jsonl"):
        (ev / f).write_text("{}\n")
    monkeypatch.setattr(hs, "EVAL", ev)
    (tmp_path / "best.pt").write_bytes(b"ckpt")
    old = {s: signal.getsignal(s) for s in hs.SIGNALS}
    yield tmp_path
    for s, h in old.items():
        signal.signal(s, h)


def launch(env, monkeypatch, cli, limit="0"):
    monkeypatch.setattr(tj, "_run", cli)
    return hs.main([str(env / "best.pt"), str(env / "out.json"), "--limit-cases", limit])


def test_happy_path_scores_and_deletes_the_pod(env, monkeypatch):
    cli = FakeCli()
    launch(env, monkeypatch, cli, "7")
    create = next(a for a in cli.calls if a[:3] == ["runpodctl", "pod", "create"])
    for flag, val in (("--image", tj.IMAGE), ("--gpu-id", "NVIDIA H100 80GB HBM3"), ("--cloud-type", "SECURE"),
                      ("--container-disk-in-gb", "80"), ("--ports", "22/tcp"), ("--min-cuda-version", "12.8")):
        assert create[create.index(flag) + 1] == val
    names = cli.uploaded[0]
    assert "pcdm/README.md" in names and "pcdm/site/server.py" not in names   # NOT_SHIPPED stays home
    assert {"pkg/best.pt", "score.sh", *(f"pkg/eval/{f}" for f in hs.FILES)} <= set(names)
    run = next(a[-1] for a in cli.calls if a[0] == "ssh" and "score.sh" in a[-1])
    assert run.endswith("score.sh 7")
    assert "set -eo pipefail" in hs.REMOTE_SH and "--limit-cases \"$1\"" in hs.REMOTE_SH
    assert json.loads((env / "out.json").read_text()) == {"ok": 1}
    assert cli.deleted() == ["pod1"] and not cli.pods


def test_unparseable_create_output_still_deletes_by_name(env, monkeypatch):
    cli = FakeCli(create=lambda: "Error: something\nnot json")
    launch(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] and not cli.pods


def test_create_that_errors_after_renting_still_deletes(env, monkeypatch):
    def create():
        raise RuntimeError("runpodctl pod create timed out after 180s")
    cli = FakeCli(create=create)
    with pytest.raises(RuntimeError, match="timed out"):
        launch(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] and not cli.pods


def test_sigterm_during_create_still_deletes(env, monkeypatch):
    def create():
        os.kill(os.getpid(), signal.SIGTERM)
        return "{}"
    cli = FakeCli(create=create)
    with pytest.raises(KeyboardInterrupt):
        launch(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] and not cli.pods


def test_exception_after_create_deletes(env, monkeypatch):
    cli = FakeCli(fail_on=lambda a: a[0] == "scp")
    with pytest.raises(RuntimeError, match="boom"):
        launch(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] and not cli.pods


def test_failed_delete_is_retried_until_gone(env, monkeypatch):
    cli = FakeCli(delete_fails=2)
    launch(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] * 3 and not cli.pods


def test_unconfirmed_delete_exits_nonzero(env, monkeypatch):
    cli = FakeCli(delete_fails=99)
    with pytest.raises(SystemExit):
        launch(env, monkeypatch, cli)


def test_missing_eval_file_fails_before_any_pod(env, monkeypatch):
    (hs.EVAL / hs.FILES[0]).unlink()
    cli = FakeCli()
    with pytest.raises(SystemExit, match=hs.FILES[0]):
        launch(env, monkeypatch, cli)
    assert not [a for a in cli.calls if a[0] == "runpodctl"]


def test_name_is_reclaimed_by_typically_job_reconcile(env, monkeypatch):
    """A killed launcher leaves its pod; `typically_job.py --reconcile` must delete it (no status file knows it)."""
    cli = FakeCli(fail_on=lambda a: a[:3] == ["runpodctl", "pod", "delete"])   # the launcher's own cleanup never lands
    with pytest.raises(SystemExit):
        launch(env, monkeypatch, cli)
    name = cli.pods[0]["name"]
    assert name.startswith(tj.POD_PREFIX + "helipadscore-")
    cli.fail_on = None
    assert tj.reconcile(log=lambda m: None) == ["pod1"] and not cli.pods


def test_no_secret_beyond_the_public_key_in_any_argv(env, monkeypatch):
    cli = FakeCli()
    launch(env, monkeypatch, cli)
    create = next(a for a in cli.calls if a[:3] == ["runpodctl", "pod", "create"])
    assert json.loads(create[create.index("--env") + 1]) == {"PUBLIC_KEY": "ssh-ed25519 AAAApublic me"}
    flat = "\n".join(" ".join(a) for a in cli.calls)
    assert not [s for s in SECRETS if s in flat]
