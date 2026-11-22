"""scripts/helipad_pod.py (generic one-pod runner) with runpodctl / ssh / scp / git faked at typically_job._run.
No network, no pods. uv run pytest tests/test_helipad_pod.py -q"""
import json
import os
import signal
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import helipad_pod as hp  # noqa: E402
import helipad_score_pod as hs  # noqa: E402
import typically_job as tj  # noqa: E402
from test_helipad_score_pod import SECRETS, FakeCli  # noqa: E402


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
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "rows.jsonl").write_text("{}\n")
    old = {s: signal.getsignal(s) for s in hp.SIGNALS}
    yield tmp_path
    for s, h in old.items():
        signal.signal(s, h)


def go(env, monkeypatch, cli, **kw):
    monkeypatch.setattr(tj, "_run", cli)
    hp.run("vtest", {env / "data": "pkg/data", **hp.repo_uploads()}, "echo hi\n",
           {"/workspace/out/eval.json": env / "got" / "eval.json"}, **kw)


def test_happy_path_uploads_runs_fetches_and_deletes(env, monkeypatch):
    cli = FakeCli()
    go(env, monkeypatch, cli, disk_gb=150, args="7")
    create = next(a for a in cli.calls if a[:3] == ["runpodctl", "pod", "create"])
    for flag, val in (("--image", tj.IMAGE), ("--gpu-id", "NVIDIA H100 80GB HBM3"), ("--container-disk-in-gb", "150"),
                      ("--min-cuda-version", "12.8")):
        assert create[create.index(flag) + 1] == val
    assert create[create.index("--name") + 1].startswith(tj.POD_PREFIX + "vtest-")
    names = set(cli.uploaded[0])
    assert {"run.sh", "pkg/data/rows.jsonl", "pcdm/README.md"} <= names and "pcdm/site/server.py" not in names
    run = next(a[-1] for a in cli.calls if a[0] == "ssh" and "run.sh" in a[-1])
    assert run.endswith("bash /workspace/run.sh 7")
    assert json.loads((env / "got" / "eval.json").read_text()) == {"ok": 1}
    assert cli.deleted() == ["pod1"] and not cli.pods


def test_remote_script_fails_fast(env, monkeypatch, tmp_path):
    seen = {}
    real = hp.pack

    def spy(tmp, uploads, script_text, script):
        seen["text"] = script_text
        return real(tmp, uploads, script_text, script)
    monkeypatch.setattr(hp, "pack", spy)
    go(env, monkeypatch, FakeCli())
    assert seen["text"].startswith("set -eo pipefail\n") and seen["text"].rstrip().endswith("echo hi")


def test_runner_deletes_pod_on_error_and_signal(env, monkeypatch):
    cli = FakeCli(fail_on=lambda a: a[0] == "scp")
    with pytest.raises(RuntimeError, match="boom"):
        go(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] and not cli.pods

    def create():
        os.kill(os.getpid(), signal.SIGTERM)
        return "{}"
    cli = FakeCli(create=create)
    with pytest.raises(KeyboardInterrupt):
        go(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] and not cli.pods

    def boom():
        raise RuntimeError("runpodctl pod create timed out after 180s")
    cli = FakeCli(create=boom)
    with pytest.raises(RuntimeError, match="timed out"):
        go(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] and not cli.pods


def test_failed_delete_is_retried_and_an_unconfirmed_one_exits_nonzero(env, monkeypatch):
    cli = FakeCli(delete_fails=2)
    go(env, monkeypatch, cli)
    assert cli.deleted() == ["pod1"] * 3 and not cli.pods
    with pytest.raises(SystemExit):
        go(env, monkeypatch, FakeCli(delete_fails=99))


def test_no_secret_beyond_the_public_key_in_any_argv(env, monkeypatch):
    cli = FakeCli()
    go(env, monkeypatch, cli)
    flat = "\n".join(" ".join(a) for a in cli.calls)
    assert not [s for s in SECRETS if s in flat]


def test_vision_train_ships_data_and_token_file_never_argv(tmp_path, monkeypatch):
    import helipad_vision_train as vt
    (tmp_path / "v0").mkdir()
    (tmp_path / "v0" / "build.json").write_text(json.dumps({"steps_3_epochs": 1620}))
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET_TOKEN")
    seen = {}

    def fake_run(tag, uploads, script, fetch, **kw):
        env = next(p for p, arc in uploads.items() if arc == "hf.env")
        seen.update(tag=tag, token=env.read_text(), mode=env.stat().st_mode & 0o777, kw=kw, fetch=fetch,
                    data=uploads[tmp_path / "v0"], script=script)
    monkeypatch.setattr(hp, "run", fake_run)
    monkeypatch.setattr(hp, "repo_uploads", lambda: {})
    vt.main(["--data", str(tmp_path / "v0"), "--out", str(tmp_path / "res")])
    assert seen["tag"] == "helipadvision" and seen["data"] == "pkg/vision_v0" and seen["kw"]["args"] == "1620"
    assert seen["token"] == "HF_TOKEN=hf_SECRET_TOKEN\n" and seen["mode"] == 0o600
    assert "hf_SECRET" not in seen["script"] and "--steps $1" in seen["script"] and "rm -f /workspace/hf.env" in seen["script"]
    assert set(seen["fetch"].values()) == {tmp_path / "res" / n for n in ("eval", "adapter", "metrics.jsonl", "train.log")}


def test_score_pod_uses_the_generic_runner(tmp_path, monkeypatch):
    ev = tmp_path / "eval"
    ev.mkdir()
    for f in hs.FILES:
        (ev / f).write_text("{}\n")
    monkeypatch.setattr(hs, "EVAL", ev)
    (tmp_path / "best.pt").write_bytes(b"ckpt")
    calls = []
    monkeypatch.setattr(hp, "run", lambda *a, **k: calls.append((a, k)))
    hs.main([str(tmp_path / "best.pt"), str(tmp_path / "out.json"), "--limit-cases", "5"])
    (tag, uploads, script, fetch), kw = calls[0]
    assert tag == "helipadscore" and script == hs.REMOTE_SH and kw["args"] == "5" and kw["disk_gb"] == 80
    assert uploads[tmp_path / "best.pt"] == "pkg/best.pt" and fetch == {"/workspace/score.json": tmp_path / "out.json"}
    assert {f"pkg/eval/{f}" for f in hs.FILES} <= set(uploads.values())
