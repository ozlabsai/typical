"""The hosted demo's live server: server.py + Typical v8 on a small RunPod GPU, behind the site's Worker.

    uv run --no-project python airdef/deploy/live_pod.py up [--scale-from 2026-11-19T19:05Z]   # rent, set up, start; point the Worker at it
    uv run --no-project python airdef/deploy/live_pod.py down     # delete the pod; the site plays its recordings again
    uv run --no-project python airdef/deploy/live_pod.py status

The page asks /api/meta at load: answered, it runs live (any settings, one world per viewer); not, it plays the tapes.
Only the Worker can call the server (a token it adds). State: .context/airdef/live.json.
Scale-down: the Worker stops the pod after IDLE_MIN (30) minutes without a viewer (from --scale-from on) and starts it
on the next visit; everything lives on the pod's volume and the server starts on boot, so a wake takes ~2 minutes
(the visitor sees the recordings meanwhile; the page switches to live at their next scenario). `up` on a running
setup brings up the new pod first, then switches the Worker and deletes the old one.
"""
import json
import os
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["TYPICALLY_POD_PREFIX"] = "airdef-live-"   # not typically-job-*: the training reconcile must never delete it
import typically_job as tj   # noqa: E402

STATE = ROOT / ".context/airdef/live.json"
CKPT = ".context/typically/results/co_airdef_v8/best.pt"
PORT = 8790
GPUS = ["NVIDIA RTX 2000 Ada Generation", "NVIDIA GeForce RTX 3090", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA GeForce RTX 4090", "NVIDIA RTX 6000 Ada Generation"]   # cheapest first ($0.24 .. $1.09 / h, secure cloud)
# everything on the volume (/workspace), so a stopped pod wakes with it: python, the venv, the model cache, the code
SETUP = """set -e
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=300 UV_PROJECT_ENVIRONMENT=/workspace/venv UV_PYTHON_INSTALL_DIR=/workspace/uvpy \\
  UV_CACHE_DIR=/workspace/uvcache HF_HOME=/workspace/hf
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/pcdm
uv sync
if [[ "$(nvidia-smi)" == *"CUDA Version: 12.8"* ]]; then
  uv pip install --python /workspace/venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B-Base')"
cat > /workspace/start.sh <<'EOS'
cd /workspace/pcdm
export HF_HOME=/workspace/hf AIRDEF_CACHE=1 AIRDEF_LIVE_TOKEN={token}
setsid nohup /workspace/venv/bin/python -m uvicorn --app-dir airdef server:app --host 0.0.0.0 --port {port} > /workspace/server.log 2>&1 < /dev/null &
EOS
bash /workspace/start.sh
"""
BOOT = "bash -c 'bash /workspace/start.sh; exec /start.sh'"   # on every boot: the server, then the image's own start (ssh)


UA = {"User-Agent": "Mozilla/5.0 (Macintosh) airdef-live"}   # RunPod's proxy turns away bare script clients


def answers(url, token, timeout=10):
    req = urllib.request.Request(url + "/api/meta", headers={"x-live-token": token, **UA})
    return bool(json.loads(urllib.request.urlopen(req, timeout=timeout).read()).get("scenarios"))


def key():
    import re
    return re.search(r"apikey\s*=\s*['\"]?([^'\"\n]+)", (Path.home() / ".runpod/config.toml").read_text(), re.I).group(1)


def runpod(pod, action=None):   # POST start / stop (the same call the Worker makes); no action: the pod's status
    req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}" + ("/action" if action else ""), method="POST" if action else "GET",
                                 data=json.dumps({"action": action}).encode() if action else None,
                                 headers={"Authorization": f"Bearer {key()}", "content-type": "application/json", **UA})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod {action or 'status'}: {e.code} {e.read().decode()[:300]}") from None


def settle(pod, want, tries=30):   # wait until RunPod reports the pod RUNNING / EXITED
    for _ in range(tries):
        if runpod(pod).get("status") == want:
            return
        time.sleep(5)
    raise RuntimeError(f"pod {pod} never reached {want}")


def wait_up(url, token, tries=40):
    for _ in range(tries):
        try:
            if answers(url, token):
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def wrangler(*args, stdin=None):
    subprocess.run(["npx", "wrangler", *args], cwd=ROOT / "airdef/deploy", input=stdin, text=True, check=True, capture_output=True)


def up():
    old = json.loads(STATE.read_text()) if STATE.exists() else None   # replaced once the new one answers
    scale_from = sys.argv[sys.argv.index("--scale-from") + 1] if "--scale-from" in sys.argv else time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    token = secrets.token_urlsafe(24)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tj.job_dir("airdef_live").mkdir(parents=True, exist_ok=True)
        j = tj.Job("airdef_live", tj.file_log("airdef_live"), tmp)
        files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        with tarfile.open(j.tar, "w:gz") as t:
            for f in files:
                if not f.startswith(tj.NOT_SHIPPED) and (ROOT / f).is_file():
                    t.add(ROOT / f, arcname=f)
            t.add(ROOT / CKPT, arcname=CKPT)   # where server.py looks for it
        os.environ.update(RUNPOD_PORTS=f"{PORT}/http,22/tcp", RUNPOD_VOLUME_GB="60", RUNPOD_CONTAINER_GB="30", RUNPOD_DOCKER_ARGS=BOOT)
        for gpu in GPUS:
            os.environ["RUNPOD_GPU"] = gpu
            try:
                j.create()
                break
            except tj.NoGpu:
                print(f"no {gpu} free; next", flush=True)
        else:
            sys.exit("no GPU free right now")
        try:
            j.deadline = time.monotonic() + 3600
            j.wait_ssh()
            scp, dest = ["scp", *tj.SSH_OPTS, "-P", j.host[1]], f"root@{j.host[0]}:"
            (tmp / "setup.sh").write_text(SETUP.replace("{token}", token).replace("{port}", str(PORT)))
            j.ssh("mkdir -p /workspace/pcdm")
            j.call([*scp, str(j.tar), dest + "/workspace/repo.tar.gz"], 900)
            j.call([*scp, str(tmp / "setup.sh"), dest + "/workspace/setup.sh"], 120)
            j.ssh("tar xzf /workspace/repo.tar.gz --no-same-owner -C /workspace/pcdm", 300)   # the volume refuses chown
            print(f"setting up on {j.gpu or gpu} (pod {j.pod_id})", flush=True)
            j.ssh("bash /workspace/setup.sh > /workspace/setup.log 2>&1", 1800)
            for _ in range(90):   # the model loads, then the server answers -- on the pod first
                if '"scenarios"' in j.ssh(f"curl -s -H 'x-live-token: {token}' localhost:{PORT}/api/meta | head -c 200 || true", 60):
                    break
                time.sleep(10)
            else:
                raise RuntimeError("the server did not start on the pod")
            url = f"https://{j.pod_id}-{PORT}.proxy.runpod.net"
            if not wait_up(url, token, 30):   # then through RunPod's proxy (it takes a minute to route a new pod)
                raise RuntimeError(f"the server runs on the pod but {url} does not reach it")
            print("answers; testing a stop and a wake", flush=True)   # what the Worker will do: it must come back by itself
            runpod(j.pod_id, "stop")
            settle(j.pod_id, "EXITED")
            runpod(j.pod_id, "start")
            t0 = time.monotonic()
            if not wait_up(url, token, 40):
                raise RuntimeError("stopped and started, the server did not come back")
            print(f"woke in {time.monotonic() - t0 + 10:.0f} s", flush=True)
        except BaseException:
            logs = ROOT / ".context/airdef/live_fail"
            logs.mkdir(parents=True, exist_ok=True)
            for f in ("setup.log", "server.log"):   # what went wrong, before the pod (and its logs) go
                try:
                    j.call(["scp", *tj.SSH_OPTS, "-P", j.host[1], f"root@{j.host[0]}:/workspace/{f}", str(logs / f)], 60)
                except Exception:
                    pass
            print(f"failed; pod logs in {logs}", flush=True)
            j.delete()
            raise
    STATE.write_text(json.dumps(dict(pod=j.pod_id, url=url, token=token, gpu=gpu, since=time.strftime("%Y-%m-%d %H:%M"), scale_from=scale_from)))
    for name, value in [("LIVE_TOKEN", token), ("LIVE_URL", url), ("LIVE_POD", j.pod_id), ("RUNPOD_API_KEY", key()), ("SCALE_FROM", scale_from)]:
        wrangler("secret", "put", name, stdin=value)
    if old:
        tj.delete_pod(old["pod"])
        print(f"replaced pod {old['pod']}", flush=True)
    print(f"live: {url} on {gpu} (pod {j.pod_id}); scale-down from {scale_from}", flush=True)


def down():
    st = json.loads(STATE.read_text()) if STATE.exists() else None
    try:
        wrangler("secret", "delete", "LIVE_URL", stdin="y\n")   # first: the site falls back to its recordings at once
    except subprocess.CalledProcessError:
        pass
    if st:
        tj.delete_pod(st["pod"])
        STATE.unlink()
        print(f"deleted pod {st['pod']}; the site plays its recordings", flush=True)
    else:
        print("nothing up", flush=True)


def status():
    if not STATE.exists():
        return print("down: the site plays its recordings")
    st = json.loads(STATE.read_text())
    try:
        ok = answers(st["url"], st["token"], 15)
    except Exception:
        ok = False
    print(f"{'up' if ok else 'asleep or NOT ANSWERING'}: {st['url']} on {st['gpu']} since {st['since']} (pod {st['pod']}); "
          f"scale-down from {st.get('scale_from', '-')}")


if __name__ == "__main__":
    {"up": up, "down": down, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
