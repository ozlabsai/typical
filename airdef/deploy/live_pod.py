"""The hosted demo's live server: server.py + Typical v8 on a small RunPod GPU, behind the site's Worker.

    uv run --no-project python airdef/deploy/live_pod.py up       # rent, set up, start; point the Worker at it
    uv run --no-project python airdef/deploy/live_pod.py down     # delete the pod; the site plays its recordings again
    uv run --no-project python airdef/deploy/live_pod.py status

The page asks /api/meta at load: answered, it runs live (any settings, one world per viewer); not, it plays the tapes.
Only the Worker can call the server (a token it adds). State: .context/airdef/live.json.
"""
import json
import os
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
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
SETUP = """set -e
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=300 UV_PROJECT_ENVIRONMENT=/workspace/venv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/pcdm
uv sync
if [[ "$(nvidia-smi)" == *"CUDA Version: 12.8"* ]]; then
  uv pip install --python /workspace/venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B-Base')"
AIRDEF_CACHE=1 AIRDEF_LIVE_TOKEN={token} setsid nohup uv run --no-sync uvicorn --app-dir airdef server:app --host 0.0.0.0 --port {port} \\
  > /workspace/server.log 2>&1 < /dev/null &
"""


def wrangler(*args, stdin=None):
    subprocess.run(["npx", "wrangler", *args], cwd=ROOT / "airdef/deploy", input=stdin, text=True, check=True, capture_output=True)


def up():
    if STATE.exists():
        sys.exit(f"already up: {json.loads(STATE.read_text())['url']} (live_pod.py down first)")
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
        os.environ["RUNPOD_PORTS"] = f"{PORT}/http,22/tcp"
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
            j.ssh("tar xzf /workspace/repo.tar.gz -C /workspace/pcdm", 300)
            print(f"setting up on {j.gpu or gpu} (pod {j.pod_id})", flush=True)
            j.ssh("bash /workspace/setup.sh > /workspace/setup.log 2>&1", 1800)
            url = f"https://{j.pod_id}-{PORT}.proxy.runpod.net"
            for _ in range(60):   # the model loads, then the server answers
                try:
                    req = urllib.request.Request(url + "/api/meta", headers={"x-live-token": token})
                    if json.loads(urllib.request.urlopen(req, timeout=10).read()).get("scenarios"):
                        break
                except Exception:
                    pass
                time.sleep(10)
            else:
                raise RuntimeError("the server did not answer: see /workspace/server.log on the pod")
        except BaseException:
            j.delete()
            raise
    STATE.write_text(json.dumps(dict(pod=j.pod_id, url=url, token=token, gpu=gpu, since=time.strftime("%Y-%m-%d %H:%M"))))
    wrangler("secret", "put", "LIVE_TOKEN", stdin=token)
    wrangler("secret", "put", "LIVE_URL", stdin=url)
    print(f"live: {url} on {gpu} (pod {j.pod_id}); the site now runs live", flush=True)


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
        req = urllib.request.Request(st["url"] + "/api/meta", headers={"x-live-token": st["token"]})
        ok = bool(json.loads(urllib.request.urlopen(req, timeout=15).read()).get("scenarios"))
    except Exception:
        ok = False
    print(f"{'up' if ok else 'NOT ANSWERING'}: {st['url']} on {st['gpu']} since {st['since']} (pod {st['pod']})")


if __name__ == "__main__":
    {"up": up, "down": down, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
