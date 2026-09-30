"""Teach it: rent one H100 on RunPod, fine-tune typical-small on .context/typically/jobs/<slug>/, fetch the result, delete the pod.

uv run --no-project python scripts/typically_job.py <slug>      (stdlib + the runpodctl/ssh/scp/git binaries only)
Progress for the UI: .context/typically/jobs/<slug>/status.json; full log: jobs/<slug>/job.log (never contains the HF token).
The pod (named typically-job-<slug>-<job_id>, job_id in status.json) is deleted in a `finally` on every path; a single 75 min
deadline caps every subprocess call. `--reconcile` (also run at server start and before every job) deletes orphaned typically-job-* pods.
"""
import json
import re
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TYPICALLY = REPO / ".context" / "typically"
KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
SSH_OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20"]
NOT_SHIPPED = ("site/", "figures/", "paper/", "blog/", "docs/", "vendor/")   # ponytail: nothing the training chain reads
POLL_S, READY_S, DEADLINE_S, STALE_S, STEPS = 60, 300, 75 * 60, 90 * 60, 400
PHASES = ("queued", "starting_gpu", "uploading", "training", "evaluating", "downloading", "done", "failed")
ACTIVE = PHASES[:6]
SLUG_RE = re.compile(r"^[a-z0-9_]{1,40}$")
POD_PREFIX = "typically-job-"
_ACTIVE: set[str] = set()   # job_ids running in THIS process; ponytail: reconcile assumes one process owns all typically-job-* pods
REMOTE = "/workspace/pcdm"
JOB_SH = """trap 'echo $? > /workspace/job.exit' EXIT   # the poller reads the chain's exit code, not log text
set -e
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd {REMOTE}
bash scripts/typically_spike_pod.sh job {slug}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Deadline(Exception):   # not a RuntimeError: the poll/ssh-retry loops must not swallow it
    pass


def job_dir(slug: str) -> Path:
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"bad slug {slug!r}: need ^[a-z0-9_]{{1,40}}$")
    return TYPICALLY / "jobs" / slug


def read_status(slug: str) -> dict | None:
    p = job_dir(slug) / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_status(slug: str, phase: str, message: str, **extra) -> dict:
    assert phase in PHASES
    prev = (read_status(slug) or {}) if phase != "queued" else {}
    st = {"phase": phase, "started_at": prev.get("started_at", _now()), "updated_at": _now(), "message": message,
          "pod_id": prev.get("pod_id"), **extra}
    (job_dir(slug) / "status.json").write_text(json.dumps(st))
    return st


def _run(argv: list[str], timeout: int, out: Path | None = None) -> str:
    """The one subprocess seam (tests replace it). Returns stdout; with `out` the stdout is also saved there."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''} timed out after {timeout:.0f}s")
    if out:
        out.write_text(r.stdout)
    if r.returncode:
        raise RuntimeError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''} failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def list_pods() -> list[dict]:
    return json.loads(_run(["runpodctl", "pod", "list", "-o", "json"], 60))


def delete_pod(pod_id: str) -> None:
    _run(["runpodctl", "pod", "delete", pod_id], 60)


class Job:
    def __init__(self, slug: str, log, tmp: Path):
        self.slug, self.log, self.tmp, self.id = slug, log, tmp, secrets.token_hex(4)
        self.name = f"{POD_PREFIX}{slug}-{self.id}"   # unique per job: nothing else can ever match it
        self.dir, self.deadline, self.pod_id, self.host = job_dir(slug), time.monotonic() + DEADLINE_S, None, None
        self.tar = tmp / "repo.tar.gz"

    def call(self, argv: list[str], cap: float, out: Path | None = None) -> str:
        """_run under the job's single deadline. Cleanup (delete) calls _run directly so it still works after the deadline."""
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise Deadline("the job took longer than 75 minutes")
        return _run(argv, min(cap, left), out)

    def to(self, phase: str, message: str, **extra):
        self.log(f"[{phase}] {message}")
        write_status(self.slug, phase, message, pod_id=self.pod_id, job_id=self.id, **extra)

    def ssh(self, cmd: str, timeout: int = 60) -> str:
        return self.call(["ssh", "-n", *SSH_OPTS, "-p", self.host[1], f"root@{self.host[0]}", cmd], timeout)

    # -- pod lifecycle
    def mine(self) -> list[str]:
        """Only the recorded pod id or the exact unique name; never a prefix match."""
        return [p["id"] for p in list_pods() if p["id"] == self.pod_id or p.get("name") == self.name]

    def create(self):
        pub = (KEY.parent / (KEY.name + ".pub")).read_text().strip()
        out = self.dir / "create.out"   # the "no instances" error also arrives on stdout
        try:
            stdout = self.call(["runpodctl", "pod", "create", "--name", self.name, "--image", IMAGE, "--gpu-id", "NVIDIA H100 80GB HBM3",
                           "--cloud-type", "SECURE", "--container-disk-in-gb", "150", "--ports", "22/tcp",
                           "--env", json.dumps({"PUBLIC_KEY": pub})], 180, out)
            self.pod_id = json.loads(stdout)["id"]
        except (RuntimeError, ValueError, KeyError, TypeError):
            raise RuntimeError("no GPU was available right now: " + out.read_text().strip()[:200])
        self.log(f"pod {self.pod_id} created")
        if len(self.mine()) != 1:
            raise RuntimeError(f"expected exactly one {self.name} pod, found {self.mine()}")

    def wait_ssh(self):
        end = time.monotonic() + READY_S
        while time.monotonic() < end:
            try:
                s = json.loads(self.call(["runpodctl", "pod", "get", self.pod_id, "-o", "json"], 60)).get("ssh") or {}
                if s.get("ip") and s.get("port"):
                    self.host = (s["ip"], str(s["port"]))
                    self.ssh("true", 30)
                    return self.log(f"ssh up at {s['ip']}:{s['port']}")
            except (RuntimeError, ValueError, KeyError) as e:   # not ready yet (a Deadline must escape)
                self.log(f"waiting for ssh: {e}")
            time.sleep(10)
        raise RuntimeError("the GPU machine did not come up in 5 minutes")

    def delete(self):
        for _ in range(3):
            try:
                ids = self.mine()
            except Exception as e:
                self.log(f"pod list failed: {e}")
                ids = [self.pod_id] if self.pod_id else []
            for i in ids:
                try:
                    delete_pod(i)
                except Exception as e:
                    self.log(f"pod delete {i} failed: {e}")
            try:
                if not self.mine():
                    return self.log("pod deleted")
            except Exception as e:
                self.log(f"pod list failed: {e}")
        self.log(f"WARNING: pod may still be running ({self.name}, {self.pod_id}) - delete it by hand")

    # -- work
    def snapshot(self):
        """Immutable upload: the tarball is built once, before any pod exists, from the dataset as it is right now."""
        files = self.call(["git", "ls-files", "-co", "--exclude-standard"], 60).splitlines()
        with tarfile.open(self.tar, "w:gz") as t:
            for f in files:
                if not f.startswith(NOT_SHIPPED) and (REPO / f).is_file():
                    t.add(REPO / f, arcname=f)
            for f in ("train.jsonl", "val.jsonl", *(f"eval/{p.name}" for p in (self.dir / "eval").glob("*.jsonl"))):
                t.add(self.dir / f, arcname=f"data_co_{self.slug}/{f}")
        self.log(f"tar {self.tar.stat().st_size / 1e6:.1f} MB, {len(files)} tracked files")

    def upload(self):
        # ponytail: HF token goes through a local temp file (0600) -> scp -> /workspace/.env; never argv, never logged
        token = (Path.home() / ".cache/huggingface/token").read_text().strip()
        env, sh = self.tmp / "env", self.tmp / "job.sh"
        env.write_text(f"HF_TOKEN={token}\n")
        env.chmod(0o600)
        sh.write_text(JOB_SH.format(REMOTE=REMOTE, slug=self.slug))
        addr = f"root@{self.host[0]}:"
        scp = ["scp", *SSH_OPTS, "-P", self.host[1]]
        self.ssh(f"mkdir -p {REMOTE}")
        self.call([*scp, str(self.tar), addr + "/workspace/repo.tar.gz"], 900)
        self.call([*scp, str(env), addr + "/workspace/.env"], 120)
        self.call([*scp, str(sh), addr + "/workspace/job.sh"], 120)
        self.ssh(f"chmod 600 /workspace/.env && tar xzf /workspace/repo.tar.gz -C {REMOTE}", 300)

    def launch(self):
        self.ssh("rm -f /workspace/job.exit; setsid nohup bash /workspace/job.sh > /workspace/job.log 2>&1 < /dev/null & echo $! > /workspace/job.pid")

    def poll(self):
        bad = 0
        while True:
            time.sleep(POLL_S)
            try:   # order matters: job.exit is written before the pid dies, so check it only once the pid is gone
                out = self.ssh('if kill -0 "$(cat /workspace/job.pid)" 2>/dev/null; then echo ALIVE; elif [ -f /workspace/job.exit ]; '
                               'then echo EXIT $(cat /workspace/job.exit); else echo DEAD; fi; tail -n 300 /workspace/job.log', 90)
                bad = 0
            except (RuntimeError, subprocess.SubprocessError) as e:   # a dropped ssh must not kill a 20 minute run
                bad += 1
                self.log(f"poll failed ({bad}/3): {e}")
                if bad >= 3:
                    raise
                continue
            (self.dir / "pod.log").write_text(out)
            steps = [int(s) for s in re.findall(r"^step (\d+) ", out, re.M)]
            head = out.split("\n", 1)[0]
            if head != "ALIVE":   # EXIT <code> | DEAD (pid gone, no exit code: killed)
                code = head.removeprefix("EXIT").strip() if head.startswith("EXIT") else ""
                if code == "0":
                    return
                hint = " (Python traceback in the log)" if "Traceback" in out else ""
                raise RuntimeError(f"the training run failed (exit {code or 'unknown'}){hint}; see job.log")
            if "best step" in out:
                self.to("evaluating", "Training finished. Scoring your model on held-out examples.", progress=1.0)
            else:
                p = min(max(steps, default=0) / STEPS, 1.0)
                self.to("training", f"Learning from your examples ({p:.0%} done)." if steps else
                        "Setting up and measuring the starting model.", progress=p)

    def download(self):
        dest = TYPICALLY / "results" / f"co_{self.slug}"
        dest.mkdir(parents=True, exist_ok=True)
        for remote, name in ((f"runs/co_{self.slug}/eval_co.json", "eval_co.json"), ("runs/base/eval_co.json", "base_eval_co.json"),
                             (f"runs/co_{self.slug}/best.pt", "best.pt")):   # best.pt last: a half-done run is never offered as a model
            self.call(["scp", *SSH_OPTS, "-P", self.host[1], f"root@{self.host[0]}:{REMOTE}/{remote}", str(dest / name)], 600)
        self.log(f"downloaded to {dest}")


def reconcile(log=print) -> list[str]:
    """Delete typically-job-* pods this process is not actively running: no status file, not an active phase / not ours, or stale
    (> 90 min). Never touches pods with other names. Returns the deleted pod ids."""
    known = {}   # job_id / pod_id -> status, from every job dir (the latest job per slug wins: status.json is overwritten)
    for p in (TYPICALLY / "jobs").glob("*/status.json"):
        try:
            st = json.loads(p.read_text())
        except ValueError:
            continue
        known.update({k: st for k in (st.get("job_id"), st.get("pod_id")) if k})
    gone = []
    for pod in list_pods():
        name = pod.get("name") or ""
        if not name.startswith(POD_PREFIX):
            continue
        st = known.get(pod["id"]) or known.get(name.rsplit("-", 1)[-1])
        if not st:
            why = "no status file"
        elif st.get("phase") not in ACTIVE or st.get("job_id") not in _ACTIVE:
            why = f"not running here (phase {st.get('phase')})"
        elif datetime.now(timezone.utc) - datetime.fromisoformat(st["updated_at"]) > timedelta(seconds=STALE_S):
            why = "status older than 90 min"
        else:
            continue
        log(f"reconcile: deleting {name} ({pod['id']}): {why}")
        try:
            delete_pod(pod["id"])
            gone.append(pod["id"])
        except Exception as e:
            log(f"reconcile: delete {pod['id']} failed: {e}")
    return gone


def run(slug: str, log=None) -> None:
    """Raises ValueError on a bad slug; otherwise never raises: the outcome is in status.json (done / failed)."""
    job_dir(slug).mkdir(parents=True, exist_ok=True)   # validates the slug
    log = log or file_log(slug)
    with tempfile.TemporaryDirectory() as tmp:
        j = Job(slug, log, Path(tmp))
        _ACTIVE.add(j.id)
        try:
            j.to("queued", "Waiting to start.")
            j.snapshot()
            reconcile(log)
            j.to("starting_gpu", "Getting a GPU ready (about 3 minutes).")
            j.create()
            j.to("starting_gpu", "Getting a GPU ready (about 3 minutes).")   # records the pod id
            j.wait_ssh()
            j.to("uploading", "Sending your examples to the GPU.")
            j.upload()
            j.launch()
            j.to("training", "Setting up and measuring the starting model.", progress=0.0)
            j.poll()
            j.to("downloading", "Bringing your model back.")
            j.download()
            j.to("done", "Your model is ready.", progress=1.0)
        except BaseException as e:   # incl. KeyboardInterrupt from the CLI; the pod must still go
            log(f"FAILED: {type(e).__name__}: {e}")
            j.to("failed", f"Training did not finish: {e}")
        finally:
            j.delete()
            _ACTIVE.discard(j.id)


def file_log(slug: str):
    def log(msg: str):
        line = f"{_now()} {msg}"
        print(line, flush=True)
        with open(job_dir(slug) / "job.log", "a") as f:
            f.write(line + "\n")
    return log


if __name__ == "__main__":
    try:
        reconcile() if sys.argv[1:] == ["--reconcile"] else run(sys.argv[1])
    except ValueError as e:
        sys.exit(str(e))
