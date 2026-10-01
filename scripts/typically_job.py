"""Teach it: rent one H100 on RunPod, fine-tune typical-small|medium on .context/typically/jobs/<slug>/, fetch the result, delete the pod.

uv run --no-project python scripts/typically_job.py <slug> [small|medium] [steps]   (stdlib + the runpodctl/ssh/scp/git binaries only)
Progress for the UI: .context/typically/jobs/<slug>/status.json; full log: jobs/<slug>/job.log (never contains the HF token).
The pod (named typically-job-<slug>-<job_id>, job_id in status.json) is deleted in a `finally` on every path; a single deadline (75 min small,
150 min medium) caps every subprocess call. `--reconcile` (also run at server start and before every job) deletes orphaned typically-job-* pods.
`--train-flags <company> <steps> <run>` (run on the pod by scripts/typically_spike_pod.sh) prints the pcdm/train.py flags.
"""
import json
import math
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
POLL_S, READY_S, DEADLINE_S, STALE_S = 60, 300, 75 * 60, 90 * 60   # DEADLINE_S / STALE_S are the small base's; x BASES[base]["scale"]
STEPS = (200, 400, 800)   # Quick / Balanced / Thorough
# scale: deadline + stale multiplier. mem: the memory profile of each release's own recipe (medium = 4B: grad-checkpointing, smaller micro-batch).
BASES = {"small": {"scale": 1, "mem": ["--grad_accum", "8"]}, "medium": {"scale": 2, "mem": ["--grad_ckpt", "--grad_accum", "16"]}}
# The released checkpoints' ckpt["args"] the fine-tune must match. The pod reads the real ones from runs/base/best.pt (train_flags);
# this table only feeds the command the UI displays -- ponytail: verified against both releases 2026-09-30, re-check when a release changes.
RELEASED_ARGS = {
    "small": {"backbone": "Qwen/Qwen3-1.7B-Base", "tap_layer": 20, "nc_render": "semif", "lora_r": 16, "lora_layers": 8,
              "nc_head": "n3", "null": "factored", "noul_head": "bern", "score_head": "choice"},
    "medium": {"backbone": "Qwen/Qwen3.5-4B-Base", "tap_layer": 23, "nc_render": "letters_nonull", "lora_r": 16, "lora_layers": 8,
               "nc_head": "n3", "null": "factored", "noul_head": "bern", "score_head": "choice"},
}
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
bash scripts/typically_spike_pod.sh job {slug} {base} {steps}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrainingFailed(RuntimeError):   # the run on the pod exited badly (vs. infrastructure trouble)
    pass


class Deadline(Exception):   # not a RuntimeError: the poll/ssh-retry loops must not swallow it
    pass


def job_dir(slug: str) -> Path:
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"bad slug {slug!r}: need ^[a-z0-9_]{{1,40}}$")
    return TYPICALLY / "jobs" / slug


def read_status(slug: str) -> dict | None:
    p = job_dir(slug) / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


CODES = {"starting_gpu": "gpu_starting"}   # phase -> code when they differ; `code` is the stable key the frontend localises, `message` stays English for logs


def write_status(slug: str, phase: str, message: str, code: str | None = None, **extra) -> dict:
    assert phase in PHASES
    prev = (read_status(slug) or {}) if phase != "queued" else {}
    st = {"phase": phase, "started_at": prev.get("started_at", _now()), "updated_at": _now(), "message": message,
          "code": code or CODES.get(phase, phase), "pod_id": prev.get("pod_id"), **extra}
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
    def __init__(self, slug: str, log, tmp: Path, base: str = "small", steps: int = 400):
        if base not in BASES or steps not in STEPS:
            raise ValueError(f"bad base/steps {base!r}/{steps!r}: need one of {list(BASES)} and {list(STEPS)}")
        self.slug, self.log, self.tmp, self.id, self.base, self.steps = slug, log, tmp, secrets.token_hex(4), base, steps
        self.name = f"{POD_PREFIX}{slug}-{self.id}"   # unique per job: nothing else can ever match it
        self.dir, self.pod_id, self.host = job_dir(slug), None, None
        self.deadline = time.monotonic() + DEADLINE_S * BASES[base]["scale"]
        self.tar = tmp / "repo.tar.gz"

    def call(self, argv: list[str], cap: float, out: Path | None = None) -> str:
        """_run under the job's single deadline. Cleanup (delete) calls _run directly so it still works after the deadline."""
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise Deadline(f"the job took longer than {DEADLINE_S * BASES[self.base]['scale'] / 60:.0f} minutes")
        return _run(argv, min(cap, left), out)

    def to(self, phase: str, message: str, code: str | None = None, **extra):
        self.log(f"[{phase}] {message}")
        write_status(self.slug, phase, message, code, pod_id=self.pod_id, job_id=self.id, base=self.base, steps=self.steps, **extra)

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
        sh.write_text(JOB_SH.format(REMOTE=REMOTE, slug=self.slug, base=self.base, steps=self.steps))
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
                raise TrainingFailed(f"the training run failed (exit {code or 'unknown'}){hint}; see job.log")
            if "best step" in out:
                self.to("evaluating", "Training finished. Scoring your model on held-out examples.", progress=1.0)
            else:
                p = min(max(steps, default=0) / self.steps, 1.0)
                self.to("training", f"Learning from your examples ({p:.0%} done)." if steps else
                        "Setting up and measuring the starting model.", None if steps else "baseline", progress=p)

    def download(self):
        dest = TYPICALLY / "results" / f"co_{self.slug}"
        dest.mkdir(parents=True, exist_ok=True)
        for remote, name in ((f"runs/co_{self.slug}/eval_co.json", "eval_co.json"), ("runs/base/eval_co.json", "base_eval_co.json"),
                             (f"runs/co_{self.slug}/best.pt", "best.pt")):   # best.pt last: a half-done run is never offered as a model
            self.call(["scp", *SSH_OPTS, "-P", self.host[1], f"root@{self.host[0]}:{REMOTE}/{remote}", str(dest / name)], 600)
        self.log(f"downloaded to {dest}")


def reconcile(log=print) -> list[str]:
    """Delete typically-job-* pods this process is not actively running: no status file, not an active phase / not ours, or stale
    (> 90 min small / 180 min medium). Never touches pods with other names. Returns the deleted pod ids."""
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
        elif datetime.now(timezone.utc) - datetime.fromisoformat(st["updated_at"]) > timedelta(seconds=STALE_S * BASES.get(st.get("base"), BASES["small"])["scale"]):
            why = "status older than its deadline + 15 min"
        else:
            continue
        log(f"reconcile: deleting {name} ({pod['id']}): {why}")
        try:
            delete_pod(pod["id"])
            gone.append(pod["id"])
        except Exception as e:
            log(f"reconcile: delete {pod['id']} failed: {e}")
    return gone


def run(slug: str, log=None, base: str = "small", steps: int = 400) -> None:
    """Raises ValueError on a bad slug/base/steps; otherwise never raises: the outcome is in status.json (done / failed)."""
    job_dir(slug).mkdir(parents=True, exist_ok=True)   # validates the slug
    (job_dir(slug) / "pod.log").unlink(missing_ok=True)   # a retrain must not chart the previous run's log
    log = log or file_log(slug)
    with tempfile.TemporaryDirectory() as tmp:
        j = Job(slug, log, Path(tmp), base, steps)
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
            j.to("training", "Setting up and measuring the starting model.", "baseline", progress=0.0)
            j.poll()
            j.to("downloading", "Bringing your model back.")
            j.download()
            j.to("done", "Your model is ready.", progress=1.0)
        except BaseException as e:   # incl. KeyboardInterrupt from the CLI; the pod must still go
            log(f"FAILED: {type(e).__name__}: {e}")
            j.to("failed", f"Training did not finish: {e}", "failed_timeout" if isinstance(e, Deadline) else "failed_training" if isinstance(e, TrainingFailed) else "failed_infra")
        finally:
            j.delete()
            _ACTIVE.discard(j.id)


def train_flags(company: str, steps: int, run: str, base_args: dict, base: str = "small") -> list[str]:
    """pcdm/train.py argv for a fine-tune of data_co_<company>, warm-started from runs/base/best.pt. The architecture flags come
    from that checkpoint's own ckpt["args"] (`base_args`), so small and medium share this one code path. No value has a space."""
    d = f"data_co_{company}"
    arch = [x for k in RELEASED_ARGS[base] for x in (f"--{k}", str(base_args[k]))]   # the keys, not the values, of the table
    return ["--name", run, "--init_from", "runs/base/best.pt", "--readout", "native", *arch, "--zscore", "--ordinal_smooth", "0.7",
            "--max_state", "1024", *BASES[base]["mem"], "--data", "data_v5", "--extra_data", f"data_wf,data_wh,data_u,{d}",
            "--bucket_map", f"data_wh=W,data_u=U,{d}=C", "--family_weights", "C:0.5,W:0.3,E:0.15,U:0.05", "--null_aug", "W:0.20",
            "--steps", str(steps), "--bs", "64", "--val_every", "50", "--ckpt_every", "100", "--eval_every", str(steps),
            "--eval_limit", "200", "--eval_bs", "8", "--best_on", f"{d}_val"]


def log_series(path: Path, limit: int = 200, tail: int = 64_000) -> dict:
    """The last `limit` logged steps of a pod.log (only its last `tail` bytes are read): train loss, val_nll and the
    company-val nll best.pt is chosen on, aligned on `step` (None where a step has no value), and the mean recent step_time."""
    try:
        with path.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - tail))
            lines = f.read().decode("utf-8", "replace").splitlines()[1 if size > tail else 0:]   # drop the cut first line
    except FileNotFoundError:
        lines = []
    cols: dict[str, dict[int, float]] = {"loss": {}, "val": {}, "best_on": {}}
    times = []
    for line in lines:
        if not (m := re.match(r"step (\d+) (?:val_nll (\S+)|best_on_nll (\S+)|.*?\bloss (\S+)(?:.*?\bstep_time ([\d.]+)s)?)", line)):
            continue
        for col, v in zip(cols, (m[4], m[2], m[3])):
            try:
                if v is not None and math.isfinite(x := float(v)):   # a nan loss is dropped: JSON has no NaN
                    cols[col][int(m[1])] = x
            except ValueError:
                pass
        if m[5]:
            times.append(float(m[5]))
    steps = sorted({s for c in cols.values() for s in c})[-limit:]
    return {"step": steps, **{k: [c.get(s) for s in steps] for k, c in cols.items()},
            "step_time": sum(times[-5:]) / len(times[-5:]) if times else None}


def file_log(slug: str):
    def log(msg: str):
        line = f"{_now()} {msg}"
        print(line, flush=True)
        with open(job_dir(slug) / "job.log", "a") as f:
            f.write(line + "\n")
    return log


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--train-flags"]:   # on the pod, after runs/base/best.pt is downloaded (needs torch: uv run --no-sync)
            import torch
            company, steps, name, *base = sys.argv[2:]
            print(" ".join(train_flags(company, int(steps), name, torch.load("runs/base/best.pt", map_location="cpu", weights_only=True)["args"], *base)))
        elif sys.argv[1:] == ["--reconcile"]:
            reconcile()
        else:
            run(sys.argv[1], None, (sys.argv[2:3] or ["small"])[0], int((sys.argv[3:4] or ["400"])[0]))
    except ValueError as e:
        sys.exit(str(e))
