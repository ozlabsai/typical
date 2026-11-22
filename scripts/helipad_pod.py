"""Run one bash script on one rented RunPod GPU: one tarball up, the script, files back, and the pod always deleted.

helipad_pod.run(name_tag, uploads={local path: arcname}, remote_script, fetch={remote path: local path})
Needs runpodctl >= 3 (REST v2) configured, and the ~/.runpod/ssh/RunPod-Key-Go key.

The pod is named <POD_PREFIX><name_tag>-<token> (typically_job.Job) and deleted by that exact name (never by the create
output's id) in a `finally`, with SIGINT / SIGTERM / SIGHUP raised into it, retried until the pod list confirms it is gone.
If the launcher is SIGKILLed, `typically_job.py --reconcile` deletes a <POD_PREFIX>* pod (no status file knows it); a
running Decision Factory server reconciles too, so it can also kill a LIVE pod. Set TYPICALLY_POD_PREFIX to keep a long
run out of other processes' reconcile (then nothing reclaims it after a SIGKILL: check `runpodctl pod list`).
"""
from __future__ import annotations

import json
import signal
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typically_job as tj  # noqa: E402

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
GPU = "NVIDIA H100 80GB HBM3"
SSH_TRIES = 90   # x 10 s: 15 min for the pod to boot and sshd to answer
RUN_S = 2 * 3600   # default cap on the remote script: a hung ssh can't hold the pod forever
KEEPALIVE = ["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]   # a quiet multi-hour ssh stays up


def _interrupt(sig, _frame):
    raise KeyboardInterrupt(signal.Signals(sig).name)


def repo_uploads(prefix: str = "pcdm") -> dict[Path, str]:
    """The repo as typically_job ships it (tracked + untracked, not ignored, minus NOT_SHIPPED) under <prefix>/."""
    return {tj.REPO / f: f"{prefix}/{f}" for f in tj._run(["git", "ls-files", "-co", "--exclude-standard"], 60).splitlines()
            if not f.startswith(tj.NOT_SHIPPED) and (tj.REPO / f).is_file()}


def pack(tmp: Path, uploads: dict[Path, str], script_text: str, script: str) -> Path:
    (sh := tmp / script).write_text(script_text)
    up = tmp / "up.tgz"
    with tarfile.open(up, "w:gz") as t:
        for src, arc in uploads.items():
            t.add(src, arcname=arc)
        t.add(sh, arcname=script)
    return up


def wait_ssh(pod_id: str) -> tuple[str, str]:
    """`pod get`'s ssh {ip, port} (what typically_job reads, proven on runpodctl 3.1.1), then an ssh `true`."""
    for _ in range(SSH_TRIES):
        try:
            s = json.loads(tj._run(["runpodctl", "pod", "get", pod_id, "-o", "json"], 60)).get("ssh") or {}
            if s.get("ip") and s.get("port"):
                tj._run(["ssh", "-n", *tj.SSH_OPTS, "-p", str(s["port"]), f"root@{s['ip']}", "true"], 30)
                print(f"ssh up at {s['ip']}:{s['port']}", flush=True)
                return s["ip"], str(s["port"])
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"waiting for ssh: {e}", flush=True)
        time.sleep(10)
    raise RuntimeError(f"pod {pod_id} never became reachable over ssh in {SSH_TRIES * 10 // 60} min")


def _go(j: tj.Job, up: Path, script: str, args: str, fetch: dict[str, Path], gpu: str, disk_gb: int, timeout_s: int):
    pub = (tj.KEY.parent / (tj.KEY.name + ".pub")).read_text().strip()
    tj._run(["runpodctl", "pod", "create", "--name", j.name, "--image", tj.IMAGE, "--gpu-id", gpu,
             "--cloud-type", "SECURE", "--container-disk-in-gb", str(disk_gb), "--ports", "22/tcp", "--min-cuda-version", "12.8",
             "--env", json.dumps({"PUBLIC_KEY": pub})], 180)   # stdout ignored: the pod is found (and deleted) by name

    def listed():
        if ids := j.mine(tj._pods()):
            return ids[0]
        raise RuntimeError(f"{j.name} not listed yet")
    j.pod_id = tj.retry(listed, "pod list")
    print(f"pod {j.name} ({j.pod_id}) created", flush=True)
    ip, port = wait_ssh(j.pod_id)
    scp, addr = ["scp", *tj.SSH_OPTS, "-P", port], f"root@{ip}"
    tj._run([*scp, str(up), f"{addr}:/workspace/up.tgz"], 1800)
    cmd = f"tar xzf /workspace/up.tgz -C /workspace && bash /workspace/{script}" + (f" {args}" if args else "")
    print(tj._run(["ssh", "-n", *tj.SSH_OPTS, *KEEPALIVE, "-p", port, addr, cmd], timeout_s), end="", flush=True)
    for remote, local in fetch.items():
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        tj._run([*scp, "-r", f"{addr}:{remote}", str(local)], 1800)
        print(f"fetched {remote} -> {local}", flush=True)


def run(name_tag: str, uploads: dict[Path, str], remote_script: str, fetch: dict[str, Path], gpu: str = GPU,
        disk_gb: int = 150, timeout_s: int = RUN_S, script: str = "run.sh", args: str = "") -> None:
    """Upload, run `bash <script> <args>` under `set -eo pipefail`, scp each fetch back, then delete the pod (always)."""
    with tempfile.TemporaryDirectory() as tmp:
        up = pack(Path(tmp), uploads, "set -eo pipefail\n" + remote_script, script)
        j = tj.Job(name_tag, lambda m: print(m, flush=True), Path(tmp))   # only its name + retried, name-matching delete
        old = {s: signal.signal(s, _interrupt) for s in SIGNALS}
        try:
            _go(j, up, script, args, fetch, gpu, disk_gb, timeout_s)
        finally:
            for s in SIGNALS:
                signal.signal(s, signal.SIG_IGN)   # a second Ctrl-C must not cut the cleanup short
            gone = j.delete()
            for s, h in old.items():
                signal.signal(s, h)
            if not gone:
                raise SystemExit(f"pod {j.name} may still be running: delete it by hand")
