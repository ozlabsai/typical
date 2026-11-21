"""Score a helipad checkpoint with scripts/helipad_score.py on a rented H100.
Qwen3.5's linear-attention layers need the fla/causal-conv1d kernels; on MPS they fall back to a reference path that did not finish
150 cases in 40 min. Same image and setup as Decision Factory jobs (scripts/typically_job.py, typically_spike_pod.sh).

uv run --no-project python scripts/helipad_score_pod.py <best.pt> <out.json> [--limit-cases N, 0 = all]
Needs runpodctl >= 3 (REST v2) configured, and the ~/.runpod/ssh/RunPod-Key-Go key.

The pod can't be left running: it is named <POD_PREFIX>helipadscore-<token> and deleted by that exact name (never by the create
output's id) in a `finally`, with SIGINT / SIGTERM / SIGHUP raised into it, retried until the pod list confirms it is gone.
If the launcher is SIGKILLed, `typically_job.py --reconcile` deletes the pod: no status file knows it.
Caveat: a running Decision Factory server also reconciles (at start, every 5 min, before every job), so it kills a LIVE scoring pod
too. That direction loses the run, never leaks a pod. Don't score while the server runs locally.
"""
import argparse
import json
import signal
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typically_job as tj  # noqa: E402

EVAL = tj.REPO / ".context" / "typically" / "jobs" / "helipad" / "eval"
FILES = ("helipad_real.jsonl", "helipad_expert.jsonl", "import_oneliner.jsonl", "helipad_contrast.jsonl")
SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
SSH_TRIES = 90   # x 10 s: 15 min for the pod to boot and sshd to answer
RUN_S = 2 * 3600   # cap on setup + scoring (about 15 min today): a hung ssh can't hold the pod forever
REMOTE_SH = f"""set -eo pipefail
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=300 UV_PROJECT_ENVIRONMENT=/workspace/venv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/pcdm
uv sync -q
if [[ "$(nvidia-smi)" == *"CUDA Version: 12.8"* ]]; then
  uv pip install -q --python /workspace/venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
uv pip install -q --python /workspace/venv/bin/python flash-linear-attention || echo "WARN: fla install failed"
uv pip install -q --python /workspace/venv/bin/python --no-build-isolation causal-conv1d || echo "WARN: causal-conv1d build failed"
E=/workspace/pkg/eval
uv run --no-sync python scripts/helipad_score.py --ckpt /workspace/pkg/best.pt --limit-cases "$1" --out /workspace/score.json \\
  --files {" ".join(f"$E/{f}" for f in FILES)} 2>&1 | {{ grep -v "it/s\\]" || true; }} | tail -20
"""


def _interrupt(sig, _frame):
    raise KeyboardInterrupt(signal.Signals(sig).name)


def pack(tmp: Path, ckpt: Path) -> Path:
    """One upload: the repo (as typically_job ships it) under pcdm/, the eval files + checkpoint under pkg/, and the remote script."""
    if missing := [f for f in FILES if not (EVAL / f).is_file()]:
        raise SystemExit(f"missing eval files in {EVAL}: {missing}")
    (sh := tmp / "score.sh").write_text(REMOTE_SH)
    up = tmp / "up.tgz"
    with tarfile.open(up, "w:gz") as t:
        for f in tj._run(["git", "ls-files", "-co", "--exclude-standard"], 60).splitlines():
            if not f.startswith(tj.NOT_SHIPPED) and (tj.REPO / f).is_file():
                t.add(tj.REPO / f, arcname=f"pcdm/{f}")
        for p in sorted(EVAL.glob("*.jsonl")):
            t.add(p, arcname=f"pkg/eval/{p.name}")
        t.add(ckpt, arcname="pkg/best.pt")
        t.add(sh, arcname="score.sh")
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


def score(j: tj.Job, up: Path, limit: int, out: Path) -> None:
    pub = (tj.KEY.parent / (tj.KEY.name + ".pub")).read_text().strip()
    tj._run(["runpodctl", "pod", "create", "--name", j.name, "--image", tj.IMAGE, "--gpu-id", "NVIDIA H100 80GB HBM3",
             "--cloud-type", "SECURE", "--container-disk-in-gb", "80", "--ports", "22/tcp", "--min-cuda-version", "12.8",
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
    print(tj._run(["ssh", "-n", *tj.SSH_OPTS, "-p", port, addr,
                   f"tar xzf /workspace/up.tgz -C /workspace && bash /workspace/score.sh {limit}"], RUN_S), end="", flush=True)
    tj._run([*scp, f"{addr}:/workspace/score.json", str(out)], 300)
    print(f"wrote {out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit-cases", type=int, default=0)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        up = pack(Path(tmp), a.ckpt)
        j = tj.Job("helipadscore", lambda m: print(m, flush=True), Path(tmp))   # only its name + retried, name-matching delete
        for s in SIGNALS:
            signal.signal(s, _interrupt)
        try:
            score(j, up, a.limit_cases, a.out)
        finally:
            for s in SIGNALS:
                signal.signal(s, signal.SIG_IGN)   # a second Ctrl-C must not cut the cleanup short
            if not j.delete():
                raise SystemExit(f"pod {j.name} may still be running: delete it by hand")


if __name__ == "__main__":
    main()
