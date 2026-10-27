"""Run harness.py for base and tuned Typical over every scenario x seed on one rented H100, fetch the ledgers, delete the pod.

    uv run --no-project python airdef/remote_harness.py [--ckpts v3=.context/typically/results/co_airdef_v3/best.pt v4=...] [--models v3 v4] [--seeds 1 2 3]

Reuses scripts/typically_job.Job for the pod lifecycle (create / ssh / delete-in-finally). Results land in
.context/airdef/remote/<run id>/*.json; the pod log in .context/typically/jobs/airdef_eval/job.log.
"""
import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT_S = 150 * 60   # whole remote run, pod create to fetch
sys.path.insert(0, str(ROOT / "scripts"))
# own pod-name prefix: typically_job's reconcile (run before every training job) deletes "typically-job-*" pods it
# doesn't own -- on 2026-10-27 it killed this eval pod a minute in. Must be set before the import reads it.
os.environ["TYPICALLY_POD_PREFIX"] = "airdef-eval-"
import typically_job as tj   # noqa: E402

RUN_SH = """set -e
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=300 UV_PROJECT_ENVIRONMENT=/workspace/venv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/pcdm
uv sync
if [[ "$(nvidia-smi)" == *"CUDA Version: 12.8"* ]]; then
  uv pip install --python /workspace/venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
uv run --no-sync python -c "from huggingface_hub import snapshot_download, hf_hub_download; snapshot_download('Qwen/Qwen3-1.7B-Base'); hf_hub_download('OzLabs/typical-small', 'best.pt')"
mkdir -p out
# one job per (scenario, seed, model), longest scenarios first, so 10 parallel jobs stay busy to the end
for sc in {scenarios}; do for sd in {seeds}; do for m in {models}; do echo "$m $sc $sd"; done; done; done | \\
  xargs -P 10 -L 1 bash -c 'uv run --no-sync python airdef/harness.py "$1" --model "$(cat ckpt/$0.path)" --seeds $2 --json out/$0__$1__$2.json > out/$0__$1__$2.log 2>&1'
touch /workspace/done
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[f"tuned={ROOT / '.context/typically/results/co_airdef_t/best.pt'}"],
                    help="name=path/to/best.pt ... (base = the release, always available)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--models", nargs="+", default=["base", "tuned"])
    ap.add_argument("--scenarios", nargs="+", default=["northern_barrage", "april_mixed", "october_saturation", "june_attrition", "houthi_eilat"])
    a = ap.parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / ".context/airdef/remote" / run_id
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tj.job_dir("airdef_eval").mkdir(parents=True, exist_ok=True)   # the Job writes its pod log + API outputs here
        j = tj.Job("airdef_eval", tj.file_log("airdef_eval"), tmp)
        try:
            files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split()
            with tarfile.open(j.tar, "w:gz") as t:
                for f in files:
                    if not f.startswith(tj.NOT_SHIPPED) and (ROOT / f).is_file():
                        t.add(ROOT / f, arcname=f)
                ckpts = [("base", "OzLabs/typical-small")]
                for name, path in (c.split("=", 1) for c in a.ckpts):
                    t.add(path, arcname=f"ckpt/{name}/best.pt")
                    ckpts.append((name, f"ckpt/{name}/best.pt"))
                for name, path in ckpts:
                    (tmp / f"{name}.path").write_text(path)
                    t.add(tmp / f"{name}.path", arcname=f"ckpt/{name}.path")
            (tmp / "run.sh").write_text(RUN_SH.format(models=" ".join(a.models), scenarios=" ".join(a.scenarios), seeds=" ".join(map(str, a.seeds))))
            print(f"tarball {j.tar.stat().st_size / 1e6:.0f} MB; asking RunPod for an H100", flush=True)
            j.create()
            j.wait_ssh()
            scp = ["scp", *tj.SSH_OPTS, "-P", j.host[1]]
            dest = f"root@{j.host[0]}:"
            j.ssh("mkdir -p /workspace/pcdm")
            j.call([*scp, str(j.tar), dest + "/workspace/repo.tar.gz"], 900)
            j.call([*scp, str(tmp / "run.sh"), dest + "/workspace/run.sh"], 120)
            j.ssh("tar xzf /workspace/repo.tar.gz -C /workspace/pcdm && rm -f /workspace/done", 300)
            # launch on its own line: backgrounding a whole `a && b &` chain keeps ssh's stdout open until it ends
            j.ssh("setsid nohup bash /workspace/run.sh > /workspace/run.log 2>&1 < /dev/null & echo $! > /workspace/run.pid", 60)
            print(f"running on {j.gpu}; polling", flush=True)
            j.deadline = time.monotonic() + LIMIT_S   # this harness's own limit, not the training job's 75 min
            bad = 0
            while time.monotonic() < j.deadline - 300:   # leave time to fetch whatever finished
                time.sleep(30)
                try:   # a dropped ssh must not kill a long run (same rule as typically_job's poller)
                    state = j.ssh("test -f /workspace/done && echo DONE || (ls /workspace/pcdm/out 2>/dev/null | grep -c json || true)", 60)   # grep -c exits 1 on 0
                    bad = 0
                except RuntimeError as e:
                    bad += 1
                    print(f"poll failed ({bad}/3): {e}", flush=True)
                    if bad >= 3:
                        raise
                    continue
                print(f"finished runs: {state.strip()}", flush=True)
                if state.startswith("DONE"):
                    break
            else:
                print(f"not finished in {LIMIT_S // 60} minutes: fetching the runs that did", flush=True)
        finally:
            if j.host:   # fetch before delete on every path: partial results are still results
                j.deadline = max(j.deadline, time.monotonic() + 400)
                scp, dest = ["scp", *tj.SSH_OPTS, "-P", j.host[1]], f"root@{j.host[0]}:"
                try:
                    j.call([*scp, "-r", dest + "/workspace/pcdm/out/", str(out)], 240)
                    j.call([*scp, dest + "/workspace/run.log", str(out / "run.log")], 60)
                    print(f"results in {out}", flush=True)
                except Exception as e:
                    print(f"fetch failed: {e}", flush=True)
            print("pod deleted" if j.delete() else "pod deletion unconfirmed: check runpodctl pod list", flush=True)


if __name__ == "__main__":
    main()
