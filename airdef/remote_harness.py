"""Run harness.py for base and tuned Typical over every scenario x seed on one rented H100, fetch the ledgers, delete the pod.

    uv run --no-project python airdef/remote_harness.py [--tuned .context/typically/results/co_airdef_t/best.pt] [--seeds 1 2 3]

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
for m in {models}; do for sc in {scenarios}; do echo "$m $sc"; done; done | \\
  xargs -P 8 -L 1 bash -c 'uv run --no-sync python airdef/harness.py "$1" --model "$(cat ckpt/$0.path)" --seeds {seeds} --json out/$0__$1.json > out/$0__$1.log 2>&1'
touch /workspace/done
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", default=str(ROOT / ".context/typically/results/co_airdef_t/best.pt"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
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
                t.add(a.tuned, arcname="ckpt/tuned/best.pt")
                for name, path in (("base", "OzLabs/typical-small"), ("tuned", "ckpt/tuned/best.pt")):
                    (tmp / f"{name}.path").write_text(path)
                    t.add(tmp / f"{name}.path", arcname=f"ckpt/{name}.path")
            (tmp / "run.sh").write_text(RUN_SH.format(models="base tuned", scenarios=" ".join(a.scenarios), seeds=" ".join(map(str, a.seeds))))
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
            deadline = time.monotonic() + 90 * 60
            while time.monotonic() < deadline:
                time.sleep(30)
                state = j.ssh("test -f /workspace/done && echo DONE || (tail -n 2 /workspace/run.log; ls /workspace/pcdm/out 2>/dev/null | grep -c json)", 60)
                print(state.strip().splitlines()[-1] if state.strip() else "", flush=True)
                if state.startswith("DONE"):
                    break
            else:
                raise RuntimeError("remote harness did not finish in 90 minutes")
            j.call([*scp, "-r", dest + "/workspace/pcdm/out/", str(out)], 600)
            j.call([*scp, dest + "/workspace/run.log", str(out / "run.log")], 120)
            print(f"results in {out}", flush=True)
        finally:
            print("pod deleted" if j.delete() else "pod deletion unconfirmed: check runpodctl pod list", flush=True)


if __name__ == "__main__":
    main()
