"""Score a helipad checkpoint with scripts/helipad_score.py on a rented H100.
Qwen3.5's linear-attention layers need the fla/causal-conv1d kernels; on MPS they fall back to a reference path that did not finish
150 cases in 40 min. Same image and setup as Decision Factory jobs (scripts/typically_job.py, typically_spike_pod.sh).

uv run --no-project python scripts/helipad_score_pod.py <best.pt> <out.json> [--limit-cases N, 0 = all]
Needs runpodctl >= 3 (REST v2) configured, and the ~/.runpod/ssh/RunPod-Key-Go key.

The pod (<POD_PREFIX>helipadscore-<token>) is run and always deleted by scripts/helipad_pod.py; see its docstring for the
cleanup guarantees and the reconcile caveat. Don't score while the Decision Factory server runs locally.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helipad_pod as hp  # noqa: E402
import typically_job as tj  # noqa: E402

EVAL = tj.REPO / ".context" / "typically" / "jobs" / "helipad" / "eval"
FILES = ("helipad_real.jsonl", "helipad_expert.jsonl", "import_oneliner.jsonl", "helipad_contrast.jsonl")
SIGNALS = hp.SIGNALS
RUN_S = 2 * 3600   # cap on setup + scoring (about 15 min today): a hung ssh can't hold the pod forever


def remote_sh(files) -> str:
    return f"""set -eo pipefail
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
  --files {" ".join(f"$E/{f}" for f in files)} 2>&1 | {{ grep -v "it/s\\]" || true; }} | tail -20
"""


REMOTE_SH = remote_sh(FILES)


def uploads(ckpt: Path, eval_dir: Path, files) -> dict[Path, str]:
    """The repo under pcdm/, the eval files + checkpoint under pkg/."""
    if missing := [f for f in files if not (eval_dir / f).is_file()]:
        raise SystemExit(f"missing eval files in {eval_dir}: {missing}")
    return {**hp.repo_uploads(), **{p: f"pkg/eval/{p.name}" for p in sorted(eval_dir.glob("*.jsonl"))}, ckpt: "pkg/best.pt"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--eval-dir", type=Path, help=f"default {EVAL}")
    ap.add_argument("--files", nargs="+", default=list(FILES), help="file names in --eval-dir to score")
    a = ap.parse_args(argv)
    hp.run("helipadscore", uploads(a.ckpt, a.eval_dir or EVAL, a.files), remote_sh(a.files), {"/workspace/score.json": a.out},
           disk_gb=80, timeout_s=RUN_S, script="score.sh", args=str(a.limit_cases))


if __name__ == "__main__":
    main()
