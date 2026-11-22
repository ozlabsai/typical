"""Train the vision student (Gemma-4-12B-it LoRA, R056 recipe) on vision_v0 on one rented H100, then evaluate base vs tuned.

uv run --no-project python scripts/helipad_vision_train.py --data <vision_v0> --out <local results dir> [--steps 1620]
The pod (scripts/helipad_pod.py) gets the repo, vision_v0 and the HF token (a 0600 file inside the upload, never argv);
it trains, picks the saved step with the lowest dev NLL, runs `helipad_vision_eval.py score` and sends back eval/,
the chosen adapter, metrics.jsonl and train.log. The pod is deleted on every exit path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helipad_pod as hp  # noqa: E402

GEMMA = ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")   # scripts/hejev_pod_run.sh G
PINNED = ("torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 peft==0.21.0 accelerate==1.13.0 safetensors==0.8.0 "
          "sentencepiece==0.2.2")   # scripts/r055_pod.sh rsetup's pinned train venv
O = "/workspace/out/vision_v0"
REMOTE_SH = f"""export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=300 HF_HOME=/workspace/hf
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
set -a; . /workspace/hf.env; set +a; rm -f /workspace/hf.env
R=/workspace/pcdm V=/workspace/pkg/vision_v0 O={O} PY=/workspace/venv/bin/python
mkdir -p $O
(uv venv -q /workspace/venv && uv pip install -q --python $PY --torch-backend auto {PINNED} pillow) >$O/setup.log 2>&1 & a=$!
uvx --from huggingface_hub hf download {GEMMA[0]} --revision {GEMMA[1]} >$O/download.log 2>&1 & b=$!
wait $a || {{ tail -30 $O/setup.log; exit 1; }}
wait $b || {{ tail -30 $O/download.log; exit 1; }}
GM=$HF_HOME/hub/models--google--gemma-4-12B-it/snapshots/{GEMMA[1]}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cd $R
PYTHONPATH=$R $PY scripts/helipad_vision_eval.py score --model $GM --data $V --out $O/smoke --limit 2 >$O/smoke.log 2>&1 \\
  || {{ tail -40 $O/smoke.log; exit 1; }}   # the eval path works before any training time is spent
echo "eval smoke ok: $(head -c 300 $O/smoke/report.json | tr -d '\\n ')"
echo "train start $(date -u +%FT%TZ)"
PYTHONPATH=$R /workspace/venv/bin/torchrun --standalone --nproc_per_node 1 -m scripts.train_gemma_decision_lora --model $GM \\
  --train $V/train.jsonl --val dev=$V/dev.jsonl --out $O --steps $1 --microbatch 1 --grad-accum 8 --rank 32 --alpha 64 \\
  --lr 5e-5 --min-lr 5e-6 --max-length 4096 --eval-every 100 --save-every 100 --seed 56 >$O/train.log 2>&1 \\
  || {{ tail -40 $O/train.log; exit 1; }}
echo "train done $(date -u +%FT%TZ)"
BEST=$($PY -c '
import json, os
rows = [json.loads(x) for x in open("{O}/metrics.jsonl")]
s = [(r["validation"]["dev"]["nll"], r["step"]) for r in rows if r.get("step") and "validation" in r
     and os.path.isdir("{O}/step-%06d" % r["step"])]
print("step-%06d" % min(s)[1])')
echo "best dev checkpoint: $BEST"
cp -r $O/$BEST $O/adapter
PYTHONPATH=$R $PY scripts/helipad_vision_eval.py score --model $GM --adapter $O/adapter --data $V --out $O/eval \\
  >$O/eval.log 2>&1 || {{ tail -40 $O/eval.log; echo "EVAL FAILED: the adapter still comes back"; }}
echo "eval done $(date -u +%FT%TZ)"
"""


def hf_token() -> str:
    return os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, help="default: build.json's 3-epoch count")
    ap.add_argument("--timeout-h", type=float, default=5)
    a = ap.parse_args(argv)
    steps = a.steps or json.loads((a.data / "build.json").read_text())["steps_3_epochs"]
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / "hf.env"
        env.touch(0o600)
        env.write_text(f"HF_TOKEN={hf_token()}\n")
        fetch = {f"{O}/{n}": a.out / n for n in ("adapter", "metrics.jsonl", "train.log", "eval")}   # eval last: may be missing
        hp.run("helipadvision", {**hp.repo_uploads(), a.data: "pkg/vision_v0", env: "hf.env"}, REMOTE_SH, fetch,
               timeout_s=int(a.timeout_h * 3600), args=str(steps))


if __name__ == "__main__":
    main()
