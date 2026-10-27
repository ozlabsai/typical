"""Load a Typical checkpoint for airdef through the training repo's own decider.

Why not inference/typical: on 2026-10-27 its native_kv_decide port disagreed with the training path on this task --
max |dp| 0.004-0.09 on the released typical-small and up to 0.7 on the airdef fine-tune whenever K >= 3
(training-path eval 88% vs port 3% on the same rows; .context/airdef/dbg_row*.py). Until that port is fixed, score
the way the checkpoint was trained and evaluated.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "pcdm"), str(ROOT / "pcdm_jev")]


class Scorer:
    """model.choice(state, question, labels) -> {label: p, ..., "p_null": p} -- the interface engine.py uses."""

    def __init__(self, model_id, device="auto"):
        from pcdm_jev.decider import PCDMDecider
        if os.path.isfile(model_id):
            run_dir = os.path.dirname(model_id)
        else:
            from huggingface_hub import hf_hub_download
            run_dir = os.path.dirname(hf_hub_download(model_id, "best.pt"))
        self.d = PCDMDecider(run_dir, mode="native", device=device)

    def choice(self, state, question, labels):
        p = self.d._probs("native", state, question, labels).tolist()
        return {**dict(zip(labels, p[:-1])), "p_null": p[-1]}


def load(model_id, device="auto"):
    return Scorer(model_id, device)
