"""R054 P0e: which saved Typical WorkflowEvals runs would WorkflowEvals' own plot.py merge?

`plot.py average` groups runs by (model.name, model.effort, question_mode). Every saved Typical run
was made with `--model typesafe:jev-1.13.0 --base-url <ours>`, so it is recorded as Jev.
This replays plot.py's grouping over the saved runs plus the published HF runs; nothing is written.

uv run --project .context/research_sources/WorkflowEvals python scripts/r054_p0e_merge_check.py
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WE = ROOT / ".context" / "research_sources" / "WorkflowEvals"
LOCAL = ["r044-workflows", "g0-cygnet", "g0-intern", "g0-winnow"] + sys.argv[1:]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(WE))
from core.workflow import WORKFLOWS  # noqa: E402
from plot import combine, published_runs  # noqa: E402

groups = defaultdict(set)
for wf in WORKFLOWS:
    paths = [ROOT / ".context" / d / wf for d in LOCAL if (ROOT / ".context" / d / wf / "results.json").exists()]
    bundle = combine(paths, "consensus", published_runs(wf))
    for run in bundle.runs:
        key = (run.model.name, run.model.effort, run.configuration.get("model", {}).get("question_mode"))
        groups[key].add(run.run_id)
for key, runs in sorted(groups.items(), key=str):
    flag = "MERGED" if len(runs) > 1 else ""
    print(f"{flag:7s} {key}: {sorted(runs)}")
