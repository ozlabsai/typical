"""Typical's answers, recorded once from the real model and replayed by the deployed demo (no model, no GPU).

    uv run --no-sync python airdef/record.py [--only northern_barrage ...] [--out airdef/recorded/answers.json.gz]

Every scenario and every drill case is played through the live demo's own clock (World.run_for: fixed DT steps, the
same quiet-time skips) with the real model, so the deployed server meets exactly these situations and asks exactly
these questions, whatever the speed, pauses, modes or replays. A situation never recorded (Settings changed) gets the
planner's call, as a late answer does in the live demo.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = Path(__file__).resolve().parent / "recorded" / "answers.json.gz"
DRILLS = Path(__file__).resolve().parent / "recorded" / "drill_ui.jsonl"


class Recorded:
    """model.choice(state, question, labels) from the recording; None for a situation it never saw."""

    def __init__(self, path=None):
        with gzip.open(path or DEFAULT, "rt") as f:
            self.memo = {(q, s, tuple(lab)): d for q, s, lab, d in json.load(f)}
        self.misses = 0

    def choice(self, state, question, labels):
        d = self.memo.get((question, state, tuple(labels)))
        self.misses += d is None
        return d


def scenario_world(name, model):
    from engine import DEFAULT_CONFIG, World
    return World(name, DEFAULT_CONFIG, model, 7, policies=("typical",), async_model=False)   # as server.new_world


def drill_world(fam, inst, model):
    import situations
    w, th, t = situations.case(fam, inst)   # as server.new_drill
    w.t, w.model, w.async_model = t, model, False
    w.policies["typical"] = copy.deepcopy(w.policies["planner"])
    w.policies["typical"].name = "typical"
    return w


def save(rows, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump([[q, s, list(lab), d] for (q, s, lab), d in rows.items()], f)
    tmp.replace(out)


def merge(paths, out):
    """Several recordings (one per remote job) -> one file."""
    rows = {}
    for p in paths:
        with gzip.open(p, "rt") as f:
            rows.update({(q, s, tuple(lab)): d for q, s, lab, d in json.load(f)})
    save(rows, Path(out))
    return len(rows)


def play(w, paused_first=False):
    if paused_first:
        w.step(0.0)   # a drill loads paused and makes its call at the frozen moment
    while w.t < w.end:
        w.run_for(600.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario names or drill:<family>")
    ap.add_argument("--out", default=str(DEFAULT))
    ap.add_argument("--model", default=str(ROOT / ".context/typically/results/co_airdef_v3/best.pt"))
    a = ap.parse_args()
    import scorer
    from sim import SCENARIOS
    m = scorer.load(a.model)
    out = Path(a.out)
    rows = {}
    if out.exists():   # add to an existing recording
        with gzip.open(out, "rt") as f:
            rows = {(q, s, tuple(lab)): d for q, s, lab, d in json.load(f)}
    jobs = [("scenario", sc, None) for sc in SCENARIOS]
    jobs += [("drill", r["meta"]["family"], r["meta"]["inst"]) for r in map(json.loads, DRILLS.read_text().splitlines())]
    for kind, name, inst in jobs:
        if a.only and name not in a.only and f"drill:{name}" not in a.only:
            continue
        w = scenario_world(name, m) if kind == "scenario" else drill_world(name, inst, m)
        play(w, paused_first=kind == "drill")
        n0 = len(rows)
        rows.update({k: d for k, (d, _) in w.memo.items()})
        print(f"{kind:8s} {name:22s} {inst if inst is not None else '':>2}  {len(w.memo):6d} questions, {len(rows) - n0:6d} new", flush=True)
        save(rows, out)   # after every job: a stopped run keeps what it finished
    print(f"{len(rows)} answers -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
