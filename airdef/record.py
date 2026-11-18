"""Typical's answers, recorded once from the real model and replayed by the deployed demo (no model, no GPU).

    uv run --no-sync python airdef/record.py [--only northern_barrage ...] [--out airdef/recorded/answers.json.gz]

Every scenario (per recorded policy) and every drill case is played through bake.bake -- the hosted build's own loop:
the server's clock, the director and its stories -- with the real model, so the build asks exactly these questions.
Recording on any other clock misses re-asks: a story disables the quiet-time skip, which moves when calls are re-asked.
"""
import argparse
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario names, name@r<posture>@s<priority option> (those policy variants), or drill:<family>")
    ap.add_argument("--out", default=str(DEFAULT))
    ap.add_argument("--model", default=str(ROOT / ".context/typically/results/co_airdef_v5/best.pt"))
    a = ap.parse_args()
    import scorer
    from sim import SCENARIOS
    m = scorer.load(a.model)
    out = Path(a.out)
    rows = {}
    if out.exists():   # add to an existing recording
        with gzip.open(out, "rt") as f:
            rows = {(q, s, tuple(lab)): d for q, s, lab, d in json.load(f)}
    from sim import policy_variants
    jobs = []
    for spec in a.only or list(SCENARIOS) + ["drill:*"]:
        name, *f = spec.split("@")   # name@r<posture> and/or name@s<priority option>
        f = {x[0]: int(x[1:]) for x in f}
        if name.startswith("drill:"):
            fam = name.split(":")[1]
            jobs += [("drill", r["meta"]["family"], r["meta"]["inst"]) for r in map(json.loads, DRILLS.read_text().splitlines())
                     if fam in ("*", r["meta"]["family"])]
        else:   # every recorded policy for the scenario (or for one magazine posture)
            jobs += [("scenario", name, v) for v in policy_variants(name, f.get("r"), f.get("s"))]
    import os
    os.environ.setdefault("AIRDEF_MODEL", "recorded")   # server.ASYNC off: the model answers in step, as the build replays it
    import bake
    for kind, name, inst in jobs:
        bake.bake(name, None, inst, model=m) if kind == "scenario" else bake.bake(None, (name, inst), model=m)
        w = bake.server.S["world"]
        n0 = len(rows)
        rows.update({k: d for k, (d, _) in w.memo.items()})
        print(f"{kind:8s} {name:22s} {inst if inst is not None else '':>2}  {len(w.memo):6d} questions, {len(rows) - n0:6d} new", flush=True)
        save(rows, out)   # after every job: a stopped run keeps what it finished
    print(f"{len(rows)} answers -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
