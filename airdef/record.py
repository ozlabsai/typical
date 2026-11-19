"""Typical's answers, recorded once from the real model and replayed by the deployed demo (no model, no GPU).

    uv run --no-sync python airdef/record.py [--only northern_barrage ...] [--out airdef/recorded/answers.json.gz]

Every scenario (per recorded policy) and every drill case is played through bake.bake -- the hosted build's own loop:
the server's clock, the director and its stories -- with the real model, so the build asks exactly these questions.
Recording on any other clock misses re-asks: a story disables the quiet-time skip, which moves when calls are re-asked.
"""
import argparse
import functools
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = Path(__file__).resolve().parent / "recorded" / "answers.json.gz"
DRILLS = Path(__file__).resolve().parent / "recorded" / "drill_ui.jsonl"


@functools.lru_cache(maxsize=2)
def _load(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    jobs = data["jobs"] if isinstance(data, dict) else {"*": data}   # a plain list: one recording, no per-job split
    return {spec: {(q, s, tuple(lab)): d for q, s, lab, d in rows} for spec, rows in jobs.items()}


class Recorded:
    """model.choice(state, question, labels) from the recording; None for a situation it never saw.
    tape: replay one tape from the job that recorded it. The GPU answers a question slightly differently from one job to
    the next (up to ~0.03), and a near-tie that flips sends the replay down a path nobody recorded."""

    def __init__(self, path=None, tape=None):
        jobs = _load(str(path or DEFAULT))
        own = next((m for spec, m in jobs.items() if tape and tape in tapes_of(spec)), None)
        self.memo = own if own is not None else {k: d for m in reversed(list(jobs.values())) for k, d in m.items()}
        self.misses = 0

    def choice(self, state, question, labels):
        d = self.memo.get((question, state, tuple(labels)))
        self.misses += d is None
        return d


def tape_name(scenario, variant=None, drill=None):
    from sim import DEFAULT_VARIANT
    if drill:
        return f"drill_{drill[0]}_{drill[1]}"
    return scenario if variant in (None, DEFAULT_VARIANT) else f"{scenario}@{variant}"


@functools.lru_cache(maxsize=None)
def tapes_of(spec):
    """The tapes a recording job covered, from its spec (= its file name): name@r<posture>@f<floor>@s<priority>, drill_<family>."""
    from sim import policy_variants
    if spec.startswith("drill"):
        fam = spec.replace("drill:", "drill_").split("_", 1)[1]
        return frozenset(tape_name(None, drill=(r["family"], r["inst"])) for r in (json.loads(x)["meta"] for x in DRILLS.read_text().splitlines())
                         if fam in ("*", r["family"]))
    name, *f = spec.split("@")
    f = {x[0]: int(x[1:]) for x in f}
    return frozenset(tape_name(name, v) for v in policy_variants(name, f.get("r"), f.get("s"), f.get("f")))


def save(rows, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump([[q, s, list(lab), d] for (q, s, lab), d in rows.items()], f)
    tmp.replace(out)


def merge(paths, out):
    """Recordings (one per remote job, named by its spec) -> one file that keeps each job's answers apart."""
    jobs = {}
    for p in paths:
        with gzip.open(p, "rt") as f:
            jobs[Path(p).name.removesuffix(".json.gz")] = json.load(f)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump({"jobs": jobs}, f)
    return sum(map(len, jobs.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario names, name@r<posture>@f<floor>@s<priority option> (those policy variants), or drill:<family>")
    ap.add_argument("--out", default=str(DEFAULT))
    ap.add_argument("--model", default=str(ROOT / ".context/typically/results/co_airdef_v8/best.pt"))
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
        name, *f = spec.split("@")   # name@r<posture>, @f<floor>, @s<priority option>, in any combination
        f = {x[0]: int(x[1:]) for x in f}
        if name.startswith("drill:"):
            fam = name.split(":")[1]
            jobs += [("drill", r["meta"]["family"], r["meta"]["inst"]) for r in map(json.loads, DRILLS.read_text().splitlines())
                     if fam in ("*", r["meta"]["family"])]
        else:   # every recorded policy for the scenario (or for one magazine posture)
            jobs += [("scenario", name, v) for v in policy_variants(name, f.get("r"), f.get("s"), f.get("f"))]
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
