"""Bake the demo into static tapes: every scenario and drill case, played once through the live server's own clock and
director with Typical's recorded answers, written as a compact timeline the browser plays back (web/tape.js).

    AIRDEF_MODEL=recorded uv run --no-sync python airdef/bake.py [--out airdef/static]

A tape holds: the paths; one frame per sim step (the step's time, and only what changed: tracks, fire units, the
ledger, the tally; aircraft every frame); each threat's detail as it changed (status, call, Typical's answer, the fire
unit's call); the full call history, decision log and every shot; the key moments Focus tells, as live autoplay at 20x
would pick them. The output directory is the whole static site (page, scripts, brand, tapes).
"""
import argparse
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("AIRDEF_MODEL", "recorded")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import director   # noqa: E402
import record     # noqa: E402
import server     # noqa: E402
from engine import DONE   # noqa: E402
from sim import SCENARIOS, consequence   # noqa: E402

TICK = 0.25        # wall seconds per server tick, as server.loop
SPEED = 20.0       # the autoplay speed the Focus stories are picked at


class Clock:   # the director's wall clock, advanced by the bake loop instead of time passing
    now = 0.0

    def monotonic(self):
        return self.now


def changed(prev, cur):
    return json.dumps(prev, sort_keys=True, default=str) != json.dumps(cur, sort_keys=True, default=str)


def detail_of(w, th):
    s = w.policies["typical"].st[th.id]
    _, sid, _, _, dist, sigma = consequence(th, w.t, w.cfg)
    return dict(status=s["status"], choice=s.get("choice"), info=s["info"], local=s.get("local"), assign=s.get("assign"),
                site=sid, open_ground=bool(dist > sigma), priority=sid in w.cfg["priority_sites"])


def bake(scenario=None, drill=None):
    clock = Clock()
    director.time = clock   # the director only reads time.monotonic()
    S = server.S
    S.update(model=record.Recorded(), model_name="typical-small · tuned", batch=1, mode="focus", speed=SPEED)
    if drill:
        server.new_drill(*drill)
    else:
        server.new_world(scenario)
    w = S["world"]
    frames, prev_tracks, prev = [], {}, dict(bat={}, ledger=None, stats=None)
    details = {th.id: [] for th in w.threats}
    last_detail = {}

    def capture(w):
        st = server.state()
        f = dict(t=round(w.t, 3))
        tracks = {x["id"]: {k: v for k, v in x.items() if k != "tti"} for x in st["tracks"]}
        upd = {i: x for i, x in tracks.items() if changed(prev_tracks.get(i), x)}
        gone = [i for i in prev_tracks if i not in tracks]
        if upd:
            f["upd"] = upd
        if gone:
            f["gone"] = gone
        prev_tracks.clear(); prev_tracks.update(tracks)
        bat = [b for b in st["batteries"] if changed(prev["bat"].get(b["id"]), b)]
        if bat:
            f["bat"] = bat
            prev["bat"].update({b["id"]: b for b in bat})
        f["units"] = st["units"]
        for k in ("ledger", "stats"):
            if st.get(k) is not None and changed(prev[k], st[k]):
                f[k] = st[k]; prev[k] = json.loads(json.dumps(st[k]))
        frames.append(f)
        for th in w.threats:
            if th.t0 <= w.t:
                d = detail_of(w, th)
                if changed(last_detail.get(th.id), d):
                    last_detail[th.id] = json.loads(json.dumps(d, default=str))
                    details[th.id].append(dict(t=round(w.t, 3), **d))

    if drill:
        w.step(0.0)   # loaded paused: the call at the frozen moment
    capture(w)
    S["running"] = True
    while w.t < w.end:   # server.loop, tick by tick, on the virtual clock
        clock.now += TICK
        d = S["director"]
        rate = d.speed(w, S["speed"]) if d else S["speed"]

        def on_step(w):
            if S["director"]:
                S["director"].tick(w)
                S["director"].offer(w, S["director"].scan(w))
            capture(w)
        w.run_for(TICK * rate, on_step)
        if S["director"]:
            S["director"].tick(w)
    final = server.state()
    pol = w.policies["typical"]
    moments = final.get("moments", [])
    for m in moments:   # when each story ends: its threat resolved (follow) or a short beat
        res = next((x["t"] for x in details[m["tid"]] if x["status"] in DONE), None)
        m["t_done"] = res if m["type"] in director.FOLLOW and res is not None else m["t"] + director.BEAT_S
    return dict(
        scenario=w.scenario, end=w.end, total=len(w.threats), cfg=w.cfg, attacker_cost=w.attacker_cost, drill=final.get("drill"),
        paths=server.paths(), frames=frames, flights=final["flights"], killed=final["killed"], moments=moments,
        threats={th.id: dict(id=th.id, kind=th.kind, cls=th.cls, t0=th.t0, impact=th.t0 + th.T, timeline=details[th.id],
                             history=pol.st[th.id]["history"], log=pol.st[th.id].get("log", [])) for th in w.threats})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "static"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "data").mkdir(parents=True)
    shutil.copytree(HERE / "web", out / "web")
    shutil.copytree(HERE / "brand", out / "brand")
    page = (HERE / "web" / "index.html").read_text()   # the static page: flag + the tape player before app.js
    page = page.replace('<script src="/web/app.js"></script>',
                        '<script>window.AIRDEF_STATIC = true;</script><script src="/web/tape.js"></script><script src="/web/app.js"></script>')
    (out / "index.html").write_text(page)
    (out / "data" / "meta.json").write_text(json.dumps(server.meta()))
    jobs = [(sc, None) for sc in SCENARIOS]
    jobs += [(None, (r["meta"]["family"], n)) for fam in sorted({json.loads(x)["meta"]["family"] for x in server.DRILL.read_text().splitlines()})
             for n, r in enumerate([json.loads(x) for x in server.DRILL.read_text().splitlines() if json.loads(x)["meta"]["family"] == fam])]
    for sc, dr in jobs:
        name = sc or f"drill_{dr[0]}_{dr[1]}"
        if a.only and name not in a.only:
            continue
        tape = bake(sc, dr)
        p = out / "data" / f"{name}.json.gz"
        with gzip.open(p, "wt") as f:
            json.dump(tape, f, separators=(",", ":"), default=str)
        print(f"{name:32s} {len(tape['frames']):5d} frames  {len(tape['moments']):2d} moments  {p.stat().st_size / 1e6:5.2f} MB", flush=True)
