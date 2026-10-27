"""Live air-defence decision demo server.

    uv run --no-sync uvicorn --app-dir airdef server:app --port 8790        (AIRDEF_MODEL=none: planner only)
"""
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import planner
from engine import DONE, World, label
from sim import (AIR_KINDS, BATTERIES, DEFAULT_CONFIG, EFFECTOR_HE, EFFECTORS, SCENARIOS, SITE_HE, SITES, THREAT_HE, THREATS,
                 consequence)

ROOT = Path(__file__).parent
TUNED = ROOT.parent / ".context/typically/results/co_airdef_v3/best.pt"   # v3: both decision levels (typically_job.py airdef_v3)
MODEL_ID = os.environ.get("AIRDEF_MODEL") or (str(TUNED) if TUNED.exists() else "OzLabs/typical-small")
TICK_S = 0.25          # real seconds per loop tick
BUDGET = 6             # Typical calls per tick; the sim slows down when decisions back up

app = FastAPI()
lock = threading.Lock()   # one lock for world + model (MPS crashes on concurrent calls)
S = dict(world=None, running=False, speed=10.0, model=None, model_name="planner only", backlog=False, skipped=0.0)
QUIET_S = 15           # auto-skip: jump when nothing happens for longer than this (sim seconds)


def load_model():
    if MODEL_ID == "none":
        return
    import scorer   # the training repo's decider; see scorer.py for why not inference/typical
    m = scorer.load(MODEL_ID)
    m.choice("warm up", "Is this a warm-up?", ["yes", "no"])
    S["model"], S["model_name"] = m, ("typical-small · tuned" if MODEL_ID.endswith(".pt") else MODEL_ID.split("/")[-1])


def new_world(scenario, seed=7, cfg=None):
    S["world"] = World(scenario, cfg or DEFAULT_CONFIG, S["model"], seed, async_model=True)
    S["running"] = False


def loop():
    """The sim clock: real elapsed time x speed, never paused for the model (a real battery won't wait either)."""
    last = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        real, last = min(t0 - last, 1.0), t0
        with lock:
            w = S["world"]
            if S["running"] and w and w.t < w.end:
                w.step(real * S["speed"])              # decide (from answers already in), resolve, fly aircraft, advance
                if (nxt := w.next_event()) - w.t > QUIET_S:
                    S["skipped"] = nxt - 5 - w.t       # the UI shows "skipped mm:ss"
                    w.advance(nxt - 5 - w.t)
            S["backlog"] = len(w.requests) if w else 0
        time.sleep(max(0.0, TICK_S - (time.perf_counter() - t0)))


def model_worker():
    """Scores queued situations outside the world lock, most urgent first, and drops each answer into the memo."""
    while True:
        with lock:
            w = S["world"]
            job = next(iter(w.requests.items()), None) if w and S["model"] else None
        if job is None:
            time.sleep(0.02)
            continue
        key, (q, text, labels) = job
        t0 = time.perf_counter()
        d = S["model"].choice(text, q, labels)
        ms = (time.perf_counter() - t0) * 1000
        with lock:
            if S["world"] is w:
                w.memo[key] = (d, ms)
                w.requests.pop(key, None)
                L = w.policies["typical"].L
                L["calls"] += 1
                L["ms"] += ms


@app.on_event("startup")
def startup():
    load_model()
    new_world("houthi_eilat")
    threading.Thread(target=loop, daemon=True).start()
    threading.Thread(target=model_worker, daemon=True).start()


# ------------------------------------------------------------------ api
@app.get("/api/meta")
def meta():
    return dict(
        scenarios={k: dict(title=v["title"], blurb=v["blurb"], title_he=v["title_he"], blurb_he=v["blurb_he"])
                   for k, v in SCENARIOS.items()},
        sites=[dict(id=s[0], name=s[1], name_he=SITE_HE[s[0]], lat=s[2], lon=s[3], r=s[4], tier=s[5]) for s in SITES],
        batteries=[dict(id=b[0], eff=b[1], lat=b[2], lon=b[3]) for b in BATTERIES],
        effectors={k: dict(name=e["name"], name_he=EFFECTOR_HE.get(k, e["name"]), cost=e["cost"], layer=e["layer"],
                           allied=e.get("allied", False), air=e.get("air", False)) for k, e in EFFECTORS.items()},
        air_kinds={k: dict(label=v["label"]) for k, v in AIR_KINDS.items()},
        threat_types={k: dict(name=v[0], name_he=THREAT_HE[k], cls=v[1], cost=v[2]) for k, v in THREATS.items()},
        config=DEFAULT_CONFIG, model=S["model_name"])


@app.get("/api/paths")
def paths():
    """Full sampled trajectories, once per scenario; the client interpolates positions by sim time."""
    w = S["world"]
    out = []
    for th in w.threats:
        ts = np.linspace(th.t0, th.t0 + th.T, 48 if th.cls in "BR" else 96)
        lat, lon, alt = th.pos(ts)
        out.append(dict(id=th.id, kind=th.kind, cls=th.cls, t0=th.t0, T=th.T,
                        p=[[round(float(a), 4), round(float(b), 4), round(float(c) * 1000), round(float(t), 1)]
                           for a, b, c, t in zip(lon, lat, alt, ts)]))
    return dict(scenario=w.scenario, end=w.end, threats=out)


@app.get("/api/state")
def state(policy: str = "typical"):
    with lock:
        w = S["world"]
        pol = w.policies[policy]
        tracks = []
        for th in w.threats:
            s = pol.st[th.id]
            if th.t0 > w.t or (s["status"] in DONE and w.t - (s.get("killed_at") or th.t0 + th.T) > 120 * max(1, S["speed"] / 10)):
                continue
            plat, plon, sigma = th.predicted_impact(w.t)
            d = (s["info"] or {}).get("dist")
            tracks.append(dict(id=th.id, kind=th.kind, cls=th.cls, status=s["status"], choice=s.get("choice"),
                               conf=max(d.values()) if d else None, escalated=s["escalated"], tti=th.t0 + th.T - w.t,
                               impact=th.t0 + th.T, ipp=[round(plon, 4), round(plat, 4), round(sigma, 2)], site=s.get("site")))
        return dict(t=w.t, end=w.end, running=S["running"], speed=S["speed"], backlog=S["backlog"],
                    skipped=S["skipped"],
                    scenario=w.scenario, model=S["model_name"], cfg=w.cfg, tracks=tracks,
                    flights=[dict(b=f[1], th=f[2], t0=f[3], t1=f[4], lat=f[5], lon=f[6], alt=f[7] * 1000, hit=f[8],
                                  slat=f[9], slon=f[10], eff=f[11]) for f in w.flights if f[0] == policy],
                    units=[dict(id=u.id, kind=u.kind, name=u.name, lat=u.lat, lon=u.lon, state=u.state, task=u.task,
                                goal=u.goal) for u in pol.units.values()],
                    killed={tid: s["killed_at"] for tid, s in pol.st.items() if s.get("killed_at")},
                    ledger=w.ledger(), attacker_cost=w.attacker_cost, launched=sum(th.t0 <= w.t for th in w.threats),
                    total=len(w.threats))


@app.get("/api/track/{tid}")
def track(tid: str, policy: str = "typical"):
    """Everything the threat card shows: identity, the timeline (windows, shots, past calls) and the current call."""
    with lock:
        w = S["world"]
        if tid not in w.by_id:
            raise HTTPException(404)
        th, pol = w.by_id[tid], w.policies[policy]
        s = pol.st[tid]
        _, sid, _, _, dist, sigma = consequence(th, w.t, w.cfg)
        return dict(id=tid, kind=th.kind, cls=th.cls, status=s["status"], t=w.t, t0=th.t0, impact=th.t0 + th.T,
                    site=sid, open_ground=bool(dist > sigma), priority=sid in w.cfg["priority_sites"],
                    windows=planner.spans(th, w.t, pol.stock, w.cfg),
                    shots=[dict(b=f[1], eff=f[11], t0=f[3], t1=f[4], hit=f[8]) for f in w.flights if f[0] == policy and f[2] == tid],
                    history=s["history"], info=s["info"], local=s.get("local"), assign=s.get("assign"),
                    escalated=s["escalated"], choice=s.get("choice"),
                    others={n: p.st[tid].get("choice") for n, p in w.policies.items()},
                    labels={o[0]: o[1] for o in (s["info"] or {}).get("opts", [])})


class Control(BaseModel):
    action: str
    scenario: str | None = None
    speed: float | None = None
    seed: int | None = None


@app.post("/api/control")
def control(c: Control):
    with lock:
        w = S["world"]
        if c.action == "play":
            S["running"] = True
        elif c.action == "pause":
            S["running"] = False
        elif c.action == "speed" and c.speed:
            S["speed"] = max(0.25, min(c.speed, 600))
        elif c.action == "skip":   # jump to just before the next thing worth watching
            w.t = max(w.t, w.next_event() - 5)
        elif c.action == "reset":
            if c.scenario and c.scenario not in SCENARIOS:
                raise HTTPException(400, "unknown scenario")
            new_world(c.scenario or w.scenario, c.seed or w.seed, w.cfg)
        return {"ok": True}


@app.post("/api/config")
def config(patch: dict):
    bad = set(patch) - set(DEFAULT_CONFIG)
    if bad:
        raise HTTPException(400, f"unknown keys {sorted(bad)}")
    for k in ("site_value", "warhead", "tier_value"):   # numeric maps: keep known keys, clamp to sane ranges
        if k in patch:
            known = {"site_value": {s[0] for s in SITES}, "warhead": set(DEFAULT_CONFIG["warhead"]),
                     "tier_value": set(DEFAULT_CONFIG["tier_value"])}[k]
            patch[k] = {kk: min(max(float(v), 0.0), 100_000.0) for kk, v in dict(patch[k]).items() if kk in known}
    with lock:
        S["world"].set_config(patch)
        return {"ok": True, "cfg": S["world"].cfg}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")
app.mount("/brand", StaticFiles(directory=ROOT / "brand"), name="brand")   # the taste index: tokens + components
