"""Live air-defence FinOps demo server.

    uv run --no-sync uvicorn --app-dir airdef server:app --port 8790        (AIRDEF_MODEL=none for optimizer-only)
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

from engine import World, parse_ok
from sim import BATTERIES, DEFAULT_CONFIG, EFFECTORS, SCENARIOS, SITES, THREATS

ROOT = Path(__file__).parent
MODEL_ID = os.environ.get("AIRDEF_MODEL", "OzLabs/typical-small")
TICK_S = 0.25          # real seconds per loop tick
BUDGET = 6             # Typical calls per tick; the sim slows down when decisions back up

app = FastAPI()
lock = threading.Lock()   # one lock for world + model (MPS crashes on concurrent calls)
S = dict(world=None, running=False, speed=10.0, model=None, model_name="optimizer only", backlog=False)


def load_model():
    if MODEL_ID == "none":
        return
    sys.path.insert(0, str(ROOT.parent / "inference"))
    from typical import Typical
    m = Typical.from_pretrained(MODEL_ID, device="auto", max_states=64)
    m.choice("warm up", "Is this a warm-up?", ["yes", "no"])
    S["model"], S["model_name"] = m, MODEL_ID.split("/")[-1]


def new_world(scenario, seed=7, cfg=None):
    S["world"] = World(scenario, cfg or DEFAULT_CONFIG, S["model"], seed)
    S["running"] = False


def loop():
    while True:
        t0 = time.perf_counter()
        with lock:
            w = S["world"]
            if S["running"] and w and w.t < w.end:
                calls = {n: w.plan_threats(p, BUDGET) for n, p in w.policies.items()}
                for p in w.policies.values():
                    w.execute(p)
                S["backlog"] = calls["typical"] >= BUDGET
                dt = TICK_S * S["speed"]
                w.t = min(w.t + (min(dt, 1.0) if S["backlog"] else dt), w.end)
        time.sleep(max(0.0, TICK_S - (time.perf_counter() - t0)))


@app.on_event("startup")
def startup():
    load_model()
    new_world("houthi_eilat")
    threading.Thread(target=loop, daemon=True).start()


# ------------------------------------------------------------------ api
@app.get("/api/meta")
def meta():
    return dict(
        scenarios={k: dict(title=v["title"], blurb=v["blurb"]) for k, v in SCENARIOS.items()},
        sites=[dict(id=s[0], name=s[1], lat=s[2], lon=s[3], r=s[4], tier=s[5]) for s in SITES],
        batteries=[dict(id=b[0], eff=b[1], lat=b[2], lon=b[3]) for b in BATTERIES],
        effectors={k: dict(name=e["name"], cost=e["cost"], rmax=e["rmax"], layer=e["layer"],
                           allied=e.get("allied", False)) for k, e in EFFECTORS.items()},
        threat_types={k: dict(name=v[0], cls=v[1], cost=v[2]) for k, v in THREATS.items()},
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


def track_row(w, th, s):
    plat, plon, sigma = th.predicted_impact(w.t)
    i = s["info"] or {}
    d = i.get("dist")
    return dict(id=th.id, kind=th.kind, status=s["status"], tti=th.t0 + th.T - w.t,
                plan=s["plan"].key if s["plan"] else None, label=s["plan"].label if s["plan"] else None,
                conf=(max(d.values()) if d else None), p_null=i.get("p_null"), opt=i.get("opt"),
                escalated=s["escalated"], ipp=[round(plon, 4), round(plat, 4), round(sigma, 2)], site=s.get("site"))


@app.get("/api/state")
def state(policy: str = "typical"):
    with lock:
        w = S["world"]
        pol = w.policies[policy]
        live = [track_row(w, th, pol.st[th.id]) for th in w.threats
                if th.t0 <= w.t and (pol.st[th.id]["status"] not in ("killed", "leaked", "fell_open")
                                     or w.t - (th.t0 + th.T) < 600 * max(1, S["speed"] / 10))]
        flights = [f for f in w.flights if f[0] == policy]
        return dict(t=w.t, end=w.end, running=S["running"], speed=S["speed"], backlog=S["backlog"],
                    scenario=w.scenario, model=S["model_name"], cfg=w.cfg, tracks=live,
                    flights=[dict(b=f[1], th=f[2], t0=f[3], t1=f[4], lat=f[5], lon=f[6], alt=f[7] * 1000, hit=f[8])
                             for f in flights],
                    killed={th.id: pol.st[th.id].get("killed_at") for th in w.threats if pol.st[th.id].get("killed_at")},
                    ledger=w.ledger(), attacker_cost=w.attacker_cost, launched=sum(th.t0 <= w.t for th in w.threats),
                    total=len(w.threats))


@app.get("/api/track/{tid}")
def track(tid: str, policy: str = "typical"):
    with lock:
        w = S["world"]
        if tid not in w.by_id:
            raise HTTPException(404)
        s = w.policies[policy].st[tid]
        th = w.by_id[tid]
        others = {n: (p.st[tid]["plan"].key if p.st[tid]["plan"] else None) for n, p in w.policies.items()}
        return dict(id=tid, name=THREATS[th.kind][0], status=s["status"], info=s["info"], escalated=s["escalated"],
                    plan=s["plan"].key if s["plan"] else None, others=others)


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
        elif c.action == "skip":   # fast-forward through transit to the next engagement opportunity
            w.t = max(w.t, w.idle_until())
        elif c.action == "reset":
            if c.scenario and c.scenario not in SCENARIOS:
                raise HTTPException(400, "unknown scenario")
            new_world(c.scenario or w.scenario, c.seed or w.seed, w.cfg)
        return {"ok": True}


@app.post("/api/config")
def config(patch: dict):
    allowed = set(DEFAULT_CONFIG)
    bad = set(patch) - allowed
    if bad:
        raise HTTPException(400, f"unknown keys {sorted(bad)}")
    if "orders" in patch:   # bounded so the state stays inside the trained length
        patch["orders"] = [str(o)[:160] for o in patch["orders"] if str(o).strip()][:5]
    with lock:
        S["world"].set_config(patch)
        return {"ok": True, "cfg": S["world"].cfg, "parsed": [parse_ok(o) for o in S["world"].cfg["orders"]]}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")
