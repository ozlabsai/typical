"""Curated decision situations with a simulated optimum -- the benchmark the planner (and Typical) is judged on.

    uv run --no-sync python airdef/situations.py [--instances 10] [--rollouts 60] [--out .context/airdef/situations.jsonl]

Each family is a small hand-built world (1-3 threats, chosen stock / channels / aircraft) frozen at one decision moment.
Every option at that moment is forced in turn and the world is simulated to the end `rollouts` times with different
outcome draws (the planner plays every later decision). The option with the lowest mean total cost is the optimum.
Reported: how often the planner and doctrine pick it, and their regret. Rows are written in the training schema so the
same file is a Typical eval set ("situation drill").
"""
import argparse
import copy
import json
import math
import random
import statistics as st
import sys
from multiprocessing import Pool
from pathlib import Path

import engine
import planner
from sim import AIR_KINDS, BATTERIES, ORIGINS, SITE, SITES, THREATS, Threat, _flight, haversine, precompute_windows

ROOT = Path(__file__).resolve().parent.parent


def make_threat(tid, kind, origin, site, t0, sigma0=None, miss_km=0.0, rng=None):
    """A threat from an origin region to (near) a site; miss_km pushes the true impact off the site (open ground)."""
    o = ORIGINS[origin][:2]
    s = SITE[site]
    ang = (rng or random).random() * 2 * math.pi
    impact = (s[2] + miss_km / 111 * math.sin(ang), s[3] + miss_km / 111 * math.cos(ang) / 0.85)
    rng_km = float(haversine(*o, *impact))
    T, apo = _flight(kind, rng_km)
    a = (rng or random).random() * 2 * math.pi
    th = Threat(id=tid, kind=kind, origin=o, impact=impact, t0=t0, T=T, apogee=apo, ipp_offset=(math.sin(a), math.cos(a)),
                ipp_sigma0=sigma0 if sigma0 is not None else min(25.0, 3 + rng_km * 0.02))
    precompute_windows(th, BATTERIES)
    return th


def world_with(threats, cfg=None, seed=1, stock=None, units_off=(), busy=None):
    w = engine.World("houthi_eilat", cfg, None, seed, policies=("doctrine", "planner"))
    w.threats, w.by_id = threats, {t.id: t for t in threats}
    w.end = max(t.t0 + t.T for t in threats) + 5
    for p in w.policies.values():
        p.st = {t.id: dict(status="pending", next=-1.0, cfgv=-1, shots=0, info=None, local=None, history=[], escalated=False,
                           choice=None, assign=None) for t in threats}
        for bid, n in (stock or {}).items():   # stock_init stays at the full load: a low number reads as scarce
            p.stock[bid], p.ready[bid] = n, min(n, p.capacity(bid))
        for uid in units_off:
            p.units[uid].weapons = {k: 0 for k in p.units[uid].weapons}
        for bid, times in (busy or {}).items():
            p.busy[bid] = list(times)
    return w


# ------------------------------------------------------------------------------- families
def fam_wait_for_cheaper(rng):
    """Arrow 3 can fire now; a cheaper layer reaches it later; Arrow 3 is scarce for the campaign."""
    site = rng.choice(["tel_aviv", "ben_gurion", "jerusalem", "ashdod"])
    th = make_threat("F1", rng.choice(["emad", "ghadr"]), rng.choice(["iran_w", "iran_nw"]), site, 0.0, sigma0=1.0, rng=rng)
    cfg = {**engine.DEFAULT_CONFIG, "reserve": 2}
    return world_with([th], cfg, stock={"A3-C": 1, "A3-S": 0, "DDG": 0}), th, th.t0 + 0.30 * th.T


def fam_last_chance(rng):
    """Only Iron Dome still reaches it, seconds before impact."""
    site = rng.choice(["haifa", "nahariya", "kiryat_shmona", "tiberias"])
    th = make_threat("F1", rng.choice(["grad", "fajr5"]), "lebanon_s", site, 0.0, sigma0=0.5, rng=rng)
    return world_with([th]), th, th.t0 + 0.70 * th.T


def fam_let_it_fall(rng):
    """The impact point is well clear of any defended site."""
    site = rng.choice(["kiryat_shmona", "nahariya", "tiberias", "haifa"])
    th = make_threat("F1", "grad", "lebanon_s", site, 0.0, sigma0=0.5, miss_km=rng.uniform(15, 25), rng=rng)
    return world_with([th]), th, th.t0 + 0.45 * th.T


def fam_track_not_firm(rng):
    """Early in a ballistic flight, the impact prediction straddles a site edge; later layers will still reach it."""
    site = rng.choice(["tel_aviv", "beer_sheva", "haifa"])
    th = make_threat("F1", "emad", "iran_w", site, 0.0, sigma0=25.0, miss_km=rng.uniform(4, 9), rng=rng)
    return world_with([th]), th, th.t0 + 0.25 * th.T


def fam_coverage_gap(rng):
    """A drone heads for Eilat; the Eilat battery is empty; a fighter CAP over the Red Sea can meet it."""
    th = make_threat("F1", "shahed", "yemen", "eilat", 0.0, sigma0=1.0, rng=rng)
    return world_with([th], stock={"ID-E": 0}, units_off=("F-N", "F-S", "F-E")), th, th.t0 + 0.75 * th.T


def fam_helicopter(rng):
    """A long-range drone heads for the north; the northern batteries are empty, the laser is clouded out and the
    fighters are committed elsewhere; Apaches are on alert."""
    site = rng.choice(["ramat_david", "haifa_bay", "tiberias"])
    th = make_threat("F1", "shahed", "iraq", site, 0.0, sigma0=1.0, rng=rng)
    cfg = {**engine.DEFAULT_CONFIG, "iron_beam_weather": "cloudy"}
    return (world_with([th], cfg, stock={"ID-N": 0, "ID-H": 0, "DS-N": 0, "DS-C": 0}, units_off=("F-N", "F-S", "F-E", "F-R")),
            th, th.t0 + th.T - 1800)


FAMILIES = {"wait_for_cheaper": fam_wait_for_cheaper, "last_chance": fam_last_chance, "let_it_fall": fam_let_it_fall,
            "track_not_firm": fam_track_not_firm, "coverage_gap": fam_coverage_gap, "helicopter": fam_helicopter}


def case(fam, inst):
    """-> (world, threat, decision time) for instance `inst` of a family; the same case every run (the live drill reloads it)."""
    return FAMILIES[fam](random.Random(f"{fam}:{inst}"))


# ------------------------------------------------------------------------------- evaluation
def options_at(w, th, t, pol_name="planner"):
    w.t = t
    pol = w.policies[pol_name]
    w.refresh_prices(pol)
    dmg, sid, p_hit, *_ = engine.consequence(th, t, w.cfg)
    opts = planner.option_values(th, t, pol.stock, w.cfg, pol.prices, dmg, pol.units.values(),
                                 engine.site_value(sid, th.kind, w.cfg), p_hit, pol.free_at(t), {})
    left, _ = pol.type_left()
    text = engine.global_text(th, t, opts, pol.prices, left, 0, w.cfg, pol.units)
    return opts, text, p_hit


def rollout(w, th, t, key, seed):
    """Force option `key` for th at t (planner policy), then let the planner play to the end; -> total cost."""
    w = copy.deepcopy(w)
    w.seed = seed
    del w.policies["doctrine"]
    pol = w.policies["planner"]
    left0, init0 = pol.type_left()   # the campaign lens: every round used is worth its campaign price at decision time
    camp = {e: planner.campaign_price(e, left0, init0, w.cfg) for e in left0}
    opts, _, _ = options_at(w, th, t)
    opt = next(o for o in opts if o.key == key)
    w.apply_global(pol, th, opt)
    pol.st[th.id].update(choice=key, next=t + engine.RECHECK_S[th.cls],
                         commit="hold" if key == "hold" else f"cheaper:{opt.eff}" if key == "cheaper" else None)
    if key.startswith("now:"):   # fire now means now: the unit engages this tick (salvo size from the local teacher)
        w.decide_local(pol, 10_000)
    while w.t < w.end:
        w.step(2.0)
        if (nxt := w.idle_until()) > w.t:
            w.advance(nxt - w.t)
    L = w.ledger()["planner"]
    return L["total"] + sum(n * camp.get(e, 0.0) for e, n in L["rounds"].items())


def one(job):
    fam, inst, n_roll = job
    w, th, t = case(fam, inst)
    opts, text, p_hit = options_at(w, th, t)
    if len(opts) < 2:
        return None
    costs = {o.key: st.mean(rollout(w, th, t, o.key, 10_000 + r) for r in range(n_roll)) for o in opts}
    best = min(costs, key=costs.get)
    doc = w.doctrine_global(opts, p_hit).key
    pl = planner.best(opts).key
    labels = [engine.label(o.key) for o in opts]
    cmin = costs[best]
    tau = 0.15 * max(cmin, 0) + 0.02
    wts = [math.exp(-(costs[o.key] - cmin) / tau) for o in opts]
    z = sum(wts)
    row = {"state": text, "query": engine.Q_GLOBAL, "candidates": labels, "target": [x / z for x in wts], "p_null": 0.0,
           "task": "airdef_situation", "label": [o.key for o in opts].index(best),
           "meta": {"qtype": "choice", "family": fam, "inst": inst, "best": best, "costs": costs, "planner": pl, "doctrine": doc}}
    return fam, best, pl, doc, costs, row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--rollouts", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / ".context/airdef/situations.jsonl"))
    a = ap.parse_args()
    jobs = [(f, i, a.rollouts) for f in FAMILIES for i in range(a.instances)]
    with Pool(10) as p:
        res = [r for r in p.map(one, jobs) if r]
    Path(a.out).write_text("".join(json.dumps(r[5]) + "\n" for r in res))
    print(f"{'family':18s} {'n':>3s}  {'optimum (most common)':24s} planner-hit  doctrine-hit  planner-regret  doctrine-regret")
    for fam in FAMILIES:
        r = [x for x in res if x[0] == fam]
        if not r:
            continue
        common = max(set(x[1] for x in r), key=[x[1] for x in r].count)
        reg = lambda who: st.mean(x[4][x[who]] - x[4][x[1]] for x in r)
        print(f"{fam:18s} {len(r):3d}  {common:24s} {sum(x[2] == x[1] for x in r) / len(r):10.0%}  {sum(x[3] == x[1] for x in r) / len(r):11.0%}"
              f"  {reg(2):14.2f}  {reg(3):15.2f}")
    print(f"rows -> {a.out}")
