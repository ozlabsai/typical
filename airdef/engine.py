"""World loop for time-indexed engagement decisions: three policies on one raid with common random numbers.

At every decision point a threat gets one of: fire E now, fire two E now, wait for E, let it fall. Decision points
recur (every RECHECK_S, on a miss, when a waited-for window opens, on a config change), so the call can change as
the threat flies. Policies: naive (most reliable shot now), planner (planner.py, the multi-lens teacher), typical
(the model reading the verbalised lenses; low confidence escalates to the operator, who takes the planner's call).
"""
import copy
import math
import random
import re
import time

import planner
from sim import (BATTERIES, EFFECTORS, ORIGINS, SCENARIOS, SITE, THREATS, DEFAULT_CONFIG, build_raid, consequence,
                 effective_pk, first_shot, money, precompute_windows, site_hit, site_value, u01)

HORIZON_S = 300        # a threat is decided once some interceptor can launch at it within this horizon
RECHECK_S = {"B": 8, "R": 6, "C": 20, "D": 30}   # re-decide at least this often (sim s), per threat class
PRICE_S = 10           # recompute raid shadow prices at most this often
ORIGIN_NAME = {"iran_w": "western Iran", "iran_nw": "north-western Iran", "iran_c": "central Iran", "iraq": "Iraq",
               "yemen": "Yemen", "lebanon_s": "southern Lebanon", "lebanon_bq": "the Beqaa valley", "gaza": "Gaza"}
CLASS_WORD = {"B": "ballistic missile", "R": "rocket", "C": "cruise missile", "D": "drone"}
Q = "What should the air defence do about this threat right now?"
DONE = ("killed", "leaked", "fell_open")


def short_name(kind):
    return re.sub(r"\s*\b(drone|MRBM|SRBM|rocket|cruise missile)\b", "", THREATS[kind][0]).strip()


def label(key):
    """Option key -> the candidate text Typical chooses among."""
    if key == "hold":
        return "Let it fall"
    act, e = key.split(":")
    n = EFFECTORS[e]["name"]
    return {"fire": f"Fire {n} now", "fire2": f"Fire two {n} now", "wait": f"Wait for {n}"}[act]


def in_words(s):
    if s < 20:
        return "seconds"
    if s < 5400:
        m = max(1, round(s / 60))
        return f"about {m} minute{'s' if m > 1 else ''}"
    return f"about {round(s / 3600)} hours"


def pk_word(p):
    return "almost certain to stop it" if p >= .95 else "very likely to stop it" if p >= .85 else \
        "likely to stop it" if p >= .7 else "may stop it" if p >= .5 else "unlikely to stop it"


def origin_key(th):
    return min(ORIGINS, key=lambda k: (ORIGINS[k][0] - th.origin[0]) ** 2 + (ORIGINS[k][1] - th.origin[1]) ** 2)


def verbalise(th, t, opts, prices, left, inbound, cfg):
    """The state Typical reads: the three lenses as sentences that pre-compute the comparisons (no arithmetic left)."""
    dmg, sid, p_hit, conf, dist, sigma = consequence(th, t, cfg)
    site = SITE[sid][1]
    prio = sid in cfg["priority_sites"]
    name = short_name(th.kind)
    lines = [f"Threat: {'an' if name[:1] in 'AEIOU' else 'a'} {name}, a {CLASS_WORD[th.cls]} from {ORIGIN_NAME[origin_key(th)]}."]
    if dist > sigma:
        lines.append(f"It will land in open ground near {site}.")
    else:
        heavy = ", a heavy warhead" if site_value(sid, th.kind, cfg) >= 400 else ""
        lines.append(f"It will hit {site}, a {'priority' if prio else 'secondary'} site{heavy}.")
    lines.append(f"It arrives in {in_words(th.t0 + th.T - t)}.")
    now = [o for o in opts if o.key.startswith("fire:")]
    later = sorted((o for o in opts if o.key.startswith("wait:")), key=lambda o: o.launch)
    if now:
        lines.append("Can fire now: " + "; ".join(
            f"{EFFECTORS[o.eff]['name']}, {pk_word(o.pk)}, {money(EFFECTORS[o.eff]['cost'])}" for o in now) + ".")
    if later:
        lines.append("Can fire later: " + "; ".join(
            f"{EFFECTORS[o.eff]['name']} in {in_words(o.launch - t)}{' (the last chance)' if o is later[-1] else ''}, "
            f"{pk_word(o.pk)}, {money(EFFECTORS[o.eff]['cost'])}" for o in later) + ".")
    used = sorted({o.eff for o in opts if o.eff and not EFFECTORS[o.eff].get("unlimited")}, key=lambda e: -EFFECTORS[e]["layer"])
    raid = [f"{EFFECTORS[e]['name']} is {'short' if prices.get(e, (0, 0))[0] > 0 else 'enough'} tonight ({left.get(e, 0):.0f} left)"
            for e in used]
    lines.append(f"Tonight: {inbound} more {CLASS_WORD[th.cls]}s in flight or expected" + (". " + "; ".join(raid) if raid else "") + ".")
    camp = [EFFECTORS[e]["name"] for e in used if prices.get(e, (0, 0))[1] > EFFECTORS[e]["cost"] * 2]
    if camp:
        lines.append(f"The campaign: {', '.join(camp)} must last more nights.")
    return "\n".join(lines)


class Policy:
    def __init__(self, name, threats, stock_scale):
        self.name = name
        self.stock = {bid: math.inf if EFFECTORS[eff].get("unlimited") else round(n * stock_scale.get(eff, 1))
                      for bid, eff, *_, n in BATTERIES}
        self.stock_init = dict(self.stock)
        self.st = {t.id: dict(status="pending", next=-1.0, cfgv=-1, shots=0, info=None, history=[], escalated=False)
                   for t in threats}
        self.prices, self.priced_at, self.priced_cfgv = {}, -1e9, -1
        self.L = dict(spent=0.0, rounds={}, kills=0, leaks=0, damage=0.0, fell_open=0, escalated=0,
                      decisions=0, agree=0, calls=0, ms=0.0)

    def type_left(self):
        left, init = {}, {}
        for bid, eff, *_ in BATTERIES:
            if self.stock[bid] != math.inf:
                left[eff] = left.get(eff, 0) + self.stock[bid]
                init[eff] = init.get(eff, 0) + self.stock_init[bid]
        return left, init


class World:
    def __init__(self, scenario="houthi_eilat", cfg=None, model=None, seed=7, policies=("naive", "planner", "typical"),
                 explore=0.0, record=None):
        self.scenario, self.seed = scenario, seed
        self.cfg = copy.deepcopy(cfg or DEFAULT_CONFIG)
        self.cfg_version = 0
        self.model = model
        self.memo = {}       # state text + options -> Typical's answer
        self.explore, self.record, self.rng = explore, record, random.Random(seed)
        self.threats = build_raid(scenario, seed)
        for th in self.threats:
            precompute_windows(th, BATTERIES)
        self.by_id = {t.id: t for t in self.threats}
        scale = SCENARIOS[scenario].get("stock_scale", {})
        self.policies = {n: Policy(n, self.threats, scale) for n in policies}
        self.t = 0.0
        self.end = max(t.t0 + t.T for t in self.threats) + 5
        self.attacker_cost = sum(THREATS[t.kind][2] for t in self.threats)
        self.flights = []   # (policy, battery, threat, launch, intercept, lat, lon, alt, hit)

    def live(self, pol, th):
        return pol.st[th.id]["status"] not in DONE and th.t0 <= self.t < th.t0 + th.T

    def expected(self, pol):
        """Threats in flight + those early warning expects to launch within cfg['forecast_s'] (the raid lens's demand)."""
        return [th for th in self.threats if pol.st[th.id]["status"] not in DONE and self.t < th.t0 + th.T
                and th.t0 <= self.t + self.cfg["forecast_s"]]

    def in_horizon(self, th):
        return self.t >= th.first_launch - HORIZON_S

    # ---------------------------------------------------------- deciding
    def refresh_prices(self, pol):
        if self.t - pol.priced_at < PRICE_S and pol.priced_cfgv == self.cfg_version:
            return
        left, init = pol.type_left()
        busy = {tid for tid, s in pol.st.items() if s["status"] == "inflight"}
        # every threat in flight, plus those early warning says launch within forecast_s, competes for tonight's stock
        pol.prices = planner.raid_prices(self.expected(pol), self.t, pol.stock, left, init, self.cfg, busy)
        pol.priced_at, pol.priced_cfgv = self.t, self.cfg_version

    def choose(self, pol, th, opts, text):
        """-> (option, info). naive / planner are rules; typical asks the model and escalates when unsure."""
        top = planner.best(opts)
        info = dict(planner=top.key, state=text, opts=[(o.key, label(o.key), o.total, o.lenses, o.pk) for o in opts])
        if pol.name == "naive":
            now = [o for o in opts if o.key.startswith(("fire:", "fire2:"))]
            pool = now or [o for o in opts if o.key.startswith("wait:")] or opts
            return max(pool, key=lambda o: (round(o.pk, 2), -(o.launch or 0))), info
        if pol.name == "planner" or self.model is None:
            if self.explore and self.rng.random() < self.explore:
                return self.rng.choice(opts), info
            return top, info
        labels = [label(o.key) for o in opts]
        key = text + "\n" + "|".join(labels)
        if key not in self.memo:
            t0 = time.perf_counter()
            d = self.model.choice(text, Q, labels)
            self.memo[key] = (d, (time.perf_counter() - t0) * 1000)
            pol.L["calls"] += 1
            pol.L["ms"] += self.memo[key][1]
        d, ms = self.memo[key]
        dist = {o.key: d[label(o.key)] for o in opts}
        pick = max(opts, key=lambda o: dist[o.key])
        info.update(dist=dist, p_null=d["p_null"], ms=ms)
        pol.L["agree"] += pick.key == top.key
        if dist[pick.key] < self.cfg["conf_threshold"] or d["p_null"] > self.cfg["abstain_threshold"]:
            info["escalated"] = True   # operator takes the planner's call
            return top, info
        return pick, info

    def decide_all(self, pol, budget):
        self.refresh_prices(pol)
        left, _ = pol.type_left()
        due = [th for th in self.threats if self.live(pol, th) and pol.st[th.id]["status"] != "inflight" and self.in_horizon(th)
               and (self.t >= pol.st[th.id]["next"] or pol.st[th.id]["cfgv"] != self.cfg_version)]
        due.sort(key=lambda th: th.t0 + th.T)
        exp = self.expected(pol)
        inbound = {c: sum(1 for x in exp if x.cls == c) for c in CLASS_WORD}
        calls = 0
        for th in due:
            s = pol.st[th.id]
            dmg = consequence(th, self.t, self.cfg)[0]
            opts = planner.option_values(th, self.t, pol.stock, self.cfg, pol.prices, dmg)
            text = verbalise(th, self.t, opts, pol.prices, left, inbound[th.cls] - 1, self.cfg)
            if pol.name == "typical" and self.model is not None and text + "\n" + "|".join(label(o.key) for o in opts) not in self.memo:
                if calls >= budget:
                    break
                calls += 1
            opt, info = self.choose(pol, th, opts, text)
            if self.record is not None and pol.name == "planner":
                self.record.append(dict(state=text, opts=[(o.key, o.total) for o in opts], kind=th.kind,
                                        scenario=self.scenario))
            if info.get("escalated") and not s["escalated"]:
                pol.L["escalated"] += 1
                s["escalated"] = True
            pol.L["decisions"] += 1
            if not s["history"] or s["history"][-1][1] != opt.key:
                s["history"].append((self.t, opt.key))
            s.update(info=info, cfgv=self.cfg_version, choice=opt.key)
            self.apply(pol, th, opt)
            left, _ = pol.type_left()
        return calls

    def apply(self, pol, th, opt):
        s = pol.st[th.id]
        recheck = RECHECK_S[th.cls]
        if opt.key == "hold":
            s.update(status="held", next=self.t + recheck)
            return
        act, e = opt.key.split(":")
        if act == "wait":
            s.update(status="waiting", next=max(self.t + 1, min(opt.launch - planner.OPEN_S / 2, self.t + recheck)))
            return
        bid = planner.windows(th, self.t, pol.stock, self.cfg)[e][0]
        shot = first_shot(th, bid, self.t + 4.0)
        hit = False
        for _ in range(2 if act == "fire2" else 1):
            if pol.stock[bid] < 1:
                break
            pol.stock[bid] -= 1
            pol.L["spent"] += EFFECTORS[e]["cost"]
            pol.L["rounds"][e] = pol.L["rounds"].get(e, 0) + 1
            h = u01(self.seed, th.id, s["shots"]) < effective_pk(e, th, self.cfg)   # same draw for every policy's n-th shot
            s["shots"] += 1
            hit = hit or h
            self.flights.append((pol.name, bid, th.id, shot[0], shot[1], shot[2], shot[3], shot[4], h))
        s.update(status="inflight", resolve=shot[1], hit=hit)

    def resolve(self, pol):
        for th in self.threats:
            s = pol.st[th.id]
            if s["status"] == "inflight" and self.t >= s["resolve"]:
                if s["hit"]:
                    s.update(status="killed", killed_at=s["resolve"])
                    pol.L["kills"] += 1
                else:
                    s.update(status="pending", next=self.t)   # missed: decide again now
            if s["status"] not in DONE and self.t >= th.t0 + th.T:
                sid = site_hit(*th.impact)
                s["site"] = sid
                if sid:
                    pol.L["leaks"] += 1
                    pol.L["damage"] += site_value(sid, th.kind, self.cfg)
                    s["status"] = "leaked"
                else:
                    pol.L["fell_open"] += 1
                    s["status"] = "fell_open"

    def step(self, dt, budget=10_000):
        calls = {n: self.decide_all(p, budget) for n, p in self.policies.items()}
        for p in self.policies.values():
            self.resolve(p)
        self.t = min(self.t + dt, self.end)
        return calls

    def idle_until(self):
        """Next sim time worth simulating (fast-forward through hours of drone transit)."""
        if any(s["status"] in ("inflight", "waiting", "held") for p in self.policies.values() for s in p.st.values()):
            return self.t
        nxt = [max(th.t0, th.first_launch - HORIZON_S) for th in self.threats if max(th.t0, th.first_launch - HORIZON_S) > self.t]
        return min(nxt + [self.end])

    def set_config(self, patch):
        self.cfg.update(patch)
        self.cfg_version += 1

    def ledger(self):
        out = {}
        for n, p in self.policies.items():
            left, init = p.type_left()
            out[n] = {**p.L, "rounds": dict(p.L["rounds"]), "left": left, "init": init, "total": p.L["spent"] + p.L["damage"]}
        return out


def run_headless(scenario, model=None, cfg=None, dt=2.0, seed=7, **kw):
    w = World(scenario, cfg, model, seed, **kw)
    while w.t < w.end:
        w.step(dt)
        w.t = max(w.t, w.idle_until())
    return w


if __name__ == "__main__":   # ponytail: the self-check -- the planner must beat fire-at-everything on total cost over seeds
    for sc in SCENARIOS:
        tot, hits, spent = {"naive": 0.0, "planner": 0.0}, {"naive": 0, "planner": 0}, {"naive": 0.0, "planner": 0.0}
        t0 = time.time()
        for seed in (1, 2, 3):
            w = run_headless(sc, seed=seed, policies=("naive", "planner"))
            assert all(v >= 0 for p in w.policies.values() for v in p.stock.values()), sc
            for n, x in w.ledger().items():
                tot[n] += x["total"]
                hits[n] += x["leaks"]
                spent[n] += x["spent"]
        print(f"{sc:20s} total  naive {tot['naive']:8.0f}  planner {tot['planner']:8.0f} | spent {money(spent['naive'])} vs "
              f"{money(spent['planner'])} | hits {hits['naive']} vs {hits['planner']}  ({time.time() - t0:.0f}s)")
        assert tot["planner"] <= tot["naive"] * 1.05, sc
