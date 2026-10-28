"""World loop, two decision levels (DECISIONS.md): GLOBAL assigns each threat an interceptor (or waits / lets it
fall); LOCAL is the assigned unit's fire one / fire two / hold, asked every tick it can engage.

Policies on one raid with common random numbers: doctrine (layered, cheapest effective first, fires when it can),
planner (planner.py at both levels -- the teacher), typical (the model at both levels; low confidence escalates to the
operator, who takes the planner's call).
"""
import copy
import math
import random
import re
import time

import air
import planner
from sim import (AIR, AIR_KINDS, BATTERIES, EFFECTORS, ORIGINS, SCENARIOS, SITE, THREATS, TURN_S, DEFAULT_CONFIG, build_raid,
                 consequence, effective_pk, first_shot, haversine, money, precompute_windows, radar_turn, site_hit,
                 site_value, u01)

HORIZON_S = 300        # a threat is decided once some interceptor can launch at it within this horizon
RECHECK_S = {"B": 8, "R": 6, "C": 20, "D": 30}
PRICE_S = 10
DT = 4.0               # live demo: the world always moves in steps of DT sim seconds, so pacing never changes the trajectory
QUIET_S = 15           # auto-skip: jump ahead when nothing happens for longer than this (sim seconds)
BUDGET_S = 1.0   # the model's latency budget in wall seconds (live demo): past it the planner's call stands in, flagged
ORIGIN_NAME = {"iran_w": "western Iran", "iran_nw": "north-western Iran", "iran_c": "central Iran", "iraq": "Iraq",
               "yemen": "Yemen", "lebanon_s": "southern Lebanon", "lebanon_bq": "the Beqaa valley", "gaza": "Gaza"}
CLASS_WORD = {"B": "ballistic missile", "R": "rocket", "C": "cruise missile", "D": "drone"}
Q_GLOBAL = "Which interceptor should take this threat?"
Q_LOCAL = "How many interceptors should this unit fire at the threat now?"
LEVELS = ["0", "1", "2", "3"]   # a Score: 0 = hold
DONE = ("killed", "leaked", "fell_open")
SHORT = {e: d["name"].replace(" laser", "").replace("GPS jamming (EW)", "EW jamming") for e, d in EFFECTORS.items()}
BAT = {b[0]: b for b in BATTERIES}


def short_name(kind):
    return re.sub(r"\s*\b(drone|MRBM|SRBM|rocket|cruise missile)\b", "", THREATS[kind][0]).strip()


def label(key):
    """Global option key -> the operative candidate Typical chooses among."""
    if key in ("hold", "wait", "cheaper"):
        return {"hold": "Let it fall", "wait": "Wait", "cheaper": "Wait for cheaper"}[key]
    kind, x = key.split(":")
    return SHORT[x] if kind == "now" else AIR_KINDS[x]["label"]


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


def threat_line(th):
    name = short_name(th.kind)
    return f"Threat: {'an' if name[:1] in 'AEIOU' else 'a'} {name}, a {CLASS_WORD[th.cls]} from {ORIGIN_NAME[origin_key(th)]}."


def target_line(th, t, cfg):
    dmg, sid, p_hit, conf, dist, sigma = consequence(th, t, cfg)
    site, prio = SITE[sid][1], sid in cfg["priority_sites"]
    if p_hit < planner.FIRM:
        return f"It will land in open ground near {site}."
    heavy = ", a heavy warhead" if site_value(sid, th.kind, cfg) >= 400 else ""
    if planner.firm(p_hit):
        return f"It will hit {site}, a {'priority' if prio else 'secondary'} site{heavy}."
    return f"It may hit {site}, a {'priority' if prio else 'secondary'} site{heavy}; the track is not firm yet."


def global_text(th, t, opts, prices, left, inbound, cfg, units):
    """What the battle manager reads: the threat, every operative option, the raid and campaign lenses."""
    lines = [threat_line(th), target_line(th, t, cfg), f"It arrives in {in_words(th.t0 + th.T - t)}."]
    now = [o for o in opts if o.key.startswith("now:")]
    later = sorted((o for o in opts if o.key == "cheaper"), key=lambda o: o.launch)
    airo = [o for o in opts if o.key.startswith("air:")]
    if now:
        lines.append("Can fire now: " + "; ".join(f"{SHORT[o.eff]}, {pk_word(o.pk)}, {money(EFFECTORS[o.eff]['cost'])}" for o in now) + ".")
    if later:
        o = later[0]
        lines.append(f"Cheaper later: {SHORT[o.eff]} in {in_words(o.launch - t)}, {pk_word(o.pk)}, {money(EFFECTORS[o.eff]['cost'])}.")
    for o in airo:
        u = units[o.unit]
        lines.append(f"Aircraft: {AIR_KINDS[u.kind]['label'].lower()} ({u.name}) can intercept it in {in_words(o.launch - t)}, "
                     f"{pk_word(o.pk)}, about {money(o.total - o.lenses.get('raid', 0) - o.lenses.get('campaign', 0) - (1 - o.pk) * 0)} with flight time.")
    if not now and not later:
        lines.append("No ground interceptor can reach it.")
    elif not now and later and later[0].launch - t > 600:
        lines.append(f"No ground interceptor can reach it for {in_words(later[0].launch - t)}.")
    # name breaks layer ties: a set's string order changes with every process (hash randomisation), and the text must not
    used = sorted({o.eff for o in now + later if not EFFECTORS[o.eff].get("unlimited")}, key=lambda e: (-EFFECTORS[e]["layer"], e))
    raid = [f"{SHORT[e]} is {'short' if prices.get(e, (0, 0))[0] > 0 else 'enough'} tonight ({left.get(e, 0):.0f} left)" for e in used]
    lines.append(f"Tonight: {inbound} more {CLASS_WORD[th.cls]}s in flight or expected" + (". " + "; ".join(raid) if raid else "") + ".")
    camp = [SHORT[e] for e in used if prices.get(e, (0, 0))[1] > EFFECTORS[e]["cost"] * 2]
    if camp:
        lines.append(f"The campaign: {', '.join(camp)} must last more nights.")
    return "\n".join(lines)


def local_text(th, t, cfg, unit_line, f):
    """What the fire unit reads before it fires."""
    _, sid, p_hit, *_ = consequence(th, t, cfg)
    lines = [unit_line, threat_line(th), target_line(th, t, cfg), f"Impact in {in_words(th.t0 + th.T - t)}."]
    if f["channels"] > 1:
        lines.append(f"Launch channels: {f['channels_free']} of {f['channels']} free." if f["channels_free"] else "All launch channels are busy.")
    if f["stock"] == math.inf:
        lines.append("Shots are effectively unlimited.")
    elif f["ready"] < 1:
        lines.append(f"No rounds on the launchers; the next launcher is reloaded in {in_words(f['reload_eta'] or 0)}." if f["reload_eta"]
                     else "No rounds left.")
    else:
        lines.append(f"Ready on the launchers: {f['ready']} of {f['stock']} left"
                     + (f"; a launcher is reloading, ready in {in_words(f['reload_eta'])}." if f["reload_eta"] else "."))
    if f["turn"]:
        lines.append("The radar is staring away from it and has to turn first (about 10 seconds).")
    if f["friendly"]:
        lines.append("Own aircraft are near the intercept point.")
    if f["debris"]:
        lines.append(f"The intercept point is over {SITE[f['debris']][1]}: debris would fall there."
                     + (" A later intercept would be over open ground." if f["later_open"] else ""))
    else:
        lines.append("The intercept point is over open ground or high enough for debris to burn up.")
    lines.append("This is the last chance to engage it." if f["last"] else
                 "There is time to look before a second shot." if f["look"] else "There is no time for a second look.")
    return "\n".join(lines)


class Policy:
    def __init__(self, name, threats, stock_scale):
        self.name = name
        self.stock = {bid: math.inf if EFFECTORS[eff].get("unlimited") else round(n * stock_scale.get(eff, 1))
                      for bid, eff, *_, n in BATTERIES}
        self.stock_init = dict(self.stock)
        self.busy = {bid: [] for bid in self.stock}           # release times of channels in use
        self.ready = {bid: min(self.stock[bid], self.capacity(bid)) for bid in self.stock}   # rounds on launchers
        self.reload_until = {bid: None for bid in self.stock}
        self.units = {u.id: u for u in air.fleet()}
        self.st = {t.id: dict(status="pending", next=-1.0, cfgv=-1, shots=0, info=None, local=None, history=[],
                              escalated=False, choice=None, assign=None) for t in threats}
        self.prices, self.priced_at, self.priced_cfgv = {}, -1e9, -1
        self.L = dict(spent=0.0, rounds={}, kills=0, leaks=0, damage=0.0, fell_open=0, escalated=0, decisions=0,
                      agree=0, calls=0, ms=0.0, local_decisions=0, local_agree=0, sorties=0, late=0)

    def type_left(self):
        left, init = {}, {}
        for bid, eff, *_ in BATTERIES:
            if self.stock[bid] != math.inf:
                left[eff] = left.get(eff, 0) + self.stock[bid]
                init[eff] = init.get(eff, 0) + self.stock_init[bid]
        return left, init

    @staticmethod
    def capacity(bid):
        e = EFFECTORS[BAT[bid][1]]
        return e["launchers"] * e["per_launcher"] if e.get("launchers") else math.inf

    def reload(self, t):
        """One launcher at a time: when a launcher's worth is spent and reserve remains, reload it (notional times)."""
        for bid, ready in self.ready.items():
            e = EFFECTORS[BAT[bid][1]]
            if not e.get("launchers"):
                continue
            if self.reload_until[bid] is not None and t >= self.reload_until[bid]:
                self.ready[bid] = min(ready + e["per_launcher"], self.stock[bid], self.capacity(bid))
                self.reload_until[bid] = None
            if self.reload_until[bid] is None and self.ready[bid] <= self.capacity(bid) - e["per_launcher"] and self.stock[bid] > self.ready[bid]:
                self.reload_until[bid] = t + e["reload"]

    def ready_n(self, bid):
        return self.ready[bid] if self.capacity(bid) != math.inf else self.stock[bid]

    def channels_free(self, bid, t):
        self.busy[bid] = [r for r in self.busy[bid] if r > t]
        return EFFECTORS[BAT[bid][1]].get("channels", 2) - len(self.busy[bid])

    def free_at(self, t):
        """battery -> when it can next launch: a free channel and a loaded launcher (t if both now)."""
        out = {}
        for bid in self.busy:
            ch = t if self.channels_free(bid, t) > 0 else min(self.busy[bid])
            ld = t if self.ready_n(bid) > 0 else (self.reload_until[bid] or math.inf)
            out[bid] = max(ch, ld)
        return out


class World:
    def __init__(self, scenario="houthi_eilat", cfg=None, model=None, seed=7, policies=("doctrine", "planner", "typical"),
                 explore=0.0, record=None, async_model=False):
        self.scenario, self.seed = scenario, seed
        self.cfg = copy.deepcopy(cfg or DEFAULT_CONFIG)
        self.cfg_version = 0
        self.model = model
        self.memo = {}       # (question, state text, labels) -> Typical's answer
        # async_model (live demo): the clock never waits for Typical; a worker drains `requests`, most urgent first.
        self.async_model, self.requests, self.focus = async_model, {}, None
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
        self.flights = []   # (policy, shooter, threat, launch, intercept, lat, lon, alt, hit, shooter lat, shooter lon, weapon)

    def live(self, pol, th):
        return pol.st[th.id]["status"] not in DONE and th.t0 <= self.t < th.t0 + th.T

    def expected(self, pol):
        return [th for th in self.threats if pol.st[th.id]["status"] not in DONE and self.t < th.t0 + th.T
                and th.t0 <= self.t + self.cfg["forecast_s"]]

    def in_horizon(self, th):
        return self.t >= th.first_launch - HORIZON_S or th.cls in "CD"   # aircraft can meet slow threats early

    def refresh_prices(self, pol):
        if self.t - pol.priced_at < PRICE_S and pol.priced_cfgv == self.cfg_version:
            return
        left, init = pol.type_left()
        busy = {tid for tid, s in pol.st.items() if s["status"] in ("inflight", "assigned")}
        pol.prices = planner.raid_prices(self.expected(pol), self.t, pol.stock, left, init, self.cfg, busy)
        pol.priced_at, pol.priced_cfgv = self.t, self.cfg_version

    # ---------------------------------------------------------- the model, shared by both levels
    def ask(self, pol, q, text, labels, due=0.0, tid=None):
        """-> (dist over labels, p_null, ms) or None while an async answer is pending; the threat on screen
        (`focus`) goes first, then `due` orders the queue."""
        key = (q, text, tuple(labels))
        if key not in self.memo:
            if self.async_model:
                self.requests.setdefault(key, (q, text, list(labels), (tid != self.focus, due)))
                return None
            t0 = time.perf_counter()
            d = self.model.choice(text, q, list(labels))
            if d is None:   # a recorded-answers model that never saw this situation: the caller's stand-in takes it
                return None
            self.memo[key] = (d, (time.perf_counter() - t0) * 1000)
            pol.L["calls"] += 1
            pol.L["ms"] += self.memo[key][1]
        d, ms = self.memo[key]
        return {lab: d[lab] for lab in labels}, d["p_null"], ms

    def late(self, pol, s, level, urgent=False):
        """The clock never waits for the model: past the latency budget, or at the last chance, the planner's call stands in."""
        k = "asked_" + level
        s.setdefault(k, time.perf_counter())
        if urgent or time.perf_counter() - s[k] > BUDGET_S:
            del s[k]
            pol.L["late"] += 1
            return True
        return False

    def unsure(self, top_p, p_null):
        return top_p < self.cfg["conf_threshold"] or p_null > self.cfg["abstain_threshold"]

    # ---------------------------------------------------------- GLOBAL
    def doctrine_global(self, opts, p_hit):
        """Layered doctrine: let open-ground impacts fall, wait out an unclear track, else the cheapest effective
        interceptor -- the cheaper later layer if one is coming, aircraft when nothing on the ground reaches."""
        by = {o.key: o for o in opts}
        if p_hit < planner.FIRM:
            return by["hold"]
        if "wait" in by:
            return by["wait"]
        good = [o for o in opts if o.key.startswith("now:") and o.pk >= 0.7]
        if "cheaper" in by and by["cheaper"].pk >= 0.7 and (not good or EFFECTORS[by["cheaper"].eff]["cost"] < min(EFFECTORS[o.eff]["cost"] for o in good)):
            return by["cheaper"]
        if good:
            return min(good, key=lambda o: EFFECTORS[o.eff]["cost"])
        airo = [o for o in opts if o.key.startswith("air:")]
        if airo:
            return min(airo, key=lambda o: o.total)
        now = [o for o in opts if o.key.startswith("now:")]
        return max(now, key=lambda o: o.pk) if now else by.get("cheaper", by["hold"])

    def decide_global(self, pol, budget):
        self.refresh_prices(pol)
        left, _ = pol.type_left()
        # assigned-but-unfired threats are revisable too: the track firms up, stock moves, an upgrade may pay off
        due = [th for th in self.threats if self.live(pol, th) and pol.st[th.id]["status"] in ("pending", "waiting", "held", "assigned")
               and self.in_horizon(th) and (self.t >= pol.st[th.id]["next"] or pol.st[th.id]["cfgv"] != self.cfg_version)]
        due.sort(key=lambda th: th.t0 + th.T)
        exp = self.expected(pol)
        inbound = {c: sum(1 for x in exp if x.cls == c) for c in CLASS_WORD}
        free_at = pol.free_at(self.t)
        bookings = {}   # layer -> planned launch times already committed by other threats (the congestion lens)
        for x in pol.st.values():
            if x["status"] == "waiting" and x.get("wait_eff"):
                bookings.setdefault(x["wait_eff"], []).append(x.get("wait_until", self.t))
            elif x["status"] == "assigned" and x["assign"] and x["assign"]["kind"] == "ground":
                bookings.setdefault(x["assign"]["eff"], []).append(x.get("eng_at", self.t))
        calls = 0
        for th in due:
            s = pol.st[th.id]
            dmg, sid, p_hit, *_ = consequence(th, self.t, self.cfg)
            opts = planner.option_values(th, self.t, pol.stock, self.cfg, pol.prices, dmg, pol.units.values(),
                                         site_value(sid, th.kind, self.cfg), p_hit, free_at, bookings)
            if s.get("ew_done"):   # jamming works or it doesn't: one attempt per threat
                opts = [o for o in opts if o.eff != "ew"]
            c = s.get("commit")   # a forced commitment (situations.py rollouts): the option keeps its meaning to the end
            if c == "hold":
                s["next"] = math.inf
                continue
            if c and c.startswith("cheaper:"):
                opts = [o for o in opts if o.key == "hold" or o.key == "now:" + c.split(":")[1] or (o.key == "cheaper" and o.eff == c.split(":")[1])]
            if len(opts) == 1:   # nothing can reach it (yet)
                s.update(status="pending", next=self.t + RECHECK_S[th.cls])
                continue
            text = global_text(th, self.t, opts, pol.prices, left, inbound[th.cls] - 1, self.cfg, pol.units)
            top = planner.best(opts, s.get("choice"))
            info = dict(planner=top.key, state=text, t=self.t,
                        opts=[(o.key, label(o.key), o.total, o.lenses, o.pk, o.launch, o.unit) for o in opts])
            if pol.name == "doctrine":
                opt = self.doctrine_global(opts, p_hit)
            elif pol.name == "planner" or self.model is None:
                opt = self.rng.choice(opts) if self.explore and self.rng.random() < self.explore else top
            else:
                if self.async_model is False and (q := (Q_GLOBAL, text, tuple(label(o.key) for o in opts))) not in self.memo:
                    if calls >= budget:
                        break
                    calls += 1
                ans = self.ask(pol, Q_GLOBAL, text, [label(o.key) for o in opts], th.t0 + th.T, th.id)
                if ans is None:
                    if not self.late(pol, s, "g", urgent=not self.async_model):
                        continue
                    opt = top
                    info["late"] = True
                else:
                    s.pop("asked_g", None)
                    d, p_null, ms = ans
                    dist = {o.key: d[label(o.key)] for o in opts}
                    opt = max(opts, key=lambda o: dist[o.key])
                    info.update(dist=dist, p_null=p_null, ms=ms)
                    pol.L["agree"] += opt.key == top.key
                if ans and self.unsure(dist[opt.key], p_null):
                    info["escalated"] = True
                    opt = top
                    if not s["escalated"]:
                        pol.L["escalated"] += 1
                        s["escalated"] = True
            if self.record is not None and pol.name == "planner":
                self.record.append(dict(level="global", state=text, q=Q_GLOBAL, opts=[(label(o.key), o.total) for o in opts],
                                        best=top.key, kind=th.kind, scenario=self.scenario))
            pol.L["decisions"] += 1
            if not s["history"] or s["history"][-1][1] != opt.key:
                s["history"].append((self.t, opt.key))
            s.update(info=info, cfgv=self.cfg_version, choice=opt.key)
            log, p = s.setdefault("log", []), (info.get("dist") or {}).get(opt.key)
            last = next((x for x in reversed(log) if "key" in x), None)   # the progression: a row when the call or its probability moves
            if not last or last["key"] != opt.key or (p is not None and last["p"] is not None and abs(p - last["p"]) >= 0.05):
                log.append(dict(t=self.t, key=opt.key, p=p, opts=info["opts"] if not last or last["key"] != opt.key else None))
                del log[:-60]
            self.apply_global(pol, th, opt)
            if opt.key == "cheaper" or opt.key.startswith("now:"):   # later threats in this same tick must see this booking
                bookings.setdefault(opt.eff, []).append(opt.launch)
            left, _ = pol.type_left()
        return calls

    def apply_global(self, pol, th, opt):
        s = pol.st[th.id]
        recheck = RECHECK_S[th.cls]
        prev = s.get("assign")
        if prev and prev["kind"] == "air" and not (opt.key.startswith("air:") and opt.unit == prev["unit"]):
            pol.units[prev["unit"]].release(self.t)   # re-assigned away from an aircraft: send it back
        if prev and opt.key.startswith(("now:", "air:")) and prev["unit"] == opt.unit:
            s["next"] = self.t + recheck               # same unit as before: keep the assignment, just re-check later
            return
        if opt.key == "hold":
            s.update(status="held", next=self.t + recheck, assign=None, wait_eff=None)
        elif opt.key == "wait":
            s.update(status="waiting", next=self.t + recheck, assign=None, wait_eff=None)
        elif opt.key == "cheaper":
            s.update(status="waiting", next=max(self.t + 1, min(opt.launch - planner.OPEN_S / 2, self.t + recheck)),
                     wait_until=opt.launch, wait_eff=opt.eff, assign=None)
        elif opt.key.startswith("now:"):
            s.update(status="assigned", assign=dict(kind="ground", unit=opt.unit, eff=opt.eff), eng_at=opt.launch,
                     next=self.t + recheck)
        else:
            u = pol.units[opt.unit]
            r = u.reach(th, self.t, self.cfg)
            u.state, u.task, u.goal = "tasked", th.id, (r[1], r[2])
            pol.L["sorties"] += 1
            pol.L["spent"] += AIR_KINDS[u.kind]["hour_cost"] * 2 * r[6] / 3600
            s.update(status="assigned", assign=dict(kind="air", unit=u.id, eff=r[4]), eng_at=r[0], next=self.t + recheck)

    # ---------------------------------------------------------- LOCAL
    def local_situation(self, pol, th):
        """-> (facts, shot, unit line) if the assigned unit can engage now; ("gone", ...) if the assignment broke; None to keep waiting."""
        s = pol.st[th.id]
        a = s["assign"]
        _, sid, p_hit, *_ = consequence(th, self.t, self.cfg)
        if a["kind"] == "ground":
            bid = a["unit"]
            lat, lon, _ = th.pos(self.t)
            turn = radar_turn(bid, a["eff"], float(lat), float(lon))
            shot = first_shot(th, bid, self.t + planner.REACT_S + (TURN_S if turn else 0))
            if shot is None or pol.stock[bid] < 1:
                return "gone"
            free = pol.free_at(self.t)[bid]
            if free > self.t and first_shot(th, bid, max(self.t + planner.REACT_S, free)) is None:
                return "gone"   # no channel / loaded launcher before this battery's window closes: back to the battle manager
            if shot[0] > self.t + planner.OPEN_S + (TURN_S if turn else 0):
                return None
            later = first_shot(th, bid, self.t + planner.OPEN_S + 20)
            after_look = planner.windows(th, shot[1] + planner.LOOK_S - planner.REACT_S, pol.stock, self.cfg)
            rl = pol.reload_until[bid]
            f = planner.local_facts(th, self.t, "ground", pol.channels_free(bid, self.t), EFFECTORS[a["eff"]].get("channels", 2),
                                    pol.stock[bid], False, p_hit, shot, later, bool(later) or bool(after_look),
                                    ready=pol.ready_n(bid), reload_eta=(rl - self.t) if rl else None, turn=turn,
                                    friendly=self.friendly_near(pol, shot[2], shot[3], shot[4]))
            f["look"] = bool(after_look)
            line = f"Unit: battery {bid} ({SHORT[a['eff']]}), assigned to this threat; it can launch now."
            return f, shot, line
        u = pol.units[a["unit"]]
        r = u.reach(th, self.t, self.cfg)
        if r is None:
            return "gone"
        u.goal = (r[1], r[2])
        if r[0] > self.t + planner.OPEN_S:
            return None
        shot = (self.t, r[0], r[1], r[2], r[3])
        later = u.reach(th, self.t + 60, self.cfg)
        after_look = planner.windows(th, r[0] + planner.LOOK_S - planner.REACT_S, pol.stock, self.cfg)
        f = planner.local_facts(th, self.t, "air", 1, 1, u.weapons.get(r[4], 0), False, p_hit, shot,
                                (0, later[0], later[1], later[2], later[3]) if later else None, bool(later) or bool(after_look))
        line = f"Unit: {u.name} ({AIR_KINDS[u.kind]['label'].lower()}), assigned to this threat; it is in weapon range."
        return f, shot, line

    def friendly_near(self, pol, lat, lon, alt, skip=None):
        """Own aircraft within 15 km of a low intercept point: the ground unit must hold (weapons tight)."""
        return alt < 15 and any(u.id != skip and u.state in ("tasked", "cap", "rtb") and haversine(u.lat, u.lon, lat, lon) < 15
                                for u in pol.units.values())

    def decide_local(self, pol, budget):
        calls = 0
        for th in sorted((x for x in self.threats if pol.st[x.id]["status"] == "assigned"), key=lambda x: x.t0 + x.T):
            s = pol.st[th.id]
            sit = self.local_situation(pol, th)
            if sit is None:
                continue
            if sit == "gone":
                if s["assign"]["kind"] == "air":
                    pol.units[s["assign"]["unit"]].release(self.t)
                s.update(status="pending", next=self.t, assign=None)
                continue
            f, shot, line = sit
            dmg_full = site_value(consequence(th, self.t, self.cfg)[1], th.kind, self.cfg)
            eff = s["assign"]["eff"]
            dmg = consequence(th, self.t, self.cfg)[0]
            if f["look"]:   # a miss leaves the best follow-up, valued from the intercept time + a look
                v_after = min(o.total for o in planner.option_values(th, shot[1] + planner.LOOK_S - planner.REACT_S, pol.stock,
                                                                       self.cfg, pol.prices, dmg))
            else:
                v_after = dmg
            teach = planner.local_teacher(f, EFFECTORS[eff]["cost"] + max(pol.prices.get(eff, (0, 0))),
                                          effective_pk(eff, th, self.cfg), v_after)
            text = local_text(th, self.t, self.cfg, line, f)
            info = dict(teacher=teach, state=text, t=self.t, unit=s["assign"]["unit"],
                        facts={k: (None if v == math.inf else v) for k, v in f.items()})
            if pol.name == "doctrine":
                n = 1 if f["channels_free"] >= 1 and f["stock"] >= 1 else 0
            elif pol.name == "planner" or self.model is None:
                n = teach
            else:
                if not self.async_model and (Q_LOCAL, text, tuple(LEVELS)) not in self.memo:
                    if calls >= budget:
                        break
                    calls += 1
                ans = self.ask(pol, Q_LOCAL, text, LEVELS, shot[0], th.id)
                if ans is None:
                    if not self.late(pol, s, "l", urgent=f["last"] or not self.async_model):
                        continue
                    n = teach
                    info["late"] = True
                else:
                    s.pop("asked_l", None)
                    d, p_null, ms = ans
                    n = int(max(d, key=d.get))
                    info.update(dist={int(k): v for k, v in d.items()}, p_null=p_null, ms=ms,
                                expected=sum(int(k) * v for k, v in d.items()) / max(sum(d.values()), 1e-9))
                    pol.L["local_agree"] += n == teach
                if ans and self.unsure(max(d.values()), p_null):
                    info["escalated"] = True
                    n = teach
            cap = 1 if s["assign"]["kind"] == "air" else f["channels_free"]
            n = int(max(0, min(n, f["stock"], cap)))   # a physical guardrail, not a decision
            if self.record is not None and pol.name == "planner":
                self.record.append(dict(level="local", state=text, q=Q_LOCAL, best=teach, kind=th.kind, scenario=self.scenario))
            pol.L["local_decisions"] += 1
            info["fired"] = n
            s["local"] = info
            if n:
                s.setdefault("log", []).append(dict(t=self.t, fire=n, unit=s["assign"]["unit"], eff=eff))
                self.fire(pol, th, shot, n)
        return calls

    def fire(self, pol, th, shot, n):
        s, a = pol.st[th.id], pol.st[th.id]["assign"]
        e, hit = a["eff"], False
        if a["kind"] == "ground":
            bid = a["unit"]
            src = BAT[bid][2:4]
        else:
            u = pol.units[a["unit"]]
            bid, src = u.id, (u.lat, u.lon)
        for _ in range(n):
            if a["kind"] == "ground":
                pol.stock[bid] -= 1
                pol.ready[bid] -= 1
                pol.busy[bid].append(shot[1])
            else:
                pol.units[bid].weapons[e] -= 1
            pol.L["spent"] += EFFECTORS[e]["cost"]
            pol.L["rounds"][e] = pol.L["rounds"].get(e, 0) + 1
            if e == "ew":
                s["ew_done"] = True
            h = u01(self.seed, th.id, s["shots"]) < effective_pk(e, th, self.cfg)   # same draw for every policy's n-th shot
            s["shots"] += 1
            hit = hit or h
            self.flights.append((pol.name, bid, th.id, shot[0], shot[1], shot[2], shot[3], shot[4], h, src[0], src[1], e))
        s.update(status="inflight", resolve=shot[1], hit=hit)

    # ---------------------------------------------------------- time
    def resolve(self, pol):
        for th in self.threats:
            s = pol.st[th.id]
            if s["status"] == "inflight" and self.t >= s["resolve"]:
                if s["assign"] and s["assign"]["kind"] == "air":
                    pol.units[s["assign"]["unit"]].release(self.t)
                if s["hit"]:
                    s.update(status="killed", killed_at=s["resolve"])
                    pol.L["kills"] += 1
                else:
                    s.update(status="pending", next=self.t, assign=None)   # missed: back to the battle manager
            if s["status"] not in DONE and self.t >= th.t0 + th.T:
                if s["assign"] and s["assign"]["kind"] == "air" and s["status"] == "assigned":
                    pol.units[s["assign"]["unit"]].release(self.t)
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
        if self.async_model:
            self.requests = {}   # rebuilt every tick: only situations that are still current get scored
        calls = {}
        for n, p in self.policies.items():
            p.reload(self.t)
            calls[n] = self.decide_global(p, budget) + self.decide_local(p, budget)
            self.resolve(p)
        self.advance(dt)
        return calls

    def run_for(self, sim_dt, on_step=None):
        """Live demo clock: owe sim_dt seconds, pay them in fixed DT steps, jump over quiet stretches. The steps (and so
        every decision and every question Typical is asked) are the same whatever the speed, pauses or replays.
        -> sim seconds skipped."""
        self.owed = getattr(self, "owed", 0.0) + sim_dt
        skipped = 0.0
        while self.owed >= DT and self.t < self.end:
            self.owed -= DT
            self.step(DT)
            if on_step:
                on_step(self)
            if (nxt := self.next_event()) - self.t > QUIET_S:
                skipped += nxt - 5 - self.t
                self.advance(nxt - 5 - self.t)
        return skipped

    def advance(self, dt):
        for p in self.policies.values():
            for u in p.units.values():
                u.move(self.t, dt)
        self.t = min(self.t + dt, self.end)

    def idle_until(self):
        if any(s["status"] in ("inflight", "waiting", "held", "assigned") for p in self.policies.values() for s in p.st.values()):
            return self.t
        nxt = [max(th.t0, th.first_launch - HORIZON_S) for th in self.threats if max(th.t0, th.first_launch - HORIZON_S) > self.t]
        # a live "pending" threat (nothing reaches it yet) still has its recheck and its impact to resolve: never jump past those
        nxt += [max(self.t, min(p.st[th.id]["next"], th.t0 + th.T)) for p in self.policies.values() for th in self.threats
                if th.t0 <= self.t and p.st[th.id]["status"] == "pending"]
        return min(nxt + [self.end])

    def next_event(self):
        """Next sim time worth watching; the live demo jumps the quiet time in between."""
        ev = [self.end]
        for th in self.threats:
            if th.t0 > self.t:
                ev.append(th.t0 if th.cls in "BR" else th.t0 + 60)
                continue
            for pol in self.policies.values():
                s = pol.st[th.id]
                if s["status"] in DONE:
                    continue
                ev.append(th.t0 + th.T)
                if s["status"] == "inflight":
                    ev.append(s["resolve"])
                elif s["status"] == "assigned":
                    ev.append(s.get("eng_at", self.t) - planner.OPEN_S)
                elif s["status"] == "waiting":   # a plain wait hours out is re-asked at ~10% of its remaining flight, not every recheck
                    ev.append(s["wait_until"] - planner.OPEN_S if s.get("wait_until") else max(s["next"], self.t + 0.1 * (th.t0 + th.T - self.t)))
                elif s["status"] == "pending":
                    ev.append(th.first_launch - HORIZON_S if not self.in_horizon(th) else self.t)
        return max(self.t, min(ev))

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
        if (nxt := w.idle_until()) > w.t:
            w.advance(nxt - w.t)
    return w


if __name__ == "__main__":   # ponytail: the self-check -- the planner must not lose to doctrine on total cost over seeds
    for sc in SCENARIOS:
        tot, hits, spent = {"doctrine": 0.0, "planner": 0.0}, {"doctrine": 0, "planner": 0}, {"doctrine": 0.0, "planner": 0.0}
        t0 = time.time()
        for seed in (1, 2, 3):
            w = run_headless(sc, seed=seed, policies=("doctrine", "planner"))
            assert all(v >= 0 for p in w.policies.values() for v in p.stock.values()), sc
            for n, x in w.ledger().items():
                tot[n] += x["total"]; hits[n] += x["leaks"]; spent[n] += x["spent"]
        print(f"{sc:20s} total  doctrine {tot['doctrine']:8.0f}  planner {tot['planner']:8.0f} | spent {money(spent['doctrine'])} vs "
              f"{money(spent['planner'])} | hits {hits['doctrine']} vs {hits['planner']}  ({time.time() - t0:.0f}s)")
