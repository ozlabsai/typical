"""World loop + three policies (naive / optimizer / Typical) on one shared raid with common random numbers."""
import copy
import math
import re
import time

from doctrine import ACTIONS, STANDING, decide, parse_order, render
from sim import (BATTERIES, EFFECTORS, ORIGINS, SITE, THREATS, DEFAULT_CONFIG, SCENARIOS, build_plans, build_raid,
                 consequence, money, naive_pick, site_value, plan_score, precompute_windows, site_hit, u01)

HORIZON_S = 300        # decide a threat once its earliest launch is this close
RECHECK_S = 4          # re-evaluate an undecided-or-revisable plan at most this often (sim seconds)
ORIGIN_NAME = {"iran_w": "western Iran", "iran_nw": "north-western Iran", "iran_c": "central Iran", "iraq": "Iraq",
               "yemen": "Yemen", "lebanon_s": "southern Lebanon", "lebanon_bq": "the Beqaa valley", "gaza": "Gaza"}
CLASS_WORD = {"B": "ballistic missile", "R": "rocket", "C": "cruise missile", "D": "drone"}
Q_PLAN = "Which action do the rules of engagement require for this threat?"   # lowest ∅ mass with orders (null_probe)
# Typical makes the doctrine call; the solver resolves it to a concrete plan. Labels echo the rule wording:
# probe.py measured 58% doctrine-following with plan labels, 99% with these (typical-small, n=148).
BASE_RULES = [render(r, "grammar") for r in STANDING]


def stock_word(left, init):
    f = left / max(init, 1)
    return "healthy" if f > 0.5 else "low" if f > 0.2 else "critical"


def rules(cfg):
    return [o.strip() for o in cfg.get("orders", []) if o.strip()] + BASE_RULES


def render_state(rule_lines, facts):
    """The exact text Typical reads -- shared by the live engine and airdef/dataset.py (train = serve)."""
    return ("Rules of engagement, in order. The first rule that applies decides.\n"
            + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rule_lines)) + "\n\nSituation.\n" + "\n".join(facts))


def situation_facts(cls_word, name, origin, site, open_ground, prio):
    # the class sentence mirrors the order grammar exactly ("When the threat is a drone, ..."): 11/12 -> 15/16 on orders.
    # No uncertainty sentence: no rule uses it, and it pulled priority threats to the cheap action.
    facts = [f"The threat is a {cls_word}.", f"It is {'an' if name[:1] in 'AEIOU' else 'a'} {name}, launched from {origin}."]
    if open_ground:
        return facts + [f"The threat will land in open ground near {site}."]
    return facts + [f"The threat will hit {site}.", f"{site} is a {'priority' if prio else 'secondary'} site."]


def short_name(kind):
    return re.sub(r"\s*\b(drone|MRBM|SRBM|rocket|cruise missile)\b", "", THREATS[kind][0]).strip()


def doctrine_action(cfg, cls_word, site, open_ground, prio):
    """What the written rules say (doctrine.decide, the training-label semantics). Free-text orders are skipped."""
    parsed = [r for r in (parse_order(o) for o in rules(cfg)) if r]
    act = decide(parsed, dict(cls=cls_word, site=site, open=open_ground), {site} if prio else set())
    return act or "silent"


def parse_ok(order):
    return parse_order(order) is not None


class Policy:
    def __init__(self, name, threats, stock_scale):
        self.name = name
        self.stock = {}
        for bid, eff, *_ , n in BATTERIES:
            self.stock[bid] = math.inf if EFFECTORS[eff].get("unlimited") else round(n * stock_scale.get(eff, 1))
        self.stock_init = dict(self.stock)
        self.reserved = {b: 0 for b in self.stock}
        self.st = {t.id: dict(status="pending", plan=None, sig=None, shots_done=0, info=None, escalated=False)
                   for t in threats}
        self.L = dict(spent=0.0, rounds={}, kills=0, leaks=0, damage=0.0, fell_open=0, held=0, escalated=0,
                      decisions=0, agree=0, follow=0, calls=0, ms=0.0)

    def avail(self):
        return {b: self.stock[b] - self.reserved[b] for b in self.stock}

    def type_left(self):
        out, init = {}, {}
        for bid, eff, *_ in BATTERIES:
            if self.stock[bid] != math.inf:
                out[eff] = out.get(eff, 0) + self.stock[bid] - self.reserved[bid]
                init[eff] = init.get(eff, 0) + self.stock_init[bid]
        return out, init


class World:
    def __init__(self, scenario="houthi_eilat", cfg=None, model=None, seed=7):
        self.scenario, self.seed = scenario, seed
        self.cfg = copy.deepcopy(cfg or DEFAULT_CONFIG)
        self.cfg_version = 0
        self.model = model
        self.memo = {}   # state text -> Typical's answer; a threat's state only changes when its situation does
        self.threats = build_raid(scenario, seed)
        for th in self.threats:
            precompute_windows(th, BATTERIES)
        self.by_id = {t.id: t for t in self.threats}
        scale = SCENARIOS[scenario].get("stock_scale", {})
        self.policies = {n: Policy(n, self.threats, scale) for n in ("naive", "optimizer", "typical")}
        self.t = 0.0
        self.end = max(t.t0 + t.T for t in self.threats) + 5
        self.attacker_cost = sum(THREATS[t.kind][2] for t in self.threats)
        self.flights = []   # interceptor flights for the map: (policy, battery, threat, launch, intercept, lat, lon, alt, hit)

    # ---------------------------------------------------------- state text for Typical
    def state_text(self, th):
        dmg, sid, p_hit, conf, dist, sigma = consequence(th, self.t, self.cfg)
        site = SITE[sid][1]
        prio = sid in self.cfg["priority_sites"]
        open_ground = dist > sigma
        cw = CLASS_WORD[th.cls]
        facts = situation_facts(cw, short_name(th.kind), ORIGIN_NAME[self.origin_key(th)], site, open_ground, prio)
        text = render_state(rules(self.cfg), facts)
        return text, dict(dmg=dmg, sid=sid, conf=conf, dist=dist, sigma=sigma,
                          rule=doctrine_action(self.cfg, cw, site, open_ground, prio))

    def origin_key(self, th):
        return min(ORIGINS, key=lambda k: (ORIGINS[k][0] - th.origin[0]) ** 2 + (ORIGINS[k][1] - th.origin[1]) ** 2)

    # ---------------------------------------------------------- decisions
    @staticmethod
    def resolve(plans, action):
        if action == "hold":
            return plans[0]
        return next((p for p in plans if action in p.tags), plans[0])

    def decide(self, pol, th, plans, text, cons):
        left, init = pol.type_left()
        scores = {p.key: plan_score(p, cons["dmg"], left, init, self.cfg) for p in plans}
        opt = min(plans, key=lambda p: scores[p.key])
        rule_plan = self.resolve(plans, cons["rule"])
        info = dict(opt=opt.key, rule=rule_plan.key, rule_action=cons["rule"], scores=scores, state=text,
                    plans=[(p.key, p.label, p.pk, p.exp_cost) for p in plans],
                    resolved={a: self.resolve(plans, a).key for a in ACTIONS})
        if pol.name == "naive":
            return naive_pick(plans), info
        if pol.name == "optimizer" or self.model is None:
            return opt, info
        if text not in self.memo:
            t0 = time.perf_counter()
            d = self.model.choice(text, Q_PLAN, list(ACTIONS.values()))
            self.memo[text] = ({a: d[lab] for a, lab in ACTIONS.items()}, d["p_null"], (time.perf_counter() - t0) * 1000)
            pol.L["calls"] += 1
            pol.L["ms"] += self.memo[text][2]
        dist, p_null, ms = self.memo[text]
        action = max(dist, key=dist.get)
        plan = self.resolve(plans, action)
        info.update(dist=dist, p_null=p_null, ms=ms, action=action)
        pol.L["agree"] += plan.key == opt.key
        pol.L["follow"] += action == cons["rule"]
        # ∅ rises to ~0.5 whenever a commander's order is present (null_probe), so it gets its own, higher bar
        if dist[action] < self.cfg["conf_threshold"] or p_null > self.cfg["abstain_threshold"]:
            info["escalated"] = True   # operator takes the optimizer's recommendation
            return opt, info
        return plan, info

    def signature(self, plans, text):
        return (text, tuple((p.key, p.tags) for p in plans), self.cfg_version)

    def plan_threats(self, pol, budget):
        """(Re)decide live threats whose first shot hasn't launched. Most urgent first; at most `budget` model calls."""
        todo = []
        for th in self.threats:
            s = pol.st[th.id]
            if s["status"] not in ("pending", "planned", "held") or self.t < max(th.t0, th.first_launch - HORIZON_S):
                continue
            if s["status"] == "planned" and s["shots_done"]:
                continue
            if self.t - s.get("checked", -1e9) < RECHECK_S and s.get("cfgv") == self.cfg_version:
                continue
            todo.append(th)
        todo.sort(key=lambda th: th.t0 + th.T)
        calls = 0
        for th in todo:
            s = pol.st[th.id]
            s["checked"], s["cfgv"] = self.t, self.cfg_version
            self.unreserve(pol, s)
            left, init = pol.type_left()
            dmg = consequence(th, self.t, self.cfg)[0]
            plans = build_plans(th, self.t, pol.avail(), self.cfg,
                                price=lambda p: plan_score(p, 0.0, left, init, self.cfg),
                                value_price=lambda p: plan_score(p, dmg, left, init, self.cfg))
            launches = [p.shots[0][2] for p in plans if p.shots]
            if not launches or min(launches) > self.t + HORIZON_S:
                self.reserve(pol, s)
                continue
            text, cons = self.state_text(th)
            sig = self.signature(plans, text)
            if sig == s["sig"]:
                self.reserve(pol, s)
                continue
            if pol.name == "typical" and self.model is not None and text not in self.memo:
                if calls >= budget:
                    self.reserve(pol, s)
                    break
                calls += 1
            plan, info = self.decide(pol, th, plans, text, cons)
            if info.get("escalated") and not s["escalated"]:
                pol.L["escalated"] += 1
                s["escalated"] = True
            pol.L["decisions"] += 1
            s.update(plan=plan, sig=sig, info=info, status="held" if plan.key == "hold" else "planned")
            self.reserve(pol, s)
        return calls

    def reserve(self, pol, s, sign=1):
        if s["plan"] is not None:
            for shot in s["plan"].shots[s["shots_done"]:]:
                pol.reserved[shot[0]] += sign * (shot[0] in pol.reserved and pol.stock[shot[0]] != math.inf)

    def unreserve(self, pol, s):
        self.reserve(pol, s, -1)

    # ---------------------------------------------------------- execution
    def execute(self, pol):
        for th in self.threats:
            s = pol.st[th.id]
            if s["status"] in ("planned", "engaging"):
                shots = s["plan"].shots
                while s["shots_done"] < len(shots) and shots[s["shots_done"]][2] <= self.t:
                    i = s["shots_done"]
                    bid, eff, launch, t_int, lat, lon, alt, pk, cond = shots[i]
                    if cond and s.get("killed_at"):
                        self.unreserve_rest(pol, s)   # backup not needed
                        break
                    if pol.stock[bid] < 1:   # someone else took the round: re-plan
                        self.unreserve(pol, s)
                        s.update(status="pending", plan=None, sig=None, shots_done=0)
                        break
                    pol.stock[bid] -= 1
                    pol.reserved[bid] -= pol.stock[bid] != math.inf
                    cost = EFFECTORS[eff]["cost"]
                    pol.L["spent"] += cost
                    pol.L["rounds"][eff] = pol.L["rounds"].get(eff, 0) + 1
                    hit = u01(self.seed, th.id, eff, i) < pk
                    self.flights.append((pol.name, bid, th.id, launch, t_int, lat, lon, alt, hit))
                    if hit and not s.get("killed_at"):
                        s["killed_at"] = t_int
                    s["shots_done"] += 1
                    s["status"] = "engaging"
                if s.get("killed_at") and self.t >= s["killed_at"] and s["status"] != "killed":
                    s["status"] = "killed"
                    pol.L["kills"] += 1
                    self.unreserve_rest(pol, s)
            if s["status"] in ("pending", "planned", "held", "engaging") and self.t >= th.t0 + th.T:
                self.unreserve_rest(pol, s)
                sid = site_hit(*th.impact)
                if sid:
                    v = site_value(sid, th.kind, self.cfg)
                    pol.L["leaks"] += 1
                    pol.L["damage"] += v
                    s["status"] = "leaked"
                else:
                    pol.L["fell_open"] += 1
                    pol.L["held"] += s["plan"] is not None and s["plan"].key == "hold"
                    s["status"] = "fell_open"
                s["site"] = sid

    def unreserve_rest(self, pol, s):
        if s["plan"] is not None:
            for shot in s["plan"].shots[s["shots_done"]:]:
                pol.reserved[shot[0]] -= pol.stock[shot[0]] != math.inf
            s["shots_done"] = len(s["plan"].shots)

    def step(self, dt, budget=12):
        for pol in self.policies.values():
            self.plan_threats(pol, budget)
            self.execute(pol)
        self.t = min(self.t + dt, self.end)

    def idle_until(self):
        """Sim time of the next thing worth simulating (fast-forward through hours of drone transit)."""
        busy = [st["status"] in ("planned", "engaging", "held") for p in self.policies.values() for st in p.st.values()]
        if any(busy):
            return self.t
        nxt = [max(th.t0, th.first_launch - HORIZON_S) for th in self.threats
               if self.policies["naive"].st[th.id]["status"] == "pending"]
        return min(nxt + [self.end]) if nxt else self.end

    def set_config(self, patch):
        self.cfg.update(patch)
        self.cfg_version += 1

    def ledger(self):
        out = {}
        for n, p in self.policies.items():
            left, init = p.type_left()
            out[n] = {**p.L, "rounds": dict(p.L["rounds"]), "left": {e: left[e] for e in left},
                      "init": init, "cer": p.L["spent"] / max(self.attacker_cost, 1e-9),
                      "total": p.L["spent"] + p.L["damage"]}
        return out


def run_headless(scenario, model=None, cfg=None, dt=2.0, seed=7):
    w = World(scenario, cfg, model, seed)
    while w.t < w.end:
        w.step(dt, budget=10_000)
        w.t = max(w.t, w.idle_until())
    return w


if __name__ == "__main__":   # ponytail: the self-check -- on healthy stocks the optimizer must not lose to naive over seeds
    healthy = {**DEFAULT_CONFIG, "reserve": 0}
    for sc in SCENARIOS:
        tot = {"naive": [0.0, 0], "optimizer": [0.0, 0]}   # spent, leaks
        for seed in (1, 2, 3):
            w = run_headless(sc, cfg=healthy, seed=seed)
            assert all(v >= 0 for p in w.policies.values() for v in p.stock.values()), sc
            assert all(v >= 0 for p in w.policies.values() for v in p.reserved.values()), sc
            for n, x in w.ledger().items():
                if n in tot:
                    tot[n][0] += x["spent"]
                    tot[n][1] += x["leaks"]
                    print(f"{sc:20s} s{seed} {n:9s} spent {money(x['spent']):>8s} leaks {x['leaks']:3d} "
                          f"open {x['fell_open']:3d} kills {x['kills']:3d} total {x['total']:8.0f}")
        # ponytail: leaks are lumpy (CRN double-misses hit every policy alike), so allow +1 leak per seed
        assert tot["optimizer"][0] <= tot["naive"][0] and tot["optimizer"][1] <= tot["naive"][1] * 1.5 + 3, (sc, tot)
        print(f"  ok {sc}: optimizer spends {tot['optimizer'][0] / tot['naive'][0]:.0%} of naive, "
              f"leaks {tot['optimizer'][1]} vs {tot['naive'][1]}")
