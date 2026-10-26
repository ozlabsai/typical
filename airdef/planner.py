"""The multi-lens teacher: what to do about one threat *right now*, priced through three lenses.

- local    : this threat. A small DP over its remaining engagement windows (fire / look / fire again), pricing
             interceptor $ and the expected damage of a leak. Shoot-look-shoot falls out, it isn't hard-coded.
- raid     : tonight. Each interceptor type gets a shadow price from the competition among the threats still
             inbound: if demand exceeds stock, a round is worth what the marginal threat loses without it
             (Lagrangian relaxation of the weapon-target assignment coupling).
- campaign : the war. A round that survives tonight is worth replacement cost x scarcity x posture.
A round's binding price is max(raid, campaign): it is either used tonight or kept.

Options at a decision point: fire E now (E can launch within OPEN_S), fire two E now, wait for E (its window
opens later), let it fall. Re-decided whenever something changes, so "wait" now can become "fire" later.
"""
import math
from dataclasses import dataclass, field

from sim import BATTERIES, EFFECTORS, LOOK_S, REACT_S, THREATS, consequence, effective_pk, first_shot

OPEN_S = 12.0          # "fire now" = the effector can launch within this many seconds
MAX_SHOTS = 3          # DP depth: at most three engagements per threat
POSTURE = [0.0, 1.0, 4.0]   # campaign lens multiplier for reserve 0 healthy / 1 conserve / 2 critical
BAT_EFF = {b[0]: b[1] for b in BATTERIES}


@dataclass
class Option:
    key: str                 # "fire:arrow3" | "fire2:arrow3" | "wait:dsling" | "hold"
    eff: str | None
    launch: float | None     # earliest launch time of the first shot
    total: float             # expected cost, $M-equivalent, all lenses
    lenses: dict = field(default_factory=dict)   # local / raid / campaign parts of `total`
    pk: float = 0.0


def windows(th, t, avail, cfg):
    """effector -> (battery, launch, intercept) of the earliest shot from a battery with stock, launch >= t + REACT."""
    out = {}
    for bid in th.samples:
        eff = BAT_EFF[bid]
        if (EFFECTORS[eff].get("allied") and not cfg["allied"]) or avail.get(bid, 0) < 1:
            continue
        s = first_shot(th, bid, t + REACT_S)
        if s and (eff not in out or s[0] < out[eff][1]):
            out[eff] = (bid, s[0], s[1])
    return out


def campaign_price(eff, left, init, cfg):
    if EFFECTORS[eff].get("unlimited"):
        return 0.0
    return EFFECTORS[eff]["cost"] * POSTURE[cfg["reserve"]] * init.get(eff, 1) / max(left.get(eff, 0), 0.5)


def option_values(th, t, avail, cfg, prices, dmg):
    """All options for this threat at time t. prices: eff -> (raid, campaign) shadow prices, $M per round."""
    win = windows(th, t, avail, cfg)
    if not win:
        return [Option("hold", None, None, dmg, {"local": dmg, "raid": 0.0, "campaign": 0.0})]
    lam = {e: max(prices.get(e, (0, 0))) for e in win}
    memo = {}

    def follow(after, depth):
        """Expected cost from `after` on, acting optimally: the local DP (prices folded in)."""
        if depth >= MAX_SHOTS:
            return dmg
        key = (round(after), depth)
        if key not in memo:
            best = dmg
            for e, (bid, *_ ) in win.items():
                s = first_shot(th, bid, after)
                if s:
                    pk = effective_pk(e, th, cfg)
                    best = min(best, EFFECTORS[e]["cost"] + lam[e] + (1 - pk) * follow(s[1] + LOOK_S, depth + 1))
            memo[key] = best
        return memo[key]

    def priced(key, e, n, launch, t_int, pk):
        nxt = follow(t_int + LOOK_S, 1 if n == 1 else 2)
        raid, camp = prices.get(e, (0, 0))
        bind = "raid" if raid >= camp else "campaign"
        local = n * EFFECTORS[e]["cost"] + (1 - pk) ** n * nxt
        lenses = {"local": local, "raid": n * lam[e] if bind == "raid" else 0.0, "campaign": n * lam[e] if bind == "campaign" else 0.0}
        return Option(key, e, launch, local + n * lam[e], lenses, 1 - (1 - pk) ** n)

    opts = [Option("hold", None, None, dmg, {"local": dmg, "raid": 0.0, "campaign": 0.0})]
    for e, (bid, launch, t_int) in win.items():
        pk = effective_pk(e, th, cfg)
        if launch <= t + OPEN_S:
            opts.append(priced(f"fire:{e}", e, 1, launch, t_int, pk))
            if avail[bid] >= 2 and not EFFECTORS[e].get("unlimited"):
                opts.append(priced(f"fire2:{e}", e, 2, launch, t_int, pk))
        else:
            opts.append(priced(f"wait:{e}", e, 1, launch, t_int, pk))
    return opts


def raid_prices(threats, t, avail, left, init, cfg, engaged=()):
    """Shadow price per effector type from tonight's remaining demand (greedy marginal; campaign part alongside).
    ponytail: one pass, each threat demands only its single best effector; a full LP if the demo ever needs it."""
    camp = {e: campaign_price(e, left, init, cfg) for e in EFFECTORS}
    zero = {e: (0.0, camp[e]) for e in EFFECTORS}
    demand = {}
    for th in threats:
        if th.id in engaged or t > th.t0 + th.T:   # not-yet-launched threats count: the caller already filtered by forecast
            continue
        dmg = consequence(th, t, cfg)[0]
        opts = [o for o in option_values(th, t, avail, cfg, zero, dmg) if o.eff]
        if not opts:
            continue
        opts.sort(key=lambda o: o.total)
        best = opts[0]
        alt = min([o.total for o in opts if o.eff != best.eff] + [dmg])
        demand.setdefault(best.eff, []).append(alt - best.total)
    out = {}
    for e in EFFECTORS:
        gaps = sorted(demand.get(e, []), reverse=True)
        supply = math.inf if EFFECTORS[e].get("unlimited") else int(max(left.get(e, 0), 0))
        out[e] = (gaps[supply] if len(gaps) > supply else 0.0, camp[e])
    return out


def best(opts):
    return min(opts, key=lambda o: o.total)


def spans(th, t, avail, cfg):
    """effector -> (first launch, last launch) still reachable from now: the bars on the threat's timeline."""
    out = {}
    for bid, (ts, fly, *_ ) in th.samples.items():
        eff = BAT_EFF[bid]
        if (EFFECTORS[eff].get("allied") and not cfg["allied"]) or avail.get(bid, 0) < 1:
            continue
        launch = ts - fly
        ok = launch >= t
        if ok.any():
            a, b = float(launch[ok].min()), float(launch[ok].max())
            lo, hi = out.get(eff, (a, b))
            out[eff] = (min(lo, a), max(hi, b))
    return out
