"""The teacher at both decision levels (DECISIONS.md), priced through three lenses.

- local lens    : this threat. A small DP over its remaining engagement windows (fire / look / fire again), pricing
                  interceptor $ and the expected damage of a leak. Shoot-look-shoot falls out, it isn't hard-coded.
- raid lens     : tonight. Each interceptor type gets a shadow price from the competition among threats still inbound
                  or expected (Lagrangian relaxation of the weapon-target assignment coupling).
- campaign lens : the war. A round that survives tonight is worth replacement cost x scarcity x posture.

GLOBAL (battle manager), one threat at time t -> option keys:
  "now:<eff>"      a ground interceptor that can launch now (the label is its name)
  "air:<kind>"     scramble / vector the best free fighter or helicopter
  "cheaper"        commit to a cheaper ground layer that reaches it later
  "wait"           no commitment yet -- valued as deciding after the track firms up (value of information)
  "hold"           let it fall
LOCAL (the assigned unit), every tick it can engage: how many interceptors to fire now, 0-3 (a Score) -- local_teacher.
"""
import math
from dataclasses import dataclass, field

from sim import AIR_KINDS, BATTERIES, EFFECTORS, LOOK_S, REACT_S, SITES, consequence, effective_pk, first_shot, haversine

OPEN_S = 12.0          # "can fire now" = launch within this many seconds
MAX_SHOTS = 3          # DP depth: at most three engagements per threat
POSTURE = [0.0, 1.0, 4.0]   # campaign multiplier for reserve 0 healthy / 1 conserve / 2 critical
FIRM = 0.15            # the track is firm when P(hit a site) is below FIRM or above 1 - FIRM
DEBRIS = 0.05          # share of a site's value lost to intercept debris falling on it (low-altitude intercepts)
BAT_EFF = {b[0]: b[1] for b in BATTERIES}


@dataclass
class Option:
    key: str
    eff: str | None                  # ground effector or air weapon
    launch: float | None             # earliest launch / engagement time
    total: float                     # expected cost, $M-equivalent, all lenses
    lenses: dict = field(default_factory=dict)
    pk: float = 0.0
    unit: str | None = None          # battery id or air unit id
    detail: dict = field(default_factory=dict)


def windows(th, t, avail, cfg, free_at=None):
    """ground effector -> (battery, launch, intercept) of the earliest shot from a battery with stock and, when
    free_at is given (battery -> when a launch channel frees), a channel free at launch time."""
    out = {}
    for bid in th.samples:
        eff = BAT_EFF[bid]
        if (EFFECTORS[eff].get("allied") and not cfg["allied"]) or avail.get(bid, 0) < 1:
            continue
        s = first_shot(th, bid, max(t + REACT_S, (free_at or {}).get(bid, 0.0)))
        if s and (eff not in out or s[0] < out[eff][1]):
            out[eff] = (bid, s[0], s[1])
    return out


def campaign_price(eff, left, init, cfg):
    if EFFECTORS[eff].get("unlimited") or EFFECTORS[eff].get("air"):
        return 0.0
    return EFFECTORS[eff]["cost"] * POSTURE[cfg["reserve"]] * init.get(eff, 1) / max(left.get(eff, 0), 0.5)


def firm(p_hit):
    return p_hit < FIRM or p_hit > 1 - FIRM


def p_channel(eff, launch, bookings, cfg, span=60.0):
    """Chance a channel of this layer is free at `launch`, from what other threats have already booked around then."""
    if not bookings:
        return 1.0
    channels = sum(EFFECTORS[eff].get("channels", 2) for b in BATTERIES if b[1] == eff
                   and not (EFFECTORS[eff].get("allied") and not cfg["allied"]))
    load = sum(1 for x in bookings.get(eff, ()) if abs(x - launch) <= span)
    return min(1.0, channels / (load + 1))


def option_values(th, t, avail, cfg, prices, dmg, units=(), full_dmg=None, p_hit=1.0, free_at=None, bookings=None):
    """Global options for one threat at time t. prices: eff -> (raid, campaign) $M per round. units: air units.
    full_dmg / p_hit: damage if it does hit a site, and how likely that is now (for the value of waiting)."""
    win = windows(th, t, avail, cfg, free_at)
    lam = {e: max(prices.get(e, (0, 0))) for e in win}

    def dp(d):
        memo = {}

        def follow(after, depth):
            if depth >= MAX_SHOTS:
                return d
            key = (round(after), depth)
            if key not in memo:
                best = d
                for e, (bid, *_ ) in win.items():
                    s = first_shot(th, bid, after)
                    if s:
                        best = min(best, EFFECTORS[e]["cost"] + lam[e] + (1 - effective_pk(e, th, cfg)) * follow(s[1] + LOOK_S, depth + 1))
                memo[key] = best
            return memo[key]
        return follow

    follow = dp(dmg)

    def priced(key, e, t_int, launch, unit, extra_cost=0.0, pfree=1.0):
        pk = effective_pk(e, th, cfg) * pfree   # a later layer only helps if a channel is free by then (congestion)
        lam_e = lam.get(e, 0.0)
        raid, camp = prices.get(e, (0, 0))
        best_n, best = 1, None
        for n in (1, 2) if not EFFECTORS[e].get("air") else (1,):   # the salvo size is the local level's call; the
            tot = n * (EFFECTORS[e]["cost"] + lam_e) + extra_cost + (1 - pk) ** n * follow(t_int + LOOK_S, n)   # value takes the better
            if best is None or tot < best:
                best_n, best = n, tot
        local = best - best_n * lam_e
        bind = "raid" if raid >= camp else "campaign"
        return Option(key, e, launch, best, {"local": local, "raid": best_n * lam_e if bind == "raid" else 0.0,
                                             "campaign": best_n * lam_e if bind == "campaign" else 0.0},
                      1 - (1 - pk) ** best_n, unit, {"salvo": best_n, "intercept": t_int})

    opts = [Option("hold", None, None, dmg, {"local": dmg, "raid": 0.0, "campaign": 0.0})]
    now, later = [], []
    for e, (bid, launch, t_int) in win.items():
        later_one = launch > t + OPEN_S
        o = priced(f"later:{e}" if later_one else f"now:{e}", e, t_int, launch, bid,
                   pfree=p_channel(e, launch, bookings, cfg) if later_one else 1.0)
        (now if launch <= t + OPEN_S else later).append(o)
    opts += now
    for kind in AIR_KINDS:   # the best free unit of each kind
        best = None
        for u in units:
            if u.kind != kind or not (u.available() or (u.state == "tasked" and u.task == th.id)):   # incl. the unit already on it
                continue
            r = u.reach(th, t, cfg)
            if r:
                t_eng, *_, w, pk, fly = r
                o = priced(f"air:{kind}", w, t_eng, t_eng, u.id, extra_cost=AIR_KINDS[kind]["hour_cost"] * 2 * fly / 3600)
                if best is None or o.total < best.total:
                    best = o
        if best:
            opts.append(best)
    if later:
        cheap = min(later, key=lambda o: o.total)
        cheapest_now = min((o.total for o in now), default=math.inf)
        if cheap.total <= cheapest_now or not now:   # "cheaper" only when it is: a later layer that beats firing now
            opts.append(Option("cheaper", cheap.eff, cheap.launch, cheap.total, cheap.lenses, cheap.pk, cheap.unit, cheap.detail))
        # value of information: decide once the track firms up, engaging only if it turns out to be a hit -- counting
        # only the layers still reachable by then (ponytail: the track is assumed firm after 35% of the remaining flight)
        firm_at = t + 0.35 * (th.t0 + th.T - t)
        after = [o for o in later if o.launch >= firm_at]
        if not firm(p_hit) and full_dmg is not None and after:
            follow_full = dp(full_dmg)
            best_later = min(EFFECTORS[o.eff]["cost"] + lam.get(o.eff, 0) + (1 - o.pk) * follow_full(o.detail["intercept"] + LOOK_S, 1)
                             for o in after)
            v = p_hit * min(best_later, full_dmg)
            opts.append(Option("wait", None, None, v, {"local": v, "raid": 0.0, "campaign": 0.0}))
    return opts


def raid_prices(threats, t, avail, left, init, cfg, engaged=()):
    """Shadow price per ground effector from tonight's remaining demand (greedy marginal; campaign part alongside).
    ponytail: one pass, each threat demands only its single best effector; a full LP if the demo ever needs it."""
    camp = {e: campaign_price(e, left, init, cfg) for e in EFFECTORS}
    zero = {e: (0.0, camp[e]) for e in EFFECTORS}
    demand = {}
    for th in threats:
        if th.id in engaged or t > th.t0 + th.T:
            continue
        dmg = consequence(th, t, cfg)[0]
        opts = [o for o in option_values(th, t, avail, cfg, zero, dmg) if o.key.startswith(("now:", "cheaper"))]
        if not opts:
            continue
        opts.sort(key=lambda o: o.total)
        b = opts[0]
        alt = min([o.total for o in opts if o.eff != b.eff] + [dmg])
        demand.setdefault(b.eff, []).append(alt - b.total)
    out = {}
    for e in EFFECTORS:
        gaps = sorted(demand.get(e, []), reverse=True)
        supply = math.inf if EFFECTORS[e].get("unlimited") or EFFECTORS[e].get("air") else int(max(left.get(e, 0), 0))
        out[e] = (gaps[supply] if len(gaps) > supply else 0.0, camp[e])
    return out


SWITCH = 0.10   # hysteresis: keep the standing decision unless an alternative is this much cheaper


def best(opts, current=None):
    top = min(opts, key=lambda o: o.total)
    keep = next((o for o in opts if o.key == current), None)
    return keep if keep and top.total > keep.total * (1 - SWITCH) else top


def spans(th, t, avail, cfg):
    """ground effector -> (first launch, last launch) still reachable from now: the bars on the threat's timeline."""
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


# ------------------------------------------------------------------------------- local level
def debris_site(lat, lon, alt):
    """A populated or infrastructure site under a low-altitude intercept point, else None."""
    if alt > 20:
        return None
    for sid, _, slat, slon, r, tier in SITES:
        if tier != "strategic" and haversine(lat, lon, slat, slon) <= r + 3:
            return sid
    return None


def local_facts(th, t, unit_kind, channels_free, channels, stock, engaging, p_hit, shot, later_shot, any_later,
                ready=None, reload_eta=None, turn=False, friendly=False):
    """Everything the fire unit knows when asked "how many now?". shot / later_shot: (launch, intercept, lat, lon, alt).
    ready: rounds on launchers (None = no launcher limit); reload_eta: seconds until the next launcher is reloaded;
    turn: the sector radar had to turn first; friendly: own aircraft near the intercept point."""
    return dict(channels_free=channels_free, channels=channels, stock=stock, engaging=engaging, firm=firm(p_hit),
                p_hit=p_hit, debris=debris_site(*shot[2:5]) if shot else None,
                later_open=bool(later_shot) and debris_site(*later_shot[2:5]) is None,
                last=not any_later, look=any_later, unit_kind=unit_kind, ready=stock if ready is None else ready,
                reload_eta=reload_eta, turn=turn, friendly=friendly)


def local_teacher(f, cost, pk, v_after):
    """How many interceptors to fire now (a 0-3 Score). Hold for the rules a fire unit obeys; otherwise the n that
    minimises n * cost + (1 - pk)^n * v_after, where v_after is what a miss leaves: the next layer's expected cost when
    there is time to look, the expected leak damage when there isn't. Same economics as the global level's DP."""
    if f["channels_free"] < 1 or f["stock"] < 1 or f["ready"] < 1 or f["engaging"]:
        return 0
    if f["friendly"] and not f["last"]:
        return 0                                       # own aircraft near the intercept point: weapons hold
    # (track firmness is the battle manager's call, priced by its Wait option -- the fire unit doesn't second-guess it)
    if f["debris"] and f["later_open"] and not f["last"]:
        return 0                                       # a later intercept drops the debris on open ground
    n = 1
    while n < 3 and (1 - pk) ** n * pk * v_after > cost:   # another round still buys more than it costs
        n += 1
    return n
