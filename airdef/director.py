"""Key moments: what is worth watching in a raid. The server slows down on each one and keeps a snapshot to replay it.

A moment is offered every time Typical's call on a threat changes (or a fire unit holds fire, or a threat hits a
site); it is featured -- slowed down, captioned, put on the timeline -- if it is worth stopping for and the screen is
free. Worth stopping for: aircraft sent, a wait for a cheaper layer (worth at least MIN_DELTA against firing now) and its payoff,
a salvo let fall, a unit holding fire, a hit.
"""
import copy
import time

from sim import EFFECTORS, consequence

BEAT_S = 7.0        # wall seconds a featured moment stays on screen
BEAT_SPEED = 1.0    # sim speed during a beat
GAP_S = 3.0         # wall seconds after a beat before the next one may start
SEEK_SPEED = 60.0   # Focus between moments (and "next"): fast-forward until the next featured moment
MIN_DELTA = 0.3     # $M of expected cost: waiting for a cheaper layer that saves less than this is not worth stopping for
SALVO_S = 120.0     # holds this close together (sim seconds) count as one "let fall" moment


class Director:
    def __init__(self):
        self.moments, self.snaps = [], []
        self.seen = dict(h={}, local={}, status={}, waited={})
        self.salvo = None
        self.stats = dict(fall=0, cheaper=0, kept=0.0, sorties=0)   # tonight's held-back tally, from Typical's own pricing
        self.beat, self.seek, self.free_at = None, False, 0.0
        self.slow = True   # Focus mode: a moment slows the clock and takes the screen; General mode only marks it

    # ------------------------------------------------------------ detection
    def scan(self, w):
        """-> candidate moments since the last tick, best first."""
        typ = w.policies["typical"]
        out = []
        for th in w.threats:
            s = typ.st[th.id]
            h, n0 = s["history"], self.seen["h"].get(th.id, 0)
            for i in range(n0, len(h)):
                m = self._call(w, th, s, h[i][1], h[i - 1][1] if i else None)
                if m:
                    out.append(m)
            self.seen["h"][th.id] = len(h)
            L = s.get("local")
            if L and L["t"] != self.seen["local"].get(th.id):
                self.seen["local"][th.id] = L["t"]
                f = L.get("facts") or {}
                if not L.get("fired") and (f.get("friendly") or f.get("debris")):
                    out.append(self._new(w, th, s, "holdfire", 2.0, unit=L["unit"], why="friendly" if f.get("friendly") else "debris",
                                         debris=f.get("debris")))
            if s["status"] != self.seen["status"].get(th.id):
                self.seen["status"][th.id] = s["status"]
                if s["status"] == "leaked":
                    shots = [[f[11], f[8]] for f in w.flights if f[0] == "typical" and f[2] == th.id]
                    out.append(self._new(w, th, s, "hit", 5.0, shots=shots))
        if self.salvo and "id" not in self.salvo and self.salvo["count"] >= 3 and self.salvo not in out:
            out.append(self.salvo)   # a salvo that arrived during another beat is offered again while it grows
        return sorted(out, key=lambda m: -m["score"])

    def _call(self, w, th, s, key, prev):
        info = s.get("info") or {}
        cost = {o[0]: o[2] for o in info.get("opts", [])}
        now = [c for k, c in cost.items() if k.startswith("now:")]
        delta = min(now) - cost[key] if key in cost and now else None   # what waiting saves against firing now
        waited = self.seen["waited"]
        if key == "hold" and prev != "hold":
            self.stats["fall"] += 1
        if key.startswith("air:"):
            self.stats["sorties"] += 1
        if key == "cheaper" and s.get("wait_eff"):
            if th.id not in waited:
                self.stats["cheaper"] += 1
            # what the wait keeps, in interceptor prices: the cheapest round that could fire now minus the one waited for
            price = [EFFECTORS[k.split(":")[1]]["cost"] for k in cost if k.startswith("now:")]
            waited[th.id] = (s["wait_eff"], max(min(price) - EFFECTORS[s["wait_eff"]]["cost"], 0.0) if price else 0.0)
        if key == "hold":
            if self.salvo and w.t - self.salvo["t_last"] < SALVO_S and self.salvo["cls"] == th.cls:
                self.salvo["count"] += 1          # one moment per salvo; a featured one updates on screen
                self.salvo["t_last"] = w.t
                return None
            self.salvo = self._new(w, th, s, "letfall", 1.5, count=1, t_last=w.t)
            return self.salvo
        if key.startswith("air:"):
            return self._new(w, th, s, "air", 3.0)
        if prev == "cheaper" and key.startswith("now:") and th.id in waited:
            eff, kept = waited.pop(th.id)
            if key == "now:" + eff:   # the wait paid off only if the interceptor it waited for is the one that fires
                self.stats["kept"] += kept
                return self._new(w, th, s, "payoff", 2.5, kept=kept)
            return None
        if key == "cheaper":
            return self._new(w, th, s, "cheaper", 1.0 + (delta or 0), delta=delta)
        return None

    @staticmethod
    def _new(w, th, s, kind, score, **kw):
        info = s.get("info") or {}
        _, sid, _, _, dist, sig = consequence(th, w.t, w.cfg)
        return {**dict(type=kind, score=score, t=w.t, tid=th.id, kind=th.kind, cls=th.cls, site=sid, open=bool(dist > sig),
                       key=s.get("choice"), info=dict(t=info.get("t"), opts=info.get("opts", []), dist=info.get("dist")), worth=True), **kw}

    # ------------------------------------------------------------ featuring
    def offer(self, w, cands):
        """Feature the best candidate if the screen is free; snapshot it so the timeline can replay it."""
        if self.beat or time.monotonic() < self.free_at:
            return None
        recent = [x["type"] for x in self.moments[-4:]]
        for m in cands:
            again = next((x for x in self.moments if x["type"] == m["type"] and x["tid"] == m["tid"]), None)
            if again:   # replaying: the same moment comes round again -- show it, do not add a duplicate marker
                if self.slow:
                    self.beat, self.seek = dict(id=again["id"], until=time.monotonic() + BEAT_S), False
                return again
            if m["type"] != "hit" and m["type"] in recent:   # variety: no type twice in four (a hit always shows)
                continue
            if m["type"] == "letfall" and m["count"] < 3:   # a salvo is a salvo from its third rocket on
                continue
            if m["worth"] and (m["type"] not in ("cheaper",) or (m.get("delta") or 0) >= MIN_DELTA):
                m["id"] = len(self.moments)
                self.moments.append(m)
                self.snaps.append(self.snapshot(w))
                if self.slow:
                    self.beat, self.seek = dict(id=m["id"], until=time.monotonic() + BEAT_S), False
                else:
                    self.free_at = time.monotonic() + BEAT_S + GAP_S   # same spacing on the timeline, no stop
                return m
        return None

    def tick(self):
        if self.beat and time.monotonic() > self.beat["until"]:
            self.end_beat()
        if self.slow and not self.beat and not self.seek and time.monotonic() > self.free_at:
            self.seek = True   # Focus is moment to moment: nothing between moments is worth watching at play speed

    def hold(self, dt):
        """Paused: the moment on screen keeps its remaining time."""
        if self.beat:
            self.beat["until"] += dt

    def end_beat(self):
        self.beat, self.free_at = None, time.monotonic() + GAP_S

    def speed(self, normal):
        return BEAT_SPEED if self.beat else SEEK_SPEED if self.seek else normal

    def next(self):
        self.beat, self.free_at, self.seek = None, 0.0, True

    # ------------------------------------------------------------ replay
    def snapshot(self, w):
        keep = {id(w.model): w.model} if w.model is not None else {}   # never copy the model
        return copy.deepcopy(w, keep), copy.deepcopy((self.seen, self.salvo, self.stats))

    def restore(self, i):
        """-> a fresh copy of the world at moment i (the snapshot itself stays reusable). Later moments stay on the
        timeline: they can still be jumped to, and coming round again they are shown, not duplicated."""
        snap, (seen, salvo, stats) = self.snaps[i]
        keep = {id(snap.model): snap.model} if snap.model is not None else {}
        w = copy.deepcopy(snap, keep)
        self.seen, self.salvo, self.stats = copy.deepcopy(seen), copy.deepcopy(salvo), copy.deepcopy(stats)
        self.beat, self.seek = dict(id=i, until=time.monotonic() + BEAT_S), False
        return w

    def public(self):
        return dict(moments=[{k: v for k, v in m.items() if k not in ("score", "worth", "t_last")} for m in self.moments],
                    beat=self.beat and dict(id=self.beat["id"], left=max(0.0, self.beat["until"] - time.monotonic())),
                    seek=self.seek, stats=self.stats)
