"""Mobile air units: fighters and attack helicopters that are scrambled or vectored onto a threat.

A unit's engagement window is computed against the threat's path: the earliest time t at which
(scramble delay if on the ground) + (distance to the threat at t - weapon reach) / speed <= t - now. That is what makes
a coverage gap answerable: where no ground battery reaches, a unit can still meet the threat along its path.
"""
from dataclasses import dataclass, field

import numpy as np

from sim import AIR_KINDS, AIR_UNITS, EFFECTORS, effective_pk, haversine

REARM_S = 20 * 60      # turnaround at base after landing


@dataclass
class Unit:
    id: str
    kind: str
    name: str
    base: tuple
    station: tuple | None
    lat: float
    lon: float
    state: str                    # ground | cap | tasked | rtb | rearm
    weapons: dict = field(default_factory=dict)
    sortie_end: float = float("inf")
    task: str | None = None       # threat id
    goal: tuple | None = None     # (lat, lon) it flies toward
    ready_at: float = 0.0

    @property
    def spec(self):
        return AIR_KINDS[self.kind]

    def available(self):
        return self.state in ("ground", "cap") and sum(self.weapons.values()) > 0

    def weapon_for(self, th, cfg):
        """Cheapest loaded weapon that can stop this threat class, else None."""
        usable = [w for w, n in self.weapons.items() if n > 0 and th.cls in EFFECTORS[w]["pk"]]
        return min(usable, key=lambda w: EFFECTORS[w]["cost"] / max(effective_pk(w, th, cfg), 1e-3), default=None)

    def reach(self, th, t, cfg, n=320):
        """Earliest engagement (time, lat, lon, alt, weapon, pk, flight seconds) on th from now, or None."""
        w = self.weapon_for(th, cfg)
        if w is None or t >= th.t0 + th.T:
            return None
        ts = np.linspace(max(t, th.t0), th.t0 + th.T - 1, n)
        lat, lon, alt = th.pos(ts)
        speed_threat = np.r_[np.hypot(np.diff(lat), np.diff(lon)) * 111 / np.maximum(np.diff(ts), 1e-6), 0]
        delay = self.spec["scramble"] if self.state == "ground" else 0.0
        need = delay + np.maximum(haversine(self.lat, self.lon, lat, lon) - EFFECTORS[w]["arange"], 0) / self.spec["speed"]
        ok = (need <= ts - t) & (alt <= EFFECTORS[w]["hmax"]) & (speed_threat <= self.spec["max_threat_speed"]) \
            & (ts - t <= min(self.sortie_end, t + self.spec["sortie"]) - t)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            return None
        i = idx[0]
        return float(ts[i]), float(lat[i]), float(lon[i]), float(alt[i]), w, effective_pk(w, th, cfg), float(ts[i] - t)

    def move(self, t, dt):
        if self.state == "rearm" and t >= self.ready_at:
            self.weapons = dict(self.spec["weapons"])
            self.state = "ground" if self.station is None else "rtb"
            self.goal = self.station
        if self.state in ("tasked", "rtb") and self.goal:
            d = float(haversine(self.lat, self.lon, *self.goal))
            step = self.spec["speed"] * dt
            if d <= step:
                self.lat, self.lon = self.goal
                if self.state == "rtb":
                    if self.goal == self.station and sum(self.weapons.values()) > 0:
                        self.state = "cap"
                    else:
                        self.state, self.ready_at = "rearm", t + REARM_S
            else:
                f = step / d
                self.lat += (self.goal[0] - self.lat) * f
                self.lon += (self.goal[1] - self.lon) * f

    def release(self, t):
        """After an engagement (or a cancelled task): back to the station if armed, else home to rearm."""
        self.task = None
        armed = sum(self.weapons.values()) > 0
        self.state = "rtb"
        self.goal = self.station if (armed and self.station) else self.base
        if armed and self.station is None:
            self.goal = self.base


def fleet():
    out = []
    for uid, kind, name, blat, blon, station in AIR_UNITS:
        lat, lon = station if station else (blat, blon)
        out.append(Unit(uid, kind, name, (blat, blon), station, lat, lon, "cap" if station else "ground",
                        dict(AIR_KINDS[kind]["weapons"])))
    return out
