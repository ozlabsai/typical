"""Air-defence FinOps simulator: raid kinematics, engagement windows, plan builder, three policies.

Everything here is notional / open-source (see .context/airdef/PLAN.md). Battery positions are
NOT real deployments. Units: km, seconds, USD millions.
"""
import hashlib
import math
import random
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- catalog
# cls: B = ballistic (MRBM/SRBM), R = rocket, C = cruise missile, D = drone
THREATS = {
    #            label                         cls  cost $M
    "emad":     ("Emad MRBM",                  "B", 0.25),
    "ghadr":    ("Ghadr MRBM",                 "B", 5.0),
    "kheibar":  ("Kheibar Shekan MRBM",        "B", 3.0),   # ponytail: cost not public, guess
    "fattah":   ("Fattah-1 manoeuvring MRBM",  "B", 5.0),   # ponytail: cost not public, guess
    "palestine2": ("Palestine-2 MRBM (Houthi)", "B", 3.0),
    "fateh110": ("Fateh-110 SRBM",             "B", 0.5),
    "grad":     ("Grad 122 mm rocket",         "R", 0.002),
    "fajr5":    ("Fajr-5 heavy rocket",        "R", 0.02),
    "burkan":   ("Burkan heavy short-range rocket", "R", 0.01),
    "paveh":    ("Paveh cruise missile",       "C", 0.5),
    "shahed":   ("Shahed-136 drone",           "D", 0.05),
    "uav_lb":   ("Hezbollah attack drone",     "D", 0.02),
}
# Warhead weight relative to a ~500 kg MRBM warhead: scales the consequence of a leak.
WARHEAD = {"B": 1.0, "R": 0.1, "fateh110": 0.6, "grad": 0.02, "fajr5": 0.2, "burkan": 0.3, "C": 0.2, "D": 0.05}
MANOEUVRING = {"fattah"}   # Pk penalty + late-converging impact prediction

# Effectors. rmin/rmax km ground range from battery, hmin/hmax km target altitude, v km/s mean flyout,
# pk vs class, cost $M per shot. ponytail: notional envelopes from public ranges; tune live via config.pk_scale.
EFFECTORS = {
    "arrow3": dict(name="Arrow 3", rmin=50, rmax=900, hmin=100, hmax=600, v=3.0, cost=2.5,
                   pk={"B": 0.85}, layer=4),
    "thaad":  dict(name="THAAD", rmin=10, rmax=200, hmin=40, hmax=150, v=2.8, cost=12.7,
                   pk={"B": 0.85}, layer=3, allied=True),
    "sm3":    dict(name="SM-3", rmin=50, rmax=700, hmin=100, hmax=600, v=3.0, cost=12.0,
                   pk={"B": 0.8}, layer=4, allied=True),
    "arrow2": dict(name="Arrow 2", rmin=5, rmax=100, hmin=10, hmax=60, v=2.5, cost=1.5,
                   pk={"B": 0.85}, layer=3),
    "sm6":    dict(name="SM-6", rmin=5, rmax=300, hmin=0, hmax=30, v=1.2, cost=4.0,
                   pk={"B": 0.6, "C": 0.85, "D": 0.85}, layer=2, allied=True),
    "dsling": dict(name="David's Sling", rmin=5, rmax=200, hmin=0.05, hmax=30, v=1.5, cost=1.0,
                   pk={"B": 0.65, "R": 0.85, "C": 0.85, "D": 0.8}, layer=2),
    "ironDome": dict(name="Iron Dome", rmin=4, rmax=70, hmin=0, hmax=10, v=0.7, cost=0.05,
                   pk={"B": 0.3, "R": 0.9, "C": 0.85, "D": 0.85}, layer=1),
    "ironBeam": dict(name="Iron Beam laser", rmin=0, rmax=10, hmin=0, hmax=5, v=0, cost=0.000003,
                   pk={"R": 0.6, "D": 0.75}, layer=0, dwell=6, unlimited=True),
    "aim9x":  dict(name="F-15/F-35 AIM-9X", rmin=0, rmax=300, hmin=0, hmax=12, v=0.3, cost=0.42,
                   pk={"C": 0.85, "D": 0.9}, layer=2),
    "apkws":  dict(name="F-16 laser-guided rocket (APKWS)", rmin=0, rmax=250, hmin=0, hmax=5, v=0.25,
                   cost=0.035, pk={"D": 0.8}, layer=1),
    "ew":     dict(name="GPS jamming (EW)", rmin=0, rmax=40, hmin=0, hmax=5, v=0, cost=0.0,
                   pk={"D": 0.35}, layer=0, dwell=20, unlimited=True),
}

# Defended sites (public places, approximate centres). tier: strategic | city | infra
SITES = [
    ("tel_aviv", "Tel Aviv metro", 32.08, 34.80, 10, "city"),
    ("jerusalem", "Jerusalem", 31.77, 35.21, 7, "city"),
    ("haifa", "Haifa", 32.79, 34.99, 6, "city"),
    ("haifa_bay", "Haifa Bay refineries", 32.80, 35.06, 3, "infra"),
    ("beer_sheva", "Be'er Sheva", 31.25, 34.79, 6, "city"),
    ("ashdod", "Ashdod port", 31.81, 34.65, 5, "infra"),
    ("ashkelon", "Ashkelon", 31.67, 34.57, 4, "city"),
    ("ben_gurion", "Ben Gurion airport", 32.00, 34.87, 3, "infra"),
    ("nevatim", "Nevatim airbase", 31.21, 35.01, 4, "strategic"),
    ("ramat_david", "Ramat David airbase", 32.67, 35.18, 3, "strategic"),
    ("dimona", "Dimona research centre", 31.00, 35.14, 3, "strategic"),
    ("eilat", "Eilat", 29.56, 34.95, 5, "city"),
    ("kiryat_shmona", "Kiryat Shmona", 33.21, 35.57, 3, "city"),
    ("nahariya", "Nahariya", 33.01, 35.10, 3, "city"),
    ("tiberias", "Tiberias", 32.79, 35.53, 3, "city"),
    ("sderot", "Sderot", 31.52, 34.60, 2, "city"),
    ("hadera_power", "Hadera power station", 32.47, 34.88, 2, "infra"),
    ("netanya", "Netanya", 32.33, 34.86, 4, "city"),
]
SITE = {s[0]: s for s in SITES}
THREAT_HE = {"emad": "עמאד", "ghadr": "קדר", "kheibar": "ח׳ייבר שכן", "fattah": "פתאח-1 מתמרן", "palestine2": "פלסטין-2 (חות׳ים)",
             "fateh110": "פאתח-110", "grad": "גראד 122 מ״מ", "fajr5": "פג׳ר-5", "burkan": "בורקאן", "paveh": "טיל שיוט פאווה",
             "shahed": "שאהד-136", "uav_lb": "כטב״ם תקיפה של חזבאללה"}
EFFECTOR_HE = {"arrow3": "חץ 3", "thaad": "THAAD", "sm3": "SM-3", "arrow2": "חץ 2", "sm6": "SM-6", "dsling": "קלע דוד",
               "ironDome": "כיפת ברזל", "ironBeam": "מגן אור", "aim9x": "AIM-9X", "apkws": "APKWS", "ew": "שיבוש GPS"}
SITE_HE = {"tel_aviv": "גוש דן", "jerusalem": "ירושלים", "haifa": "חיפה", "haifa_bay": "בתי הזיקוק במפרץ חיפה",
           "beer_sheva": "באר שבע", "ashdod": "נמל אשדוד", "ashkelon": "אשקלון", "ben_gurion": "נמל התעופה בן גוריון",
           "nevatim": "בסיס נבטים", "ramat_david": "בסיס רמת דוד", "dimona": "קריית המחקר בדימונה", "eilat": "אילת",
           "kiryat_shmona": "קריית שמונה", "nahariya": "נהריה", "tiberias": "טבריה", "sderot": "שדרות",
           "hadera_power": "תחנת הכוח בחדרה", "netanya": "נתניה"}

# Notional batteries -- NOT real deployments.
BATTERIES = [
    ("A3-C", "arrow3", 31.90, 34.90, 10), ("A3-S", "arrow3", 30.90, 34.80, 8),
    ("A2-C", "arrow2", 32.35, 35.00, 12),
    ("DS-C", "dsling", 32.10, 35.00, 16), ("DS-N", "dsling", 32.75, 35.20, 16),
    ("ID-N", "ironDome", 33.00, 35.30, 60), ("ID-H", "ironDome", 32.75, 35.05, 60),
    ("ID-TA", "ironDome", 32.05, 34.85, 60), ("ID-AS", "ironDome", 31.75, 34.70, 60),
    ("ID-BS", "ironDome", 31.30, 34.85, 60), ("ID-E", "ironDome", 29.62, 34.95, 40),
    ("ID-G", "ironDome", 31.50, 34.55, 60), ("ID-J", "ironDome", 31.75, 35.15, 40),
    ("ID-NV", "ironDome", 31.15, 35.00, 40),
    ("IB-N", "ironBeam", 33.05, 35.40, 0), ("IB-H", "ironBeam", 32.78, 35.05, 0),
    ("EW-N", "ew", 32.95, 35.40, 0), ("EW-S", "ew", 31.20, 34.90, 0),
    ("CAP-E", "aim9x", 31.90, 36.80, 16), ("CAP-E2", "apkws", 31.90, 36.80, 28),
    ("CAP-S", "aim9x", 29.50, 35.20, 8), ("CAP-S2", "apkws", 29.50, 35.20, 14),
    ("THAAD", "thaad", 31.25, 34.95, 48),
    ("DDG", "sm3", 32.60, 33.90, 8), ("DDG6", "sm6", 32.60, 33.90, 24),
]

ORIGINS = {
    "iran_w": (34.3, 47.1, 1.2), "iran_nw": (38.0, 46.3, 1.0), "iran_c": (32.6, 51.6, 1.0),
    "iraq": (33.3, 43.5, 0.8), "yemen": (15.4, 44.2, 0.6),
    "lebanon_s": (33.28, 35.40, 0.12), "lebanon_bq": (33.85, 36.00, 0.15), "gaza": (31.45, 34.40, 0.08),
}

TIERS = ("strategic", "city", "infra")
DEFAULT_CONFIG = {
    "tier_value": {"strategic": 1000.0, "city": 500.0, "infra": 300.0},  # $M-equivalent of a 500 kg hit
    "priority_sites": ["nevatim", "dimona", "tel_aviv"],
    "reserve": 1,              # 0 healthy, 1 conserve, 2 critical
    "allied": True,            # THAAD + US Navy available
    "iron_beam_weather": "clear",  # clear | cloudy (laser Pk x0.4)
    "pk_scale": 1.0,
    "orders": [],              # commander's orders, prepended to the rules (rule grammar: see engine.RULE_RE)
    "conf_threshold": 0.42,    # escalate to the operator when Typical's top action is below this...
    "abstain_threshold": 0.9,  # ...or when its ∅ ("none of these") mass is above this
    # damage model, all user-configurable: value of a site hit = site_value[sid] (else tier_value[tier])
    #   x warhead weight (per threat kind, else per class) x priority_multiplier if it's a priority site
    "site_value": {},          # sid -> $M override
    "warhead": dict(WARHEAD),
    "priority_multiplier": 3.0,
}

EARTH_KM = 6371.0
LOOK_S = 8.0       # shoot-look-shoot assessment time
REACT_S = 4.0      # launch reaction time


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def u01(*key):
    """Common random number shared by every policy: same (seed, threat, shot) -> same draw."""
    h = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


# ---------------------------------------------------------------- threats
@dataclass
class Threat:
    id: str
    kind: str
    origin: tuple
    impact: tuple          # true impact lat, lon
    t0: float              # launch sim time
    T: float               # flight time
    apogee: float          # km (B/R) or cruise altitude
    waypoint: tuple | None = None
    ipp_offset: tuple = (0.0, 0.0)   # unit offset direction of the prediction error, km
    ipp_sigma0: float = 20.0
    samples: dict = field(default_factory=dict)   # battery id -> (t array, flyout, lat, lon, alt) where feasible
    first_launch: float = float("inf")             # earliest launch any battery could make (stock ignored)

    @property
    def cls(self):
        return THREATS[self.kind][1]

    def pos(self, t):
        """(lat, lon, alt_km) at sim time t (vectorised over t). ponytail: lat/lon lerp, fine at Levant scale."""
        s = np.clip((np.asarray(t, float) - self.t0) / self.T, 0, 1)
        if self.waypoint is not None:   # cruise: two legs, via waypoint at s=0.5
            a = np.where(s < 0.5, s * 2, s * 2 - 1)[..., None]
            p0 = np.where((s < 0.5)[..., None], self.origin, self.waypoint)
            p1 = np.where((s < 0.5)[..., None], self.waypoint, self.impact)
            ll = p0 + (np.array(p1) - p0) * a
        else:
            ll = np.array(self.origin) + (np.array(self.impact) - np.array(self.origin)) * s[..., None]
        if self.cls in "BR":
            alt = 4 * self.apogee * s * (1 - s)
        else:   # climb to cruise altitude, dive at the end
            alt = self.apogee * np.minimum(1, np.minimum(s, 1 - s) * 40)
        return ll[..., 0], ll[..., 1], alt

    def predicted_impact(self, t):
        """Impact-point prediction: converges onto the true impact as the threat closes."""
        s = min(max((t - self.t0) / self.T, 0), 1)
        late = 0.15 if self.kind in MANOEUVRING else 0.0
        sigma = self.ipp_sigma0 * max(1 - s, late) + 0.3
        k = sigma / 111.0
        return (self.impact[0] + self.ipp_offset[0] * k, self.impact[1] + self.ipp_offset[1] * k, sigma)


def _flight(kind, rng_km):
    cls = THREATS[kind][1]
    if cls in "BR":
        H = 0.25 * rng_km
        boost = 60 if rng_km > 500 else (20 if cls == "B" else 0)
        return 2 * math.sqrt(2 * H * 1000 / 9.81) + boost, H
    speed = {"paveh": 0.25, "shahed": 0.052, "uav_lb": 0.05}[kind]
    return rng_km / speed, {"paveh": 0.05, "shahed": 0.3, "uav_lb": 0.2}[kind]


def site_hit(lat, lon):
    """Site id whose radius contains the point, else None."""
    for sid, _, slat, slon, r, _ in SITES:
        if haversine(lat, lon, slat, slon) <= r:
            return sid
    return None


def nearest_site(lat, lon):
    d = [(float(haversine(lat, lon, s[2], s[3])) - s[4], s[0]) for s in SITES]
    return min(d)


# ---------------------------------------------------------------- scenarios
SCENARIOS = {
    "northern_barrage": dict(
        title_he="חזית צפון: מטח חזבאללה", blurb_he="150 רקטות, 20 כטב״מים ו-4 פאתח-110 בתוך 6 דקות. רוב הרקטות נופלות בשטח פתוח.",
        title="Northern front: Hezbollah barrage",
        blurb="150 rockets, 20 attack drones and 4 Fateh-110 in 6 minutes. Most rockets fall in open ground.",
        waves=[("grad", "lebanon_s", ["kiryat_shmona", "nahariya", "tiberias", "haifa", "ramat_david"], 110, 14, 0, 360),
               ("fajr5", "lebanon_s", ["haifa", "haifa_bay", "ramat_david", "hadera_power"], 25, 6, 30, 300),
               ("burkan", "lebanon_s", ["kiryat_shmona", "nahariya"], 15, 4, 60, 300),
               ("uav_lb", "lebanon_s", ["kiryat_shmona", "ramat_david", "haifa_bay", "nahariya"], 20, 1.5, 0, 400),
               ("fateh110", "lebanon_bq", ["haifa_bay", "ramat_david", "haifa"], 4, 1.5, 200, 360)]),
    "april_mixed": dict(
        title_he="13 באפריל 2024: גל משולב מאיראן", blurb_he="כ-60 כטב״מי שאהד, 15 טילי שיוט פאווה ו-60 טילים בליסטיים, מתוזמנים להגיע יחד.",
        title="13 April 2024: mixed wave from Iran",
        blurb="~60 Shahed drones, 15 Paveh cruise missiles and 60 ballistic missiles timed to arrive together.",
        waves=[("shahed", "iran_w", ["nevatim", "ramat_david", "dimona", "jerusalem"], 45, 3, 0, 900),
               ("shahed", "iraq", ["nevatim", "eilat", "ramat_david"], 15, 3, 0, 900),
               ("paveh", "iraq", ["nevatim", "ramat_david", "haifa_bay"], 15, 1.5, 400, 900),
               ("emad", "iran_w", ["nevatim", "ramat_david", "tel_aviv"], 30, 2.5, 700, 900),
               ("ghadr", "iran_nw", ["nevatim", "ramat_david"], 30, 2.5, 700, 900)]),
    "october_saturation": dict(
        title_he="1 באוקטובר 2024: רוויה בליסטית", blurb_he="כ-120 טילים בליסטיים ב-12 דקות, כולל פתאח-1 מתמרן, לעבר בסיסי חיל האוויר וגוש דן.",
        title="1 October 2024: ballistic saturation",
        blurb="~120 ballistic missiles in 12 minutes, including manoeuvring Fattah-1, aimed at airbases and Tel Aviv.",
        waves=[("emad", "iran_w", ["nevatim", "tel_aviv", "ramat_david"], 40, 3, 0, 720),
               ("ghadr", "iran_nw", ["nevatim", "tel_aviv", "dimona"], 40, 3, 0, 720),
               ("kheibar", "iran_c", ["nevatim", "tel_aviv", "ben_gurion"], 25, 2.5, 0, 720),
               ("fattah", "iran_c", ["nevatim", "tel_aviv"], 15, 2, 0, 720)]),
    "june_attrition": dict(
        title_he="יוני 2025: לילה 9 במלחמת התשה", blurb_he="המלאים כבר נמוכים. 35 טילים ו-30 כטב״מים; כל חץ חשוב.",
        title="June 2025: night 9 of an attrition war",
        blurb="Magazines are already low. 35 missiles and 30 drones; every Arrow counts.",
        stock_scale={"arrow3": 0.3, "arrow2": 0.4, "thaad": 0.35, "dsling": 0.6},
        waves=[("kheibar", "iran_c", ["haifa_bay", "tel_aviv", "beer_sheva", "nevatim"], 15, 2.5, 0, 300),
               ("emad", "iran_w", ["tel_aviv", "haifa", "beer_sheva", "jerusalem"], 20, 3, 600, 900),
               ("shahed", "iran_w", ["beer_sheva", "haifa_bay", "tiberias"], 30, 3, 0, 1200)]),
    "houthi_eilat": dict(
        title_he="ים סוף: ירי חות׳י ארוך טווח", blurb_he="3 טילי פלסטין-2 ו-8 כטב״מים לעבר אילת והמרכז. מטח קטן, כל החלטה גלויה.",
        title="Red Sea: Houthi long-range",
        blurb="3 Palestine-2 missiles and 8 drones towards Eilat and the centre. Small raid, every decision visible.",
        waves=[("palestine2", "yemen", ["tel_aviv", "eilat", "ben_gurion"], 3, 3, 0, 400),
               ("shahed", "yemen", ["eilat"], 8, 2, 0, 600)]),
}


def build_raid(name, seed=7):
    """Arrival-synchronised raid: each wave = (kind, origin, target sites, count, scatter km, t_first, t_last arrival)."""
    rng = random.Random(seed)
    sc = SCENARIOS[name]
    threats = []
    for kind, origin, targets, n, scatter_km, a0, a1 in sc["waves"]:
        olat, olon, ojit = ORIGINS[origin]
        for _ in range(n):
            tgt = SITE[rng.choice(targets)]
            # rockets are inaccurate: many land in open ground; guided threats mostly hit near the aim
            sc_km = scatter_km * (rng.random() * 2.5 if THREATS[kind][1] == "R" else abs(rng.gauss(0, 1)))
            ang = rng.random() * 2 * math.pi
            impact = (tgt[2] + sc_km / 111 * math.sin(ang), tgt[3] + sc_km / 111 * math.cos(ang) / 0.85)
            o = (olat + rng.uniform(-ojit, ojit), olon + rng.uniform(-ojit, ojit))
            rng_km = float(haversine(*o, *impact))
            wp = None
            if THREATS[kind][1] == "C":   # cruise missiles dog-leg through the south
                wp = ((o[0] + impact[0]) / 2 - rng.uniform(0.5, 1.5), (o[1] + impact[1]) / 2)
                rng_km *= 1.08
            T, apo = _flight(kind, rng_km)
            arrive = rng.uniform(a0, a1)
            a = rng.random() * 2 * math.pi
            th = Threat(id=f"T{len(threats) + 1:03d}", kind=kind, origin=o, impact=impact, t0=arrive - T, T=T,
                        apogee=apo, waypoint=wp, ipp_offset=(math.sin(a), math.cos(a)),
                        ipp_sigma0=min(25.0, 3 + rng_km * 0.02))
            threats.append(th)
    t_start = min(t.t0 for t in threats)
    for t in threats:   # sim clock starts at the first launch
        t.t0 -= t_start
    return threats


# ---------------------------------------------------------------- engagement geometry
def precompute_windows(th: Threat, batteries):
    n = int(min(max(300, th.T / 4), 3000))
    ts = np.linspace(th.t0, th.t0 + th.T, n)
    lat, lon, alt = th.pos(ts)
    for bid, eff, blat, blon, _ in batteries:
        e = EFFECTORS[eff]
        if th.cls not in e["pk"]:
            continue
        d = haversine(blat, blon, lat, lon)
        ok = (d >= e["rmin"]) & (d <= e["rmax"]) & (alt >= e["hmin"]) & (alt <= e["hmax"])
        ok[-1] = False   # can't intercept at impact
        if not ok.any():
            continue
        fly = e.get("dwell") or np.sqrt(d ** 2 + alt ** 2) / e["v"]
        th.samples[bid] = (ts[ok], np.broadcast_to(fly, ts.shape)[ok], lat[ok], lon[ok], alt[ok])
        th.first_launch = min(th.first_launch, float((th.samples[bid][0] - th.samples[bid][1]).min()))


def first_shot(th, bid, after):
    """Earliest (launch, intercept, lat, lon, alt) for battery bid with launch >= after, or None."""
    s = th.samples.get(bid)
    if s is None:
        return None
    ts, fly, lat, lon, alt = s
    idx = np.nonzero(ts - fly >= after)[0]
    if not len(idx):
        return None
    i = idx[0]
    return float(ts[i] - fly[i]), float(ts[i]), float(lat[i]), float(lon[i]), float(alt[i])


# ---------------------------------------------------------------- plans
@dataclass
class Plan:
    key: str
    label: str
    shots: list            # [(battery, effector, launch, intercept, lat, lon, alt, pk, conditional)]
    pk: float
    exp_cost: float        # expected $M (SLS backup weighted by miss prob)
    exp_rounds: dict       # effector -> expected rounds
    tags: tuple = ()       # "protection" / "value": which doctrine rule points at this plan


def pk_word(p):
    return ("almost certainly stops it" if p >= 0.95 else "very likely stops it" if p >= 0.85 else
            "likely stops it" if p >= 0.7 else "may stop it" if p >= 0.5 else "unlikely to stop it")


def cost_word(c):
    return ("nearly free" if c < 0.005 else "cheap" if c < 0.2 else "costly" if c < 2 else
            "very costly" if c < 8 else "extremely costly")


def money(c):
    return "under $1k" if c < 0.001 else f"${c * 1000:,.0f}k" if c < 1 else f"${c:,.1f}M"


def effective_pk(eff, th, cfg):
    p = EFFECTORS[eff]["pk"].get(th.cls, 0) * cfg["pk_scale"]
    if th.kind in MANOEUVRING:
        p *= 0.7
    if eff == "ironBeam" and cfg["iron_beam_weather"] == "cloudy":
        p *= 0.4
    return min(p, 0.99)


def build_plans(th, now, stock, cfg, max_plans=7, price=None, value_price=None):
    """Feasible engagement plans for one threat. stock: battery -> rounds available (inf = unlimited)."""
    bat = {b[0]: b for b in BATTERIES}
    best = {}   # effector -> (battery, shot) earliest feasible battery with stock
    for bid in th.samples:
        eff = bat[bid][1]
        if EFFECTORS[eff].get("allied") and not cfg["allied"]:
            continue
        if stock.get(bid, 0) < 1:
            continue
        s = first_shot(th, bid, now + REACT_S)
        if s and (eff not in best or s[1] < best[eff][1][1]):
            best[eff] = (bid, s)
    plans = [Plan("hold", "Hold fire and let it fall", [], 0.0, 0.0, {})]

    def shot(eff, bid, s, cond=False):
        return (bid, eff, s[0], s[1], s[2], s[3], s[4], effective_pk(eff, th, cfg), cond)

    for eff, (bid, s) in best.items():
        e, p = EFFECTORS[eff], effective_pk(eff, th, cfg)
        plans.append(Plan(f"{eff}x1", f"One {e['name']} interceptor", [shot(eff, bid, s)], p, e["cost"], {eff: 1}))
        if stock[bid] >= 2:
            p2 = 1 - (1 - p) ** 2
            plans.append(Plan(f"{eff}x2", f"Two {e['name']} interceptors fired together", [shot(eff, bid, s), shot(eff, bid, s)],
                              p2, 2 * e["cost"], {eff: 2}))
    # shoot-look-shoot: first effector, then (if it misses) a backup that can still reach it
    for e1, (b1, s1) in best.items():
        for e2, (b2, _) in best.items():
            if EFFECTORS[e2]["layer"] > EFFECTORS[e1]["layer"]:
                continue
            s2 = first_shot(th, b2, s1[1] + LOOK_S)
            if s2 is None or (b1 == b2 and stock[b1] < 2):
                continue
            p1, p2 = effective_pk(e1, th, cfg), effective_pk(e2, th, cfg)
            n1, n2 = EFFECTORS[e1]["name"], EFFECTORS[e2]["name"]
            lab = (f"{n1}, then a second {n1} only if the first misses" if e1 == e2
                   else f"{n1}, then {n2} only if the first misses")
            plans.append(Plan(f"{e1}>{e2}", lab, [shot(e1, b1, s1), shot(e2, b2, s2, True)], 1 - (1 - p1) * (1 - p2),
                              EFFECTORS[e1]["cost"] + (1 - p1) * EFFECTORS[e2]["cost"],
                              {e1: 1, e2: (1 - p1)} if e1 != e2 else {e1: 2 - p1}))
    # Tag the two plans doctrine refers to. "Cheapest" = effective cost (price adds the magazine's shadow value).
    hold, rest = plans[0], plans[1:]
    price = price or (lambda p: p.exp_cost)
    if not rest:
        return [hold]
    sure = [p for p in rest if p.pk >= 0.95]
    likely = [p for p in rest if p.pk >= 0.85]
    best_prot = min(sure, key=price) if sure else max(rest, key=lambda p: (round(p.pk, 2), -price(p)))
    # "cheapest reliable" = the solver's cost-aware pick (leak risk priced by what's at stake), else Pk >= 0.85
    best_val = min(rest, key=value_price) if value_price else (min(likely, key=price) if likely else best_prot)
    # ponytail: shortlist one plan per (pk band, cost word) so K <= max_plans; tagged plans always kept
    rest.sort(key=lambda p: (-round(p.pk, 1), price(p)))
    seen, short = set(), [best_prot] + ([best_val] if best_val is not best_prot else [])
    for p in short:
        seen.add((round(p.pk, 1), cost_word(p.exp_cost)))
    for p in rest:
        band = (round(p.pk, 1), cost_word(p.exp_cost))
        if band not in seen and len(short) < max_plans - 1:
            seen.add(band)
            short.append(p)
    short.sort(key=lambda p: p.exp_cost)
    for p in short:
        p.tags = tuple(t for t, q in (("protection", best_prot), ("value", best_val)) if q is p)
        tag = f", {' and '.join('the best ' + t for t in p.tags)}" if p.tags else ""
        p.label = f"{p.label} ({money(p.exp_cost)}, {pk_word(p.pk)}){tag}"
    return [hold] + short


# ---------------------------------------------------------------- consequence + optimizer
def site_value(sid, kind, cfg):
    """$M-equivalent consequence of this threat landing inside site sid."""
    wh = cfg["warhead"]
    w = wh.get(kind, wh[THREATS[kind][1]])
    base = cfg["site_value"].get(sid, cfg["tier_value"][SITE[sid][5]])
    return base * w * (cfg["priority_multiplier"] if sid in cfg["priority_sites"] else 1)


def consequence(th, now, cfg):
    """(expected damage if it leaks $M, predicted site id or None, confidence word, distance-to-site km)."""
    plat, plon, sigma = th.predicted_impact(now)
    dist, sid = nearest_site(plat, plon)
    # ponytail: P(hit) from a 1-D normal on the distance to the site edge; good enough for a demo
    p_hit = 0.5 * math.erfc(max(dist, -50) / (sigma * math.sqrt(2)))
    val = site_value(sid, th.kind, cfg)
    conf = "high" if sigma < 2 else "medium" if sigma < 8 else "low"
    return p_hit * val, sid, p_hit, conf, dist, sigma


def shadow_price(eff, stock_left, stock_init, cfg):
    """Value of keeping one round for later waves, $M. Scarcer + more conservative posture -> pricier."""
    if EFFECTORS[eff].get("unlimited"):
        return 0.0
    scarcity = stock_init / max(stock_left, 0.5)
    return EFFECTORS[eff]["cost"] * [0.0, 1.0, 4.0][cfg["reserve"]] * scarcity


def plan_score(plan, damage, type_left, type_init, cfg):
    """Expected total cost of a plan, $M: rounds spent + shadow price of the magazine + expected leak damage."""
    mag = sum(n * shadow_price(e, type_left.get(e, 0), type_init.get(e, 1), cfg) for e, n in plan.exp_rounds.items())
    return plan.exp_cost + mag + (1 - plan.pk) * damage


def naive_pick(plans):
    """Baseline: engage everything with the most reliable plan, cost be damned."""
    return max(plans[1:], key=lambda p: (round(p.pk, 2), -p.exp_cost)) if len(plans) > 1 else plans[0]
