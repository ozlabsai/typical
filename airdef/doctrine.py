"""Rules of engagement as data: first-match semantics + English/Hebrew order templates.

One source of truth for the live engine (doctrine marker, state text) and dataset.py (training labels).
rule = (cond, value, action): cond in cls | site | open | prio | secondary; ("mod", site, None) = site counts as priority.
"""
import re

from sim import SITE_HE, SITES

ACTIONS = {"hold": "Hold fire", "protection": "Fire the sure-kill plan", "value": "Fire the cheapest reliable plan"}
STANDING = [("open", None, "hold"), ("prio", None, "protection"), ("secondary", None, "value")]
CLASSES = ["drone", "rocket", "ballistic missile", "cruise missile"]


def applies(rule, sit, prio):
    cond, val, _ = rule
    return {"cls": lambda: sit["cls"] == val, "site": lambda: not sit["open"] and sit["site"] == val,
            "open": lambda: sit["open"], "prio": lambda: not sit["open"] and prio,
            "secondary": lambda: not sit["open"] and not prio}[cond]()


def decide(rules, sit, prio_sites):
    """First rule that applies decides; modifiers apply wherever they sit. None = the doctrine is silent."""
    prio = sit["site"] in prio_sites or any(r[0] == "mod" and r[1] == sit["site"] for r in rules)
    return next((r[2] for r in rules if r[0] != "mod" and applies(r, sit, prio)), None)


# (train templates, held-out templates). The first train template + first phrase = the canonical grammar.
EN_ACT = {"hold": (["hold fire", "do not engage", "let it fall"], ["hold your fire"]),
          "protection": (["fire the sure-kill plan", "use the sure-kill plan", "stop it at any cost"], ["make sure it is destroyed"]),
          "value": (["fire the cheapest reliable plan", "engage it cheaply", "use the cheapest reliable plan"],
                    ["spend as little as possible on it"])}
EN_T = {"cls": (["When the threat is a {c}, {a}.", "{C}s: {a}.", "For every {c}, {a}.", "If it is a {c}, {a}."],
                ["Any {c} tonight: {a}.", "{A} on every {c}."]),
        "site": (["When the threat will hit {s}, {a}.", "Threats to {s}: {a}.", "For anything aimed at {s}, {a}."],
                 ["If {s} is the target, {a}.", "{s} targets: {a}."]),
        "open": (["When the threat will land in open ground, {a}.", "Open-ground impacts: {a}."], ["If it falls in open ground, {a}."]),
        "prio": (["When the threat will hit a priority site, {a}.", "Priority sites: {a}."], ["Anything aimed at a priority site: {a}."]),
        "secondary": (["When the threat will hit a secondary site, {a}.", "Secondary sites: {a}."],
                      ["Anything aimed at a secondary site: {a}."]),
        "mod": (["Treat {s} as a priority site.", "{s} is a priority site tonight."], ["Add {s} to the priority sites."])}
HE_CLS = {"drone": "כטב״ם", "rocket": "רקטה", "ballistic missile": "טיל בליסטי", "cruise missile": "טיל שיוט"}
HE_ACT = {"hold": (["להחזיק אש", "לא ליירט"], ["לתת לו ליפול"]),
          "protection": (["לירות את תוכנית ההשמדה הוודאית", "ליירט בכל מחיר"], ["להבטיח יירוט ודאי"]),
          "value": (["לירות את התוכנית הזולה והאמינה", "ליירט בזול"], ["לחסוך במיירטים יקרים"])}
HE_T = {"cls": (["כאשר האיום הוא {c}: {a}.", "על {c} — {a}."], ["כל {c} הלילה: {a}."]),
        "site": (["איומים לעבר {s}: {a}.", "כאשר האיום יפגע ב{s}: {a}."], ["אם המטרה היא {s}, {a}."]),
        "open": (["כאשר האיום ייפול בשטח פתוח: {a}."], ["נפילה בשטח פתוח: {a}."]),
        "prio": (["כאשר האיום יפגע באתר בעדיפות: {a}."], ["אתרים בעדיפות: {a}."]),
        "secondary": (["כאשר האיום יפגע באתר משני: {a}."], ["אתרים משניים: {a}."]),
        "mod": (["{s} הוא אתר בעדיפות הלילה."], ["יש להוסיף את {s} לאתרים בעדיפות."])}
SITE_ID = {s[1]: s[0] for s in SITES}


def render(rule, style, held=False, rng=None):
    """style: grammar (canonical English) | en | he. held: draw from the held-out templates."""
    cond, val, act = rule
    if style == "grammar":
        t, a = EN_T[cond][0][0], EN_ACT[act][0][0] if act else ""
    else:
        T, A = (EN_T, EN_ACT) if style == "en" else (HE_T, HE_ACT)
        t = rng.choice(T[cond][held])
        a = rng.choice(A[act][held]) if act else ""
    c = (HE_CLS[val] if style == "he" else val) if cond == "cls" else ""
    s = (SITE_HE[SITE_ID[val]] if style == "he" else val) if cond in ("site", "mod") else ""
    out = t.format(c=c, C=c[:1].upper() + c[1:], s=s, a=a, A=a[:1].upper() + a[1:])
    return out[:1].upper() + out[1:]


def _parsers():
    """Every known template (both languages, train + held-out) compiled to a regex -> structured rule."""
    out = []
    for T, A, cls_names, site_names in ((EN_T, EN_ACT, {c: c for c in CLASSES}, {s[1]: s[1] for s in SITES}),
                                        (HE_T, HE_ACT, {v: k for k, v in HE_CLS.items()}, {SITE_HE[s[0]]: s[1] for s in SITES})):
        acts = {p.lower(): k for k, (tr, ho) in A.items() for p in tr + ho}
        cls_names = {k.lower(): v for k, v in cls_names.items()}
        sites = {k.lower(): v for k, v in site_names.items()}
        for cond, (tr, ho) in T.items():
            for t in tr + ho:
                rx = re.escape(t)
                for key, grp in (("c", "c"), ("C", "c"), ("s", "s"), ("a", "a"), ("A", "a")):
                    rx = rx.replace(re.escape("{" + key + "}"), f"(?P<{grp}>.+?)", 1)
                out.append((re.compile(f"^{rx}$", re.I), cond, acts, cls_names, sites))
    return out


PARSERS = _parsers()


def parse_order(text):
    """Order text -> rule tuple, or None when it is free text outside every template."""
    text = text.strip()
    for rx, cond, acts, cls_names, sites in PARSERS:
        m = rx.match(text)
        if not m:
            continue
        g = {k: v.strip().lower() for k, v in m.groupdict().items()}
        act = acts.get(g["a"]) if "a" in g else None
        if ("a" in g and act is None) or ("c" in g and g["c"] not in cls_names) or ("s" in g and g["s"] not in sites):
            continue
        val = cls_names[g["c"]] if "c" in g else sites[g["s"]] if "s" in g else None
        return (cond, val, act)
    return None


if __name__ == "__main__":   # ponytail: the self-check -- every template renders and parses back to the same rule
    import random
    rng = random.Random(0)
    for style in ("grammar", "en", "he"):
        for held in (0, 1):
            for cond in EN_T:
                for _ in range(20):
                    act = None if cond == "mod" else rng.choice(list(ACTIONS))
                    val = rng.choice(CLASSES) if cond == "cls" else rng.choice(SITES)[1] if cond in ("site", "mod") else None
                    r = (cond, val, act)
                    assert parse_order(render(r, style, held, rng)) == r, (style, held, render(r, style, held, rng))
    assert parse_order("Save the Arrows for Tel Aviv") is None
    print("doctrine: every template round-trips in both languages")
