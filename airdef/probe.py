"""Fast wording probe: capture decision cases from optimizer runs, score state/label variants with Typical.

    uv run --no-sync python airdef/probe.py [--n 120]
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
import engine   # noqa: E402
from sim import SCENARIOS   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--variants", default="")
a = ap.parse_args()

cases = []
orig = engine.World.decide


def capture(self, pol, th, plans, text, cons):
    plan, info = orig(self, pol, th, plans, text, cons)
    if pol.name == "optimizer":
        cases.append((self, th, plans, text, info["rule"]))
    return plan, info


engine.World.decide = capture
for sc in SCENARIOS:
    for cfg in ({}, {"priority_sites": ["haifa", "haifa_bay", "eilat", "tel_aviv", "nevatim"]}):
        engine.run_headless(sc, cfg={**engine.DEFAULT_CONFIG, **cfg})
random.seed(0)
by_rule = {}
for c in cases:
    by_rule.setdefault(c[4] == "hold" and "hold" or c[2][[p.key for p in c[2]].index(c[4])].tags, []).append(c)
per = max(1, a.n // len(by_rule))
sample = [c for v in by_rule.values() for c in random.sample(v, min(per, len(v)))]
print(f"{len(cases)} cases captured; probing {len(sample)}:", {str(k): len(v) for k, v in by_rule.items()})

from typical import Typical   # noqa: E402
m = Typical.from_pretrained("OzLabs/typical-small", device="auto", max_states=8)
Q = engine.Q_PLAN


def run(name, render):
    """render(case) -> (state, labels, answer_label)"""
    hit, miss, per = 0, Counter(), Counter()
    for c in sample:
        state, labels, ans = render(c)
        d = m.choice(state, Q, labels)
        d.pop("p_null")
        top = max(d, key=d.get)
        hit += top == ans
        per[(labels.index(ans), labels.index(top))] += 1
        if top != ans:
            miss[("hold" if ans.startswith("Hold") else "fire") + "->" + ("hold" if top.startswith("Hold") else "fire")] += 1
    print(f"{name:28s} follows {hit / len(sample):.0%}  confusion(ans,pred) {dict(sorted(per.items()))}")


def base(c):
    w, th, plans, text, rule = c
    return text, [p.label for p in plans], next(p.label for p in plans if p.key == rule)


def strip_reserve(text):
    return "\n".join(ln for ln in text.split("\n") if "tonight" not in ln or "order" in ln)


def tagged(plans, style):
    out = []
    for p in plans:
        core = p.label.split(", the best")[0]
        if p.key == "hold":
            out.append("Hold fire")
        elif style == "front" and p.tags:
            out.append(f"{' and '.join('the best ' + t for t in p.tags).capitalize()}: {core}")
        elif style == "front":
            out.append(core)
        else:
            out.append(p.label)
    return out


def v_front(c):
    w, th, plans, text, rule = c
    labels = tagged(plans, "front")
    return strip_reserve(text), labels, labels[[p.key for p in plans].index(rule)]


def explicit_rules(text):
    text = strip_reserve(text)
    text = text.replace("3. Otherwise, fire the best value.",
                        "3. When the threat will hit a secondary site, fire the best value.\n"
                        "4. When the impact prediction is uncertain, fire the best value.")
    return text


def v_explicit(c):
    w, th, plans, text, rule = c
    labels = tagged(plans, "front")
    return explicit_rules(text), labels, labels[[p.key for p in plans].index(rule)]


def v_merged(c):
    """Fact sentence pre-computes the site class: 'The threat will hit Haifa, a secondary site.'"""
    w, th, plans, text, rule = c
    lines = explicit_rules(text).split("\n")
    out = []
    for ln in lines:
        if ln.endswith("is a secondary site.") or ln.endswith("is a priority site."):
            continue
        if ln.startswith("The threat will hit "):
            site = ln[len("The threat will hit "):-1]
            prio = f"{site} is a priority site." in text
            ln = f"The threat will hit {site}, a {'priority' if prio else 'secondary'} site."
        out.append(ln)
    labels = tagged(plans, "front")
    return "\n".join(out), labels, labels[[p.key for p in plans].index(rule)]


ACTIONS = {"hold": "Hold fire", "protection": "Fire the best protection", "value": "Fire the best value"}


def answer_action(plans, rule):
    if rule == "hold":
        return "hold"
    tags = next(p.tags for p in plans if p.key == rule)
    return tags[0] if len(tags) == 1 else tags


def v_actions(render_text):
    def f(c):
        w, th, plans, text, rule = c
        ans = answer_action(plans, rule)
        labels = list(ACTIONS.values())
        # both tags on one plan: either action is correct; score against the one doctrine names
        if isinstance(ans, tuple):
            ans = "protection" if "priority site." in text and "a priority site" in render_text(text) else "value"
        return render_text(text), labels, ACTIONS[ans]
    return f


def v_actions_plan(render_text):
    def f(c):
        w, th, plans, text, rule = c
        st, labels, ans = v_actions(render_text)(c)
        conc = {}
        for p in plans:
            for t in p.tags:
                conc[t] = p.label.split(" (")[0]
        out = [lab if lab == "Hold fire" else f"{lab}: {conc[[k for k, v in ACTIONS.items() if v == lab][0]]}" for lab in labels]
        return st, out, out[labels.index(ans)]
    return f


EXACT_RULES = ("Rules of engagement.\n1. When the threat will land in open ground, hold fire.\n"
               "2. When the threat will hit a priority site, fire the best protection.\n"
               "3. When the threat will hit a secondary site, fire the best value.")


def v_exact(mags, intent_first=False):
    def f(c):
        w, th, plans, text, rule = c
        lines = text.split("Situation.\n")[1].split("\n")
        kind, where = lines[0], lines[1]
        site = where.split(" near ")[-1].rstrip(".") if "open ground" in where else where.split(" hit ")[1].split(";")[0].rstrip(".")
        prio = f"{site} is a priority site." in text
        if "open ground" in where:
            fact, ans = f"The threat will land in open ground near {site}.", "hold"
        else:
            unc = " The impact prediction is still uncertain." if "uncertain" in where else ""
            fact, ans = f"The threat will hit a {'priority' if prio else 'secondary'} site: {site}.{unc}", ("protection" if prio else "value")
        rest = [ln for ln in lines[2:] if ln.startswith("The threat arrives")]
        if mags:
            rest += [ln for ln in lines[2:] if "magazine" in ln]
        return EXACT_RULES + "\n\nSituation.\n" + "\n".join([kind, fact] + rest), list(ACTIONS.values()), ACTIONS[ans]
    return f


def relabel(f, labels, rules=None):
    def g(c):
        st, old, ans = f(c)
        if rules:
            st = st.replace(EXACT_RULES, rules)
        return st, labels, labels[old.index(ans)]
    return g


LEX = ["Hold fire, it falls in open ground", "Fire the sure-kill plan to save the priority site",
       "Fire the cheapest reliable plan at the secondary site"]
LEX_RULES = ("Rules of engagement.\n1. When the threat will land in open ground, hold fire.\n"
             "2. When the threat will hit a priority site, fire the sure-kill plan.\n"
             "3. When the threat will hit a secondary site, fire the cheapest reliable plan.")
LEX2 = ["Hold fire", "Fire the sure-kill plan", "Fire the cheapest reliable plan"]


VARIANTS = {"lex": relabel(v_exact(False), LEX, LEX_RULES), "lex2": relabel(v_exact(False), LEX2, LEX_RULES),
            "exact": v_exact(False), "exact+mags": v_exact(True), "actions": v_actions(explicit_rules), "actions+plan": v_actions_plan(explicit_rules),
            "actions-otherwise": v_actions(strip_reserve),"base": base, "front-tags": v_front, "explicit-rules": v_explicit, "merged-facts": v_merged}
for k, f in VARIANTS.items():
    if not a.variants or k in a.variants.split(","):
        run(k, f)
print("\nexample:\n" + v_exact(True)(sample[3])[0], "\n", v_exact(True)(sample[3])[1:])
