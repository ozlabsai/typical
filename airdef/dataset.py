"""Synthetic doctrine-decision data for fine-tuning Typical on air-defence rules of engagement.

    uv run --no-sync python airdef/dataset.py [--out .context/typically/jobs/airdef] [--train 24000]

Each row: randomised rules of engagement (standing doctrine + 0-3 commander's orders, half of them distractors)
+ a situation from real sim trajectories (or procedural long-tail) -> one of the three doctrine actions, or ∅
when no rule covers the situation (escalate). Labels come from a first-match evaluator, so they are never
ambiguous. Order paraphrases (English + Hebrew) are split into train / held-out templates; eval/ uses only held-out.
The state text is rendered by engine.render_state / situation_facts: byte-identical to what the live demo sends.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import engine
from doctrine import ACTIONS, STANDING, decide, render
from engine import CLASS_WORD, ORIGIN_NAME, Q_PLAN, render_state, short_name, situation_facts
from sim import EFFECTORS, SCENARIOS, SITE, SITES, THREATS

# ---------------------------------------------------------------- situations
def sim_situations(seeds=(1, 2, 3, 4)):
    """Every (threat, origin, predicted site, open/hit) the live engine actually rendered, across scenarios and seeds."""
    seen, orig = [], engine.World.state_text

    def grab(self, th):
        text, cons = orig(self, th)
        seen.append(dict(cls=CLASS_WORD[th.cls], name=short_name(th.kind), origin=ORIGIN_NAME[self.origin_key(th)],
                         site=SITE[cons["sid"]][1], open=cons["dist"] > cons["sigma"]))
        return text, cons
    engine.World.state_text = grab
    try:
        for sc in SCENARIOS:
            for seed in seeds:
                engine.run_headless(sc, seed=seed)
    finally:
        engine.World.state_text = orig
    return list({json.dumps(s, sort_keys=True): s for s in seen}.values())


PLAUSIBLE_ORIGIN = {"B": ["iran_w", "iran_nw", "iran_c", "yemen", "lebanon_bq"], "R": ["lebanon_s", "gaza"],
                    "C": ["iran_w", "iraq", "yemen"], "D": ["iran_w", "iraq", "yemen", "lebanon_s"]}


def procedural_situation(rng):
    kind = rng.choice(list(THREATS))
    cls = THREATS[kind][1]
    return dict(cls=CLASS_WORD[cls], name=short_name(kind), origin=ORIGIN_NAME[rng.choice(PLAUSIBLE_ORIGIN[cls])],
                site=rng.choice(SITES)[1], open=rng.random() < 0.3)


# ---------------------------------------------------------------- one row
def random_rule(rng, sit, relevant):
    """A rule whose condition does (relevant) or does not match this situation. Action is random."""
    act = rng.choice(list(ACTIONS))
    if relevant:
        cond = rng.choice(["cls", "cls", "site", "site"] + (["open"] if sit["open"] else ["prio", "secondary"]))
        val = sit["cls"] if cond == "cls" else sit["site"] if cond == "site" else None
        return (cond, val, act)
    cond = rng.choice(["cls", "cls", "site", "site", "open"])
    if cond == "cls":
        return ("cls", rng.choice([c for c in CLASS_WORD.values() if c != sit["cls"]]), act)
    if cond == "site":
        return ("site", rng.choice([s[1] for s in SITES if s[1] != sit["site"]]), act)
    return ("open", None, act) if not sit["open"] else ("site", rng.choice([s[1] for s in SITES if s[1] != sit["site"]]), act)


DISTRACTORS = [lambda r: f"The threat arrives in about {r.randint(1, 14)} minutes.",
               lambda r: f"The {EFFECTORS[r.choice(['arrow3', 'arrow2', 'thaad', 'dsling', 'ironDome'])]['name']} magazine is "
                         f"{r.choice(['healthy', 'low', 'critical'])}.",
               lambda r: "The impact prediction is uncertain."]


def make_row(rng, sit, style_w, held, variant_p, orders_w, silent_ok=True):
    prio_sites = set(rng.sample([s[1] for s in SITES], rng.randint(1, 4)))
    if rng.random() < 0.4:
        prio_sites.add(sit["site"])
    standing = list(STANDING)
    if rng.random() < variant_p:   # a different standing doctrine: the model must read the rules, not memorise them
        standing = [(c, v, rng.choice(list(ACTIONS)) if rng.random() < 0.5 else a) for c, v, a in standing]
        if silent_ok and rng.random() < 0.35:
            standing.pop(rng.randrange(len(standing)))
        rng.shuffle(standing)
    orders = [random_rule(rng, sit, rng.random() < 0.5) for _ in range(rng.choices([0, 1, 2, 3], orders_w)[0])]
    if rng.random() < 0.12:
        orders.insert(rng.randint(0, len(orders)), ("mod", sit["site"] if rng.random() < 0.6 else rng.choice(SITES)[1], None))
    style = rng.choices(["grammar", "en", "he"], style_w)[0]
    lines = [render(o, style, held, rng) for o in orders] + [render(r, "grammar", False, rng) for r in standing]
    facts = situation_facts(sit["cls"], sit["name"], sit["origin"], sit["site"], sit["open"], sit["site"] in prio_sites)
    if rng.random() < 0.4:
        facts += [d(rng) for d in rng.sample(DISTRACTORS, rng.randint(1, 2))]
    act = decide(orders + standing, sit, prio_sites)
    cands = list(ACTIONS.values())
    if rng.random() < 0.5:
        rng.shuffle(cands)
    meta = {"qtype": "choice", "family": "airdef_doctrine", "style": style, "n_orders": len(orders),
            "variant": standing != STANDING, "gold": act or "silent"}
    if act is None:
        return {"state": render_state(lines, facts), "query": Q_PLAN, "candidates": cands, "target": [1 / 3] * 3,
                "p_null": 1.0, "task": "airdef_doctrine", "label": -1, "meta": meta}
    i = cands.index(ACTIONS[act])
    return {"state": render_state(lines, facts), "query": Q_PLAN, "candidates": cands,
            "target": [float(j == i) for j in range(3)], "p_null": 0.0, "task": "airdef_doctrine", "label": i, "meta": meta}


def build(n, rng, sims, **kw):
    return [make_row(rng, rng.choice(sims) if rng.random() < 0.7 else procedural_situation(rng), **kw) for _ in range(n)]


TRAIN_KW = dict(style_w=[0.45, 0.35, 0.20], held=False, variant_p=0.3, orders_w=[0.3, 0.35, 0.22, 0.13])
EVALS = {   # name -> kwargs; all held-out templates where a template is involved
    "airdef_standard": dict(style_w=[1, 0, 0], held=False, variant_p=0.0, orders_w=[1, 0, 0, 0]),
    "airdef_orders_grammar": dict(style_w=[1, 0, 0], held=False, variant_p=0.0, orders_w=[0, 0.5, 0.3, 0.2]),
    "airdef_orders_en_heldout": dict(style_w=[0, 1, 0], held=True, variant_p=0.0, orders_w=[0, 0.5, 0.3, 0.2]),
    "airdef_orders_he_heldout": dict(style_w=[0, 0, 1], held=True, variant_p=0.0, orders_w=[0, 0.5, 0.3, 0.2]),
    "airdef_doctrine_variants": dict(style_w=[1, 0, 0], held=False, variant_p=1.0, orders_w=[0.5, 0.5, 0, 0]),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / ".context/typically/jobs/airdef"))
    ap.add_argument("--train", type=int, default=24000)
    ap.add_argument("--val", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    sims = sim_situations()
    print(f"{len(sims)} distinct sim situations")
    rng = random.Random(0)
    train, val = build(a.train, rng, sims, **TRAIN_KW), build(a.val, random.Random(1), sims, **TRAIN_KW)
    evals = {k: build(a.eval, random.Random(100 + i), sims, **kw) for i, (k, kw) in enumerate(EVALS.items())}
    train_states = {r["state"] for r in train}
    for k, rows in evals.items():   # no eval state may appear verbatim in train
        evals[k] = [r for r in rows if r["state"] not in train_states]
    for name, rows in [("train", train), ("val", val)] + [(f"eval/{k}", v) for k, v in evals.items()]:
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{name:32s} {len(rows):6d}  gold {dict(Counter(r['meta']['gold'] for r in rows))}")
    # ponytail: the self-check -- the evaluator must agree with the live engine's parser on grammar-only rows
    for r in evals["airdef_orders_grammar"][:200]:
        rule_lines = r["state"].split("\n\nSituation.\n")[0].split("\n")[1:]
        cfg = {"orders": [ln.split(". ", 1)[1] for ln in rule_lines][:-3], "priority_sites": []}
        f = r["state"].split("Situation.\n")[1].split("\n")
        open_ground = "open ground" in f[2]
        site = f[2].split(" near ")[-1].rstrip(".") if open_ground else f[2][len("The threat will hit "):-1]
        prio = (not open_ground) and "a priority site" in f[3]
        want = engine.doctrine_action(cfg, f[0][len("The threat is a "):-1], site, open_ground, prio)
        assert r["meta"]["gold"] == want, (r["state"], r["meta"]["gold"], want)
    print("evaluator agrees with engine.doctrine_action on 200 grammar rows")
