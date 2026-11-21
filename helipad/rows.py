"""SiteCase -> Decision Factory training rows (the scripts/typically_spike.row schema). One row per protocol item plus
one for the site verdict. Each query carries that item's rule in plain English: the policy has to be in the model's text."""
from __future__ import annotations

import random

from helipad.case import ALLOWED, ITEMS, VERDICTS, SiteCase, label, render_case, visible
from helipad.rules import judge, verdict

QUESTIONS = {
    "pad": "Is the 35x35 m pad clear of rotor obstacles?",
    "very_low_10": "Very-low obstacles (1.2-4.5 m tall) within 10 m of the pad edge?",
    "low_50": "Low obstacles (4.5-24 m tall) within 50 m of the pad edge?",
    "high_300": "High obstacles (24-76 m tall) within 300 m of the pad edge?",
    "ultra_1km": "Ultra-high obstacles (76 m and taller) within 1 km?",
    "slope": "Is the slope in the pad within the aircraft's limit?",
    "zones": "Firing zones, military areas or nature reserves around the site?",
    "belly": "Is the belly rectangle clear (aircraft 1: 17x4 m, nothing above 10 cm; aircraft 2: 15x8 m, nothing above 20 cm)?",
    "dust": "Is the dust level light?",
    "low_wires": "Low wires (power, phone, irrigation) near the pad?",
    "minefield_500": "Minefield within 500 m of the touchdown point?",
    "verdict": "Site verdict for this landing zone?",
}
CERTAIN = "fails if its height is certain (tagged, levels, IEC or measured), probably fails if estimated or unknown"
NO_HIT = "With none it passes, or it is not checked if the map (OSM) is missing."
POLICY = {
    "pad": "Fails if any obstacle inside the pad, or the tree canopy in the pad, is taller than the rotor limit (aircraft 1: "
           "1 m, aircraft 2: 2 m). Otherwise it passes, or it is not checked if the map (OSM) is missing.",
    "very_low_10": f"An obstacle 1.2-4.5 m tall closer than 10 m to the pad edge {CERTAIN}; exactly 10 m is outside. "
                   f"Tree-canopy layer values do not count. {NO_HIT}",
    "low_50": f"An obstacle 4.5-24 m tall closer than 50 m to the pad edge {CERTAIN}; exactly 50 m is outside. Tree-canopy "
              f"layer values do not count; a lit road counts as a row of ~10 m lighting poles. {NO_HIT}",
    "high_300": f"An obstacle 24-76 m tall closer than 300 m to the pad edge {CERTAIN}, and a planned high-voltage corridor "
                f"(TAMA 10) within 317.5 m of touchdown (300 m of the pad edge) probably fails. Tree-canopy layer values "
                f"do not count. {NO_HIT}",
    "ultra_1km": "An obstacle 76 m or taller within 1,000 m of touchdown makes a forbidden approach sector: coordination, "
                 f"never a failure. {NO_HIT}",
    "slope": "Fails if the slope minus its uncertainty exceeds the limit (aircraft 1: 15 deg, aircraft 2: 10 deg). If "
             "slope plus uncertainty exceeds it, or the slope is unavailable, it is not checked.",
    "zones": "A firing zone, military area, nature reserve or a minefield beyond 500 m means coordination, never a failure. "
             "Otherwise it passes, or it is not checked if the zones layer (GovMap) is unavailable.",
    "belly": "Ground condition is preparation, never a failure: obstacles in the belly area mean prep. Not assessed "
             "means not checked.",
    "dust": "Light dust passes; medium or heavy dust means prep (wetting), never a failure. Not assessed means not checked.",
    "low_wires": "No map shows low wires. Fails if a wire crosses the pad, probably fails if one is seen within 50 m of the "
                 "pad edge, otherwise not checked.",
    "minefield_500": "There is no public minefield layer: fails if a minefield is known within 500 m of touchdown, "
                     "otherwise not checked.",
    "verdict": "Any item that fails makes the site fail. Otherwise, missing core map data (OSM or DEM) makes it not "
               "checked. Otherwise, any item that probably fails makes the site probably fail; else it is approved, "
               "and prep and coordination never change it.",
}
EXPERT = {"approved": {"approved": 1.0}, "not_approved": {"probably_fails": 0.5, "fails": 0.5}}


def _row(state: str, key: str, target: dict[str, float], c: SiteCase, seed, **meta) -> dict:
    """Candidates in a per-row seeded order (ARCH-2); `label` points at the gold one."""
    order = list(VERDICTS if key == "verdict" else ALLOWED[key])
    random.Random(f"{c.name if seed is None else seed}|{key}").shuffle(order)
    t = [float(target.get(s, 0.0)) for s in order]
    return {"state": state, "query": f"{QUESTIONS[key]}\nRule: {POLICY[key]}", "candidates": [label(s) for s in order],
            "target": t, "p_null": 0.0, "task": f"import_{key}", "label": max(range(len(t)), key=t.__getitem__),
            "meta": {"qtype": "choice", "family": key, "case": c.name, "core_missing": list(c.core_missing), **meta}}


def case_rows(c: SiteCase, expert: str | None = None, expert_items: dict[str, tuple[str, ...]] | None = None,
              rng_seed=None) -> list[dict]:
    """Rule rows (11 items + verdict) labelled on the visible case, or, with `expert`, the expert's verdict row plus
    one row per expert item, the target spread over that item's acceptable statuses."""
    v = visible(c)
    state = render_case(v)
    if expert is None:
        items = judge(v)
        rows = [_row(state, k, {items[k]: 1.0}, v, rng_seed) for k in ITEMS]
        return rows + [_row(state, "verdict", {verdict(v, items): 1.0}, v, rng_seed)]
    rows = [_row(state, "verdict", EXPERT[expert], v, rng_seed, expert=expert)]
    for k, ok in (expert_items or {}).items():
        assert ok and set(ok) <= set(ALLOWED[k]), (k, ok)
        rows.append(_row(state, k, {s: 1 / len(ok) for s in ok}, v, rng_seed, expert=expert))
    return rows
