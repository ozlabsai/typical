"""Helipad decision dataset in the Decision Factory job layout (.context/typically/jobs/helipad/), trained unchanged by
`uv run --no-project python scripts/typically_job.py helipad medium 800`.

train / val / eval/import_oneliner (rule holdout; the pod's base-vs-tuned reveal uses this exact name) come from
helipad.generate with distinct seeds. eval/helipad_real holds every fetched real site labelled by the rules on its lz_check
case. eval/helipad_expert holds the September verdict rows with the expert's call (plus soft item rows from expert_items);
sites whose OSM or DEM fetch failed are left out of both. eval/helipad_contrast holds minimal pairs (`<case>~k` varies one
thing from `<case>`), eval/helipad_norule the same questions with the rule text removed. Every file is shuffled (seeded);
an empty eval file is skipped because the pod job dies on one.

uv run scripts/helipad_build.py [--n-train 6000 --n-val 600 --n-holdout 600 --seed 0 --sites-json $HELIPAD_PRIVATE/sites.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from helipad.case import render_case, visible  # noqa: E402
from helipad.contrast import minimal_pairs  # noqa: E402
from helipad.from_lz import case_from_lz  # noqa: E402
from helipad.generate import sample_balanced  # noqa: E402
from helipad.paths import SITES_JSON, require  # noqa: E402
from helipad.rows import case_rows  # noqa: E402
from scripts.helipad_sites import is_complete  # noqa: E402

N_CONTRAST, N_NORULE = 300, 200


def _write(path: Path, rows: list[dict], rng: random.Random) -> None:
    if not rows:
        print(f"{path.resolve()}: no rows, skipped")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rng.sample(rows, len(rows))   # shuffled, so --eval_limit takes a random slice
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path.resolve()}: {len(rows)} rows")


def _rows(cases: list, seed: int) -> list[dict]:
    return [r for i, c in enumerate(cases) for r in case_rows(c, rng_seed=seed * 100_003 + i)]


def _split(n: int, seed: int, seen: set[str]) -> list:
    """Cases whose state body is not already in `seen` (a held-out case equal to an earlier split's is dropped)."""
    return [c for c, _, _ in sample_balanced(n, seed) if render_case(visible(c)) not in seen]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-holdout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sites", default=str(ROOT / ".context/helipads/sites"))
    ap.add_argument("--out", default=str(ROOT / ".context/typically/jobs/helipad"))
    ap.add_argument("--sites-json", default=str(SITES_JSON))
    a = ap.parse_args(argv)
    out, rng = Path(a.out), random.Random(a.seed)

    train = _rows(_split(a.n_train, a.seed, set()), a.seed)
    seen = {r["state"] for r in train}
    val_cases = _split(a.n_val, a.seed + 1, seen)
    val = _rows(val_cases, a.seed + 1)
    seen |= {r["state"] for r in val}
    hold_cases = _split(a.n_holdout, a.seed + 2, seen)
    _write(out / "train.jsonl", train, rng)
    _write(out / "val.jsonl", val, rng)
    _write(out / "eval/import_oneliner.jsonl", _rows(hold_cases, a.seed + 2), rng)

    pool = rng.sample(hold_cases, len(hold_cases))
    variants = minimal_pairs(pool, rng)[:N_CONTRAST]
    bases = {c.name: c for c in pool}
    pair_cases = [bases[n] for n in dict.fromkeys(v.name.split("~")[0] for v in variants)] + variants
    _write(out / "eval/helipad_contrast.jsonl", _rows(pair_cases, a.seed + 3), rng)
    norule = _rows(pool[:N_NORULE], a.seed + 2)
    for r in norule:
        r["query"] = r["query"].split("\nRule:")[0]
    _write(out / "eval/helipad_norule.jsonl", norule, rng)

    fetched = sorted(Path(a.sites).glob("*/lz.json"))
    sites = {s["slug"]: s for s in json.loads(require(a.sites_json).read_text())} if fetched else {}
    real, expert, excluded = [], [], []
    for f in fetched:
        slug = f.parent.name
        lz = json.loads(f.read_text())
        if slug not in sites:
            print(f"warning: {slug} is not in {a.sites_json}, skipped", file=sys.stderr)
        elif not is_complete(lz):
            excluded.append(slug)
        else:
            s = sites[slug]
            c = case_from_lz(lz, s.get("public_id") or slug)   # never the real name
            real += case_rows(c)
            if s["expert"]:
                expert += case_rows(c, expert=s["expert"], expert_items=s.get("expert_items"))
    if excluded:
        print(f"excluded from real/expert (OSM or DEM fetch failed): {', '.join(excluded)}")
    _write(out / "eval/helipad_real.jsonl", real, rng)
    _write(out / "eval/helipad_expert.jsonl", expert, rng)


if __name__ == "__main__":
    main()
