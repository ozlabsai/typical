"""Score a helipad decision checkpoint on helipad eval files and report item agreement, verdict agreement, verdict
consistency with the model's own item answers, safety (false clears), contrast pairs, calibration and expert direction.

uv run scripts/helipad_score.py --ckpt .context/typically/results/co_helipad/best.pt \
    --files .context/typically/jobs/helipad/eval/*.jsonl --out .context/helipads/eval/plan1.json
Re-score a previous output without a model: --preds-in <that output.json> instead of --ckpt.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from helipad import rules  # noqa: E402
from helipad.case import ALLOWED, RING_ITEMS  # noqa: E402

CLEAR_FAMILIES = frozenset(f for f, allowed in ALLOWED.items() if "passes" in allowed)   # a false clear is possible
RING_FAMILIES = RING_ITEMS | {"pad"}
FAILING = {"fails", "probably fails"}


def _status(label: str) -> str:
    return label.replace(" ", "_")


def _mean(xs: list) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _ece(hits: list[tuple[float | None, bool]], bins: int = 10) -> float | None:
    """Expected calibration error: bin by confidence, weight |accuracy - mean confidence| by bin size.
    None without full probabilities (an old-format output)."""
    if not hits or any(conf is None for conf, _ in hits):
        return None
    groups = defaultdict(list)
    for conf, ok in hits:
        groups[min(int(conf * bins), bins - 1)].append((conf, ok))
    return sum(abs(sum(ok for _, ok in g) / len(g) - sum(c for c, _ in g) / len(g)) * len(g) for g in groups.values()) / len(hits)


def _per_family(scored, keep, hit, agg=_mean, overall_of=None) -> dict:
    """{family: agg(hits), "overall": agg(pooled hits)} over the (meta, gold, pred) rows that keep(family, gold) selects."""
    fam = defaultdict(list)
    for m, gold, p in scored:
        if keep(m["family"], gold):
            fam[m["family"]].append(hit(gold, p))
    out = {k: agg(v) for k, v in sorted(fam.items())}
    out["overall"] = agg([x for k, v in fam.items() if overall_of is None or k in overall_of for x in v])
    return out


def _cases(scored) -> dict:
    """Per case: item argmaxes, the model's verdict, the gold verdict and the verdict of record (derived in code)."""
    cases = defaultdict(lambda: {"items": {}, "verdict": None, "gold_verdict": None, "core": False})
    for m, gold, p in scored:
        c = cases[m["case"]]
        c["core"] = bool(m.get("core_missing"))
        if m["family"] == "verdict":
            c["verdict"], c["gold_verdict"] = p["argmax"], gold
        else:
            c["items"][m["family"]] = p["argmax"]
    for c in cases.values():
        c["record"] = rules.derive_verdict([_status(s) for s in c["items"].values()], c["core"]).replace("_", " ") if c["items"] else None
    return cases


def _split(rows, preds):
    """-> (meta, gold, pred) for rule rows; expert rows apart."""
    scored, expert = [], []
    for r, p in zip(rows, preds):
        (expert if r["meta"].get("expert") else scored).append((r["meta"], r["candidates"][r["label"]], p, r))
    return [s[:3] for s in scored], expert


def pooled_records(files) -> dict[str, str]:
    """Verdicts of record by case across several (rows, preds) files: the real-site file carries the expert sites' items."""
    out = {}
    for rows, preds in files:
        out.update({k: c["record"] for k, c in _cases(_split(rows, preds)[0]).items() if c["record"]})
    return out


def _direction(answers) -> dict:
    """answers: (verdict, expert verdict) pairs -> approved-vs-not agreement; "not checked" abstains."""
    out = {"n": len(answers), "direction_match": 0, "abstained": 0}
    for pred, exp in answers:
        if pred == "not checked":
            out["abstained"] += 1
        else:
            out["direction_match"] += int((pred == "approved") == (exp == "approved"))
    return out


def _expert(expert, records) -> dict:
    verdicts = [(m["case"], p["argmax"], m["expert"]) for m, _, p, _ in expert if m["family"] == "verdict"]
    items = [p["argmax"] in {c for c, t in zip(r["candidates"], r["target"]) if t > 0}
             for m, _, p, r in expert if m["family"] != "verdict"]
    return {"expert": _direction([(records.get(c, a), e) for c, a, e in verdicts]),
            "expert_record": _direction([(records[c], e) for c, _, e in verdicts if c in records]),
            "expert_model_verdict": _direction([(a, e) for _, a, e in verdicts]),
            "expert_items": {"n": len(items), "match": sum(items)}}


def _contrast(scored) -> dict:
    """Variants `<base>~k` against their base, on the items whose gold the variant flips."""
    by = {(m["case"], m["family"]): (gold, p["argmax"]) for m, gold, p in scored if m["family"] != "verdict"}
    flips = []
    for (case, f), var in by.items():
        base = by.get((case.split("~")[0], f)) if "~" in case else None   # --limit-cases can drop the base
        if base and base[0] != var[0]:
            flips.append((base, var))
    return {"contrast_flipped_n": len(flips),
            "contrast_flipped_agreement": _mean([v[0] == v[1] for _, v in flips]),
            "contrast_pair_accuracy": _mean([b[0] == b[1] and v[0] == v[1] for b, v in flips])}


def report(rows: list[dict], preds: list[dict], records: dict[str, str] | None = None) -> dict:
    """records: verdicts of record from other files (pooled_records), used only for expert rows without items here."""
    scored, expert = _split(rows, preds)
    cases = _cases(scored)
    own = {k: c["record"] for k, c in cases.items() if c["record"]}
    with_record = [c for c in cases.values() if c["verdict"] and c["record"]]
    items = lambda f, g: f != "verdict"   # noqa: E731
    conf = lambda p: p["probs"][p["argmax"]] if p.get("probs") else None   # noqa: E731
    calib = lambda g, p: (conf(p), p["argmax"] == g)   # noqa: E731
    return {"item_agreement": _per_family(scored, items, lambda g, p: p["argmax"] == g),
            "verdict_agreement": _mean([c["verdict"] == c["gold_verdict"] for c in cases.values() if c["verdict"]]),
            "verdict_of_record_agreement": _mean([c["record"] == c["gold_verdict"] for c in with_record]),
            "consistency": _mean([c["verdict"] == c["record"] for c in with_record]),
            "unsafe_item_rate": _per_family(scored, lambda f, g: f in CLEAR_FAMILIES and g in FAILING,
                                            lambda g, p: p["argmax"] == "passes"),
            "false_clear_on_not_checked": _per_family(scored, lambda f, g: f in CLEAR_FAMILIES and g == "not checked",
                                                      lambda g, p: p["argmax"] == "passes"),
            "unsafe_record_rate": _mean([c["record"] == "approved" for c in with_record if c["gold_verdict"] != "approved"]),
            # low_wires and minefield_500 are always "not checked" on these cases, so they stay out of the overall
            "not_checked_recall": _per_family(scored, lambda f, g: items(f, g) and g == "not checked",
                                              lambda g, p: p["argmax"] == "not checked", overall_of=CLEAR_FAMILIES),
            "ece": _per_family(scored, lambda f, g: True, calib, agg=_ece),
            "ece_rings": _ece([calib(g, p) for m, g, p in scored if m["family"] in RING_FAMILIES]),
            **_contrast(scored), **_expert(expert, {**(records or {}), **own}), "n_cases": len(cases)}


def _load(ckpt: str):
    sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "scripts")]
    from typical import Typical
    m = Typical.from_pretrained(ckpt, backbones={})
    m.activate()
    return m


def _predict(m, rows: list[dict]) -> list[dict]:
    from typically_reveal import queries, score
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["state"]].append(i)
    preds = [None] * len(rows)
    for n, (state, idx) in enumerate(groups.items(), 1):
        for i, out in zip(idx, score(m, state, queries([rows[i] for i in idx]))):
            preds[i] = out
        if n % 50 == 0:
            print(f"{n}/{len(groups)} cases", file=sys.stderr)
    return preds


def _explain(rows, preds) -> dict:
    """Per-case records with full probabilities (the safety threshold is fitted on them; --preds-in reads them back)."""
    out = defaultdict(lambda: {"items": {}})
    for r, p in zip(rows, preds):
        val = {"argmax": p["argmax"], "probs": p["probs"] and {k: round(v, 4) for k, v in p["probs"].items()}}
        e, fam = out[r["meta"]["case"]], r["meta"]["family"]
        if fam == "verdict":
            e["verdict"] = val
        else:
            e["items"][fam] = val
    return dict(out)


def preds_from_cases(rows: list[dict], cases: dict) -> tuple[list[dict], list[dict]]:
    """A previous output's case records -> (rows that have a prediction, preds). The old [argmax, p] format has no
    full probabilities, so its preds carry probs=None (calibration -> None)."""
    kept, preds = [], []
    for r in rows:
        e, fam = cases.get(r["meta"]["case"], {}), r["meta"]["family"]
        v = e.get("verdict") if fam == "verdict" else e.get("items", {}).get(fam)
        if v is not None:
            kept.append(r)
            preds.append(v if isinstance(v, dict) else {"argmax": v[0], "probs": None})
    return kept, preds


def _limit(rows: list[dict], n: int) -> list[dict]:
    """n cases, sampled by base name so a contrast pair `<base>~k` stays with its base."""
    base = lambda r: r["meta"]["case"].split("~")[0]   # noqa: E731
    keep = set(random.Random(0).sample(sorted({base(r) for r in rows}), min(n, len({base(r) for r in rows}))))
    return [r for r in rows if base(r) in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--preds-in", help="re-score a previous --out file's case records: no model, no GPU")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--limit-cases", type=int, default=300, help="cases (or contrast bases) per file; 0 = all; "
                                                                "ignored with --preds-in")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old = json.loads(Path(a.preds_in).read_text()) if a.preds_in else None
    m = None if old else _load(a.ckpt)
    scored = {}
    for f in a.files:
        rows = [json.loads(line) for line in open(f)]
        if old:
            scored[f] = preds_from_cases(rows, old.get(Path(f).name, {}).get("cases", {}))
        else:
            rows = _limit(rows, a.limit_cases) if a.limit_cases else rows
            scored[f] = rows, _predict(m, rows)
    records, result = pooled_records(scored.values()), {}
    for f, (rows, preds) in scored.items():
        result[Path(f).name] = {"report": report(rows, preds, records), "cases": _explain(rows, preds)}
        print(Path(f).name, json.dumps(result[Path(f).name]["report"], ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
