"""A cautious decision rule on calibrated item probabilities. An item the model answers "passes" is reported as
"probably fails" when P(fails) + P(probably fails) >= tau. The probabilities themselves are never changed:
every option and its probability is always shown, and the rule only picks the reported status.

tau is fitted on one half of the rule holdout (split by case) as the largest value whose unsafe-approval rate (verdict of
record) stays within the target. It is then reported, with its false-alarm cost, on the other half, the contrast set and
the real sites.

uv run scripts/helipad_threshold.py --scored .context/helipads/eval/run800_full.json \
    --eval-dir .context/helipads/eval_oldformat/eval --out .context/helipads/eval/threshold_run800.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.helipad_score import CLEAR_FAMILIES, _cases, _split, preds_from_cases, report  # noqa: E402

GRID = [round(0.01 * i, 2) for i in range(1, 51)]
FILES = ("import_oneliner.jsonl", "helipad_contrast.jsonl", "helipad_real.jsonl")


def cautious(meta: dict, pred: dict, tau: float) -> dict:
    probs = pred.get("probs") or {}
    if meta["family"] not in CLEAR_FAMILIES or pred["argmax"] != "passes" or "probably fails" not in probs:
        return pred
    risk = probs.get("fails", 0.0) + probs["probably fails"]
    return {**pred, "argmax": "probably fails"} if risk >= tau else pred


def _apply(rows: list[dict], preds: list[dict], tau: float) -> list[dict]:
    return [cautious(r["meta"], p, tau) for r, p in zip(rows, preds)]


def evaluate(rows: list[dict], preds: list[dict], tau: float) -> dict:
    adj = _apply(rows, preds, tau)
    rep = report(rows, adj)
    cases = [c for c in _cases(_split(rows, adj)[0]).values() if c["record"]]
    clear = [c for c in cases if c["gold_verdict"] == "approved"]
    rep["false_alarm_rate"] = sum(c["record"] != "approved" for c in clear) / len(clear) if clear else None
    return rep


def fit_tau(rows: list[dict], preds: list[dict], target: float = 0.01, grid: list[float] = GRID) -> tuple[float, dict]:
    """Largest tau whose unsafe-approval rate is <= target (fewest false alarms); the smallest tau if none qualifies."""
    best = None
    for tau in sorted(grid):
        rep = evaluate(rows, preds, tau)
        if (rep["unsafe_record_rate"] or 0.0) <= target:
            best = (tau, rep)
    return best or (min(grid), evaluate(rows, preds, min(grid)))


def split_cases(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic half split by case (contrast variants follow their base)."""
    side = lambda r: int(hashlib.sha1(r["meta"]["case"].split("~")[0].encode()).hexdigest(), 16) % 2   # noqa: E731
    return [r for r in rows if side(r) == 0], [r for r in rows if side(r) == 1]


def _summary(rep: dict) -> dict:
    keys = ("unsafe_record_rate", "false_alarm_rate", "verdict_of_record_agreement")
    out = {k: rep.get(k) for k in keys}
    out["unsafe_item_rate"] = (rep.get("unsafe_item_rate") or {}).get("overall")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="helipad_score.py output with full probabilities")
    ap.add_argument("--eval-dir", required=True, help="the eval files the scored run used")
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scored = json.loads(Path(a.scored).read_text())
    data = {}
    for f in FILES:
        rows = [json.loads(line) for line in open(Path(a.eval_dir) / f)]
        data[f] = preds_from_cases(rows, scored[f]["cases"])
    if any(p.get("probs") is None for _, ps in data.values() for p in ps):
        sys.exit("the scored file has no full probabilities; re-score with the current helipad_score.py")
    fit_ids = {id(r) for r in split_cases(data[FILES[0]][0])[0]}
    pairs = list(zip(*data[FILES[0]]))
    fit_rows, fit_preds = [r for r, _ in pairs if id(r) in fit_ids], [p for r, p in pairs if id(r) in fit_ids]
    test_rows, test_preds = [r for r, _ in pairs if id(r) not in fit_ids], [p for r, p in pairs if id(r) not in fit_ids]
    tau, fit_rep = fit_tau(fit_rows, fit_preds, a.target)
    sets = {"holdout_fit": (fit_rows, fit_preds), "holdout_test": (test_rows, test_preds),
            "contrast": data[FILES[1]], "real": data[FILES[2]]}
    out = {"tau": tau, "target": a.target,
           "argmax": {k: _summary(evaluate(r, p, 1.01)) for k, (r, p) in sets.items()},   # tau > 1: nothing flagged
           "cautious": {k: _summary(evaluate(r, p, tau)) for k, (r, p) in sets.items()},
           "curve": {str(t): _summary(evaluate(test_rows, test_preds, t)) for t in GRID[::5]}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"tau": tau, "argmax": out["argmax"], "cautious": out["cautious"]}, indent=1))


if __name__ == "__main__":
    main()
