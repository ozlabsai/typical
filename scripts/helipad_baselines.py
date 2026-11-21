"""Shortcut baselines for the helipad decision rows: per family, how far majority class and a depth-8 tree on state line
counts get against the rule labels. A trained model has to beat the tree on the hard items, not the majority class.

uv run scripts/helipad_baselines.py --train .context/typically/jobs/helipad/train.jsonl \
    --eval .context/typically/jobs/helipad/eval/import_oneliner.jsonl --out .context/helipads/eval/baselines.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

PHRASES = ("height unknown", "inside the pad", "Missing core data", "est.")


def features(state: str) -> dict[str, int]:
    f = Counter(f"prefix:{line.split(':')[0]}" for line in state.split("\n") if ":" in line)
    f.update({p: state.count(p) for p in PHRASES})
    tail = state.split("\nObstacles (", 1)   # obstacle lines are everything after the "Obstacles (...)" header
    f["obstacles"] = tail[1].count("\n") if len(tail) == 2 else 0
    return dict(f)


def _by_family(rows: list[dict]) -> dict[str, list[tuple[dict, str]]]:
    out = defaultdict(list)
    for r in rows:
        if not r["meta"].get("expert"):   # soft-target expert rows have no hard gold
            out[r["meta"]["family"]].append((features(r["state"]), r["candidates"][r["label"]]))
    return out


def baselines(train: list[dict], evals: list[dict]) -> dict:
    tr, ev = _by_family(train), _by_family(evals)
    out, hits = {}, {"majority": [], "tree": []}
    for fam in sorted(set(tr) & set(ev)):
        (X, y), (Xe, ye) = zip(*tr[fam]), zip(*ev[fam])
        vec = DictVectorizer()
        tree = DecisionTreeClassifier(max_depth=8, random_state=0).fit(vec.fit_transform(X), y)
        top = Counter(y).most_common(1)[0][0]
        ok = {"majority": [t == top for t in ye], "tree": [p == t for p, t in zip(tree.predict(vec.transform(Xe)), ye)]}
        out[fam] = {k: sum(v) / len(v) for k, v in ok.items()} | {"n": len(ye)}
        if fam != "verdict":
            for k in hits:
                hits[k] += ok[k]
    out["overall"] = {k: sum(v) / len(v) if v else None for k, v in hits.items()}   # item families only
    out["rule_engine"] = "1.0 by construction (the labels are the rules); on real sites, agreement with lz_check is the ceiling"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    read = lambda f: [json.loads(line) for line in open(f)]
    res = baselines(read(a.train), read(a.eval))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
