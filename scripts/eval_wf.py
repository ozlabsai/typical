"""PLAN6 Phase 6 / PLAN7 §3x: post-hoc evaluation of a checkpoint on data_wf / data_wf_hf eval files at
full state length (train-time evals truncate states to 256 tokens and are head-capped by --eval_cap).

    uv run python scripts/eval_wf.py --run runs/nc_v3_tap20_wf --mode native \
        --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl --limit 0 --out runs/nc_v3_tap20_wf/eval_wf_full.json

Rows are the training schema (state/query/candidates/target/p_null/label), so the query is used verbatim
(no query_text re-rendering). Per file: metrics.summarize (raw, T=1) + per-family/qtype/label-class
breakdowns + trivial baselines + metrics.rubric_flip (overall and per family) for wf_rubric_flip*.

--limit is a *stratified seeded sample* (random.Random(0)), not head-N: several eval files are family-
or label-ordered (REPORT.md §3w correction), so a naive head-N is not representative. --limit 0 (or
>= the file's row count) = every row. meta.flip_pair pairs are sampled and emitted as atomic units so
they always stay adjacent; strata are meta.family (meta.qtype when a row has no family).

Scoring (native mode) batches BATCH rows per forward through run_batch_native (mcq.collate_mcq + the
same batched call pcdm/train.py uses), instead of one native_kv_decide call per row -- ~3 rows/s per-row vs.
30+ rows/s batched on H100 at K<=10 (test_native_kv_decide_matches_run_batch already proves the two
paths agree per-row; tests/test_pipeline.py adds a batched-vs-single-row check on 6 rows here). A row
whose rendered suffix exceeds native.MAX_SUFFIX is chunked hierarchically INSIDE run_batch_native (see
pcdm/native.py's module docstring) rather than failing, so "suffix_overflow" below should now read ~0 even for
big-K files like tree_choice_cap (K=320) -- previously native_kv_decide had no chunking fallback and
those rows were scored as chance (REPORT.md §3w correction). The counter stays, as a rescue-path
diagnostic: it only increments if a row still can't be scored (a genuine RuntimeError/AssertionError),
in which case that one row falls back to a uniform distribution, same as before.
Soft-gold sets (typed_decisions) report NLL/Brier; acc there is argmax-vs-argmax agreement, not truth.
"""
import argparse
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from mcq import collate_mcq  # noqa: E402
from metrics import rubric_flip, summarize  # noqa: E402
from native import run_batch_native  # noqa: E402
from pcdm_jev.decider import PCDMDecider  # noqa: E402

BATCH = 16  # ponytail: fixed native forward-pass batch size; lower it if a huge-K file (e.g. K=320) OOMs


def stratified_limit(rows, limit, seed=0):
    """rows[:limit] head-N is not representative for family-/label-ordered files (REPORT.md §3w
    correction) -- sample instead. Units: a meta.flip_pair pair of rows (kept atomic so pairs stay
    adjacent in the output) or a lone row. Strata: meta.family, falling back to meta.qtype for rows
    with no family. Row quota per stratum is proportional to the stratum's row share of the file
    (largest-remainder rounding); any shortfall (a tiny stratum runs out of units first) is topped
    up from the seeded-shuffled remainder across all strata so the total still hits `limit`.
    limit<=0 or >= len(rows): return rows unchanged (every row, original order)."""
    if not limit or limit <= 0 or limit >= len(rows):
        return rows

    pair_rows = defaultdict(list)
    for i, r in enumerate(rows):
        fp = r.get("meta", {}).get("flip_pair")
        if fp is not None:
            pair_rows[fp].append(i)
    paired = {i for idxs in pair_rows.values() if len(idxs) == 2 for i in idxs}
    units = [idxs for idxs in pair_rows.values() if len(idxs) == 2]
    units += [[i] for i in range(len(rows)) if i not in paired]

    def strat_key(u):
        m = rows[u[0]].get("meta", {})
        return m.get("family") or m.get("qtype") or "_none"

    by_strat = defaultdict(list)
    for u in units:
        by_strat[strat_key(u)].append(u)
    rng = random.Random(seed)
    for u_list in by_strat.values():
        rng.shuffle(u_list)

    strata = sorted(by_strat)
    sizes = {s: sum(len(u) for u in by_strat[s]) for s in strata}
    total = sum(sizes.values())
    raw = {s: limit * sizes[s] / total for s in strata}
    quota = {s: int(raw[s]) for s in strata}
    short = limit - sum(quota.values())
    for s in sorted(strata, key=lambda s: raw[s] - quota[s], reverse=True)[:max(short, 0)]:
        quota[s] += 1

    selected, have = [], 0
    for s in strata:
        n = 0
        for u in by_strat[s]:
            if n >= quota[s]:
                break
            selected.append(u); n += len(u)
        have += n
    used = {tuple(u) for u in selected}
    leftover = [u for u_list in by_strat.values() for u in u_list if tuple(u) not in used]
    rng.shuffle(leftover)
    for u in leftover:
        if have >= limit:
            break
        selected.append(u); have += len(u)

    selected.sort(key=lambda u: u[0])  # stable, roughly-original-order output; units never split
    return [rows[i] for u in selected for i in u]


def _score_native(dec, rows, batch_size=BATCH):
    """-> (probs [N, Kmax+1], suffix_overflow). Kmax over `rows`; null column always last regardless
    of a mini-batch's own (smaller) Kmax, since run_batch_native pads with finfo.min there."""
    head, model = dec.native["bb"], dec.native["m"]
    kmax = max(len(r["candidates"]) for r in rows)
    probs = np.zeros((len(rows), kmax + 1))
    overflow = 0

    def score_into(i, chunk):
        nonlocal overflow
        batch = collate_mcq(chunk)
        with torch.inference_mode():
            p = torch.softmax(run_batch_native(head, model, batch, chunk, max_state=dec.max_state), dim=-1)
        p = p.float().cpu().numpy()
        for j, r in enumerate(chunk):
            k = len(r["candidates"])
            probs[i + j, :k], probs[i + j, -1] = p[j, :k], p[j, -1]

    for i in tqdm(range(0, len(rows), batch_size), desc="native", leave=False):
        chunk = rows[i:i + batch_size]
        try:
            score_into(i, chunk)
        except (AssertionError, RuntimeError):
            # ponytail: one unrecoverable row (e.g. a single candidate longer than MAX_SUFFIX even
            # alone) shouldn't sink the whole batch -- rescue one row at a time.
            for j, r in enumerate(chunk):
                k = len(r["candidates"])
                try:
                    score_into(i + j, [r])
                except (AssertionError, RuntimeError):
                    overflow += 1
                    probs[i + j, :k + 1] = 1.0 / (k + 1)
    return probs, overflow


def _score_energy(dec, rows):
    """Per-row fallback (model.decide's KV-cache path isn't batched across different states the way
    run_batch_native is); energy isn't the architecture under test here (§3w/§3x use native
    checkpoints) so this stays the slow path rather than growing a second batched implementation."""
    kmax = max(len(r["candidates"]) for r in rows)
    probs, overflow = np.zeros((len(rows), kmax + 1)), 0
    for i, r in enumerate(tqdm(rows, desc="energy", leave=False)):
        k = len(r["candidates"])
        try:
            p = dec._probs("energy", r["state"], r["query"], r["candidates"]).float().cpu().numpy()
        except AssertionError:
            overflow += 1
            p = np.full(k + 1, 1.0 / (k + 1))
        probs[i, :k], probs[i, -1] = p[:k], p[-1]
    return probs, overflow


def _build_target_label(rows):
    kmax = max(len(r["candidates"]) for r in rows)
    target = np.zeros((len(rows), kmax + 1))
    label = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        k = len(r["candidates"])
        target[i, :k] = (1 - r["p_null"]) * np.asarray(r["target"], dtype=float)
        target[i, -1] = r["p_null"]
        if r.get("label") is not None:
            label[i] = r["label"]
    return target, label


def breakdown(rows, probs, target, label, meta_key):
    """{value -> summarize(...) + n} grouped by rows[i]["meta"][meta_key]; rows missing that key are
    left out of this particular breakdown (they may show up in another one, e.g. by_qtype)."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        v = r.get("meta", {}).get(meta_key)
        if v is not None:
            groups[str(v)].append(i)
    out = {}
    for g, idx in groups.items():
        idx = np.array(idx)
        d = summarize(probs[idx], target[idx], [label[i] for i in idx])
        d["n"] = len(idx)
        out[g] = d
    return out


def breakdown_label_class(rows, probs, target, label):
    """{"answer"|"null" -> summarize(...) + n}: label==-1 is a null-target row, label>=0 answers a
    real candidate -- the two floors (majority baseline, always-positive) mean different things for
    each, so acc/nll/brier are reported split rather than pooled."""
    groups = defaultdict(list)
    for i, l in enumerate(label):
        if np.isnan(l):
            continue
        groups["null" if int(l) == -1 else "answer"].append(i)
    out = {}
    for g, idx in groups.items():
        idx = np.array(idx)
        d = summarize(probs[idx], target[idx], [label[i] for i in idx])
        d["n"] = len(idx)
        out[g] = d
    return out


def trivial_baselines(rows, label):
    """Floors so every accuracy number has something to beat: majority_acc = accuracy of always
    predicting the single most common label value (null counts as its own class); for noul files
    (fixed candidates=["no","yes"]) also always_positive_acc = accuracy of always answering "yes"."""
    valid = [int(l) for l in label if not np.isnan(l)]
    out = {}
    if valid:
        maj, cnt = Counter(valid).most_common(1)[0]
        out["majority_class"] = maj
        out["majority_acc"] = cnt / len(valid)
    qtypes = {r.get("meta", {}).get("qtype") for r in rows}
    if qtypes == {"noul"}:
        pos = sum(1 for l in valid if l == 1)
        out["always_positive_acc"] = pos / len(valid) if valid else float("nan")
    return out


def flip_by_family(probs, rows):
    families = sorted({r["meta"]["family"] for r in rows if r.get("meta", {}).get("flip_pair") is not None})
    out = {}
    for fam in families:
        idx = [i for i, r in enumerate(rows) if r.get("meta", {}).get("family") == fam]
        out[fam] = rubric_flip(probs[np.array(idx)], [rows[i] for i in idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="native", choices=["native", "energy"])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="stratified seeded sample size per file; 0 = all rows")
    ap.add_argument("--max_state", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="native forward-pass batch size; lower it for very long states or huge K "
                         "(BATCH=16 OOMs a 14B at 12k-token states)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dec = PCDMDecider(args.run, mode=args.mode, max_state=args.max_state)
    out = {"run": args.run, "mode": args.mode, "max_state": args.max_state, "limit": args.limit,
           "batch": args.batch, "eval": {}}
    for f in sorted(p for pat in args.files for p in glob.glob(pat)):
        rows_all = [json.loads(l) for l in open(f)]
        rows = stratified_limit(rows_all, args.limit, seed=0)
        target, label = _build_target_label(rows)
        probs, overflow = (_score_native(dec, rows, batch_size=args.batch) if args.mode == "native"
                           else _score_energy(dec, rows))

        name = Path(f).stem
        result = {
            "n": len(rows), "n_total": len(rows_all), "suffix_overflow": overflow,
            "raw": summarize(probs, target, label),
            "by_family": breakdown(rows, probs, target, label, "family"),
            "by_qtype": breakdown(rows, probs, target, label, "qtype"),
            "by_label_class": breakdown_label_class(rows, probs, target, label),
            "baselines": trivial_baselines(rows, label),
        }
        if name.startswith("wf_rubric_flip"):
            result["flip"] = rubric_flip(probs, rows)
            result["flip_by_family"] = flip_by_family(probs, rows)
        out["eval"][name] = result
        print(name, f"n={result['n']}/{result['n_total']}", f"overflow={overflow}",
              json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["raw"].items()}),
              result.get("flip", ""), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))  # checkpoint after every file


if __name__ == "__main__":
    main()
