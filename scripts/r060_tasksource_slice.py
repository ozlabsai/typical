#!/usr/bin/env python3
"""R060 anchor/replay slice: commercial-licensed tasksource typed decisions -> trainer rows (meta.anchor=true).

Pool: tasksource/tasksource-jev-typed-decisions `default` train (+ `full` for Hebrew rows) and
tasksource/procedural-typed-decisions `all` train (levels 1-4 = the upper four of five levels).
License: a row is kept only if its own `license_use` AND tasksource.task_licenses() both say commercial.
Selection: ~10% procedural, ~8% multilingual (every Hebrew-language row included), rest English; choice/noul/score
balanced; round-robin over tasks (soft targets first); any task <= 2% of a split; state <= 8k Gemma-4 tokens.
Contamination: 13-gram (R055/R056 boilerplate rules) + MinHash (char 5-gram Jaccard > 0.5) vs public JevBench
(everything under .context/jevbench-v154/datasets), WorkflowEvals cases.parquet input_json, Hebrew JevBench v0 and
the external Hebrew panel. Source datasets whose rows share a 13-gram with a public JevBench item are dropped whole.
Vision: skipped unless the vision parquet is already in the local HF cache.

  uv run --with tasksource scripts/r060_tasksource_slice.py [--push]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r055_hebrew_open as ho  # noqa: E402  (13-gram rules + Hebrew bench paths, shared with R055/R056 decon)

OUT = ROOT / ".context/r060/tasksource"
JEV = ROOT / ".context/jevbench-v154/datasets"
TD, PD = "tasksource/tasksource-jev-typed-decisions", "tasksource/procedural-typed-decisions"
REPO = "guychuk/typical-r060-data"
GEMMA = ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
N_TRAIN, N_DEV, CAP, PROC, ML = 25_000, 1_000, 0.02, 0.10, 0.08
MAX_TOK, MAX_OPTS, PER_TASK_CANDIDATES, SEED = 8192, 26, 800, 60
KINDS = ("noul", "score", "choice")  # thinnest first: shortfalls spill into choice
HE = re.compile("[א-ת]")
ISO3 = {"eng": "en", "jap": "ja", "fra": "fr", "spa": "es", "por": "pt", "nld": "nl", "deu": "de", "rus": "ru",
        "zho": "zh", "tha": "th", "fas": "fa", "eus": "eu", "swa": "sw"}  # source-name suffixes -> ISO 639-1
YIDDISH = re.compile("[װ-ײ]|[א][ַָ]| (?:איז|דער|דעם|און|געווען|האט) ")


def h(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def lang_of(source: str, state: str) -> str:
    letters = len(re.findall(r"[^\W\d_]", state))
    if letters and len(HE.findall(state)) / letters > 0.5:
        return "yi" if YIDDISH.search(state) else "he"
    if not source.startswith("multilingual/"):
        return "en"
    tail = re.sub(r"^(X-CSQA-|X-CODAH-|translation-)", "", source.rsplit("/", 1)[-1]).split(".")[0]
    tail = ISO3.get(tail, tail)
    return tail if re.fullmatch(r"[a-z]{2,3}", tail) and tail not in ("all", "nli", "sts") else "mul"


# ---------------------------------------------------------------- licenses + pools
def commercial_tasks() -> dict[str, str]:
    import tasksource as ts
    lic = pd.concat([ts.task_licenses(), ts.task_licenses(multilingual=True)]).drop_duplicates("id")
    return dict(zip(lic.id, lic.license_use))


def typed_pool(snap: Path, task_use: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Commercial rows of `default` train, plus the Hebrew-language rows of `full` (dedup by id)."""
    cols = ["state", "kind", "id", "options", "target", "question", "source", "variant", "group_id", "example_id",
            "license", "license_use"]
    df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted((snap / "data").glob("train-*"))])
    full = [pq.read_table(f, columns=cols).to_pandas() for f in sorted((snap / "full").glob("train-*"))]
    full = pd.concat([d[d.state.str.contains("[א-ת]{3}")] for d in full])
    df = pd.concat([df, full]).drop_duplicates("id").reset_index(drop=True)
    df = df[~df.source.str.startswith("procedural-typed-decisions/")]  # procedural comes from PD, with levels
    stats = {"rows_default_plus_full_hebrew": len(df)}
    task = df.source.str.replace("multilingual/", "", regex=False)
    ok = (df.license_use == "commercial") & task.map(lambda t: task_use.get(t) == "commercial")
    stats["row_license_use"] = df.license_use.value_counts().to_dict()
    stats["commercial_rows_failing_task_licenses"] = int(((df.license_use == "commercial") & ~ok).sum())
    stats["commercial_sources_failing_task_licenses"] = sorted(df[(df.license_use == "commercial") & ~ok].source.unique())
    df = df[ok & df.target.map(lambda t: len(t) > 0 and np.isfinite(t).all() and min(t) >= 0 and sum(t) > 0)].copy()
    wide = (df.kind == "choice") & (df.options.map(len) > MAX_OPTS)
    for i in df.index[wide]:
        df.at[i, "options"], df.at[i, "target"] = narrow(df.at[i, "id"], df.at[i, "options"], df.at[i, "target"])
    stats["choice_rows_narrowed_to_26_options"] = int(wide.sum())
    df["narrowed"] = wide
    n_opt = df.options.map(len)
    df = df[(df.kind == "noul") | ((n_opt >= 2) & (n_opt <= MAX_OPTS))]
    df = df.assign(lang=[lang_of(s, t) for s, t in zip(df.source, df.state)],
                   soft=df.target.map(lambda t: bool(((np.asarray(t) > 0) & (np.asarray(t) < 1)).any())))
    return df.reset_index(drop=True), stats


def narrow(row_id: str, options, target) -> tuple[list, list]:
    """>26-option choice (wili's 235 languages, ...) -> every option with mass plus seeded distractors, original order.
    Rows whose mass spans > 26 options stay wide and are dropped by the caller."""
    live = [i for i, t in enumerate(target) if t > 0]
    rest = [i for i, t in enumerate(target) if t <= 0]
    random.Random(row_id).shuffle(rest)
    keep = sorted(live + rest[:max(0, MAX_OPTS - len(live))])
    return [options[i] for i in keep], [float(target[i]) for i in keep]


def procedural_pool(snap: Path) -> pd.DataFrame:
    df = pd.concat([pq.read_table(f).to_pandas() for f in sorted((snap / "all").glob("train-*"))])
    return df[df.level >= 1].reset_index(drop=True)


# ---------------------------------------------------------------- contamination
def test_items() -> dict[str, list[list[str]]]:
    """{test set: [item texts]}; JevBench = every file under datasets/ (jsonl rows, other files whole)."""
    jev = []
    for p in sorted(JEV.rglob("*")):
        if p.is_file():
            jev += ([ho.texts_of_task(r) for r in ho.jsonl(p)] if p.suffix == ".jsonl" else [[p.read_text()]])
    wf = {}
    for p in sorted(Path.home().glob(".cache/huggingface/hub/datasets--typesafe--evalsafe-*/snapshots/*/data/cases.parquet")):
        wf[p.parts[-5]] = [ho.texts_of_task(json.loads(s)) for s in pq.read_table(p, columns=["input_json"])["input_json"].to_pylist()]
    he = [ho.texts_of_task(r) for p in [*sorted(ho.HEJEV.glob("*.jsonl")), *sorted(ho.EXT.glob("*.jsonl"))] for r in ho.jsonl(p)]
    assert jev and len(wf) == 4 and he, "missing a test set"
    return {"jevbench": jev, "hebrew": he, **{f"wf:{k.split('--')[-1]}": v for k, v in wf.items()}}


def gram_index(items: dict) -> dict[int, str]:
    """13-gram -> test set. R056 rules: WF grams in >= 5% of a workflow's cases and JevBench/Hebrew grams in >= 20
    items are prompt boilerplate, not content."""
    index = {}
    for name, rows in items.items():
        freq = Counter(g for texts in rows for g in set().union(*map(ho.grams, texts)))
        limit = max(2, 0.05 * len(rows)) if name.startswith("wf:") else 20
        index.update({g: name for g, c in freq.items() if c < limit})
    return index


def shingle_hash(text: str) -> MinHash:
    text = ho.norm(text).lower()
    m = MinHash(num_perm=128, seed=SEED)
    m.update_batch([text[i:i + 5].encode() for i in range(max(1, len(text) - 4))])
    return m


def minhash_index(items: dict) -> tuple[MinHashLSH, dict]:
    lsh, sigs = MinHashLSH(threshold=0.5, num_perm=128), {}
    for name, rows in items.items():
        for i, texts in enumerate(rows):
            key = f"{name}#{i}"
            sigs[key] = shingle_hash(" ".join(texts))
            lsh.insert(key, sigs[key])
    return lsh, sigs


def contaminated(texts: list[str], grams: dict, lsh: MinHashLSH, sigs: dict) -> str | None:
    for g in set().union(*map(ho.grams, texts)):
        if g in grams:
            return f"13gram:{grams[g]}"
    m = shingle_hash(" ".join(texts))
    for key in lsh.query(m):
        if m.jaccard(sigs[key]) > 0.5:
            return f"minhash:{key.split('#')[0]}"
    return None


def derived_sources(states: pd.Series, sources: pd.Series, jev: list[list[str]]) -> dict[str, list[int]]:
    """Source -> public JevBench items whose (non-boilerplate) 13-grams appear in that source's states."""
    owner = defaultdict(set)
    freq = Counter(g for texts in jev for g in set().union(*map(ho.grams, texts)))
    for i, texts in enumerate(jev):
        for g in set().union(*map(ho.grams, texts)):
            if freq[g] < 20:
                owner[g].add(i)
    hits = defaultdict(set)
    for state, src in zip(states, sources):
        for g in ho.grams(state):
            if g in owner:
                hits[src] |= owner[g]
    return {s: sorted(v) for s, v in hits.items()}


# ---------------------------------------------------------------- selection
def round_robin(df: pd.DataFrame, n: int, cap: int, rng: random.Random) -> pd.DataFrame:
    """n rows, kinds balanced (shortfall of a thin kind spills to the others), one row per task per turn, no task
    above `cap` rows. Soft-target rows go first: a pass over soft rows only, then a pass over the rest."""
    queues = {}
    for (kind, src, soft), g in df.groupby(["kind", "source", "soft"]):
        queues[kind, src, soft] = list(g.index)
        rng.shuffle(queues[kind, src, soft])
    taken, per_task = [], Counter()
    for k, kind in enumerate(KINDS):
        quota = (n - len(taken)) // (len(KINDS) - k)
        got = 0
        for soft in (True, False):
            live = [key for key in sorted(queues) if key[0] == kind and key[2] == soft]
            rng.shuffle(live)
            while live and got < quota:
                for key in list(live):
                    if got >= quota:
                        break
                    if not queues[key] or per_task[key[1]] >= cap:
                        live.remove(key)
                        continue
                    taken.append(queues[key].pop())
                    per_task[key[1]] += 1
                    got += 1
    return df.loc[taken]


def pick_procedural(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    """n rows, round-robin over (task, level)."""
    queues = {key: list(g.index) for key, g in df.groupby(["task", "level"])}
    for q in queues.values():
        rng.shuffle(q)
    taken = []
    while len(taken) < n and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(taken) < n:
                taken.append(queues[key].pop())
    return df.loc[taken]


# ---------------------------------------------------------------- conversion
def typed_row(r) -> dict:
    t = [float(x) for x in r.target]
    if r.kind == "noul":
        p = min(1.0, max(0.0, t[0]))
        q = {"type": "noul", "instructions": r.question, "criteria": {"true": "Yes", "false": "No"}}
        soft, label = [p, 1 - p], "true" if p >= 0.5 else "false"
    elif r.kind == "choice":
        keys = [f"o{i:02d}" for i in range(len(r.options))]
        q = {"type": "choice", "instructions": r.question, "criteria": dict(zip(keys, map(str, r.options)))}
        soft, label = t, keys[int(np.argmax(t))]
    else:
        q = {"type": "score", "instructions": r.question, "criteria": [str(o) for o in r.options]}
        soft, label = t, str(int(np.argmax(t)))
    return {"id": f"ts-{r.id}", "state": r.state, "questions": {"decision": q},
            "targets": {"decision": {"label": label}}, "soft_targets": {"decision": soft},
            "meta": {"anchor": True, "source": f"tasksource:{r.source}", "license": r.license,
                     "license_use": "commercial", "lang": r.lang, "kind": r.kind, "variant": r.variant,
                     "soft_target": bool(r.soft), "options_narrowed": r.narrowed, "group_id": r.group_id, "example_id": r.example_id}}


def procedural_row(r) -> dict | None:
    questions, answers, soft, targets = json.loads(r.questions), json.loads(r.answers), {}, {}
    out_q = {}
    for qid, q in questions.items():
        a = answers[qid]
        if q["type"] == "noul":
            p = float(a["noul"])
            out_q[qid] = {"type": "noul", "instructions": q["instructions"], "criteria": {"true": "Yes", "false": "No"}}
            soft[qid], targets[qid] = [p, 1 - p], "true" if p >= 0.5 else "false"
            continue
        crit = q["criteria"]
        if len(crit) > MAX_OPTS:
            continue
        if q["type"] == "choice":
            keys = list(crit)
            out_q[qid] = {"type": "choice", "instructions": q["instructions"],
                          "criteria": {k: k if v in (None, "") else f"{k}: {v}" for k, v in crit.items()}}
        else:
            keys = [str(i) for i in range(len(crit))]
            out_q[qid] = {"type": "score", "instructions": q["instructions"], "criteria": [str(c) for c in crit]}
        soft[qid] = [float(a["probabilities"].get(k, 0.0)) for k in keys]
        targets[qid] = keys[int(np.argmax(soft[qid]))]
    if not out_q:
        return None
    return {"id": f"ts-proc-{r.id}", "state": r.state, "questions": out_q,
            "targets": {qid: {"label": lab} for qid, lab in targets.items()}, "soft_targets": soft,
            "meta": {"anchor": True, "source": f"tasksource:procedural-typed-decisions/{r.task}",
                     "license": "apache-2.0", "license_use": "commercial", "lang": "en", "kind": "procedural",
                     "level": int(r.level), "soft_target": any(0 < x < 1 for v in soft.values() for x in v)}}


def row_texts(row: dict) -> list[str]:
    return [row["state"] if isinstance(row["state"], str) else json.dumps(row["state"]), *ho.texts_of_task(row["questions"])]


# ---------------------------------------------------------------- main
def snapshot(repo: str, patterns: list[str]) -> Path:
    return Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns))


def counts(rows: list[dict], key) -> dict:
    return dict(Counter(map(key, rows)).most_common())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true", help=f"upload OUT to {REPO} under tasksource/")
    args = ap.parse_args(argv)
    import tasksource
    from importlib.metadata import version
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT))
    from scripts.train_gemma_decision_lora import load_examples

    rng = random.Random(SEED)
    td = snapshot(TD, ["data/train-*", "full/train-*", "sources.yaml", "README.md"])
    pdir = snapshot(PD, ["all/train-*", "README.md", "provenance.json"])
    vision_local = any(r.repo_id == TD and any("vision/" in str(f.file_path) and f.file_path.suffix == ".parquet"
                                                 for rev in r.revisions for f in rev.files)
                       for r in scan_cache_dir().repos)
    sources_yaml = __import__("yaml").safe_load((td / "sources.yaml").read_text())["sources"]

    task_use = commercial_tasks()
    pool, lic_stats = typed_pool(td, task_use)
    proc = procedural_pool(pdir)
    print(f"pool: {len(pool)} commercial typed rows, {len(proc)} procedural rows (levels 1-4)")

    # Source datasets public JevBench derives from (13-gram over every state of `full` + procedural), dropped whole.
    items = test_items()
    states, srcs = [], []
    for f in sorted((td / "full").glob("train-*")):
        t = pq.read_table(f, columns=["state", "source"]).to_pandas().drop_duplicates()
        states += t.state.tolist()
        srcs += t.source.tolist()
    states += proc.state.tolist()
    srcs += ("procedural-typed-decisions/" + proc.task).tolist()
    derived = derived_sources(pd.Series(states), pd.Series(srcs), items["jevbench"])
    del states, srcs
    hub = {s: sources_yaml.get(s, {}).get("dataset", s) for s in derived}
    banned_ds = set(hub.values())
    banned = {s for s, v in sources_yaml.items() if v.get("dataset") in banned_ds} | set(derived)
    print(f"JevBench-derived sources: {derived} -> excluding datasets {sorted(banned_ds)} ({len(banned)} sources)")
    pool = pool[~pool.source.isin(banned)]
    proc = proc[~("procedural-typed-decisions/" + proc.task).isin(banned)]

    # Candidates (top PER_TASK_CANDIDATES per (task, kind) in priority order + every Hebrew row), then filters.
    pool = pool.assign(rand=[rng.random() for _ in range(len(pool))]).sort_values(["soft", "rand"], ascending=[False, True])
    cand = pd.concat([pool.groupby(["source", "kind"]).head(PER_TASK_CANDIDATES), pool[pool.lang == "he"]])
    cand = cand[~cand.index.duplicated()]
    pcand = pick_procedural(proc, int(1.4 * PROC * (N_TRAIN + N_DEV)), rng)
    tok = AutoTokenizer.from_pretrained(GEMMA[0], revision=GEMMA[1])
    n_tok = lambda xs: [len(ids) for ids in tok(list(xs), add_special_tokens=False)["input_ids"]]  # noqa: E731
    cand = cand[cand.state.str.len() < 60_000]
    cand = cand[np.array(n_tok(cand.state)) <= MAX_TOK]
    pcand = pcand[np.array(n_tok(pcand.state)) <= MAX_TOK]

    grams = gram_index(items)
    lsh, sigs = minhash_index(items)
    drops = Counter()
    keep = []
    for row in map(typed_row, cand.itertuples()):
        why = contaminated(row_texts(row), grams, lsh, sigs)
        drops[why] += why is not None
        keep.append(why is None)
    cand = cand[keep]
    prows = []
    for r in pcand.itertuples():
        row = procedural_row(r)
        why = row and contaminated(row_texts(row), grams, lsh, sigs)
        drops[why] += bool(why)
        if row and not why:
            prows.append(row)
    drops.pop(None, None)
    print(f"contamination drops: {dict(drops)}")

    # Split by question group (dev = 4% of groups), then select per split.
    cand = cand.assign(dev=[h(g) % 25 == 0 for g in cand.group_id])
    out = {}
    for split, n in (("train", N_TRAIN), ("dev", N_DEV)):
        part = cand[cand.dev == (split == "dev")]
        n_proc, n_ml = round(PROC * n), round(ML * n)
        he = part[part.lang == "he"]
        ml = round_robin(part[(part.lang != "en") & (part.lang != "he")], n_ml - len(he), round(CAP * n), rng)
        en = round_robin(part[part.lang == "en"], n - n_proc - len(he) - len(ml), round(CAP * n), rng)
        p = [r for r in prows if (h(r["id"]) % 25 == 0) == (split == "dev")][:n_proc]
        out[split] = [*map(typed_row, pd.concat([en, ml, he]).itertuples()), *p]
        rng.shuffle(out[split])
    train_examples = {r["meta"].get("example_id") for r in out["train"]} - {None}
    out["dev"] = [r for r in out["dev"] if r["meta"].get("example_id") not in train_examples]

    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        # not ho.write: sort_keys would reorder criteria away from their soft_targets
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        n_ex = len(load_examples([OUT / f"{split}.jsonl"]))
        n_q = sum(len(r["questions"]) for r in rows)
        assert n_ex == n_q, f"{split}: load_examples gave {n_ex} examples for {n_q} questions"
        print(f"{split}: {len(rows)} rows, {n_ex} trainer examples (load_examples OK)")
    assert not {r["meta"].get("group_id") for r in out["dev"]} & {r["meta"].get("group_id") for r in out["train"]} - {None}

    task = lambda r: r["meta"]["source"].removeprefix("tasksource:")  # noqa: E731
    used = sorted({task(r) for rows in out.values() for r in rows})
    manifest = {
        "purpose": "R060 anchor/replay rows (meta.anchor=true) for the KL-to-base / soft-label term",
        "sources": {TD: td.name, PD: pdir.name, "tasksource_pkg": version("tasksource"), "tokenizer": list(GEMMA)},
        "rules": {"license": "row license_use == commercial AND tasksource.task_licenses() == commercial",
                  "task_cap": CAP, "procedural_share": PROC, "multilingual_share": ML, "max_state_tokens": MAX_TOK,
                  "procedural_levels": "1-4 (upper four of 0-4)", "dev_split": "md5(group_id) % 25 == 0",
                  "contamination": "13-gram (R056 boilerplate rules) + MinHash char-5gram Jaccard > 0.5"},
        "license_audit": lic_stats,
        "contamination": {"test_sets": {k: len(v) for k, v in items.items()}, "dropped": dict(drops),
                          "jevbench_derived_sources": {s: {"hub_dataset": hub[s], "jevbench_items": v} for s, v in derived.items()},
                          "excluded_hub_datasets": sorted(banned_ds), "excluded_sources": sorted(banned)},
        "hebrew": {"rows": {s: sum(r["meta"]["lang"] == "he" for r in rows) for s, rows in out.items()},
                   "catalog": "multilingual catalog Hebrew: mms/he (unspecified), multilingual-NLI-26lang-2mil7/he "
                              "(unspecified), universal_dependencies/he_{htb,iahltwiki,iahltknesset,postrab}/pos "
                              "(other -> unspecified), wili_2018 Hebrew class (ODbL, commercial: the only one kept)"},
        "vision": "included" if vision_local else "skipped: vision parquet not in the local HF cache",
        "licenses": {t: ("apache-2.0" if t.startswith("procedural") else sources_yaml[t]["license"]) for t in used},
        "splits": {},
    }
    for split, rows in out.items():
        path = OUT / f"{split}.jsonl"
        manifest["splits"][split] = {
            "rows": len(rows), "questions": sum(len(r["questions"]) for r in rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "by_type": counts([q["type"] for r in rows for q in r["questions"].values()], lambda x: x),
            "by_lang": counts(rows, lambda r: r["meta"]["lang"]),
            "by_license": counts(rows, lambda r: r["meta"]["license"]),
            "soft_target_rows": sum(r["meta"]["soft_target"] for r in rows),
            "procedural_by_level": counts([r for r in rows if "level" in r["meta"]], lambda r: r["meta"]["level"]),
            "by_task": counts(rows, task)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    tr = manifest["splits"]["train"]
    (OUT / "README.md").write_text(f"""# R060 tasksource anchor slice

Anchor/replay rows for R060's KL-to-base / soft-label term (`--kl-to-base-on anchor`). Built by
`scripts/r060_tasksource_slice.py` from `{TD}` (default train; Hebrew rows also from `full`) and `{PD}`
(levels 1-4 of 0-4). Every row has `meta.anchor=true`, `meta.source='tasksource:<task>'`, `meta.license`, `meta.lang`;
`soft_targets` hold annotator/rating distributions where the source has them, else one-hot.

- train.jsonl: {tr['rows']} rows / {tr['questions']} questions; dev.jsonl: {manifest['splits']['dev']['rows']} rows (split by question group).
- Commercial licenses only (row `license_use` and `tasksource.task_licenses()` must both say commercial).
- Task cap 2%, choice/noul/score balanced, soft rows preferred, ~10% procedural, ~8% multilingual incl. all
  Hebrew-language rows ({manifest['hebrew']['rows']['train']} train), states <= 8k Gemma-4 tokens.
- Decontaminated (13-gram + MinHash > 0.5) against public JevBench, WorkflowEvals cases, Hebrew JevBench v0 and the
  external Hebrew panel; source datasets matching public JevBench excluded whole: {sorted(banned_ds)}.
- Vision: {manifest['vision']}.

Counts, licenses, exclusions and sha256 in manifest.json. Licenses are tasksource's best-effort labels, not legal advice.
""")
    print(json.dumps({s: {k: v for k, v in m.items() if k != "by_task"} for s, m in manifest["splits"].items()}, indent=1))
    if args.push:
        api = HfApi()
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(OUT), path_in_repo="tasksource",
                          commit_message="R060: tasksource anchor/replay slice")
        print(f"pushed -> {REPO}/tasksource")


if __name__ == "__main__":
    main()
