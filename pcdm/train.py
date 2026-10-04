"""Train the decision tower + LoRA backbone and evaluate it.

uv run pcdm/train.py --name X [--backbone Qwen/Qwen3-1.7B-Base] [--lora_layers 8] [--lora_r 16]
                 [--steps 12000] [--bs 64] [--lr 3e-4] [--lora_lr 1e-4] [--seed 0]
                 [--no_hybrid] [--no_cand_null] [--no_null] [--null softmax|factored] [--hard_only] [--mix full|nlionly]
                 [--eval_every 4000] [--val_every 1000] [--ckpt_every 2000] [--eval_bs 128]
                 [--device auto] [--wandb] [--hf_repo guychuk/pcdm-runs] [--data data]
                 [--smoke] [--eval_only] [--dump_logits DIR]
                 [--head mlp|z1|zr|zr_set] [--z_dim 128] [--z_probes 8] [--tiny_layers 2]
                 [--cand_encoder backbone|qwen3emb|tiny]
                 [--readout energy|mcq|native] [--nc_head n2|n3|n2n3] [--nc_render letters|tags|letters_nonull|semif] [--perm_lambda L] [--no_shuffle]
                 [--score_head choice|cumlink] [--noul_head choice|bern] [--qtype_filter choice|score|noul] [--ordinal_smooth TAU]
"""
import argparse
import sys
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from encode import Backbone, FeatureCache, pick_device
from model import DecisionModel, decision_loss, collate, run_batch, split_joint
from mcq import MCQHead, collate_mcq, run_batch_mcq
from native import NativeHead, run_batch_native, native_features, shuffle_options, perm_consistency_loss
from metrics import summarize, choice_set_effects, ksweep, counterfactual, rubric_flip, ordinal_metrics, noul_reversed_check
import data as data_mod

FAMILY = getattr(data_mod, "FAMILY", None)
NLI_ONLY_TASKS = {"snli", "mnli", "anli", "snli_soft", "unli", "boolq", "squad", "clinc"}

SMOKE_STATES = [
    "A man is playing guitar on a busy city street while people walk by.",
    "The research team published their findings after months of careful analysis.",
    "A dog is chasing a ball across a green park near the river.",
    "The customer asked to cancel their subscription due to billing issues.",
    "Scientists discovered a new species of frog in the rainforest.",
]
SMOKE_QUERIES = [
    "what is the relationship between the state and the hypothesis?",
    "what is the intent behind this request?",
    "what topic does this text discuss?",
]
SMOKE_LABEL_POOL = {
    "snli": ["entailment", "neutral", "contradiction"],
    "clinc": ["cancel subscription", "book a flight", "check account balance", "play music", "set an alarm"],
}


def family_of(task):
    return FAMILY.get(task, task) if FAMILY else task


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_hard_only(examples):
    out = []
    for ex in examples:
        ex = dict(ex)
        if ex["task"] == "unli":
            ex["p_null"] = float(round(ex["p_null"]))
            ex["target"] = [1.0]
        else:
            gold = max(range(len(ex["target"])), key=lambda j: ex["target"][j])
            ex["target"] = [1.0 if j == gold else 0.0 for j in range(len(ex["target"]))]
        out.append(ex)
    return out


def apply_no_null(examples):
    out = []
    for ex in examples:
        if ex["p_null"] == 1:
            continue
        ex = dict(ex)
        ex["p_null"] = 0.0
        out.append(ex)
    return out


def apply_qtype_filter(examples, qtype):
    """PLAN7 track C --qtype_filter: keep only meta.qtype == qtype rows (score_A_kway etc. train
    3k steps on score-only data_wf/data_wf_hf rows; noul arms on noul-only rows)."""
    return [ex for ex in examples if ex.get("meta", {}).get("qtype") == qtype]


def apply_ordinal_smooth(examples, tau):
    """PLAN7 track C score_B_smooth: replace a score-qtype row's hard one-hot target with an
    ordinal-smoothed distribution over its K rendered (ordered) levels -- target_j proportional
    to exp(-|j-y|/tau), y = the gold level index (candidate order == level order). Rows that
    aren't meta.qtype == "score", or lack a hard candidate label, pass through unchanged."""
    out = []
    for ex in examples:
        lbl = ex.get("label")
        if ex.get("meta", {}).get("qtype") != "score" or not (isinstance(lbl, int) and lbl >= 0):
            out.append(ex)
            continue
        w = [math.exp(-abs(j - lbl) / tau) for j in range(len(ex["candidates"]))]
        s = sum(w)
        out.append(dict(ex, target=[wi / s for wi in w]))
    return out


def apply_drop_truncated(examples, tokenizer, max_state, label):
    """--drop_truncated (tl1b item 1): drop train rows whose state exceeds --max_state backbone
    tokens (no special tokens) -- states are right-truncated at train/eval time (pcdm/native.py),
    so a row whose facts land past max_state was training the model to answer confidently from
    a state it never actually saw (REPORT S3ab/S3af: data_wf_long p50 1,845 tokens, 98.8% of
    long rows > 1,024). Cheap pre-filter: only tokenize rows already longer than 3.5*max_state
    chars (this repo's states are English; that many chars can't possibly under-run max_state
    tokens) -- tokenizing every row of a ~1M-row corpus for a mostly-irrelevant check is the
    expensive path this half-measure exists to avoid. Prints how many of `label`'s rows were
    dropped."""
    char_cut = 3.5 * max_state
    suspect = [i for i, ex in enumerate(examples) if len(ex["state"]) > char_cut]
    drop = set()
    if suspect:
        lens = [len(ids) for ids in tokenizer([examples[i]["state"] for i in suspect],
                                                add_special_tokens=False)["input_ids"]]
        drop = {suspect[j] for j, n in enumerate(lens) if n > max_state}
    print(f"--drop_truncated {label}: {len(drop)}/{len(examples)} rows dropped "
          f"({len(suspect)} tokenized, {len(examples) - len(suspect)} skipped by the char pre-filter)")
    return [ex for i, ex in enumerate(examples) if i not in drop]


def make_smoke_examples(n, seed):
    rng = random.Random(seed)
    tasks = ["snli", "unli", "clinc"]
    examples = []
    for i in range(n):
        task = rng.choice(tasks)
        state = f"{rng.choice(SMOKE_STATES)} ({i})"
        query = rng.choice(SMOKE_QUERIES)
        if task == "unli":
            p = rng.random()
            examples.append({
                "state": state, "query": query, "candidates": ["the hypothesis is plausible"],
                "target": [1.0], "p_null": 1 - p, "task": task, "label": None,
            })
        else:
            pool = SMOKE_LABEL_POOL[task]
            k = rng.randint(2, len(pool))
            candidates = rng.sample(pool, k)
            gold = rng.randrange(k)
            is_null = rng.random() < 0.25
            target = [1.0 if j == gold else 0.0 for j in range(k)]
            examples.append({
                "state": state, "query": query, "candidates": candidates, "target": target,
                "p_null": 1.0 if is_null else 0.0, "task": task, "label": -1 if is_null else gold,
            })
    return examples


def prefetch(gen, cache, depth=3, readout="energy", shuffle=False, seed=0):
    """Yield (examples, collate(cache, examples)) with the collate done on a worker thread.
    shuffle (mcq/native readouts, PLAN4 sec 7): re-order every example's options first.
    ponytail: the shuffle rng is not checkpointed -- a resumed run re-draws option orders."""
    import queue, threading
    q = queue.Queue(maxsize=depth)
    rng = random.Random(seed)
    def work():
        try:
            for ex in gen:
                if shuffle:
                    ex = shuffle_options(ex, rng)
                batch = collate_mcq(ex) if readout in ("mcq", "native") else collate(cache, ex)
                q.put((ex, batch))
        except BaseException as e:  # a collate error must kill the run, not leave the main loop waiting at 0% GPU
            q.put(e)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, BaseException):
            raise item
        yield item


def group_units(examples):
    """Rows sharing meta.ms_group (scripts/multiset.py) form one unit, in first-appearance
    order; every other row is its own unit."""
    units, by_group = [], {}
    for ex in examples:
        g = ex.get("meta", {}).get("ms_group")
        if g is None:
            units.append([ex])
        elif g in by_group:
            by_group[g].append(ex)
        else:
            by_group[g] = [ex]
            units.append(by_group[g])
    return units


def data_generator(examples, bs, seed):
    """Infinite stream of length-bucketed batches: shuffle -> chunk (bs*50 units) -> sort by
    len(state) -> pack units into batches of <= bs rows -> shuffle batch order. Cycles forever.
    PLAN3 E3-ms: a multiset group (group_units) never straddles a batch, so decision_loss can
    pair every variant with its orig row; with no groups this is exactly the old per-row stream.
    ponytail: a group larger than bs becomes one oversize batch (cap --variants, not this)."""
    rng = random.Random(seed)
    units = group_units(examples)
    chunk_size = bs * 50
    while True:
        order = units[:]
        rng.shuffle(order)
        for i in range(0, len(order), chunk_size):
            chunk = sorted(order[i:i + chunk_size], key=lambda u: len(u[0]["state"]))
            batches, cur = [], []
            for u in chunk:
                if cur and len(cur) + len(u) > bs:
                    batches.append(cur)
                    cur = []
                cur.extend(u)
            if cur:
                batches.append(cur)
            rng.shuffle(batches)
            for b in batches:
                yield b


def base_backbone(backbone):
    """The underlying encode.Backbone: --readout mcq/native wrap it in MCQHead (attribute
    `.backbone`); --readout energy uses it directly."""
    return getattr(backbone, "backbone", backbone)


def bucket_for_dir(dirname, bucket_map):
    """PLAN6 Queue review: dir -> family bucket for --family_weights. --bucket_map overrides
    win; else data_kb*->K, data_wf*->W; anything else (including --data) falls back to E."""
    if dirname in bucket_map:
        return bucket_map[dirname]
    if dirname.startswith("data_kb"):
        return "K"
    if dirname.startswith("data_wf"):
        return "W"
    return "E"


def tag_bucket(rows, dirname, bucket_map):
    """Stamp meta.fam_bucket on rows lacking it (rows that already carry one, e.g. from a
    generator that assigns it per-row, are left alone)."""
    b = bucket_for_dir(dirname, bucket_map)
    for ex in rows:
        ex.setdefault("meta", {}).setdefault("fam_bucket", b)


def bucket_of(ex):
    return ex.get("meta", {}).get("fam_bucket", "E")


def parse_bucket_map(s):
    return dict(kv.split("=", 1) for kv in s.split(",")) if s else {}


NULL_AUG_CATCHALL = {"other", "not_stated", "skip", "none"}


def parse_null_aug(s):
    if not s:
        return None
    bucket, frac = s.split(":", 1)
    return bucket, float(frac)


def apply_null_aug(examples, bucket, frac, seed):
    """--null_aug BUCKET:FRAC (PLAN7 track B null control): for a seeded FRAC of `bucket` rows
    (meta.fam_bucket, see bucket_of/tag_bucket) with a hard gold label and >= 3 candidates, add an
    augmented COPY with the gold candidate -- and any rendered catch-all option (NULL_AUG_CATCHALL)
    -- removed, so the only correct read left is null: target uniform-zero over what remains,
    p_null=1.0, label=-1, meta.null_aug=True. The original row is untouched (not deleted, not
    mutated), so ms_group/rubric_group keep meaning unchanged rows they already had. A row is
    skipped (no copy) if removing the gold + catch-alls would leave under 2 candidates."""
    rng = random.Random(seed)
    eligible = [ex for ex in examples if bucket_of(ex) == bucket and ex.get("p_null", 0) == 0
                and isinstance(ex.get("label"), int) and 0 <= ex["label"] < len(ex["candidates"])
                and abs(ex["target"][ex["label"]] - 1.0) < 1e-6 and len(ex["candidates"]) >= 3]
    n = round(len(eligible) * frac)
    extra = []
    for ex in rng.sample(eligible, min(n, len(eligible))):
        cands = [c for i, c in enumerate(ex["candidates"]) if i != ex["label"]]
        cands = [c for c in cands if c.strip().lower() not in NULL_AUG_CATCHALL]
        if len(cands) < 2:
            continue
        aug = dict(ex, candidates=cands, target=[1.0 / len(cands)] * len(cands), p_null=1.0, label=-1)
        aug["meta"] = dict(ex.get("meta", {}), null_aug=True)
        extra.append(aug)
    return examples + extra


def parse_family_weights(s):
    return {k: float(v) for k, v in (kv.split(":", 1) for kv in s.split(","))} if s else None


def bucketed_data_generator(examples, bs, seed, weights):
    """PLAN6 Queue review: family-balanced sampler. One plain data_generator per bucket (so
    groups and length-bucketing are exactly data_generator's, per bucket) interleaved by a
    seeded RNG drawing from --family_weights; each yielded batch is therefore single-bucket."""
    buckets = {}
    for ex in examples:
        buckets.setdefault(bucket_of(ex), []).append(ex)
    missing = [b for b in weights if not buckets.get(b)]
    assert not missing, f"--family_weights names bucket(s) with no rows: {missing}"
    names = list(weights)
    print("family-balanced sampler: " + ", ".join(f"{b}={len(buckets[b])} rows (w={weights[b]})" for b in names))
    gens = {b: data_generator(buckets[b], bs, seed) for b in names}
    rng = random.Random(seed)
    ws = [weights[b] for b in names]
    while True:
        yield next(gens[rng.choices(names, weights=ws)[0]])


def add_extra_data(args, train_examples, eval_sets, tokenizer=None):
    """PLAN3 E3: mix --extra_data corpora (e.g. data_kb, closed-book MCQ with teacher labels) into
    training in place; each dir's val becomes an eval set (<dir>_val) so distillation quality is
    reported separately. multiset groups (meta.ms_group) are namespaced by dir so two expanded
    corpora never pair across files."""
    bucket_map = parse_bucket_map(args.bucket_map)
    for extra in (args.extra_data.split(",") if args.extra_data else []):
        ed = Path(extra)
        # a dir may hold one train.jsonl/val.jsonl or split train/*.jsonl, val/*.jsonl (data_wf_hf)
        load_split = lambda name: sum((load_jsonl(p) for p in sorted(glob.glob(str(ed / name / "*.jsonl")))), []) \
            if (ed / name).is_dir() else load_jsonl(ed / f"{name}.jsonl") if (ed / f"{name}.jsonl").exists() else []
        rows = load_split("train")
        assert rows, f"{ed}: no train rows"
        for ex in rows:
            if "ms_group" in ex.get("meta", {}):
                ex["meta"]["ms_group"] = f"{ed.name}/{ex['meta']['ms_group']}"
        if args.family_weights or args.null_aug:
            tag_bucket(rows, ed.name, bucket_map)
        if args.drop_truncated:
            rows = apply_drop_truncated(rows, tokenizer, args.max_state, ed.name)
        train_examples += rows
        # --eval_cap: head-N (not a random sample) so adjacent rubric-flip pairs stay intact (PLAN6 §W)
        cap = lambda rows: rows[:args.eval_cap] if args.eval_cap else rows
        if val := load_split("val"):
            eval_sets[f"{ed.name}_val"] = cap(val)
        eval_sets.update({Path(p).stem: cap(load_jsonl(p)) for p in sorted(glob.glob(str(ed / "eval" / "*.jsonl")))})


def load_run_data(args, backbone):
    # readout=mcq never touches the cached frozen-space candidate features (candidates
    # are read straight into the prompt text) -> skip building/loading that cache entirely.
    # cand_encoder=tiny caches token ids only (TokenCandCache, built below; the encoder is
    # trained inside the model) -> no frozen-feature cache either.
    # readout=native needs the embedding VecCache only for n2 heads (n3 pools the suffix itself).
    build_cache_ = (args.readout == "energy" and args.cand_encoder != "tiny") or \
        (args.readout == "native" and "n2" in args.nc_head)
    use_vec = args.cand_encoder == "qwen3emb"
    if args.smoke:
        train_examples = make_smoke_examples(64, seed=0)
        val_examples = make_smoke_examples(16, seed=1)
        eval_sets = {"smoke_eval": make_smoke_examples(16, seed=2)}
        if args.family_weights or args.null_aug:
            tag_bucket(train_examples, Path(args.data).name, parse_bucket_map(args.bucket_map))
        add_extra_data(args, train_examples, eval_sets, tokenizer=base_backbone(backbone).tokenizer)  # e.g. a tiny multiset file for an E3-ms smoke
        cache = None
        if build_cache_:
            all_ex = train_examples + val_examples + eval_sets["smoke_eval"]
            if use_vec:
                from encode import EmbedEncoder, VecCache, CAND_RENDER
                encoder = EmbedEncoder(device=backbone.device)
                cache = VecCache(device="cpu")
                cache.add(encoder, sorted({c for ex in all_ex for c in ex["candidates"]}), render=CAND_RENDER, max_len=32)
                cache.add(encoder, sorted({ex["state"] for ex in all_ex}), max_len=128)
                cache.add(encoder, sorted({ex["query"] for ex in all_ex}), max_len=32)
            else:
                cache = FeatureCache(device="cpu")
                cache.add(backbone, sorted({c for ex in all_ex for c in ex["candidates"]}), max_len=16)
    else:
        data_dir = Path(args.data)
        train_examples = load_jsonl(data_dir / "train.jsonl")
        val_examples = load_jsonl(data_dir / "val.jsonl")
        eval_sets = {Path(p).stem: load_jsonl(p) for p in sorted(glob.glob(str(data_dir / "eval" / "*.jsonl")))}
        if args.family_weights or args.null_aug:
            tag_bucket(train_examples, data_dir.name, parse_bucket_map(args.bucket_map))
        if args.drop_truncated:
            train_examples = apply_drop_truncated(train_examples, base_backbone(backbone).tokenizer,
                                                   args.max_state, data_dir.name)
        add_extra_data(args, train_examples, eval_sets, tokenizer=base_backbone(backbone).tokenizer)

        cache = None
        if build_cache_ and use_vec:
            from encode import build_vec_cache, VecCache
            cache_path = data_dir / "veccache_qwen3emb.pt"
            if not cache_path.exists():
                build_vec_cache(str(data_dir), str(cache_path))
            cache = VecCache.load(str(cache_path), "cpu")

            # states/queries/candidates all live in this one cache -> top up with everything,
            # same "may predate new rows" reasoning as the frozen-feature cache below.
            # ... including --extra_data rows, which build_vec_cache(data_dir) would never see: embed
            # exactly the missing strings per category (candidates are rendered, states/queries raw).
            cands, states, queries = set(), set(), set()
            for exs in (train_examples, val_examples, *eval_sets.values()):
                for ex in exs:
                    cands.update(ex["candidates"]); states.add(ex["state"]); queries.add(ex["query"])
            have = cache.index.keys()
            missing = [sorted(s - have) for s in (cands, states, queries)]
            if any(missing):
                from encode import EmbedEncoder, CAND_RENDER
                enc = EmbedEncoder(device=backbone.device)
                cache.add(enc, missing[0], render=CAND_RENDER, max_len=32)
                cache.add(enc, missing[1], max_len=128)
                cache.add(enc, missing[2], max_len=32)
                cache.save(str(cache_path))
                print(f"vec cache topped up: +{len(missing[0])} candidates, +{len(missing[1])} states, +{len(missing[2])} queries")
                del enc
        elif build_cache_:
            # candidate features live in the backbone's frozen layer-L space -> one cache per backbone
            cache_path = data_dir / f"cache_{args.backbone.split('/')[-1]}_L{backbone.split_layer}.pt"
            if not cache_path.exists():
                from encode import build_cache
                build_cache(str(data_dir), str(cache_path), backbone)
            cache = FeatureCache.load(str(cache_path), "cpu")

            # the on-disk cache may predate rows added to train/val/eval since it was built;
            # top it up (build_cache merges into the existing file) rather than KeyError-ing
            # the first time collate() hits a candidate string it hasn't seen.
            all_cands = {c for ex in train_examples + val_examples for c in ex["candidates"]}
            all_cands.update(c for exs in eval_sets.values() for ex in exs for c in ex["candidates"])
            if all_cands - cache.index.keys():
                from encode import build_cache
                build_cache(str(data_dir), str(cache_path), backbone)
                cache = FeatureCache.load(str(cache_path), "cpu")

        if args.eval_limit:  # smoke runs: first N rows of every eval set
            eval_sets = {k: v[:args.eval_limit] for k, v in eval_sets.items()}
            val_examples = val_examples[:args.eval_limit]

    if args.null_aug:
        bucket, frac = parse_null_aug(args.null_aug)
        train_examples = apply_null_aug(train_examples, bucket, frac, args.seed)

    if args.readout == "energy" and args.cand_encoder == "tiny":
        from encode import TokenCandCache
        cache = TokenCandCache(backbone)
        cache.add(sorted({c for exs in (train_examples, val_examples, *eval_sets.values()) for ex in exs for c in ex["candidates"]}))

    if args.qtype_filter:
        train_examples = apply_qtype_filter(train_examples, args.qtype_filter)
        val_examples = apply_qtype_filter(val_examples, args.qtype_filter)
        assert train_examples, f"--qtype_filter {args.qtype_filter}: no train rows"
        # eval_sets: filter mixed-qtype files down to this arm's rows too (PLAN7 track C's own
        # eval-set list, e.g. "typed_decisions_test (filter score / noul rows)") -- required for
        # a matched comparison, and --score_head cumlink still structurally can't score a row of
        # the wrong qtype (it globally replaces the K-way scorer, no letters/candidates to fall
        # back on). --noul_head bern routes per row now (native._is_bern_row) so it no longer
        # needs this to avoid crashing, but the filter still gives every arm the same rows. Sets
        # left with zero matching rows (e.g. cua_s1_forms under qtype_filter=score) are dropped.
        eval_sets = {k: f for k, v in eval_sets.items() if (f := apply_qtype_filter(v, args.qtype_filter))}
    if args.ordinal_smooth:
        train_examples = apply_ordinal_smooth(train_examples, args.ordinal_smooth)

    if args.mix == "nlionly":
        train_examples = [ex for ex in train_examples if ex["task"] in NLI_ONLY_TASKS]
    if args.hard_only:
        train_examples = apply_hard_only(train_examples)
        val_examples = apply_hard_only(val_examples)
    if args.no_null:
        train_examples = apply_no_null(train_examples)
        val_examples = apply_no_null(val_examples)
    return cache, train_examples, val_examples, eval_sets


def probs_from_logits(logits, cmask, T=1.0, null="softmax", b=0.0):
    """b: additive offset on logit(P(null)) -- the eval-time null-logit-offset knob (REPORT
    S3w calibration experiment #1). For the plain-softmax null, P(null) = sigmoid(s_null/T -
    logsumexp(s_valid/T)), i.e. a two-outcome softmax between the null score and everything
    else -- so adding b to the (already-tempered) null score IS adding b to logit(P(null)),
    with no change to the relative odds among candidates. For the factored null, r is composed
    T-invariantly (see below) from log_r/log_1mr, so the same b is added directly to
    logit(r) = log_r - log_1mr. One offset, same operation, both null forms -- that's why a
    single --null_offset covers either --null softmax or --null factored.
    b=0.0 (default) reproduces the pre-offset probs_from_logits exactly."""
    null_col = torch.ones(cmask.shape[0], 1, dtype=torch.bool, device=cmask.device)
    valid = torch.cat([cmask, null_col], dim=1)
    if null == "factored" and (T != 1.0 or b != 0.0):
        # factored null composes p_j = softmax(s/T) *inside* the model (DecisionModel.
        # _factored_logits) -- T can't be applied to the already-composed logits like the
        # plain-softmax null below. But r=P(null) only sees untempered s-statistics, so it's
        # exactly T-invariant and recoverable unchanged from these (T=1) logits; only the K
        # candidate columns need re-softmaxing at the new T. This algebraic identity lets the
        # T grid search in fit_temperature reuse one forward pass instead of one per T.
        K = logits.shape[-1] - 1
        log_r = logits[:, K]
        log_1mr = torch.log1p(-log_r.exp().clamp(max=1 - 1e-7))
        if b != 0.0:
            logit_r = log_r - log_1mr + b
            log_r, log_1mr = F.logsigmoid(logit_r), F.logsigmoid(-logit_r)
        adj = (logits[:, :K] - log_1mr.unsqueeze(-1)).masked_fill(~cmask, float("-inf"))
        logp = torch.log_softmax(adj / T, dim=-1)
        cand = (log_1mr.unsqueeze(-1) + logp).exp().masked_fill(~cmask, 0.0)
        return torch.cat([cand, log_r.exp().unsqueeze(-1)], dim=-1)
    masked = logits.masked_fill(~valid, float("-inf")) / T
    if b != 0.0:
        masked = torch.cat([masked[:, :-1], masked[:, -1:] + b], dim=-1)
    return torch.softmax(masked, dim=-1)


def full_target_from(target, p_null):
    return torch.cat([(1 - p_null).unsqueeze(-1) * target, p_null.unsqueeze(-1)], dim=-1)


def run_readout(readout, backbone, model, batch, examples, joint=False, vec_cache=None, shots=0, max_state=256):
    """One forward through whichever readout is configured -> logits [B, Kmax+1].
    max_state (--max_state, native only): decision-state truncation length; 256 = today's default."""
    if readout == "mcq":
        return run_batch_mcq(backbone, batch, examples, shots=shots)
    if readout == "native":
        return run_batch_native(backbone, model, batch, examples, max_state=max_state, vec_cache=vec_cache)
    return run_batch(backbone, model, batch, joint=joint, vec_cache=vec_cache)


def forward_batches(backbone, model, cache, examples, bs, joint=False, readout="energy", vec_cache=None, shots=0,
                     max_state=256):
    """(logits, cmask, target, p_null, batch) per batch, no grad, single forward each."""
    model.eval(); backbone.eval()  # LoRALinear modules default to train mode (LoRA dropout active at eval otherwise)
    out = []
    with torch.inference_mode():
        for i in range(0, len(examples), bs):
            ex_batch = examples[i:i + bs]
            batch = collate_mcq(ex_batch) if readout in ("mcq", "native") else collate(cache, ex_batch)
            logits = run_readout(readout, backbone, model, batch, ex_batch, joint=joint, vec_cache=vec_cache,
                                 shots=shots, max_state=max_state)
            # run_batch only moves state/query/C to backbone.device internally; target/p_null/cmask
            # stay on the collate()-produced CPU tensors, so callers must move them themselves.
            dev = logits.device
            out.append((logits, batch["cmask"].to(dev), batch["target"].to(dev), batch["p_null"].to(dev), ex_batch))
    model.train(); backbone.train()  # Qwen3 has no dropout, so train mode only re-enables LoRA dropout
    return out


def eval_val_loss(backbone, model, cache, examples, bs, joint=False, readout="energy", vec_cache=None, shots=0,
                   max_state=256):
    total, n = 0.0, 0
    for logits, cmask, target, p_null, ex_batch in forward_batches(backbone, model, cache, examples, bs,
                                                                     joint=joint, readout=readout, vec_cache=vec_cache,
                                                                     shots=shots, max_state=max_state):
        loss = decision_loss(logits, target, p_null, cmask)
        total += loss.item() * len(ex_batch)
        n += len(ex_batch)
    return total / max(n, 1)


def fit_temperature(backbone, model, cache, val_examples, bs, joint=False, readout="energy", vec_cache=None,
                     shots=0, b_grid=None, max_state=256):
    # ponytail: this doesn't set model.temperature even though DecisionModel has the attribute
    # (used by null="factored"'s forward composition) -- doing so would leak into eval_dataset's
    # own forward pass right after (which wants T=1 cached logits to sweep both raw and scaled
    # metrics via probs_from_logits) and into the next training step's run_batch (which doesn't
    # go through forward_batches at all). T stays a reporting-time overlay; a deployment script
    # that wants it baked into forward() can set model.temperature = fit_temperature(...)[0] itself.
    #
    # b_grid (REPORT S3w calib #1): candidate null-logit-offset values to search jointly with T,
    # minimising NLL on val_examples (or whatever calibration set the caller passes in). None
    # (default) fixes b=0 and searches T alone over exactly the same grid as before -- so every
    # existing caller reproduces its old (T, nll) numbers exactly; only the return arity grew.
    null_mode = getattr(model, "null", "softmax")
    batches = forward_batches(backbone, model, cache, val_examples, bs, joint=joint, readout=readout, vec_cache=vec_cache,
                              shots=shots, max_state=max_state)
    b_candidates = [0.0] if b_grid is None else list(b_grid)
    best_T, best_b, best_nll = 1.0, 0.0, float("inf")
    for T in np.geomspace(0.1, 10, 60):
        for off in b_candidates:
            total, n = 0.0, 0
            for logits, cmask, target, p_null, _ in batches:
                probs = probs_from_logits(logits, cmask, T, null=null_mode, b=off)
                tgt = full_target_from(target, p_null)
                nll = -(tgt * torch.log(probs.clamp_min(1e-12))).sum(-1)
                total += nll.sum().item()
                n += nll.shape[0]
            avg = total / n
            if avg < best_nll:
                best_nll, best_T, best_b = avg, float(T), float(off)
    return best_T, best_b, best_nll


def eval_dataset(backbone, model, cache, examples, bs, Ts=(1.0,), joint=False, readout="energy", vec_cache=None,
                  shots=0, max_state=256):
    """Ts: iterable of T floats, or (T, offset) pairs -- bare floats default offset to 0.0.
    probs_all is keyed by T alone (as before the offset knob existed), so every existing
    `probs[T]` lookup keeps working unchanged.
    # ponytail: if T itself lands exactly on 1.0 with a nonzero offset, that entry's key
    # collides with the raw (1.0, offset=0) entry and the offset one wins (dict overwrite) --
    # a pre-existing risk of this T-keyed design (harmless before b existed, since two b=0
    # entries at the same T are identical); not worth a bigger key for a probability-zero grid
    # coincidence. Give T and offset separately (not equal T) if that ever matters.
    """
    null_mode = getattr(model, "null", "softmax")
    Ts_pairs = [t if isinstance(t, tuple) else (t, 0.0) for t in Ts]
    n = len(examples)
    kmax = max(len(ex["candidates"]) for ex in examples)
    probs_all = {T: np.zeros((n, kmax + 1)) for T, _ in Ts_pairs}
    target_all = np.zeros((n, kmax + 1))
    label_all = np.full(n, np.nan)
    idx = 0
    for logits, cmask, target, p_null, ex_batch in forward_batches(backbone, model, cache, examples, bs,
                                                                     joint=joint, readout=readout, vec_cache=vec_cache,
                                                                     shots=shots, max_state=max_state):
        tgt = full_target_from(target, p_null).cpu().numpy()
        k_local = target.shape[1]
        bsz = len(ex_batch)
        for T, off in Ts_pairs:
            probs = probs_from_logits(logits, cmask, T, null=null_mode, b=off).cpu().numpy()
            probs_all[T][idx:idx + bsz, :k_local] = probs[:, :k_local]
            probs_all[T][idx:idx + bsz, -1] = probs[:, -1]
        target_all[idx:idx + bsz, :k_local] = tgt[:, :k_local]
        target_all[idx:idx + bsz, -1] = tgt[:, -1]
        for j, ex in enumerate(ex_batch):
            label_all[idx + j] = ex["label"] if ex.get("label") is not None else np.nan
        idx += bsz
    return probs_all, target_all, label_all


def dump_logits(backbone, model, cache, examples, bs, joint=False, readout="energy", vec_cache=None, shots=0,
                 max_state=256):
    """Like eval_dataset but keeps raw (pre-softmax, pre-T) logits -- padded with finfo.min --
    plus the real candidate count K per row, for scripts/null_bias.py to fit a post-hoc
    K-aware null-bias correction (s_null' = s_null + alpha*log(K) + beta) without retraining."""
    n = len(examples)
    kmax = max(len(ex["candidates"]) for ex in examples)
    logits_all = np.full((n, kmax + 1), np.finfo(np.float64).min)
    target_all = np.zeros((n, kmax + 1))
    label_all = np.full(n, np.nan)
    K_all = np.zeros(n, dtype=np.int64)
    idx = 0
    for logits, cmask, target, p_null, ex_batch in forward_batches(backbone, model, cache, examples, bs,
                                                                     joint=joint, readout=readout, vec_cache=vec_cache,
                                                                     shots=shots, max_state=max_state):
        tgt = full_target_from(target, p_null).cpu().numpy()
        lg = logits.cpu().numpy()
        k_local = target.shape[1]
        b = len(ex_batch)
        logits_all[idx:idx + b, :k_local] = lg[:, :k_local]
        logits_all[idx:idx + b, -1] = lg[:, -1]
        target_all[idx:idx + b, :k_local] = tgt[:, :k_local]
        target_all[idx:idx + b, -1] = tgt[:, -1]
        K_all[idx:idx + b] = cmask.sum(-1).cpu().numpy()
        for j, ex in enumerate(ex_batch):
            label_all[idx + j] = ex["label"] if ex.get("label") is not None else np.nan
        idx += b
    return logits_all, target_all, label_all, K_all


def dump_eval_logits(args, backbone, model, cache, val_examples, eval_sets):
    """--dump_logits DIR: for val and every eval set, write DIR/<set>.npz (logits/target/label/K)
    and DIR/<set>.meta.json (the ordered examples, minus the "state" text, for choice_set_effects
    /ksweep which read meta.pair/variant/K/u/present and candidates off them)."""
    vec_cache = cache if args.cand_encoder == "qwen3emb" else None
    out_dir = Path(args.dump_logits)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in {"val": val_examples, **eval_sets}.items():
        logits, target, label, K = dump_logits(backbone, model, cache, examples, args.eval_bs,
                                                 joint=args.joint, readout=args.readout, vec_cache=vec_cache,
                                                 shots=args.shots, max_state=args.max_state)
        np.savez(out_dir / f"{name}.npz", logits=logits, target=target, label=label, K=K)
        meta = [{k: v for k, v in ex.items() if k != "state"} for ex in examples]
        with open(out_dir / f"{name}.meta.json", "w") as f:
            json.dump(meta, f)


def checkpoint_metric(backbone, model, cache, eval_sets, best_on, args, vec_cache, val_nll, wb, step):
    """--best_on (tl1b item 4): the metric checkpoint selection watches. None (default) ->
    val_nll, today's behaviour unchanged. Else the unweighted mean NLL of the named eval sets
    (val_nll is still printed/logged by the caller either way -- this only changes what best.pt
    tracks)."""
    if not best_on:
        return val_nll
    per_set = {name: eval_val_loss(backbone, model, cache, eval_sets[name], args.eval_bs, joint=args.joint,
                                    readout=args.readout, vec_cache=vec_cache, shots=args.shots)
               for name in best_on}
    metric = sum(per_set.values()) / len(per_set)
    print(f"step {step} best_on_nll {metric:.4f} (" + ", ".join(f"{k}={v:.4f}" for k, v in per_set.items()) + ")")
    if wb:
        wb.log({"val/best_on_nll": metric, **{f"val/best_on/{k}": v for k, v in per_set.items()}}, step=step)
    return metric


def resolve_best_on(best_on_arg, eval_sets):
    """--best_on NAME[,NAME...] (tl1b item 4): checkpoint-selection eval sets, e.g.
    'data_u_val,data_wh_val' -- names as they appear in eval_sets (add_extra_data names an
    extra dir's val split '<dir>_val'). None (default) -> None, today's in-distribution
    val_nll selection is untouched."""
    if not best_on_arg:
        return None
    names = [n.strip() for n in best_on_arg.split(",") if n.strip()]
    missing = [n for n in names if n not in eval_sets]
    assert not missing, f"--best_on: unknown eval set(s) {missing}; available: {sorted(eval_sets)}"
    return names


def resolve_calib_examples(calib_sets_arg, val_examples, eval_sets):
    """--calib_sets NAME[,NAME...] (REPORT S3w calib #1): concat named eval sets ("val" is an
    alias for val_examples) into one calibration pool for the joint (T, null_offset) fit.
    Unset (None) -> (val_examples, ["val"], None) so run_full_eval's default path is untouched:
    None as the third element tells fit_temperature to skip the b grid (b fixed at 0)."""
    if not calib_sets_arg:
        return val_examples, ["val"], None
    names = [n.strip() for n in calib_sets_arg.split(",") if n.strip()]
    lookup = {"val": val_examples, **eval_sets}
    missing = [n for n in names if n not in lookup]
    assert not missing, f"--calib_sets: unknown eval set(s) {missing}; available: {sorted(lookup)}"
    examples = sum((lookup[n] for n in names), [])
    return examples, names, np.arange(-4.0, 4.0 + 1e-9, 0.25)


def run_full_eval(args, backbone, model, cache, val_examples, eval_sets, val_nll):
    vec_cache = cache if args.cand_encoder == "qwen3emb" else None
    calib_examples, calib_names, b_grid = resolve_calib_examples(getattr(args, "calib_sets", None), val_examples, eval_sets)
    best_T, best_b, _ = fit_temperature(backbone, model, cache, calib_examples, args.eval_bs, joint=args.joint,
                                         readout=args.readout, vec_cache=vec_cache, shots=args.shots, b_grid=b_grid,
                                         max_state=args.max_state)
    # args: so a run is reproducible from results.json alone
    results = {"T": best_T, "null_offset": best_b, "calib_sets": calib_names, "val_nll": val_nll,
               "args": vars(args), "eval": {}}
    for name, examples in eval_sets.items():
        probs, target, label = eval_dataset(backbone, model, cache, examples, args.eval_bs,
                                             Ts=(1.0, (best_T, best_b)),
                                             joint=args.joint, readout=args.readout, vec_cache=vec_cache,
                                             shots=args.shots, max_state=args.max_state)
        raw_m, scaled_m = summarize(probs[1.0], target, label), summarize(probs[best_T], target, label)
        if args.no_null:
            raw_m["auroc_null"] = scaled_m["auroc_null"] = "n/a"
        results["eval"][name] = {"raw": raw_m, "scaled": scaled_m}
        if name.startswith("cse_"):      # paired choice-set-effect battery (data v4)
            results["eval"][name]["cse"] = choice_set_effects(probs[best_T], examples)
        elif name.startswith("ksweep_"):  # P(null) as a function of K
            results["eval"][name]["ksweep"] = ksweep(probs[best_T], examples)
        elif name.startswith("mmlu_cf"):  # PLAN3 E3-cf counterfactual option-set battery (scripts/mmlu_counterfactual.py)
            results["eval"][name]["cf"] = counterfactual(probs[best_T], examples)
        elif name.startswith("wf_rubric_flip"):  # PLAN6 rubric-group probe (scripts/workflow_corpus.py)
            results["eval"][name]["flip"] = rubric_flip(probs[best_T], examples)

        # PLAN7 track C: ordinal MAE / expected-score error on whichever rows of this (possibly
        # mixed-qtype) set are meta.qtype == "score" -- content-keyed, not name-keyed, since
        # typed_decisions_test/systemone_lite_hard/etc. mix choice/noul/score rows in one file.
        score_idx = [j for j, ex in enumerate(examples) if ex.get("meta", {}).get("qtype") == "score"]
        if score_idx:
            results["eval"][name]["ordinal"] = ordinal_metrics(probs[best_T][score_idx],
                                                                [examples[j] for j in score_idx])
        # Reversed-label control on this set's noul rows (candidate order ["yes","no"] vs
        # ["no","yes"]): run for BOTH noul arms, not just --noul_head bern, so the matched
        # comparison is explicit in results.json -- bern's P(yes) is untouched (the suffix
        # never renders candidates, see native._render_query_only), while the K-way Choice
        # control (noul_A_2way) is expected to move, since it does read the rendered letters.
        if args.readout == "native":
            noul_idx = [j for j, ex in enumerate(examples) if ex.get("meta", {}).get("qtype") == "noul"]
            if noul_idx:
                noul_ex = [examples[j] for j in noul_idx]
                rev_ex = [dict(ex, candidates=list(reversed(ex["candidates"])), target=list(reversed(ex["target"])))
                          for ex in noul_ex]
                rev_probs, _, _ = eval_dataset(backbone, model, cache, rev_ex, args.eval_bs, Ts=((best_T, best_b),),
                                               joint=args.joint, readout=args.readout, vec_cache=vec_cache,
                                               shots=args.shots, max_state=args.max_state)
                results["eval"][name]["noul_reversed"] = noul_reversed_check(
                    probs[best_T][noul_idx], noul_ex, rev_probs[best_T], rev_ex)
    return results


def calibrate_zscore_native(head, model, examples, seed, n=256, max_state=256):
    """NativeHead.mu_h/sd_h from n training rows' decision states, mu_c/sd_c from their pooled
    option spans (valid options + the null line) -- same rogue-dim reasoning as calibrate_zscore.
    max_state must match the run's --max_state so the calibration states aren't truncated
    differently than training/eval."""
    sample = random.Random(seed).sample(examples, min(n, len(examples)))
    hs, cs = [], []
    with torch.inference_mode():
        for i in range(0, len(sample), 32):
            chunk = sample[i:i + 32]
            h, C3, cmask, _ = native_features(head, [ex["state"] for ex in chunk], [ex["query"] for ex in chunk],
                                              [ex["candidates"] for ex in chunk], max_state=max_state, render=model.render)
            valid = torch.cat([cmask, torch.ones_like(cmask[:, :1])], 1)
            hs.append(h); cs.append(C3[valid])
    model.calibrate(torch.cat(hs), torch.cat(cs))


def calibrate_zscore(backbone, model, examples, cache, seed, joint=False, n=256, vec_cache=None):
    """mu_top/sd_top: n states + n queries (top-layer LoRA-adapted features).
    mu_fz/sd_fz: same states + queries in frozen space, plus 512 cached candidate
    vectors -- C is compared against Uf/Vf in frozen space, so it needs to be covered
    by the same stats (fixes the v1 defect: 512 STATE texts only, candidates never seen).
    vec_cache (--cand_encoder qwen3emb): frozen space IS the embedding cache, so mu_fz/sd_fz
    sample straight from it (states+queries+candidates) instead of a backbone forward pass."""
    rng = random.Random(seed)
    sample = rng.sample(examples, min(n, len(examples)))
    tops, fzs = [], []
    with torch.inference_mode():
        for i in range(0, len(sample), 64):
            chunk = sample[i:i + 64]
            states, queries = [ex["state"] for ex in chunk], [ex["query"] for ex in chunk]
            if joint:
                ids, am, ls, lq = backbone.tokenize_joint(states, queries, 256, 64)
                h_top, h_fz, mask = backbone(ids.to(backbone.device), am.to(backbone.device))
                H, hmask, Q, qmask, Hf, Qf = split_joint(h_top, h_fz, mask, ls, lq)
            else:
                ids_s, am_s = backbone.tokenize(states, 256)
                ids_q, am_q = backbone.tokenize(queries, 64)
                H, Hf, hmask = backbone(ids_s.to(backbone.device), am_s.to(backbone.device))[:3]
                Q, Qf, qmask = backbone(ids_q.to(backbone.device), am_q.to(backbone.device))[:3]
            tops.append(H[hmask]); tops.append(Q[qmask])
            if vec_cache is None:
                fzs.append(Hf[hmask]); fzs.append(Qf[qmask])
    if vec_cache is not None:
        fz_texts = rng.sample(sorted(vec_cache.index), min(n * 2 + 512, len(vec_cache.index)))
        fzs.append(torch.stack([vec_cache.pooled(t) for t in fz_texts]).to(backbone.device))
    else:
        cand_sample = rng.sample(sorted(cache.index), min(512, len(cache.index)))
        fzs.append(torch.stack([cache.pooled(c) for c in cand_sample]).to(backbone.device))
    model.calibrate(torch.cat(tops), torch.cat(fzs))


def device_memory(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    if device == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def per_family_loss(logits, batch, examples):
    with torch.no_grad():
        t = full_target_from(batch["target"], batch["p_null"])
        logp = torch.log_softmax(logits, dim=-1)
        per_ex = (-(t * logp).sum(-1)).tolist()
    fam = {}
    for ex, l in zip(examples, per_ex):
        fam.setdefault(family_of(ex["task"]), []).append(l)
    return {k: sum(v) / len(v) for k, v in fam.items()}


def build_optimizer(model, backbone, args):
    opt = AdamW([
        {"params": model.parameters(), "lr": args.lr, "weight_decay": 0.01},
        {"params": backbone.trainable_parameters(), "lr": args.lora_lr, "weight_decay": 0.0},
    ])
    warmup_steps = min(500, max(1, args.steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, args.steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return opt, LambdaLR(opt, lr_lambda)


def rng_state_dict():
    d = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        d["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        d["mps"] = torch.mps.get_rng_state()
    return d


def restore_rng(d):
    # the CPU default generator needs a CPU ByteTensor even if the checkpoint was
    # loaded with map_location="mps"/"cuda" (which moves every tensor in the file).
    torch.set_rng_state(d["torch"].cpu())
    random.setstate(d["python"])
    if "cuda" in d and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in d["cuda"]])
    if "mps" in d and torch.backends.mps.is_available():
        torch.mps.set_rng_state(d["mps"].cpu())


def save_checkpoint(path, model, backbone, step, opt, sched, best_val, args, tower_lora_only=False):
    ckpt = {"tower": model.state_dict(), "lora": backbone.lora_state_dict(), "step": step,
            "best_val": best_val, "args": vars(args)}
    if not tower_lora_only:
        ckpt.update({"opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng_state_dict()})
    torch.save(ckpt, path)


def upload_ckpt(args, run_dir):
    """--ckpt_upload: push last.pt/best.pt to hf_repo/<name>/ so a pod that gets pre-empted (no local
    volume) can resume elsewhere -- the boot script hf-downloads <name>/last.pt before this process
    starts, and main()'s own last_path.exists() check resumes from it. Best-effort: a transient upload
    failure shouldn't kill an otherwise-healthy training step."""
    import huggingface_hub
    for fname in ("last.pt", "best.pt"):
        p = run_dir / fname
        if not p.exists():
            continue
        try:
            huggingface_hub.upload_file(path_or_fileobj=str(p), path_in_repo=f"{args.name}/{fname}",
                                        repo_id=args.hf_repo, repo_type="model")
        except Exception as e:  # noqa: BLE001
            print(f"warning: ckpt upload failed ({fname}): {e}")


def load_checkpoint(path, model, backbone, opt, sched, device):
    # weights_only=False: trusted, self-produced checkpoint (opt/sched/rng state, not third-party input)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["tower"])
    backbone.load_lora_state_dict(ckpt["lora"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    restore_rng(ckpt["rng"])
    return ckpt["step"], ckpt["best_val"]


def fmt(v, width=8, prec=3):
    if v is None:
        return f"{'-':>{width}}"
    if v == "n/a":
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{prec}f}"


def print_table(results):
    calib_note = f"  null_offset={results['null_offset']:+.2f} (calib_sets={','.join(results['calib_sets'])})" \
        if results.get("null_offset") else ""
    print(f"\nT={results['T']:.3f}  val_nll={results['val_nll']:.4f}{calib_note}")
    print(f"{'set':<20}{'acc':>8}{'acc_k':>8}{'nll':>8}{'brier':>8}{'ece':>8}{'auroc_null':>12}")
    for name, ev in results["eval"].items():
        m = ev["scaled"]
        print(f"{name:<20}{fmt(m.get('acc'))}{fmt(m.get('acc_k'))}{fmt(m.get('nll'))}"
              f"{fmt(m.get('brier'))}{fmt(m.get('ece'))}{fmt(m.get('auroc_null'), width=12)}")


def init_wandb(args):
    import wandb
    wandb.init(project="pcdm", name=args.name, id=args.name, resume="allow", config=vars(args))
    return wandb


def log_eval_wandb(wb, results, step):
    if not wb:
        return
    log = {"T": results["T"]}
    for name, ev in results["eval"].items():
        for k, v in ev["scaled"].items():
            if v != "n/a":
                log[f"eval/{name}/{k}"] = v
        for k, v in ev["raw"].items():
            if v != "n/a":
                log[f"eval_raw/{name}/{k}"] = v
        for extra in ("cse", "ksweep", "flip"):
            for k, v in ev.get(extra, {}).items():
                if isinstance(v, (int, float)):
                    log[f"{extra}/{name}/{k}"] = v
    wb.log(log, step=step)


def finish_wandb(wb, results):
    if not wb:
        return
    cols = ["set", "acc", "acc_k", "nll", "brier", "ece", "auroc_null"]
    num = lambda v: float(v) if isinstance(v, (int, float)) else float("nan")  # wandb.Table needs one type per column
    rows = [[name] + [num(ev["scaled"].get(c)) for c in cols[1:]] for name, ev in results["eval"].items()]
    wb.log({"report": wb.Table(columns=cols, data=rows)})
    wb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--backbone", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--lora_layers", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--bs", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=1,
                    help="split --bs into --bs/--grad_accum micro-batches, accumulate gradients over "
                         "them, one optimizer step per outer step -- keeps the effective batch and "
                         "total step count identical while cutting peak activation memory (larger "
                         "backbones / long W-family sequences at bs=64 can OOM an 80GB GPU); "
                         "1 = off (default, unchanged behavior)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_hybrid", action="store_true")
    p.add_argument("--no_cand_null", action="store_true")
    p.add_argument("--no_null", action="store_true")
    p.add_argument("--null", choices=["softmax", "factored"], default="softmax",
                    help="factored: r=sigmoid(g(z)) over O(K) candidate-score set statistics, "
                         "P(a_j)=(1-r)*softmax(s)_j, P(null)=r (exact IIA); softmax: current behavior")
    p.add_argument("--hard_only", action="store_true")
    p.add_argument("--mix", choices=["full", "nlionly"], default="full")
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=None)
    p.add_argument("--eval_bs", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--hf_repo", default=None)
    p.add_argument("--ckpt_upload", action="store_true",
                    help="push last.pt/best.pt (LoRA+tower only, small) to --hf_repo/<name>/ at every "
                         "--ckpt_every, so a pre-empted pod can resume: boot script hf-downloads "
                         "<name>/last.pt into runs/<name>/ before this invocation, and the existing "
                         "last_path.exists() check above picks it up")
    p.add_argument("--data", default="data")
    p.add_argument("--extra_data", default=None, help="comma-separated extra data dirs mixed into train (their val/eval become eval sets)")
    p.add_argument("--eval_cap", type=int, default=None, help="head-N cap per --extra_data val/eval set (big external evals run post-hoc via scripts/eval_wf.py)")
    p.add_argument("--calib_sets", default=None,
                    help="comma-separated eval-set names (as in results.json['eval'], plus 'val') to fit T and "
                         "the null_offset b jointly on by grid search; unset = today's behaviour (val only, b=0)")
    p.add_argument("--best_on", default=None,
                    help="tl1b item 4: comma-separated eval-set names (as in results.json['eval'], e.g. "
                         "'data_u_val,data_wh_val') -- best.pt is picked on the unweighted mean NLL of these sets "
                         "instead of the in-distribution val_nll (which is still logged/printed unchanged)")
    p.add_argument("--family_weights", default=None,
                   help="PLAN6 Queue review: e.g. 'E:0.35,K:0.25,W:0.40' -- family-balanced sampler over "
                        "meta.fam_bucket (else --data->E, a data_kb*/data_wf* --extra_data dir->K/W); "
                        "default None = today's uniform concatenation, byte-identical stream")
    p.add_argument("--bucket_map", default=None,
                   help="--family_weights only: comma-separated dirname=bucket overrides, e.g. data_kbt=K,data_wf_hf=W")
    p.add_argument("--null_aug", default=None,
                   help="PLAN7 track B null control: 'BUCKET:FRAC', e.g. 'W:0.20' -- for a seeded FRAC of that "
                        "meta.fam_bucket's hard-labelled, >=3-candidate rows, add a copy with the gold candidate "
                        "(and any rendered catch-all option -- other/not_stated/skip/none) removed so only null "
                        "is correct (p_null=1.0, label=-1, meta.null_aug=True); the original row is kept as-is")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval_limit", type=int, default=0, help="cap each eval set (smoke runs)")
    p.add_argument("--no_final_eval", action="store_true", help="skip the closing eval of best.pt over every eval set (~10 min on an H100)")
    p.add_argument("--tap_layer", type=int, default=0, help="tower memory from layer T (0 = last layer)")
    p.add_argument("--zscore", action="store_true", help="per-dim standardise backbone features (stats from 512 train states)")
    p.add_argument("--grad_ckpt", action="store_true",
                    help="tl1b: backbone.model.gradient_checkpointing_enable (use_reentrant=False) -- trades "
                         "recompute for activation memory on long W-family states at large backbones; LoRA "
                         "layers still get grads (non-reentrant checkpointing needs no frozen-embedding trick)")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--dump_logits", default=None,
                    help="with --eval_only: dump raw per-set logits/target/label/K + meta to DIR "
                         "for scripts/null_bias.py (post-hoc K-aware null-bias fit)")
    p.add_argument("--tower_d", type=int, default=512)
    p.add_argument("--tower_layers", type=int, default=2)
    p.add_argument("--tower_heads", type=int, default=8)
    p.add_argument("--joint", action="store_true", help="encode query as a causal prefix continuation of state (one backbone pass)")
    p.add_argument("--listwise", action="store_true", help="set-attention mixer over [h; candidates] before scoring (zero-init)")
    p.add_argument("--extra_tap", type=int, default=0, help="also concatenate layer E's state onto the top state ([h_E; h_top]); needs --cand_encoder qwen3emb")
    p.add_argument("--init_from", default=None, help="warm-start tower+lora from a checkpoint (tower loaded strict=False, e.g. adding the listwise mixer)")
    p.add_argument("--readout", choices=["energy", "mcq", "native"], default="energy",
                    help="mcq: options enumerated in the suffix, answer read from next-token letter logits (see pcdm/mcq.py); "
                         "native: same suffix + terminal decision token, direct scorer over candidates (see pcdm/native.py)")
    p.add_argument("--nc_head", choices=["n2", "n3", "n2n3"], default="n2n3",
                    help="--readout native candidates: n2 = Qwen3-Embedding vectors, n3 = pooled option spans "
                         "from the suffix, n2n3 = both (PLAN4 sec 12)")
    p.add_argument("--nc_render", choices=["letters", "tags", "letters_nonull", "semif"], default="letters",
                    help="--readout native suffix: letters = mcq's 'A. opt' lines + null line + 'Answer:' (unchanged); "
                         "tags = native_v2 (PLAN5 sec 2) letter-free '<choice>\\n opt \\n</choice>' blocks, no null line; "
                         "semif = SemIf-structured chat-template prefix/suffix around the state (pcdm/native.py module docstring)")
    p.add_argument("--score_head", choices=["choice", "cumlink"], default="choice",
                    help="PLAN7 track C: score_C_cumlink -- cumulative-link ordinal head over the rendered "
                         "levels instead of the K-way Choice scorer (choice = today's default, unchanged)")
    p.add_argument("--noul_head", choices=["choice", "bern"], default="choice",
                    help="PLAN7 track C: noul_B_bern -- Bernoulli P(yes) from h_D, no candidates rendered "
                         "in the suffix (choice = today's 2-way Choice, unchanged)")
    p.add_argument("--qtype_filter", choices=["choice", "score", "noul"], default=None,
                    help="PLAN7 track C: keep only meta.qtype == X train/val rows (--data/--extra_data); "
                         "eval sets are untouched")
    p.add_argument("--ordinal_smooth", type=float, default=0.0,
                    help="PLAN7 track C score_B_smooth: tau for the ordinal-smoothed train target on "
                         "meta.qtype == 'score' rows, target_j ~ exp(-|j-y|/tau) (0 = off, today's hard target)")
    p.add_argument("--max_state", type=int, default=256,
                    help="--readout native: decision-state truncation length in tokens, threaded to "
                         "run_batch_native/native_features/calibrate_zscore_native (256 = today's default)")
    p.add_argument("--drop_truncated", action="store_true",
                    help="tl1b item 1: drop --data/--extra_data train rows whose state tokenizes past "
                         "--max_state (a right-truncated state can't carry the facts it was labelled on -- "
                         "REPORT S3ab/S3af); val/eval rows are untouched. Prints a per-source-dir drop count")
    p.add_argument("--perm_lambda", type=float, default=0.0,
                    help="--readout native: weight on native.perm_consistency_loss -- a second forward of a random 1/4 of "
                         "the batch under a fresh option order, KL between the two candidate distributions aligned by "
                         "identity (PLAN5 sec 2 F(pi(A)) = pi(F(A))); 0 = off (default, one forward per step)")
    p.add_argument("--no_shuffle", action="store_true",
                    help="mcq/native: keep the given option order in training (default: random order per example per step)")
    p.add_argument("--zero_shot", action="store_true", help="eval only, no checkpoint (e.g. a frozen --lora_r 0 backbone)")
    p.add_argument("--shots", type=int, default=0, help="--readout mcq: prepend N fixed exemplars (mcq.build_shots) so a base model picks up the letter-answer format")
    p.add_argument("--cand_encoder", choices=["backbone", "qwen3emb", "tiny"], default="backbone",
                    help="qwen3emb: candidate/state/query frozen-space features come from a separate "
                         "Qwen3-Embedding-0.6B cache instead of the backbone's own frozen layer; "
                         "tiny: token-level candidates from the backbone's frozen input-embedding table "
                         "+ --tiny_layers trained blocks inside the model (PLAN3 E3; cold-capable)")
    p.add_argument("--head", choices=["mlp", "z1", "zr", "zr_set"], default="mlp",
                    help="PLAN3 E3 students: z1 = one decision vector Z from h, bilinear late interaction; "
                         "zr = R probe vectors over the query tokens + token MaxSim; zr_set = zr + one "
                         "cross-attention from the probes over all candidate tokens (O(K), IIA-breaking)")
    p.add_argument("--z_dim", type=int, default=128, help="d' of the z heads")
    p.add_argument("--z_probes", type=int, default=8, help="R probes for zr/zr_set")
    p.add_argument("--tiny_layers", type=int, default=2, help="--cand_encoder tiny: trained blocks over the embedding table (0 = table only)")
    p.add_argument("--distill_alpha", type=float, default=1.0, help="weight on the existing gold soft-CE")
    p.add_argument("--distill_beta", type=float, default=0.0,
                    help="PLAN3 E3-T: weight on beta*T^2*KL(teacher_T || student_T) for rows with a "
                         "`teacher` field (scripts/teacher_label.py); 0 = off (default, no behavior change)")
    p.add_argument("--distill_T", type=float, default=2.0, help="temperature for the distillation KL term")
    p.add_argument("--delta_gamma", type=float, default=0.0,
                    help="PLAN3 E3-ms: weight on the Huber log-odds-shift term over meta.delta_t triples "
                         "(scripts/ms_targets.py); 0 = off (default, loss unchanged)")
    p.add_argument("--brier_lambda", type=float, default=0.0,
                    help="tl1b item 5: weight on lambda*((softmax(logits)-target)**2).sum(-1).mean() added to "
                         "decision_loss (target includes the null column); 0 = off (default, loss unchanged)")
    args = p.parse_args()

    if args.readout == "native":
        # n2 needs the embedding cache; n3 needs nothing beyond the backbone
        args.cand_encoder = "qwen3emb" if "n2" in args.nc_head else "backbone"
    if args.smoke:
        # Smoke defaults -- only fill these in when the caller didn't explicitly pick a
        # backbone/LoRA size of their own (e.g. `--smoke --backbone Qwen/Qwen3.5-0.8B-Base`
        # for a from-scratch-family smoke run must keep that backbone, not silently fall back
        # to Qwen3-0.6B-Base). Same "only if still at the parser's own default" rule the
        # steps/bs/etc smoke defaults below use, just checked per-field since these three
        # have real (non-None) defaults of their own.
        if args.backbone == p.get_default("backbone"):
            args.backbone = "Qwen/Qwen3-0.6B-Base"
        if args.lora_layers == p.get_default("lora_layers"):
            args.lora_layers = 4
        if args.lora_r == p.get_default("lora_r"):
            args.lora_r = 8
        if args.wandb:
            os.environ["WANDB_MODE"] = "offline"
    defaults = {"steps": 20, "bs": 8, "val_every": 10, "eval_every": 20, "ckpt_every": 10} if args.smoke else \
        {"steps": 12000, "bs": 64, "val_every": 1000, "eval_every": 4000, "ckpt_every": 2000}  # 12k = 1 epoch of ~758k
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def main():
    sys.stdout.reconfigure(line_buffering=True)  # live logs when piped/tee'd (pod, nohup)
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, results_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "results.json"

    ckpt = None
    if args.eval_only and not args.zero_shot:
        # reconstruct the architecture from the checkpoint's own args, not whatever flags
        # this invocation happened to be called with -- otherwise a mismatched
        # --lora_layers/--tap_layer/--joint silently loads weights into the wrong shapes.
        ckpt = torch.load(best_path, map_location=device, weights_only=False)  # trusted, self-produced
        for k in ("backbone", "tap_layer", "lora_layers", "lora_r", "no_hybrid", "no_cand_null", "joint", "tower_d", "tower_layers", "tower_heads", "listwise", "readout", "cand_encoder", "extra_tap", "null", "head", "z_dim", "z_probes", "tiny_layers", "nc_head", "nc_render", "score_head", "noul_head", "max_state"):
            if k in ckpt.get("args", {}):
                setattr(args, k, ckpt["args"][k])

    if args.readout == "mcq":
        assert not args.zscore and not args.joint and not args.listwise, \
            "--zscore/--joint/--listwise are energy-only (mcq mode has no tower)"
        backbone = MCQHead(args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r, device=device,
                           tap_layer=args.tap_layer)
        model = nn.Module()  # no tower: optimiser/checkpoint code below stays generic (empty param group / state dict)
    elif args.readout == "native":
        assert not args.joint and not args.listwise and not args.shots, "--joint/--listwise/--shots are not native options"
        backbone = MCQHead(args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r, device=device,
                           tap_layer=args.tap_layer)  # same backbone + LoRA as mcq; its lm_head is simply unused
        model = NativeHead(backbone.backbone.d, nc_head=args.nc_head, null=args.null, render=args.nc_render,
                           score_head=args.score_head, noul_head=args.noul_head).to(device)
    else:
        assert not args.extra_tap or args.cand_encoder == "qwen3emb", "--extra_tap doubles the top width; candidates must come from the embedder"
        backbone = Backbone(args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r, device=device,
                            tap_layer=args.tap_layer, extra_tap=args.extra_tap)
        from encode import EMBED_DIM
        d_cand = {"qwen3emb": EMBED_DIM, "tiny": backbone.model.config.hidden_size}.get(args.cand_encoder)
        model = DecisionModel(d_in=backbone.d, d=args.tower_d, num_layers=args.tower_layers, nhead=args.tower_heads,
                              dim_feedforward=2 * args.tower_d, no_null=args.no_null, cand_null=not args.no_cand_null,
                              hybrid=not args.no_hybrid, mps_safe=(device == "mps"), listwise=args.listwise,
                              d_cand=d_cand, null=args.null, head=args.head, z_dim=args.z_dim, z_probes=args.z_probes,
                              tiny_layers=args.tiny_layers if args.cand_encoder == "tiny" else 0).to(device)
    if args.grad_ckpt:
        base_backbone(backbone).model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cache, train_examples, val_examples, eval_sets = load_run_data(args, backbone)
    vec_cache = cache if args.cand_encoder == "qwen3emb" else None
    if args.zscore and args.readout == "native":
        calibrate_zscore_native(backbone, model, train_examples, args.seed, max_state=args.max_state)
    elif args.zscore:
        calibrate_zscore(backbone, model, train_examples, cache, args.seed, joint=args.joint, vec_cache=vec_cache)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)  # trusted, self-produced
        model.load_state_dict(ck["tower"], strict=False)
        backbone.load_lora_state_dict(ck["lora"])
    wb = init_wandb(args) if args.wandb else None

    if args.zero_shot:
        val_nll = eval_val_loss(backbone, model, cache, val_examples, args.eval_bs, joint=args.joint,
                                 readout=args.readout, vec_cache=vec_cache, shots=args.shots)
        results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, val_nll)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print_table(results)
        finish_wandb(wb, results)
        return

    if args.eval_only:
        model.load_state_dict(ckpt["tower"])
        backbone.load_lora_state_dict(ckpt["lora"])
        val_nll = eval_val_loss(backbone, model, cache, val_examples, args.eval_bs, joint=args.joint,
                                 readout=args.readout, vec_cache=vec_cache, shots=args.shots)
        results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, val_nll)
        if args.dump_logits:
            dump_eval_logits(args, backbone, model, cache, val_examples, eval_sets)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print_table(results)
        finish_wandb(wb, results)
        return

    best_on = resolve_best_on(args.best_on, eval_sets)
    opt, sched = build_optimizer(model, backbone, args)
    all_params = list(model.parameters()) + list(backbone.trainable_parameters())

    start_step, best_val = 0, float("inf")
    if last_path.exists() and not results_path.exists():
        start_step, best_val = load_checkpoint(last_path, model, backbone, opt, sched, device)
        print(f"resumed from step {start_step}")

    family_weights = parse_family_weights(args.family_weights)
    micro_bs = max(1, args.bs // args.grad_accum)  # --grad_accum micro-batches accumulate to one --bs step
    gen = bucketed_data_generator(train_examples, micro_bs, args.seed, family_weights) if family_weights \
        else data_generator(train_examples, micro_bs, args.seed)
    for _ in range(start_step * args.grad_accum):
        next(gen)
    batches = prefetch(gen, cache, readout=args.readout, seed=args.seed,
                       shuffle=args.readout in ("mcq", "native") and not args.no_shuffle)  # collate on a background thread
    perm_rng = random.Random(args.seed + 1)  # ponytail: not checkpointed (like the shuffle rng)
    assert args.perm_lambda == 0 or args.readout == "native", "--perm_lambda is a native-readout option"

    results = None
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        opt.zero_grad()
        loss_sum, n_tokens, bucket_losses = 0.0, 0, {}
        for _ in range(args.grad_accum):  # 1 iteration, byte-identical to pre-grad_accum behavior
            examples, batch = next(batches)
            logits = run_readout(args.readout, backbone, model, batch, examples, joint=args.joint, vec_cache=vec_cache,
                                 shots=args.shots, max_state=args.max_state)
            # run_batch moves state/query/C to backbone.device but leaves target/p_null/cmask/
            # teacher/has_teacher/delta_* on the CPU tensors collate()/collate_mcq() built; move
            # them here so loss math matches logits' device (delta_* only exist for collate()).
            for k in ("target", "p_null", "cmask", "teacher", "has_teacher", "delta_idx", "delta_tgt"):
                if k in batch:
                    batch[k] = batch[k].to(logits.device)
            loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"],
                                  teacher=batch["teacher"], has_teacher=batch["has_teacher"],
                                  alpha=args.distill_alpha, beta=args.distill_beta, T=args.distill_T,
                                  delta_idx=batch.get("delta_idx"), delta_tgt=batch.get("delta_tgt"),
                                  gamma=args.delta_gamma, brier_lambda=args.brier_lambda)
            if args.perm_lambda > 0:
                loss = loss + args.perm_lambda * perm_consistency_loss(
                    lambda ex, b: run_batch_native(backbone, model, b, ex, max_state=args.max_state, vec_cache=vec_cache),
                    logits, examples, perm_rng)
            (loss / args.grad_accum).backward()
            loss_sum += loss.item()
            n_tokens += batch.get("n_tokens", 0)
            if family_weights:  # each micro-batch is single-bucket under the bucketed sampler, but
                bucket_losses[bucket_of(examples[0])] = loss.item()  # accumulation can span buckets
        # logits/batch/examples below are the LAST micro-batch only (per_family_loss's breakdown is a
        # diagnostic, not the training signal -- fine as an approximation under grad_accum > 1)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()
        step_time = time.time() - t0
        loss_value = loss_sum / args.grad_accum

        log = {
            "train/loss": loss_value, "train/step_time": step_time,
            "train/tokens_per_s": n_tokens / step_time if step_time > 0 else 0.0,
            "train/lr_tower": opt.param_groups[0]["lr"], "train/lr_lora": opt.param_groups[1]["lr"],
            "train/mem": device_memory(device),
        }
        if step % 50 == 0:
            fam = per_family_loss(logits, batch, examples)
            log.update({f"train/family_loss/{k}": v for k, v in fam.items()})
            log.update({f"train/bucket_loss/{b}": v for b, v in bucket_losses.items()})
            bucket_note = f"bucket {','.join(bucket_losses)} " if bucket_losses else ""
            print(f"step {step} {bucket_note}loss {loss_value:.4f} step_time {step_time:.2f}s "
                  f"tok/s {log['train/tokens_per_s']:.0f} lr_tower {log['train/lr_tower']:.2e} "
                  f"lr_lora {log['train/lr_lora']:.2e}")
        if wb:
            wb.log(log, step=step)

        if step % args.val_every == 0 or step == args.steps:
            val_nll = eval_val_loss(backbone, model, cache, val_examples, args.eval_bs, joint=args.joint,
                                     readout=args.readout, vec_cache=vec_cache, shots=args.shots)
            print(f"step {step} val_nll {val_nll:.4f}")
            if wb:
                wb.log({"val/nll": val_nll}, step=step)
            ckpt_metric = checkpoint_metric(backbone, model, cache, eval_sets, best_on, args, vec_cache, val_nll, wb, step)
            if ckpt_metric < best_val:
                best_val = ckpt_metric
                save_checkpoint(best_path, model, backbone, step, opt, sched, best_val, args, tower_lora_only=True)

        if step % args.eval_every == 0 and step != args.steps:  # final eval below uses best.pt
            results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, best_val)
            log_eval_wandb(wb, results, step)

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(last_path, model, backbone, step, opt, sched, best_val, args)
            if args.ckpt_upload and args.hf_repo:
                upload_ckpt(args, run_dir)

    if args.no_final_eval:   # typically jobs: scripts/eval_wf.py scores best.pt on the company's own rows instead
        return
    # final eval on the best-by-val checkpoint (as v0), not the last step
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["tower"])
        backbone.load_lora_state_dict(ckpt["lora"])
    results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, best_val)
    log_eval_wandb(wb, results, args.steps)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print_table(results)
    finish_wandb(wb, results)

    if args.hf_repo:
        try:
            import huggingface_hub
            huggingface_hub.upload_folder(folder_path=str(run_dir), repo_id=args.hf_repo,
                                           path_in_repo=args.name, repo_type="model", ignore_patterns=["last.pt"])
        except Exception as e:
            print(f"warning: hf upload failed: {e}")


if __name__ == "__main__":
    main()
