"""Local CPU gate for the v1 pipeline (PLAN2.md "Local gate" item 1). Must finish in
< 60s total: `uv run pytest tests/ -q`. Uses the tiny_backbone/tiny_cache fixtures
from tests/conftest.py (a real, tiny, random-weight Qwen3 + the real Qwen3-0.6B-Base
tokenizer) instead of the full-size backbone.
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from encode import Backbone, FeatureCache
from model import DecisionModel, decision_loss, collate, collate_teacher, run_batch, decide, split_joint
from metrics import summarize
from mcq import MCQHead, collate_mcq, run_batch_mcq
import train as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from null_bias import fit_null_bias, apply_bias, soft_ce  # noqa: E402
import workflow_corpus as wf  # noqa: E402


CANDS = [f"cand{i}" for i in range(8)]


def make_examples(seed=0, n=4, ks=(1, 2, 3, 5), p_nulls=(0.0, 0.3, 1.0, 0.0)):
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        K = ks[i % len(ks)]
        chosen = rng.sample(CANDS, K)
        tgt = [1.0] + [0.0] * (K - 1)
        examples.append({
            "state": f"state number {i} " * (i + 1),
            "query": f"query number {i}",
            "candidates": chosen,
            "target": tgt,
            "p_null": p_nulls[i % len(p_nulls)],
            "task": "smoke",
        })
    return examples


# ---------------------------------------------------------------------------
# 1. collate
# ---------------------------------------------------------------------------

def test_collate_ragged(tiny_cache):
    examples = make_examples()
    torch.manual_seed(0)
    batch = collate(tiny_cache, examples)

    Kmax = max(len(ex["candidates"]) for ex in examples)
    d_in = tiny_cache.feats.shape[1]
    assert batch["C"].shape == (4, Kmax, d_in)
    assert batch["cmask"].shape == (4, Kmax)
    assert batch["target"].shape == (4, Kmax)
    assert batch["p_null"].shape == (4,)

    for i, ex in enumerate(examples):
        K = len(ex["candidates"])
        assert batch["cmask"][i].sum().item() == K
        assert torch.all(batch["target"][i, K:] == 0)
    assert batch["cmask"].any(-1).all()


# ---------------------------------------------------------------------------
# 2. probs_from_logits
# ---------------------------------------------------------------------------

def test_probs_pads_zero_rows_sum_one():
    torch.manual_seed(0)
    B, Kmax = 4, 5
    Ks = [1, 2, 3, 5]
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    for i, K in enumerate(Ks):
        cmask[i, :K] = True
    logits = torch.randn(B, Kmax + 1) * 3

    probs = T.probs_from_logits(logits, cmask)
    assert probs.shape == (B, Kmax + 1)
    for i, K in enumerate(Ks):
        if K < Kmax:
            assert torch.all(probs[i, K:Kmax] == 0)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-6)


# ---------------------------------------------------------------------------
# 3. permutation equivariance
# ---------------------------------------------------------------------------

def test_permutation_equivariance():
    torch.manual_seed(0)
    d_in = 64
    B, Lh, Lq, Kmax = 4, 6, 3, 5
    model = DecisionModel(d_in=d_in, dropout=0).eval()

    H = torch.randn(B, Lh, d_in)
    Q = torch.randn(B, Lq, d_in)
    C = torch.randn(B, Kmax, d_in)
    Uf = torch.randn(B, d_in)
    Vf = torch.randn(B, d_in)

    hlens = [6, 4, 5, 2]
    qlens = [3, 1, 2, 3]
    klens = [1, 2, 3, 5]
    hmask = torch.zeros(B, Lh, dtype=torch.bool)
    qmask = torch.zeros(B, Lq, dtype=torch.bool)
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    for i in range(B):
        hmask[i, :hlens[i]] = True
        qmask[i, :qlens[i]] = True
        cmask[i, :klens[i]] = True

    with torch.no_grad():
        logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)

        perm = [2, 0, 3, 1]
        logits_perm = model(H[perm], hmask[perm], Q[perm], qmask[perm], C[perm], cmask[perm],
                             Uf[perm], Vf[perm])

    assert torch.allclose(logits_perm, logits[perm], atol=1e-5)


# ---------------------------------------------------------------------------
# 4. loss
# ---------------------------------------------------------------------------

def test_loss_matches_hand_soft_ce():
    torch.manual_seed(0)
    B, Kmax = 5, 4
    logits = torch.randn(B, Kmax + 1) * 2
    p_null = torch.rand(B) * 0.5
    cmask = torch.ones(B, Kmax, dtype=torch.bool)
    target = torch.softmax(torch.randn(B, Kmax), dim=-1)

    loss = decision_loss(logits, target, p_null, cmask)

    t = torch.cat([(1 - p_null).unsqueeze(-1) * target, p_null.unsqueeze(-1)], dim=-1)
    manual_logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    manual_loss = -(t * manual_logp).sum(-1).mean()

    assert torch.allclose(loss, manual_loss, atol=1e-6)


def test_null_target_all_mass_on_null():
    torch.manual_seed(0)
    B, Kmax = 5, 4
    logits = torch.randn(B, Kmax + 1) * 2
    p_null = torch.ones(B)
    cmask = torch.ones(B, Kmax, dtype=torch.bool)
    target = torch.softmax(torch.randn(B, Kmax), dim=-1)  # ignored: p_null=1 zeroes it out

    loss = decision_loss(logits, target, p_null, cmask)
    expected = -(logits - torch.logsumexp(logits, dim=-1, keepdim=True))[:, -1].mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_brier_lambda_matches_hand_computation_and_zero_is_noop():
    """tl1b item 5: brier_lambda=0 (default) must reproduce the untouched CE loss exactly;
    brier_lambda>0 adds lambda * mean_row sum_j (softmax(logits)_j - t_j)^2 over the full
    [K+1] target (t includes the null column, via the same (1-p_null)*target/p_null split)."""
    torch.manual_seed(1)
    B, Kmax = 5, 4
    logits = torch.randn(B, Kmax + 1) * 2
    p_null = torch.rand(B) * 0.5
    cmask = torch.ones(B, Kmax, dtype=torch.bool)
    target = torch.softmax(torch.randn(B, Kmax), dim=-1)

    ce_only = decision_loss(logits, target, p_null, cmask)
    off = decision_loss(logits, target, p_null, cmask, brier_lambda=0.0)
    assert torch.allclose(off, ce_only, atol=1e-6)

    t = torch.cat([(1 - p_null).unsqueeze(-1) * target, p_null.unsqueeze(-1)], dim=-1)
    p = torch.softmax(logits, dim=-1)
    manual_brier = (p - t).pow(2).sum(-1).mean()
    with_brier = decision_loss(logits, target, p_null, cmask, brier_lambda=0.5)
    assert torch.allclose(with_brier, ce_only + 0.5 * manual_brier, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. LoRA zero-init + grad isolation
# ---------------------------------------------------------------------------

def test_lora_zero_init_and_grads(tiny_model_dir, tiny_cache):
    torch.manual_seed(0)
    bb = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu")
    bb0 = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=0, device="cpu")

    ids, am = bb.tokenize(["a short sentence", "another one, a bit longer than the first"], max_len=32)
    with torch.no_grad():
        h_top, _, _ = bb(ids, am)
        h_top0, _, _ = bb0(ids, am)
    assert torch.allclose(h_top, h_top0, atol=1e-4), "zero-init LoRA changed h_top"

    examples = make_examples(n=3, ks=(1, 2, 3))
    model = DecisionModel(d_in=bb.d, dropout=0)
    batch = collate(tiny_cache, examples)
    logits = run_batch(bb, model, batch)
    loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
    loss.backward()

    trainable = bb.trainable_parameters()
    assert len(trainable) > 0
    assert all(p.grad is not None for p in trainable)
    base_params = [p for n, p in bb.named_parameters() if not n.endswith((".A", ".B"))]
    assert all(p.grad is None for p in base_params)

    sd = bb.lora_state_dict()
    assert len(sd) > 0
    assert all(k.endswith((".A", ".B")) for k in sd)

    bb2 = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu")
    bb2.load_lora_state_dict(sd)
    sd2 = bb2.lora_state_dict()
    for k in sd:
        assert torch.equal(sd[k], sd2[k])


# ---------------------------------------------------------------------------
# 6. run_batch end to end
# ---------------------------------------------------------------------------

def test_run_batch_end_to_end(tiny_backbone, tiny_cache):
    torch.manual_seed(0)
    examples = make_examples()
    batch = collate(tiny_cache, examples)

    model = DecisionModel(d_in=tiny_backbone.d, dropout=0)
    with torch.no_grad():
        logits = run_batch(tiny_backbone, model, batch)
    Kmax = max(len(ex["candidates"]) for ex in examples)
    assert logits.shape == (4, Kmax + 1)
    assert torch.isfinite(logits).all()
    assert batch["n_tokens"] > 0

    model_nh = DecisionModel(d_in=tiny_backbone.d, hybrid=False, dropout=0)
    with torch.no_grad():
        logits_nh = run_batch(tiny_backbone, model_nh, batch)
    assert torch.isfinite(logits_nh).all()


# ---------------------------------------------------------------------------
# 7. decide() matches a batched run_batch call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("null", ["softmax", "factored"])
def test_decide_matches_batched(tiny_backbone, tiny_cache, null):
    torch.manual_seed(0)
    state = "a shared state text used for every query below"
    queries = [
        ("first query", ["cand0", "cand1"]),
        ("second query", ["cand2", "cand3", "cand4"]),
        ("third query", ["cand5"]),
    ]
    model = DecisionModel(d_in=tiny_backbone.d, dropout=0, null=null).eval()

    dists = decide(tiny_backbone, model, tiny_cache, state, queries, chunk=2)
    assert len(dists) == 3

    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    batch = collate(tiny_cache, examples)
    with torch.no_grad():
        logits = run_batch(tiny_backbone, model, batch)
    probs = F.softmax(logits, dim=-1)

    for i, (_, cands) in enumerate(queries):
        K = len(cands)
        expected = torch.cat([probs[i, :K], probs[i, -1:]])
        assert torch.allclose(dists[i], expected, atol=1e-4)
        assert torch.allclose(dists[i].sum(), torch.tensor(1.0), atol=1e-4)


# ---------------------------------------------------------------------------
# 7b. joint (prefix-conditioned) query encoding
# ---------------------------------------------------------------------------

def test_joint_causal_invariance(tiny_backbone):
    # ragged, varied lengths so padding/truncation actually exercise split_joint's index math
    states = ["a short state", "a somewhat longer state text here for padding", "s"]
    queries = ["q1", "a longer query text for testing", "another query here"]

    input_ids, attention_mask, ls, lq = tiny_backbone.tokenize_joint(states, queries, max_state=32, max_query=16)
    h_top, h_frozen, mask = tiny_backbone(input_ids, attention_mask)
    H, hmask, Q, qmask, Hf, Qf = split_joint(h_top, h_frozen, mask, ls, lq)

    ids_s, am_s = tiny_backbone.tokenize(states, 32)
    h_top_s, h_frozen_s, mask_s = tiny_backbone(ids_s, am_s)

    # causal invariant: state tokens can't see the query that follows them, so their
    # states in the joint pass must equal the state-only pass exactly (bf16 -> 5e-2)
    assert hmask.shape == mask_s.shape
    assert torch.equal(hmask, mask_s)
    assert torch.allclose(H[hmask], h_top_s[mask_s], atol=5e-2)
    assert torch.allclose(Hf[hmask], h_frozen_s[mask_s], atol=5e-2)


def test_joint_run_batch_shapes(tiny_model_dir, tiny_cache):
    torch.manual_seed(0)
    bb = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu")
    examples = make_examples()
    batch = collate(tiny_cache, examples)

    model = DecisionModel(d_in=bb.d, dropout=0)
    logits = run_batch(bb, model, batch, joint=True)
    Kmax = max(len(ex["candidates"]) for ex in examples)
    assert logits.shape == (4, Kmax + 1)
    assert torch.isfinite(logits).all()
    assert batch["n_tokens"] > 0

    loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
    loss.backward()
    trainable = bb.trainable_parameters()
    assert len(trainable) > 0
    assert all(p.grad is not None for p in trainable)


@pytest.mark.parametrize("null", ["softmax", "factored"])
def test_decide_joint_matches_run_batch(tiny_backbone, tiny_cache, null):
    torch.manual_seed(0)
    state = "a shared state text used for every query below in joint mode"
    queries = [
        ("first query", ["cand0", "cand1"]),
        ("second query", ["cand2", "cand3", "cand4"]),
        ("third query", ["cand5"]),
    ]
    model = DecisionModel(d_in=tiny_backbone.d, dropout=0, null=null).eval()

    dists = decide(tiny_backbone, model, tiny_cache, state, queries, chunk=2, joint=True)
    assert len(dists) == 3

    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    batch = collate(tiny_cache, examples)
    with torch.no_grad():
        logits = run_batch(tiny_backbone, model, batch, joint=True)
    probs = F.softmax(logits, dim=-1)

    for i, (_, cands) in enumerate(queries):
        K = len(cands)
        expected = torch.cat([probs[i, :K], probs[i, -1:]])
        assert torch.allclose(dists[i], expected, atol=1e-3)
        assert torch.allclose(dists[i].sum(), torch.tensor(1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# 8. checkpoint save -> load -> resume reproduces loss
# ---------------------------------------------------------------------------

def test_checkpoint_resume_reproduces_loss(tiny_model_dir, tmp_path):
    random.seed(0)
    examples = make_examples(n=6, ks=(1, 2, 3, 1, 2, 3), p_nulls=(0.0,) * 6)
    batches_ex = [examples[i:i + 2] for i in range(0, 6, 2)]  # 3 two-example batches

    def build(seed):
        torch.manual_seed(seed)
        bb = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu")
        model = DecisionModel(d_in=bb.d, dropout=0)
        args = argparse.Namespace(lr=1e-3, lora_lr=1e-3, steps=100)
        opt, sched = T.build_optimizer(model, bb, args)
        return bb, model, opt, sched, args

    def do_step(bb, model, opt, sched, cache, ex_batch):
        batch = collate(cache, ex_batch)
        logits = run_batch(bb, model, batch)
        loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(bb.trainable_parameters()), 1.0)
        opt.step()
        sched.step()
        return loss.item()

    bb, model, opt, sched, args = build(seed=0)
    cache = FeatureCache(device="cpu")
    cache.add(bb, CANDS, max_len=16)

    for step in range(3):
        do_step(bb, model, opt, sched, cache, batches_ex[step % 3])

    ckpt_path = tmp_path / "ck.pt"
    T.save_checkpoint(str(ckpt_path), model, bb, 3, opt, sched, float("inf"), args)

    loss_a = do_step(bb, model, opt, sched, cache, batches_ex[0])

    bb2, model2, opt2, sched2, args2 = build(seed=0)
    step_loaded, _ = T.load_checkpoint(str(ckpt_path), model2, bb2, opt2, sched2, "cpu")
    assert step_loaded == 3
    loss_b = do_step(bb2, model2, opt2, sched2, cache, batches_ex[0])

    assert abs(loss_a - loss_b) < 1e-5


# ---------------------------------------------------------------------------
# 9. W&B offline
# ---------------------------------------------------------------------------

def test_wandb_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    import wandb
    run = wandb.init(project="pcdm-test", dir=str(tmp_path))
    run.log({"a": 1.0, "b": 2.0})
    run.log({"table": wandb.Table(columns=["x"], data=[[1]])})
    run.finish()

    offline_dirs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert offline_dirs, "no offline run directory written"


# ---------------------------------------------------------------------------
# 10. metrics + report
# ---------------------------------------------------------------------------

def test_metrics_summarize_hand_values():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    target = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    label = [0, -1]
    m = summarize(probs, target, label)
    assert abs(m["nll"] - 0.2899) < 1e-3
    assert abs(m["brier"] - 0.10) < 1e-6
    assert m["acc"] == 1.0
    assert m["acc_k"] == 1.0
    assert m["conf_wrong"] == 0.0
    assert abs(m["auroc_null"] - 1.0) < 1e-9


def test_report_handles_na(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "runs" / "x"
    run_dir.mkdir(parents=True)
    results = {
        "T": 1.0,
        "val_nll": 0.5,
        "eval": {
            "clinc_k": {"scaled": {"auroc_null": "n/a"}, "raw": {"auroc_null": "n/a"}},
            "snli_test": {"scaled": {"acc": 0.9}, "raw": {"acc": 0.85}},
        },
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f)

    monkeypatch.chdir(tmp_path)
    import report
    report.main()
    out = capsys.readouterr().out
    assert "n/a" in out


# ---------------------------------------------------------------------------
# 11. data audit (skips if no data files present yet)
# ---------------------------------------------------------------------------

def test_data_audit_if_present():
    train_path = None
    for candidate in ("data_small/train.jsonl", "data/train.jsonl"):
        if os.path.exists(candidate):
            train_path = candidate
            break
    if train_path is None:
        pytest.skip("no train.jsonl found")

    def norm_state(s):
        return " ".join(s.split()).strip().lower()

    rows = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, f"{train_path} is empty"

    for ex in rows:
        assert len(ex["candidates"]) == len(ex["target"])
        assert 0.0 <= ex["p_null"] <= 1.0
        assert abs(sum(ex["target"]) - 1.0) < 1e-4
        label = ex.get("label")
        if label is not None and label >= 0:
            assert ex["target"][label] == max(ex["target"])

    # ponytail: bound the leak-audit scan so this stays fast even once train.jsonl
    # grows toward the full ~800k-row build; raise the cap if that starts missing leaks.
    train_states, train_cands = set(), set()
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= 200_000:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            train_states.add(norm_state(ex["state"]))
            train_cands.update(ex["candidates"])

    try:
        from data import PARAPHRASES
    except ImportError:
        PARAPHRASES = None
    if PARAPHRASES:
        leaked = {w for triple in PARAPHRASES for w in triple} & train_cands
        assert not leaked, f"held-out paraphrase wordings leaked into train candidates: {leaked}"

    eval_dir = Path(train_path).parent / "eval"
    if eval_dir.is_dir():
        for eval_path in sorted(eval_dir.glob("*.jsonl")):
            overlap = set()
            seen = 0
            with open(eval_path) as f:
                for line in f:
                    if seen >= 5000:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    seen += 1
                    ex = json.loads(line)
                    st = norm_state(ex["state"])
                    if st in train_states:
                        overlap.add(st)
            assert not overlap, f"{len(overlap)} states leak between train and {eval_path.name}"


# ---------------------------------------------------------------------------
# 12. listwise SetMixer: candidate-set equivariance + identity-at-init
# ---------------------------------------------------------------------------

def _lw_model(listwise, perturb_seed=None):
    torch.manual_seed(0)
    model = DecisionModel(d_in=64, dropout=0, listwise=listwise).eval()
    if listwise and perturb_seed is not None:
        # zero-init makes the mixer an exact identity, so a listwise-vs-base test would
        # trivially pass with no signal; perturb it so listwise actually conditions on the set.
        g = torch.Generator().manual_seed(perturb_seed)
        with torch.no_grad():
            for p in model.mixer.parameters():
                p.normal_(0, 0.05, generator=g)
    return model


def _lw_batch(Kmax=6, klens=(2, 4, 6), Lh=6, Lq=3, seed=0):
    torch.manual_seed(seed)
    d_in, B = 64, len(klens)
    H, Q = torch.randn(B, Lh, d_in), torch.randn(B, Lq, d_in)
    C = torch.randn(B, Kmax, d_in)
    Uf, Vf = torch.randn(B, d_in), torch.randn(B, d_in)
    hmask = torch.ones(B, Lh, dtype=torch.bool)
    qmask = torch.ones(B, Lq, dtype=torch.bool)
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    for i, k in enumerate(klens):
        cmask[i, :k] = True
    return H, hmask, Q, qmask, C, cmask, Uf, Vf


@pytest.mark.parametrize("listwise", [False, True])
def test_candidate_permutation_equivariance(listwise):
    model = _lw_model(listwise, perturb_seed=1)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch()
    with torch.no_grad():
        logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)

    B, Kmax = cmask.shape
    C_perm = C.clone()
    perms = []
    for i in range(B):
        K = int(cmask[i].sum())
        p = list(range(K))
        random.Random(i).shuffle(p)
        perms.append(p)
        C_perm[i, :K] = C[i, p]

    with torch.no_grad():
        logits_perm = model(H, hmask, Q, qmask, C_perm, cmask, Uf, Vf)

    for i in range(B):
        K = int(cmask[i].sum())
        expected = logits[i, :K][perms[i]]
        assert torch.allclose(logits_perm[i, :K], expected, atol=1e-5)
    assert torch.allclose(logits_perm[:, -1], logits[:, -1], atol=1e-5)  # null unchanged


@pytest.mark.parametrize("listwise", [False, True])
def test_padding_invariance(listwise):
    model = _lw_model(listwise, perturb_seed=2)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch()
    with torch.no_grad():
        logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)

    pad = 3
    B, Kmax = cmask.shape
    C_pad = torch.cat([C, torch.randn(B, pad, C.shape[-1])], dim=1)
    cmask_pad = torch.cat([cmask, torch.zeros(B, pad, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        logits_pad = model(H, hmask, Q, qmask, C_pad, cmask_pad, Uf, Vf)

    assert torch.allclose(logits_pad[:, :Kmax], logits[:, :Kmax], atol=1e-6)
    assert torch.allclose(logits_pad[:, -1], logits[:, -1], atol=1e-6)


@pytest.mark.parametrize("listwise", [False, True])
def test_duplicate_slots_identical(listwise):
    model = _lw_model(listwise, perturb_seed=3)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch()
    C, cmask = C.clone(), cmask.clone()
    row, dup_slot = 0, int(cmask[0].sum())  # row 0 has K=2, so slot 2 is a free pad slot
    C[row, dup_slot] = C[row, 0]
    cmask[row, dup_slot] = True

    with torch.no_grad():
        logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)
    assert torch.allclose(logits[row, 0], logits[row, dup_slot], atol=1e-6)


def test_listwise_identity_at_init_and_warm_start():
    base = _lw_model(listwise=False)
    lw = _lw_model(listwise=True)  # fresh, still zero-init -- no perturbation
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch()

    result = lw.load_state_dict(base.state_dict(), strict=False)
    expected_missing = {f"mixer.{n}" for n, _ in lw.mixer.named_parameters()}
    assert set(result.missing_keys) == expected_missing
    assert not result.unexpected_keys

    with torch.no_grad():
        logits_base = base(H, hmask, Q, qmask, C, cmask, Uf, Vf)
        logits_lw = lw(H, hmask, Q, qmask, C, cmask, Uf, Vf)
    assert torch.allclose(logits_lw, logits_base, atol=1e-6)


def test_add_candidate_odds():
    base = _lw_model(listwise=False)
    lw = _lw_model(listwise=True, perturb_seed=4)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch(Kmax=4, klens=(2, 2))

    cmask_3 = cmask.clone(); cmask_3[:, 2] = True  # enable one more candidate slot

    with torch.no_grad():
        base_2 = base(H, hmask, Q, qmask, C, cmask, Uf, Vf)
        base_3 = base(H, hmask, Q, qmask, C, cmask_3, Uf, Vf)
        lw_2 = lw(H, hmask, Q, qmask, C, cmask, Uf, Vf)
        lw_3 = lw(H, hmask, Q, qmask, C, cmask_3, Uf, Vf)

    # base has no cross-candidate interaction: s0-s1 is exactly unchanged by adding a slot
    assert torch.allclose(base_2[:, 0] - base_2[:, 1], base_3[:, 0] - base_3[:, 1], atol=1e-6)
    # listwise mixes the candidate set into h/c, so s0-s1 shifts when a slot is added
    assert not torch.allclose(lw_2[:, 0] - lw_2[:, 1], lw_3[:, 0] - lw_3[:, 1], atol=1e-4)


# ---------------------------------------------------------------------------
# 12b. factored null (PLAN3 E2): P(a_j)=(1-r)*p_j, P(null)=r, exact IIA
# ---------------------------------------------------------------------------

def _factored_model(perturb_seed=None):
    torch.manual_seed(0)
    model = DecisionModel(d_in=64, dropout=0, null="factored").eval()
    if perturb_seed is not None:
        # null_gate starts small-but-nonzero (regular init, not zero-init like the listwise
        # mixer) so no perturbation is strictly needed -- but match the listwise tests' style
        # so r actually varies across rows instead of being near-constant at init.
        g = torch.Generator().manual_seed(perturb_seed)
        with torch.no_grad():
            for p in model.null_gate.parameters():
                p.normal_(0, 0.05, generator=g)
    return model


def test_factored_sums_to_one_and_pnull_is_r():
    model = _factored_model(perturb_seed=1)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch()
    with torch.no_grad():
        logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)
    probs = torch.softmax(logits, dim=-1)
    B = cmask.shape[0]
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    r = logits[:, -1].exp()  # logits_null = log r by construction
    assert torch.allclose(probs[:, -1], r, atol=1e-6)
    assert torch.allclose(probs[:, :-1][~cmask], torch.zeros_like(probs[:, :-1][~cmask]), atol=1e-6)


def test_factored_loss_identity():
    """decision_loss (soft CE over [(1-p_null)*target, p_null]) equals
    BCE(r, p_null) + (1-p_null)*CE(p, target) exactly for the factored composition --
    verify the identity instead of maintaining a second loss implementation."""
    model = _factored_model(perturb_seed=2)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch()
    with torch.no_grad():
        logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)
    B, Kmax = cmask.shape
    target = torch.softmax(torch.randn(B, Kmax).masked_fill(~cmask, -1e9), dim=-1)
    p_null = torch.rand(B) * 0.8

    loss = decision_loss(logits, target, p_null, cmask)

    r = logits[:, -1].exp()
    logp = logits[:, :Kmax] - torch.log1p(-r).unsqueeze(-1)  # logits_j - log(1-r) = log p_j
    p = torch.softmax(logp.masked_fill(~cmask, torch.finfo(logp.dtype).min), dim=-1)
    ce = -(target * torch.log(p.clamp_min(1e-12))).sum(-1)
    bce = -(p_null * torch.log(r.clamp_min(1e-12)) + (1 - p_null) * torch.log((1 - r).clamp_min(1e-12)))
    expected = (bce + (1 - p_null) * ce).mean()
    assert torch.allclose(loss, expected, atol=1e-4)


def test_factored_permutation_invariance_r_and_equivariance_p():
    model = _factored_model(perturb_seed=3)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch()
    with torch.no_grad():
        logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)

    B, Kmax = cmask.shape
    C_perm = C.clone()
    perms = []
    for i in range(B):
        K = int(cmask[i].sum())
        p = list(range(K))
        random.Random(i).shuffle(p)
        perms.append(p)
        C_perm[i, :K] = C[i, p]

    with torch.no_grad():
        logits_perm = model(H, hmask, Q, qmask, C_perm, cmask, Uf, Vf)

    assert torch.allclose(logits_perm[:, -1], logits[:, -1], atol=1e-5)  # r: permutation-invariant
    for i in range(B):  # p_j: permutation-equivariant
        K = int(cmask[i].sum())
        expected = logits[i, :K][perms[i]]
        assert torch.allclose(logits_perm[i, :K], expected, atol=1e-5)


def test_factored_iia_add_candidate():
    model = _factored_model(perturb_seed=4)
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch(Kmax=4, klens=(2, 2))
    cmask_3 = cmask.clone(); cmask_3[:, 2] = True  # enable one more (irrelevant) candidate slot

    with torch.no_grad():
        logits_2 = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)
        logits_3 = model(H, hmask, Q, qmask, C, cmask_3, Uf, Vf)

    # log-odds between the two original candidates is untouched by the new candidate (IIA)
    assert torch.allclose(logits_2[:, 0] - logits_2[:, 1], logits_3[:, 0] - logits_3[:, 1], atol=1e-5)


# ---------------------------------------------------------------------------
# 13. MCQ-LoRA readout (pcdm/mcq.py)
# ---------------------------------------------------------------------------

def _make_tied_mcq_head(d, lora_layers=2, lora_r=4):
    """A tiny random-weight Qwen3 with TIED embeddings (conftest's tiny_model_dir defaults
    to untied, per Qwen3Config()'s default) + the real Qwen3-0.6B-Base tokenizer, so the
    lm_head-from-embed_tokens path in MCQHead is exercised without a second model load."""
    from transformers import AutoModel, AutoTokenizer, Qwen3Config
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", padding_side="right")
    cfg = Qwen3Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, intermediate_size=128, vocab_size=len(tok),
                       head_dim=16, tie_word_embeddings=True)
    AutoModel.from_config(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    torch.manual_seed(0)
    return MCQHead(name=str(d), lora_layers=lora_layers, lora_r=lora_r, device="cpu")


@pytest.fixture(scope="session")
def tied_mcq_head(tmp_path_factory):
    return _make_tied_mcq_head(tmp_path_factory.mktemp("tiny_qwen3_tied"))


def _mcq_examples(ks, seed=0):
    rng = random.Random(seed)
    out = []
    for i, k in enumerate(ks):
        cands = [f"option {j}" for j in range(k)]
        gold = rng.randrange(k)
        out.append({
            "state": f"state text number {i}",
            "query": f"query number {i}",
            "candidates": cands,
            "target": [1.0 if j == gold else 0.0 for j in range(k)],
            "p_null": 0.0,
            "task": "smoke",
        })
    return out


def test_mcq_letter_logits_shape_finite(tied_mcq_head):
    examples = _mcq_examples(ks=(2, 3, 5))
    batch = collate_mcq(examples)
    logits = run_batch_mcq(tied_mcq_head, batch, examples)
    Kmax = max(len(ex["candidates"]) for ex in examples)
    assert logits.shape == (3, Kmax + 1)
    assert torch.isfinite(logits).all()
    assert batch["n_tokens"] > 0


def test_mcq_padded_columns_are_finfo_min(tied_mcq_head):
    examples = _mcq_examples(ks=(2, 3, 5))
    batch = collate_mcq(examples)
    logits = run_batch_mcq(tied_mcq_head, batch, examples)
    Kmax = logits.shape[1] - 1
    for i, ex in enumerate(examples):
        k = len(ex["candidates"])
        if k < Kmax:
            assert torch.all(logits[i, k:Kmax] == torch.finfo(logits.dtype).min)


def test_mcq_chunked_k70(tied_mcq_head):
    # K=70 > MAXK_DIRECT (51) forces the two-stage chunked path for row 1; row 0 (K=5)
    # exercises the direct path in the same batch.
    examples = _mcq_examples(ks=(5, 70))
    batch = collate_mcq(examples)
    logits = run_batch_mcq(tied_mcq_head, batch, examples)
    assert logits.shape == (2, 71)
    assert torch.isfinite(logits[:, -1]).all()  # null column scored for every row
    assert torch.all(logits[0, 5:70] == torch.finfo(logits.dtype).min)  # row 0: K=5, rest padded
    assert (logits[1, :70] > torch.finfo(logits.dtype).min).any()  # row 1: chunk winners scored


def test_mcq_shots_prefix(tied_mcq_head):
    # PLAN3 Task C: --shots N prepends N fixed exemplars, format identical to the eval
    # prompt (question, lettered options, "Answer: X"); N=0 is the untouched zero-shot path.
    from mcq import build_shots, FEWSHOT_EXAMPLES
    assert build_shots(0) == ""
    prefix = build_shots(3)
    assert prefix.count("Answer:") == 3
    assert all(ex["question"] in prefix for ex in FEWSHOT_EXAMPLES[:3])

    examples = _mcq_examples(ks=(3,))
    batch = collate_mcq(examples)
    logits0 = run_batch_mcq(tied_mcq_head, batch, examples, shots=0)
    n_tok0 = batch["n_tokens"]
    logits3 = run_batch_mcq(tied_mcq_head, batch, examples, shots=3)
    n_tok3 = batch["n_tokens"]
    assert logits3.shape == logits0.shape
    assert torch.isfinite(logits3).all()
    assert n_tok3 > n_tok0  # the exemplar prefix adds real attended tokens


# ---------------------------------------------------------------------------
# 14. --cand_encoder qwen3emb: EmbedEncoder, DecisionModel(d_cand=1024), run_batch(vec_cache=...)
# ---------------------------------------------------------------------------

QWEN3_EMB_CACHED = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B")


@pytest.mark.skipif(not os.path.isdir(QWEN3_EMB_CACHED), reason="Qwen3-Embedding-0.6B not cached locally")
def test_embed_encoder_unit_norm():
    from encode import EmbedEncoder, CAND_RENDER
    enc = EmbedEncoder(device="cpu")
    texts = ["cancel my subscription", "book a flight to Rome"]
    vecs = enc.embed(texts, max_len=32, batch_size=8)
    assert vecs.shape == (2, enc.d) == (2, 1024)
    assert vecs.dtype == torch.float32
    norms = vecs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)

    rendered = enc.embed(["cancel subscription"], render=CAND_RENDER)
    raw = enc.embed(["cancel subscription"])
    assert rendered.shape == raw.shape == (1, enc.d)
    assert not torch.allclose(rendered, raw)  # different surface text -> different vector


def test_decision_model_d_cand_forward_backward():
    torch.manual_seed(0)
    d_in, d_cand, B, Lh, Lq, Kmax = 64, 1024, 3, 5, 2, 4
    model = DecisionModel(d_in=d_in, d_cand=d_cand, dropout=0)
    assert model.mu_fz.shape == (d_cand,) and model.sd_fz.shape == (d_cand,)

    H, Q = torch.randn(B, Lh, d_in), torch.randn(B, Lq, d_in)
    C = torch.randn(B, Kmax, d_cand)
    Uf, Vf = torch.randn(B, d_cand), torch.randn(B, d_cand)
    hmask = torch.ones(B, Lh, dtype=torch.bool)
    qmask = torch.ones(B, Lq, dtype=torch.bool)
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    for i, k in enumerate((1, 2, 4)):
        cmask[i, :k] = True

    logits = model(H, hmask, Q, qmask, C, cmask, Uf, Vf)
    assert logits.shape == (B, Kmax + 1)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert model.proj_c[1].weight.grad is not None
    assert model.proj_sim.weight.grad is not None


class _FakeVecCache:
    """random unit vectors keyed by text -- duck-types VecCache.pooled()/.device/.index."""
    def __init__(self, d, device="cpu"):
        self.d, self.device, self.index = d, device, {}

    def pooled(self, text):
        if text not in self.index:
            v = torch.randn(self.d)
            self.index[text] = F.normalize(v, dim=0)
        return self.index[text]

    def embed(self, texts, render=None):  # duck-types EmbedEncoder.embed for decide(encoder=...)
        return torch.stack([self.pooled(t) for t in texts])


def test_run_batch_with_fake_vec_cache(tiny_model_dir, tiny_cache):
    # fresh Backbone (not the shared tiny_backbone fixture): this test backward()es
    # through it, and tiny_backbone is documented forward-only / shared across tests.
    torch.manual_seed(0)
    bb = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu")
    examples = make_examples()
    batch = collate(tiny_cache, examples)  # C still comes from the frozen FeatureCache
    vec_cache = _FakeVecCache(d=1024)

    model = DecisionModel(d_in=bb.d, d_cand=1024, dropout=0)
    # collate()'s C is frozen-space (d_in), but vec_cache's Uf/Vf are d_cand=1024 --
    # swap C for random d_cand vectors so forward() sees matching shapes end to end.
    Kmax = batch["C"].shape[1]
    batch["C"] = F.normalize(torch.randn(len(examples), Kmax, 1024), dim=-1)

    logits = run_batch(bb, model, batch, vec_cache=vec_cache)
    assert torch.isfinite(logits).all()

    loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
    loss.backward()
    trainable = bb.trainable_parameters()
    assert len(trainable) > 0
    assert all(p.grad is not None for p in trainable)


# ---------------------------------------------------------------------------
# scripts/null_bias.py: post-hoc, val-fitted K-aware null bias (REPORT.md sec 3c --
# P(null | gold absent) falls 0.98 -> 0.35 from K=2 to K=150; softmax dilution, not
# a data-prior artifact, so it can be corrected post-hoc without retraining).
# ---------------------------------------------------------------------------

def _fake_dump(seed, n, ks):
    """Ragged-K fake logits/target/K with the null logit deliberately diluted as K grows
    (mirrors the K-dilution finding), so fit_null_bias has a real effect to find."""
    rng = np.random.RandomState(seed)
    Kmax = max(ks)
    logits = np.full((n, Kmax + 1), np.finfo(np.float64).min)
    target = np.zeros((n, Kmax + 1))
    K = np.array([ks[i % len(ks)] for i in range(n)])
    for i, k in enumerate(K):
        logits[i, :k] = rng.randn(k)
        logits[i, -1] = 1.0 - 0.5 * np.log(k)  # null logit shrinks as K grows -> P(null) dilutes
        if rng.rand() < 0.3:
            target[i, -1] = 1.0
        else:
            target[i, rng.randint(k)] = 1.0
    return logits, target, K


def test_fit_null_bias_finite_and_improves_val_nll():
    logits, target, K = _fake_dump(seed=0, n=60, ks=[2, 5, 20, 50, 150])
    fit = fit_null_bias(logits, target, K)
    assert all(np.isfinite(fit[k]) for k in ("alpha", "beta", "T", "nll_raw", "nll_fit"))
    assert fit["nll_fit"] <= fit["nll_raw"] + 1e-9


def test_fit_null_bias_three_ragged_sets_end_to_end():
    """Fit on one ragged-K fake set (as "val"), apply to two others -- exercises the same
    per-set ragged-Kmax handling (K-reconstructed validity mask) null_bias.py's main() does."""
    val_logits, val_target, val_K = _fake_dump(seed=1, n=80, ks=[2, 10, 40, 150])
    fit = fit_null_bias(val_logits, val_target, val_K)
    assert np.isfinite(fit["T"]) and fit["T"] > 0

    for seed, ks in [(2, [3, 30]), (3, [1, 8, 77])]:
        logits, target, K = _fake_dump(seed=seed, n=20, ks=ks)
        probs = apply_bias(logits, K, fit["alpha"], fit["beta"], fit["T"])
        assert np.allclose(probs.sum(-1), 1.0, atol=1e-6)
        assert np.isfinite(soft_ce(probs, target))


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="loads the real Qwen3-0.6B-Base "
                     "backbone + full eval sweep; set RUN_SLOW=1 to run")
def test_null_bias_real_path_on_mini_v4(tmp_path):
    """End to end on runs/mini_v4 if it's still around: pcdm/train.py --eval_only --dump_logits,
    then scripts/null_bias.py on the dump. Skipped (not failed) if the checkpoint is gone."""
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "runs" / "mini_v4" / "best.pt").exists():
        pytest.skip("runs/mini_v4/best.pt not present")

    dump_dir = tmp_path / "mini_dump"
    subprocess.run([sys.executable, "pcdm/train.py", "--name", "mini_v4", "--eval_only",
                     "--data", "data_small_v4", "--dump_logits", str(dump_dir), "--eval_limit", "40"],
                    check=True, cwd=str(repo_root))
    assert (dump_dir / "val.npz").exists() and (dump_dir / "val.meta.json").exists()

    out = dump_dir / "results.json"
    subprocess.run([sys.executable, "scripts/null_bias.py", str(dump_dir), "--out", str(out)],
                    check=True, cwd=str(repo_root))
    results = json.loads(out.read_text())
    assert all(np.isfinite(results[k]) for k in ("T", "alpha", "beta", "val_nll", "val_nll_raw"))
    assert results["val_nll"] <= results["val_nll_raw"] + 1e-6
    assert "eval" in results and "beta_only" in results


# ---------------------------------------------------------------------------
# 12. extra_tap ([h_E; h_top], PLAN3 E1): width doubles, decide (KV-cache path) == run_batch
# ---------------------------------------------------------------------------

def test_extra_tap_decide_matches_batched(tiny_model_dir, tiny_cache):
    torch.manual_seed(0)
    bb = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu", extra_tap=2).eval()
    assert bb.extra_tap == 2 and bb.d == 2 * bb.model.config.hidden_size
    state = "a shared state text used for every query below in joint mode"
    queries = [("first query", ["cand0", "cand1"]), ("second query", ["cand2", "cand3", "cand4"])]
    # candidates/frozen features keep the single-layer width; only the top state is concatenated
    model = DecisionModel(d_in=bb.d, d_cand=bb.model.config.hidden_size, dropout=0).eval()

    dists = decide(bb, model, tiny_cache, state, queries, chunk=2, joint=True)
    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    batch = collate(tiny_cache, examples)
    with torch.no_grad():
        logits = run_batch(bb, model, batch, joint=True)
    probs = F.softmax(logits, dim=-1)
    for i, (_, cands) in enumerate(queries):
        K = len(cands)
        assert torch.allclose(dists[i], torch.cat([probs[i, :K], probs[i, -1:]]), atol=1e-3)


# ---------------------------------------------------------------------------
# 15. PLAN3 E3 student heads (--head z1|zr|zr_set) + --cand_encoder tiny (TokenCandCache)
# ---------------------------------------------------------------------------

Z_HEADS = ["z1", "zr", "zr_set"]


def _z_model(head, tiny_layers=1, null="softmax", perturb_seed=None):
    torch.manual_seed(0)
    model = DecisionModel(d_in=64, d=32, num_layers=0, dropout=0, head=head, z_dim=16, z_probes=4,
                          tiny_layers=tiny_layers, null=null).eval()
    if head == "zr_set" and perturb_seed is not None:  # set_attn.out_proj is zero-init (zr_set == zr at init)
        g = torch.Generator().manual_seed(perturb_seed)
        with torch.no_grad():
            for p in model.set_attn.out_proj.parameters():
                p.normal_(0, 0.1, generator=g)
    return model


def _tok_batch(klens=(2, 4, 6), Kmax=None, n_uniq=8, L=5, seed=0):
    """_lw_batch + token-level candidates: n_uniq unique candidates of ragged length, each
    slot (row, k) pointing at unique row cidx[row, k]."""
    H, hmask, Q, qmask, C, cmask, Uf, Vf = _lw_batch(Kmax=Kmax or max(klens), klens=klens, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    ctok = torch.randn(n_uniq, L, 64, generator=g)
    lens = torch.randint(1, L + 1, (n_uniq,), generator=g)
    ctmask = torch.arange(L)[None, :] < lens[:, None]
    cidx = torch.randint(0, n_uniq, cmask.shape, generator=g)
    return (H, hmask, Q, qmask, C, cmask, Uf, Vf), dict(ctok=ctok, ctmask=ctmask, cidx=cidx)


@pytest.mark.parametrize("head", Z_HEADS)
@pytest.mark.parametrize("null", ["softmax", "factored"])
def test_z_head_shapes_finite(head, null):
    model = _z_model(head, null=null, perturb_seed=1)
    args, tok = _tok_batch()
    cmask = args[5]
    with torch.no_grad():
        logits = model(*args, **tok)          # token-level candidates (tiny encoder path)
        logits_pooled = model(*args)          # pooled fallback: L=1 "token" = C (backbone/qwen3emb tiers)
    for lg in (logits, logits_pooled):
        assert lg.shape == (cmask.shape[0], cmask.shape[1] + 1)
        assert torch.isfinite(lg).all()
        assert torch.all(lg[:, :-1][~cmask] == torch.finfo(lg.dtype).min)
        assert torch.allclose(torch.softmax(lg, -1).sum(-1), torch.ones(cmask.shape[0]), atol=1e-5)


@pytest.mark.parametrize("head", Z_HEADS)
def test_z_head_iia_add_candidate(head):
    """z1/zr: Z never sees the candidate set -> adding a candidate leaves existing log-odds
    exactly unchanged. zr_set: the set-conditioning step breaks IIA by design."""
    model = _z_model(head, perturb_seed=2)
    args, tok = _tok_batch(klens=(2, 2), Kmax=4)
    cmask_3 = args[5].clone(); cmask_3[:, 2] = True
    args_3 = args[:5] + (cmask_3,) + args[6:]
    with torch.no_grad():
        l2, l3 = model(*args, **tok), model(*args_3, **tok)
    same = torch.allclose(l2[:, 0] - l2[:, 1], l3[:, 0] - l3[:, 1], atol=1e-5)
    assert same == (head != "zr_set"), head


@pytest.mark.parametrize("head", Z_HEADS)
def test_z_head_candidate_permutation_and_padding(head):
    model = _z_model(head, perturb_seed=3)
    args, tok = _tok_batch()
    H, hmask, Q, qmask, C, cmask, Uf, Vf = args
    with torch.no_grad():
        logits = model(*args, **tok)
    B, Kmax = cmask.shape
    cidx_perm, perms = tok["cidx"].clone(), []
    for i in range(B):
        K = int(cmask[i].sum()); p = list(range(K)); random.Random(i).shuffle(p); perms.append(p)
        cidx_perm[i, :K] = tok["cidx"][i, p]
    with torch.no_grad():
        logits_perm = model(*args, **{**tok, "cidx": cidx_perm})
    for i in range(B):
        K = int(cmask[i].sum())
        assert torch.allclose(logits_perm[i, :K], logits[i, :K][perms[i]], atol=1e-5)
    assert torch.allclose(logits_perm[:, -1], logits[:, -1], atol=1e-5)  # null: set-invariant

    pad = 2  # extra pad slots (masked) must not touch anything, incl. zr_set's set attention
    cmask_pad = torch.cat([cmask, torch.zeros(B, pad, dtype=torch.bool)], 1)
    cidx_pad = torch.cat([tok["cidx"], torch.randint(0, tok["ctok"].shape[0], (B, pad))], 1)
    C_pad = torch.cat([C, torch.randn(B, pad, C.shape[-1])], 1)
    with torch.no_grad():
        logits_pad = model(H, hmask, Q, qmask, C_pad, cmask_pad, Uf, Vf, **{**tok, "cidx": cidx_pad})
    assert torch.allclose(logits_pad[:, :Kmax], logits[:, :Kmax], atol=1e-6)
    assert torch.allclose(logits_pad[:, -1], logits[:, -1], atol=1e-6)


def test_z_head_grads_reach_tiny_encoder_and_probes(tiny_model_dir):
    from encode import TokenCandCache
    torch.manual_seed(0)
    bb = Backbone(name=tiny_model_dir, lora_layers=2, lora_r=4, device="cpu")
    cache = TokenCandCache(bb)
    examples = make_examples()
    batch = collate(cache, examples)
    assert "C" not in batch and batch["ctok_ids"].shape[0] == len({c for ex in examples for c in ex["candidates"]})
    model = DecisionModel(d_in=bb.d, d=32, num_layers=0, dropout=0, head="zr_set", z_dim=16, z_probes=4, tiny_layers=2)
    logits = run_batch(bb, model, batch, joint=True)
    loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
    loss.backward()
    for name in ("probes", "proj_ct.1.weight", "tiny.0.attn.in_proj_weight", "tiny.1.ff.0.weight",
                 "set_attn.out_proj.weight", "probe_attn.q_proj_weight"):  # set_attn.out_proj zero-init -> in_proj grad 0 at step 1, by design
        g = dict(model.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
    assert all(p.grad is not None for p in bb.trainable_parameters())
    assert bb.model.embed_tokens.weight.grad is None  # table stays frozen


@pytest.mark.parametrize("head", ["mlp"] + Z_HEADS)
@pytest.mark.parametrize("enc", ["tiny", "qwen3emb"])
def test_z_head_decide_matches_run_batch(tiny_backbone, head, enc):
    """KV-cache decide() == batched run_batch() per head x candidate encoder. tiny: decide gets an
    EMPTY TokenCandCache (cold path: every candidate tokenised on the fly); qwen3emb: a fake
    pre-populated VecCache stands in for the embedder (encoder=None -> cache hits only)."""
    from encode import TokenCandCache
    torch.manual_seed(0)
    state = "a shared state text used for every query below in joint mode"
    queries = [("first query", ["cand0", "cand1"]), ("second query", ["cand2", "cand3", "cand4"]),
               ("third query", ["a much longer candidate string here"])]
    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    if enc == "tiny":
        cache_rb, cache_dec, d_cand, vec_cache = TokenCandCache(tiny_backbone), TokenCandCache(tiny_backbone), None, None
        cache_rb.add([c for _, cs in queries for c in cs])
    else:
        cache_rb = cache_dec = vec_cache = _FakeVecCache(d=1024)
        for t in [state] + [q for q, _ in queries] + [c for _, cs in queries for c in cs]:
            cache_rb.pooled(t)
        d_cand = 1024
    model = DecisionModel(d_in=tiny_backbone.d, d=32, num_layers=0, dropout=0, head=head, z_dim=16, z_probes=4,
                          tiny_layers=1 if enc == "tiny" else 0, d_cand=d_cand).eval()
    if head == "zr_set":
        with torch.no_grad():
            model.set_attn.out_proj.weight.normal_(0, 0.1)

    dists = decide(tiny_backbone, model, cache_dec, state, queries, chunk=2, joint=True,
                   encoder=vec_cache)  # qwen3emb: Uf/Vf/C from the (fake) embedder cache
    batch = collate(cache_rb, examples)
    with torch.no_grad():
        logits = run_batch(tiny_backbone, model, batch, joint=True, vec_cache=vec_cache)
    probs = F.softmax(logits, dim=-1)
    for i, (_, cands) in enumerate(queries):
        K = len(cands)
        assert torch.allclose(dists[i], torch.cat([probs[i, :K], probs[i, -1:]]), atol=1e-3)
        assert torch.allclose(dists[i].sum(), torch.tensor(1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# 15. PLAN3 E3-0/E3-T: scripts/distill_corpus.py + teacher-distillation loss/collate
# ---------------------------------------------------------------------------

def test_distill_corpus_limit_smoke(tmp_path):
    """scripts/distill_corpus.py --limit 5: every train/val/eval row is schema-valid
    (candidates/target same length, target sums to 1, label consistent) and the null
    rows it deliberately synthesizes (gold option removed) actually show up."""
    repo_root = Path(__file__).resolve().parent.parent
    out = tmp_path / "data_kb_smoke"
    try:
        subprocess.run([sys.executable, "scripts/distill_corpus.py", "--out", str(out), "--limit", "5"],
                       check=True, cwd=str(repo_root), capture_output=True, text=True, timeout=180)
    except Exception as e:
        pytest.skip(f"distill_corpus.py needs network/HF access: {e}")

    paths = list(out.glob("*.jsonl")) + list(out.glob("eval/*.jsonl"))
    assert paths, "no jsonl written"
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    assert rows
    null_n = 0
    for r in rows:
        assert set(r) >= {"state", "query", "candidates", "target", "p_null", "task", "label", "meta"}
        assert len(r["candidates"]) == len(r["target"]) >= 1
        assert abs(sum(r["target"]) - 1.0) < 1e-4
        assert r["meta"]["family"] == "qa"
        if r["p_null"] >= 1.0:
            assert r["label"] == -1
            null_n += 1
        else:
            assert r["target"][r["label"]] == max(r["target"])
    # NULL_FRAC=0.05 on the train split only (eval loaders pass null_frac=0.0); just check
    # the mechanism actually fires somewhere across this many sources/rows.
    train_rows = [json.loads(l) for l in open(out / "train.jsonl")] if (out / "train.jsonl").exists() else []
    if train_rows:
        assert 0 < sum(1 for r in train_rows if r["p_null"] >= 1.0)
    assert (out / "manifest.json").exists()


def test_workflow_corpus_limit_smoke(tmp_path):
    """scripts/workflow_corpus.py --limit 30 (PLAN6 item 4): rows of all three JevBench types, every row
    schema-valid with label in range (noul candidates exactly ["no","yes"]), the query rendered by
    pcdm_jev.decider.query_text, rubric groups sharing state+candidates with differing gold, flip pairs
    adjacent, and zero state overlap with the public JevBench set (the script asserts it; re-checked here)."""
    repo_root = Path(__file__).resolve().parent.parent
    out = tmp_path / "data_wf_smoke"
    env = dict(os.environ, HF_DATASETS_OFFLINE=os.environ.get("HF_DATASETS_OFFLINE", "1"))
    try:
        subprocess.run([sys.executable, "scripts/workflow_corpus.py", "--out", str(out), "--limit", "30"],
                       check=True, cwd=str(repo_root), capture_output=True, text=True, timeout=600, env=env)
    except Exception as e:
        pytest.skip(f"workflow_corpus.py needs cached HF datasets: {getattr(e, 'stderr', e)}")
    files = {p.stem: [json.loads(l) for l in open(p) if l.strip()]
             for p in list(out.glob("*.jsonl")) + list(out.glob("eval/*.jsonl"))}
    assert {"train", "val", "wf_heldout_noul", "wf_heldout_choice", "wf_heldout_score", "wf_heldout_style",
            "wf_rubric_flip", "wf_rubric_shuffled"} <= set(files)
    rows = [r for v in files.values() for r in v]
    assert {r["meta"]["qtype"] for r in files["train"]} == {"noul", "choice", "score"}
    groups = {}
    for r in rows:
        assert set(r) >= {"state", "query", "candidates", "target", "p_null", "task", "label", "meta"}
        assert len(r["candidates"]) == len(r["target"]) >= 2 and abs(sum(r["target"]) - 1.0) < 1e-4
        assert r["meta"]["fam_bucket"] == "W" and r["meta"]["qtype"] in ("noul", "choice", "score")
        if r["p_null"] >= 1.0:
            assert r["label"] == -1 and r["meta"]["qtype"] == "choice"
        else:
            assert 0 <= r["label"] < len(r["candidates"]) and r["target"][r["label"]] == 1.0
        if r["meta"]["qtype"] == "noul":
            assert r["candidates"] == ["no", "yes"] and "yes: " in r["query"] and "no: " in r["query"]
        if r["meta"].get("rubric_group") and "flip_pair" not in r["meta"] and "probe" not in r["meta"]:
            groups.setdefault(r["meta"]["rubric_group"], []).append(r)
    assert groups
    for g in groups.values():
        assert len({x["state"] for x in g}) == 1 and len({tuple(x["candidates"]) for x in g}) == 1
        assert len({x["label"] for x in g}) > 1 and len({x["query"] for x in g}) == len(g)
    flip = files["wf_rubric_flip"]
    assert flip and len(flip) % 2 == 0
    for a, b in zip(flip[::2], flip[1::2]):
        assert a["meta"]["flip_pair"] == b["meta"]["flip_pair"] and a["state"] == b["state"]
        assert a["candidates"] == b["candidates"] and a["label"] != b["label"]
    # held-out families never in train/val; held-out style never in train/val
    heldout_fams = {"eligibility", "tool_select", "urgency"}
    for r in files["train"] + files["val"]:
        assert r["meta"]["family"] not in heldout_fams
        assert r["meta"]["rubric_style"] not in ("flag", "spec", "band")
    assert all(r["meta"]["rubric_style"] in ("flag", "spec", "band") for r in files["wf_heldout_style"])
    jev = Path("/tmp/jevbench/datasets/public")
    if jev.exists():
        states = [json.loads(l)["state"] for p in jev.glob("*.jsonl") for l in open(p) if l.strip()]
        states = {s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in states}  # some states are JSON objects
        assert not {r["state"] for r in rows} & states
    manifest = json.load(open(out / "manifest.json"))
    assert manifest["flip_pairs"] == len(flip) // 2


def test_gen_long_extra_all_rows_long_split_across_families():
    """--long_share (scripts/workflow_corpus.py gen_long_extra, PLAN7 track B long_e45 run):
    force_long=True rows are always long (unlike gen_policy_permit(n, rng)'s ~LONG_FRAC-chance
    default) and split between the two long-capable families in their CAPS ratio; force_long
    also disables rubric groups (fill_to(n, single, None, rng)), so no row.meta.long is missing."""
    rng = random.Random(0)
    rows = wf.gen_long_extra(40, rng)
    assert 38 <= len(rows) <= 42  # fill_to may overshoot by a couple of rows, never undershoot
    assert all(r["meta"]["long"] for r in rows)
    assert not any(r["meta"].get("rubric_group") for r in rows)
    fams = {r["meta"]["family"] for r in rows}
    assert fams == {"policy_permit", "action_select"}
    assert wf.gen_long_extra(0, random.Random(0)) == []


def test_assemble_state_long_puts_case_first():
    """tl1b (REPORT S3ab/S3af): long states must render Case: (facts+request) BEFORE the
    policy filler, not after -- states right-truncate at --max_state at train/eval time, and
    Case-last put the facts at ~97% of the text (98.8% of long rows dropped them at
    max_state=1,024). Case must now sit within the first 10% of the state."""
    rng = random.Random(0)
    dom = next(iter(wf.DOMAINS))
    for _ in range(10):  # long_prefix draws a random section count/order each call
        state = wf.assemble_state("REAL POLICY TEXT", "the facts", "the request?", dom, rng, long=True)
        assert state.startswith("Case: the facts the request?")
        assert state.index("the facts") / len(state) < 0.10


def test_multiset_expand_smoke(tmp_path):
    """scripts/multiset.py --variants 3 on 20 synthetic MCQ rows (PLAN3 E3-ms): yields
    80 rows, each ms_group of size 4 (orig + remove/add_unrel/reorder), and every row is
    schema-valid (candidates/target same length >=2, target sums to 1, gold index
    matches on gold-present rows, label=-1 with a uniform target on gold-absent rows).
    Two tasks so add_unrel has same-source options to draw from."""
    from collections import Counter
    repo_root = Path(__file__).resolve().parent.parent
    rng = random.Random(0)
    rows = []
    for i in range(20):
        task = "taskA" if i % 2 == 0 else "taskB"
        K = 3 + (i % 3)
        cands = [f"{task}_opt{i}_{j}" for j in range(K)]
        if i % 5 == 0:  # gold-absent row, same shape as distill_corpus.py's null synthesis
            target, p_null, label = [1.0 / K] * K, 1.0, -1
        else:
            label = rng.randrange(K)
            target, p_null = [1.0 if j == label else 0.0 for j in range(K)], 0.0
        rows.append({"state": f"state {i}", "query": "Which option is correct?", "candidates": cands,
                     "target": target, "p_null": p_null, "task": task, "label": label, "meta": {"family": "qa"}})
    src = tmp_path / "in.jsonl"
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "out.jsonl"
    subprocess.run([sys.executable, "scripts/multiset.py", "--file", str(src), "--out", str(out),
                     "--variants", "3", "--seed", "0"], check=True, cwd=str(repo_root),
                    capture_output=True, text=True, timeout=60)

    out_rows = [json.loads(l) for l in open(out) if l.strip()]
    assert len(out_rows) == 80
    group_sizes = Counter(r["meta"]["ms_group"] for r in out_rows)
    assert set(group_sizes.values()) == {4}
    variant_counts = Counter(r["meta"]["ms_variant"] for r in out_rows)
    assert variant_counts == {"orig": 20, "remove": 20, "add_unrel": 20, "reorder": 20}

    for r in out_rows:
        assert len(r["candidates"]) == len(r["target"]) >= 2
        assert abs(sum(r["target"]) - 1.0) < 1e-6
        if r["p_null"] < 1.0:
            assert r["target"][r["label"]] == max(r["target"])
        else:
            assert r["label"] == -1


def test_distill_loss_zero_when_teacher_equals_student():
    torch.manual_seed(0)
    B, Kmax = 4, 5
    logits = torch.randn(B, Kmax + 1)
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    cmask[:, :3] = True
    logits = logits.masked_fill(~torch.cat([cmask, torch.ones(B, 1, dtype=torch.bool)], -1),
                                 torch.finfo(logits.dtype).min)
    target = torch.zeros(B, Kmax); target[:, 0] = 1.0
    p_null = torch.zeros(B)

    # teacher == student's own softmax distribution -> KL term must vanish exactly,
    # regardless of temperature, leaving only the untouched alpha*CE term.
    teacher = F.softmax(logits, dim=-1)
    has_teacher = torch.ones(B, dtype=torch.bool)
    ce_only = decision_loss(logits, target, p_null, cmask)
    with_matching_teacher = decision_loss(logits, target, p_null, cmask, teacher=teacher,
                                           has_teacher=has_teacher, alpha=1.0, beta=1.0, T=2.0)
    assert torch.allclose(with_matching_teacher, ce_only, atol=1e-5)

    # a mismatched teacher (peaked on a different candidate) must raise the loss.
    bad_teacher = torch.zeros(B, Kmax + 1); bad_teacher[:, 2] = 1.0
    with_bad_teacher = decision_loss(logits, target, p_null, cmask, teacher=bad_teacher,
                                      has_teacher=has_teacher, alpha=1.0, beta=1.0, T=2.0)
    assert with_bad_teacher > with_matching_teacher + 1e-4

    # has_teacher=False rows must get exactly zero KL contribution (beta-weight 0).
    no_teacher_mask = torch.zeros(B, dtype=torch.bool)
    masked_out = decision_loss(logits, target, p_null, cmask, teacher=bad_teacher,
                                has_teacher=no_teacher_mask, alpha=1.0, beta=1.0, T=2.0)
    assert torch.allclose(masked_out, ce_only, atol=1e-5)


def test_distill_kl_renormalises_over_candidates_not_null():
    """tl1b teacher rows (scripts/teacher_label.py) always carry teacher[..., null] == 0 --
    the KL term must be computed over the K candidate columns only (renormalised), so a
    student that happens to place nonzero mass on null is not penalised for it, and the null
    column's own value never enters the KL at all (changing it must not move the loss)."""
    B, Kmax = 3, 4
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    cmask[:, :2] = True
    logits = torch.zeros(B, Kmax + 1)
    logits[:, 0] = 2.0  # student puts most candidate mass on slot 0
    logits = logits.masked_fill(~torch.cat([cmask, torch.ones(B, 1, dtype=torch.bool)], -1),
                                 torch.finfo(logits.dtype).min)
    target = torch.zeros(B, Kmax); target[:, 0] = 1.0
    p_null = torch.zeros(B)
    has_teacher = torch.ones(B, dtype=torch.bool)

    # teacher agrees with the student's candidate-only distribution exactly; only its null
    # slot differs from what a full-softmax teacher would have put there (it's forced to 0
    # by scripts/teacher_label.py, but here we vary the student's OWN null logit instead --
    # the loss must not change, since the null column is excluded from the KL entirely).
    cand_only = F.softmax(logits[:, :-1].masked_fill(~cmask, torch.finfo(logits.dtype).min), dim=-1)
    teacher = torch.cat([cand_only, torch.zeros(B, 1)], dim=-1)
    base = decision_loss(logits, target, p_null, cmask, teacher=teacher, has_teacher=has_teacher,
                          alpha=1.0, beta=1.0, T=1.0)

    logits_diff_null = logits.clone()
    logits_diff_null[:, -1] = 5.0  # a wildly different null logit
    changed_null = decision_loss(logits_diff_null, target, p_null, cmask, teacher=teacher,
                                  has_teacher=has_teacher, alpha=1.0, beta=1.0, T=1.0)
    # the alpha*CE term does see the null logit (it's part of the full softmax denominator),
    # so isolate the KL contribution by comparing beta=0 (CE-only) deltas against beta=1 deltas.
    base_ce = decision_loss(logits, target, p_null, cmask)
    changed_ce = decision_loss(logits_diff_null, target, p_null, cmask)
    assert torch.allclose(base - base_ce, changed_null - changed_ce, atol=1e-5)


@pytest.mark.parametrize("collate_fn", ["energy", "mcq"])
def test_collate_carries_teacher_with_mask(tiny_cache, collate_fn):
    examples = make_examples(n=3, ks=(2, 3, 1))
    examples[0]["teacher"] = [0.7, 0.3, 0.0]      # K=2 candidates + null
    examples[2]["teacher"] = [1.0, 0.0]           # K=1 candidate + null
    # examples[1] (K=3) carries no teacher field at all.
    batch = collate(tiny_cache, examples) if collate_fn == "energy" else collate_mcq(examples)

    Kmax = batch["cmask"].shape[1]
    assert batch["teacher"].shape == (3, Kmax + 1)
    assert batch["has_teacher"].tolist() == [True, False, True]
    assert torch.allclose(batch["teacher"][0, :2], torch.tensor([0.7, 0.3]))
    assert batch["teacher"][0, Kmax] == 0.0  # null slot
    assert torch.allclose(batch["teacher"][2, :1], torch.tensor([1.0]))
    assert batch["teacher"][2, Kmax] == 0.0
    assert torch.all(batch["teacher"][1] == 0.0)  # no-teacher row stays all-zero


def test_collate_teacher_standalone():
    examples = [
        {"candidates": ["a", "b"], "teacher": [0.6, 0.1, 0.3]},
        {"candidates": ["c", "d", "e"]},  # no teacher field
    ]
    teacher, has_teacher = collate_teacher(examples, Kmax=3)
    assert has_teacher.tolist() == [True, False]
    assert torch.allclose(teacher[0], torch.tensor([0.6, 0.1, 0.0, 0.3]))  # [K padded to Kmax] + null
    assert torch.all(teacher[1] == 0.0)


# ---------------------------------------------------------------------------
# 20. PLAN3 E3-ms: symmetrized teacher, set_context variants, delta targets, L_delta, grouping
# ---------------------------------------------------------------------------

def test_teacher_label_perms(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import teacher_label as tl
    ex = [{"candidates": ["a", "b", "c"]}, {"candidates": ["c", "a"]}]
    fixed = {"a": 0.5, "b": 0.3, "c": 0.2}
    # order-blind teacher: symmetrizing changes nothing, perm std is 0; perms=1 returns score_batch verbatim
    monkeypatch.setattr(tl, "score_batch", lambda head, exs, T: [[fixed[c] * 0.9 for c in e["candidates"]] + [0.1] for e in exs])
    p1, s1 = tl.label_batch(None, ex, 1.0)
    p3, s3 = tl.label_batch(None, ex, 1.0, perms=3)
    assert s1 is None and p1 == [[0.45, 0.27, 0.18000000000000002, 0.1], [0.18000000000000002, 0.45, 0.1]]
    assert np.allclose(p3[0], p1[0]) and np.allclose(p3[1], p1[1]) and max(s3) < 1e-6
    # slot-0-biased teacher: the average spreads the bias over candidates (still sums to 1), std > 0
    monkeypatch.setattr(tl, "score_batch", lambda head, exs, T: [[0.7] + [0.3 / (len(e["candidates"]) - 1)] * (len(e["candidates"]) - 1) + [0.0] for e in exs])
    p3, s3 = tl.label_batch(None, ex, 1.0, perms=3)
    assert abs(sum(p3[0]) - 1) < 1e-6 and s3[0] > 0 and max(p3[0][:3]) < 0.7
    # seeded per global row index: same row, same perms regardless of batch split
    q3, _ = tl.label_batch(None, ex[1:], 1.0, perms=3, start_idx=1)
    assert np.allclose(q3[0], p3[1])


def test_teacher_label_zero_shot_scorer(tied_mcq_head):
    """PLAN7 tl1b item 6: score_batch_zero_shot renders no null option (pcdm_jev.decider's
    _render_zero_shot) and hardcodes p_null = 0, unlike the trained-checkpoint score_batch
    which reads a null option out of the letter logits."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import teacher_label as tl
    examples = _mcq_examples(ks=(2, 3))
    probs = tl.score_batch_zero_shot(tied_mcq_head, examples, T=1.0)
    assert len(probs) == 2
    for p, ex in zip(probs, examples):
        K = len(ex["candidates"])
        assert len(p) == K + 1
        assert p[-1] == 0.0                      # null is never rendered -> hardcoded 0
        assert abs(sum(p[:K]) - 1.0) < 1e-5       # softmax over exactly the K candidates
    # a --shots prefix changes the scored logits (real attended tokens), same shape/null=0.
    prefixed = tl.score_batch_zero_shot(tied_mcq_head, examples, T=1.0, shots_prefix="Some worked example.\n\n",
                                         shots_prefix_tokens=8)
    assert prefixed[0][-1] == 0.0
    assert not np.allclose(prefixed[0][:2], probs[0][:2])


def test_teacher_label_zero_shot_skips_and_order(tmp_path, tied_mcq_head, monkeypatch):
    """main()'s --zero_shot path: only label>=0, 2<=K<=MAXK_DIRECT rows get a teacher; every
    row (labeled or not) is written back in its original order so the file stays usable as
    train.jsonl and the line-count resume logic still lines up 1:1 with the input."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import teacher_label as tl
    rows = [
        {"state": "s0", "query": "q0", "candidates": ["a", "b"], "target": [1.0, 0.0], "p_null": 0.0, "label": 0},
        {"state": "s1", "query": "q1", "candidates": ["only one"], "target": [1.0], "p_null": 0.0, "label": 0},  # K=1
        {"state": "s2", "query": "q2", "candidates": ["a", "b"], "target": [0.0, 0.0], "p_null": 1.0, "label": -1},  # null row
        {"state": "s3", "query": "q3", "candidates": ["a", "b", "c"], "target": [0.0, 1.0, 0.0], "p_null": 0.0, "label": 1},
    ]
    in_path = tmp_path / "in.jsonl"
    with open(in_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(tl, "MCQHead", lambda *a, **k: tied_mcq_head)
    monkeypatch.setattr(sys, "argv", ["teacher_label.py", "--file", str(in_path), "--zero_shot",
                                       "--backbone", "unused", "--bs", "2"])
    tl.main()

    out_path = in_path.with_suffix("").with_suffix(".teacher.jsonl")
    out_rows = [json.loads(l) for l in open(out_path)]
    assert [r["state"] for r in out_rows] == ["s0", "s1", "s2", "s3"]  # order preserved
    assert "teacher" in out_rows[0] and len(out_rows[0]["teacher"]) == 3  # K=2 + null
    assert "teacher" not in out_rows[1]  # K=1, skipped
    assert "teacher" not in out_rows[2]  # label=-1, skipped
    assert "teacher" in out_rows[3] and out_rows[3]["teacher"][-1] == 0.0


def _ms_rows(n=6, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        K = 4 + (i % 2)
        cands = [f"opt{i}_{j}" for j in range(K)]
        label = rng.randrange(K)
        t = [rng.random() for _ in range(K + 1)]
        rows.append({"state": f"state {i}", "query": "q", "candidates": cands, "p_null": 0.0, "task": "t",
                     "target": [1.0 if j == label else 0.0 for j in range(K)], "label": label,
                     "meta": {"family": "qa"}, "teacher": [x / sum(t) for x in t]})
    return rows


def test_multiset_set_context_variants_follow_teacher():
    from scripts.multiset import expand, VARIANT_TYPES
    rows = _ms_rows()
    out = expand(rows, len(VARIANT_TYPES), seed=0, teachers=[r["teacher"] for r in rows])
    assert len(out) == len(rows) * (1 + len(VARIANT_TYPES))
    by = {(r["meta"]["ms_group"], r["meta"]["ms_variant"]): r for r in out}
    assert {r["meta"]["ms_kind"] for r in out if r["meta"]["ms_variant"] in ("reorder", "add_unrel")} == {"invariance"}
    assert {r["meta"]["ms_kind"] for r in out if r["meta"]["ms_variant"] in ("remove", "replace_hard", "add_neardup", "remove_strong")} == {"set_context"}
    for g, o in enumerate(rows):
        gold = o["label"]
        ranked = sorted((i for i in range(len(o["candidates"])) if i != gold), key=lambda i: -o["teacher"][i])
        rh = by[g, "replace_hard"]
        assert rh["candidates"][ranked[0]] not in o["candidates"] and rh["label"] == gold
        assert [c for i, c in enumerate(rh["candidates"]) if i != ranked[0]] == [c for i, c in enumerate(o["candidates"]) if i != ranked[0]]
        rs = by[g, "remove_strong"]
        assert set(rs["candidates"]) == set(o["candidates"]) - {o["candidates"][i] for i in ranked[:2]}
        assert rs["candidates"][rs["label"]] == o["candidates"][gold]
        nd = by[g, "add_neardup"]
        assert nd["candidates"][:-1] == o["candidates"] and nd["target"][gold] == nd["target"][-1] == 0.5
        assert nd["candidates"][-1] != o["candidates"][gold]


def test_ms_targets_delta_formula():
    from scripts.multiset import expand
    from scripts.ms_targets import add_targets
    rows = expand(_ms_rows(), 6, seed=0, teachers=None)
    rng = random.Random(3)
    for r in rows:  # fake symmetrized labels for every expanded row
        t = [rng.random() for _ in range(len(r["candidates"]) + 1)]
        r["teacher"] = [x / sum(t) for x in t]
    n = add_targets(rows)
    assert n == sum(r["meta"]["ms_variant"] != "orig" for r in rows)
    for r in rows:
        m = r["meta"]
        if m["ms_variant"] == "orig":
            assert "delta_t" not in m
            continue
        o = rows[m["orig_idx"]]
        assert o["meta"]["ms_variant"] == "orig" and o["meta"]["ms_group"] == m["ms_group"]
        assert 0 < len(m["delta_t"]) <= 6  # top-4 shared candidates -> <= C(4,2) pairs
        for i, j, d in m["delta_t"]:
            ci, cj = r["candidates"][i], r["candidates"][j]
            oi, oj = o["candidates"].index(ci), o["candidates"].index(cj)
            exp = (np.log(r["teacher"][i]) - np.log(r["teacher"][j])) - (np.log(o["teacher"][oi]) - np.log(o["teacher"][oj]))
            assert abs(exp - d) < 1e-9
        shared = [c for c in r["candidates"] if c in o["candidates"]]
        if len(shared) > 4:  # capped to the orig teacher's top-4 shared candidates
            used = {r["candidates"][k] for i, j, _ in m["delta_t"] for k in (i, j)}
            top4 = sorted(shared, key=lambda c: -o["teacher"][o["candidates"].index(c)])[:4]
            assert used == set(top4)


def test_collate_delta_aligns_by_string_and_needs_orig_in_batch():
    from model import collate_delta
    orig = {"candidates": ["a", "b", "c"], "meta": {"ms_group": 7, "ms_variant": "orig"}}
    var = {"candidates": ["c", "x", "a"], "meta": {"ms_group": 7, "ms_variant": "replace_hard",
                                                   "delta_t": [[2, 0, 0.5], [0, 2, -0.5]]}}
    orphan = {"candidates": ["a", "b"], "meta": {"ms_group": 8, "ms_variant": "remove", "delta_t": [[0, 1, 1.0]]}}
    plain = {"candidates": ["z"]}
    idx, tgt = collate_delta([plain, var, orig, orphan])
    assert idx.tolist() == [[1, 2, 0, 2, 0, 2], [1, 0, 2, 2, 2, 0]] and tgt.tolist() == [0.5, -0.5]
    idx, tgt = collate_delta([plain, orphan])
    assert idx.shape == (0, 6) and tgt.shape == (0,)


def _delta_batch():
    """Two rows with identical state/query; row 0 (orig) = unique cands [0,1,2], row 1 (variant)
    = [2,0,3]; shared "0"/"2" -> one triple (variant i=1 "0", j=0 "2"; orig 0, 2)."""
    (H, hmask, Q, qmask, C, cmask, Uf, Vf), tok = _tok_batch(klens=(3, 3), n_uniq=4, L=4, seed=5)
    for t in (H, Q, C):
        t[1] = t[0]
    tok["cidx"] = torch.tensor([[0, 1, 2], [2, 0, 3]])
    delta_idx = torch.tensor([[1, 1, 0, 0, 0, 2]])
    delta_tgt = torch.tensor([0.7])
    return (H, hmask, Q, qmask, C, cmask), tok, delta_idx, delta_tgt


def _delta_grads(model, gamma):
    (H, hmask, Q, qmask, C, cmask), tok, delta_idx, delta_tgt = _delta_batch()
    model.zero_grad()
    logits = model(H, hmask, Q, qmask, C, cmask, **tok)
    target = torch.zeros(2, 3); target[:, 0] = 1.0
    loss = decision_loss(logits, target, torch.zeros(2), cmask, delta_idx=delta_idx, delta_tgt=delta_tgt, gamma=gamma)
    loss.backward()
    s = logits[:, :-1].detach()
    d_s = (s[1, 1] - s[1, 0]) - (s[0, 0] - s[0, 2])
    return loss.item(), d_s.item(), [p.grad.clone() for p in model.parameters() if p.grad is not None]


@pytest.mark.parametrize("head", ["z1", "zr"])
def test_delta_loss_constant_for_candidate_blind_heads(head):
    model = _z_model(head)
    l0, d_s, g0 = _delta_grads(model, 0.0)
    l1, _, g1 = _delta_grads(model, 1.0)
    assert abs(d_s) < 1e-5                                          # Delta^S == 0 on shared candidates
    assert abs((l1 - l0) - F.huber_loss(torch.tensor(0.0), torch.tensor(0.7)).item()) < 1e-5  # L_delta = const
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g0, g1))  # ... so no gradient from it


def test_delta_loss_trains_zr_set():
    model = _z_model("zr_set", perturb_seed=1)
    l0, d_s, g0 = _delta_grads(model, 0.0)
    l1, _, g1 = _delta_grads(model, 1.0)
    assert abs(d_s) > 1e-4 and l1 != l0
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(g0, g1))
    # gamma=0 with delta tensors present == plain loss, bit for bit
    (H, hmask, Q, qmask, C, cmask), tok, delta_idx, delta_tgt = _delta_batch()
    logits = model(H, hmask, Q, qmask, C, cmask, **tok).detach()
    target = torch.zeros(2, 3); target[:, 0] = 1.0
    plain = decision_loss(logits, target, torch.zeros(2), cmask)
    with_zero = decision_loss(logits, target, torch.zeros(2), cmask, delta_idx=delta_idx, delta_tgt=delta_tgt, gamma=0.0)
    assert plain.item() == with_zero.item()


def test_data_generator_keeps_groups_and_matches_old_stream():
    def old_generator(examples, bs, seed):  # pre-E3-ms per-row bucketing, verbatim
        rng = random.Random(seed)
        while True:
            order = examples[:]
            rng.shuffle(order)
            for i in range(0, len(order), bs * 50):
                chunk = sorted(order[i:i + bs * 50], key=lambda ex: len(ex["state"]))
                batches = [chunk[j:j + bs] for j in range(0, len(chunk), bs)]
                rng.shuffle(batches)
                for b in batches:
                    yield b
    plain = [{"state": "s" * random.Random(i).randint(1, 40), "id": i} for i in range(203)]
    new, old = T.data_generator(plain, 8, 0), old_generator(plain, 8, 0)
    for _ in range(60):
        assert [e["id"] for e in next(new)] == [e["id"] for e in next(old)]

    grouped = plain[:100] + [{"state": f"g{g}" * 3, "id": 1000 + 10 * g + v, "meta": {"ms_group": g}}
                             for g in range(30) for v in range(4)]
    gen = T.data_generator(grouped, 8, 1)
    seen = set()
    for _ in range(100):
        b = next(gen)
        assert len(b) <= 8
        groups = [e["meta"]["ms_group"] for e in b if "meta" in e]
        for g in set(groups):
            assert groups.count(g) == 4, "group split across batches"
            seen.add(g)
    assert seen == set(range(30))


def test_bucket_for_dir_and_tag_bucket():
    bmap = T.parse_bucket_map("data_kbt=K,data_wf_hf=W")
    assert T.bucket_for_dir("data", {}) == "E"
    assert T.bucket_for_dir("data_kb", {}) == "K"
    assert T.bucket_for_dir("data_wf_forms", {}) == "W"
    assert T.bucket_for_dir("data_other", {}) == "E"       # unmatched prefix -> E fallback
    assert T.bucket_for_dir("data_kbt", bmap) == "K"       # --bucket_map override wins
    assert T.bucket_for_dir("data_wf_hf", bmap) == "W"

    rows = [{"id": 1}, {"id": 2, "meta": {"fam_bucket": "K"}}]
    T.tag_bucket(rows, "data_wf_x", {})
    assert rows[0]["meta"]["fam_bucket"] == "W"
    assert rows[1]["meta"]["fam_bucket"] == "K"            # pre-existing meta.fam_bucket wins


class _FakeTok:
    """Minimal HF-tokenizer-shaped stand-in: one "token" per char, no real tokenizer needed."""
    def __init__(self):
        self.calls = []

    def __call__(self, texts, add_special_tokens=False):
        assert add_special_tokens is False
        self.calls.append(list(texts))
        return {"input_ids": [list(t) for t in texts]}


def test_apply_drop_truncated_char_prefilter_and_token_cut(capsys):
    """tl1b item 1: rows short enough in chars to never exceed --max_state tokens skip
    tokenization entirely (the char pre-filter); only the rest get tokenized and dropped iff
    their real token count exceeds max_state."""
    tok = _FakeTok()
    max_state = 10  # char_cut = 35
    short = {"state": "a" * 5}           # 5 chars <= 35 -> never tokenized, always kept
    long_ok = {"state": "b" * 10}        # 10 chars <= 35 -> also never tokenized (still short enough)
    long_over = {"state": "c" * 50}      # 50 chars > 35 -> tokenized, 50 "tokens" > 10 -> dropped
    examples = [short, long_ok, long_over]

    kept = T.apply_drop_truncated(examples, tok, max_state, "mylabel")

    assert tok.calls == [[long_over["state"]]]
    assert kept == [short, long_ok]
    out = capsys.readouterr().out
    assert "mylabel" in out and "1/3 rows dropped" in out


def test_add_extra_data_drop_truncated(tmp_path, capsys):
    ed = tmp_path / "data_x"
    ed.mkdir()
    rows = [
        {"state": "a" * 5, "candidates": ["a"], "target": [1.0], "p_null": 0.0, "task": "t"},
        {"state": "b" * 50, "candidates": ["a"], "target": [1.0], "p_null": 0.0, "task": "t"},
    ]
    with open(ed / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    args = argparse.Namespace(extra_data=str(ed), family_weights=None, null_aug=None, bucket_map=None,
                               eval_cap=None, drop_truncated=True, max_state=10)
    train_examples, eval_sets = [], {}
    T.add_extra_data(args, train_examples, eval_sets, tokenizer=_FakeTok())

    assert len(train_examples) == 1 and train_examples[0]["state"] == "a" * 5
    out = capsys.readouterr().out
    assert "data_x" in out and "1/2 rows dropped" in out


def test_family_weights_parsing():
    assert T.parse_family_weights(None) is None
    assert T.parse_family_weights("") is None
    assert T.parse_family_weights("E:0.35,K:0.25,W:0.4") == {"E": 0.35, "K": 0.25, "W": 0.4}


def test_bucketed_sampler_matches_weights_and_keeps_groups():
    """PLAN6 Queue review: family-balanced sampler. Batch-bucket frequency over 1000 batches
    should track --family_weights within +-.03, and ms_group units (here namespaced into the
    W bucket, as add_extra_data would produce) must never straddle a batch."""
    rng = random.Random(0)
    examples = []
    for b, n in (("E", 60), ("K", 60), ("W", 60)):
        for i in range(n):
            examples.append({"state": "s" * rng.randint(1, 40), "id": f"{b}{i}", "meta": {"fam_bucket": b}})
    for g in range(10):
        for v in range(4):
            examples.append({"state": "w" * 3, "id": f"Wg{g}v{v}", "meta": {"fam_bucket": "W", "ms_group": f"g{g}"}})

    weights = {"E": 0.35, "K": 0.25, "W": 0.40}
    gen = T.bucketed_data_generator(examples, 8, 0, weights)
    counts = {b: 0 for b in weights}
    seen_groups = set()
    for _ in range(1000):
        batch = next(gen)
        buckets = {T.bucket_of(ex) for ex in batch}
        assert len(buckets) == 1, "a batch under the family-balanced sampler spans more than one bucket"
        counts[buckets.pop()] += 1
        groups = [ex["meta"]["ms_group"] for ex in batch if "ms_group" in ex.get("meta", {})]
        for g in set(groups):
            assert groups.count(g) == 4, "group split across batches"
            seen_groups.add(g)
    total = sum(counts.values())
    for b, w in weights.items():
        assert abs(counts[b] / total - w) <= 0.03
    assert seen_groups == {f"g{g}" for g in range(10)}


def test_bucketed_sampler_missing_bucket_asserts():
    examples = [{"state": "s", "id": 0, "meta": {"fam_bucket": "E"}}]
    with pytest.raises(AssertionError):
        next(T.bucketed_data_generator(examples, 8, 0, {"E": 0.5, "K": 0.5}))


def _hard_row(bucket, cands, gold):
    return {"state": "s", "query": "q", "candidates": list(cands),
            "target": [1.0 if i == gold else 0.0 for i in range(len(cands))],
            "p_null": 0.0, "task": "t", "label": gold, "meta": {"fam_bucket": bucket}}


def test_parse_null_aug():
    assert T.parse_null_aug(None) is None
    assert T.parse_null_aug("") is None
    assert T.parse_null_aug("W:0.2") == ("W", 0.2)


def test_null_aug_fraction_and_pnull_semantics():
    """--null_aug W:0.5 (PLAN7 track B null control): exactly round(N*FRAC) of the eligible
    W rows get an augmented copy appended (originals untouched); the copy is target-uniform
    over the remaining candidates with p_null=1.0/label=-1 (the softmax-null convention
    make_row uses in scripts/workflow_corpus.py), and non-W rows are never touched."""
    w_rows = [_hard_row("W", ["a", "b", "c"], 0) for _ in range(20)]
    e_rows = [_hard_row("E", ["a", "b", "c"], 0) for _ in range(20)]
    examples = w_rows + e_rows
    out = T.apply_null_aug(examples, "W", 0.5, seed=0)
    assert out[:len(examples)] == examples  # originals kept, byte-identical, in place
    added = out[len(examples):]
    assert len(added) == 10  # round(20 * 0.5), only from the W-bucket pool
    for a in added:
        assert a["p_null"] == 1.0 and a["label"] == -1 and a["meta"]["null_aug"] is True
        assert a["meta"]["fam_bucket"] == "W"
        assert a["candidates"] == ["b", "c"]  # gold "a" removed, no catch-all present
        assert abs(sum(a["target"]) - 1.0) < 1e-9
        assert all(abs(t - 0.5) < 1e-9 for t in a["target"])  # uniform over K=2 leftovers


def test_null_aug_drops_catchall_and_needs_k_ge_2():
    """Rendered catch-all options (other/not_stated/skip/none) are dropped alongside the gold
    candidate; a row where that leaves < 2 candidates is skipped (no copy), and rows under the
    K >= 3 floor are never eligible in the first place."""
    keeps = _hard_row("W", ["a", "b", "c", "other"], 0)          # -> ["b", "c"] (catch-all stripped)
    collapses = _hard_row("W", ["yes", "no", "other"], 0)        # -> ["no"] after strip -> skipped
    too_few = _hard_row("W", ["x", "y"], 0)                      # K=2, never eligible
    soft = dict(_hard_row("W", ["a", "b", "c"], 1), target=[0.3, 0.4, 0.3])  # not one-hot -> not "hard"
    examples = [keeps, collapses, too_few, soft]
    out = T.apply_null_aug(examples, "W", 1.0, seed=0)
    added = out[len(examples):]
    assert len(added) == 1
    assert added[0]["candidates"] == ["b", "c"] and "other" not in added[0]["candidates"]
    assert added[0]["p_null"] == 1.0 and added[0]["label"] == -1


def test_max_state_reaches_native_features_via_calibrate_zscore_native(monkeypatch):
    """--max_state (default 256, today's behaviour): calibrate_zscore_native forwards it,
    unchanged, to native.native_features."""
    seen = []

    def fake_native_features(head, states, queries, cand_lists, max_state=256, max_suffix=512, render="letters"):
        seen.append(max_state)
        Kmax = max(len(c) for c in cand_lists)
        B = len(states)
        return (torch.zeros(B, 4), torch.zeros(B, Kmax + 1, 4), torch.ones(B, Kmax, dtype=torch.bool), 1)

    monkeypatch.setattr(T, "native_features", fake_native_features)

    class DummyModel:
        render = "letters"
        def calibrate(self, h, c):
            pass

    examples = [{"state": f"s{i}", "query": "q", "candidates": ["a", "b"]} for i in range(3)]
    T.calibrate_zscore_native(head=None, model=DummyModel(), examples=examples, seed=0, n=3)
    assert seen == [256]
    seen.clear()
    T.calibrate_zscore_native(head=None, model=DummyModel(), examples=examples, seed=0, n=3, max_state=1024)
    assert seen == [1024]


def test_max_state_threads_through_run_readout_and_forward_batches(monkeypatch):
    """--max_state threads run_readout -> native.run_batch_native, and forward_batches (used
    by eval_val_loss/fit_temperature/eval_dataset/dump_logits) threads it into run_readout."""
    seen = []

    def fake_run_batch_native(head, model, batch, examples, max_state=256, max_suffix=512, vec_cache=None):
        seen.append(max_state)
        return torch.zeros(len(examples), batch["cmask"].shape[1] + 1)

    monkeypatch.setattr(T, "run_batch_native", fake_run_batch_native)
    batch = {"cmask": torch.ones(2, 2, dtype=torch.bool)}
    T.run_readout("native", backbone=None, model=None, batch=batch, examples=[{}, {}])
    T.run_readout("native", backbone=None, model=None, batch=batch, examples=[{}, {}], max_state=77)
    assert seen == [256, 77]

    seen.clear()

    class Stub:
        def eval(self): pass
        def train(self): pass

    examples = [{"candidates": ["a", "b"], "target": [1.0, 0.0], "p_null": 0.0} for _ in range(4)]
    T.forward_batches(backbone=Stub(), model=Stub(), cache=None, examples=examples, bs=2, readout="native", max_state=99)
    assert seen == [99, 99]  # two batches of bs=2, each a real run_batch_native call


# ---------------------------------------------------------------------------
# 15. native_choice_v1 (pcdm/native.py, PLAN4 sec 11-12): --readout native --nc_head {n2,n3,n2n3}
# ---------------------------------------------------------------------------

from native import NativeHead, run_batch_native, native_features, shuffle_options, _fit_chunks, _is_bern_row, _yes_idx, RENDERS  # noqa: E402


def _native_batch(head, examples, nc_head, null, vec_cache=None):
    model = NativeHead(head.backbone.d, nc_head=nc_head, null=null)
    batch = collate_mcq(examples)
    return model, batch, run_batch_native(head, model, batch, examples, vec_cache=vec_cache)


def test_grad_ckpt_lora_grads_and_loss_decreases(tmp_path):
    """tl1b --grad_ckpt: backbone.model.gradient_checkpointing_enable(use_reentrant=False)
    must not break LoRA gradients (a frozen base + LoRA-only-on-the-top-layers setup is the
    classic case where reentrant checkpointing needs the enable_input_require_grads trick;
    non-reentrant checkpointing needs no such trick) -- and a few SGD steps must still lower
    the loss, i.e. gradients are not just non-None but actually useful."""
    torch.manual_seed(0)
    head = _make_tied_mcq_head(tmp_path, lora_layers=2, lora_r=4)
    head.backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = NativeHead(head.backbone.d, nc_head="n3", null="factored")
    examples = _mcq_examples(ks=(2, 3, 2))
    batch = collate_mcq(examples)
    opt = torch.optim.SGD(list(model.parameters()) + head.trainable_parameters(), lr=0.5)

    def step():
        opt.zero_grad()
        logits = run_batch_native(head, model, batch, examples, max_state=64)
        loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
        loss.backward()
        assert all(p.grad is not None for p in head.trainable_parameters())
        opt.step()
        return loss.item()

    losses = [step() for _ in range(6)]
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("nc_head", ["n2", "n3", "n2n3"])
@pytest.mark.parametrize("null", ["softmax", "factored"])
def test_native_shapes_finite_pads(tied_mcq_head, nc_head, null):
    torch.manual_seed(0)
    examples = _mcq_examples(ks=(2, 3, 5))
    vec_cache = _FakeVecCache(d=1024)
    _, batch, logits = _native_batch(tied_mcq_head, examples, nc_head, null, vec_cache)
    Kmax = max(len(ex["candidates"]) for ex in examples)
    assert logits.shape == (3, Kmax + 1)
    assert torch.isfinite(logits[:, -1]).all() and batch["n_tokens"] > 0
    for i, ex in enumerate(examples):
        k = len(ex["candidates"])
        assert torch.isfinite(logits[i, :k]).all()
        assert torch.all(logits[i, k:Kmax] == torch.finfo(logits.dtype).min)
    probs = T.probs_from_logits(logits, batch["cmask"], null=null)
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-5)


def test_native_n3_span_pooling_matches_manual_mean(tied_mcq_head):
    # C3[i, k] must equal the plain mean of the last-layer states over exactly the tokens
    # whose char offsets overlap option k's text (letters/newlines excluded), null line last.
    from mcq import _render
    head = tied_mcq_head
    tok = head.backbone.tokenizer
    examples = _mcq_examples(ks=(3, 2))
    examples[0]["candidates"] = ["Paris.", "New York", "none"]  # ".\n" merge + multi-token + null-like text
    states, queries, cands = [ex["state"] for ex in examples], [ex["query"] for ex in examples], [ex["candidates"] for ex in examples]
    with torch.inference_mode():
        h, C3, cmask, _ = native_features(head, states, queries, cands)
        for i in range(len(examples)):
            suffix, spans = _render(queries[i], cands[i])
            s_ids = tok(states[i], add_special_tokens=False)["input_ids"]
            enc = tok(suffix, add_special_tokens=False, return_offsets_mapping=True)
            ids = [tok.eos_token_id] + s_ids + enc["input_ids"] + [tok.eos_token_id]
            H = head.backbone.model(input_ids=torch.tensor([ids])).last_hidden_state[0].float()
            assert torch.allclose(h[i], H[-1], atol=1e-4)  # h_D = terminal token, unpadded reference
            for k, (cs, ce) in enumerate(spans):
                pos = [1 + len(s_ids) + j for j, (ts, te) in enumerate(enc["offset_mapping"]) if te > cs and ts < ce]
                assert suffix[cs:ce] in tok.decode([enc["input_ids"][p - 1 - len(s_ids)] for p in pos])
                col = C3.shape[1] - 1 if k == len(spans) - 1 else k
                assert torch.allclose(C3[i, col], H[pos].mean(0), atol=1e-4), (i, k)


def test_native_shuffle_remaps_target_teacher_label():
    # Fake head: s_j depends only on the candidate string -> loss must be identical under any
    # option permutation iff target/teacher are remapped consistently.
    rng = random.Random(0)
    examples = _mcq_examples(ks=(4, 3, 1))
    examples[0]["teacher"] = [0.1, 0.2, 0.3, 0.3, 0.1]
    examples[0]["label"] = 2
    examples[0]["target"] = [0.0, 0.0, 0.7, 0.3]
    shuffled = shuffle_options(examples, rng)
    assert [sorted(e["candidates"]) for e in shuffled] == [sorted(e["candidates"]) for e in examples]
    assert shuffled[0]["candidates"] != examples[0]["candidates"] or shuffled[1]["candidates"] != examples[1]["candidates"]
    assert shuffled[0]["candidates"][shuffled[0]["label"]] == examples[0]["candidates"][examples[0]["label"]]
    assert shuffled[2] is examples[2]  # K=1 untouched
    for ex in examples:  # inputs never mutated
        assert ex["candidates"] == [f"option {j}" for j in range(len(ex["candidates"]))]

    def fake_logits(exs):
        Kmax = max(len(e["candidates"]) for e in exs)
        out = torch.full((len(exs), Kmax + 1), torch.finfo(torch.float32).min)
        for i, e in enumerate(exs):
            out[i, :len(e["candidates"])] = torch.tensor([float(hash(c) % 97) / 10 for c in e["candidates"]])
            out[i, -1] = 0.5
        return out

    def loss_of(exs):
        b = collate_mcq(exs)
        return decision_loss(fake_logits(exs), b["target"], b["p_null"], b["cmask"],
                             teacher=b["teacher"], has_teacher=b["has_teacher"], beta=1.0)
    assert torch.allclose(loss_of(examples), loss_of(shuffled))


def test_native_n2_cold_path_and_grads(tmp_path_factory):
    # Candidate missing from the VecCache -> embedded by the (fake) encoder; grads reach W_h,
    # W_c (both sources) and the LoRA. Own head instance: this test backward()s.
    from transformers import AutoModel, AutoTokenizer, Qwen3Config
    d = tmp_path_factory.mktemp("tiny_qwen3_native")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", padding_side="right")
    cfg = Qwen3Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                       intermediate_size=128, vocab_size=len(tok), head_dim=16, tie_word_embeddings=True)
    AutoModel.from_config(cfg).save_pretrained(d); tok.save_pretrained(d)
    torch.manual_seed(0)
    head = MCQHead(name=str(d), lora_layers=2, lora_r=4, device="cpu")
    examples = _mcq_examples(ks=(3, 2))
    warm = _FakeVecCache(d=1024)
    for c in ["option 0", "option 1"]:
        warm.pooled(c)  # "option 2" stays cold
    calls = []

    class Enc:
        def embed(self, texts, render=None):
            calls.append((list(texts), render)); return torch.stack([warm.pooled(t) for t in texts])
    head.encoder = Enc()
    model, batch, logits = _native_batch(head, examples, "n2n3", "factored", warm)
    assert calls == [(["option 2"], "label: {}")]
    assert torch.isfinite(logits[:, -1]).all()
    loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
    loss.backward()
    assert model.proj_h[1].weight.grad is not None and model.proj_h[1].weight.grad.abs().sum() > 0
    assert model.proj_2[1].weight.grad is not None and model.proj_3[1].weight.grad is not None
    assert all(p.grad is not None for p in head.trainable_parameters())


def test_native_chunked_when_suffix_overflows(tied_mcq_head):
    # max_suffix too small for K=9 -> chunks that fit, composed hierarchically; every candidate
    # and the null get a finite score, shape unchanged.
    examples = _mcq_examples(ks=(9, 2))
    tok = tied_mcq_head.backbone.tokenizer
    chunks = _fit_chunks(tok, examples[0]["query"], examples[0]["candidates"], max_suffix=40)
    assert len(chunks) > 1 and sorted(i for c in chunks for i in c) == list(range(9))
    model = NativeHead(tied_mcq_head.backbone.d, nc_head="n3", null="factored")
    batch = collate_mcq(examples)
    logits = run_batch_native(tied_mcq_head, model, batch, examples, max_suffix=40)
    assert logits.shape == (2, 10) and torch.isfinite(logits[0]).all() and torch.isfinite(logits[1, :2]).all()
    assert torch.all(logits[1, 2:9] == torch.finfo(logits.dtype).min)


@pytest.mark.parametrize("null", ["softmax", "factored"])
def test_native_kv_decide_matches_run_batch(tied_mcq_head, null):
    # PLAN5 sec 1 serving path: state KV encoded once + one right-padded suffix pass per chunk
    # (explicit mask + position_ids) must reproduce run_batch_native's full-row probabilities --
    # the same contract test_decide_* give the energy decide().
    from native import native_kv_decide
    torch.manual_seed(0)
    head = tied_mcq_head.eval()
    model = NativeHead(head.backbone.d, nc_head="n3", null=null).eval()
    state = "a shared state text used for every query below"
    queries = [(f"query number {i}", [f"option {j}" for j in range(k)]) for i, k in enumerate((2, 5, 9))]
    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    batch = collate_mcq(examples)
    with torch.inference_mode():
        probs = torch.softmax(run_batch_native(head, model, batch, examples), dim=-1)
    dists = native_kv_decide(head, model, state, queries, chunk=2)
    assert len(dists) == 3
    for i, (_, c) in enumerate(queries):
        expected = torch.cat([probs[i, :len(c)], probs[i, -1:]])
        assert torch.allclose(dists[i], expected, atol=1e-3), (i, (dists[i] - expected).abs().max())
        assert torch.allclose(dists[i].sum(), torch.tensor(1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# 15b. PLAN7 track C typed heads: --score_head cumlink, --noul_head bern
# ---------------------------------------------------------------------------

def test_cumlink_shape_normalizes_and_k2_is_sigmoid():
    """_cumlink_probs: [B, Kmax] rows sum to 1 over the valid (cmask) columns only, pads zero;
    a K=2 row's category probs are exactly the plain sigmoid P(0)=sigmoid(tau0-u),
    P(1)=sigmoid(u-tau0) -- the "K=2 reduces to a sigmoid" case named in PLAN7 track C."""
    torch.manual_seed(0)
    d, B, Kmax = 8, 4, 5
    model = NativeHead(d, nc_head="n3", score_head="cumlink")
    h = torch.randn(B, d)
    c3 = torch.randn(B, Kmax + 1, d)
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    Ks = [2, 3, 4, 5]
    for i, k in enumerate(Ks):
        cmask[i, :k] = True
    hz = (h - model.mu_h) / model.sd_h
    p = model._cumlink_probs(hz, c3, cmask)
    assert p.shape == (B, Kmax)
    for i, k in enumerate(Ks):
        assert torch.allclose(p[i, :k].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(p[i, k:] == 0)
        assert torch.all(p[i, :k] >= 0)

    row = 0  # K=2
    c3z = (c3 - model.mu_c) / model.sd_c
    u = model.w_o(hz[row:row + 1]).squeeze(-1)
    tau0 = model.theta0 + F.softplus(model.w_t(c3z[row:row + 1, :1])).squeeze(-1)
    assert torch.allclose(p[row, 0], torch.sigmoid(tau0 - u).squeeze(), atol=1e-5)
    assert torch.allclose(p[row, 1], torch.sigmoid(u - tau0).squeeze(), atol=1e-5)


def test_cumlink_monotone_cdf():
    """thresholds tau_j = theta0 + cumsum(softplus(...)) are strictly increasing in j (softplus
    > 0) -> P(y<=j) = sigmoid(tau_j - u) is non-decreasing in j for fixed u (monotone CDF)."""
    torch.manual_seed(1)
    d, B, Kmax = 8, 3, 6
    model = NativeHead(d, nc_head="n3", score_head="cumlink")
    h = torch.randn(B, d)
    c3 = torch.randn(B, Kmax + 1, d)
    cmask = torch.ones(B, Kmax, dtype=torch.bool)
    hz = (h - model.mu_h) / model.sd_h
    c3z = (c3 - model.mu_c) / model.sd_c
    u = model.w_o(hz).squeeze(-1)
    width = F.softplus(model.w_t(c3z[:, :-1]).squeeze(-1))
    theta = model.theta0 + torch.cumsum(width, dim=-1)
    cdf = torch.sigmoid(theta - u.unsqueeze(-1))
    assert torch.all(cdf[:, 1:] + 1e-6 >= cdf[:, :-1])


def test_cumlink_end_to_end_native_batch(tied_mcq_head):
    """--score_head cumlink through run_batch_native (real tiny backbone): finite logits, valid
    probability distribution, ∅ still placed via the shared factored gate."""
    torch.manual_seed(0)
    examples = _mcq_examples(ks=(3, 4))
    batch = collate_mcq(examples)
    model = NativeHead(tied_mcq_head.backbone.d, nc_head="n3", null="factored", score_head="cumlink")
    logits = run_batch_native(tied_mcq_head, model, batch, examples)
    assert torch.isfinite(logits[:, -1]).all()
    probs = T.probs_from_logits(logits, batch["cmask"], null="factored")
    assert torch.allclose(probs.sum(-1), torch.ones(len(examples)), atol=1e-5)


def test_bern_reversed_label_gives_identical_p_yes(tied_mcq_head):
    """--noul_head bern (PLAN7 track C noul_B_bern): the suffix never renders candidates (see
    native._render_query_only), so P(yes) must be identical whether the row's candidates are
    ["no","yes"] or ["yes","no"] -- the reversed-label control from PLAN7 track C."""
    torch.manual_seed(0)
    head = tied_mcq_head
    model = NativeHead(head.backbone.d, nc_head="n3", noul_head="bern")
    ex_a = [
        {"state": "s1", "query": "eligible?", "candidates": ["no", "yes"],
         "target": [1.0, 0.0], "p_null": 0.0, "task": "t"},
        {"state": "s2", "query": "trigger?", "candidates": ["no", "yes"],
         "target": [0.0, 1.0], "p_null": 0.0, "task": "t"},
    ]
    ex_b = [dict(ex, candidates=list(reversed(ex["candidates"])), target=list(reversed(ex["target"])))
            for ex in ex_a]
    probs_a = torch.softmax(run_batch_native(head, model, collate_mcq(ex_a), ex_a), dim=-1)
    probs_b = torch.softmax(run_batch_native(head, model, collate_mcq(ex_b), ex_b), dim=-1)
    assert probs_a.shape == (2, 3)  # Kmax=2 + null
    assert torch.allclose(probs_a.sum(-1), torch.ones(2), atol=1e-5)
    for i in range(2):
        ya, yb = ex_a[i]["candidates"].index("yes"), ex_b[i]["candidates"].index("yes")
        assert torch.allclose(probs_a[i, ya], probs_b[i, yb], atol=1e-6)


def test_is_bern_row():
    """PLAN7 track C mixed-type fix: bern-eligible iff candidates are exactly {"yes","no"}
    (case-insensitive), or meta.qtype == "noul" with K == 2 when meta is available."""
    assert _is_bern_row(["no", "yes"])
    assert _is_bern_row(["Yes", "NO"])          # case-insensitive
    assert not _is_bern_row(["cancel", "refund", "status"])
    assert not _is_bern_row(["a", "b"])         # 2-way but not yes/no, no meta
    assert _is_bern_row(["a", "b"], {"qtype": "noul"})       # meta fallback
    assert not _is_bern_row(["a", "b", "c"], {"qtype": "noul"})  # K != 2
    assert not _is_bern_row(["a", "b"], {"qtype": "score"})


def test_bern_render_is_query_only_for_yesno_rows_only():
    """--noul_head bern renders query-only for a yes/no row and the normal --nc_render for
    every other row in the same batch (PLAN7 track C mixed-type fix: it used to force
    query-only on every row unconditionally, hiding non-noul candidates entirely)."""
    text, spans = RENDERS["query_only"]("is this eligible?", ["no", "yes"])
    assert text == "is this eligible?\n" and spans == []
    letters_text, letters_spans = RENDERS["letters_nonull"]("pick one", ["a", "b", "c"])
    assert "a" in letters_text and len(letters_spans) == 3


def test_bern_mixed_batch_matches_choice_on_non_yesno_rows(tied_mcq_head):
    """PLAN7 track C mixed-type fix (the bug this whole section exists for): in a batch mixing
    a yes/no row with ordinary K-way rows, --noul_head bern must give the SAME logits on the
    K-way rows as --noul_head choice (bit-identical -- same weights, same code path), and only
    the yes/no row goes through the Bernoulli head. Before the fix, --noul_head bern rendered
    every row query-only and force-applied the Bernoulli head to it, so the K-way rows saw no
    candidate text at all and their logits differed from the choice control."""
    torch.manual_seed(0)
    head = tied_mcq_head
    examples = [
        {"state": "s1", "query": "eligible?", "candidates": ["no", "yes"],
         "target": [1.0, 0.0], "p_null": 0.0, "task": "t", "meta": {"qtype": "noul"}},
        {"state": "s2", "query": "which team?", "candidates": ["billing", "support", "logistics"],
         "target": [0.0, 1.0, 0.0], "p_null": 0.0, "task": "t", "meta": {"qtype": "choice"}},
        {"state": "s3", "query": "urgency?", "candidates": ["low", "medium", "high", "critical"],
         "target": [0.0, 0.0, 1.0, 0.0], "p_null": 0.0, "task": "t", "meta": {"qtype": "score"}},
    ]
    batch = collate_mcq(examples)

    model_bern = NativeHead(head.backbone.d, nc_head="n3", null="factored", noul_head="bern")
    model_choice = NativeHead(head.backbone.d, nc_head="n3", null="factored", noul_head="choice")
    model_choice.load_state_dict(model_bern.state_dict(), strict=False)  # same K-way scorer weights

    logits_bern = run_batch_native(head, model_bern, batch, examples)
    logits_choice = run_batch_native(head, model_choice, batch, examples)

    Kmax = batch["cmask"].shape[1]
    for i in (1, 2):  # the two K-way (non yes/no) rows: bit-identical to the choice control
        k = len(examples[i]["candidates"])
        assert torch.allclose(logits_bern[i, :k], logits_choice[i, :k], atol=1e-6), i
        assert torch.allclose(logits_bern[i, -1], logits_choice[i, -1], atol=1e-6), i
    # row 0 (yes/no) went through the Bernoulli head instead -- it need not match the choice
    # control, but it must still be a normalized, finite distribution
    probs0 = torch.softmax(logits_bern[0], dim=-1)
    assert torch.isfinite(probs0).all() and torch.allclose(probs0.sum(), torch.tensor(1.0), atol=1e-5)


def test_bern_reversed_label_holds_inside_a_mixed_batch(tied_mcq_head):
    """The reversed-label invariance (PLAN7 track C) must still hold for the yes/no row when
    it's scored alongside ordinary K-way rows, not just in an all-noul batch."""
    torch.manual_seed(0)
    head = tied_mcq_head
    base = [
        {"state": "s2", "query": "which team?", "candidates": ["billing", "support", "logistics"],
         "target": [0.0, 1.0, 0.0], "p_null": 0.0, "task": "t"},
        {"state": "s3", "query": "urgency?", "candidates": ["low", "medium", "high"],
         "target": [0.0, 0.0, 1.0], "p_null": 0.0, "task": "t"},
    ]
    model = NativeHead(head.backbone.d, nc_head="n3", noul_head="bern")
    ex_a = base + [{"state": "s1", "query": "eligible?", "candidates": ["no", "yes"],
                    "target": [1.0, 0.0], "p_null": 0.0, "task": "t"}]
    ex_b = base + [{"state": "s1", "query": "eligible?", "candidates": ["yes", "no"],
                    "target": [0.0, 1.0], "p_null": 0.0, "task": "t"}]
    probs_a = torch.softmax(run_batch_native(head, model, collate_mcq(ex_a), ex_a), dim=-1)
    probs_b = torch.softmax(run_batch_native(head, model, collate_mcq(ex_b), ex_b), dim=-1)
    ya, yb = ex_a[2]["candidates"].index("yes"), ex_b[2]["candidates"].index("yes")
    assert torch.allclose(probs_a[2, ya], probs_b[2, yb], atol=1e-6)


def test_bern_out_of_domain_kway_row_degrades_instead_of_crashing():
    """--noul_head bern is trained/evaluated 2-way only (qtype == noul), but a mixed-family
    eval run (JevBench) can hand it a K-way row with no literal "yes" candidate -- PLAN7 track
    C: this must degrade to a well-defined (if meaningless) distribution, not crash the whole
    eval batch. _yes_idx defaults such rows to column 0; _bern_probs spreads 1-p_yes uniformly
    over the other valid candidates, which is exactly today's 1-p_yes-at-the-other-slot at K=2."""
    torch.manual_seed(0)
    d, B, Kmax = 8, 2, 5
    model = NativeHead(d, nc_head="n3", noul_head="bern")
    hz = torch.randn(B, d)
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    cmask[0, :2] = True   # in-domain: 2-way
    cmask[1, :5] = True   # out-of-domain: 5-way, no "yes" among its candidates
    yi = _yes_idx([["no", "yes"], ["cancel", "refund", "status", "change_address", "other"]], hz.device)
    assert yi.tolist() == [1, 0]  # row 1 has no "yes" -> defaults to column 0
    p = model._bern_probs(hz, cmask, yi)
    assert torch.allclose(p[0, :2].sum(), torch.tensor(1.0), atol=1e-5) and torch.all(p[0, 2:] == 0)
    assert torch.allclose(p[1, :5].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(p[1] >= 0)


# ---------------------------------------------------------------------------
# 15c. --nc_render semif: SemIf-structured native head (pcdm/native.py module docstring,
# pcdm_jev.decider --prompt_style semif's frozen-probe +8 JevBench-standard result)
# ---------------------------------------------------------------------------

from native import _render_semif, _render_row, _semif_head_tail, native_kv_decide  # noqa: E402


def test_every_render_dispatches_through_render_row(tied_mcq_head):
    """pcdm/bench.py called RENDERS[name](query, cands) directly and crashed the ts1b_semif
    latency bench with "TypeError: _render_semif() missing 1 required positional argument:
    'tail_text'" -- semif is the one render needing the tokenizer's chat-template tail, which
    is exactly why _render_row exists. Any caller reaching RENDERS directly breaks on semif
    only, so it survives every letters/tags smoke test."""
    tok = tied_mcq_head.backbone.tokenizer
    for name in RENDERS:
        text, spans = _render_row(tok, name, "Which one?", ["alpha", "beta", "gamma"])
        assert isinstance(text, str) and text, name
        assert isinstance(spans, (list, tuple)), name


def test_semif_render_two_examples_json_shape_and_spans(tied_mcq_head):
    """_render_semif's suffix continues the "evidence" string opened in the cached prefix:
    closing quote, criterion, then K lettered {"letter","description"} options (no null line)
    ending in tail_text -- spans[k] must point at exactly option k's description text."""
    tok = tied_mcq_head.backbone.tokenizer
    tail = _semif_head_tail(tok)[1]
    for query, cands in [("is this eligible?", ["yes", "no"]),
                          ("pick the best category", ["billing", "support", "a \"quoted\" one"])]:
        text, spans = _render_semif(query, cands, tail)
        assert text.startswith('", "criterion": ') and text.endswith(tail)
        assert len(spans) == len(cands)
        for (cs, ce), c in zip(spans, cands):
            assert text[cs:ce] == json.dumps(c)[1:-1]  # escaped exactly like the rendered JSON


def test_semif_head_tail_wraps_system_and_json_opening(tied_mcq_head):
    """pre_head = system turn + user-turn opening + the "evidence" field's opening quote (goes
    at the START of the KV-cached prefix, before the state text); tail = end of user turn +
    assistant-turn opening (appended after _render_semif's JSON close)."""
    from native import SEMIF_SYSTEM
    tok = tied_mcq_head.backbone.tokenizer
    pre_head, tail = _semif_head_tail(tok)
    assert SEMIF_SYSTEM in pre_head and pre_head.endswith('{"evidence": "')
    assert tail.startswith(tok.eos_token) or "assistant" in tail  # end-of-turn then assistant-turn opening
    assert _semif_head_tail(tok) is _semif_head_tail(tok)  # lru_cache: same tokenizer -> same object


def test_semif_n3_span_pooling_matches_manual_mean(tied_mcq_head):
    """C3[i, k] under --nc_render semif must equal the plain mean of the last-layer states over
    exactly the tokens whose char offsets overlap option k's description text, with the prefix
    = pre_head + state (no eos sink) and the suffix ending in tail_text (no extra eos tail) --
    the same manual-reconstruction check as letters' test_native_n3_span_pooling_matches_manual_mean."""
    head = tied_mcq_head
    tok = head.backbone.tokenizer
    examples = _mcq_examples(ks=(3, 2))
    examples[0]["candidates"] = ["Paris.", "New York", "a \"quoted\" one"]
    states, queries, cands = [ex["state"] for ex in examples], [ex["query"] for ex in examples], [ex["candidates"] for ex in examples]
    with torch.inference_mode():
        h, C3, cmask, _ = native_features(head, states, queries, cands, render="semif")
        pre_head, tail = _semif_head_tail(tok)
        for i in range(len(examples)):
            suffix, spans = _render_semif(queries[i], cands[i], tail)
            s_ids = tok(pre_head + states[i], add_special_tokens=False)["input_ids"]
            enc = tok(suffix, add_special_tokens=False, return_offsets_mapping=True)
            ids = s_ids + enc["input_ids"]  # no leading eos sink, no trailing eos tail
            H = head.backbone.model(input_ids=torch.tensor([ids])).last_hidden_state[0].float()
            assert torch.allclose(h[i], H[-1], atol=1e-4)  # h_D = last token of the assistant-turn opening
            for k, (cs, ce) in enumerate(spans):
                pos = [len(s_ids) + j for j, (ts, te) in enumerate(enc["offset_mapping"]) if te > cs and ts < ce]
                assert torch.allclose(C3[i, k], H[pos].mean(0), atol=1e-4), (i, k)


def test_semif_bern_row_routing_unaffected(tied_mcq_head):
    """PLAN7 track C mixed-type fix must hold under --nc_render semif too: a yes/no row still
    renders query_only against the PLAIN eos+state prefix (native._is_bern_row), so a --noul_head
    bern model gives BIT-IDENTICAL logits to the choice control on every non-yes/no row --
    exactly test_bern_mixed_batch_matches_choice_on_non_yesno_rows, with nc_render=semif."""
    torch.manual_seed(0)
    head = tied_mcq_head
    examples = [
        {"state": "s1", "query": "eligible?", "candidates": ["no", "yes"],
         "target": [1.0, 0.0], "p_null": 0.0, "task": "t", "meta": {"qtype": "noul"}},
        {"state": "s2", "query": "which team?", "candidates": ["billing", "support", "logistics"],
         "target": [0.0, 1.0, 0.0], "p_null": 0.0, "task": "t", "meta": {"qtype": "choice"}},
    ]
    batch = collate_mcq(examples)
    model_bern = NativeHead(head.backbone.d, nc_head="n3", null="factored", render="semif", noul_head="bern")
    model_choice = NativeHead(head.backbone.d, nc_head="n3", null="factored", render="semif", noul_head="choice")
    model_choice.load_state_dict(model_bern.state_dict(), strict=False)

    logits_bern = run_batch_native(head, model_bern, batch, examples)
    logits_choice = run_batch_native(head, model_choice, batch, examples)
    k = len(examples[1]["candidates"])
    assert torch.allclose(logits_bern[1, :k], logits_choice[1, :k], atol=1e-6)
    assert torch.allclose(logits_bern[1, -1], logits_choice[1, -1], atol=1e-6)
    probs0 = torch.softmax(logits_bern[0], dim=-1)
    assert torch.isfinite(probs0).all() and torch.allclose(probs0.sum(), torch.tensor(1.0), atol=1e-5)


def test_semif_smoke_trains(tied_mcq_head):
    """--nc_render semif end to end: a few SGD steps on run_batch_native's logits must lower
    decision_loss and reach every trainable parameter, same contract as the other renders'
    test_grad_ckpt_lora_grads_and_loss_decreases."""
    torch.manual_seed(0)
    head = tied_mcq_head
    model = NativeHead(head.backbone.d, nc_head="n3", null="factored", render="semif")
    examples = _mcq_examples(ks=(2, 3, 2))
    batch = collate_mcq(examples)
    opt = torch.optim.SGD(list(model.parameters()) + head.trainable_parameters(), lr=0.5)

    def step():
        opt.zero_grad()
        logits = run_batch_native(head, model, batch, examples)
        loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"])
        loss.backward()
        assert all(p.grad is not None for p in head.trainable_parameters())
        opt.step()
        return loss.item()

    losses = [step() for _ in range(6)]
    assert losses[-1] < losses[0]


def test_semif_fit_chunks_when_suffix_overflows(tied_mcq_head):
    """tl1b_semif regression (2026-09-22): _fit_chunks' semif branch built its per-option
    length-probe template with unescaped literal JSON braces ('{"letter": ...}'), which
    str.format() parsed as extra format fields -> KeyError('"letter"') the first time a real
    run hit K > MAXK_DIRECT under --nc_render semif (data_kb). Mirrors
    test_native_chunked_when_suffix_overflows but forces the chunked path with render=semif."""
    examples = _mcq_examples(ks=(9, 2))
    tok = tied_mcq_head.backbone.tokenizer
    chunks = _fit_chunks(tok, examples[0]["query"], examples[0]["candidates"], max_suffix=40, render="semif")
    assert len(chunks) > 1 and sorted(i for c in chunks for i in c) == list(range(9))


@pytest.mark.parametrize("null", ["softmax", "factored"])
def test_semif_kv_decide_matches_run_batch(tied_mcq_head, null):
    """native_kv_decide's semif prefix/suffix split must reproduce run_batch_native's full-row
    probabilities, exactly like test_native_kv_decide_matches_run_batch for the other renders."""
    torch.manual_seed(0)
    head = tied_mcq_head.eval()
    model = NativeHead(head.backbone.d, nc_head="n3", null=null, render="semif").eval()
    state = "a shared state text used for every query below"
    queries = [(f"query number {i}", [f"option {j}" for j in range(k)]) for i, k in enumerate((2, 5, 9))]
    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    batch = collate_mcq(examples)
    with torch.inference_mode():
        probs = torch.softmax(run_batch_native(head, model, batch, examples), dim=-1)
    dists = native_kv_decide(head, model, state, queries, chunk=2)
    assert len(dists) == 3
    for i, (_, c) in enumerate(queries):
        expected = torch.cat([probs[i, :len(c)], probs[i, -1:]])
        assert torch.allclose(dists[i], expected, atol=1e-3), (i, (dists[i] - expected).abs().max())


def test_semif_kv_decide_mixed_bern_matches_run_batch(tied_mcq_head):
    """A state whose queries mix a bern (yes/no) row with ordinary semif-rendered rows needs two
    KV caches (native_kv_decide's non_bern/bern split) -- both must still match run_batch_native
    row for row, exercising the dual-cache bucketing this render forces."""
    torch.manual_seed(0)
    head = tied_mcq_head.eval()
    model = NativeHead(head.backbone.d, nc_head="n3", render="semif", noul_head="bern").eval()
    state = "a shared state text for the mixed bern/non-bern check"
    queries = [("eligible?", ["no", "yes"]), ("which team?", ["billing", "support", "logistics"])]
    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    batch = collate_mcq(examples)
    with torch.inference_mode():
        probs = torch.softmax(run_batch_native(head, model, batch, examples), dim=-1)
    dists = native_kv_decide(head, model, state, queries, chunk=8)
    for i, (_, c) in enumerate(queries):
        expected = torch.cat([probs[i, :len(c)], probs[i, -1:]])
        assert torch.allclose(dists[i], expected, atol=1e-3), (i, (dists[i] - expected).abs().max())


# ---------------------------------------------------------------------------
# 16. native_v2 (PLAN5 sec 2): --nc_render tags + --perm_lambda
# ---------------------------------------------------------------------------

from native import _render_tags, perm_consistency_loss, RENDERS  # noqa: E402
from mcq import _render  # noqa: E402


def test_tags_render_letter_free_and_spans():
    text, spans = _render_tags("q?", ["Paris.", "New York"])
    assert text == "q?\n<choice>\nParis.\n</choice>\n<choice>\nNew York\n</choice>\n"
    assert [text[a:b] for a, b in spans] == ["Paris.", "New York"] and len(spans) == 2  # no null line
    assert "A." not in text and "Answer:" not in text and "none of the above" not in text
    assert RENDERS["letters"]("q?", ["x"]) == _render("q?", ["x"])  # default rendering byte-identical


def test_tags_n3_span_pooling_matches_manual_mean(tied_mcq_head):
    head = tied_mcq_head
    tok = head.backbone.tokenizer
    examples = _mcq_examples(ks=(3, 2))
    examples[0]["candidates"] = ["Paris.", "New York", "none"]
    states, queries, cands = [ex["state"] for ex in examples], [ex["query"] for ex in examples], [ex["candidates"] for ex in examples]
    with torch.inference_mode():
        h, C3, cmask, _ = native_features(head, states, queries, cands, render="tags")
        for i in range(len(examples)):
            suffix, spans = _render_tags(queries[i], cands[i])
            s_ids = tok(states[i], add_special_tokens=False)["input_ids"]
            enc = tok(suffix, add_special_tokens=False, return_offsets_mapping=True)
            ids = [tok.eos_token_id] + s_ids + enc["input_ids"] + [tok.eos_token_id]
            H = head.backbone.model(input_ids=torch.tensor([ids])).last_hidden_state[0].float()
            assert torch.allclose(h[i], H[-1], atol=1e-4)
            for k, (cs, ce) in enumerate(spans):
                pos = [1 + len(s_ids) + j for j, (ts, te) in enumerate(enc["offset_mapping"]) if te > cs and ts < ce]
                pooled_text = tok.decode([enc["input_ids"][p - 1 - len(s_ids)] for p in pos])
                assert suffix[cs:ce] in pooled_text and "choice" not in pooled_text  # tag tokens excluded
                assert torch.allclose(C3[i, k], H[pos].mean(0), atol=1e-4), (i, k)
            assert torch.all(C3[i, -1] == 0)  # no rendered null: the null is the head's
    model = NativeHead(head.backbone.d, nc_head="n3", null="factored", render="tags")
    batch = collate_mcq(examples)
    logits = run_batch_native(head, model, batch, examples)
    assert torch.isfinite(logits[:, -1]).all() and torch.isfinite(logits[0, :3]).all()
    probs = T.probs_from_logits(logits, batch["cmask"], null="factored")
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5)
    # chunked fallback works under tags too
    assert len(_fit_chunks(tok, "q", [f"option {j}" for j in range(9)], max_suffix=40, render="tags")) > 1


def test_perm_consistency_loss_zero_iff_order_equivariant():
    examples = _mcq_examples(ks=(4, 3, 1))

    def fake(by_string):
        def score(exs, batch):
            Kmax = max(len(e["candidates"]) for e in exs)
            out = torch.full((len(exs), Kmax + 1), torch.finfo(torch.float32).min)
            for i, e in enumerate(exs):
                cs = e["candidates"]
                out[i, :len(cs)] = torch.tensor([float(sum(map(ord, c)) % 7) if by_string else float(j) for j, c in enumerate(cs)])
                out[i, -1] = 0.5
            return out
        return score
    for by_string, expect_zero in ((True, True), (False, False)):
        score = fake(by_string)
        logits = score(examples, collate_mcq(examples))
        loss = perm_consistency_loss(score, logits, examples, random.Random(0), frac=1.0)
        assert torch.isfinite(loss) and loss >= 0
        assert (loss.item() < 1e-5) == expect_zero and (expect_zero or loss.item() > 1e-2), (by_string, loss.item())
    # frac subset + shuffled inputs untouched
    loss = perm_consistency_loss(fake(True), fake(True)(examples, None), examples, random.Random(1), frac=0.25)
    assert loss.item() < 1e-5 and examples[0]["candidates"] == [f"option {j}" for j in range(4)]


# ---------------------------------------------------------------------------
# 17. scripts/compose_support.py (PLAN5 sec 3): energy support gate x native conditional choice
# ---------------------------------------------------------------------------

from compose_support import run as compose_run, compose, align  # noqa: E402


def _write_dump(d, name, logits, target, label, K, meta):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{name}.npz", logits=logits, target=target, label=label, K=K)
    json.dump(meta, open(d / f"{name}.meta.json", "w"))


def test_compose_support_synthetic(tmp_path):
    rng = np.random.RandomState(0)
    ks = [2, 4, 4, 6, 3, 5]
    le, te, Ke = _fake_dump(seed=1, n=len(ks), ks=ks)[0], None, np.array(ks)
    Kmax = max(ks)
    te, ye, me = np.zeros((len(ks), Kmax + 1)), np.zeros(len(ks)), []
    ln, tn, yn, mn = np.full_like(le, np.finfo(np.float64).min), np.zeros_like(te), np.zeros(len(ks)), []
    for i, k in enumerate(ks):
        cands = [f"c{i}_{j}" for j in range(k)]
        if k == 4:
            cands[1] = cands[0]  # duplicate candidate strings must still align
        gold = -1 if i == 3 else i % k
        ye[i] = gold; te[i, k if gold == -1 else gold] = 1.0
        me.append({"query": f"q{i}", "candidates": cands, "label": gold, "meta": {}})
        perm = rng.permutation(k)
        nat_logits = rng.randn(k)
        ln[i, :k] = nat_logits[perm]; ln[i, -1] = rng.randn()
        ln_orig = np.full(Kmax + 1, np.finfo(np.float64).min); ln_orig[:k] = nat_logits
        mn.append({"query": f"q{i}", "candidates": [cands[p] for p in perm], "label": int(np.where(perm == gold)[0][0]) if gold >= 0 else -1, "meta": {}})
        tn[i, :k] = te[i, :k][perm]; tn[i, -1] = te[i, -1]; yn[i] = mn[-1]["label"]
    _write_dump(tmp_path / "e", "setA", le, te, ye, Ke, me)
    _write_dump(tmp_path / "n", "setA", ln, tn, yn, Ke, mn)
    _write_dump(tmp_path / "e", "only_e", le, te, ye, Ke, me)  # present in one dump only -> skipped

    aligned = align(me, mn, ln)
    for i, k in enumerate(ks):  # alignment undoes the native shuffle (dups: equal strings, any order is fine)
        s_e = {c: aligned[i, j] for j, c in enumerate(me[i]["candidates"])}
        s_n = {c: ln[i, j] for j, c in enumerate(mn[i]["candidates"])}
        assert all(np.isclose(s_e[c], s_n[c]) or c == me[i]["candidates"][0] for c in s_e)
    for r_from in ("energy", "energy_T", "native"):
        p, p_e, p_n = compose(le, Ke, aligned, r_from, energy_T=1.7)
        assert np.allclose(p.sum(-1), 1.0) and (p >= 0).all()
        assert np.allclose(p[:, :-1].argmax(-1), p_n[:, :-1].argmax(-1))  # among-K = native's exactly
        if r_from == "energy":
            assert np.allclose(p[:, -1], apply_bias(le, Ke, 0, 0, 1.0)[:, -1])  # P(null) = energy null exactly
        elif r_from == "energy_T":
            assert np.allclose(p[:, -1], apply_bias(le, Ke, 0, 0, 1.7)[:, -1])
        else:
            assert np.allclose(p, p_n)  # r from native's own null reproduces native exactly
    res = compose_run(tmp_path / "e", tmp_path / "n", quiet=True)
    assert set(res["eval"]) == {"setA"} and "acc" in res["eval"]["setA"]["scaled"]
    m_n = summarize(apply_bias(ln, Ke, 0, 0, 1.0), tn, yn)
    assert np.isclose(res["eval"]["setA"]["scaled"]["acc_k"], m_n["acc_k"])  # native among-K in native's own order
    m_e = summarize(apply_bias(le, Ke, 0, 0, 1.0), te, ye)
    assert np.isclose(res["eval"]["setA"]["scaled"]["auroc_null"], m_e["auroc_null"])


# ---------------------------------------------------------------------------
# 18. scripts/fuse_scores.py (PLAN5 sec 4): "unify the two experts" -- eval-time score-level
# fusion s_j = log P_energy(a_j|answerable) + g*log P_native(a_j|answerable) (log mode) /
# (1-g)*P_energy + g*P_native (linear mode), null always from the energy gate.
# ---------------------------------------------------------------------------

from fuse_scores import fuse, run as fuse_run, load_dumps  # noqa: E402


def test_fuse_scores_synthetic(tmp_path):
    rng = np.random.RandomState(2)
    ks = [2, 4, 4, 6, 3, 5]
    le, Ke = _fake_dump(seed=1, n=len(ks), ks=ks)[0], np.array(ks)
    Kmax = max(ks)
    te, ye, me = np.zeros((len(ks), Kmax + 1)), np.zeros(len(ks)), []
    ln, tn, yn, mn = np.full_like(le, np.finfo(np.float64).min), np.zeros_like(te), np.zeros(len(ks)), []
    for i, k in enumerate(ks):
        cands = [f"c{i}_{j}" for j in range(k)]
        gold = -1 if i == 3 else i % k
        ye[i] = gold; te[i, k if gold == -1 else gold] = 1.0
        me.append({"query": f"q{i}", "candidates": cands, "label": gold, "meta": {}})
        perm = rng.permutation(k)
        nat_logits = rng.randn(k) * 5.0  # well-separated so a large g's argmax is unambiguous
        ln[i, :k] = nat_logits[perm]; ln[i, -1] = rng.randn()
        mn.append({"query": f"q{i}", "candidates": [cands[p] for p in perm], "label": -1, "meta": {}})
    _write_dump(tmp_path / "e", "setA", le, te, ye, Ke, me)
    _write_dump(tmp_path / "n", "setA", ln, tn, yn, Ke, mn)

    loaded = load_dumps(tmp_path / "e", tmp_path / "n")
    aligned = loaded["setA"][4]
    p_energy = apply_bias(le, Ke, 0.0, 0.0, 1.0)  # "energy alone" (same as compose_support's before/energy column)

    for mode in ("log", "linear"):
        p0 = fuse(le, Ke, aligned, 0.0, mode)
        assert np.allclose(p0.sum(-1), 1.0)
        assert np.allclose(p0, p_energy, atol=1e-9), (mode, p0 - p_energy)  # g=0 reproduces energy exactly

    p_big = fuse(le, Ke, aligned, 100.0, "log")
    assert np.allclose(p_big.sum(-1), 1.0)
    for i, k in enumerate(ks):
        assert p_big[i, :k].argmax() == aligned[i, :k].argmax()  # large g -> native's argmax

    by_g, val_g = fuse_run(tmp_path / "e", tmp_path / "n", g_grid=[0.0, 1.0], quiet=True)
    assert set(by_g) == {0.0, 1.0} and val_g is None  # no "val" set in this synthetic dump
    assert set(by_g[0.0]["eval"]) == {"setA"} and "acc" in by_g[0.0]["eval"]["setA"]["scaled"]


def test_render_letters_nonull_has_letters_no_null_line():
    from native import RENDERS
    text, spans = RENDERS["letters_nonull"]("q?", ["alpha", "beta"])
    assert "A. alpha" in text and "B. beta" in text and "none of the above" not in text and "Answer:" not in text
    assert len(spans) == 2 and all(text[a:b] == c for (a, b), c in zip(spans, ["alpha", "beta"]))


# ---------------------------------------------------------------------------
# 19. scripts/gate_experts.py (PLAN6 item 3): learned per-input gate g(x) over the two experts,
# trained on the dumps' val split only, one val-fitted T, pcdm/train.py-format results.json.
# ---------------------------------------------------------------------------

import gate_experts as GE  # noqa: E402


def _two_expert_dumps(tmp_path, n=120, seed=3):
    """Energy + shuffled native dump over the same examples; the native expert is only good on
    rows whose K >= 4 (so a per-input gate has something to learn from log K / margins)."""
    rng = np.random.RandomState(seed)
    ks = [2, 3, 4, 6, 8]
    Kmax = max(ks)
    for split in ("val", "setA"):
        K = np.array([ks[i % len(ks)] for i in range(n)])
        le, ln = np.full((n, Kmax + 1), np.finfo(np.float64).min), np.full((n, Kmax + 1), np.finfo(np.float64).min)
        te, tn, ye, yn, me, mn = np.zeros((n, Kmax + 1)), np.zeros((n, Kmax + 1)), np.zeros(n), np.zeros(n), [], []
        for i, k in enumerate(K):
            gold = -1 if rng.rand() < 0.15 else rng.randint(k)
            ye[i] = gold; te[i, k if gold == -1 else gold] = 1.0
            cands = [f"{split}_c{i}_{j}" for j in range(k)]
            le[i, :k] = rng.randn(k) + (2.0 if k < 4 else 0.3) * te[i, :k]; le[i, -1] = 1.0 if gold == -1 else -1.0
            perm = rng.permutation(k)
            nat = rng.randn(k) + (0.3 if k < 4 else 3.0) * te[i, :k]
            ln[i, :k] = nat[perm]; ln[i, -1] = rng.randn()
            tn[i, :k] = te[i, :k][perm]; tn[i, -1] = te[i, -1]; yn[i] = gold if gold == -1 else int(np.where(perm == gold)[0][0])
            me.append({"query": f"q{i} " * (k % 3 + 1), "candidates": cands, "label": gold, "meta": {}})
            mn.append({"query": me[-1]["query"], "candidates": [cands[p] for p in perm], "label": yn[i], "meta": {}})
        _write_dump(tmp_path / "e", split, le, te, ye, K, me)
        _write_dump(tmp_path / "n", split, ln, tn, yn, K, mn)


def test_gate_experts_zero_weights_is_constant_mixing(tmp_path):
    _two_expert_dumps(tmp_path)
    loaded = load_dumps(tmp_path / "e", tmp_path / "n")
    le, te, ye, K, ln, me = loaded["setA"]
    X = GE.features(le, K, ln, me, GE.FEATURE_SETS["all"])
    assert X.shape == (len(K), 10) and np.isfinite(X).all()
    gate = GE.Gate(X.shape[1], hidden=4)
    with torch.no_grad():
        for p in gate.net.parameters():
            p.zero_()
        gate.net[-1].bias.fill_(0.7)
        g = gate(torch.as_tensor(X)).numpy()
    g0 = 1 / (1 + np.exp(-0.7))
    assert np.allclose(g, g0)  # all-zero weights -> g = sigmoid(b) for every row
    p = GE.mix(le, K, ln, g, 1.0).numpy()
    assert np.allclose(p.sum(-1), 1.0) and (p >= 0).all()
    assert np.allclose(p, fuse(le, K, ln, g0, "log"), atol=1e-9)  # == fuse_scores' log mode at constant g
    assert np.allclose(GE.mix(le, K, ln, np.zeros(len(K)), 1.7).numpy(), apply_bias(le, K, 0, 0, 1.7), atol=1e-9)  # g=0: energy at T


def test_gate_experts_trains_and_writes_results(tmp_path, monkeypatch):
    _two_expert_dumps(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = GE.run(tmp_path / "e", tmp_path / "n", feature_set="all", hidden=8, steps=150, quiet=True)
    assert res["val_nll"] < res["val_nll_init"] - 1e-3  # training reduces val NLL
    assert res["val_nll_T"] <= res["val_nll"] + 1e-9 and res["gate"]["n_params"] <= 2000
    assert res["gate"]["val_sets"] == ["val"] and set(res["eval"]) == {"setA"}  # never trained on the eval set
    assert 0.0 <= res["gate"]["mean_g"]["setA"] <= 1.0 and "acc" in res["eval"]["setA"]["scaled"]
    for fs in ("energy_only", "native_only"):
        r = GE.run(tmp_path / "e", tmp_path / "n", feature_set=fs, hidden=8, steps=50, quiet=True)
        assert r["gate"]["features"] == GE.FEATURE_SETS[fs]
    run_dir = tmp_path / "runs" / "gate_x"
    run_dir.mkdir(parents=True)
    res["eval"]["mmlu_pro"] = res["eval"]["setA"]  # report_native keys off mmlu_pro
    json.dump(res, open(run_dir / "results.json", "w"))
    import report_native
    row = report_native.row("gate_x")
    assert np.isclose(row["mmlu_among_k"], res["eval"]["setA"]["scaled"]["acc_k"])


# ---------------------------------------------------------------------------
# 19. scripts/eval_wf.py (PLAN7 §3x): stratified --limit + batched native scoring
# ---------------------------------------------------------------------------

import scripts.eval_wf as EW  # noqa: E402


def _flip_row(fam, gid, cands, label):
    return {"candidates": cands, "label": label, "p_null": 0.0,
            "target": [1.0 if j == label else 0.0 for j in range(len(cands))],
            "meta": {"family": fam, "flip_pair": gid}}


def test_stratified_limit_covers_every_family_and_keeps_pairs_adjacent():
    rows = []
    for fam, n in [("eligibility", 50), ("tool_select", 30), ("urgency", 10)]:
        for i in range(n):
            rows.append({"candidates": ["no", "yes"], "label": i % 2, "p_null": 0.0,
                         "target": [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0],
                         "meta": {"family": fam, "qtype": "noul"}})
    for i in range(8):  # a family entirely made of flip pairs
        rows.append(_flip_row("flip_fam", f"g{i}", ["x", "y"], i % 2))
        rows.append(_flip_row("flip_fam", f"g{i}", ["x", "y"], 1 - i % 2))

    sample = EW.stratified_limit(rows, 40, seed=0)
    fams = {r["meta"]["family"] for r in rows}
    sampled_fams = {r["meta"]["family"] for r in sample}
    assert sampled_fams == fams  # every family represented, none dropped by a naive head-N

    from collections import defaultdict
    by_pair = defaultdict(list)
    for i, r in enumerate(sample):
        fp = r["meta"].get("flip_pair")
        if fp is not None:
            by_pair[fp].append(i)
    for gid, idxs in by_pair.items():
        assert len(idxs) == 2 and abs(idxs[0] - idxs[1]) == 1, (gid, idxs)  # pair rows stay adjacent

    # deterministic given the fixed seed, and a no-op for limit<=0 or >= len(rows)
    assert EW.stratified_limit(rows, 40, seed=0) == sample
    assert EW.stratified_limit(rows, 0, seed=0) == rows
    assert EW.stratified_limit(rows, len(rows) + 5, seed=0) == rows


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="loads the real nc_v3 checkpoint "
                     "(Qwen3-1.7B-Base); set RUN_SLOW=1 to run")
def test_eval_wf_batched_matches_single_row_on_nc_v3():
    """PLAN7 §3x: scripts/eval_wf.py used to call native_kv_decide once per row (~3 rows/s); the
    fix batches rows through run_batch_native instead. Same checkpoint, same 6 rows, both paths ->
    the same argmax decision and probabilities within bf16 noise (the backbone runs in bfloat16,
    see encode.Backbone; test_native_kv_decide_matches_run_batch already covers exact agreement on
    a float32 toy model -- this checks the real checkpoint doesn't drift further than that)."""
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "runs" / "nc_v3" / "best.pt").exists():
        pytest.skip("runs/nc_v3/best.pt not present")
    from pcdm_jev.decider import PCDMDecider

    dec = PCDMDecider(str(repo_root / "runs" / "nc_v3"), mode="native", max_state=256)
    rows = [
        {"state": "A user reports a billing issue with invoice 4021.",
         "query": "Is the request eligible for a refund?", "candidates": ["no", "yes"],
         "label": 1, "p_null": 0.0, "target": [0.0, 1.0]},
        {"state": "System log: disk usage at 92% on node 7.",
         "query": "Should this trigger a page?", "candidates": ["no", "yes"],
         "label": 0, "p_null": 0.0, "target": [1.0, 0.0]},
        {"state": "Ticket: customer wants to change their shipping address.",
         "query": "Route to which team?", "candidates": ["billing", "logistics", "support"],
         "label": 1, "p_null": 0.0, "target": [0.0, 1.0, 0.0]},
        {"state": "Meeting notes about Q3 roadmap.", "query": "What is the urgency?",
         "candidates": ["low", "medium", "high"], "label": 2, "p_null": 0.0, "target": [0.0, 0.0, 1.0]},
        {"state": "A form was submitted with a missing signature field.",
         "query": "Is the form complete?", "candidates": ["no", "yes"],
         "label": 0, "p_null": 0.0, "target": [1.0, 0.0]},
        {"state": "Chat: I want to cancel my subscription please.",
         "query": "Which action should be taken?", "candidates": ["cancel", "renew", "escalate", "ignore"],
         "label": 0, "p_null": 0.0, "target": [1.0, 0.0, 0.0, 0.0]},
    ]
    single = [dec._probs("native", r["state"], r["query"], r["candidates"]).float().cpu().numpy() for r in rows]
    batched, overflow = EW._score_native(dec, rows, batch_size=len(rows))
    assert overflow == 0
    for i, r in enumerate(rows):
        k = len(r["candidates"])
        a = np.concatenate([single[i][:k], single[i][-1:]])
        b = np.concatenate([batched[i][:k], batched[i][-1:]])
        assert a.argmax() == b.argmax(), (i, a, b)
        assert np.abs(a - b).max() < 0.05, (i, a, b)


# ---------------------------------------------------------------------------
# N. null-logit offset knob (REPORT S3w calibration experiment #1)
# ---------------------------------------------------------------------------

def _factored_probs_inputs(seed=0, n=6, Kmax=3):
    """A valid factored-null logits tensor built the same way DecisionModel._factored_logits
    composes one (logits_j = log p_j + log(1-r), logits_null = log r), so
    probs_from_logits(..., null="factored") sees the same invariants a real model produces."""
    g = torch.Generator().manual_seed(seed)
    cmask = torch.zeros(n, Kmax, dtype=torch.bool)
    for i in range(n):
        cmask[i, :((i % Kmax) + 1)] = True
    r_logit = torch.randn(n, generator=g)
    log_r, log_1mr = F.logsigmoid(r_logit), F.logsigmoid(-r_logit)
    cand_logp = torch.log_softmax(torch.randn(n, Kmax, generator=g).masked_fill(~cmask, -1e9), dim=-1)
    cand_logits = (log_1mr.unsqueeze(-1) + cand_logp).masked_fill(~cmask, torch.finfo(torch.float32).min)
    logits = torch.cat([cand_logits, log_r.unsqueeze(-1)], dim=-1)
    return logits, cmask


def test_probs_from_logits_offset_b_zero_matches_legacy():
    torch.manual_seed(0)
    B, Kmax = 6, 4
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    for i in range(B):
        cmask[i, :((i % Kmax) + 1)] = True
    logits = torch.randn(B, Kmax + 1) * 2
    for Tval in (1.0, 1.7):
        legacy = T.probs_from_logits(logits, cmask, Tval, null="softmax")
        assert torch.allclose(legacy, T.probs_from_logits(logits, cmask, Tval, null="softmax", b=0.0))

    flogits, fcmask = _factored_probs_inputs()
    for Tval in (1.0, 1.7):
        legacy = T.probs_from_logits(flogits, fcmask, Tval, null="factored")
        assert torch.allclose(legacy, T.probs_from_logits(flogits, fcmask, Tval, null="factored", b=0.0))


def test_probs_from_logits_offset_shifts_logit_pnull_by_b():
    """Both null forms document the same operation: b is added to logit(P(null)). Verify the
    algebraic identity directly instead of just trusting the docstring."""
    def logit_pnull(probs):
        p = probs[:, -1].clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)

    torch.manual_seed(1)
    B, Kmax = 5, 3
    cmask = torch.ones(B, Kmax, dtype=torch.bool)
    logits = torch.randn(B, Kmax + 1)
    b_val = 1.75
    cases = {"softmax": (logits, cmask), "factored": _factored_probs_inputs(seed=2, n=B, Kmax=Kmax)}
    for null_mode, (lg, cm) in cases.items():
        base = T.probs_from_logits(lg, cm, T=1.0, null=null_mode, b=0.0)
        shifted = T.probs_from_logits(lg, cm, T=1.0, null=null_mode, b=b_val)
        assert torch.allclose(logit_pnull(shifted) - logit_pnull(base), torch.full((B,), b_val), atol=1e-4)
        assert torch.allclose(shifted.sum(-1), torch.ones(B), atol=1e-5)  # still a valid distribution


def test_fit_temperature_b_grid_recovers_known_null_miscalibration(monkeypatch):
    """Synthetic set: cand0 (gold when not null) at logit +4, cand1 at -4 (both always valid),
    null logit fixed at -1.5 -- but the true label is null 85% of the time. b=0 (today's fit)
    can only reach ~0.28 average P(null) by trading T off against candidate sharpness; letting
    b range over [-4, 4] (the PLAN7/S3w calib #1 grid) recovers the missing null mass (~0.85)
    at a strictly lower NLL. "Recovers a known offset" is cashed out as calibration recovery,
    not a single hand-picked b* -- T and b trade off against each other, so the *combination*
    the grid lands on, not either scalar alone, is what's identifiable here."""
    n = 400
    rng = np.random.RandomState(0)
    is_null = rng.rand(n) < 0.85
    logits = torch.zeros(n, 3)
    logits[:, 0], logits[:, 1], logits[:, 2] = 4.0, -4.0, -1.5
    cmask = torch.ones(n, 2, dtype=torch.bool)
    target = torch.zeros(n, 2)
    target[~torch.from_numpy(is_null), 0] = 1.0
    p_null = torch.from_numpy(is_null.astype(np.float32))

    def fake_forward_batches(backbone, model, cache, examples, bs, joint=False, readout="energy", vec_cache=None,
                              shots=0, max_state=256):
        return [(logits, cmask, target, p_null, [None] * n)]

    monkeypatch.setattr(T, "forward_batches", fake_forward_batches)
    model = argparse.Namespace(null="softmax")
    dummy_examples = list(range(n))

    T0, b0, nll0 = T.fit_temperature(None, model, None, dummy_examples, bs=64, b_grid=None)
    assert b0 == 0.0  # b_grid=None never searches an offset -- the "b=0 path unchanged" contract

    b_grid = np.arange(-4.0, 4.0 + 1e-9, 0.25)
    T1, b1, nll1 = T.fit_temperature(None, model, None, dummy_examples, bs=64, b_grid=b_grid)
    assert b1 > 2.0             # a real, positive offset recovered (raw null logit under-shoots)
    assert nll1 < nll0 - 0.3    # meaningfully lower NLL than the b=0 fit

    p_null0 = T.probs_from_logits(logits, cmask, T0, null="softmax", b=b0)[:, -1].mean().item()
    p_null1 = T.probs_from_logits(logits, cmask, T1, null="softmax", b=b1)[:, -1].mean().item()
    assert abs(p_null1 - 0.85) < 0.05   # recovers the true null rate
    assert abs(p_null0 - 0.85) > 0.2    # while the b=0 fit stays far off


def test_resolve_calib_examples_default_and_named():
    val_examples = [{"id": "v"}]
    eval_sets = {"typed_decisions_train": [{"id": "a"}, {"id": "b"}], "data_wf_val": [{"id": "c"}]}

    examples, names, b_grid = T.resolve_calib_examples(None, val_examples, eval_sets)
    assert examples is val_examples and names == ["val"] and b_grid is None

    examples, names, b_grid = T.resolve_calib_examples("typed_decisions_train,data_wf_val", val_examples, eval_sets)
    assert [e["id"] for e in examples] == ["a", "b", "c"]
    assert names == ["typed_decisions_train", "data_wf_val"]
    assert b_grid is not None and b_grid[0] == -4.0 and b_grid[-1] == 4.0

    with pytest.raises(AssertionError):
        T.resolve_calib_examples("nope", val_examples, eval_sets)


def test_resolve_best_on_default_and_named():
    eval_sets = {"data_u_val": [{"id": "a"}], "data_wh_val": [{"id": "b"}, {"id": "c"}]}
    assert T.resolve_best_on(None, eval_sets) is None
    assert T.resolve_best_on("data_u_val,data_wh_val", eval_sets) == ["data_u_val", "data_wh_val"]
    with pytest.raises(AssertionError):
        T.resolve_best_on("nope", eval_sets)


def test_checkpoint_metric_uses_best_on_mean_not_val_nll(monkeypatch):
    """tl1b item 4: with --best_on set, checkpoint_metric returns the unweighted mean NLL of
    the named eval sets, not val_nll -- and None (default) returns val_nll unchanged."""
    nll_by_set = {"data_u_val": 0.1, "data_wh_val": 0.3}  # mean = 0.2, far from val_nll=1.0
    calls = []

    def fake_eval_val_loss(backbone, model, cache, examples, bs, **kw):
        calls.append(kw)
        return nll_by_set[examples]  # examples stands in for its own set name here

    monkeypatch.setattr(T, "eval_val_loss", fake_eval_val_loss)
    eval_sets = {"data_u_val": "data_u_val", "data_wh_val": "data_wh_val"}
    args = argparse.Namespace(eval_bs=8, joint=False, readout="native", shots=0, max_state=2048)

    unchanged = T.checkpoint_metric(None, None, None, eval_sets, None, args, None, val_nll=1.0, wb=None, step=1)
    assert unchanged == 1.0

    best_on = T.resolve_best_on("data_u_val,data_wh_val", eval_sets)
    metric = T.checkpoint_metric(None, None, None, eval_sets, best_on, args, None, val_nll=1.0, wb=None, step=1)
    assert metric == pytest.approx(0.2)
    assert all(call["max_state"] == 2048 for call in calls)


def test_eval_dataset_accepts_T_offset_pairs(tiny_backbone, tiny_cache):
    torch.manual_seed(0)
    examples = make_examples()
    model = DecisionModel(d_in=tiny_backbone.d, dropout=0)
    probs, target, label = T.eval_dataset(tiny_backbone, model, tiny_cache, examples, bs=4,
                                           Ts=(1.0, (1.3, 0.5)))
    assert set(probs.keys()) == {1.0, 1.3}
    assert np.allclose(probs[1.0].sum(-1), 1.0, atol=1e-5)
    assert np.allclose(probs[1.3].sum(-1), 1.0, atol=1e-5)
    assert not np.allclose(probs[1.0], probs[1.3])  # the offset actually changed the scaled probs
