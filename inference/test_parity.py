"""Parity check: inference/typical/ vs the training repo's pcdm_jev.decider.PCDMDecider
(mode="native"), same checkpoint (guychuk/pcdm-runs/typical-small/best.pt), same device,
same 6 items. Downloads a 1.7B checkpoint -> gated like the repo's other real-checkpoint
tests (tests/test_pipeline.py, tests/test_jevbench.py): RUN_SLOW=1 required.

uv run --no-sync pytest inference/test_parity.py -v   # (with RUN_SLOW=1 set)
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "pcdm"), str(INFERENCE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT_REPO = "guychuk/pcdm-runs"
CKPT_FILE = "typical-small/best.pt"

STATE = (
    "Ticket #4821: Customer says their package arrived damaged. They want a replacement "
    "shipped overnight, not a refund. Order was placed 3 days ago, still under warranty."
)

ITEMS = [
    ("What does the customer want?", ["refund", "replacement", "repair"]),
    ("Is the order still under warranty?", ["no", "yes"]),
    ("How urgent is this ticket?", ["0", "1", "2", "3"]),
    ("What is the customer's sentiment?", ["positive", "neutral", "negative"]),
    ("Should this be escalated to a manager?", ["no", "yes"]),
    ("Which department should own this ticket?", ["billing", "shipping", "support"]),
]

NOUL_ITEMS = [
    "Is the order still under warranty?",
    "Should this be escalated to a manager?",
    "Was the package damaged?",
]


def test_expand_state_cache_matches_deepcopy_and_does_not_alias():
    """Phase A (serving speedup): `_expand_state_cache` replaces `copy.deepcopy(state_cache)`
    + `batch_repeat_interleave` with a zero-copy `.expand()`. No checkpoint needed -- this
    exercises the cache mechanics directly against transformers' real DynamicCache/DynamicLayer
    with random tensors, checking (a) bit-identical results to the old deepcopy+repeat
    reference and (b) that expanding+updating the new cache never mutates the original."""
    import torch
    from transformers.cache_utils import DynamicCache

    from typical.native import _expand_state_cache

    torch.manual_seed(0)
    B, H, Ls, D, m = 1, 4, 7, 8, 5
    k0, v0 = torch.randn(B, H, Ls, D), torch.randn(B, H, Ls, D)
    k1, v1 = torch.randn(B, H, Ls, D), torch.randn(B, H, Ls, D)
    cache = DynamicCache(ddp_cache_data=[(k0, v0), (k1, v1)])
    orig_keys = [layer.keys.clone() for layer in cache.layers]
    orig_values = [layer.values.clone() for layer in cache.layers]

    # ground truth: the old path (deepcopy, then the real repeat_interleave every layer used
    # to get regardless of whether it holds K/V or Gated-DeltaNet conv/recurrent state)
    import copy
    ref = copy.deepcopy(cache)
    ref.batch_repeat_interleave(m)

    new = _expand_state_cache(cache, m)

    q_len = 3
    torch.manual_seed(1)
    new_k = [torch.randn(m, H, q_len, D), torch.randn(m, H, q_len, D)]
    new_v = [torch.randn(m, H, q_len, D), torch.randn(m, H, q_len, D)]
    for i in range(2):
        k_ref, v_ref = ref.update(new_k[i], new_v[i], i)
        k_new, v_new = new.update(new_k[i], new_v[i], i)
        assert torch.equal(k_ref, k_new) and torch.equal(v_ref, v_new), f"layer {i} mismatch"

    # the original (batch=1) cache must be untouched by both the expand and the update above
    for i, layer in enumerate(cache.layers):
        assert torch.equal(layer.keys, orig_keys[i]) and torch.equal(layer.values, orig_values[i]), \
            f"layer {i} of the original state_cache was mutated"
    print("[expand_state_cache] matches deepcopy+repeat_interleave reference, no aliasing")


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads the real typical-small "
                     "checkpoint (Qwen3-1.7B-Base + LoRA); set RUN_SLOW=1 to run")
def _assert_parity(repo, filename):
    """-> (orig, mine) after checking every ITEMS row, scored one call each, to 1e-4."""
    from huggingface_hub import hf_hub_download

    from pcdm_jev.decider import PCDMDecider
    from typical import Typical

    run_dir = os.path.dirname(hf_hub_download(repo, filename))
    orig = PCDMDecider(run_dir=run_dir, mode="native", device="auto")
    mine = Typical.from_pretrained(repo, filename=filename, device=orig.device, max_state=orig.max_state)
    assert orig.device == mine.device

    max_diff = 0.0
    for query, labels in ITEMS:
        p_orig = orig._probs("native", STATE, query, labels)
        p_mine = mine._raw(STATE, query, labels)
        diff = (p_orig.float().cpu() - p_mine.float().cpu()).abs().max().item()
        max_diff = max(max_diff, diff)
        assert diff < 1e-4, f"{query!r} {labels}: max abs diff {diff}"
    print(f"[parity] {repo}: {len(ITEMS)} items, max abs prob diff = {max_diff:.2e}")
    return orig, mine


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads the released typical-small; set RUN_SLOW=1 to run")
def test_release_semif_matches_pcdm_decider():
    """The release renders semif. A call with no yes/no row used to get an eos tail the training path never adds
    (2026-10-27: |dp| up to 0.09 at K>=3 on the release, 0.7 on an airdef fine-tune)."""
    _assert_parity("OzLabs/typical-small", "best.pt")


def test_typical_matches_pcdm_decider():
    from typical import to_labels

    orig, mine = _assert_parity(CKPT_REPO, CKPT_FILE)

    # noul reversed-label control: the checkpoint's noul_head is bern (query-only, position-
    # invariant by construction -- the rendered suffix never contains the candidate text, so
    # P(yes) under ["no","yes"] and ["yes","no"] is the SAME forward pass mathematically).
    # ponytail: exact float equality is verified on CPU (checked separately, bit-identical);
    # on MPS the two calls can differ by ~1e-7 from non-deterministic reduction order, so this
    # check uses a tight epsilon rather than `==` to stay device-agnostic.
    eps = 1e-6
    for q in NOUL_ITEMS:
        p_yes_forward = mine.noul(STATE, q)
        probs_rev, _ = to_labels(mine._raw(STATE, q, ["yes", "no"]), ["yes", "no"])
        assert abs(probs_rev["yes"] - p_yes_forward) < eps, (q, probs_rev["yes"], p_yes_forward)

        orig_fwd, _ = to_labels(orig._probs("native", STATE, q, ["no", "yes"]), ["no", "yes"])
        orig_rev, _ = to_labels(orig._probs("native", STATE, q, ["yes", "no"]), ["yes", "no"])
        assert abs(orig_fwd["yes"] - orig_rev["yes"]) < eps, (q, orig_fwd["yes"], orig_rev["yes"])
        assert abs(orig_fwd["yes"] - p_yes_forward) < 1e-4, (q, orig_fwd["yes"], p_yes_forward)
    print(f"[parity] noul reversed-label check OK over {len(NOUL_ITEMS)} items (eps={eps})")


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads the real "
                     "typical-small-preview checkpoint; set RUN_SLOW=1 to run")
def test_typical_noul_via_choice_path_on_preview():
    """typical-small-preview trains noul_head="choice" (no Bernoulli head): m.noul() must
    still work by scoring the plain K-way ["no","yes"] choice and reading off P(yes) -- and
    must still match the original decider exactly (order-invariance is a bern-only guarantee,
    not required here)."""
    from huggingface_hub import hf_hub_download

    from pcdm_jev.decider import PCDMDecider
    from typical import Typical, to_labels

    local_best_pt = hf_hub_download(CKPT_REPO, "typical-small-preview/best.pt")
    run_dir = os.path.dirname(local_best_pt)

    orig = PCDMDecider(run_dir=run_dir, mode="native", device="auto")
    assert orig.native["m"].noul_head == "choice"
    mine = Typical.from_pretrained(CKPT_REPO, filename="typical-small-preview/best.pt",
                                   device=orig.device, max_state=orig.max_state)
    assert mine.model.noul_head == "choice"

    for q in NOUL_ITEMS:
        p_yes = mine.noul(STATE, q)
        assert 0.0 <= p_yes <= 1.0
        orig_probs, _ = to_labels(orig._probs("native", STATE, q, ["no", "yes"]), ["no", "yes"])
        assert abs(orig_probs["yes"] - p_yes) < 1e-4, (q, orig_probs["yes"], p_yes)
    print(f"[parity] preview noul() via K-way choice path OK over {len(NOUL_ITEMS)} items")


QWEN35_NAME = "Qwen/Qwen3.5-0.8B-Base"
QWEN35_CACHED = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base")


@pytest.mark.skipif(not os.path.isdir(QWEN35_CACHED), reason=f"{QWEN35_NAME} not cached locally")
def test_typical_from_pretrained_loads_qwen35_checkpoint(tmp_path, monkeypatch):
    """No trained Qwen3.5 checkpoint exists yet (see the porting task report) -- this proves
    Typical.from_pretrained would load one anyway: it rebuilds backbone+head purely from
    ckpt["args"] (backbone id, tap_layer, lora_r/lora_layers, nc_head/nc_render/null/noul_head
    -- see core.py's docstring), never hardcoding a Qwen3 backbone name. A synthetic checkpoint
    built from a real (tiny-tap, for speed) Qwen3.5-0.8B-Base backbone + NativeHead stands in
    for the real thing; hf_hub_download is monkeypatched to return it so from_pretrained's own
    code path runs unmodified, same as it would against a real Hub repo."""
    import torch

    from encode import Backbone as TrainBackbone
    from native import NativeHead as TrainNativeHead

    args = {"backbone": QWEN35_NAME, "readout": "native", "tap_layer": 8, "lora_layers": 2,
            "lora_r": 4, "nc_head": "n3", "nc_render": "letters_nonull", "null": "factored",
            "noul_head": "choice", "score_head": "choice"}
    bb = TrainBackbone(args["backbone"], lora_layers=args["lora_layers"], lora_r=args["lora_r"],
                       device="cpu", tap_layer=args["tap_layer"])
    model = TrainNativeHead(bb.d, nc_head=args["nc_head"], null=args["null"], render=args["nc_render"],
                            score_head=args["score_head"], noul_head=args["noul_head"])
    ckpt_path = tmp_path / "best.pt"
    torch.save({"tower": model.state_dict(), "lora": bb.lora_state_dict(), "args": args}, ckpt_path)

    import typical.core as core_mod
    from typical import Typical
    monkeypatch.setattr(core_mod, "hf_hub_download", lambda *a, **k: str(ckpt_path))

    mine = Typical.from_pretrained("fake/does-not-exist", device="cpu")
    assert mine.head.backbone.model.config.hidden_size == bb.d
    probs, runtime = mine.decide(
        "A customer reports their package arrived damaged and wants a replacement.",
        {"type": "noul", "instructions": "Should this be escalated?",
         "criteria": {"true": "Yes", "false": "No"}}, ["no", "yes"])
    assert abs(sum(probs.values()) - 1.0) < 1e-4  # to_labels renormalises over labels
    assert 0.0 <= runtime["p_null"] <= 1.0


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads the real typical-small "
                     "checkpoint; set RUN_SLOW=1 to run")
def test_state_cache_is_exact_and_faster():
    """Serving-latency work (PLAN item 2): Typical's per-instance state LRU (core.py's
    _state_kv_for) must make a second choice() call on the SAME state (a) bit-identical to
    the first call's probabilities and (b) meaningfully faster, since it skips
    native.encode_state's state_prefix_forward entirely on the cache hit."""
    import time

    from typical import Typical

    m = Typical.from_pretrained(CKPT_REPO, filename=CKPT_FILE, device="cpu")
    query, labels = ITEMS[0]

    m._state_kv.clear()
    t0 = time.perf_counter()
    first = m.choice(STATE, query, labels)
    cold_s = time.perf_counter() - t0
    assert len(m._state_kv) == 1, "one state text -> one cache entry"

    t0 = time.perf_counter()
    second = m.choice(STATE, query, labels)
    warm_s = time.perf_counter() - t0
    assert len(m._state_kv) == 1, "same state text -> cache hit, no growth"

    assert first == second, "cache hit must reproduce the cold call's probabilities exactly"
    assert warm_s < cold_s, f"warm ({warm_s:.4f}s) should be faster than cold ({cold_s:.4f}s)"

    # LRU eviction: max_states caps how many distinct states are held
    m2 = Typical.from_pretrained(CKPT_REPO, filename=CKPT_FILE, device="cpu", max_states=2)
    for i in range(5):
        m2.choice(f"{STATE} [{i}]", query, labels)
    assert len(m2._state_kv) == 2, "LRU must evict down to max_states"


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads the real typical-small "
                     "checkpoint; set RUN_SLOW=1 to run")
def test_max_option_tokens_is_noop_when_nothing_truncated():
    """Serving-latency work (PLAN item 5, "suffix diet"): capping each rendered option at
    max_option_tokens must not move probabilities at all when no option actually exceeds the
    cap -- the default (24) is generous enough that ITEMS' short labels never trigger it."""
    from typical import Typical
    from typical.native import native_kv_decide

    m = Typical.from_pretrained(CKPT_REPO, filename=CKPT_FILE, device="cpu")
    query, labels = ITEMS[0]
    cache = m._state_kv_for(STATE)

    p_default = native_kv_decide(m.head, m.model, STATE, [(query, labels)], max_state=m.max_state,
                                 state_cache=cache)[0]  # default max_option_tokens=24
    p_uncapped = native_kv_decide(m.head, m.model, STATE, [(query, labels)], max_state=m.max_state,
                                  state_cache=cache, max_option_tokens=None)[0]
    diff = (p_default - p_uncapped).abs().max().item()
    assert diff == 0.0, f"max abs diff {diff}"

    # a genuinely long option DOES get truncated and still returns a valid distribution
    long_labels = [labels[0], labels[1] + " extra words " * 20] + labels[2:]
    p_long = m.choice(STATE, query, long_labels)
    assert abs(sum(v for k, v in p_long.items() if k != "p_null") - 1.0) < 1e-4


if __name__ == "__main__":
    test_expand_state_cache_matches_deepcopy_and_does_not_alias()
    if not os.environ.get("RUN_SLOW"):
        os.environ["RUN_SLOW"] = "1"
    test_typical_matches_pcdm_decider()
    test_typical_noul_via_choice_path_on_preview()
    test_state_cache_is_exact_and_faster()
    test_max_option_tokens_is_noop_when_nothing_truncated()
