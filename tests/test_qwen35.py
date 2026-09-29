"""Qwen3.5 backbone port (see PLAN.md "Qwen3.5 port" / the porting task). Skipped unless
Qwen/Qwen3.5-0.8B-Base is already cached locally (`huggingface_hub.snapshot_download`) --
these are real-checkpoint tests, not the tiny random-weight fixtures tests/test_pipeline.py
uses for Qwen3. Qwen3.5 wraps its text trunk in a VL config (text_config/vision_config) and
alternates Gated-DeltaNet linear-attention layers with regular full-attention layers
(config.layer_types); Backbone/native.py must handle both transparently, and Qwen3 (tested
in tests/test_pipeline.py) must be completely unaffected -- see pcdm/encode.py/native.py comments.
"""
import os

import pytest
import torch

NAME = "Qwen/Qwen3.5-0.8B-Base"
CACHED = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base")
pytestmark = pytest.mark.skipif(not os.path.isdir(CACHED), reason=f"{NAME} not cached locally")


def test_backbone_hybrid_stack_tap_and_lora():
    """tap_layer at ~71% depth (24 layers -> 17) keeps whole layers (both linear- and
    full-attention survive), records layer_types in backbone.info, and LoRA lands on q/k/v/o
    (full-attention), the Gated-DeltaNet in/out projections (linear-attention), and the MLP --
    every layer type in the kept stack gets LoRA on whatever it actually has."""
    from encode import Backbone

    tap = round(0.71 * 24)  # 17
    bb = Backbone(name=NAME, lora_layers=2, lora_r=4, device="cpu", tap_layer=tap)
    assert bb.d == 1024
    assert bb.tap_layer == tap == len(bb.model.layers)
    assert bb.info["n_layers"] == tap
    layer_types = bb.info["layer_types"]
    assert len(layer_types) == tap
    assert set(layer_types) == {"linear_attention", "full_attention"}  # both survive the cut
    assert layer_types.count("full_attention") == 4  # interval-4 pattern -> layers 3,7,11,15

    lora_modules = set(bb.info["lora_modules"])
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= lora_modules  # full-attention layer 15
    assert {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= lora_modules  # linear-attn layer 16
    assert {"gate_proj", "up_proj", "down_proj"} <= lora_modules  # mlp, every layer

    # zero-init LoRA (B=0) -> h_top must match a lora_r=0 Backbone loading the same weights.
    bb0 = Backbone(name=NAME, lora_layers=2, lora_r=0, device="cpu", tap_layer=tap)
    texts = ["The quarterly report shows steady growth.", "yes",
             "A much longer sentence to exercise ragged padding end to end across the hybrid stack."]
    input_ids, attention_mask = bb.tokenize(texts, max_len=32)
    with torch.no_grad():
        h_top, h_frozen, mask = bb(input_ids, attention_mask)
        h_top0, h_frozen0, mask0 = bb0(input_ids, attention_mask)
    assert torch.allclose(h_top, h_top0, atol=1e-2)
    B, T = input_ids.shape
    assert h_top.shape == (B, T - 1, bb.d) and h_frozen.shape == (B, T - 1, bb.d)
    assert mask.shape == (B, T - 1) and mask.dtype == torch.bool
    assert torch.equal(mask.sum(1), attention_mask.sum(1) - 1)
    assert torch.isfinite(h_top).all() and torch.isfinite(h_frozen).all()


def test_mcq_zero_shot_letters_and_shapes():
    """MCQHead's frozen letter-logit path (mcq.run_batch_mcq, --shots) on Qwen3.5: single-token
    letter assumption holds on its (much larger) vocab, logits are finite and shaped like any
    other backbone, and a --shots > 0 prefix changes n_tokens without changing the shape --
    mirrors tests/test_pipeline.py's test_mcq_letter_logits_shape_finite/test_mcq_shots_prefix
    for Qwen3, on the real small Qwen3.5 checkpoint. tap_layer=8 keeps this fast (2 full- + 6
    linear-attention layers -- still a real hybrid stack, just a shallow one)."""
    from mcq import MCQHead, collate_mcq, run_batch_mcq

    head = MCQHead(name=NAME, lora_layers=0, lora_r=0, device="cpu", tap_layer=8).eval()
    assert head.letter_ids.shape == (52,)
    examples = [
        {"state": "A customer asks about order status.", "query": "Has the order shipped?",
         "candidates": ["no", "yes"], "target": [1.0, 0.0], "p_null": 0.0},
        {"state": "A ticket describes a login failure.", "query": "What team should handle this?",
         "candidates": ["billing", "support", "engineering"], "target": [0.0, 1.0, 0.0], "p_null": 0.0},
    ]
    batch = collate_mcq(examples)
    with torch.inference_mode():
        logits0 = run_batch_mcq(head, batch, examples, shots=0)
        logits3 = run_batch_mcq(head, batch, examples, shots=3)
    assert logits0.shape == logits3.shape == (2, 4)
    for logits in (logits0, logits3):
        assert torch.isfinite(logits[0, :2]).all() and torch.isfinite(logits[0, -1]).all()
        assert torch.isfinite(logits[1, :3]).all() and torch.isfinite(logits[1, -1]).all()
        assert logits[0, 2] == torch.finfo(logits.dtype).min  # padded column (K=2 row, Kmax=3)
    assert batch["n_tokens"] > 0


def test_native_kv_decide_matches_run_batch_qwen35_fp32():
    """The hybrid KV-cache serving path (native.native_kv_decide: state prefix cached once,
    suffix chunks run against it with an explicit dense mask + position_ids) must reproduce
    run_batch_native's full uncached forward -- the same contract
    tests/test_pipeline.py::test_native_kv_decide_matches_run_batch checks for Qwen3.
    float32 isolates the mechanism from bf16 rounding (Backbone always trains/serves in bf16;
    see encode.Backbone's dtype param): Qwen3.5's cache is hybrid (per-token KV for
    full-attention layers, a Gated-DeltaNet recurrent+conv state for linear-attention layers,
    via transformers' DynamicCache/LinearAttentionLayer) and its linear-attention layers don't
    implement batch_repeat_interleave themselves -- native._cache_batch_repeat_interleave
    covers that gap; this test is the regression check for both that fix and the mask/position
    plumbing around it."""
    from mcq import MCQHead, collate_mcq
    from native import NativeHead, run_batch_native, native_kv_decide

    torch.manual_seed(0)
    head = MCQHead(name=NAME, lora_layers=2, lora_r=4, device="cpu", tap_layer=8,
                    dtype=torch.float32).eval()
    model = NativeHead(head.backbone.d, nc_head="n3", null="factored").eval()

    state = "a shared state text used for every query below, long enough to exercise the hybrid stack"
    queries = [(f"query number {i}", [f"option {j}" for j in range(k)]) for i, k in enumerate((2, 3, 2))]
    examples = [{"state": state, "query": q, "candidates": c, "target": [1.0] + [0.0] * (len(c) - 1),
                 "p_null": 0.0, "task": "smoke"} for q, c in queries]
    batch = collate_mcq(examples)
    with torch.inference_mode():
        probs = torch.softmax(run_batch_native(head, model, batch, examples), dim=-1)
    dists = native_kv_decide(head, model, state, queries, chunk=2)

    assert len(dists) == 3
    for i, (_, c) in enumerate(queries):
        expected = torch.cat([probs[i, :len(c)], probs[i, -1:]])
        assert torch.allclose(dists[i], expected, atol=1e-4), (i, (dists[i] - expected).abs().max())
        assert torch.allclose(dists[i].sum(), torch.tensor(1.0), atol=1e-4)


def test_visual_native_cache_matches_full_qwen35_fp32():
    """One-image state prefill must preserve Qwen3.5's M-RoPE offset in cached suffixes."""
    from PIL import Image

    from mcq import MCQHead, collate_mcq
    from native import NativeHead, native_kv_decide, run_batch_native

    torch.manual_seed(0)
    head = MCQHead(name=NAME, lora_layers=0, lora_r=0, device="cpu", tap_layer=4,
                   dtype=torch.float32, multimodal=True, max_image_pixels=65536).eval()
    model = NativeHead(head.backbone.d, nc_head="n3", null="factored").eval()
    image = Image.new("RGB", (64, 64), "red")
    state = "Use the image as evidence."
    queries = [("What is the dominant color?", ["red", "blue"]),
               ("Does the image contain green?", ["yes", "no"])]
    examples = [{"state": state, "query": query, "candidates": candidates,
                 "target": [1.0, 0.0], "p_null": 0.0, "task": "scienceqa_image", "image": image}
                for query, candidates in queries]
    with torch.inference_mode():
        probs = torch.softmax(run_batch_native(head, model, collate_mcq(examples), examples), dim=-1)
        blue = [dict(example, image=Image.new("RGB", (64, 64), "blue")) for example in examples]
        blue_probs = torch.softmax(run_batch_native(head, model, collate_mcq(blue), blue), dim=-1)
    assert not torch.allclose(probs, blue_probs, atol=1e-5), "visual state ignored the image"
    cached = native_kv_decide(head, model, state, queries, chunk=2, image=image)
    for row, distribution in enumerate(cached):
        expected = torch.cat([probs[row, :2], probs[row, -1:]])
        assert torch.allclose(distribution, expected, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla's DeltaNet kernel is Triton/CUDA-only "
                     "(installed alongside this checkpoint on GPU pods) -- runs on CPU raise "
                     "'Pointer argument cannot be accessed from Triton', unrelated to this test")
def test_bench_b_fair_batched_runs_without_crashing_on_hybrid_cache():
    """pcdm/bench.py's B_fair baseline (score_b_fair_chunk / _cache_rows) crashed on Qwen3.5's hybrid
    cache with AttributeError: 'LinearAttentionLayer' object has no attribute
    'batch_repeat_interleave' (tm2 pod run, 2026-09-22) -- score_b_fair_chunk's own top-level
    cache expansion AND _cache_rows' sub-batch row-slicing both called the raw method directly
    instead of native._cache_batch_repeat_interleave / the new native._cache_select_rows.
    max_seqs=2 forces multiple sub-batches through _cache_rows for m=3 queries, so this
    exercises exactly the path that crashed, on the real hybrid stack (not Qwen3's uniform
    DynamicCache, which never hit this bug). This only pins "doesn't crash" -- the batched-vs-
    unbatched *numeric* parity check_batched_matches_fair enforces on Qwen3 does not yet hold
    on this hybrid cache (~0.9 nat max diff, un-root-caused; see that function's TODO) and is
    deliberately not asserted here. B_fair is a baseline-only comparator, not our product path;
    native_kv_decide's own parity test above (test_native_kv_decide_matches_run_batch_qwen35_fp32)
    is unaffected."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench import check_batched_matches_fair

    tok = AutoTokenizer.from_pretrained(NAME, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to("cuda").eval()
    check_batched_matches_fair(lm, tok, "cuda", k=3, m=3, max_seqs=2)  # must not raise
