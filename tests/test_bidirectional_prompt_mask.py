"""R057 prompt-bidirectional mask: rule check plus a tiny random Gemma 4 on CPU (no download)."""

import torch
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

from scripts.train_gemma_decision_lora import enable_bidirectional_prompt, prefix_bidirectional_masks


def test_mask_rule_matches_brute_force():
    lengths, total, window = [6, 4], 6, 3
    masks = prefix_bidirectional_masks(lengths, total, window)
    for b, n in enumerate(lengths):
        for q in range(total):
            for k in range(total):
                full = k < n and (k <= q or (q < n - 1 and k < n - 1))
                assert bool(masks["full_attention"][b, 0, q, k]) == full
                assert bool(masks["sliding_attention"][b, 0, q, k]) == (full and abs(q - k) < window)
    # the read-out slot (n-1) reads every token and no earlier token reads it
    assert masks["full_attention"][0, 0, 5].all() and not masks["full_attention"][0, 0, :5, 5].any()


def _tiny():
    torch.manual_seed(0)
    text = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, global_head_dim=16, sliding_window=4,
                layer_types=["sliding_attention", "full_attention"] * 2)
    config = Gemma4UnifiedConfig(text_config=text, vision_config=dict(mm_embed_dim=32, output_proj_dims=32),
                                 audio_config=dict(hidden_size=16, audio_embed_dim=16, output_proj_dims=16),
                                 image_token_id=120, boi_token_id=121, eoi_token_id=122, video_token_id=123,
                                 boa_token_id=124, eoa_token_index=125, audio_token_id=126)
    return Gemma4UnifiedForConditionalGeneration(config).eval()


@torch.no_grad()
def test_hook_off_is_causal_on_reads_the_prompt_bidirectionally():
    model = _tiny()
    ids = torch.randint(3, 100, (1, 10))
    late = ids.clone()
    late[0, 8] += 1                      # last prompt token before the read-out slot (9)
    slot = ids.clone()
    slot[0, 9] += 1                      # the read-out slot itself
    hidden = lambda x: model.model(input_ids=x, use_cache=False).last_hidden_state
    off, off_late = hidden(ids), hidden(late)
    assert torch.equal(off[0, :8], off_late[0, :8])           # stock: causal
    handle = enable_bidirectional_prompt(model)
    on, on_late, on_slot = hidden(ids), hidden(late), hidden(slot)
    assert (on[0, 0] - on_late[0, 0]).abs().max() > 0         # a late prompt token reaches position 0
    assert torch.equal(on[0, :9], on_slot[0, :9])             # the read-out slot is no key for the prompt
    padded = torch.cat([ids, torch.zeros(1, 3, dtype=ids.dtype)], 1)
    mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 3)], 1).long()
    torch.testing.assert_close(model.model(input_ids=padded, attention_mask=mask, use_cache=False)
                               .last_hidden_state[0, :10], on[0], rtol=1e-5, atol=1e-5)  # padding is never a key
    handle.remove()
    assert torch.equal(hidden(ids), off)                     # removing the hook restores the stock forward
