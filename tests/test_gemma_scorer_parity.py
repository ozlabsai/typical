"""No-GPU check that the Cygnet readout matches the Gemma LoRA scorer."""
import importlib.util
from pathlib import Path

import torch

from scripts.train_gemma_decision_lora import LETTERS, letter_token_groups, render


SHIM_PATH = (Path(__file__).parents[1] / ".context/research_sources/cygnet-recipe/shim/cygnet_shim.py")
spec = importlib.util.spec_from_file_location("cygnet_shim_parity", SHIM_PATH)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


class SurfaceTokenizer:
    """Small stand-in with several token IDs that decode to the same option letter."""

    surfaces = ("A", "A", "B", "B", "C", "C", "D", "D",
                "E", "E", "F", "F", "word", " a", "a")
    surfaces += tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] for i in range(6, 26))
    vocab_size = len(surfaces)

    def decode(self, ids, **_kwargs):
        return self.surfaces[ids[0]]


def test_hf_letter_groups_match_serving_surface_mass_after_rotation():
    tokenizer = SurfaceTokenizer()
    groups = letter_token_groups(tokenizer)
    # Rotate source options so A-F output slots map back to nontrivial input positions.
    order = [3, 4, 5, 0, 1, 2]
    example = {"state": "fixed", "instructions": "choose", "items": [(str(i), str(i)) for i in range(6)],
               "probabilities": [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]}
    prompt, rotated_target = render(example, order)
    assert "A. 3\nB. 4\nC. 5\nD. 0\nE. 1\nF. 2" in prompt
    assert rotated_target == [example["probabilities"][i] for i in order]

    logits = torch.tensor([((i * 17) % 23 - 11) / 10 for i in range(tokenizer.vocab_size)])
    logits[24] = 4.0  # A high-scoring token outside the serving choice mask.
    # A-F are the allowed surfaces; the unrelated vocabulary item is excluded by choice masking.
    allowed_ids = torch.cat(groups[:6])
    hf_scores = torch.stack([torch.logsumexp(logits[group], 0) for group in groups[:6]])
    hf_probs = torch.softmax(hf_scores, dim=0).tolist()

    allowed_logprobs = torch.log_softmax(logits[allowed_ids], dim=0)
    served_top = [(tokenizer.decode([token_id]), float(logprob))
                  for token_id, logprob in zip(allowed_ids.tolist(), allowed_logprobs.tolist())]
    served = shim.letter_probs(served_top, 6)
    assert served is not None
    served_probs = [served[i] for i in range(6)]
    assert torch.allclose(torch.tensor(hf_probs), torch.tensor(served_probs), atol=1e-7)

    # Both paths preserve the distribution when mapped from rotated A-F slots to original options.
    hf_by_option = [0.0] * 6
    served_by_option = [0.0] * 6
    for output_slot, original_option in enumerate(order):
        hf_by_option[original_option] = hf_probs[output_slot]
        served_by_option[original_option] = served_probs[output_slot]
    assert torch.allclose(torch.tensor(hf_by_option), torch.tensor(served_by_option), atol=1e-7)


def test_training_and_serving_accept_the_same_letter_surfaces():
    tokenizer = SurfaceTokenizer()
    groups = letter_token_groups(tokenizer)
    for index, group in enumerate(groups[:6]):
        surfaces = [tokenizer.decode([token_id]) for token_id in group.tolist()]
        parsed = shim.letter_probs([(surface, 0.0) for surface in surfaces], 6)
        assert len(surfaces) == 2
        assert parsed is not None and parsed[index] == 1.0
    assert all(groups[index].numel() == 2 for index in range(6))
    assert len(LETTERS) == 26
