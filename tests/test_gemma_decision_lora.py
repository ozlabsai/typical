import pytest

from scripts.train_gemma_decision_lora import flatten_row, render


def test_option_permutation_keeps_soft_target_aligned():
    row = {
        "id": "case",
        "state": {"amount": 7},
        "questions": {"decision": {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"left": "Left option", "right": "Right option"},
        }},
        "targets": {"decision": {"label": "right"}},
        "soft_targets": {"decision": [0.2, 0.8]},
    }
    example, = flatten_row(row)
    prompt, probabilities = render(example, [1, 0])
    assert "A. Right option\nB. Left option" in prompt
    assert probabilities == [0.8, 0.2]
    assert '"amount": 7' in prompt


def test_rejects_target_outside_options():
    row = {
        "id": "bad",
        "state": "state",
        "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        "targets": {"q": {"label": "c"}},
    }
    with pytest.raises(ValueError, match="target is not an option"):
        flatten_row(row)
