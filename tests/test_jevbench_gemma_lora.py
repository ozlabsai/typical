import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.jevbench_gemma_lora import (
    GemmaLoraAdapter,
    InvalidTask,
    probabilities_from_option_scores,
    render_prompt,
    task_items,
    validate_native_probs,
)


JEVBENCH = Path(".context/jevbench-v154").resolve()


def test_direct_script_help_bootstraps_repository_imports(tmp_path):
    script = Path("scripts/jevbench_gemma_lora.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Run a pinned Gemma-4" in result.stdout


def task(qtype, labels, criteria, state="state", instructions="decide"):
    return SimpleNamespace(
        id="golden", state=state, labels=labels,
        question={"type": qtype, "criteria": criteria, "instructions": instructions},
    )


def test_choice_uses_declared_label_order_under_permutations():
    criteria = {"third": "Third", "first": "First", "second": "Second"}
    for labels in (["first", "second", "third"], ["third", "first", "second"]):
        current = task("choice", labels, criteria)
        prompt, rendered_labels = render_prompt(current)
        assert rendered_labels == labels
        assert [line for line in prompt.splitlines() if line[:2] in {"A.", "B.", "C."}] == [
            f"{letter}. {criteria[label]}" for letter, label in zip("ABC", labels)
        ]
        probs = probabilities_from_option_scores(labels, [3.0, 2.0, 1.0])
        assert list(probs) == labels and max(probs, key=probs.get) == labels[0]


def test_noul_maps_no_to_false_yes_to_true_and_keeps_threshold_text():
    criteria = {"true": "eligible when score ≥ 0.72", "false": "ineligible below 0.72"}
    current = task("noul", ["yes", "no"], criteria)
    assert task_items(current) == [
        ("yes", "eligible when score ≥ 0.72"), ("no", "ineligible below 0.72")
    ]
    prompt, labels = render_prompt(current)
    assert labels == ["yes", "no"]
    assert "A. eligible when score ≥ 0.72" in prompt
    assert "B. ineligible below 0.72" in prompt


def test_score_levels_remain_ordered_and_have_the_expected_value():
    current = task("score", ["0", "1", "2", "3"], ["none", "low", "high", "critical"])
    assert task_items(current) == list(zip(current.labels, current.question["criteria"]))
    probs = probabilities_from_option_scores(current.labels, [0.0, 1.0, 2.0, 3.0])
    expected = sum(int(label) * probability for label, probability in probs.items())
    assert 2.4 < expected < 2.6
    with pytest.raises(InvalidTask, match="ordered string indices"):
        task_items(task("score", ["1", "0"], ["low", "high"]))


def test_unicode_and_hebrew_survive_rendering_without_escaping():
    current = task(
        "choice", ["אישור", "דחייה"], {"דחייה": "דחה בקשה", "אישור": "אשר בקשה"},
        state={"לקוח": "שרה", "הערה": "שלום 👋"}, instructions="מה ההחלטה?",
    )
    prompt, labels = render_prompt(current)
    assert labels == ["אישור", "דחייה"]
    assert '"לקוח": "שרה"' in prompt and "שלום 👋" in prompt and "מה ההחלטה?" in prompt


def test_jevbench_uses_shared_exact_singleton_forward_and_probabilities(monkeypatch):
    import torch
    import scripts.singleton_option_scorer as scorer

    assert probabilities_from_option_scores is scorer.probabilities_from_option_scores
    assert validate_native_probs is scorer.validate_native_probs
    ids = torch.tensor([[8, 9, 10]])
    logits = torch.tensor([[[99.0, -99.0, 0.0], [0.5, 2.0, 1.5]]], dtype=torch.bfloat16)
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        assert set(kwargs) == {"input_ids", "use_cache", "logits_to_keep"}
        assert kwargs["input_ids"].shape == (1, 3)
        assert kwargs["use_cache"] is False
        assert type(kwargs["logits_to_keep"]) is int and kwargs["logits_to_keep"] == 1
        return SimpleNamespace(logits=logits)

    adapter = GemmaLoraAdapter("base", "rev", None, None, 8192, "bfloat16", device="cpu")
    adapter._model = model
    adapter._groups = [torch.tensor([0, 2]), torch.tensor([1])]
    monkeypatch.setattr(adapter, "_input_ids", lambda prompt: (ids, 3))
    actual, count = adapter._predict(task("choice", ["a", "b"], {"a": "A", "b": "B"}))
    expected = scorer.probabilities_from_option_scores(
        ["a", "b"], [torch.logsumexp(logits[0, -1, [0, 2]].float(), 0).item(), 2.0])
    assert actual == expected and count == 3 and len(calls) == 1


@pytest.mark.parametrize("probs", [
    {"a": 0.5},
    {"a": 0.5, "b": 0.4},
    {"a": math.nan, "b": math.nan},
    {"a": -0.1, "b": 1.1},
    {"b": 0.5, "a": 0.5},
])
def test_invalid_native_probabilities_fail_closed(probs):
    with pytest.raises(ValueError):
        validate_native_probs(probs, ["a", "b"])


def test_overlength_is_a_422_and_never_truncated(monkeypatch):
    if str(JEVBENCH) not in sys.path:
        sys.path.insert(0, str(JEVBENCH))

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "encoded"

        def __call__(self, *args, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": SimpleNamespace(shape=(1, 9))}

    adapter = GemmaLoraAdapter("base", "rev", None, None, 8, "bfloat16")
    adapter._tokenizer = Tokenizer()
    adapter._model = object()
    adapter._groups = []
    monkeypatch.setattr(adapter, "load", lambda: adapter)
    result = adapter.run(task("choice", ["a", "b"], {"a": "A", "b": "B"}))
    assert not result.ok and result.status == 422
    assert result.raw["runtime"] == {"max_context": 8, "truncated": False}
    assert "9 tokens" in result.error


def test_current_jevbench_runner_accepts_the_native_result(monkeypatch, tmp_path):
    if str(JEVBENCH) not in sys.path:
        sys.path.insert(0, str(JEVBENCH))
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.tasks import Task

    current = Task(
        id="runner-golden", family="routing", state="שלום", split="public", expected="ב",
        labels=["א", "ב"], question={"type": "choice", "instructions": "בחר",
                                      "criteria": {"א": "ראשון", "ב": "שני"}},
    )
    current.validate()
    adapter = GemmaLoraAdapter("base", "rev", "adapter", "adapter-rev", 32768, "bfloat16")
    monkeypatch.setattr(adapter, "load", lambda: adapter)
    monkeypatch.setattr(adapter, "_predict", lambda unused: ({"א": 0.25, "ב": 0.75}, 17))
    record = Runner(adapter, Ledger(tmp_path / "ledger.jsonl"), tmp_path / "raw",
                    default_reserve_usd=0.0).run_task(current)
    assert record["ok"] and record["valid"] and record["strict_valid"] and record["correct"]
    assert record["probs"] == {"א": 0.25, "ב": 0.75}
