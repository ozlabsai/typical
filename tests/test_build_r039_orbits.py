from collections import Counter

from scripts.build_r039_orbits import _by_source, build, operand_shortcut_accuracy, rendered_signature
from scripts.train_gemma_decision_lora import encode, flatten_row


class _Tokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def __call__(self, *_args, **_kwargs):
        import torch
        return {"input_ids": torch.tensor([[1, 2]])}


def test_r039_arms_are_matched_but_differ_in_surface_interactions():
    complete, partial, evaluation = build(train_sources=8, eval_sources=32, seed=42)
    assert len(complete) == len(partial) == 512 and len(evaluation) == 2048
    assert not (set(map(rendered_signature, complete)) & set(map(rendered_signature, evaluation)))
    assert operand_shortcut_accuracy(complete, evaluation) <= .75
    assert all((left["meta"]["source_id"], left["meta"]["world"], left["meta"]["yes_first"]) ==
               (right["meta"]["source_id"], right["meta"]["world"], right["meta"]["yes_first"])
               for left, right in zip(complete, partial))
    for full, half in zip(_by_source(complete).values(), _by_source(partial).values()):
        full_pairs = {(row["meta"]["criteria_style"], row["meta"]["wording"]) for row in full}
        half_pairs = {(row["meta"]["criteria_style"], row["meta"]["wording"]) for row in half}
        assert len(full_pairs) == 16
        assert len(half_pairs) == 8
        assert Counter(row["targets"]["decision"]["label"] for row in full) == Counter(yes=32, no=32)
        assert Counter(row["targets"]["decision"]["label"] for row in half) == Counter(yes=32, no=32)


def test_r039_freezes_both_rendered_option_orders_without_label_shortcuts():
    complete, partial, _ = build(train_sources=8, eval_sources=32, seed=42)
    for rows in (complete, partial):
        assert all(row["meta"]["fixed_option_order"] for row in rows)
        for source_rows in _by_source(rows).values():
            for yes_first in (False, True):
                labels = {row["targets"]["decision"]["label"] for row in source_rows
                          if row["meta"]["yes_first"] == yes_first}
                assert labels == {"yes", "no"}
    example = flatten_row(partial[0])[0]
    assert example["fixed_order"]
    first = encode(example, _Tokenizer(), 16, shuffle=True, include_labels=True)[2]
    second = encode(example, _Tokenizer(), 16, shuffle=True, include_labels=True)[2]
    assert first == second == [item[0] for item in example["items"]]
