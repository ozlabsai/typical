from collections import Counter

import pytest

from scripts.build_r040_orbits import _by_source, add_replay, build, rendered_signature
from scripts.train_gemma_decision_lora import encode, flatten_row


class _Tokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def __call__(self, *_args, **_kwargs):
        import torch
        return {"input_ids": torch.tensor([[1, 2]])}


def test_r040_arms_are_matched_full_vs_balanced_partial_factorials():
    complete, partial, evaluation = build(train_sources=8, eval_sources=32)
    assert len(complete) == len(partial) == 384
    assert len(evaluation) == 1536
    assert not (set(map(rendered_signature, complete)) & set(map(rendered_signature, evaluation)))
    assert all((left["meta"]["source_id"], left["meta"]["world"], left["meta"]["polarity"]) ==
               (right["meta"]["source_id"], right["meta"]["world"], right["meta"]["polarity"])
               for left, right in zip(complete, partial))
    for full, half in zip(_by_source(complete).values(), _by_source(partial).values()):
        full_surfaces = {(r["meta"]["wording"], r["meta"]["criteria_style"], r["meta"]["yes_first"])
                         for r in full}
        half_surfaces = {(r["meta"]["wording"], r["meta"]["criteria_style"], r["meta"]["yes_first"])
                         for r in half}
        assert len(full_surfaces) == 12
        assert len(half_surfaces) == 6
        assert Counter(r["targets"]["decision"]["label"] for r in full) == Counter(yes=24, no=24)
        assert Counter(r["targets"]["decision"]["label"] for r in half) == Counter(yes=24, no=24)
        for field in ("world", "polarity", "wording", "criteria_style", "yes_first"):
            assert all({r["targets"]["decision"]["label"] for r in full if r["meta"][field] == value} == {"yes", "no"}
                       for value in {r["meta"][field] for r in full})


def test_r040_fixes_option_order_for_both_arms():
    complete, partial, _ = build(train_sources=8, eval_sources=32)
    for rows in (complete, partial):
        assert all(row["meta"]["fixed_option_order"] for row in rows)
        example = flatten_row(rows[0])[0]
        assert encode(example, _Tokenizer(), 16, shuffle=True, include_labels=True)[2] == [
            item[0] for item in example["items"]]


def test_r040_excludes_the_approval_probe_predicate_and_action_panel_replay():
    complete, partial, _ = build(train_sources=8, eval_sources=32)
    rendered = "\n".join(row["state"] + row["questions"]["decision"]["instructions"] for row in complete)
    assert "approval" not in rendered.lower()
    with pytest.raises(ValueError, match="exclude"):
        add_replay([{"id": "probe", "meta": {"panel": "action_v1"}}], complete, partial)
