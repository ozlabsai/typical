from collections import Counter

import pytest

from scripts.build_r042_record_cards import _by_source, add_replay, build, rendered_signature


def test_r042_has_matched_48_view_card_and_prose_orbits_with_disjoint_transfer_vocab():
    card, prose, card_eval, prose_eval = build()
    assert len(card) == len(prose) == 44 * 48
    assert len(card_eval) == len(prose_eval) == 64 * 48
    assert not ({row["meta"]["predicate"] for row in card} & {row["meta"]["predicate"] for row in card_eval})
    assert not (set(map(rendered_signature, card + prose)) & set(map(rendered_signature, card_eval + prose_eval)))
    for card_rows, prose_rows in zip(_by_source(card).values(), _by_source(prose).values()):
        assert [row["state"] for row in card_rows] != [row["state"] for row in prose_rows]
        assert [row["questions"] for row in card_rows] == [row["questions"] for row in prose_rows]
        assert [row["targets"] for row in card_rows] == [row["targets"] for row in prose_rows]
        assert Counter(row["targets"]["decision"]["label"] for row in card_rows) == Counter(yes=24, no=24)
        assert Counter(row["meta"]["queried_position"] for row in card_rows) == Counter({0: 12, 1: 12, 2: 12, 3: 12})
        assert all(sum(fact["queried"] for fact in row["meta"]["facts"]) == 1
                   and len(row["meta"]["facts"]) == 4 for row in card_rows)


def test_r042_replay_is_identical_in_both_arms_and_rejects_forbidden_or_duplicate_rows():
    card, prose, _, _ = build()
    replay = [{"id": "replay", "state": "independent fact", "questions": {}, "targets": {}, "meta": {}}]
    card_with_replay, prose_with_replay = add_replay(replay, card, prose)
    assert card_with_replay[0] == prose_with_replay[0] == replay[0]
    assert len(card_with_replay) == len(prose_with_replay) == 1 + 44 * 48
    with pytest.raises(ValueError, match="exclude"):
        add_replay([{"id": "x", "meta": {"panel": "action_v1"}}], card, prose)
    with pytest.raises(ValueError, match="exclude"):
        add_replay([dict(card[0], id="copy")], card, prose)
    list_criteria = {"id": "typed-replay", "state": "independent", "questions": {
        "decision": {"criteria": ["a", "b"]}}, "targets": {}, "meta": {}}
    assert add_replay([list_criteria], card, prose)[0][0] == list_criteria
