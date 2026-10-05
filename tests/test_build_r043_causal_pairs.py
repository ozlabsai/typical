import copy
import json
from collections import Counter

import pytest

from scripts.build_r043_causal_pairs import (
    EVAL_PAIRS,
    FAMILIES,
    REPLAY_ROWS,
    TRAIN_PAIRS,
    Quarantine,
    _build_split,
    _source,
    _validate_true_pair,
    build,
    execute,
    main,
    write_outputs,
)


@pytest.fixture(scope="module")
def generated():
    return build(seed=43)


def test_r043_true_pairs_are_balanced_executable_single_fact_counterfactuals(generated):
    train, _, evaluation, _ = generated
    assert len(train) == TRAIN_PAIRS
    assert len(evaluation) == EVAL_PAIRS
    assert Counter(row["meta"]["family"] for row in train) == {
        family: TRAIN_PAIRS // 8 for family in FAMILIES
    }
    train_sources = {row["meta"]["source_id"] for row in train}
    assert not train_sources & {row["meta"]["source_id"] for row in evaluation}
    presentations = []
    for pair in train:
        _validate_true_pair(pair)
        left, right = pair["paired_views"]
        assert pair["pair_mode"] == "counterfactual"
        assert pair["source_ids"][0] == pair["source_ids"][1]
        assert left["questions"] == right["questions"]
        assert sum(a != b for a, b in zip(left["meta"]["facts"], right["meta"]["facts"])) == 1
        for view in (left, right):
            meta = view["meta"]
            assert execute(meta["action_program"], meta["facts"]) == meta["action_gold"]
            assert meta["presentation_id"] == view["id"]
            assert len(meta["dependency_ids"]) == 1
        presentations.extend((left, right))
    assert Counter(view["targets"]["decision"]["label"] for view in presentations) == {
        "yes": TRAIN_PAIRS, "no": TRAIN_PAIRS
    }
    assert Counter(view["meta"]["polarity"] for view in presentations) == {
        "positive": TRAIN_PAIRS, "negative": TRAIN_PAIRS
    }
    assert Counter(view["meta"]["yes_first"] for view in presentations) == {
        False: TRAIN_PAIRS, True: TRAIN_PAIRS
    }


def test_r043_shuffled_control_is_within_family_source_derangement(generated):
    train, shuffled, _, _ = generated
    assert len(shuffled) == len(train) == TRAIN_PAIRS
    for pair in shuffled:
        assert pair["pair_mode"] == "shuffled_counterfactual"
        assert len(set(pair["source_ids"])) == 2
        assert {view["meta"]["family"] for view in pair["paired_views"]} == {pair["meta"]["family"]}
        assert {view["targets"]["decision"]["label"] for view in pair["paired_views"]} == {"yes", "no"}
        assert [view["meta"]["s2_arm"] for view in pair["paired_views"]] == ["A", "B"]


def test_r043_outputs_match_counts_hashes_and_frozen_replay(tmp_path, generated):
    train, shuffled, evaluation, rejections = generated
    replay = [{"id": f"replay-{index}", "state": f"frozen-{index}",
               "questions": {}, "targets": {}} for index in range(REPLAY_ROWS)]
    provenance = {"rejections": rejections, "inputs": {"replay": {"sha256": "input"}}}
    manifest = write_outputs(tmp_path, replay, train, shuffled, evaluation, provenance, 43)
    payload_a = (tmp_path / "arm-a.jsonl").read_bytes()
    payload_b = (tmp_path / "arm-b.jsonl").read_bytes()
    assert payload_a == payload_b
    arms = [[json.loads(line) for line in (tmp_path / f"arm-{name}.jsonl").read_text().splitlines()]
            for name in ("a", "b", "c")]
    assert all(len(arm) == 2800 for arm in arms)
    assert arms[0][:REPLAY_ROWS] == arms[1][:REPLAY_ROWS] == arms[2][:REPLAY_ROWS] == replay
    assert manifest["logical_rows"] == {"arm-a": 2800, "arm-b": 2800, "arm-c": 2800,
                                         "eval": EVAL_PAIRS}
    assert manifest["presentations"]["arm-a"] == 4000
    assert manifest["outputs"]["arm-a"]["sha256"] == manifest["outputs"]["arm-b"]["sha256"]
    with pytest.raises(ValueError, match="at least"):
        write_outputs(tmp_path, replay[:-1], train, shuffled, evaluation, provenance, 43)


def test_r043_quarantine_rejects_exact_and_near_duplicate_text():
    pair = _source("train", "invoice", 0, 43, 0)
    view = pair["paired_views"][0]
    exact = Quarantine([{"state": view["state"]}])
    assert exact.collision(view) == "exact"
    near = copy.deepcopy(view["state"]).replace("Evidence:", "Evidence record:")
    assert Quarantine([{"state": near}]).collision(view) == "trigram"
    unrelated = Quarantine([{"state": "A coral telescope observes silent moons."}])
    assert unrelated.collision(view) is None
    accepted, rejected = _build_split(8, "train", 43, 0, exact)
    assert len(accepted) == 8 and sum(rejected.values()) >= 1


def test_r043_quarantine_recognizes_nested_official_case_fields_and_rejects_empty_coverage():
    pair = _source("train", "security", 0, 43, 0)
    view = pair["paired_views"][0]
    nested = {"payload": {"case": {"input": view["state"]},
                          "request": {"instructions": "unrelated instruction"}}}
    assert Quarantine([nested]).collision(view) == "exact"
    with pytest.raises(ValueError, match="no recognized text fields"):
        Quarantine([{"id": "official-case-with-unsupported-shape", "answer": "yes"}])


def test_r043_cli_concatenates_repeatable_replay_inputs_in_order(tmp_path, monkeypatch, capsys):
    first = tmp_path / "typed.jsonl"
    second = tmp_path / "broad.jsonl"
    first_rows = [{"id": f"typed-{index}"} for index in range(1488)]
    second_rows = [{"id": f"broad-{index}"} for index in range(112)]
    first.write_text("".join(json.dumps(row) + "\n" for row in first_rows))
    second.write_text("".join(json.dumps(row) + "\n" for row in second_rows))
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["build_r043_causal_pairs.py", "--out", str(out),
                                     "--replay", str(first), "--replay", str(second)])
    main()
    capsys.readouterr()
    manifest = json.loads((out / "manifest.json").read_text())
    arm = [json.loads(line) for line in (out / "arm-a.jsonl").read_text().splitlines()]
    assert [entry["rows"] for entry in manifest["inputs"]["replay"]] == [1488, 112]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["inputs"]["replay"])
    assert [row["id"] for row in arm[:REPLAY_ROWS]] == [
        *(row["id"] for row in first_rows), *(row["id"] for row in second_rows)
    ]
    assert len(manifest["selected_replay_sha256"]) == 64
    empty_quarantine = tmp_path / "empty-quarantine.jsonl"
    empty_quarantine.write_text("")
    monkeypatch.setattr("sys.argv", ["build_r043_causal_pairs.py", "--out", str(out),
                                     "--replay", str(first), "--replay", str(second),
                                     "--quarantine", str(empty_quarantine)])
    with pytest.raises(ValueError, match="no recognized text fields"):
        main()
