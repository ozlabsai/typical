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
    _select_replay,
    _source,
    _stratum_key,
    _validate_true_pair,
    build,
    execute,
    main,
    write_outputs,
)


def _replay_row(identifier, option_count=2):
    criteria = {f"option-{index}": f"Option {index}" for index in range(option_count)}
    return {"id": identifier, "state": f"Replay state {identifier}",
            "questions": {"decision": {"type": "noul", "instructions": "Choose.",
                                         "criteria": criteria}},
            "targets": {"decision": {"label": "option-0"}}}


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
    true_presentations = Counter(view["meta"]["presentation_id"] for pair in train
                                 for view in pair["paired_views"])
    shuffled_presentations = Counter(view["meta"]["presentation_id"] for pair in shuffled
                                     for view in pair["paired_views"])
    true_yes_positions = {
        next(view["meta"]["presentation_id"] for view in pair["paired_views"]
             if view["targets"]["decision"]["label"] == "yes"):
        next(index for index, view in enumerate(pair["paired_views"])
             if view["targets"]["decision"]["label"] == "yes")
        for pair in train
    }
    for pair in shuffled:
        assert pair["pair_mode"] == "shuffled_counterfactual"
        assert len(set(pair["source_ids"])) == 2
        left, right = pair["paired_views"]
        assert _stratum_key(left) == _stratum_key(right)
        assert {left["meta"]["action_gold"], right["meta"]["action_gold"]} == {"execute", "withhold"}
        assert {left["targets"]["decision"]["label"], right["targets"]["decision"]["label"]} == {"yes", "no"}
        assert [left["meta"]["s2_arm"], right["meta"]["s2_arm"]] == ["A", "B"]
        yes_position = next(index for index, view in enumerate(pair["paired_views"])
                            if view["targets"]["decision"]["label"] == "yes")
        yes_id = pair["paired_views"][yes_position]["meta"]["presentation_id"]
        assert yes_position == true_yes_positions[yes_id]
    assert shuffled_presentations == true_presentations


def test_r043_outputs_match_counts_hashes_and_frozen_replay(tmp_path, generated):
    train, shuffled, evaluation, rejections = generated
    supported = [_replay_row(f"replay-{index}") for index in range(REPLAY_ROWS)]
    replay = [_replay_row("unsupported", 27), *supported]
    provenance = {"rejections": rejections, "inputs": {"replay": {"sha256": "input"}}}
    manifest = write_outputs(tmp_path, replay, train, shuffled, evaluation, provenance, 43)
    payload_a = (tmp_path / "arm-a.jsonl").read_bytes()
    payload_b = (tmp_path / "arm-b.jsonl").read_bytes()
    payload_c = (tmp_path / "arm-c.jsonl").read_bytes()
    payload_d = (tmp_path / "arm-d.jsonl").read_bytes()
    assert payload_a == payload_b
    assert payload_c == payload_d
    arms = [[json.loads(line) for line in (tmp_path / f"arm-{name}.jsonl").read_text().splitlines()]
            for name in ("a", "b", "c", "d")]
    assert all(len(arm) == 2800 for arm in arms)
    assert arms[0][:REPLAY_ROWS] == arms[1][:REPLAY_ROWS] == arms[2][:REPLAY_ROWS] == supported
    assert manifest["logical_rows"] == {"arm-a": 2800, "arm-b": 2800, "arm-c": 2800,
                                         "arm-d": 2800, "eval": EVAL_PAIRS}
    assert manifest["presentations"]["arm-a"] == 4000
    assert manifest["trainer_examples"]["arm-a"] == 2800
    assert manifest["trainer_examples"]["arm-d"] == 2800
    assert manifest["outputs"]["arm-a"]["sha256"] == manifest["outputs"]["arm-b"]["sha256"]
    assert manifest["outputs"]["arm-c"]["sha256"] == manifest["outputs"]["arm-d"]["sha256"]
    assert manifest["replay_selection"]["skipped"] == {"unsupported_option_count": 1}
    assert manifest["replay_selection"]["selected_positions"] == list(range(1, REPLAY_ROWS + 1))
    with pytest.raises(ValueError, match="supported frozen replay"):
        write_outputs(tmp_path, supported[:-1], train, shuffled, evaluation, provenance, 43)


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
    first_rows = [_replay_row(f"typed-{index}") for index in range(1488)]
    second_rows = [_replay_row(f"broad-{index}") for index in range(112)]
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


def test_r043_replay_selector_accepts_score_rows_and_rejects_multi_question_rows():
    score = {"id": "score", "state": "score state",
             "questions": {"decision": {"type": "score", "criteria": ["low", "high"]}},
             "targets": {"decision": {"label": 1}}}
    multi = _replay_row("multi")
    multi["questions"]["second"] = copy.deepcopy(multi["questions"]["decision"])
    multi["targets"]["second"] = {"label": "option-0"}
    rows = [multi, score, *(_replay_row(f"fill-{index}") for index in range(REPLAY_ROWS - 1))]
    selected, audit = _select_replay(rows)
    assert selected[0] == score and len(selected) == REPLAY_ROWS
    assert audit["skipped"] == {"question_count": 1}
