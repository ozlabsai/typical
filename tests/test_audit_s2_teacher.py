import hashlib
import json

import pytest

from scripts.audit_s2_teacher import audit_results


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_inputs(tmp_path, missing_pair=False):
    panel, predictions = [], []
    for index in range(128):
        gold = "no" if index < 64 else "yes"
        source_id = f"src-{index:03d}"
        for order_index, order in enumerate((("no", "yes"), ("yes", "no"))):
            example_id = f"row-{index:03d}-o{order_index}"
            panel.append({
                "id": example_id, "state": "state",
                "questions": {"decision": {"type": "noul", "instructions": "A question?",
                                             "criteria": {key: key for key in order}}},
                "targets": {"decision": {"label": gold}},
                "meta": {"source_id": source_id, "source_group": f"group-{index // 2}",
                         "question_style": "interrogative", "criteria_style": "explicit",
                         "polarity": "positive", "context_view": "compact",
                         "distractor_groups": [], "rubric_style": "plain",
                         "arm_d_train_polarity": "positive" if index % 2 else "negative"},
            })
            p_yes = ((.15, .20)[order_index] if gold == "no" else
                     (.80, .85)[order_index])
            predicted = gold
            if index % 64 >= 60 and order_index == 1:
                predicted = "yes" if gold == "no" else "no"
            predictions.append({
                "example_id": example_id, "source_id": source_id,
                "source_group": f"group-{index // 2}", "option_order": list(order),
                "gold": gold, "predicted": predicted, "correct": predicted == gold,
                "p_yes": p_yes,
            })
    if missing_pair:
        panel[1]["meta"]["source_id"] = "src-001"
    panel_path = tmp_path / "panel.jsonl"
    write_jsonl(panel_path, panel)
    raw = panel_path.read_bytes()
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "data_sha256": hashlib.sha256(raw).hexdigest(), "complete": True,
        "subset": "all", "rows": 256, "results": [{"name": "teacher",
                                                          "predictions": predictions}],
    }))
    return panel_path, result_path


def test_locked_thresholds_and_coverage_are_inclusive(tmp_path):
    panel, result = make_inputs(tmp_path)
    report = audit_results(result, panel)
    assert report["pass"] is True
    assert report["eligible"] == 120
    assert report["gold_coverage"] == {
        "no": {"eligible": 60, "total": 64, "rate": 60 / 64},
        "yes": {"eligible": 60, "total": 64, "rate": 60 / 64},
    }
    assert report["cases"][0]["min_p_gold"] == pytest.approx(.8)
    assert report["cases"][0]["p_yes_delta"] == pytest.approx(.05)
    assert report["cases"][0]["eligible"] is True
    assert report["cases"][-1]["reasons"] == ["not_both_correct"]
    assert report["train_polarity_coverage"].keys() == {"negative", "positive"}
    assert report["rubric_style_coverage"] == {
        "plain": {"eligible": 120, "total": 128, "rate": 120 / 128}}


def test_rejects_incomplete_order_pair_and_wrong_panel_hash(tmp_path):
    panel, result = make_inputs(tmp_path, missing_pair=True)
    with pytest.raises(ValueError, match="128 two-order source pairs"):
        audit_results(result, panel)

    panel, result = make_inputs(tmp_path)
    record = json.loads(result.read_text())
    record["data_sha256"] = "wrong"
    result.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="data_sha256"):
        audit_results(result, panel)

    panel, result = make_inputs(tmp_path)
    record = json.loads(result.read_text())
    record["results"][0]["predictions"][1]["option_order"] = ["no", "yes"]
    result.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity/gold mismatch"):
        audit_results(result, panel)


def test_rejects_values_beyond_locked_probability_boundaries(tmp_path):
    panel, result = make_inputs(tmp_path)
    record = json.loads(result.read_text())
    _, yes_no = record["results"][0]["predictions"][:2]
    yes_no["p_yes"] = .251
    result.write_text(json.dumps(record))
    report = audit_results(result, panel)
    assert report["cases"][0]["eligible"] is False
    assert report["cases"][0]["reasons"] == [
        "min_p_gold_below_0.80", "p_yes_delta_above_0.05"]
