import json

from scripts.audit_s2_r036 import audit
from tests.test_audit_s2_r035 import _crossed, _metrics


def _candidate_metrics():
    rows = _metrics(candidate=True)
    for row in rows:
        row["projection_cons_energy"] = 1.0
        row["projection_removed_energy"] = .2
    rows[-1]["validation"]["wf"] = {"accuracy": .85}
    rows[-1]["validation"]["replay"] = {"accuracy": .87}
    rows[-1]["projection_cumulative_removed_energy_fraction"] = .2
    return rows


def _baseline_metrics():
    rows = _metrics(False)
    rows[-1]["validation"]["wf"] = {"accuracy": .85}
    return rows


def _with_promotion_summaries(payload):
    rows = payload["results"][0]["predictions"]
    additions = []
    for row in rows:
        if row["question_style"] != "interrogative" or row["criteria_style"] != "explicit":
            continue
        for style, criteria in (("declarative", "explicit"), ("interrogative", "generic")):
            clone = {**row, "example_id": f'{row["example_id"]}-{style}-{criteria}',
                     "question_style": style, "criteria_style": criteria}
            if criteria == "generic":
                clone.update({"gold": "yes", "predicted": "yes", "p_yes": .95,
                              "polarity": "positive"})
            additions.append(clone)
    rows.extend(additions)
    payload["rows"] = len(rows)
    payload["results"][0]["order_robustness"]["both_correct_rate"] = 1.0
    return payload


def test_r036_full_gate_passes_a_retained_repair(tmp_path):
    paths = {name: tmp_path / name for name in
             ("base.json", "ref.json", "cand.json", "base.jsonl", "cand.jsonl")}
    base = _with_promotion_summaries(_crossed(False))
    candidate = _with_promotion_summaries(_crossed(True))
    for name, payload in (("base.json", base), ("ref.json", candidate),
                          ("cand.json", candidate)):
        paths[name].write_text(json.dumps(payload))
    paths["base.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in _baseline_metrics()))
    paths["cand.jsonl"].write_text(
        "".join(json.dumps(row) + "\n" for row in _candidate_metrics()))

    result = audit(paths["base.json"], paths["ref.json"], paths["cand.json"],
                   paths["base.jsonl"], paths["cand.jsonl"], expected_rows=8)
    assert result["promotion_pass"]
    assert all(result["gates"].values())


def test_r036_rejects_inactive_projection(tmp_path):
    paths = {name: tmp_path / name for name in
             ("base.json", "ref.json", "cand.json", "base.jsonl", "cand.jsonl")}
    base = _with_promotion_summaries(_crossed(False))
    candidate = _with_promotion_summaries(_crossed(True))
    for name, payload in (("base.json", base), ("ref.json", candidate),
                          ("cand.json", candidate)):
        paths[name].write_text(json.dumps(payload))
    metrics = _candidate_metrics()
    for row in metrics:
        row["projection_removed_energy"] = .009
    metrics[-1]["projection_cumulative_removed_energy_fraction"] = .009
    paths["base.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in _baseline_metrics()))
    paths["cand.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in metrics))

    try:
        audit(paths["base.json"], paths["ref.json"], paths["cand.json"],
              paths["base.jsonl"], paths["cand.jsonl"], expected_rows=8)
    except ValueError as error:
        assert "should have stopped at step 50" in str(error)
    else:
        raise AssertionError("inactive projection must fail before promotion scoring")


def test_r036_enforces_replay_and_format_floors(tmp_path):
    paths = {name: tmp_path / name for name in
             ("base.json", "ref.json", "cand.json", "base.jsonl", "cand.jsonl")}
    base = _with_promotion_summaries(_crossed(False))
    reference = _with_promotion_summaries(_crossed(True))
    candidate = _with_promotion_summaries(_crossed(True))
    for row in candidate["results"][0]["predictions"]:
        if row["question_style"] == "declarative" and row["criteria_style"] == "explicit":
            row.update({"correct": False, "predicted": "yes"})
    for name, payload in (("base.json", base), ("ref.json", reference),
                          ("cand.json", candidate)):
        paths[name].write_text(json.dumps(payload))
    metrics = _candidate_metrics()
    metrics[-1]["validation"]["replay"] = {"accuracy": .86}
    paths["base.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in _baseline_metrics()))
    paths["cand.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in metrics))

    result = audit(paths["base.json"], paths["ref.json"], paths["cand.json"],
                   paths["base.jsonl"], paths["cand.jsonl"], expected_rows=8)
    assert not result["gates"]["replay_accuracy_at_least_0.8628125"]
    assert not result["gates"]["each_format_retains_stronger_endpoint_minus_0.02"]
    assert not result["promotion_pass"]


def test_r036_rejects_incomplete_training_metrics(tmp_path):
    paths = {name: tmp_path / name for name in
             ("base.json", "ref.json", "cand.json", "base.jsonl", "cand.jsonl")}
    base = _with_promotion_summaries(_crossed(False))
    candidate = _with_promotion_summaries(_crossed(True))
    for name, payload in (("base.json", base), ("ref.json", candidate),
                          ("cand.json", candidate)):
        paths[name].write_text(json.dumps(payload))
    metrics = [row for row in _candidate_metrics() if row["step"] != 50]
    paths["base.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in _baseline_metrics()))
    paths["cand.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in metrics))

    try:
        audit(paths["base.json"], paths["ref.json"], paths["cand.json"],
              paths["base.jsonl"], paths["cand.jsonl"], expected_rows=8)
    except ValueError as error:
        assert "unique complete steps" in str(error)
    else:
        raise AssertionError("incomplete training metrics must fail closed")
