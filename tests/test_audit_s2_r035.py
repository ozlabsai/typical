import json
import math

from scripts.audit_s2_r035 import audit


def _prediction(index, style, criteria, order, correct, p_yes):
    return {
        "example_id": f"{style}-{criteria}-{order[0]}-{index}",
        "source_id": f"source-{index}", "source_group": f"group-{index}",
        "question_style": style, "criteria_style": criteria, "polarity": "negative",
        "context_view": "full", "placement": "middle", "option_order": order,
        "gold": "no", "predicted": "no" if correct else "yes", "p_yes": p_yes,
        "correct": correct, "nll": -math.log(1 - p_yes), "brier": p_yes ** 2,
    }


def _crossed(candidate=False):
    rows = []
    for index, order in enumerate((("no", "yes"), ("yes", "no"))):
        rows.append(_prediction(index, "interrogative", "explicit", order, True, .05))
        rows.append(_prediction(index, "declarative", "generic", order,
                                candidate, .10 if candidate else .80))
    nll = sum(row["nll"] for row in rows) / len(rows)
    brier = sum(row["brier"] for row in rows) / len(rows)
    return {"complete": True, "rows": 4, "data_sha256": "same", "results": [{
        "name": "candidate" if candidate else "baseline", "predictions": rows,
        "order_robustness": {"mean_abs_aligned_p_yes_delta": .03 if candidate else .04},
        "overall": {"n": 4, "accuracy": sum(r["correct"] for r in rows) / 4,
                    "nll": nll, "brier": brier},
    }]}


def _metrics(candidate=False):
    rows = []
    for step in range(1, 251):
        row = {"step": step}
        if candidate:
            row.update({"pair_consistency_active": 1, "pair_consistency_seen": 1})
        if step == 250:
            row["validation"] = {name: {"accuracy": .85} for name in ("wh", "replay")}
        rows.append(row)
    return rows


def test_preregistered_mechanism_gates_pass_on_clear_improvement(tmp_path):
    paths = {name: tmp_path / name for name in ("base.json", "cand.json", "base.jsonl", "cand.jsonl")}
    paths["base.json"].write_text(json.dumps(_crossed(False)))
    paths["cand.json"].write_text(json.dumps(_crossed(True)))
    paths["base.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in _metrics(False)))
    paths["cand.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in _metrics(True)))

    result = audit(paths["base.json"], paths["cand.json"],
                   paths["base.jsonl"], paths["cand.jsonl"], expected_rows=4)
    assert result["mechanism_pass"]
    assert all(result["gates"].values())
    assert result["active_pair_coverage_final_100_steps"]["rate"] == 1.0
