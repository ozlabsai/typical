import json

from scripts.intern_adapter import compile_files


def test_compile_grouped_hard_rows_and_audit_losses(tmp_path):
    path = tmp_path / "rows.jsonl"
    base = {"state": "מצב משותף", "p_null": 0.0, "task": "workflow"}
    rows = [
        {**base, "question": {"type": "choice", "instructions": "Choose", "criteria": {"b": "B", "a": "A"}},
         "candidates": ["a", "b"], "target": [1.0, 0.0], "label": 0,
         "meta": {"rubric_group": "g", "rubric_variant": 0}},
        {**base, "question": {"type": "choice", "instructions": "Choose again", "criteria": {"a": "A2", "b": "B2"}},
         "candidates": ["a", "b"], "target": [0.0, 1.0], "label": 1,
         "meta": {"rubric_group": "g", "rubric_variant": 1}},
        {**base, "state": "binary", "question": {"type": "noul", "instructions": "Is it true?",
                                                    "criteria": {"true": "Indeed", "false": "Nope"}},
         "candidates": ["true", "false"], "target": [1.0, 0.0], "label": 0, "meta": {}},
        {**base, "state": "rating", "question": {"type": "score", "instructions": "Rate",
                                                    "criteria": {"low": "Low", "high": "High"}},
         "candidates": ["low", "high"], "target": [0.0, 1.0], "label": 1, "meta": {}},
        {**base, "state": "null", "question": {"type": "choice", "instructions": "Pick",
                                                  "criteria": {"a": "A", "b": "B"}},
         "candidates": ["a", "b"], "target": [0.5, 0.5], "label": -1, "p_null": 1.0, "meta": {}},
        {**base, "state": "soft", "question": {"type": "choice", "instructions": "Pick",
                                                  "criteria": {"a": "A", "b": "B"}},
         "candidates": ["a", "b"], "target": [0.7, 0.3], "label": 0, "meta": {}},
        {**base, "state": "legacy", "query": "Which label?", "candidates": ["x", "y"],
         "target": [1.0, 0.0], "label": 0, "meta": {"family": "qa"}},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    records, skipped = compile_files([path])

    assert len(records) == 4
    assert list(records[0]["questions"]) == ["q000", "q001"]
    assert list(records[0]["questions"]["q000"]["criteria"]) == ["a", "b"]
    assert records[1]["questions"]["q000"]["criteria"] == {"no": "Nope", "yes": "Indeed"}
    assert records[1]["targets"]["q000"] == {"label": "yes"}
    assert records[2]["questions"]["q000"]["criteria"] == ["Low", "High"]
    assert records[2]["targets"]["q000"] == {"label": "1"}
    assert records[3]["questions"]["q000"]["criteria"] == {"x": "x", "y": "y"}
    assert skipped == {"null": 1, "soft_target": 1}
    assert json.loads(json.dumps(records, ensure_ascii=False))[0]["state"] == "מצב משותף"
