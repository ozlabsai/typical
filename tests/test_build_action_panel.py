import json

from scripts.build_action_panel import SPLITS, action_for, build, main
from scripts.score_action_panel import summarize


def test_action_panel_is_deterministic_and_split_disjoint(tmp_path, monkeypatch):
    first, report = build(42, split="eval", cases_per_cell=3)
    second, _ = build(42, split="eval", cases_per_cell=3)
    assert first == second
    assert report["rows"] == 24
    for row in first:
        assert 2 <= len(row["questions"]) <= 6
        fields = {name: row["targets"][f"q{index:02d}"]["label"] == "yes"
                  for index, name in enumerate(row["meta"]["action_trace"])}
        assert action_for(fields) == row["meta"]["action_gold"]
        assert row["meta"]["family"] in SPLITS["eval"]["families"]
        assert 0 < row["meta"]["decision_threshold"] < 1
    for axis in ("families", "templates", "bands"):
        for left in SPLITS.values():
            for right in SPLITS.values():
                if left is not right:
                    assert set(left[axis]).isdisjoint(right[axis])
    first_labels = [next(iter(question["criteria"]))
                    for row in first for question in row["questions"].values()]
    assert abs(first_labels.count("yes") - first_labels.count("no")) <= 1

    monkeypatch.setattr("sys.argv", ["build_action_panel.py", "--out", str(tmp_path),
                                      "--train-per-cell", "1", "--cal-per-cell", "1",
                                      "--eval-per-cell", "1"])
    main()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["splits"]["train"]["rows"] == 8


def test_action_report_separates_ranking_and_threshold_errors():
    rows, _ = build(42, split="eval", cases_per_cell=1)
    row = next(item for item in rows if len(item["questions"]) == 2)
    good = {qid: {"yes": float(target["label"] == "yes"),
                  "no": float(target["label"] == "no")}
            for qid, target in row["targets"].items()}
    report = summarize([row], [{"id": row["id"], "probabilities": good}])
    assert report["exact_action_accuracy"] == 1.0

    ranked_wrong = {qid: {"yes": 1 - values["yes"], "no": values["yes"]}
                    for qid, values in good.items()}
    report = summarize([row], [{"id": row["id"], "probabilities": ranked_wrong}])
    assert report["error_categories"]["ranking"] == 1
