import json

from scripts.build_action_panel import SPLITS, action_for, build, main
from scripts.probe_action_renderer import render, select_approved, summarize as summarize_renderer
from scripts.score_action_panel import calibrate, calibration_records, limited_rows, summarize


def test_action_panel_is_deterministic_and_split_disjoint(tmp_path, monkeypatch):
    first, report = build(42, split="eval", cases_per_cell=3)
    second, _ = build(42, split="eval", cases_per_cell=3)
    assert first == second
    assert report["rows"] == 24
    for row in first:
        assert 2 <= len(row["questions"]) <= 6
        fields = {name: row["targets"][f"q{index:02d}"]["label"] == "yes"
                  for index, name in enumerate(row["meta"]["field_order"])}
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
    stored = [json.loads(line) for line in (tmp_path / "eval.jsonl").read_text().splitlines()]
    stored_orders = [next(iter(question["criteria"]))
                     for row in stored for question in row["questions"].values()]
    assert abs(stored_orders.count("yes") - stored_orders.count("no")) <= 1


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

    unmatched = dict(row, id="unmatched")
    records = calibration_records([row, unmatched], [{"id": row["id"], "probabilities": good}])
    recalibrated = calibrate([{"id": row["id"], "probabilities": good}], {
        "noul": {"temperature": 1.0, "logit_intercept_true": 0.0},
    })
    assert len(records) == len(row["questions"])
    for qid, values in good.items():
        for label, value in values.items():
            assert abs(recalibrated[0]["probabilities"][qid][label] - value) < 1e-12


def test_action_panel_limit_is_balanced_across_cells():
    rows, _ = build(42, split="eval", cases_per_cell=4)
    sample = limited_rows(rows, 16)
    cells = [(row["meta"]["family"], row["meta"]["template"], row["meta"]["threshold_band"])
             for row in sample]
    assert len(sample) == 16
    assert all(cells.count(cell) == 2 for cell in set(cells))


def test_action_panel_exercises_every_available_field_polarity():
    rows, _ = build(42, split="eval", cases_per_cell=100)
    labels = {name: set() for name in ("safe", "approved", "within_limit", "timely", "evidence", "covered")}
    for row in rows:
        for index, name in enumerate(row["meta"]["field_order"]):
            labels[name].add(row["targets"][f"q{index:02d}"]["label"])
    assert all(values == {"no", "yes"} for values in labels.values())


def test_renderer_probe_selects_paired_orders_without_changing_labels():
    rows, _ = build(42, split="eval", cases_per_cell=250)
    selected = select_approved(rows, 65)
    assert [row["targets"]["q01"]["label"] for row in selected].count("no") == 65
    assert [row["targets"]["q01"]["label"] for row in selected].count("yes") == 65
    no_first = render(selected[0], canonical=True, yes_first=False)
    yes_first = render(selected[0], canonical=True, yes_first=True)
    assert list(no_first["questions"]["q00"]["criteria"]) == ["no", "yes"]
    assert list(yes_first["questions"]["q00"]["criteria"]) == ["yes", "no"]
    assert no_first["targets"] == yes_first["targets"]
    report = summarize_renderer([
        {"id": "x", "label": "no", "renderer": "bare", "p_yes": .1, "correct": True},
        {"id": "x", "label": "no", "renderer": "bare", "p_yes": .2, "correct": True},
    ])
    assert report["bare_no"]["accuracy"] == 1.0
    assert report["bare_no_both_orders"] == 1.0
    assert report["bare_mean_order_delta"] == .1
