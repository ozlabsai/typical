from scripts.analyze_r046_ladder import summarize


def test_counterfactual_margin_summary():
    def view(row_id, label, probability):
        return {
            "id": f"{row_id}-{label}",
            "targets": {"decision": {"label": label}},
            "meta": {"source_id": "source", "scenario_id": "scenario", "language": "en",
                     "decisive_kind": "decimal", "renderer": "brief"},
            "probability": probability,
        }

    rows = [{"id": "pair", "paired_views": [view("pair", "no", .9), view("pair", "yes", .9)]}]
    predictions = [
        {"id": item["id"], "probabilities": {item["targets"]["decision"]["label"]: .9,
                                               "yes" if item["targets"]["decision"]["label"] == "no" else "no": .1}}
        for item in rows[0]["paired_views"]
    ]
    report = summarize(rows, predictions)
    assert report["pair_both_correct"] == 1
    assert report["direction_rate"] == 1
    assert report["margin_rate"] == 1
    assert report["mean_delta"] > 4
    assert report["mean_hinge"] == 0
    assert report["source_complete"] == 1
    assert report["scenario_complete"] == 1
