import json

import pytest

from scripts.audit_r042_representation import audit, _summary


def _rows():
    rows = []
    for source in range(64):
        for world, label in ((False, "no"), (True, "yes")):
            for yes_first in (False, True):
                rows.append({"id": f"{source}-{world}-{yes_first}", "meta": {
                    "source_id": str(source), "world": world, "polarity": "positive", "wording": "bare",
                    "criteria_style": "generic", "yes_first": yes_first},
                    "targets": {"decision": {"label": label}}})
    return rows


def _predictions(rows, failures=0):
    predictions = []
    for row in rows:
        failed = int(row["meta"]["source_id"]) < failures
        yes = row["targets"]["decision"]["label"] == "yes"
        predictions.append({"id": row["id"], "probabilities": {"yes": .1 if yes == failed else .9}})
    return predictions


def _report(rows, predictions):
    # Reuse the implementation's scorer so report/prediction consistency is exact.
    from scripts.audit_r042_representation import _aligned
    summary = _summary(_aligned(rows, predictions))
    return {"n": summary["n"], "accuracy": summary["accuracy"], "nll": summary["nll"],
            "brier": summary["brier"], "counterfactual_twins_both_correct": summary["twins"],
            "mean_semantic_order_delta": summary["order_delta"]}


def _write(path, value, *, lines=False):
    if lines:
        path.write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        path.write_text(json.dumps(value))


def _paths(tmp_path, rows, card, prose):
    paths = {}
    for name, value, lines in (("card-data", rows, True), ("prose-data", rows, True),
                               ("card-card-predictions", card, True), ("card-prose-predictions", card, True),
                               ("prose-card-predictions", prose, True), ("prose-prose-predictions", prose, True),
                               ("card-card-report", _report(rows, card), False),
                               ("card-prose-report", _report(rows, card), False),
                               ("prose-card-report", _report(rows, prose), False),
                               ("prose-prose-report", _report(rows, prose), False)):
        path = tmp_path / name
        _write(path, value, lines=lines)
        paths[name.replace("-", "_")] = path
    return paths


def test_r042_requires_a_paired_card_win_on_both_renderers(tmp_path):
    rows, card, prose = _rows(), _predictions(_rows()), _predictions(_rows(), failures=10)
    paths = _paths(tmp_path, rows, card, prose)
    result = audit(**paths, samples=300)
    assert result["card_renderer"]["targeted"]["paired_delta"] == pytest.approx(10 / 64)
    assert all(result["gates"].values())
    assert result["promotion_pass"]


def test_r042_rejects_misaligned_renderer_sources(tmp_path):
    rows, card, prose = _rows(), _predictions(_rows()), _predictions(_rows(), failures=10)
    paths = _paths(tmp_path, rows, card, prose)
    prose_rows = _rows()
    prose_rows[0]["meta"]["source_id"] = "other"
    _write(paths["prose_data"], prose_rows, lines=True)
    with pytest.raises(ValueError, match="source IDs or source coverage differ"):
        audit(**paths, samples=20)
