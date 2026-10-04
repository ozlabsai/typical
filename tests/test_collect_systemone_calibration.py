import pytest

from scripts.collect_systemone_calibration import records_for


def test_records_preserve_soft_targets_and_noul_probability():
    row = {
        "id": "case",
        "state": "s",
        "questions": {
            "yn": {"type": "noul", "criteria": {"true": "yes", "false": "no"}},
            "pick": {"type": "choice", "criteria": {"left": "L", "right": "R"}},
        },
        "targets": {"yn": {"label": "true"}, "pick": {"label": "right"}},
        "soft_targets": {"pick": [0.25, 0.75]},
    }
    response = {"answers": {
        "yn": {"type": "noul", "noul": 0.8},
        "pick": {"type": "choice", "probabilities": {"left": 0.4, "right": 0.6}},
    }}
    yn, pick = records_for(row, response)
    assert yn["probabilities"] == pytest.approx({"true": 0.8, "false": 0.2})
    assert yn["target"] == "true"
    assert pick["target"] == {"left": 0.25, "right": 0.75}
