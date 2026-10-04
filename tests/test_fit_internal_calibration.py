import pytest

from scripts.fit_internal_calibration import fit


def test_recovers_synthetic_temperature_and_improves_proper_scores():
    rows = []
    for raw in ([.98, .02], [.9, .1], [.75, .25], [.25, .75], [.1, .9], [.02, .98]):
        softened = [p ** .5 for p in raw]
        total = sum(softened)
        probs = {"true": softened[0] / total, "false": softened[1] / total}
        rows.append({"qtype": "choice", "probabilities": probs,
                     "target": {"true": raw[0], "false": raw[1]}})

    result = fit(rows)["choice"]
    assert result["temperature"] == pytest.approx(0.5, rel=1e-4)
    assert result["after"]["nll"] < result["before"]["nll"]
    assert result["after"]["brier"] < result["before"]["brier"]
    assert result["after"]["n"] == len(rows)
