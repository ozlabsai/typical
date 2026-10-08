from scripts.r053_calibration_headroom import temper


def test_temper():
    # T=2 halves the logit: p=0.8 (logit ln4) -> sigmoid(ln2) = 2/3.
    assert abs(temper({"type": "noul", "noul": 0.8}, 2)["noul"] - 2 / 3) < 1e-9
    # Score: p^(1/T) renormalized; expectation and (pmax-1/K)/(1-1/K) confidence recomputed.
    s = temper({"type": "score", "probabilities": {"0": 0.2, "1": 0.8}}, 0.5)
    assert abs(s["probabilities"]["1"] - 16 / 17) < 1e-9
    assert abs(s["score"] - 16 / 17) < 1e-9 and abs(s["confidence"] - 15 / 17) < 1e-9
