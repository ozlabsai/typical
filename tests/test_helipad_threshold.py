from scripts.helipad_threshold import cautious, fit_tau, split_cases

H = ["passes", "fails", "probably fails", "not checked"]
V = ["approved", "probably fails", "fails", "not checked"]


def _p(**probs):
    full = {c: 0.0 for c in H} | {k.replace("_", " "): v for k, v in probs.items()}
    return {"probs": full, "argmax": max(full, key=full.get)}


def _row(case, fam, gold, cands=H):
    return {"candidates": cands, "label": cands.index(gold), "meta": {"case": case, "family": fam, "core_missing": []}}


def test_cautious_flags_passes_with_enough_failing_mass_and_keeps_the_distribution():
    p = _p(passes=0.6, probably_fails=0.3, fails=0.1)
    out = cautious({"family": "low_50"}, p, tau=0.35)
    assert out["argmax"] == "probably fails" and out["probs"] == p["probs"]
    assert cautious({"family": "low_50"}, p, tau=0.45)["argmax"] == "passes"


def test_cautious_never_touches_verdict_rows_or_items_without_a_passes_option():
    p = {"probs": {"approved": 0.6, "probably fails": 0.4, "fails": 0, "not checked": 0}, "argmax": "approved"}
    assert cautious({"family": "verdict"}, p, tau=0.1) == p
    q = {"probs": {"not checked": 0.6, "probably fails": 0.4, "fails": 0}, "argmax": "not checked"}
    assert cautious({"family": "low_wires"}, q, tau=0.1) == q


def test_fit_picks_the_largest_tau_meeting_the_unsafe_target():
    # case a: gold probably fails, model says passes with 0.3 failing mass -> needs tau <= 0.3
    rows = [_row("a", "low_50", "probably fails"), _row("a", "verdict", "probably fails", V),
            _row("b", "low_50", "passes"), _row("b", "verdict", "approved", V)]
    preds = [_p(passes=0.7, probably_fails=0.3), {"probs": {v: float(v == "approved") for v in V}, "argmax": "approved"},
             _p(passes=0.9, probably_fails=0.1), {"probs": {v: float(v == "approved") for v in V}, "argmax": "approved"}]
    tau, rep = fit_tau(rows, preds, target=0.0, grid=[0.05, 0.1, 0.2, 0.3, 0.4])
    assert tau == 0.3 and rep["unsafe_record_rate"] == 0.0 and rep["false_alarm_rate"] == 0.0


def test_split_is_deterministic_and_disjoint_by_case():
    rows = [_row(f"c{i}", "low_50", "passes") for i in range(200)]
    a1, b1 = split_cases(rows)
    a2, _ = split_cases(rows)
    assert a1 == a2 and not {r["meta"]["case"] for r in a1} & {r["meta"]["case"] for r in b1} and 60 < len(a1) < 140
