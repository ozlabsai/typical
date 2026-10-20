import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hejev_build as hb  # noqa: E402
import hejev_score as hs  # noqa: E402

ITEM = {"id": "x", "labels": ["route_to_billing", "other"], "expected": "route_to_billing",
        "state": {"ticket": "Refund 4471 of $1,200 to route_to_billing?", "status": "open", "n": 3,
                  "notes": ["Customer called twice.", "ID-77"]},
        "question": {"type": "choice", "instructions": "Where should it go?",
                     "criteria": {"route_to_billing": "Billing team", "other": "Anything else"}}}


def test_rewrite_roundtrip_keeps_ids_keys_and_labels():
    segs = hb.segments(ITEM)
    # single-token state values (enum/ID) and non-strings are not sent for translation; criteria keys never are
    assert list(segs.values()) == ["Refund 4471 of $1,200 to route_to_billing?", "Customer called twice.",
                                   "Where should it go?", "Billing team", "Anything else"]
    state, question = hb.rewrite(ITEM, lambda k, s: f"HE[{s}]")
    assert state["status"] == "open" and state["n"] == 3 and state["notes"][1] == "ID-77"
    assert state["notes"][0] == "HE[Customer called twice.]"
    assert list(question["criteria"]) == ["route_to_billing", "other"]
    assert hb.mirror_schema(segs)["required"] == list(segs)


def test_det_issues_catches_lost_digits_ids_and_untranslated_instructions():
    src = hb.segments(ITEM)
    good = {k: "בדיקה " + v for k, v in src.items()}
    assert hb.det_issues(src, good, ITEM["labels"]) == []
    bad = dict(good)
    k_ticket, k_instr = list(src)[0], list(src)[2]
    bad[k_ticket] = "החזר 4471 של $1,300 לחיוב"  # changed amount, dropped option ID
    bad[k_instr] = "Where should it go?"
    issues = hb.det_issues(src, bad, ITEM["labels"])
    assert any("digits" in i and "200" in i for i in issues)
    assert any("route_to_billing" in i for i in issues)
    assert any("instructions not translated" in i for i in issues)


def test_scorer_paired_metrics():
    assert hs.jsd({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}) == 0
    assert hs.jsd({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    tasks = {"en": {"t1": {"id": "t1", "expected": "yes", "labels": ["no", "yes"]},
                    "t2": {"id": "t2", "expected": "no", "labels": ["no", "yes"]}}}
    tasks["he"] = {"he-" + k: {**v, "id": "he-" + k} for k, v in tasks["en"].items()}
    row = lambda tid, p, ok: {"task_id": tid, "valid": True, "correct": ok,  # noqa: E731
                              "predicted": "yes" if p > .5 else "no", "probs": {"yes": p, "no": 1 - p}}
    results = {"en": {"t1": row("t1", .9, True), "t2": row("t2", .2, True)},
               "he": {"he-t1": row("he-t1", .9, True), "he-t2": row("he-t2", .7, False)}}
    rep = hs.score(results, tasks)
    assert rep["splits"]["en"]["accuracy"] == 1 and rep["splits"]["he"]["accuracy"] == .5
    assert rep["splits"]["en"]["brier"] == pytest.approx((2 * .1 ** 2 + 2 * .2 ** 2) / 2)
    p = rep["paired"]["he_vs_en"]["all"]
    assert p["n_pairs"] == 2 and p["acc_gap_other_minus_en"] == -.5 and p["prediction_agreement"] == .5
    assert p["flips_en_right_other_wrong"] == 1 and 0 < p["mean_jsd"] < 1


@pytest.mark.skipif(not (hb.OUT / "translated.jsonl").exists(), reason="benchmark not built")
def test_built_twins_keep_gold_and_load_in_runner():
    src = {t["id"]: t for t in hb.load_source()}
    for name, prefix in (("translated.jsonl", "he-"), ("mixed.jsonl", "mx-")):
        for row in map(json.loads, (hb.OUT / name).open()):
            hb.Task.from_dict(row)
            en = src[row["id"].removeprefix(prefix)]
            assert (row["labels"], row["expected"], row["question"]["type"]) == \
                   (en["labels"], en["expected"], en["question"]["type"])
            assert row["provenance"]["eval_only"] is True
