from scripts.r053_build_train_rows import train_row
from scripts.r053_generate_cases import contaminated, contamination_index, skeleton

BENCH = {"alert": "EDR-1234 | Suspicious scheduled task on host DC-NW-00 created by the nightly reporting script",
         "context": {"records": ["okta.user\n  login: b.farkas\n  status: ACTIVE"]}}


def test_contamination_filter():
    index = contamination_index([BENCH])
    copied = {"alert": "note: suspicious scheduled task on host dc nw 00 created by the nightly job"}
    assert contaminated(copied, index).startswith("ngram8:")
    assert contaminated({"alert": "Login from b.farkas at an odd hour"}, index) == "id:b.farkas"
    assert contaminated({"alert": "Host SRV-QX-17 ran an unsigned updater"}, index) is None
    # the generator's skeleton enumerates structural codes but never names
    skel = skeleton([{"workflow_name": "ridge_chat", "env": "prod"}] * 3)
    assert skel == {"workflow_name": "<free text ~10 chars>", "env": "<one of: prod>"}


def test_soft_target_order_matches_server():
    state = {"a": 1}
    label = {"case_id": "c", "node": "n", "request_id": "r", "qid": "q",
             "question": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is", "false": None}},
             "consensus": {"false": 0.3, "true": 0.7}, "teachers": {"t": {"false": 0.3, "true": 0.7}}}
    row = train_row("wf", label, state)
    assert row["questions"]["q"]["criteria"] == {"true": "Yes: yes it is", "false": "No"}
    assert row["soft_targets"]["q"] == [0.7, 0.3] and row["targets"]["q"]["label"] == "true"
    label["question"] = {"type": "score", "instructions": "How?", "criteria": ["low", "mid", "high"]}
    label["consensus"] = label["teachers"]["t"] = {"0": 0.1, "1": 0.2, "2": 0.7}
    row = train_row("wf", label, state)
    assert row["soft_targets"]["q"] == [0.1, 0.2, 0.7] and row["targets"]["q"]["label"] == "2"
