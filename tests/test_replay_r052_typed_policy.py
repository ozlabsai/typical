import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from scripts import replay_r052_typed_policy as replay

ZERO = "0" * 64
ADJUDICATION_BYTES = b"independent pre-question adjudication receipt"
ADJUDICATION_SHA256 = replay.sha256_bytes(ADJUDICATION_BYTES)


def _hash_text(text: str) -> str:
    return replay.sha256_bytes(text.encode())


def _set_causal_proof(case, question, option_index=0):
    full = {"status": "resolved", "option_index": option_index}
    removed = {"status": "unresolved"}
    decisive = question["decisive_chunk_keys"]
    question["causal_proof"] = {
        "kind": "causal_annotation",
        "removed_chunk_keys": decisive,
        "full_context_sha256": replay._causal_context_sha256(case, question, set()),
        "removed_context_sha256": replay._causal_context_sha256(
            case, question, set(decisive)
        ),
        "full_outcome_sha256": replay.sha256_json(full),
        "removed_outcome_sha256": replay.sha256_json(removed),
        "full_outcome": full,
        "removed_outcome": removed,
        "changed_fields": ["status", "option_index"],
    }


def _fixture():
    policy_text = "Always apply the frozen policy."
    evidence_text = "The permit is active."
    distractor_text = "The office is open."
    chunks = [
        {
            "chunk_key": "c.policy",
            "kind": "global_policy",
            "modality": "text",
            "text": policy_text,
            "media": None,
            "immutable": True,
            "content_sha256": ZERO,
            "image_control": None,
            "token_start": 0,
            "token_end": 5,
            "derivation": {"kind": "non_source_global_policy"},
        },
        {
            "chunk_key": "c.decisive",
            "kind": "evidence",
            "modality": "text",
            "text": evidence_text,
            "media": None,
            "immutable": False,
            "content_sha256": ZERO,
            "image_control": None,
            "token_start": 5,
            "token_end": 10,
            "derivation": {"kind": "source_text_slice"},
        },
        {
            "chunk_key": "c.distractor",
            "kind": "distractor",
            "modality": "text",
            "text": distractor_text,
            "media": None,
            "immutable": False,
            "content_sha256": ZERO,
            "image_control": None,
            "token_start": 10,
            "token_end": 15,
            "derivation": {"kind": "source_text_slice"},
        },
    ]
    for chunk in chunks:
        chunk["content_sha256"] = replay.sha256_json(
            {
                "kind": chunk["kind"],
                "modality": chunk["modality"],
                "text": chunk["text"],
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "media": None,
                "derivation": chunk["derivation"],
            }
        )
    question = {
        "question_key": "q.permit",
        "text": "Is the permit active?",
        "options": ["Yes", "No"],
        "option_ids": ["yes", "no"],
        "policy_role": "permit",
        "content_sha256": ZERO,
        "decision_type": "noul",
        "twin_anchor": "permit",
        "decisive_chunk_keys": ["c.decisive"],
        "known_distractor_chunk_keys": ["c.distractor"],
    }
    question["content_sha256"] = replay.sha256_json(
        {
            key: question[key]
            for key in (
                "text",
                "options",
                "option_ids",
                "decision_type",
                "policy_role",
            )
        }
    )
    panel = {
        "schema_version": "test-panel",
        "cases": [
            {
                "case_key": "case.en",
                "language": "en",
                "twin_group_key": None,
                "chunks": chunks,
                "questions": [question],
            }
        ],
    }
    panel_hash = replay.sha256_json(panel)
    ledger = {
        "schema_version": replay.LEDGER_VERSION,
        "panel_sha256": panel_hash,
        "facts": {"f0001": {"type": "boolean", "unit": None}},
        "bindings": [
            {
                "case_key": "case.en",
                "chunk_key": "c.decisive",
                "chunk_content_sha256": chunks[1]["content_sha256"],
                "observations": [
                    {
                        "fact": "f0001",
                        "value": True,
                        "locator": {
                            "kind": "text_span",
                            "char_start": 0,
                            "char_end": len(evidence_text),
                            "span_sha256": _hash_text(evidence_text),
                        },
                    }
                ],
                "adjudication_receipt_sha256": ADJUDICATION_SHA256,
            }
        ],
    }
    ledger_hash = replay.sha256_json(ledger)
    fact = {"op": "fact", "id": "f0001"}
    yes = {"op": "const", "type": "boolean", "unit": None, "value": True}
    no = {"op": "const", "type": "boolean", "unit": None, "value": False}
    rubric = {
        "schema_version": replay.RUBRIC_VERSION,
        "panel_sha256": panel_hash,
        "fact_ledger_sha256": ledger_hash,
        "semantic_groups": [
            {
                "group_id": "g.permit",
                "facts": {"f0001": {"type": "boolean", "unit": None}},
                "decisions": [
                    {
                        "decision_id": "d.permit",
                        "type": "noul",
                        "outputs": [
                            {"id": "yes", "label": "Yes"},
                            {"id": "no", "label": "No"},
                        ],
                        "order": ["yes", "no"],
                        "rules": [
                            {
                                "output": "yes",
                                "when": {"op": "eq", "left": fact, "right": yes},
                            },
                            {
                                "output": "no",
                                "when": {"op": "eq", "left": fact, "right": no},
                            },
                        ],
                    }
                ],
                "presentations": [
                    {
                        "presentation_id": "p.en",
                        "case_key": "case.en",
                        "question_bindings": {
                            "d.permit": {
                                "question_key": "q.permit",
                                "question_content_sha256": question["content_sha256"],
                            }
                        },
                        "views": {
                            "full": ["c.policy", "c.decisive", "c.distractor"],
                            "removed": ["c.policy", "c.distractor"],
                            "decisive_only": ["c.policy", "c.decisive"],
                            "distractor_only": ["c.policy", "c.distractor"],
                        },
                        "ablations": [
                            {
                                "decision_id": "d.permit",
                                "remove_chunk_keys": ["c.decisive"],
                                "required_transition": "resolution_loss",
                                "assert_full": {"status": "resolved", "output": "yes"},
                                "assert_removed": {"status": "unresolved"},
                            }
                        ],
                    }
                ],
            }
        ],
    }
    _set_causal_proof(panel["cases"][0], question)
    return panel, ledger, rubric


def _seal(panel, ledger, rubric):
    panel_hash = replay.sha256_json(panel)
    ledger["panel_sha256"] = panel_hash
    ledger_hash = replay.sha256_json(ledger)
    rubric["panel_sha256"] = panel_hash
    rubric["fact_ledger_sha256"] = ledger_hash
    rubric_hash = replay.sha256_json(rubric)
    return {
        "panel_sha256": panel_hash,
        "ledger_sha256": ledger_hash,
        "rubric_sha256": rubric_hash,
        "evaluator_sha256": "e" * 64,
        "schema_sha256": replay.EXPECTED_SCHEMA_SHA256,
    }


def _run(values=None):
    panel, ledger, rubric = values or _fixture()
    return replay.replay(
        panel,
        ledger,
        rubric,
        **_seal(panel, ledger, rubric),
        adjudication_receipts={ADJUDICATION_SHA256: ADJUDICATION_BYTES},
    )


def test_happy_path_receipt_is_deterministic_and_missing_noul_is_never_false():
    values = _fixture()
    first = _run(deepcopy(values))
    second = _run(deepcopy(values))
    assert first == second
    trace = first["semantic_groups"][0]["presentations"][0]["decisions"][0]
    assert trace["transition"] == "resolution_loss"
    assert trace["views"]["full"] | {"rules": []} == {
        "status": "resolved",
        "output_id": "yes",
        "option_index": 0,
        "text": "Yes",
        "question_key": "q.permit",
        "question_content_sha256": values[0]["cases"][0]["questions"][0][
            "content_sha256"
        ],
        "rules": [],
    }
    assert trace["views"]["removed"]["status"] == "unresolved"
    assert [item["truth"] for item in trace["views"]["removed"]["rules"]] == [
        "missing",
        "missing",
    ]
    digest = first.pop("receipt_sha256")
    assert digest == replay.sha256_json(first)


def test_unknown_competing_choice_and_multiple_true_fail_closed():
    panel, ledger, rubric = _fixture()
    decision = rubric["semantic_groups"][0]["decisions"][0]
    decision["type"] = "choice"
    question = panel["cases"][0]["questions"][0]
    question["decision_type"] = "choice"
    question["options"].append("Maybe")
    question["option_ids"].append("maybe")
    question["content_sha256"] = replay.sha256_json(
        {
            key: question[key]
            for key in (
                "text",
                "options",
                "option_ids",
                "decision_type",
                "policy_role",
            )
        }
    )
    rubric["semantic_groups"][0]["presentations"][0]["question_bindings"]["d.permit"][
        "question_content_sha256"
    ] = question["content_sha256"]
    _set_causal_proof(panel["cases"][0], question)
    decision["outputs"].append({"id": "maybe", "label": "Maybe"})
    decision["order"].append("maybe")
    decision["rules"].append({"output": "maybe", "when": {"op": "fact", "id": "f0002"}})
    rubric["semantic_groups"][0]["facts"]["f0002"] = {
        "type": "boolean",
        "unit": None,
    }
    ledger["facts"]["f0002"] = {"type": "boolean", "unit": None}
    ledger["bindings"][0]["observations"].append(
        {
            "fact": "f0002",
            "value": False,
            "locator": {
                "kind": "text_span",
                "char_start": 0,
                "char_end": 3,
                "span_sha256": _hash_text("The"),
            },
        }
    )
    # Move the competing fact to a distractor, so it is unknown in decisive-only.
    competing = ledger["bindings"][0]["observations"].pop()
    ledger["bindings"].append(
        {
            "case_key": "case.en",
            "chunk_key": "c.distractor",
            "chunk_content_sha256": panel["cases"][0]["chunks"][2]["content_sha256"],
            "observations": [competing],
            "adjudication_receipt_sha256": ADJUDICATION_SHA256,
        }
    )
    with pytest.raises(replay.PolicyError, match="decisive-only"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    decision = rubric["semantic_groups"][0]["decisions"][0]
    decision["rules"][1]["when"] = deepcopy(decision["rules"][0]["when"])
    with pytest.raises(replay.PolicyError, match="assert_full"):
        _run((panel, ledger, rubric))


def test_duplicate_support_conflict_and_nonminimal_removal_are_rejected():
    panel, ledger, rubric = _fixture()
    extra = deepcopy(ledger["bindings"][0])
    extra["chunk_key"] = "c.distractor"
    extra["chunk_content_sha256"] = panel["cases"][0]["chunks"][2]["content_sha256"]
    extra["observations"][0]["locator"] = {
        "kind": "text_span",
        "char_start": 0,
        "char_end": 3,
        "span_sha256": _hash_text("The"),
    }
    ledger["bindings"].append(extra)
    with pytest.raises(replay.PolicyError, match="assert_removed|ablation|distractor"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    extra = deepcopy(ledger["bindings"][0])
    extra["chunk_key"] = "c.distractor"
    extra["chunk_content_sha256"] = panel["cases"][0]["chunks"][2]["content_sha256"]
    extra["observations"][0]["value"] = False
    extra["observations"][0]["locator"] = {
        "kind": "text_span",
        "char_start": 0,
        "char_end": 3,
        "span_sha256": _hash_text("The"),
    }
    ledger["bindings"].append(extra)
    with pytest.raises(replay.PolicyError, match="assert_full"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    question = panel["cases"][0]["questions"][0]
    question["decisive_chunk_keys"].append("c.distractor")
    question["known_distractor_chunk_keys"] = []
    presentation = rubric["semantic_groups"][0]["presentations"][0]
    presentation["views"]["removed"] = ["c.policy"]
    presentation["views"]["decisive_only"] = ["c.policy", "c.decisive", "c.distractor"]
    presentation["views"]["distractor_only"] = ["c.policy"]
    presentation["ablations"][0]["remove_chunk_keys"].append("c.distractor")
    _set_causal_proof(panel["cases"][0], question)
    with pytest.raises(replay.PolicyError, match="inclusion-minimal"):
        _run((panel, ledger, rubric))


def test_exact_fraction_boundaries_count_true_and_division_by_zero():
    env = {
        "n": replay.TypedValue("decimal", "kg", Fraction("0.3")),
        "d": replay.TypedValue("decimal", "kg", Fraction("0.1")),
    }
    ratio = {
        "op": "ratio",
        "numerator": {"op": "fact", "id": "n"},
        "denominator": {"op": "fact", "id": "d"},
    }
    boundary = {
        "op": "eq",
        "left": ratio,
        "right": {"op": "const", "type": "integer", "unit": None, "value": 3},
    }
    assert replay._eval(boundary, env) is True
    env["d"] = replay.TypedValue("decimal", "kg", Fraction(0))
    assert replay._eval(ratio, env) is replay.Signal.CONFLICT
    count = {
        "op": "count_true",
        "args": [
            {"op": "const", "type": "boolean", "unit": None, "value": True},
            {"op": "const", "type": "boolean", "unit": None, "value": False},
        ],
    }
    assert replay._eval(count, {}).value == 1


@pytest.mark.parametrize("decision_type", ["choice", "noul", "score"])
def test_all_typed_decisions_use_exact_single_winner_resolution(decision_type):
    fact = {"op": "fact", "id": "x"}
    zero = {"op": "const", "type": "integer", "unit": None, "value": 0}
    outputs = ("yes", "no") if decision_type == "noul" else ("left", "right")
    decision = {
        "type": decision_type,
        "rules": [
            {"output": outputs[0], "when": {"op": "le", "left": fact, "right": zero}},
            {"output": outputs[1], "when": {"op": "gt", "left": fact, "right": zero}},
        ],
    }
    known = replay._resolve(
        decision, {"x": replay.TypedValue("integer", None, Fraction(0))}
    )
    assert (known.status, known.output) == ("resolved", outputs[0])
    assert (
        replay._resolve(decision, {"x": replay.Signal.MISSING}).status == "unresolved"
    )


def test_rule_and_binding_permutations_do_not_change_semantics():
    original = _run()
    panel, ledger, rubric = _fixture()
    rubric["semantic_groups"][0]["decisions"][0]["rules"].reverse()
    permuted = _run((panel, ledger, rubric))
    left = original["semantic_groups"][0]["presentations"][0]["decisions"][0]
    right = permuted["semantic_groups"][0]["presentations"][0]["decisions"][0]
    assert left["transition"] == right["transition"]
    assert {
        item["output"]: item["truth"] for item in left["views"]["full"]["rules"]
    } == {item["output"]: item["truth"] for item in right["views"]["full"]["rules"]}


def test_question_hash_tamper_hebrew_drift_image_leakage_and_policy_removal():
    panel, ledger, rubric = _fixture()
    panel["cases"][0]["questions"][0]["text"] = "Tampered question"
    with pytest.raises(replay.PolicyError, match="question content_sha256"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    presentation = rubric["semantic_groups"][0]["presentations"][0]
    presentation["views"]["removed"] = ["c.distractor"]
    with pytest.raises(replay.PolicyError, match="omits policy"):
        _run((panel, ledger, rubric))

    cases = {
        "he": {
            "language": "he",
            "twin_group_key": "t",
            "chunks": [
                {
                    "chunk_key": "c.he",
                    "twin_anchor": "c",
                    "kind": "evidence",
                    "modality": "text",
                    "immutable": False,
                    "image_control": None,
                }
            ],
            "questions": [
                {
                    "question_key": "q",
                    "twin_anchor": "a",
                    "decision_type": "noul",
                    "policy_role": "p",
                    "options": ["כן", "לא"],
                    "option_ids": ["yes", "no"],
                }
            ],
        },
        "en": {
            "language": "en",
            "twin_group_key": "t",
            "chunks": [
                {
                    "chunk_key": "c.en",
                    "twin_anchor": "c",
                    "kind": "evidence",
                    "modality": "text",
                    "immutable": False,
                    "image_control": None,
                }
            ],
            "questions": [
                {
                    "question_key": "q",
                    "twin_anchor": "drift",
                    "decision_type": "noul",
                    "policy_role": "p",
                    "options": ["yes", "no"],
                    "option_ids": ["yes", "no"],
                }
            ],
        },
    }
    groups = [
        {
            "group_id": "g",
            "presentations": [
                {
                    "presentation_id": "p.he",
                    "case_key": "he",
                    "views": {
                        name: ["c.he"]
                        for name in (
                            "full",
                            "removed",
                            "decisive_only",
                            "distractor_only",
                        )
                    },
                    "question_bindings": {
                        "d": {
                            "question_key": "q",
                        }
                    },
                },
                {
                    "presentation_id": "p.en",
                    "case_key": "en",
                    "views": {
                        name: ["c.en"]
                        for name in (
                            "full",
                            "removed",
                            "decisive_only",
                            "distractor_only",
                        )
                    },
                    "question_bindings": {
                        "d": {
                            "question_key": "q",
                        }
                    },
                },
            ],
        }
    ]
    traces = {
        ("g", "p.he"): {"fact_states": {}, "decisions": []},
        ("g", "p.en"): {"fact_states": {}, "decisions": []},
    }
    with pytest.raises(replay.PolicyError, match="semantic anchors drift"):
        replay._validate_hebrew(groups, cases, traces)

    case = {
        "chunks": [
            {
                "chunk_key": "real",
                "text": "neutral",
                "media": {"sha256": "1" * 64},
                "image_control": {"group_key": "i", "variant": "real"},
            },
            {
                "chunk_key": "shuffle",
                "text": "neutral",
                "media": {"sha256": "2" * 64},
                "image_control": {"group_key": "i", "variant": "shuffled"},
            },
            {
                "chunk_key": "missing",
                "text": "neutral",
                "media": None,
                "image_control": {"group_key": "i", "variant": "missing"},
            },
        ]
    }
    with pytest.raises(replay.PolicyError, match="must not bind"):
        replay._validate_images({"case": case}, [{"chunk_key": "shuffle"}])


def test_duplicate_nonfinite_noncanonical_extra_keys_and_depth_are_rejected():
    with pytest.raises(replay.PolicyError, match="duplicate JSON key"):
        replay.parse_canonical_json(b'{"a":1,"a":2}\n', "test")
    with pytest.raises(replay.PolicyError, match="non-finite"):
        replay.parse_canonical_json(b'{"a":NaN}\n', "test")
    with pytest.raises(replay.PolicyError, match="canonical"):
        replay.parse_canonical_json(b'{ "a": 1 }\n', "test")
    assert replay.parse_canonical_json(replay.canonical_json({"a": 0.1}), "test") == {
        "a": 0.1
    }

    panel, ledger, rubric = _fixture()
    ledger["extra"] = None
    with pytest.raises(replay.PolicyError, match="keys must be exactly"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    node = {"op": "fact", "id": "f0001"}
    for _ in range(replay.MAX_AST_DEPTH):
        node = {"op": "not", "arg": node}
    rubric["semantic_groups"][0]["decisions"][0]["rules"][0]["when"] = node
    with pytest.raises(replay.PolicyError, match="maximum depth"):
        _run((panel, ledger, rubric))


def test_fact_ids_are_question_blind_and_text_spans_are_exact():
    panel, ledger, rubric = _fixture()
    ledger["facts"]["correct_answer"] = ledger["facts"].pop("f0001")
    with pytest.raises(replay.PolicyError, match=r"f\[0-9\]"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    ledger["bindings"][0]["observations"][0]["locator"]["span_sha256"] = ZERO
    with pytest.raises(replay.PolicyError, match="text span hash mismatch"):
        _run((panel, ledger, rubric))


def test_question_coverage_is_an_exact_bijection():
    panel, ledger, rubric = _fixture()
    extra = deepcopy(panel["cases"][0]["questions"][0])
    extra["question_key"] = "q.unbound"
    extra["twin_anchor"] = "permit-extra"
    panel["cases"][0]["questions"].append(extra)
    with pytest.raises(replay.PolicyError, match="every panel question exactly once"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    duplicate = deepcopy(rubric["semantic_groups"][0]["presentations"][0])
    duplicate["presentation_id"] = "p.duplicate"
    rubric["semantic_groups"][0]["presentations"].append(duplicate)
    with pytest.raises(replay.PolicyError, match="reused"):
        _run((panel, ledger, rubric))


def test_option_ids_causal_proof_and_resolution_loss_are_authoritative():
    panel, ledger, rubric = _fixture()
    binding = rubric["semantic_groups"][0]["presentations"][0]["question_bindings"][
        "d.permit"
    ]
    binding["output_option_indices"] = {"yes": 1, "no": 0}
    with pytest.raises(replay.PolicyError, match="keys must be exactly"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    question = panel["cases"][0]["questions"][0]
    question["option_ids"] = ["no", "yes"]
    question["content_sha256"] = replay.sha256_json(
        {
            key: question[key]
            for key in (
                "text",
                "options",
                "option_ids",
                "decision_type",
                "policy_role",
            )
        }
    )
    rubric["semantic_groups"][0]["presentations"][0]["question_bindings"]["d.permit"][
        "question_content_sha256"
    ] = question["content_sha256"]
    _set_causal_proof(panel["cases"][0], question)
    with pytest.raises(replay.PolicyError, match="exactly bind panel option_ids"):
        _run((panel, ledger, rubric))

    decision = rubric["semantic_groups"][0]["decisions"][0]
    question["options"] = ["No", "Yes"]
    decision["outputs"].reverse()
    decision["order"].reverse()
    decision["rules"].reverse()
    question["content_sha256"] = replay.sha256_json(
        {
            key: question[key]
            for key in (
                "text",
                "options",
                "option_ids",
                "decision_type",
                "policy_role",
            )
        }
    )
    rubric["semantic_groups"][0]["presentations"][0]["question_bindings"]["d.permit"][
        "question_content_sha256"
    ] = question["content_sha256"]
    _set_causal_proof(panel["cases"][0], question, option_index=1)
    swapped = _run((panel, ledger, rubric))
    selected = swapped["semantic_groups"][0]["presentations"][0]["decisions"][0][
        "views"
    ]["full"]
    assert (selected["output_id"], selected["option_index"], selected["text"]) == (
        "yes",
        1,
        "Yes",
    )

    panel, ledger, rubric = _fixture()
    question = panel["cases"][0]["questions"][0]
    question["causal_proof"]["full_outcome"] = {
        "status": "resolved",
        "option_index": 1,
    }
    question["causal_proof"]["full_outcome_sha256"] = replay.sha256_json(
        question["causal_proof"]["full_outcome"]
    )
    with pytest.raises(replay.PolicyError, match="contradicts authoritative replay"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    rubric["semantic_groups"][0]["presentations"][0]["ablations"][0][
        "required_transition"
    ] = "label_flip"
    with pytest.raises(replay.PolicyError, match="required causal transition"):
        _run((panel, ledger, rubric))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("options", None, "options must be"),
        ("options", {"yes": "Yes"}, "options must be"),
        ("options", ["Yes", "Yes"], "options must be"),
        ("option_ids", None, "option_ids must be"),
        ("option_ids", {"yes": 0, "no": 1}, "option_ids must be"),
        ("option_ids", ["yes", "yes"], "option_ids must be"),
    ],
)
def test_malformed_panel_options_fail_as_policy_errors(field, value, message):
    panel, ledger, rubric = _fixture()
    panel["cases"][0]["questions"][0][field] = value
    with pytest.raises(replay.PolicyError, match=message):
        _run((panel, ledger, rubric))


def test_receipt_artifacts_immutability_and_policy_binding_fail_closed():
    panel, ledger, rubric = _fixture()
    hashes = _seal(panel, ledger, rubric)
    with pytest.raises(replay.PolicyError, match="no supplied artifact"):
        replay.replay(
            panel,
            ledger,
            rubric,
            **hashes,
            adjudication_receipts={},
        )
    with pytest.raises(replay.PolicyError, match="bytes/hash"):
        replay.replay(
            panel,
            ledger,
            rubric,
            **hashes,
            adjudication_receipts={ADJUDICATION_SHA256: b"tampered"},
        )

    panel, ledger, rubric = _fixture()
    panel["cases"][0]["chunks"][2]["immutable"] = True
    with pytest.raises(replay.PolicyError, match="true iff"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    policy = panel["cases"][0]["chunks"][0]
    binding = ledger["bindings"][0]
    binding["chunk_key"] = policy["chunk_key"]
    binding["chunk_content_sha256"] = policy["content_sha256"]
    with pytest.raises(replay.PolicyError, match="policy chunks cannot bind"):
        _run((panel, ledger, rubric))


def _image_control_case():
    image_chunks = [
        {
            "chunk_key": "real",
            "text": "neutral",
            "modality": "image",
            "media": {"sha256": "1" * 64},
            "image_control": {"group_key": "i", "variant": "real"},
        },
        {
            "chunk_key": "shuffle",
            "text": "neutral",
            "modality": "image",
            "media": {"sha256": "2" * 64},
            "image_control": {"group_key": "i", "variant": "shuffled"},
        },
        {
            "chunk_key": "missing",
            "text": "neutral",
            "modality": "image",
            "media": None,
            "image_control": {"group_key": "i", "variant": "missing"},
        },
    ]
    text_chunks = [
        {
            "chunk_key": f"t{index}",
            "text": f"distractor {index}",
            "modality": "text",
            "media": None,
            "image_control": None,
        }
        for index in range(5)
    ]
    return {
        "chunks": image_chunks + text_chunks,
        "questions": [
            {
                "decisive_chunk_keys": ["real"],
                "known_distractor_chunk_keys": [
                    "shuffle",
                    "missing",
                    *[f"t{index}" for index in range(5)],
                ],
            }
        ],
    }


def test_image_controls_enforce_distinct_media_single_question_and_text_controls():
    case = _image_control_case()
    replay._validate_images({"case": case}, [])

    same_media = deepcopy(case)
    same_media["chunks"][1]["media"]["sha256"] = "1" * 64
    with pytest.raises(replay.PolicyError, match="distinct media"):
        replay._validate_images({"case": same_media}, [])

    duplicate_question = deepcopy(case)
    duplicate_question["questions"].append(deepcopy(case["questions"][0]))
    with pytest.raises(replay.PolicyError, match="exactly one causal question"):
        replay._validate_images({"case": duplicate_question}, [])

    too_few = deepcopy(case)
    too_few["questions"][0]["known_distractor_chunk_keys"] = [
        "shuffle",
        "missing",
        "t0",
    ]
    with pytest.raises(replay.PolicyError, match="five text distractors"):
        replay._validate_images({"case": too_few}, [])


def test_hebrew_twins_compare_option_evidence_environment_and_replay_traces():
    def twin_case(language, suffix):
        return {
            "language": language,
            "twin_group_key": "twin",
            "chunks": [
                {
                    "chunk_key": f"c.{suffix}",
                    "twin_anchor": "evidence",
                    "kind": "evidence",
                    "modality": "text",
                    "immutable": False,
                    "image_control": None,
                }
            ],
            "questions": [
                {
                    "question_key": f"q.{suffix}",
                    "twin_anchor": "question",
                    "decision_type": "noul",
                    "policy_role": "role",
                    "options": ["yes", "no"],
                    "option_ids": ["yes", "no"],
                    "decisive_chunk_keys": [f"c.{suffix}"],
                    "known_distractor_chunk_keys": [],
                }
            ],
        }

    cases = {"he": twin_case("he", "he"), "en": twin_case("en", "en")}
    cases["he"]["questions"][0]["options"] = ["כן", "לא"]
    views = {
        name: ["c.he"]
        for name in ("full", "removed", "decisive_only", "distractor_only")
    }
    presentations = [
        {
            "presentation_id": "p.he",
            "case_key": "he",
            "views": views,
            "question_bindings": {
                "d": {
                    "question_key": "q.he",
                }
            },
        },
        {
            "presentation_id": "p.en",
            "case_key": "en",
            "views": {name: ["c.en"] for name in views},
            "question_bindings": {
                "d": {
                    "question_key": "q.en",
                }
            },
        },
    ]
    groups = [{"group_id": "g", "presentations": presentations}]
    result = {
        "status": "resolved",
        "output_id": "yes",
        "option_index": 0,
        "rules": [],
    }
    decision = {
        "decision_id": "d",
        "transition": "resolution_loss",
        "views": {
            name: deepcopy(result)
            for name in ("full", "removed", "decisive_only", "distractor_only")
        },
    }
    fact_states = {
        name: {} for name in ("full", "removed", "decisive_only", "distractor_only")
    }
    traces = {
        ("g", "p.he"): {"fact_states": fact_states, "decisions": [decision]},
        ("g", "p.en"): {
            "fact_states": deepcopy(fact_states),
            "decisions": [deepcopy(decision)],
        },
    }
    replay._validate_hebrew(groups, cases, traces)

    three_way_cases = deepcopy(cases)
    three_way_cases["he2"] = twin_case("he", "he2")
    three_way_groups = deepcopy(groups)
    extra_presentation = deepcopy(three_way_groups[0]["presentations"][0])
    extra_presentation["presentation_id"] = "p.he2"
    extra_presentation["case_key"] = "he2"
    extra_presentation["views"] = {name: ["c.he2"] for name in views}
    extra_presentation["question_bindings"]["d"]["question_key"] = "q.he2"
    three_way_groups[0]["presentations"].append(extra_presentation)
    with pytest.raises(replay.PolicyError, match="exactly one English and one Hebrew"):
        replay._validate_hebrew(three_way_groups, three_way_cases, traces)

    option_drift = deepcopy(cases)
    option_drift["en"]["questions"][0]["option_ids"] = ["no", "yes"]
    with pytest.raises(replay.PolicyError, match="semantic anchors drift"):
        replay._validate_hebrew(groups, option_drift, traces)

    evidence_drift = deepcopy(cases)
    evidence_drift["en"]["questions"][0]["decisive_chunk_keys"] = []
    with pytest.raises(replay.PolicyError, match="evidence anchor mappings drift"):
        replay._validate_hebrew(groups, evidence_drift, traces)

    environment_drift = deepcopy(traces)
    environment_drift[("g", "p.en")]["fact_states"] = {"full": {"f0001": {}}}
    with pytest.raises(replay.PolicyError, match="view fact values"):
        replay._validate_hebrew(groups, cases, environment_drift)

    trace_drift = deepcopy(traces)
    trace_drift[("g", "p.en")]["decisions"][0]["views"]["decisive_only"][
        "output_id"
    ] = "no"
    with pytest.raises(replay.PolicyError, match="replay traces drift"):
        replay._validate_hebrew(groups, cases, trace_drift)


def test_duplicate_twin_anchors_are_rejected():
    panel, ledger, rubric = _fixture()
    panel["cases"][0]["chunks"][1]["twin_anchor"] = "duplicate"
    panel["cases"][0]["chunks"][2]["twin_anchor"] = "duplicate"
    with pytest.raises(replay.PolicyError, match="duplicate chunk twin anchors"):
        _run((panel, ledger, rubric))


def test_json_and_collection_bounds_and_recursion_are_fail_closed(monkeypatch):
    with pytest.raises(replay.PolicyError, match="integer exceeds"):
        replay.parse_canonical_json(
            ('{"n":' + "9" * (replay.MAX_INTEGER_DIGITS + 1) + "}\n").encode(),
            "huge integer",
        )

    nested = "0"
    for _ in range(replay.MAX_JSON_DEPTH + 1):
        nested = f"[{nested}]"
    with pytest.raises(replay.PolicyError, match="depth bound|valid UTF-8 JSON"):
        replay.parse_canonical_json((nested + "\n").encode(), "deep")

    monkeypatch.setattr(replay, "MAX_JSON_NODES", 2)
    with pytest.raises(replay.PolicyError, match="node bound"):
        replay.parse_canonical_json(b'{"a":1,"b":2}\n', "wide")
    monkeypatch.setattr(replay, "MAX_JSON_NODES", 250_000)
    monkeypatch.setattr(replay, "MAX_JSON_OBJECTS", 1)
    with pytest.raises(replay.PolicyError, match="object bound"):
        replay.parse_canonical_json(b'{"a":{}}\n', "objects")
    monkeypatch.setattr(replay, "MAX_JSON_OBJECTS", 50_000)
    panel, ledger, rubric = _fixture()
    monkeypatch.setattr(replay, "MAX_RULES", 1)
    with pytest.raises(replay.PolicyError, match="outputs|rules"):
        _run((panel, ledger, rubric))
    monkeypatch.setattr(replay, "MAX_RULES", 32)
    monkeypatch.setattr(replay, "MAX_BINDINGS", 0)
    with pytest.raises(replay.PolicyError, match="bindings"):
        _run(_fixture())
    monkeypatch.setattr(replay, "MAX_BINDINGS", 16_384)
    monkeypatch.setattr(replay, "MAX_PRESENTATIONS", 0)
    with pytest.raises(replay.PolicyError, match="presentations"):
        _run(_fixture())
    monkeypatch.setattr(replay, "MAX_PRESENTATIONS", 2_048)
    monkeypatch.setattr(replay, "MAX_INPUT_BYTES", 2)
    with pytest.raises(replay.PolicyError, match="input byte bound"):
        replay.parse_canonical_json(b"{}\n", "oversized")
    monkeypatch.setattr(replay, "MAX_INPUT_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(
        replay.json,
        "loads",
        lambda *args, **kwargs: (_ for _ in ()).throw(RecursionError()),
    )
    with pytest.raises(replay.PolicyError, match="valid UTF-8 JSON"):
        replay.parse_canonical_json(b"{}\n", "recursive")


def test_schema_hash_is_hard_bound():
    panel, ledger, rubric = _fixture()
    hashes = _seal(panel, ledger, rubric)
    hashes["schema_sha256"] = "a" * 64
    with pytest.raises(replay.PolicyError, match="frozen schema"):
        replay.replay(
            panel,
            ledger,
            rubric,
            **hashes,
            adjudication_receipts={ADJUDICATION_SHA256: ADJUDICATION_BYTES},
        )


def test_runtime_and_schema_group_bounds_match_and_maximum_is_valid():
    schema = json.loads(Path("sources/r052/policy_replay.schema.json").read_text())
    schema_max = schema["$defs"]["policyRubric"]["properties"]["semantic_groups"][
        "maxItems"
    ]
    assert schema_max == replay.MAX_GROUPS == 64

    base_panel, base_ledger, base_rubric = _fixture()
    cases = []
    bindings = []
    groups = []
    for index in range(replay.MAX_GROUPS):
        suffix = f"{index:02d}"
        case = deepcopy(base_panel["cases"][0])
        case["case_key"] = f"case.{suffix}"
        key_map = {
            chunk["chunk_key"]: f"{chunk['chunk_key']}.{suffix}"
            for chunk in case["chunks"]
        }
        for chunk in case["chunks"]:
            chunk["chunk_key"] = key_map[chunk["chunk_key"]]
        question = case["questions"][0]
        question["question_key"] = f"q.{suffix}"
        question["decisive_chunk_keys"] = [
            key_map[key] for key in question["decisive_chunk_keys"]
        ]
        question["known_distractor_chunk_keys"] = [
            key_map[key] for key in question["known_distractor_chunk_keys"]
        ]
        _set_causal_proof(case, question)
        cases.append(case)

        binding = deepcopy(base_ledger["bindings"][0])
        binding["case_key"] = case["case_key"]
        binding["chunk_key"] = key_map["c.decisive"]
        bindings.append(binding)

        group = deepcopy(base_rubric["semantic_groups"][0])
        group["group_id"] = f"g.{suffix}"
        presentation = group["presentations"][0]
        presentation["presentation_id"] = f"p.{suffix}"
        presentation["case_key"] = case["case_key"]
        presentation["question_bindings"]["d.permit"]["question_key"] = question[
            "question_key"
        ]
        presentation["views"] = {
            name: [key_map[key] for key in keys]
            for name, keys in presentation["views"].items()
        }
        presentation["ablations"][0]["remove_chunk_keys"] = [key_map["c.decisive"]]
        groups.append(group)

    base_panel["cases"] = cases
    base_ledger["bindings"] = bindings
    base_rubric["semantic_groups"] = groups
    receipt = _run((base_panel, base_ledger, base_rubric))
    assert len(receipt["semantic_groups"]) == replay.MAX_GROUPS


def test_cli_requires_external_hashes_and_never_overwrites(tmp_path):
    panel, ledger, rubric = _fixture()
    hashes = _seal(panel, ledger, rubric)
    paths = {}
    for name, value in (("panel", panel), ("ledger", ledger), ("rubric", rubric)):
        path = tmp_path / f"{name}.json"
        path.write_bytes(replay.canonical_json(value))
        paths[name] = path
    schema = Path("sources/r052/policy_replay.schema.json")
    evaluator = Path(replay.__file__)
    adjudication = tmp_path / "adjudication.txt"
    adjudication.write_bytes(ADJUDICATION_BYTES)
    output = tmp_path / "receipt.json"
    argv = [
        "--panel",
        str(paths["panel"]),
        "--panel-sha256",
        hashes["panel_sha256"],
        "--fact-ledger",
        str(paths["ledger"]),
        "--fact-ledger-sha256",
        hashes["ledger_sha256"],
        "--rubric",
        str(paths["rubric"]),
        "--rubric-sha256",
        hashes["rubric_sha256"],
        "--schema",
        str(schema),
        "--schema-sha256",
        replay.sha256_bytes(schema.read_bytes()),
        "--evaluator-sha256",
        replay.sha256_bytes(evaluator.read_bytes()),
        "--adjudication-receipt",
        f"{ADJUDICATION_SHA256}={adjudication}",
        "--output",
        str(output),
    ]
    assert replay.main(argv) == 0
    assert replay.parse_canonical_json(output.read_bytes(), "receipt")["receipt_sha256"]
    assert replay.main(argv) == 2
    target = tmp_path / "target.txt"
    target.write_text("unchanged")
    symlink_output = tmp_path / "symlink-output.json"
    symlink_output.symlink_to(target)
    argv[-1] = str(symlink_output)
    assert replay.main(argv) == 2
    assert target.read_text() == "unchanged"


def test_atomic_publication_exposes_only_complete_receipt_bytes(tmp_path, monkeypatch):
    destination = tmp_path / "receipt.json"
    payload = b'{"complete":true}\n'
    real_link = replay.os.link
    publications = []

    def inspect_then_link(source, target, *, follow_symlinks):
        assert Path(source).parent == destination.parent
        assert Path(source).read_bytes() == payload
        assert not Path(target).exists()
        publications.append((source, target))
        return real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(replay.os, "link", inspect_then_link)
    replay._write_exclusive_atomic(destination, payload)

    assert publications
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
