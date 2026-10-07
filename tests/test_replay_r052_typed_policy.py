from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from scripts import replay_r052_typed_policy as replay

ZERO = "0" * 64


def _hash_text(text: str) -> str:
    return replay.sha256_bytes(text.encode())


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
            for key in ("text", "options", "decision_type", "policy_role")
        }
    )
    panel = {
        "schema_version": "test-panel",
        "cases": [
            {
                "case_key": "case.en",
                "language": "en",
                "twin_group_key": "twin.permit",
                "chunks": chunks,
                "questions": [question],
            }
        ],
    }
    panel_hash = replay.sha256_json(panel)
    ledger = {
        "schema_version": replay.LEDGER_VERSION,
        "panel_sha256": panel_hash,
        "facts": {"f.9f3a": {"type": "boolean", "unit": None}},
        "bindings": [
            {
                "case_key": "case.en",
                "chunk_key": "c.decisive",
                "chunk_content_sha256": chunks[1]["content_sha256"],
                "observations": [
                    {
                        "fact": "f.9f3a",
                        "value": True,
                        "locator": {
                            "kind": "text_span",
                            "char_start": 0,
                            "char_end": len(evidence_text),
                            "span_sha256": _hash_text(evidence_text),
                        },
                    }
                ],
                "adjudication_receipt_sha256": "1" * 64,
            }
        ],
    }
    ledger_hash = replay.sha256_json(ledger)
    fact = {"op": "fact", "id": "f.9f3a"}
    yes = {"op": "const", "type": "boolean", "unit": None, "value": True}
    no = {"op": "const", "type": "boolean", "unit": None, "value": False}
    rubric = {
        "schema_version": replay.RUBRIC_VERSION,
        "panel_sha256": panel_hash,
        "fact_ledger_sha256": ledger_hash,
        "semantic_groups": [
            {
                "group_id": "g.permit",
                "facts": {"f.9f3a": {"type": "boolean", "unit": None}},
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
        "schema_sha256": "a" * 64,
    }


def _run(values=None):
    panel, ledger, rubric = values or _fixture()
    return replay.replay(panel, ledger, rubric, **_seal(panel, ledger, rubric))


def test_happy_path_receipt_is_deterministic_and_missing_noul_is_never_false():
    values = _fixture()
    first = _run(deepcopy(values))
    second = _run(deepcopy(values))
    assert first == second
    trace = first["semantic_groups"][0]["presentations"][0]["decisions"][0]
    assert trace["transition"] == "resolution_loss"
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
    question["content_sha256"] = replay.sha256_json(
        {
            key: question[key]
            for key in ("text", "options", "decision_type", "policy_role")
        }
    )
    rubric["semantic_groups"][0]["presentations"][0]["question_bindings"]["d.permit"][
        "question_content_sha256"
    ] = question["content_sha256"]
    decision["outputs"].append({"id": "maybe", "label": "Maybe"})
    decision["order"].append("maybe")
    decision["rules"].append(
        {"output": "maybe", "when": {"op": "fact", "id": "f.absent"}}
    )
    rubric["semantic_groups"][0]["facts"]["f.absent"] = {
        "type": "boolean",
        "unit": None,
    }
    ledger["facts"]["f.absent"] = {"type": "boolean", "unit": None}
    ledger["bindings"][0]["observations"].append(
        {
            "fact": "f.absent",
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
            "adjudication_receipt_sha256": "2" * 64,
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
                }
            ],
        },
        "en": {
            "language": "en",
            "twin_group_key": "t",
            "chunks": [
                {
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
                    "question_bindings": {"d": {"question_key": "q"}},
                },
                {
                    "presentation_id": "p.en",
                    "case_key": "en",
                    "question_bindings": {"d": {"question_key": "q"}},
                },
            ],
        }
    ]
    traces = {
        ("g", "p.he"): {"full_fact_state": {}},
        ("g", "p.en"): {"full_fact_state": {}},
    }
    with pytest.raises(replay.PolicyError, match="semantic anchors drift"):
        replay._validate_hebrew(groups, cases, traces)

    case = {
        "chunks": [
            {
                "chunk_key": "real",
                "text": "neutral",
                "media": {},
                "image_control": {"group_key": "i", "variant": "real"},
            },
            {
                "chunk_key": "shuffle",
                "text": "neutral",
                "media": {},
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
    node = {"op": "fact", "id": "f.9f3a"}
    for _ in range(replay.MAX_AST_DEPTH):
        node = {"op": "not", "arg": node}
    rubric["semantic_groups"][0]["decisions"][0]["rules"][0]["when"] = node
    with pytest.raises(replay.PolicyError, match="maximum depth"):
        _run((panel, ledger, rubric))


def test_fact_ids_are_question_blind_and_text_spans_are_exact():
    panel, ledger, rubric = _fixture()
    ledger["facts"]["correct_answer"] = ledger["facts"].pop("f.9f3a")
    with pytest.raises(replay.PolicyError, match="question-blind"):
        _run((panel, ledger, rubric))

    panel, ledger, rubric = _fixture()
    ledger["bindings"][0]["observations"][0]["locator"]["span_sha256"] = ZERO
    with pytest.raises(replay.PolicyError, match="text span hash mismatch"):
        _run((panel, ledger, rubric))


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
        "--output",
        str(output),
    ]
    assert replay.main(argv) == 0
    assert replay.parse_canonical_json(output.read_bytes(), "receipt")["receipt_sha256"]
    assert replay.main(argv) == 2
