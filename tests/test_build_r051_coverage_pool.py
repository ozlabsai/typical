import copy
import hashlib
import json
import re
from collections import Counter

import pytest

from scripts import build_r051_coverage_pool as builder
from scripts.build_r051_coverage_pool import (
    CANDIDATE_SCHEMA,
    LICENSE_POLICY,
    QUARANTINE_SCHEMA,
    QUARANTINE_TARGETS,
    SCORE_SCHEMA,
    SOURCE_SCHEMA,
    TYPE_TOTALS,
    _canonical,
    _jsonl,
    _normalize,
    build_pool,
    write_pool,
)
from scripts.exact_policy_executor import execute_policy
from scripts.typed_policy_executor import validate_typed_policy_siblings


def _sha(value):
    if not isinstance(value, bytes):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _offsets(text):
    return [match.span() for match in re.finditer(r"\S+", text)]


def _input_tokens(row, qid, reversed_options):
    del reversed_options
    question = row["questions"][qid]
    return len(row["state"].split()) + len(question["instructions"].split()) + sum(
        len(value.split()) for value in question["criteria"].values()
    ) + 32


def _action(name):
    return {"name": name, "payload": {}}


def _policy(fields):
    return {
        "fields": fields,
        "mode": "priority",
        "rules": [
            {"when": {"field": field}, "action": _action(f"action-{index}")}
            for index, field in enumerate(fields)
        ],
        "fallback": _action("fallback"),
    }


def _question(kind, ordinal, truth_counter):
    if kind == "choice":
        criteria = {"route_a": "Route A", "route_b": "Route B", "route_c": "Route C"}
        target = "route_b" if ordinal % 2 else "route_a"
        positives = ["route_b", "route_c"]
        score_order = None
    elif kind == "noul":
        criteria = {"no": "No", "yes": "Yes"}
        target = "yes" if truth_counter % 2 else "no"
        positives = ["yes"]
        score_order = None
    else:
        criteria = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
        score_order = list(criteria)
        target = score_order[truth_counter % len(score_order)]
        positives = ["high", "critical"]
    return {
        "type": kind,
        "instructions": f"Resolve typed field {ordinal} from the dossier.",
        "criteria": criteria,
    }, target, positives, score_order


def _fixture():
    candidates = []
    documents = []
    type_sequence = (["choice", "noul", "score"] * 1_331)[:3_992]
    type_cursor = 0
    noul_cursor = 0
    score_cursor = 0
    for root_index in range(800):
        group_id = f"r051-root-{root_index:04d}"
        arity = 4 if root_index < 8 else 5
        fields = [f"field_{index:02d}" for index in range(arity)]
        policy = _policy(fields)
        questions = {}
        targets = {}
        events = {}
        score_orders = {}
        trace = {}
        for field_index, field in enumerate(fields):
            kind = type_sequence[type_cursor]
            type_cursor += 1
            counter = noul_cursor if kind == "noul" else score_cursor if kind == "score" else type_cursor
            question, target, positives, score_order = _question(kind, type_cursor, counter)
            if kind == "noul":
                noul_cursor += 1
            if kind == "score":
                score_cursor += 1
            qid = f"q{field_index:02d}"
            questions[qid] = question
            targets[qid] = {"label": target}
            events[field] = {"positive_labels": positives}
            trace[field] = target in positives
            if score_order:
                score_orders[qid] = score_order

        evidence = f"decisive_exhibit_{root_index:04d}"
        if root_index < 40:
            bucket = "early" if root_index < 14 else "middle" if root_index < 27 else "late"
            position = {"early": 800, "middle": 4_000, "late": 7_200}[bucket]
            words = [f"natural_{root_index}_{index}" for index in range(8_000)]
            words[position] = evidence
            state = " ".join(words)
        else:
            state = f"Natural licensed case document {root_index}; facts are specific to this record."
        family = f"family-{root_index // 80:02d}"
        cell_within_family = (root_index % 80) // 20
        domain = f"domain-{cell_within_family:02d}"
        template = f"policy-{cell_within_family:02d}"
        source = {
            "source_id": family,
            "document_id": f"document-{root_index:04d}",
            "upstream_id": f"upstream-{root_index:04d}",
            "source_family": family,
            "domain": domain,
            "revision": "frozen-revision-1",
            "license": "CC-BY-4.0",
            "license_url": "https://example.test/license",
            "generator_family": f"generator-train-{root_index % 7}",
            "policy_template": template,
            "lineage": [f"lineage-{root_index:04d}"],
            "raw_state_sha256": _sha(state),
            "normalized_state_sha256": _sha(_normalize(state)),
        }
        meta = {
            "panel": "typed_policy_v1",
            "semantic_group_id": group_id,
            "option_order": "canonical",
            "field_order": fields,
            "qid_to_field": {f"q{index:02d}": field for index, field in enumerate(fields)},
            "field_events": events,
            "score_orders": score_orders,
            "action_trace": trace,
            "policy": policy,
            "gold_outcome": execute_policy(policy, trace).as_dict(),
            "decision_threshold": 0.5,
            "split": "train",
            "source": source,
            "is_long_context": root_index < 40,
            "decisive_evidence": None,
        }
        row = {
            "id": f"candidate-{root_index:04d}",
            "state": state,
            "questions": questions,
            "targets": targets,
            "meta": meta,
        }
        if root_index < 40:
            start = state.index(evidence)
            qid = "q00"
            field = fields[0]
            labels = list(questions[qid]["criteria"])
            positive = set(events[field]["positive_labels"])
            original = targets[qid]["label"]
            removed = next(label for label in labels if (label in positive) != (original in positive))
            removed_trace = dict(trace)
            removed_trace[field] = removed in positive
            meta["decisive_evidence"] = {
                "text": evidence,
                "start_char": start,
                "end_char": start + len(evidence),
                "field": field,
                "removed_label": removed,
                "removed_outcome": execute_policy(policy, removed_trace).as_dict(),
                "removed_state_sha256": _sha(
                    (state[:start] + state[start + len(evidence):]).encode()
                ),
                "validation": {
                    "method": "independently_executed",
                    "receipt_sha256": _sha(f"evidence-receipt-{root_index}"),
                    "verifier_code_sha256": _sha(b"evidence-verifier-v1"),
                    "original_state_sha256": _sha(state),
                    "removed_state_sha256": _sha(
                        (state[:start] + state[start + len(evidence):]).encode()
                    ),
                    "original_label": original,
                    "removed_label": removed,
                    "original_outcome": meta["gold_outcome"],
                    "removed_outcome": execute_policy(policy, removed_trace).as_dict(),
                },
            }
        candidates.append(row)
        documents.append({
            "source_id": source["source_id"],
            "document_id": source["document_id"],
            "upstream_id": source["upstream_id"],
            "source_family": source["source_family"],
            "domain": source["domain"],
            "revision": source["revision"],
            "license": source["license"],
            "license_url": source["license_url"],
            "split": "train",
            "content_kind": "natural",
            "text": state,
            "lineage": source["lineage"],
        })
    assert Counter(
        question["type"] for row in candidates for question in row["questions"].values()
    ) == TYPE_TOTALS

    candidate_sha = _sha(b"fixture-candidates")
    candidate_manifest_sha = _sha(b"fixture-candidate-manifest")
    source_manifest_sha = _sha(b"fixture-source-manifest")
    candidate_manifest = {
        "schema": CANDIDATE_SCHEMA,
        "max_length": 16_384,
        "source_manifest_sha256": source_manifest_sha,
    }
    source_manifest = {
        "schema": SOURCE_SCHEMA,
        "license_policy": LICENSE_POLICY,
        "license_cleared": True,
        "tokenizer": {"id": "fixture-tokenizer", "revision": "fixture-revision"},
    }
    quarantine_manifest = {
        "schema": QUARANTINE_SCHEMA,
        "scope": "jevbench-and-typesafe-heldout-v1",
        "targets": QUARANTINE_TARGETS,
    }
    quarantine = [{
        "id": "heldout-1",
        "role": "benchmark",
        "texts": ["An unrelated frozen benchmark utterance with no shared dossier wording."],
        "lineage": ["benchmark-lineage-1"],
        "generator_family": "heldout-generator",
        "policy_template": "heldout-policy",
        "source_family": "heldout-family",
        "domain": "heldout-domain",
        "source_id": "heldout-source",
        "document_id": "heldout-document",
    }]
    ids = sorted(row["meta"]["semantic_group_id"] for row in candidates)
    score_receipt = {
        "schema": SCORE_SCHEMA,
        "status": "complete",
        "bindings": {
            "candidate_pool_sha256": candidate_sha,
            "candidate_manifest_sha256": candidate_manifest_sha,
            "parent_model_tree_sha256": _sha(b"model"),
            "parent_adapter_tree_sha256": _sha(b"adapter"),
            "scorer_code_sha256": _sha(b"scorer"),
            "runtime_lock_sha256": _sha(b"runtime"),
        },
        "root_inventory": ids,
        "root_inventory_sha256": _sha(_canonical(ids)),
        "scores": [{
            "semantic_group_id": identifier,
            "wrong_complete_outcome": index % 3 == 0,
            "complete_outcome_nll": 1.0 + (index % 17) / 100,
            "mean_gold_label_nll": 0.5 + (index % 11) / 100,
        } for index, identifier in enumerate(ids)],
    }
    bindings = {
        "candidate_pool_sha256": candidate_sha,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "source_snapshot_sha256": _sha(b"fixture-sources"),
        "source_manifest_sha256": source_manifest_sha,
        "quarantine_sha256": _sha(b"fixture-quarantine"),
        "quarantine_manifest_sha256": _sha(b"fixture-quarantine-manifest"),
        "score_receipt_sha256": _sha(json.dumps(score_receipt, sort_keys=True)),
    }
    return {
        "candidates": candidates,
        "documents": documents,
        "quarantine": quarantine,
        "score_receipt": score_receipt,
        "candidate_manifest": candidate_manifest,
        "source_manifest": source_manifest,
        "quarantine_manifest": quarantine_manifest,
        "bindings": bindings,
        "token_offsets": _offsets,
        "input_tokens": _input_tokens,
    }


@pytest.fixture(scope="module")
def fixture():
    return _fixture()


def test_builds_exact_deterministic_typed_coverage_pool(fixture):
    rows, manifest = build_pool(**fixture)
    rows_again, manifest_again = build_pool(
        **{**fixture, "candidates": list(reversed(fixture["candidates"]))}
    )

    assert rows == rows_again
    assert manifest == manifest_again
    assert len(rows) == 1_600
    assert manifest["distribution"] == {
        "semantic_roots": 800,
        "canonical_decisions": 3_992,
        "rendered_decisions": 7_984,
        "arities": {"4": 8, "5": 792},
        "canonical_types": {"choice": 1_331, "noul": 1_331, "score": 1_330},
        "rendered_types": {"choice": 2_662, "noul": 2_662, "score": 2_660},
        "noul_truths": {"no": 666, "yes": 665},
        "score_target_ranks": {"0": 333, "1": 333, "2": 332, "3": 332},
        "score_target_ranks_by_arity": {
            "4": {"0": 333, "1": 333, "2": 332, "3": 332},
        },
        "long_roots": 40,
        "decisive_evidence_quantiles": {"early": 14, "late": 13, "middle": 13},
        "max_length": 16_384,
        "max_observed_rendered_input_tokens": 8_045,
    }
    assert manifest["selection"]["hard_roots"] == 200
    assert manifest["selection"]["stratified_roots"] == 600
    assert manifest["contamination_audit"]["status"] == "pass"
    validate_typed_policy_siblings(rows)
    persisted = [json.loads(line) for line in _jsonl(rows).splitlines()]
    validate_typed_policy_siblings(persisted)
    for left, right in zip(rows[::2], rows[1::2]):
        assert left["state"] == right["state"]
        assert left["targets"] == right["targets"]
        assert all(
            list(left["questions"][qid]["criteria"])
            == list(reversed(list(right["questions"][qid]["criteria"])))
            for qid in left["questions"]
        )


def test_fails_closed_on_source_or_score_identity_drift(fixture):
    changed = copy.deepcopy(fixture)
    changed["documents"][0]["revision"] = "mutable-latest"
    with pytest.raises(ValueError, match="source provenance differs"):
        build_pool(**changed)

    changed = copy.deepcopy(fixture)
    changed["score_receipt"]["bindings"]["candidate_pool_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input binding mismatch"):
        build_pool(**changed)

    changed = copy.deepcopy(fixture)
    changed["source_manifest"]["license_cleared"] = False
    with pytest.raises(ValueError, match="license contract"):
        build_pool(**changed)


def test_fails_closed_on_benchmark_text_or_lineage_contamination(fixture):
    changed = copy.deepcopy(fixture)
    changed["quarantine"][0]["texts"] = [changed["candidates"][50]["state"]]
    with pytest.raises(ValueError, match="exact text fingerprint"):
        build_pool(**changed)

    changed = copy.deepcopy(fixture)
    changed["quarantine"][0]["lineage"] = ["lineage-0050"]
    with pytest.raises(ValueError, match="lineage overlaps"):
        build_pool(**changed)


def test_fails_closed_on_long_context_or_distribution_drift(fixture):
    changed = copy.deepcopy(fixture)
    long_row = changed["candidates"][0]
    short_state = " ".join(long_row["state"].split()[:100])
    long_row["state"] = short_state
    long_row["meta"]["source"]["raw_state_sha256"] = _sha(short_state)
    long_row["meta"]["source"]["normalized_state_sha256"] = _sha(_normalize(short_state))
    changed["documents"][0]["text"] = short_state
    with pytest.raises(ValueError, match="8k-12k"):
        build_pool(**changed)

    changed = copy.deepcopy(fixture)
    changed["input_tokens"] = lambda row, qid, reverse: (
        20_000 if row["meta"]["semantic_group_id"] == "r051-root-0040" else
        _input_tokens(row, qid, reverse)
    )
    with pytest.raises(ValueError, match="rendered tokens exceeds"):
        build_pool(**changed)

    changed = copy.deepcopy(fixture)
    for row in changed["candidates"]:
        if row["meta"]["is_long_context"]:
            continue
        for qid, question in row["questions"].items():
            if question["type"] == "score":
                row["targets"][qid]["label"] = row["meta"]["score_orders"][qid][0]
                field = row["meta"]["qid_to_field"][qid]
                row["meta"]["action_trace"][field] = (
                    row["targets"][qid]["label"]
                    in row["meta"]["field_events"][field]["positive_labels"]
                )
                row["meta"]["gold_outcome"] = execute_policy(
                    row["meta"]["policy"], row["meta"]["action_trace"]
                ).as_dict()
    with pytest.raises(ValueError, match="ranks are not balanced"):
        build_pool(**changed)


def test_fails_closed_on_quarantine_or_evidence_receipt_drift(fixture):
    changed = copy.deepcopy(fixture)
    changed["quarantine_manifest"]["targets"]["jevbench"]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="frozen external targets"):
        build_pool(**changed)

    changed = copy.deepcopy(fixture)
    changed["candidates"][0]["meta"]["decisive_evidence"]["validation"][
        "original_label"
    ] = "forged"
    with pytest.raises(ValueError, match="validation binding mismatch"):
        build_pool(**changed)

    changed = copy.deepcopy(fixture)
    changed["candidates"][8]["questions"].pop("q04")
    changed["candidates"][8]["targets"].pop("q04")
    with pytest.raises(ValueError, match="qid_to_field|field_order"):
        build_pool(**changed)


def test_path_api_authenticates_and_preserves_option_order(tmp_path, monkeypatch, fixture):
    values = copy.deepcopy(fixture)

    def write_jsonl(name, rows):
        path = tmp_path / name
        path.write_bytes(_jsonl(rows))
        return path

    def write_manifest(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        return path

    sources = write_jsonl("sources.jsonl", values["documents"])
    source_manifest_value = {
        **values["source_manifest"],
        "output": {"sha256": _sha(sources.read_bytes()), "rows": len(values["documents"])},
    }
    source_manifest = write_manifest("source-manifest.json", source_manifest_value)

    candidates = write_jsonl("candidates.jsonl", values["candidates"])
    candidate_manifest_value = {
        **values["candidate_manifest"],
        "source_manifest_sha256": _sha(source_manifest.read_bytes()),
        "output": {"sha256": _sha(candidates.read_bytes()), "rows": len(values["candidates"])},
    }
    candidate_manifest = write_manifest("candidate-manifest.json", candidate_manifest_value)

    quarantine = write_jsonl("quarantine.jsonl", values["quarantine"])
    quarantine_manifest_value = {
        **values["quarantine_manifest"],
        "output": {"sha256": _sha(quarantine.read_bytes()), "rows": len(values["quarantine"])},
    }
    quarantine_manifest = write_manifest("quarantine-manifest.json", quarantine_manifest_value)

    receipt = values["score_receipt"]
    receipt["bindings"]["candidate_pool_sha256"] = _sha(candidates.read_bytes())
    receipt["bindings"]["candidate_manifest_sha256"] = _sha(candidate_manifest.read_bytes())
    score_receipt = write_manifest("score-receipt.json", receipt)
    monkeypatch.setattr(builder, "_hf_token_tools", lambda *_: (_offsets, _input_tokens))
    out = tmp_path / "pool"

    manifest = write_pool(
        candidates_path=candidates,
        candidate_manifest_path=candidate_manifest,
        sources_path=sources,
        source_manifest_path=source_manifest,
        quarantine_path=quarantine,
        quarantine_manifest_path=quarantine_manifest,
        score_receipt_path=score_receipt,
        out=out,
        expected_candidate_manifest_sha256=_sha(candidate_manifest.read_bytes()),
        expected_source_manifest_sha256=_sha(source_manifest.read_bytes()),
        expected_quarantine_manifest_sha256=_sha(quarantine_manifest.read_bytes()),
        expected_score_receipt_sha256=_sha(score_receipt.read_bytes()),
    )

    persisted = [json.loads(line) for line in (out / "coverage-pool.jsonl").read_bytes().splitlines()]
    validate_typed_policy_siblings(persisted)
    assert manifest["outputs"]["coverage_pool"]["sha256"] == _sha(
        (out / "coverage-pool.jsonl").read_bytes()
    )
    assert json.loads((out / "manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError, match="overwrite"):
        write_pool(
            candidates_path=candidates,
            candidate_manifest_path=candidate_manifest,
            sources_path=sources,
            source_manifest_path=source_manifest,
            quarantine_path=quarantine,
            quarantine_manifest_path=quarantine_manifest,
            score_receipt_path=score_receipt,
            out=out,
            expected_candidate_manifest_sha256=_sha(candidate_manifest.read_bytes()),
            expected_source_manifest_sha256=_sha(source_manifest.read_bytes()),
            expected_quarantine_manifest_sha256=_sha(quarantine_manifest.read_bytes()),
            expected_score_receipt_sha256=_sha(score_receipt.read_bytes()),
        )
