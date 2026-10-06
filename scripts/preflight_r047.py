"""Fail-closed structural and token-budget preflight for frozen R047 arms."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.build_r045_arms import _read, _reject_benchmark
from scripts.build_r045_compositional_transfer import _evaluate, execute
from scripts.build_r047_clause_arms import (
    FROZEN_R045_ARM_B_SHA256,
    FROZEN_K2_SUBSET_SHA256,
    FROZEN_SORTED_SLOT_IDS_SHA256,
    _canonical_sha,
    _instruction,
    _jsonl,
    _strip_auxiliary,
    _summary,
)


TOKEN_DELTA_LIMIT = 0.005


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _decision(view):
    questions, targets = view.get("questions"), view.get("targets")
    if not isinstance(questions, dict) or len(questions) != 1 or set(questions) != set(targets or {}):
        raise ValueError(f"{view.get('id')}: expected one aligned decision question")
    qid = next(iter(questions))
    return qid, questions[qid], targets[qid]


def _base_views(row):
    return row.get("paired_views", [row])


def _all_training_views(rows):
    for row in rows:
        for view in _base_views(row):
            yield copy.deepcopy(view)
        yield from row.get("auxiliary_views", [])


def _validate_auxiliary(source, control, clause):
    expected = ((source.get("meta") or {}).get("source") == "r045_generated"
                and len((source.get("meta") or {}).get("composition") or []) > 1)
    source_views = _base_views(source)
    control_views = _base_views(control)
    clause_views = _base_views(clause)
    if not (len(source_views) == len(control_views) == len(clause_views)):
        raise ValueError(f"{source.get('id')}: base-view count mismatch")
    control_aux = control.get("auxiliary_views", [])
    clause_aux = clause.get("auxiliary_views", [])
    expected_count = sum(len((view.get("meta") or {}).get("clauses") or [])
                         for view in source_views) if expected else 0
    if len(control_aux) != expected_count or len(clause_aux) != expected_count:
        raise ValueError(f"{source.get('id')}: auxiliary-view count mismatch")
    if not expected:
        return
    cursor = 0
    for source_view in source_views:
        clauses = (source_view.get("meta") or {}).get("clauses") or []
        final_gold = execute(source_view["meta"]["program"], clauses) if expected else None
        for position, executable_clause in enumerate(clauses, 1):
            control_item, clause_item = control_aux[cursor], clause_aux[cursor]
            cursor += 1
            if control_item["id"] != clause_item["id"]:
                raise ValueError(f"{source_view['id']}: auxiliary IDs are not aligned")
            if control_item["state"] != source_view["state"] or clause_item["state"] != source_view["state"]:
                raise ValueError(f"{source_view['id']}: auxiliary state changed")
            control_qid, control_question, control_target = _decision(control_item)
            clause_qid, clause_question, clause_target = _decision(clause_item)
            source_qid, source_question, _ = _decision(source_view)
            if control_qid != source_qid or clause_qid != source_qid:
                raise ValueError(f"{source_view['id']}: auxiliary question ID changed")
            if (control_question.get("criteria") != source_question.get("criteria")
                    or clause_question.get("criteria") != source_question.get("criteria")):
                raise ValueError(f"{source_view['id']}: auxiliary options or order changed")
            language = source_view["meta"].get("language")
            if control_question.get("instructions") != _instruction(language, "whole_case", position):
                raise ValueError(f"{source_view['id']}: invalid repeated-final instruction")
            if clause_question.get("instructions") != _instruction(language, "clause", position):
                raise ValueError(f"{source_view['id']}: invalid clause instruction")
            if control_target.get("label") != ("yes" if final_gold else "no"):
                raise ValueError(f"{source_view['id']}: repeated-final target is not executable")
            clause_gold = _evaluate(executable_clause)
            if clause_target.get("label") != ("yes" if clause_gold else "no"):
                raise ValueError(f"{source_view['id']}: clause target is not executable")
            for item, scope, source_name in (
                    (control_item, "whole_case", "executable_final"),
                    (clause_item, "clause", "executable_clause")):
                aux_meta = (item.get("meta") or {}).get("r047_auxiliary") or {}
                expected_meta = {"position": position,
                                 "parent_view_id": source_view["id"],
                                 "clause_index": position - 1,
                                 "clause_kind": executable_clause["kind"],
                                 "clause_sha256": _canonical_sha(executable_clause),
                                 "scope": scope,
                                 "reference_clause_kind": executable_clause["kind"],
                                 "gold_source": source_name,
                                 "role": "auxiliary",
                                 "arm": "treatment" if scope == "clause" else "control"}
                if aux_meta != expected_meta:
                    raise ValueError(f"{source_view['id']}: invalid auxiliary provenance")
    if cursor != expected_count:
        raise AssertionError("auxiliary cursor did not consume the declared views")


def _check_manifest(manifest, input_path, control_path, clause_path, m0_path, input_rows,
                    control_rows, clause_rows, m0_rows, expected_input_sha256):
    if manifest.get("experiment") != "r047_clause_auxiliary":
        raise ValueError("wrong R047 manifest experiment")
    expected_hashes = {
        "input": manifest.get("input", {}).get("sha256"),
        "control": manifest.get("outputs", {}).get("control", {}).get("sha256"),
        "clause": manifest.get("outputs", {}).get("clause", {}).get("sha256"),
        "m0-treatment": manifest.get("outputs", {}).get("m0-treatment", {}).get("sha256"),
    }
    actual_hashes = {"input": _sha(input_path), "control": _sha(control_path),
                     "clause": _sha(clause_path), "m0-treatment": _sha(m0_path)}
    if expected_hashes != actual_hashes:
        raise ValueError(f"R047 file hash mismatch: expected {expected_hashes}, found {actual_hashes}")
    if actual_hashes["input"] != expected_input_sha256:
        raise ValueError(
            f"R045 arm-B hash mismatch: expected {expected_input_sha256}, "
            f"found {actual_hashes['input']}")
    if manifest["input"].get("rows") != len(input_rows):
        raise ValueError("manifest input row count mismatch")
    for name, rows in (("control", control_rows), ("clause", clause_rows)):
        if manifest["outputs"][name].get("rows") != len(rows):
            raise ValueError(f"manifest {name} output row count mismatch")
        if manifest.get("arms", {}).get(name) != _summary(rows):
            raise ValueError(f"manifest {name} exposure summary mismatch")
    if manifest["outputs"]["m0-treatment"].get("rows") != len(m0_rows):
        raise ValueError("manifest M0 output row count mismatch")

    source_k2 = [row for row in input_rows
                 if (row.get("meta") or {}).get("source") == "r045_generated"
                 and len((row.get("meta") or {}).get("composition") or []) == 2]
    treatment_k2 = [row for row in clause_rows if row.get("auxiliary_views")
                    and len((row.get("meta") or {}).get("composition") or []) == 2]
    if m0_rows != treatment_k2:
        raise ValueError("M0 treatment file is not the exact ordered treatment K=2 subset")
    source_k2_hash = hashlib.sha256(_jsonl(source_k2)).hexdigest()
    sorted_ids_hash = hashlib.sha256(
        ("\n".join(sorted(row["id"] for row in source_k2)) + "\n").encode()).hexdigest()
    expected_m0 = {
        "rows": len(treatment_k2),
        "canonical_views": sum(len(row["paired_views"]) for row in treatment_k2),
        "auxiliary_views": sum(len(row["auxiliary_views"]) for row in treatment_k2),
        "effective_presentations": sum(
            len(row["paired_views"]) + len(row["auxiliary_views"])
            for row in treatment_k2),
        "source_subset_sha256": source_k2_hash,
        "source_identifier_field": "id",
        "sorted_source_ids_sha256": sorted_ids_hash,
        "sorted_slot_ids_sha256": sorted_ids_hash,
        "summary": _summary(treatment_k2),
    }
    if manifest.get("m0_treatment") != expected_m0:
        raise ValueError("manifest M0 treatment summary mismatch")
    if expected_input_sha256 == FROZEN_R045_ARM_B_SHA256:
        if source_k2_hash != FROZEN_K2_SUBSET_SHA256:
            raise ValueError("frozen M0 source subset hash mismatch")
        if sorted_ids_hash != FROZEN_SORTED_SLOT_IDS_SHA256:
            raise ValueError("frozen M0 sorted slot-ID hash mismatch")
    contract = manifest.get("contract") or {}
    required_true = ("base_rows_and_order_preserved", "same_state",
                     "same_option_labels_and_order", "same_auxiliary_view_count",
                     "generic_composition_arity", "benchmark_sources_rejected",
                     "token_budget_requires_preflight")
    if any(contract.get(key) is not True for key in required_true):
        raise ValueError("manifest does not declare the full R047 contract")


def audit(input_path, control_path, clause_path, m0_path, manifest_path, token_counter=None,
          tokenizer_name=None, token_delta_limit=TOKEN_DELTA_LIMIT,
          expected_input_sha256=FROZEN_R045_ARM_B_SHA256):
    input_rows, _ = _read(input_path)
    control_rows, _ = _read(control_path)
    clause_rows, _ = _read(clause_path)
    m0_rows, _ = _read(m0_path)
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid R047 manifest") from error
    _reject_benchmark(Path(manifest_path), [manifest])
    _check_manifest(manifest, input_path, control_path, clause_path, m0_path, input_rows,
                    control_rows, clause_rows, m0_rows, expected_input_sha256)
    if not (len(input_rows) == len(control_rows) == len(clause_rows)):
        raise ValueError("R047 arm row counts differ")
    if [row.get("id") for row in input_rows] != [row.get("id") for row in control_rows]:
        raise ValueError("control row IDs or order changed")
    if [row.get("id") for row in input_rows] != [row.get("id") for row in clause_rows]:
        raise ValueError("clause row IDs or order changed")
    if [_strip_auxiliary(row) for row in control_rows] != input_rows:
        raise ValueError("control changed frozen input content")
    if [_strip_auxiliary(row) for row in clause_rows] != input_rows:
        raise ValueError("clause arm changed frozen input content")
    for source, control, clause in zip(input_rows, control_rows, clause_rows):
        _validate_auxiliary(source, control, clause)

    result = {
        "passed": True,
        "input_sha256": _sha(input_path),
        "control_sha256": _sha(control_path),
        "clause_sha256": _sha(clause_path),
        "m0_treatment_sha256": _sha(m0_path),
        "control": _summary(control_rows),
        "clause": _summary(clause_rows),
        "token_budget": None,
    }
    if token_counter is not None:
        if token_delta_limit < 0:
            raise ValueError("token delta limit must be nonnegative")
        control_tokens = sum(token_counter(view) for view in _all_training_views(control_rows))
        clause_tokens = sum(token_counter(view) for view in _all_training_views(clause_rows))
        if not isinstance(control_tokens, int) or not isinstance(clause_tokens, int):
            raise ValueError("token counter must return integer counts")
        if control_tokens <= 0 or clause_tokens <= 0:
            raise ValueError("encoded token budgets must be positive")
        denominator = max(control_tokens, clause_tokens, 1)
        delta = abs(control_tokens - clause_tokens)
        fraction = delta / denominator
        if fraction > token_delta_limit:
            raise ValueError(
                f"encoded token budget differs by {fraction:.6%}, above {token_delta_limit:.6%}")
        result["token_budget"] = {
            "tokenizer": tokenizer_name,
            "control_tokens": control_tokens,
            "clause_tokens": clause_tokens,
            "absolute_delta": delta,
            "relative_delta": fraction,
            "limit": token_delta_limit,
            "within_limit": True,
        }
    return result


def tokenizer_counter(tokenizer_path):
    from transformers import AutoTokenizer
    from scripts.train_gemma_decision_lora import encode, flatten_row

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def count(row):
        examples = flatten_row(row)
        return sum(len(encode(example, tokenizer, 1_000_000, shuffle=False)[0])
                   for example in examples)

    return count, getattr(tokenizer, "name_or_path", str(tokenizer_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--clause", required=True)
    parser.add_argument("--m0-treatment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    counter, tokenizer_name = tokenizer_counter(args.tokenizer)
    report = audit(args.input, args.control, args.clause, args.m0_treatment,
                   args.manifest, counter, tokenizer_name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
