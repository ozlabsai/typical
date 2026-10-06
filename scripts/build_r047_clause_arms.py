"""Build matched R047 clause-supervision and repeated-final control arms."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.build_r045_arms import _copy_images, _read, _tree_hash, _valid_row
from scripts.build_r045_compositional_transfer import _evaluate, execute


FROZEN_R045_ARM_B_SHA256 = "ff0e0b2c8b4ad0f11552b51db176352c21527cabe8576341e0773da85574b801"
FROZEN_ROWS = 800
FROZEN_PRESENTATIONS = 900
FROZEN_PAIR_SLOTS = 100
FROZEN_AUGMENTED_ROWS = 48
FROZEN_K2_SUBSET_SHA256 = "b143ca8888480d4474567e8645f4242b6f9f92280816ec5d6655d92d4f21f23f"
FROZEN_SORTED_SLOT_IDS_SHA256 = "7a9bba8d724e16ae3bfc96805392b78d2531756f356fb40b2bdbbb7a6a0822c0"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _canonical_sha(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha(payload)


def _decision(view):
    questions, targets = view.get("questions"), view.get("targets")
    if not isinstance(questions, dict) or len(questions) != 1 or set(questions) != set(targets or {}):
        raise ValueError(f"{view.get('id')}: R047 requires exactly one aligned decision question")
    qid = next(iter(questions))
    question = questions[qid]
    if question.get("type") != "choice" or not isinstance(question.get("criteria"), dict):
        raise ValueError(f"{view.get('id')}: R047 requires one choice question")
    if set(question["criteria"]) != {"no", "yes"}:
        raise ValueError(f"{view.get('id')}: R047 requires semantic no/yes options")
    return qid, question, targets[qid]


def _instruction(language, scope, position):
    if language == "he":
        if scope == "clause":
            return f"האם תנאי מספר {position} מתקיים?"
        return "האם בדיקת המקרה כולו מאשרת להמשיך?"
    if language == "en":
        if scope == "clause":
            return f"Is numbered condition {position} satisfied?"
        return "Does the whole-case audit authorize proceeding?"
    raise ValueError(f"unsupported R047 language: {language!r}")


def _auxiliary_view(view, clause, position, scope):
    qid, question, final_target = _decision(view)
    meta = view.get("meta") or {}
    language = meta.get("language")
    final_gold = execute(meta.get("program") or {}, meta.get("clauses") or [])
    if final_target.get("label") != ("yes" if final_gold else "no"):
        raise ValueError(f"{view.get('id')}: final target disagrees with executable gold")
    gold = _evaluate(clause) if scope == "clause" else final_gold
    auxiliary = {
        "id": f"{view['id']}::r047-aux-{position:02d}",
        "state": copy.deepcopy(view["state"]),
        "questions": {qid: {**copy.deepcopy(question),
                            "instructions": _instruction(language, scope, position)}},
        "targets": {qid: {"label": "yes" if gold else "no"}},
        "meta": {**copy.deepcopy(meta),
                 "r047_auxiliary": {
                     "position": position,
                     "parent_view_id": view["id"],
                     "clause_index": position - 1,
                     "clause_kind": clause["kind"],
                     "clause_sha256": _canonical_sha(clause),
                     "scope": scope,
                     "reference_clause_kind": clause["kind"],
                     "gold_source": ("executable_clause" if scope == "clause"
                                     else "executable_final"),
                     "role": "auxiliary",
                     "arm": "treatment" if scope == "clause" else "control",
                 }},
    }
    return auxiliary


def _augment_row(row, scope):
    result = copy.deepcopy(row)
    if (row.get("meta") or {}).get("source") != "r045_generated":
        return result
    composition = row["meta"].get("composition")
    if not isinstance(composition, list) or len(composition) <= 1:
        return result
    if not _valid_row(row) or row.get("pair_mode") != "counterfactual":
        raise ValueError(f"{row.get('id')}: invalid R045 compositional pair")
    if "auxiliary_views" in row or any("auxiliary_views" in view for view in row["paired_views"]):
        raise ValueError(f"{row.get('id')}: input already contains auxiliary views")
    result["auxiliary_views"] = []
    for view in result["paired_views"]:
        clauses = (view.get("meta") or {}).get("clauses")
        if not isinstance(clauses, list) or len(clauses) != len(composition):
            raise ValueError(f"{view.get('id')}: clauses do not match composition arity")
        if [clause.get("kind") for clause in clauses] != [
                clause.get("kind") for clause in (row["paired_views"][0]["meta"]["clauses"])]:
            raise ValueError(f"{view.get('id')}: paired clause order differs")
        result["auxiliary_views"].extend(
            _auxiliary_view(view, clause, position, scope)
            for position, clause in enumerate(clauses, 1)
        )
    return result


def _presentations(rows):
    return sum(len(row.get("paired_views", [row])) for row in rows)


def _summary(rows):
    augmented = [row for row in rows if row.get("auxiliary_views")]
    auxiliary = [aux for row in augmented for aux in row["auxiliary_views"]]
    labels = Counter(next(iter(aux["targets"].values()))["label"] for aux in auxiliary)
    languages = Counter(aux["meta"].get("language") for aux in auxiliary)
    kinds = Counter(aux["meta"]["r047_auxiliary"]["reference_clause_kind"] for aux in auxiliary)
    operators = Counter(aux["meta"].get("program", {}).get("operator") for aux in auxiliary)
    arities = Counter(len(row["meta"]["composition"]) for row in augmented)
    return {
        "logical_rows": len(rows),
        "base_presentations": _presentations(rows),
        "pair_slots": sum("paired_views" in row for row in rows),
        "augmented_rows": len(augmented),
        "augmented_base_views": sum(len(row["paired_views"]) for row in augmented),
        "auxiliary_views": len(auxiliary),
        "effective_presentations": _presentations(rows) + len(auxiliary),
        "auxiliary_labels": dict(sorted(labels.items())),
        "auxiliary_languages": dict(sorted(languages.items())),
        "auxiliary_clause_kinds": dict(sorted(kinds.items())),
        "auxiliary_operators": dict(sorted(operators.items())),
        "augmented_arities": {str(key): value for key, value in sorted(arities.items())},
    }


def _strip_auxiliary(row):
    result = copy.deepcopy(row)
    result.pop("auxiliary_views", None)
    return result


def build(r045_arm_b, out, expected_input_sha256=FROZEN_R045_ARM_B_SHA256):
    source_path = Path(r045_arm_b)
    source_payload = source_path.read_bytes()
    source_hash = _sha(source_payload)
    if source_hash != expected_input_sha256:
        raise ValueError(
            f"R045 arm-B hash mismatch: expected {expected_input_sha256}, found {source_hash}")
    source_rows, source_info = _read(source_path)
    if any(not _valid_row(row) for row in source_rows):
        raise ValueError("R045 arm-B contains an invalid trainer row")
    if any("auxiliary_views" in row or any("auxiliary_views" in view
                                           for view in row.get("paired_views", []))
           for row in source_rows):
        raise ValueError("R045 arm-B must not already contain auxiliary views")

    control = [_augment_row(row, "whole_case") for row in source_rows]
    clause = [_augment_row(row, "clause") for row in source_rows]
    if [_strip_auxiliary(row) for row in control] != source_rows:
        raise AssertionError("control changed frozen base-row content or order")
    if [_strip_auxiliary(row) for row in clause] != source_rows:
        raise AssertionError("clause arm changed frozen base-row content or order")

    control_summary, clause_summary = _summary(control), _summary(clause)
    matched_fields = ("logical_rows", "base_presentations", "pair_slots", "augmented_rows",
                      "augmented_base_views", "auxiliary_views", "effective_presentations",
                      "auxiliary_languages", "auxiliary_clause_kinds", "auxiliary_operators",
                      "augmented_arities")
    if any(control_summary[field] != clause_summary[field] for field in matched_fields):
        raise AssertionError("R047 arm exposure mismatch")
    if expected_input_sha256 == FROZEN_R045_ARM_B_SHA256:
        frozen = {"logical_rows": FROZEN_ROWS, "base_presentations": FROZEN_PRESENTATIONS,
                  "pair_slots": FROZEN_PAIR_SLOTS, "augmented_rows": FROZEN_AUGMENTED_ROWS}
        if any(control_summary[key] != value for key, value in frozen.items()):
            raise ValueError(f"frozen R045 arm-B structure mismatch: {control_summary}")

    source_k2 = [row for row in source_rows
                 if (row.get("meta") or {}).get("source") == "r045_generated"
                 and len((row.get("meta") or {}).get("composition") or []) == 2]
    treatment_k2 = [row for row in clause if row.get("auxiliary_views")
                    and len((row.get("meta") or {}).get("composition") or []) == 2]
    source_k2_hash = _sha(_jsonl(source_k2))
    sorted_slot_ids_hash = _sha(("\n".join(sorted(row["id"] for row in source_k2)) + "\n").encode())
    if len(source_k2) != len(treatment_k2):
        raise AssertionError("M0 treatment subset does not align with its frozen source slots")
    if expected_input_sha256 == FROZEN_R045_ARM_B_SHA256:
        if source_k2_hash != FROZEN_K2_SUBSET_SHA256:
            raise ValueError(
                f"frozen K=2 subset hash mismatch: {source_k2_hash}")
        if sorted_slot_ids_hash != FROZEN_SORTED_SLOT_IDS_SHA256:
            raise ValueError(
                f"frozen sorted slot-ID hash mismatch: {sorted_slot_ids_hash}")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    image_manifest = _copy_images(control + clause, source_path, out)
    image_tree_hash, image_count = _tree_hash(out, image_manifest)
    payloads = {"control": _jsonl(control), "clause": _jsonl(clause),
                "m0-treatment": _jsonl(treatment_k2)}
    for name, payload in payloads.items():
        (out / f"{name}.jsonl").write_bytes(payload)

    manifest = {
        "experiment": "r047_clause_auxiliary",
        "builder_sha256": _sha(Path(__file__).read_bytes()),
        "input": source_info,
        "contract": {
            "base_rows_and_order_preserved": True,
            "augmentation_field": "auxiliary_views",
            "augmented_scope": "r045_generated rows with composition arity greater than one",
            "control": "repeat executable whole-case decision once per clause",
            "treatment": "predict each displayed clause's executable truth",
            "same_state": True,
            "same_option_labels_and_order": True,
            "same_auxiliary_view_count": True,
            "generic_composition_arity": True,
            "benchmark_sources_rejected": True,
            "token_budget_requires_preflight": True,
        },
        "arms": {"control": control_summary, "clause": clause_summary},
        "m0_treatment": {
            "rows": len(treatment_k2),
            "canonical_views": sum(len(row["paired_views"]) for row in treatment_k2),
            "auxiliary_views": sum(len(row["auxiliary_views"]) for row in treatment_k2),
            "effective_presentations": sum(
                len(row["paired_views"]) + len(row["auxiliary_views"])
                for row in treatment_k2),
            "source_subset_sha256": source_k2_hash,
            "source_identifier_field": "id",
            "sorted_source_ids_sha256": sorted_slot_ids_hash,
            "sorted_slot_ids_sha256": sorted_slot_ids_hash,
            "summary": _summary(treatment_k2),
        },
        "images": {"files": image_count, "tree_sha256": image_tree_hash,
                   "files_by_sha256": image_manifest},
        "outputs": {name: {"rows": len(rows), "sha256": _sha(payloads[name])}
                    for name, rows in (("control", control), ("clause", clause),
                                       ("m0-treatment", treatment_k2))},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r045-arm-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = build(args.r045_arm_b, args.out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
