"""Fail-closed integrity and leakage preflight for the R047 causal panel.

The inventory manifest is external to the panel and has this small schema::

    {"version": 1,
     "training": [{"lineage": "r044", "path": "...", "sha256": "..."}, ...],
     "benchmarks": [{"path": "...", "sha256": "..."}, ...]}

Paths may be JSON/JSONL files or directories, whose hash is the SHA-256 of the
sorted ``relative-path\\0file-sha256\\n`` inventory.  Every input is therefore
content-addressed.  The caller must also pin the manifest hash;
there is no permissive "benchmark unavailable" path.  Reports contain only
benchmark hashes and aggregate counts, never benchmark paths or contents.
"""

import argparse
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from scripts.build_r047_panel import (
        CLAUSE_QUESTIONS, FINAL_QUESTION, FROZEN_COUNTS, GENERATOR_FAMILY_ID, K3,
        PROMPT_FAMILIES, RULE, SEEN_K2, UNSEEN_K2,
    )
except ModuleNotFoundError:  # Supports direct script execution.
    from build_r047_panel import (
        CLAUSE_QUESTIONS, FINAL_QUESTION, FROZEN_COUNTS, GENERATOR_FAMILY_ID, K3,
        PROMPT_FAMILIES, RULE, SEEN_K2, UNSEEN_K2,
    )


REQUIRED_TRAINING_LINEAGES = {"r044", "r045", "r047"}
FINGERPRINT_FIELDS = (
    "raw_state", "normalized_state", "raw_question", "normalized_question",
    "raw_instruction", "normalized_instruction", "clause", "ast", "multiset",
    "numeric_signature", "decisive_pair", "scenario_lineage", "template", "generator",
)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _fingerprint(value):
    return _sha(_canonical(value).encode())


def _normalize(value):
    text = value if isinstance(value, str) else _canonical(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def _read_jsonl(path):
    payload = Path(path).read_text()
    try:
        return [json.loads(line) for line in payload.splitlines() if line.strip()]
    except json.JSONDecodeError:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON/JSONL inventory or panel file: {Path(path).name}") from error
        return value if isinstance(value, list) else [value]


def _oracle_clause(clause):
    """Independent implementation: do not call the generator's evaluator."""
    kind = clause.get("kind")
    if kind == "decimal":
        return (sum((Decimal(value) for value in clause["credits"]), Decimal(0))
                - Decimal(clause["fee"]) >= Decimal(clause["threshold"]))
    if kind == "fraction":
        return Fraction(clause["numerator"], clause["denominator"]) >= Fraction(*clause["threshold"])
    if kind == "datetime":
        local = datetime.fromisoformat(clause["local_time"]).replace(
            tzinfo=ZoneInfo(clause["timezone"]))
        deadline = datetime.fromisoformat(clause["deadline"]).replace(tzinfo=timezone.utc)
        return local.astimezone(timezone.utc) <= deadline
    if kind == "ledger":
        return sum((Decimal(item["delta"]) for item in clause["events"]), Decimal(0)) >= Decimal(
            clause["threshold"])
    if kind == "conditional":
        group = next(item for item in clause["groups"]
                     if item["name"] == clause["condition_group"])
        return Fraction(group["success"], group["success"] + group["failure"]) >= Fraction(
            *clause["threshold"])
    if kind == "policy":
        candidates = [rule for rule in clause["rules"] if rule["active"] and
                      rule["scope"] in {"all", clause["request_scope"]}]
        winner = max(candidates, key=lambda rule: (rule["priority"], rule["effect"] == "deny"))
        return winner["effect"] == "allow"
    raise ValueError(f"unsupported clause kind: {kind!r}")


def _oracle_program(program, clauses):
    if program.get("operator") != "all":
        raise ValueError("R047 panel only permits the all operator")
    composition = program.get("composition")
    kinds = [clause.get("kind") for clause in clauses]
    if len(set(kinds)) != len(kinds) or set(composition or []) != set(kinds):
        raise ValueError("program composition and clauses differ")
    return all(_oracle_clause(clause) for clause in clauses)


def _decision(row):
    questions, targets = row.get("questions"), row.get("targets")
    if (not isinstance(questions, dict) or len(questions) != 1
            or not isinstance(targets, dict) or set(targets) != set(questions)):
        raise ValueError("every causal-panel row must contain one aligned question")
    qid = next(iter(questions))
    question, target = questions[qid], targets[qid]
    if (question.get("type") != "choice" or list(question.get("criteria", {}))
            not in (["yes", "no"], ["no", "yes"]) or target.get("label") not in {"yes", "no"}):
        raise ValueError("invalid binary decision row")
    return question, target


def _visible_clause_values(clause):
    """Values that must bind the executable AST to the model-visible state."""
    kind = clause["kind"]
    if kind == "decimal":
        return [*clause["credits"], clause["fee"], clause["threshold"], clause["unit"]]
    if kind == "fraction":
        return [clause["numerator"], clause["denominator"], *clause["threshold"]]
    if kind == "datetime":
        return [clause["local_time"], clause["timezone"], clause["deadline"]]
    if kind == "ledger":
        return [clause["threshold"], clause["currency"],
                *(value for event in clause["events"]
                  for value in (event["event_id"], event["delta"]))]
    if kind == "conditional":
        return [clause["condition_group"], *clause["threshold"],
                *(value for group in clause["groups"]
                  for value in (group["name"], group["success"], group["failure"]))]
    if kind == "policy":
        return [clause["request_scope"],
                *(value for rule in clause["rules"]
                  for value in (rule["name"], rule["priority"], rule["effect"], rule["scope"]))]
    raise ValueError(f"unsupported clause kind: {kind!r}")


def _numeric_signature(clause):
    values = []

    def visit(value, path=""):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            values.append((path, str(value)))
        elif isinstance(value, str) and (re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value)
                                         or re.search(r"\d{4}-\d{2}-\d{2}", value)):
            values.append((path, value))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[]" if not path else f"{path}[]")
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}.{key}" if path else key)

    visit(clause)
    return _fingerprint([clause.get("kind"), values])


def _validate_causal(rows, manifest, output_name):
    groups, clusters = defaultdict(list), defaultdict(list)
    ids = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError(f"{output_name}: missing or duplicate row ID")
        ids.add(row["id"])
        question, target = _decision(row)
        meta = row.get("meta") or {}
        required = {"semantic_cluster_id", "semantic_view_id", "scenario_id",
                    "generator_family_id", "composition", "arity", "language",
                    "option_order", "prompt_family", "truth_pattern",
                    "truth_hamming_weight", "prediction_role", "primitive",
                    "decisive_role", "seen_edge_count", "clauses", "program", "gold",
                    "renderer", "template_id", "entity_vocab", "cluster_lineage",
                    "cluster_ordinal"}
        if required - set(meta):
            raise ValueError(f"{output_name}: row metadata is incomplete")
        composition, clauses, truth = meta["composition"], meta["clauses"], meta["truth_pattern"]
        arity = meta["arity"]
        if (arity not in {2, 3} or len(composition) != arity or len(clauses) != arity
                or len(truth) != arity or any(type(value) is not bool for value in truth)):
            raise ValueError(f"{output_name}: invalid composition/truth shape")
        if composition != [clause.get("kind") for clause in clauses]:
            raise ValueError(f"{output_name}: causal-panel clause order drift")
        observed_truth = [_oracle_clause(clause) for clause in clauses]
        if observed_truth != truth or meta["truth_hamming_weight"] != sum(truth):
            raise ValueError(f"{output_name}: truth metadata disagrees with independent execution")
        program_gold = _oracle_program(meta["program"], clauses)
        role = meta["prediction_role"]
        if role == "final":
            expected_gold = program_gold
            if (meta["primitive"] is not None or meta["decisive_role"] != "final"
                    or meta["prompt_family"] != "final" or meta.get("clause_index") is not None):
                raise ValueError(f"{output_name}: invalid final prediction metadata")
        elif role == "clause":
            index = meta.get("clause_index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < arity:
                raise ValueError(f"{output_name}: invalid clause index")
            expected_gold = truth[index]
            decisive = all(truth[i] for i in range(arity) if i != index)
            if (meta["primitive"] != composition[index]
                    or meta["decisive_role"] != ("decisive" if decisive else "non_decisive")
                    or meta["prompt_family"] not in PROMPT_FAMILIES):
                raise ValueError(f"{output_name}: invalid clause prediction metadata")
        else:
            raise ValueError(f"{output_name}: unknown prediction role")
        if meta["gold"] is not expected_gold or target["label"] != ("yes" if expected_gold else "no"):
            raise ValueError(f"{output_name}: target disagrees with independently executed gold")
        if meta["generator_family_id"] != GENERATOR_FAMILY_ID:
            raise ValueError(f"{output_name}: wrong generator family")
        expected_lineage = {"root": meta["semantic_cluster_id"], "parent": None,
                            "generator_family_id": GENERATOR_FAMILY_ID}
        if meta["cluster_lineage"] != expected_lineage:
            raise ValueError(f"{output_name}: invalid cluster lineage")
        if meta["scenario_id"] != meta["semantic_cluster_id"]:
            raise ValueError(f"{output_name}: scenario/cluster lineage mismatch")
        if meta["language"] not in {"en", "he"} or meta["option_order"] not in {
                "yes_no", "no_yes"}:
            raise ValueError(f"{output_name}: invalid language/order")
        expected_order = ["yes", "no"] if meta["option_order"] == "yes_no" else ["no", "yes"]
        if list(question["criteria"]) != expected_order:
            raise ValueError(f"{output_name}: declared option order disagrees with question")
        expected_instruction = (FINAL_QUESTION[meta["language"]] if role == "final" else
                                CLAUSE_QUESTIONS[meta["prompt_family"]][meta["language"]].format(
                                    position=meta["clause_index"] + 1))
        if question.get("instructions") != expected_instruction:
            raise ValueError(f"{output_name}: prompt family/instruction mismatch")
        if RULE[meta["language"]] not in row["state"] or meta["entity_vocab"] not in row["state"]:
            raise ValueError(f"{output_name}: state contract or entity is not visible")
        if any(str(value) not in row["state"] for clause in clauses
               for value in _visible_clause_values(clause)):
            raise ValueError(f"{output_name}: executable AST value is absent from visible state")
        expected_edges = sum(tuple(pair) in SEEN_K2 for pair in itertools.combinations(composition, 2))
        if meta["seen_edge_count"] != expected_edges:
            raise ValueError(f"{output_name}: seen-edge count mismatch")
        groups[meta["semantic_view_id"]].append(row)
        clusters[meta["semantic_cluster_id"]].append(row)

    for semantic_view_id, items in groups.items():
        first = items[0]["meta"]
        arity, truth = first["arity"], first["truth_pattern"]
        if len(items) != arity + 1:
            raise ValueError(f"{output_name}: incomplete final/clause semantic view")
        if Counter(item["meta"]["prediction_role"] for item in items) != {"final": 1, "clause": arity}:
            raise ValueError(f"{output_name}: semantic view role coverage mismatch")
        if {item["state"] for item in items} != {items[0]["state"]}:
            raise ValueError(f"{output_name}: semantic-view state drift")
        invariant = {(item["meta"]["semantic_cluster_id"],
                      tuple(item["meta"]["truth_pattern"]), item["meta"]["language"],
                      item["meta"]["option_order"], _fingerprint(item["meta"]["clauses"]))
                     for item in items}
        if len(invariant) != 1:
            raise ValueError(f"{output_name}: semantic-view metadata drift")
        if {item["meta"]["clause_index"] for item in items
                if item["meta"]["prediction_role"] == "clause"} != set(range(arity)):
            raise ValueError(f"{output_name}: semantic-view clause coverage mismatch")
        expected_suffix = (f"::{''.join('1' if value else '0' for value in truth)}::"
                           f"{first['language']}::{first['option_order']}")
        if not semantic_view_id.endswith(expected_suffix):
            raise ValueError(f"{output_name}: unstable semantic-view identity")

    by_composition = Counter()
    for cluster_id, items in clusters.items():
        representative = items[0]["meta"]
        arity, composition = representative["arity"], tuple(representative["composition"])
        cluster_identity = {(item["meta"]["panel_stratum"],
                             tuple(item["meta"]["composition"]), item["meta"]["arity"],
                             item["meta"]["clause_prompt_family"], item["meta"]["renderer"],
                             item["meta"]["entity_vocab"], item["meta"].get("cluster_ordinal"))
                            for item in items}
        if len(cluster_identity) != 1:
            raise ValueError(f"{output_name}: semantic-cluster metadata drift")
        truth_patterns = {tuple(item["meta"]["truth_pattern"]) for item in items}
        if truth_patterns != set(itertools.product((False, True), repeat=arity)):
            raise ValueError(f"{output_name}: cluster does not contain the full Boolean cube")
        presentations = {(tuple(item["meta"]["truth_pattern"]), item["meta"]["language"],
                          item["meta"]["option_order"])
                         for item in items if item["meta"]["prediction_role"] == "final"}
        expected_presentations = set(itertools.product(
            itertools.product((False, True), repeat=arity), ("en", "he"), ("yes_no", "no_yes")))
        if presentations != expected_presentations:
            raise ValueError(f"{output_name}: language/order cube is incomplete")
        families = {item["meta"]["prompt_family"] for item in items
                    if item["meta"]["prediction_role"] == "clause"}
        if len(families) != 1 or not families <= set(PROMPT_FAMILIES):
            raise ValueError(f"{output_name}: cluster must use exactly one held-out prompt family")
        expected_rows = (2 ** arity) * 2 * 2 * (arity + 1)
        if len(items) != expected_rows:
            raise ValueError(f"{output_name}: cluster row count mismatch")
        clauses_by_truth = defaultdict(set)
        for item in items:
            clauses_by_truth[tuple(item["meta"]["truth_pattern"])].add(
                _fingerprint(item["meta"]["clauses"]))
        if any(len(values) != 1 for values in clauses_by_truth.values()):
            raise ValueError(f"{output_name}: AST drift across language/order twins")
        by_composition[(representative["panel_stratum"], composition)] += 1

    expected_counts = manifest["counts_per_composition"]
    expected_compositions = ({("unseen_k2", tuple(value)) for value in UNSEEN_K2}
                             | {("k3", tuple(value)) for value in K3}
                             | {("seen_k2", tuple(value)) for value in SEEN_K2})
    allowed = ({key for key in expected_compositions if output_name == "near-transfer"
                and key[0] == "seen_k2"}
               if output_name == "near-transfer" else
               {key for key in expected_compositions if key[0] != "seen_k2"})
    if set(by_composition) != allowed:
        raise ValueError(f"{output_name}: composition coverage mismatch")
    if any(count != expected_counts[stratum] for (stratum, _), count in by_composition.items()):
        raise ValueError(f"{output_name}: per-composition cluster count mismatch")
    for key in allowed:
        selected = [values[0]["meta"]["clause_prompt_family"] for values in clusters.values()
                    if (values[0]["meta"]["panel_stratum"],
                        tuple(values[0]["meta"]["composition"])) == key]
        counts = Counter(selected)
        family_coverage = (set(counts) == set(PROMPT_FAMILIES)
                           if expected_counts[key[0]] >= len(PROMPT_FAMILIES)
                           else len(counts) == expected_counts[key[0]])
        if not family_coverage or max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"{output_name}: prompt-family assignment is not balanced")
    return {"rows": len(rows), "clusters": len(clusters), "semantic_views": len(groups),
            "decisions": Counter(row["meta"]["prediction_role"] for row in rows)}


def _validate_legacy(rows):
    if len(rows) != 144:
        raise ValueError("legacy smoke must contain the 144-row seed47/domain-offset2 test slice")
    scenarios = set()
    for row in rows:
        if row.get("pair_mode") != "counterfactual" or len(row.get("paired_views", [])) != 2:
            raise ValueError("invalid legacy smoke pair")
        scenarios.add(row.get("meta", {}).get("scenario_id"))
        for view in row["paired_views"]:
            language = view.get("meta", {}).get("language")
            question, target = _decision(view)
            if (question["instructions"] != FINAL_QUESTION.get(language)
                    or view["meta"].get("question_repair") != "composition-neutral-v1"):
                raise ValueError("legacy smoke question was not repaired")
            gold = _oracle_program(view["meta"]["program"], view["meta"]["clauses"])
            if target["label"] != ("yes" if gold else "no"):
                raise ValueError("legacy smoke executable target mismatch")
    if len(scenarios) != 32:
        raise ValueError("legacy smoke must contain exactly 32 semantic scenarios")
    return {"rows": len(rows), "scenarios": len(scenarios), "causal_gate_eligible": False}


def _iter_views(rows):
    for row in rows:
        if isinstance(row, dict) and "state" in row and "questions" in row:
            yield row
        for field in ("paired_views", "auxiliary_views", "views", "presentations"):
            children = row.get(field, []) if isinstance(row, dict) else []
            if isinstance(children, list):
                yield from _iter_views(children)


def _sets():
    return {field: set() for field in FINGERPRINT_FIELDS}


def _add_view(result, view):
    state, questions, meta = view.get("state"), view.get("questions") or {}, view.get("meta") or {}
    if state is not None:
        raw = state if isinstance(state, str) else _canonical(state)
        result["raw_state"].add(_sha(raw.encode()))
        result["normalized_state"].add(_sha(_normalize(raw).encode()))
    for question in questions.values():
        result["raw_question"].add(_fingerprint(question))
        result["normalized_question"].add(_sha(_normalize(question).encode()))
        instruction = question.get("instructions")
        if isinstance(instruction, str):
            result["raw_instruction"].add(_sha(instruction.encode()))
            result["normalized_instruction"].add(_sha(_normalize(instruction).encode()))
    clauses, program = meta.get("clauses"), meta.get("program")
    if isinstance(clauses, list) and clauses and isinstance(program, dict):
        clause_fps = [_fingerprint(clause) for clause in clauses]
        result["clause"].update(clause_fps)
        result["numeric_signature"].update(_numeric_signature(clause) for clause in clauses)
        result["ast"].add(_fingerprint({"operator": program.get("operator"),
                                        "composition": program.get("composition"),
                                        "clauses": clauses}))
        result["multiset"].add(_fingerprint(sorted(clause_fps)))
    for key in ("scenario_id", "semantic_cluster_id", "source_id", "twin_id", "parent_view_id"):
        if meta.get(key) is not None:
            result["scenario_lineage"].add(_fingerprint([key, meta[key]]))
    if meta.get("cluster_lineage") is not None:
        result["scenario_lineage"].add(_fingerprint(meta["cluster_lineage"]))
    for key in ("renderer", "template_id", "rendered_contract"):
        if meta.get(key) is not None:
            result["template"].add(_fingerprint([key, meta[key]]))
    for key in ("generator_family_id", "generator", "source"):
        if meta.get(key) is not None:
            result["generator"].add(_fingerprint([key, meta[key]]))


def _add_decisive_pairs(result, rows):
    for row in rows:
        views = row.get("paired_views") if isinstance(row, dict) else None
        if isinstance(views, list) and len(views) == 2:
            multisets = []
            for view in views:
                clauses = (view.get("meta") or {}).get("clauses")
                if isinstance(clauses, list) and clauses:
                    multisets.append(_fingerprint(sorted(_fingerprint(clause) for clause in clauses)))
            if len(multisets) == 2:
                result["decisive_pair"].add(_fingerprint(sorted(multisets)))


def _panel_fingerprints(rows):
    result = _sets()
    for view in _iter_views(rows):
        _add_view(result, view)
    by_cluster = defaultdict(dict)
    for row in rows:
        meta = row["meta"]
        if meta["prediction_role"] == "final" and meta["language"] == "en" and meta["option_order"] == "yes_no":
            by_cluster[meta["semantic_cluster_id"]][tuple(meta["truth_pattern"])] = meta["clauses"]
    for cube in by_cluster.values():
        arity = len(next(iter(cube)))
        all_true = tuple([True] * arity)
        true_fp = _fingerprint(sorted(_fingerprint(clause) for clause in cube[all_true]))
        for index in range(arity):
            pattern = list(all_true)
            pattern[index] = False
            false_fp = _fingerprint(sorted(_fingerprint(clause) for clause in cube[tuple(pattern)]))
            result["decisive_pair"].add(_fingerprint(sorted((true_fp, false_fp))))
    return result


def _assert_unique_clause_ownership(rows):
    """Independently reject executable clause reuse across semantic clusters."""
    owners = defaultdict(set)
    kinds = {}
    for row in rows:
        meta = row.get("meta") or {}
        cluster_id = meta.get("semantic_cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError("causal panel row is missing semantic-cluster ownership")
        for clause in meta.get("clauses") or []:
            fingerprint = _fingerprint(clause)
            owners[fingerprint].add(cluster_id)
            kinds[fingerprint] = clause.get("kind")
    collisions = {fingerprint: values for fingerprint, values in owners.items() if len(values) > 1}
    if collisions:
        raise ValueError(
            f"executable clause ownership collision: {len(collisions)} fingerprints; "
            f"max owners={max(map(len, collisions.values()))}")
    return {"unique_clause_fingerprints": len(owners),
            "unique_clause_fingerprints_by_kind": dict(sorted(
                Counter(kinds.values()).items())),
            "cross_cluster_collisions": 0,
            "max_cluster_owners_per_fingerprint": max(map(len, owners.values()), default=0)}


def _inventory_fingerprints(rows):
    result = _sets()
    views = list(_iter_views(rows))
    for view in views:
        _add_view(result, view)
    # Public benchmark formats often use singular ``question``/``expected``
    # fields rather than trainer ``questions``/``targets``.  Workflow
    # inventories can be nested JSON.  Conservatively fingerprint those
    # surfaces too; this traversal never copies values into the report.
    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        state = value.get("state")
        if isinstance(state, (str, dict, list)):
            raw = state if isinstance(state, str) else _canonical(state)
            result["raw_state"].add(_sha(raw.encode()))
            result["normalized_state"].add(_sha(_normalize(raw).encode()))
        question = value.get("question")
        if isinstance(question, dict):
            result["raw_question"].add(_fingerprint(question))
            result["normalized_question"].add(_sha(_normalize(question).encode()))
        instruction = value.get("instructions")
        if isinstance(instruction, str):
            result["raw_instruction"].add(_sha(instruction.encode()))
            result["normalized_instruction"].add(_sha(_normalize(instruction).encode()))
        for item in value.values():
            if isinstance(item, (dict, list)):
                visit(item)
    visit(rows)
    if not views and not any(result[field] for field in (
            "raw_state", "raw_question", "raw_instruction")):
        raise ValueError("inventory entry contains no auditable decision surfaces")
    _add_decisive_pairs(result, rows)
    return result, len(views)


def _merge(target, source):
    for field in FINGERPRINT_FIELDS:
        target[field].update(source[field])


def _inventory_files(path):
    if path.is_file():
        return [path], _sha(path.read_bytes())
    if not path.is_dir():
        raise ValueError("inventory path is unavailable")
    files = sorted(item for item in path.rglob("*")
                   if item.is_file() and item.suffix.lower() in {".json", ".jsonl"})
    if not files:
        raise ValueError("inventory directory contains no JSON/JSONL files")
    payload = "".join(f"{item.relative_to(path).as_posix()}\0{_sha(item.read_bytes())}\n"
                      for item in files).encode()
    return files, _sha(payload)


def _load_inventory_manifest(path, expected_sha256):
    path = Path(path)
    payload = path.read_bytes()
    actual = _sha(payload)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "") or actual != expected_sha256:
        raise ValueError("inventory-manifest hash mismatch")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid inventory manifest") from error
    if manifest.get("version") != 1:
        raise ValueError("unsupported inventory manifest version")
    training, benchmarks = manifest.get("training"), manifest.get("benchmarks")
    if not isinstance(training, list) or not training or not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("inventory manifest requires non-empty training and benchmark lists")
    if not REQUIRED_TRAINING_LINEAGES <= {item.get("lineage") for item in training}:
        raise ValueError("inventory manifest must cover r044, r045, and r047 training")
    return manifest, actual, path.parent


def _load_inventory_entries(entries, base, benchmark=False):
    combined, receipts = _sets(), []
    for item in entries:
        expected = item.get("sha256") if isinstance(item, dict) else None
        path_value = item.get("path") if isinstance(item, dict) else None
        if not re.fullmatch(r"[0-9a-f]{64}", expected or "") or not isinstance(path_value, str):
            raise ValueError("every inventory entry requires a path and lowercase SHA-256")
        path = Path(path_value)
        if not path.is_absolute():
            path = base / path
        files, actual = _inventory_files(path)
        if actual != expected:
            kind = "benchmark" if benchmark else "training"
            raise ValueError(f"{kind} inventory hash mismatch")
        rows = [row for source in files for row in _read_jsonl(source)]
        fingerprints, views = _inventory_fingerprints(rows)
        _merge(combined, fingerprints)
        receipt = {"sha256": actual, "files": len(files), "rows": len(rows), "views": views}
        if not benchmark:
            receipt["lineage"] = item["lineage"]
        receipts.append(receipt)
    return combined, receipts


def _reject_overlap(panel, inventory, scope):
    counts = {}
    for field in FINGERPRINT_FIELDS:
        count = len(panel[field] & inventory[field])
        counts[field] = count
        if count:
            raise ValueError(f"{scope} leakage: {field} overlap count={count}")
    return counts


def audit(panel_dir, inventory_manifest, expected_inventory_manifest_sha256,
          out=None, require_frozen=True):
    panel_dir = Path(panel_dir)
    manifest_path = panel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != "r047_primary_causal_panel" or manifest.get("version") != 1:
        raise ValueError("wrong panel manifest")
    if require_frozen and (not manifest.get("contract", {}).get("frozen_production")
                           or manifest.get("counts_per_composition") != FROZEN_COUNTS
                           or manifest.get("clusters") != {"unseen_k2": 312, "k3": 320,
                                                            "seen_k2_near_transfer": 48,
                                                            "causal_primary": 632}):
        raise ValueError("panel does not match the frozen production count contract")
    rows_by_name = {}
    for name in ("primary", "near-transfer", "legacy-smoke"):
        spec = manifest.get("outputs", {}).get(name) or {}
        path = panel_dir / spec.get("file", "")
        if not path.is_file() or _sha(path.read_bytes()) != spec.get("sha256"):
            raise ValueError(f"{name}: panel output hash mismatch")
        rows = _read_jsonl(path)
        if len(rows) != spec.get("rows"):
            raise ValueError(f"{name}: panel row count mismatch")
        rows_by_name[name] = rows
    primary = _validate_causal(rows_by_name["primary"], manifest, "primary")
    near = _validate_causal(rows_by_name["near-transfer"], manifest, "near-transfer")
    legacy = _validate_legacy(rows_by_name["legacy-smoke"])

    inventory, inventory_sha, inventory_base = _load_inventory_manifest(
        inventory_manifest, expected_inventory_manifest_sha256)
    training_fp, training_receipts = _load_inventory_entries(
        inventory["training"], inventory_base, benchmark=False)
    benchmark_fp, benchmark_receipts = _load_inventory_entries(
        inventory["benchmarks"], inventory_base, benchmark=True)
    causal_rows = rows_by_name["primary"] + rows_by_name["near-transfer"]
    clause_ownership = _assert_unique_clause_ownership(causal_rows)
    if manifest.get("clause_ownership") != clause_ownership:
        raise ValueError("manifest executable-clause ownership receipt mismatch")
    cluster_ordinals = {}
    for row in causal_rows:
        meta = row["meta"]
        cluster_ordinals.setdefault(meta["semantic_cluster_id"], meta["cluster_ordinal"])
        if cluster_ordinals[meta["semantic_cluster_id"]] != meta["cluster_ordinal"]:
            raise ValueError("semantic cluster has inconsistent global ordinal")
    ordinals = list(cluster_ordinals.values())
    if (any(not isinstance(value, int) or isinstance(value, bool) for value in ordinals)
            or len(set(ordinals)) != len(ordinals) or set(ordinals) != set(range(len(ordinals)))):
        raise ValueError("semantic-cluster ordinals are not globally unique and contiguous")
    panel_fp = _panel_fingerprints(causal_rows)
    training_overlap = _reject_overlap(panel_fp, training_fp, "training")
    benchmark_overlap = _reject_overlap(panel_fp, benchmark_fp, "benchmark")

    report = {
        "experiment": "r047_primary_panel_preflight",
        "status": "pass",
        "panel_manifest_sha256": _sha(manifest_path.read_bytes()),
        "builder_sha256": manifest["builder_sha256"],
        "preflight_sha256": _sha(Path(__file__).read_bytes()),
        "inventory_manifest_sha256": inventory_sha,
        "outputs": {name: {"sha256": manifest["outputs"][name]["sha256"],
                           "rows": manifest["outputs"][name]["rows"]}
                    for name in ("primary", "near-transfer", "legacy-smoke")},
        "causal": {"primary": primary, "near_transfer": near},
        "clause_ownership": clause_ownership,
        "legacy_smoke": legacy,
        "training_inventories": training_receipts,
        # Deliberately omit benchmark path/name/content. Hashes are sufficient receipts.
        "benchmark_inventories": benchmark_receipts,
        "overlap": {"training": training_overlap, "benchmarks": benchmark_overlap},
        "benchmark_content_emitted": False,
    }
    if out:
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--inventory-manifest", required=True)
    parser.add_argument("--expected-inventory-manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = audit(args.panel_dir, args.inventory_manifest,
                   args.expected_inventory_manifest_sha256, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
