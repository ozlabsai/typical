"""Evaluate executable clause decisions and their final-decision composition."""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


CAUSAL_SCHEMA_VERSION = "r047-causal-prediction-v1"
FROZEN_CAUSAL_COUNTS = {"unseen_k2": 24, "k3": 16, "seen_k2": 24}
FROZEN_CAUSAL_CLUSTERS = {"unseen_k2": 312, "k3": 320,
                          "seen_k2_near_transfer": 48, "causal_primary": 632}
RUN_RECEIPT_FIELDS = {
    "schema_version", "slot", "train_sha256", "recipe_sha256",
    "parent_adapter_sha256", "model_revision", "adapter_sha256", "config_sha256",
    "metrics_sha256", "log_sha256", "model_inventory_sha256",
    "checkpoint_payload_sha256", "checkpoint_step", "status",
}
CAUSAL_META_FIELDS = (
    "semantic_cluster_id", "semantic_view_id", "scenario_id", "composition", "arity",
    "language", "option_order", "prompt_family", "prediction_role", "primitive",
    "decisive_role", "truth_hamming_weight", "truth_pattern", "seen_edge_count",
)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_causal_run_receipt(payload, adapter_sha256=None):
    """Validate a post-training receipt that binds one named slot to its output adapter."""
    try:
        receipt = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid causal run receipt") from error
    if (set(receipt) != RUN_RECEIPT_FIELDS
            or receipt.get("schema_version") != "r047-causal-run-receipt-v1"
            or receipt.get("slot") not in {"slot_a", "slot_b"}
            or receipt.get("status") != "complete"
            or not isinstance(receipt.get("checkpoint_step"), int)
            or isinstance(receipt.get("checkpoint_step"), bool)
            or receipt["checkpoint_step"] < 0):
        raise ValueError("causal run receipt schema/status is invalid")
    hash_fields = RUN_RECEIPT_FIELDS - {
        "schema_version", "slot", "model_revision", "checkpoint_step", "status"}
    if any(not _valid_sha256(receipt[field]) for field in hash_fields):
        raise ValueError("causal run receipt contains an invalid SHA-256")
    if not isinstance(receipt["model_revision"], str) or not receipt["model_revision"]:
        raise ValueError("causal run receipt model revision is invalid")
    if adapter_sha256 is not None and receipt["adapter_sha256"] != adapter_sha256:
        raise ValueError("causal run receipt adapter differs from loaded adapter")
    return receipt


def validate_causal_model_inventory(model_path, payload, model_revision):
    """Verify the loaded base-model tree against one sealed immutable inventory."""
    from scripts.preflight_r043_assets import tree_hash
    try:
        inventory = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid causal model inventory") from error
    if (set(inventory) != {"schema_version", "model_id", "revision", "tree_sha256", "files"}
            or inventory.get("schema_version") != "r047-model-inventory-v1"
            or inventory.get("model_id") != "google/gemma-4-12B-it"
            or inventory.get("revision") != model_revision
            or not _valid_sha256(inventory.get("tree_sha256"))
            or not isinstance(inventory.get("files"), int)
            or isinstance(inventory.get("files"), bool) or inventory["files"] <= 0):
        raise ValueError("causal model inventory schema/revision is invalid")
    observed_hash, observed_files = tree_hash(model_path)
    if (inventory["tree_sha256"] != observed_hash
            or inventory["files"] != observed_files):
        raise ValueError("loaded base-model tree differs from sealed inventory")
    return inventory


def validate_causal_panel_payload(panel_payload, manifest_payload, require_frozen=True):
    """Bind evaluation to the declared primary panel and, by default, its frozen contract."""
    try:
        rows = [json.loads(line) for line in panel_payload.decode().splitlines() if line.strip()]
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid causal panel or manifest JSON") from error
    if not isinstance(manifest, dict) or manifest.get("experiment") != "r047_primary_causal_panel":
        raise ValueError("wrong causal panel manifest")
    output = (manifest.get("outputs") or {}).get("primary") or {}
    panel_sha = hashlib.sha256(panel_payload).hexdigest()
    if (output.get("file") != "primary.jsonl" or output.get("sha256") != panel_sha
            or output.get("rows") != len(rows)):
        raise ValueError("primary panel payload disagrees with manifest")
    if require_frozen:
        if (manifest.get("contract", {}).get("frozen_production") is not True
                or manifest.get("counts_per_composition") != FROZEN_CAUSAL_COUNTS
                or manifest.get("clusters") != FROZEN_CAUSAL_CLUSTERS
                or len(manifest.get("compositions", {}).get("unseen_k2", [])) != 13
                or len(manifest.get("compositions", {}).get("k3", [])) != 20):
            raise ValueError("causal panel manifest is not the exact frozen production contract")
    clusters = defaultdict(list)
    for row in rows:
        _causal_row_metadata(row)
        meta = row["meta"]
        if meta.get("panel_stratum") not in {"unseen_k2", "k3"}:
            raise ValueError("primary panel contains a non-primary stratum")
        clusters[meta["semantic_cluster_id"]].append(row)
    by_stratum_composition = Counter()
    for cluster_id, items in clusters.items():
        signatures = {(item["meta"]["panel_stratum"],
                       tuple(item["meta"]["composition"])) for item in items}
        if len(signatures) != 1:
            raise ValueError(f"{cluster_id}: panel cluster lineage drift")
        by_stratum_composition[next(iter(signatures))] += 1
    expected_compositions = {
        stratum: {tuple(value) for value in manifest.get("compositions", {}).get(stratum, [])}
        for stratum in ("unseen_k2", "k3")}
    for stratum, compositions in expected_compositions.items():
        expected_count = manifest.get("counts_per_composition", {}).get(stratum)
        if not isinstance(expected_count, int) or expected_count <= 0:
            raise ValueError("invalid panel composition count contract")
        observed = {composition: count for (name, composition), count
                    in by_stratum_composition.items() if name == stratum}
        if set(observed) != compositions or any(count != expected_count
                                                for count in observed.values()):
            raise ValueError(f"{stratum}: panel composition coverage differs from manifest")
    expected_primary = manifest.get("clusters", {}).get("causal_primary")
    if len(clusters) != expected_primary:
        raise ValueError("primary semantic-cluster count differs from manifest")
    return rows, manifest


def _causal_row_metadata(row):
    """Validate and normalize the explicit metadata on one causal-panel question."""
    row_id = row.get("id")
    meta = row.get("meta")
    if not isinstance(row_id, str) or not row_id or not isinstance(meta, dict):
        raise ValueError("causal rows need a non-empty ID and metadata")
    missing = [field for field in CAUSAL_META_FIELDS if field not in meta]
    if missing:
        raise ValueError(f"{row_id}: causal metadata missing {', '.join(missing)}")
    questions, targets = row.get("questions"), row.get("targets")
    if (not isinstance(questions, dict) or len(questions) != 1
            or set(questions) != set(targets or {})):
        raise ValueError(f"{row_id}: causal row must contain one aligned question and target")
    qid = next(iter(questions))
    question = questions[qid]
    if question.get("type") != "choice" or set(question.get("criteria") or {}) != {"no", "yes"}:
        raise ValueError(f"{row_id}: causal question must expose semantic no/yes options")
    option_order = meta["option_order"]
    expected_order = (["yes", "no"] if option_order == "yes_no" else
                      ["no", "yes"] if option_order == "no_yes" else None)
    if expected_order is None or list(question["criteria"]) != expected_order:
        raise ValueError(f"{row_id}: option-order metadata differs from the rendered question")
    composition = meta["composition"]
    if (not isinstance(composition, list) or len(composition) not in (2, 3)
            or len(set(composition)) != len(composition)
            or meta["arity"] != len(composition)):
        raise ValueError(f"{row_id}: invalid causal composition/arity")
    clauses = meta.get("clauses")
    if (not isinstance(clauses, list) or len(clauses) != len(composition)
            or [clause.get("kind") for clause in clauses] != composition):
        raise ValueError(f"{row_id}: executable clauses do not match composition")
    by_kind = {clause["kind"]: clause for clause in clauses}
    truth_pattern = [bool(execute_clause(by_kind[kind])) for kind in composition]
    if execute_program(meta.get("program") or {}, clauses) is not all(truth_pattern):
        raise ValueError(f"{row_id}: program result disagrees with executable clauses")
    if meta["truth_pattern"] != truth_pattern:
        raise ValueError(f"{row_id}: truth pattern disagrees with executable clauses")
    if meta["truth_hamming_weight"] != sum(truth_pattern):
        raise ValueError(f"{row_id}: truth Hamming weight disagrees with executable clauses")
    if meta["language"] not in {"en", "he"}:
        raise ValueError(f"{row_id}: unsupported causal language")
    for field in ("semantic_cluster_id", "semantic_view_id", "scenario_id", "prompt_family"):
        if not isinstance(meta[field], str) or not meta[field]:
            raise ValueError(f"{row_id}: {field} must be a non-empty string")
    if (not isinstance(meta["seen_edge_count"], int) or isinstance(meta["seen_edge_count"], bool)
            or not 0 <= meta["seen_edge_count"] <= len(composition)):
        raise ValueError(f"{row_id}: invalid seen-edge count")
    role, primitive = meta["prediction_role"], meta["primitive"]
    if role == "final":
        if primitive is not None or meta["decisive_role"] != "final":
            raise ValueError(f"{row_id}: final row must use primitive=null and decisive_role=final")
        expected = all(truth_pattern)
    elif role == "clause":
        if primitive not in composition or meta["decisive_role"] not in {
                "decisive", "non_decisive"}:
            raise ValueError(f"{row_id}: invalid clause primitive/decisive role")
        primitive_index = composition.index(primitive)
        expected_role = ("decisive" if all(
            value for index, value in enumerate(truth_pattern) if index != primitive_index)
                         else "non_decisive")
        if meta["decisive_role"] != expected_role:
            raise ValueError(f"{row_id}: decisive role disagrees with truth pattern")
        expected = truth_pattern[primitive_index]
    else:
        raise ValueError(f"{row_id}: prediction_role must be final or clause")
    target = targets[qid].get("label")
    if target != ("yes" if expected else "no"):
        raise ValueError(f"{row_id}: target disagrees with executable causal gold")
    if meta.get("gold") is not expected:
        raise ValueError(f"{row_id}: metadata gold disagrees with executable causal gold")
    return {field: copy.deepcopy(meta[field]) for field in CAUSAL_META_FIELDS}, target


def seal_causal_predictions(rows, predictions):
    """Bind model outputs to all causal-panel semantics and add per-record integrity seals."""
    if not rows:
        raise ValueError("causal panel is empty")
    row_ids = [row.get("id") for row in rows]
    prediction_ids = [item.get("id") for item in predictions]
    if None in row_ids or len(row_ids) != len(set(row_ids)):
        raise ValueError("causal row IDs must be present and unique")
    if None in prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("causal prediction IDs must be present and unique")
    if len(rows) != len(predictions) or set(row_ids) != set(prediction_ids):
        raise ValueError("causal predictions must cover every row exactly once")
    by_id = {item["id"]: item for item in predictions}
    sealed = []
    for row in rows:
        metadata, gold = _causal_row_metadata(row)
        no, yes = _probabilities(by_id[row["id"]], row["id"])
        record = {
            "schema_version": CAUSAL_SCHEMA_VERSION,
            "id": row["id"],
            "row_sha256": hashlib.sha256(_canonical_json(row).encode()).hexdigest(),
            "gold": gold,
            "probabilities": {"no": no, "yes": yes},
            **metadata,
        }
        record["record_sha256"] = hashlib.sha256(_canonical_json(record).encode()).hexdigest()
        sealed.append(record)
    return sealed


def validate_sealed_causal_predictions(records):
    """Fail closed on malformed, incomplete, or tampered causal prediction records."""
    if not isinstance(records, list) or not records:
        raise ValueError("sealed causal predictions are empty")
    ids = [record.get("id") for record in records]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("sealed causal prediction IDs must be present and unique")
    expected = {"schema_version", "id", "row_sha256", "gold", "probabilities", "record_sha256",
                *CAUSAL_META_FIELDS}
    for record in records:
        if set(record) != expected or record["schema_version"] != CAUSAL_SCHEMA_VERSION:
            raise ValueError(f"{record.get('id')}: invalid sealed causal schema")
        seal = record["record_sha256"]
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        if seal != hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest():
            raise ValueError(f"{record.get('id')}: causal prediction seal mismatch")
        if (not isinstance(record["row_sha256"], str) or len(record["row_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in record["row_sha256"])):
            raise ValueError(f"{record.get('id')}: invalid causal source-row hash")
        _probabilities(record, record["id"])
        if record["gold"] not in {"no", "yes"}:
            raise ValueError(f"{record['id']}: invalid sealed gold")
        composition = record["composition"]
        pattern = record["truth_pattern"]
        if (not isinstance(composition, list) or record["arity"] != len(composition)
                or len(composition) not in (2, 3) or len(pattern) != len(composition)
                or len(set(composition)) != len(composition)
                or any(type(value) is not bool for value in pattern)
                or record["truth_hamming_weight"] != sum(pattern)):
            raise ValueError(f"{record['id']}: invalid sealed truth metadata")
        if (record["language"] not in {"en", "he"}
                or record["option_order"] not in {"no_yes", "yes_no"}
                or not isinstance(record["seen_edge_count"], int)
                or isinstance(record["seen_edge_count"], bool)
                or not 0 <= record["seen_edge_count"] <= len(composition)
                or any(not isinstance(record[field], str) or not record[field]
                       for field in ("semantic_cluster_id", "semantic_view_id", "scenario_id",
                                     "prompt_family"))):
            raise ValueError(f"{record['id']}: invalid sealed causal metadata")
        role, primitive = record["prediction_role"], record["primitive"]
        if role == "final":
            expected_gold = "yes" if all(pattern) else "no"
            if primitive is not None or record["decisive_role"] != "final":
                raise ValueError(f"{record['id']}: invalid sealed final metadata")
        elif role == "clause" and primitive in composition:
            primitive_index = composition.index(primitive)
            expected_role = ("decisive" if all(
                value for index, value in enumerate(pattern) if index != primitive_index)
                             else "non_decisive")
            if record["decisive_role"] != expected_role:
                raise ValueError(f"{record['id']}: invalid sealed decisive role")
            expected_gold = "yes" if pattern[primitive_index] else "no"
        else:
            raise ValueError(f"{record['id']}: invalid sealed clause metadata")
        if record["gold"] != expected_gold:
            raise ValueError(f"{record['id']}: sealed gold disagrees with truth pattern")
    return records


def causal_coverage_report(records):
    """Summarize every bound causal dimension without replacing metric recomputation."""
    validate_sealed_causal_predictions(records)
    dimensions = {}
    for field in CAUSAL_META_FIELDS:
        values = Counter(_canonical_json(item[field]) for item in records)
        dimensions[field] = {key: values[key] for key in sorted(values)}
    projection = [{field: item[field]
                   for field in ("id", "row_sha256", "gold", *CAUSAL_META_FIELDS)}
                  for item in records]
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "records": len(records),
        "semantic_clusters": len({item["semantic_cluster_id"] for item in records}),
        "scenarios": len({item["scenario_id"] for item in records}),
        "dimensions": dimensions,
        "metadata_sha256": hashlib.sha256(_canonical_json(projection).encode()).hexdigest(),
    }


def _probabilities(item, row_id):
    values = item.get("probabilities")
    if not isinstance(values, dict) or set(values) != {"no", "yes"}:
        raise ValueError(f"{row_id}: prediction must contain exactly no/yes probabilities")
    no, yes = (float(values[label]) for label in ("no", "yes"))
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (no, yes)):
        raise ValueError(f"{row_id}: invalid probability")
    if not math.isclose(no + yes, 1, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{row_id}: probabilities must sum to one")
    return no, yes


def execute_clause(clause):
    """Independent executable gold; intentionally does not import the data builder."""
    kind = clause["kind"]
    if kind == "decimal":
        net = sum((Decimal(value) for value in clause["credits"]), Decimal("0"))
        return net - Decimal(clause["fee"]) >= Decimal(clause["threshold"])
    if kind == "fraction":
        return Fraction(clause["numerator"], clause["denominator"]) >= Fraction(*clause["threshold"])
    if kind == "datetime":
        local = datetime.fromisoformat(clause["local_time"]).replace(
            tzinfo=ZoneInfo(clause["timezone"]))
        deadline = datetime.fromisoformat(clause["deadline"]).replace(tzinfo=timezone.utc)
        return local.astimezone(timezone.utc) <= deadline
    if kind == "ledger":
        balance = sum((Decimal(event["delta"]) for event in clause["events"]), Decimal("0"))
        return balance >= Decimal(clause["threshold"])
    if kind == "conditional":
        target = next(group for group in clause["groups"]
                      if group["name"] == clause["condition_group"])
        probability = Fraction(target["success"], target["success"] + target["failure"])
        return probability >= Fraction(*clause["threshold"])
    if kind == "policy":
        applicable = [rule for rule in clause["rules"] if rule["active"] and
                      rule["scope"] in {"all", clause["request_scope"]}]
        winner = max(applicable, key=lambda rule: (rule["priority"], rule["effect"] == "deny"))
        return winner["effect"] == "allow"
    raise ValueError(f"unknown clause kind: {kind}")


def execute_program(program, clauses):
    by_kind = {clause["kind"]: clause for clause in clauses}
    if set(by_kind) != set(program.get("composition", [])):
        raise ValueError("program and evidence composition differ")
    return all(execute_clause(by_kind[kind]) for kind in program["composition"])


def clause_view(view, index):
    """Project one executable clause into the unchanged decision-serving schema."""
    meta = view.get("meta") or {}
    clauses = meta.get("clauses")
    if not isinstance(clauses, list) or not 0 <= index < len(clauses):
        raise ValueError(f"{view.get('id')}: missing executable clause {index}")
    clause = clauses[index]
    language = meta.get("language")
    if language not in {"en", "he"}:
        raise ValueError(f"{view.get('id')}: unsupported language")
    question = view.get("questions", {}).get("decision")
    if not isinstance(question, dict) or not isinstance(question.get("criteria"), dict):
        raise ValueError(f"{view.get('id')}: malformed decision question")
    instructions = (f"Evaluate only numbered condition {index + 1}. Does it satisfy its stated requirement?"
                    if language == "en" else
                    f"יש להעריך רק את תנאי מספר {index + 1}. האם הוא עומד בדרישה המוצהרת שלו?")
    result = copy.deepcopy(view)
    result["id"] = f"{view['id']}::clause-{index + 1}"
    result["questions"] = {"decision": {"type": "choice", "instructions": instructions,
                                          "criteria": copy.deepcopy(question["criteria"])}}
    result["targets"] = {"decision": {"label": "yes" if execute_clause(clause) else "no"}}
    result["meta"].update({"r047_parent_view_id": view["id"],
                           "r047_clause_index": index,
                           "r047_clause_kind": clause["kind"]})
    return result


def row_clause_views(row):
    """Return exact training auxiliaries when present; otherwise project evaluation views."""
    canonical = row.get("paired_views") or []
    auxiliaries = row.get("auxiliary_views")
    if auxiliaries is None:
        return {view["id"]: [clause_view(view, index)
                              for index in range(len(view["meta"]["clauses"]))]
                for view in canonical}
    if not isinstance(auxiliaries, list):
        raise ValueError(f"{row.get('id')}: auxiliary_views must be a list")
    by_parent = defaultdict(dict)
    canonical_by_id = {view["id"]: view for view in canonical}
    for auxiliary in auxiliaries:
        provenance = (auxiliary.get("meta") or {}).get("r047_auxiliary") or {}
        parent_id = provenance.get("parent_view_id")
        index = provenance.get("clause_index")
        if parent_id not in canonical_by_id or not isinstance(index, int):
            raise ValueError(f"{auxiliary.get('id')}: invalid auxiliary parent/index")
        parent = canonical_by_id[parent_id]
        clauses = parent["meta"]["clauses"]
        if not 0 <= index < len(clauses) or index in by_parent[parent_id]:
            raise ValueError(f"{auxiliary.get('id')}: duplicate or out-of-range clause index")
        if auxiliary.get("state") != parent.get("state"):
            raise ValueError(f"{auxiliary.get('id')}: auxiliary state differs from parent")
        if provenance.get("role") != "auxiliary" or provenance.get("clause_kind") != clauses[index]["kind"]:
            raise ValueError(f"{auxiliary.get('id')}: invalid auxiliary provenance")
        question = auxiliary.get("questions", {}).get("decision")
        parent_question = parent.get("questions", {}).get("decision")
        if not question or not parent_question or question.get("criteria") != parent_question.get("criteria"):
            raise ValueError(f"{auxiliary.get('id')}: auxiliary options differ from parent")
        expected = "yes" if execute_clause(clauses[index]) else "no"
        if auxiliary.get("targets", {}).get("decision", {}).get("label") != expected:
            raise ValueError(f"{auxiliary.get('id')}: auxiliary target disagrees with executable gold")
        by_parent[parent_id][index] = auxiliary
    result = {}
    for parent in canonical:
        expected = len(parent["meta"]["clauses"])
        if set(by_parent[parent["id"]]) != set(range(expected)):
            raise ValueError(f"{parent['id']}: auxiliary coverage is incomplete")
        result[parent["id"]] = [by_parent[parent["id"]][index] for index in range(expected)]
    return result


def _score(item, gold, row_id):
    no, yes = _probabilities(item, row_id)
    predicted = "yes" if yes > no else "no" if no > yes else "tie"
    probability = yes if gold == "yes" else no
    return {"gold": gold, "predicted": predicted, "correct": predicted == gold,
            "yes_logit": math.log(max(yes, 1e-12) / max(no, 1e-12)),
            "gold_probability": probability, "confidence": max(no, yes),
            "nll": -math.log(max(probability, 1e-12)),
            "brier": (yes - float(gold == "yes")) ** 2}


def _rate(values):
    return sum(values) / len(values) if values else None


def _balanced_accuracy(items):
    recalls = []
    for label in ("no", "yes"):
        group = [item for item in items if item["gold"] == label]
        if group:
            recalls.append(_rate([item["correct"] for item in group]))
    return _rate(recalls)


def _p05(values):
    values = sorted(values)
    return values[max(0, math.ceil(0.05 * len(values)) - 1)] if values else None


def _ece(items, bins=10):
    total, error = len(items), 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        group = [item for item in items if lower <= item["confidence"]
                 and (item["confidence"] < upper or index == bins - 1)]
        if group:
            error += len(group) / total * abs(
                _rate([item["confidence"] for item in group])
                - _rate([item["correct"] for item in group]))
    return error


def _clause_summary(items):
    no_items = [item for item in items if item["gold"] == "no"]
    yes_items = [item for item in items if item["gold"] == "yes"]
    gold_probabilities = [item["gold_probability"] for item in items]
    return {"n": len(items), "accuracy": _rate([item["correct"] for item in items]),
            "balanced_accuracy": _balanced_accuracy(items),
            "no_n": len(no_items), "yes_n": len(yes_items),
            "tn": sum(item["predicted"] == "no" for item in no_items),
            "fp": sum(item["predicted"] != "no" for item in no_items),
            "tp": sum(item["predicted"] == "yes" for item in yes_items),
            "fn": sum(item["predicted"] != "yes" for item in yes_items),
            "tnr": _rate([item["correct"] for item in no_items]),
            "tpr": _rate([item["correct"] for item in yes_items]),
            "nll": _rate([item["nll"] for item in items]),
            "brier": _rate([item["brier"] for item in items]),
            "ece_10bin": _ece(items),
            "gold_probability_mean": _rate(gold_probabilities),
            "gold_probability_p05": _p05(gold_probabilities),
            "gold_probability_min": min(gold_probabilities)}


def _final_summary(items):
    return {"presentations": len(items),
            "accuracy": _rate([item["correct"] for item in items]),
            "balanced_accuracy": _balanced_accuracy(items),
            "nll": _rate([item["nll"] for item in items]),
            "brier": _rate([item["brier"] for item in items]),
            "ece_10bin": _ece(items)}


def audit(rows, predictions, min_clauses=1):
    """Fail closed and summarize final, per-clause, vector, and coherence metrics."""
    selected = [row for row in rows if len((row.get("meta") or {}).get("composition", []))
                >= min_clauses]
    if not selected:
        raise ValueError("no rows satisfy min_clauses")
    expected = []
    for row in selected:
        views = row.get("paired_views")
        if row.get("pair_mode") != "counterfactual" or not isinstance(views, list) or len(views) != 2:
            raise ValueError(f"{row.get('id')}: expected one two-view counterfactual pair")
        projected = row_clause_views(row)
        for view in views:
            expected.append(view["id"])
            expected.extend(item["id"] for item in projected[view["id"]])
    prediction_ids = [item.get("id") for item in predictions]
    if None in prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction IDs must be present and unique")
    if set(prediction_ids) != set(expected) or len(prediction_ids) != len(expected):
        raise ValueError("predictions must cover every final and clause view exactly once")
    by_id = {item["id"]: item for item in predictions}

    finals, clauses, pairs, vectors, coherences = [], [], [], [], []
    for row in selected:
        pair = []
        projected_by_parent = row_clause_views(row)
        for view in row["paired_views"]:
            meta = view["meta"]
            final_value = execute_program(meta["program"], meta["clauses"])
            final_gold = view["targets"]["decision"]["label"]
            if final_gold != ("yes" if final_value else "no") or final_value is not meta.get("gold"):
                raise ValueError(f"{view['id']}: final target disagrees with executable gold")
            final = _score(by_id[view["id"]], final_gold, view["id"])
            final.update({"id": view["id"], "language": meta["language"],
                          "decisive_kind": meta["decisive_kind"]})
            finals.append(final)
            pair.append(final)
            view_clauses = []
            for index, (clause, projected) in enumerate(zip(
                    meta["clauses"], projected_by_parent[view["id"]])):
                clause_gold = projected["targets"]["decision"]["label"]
                item = _score(by_id[projected["id"]], clause_gold, projected["id"])
                item.update({"id": projected["id"], "parent_id": view["id"],
                             "language": meta["language"], "kind": clause["kind"],
                             "role": ("decisive" if clause["kind"] == meta["decisive_kind"]
                                      else "non_decisive")})
                clauses.append(item)
                view_clauses.append(item)
            vectors.append(all(item["correct"] for item in view_clauses))
            composed = "yes" if all(item["predicted"] == "yes" for item in view_clauses) else "no"
            coherences.append(final["predicted"] == composed)
        if {item["gold"] for item in pair} != {"no", "yes"}:
            raise ValueError(f"{row['id']}: final counterfactual labels must flip")
        positive = next(item for item in pair if item["gold"] == "yes")
        negative = next(item for item in pair if item["gold"] == "no")
        delta = positive["yes_logit"] - negative["yes_logit"]
        pairs.append({"direction": delta > 0,
                      "both_correct": all(item["correct"] for item in pair),
                      "delta": delta,
                      "language": pair[0]["language"],
                      "decisive_kind": pair[0]["decisive_kind"]})

    clause_groups = {}
    for key in ("kind", "language", "role"):
        groups = defaultdict(list)
        for item in clauses:
            groups[item[key]].append(item)
        clause_groups[key] = {name: _clause_summary(items) for name, items in sorted(groups.items())}
    pair_groups = {}
    for key in ("decisive_kind", "language"):
        groups = defaultdict(list)
        for item in pairs:
            groups[item[key]].append(item)
        pair_groups[key] = {name: {"pairs": len(items),
                                  "direction_rate": _rate([item["direction"] for item in items]),
                                  "pair_both_correct": _rate([item["both_correct"] for item in items]),
                                  "delta_mean": _rate([item["delta"] for item in items]),
                                  "delta_p05": _p05([item["delta"] for item in items]),
                                  "delta_min": min(item["delta"] for item in items),
                                  "margin_rate": _rate([item["delta"] >= 2 for item in items]),
                                  "mean_hinge": _rate([max(0, 2 - item["delta"])
                                                       for item in items])}
                            for name, items in sorted(groups.items())}
    deltas = [item["delta"] for item in pairs]
    result = {"rows": len(selected), "final": _final_summary(finals),
              "final_pairs": {"pairs": len(pairs),
                              "direction_rate": _rate([item["direction"] for item in pairs]),
                              "pair_both_correct": _rate([item["both_correct"] for item in pairs]),
                              "delta_mean": _rate(deltas), "delta_p05": _p05(deltas),
                              "delta_min": min(deltas),
                              "margin_rate": _rate([delta >= 2 for delta in deltas]),
                              "mean_hinge": _rate([max(0, 2 - delta) for delta in deltas]),
                              **pair_groups},
              "clauses": {**_clause_summary(clauses), **clause_groups,
                          "vectors": len(vectors), "vector_accuracy": _rate(vectors)},
              "coherence": {"n": len(coherences), "accuracy": _rate(coherences)}}
    return result


def m0_gate(report, required_kinds, threshold=0.98):
    """Machine-check the preregistered can-fit threshold without holdout selection."""
    failures = []
    clause = report["clauses"]
    pairs = report["final_pairs"]
    if clause["balanced_accuracy"] < threshold:
        failures.append("overall clause balanced accuracy")
    if pairs["direction_rate"] < threshold:
        failures.append("overall final direction")
    if report["final"]["balanced_accuracy"] < threshold:
        failures.append("overall final balanced accuracy")
    if pairs["pair_both_correct"] < threshold:
        failures.append("overall final pair-both correctness")
    for kind in required_kinds:
        if kind not in clause["kind"] or clause["kind"][kind]["balanced_accuracy"] < threshold:
            failures.append(f"{kind} clause balanced accuracy")
        if (kind not in pairs["decisive_kind"]
                or pairs["decisive_kind"][kind]["direction_rate"] < threshold):
            failures.append(f"{kind} final direction")
    return {"passed": not failures, "threshold": threshold,
            "required_kinds": list(required_kinds), "failures": failures}


def score_causal_rows_batched(model, tokenizer, groups, rows, device, max_length,
                              max_batch_tokens=32768, max_batch_rows=64):
    """Score standalone text rows in stable bounded batches using the legacy letter logits."""
    import torch
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pad_sequence
    from scripts.train_gemma_decision_lora import encode, flatten_row

    if max_batch_tokens <= 0 or max_batch_rows <= 0:
        raise ValueError("causal batch limits must be positive")
    predictions, batch = [], []

    def flush():
        if not batch:
            return
        ids = pad_sequence([item[1][0] for item in batch], batch_first=True,
                           padding_value=tokenizer.pad_token_id)
        mask = ids.ne(tokenizer.pad_token_id)
        positions = torch.tensor([len(item[1][0]) - 1 for item in batch])
        unique_positions = torch.unique(positions, sorted=True)
        output = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False,
                       logits_to_keep=unique_positions.to(device))
        offsets = {position.item(): index for index, position in enumerate(unique_positions)}
        for index, (row_id, (_, _, labels)) in enumerate(batch):
            logits = output.logits[index, offsets[positions[index].item()]].float()
            scores = torch.stack([torch.logsumexp(logits[token_ids.to(device)], dim=0)
                                  for token_ids in groups[:len(labels)]])
            values = functional.softmax(scores, dim=0).tolist()
            probabilities = {label: float(value) for label, value in zip(labels, values)}
            predictions.append({"id": row_id, "probabilities": probabilities,
                                "semantic_margin": math.log(
                                    max(probabilities["yes"], 1e-12)
                                    / max(probabilities["no"], 1e-12))})
        batch.clear()

    for row in rows:
        examples = flatten_row(row)
        if len(examples) != 1 or examples[0].get("image"):
            raise ValueError(f"{row.get('id')}: causal batched scorer needs one text question")
        encoded = encode(examples[0], tokenizer, max_length, shuffle=False, include_labels=True)
        proposed_rows = len(batch) + 1
        proposed_length = max([len(item[1][0]) for item in batch] + [len(encoded[0])])
        if batch and (proposed_rows > max_batch_rows
                      or proposed_rows * proposed_length > max_batch_tokens):
            flush()
        if len(encoded[0]) > max_batch_tokens:
            raise ValueError(
                f"{row.get('id')}: encoded length exceeds causal max-batch-tokens")
        batch.append((row["id"], encoded))
    flush()
    return predictions


def causal_parity_subset(rows):
    """Select the first frozen row in every arity/language/order/role/family/primitive cell."""
    selected, seen = [], set()
    for row in rows:
        meta = row["meta"]
        key = (meta["arity"], meta["language"], meta["option_order"],
               meta["prediction_role"], meta["prompt_family"], meta["primitive"])
        if key not in seen:
            seen.add(key)
            selected.append(row)
    return selected


def audit_causal_batch_parity(rows, batched, scalar, tolerance=1e-5):
    """Require actual-model batched scoring to preserve scalar decisions and probabilities."""
    if tolerance != 1e-5:
        raise ValueError("causal parity tolerance is frozen at 1e-5")
    expected_ids = [row["id"] for row in rows]
    if ([item.get("id") for item in batched] != expected_ids
            or [item.get("id") for item in scalar] != expected_ids):
        raise ValueError("causal parity predictions do not preserve frozen subset order")
    maximum = 0.0
    for left, right in zip(batched, scalar):
        left_no, left_yes = _probabilities(left, left["id"])
        right_no, right_yes = _probabilities(right, right["id"])
        left_label = "yes" if left_yes > left_no else "no" if left_no > left_yes else "tie"
        right_label = "yes" if right_yes > right_no else "no" if right_no > right_yes else "tie"
        if left_label != right_label:
            raise ValueError(f"{left['id']}: batched/scalar discrete prediction differs")
        maximum = max(maximum, abs(left_no - right_no), abs(left_yes - right_yes))
    if maximum > tolerance:
        raise ValueError(f"batched/scalar probability delta {maximum} exceeds {tolerance}")
    subset_hash = hashlib.sha256(
        ("\n".join(expected_ids) + "\n").encode()).hexdigest()
    return {"status": "pass", "rows": len(rows), "subset_ids_sha256": subset_hash,
            "exact_discrete_outputs": True, "max_abs_probability_delta": maximum,
            "tolerance": tolerance,
            "selection": "first_frozen_row_per_arity_language_order_role_family_primitive"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--min-clauses", type=int, default=1)
    parser.add_argument("--m0-threshold", type=float, default=0.98)
    parser.add_argument("--required-kind", action="append", default=[])
    parser.add_argument("--causal-standalone", action="store_true")
    parser.add_argument("--panel-manifest")
    parser.add_argument("--run-receipt")
    parser.add_argument("--reference-provenance")
    parser.add_argument("--base-model-inventory")
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-batch-rows", type=int, default=64)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from scripts.score_action_panel import score_group
    from scripts.train_gemma_decision_lora import letter_token_groups

    rows = _read(args.data)
    selected = (rows if args.causal_standalone else
                [row for row in rows if len((row.get("meta") or {}).get("composition", []))
                 >= args.min_clauses])
    if args.causal_standalone:
        if (not args.panel_manifest or bool(args.run_receipt) == bool(args.reference_provenance)
                or not args.base_model_inventory or not args.adapter):
            raise ValueError(
                "causal evaluation requires panel/base inventory, adapter, and exactly one "
                "candidate receipt or reference provenance")
        manifest_payload = Path(args.panel_manifest).read_bytes()
        run_receipt_payload = Path(args.run_receipt).read_bytes() if args.run_receipt else None
        reference_provenance_payload = (Path(args.reference_provenance).read_bytes()
                                        if args.reference_provenance else None)
        if reference_provenance_payload == b"":
            raise ValueError("reference provenance artifact must not be empty")
        base_inventory_payload = Path(args.base_model_inventory).read_bytes()
        model_inventory = validate_causal_model_inventory(
            args.model, base_inventory_payload, args.model_revision)
        checked_rows, _ = validate_causal_panel_payload(
            Path(args.data).read_bytes(), manifest_payload, require_frozen=True)
        if checked_rows != rows:
            raise AssertionError("causal panel parser disagreement")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = letter_token_groups(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to("cuda").eval()
    predictions = []
    parity_report = None
    with torch.no_grad():
        if args.causal_standalone:
            parity_rows = causal_parity_subset(selected)
            parity_batched = score_causal_rows_batched(
                model, tokenizer, groups, parity_rows, torch.device("cuda"), args.max_length,
                args.max_batch_tokens, args.max_batch_rows)
            parity_scalar = []
            for row in parity_rows:
                probabilities = score_group(
                    model, tokenizer, groups, row, torch.device("cuda"), args.max_length)["q00"]
                parity_scalar.append({"id": row["id"], "probabilities": probabilities})
            parity_report = audit_causal_batch_parity(
                parity_rows, parity_batched, parity_scalar)
            predictions = score_causal_rows_batched(
                model, tokenizer, groups, selected, torch.device("cuda"), args.max_length,
                args.max_batch_tokens, args.max_batch_rows)
        else:
            for row in selected:
                projected_by_parent = row_clause_views(row)
                for view in row["paired_views"]:
                    to_score = [view] + projected_by_parent[view["id"]]
                    for projected in to_score:
                        probabilities = score_group(
                            model, tokenizer, groups, projected, torch.device("cuda"),
                            args.max_length)["q00"]
                        predictions.append({"id": projected["id"], "probabilities": probabilities,
                                            "semantic_margin": math.log(
                                                max(probabilities["yes"], 1e-12)
                                                / max(probabilities["no"], 1e-12))})
    if args.causal_standalone:
        predictions = seal_causal_predictions(rows, predictions)
        report = causal_coverage_report(predictions)
        prediction_payload = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions)
    else:
        report = audit(rows, predictions, args.min_clauses)
        if args.required_kind:
            report["m0_gate"] = m0_gate(report, args.required_kind, args.m0_threshold)
        prediction_payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions)
    adapter_path = Path(args.adapter) / "adapter_model.safetensors" if args.adapter else None
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest() if adapter_path else None
    if args.causal_standalone:
        model_inventory_sha256 = hashlib.sha256(base_inventory_payload).hexdigest()
        if run_receipt_payload is not None:
            run_receipt = validate_causal_run_receipt(run_receipt_payload, adapter_sha256)
            if run_receipt["model_revision"] != args.model_revision:
                raise ValueError("run receipt model revision differs from evaluator revision")
            if run_receipt["model_inventory_sha256"] != model_inventory_sha256:
                raise ValueError("run receipt model inventory differs from supplied inventory")
    report.update({"input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                   "predictions_sha256": hashlib.sha256(prediction_payload.encode()).hexdigest(),
                   "adapter_sha256": adapter_sha256,
                   "model_revision": args.model_revision})
    if args.causal_standalone:
        report.update({
            "panel_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "model_inventory_sha256": model_inventory_sha256,
            "model_tree_sha256": model_inventory["tree_sha256"],
            "model_tree_files": model_inventory["files"],
            "panel_rows": len(rows),
            "batch_scalar_parity": parity_report,
        })
        if run_receipt_payload is not None:
            report.update({
                "provenance_role": "candidate",
                "run_receipt_sha256": hashlib.sha256(run_receipt_payload).hexdigest(),
                "train_sha256": run_receipt["train_sha256"],
                "recipe_sha256": run_receipt["recipe_sha256"],
                "parent_adapter_sha256": run_receipt["parent_adapter_sha256"],
            })
        else:
            report.update({
                "provenance_role": "reference",
                "reference_provenance_sha256": hashlib.sha256(
                    reference_provenance_payload).hexdigest(),
            })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions.jsonl").write_text(prediction_payload)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
