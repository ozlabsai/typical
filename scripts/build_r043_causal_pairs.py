"""Build R043's matched causal-pair pilot arms without public benchmark templates."""

import argparse
import copy
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path


TRAIN_PAIRS = 1_200
EVAL_PAIRS = 256
REPLAY_ROWS = 1_600
FAMILIES = (
    "invoice",
    "security",
    "agent_trace",
    "customer_support",
    "policy_routing",
    "evidence_selection",
    "temporal_numeric_probability",
    "ordinal",
)
GENERIC_CRITERIA = {"no": "No", "yes": "Yes"}
WORD_RE = re.compile(r"\w+", re.UNICODE)
QUARANTINE_TEXT_KEYS = {
    "state", "question", "questions", "policy", "policy_text",
    "input", "input_json", "instructions", "criteria", "content", "text", "prompt", "description",
}


SPECS = {
    "invoice": {
        "field": "invoice_match", "pass": "matched", "fail": "mismatched",
        "action": "release the remittance", "condition": "the invoice match is matched",
        "nuisance": ("ledger_region", "batch_channel", "currency_class", "archive_lane"),
    },
    "security": {
        "field": "session_attestation", "pass": "trusted", "fail": "untrusted",
        "action": "open the protected session", "condition": "the session attestation is trusted",
        "nuisance": ("network_zone", "client_family", "request_class", "audit_lane"),
    },
    "agent_trace": {
        "field": "required_tool_result", "pass": "succeeded", "fail": "failed",
        "action": "commit the agent outcome", "condition": "the required tool result succeeded",
        "nuisance": ("planner_version", "trace_channel", "worker_pool", "logging_mode"),
    },
    "customer_support": {
        "field": "identity_check", "pass": "confirmed", "fail": "unconfirmed",
        "action": "reveal the account detail", "condition": "the identity check is confirmed",
        "nuisance": ("contact_channel", "queue_color", "locale_group", "ticket_class"),
    },
    "policy_routing": {
        "field": "route_eligibility", "pass": "eligible", "fail": "ineligible",
        "action": "send the item to the express route", "condition": "route eligibility is eligible",
        "nuisance": ("origin_cluster", "service_band", "payload_class", "routing_epoch"),
    },
    "evidence_selection": {
        "field": "citation_support", "pass": "supports", "fail": "contradicts",
        "action": "accept the cited finding", "condition": "the citation supports the finding",
        "nuisance": ("collection_name", "index_version", "retrieval_lane", "document_class"),
    },
    "ordinal": {
        "field": "review_rank", "pass": "tier_four", "fail": "tier_two",
        "action": "advance the ranked submission", "condition": "the review rank is at least tier three",
        "nuisance": ("contest_track", "judge_panel", "submission_class", "ranking_epoch"),
    },
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _read_jsonl(path):
    data = Path(path).read_bytes()
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()], _sha256_bytes(data)


def _normalize(text):
    return " ".join(WORD_RE.findall(unicodedata.normalize("NFKC", str(text)).casefold()))


def _trigrams(text):
    words = text.split()
    return {tuple(words[index:index + 3]) for index in range(len(words) - 2)}


def _field_value_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_field_value_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_field_value_text(item) for item in value)
    return str(value) if value is not None else ""


def _selected_texts(row):
    texts = []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold() in QUARANTINE_TEXT_KEYS:
                    text = _normalize(_field_value_text(item))
                    if text:
                        texts.append(text)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(row)
    return list(dict.fromkeys(texts))


class Quarantine:
    def __init__(self, rows=()):
        rows = list(rows)
        self.texts = list(dict.fromkeys(text for row in rows for text in _selected_texts(row)))
        if rows and not self.texts:
            raise ValueError("quarantine input contains no recognized text fields")
        self._trigrams = [_trigrams(text) for text in self.texts]

    def collision(self, row):
        for candidate in _selected_texts(row):
            candidate_trigrams = _trigrams(candidate)
            for reference, reference_trigrams in zip(self.texts, self._trigrams):
                if candidate == reference:
                    return "exact"
                union = candidate_trigrams | reference_trigrams
                if (candidate_trigrams and reference_trigrams
                        and len(candidate_trigrams & reference_trigrams) / len(union) >= 0.8):
                    return "trigram"
        return None


def execute(program, facts):
    values = {fact["field"]: fact["value"] for fact in facts}
    actual = values[program["field"]]
    op, expected = program["op"], program["value"]
    if op == "eq":
        passed = actual == expected
    elif op == "lte":
        passed = float(actual) <= float(expected)
    elif op == "before_or_on":
        passed = str(actual) <= str(expected)
    elif op == "gte_probability":
        passed = float(actual) >= float(expected)
    elif op == "gte_rank":
        passed = program["order"].index(actual) >= program["order"].index(expected)
    else:
        raise ValueError(f"unsupported R043 program operation: {op}")
    return program["on_true"] if passed else program["on_false"]


def _temporal_spec(index):
    mode = index % 3
    if mode == 0:
        return {
            "field": "received_date", "pass": "2031-04-14", "fail": "2031-04-16",
            "action": "admit the timed request", "condition": "the received date is on or before 2031-04-15",
            "op": "before_or_on", "value": "2031-04-15",
        }
    if mode == 1:
        return {
            "field": "measured_load", "pass": 48, "fail": 52,
            "action": "accept the measured load", "condition": "the measured load is at most 50",
            "op": "lte", "value": 50,
        }
    return {
        "field": "success_probability", "pass": 0.72, "fail": 0.68,
        "action": "select the probabilistic branch", "condition": "success probability is at least 0.70",
        "op": "gte_probability", "value": 0.70,
    }


def _spec(family, index):
    spec = dict(_temporal_spec(index) if family == "temporal_numeric_probability" else SPECS[family])
    spec.setdefault("op", "gte_rank" if family == "ordinal" else "eq")
    spec.setdefault("value", "tier_three" if family == "ordinal" else spec["pass"])
    if family == "ordinal":
        spec["order"] = ["tier_one", "tier_two", "tier_three", "tier_four", "tier_five"]
    spec.setdefault("nuisance", ("sampling_lane", "measurement_batch", "observer_group", "record_epoch"))
    return spec


def _source(split, family, index, seed, range_start, design_index=None):
    candidate = range_start + index
    design_index = index if design_index is None else design_index
    source_id = f"r043-{split}-{seed}-{family}-{candidate:07d}"
    entity = f"unit-{seed % 997:03d}-{candidate:07d}"
    rng = random.Random(f"r043:{seed}:{split}:{family}:{candidate}")
    spec = _spec(family, design_index)
    nuisance = [
        {"id": f"ev-{source_id}-n{number}", "field": field,
         "value": f"code_{rng.randrange(10_000, 99_999)}", "decisive": False}
        for number, field in enumerate(spec["nuisance"])
    ]
    decisive_id = f"ev-{source_id}-decisive"
    program = {"language": "r043-dsl-v1", "op": spec["op"], "field": spec["field"],
               "value": spec["value"], "on_true": "execute", "on_false": "withhold"}
    if "order" in spec:
        program["order"] = spec["order"]
    polarity = "positive" if design_index % 2 == 0 else "negative"
    # Orthogonal four-cycle: balance option order without making it a polarity cue.
    yes_first = bool((design_index // 2 + design_index % 2) % 2)
    decisive_position = design_index % 5
    policy = (f"Policy for {entity}: execute {spec['action']} only when {spec['condition']}; "
              "otherwise withhold it.")

    def view(passed):
        decisive = {"id": decisive_id, "field": spec["field"],
                    "value": spec["pass"] if passed else spec["fail"], "decisive": True}
        facts = copy.deepcopy(nuisance)
        facts.insert(decisive_position, decisive)
        action_gold = execute(program, facts)
        queried_action = "execute" if polarity == "positive" else "withhold"
        label = "yes" if action_gold == queried_action else "no"
        criteria = dict(reversed(list(GENERIC_CRITERIA.items()))) if yes_first else dict(GENERIC_CRITERIA)
        evidence_lines = [f"[{fact['id']}] {fact['field']}: {fact['value']}." for fact in facts]
        instruction = (f"For {entity}, should the system {queried_action} the action: {spec['action']}? "
                       "Apply only the stated policy and evidence.")
        presentation_id = f"{source_id}-{'pass' if passed else 'fail'}-{polarity}"
        return {
            "id": presentation_id,
            "state": policy + "\nEvidence:\n" + "\n".join(evidence_lines),
            "questions": {"decision": {"type": "noul", "instructions": instruction,
                                         "criteria": criteria}},
            "targets": {"decision": {"label": label}},
            "meta": {"source": "r043_programmatic", "source_id": source_id, "split": split,
                     "family": family, "decisive_field": spec["field"],
                     "evidence_ids": [fact["id"] for fact in facts],
                     "dependency_ids": [decisive_id], "action_program": program,
                     "action_gold": action_gold, "presentation_id": presentation_id,
                     "pair_mode": "counterfactual", "polarity": polarity,
                     "evidence_position": decisive_position, "yes_first": yes_first,
                     "fixed_option_order": True, "policy_text": policy, "facts": facts},
        }

    views = [view(True), view(False)]
    by_label = {item["targets"]["decision"]["label"]: item for item in views}
    if set(by_label) != {"yes", "no"}:
        raise AssertionError("counterfactual did not flip the semantic action label")
    ordered = ([by_label["yes"], by_label["no"]] if design_index % 2 == 0
               else [by_label["no"], by_label["yes"]])
    for arm, item in zip(("A", "B"), ordered):
        item["meta"]["s2_arm"] = arm
    return {
        "id": f"r043-pair-{split}-{family}-{candidate:07d}",
        "pair_mode": "counterfactual", "source_ids": [source_id, source_id],
        "paired_views": ordered,
        "meta": {"source": "r043_programmatic", "source_id": source_id, "split": split,
                 "family": family, "decisive_field": spec["field"],
                 "evidence_ids": [fact["id"] for fact in ordered[0]["meta"]["facts"]],
                 "dependency_ids": [decisive_id], "action_program": program,
                 "presentation_ids": [item["meta"]["presentation_id"] for item in ordered]},
    }


def _validate_true_pair(pair):
    left, right = pair["paired_views"]
    if pair["pair_mode"] != "counterfactual" or pair["source_ids"][0] != pair["source_ids"][1]:
        raise ValueError(f"{pair['id']}: invalid true-pair identity")
    if {left["targets"]["decision"]["label"], right["targets"]["decision"]["label"]} != {"yes", "no"}:
        raise ValueError(f"{pair['id']}: label does not flip")
    left_meta, right_meta = left["meta"], right["meta"]
    if left["questions"] != right["questions"] or left_meta["policy_text"] != right_meta["policy_text"]:
        raise ValueError(f"{pair['id']}: paired policy or question differs")
    if len(_normalize(left["state"]).split()) != len(_normalize(right["state"]).split()):
        raise ValueError(f"{pair['id']}: counterfactual state lengths differ")
    left_facts = {fact["id"]: fact for fact in left_meta["facts"]}
    right_facts = {fact["id"]: fact for fact in right_meta["facts"]}
    differences = [fact_id for fact_id in left_facts if left_facts[fact_id] != right_facts.get(fact_id)]
    if differences != left_meta["dependency_ids"] or differences != right_meta["dependency_ids"]:
        raise ValueError(f"{pair['id']}: pair must differ in exactly its decisive dependency")
    for view in (left, right):
        meta = view["meta"]
        if execute(meta["action_program"], meta["facts"]) != meta["action_gold"]:
            raise ValueError(f"{view['id']}: executable action does not match gold")
        queried = "execute" if meta["polarity"] == "positive" else "withhold"
        expected = "yes" if meta["action_gold"] == queried else "no"
        if view["targets"]["decision"]["label"] != expected:
            raise ValueError(f"{view['id']}: semantic target does not match executable action")


def _build_split(total_pairs, split, seed, range_start, quarantine):
    if total_pairs % len(FAMILIES):
        raise ValueError("pair count must be divisible across exactly eight families")
    wanted = total_pairs // len(FAMILIES)
    rows, rejected = [], Counter()
    for family in FAMILIES:
        accepted = 0
        for candidate_index in range(wanted * 100):
            pair = _source(split, family, candidate_index, seed, range_start, accepted)
            reason = next((reason for view in pair["paired_views"]
                           if (reason := quarantine.collision(view))), None)
            if reason:
                rejected[reason] += 1
                continue
            _validate_true_pair(pair)
            rows.append(pair)
            accepted += 1
            if accepted == wanted:
                break
        if accepted != wanted:
            raise ValueError(f"quarantine left too few safe {split}/{family} sources")
    return rows, rejected


def shuffled_control(true_pairs):
    by_family = defaultdict(list)
    for pair in true_pairs:
        by_family[pair["meta"]["family"]].append(pair)
    shuffled = []
    for family in FAMILIES:
        pairs = by_family[family]
        yes_views = [{view["targets"]["decision"]["label"]: view for view in pair["paired_views"]}["yes"]
                     for pair in pairs]
        no_views = [{view["targets"]["decision"]["label"]: view for view in pair["paired_views"]}["no"]
                    for pair in pairs]
        for index, (yes_view, no_view) in enumerate(zip(yes_views, no_views[1:] + no_views[:1])):
            views = [copy.deepcopy(yes_view), copy.deepcopy(no_view)]
            if index % 2:
                views.reverse()
            for arm, view in zip(("A", "B"), views):
                view["meta"].update({"s2_arm": arm, "pair_mode": "shuffled_counterfactual"})
            source_ids = [view["meta"]["source_id"] for view in views]
            if source_ids[0] == source_ids[1]:
                raise AssertionError("within-family derangement retained a source")
            shuffled.append({
                "id": f"r043-shuffled-{family}-{index:04d}",
                "pair_mode": "shuffled_counterfactual", "source_ids": source_ids,
                "paired_views": views,
                "meta": {"source": "r043_programmatic", "split": "train", "family": family,
                         "source_ids": source_ids,
                         "presentation_ids": [view["meta"]["presentation_id"] for view in views]},
            })
    return shuffled


def validate(train_pairs, shuffled, eval_pairs):
    if len(train_pairs) != TRAIN_PAIRS or len(shuffled) != TRAIN_PAIRS or len(eval_pairs) != EVAL_PAIRS:
        raise ValueError("R043 requires exactly 1,200 train and 256 evaluation pairs")
    expected_train = {family: TRAIN_PAIRS // 8 for family in FAMILIES}
    expected_eval = {family: EVAL_PAIRS // 8 for family in FAMILIES}
    if Counter(row["meta"]["family"] for row in train_pairs) != expected_train:
        raise ValueError("train families are not balanced")
    if Counter(row["meta"]["family"] for row in eval_pairs) != expected_eval:
        raise ValueError("evaluation families are not balanced")
    for pair in train_pairs + eval_pairs:
        _validate_true_pair(pair)
    train_sources = {pair["meta"]["source_id"] for pair in train_pairs}
    eval_sources = {pair["meta"]["source_id"] for pair in eval_pairs}
    if train_sources & eval_sources:
        raise ValueError("train/evaluation source leak")
    for pair in shuffled:
        views = pair["paired_views"]
        if (pair["pair_mode"] != "shuffled_counterfactual" or len(set(pair["source_ids"])) != 2
                or len({view["meta"]["family"] for view in views}) != 1
                or {view["targets"]["decision"]["label"] for view in views} != {"yes", "no"}):
            raise ValueError(f"{pair['id']}: invalid shuffled control")
    presentations = [view for pair in train_pairs for view in pair["paired_views"]]
    for field in ("polarity", "yes_first", "evidence_position"):
        counts = Counter(view["meta"][field] for view in presentations)
        if max(counts.values()) - min(counts.values()) > 2:
            raise ValueError(f"unbalanced presentation field: {field}")
    if Counter(view["targets"]["decision"]["label"] for view in presentations) != {"yes": 1200, "no": 1200}:
        raise ValueError("labels are not balanced")


def build(seed=43, quarantine_rows=()):
    quarantine = Quarantine(quarantine_rows)
    train, train_rejected = _build_split(TRAIN_PAIRS, "train", seed, 0, quarantine)
    evaluation, eval_rejected = _build_split(EVAL_PAIRS, "eval", seed + 104_729, 1_000_000, quarantine)
    shuffled = shuffled_control(train)
    validate(train, shuffled, evaluation)
    return train, shuffled, evaluation, {"train": dict(train_rejected), "eval": dict(eval_rejected)}


def _replay_skip_reason(row):
    if not isinstance(row, Mapping):
        return "invalid_row"
    if "paired_views" in row:
        return "paired_views"
    questions = row.get("questions")
    if not isinstance(questions, Mapping) or len(questions) != 1:
        return "question_count"
    qid, question = next(iter(questions.items()))
    if not isinstance(question, Mapping):
        return "invalid_question"
    criteria = question.get("criteria")
    if question.get("type") == "score":
        if not isinstance(criteria, list):
            return "invalid_criteria"
        labels = [str(index) for index in range(len(criteria))]
    else:
        if not isinstance(criteria, Mapping):
            return "invalid_criteria"
        labels = [str(label) for label in criteria]
    if not 2 <= len(labels) <= 26:
        return "unsupported_option_count"
    targets = row.get("targets")
    target = targets.get(qid) if isinstance(targets, Mapping) else None
    if not isinstance(target, Mapping) or str(target.get("label")) not in labels:
        return "missing_target_label"
    if "state" not in row:
        return "missing_state"
    soft_targets = row.get("soft_targets") or {}
    if not isinstance(soft_targets, Mapping):
        return "invalid_soft_target"
    probabilities = soft_targets.get(qid)
    if probabilities is not None:
        try:
            valid = (len(probabilities) == len(labels) and all(value >= 0 for value in probabilities)
                     and sum(probabilities) > 0)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            return "invalid_soft_target"
    return None


def _select_replay(rows):
    selected, positions, skipped = [], [], Counter()
    for position, row in enumerate(rows):
        reason = _replay_skip_reason(row)
        if reason:
            skipped[reason] += 1
            continue
        selected.append(copy.deepcopy(row))
        positions.append(position)
        if len(selected) == REPLAY_ROWS:
            return selected, {"selected_positions": positions, "position_base": 0,
                              "scanned_rows": position + 1,
                              "skipped": dict(skipped)}
    raise ValueError(f"R043 requires at least {REPLAY_ROWS} supported frozen replay singles; "
                     f"found {len(selected)} after scanning {len(rows)} rows")


def write_outputs(out, replay, train_pairs, shuffled, evaluation, provenance, seed):
    replay, replay_selection = _select_replay(replay)
    arm_a = [*copy.deepcopy(replay), *copy.deepcopy(train_pairs)]
    arm_b = copy.deepcopy(arm_a)
    arm_c = [*copy.deepcopy(replay), *copy.deepcopy(shuffled)]
    named = {"arm-a": arm_a, "arm-b": arm_b, "arm-c": arm_c, "eval": evaluation}
    payloads = {name: _jsonl_bytes(rows) for name, rows in named.items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (out / f"{name}.jsonl").write_bytes(payload)
    manifest = {
        "experiment": "r043_causal_pairs", "seed": seed,
        "families": list(FAMILIES), "train_pairs": TRAIN_PAIRS, "eval_pairs": EVAL_PAIRS,
        "replay_rows": REPLAY_ROWS,
        "logical_rows": {"arm-a": 2800, "arm-b": 2800, "arm-c": 2800, "eval": EVAL_PAIRS},
        "trainer_examples": {"arm-a": 2800, "arm-b": 2800, "arm-c": 2800,
                             "eval": EVAL_PAIRS},
        "presentations": {"arm-a": 4000, "arm-b": 4000, "arm-c": 4000,
                          "eval": EVAL_PAIRS * 2},
        "pair_modes": {"arm-a": "counterfactual", "arm-b": "counterfactual",
                       "arm-c": "shuffled_counterfactual", "eval": "counterfactual"},
        "source_ranges": {"train": [0, 999_999], "eval": [1_000_000, None]},
        "eval_seed": seed + 104_729, "quarantine_rejections": provenance["rejections"],
        "inputs": provenance["inputs"],
        "selected_replay_sha256": _sha256_bytes(_jsonl_bytes(replay)),
        "replay_selection": replay_selection,
        "outputs": {name: {"rows": len(named[name]), "sha256": _sha256_bytes(payload)}
                    for name, payload in payloads.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay", action="append", required=True)
    parser.add_argument("--quarantine", action="append", default=[])
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    replay, replay_inputs = [], []
    for path in args.replay:
        rows, digest = _read_jsonl(path)
        replay.extend(rows)
        replay_inputs.append({"path": str(path), "rows": len(rows), "sha256": digest})
    quarantine_rows, quarantine_inputs = [], []
    for path in args.quarantine:
        rows, digest = _read_jsonl(path)
        selected_texts = len(Quarantine(rows).texts)
        if not selected_texts:
            raise ValueError(f"quarantine input contains no recognized text fields: {path}")
        quarantine_rows.extend(rows)
        quarantine_inputs.append({"path": str(path), "rows": len(rows), "selected_texts": selected_texts,
                                  "sha256": digest})
    train, shuffled, evaluation, rejections = build(args.seed, quarantine_rows)
    provenance = {"rejections": rejections,
                  "inputs": {"replay": replay_inputs, "replay_rows": len(replay),
                             "quarantine": quarantine_inputs,
                             "quarantine_rows": len(quarantine_rows)}}
    manifest = write_outputs(args.out, replay, train, shuffled, evaluation, provenance, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
