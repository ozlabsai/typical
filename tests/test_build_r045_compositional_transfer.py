import json
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from zoneinfo import ZoneInfo

import pytest

from scripts.build_r045_compositional_transfer import (
    CASES_PER_COMPOSITION,
    COMPOSITIONS,
    DECISION_RULE,
    HE_EFFECT,
    HE_SCOPE,
    SPLITS,
    _evaluate,
    build,
    execute,
)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rendered_oracle(state):
    """Recompute English targets from model-visible text, without hidden metadata."""
    results = []
    for line in (line for line in state.splitlines() if line.startswith("Condition ")):
        if ": decimal:" in line:
            values = re.search(r"credits ([\d.]+) \+ ([\d.]+), fee ([\d.]+) credits;.*>= ([\d.]+)", line)
            a, b, fee, threshold = map(Decimal, values.groups())
            results.append(a + b - fee >= threshold)
        elif ": fraction:" in line:
            values = re.search(r"ratio (\d+)/(\d+);.*>= (\d+)/(\d+)", line)
            a, b, c, d = map(int, values.groups())
            results.append(Fraction(a, b) >= Fraction(c, d))
        elif ": datetime:" in line:
            values = re.search(r"time ([\d:T-]+) in ([\w/]+);.*<= ([\d:T-]+) UTC", line)
            local, zone, deadline = values.groups()
            instant = datetime.fromisoformat(local).replace(tzinfo=ZoneInfo(zone)).astimezone(timezone.utc)
            results.append(instant <= datetime.fromisoformat(deadline).replace(tzinfo=timezone.utc))
        elif ": ledger:" in line:
            entries, threshold = re.search(r"deltas \[(.+)\];.*>= ([\d.]+) units", line).groups()
            results.append(sum(Decimal(item.rsplit(":", 1)[1]) for item in entries.split(", "))
                           >= Decimal(threshold))
        elif ": conditional:" in line:
            groups, conditioned, a, b = re.search(
                r"groups (.+); condition on ([\w-]+);.*>= (\d+)/(\d+)", line).groups()
            parsed = {name: (int(success), int(failure)) for name, success, failure in
                      re.findall(r"([\w-]+)\[success=(\d+),failure=(\d+)\]", groups)}
            success, failure = parsed[conditioned]
            results.append(Fraction(success, success + failure) >= Fraction(int(a), int(b)))
        elif ": policy:" in line:
            scope = re.search(r"request scope ([\w-]+);", line).group(1)
            rules = [(int(priority), effect) for _, effect, priority, rule_scope in
                     re.findall(r"([\w-]+)=(allow|deny)@(\d+) active scope=([\w-]+)", line)
                     if rule_scope in {"all", scope}]
            results.append(max(rules, key=lambda item: (item[0], item[1] == "deny"))[1] == "allow")
        else:
            raise AssertionError(f"unparsed condition: {line}")
    assert results and "if and only if EVERY" in state
    return all(results)


def test_build_is_deterministic_and_emits_trainer_rows_with_executable_gold(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build(first, seed=45)
    build(second, seed=45)
    assert manifest["experiment"] == "r045_compositional_transfer"
    assert manifest["source_inputs"] == []
    assert len(manifest["builder_sha256"]) == 64
    for split in SPLITS:
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
        rows = _rows(first / f"{split}.jsonl")
        assert len(rows) == manifest["splits"][split]["rows"]
        assert manifest["splits"][split]["scenarios"] == len(
            {row["meta"]["scenario_id"] for row in rows})
        assert manifest["splits"][split]["sha256"]
        for row in rows:
            assert row["pair_mode"] == "counterfactual"
            assert len(row["paired_views"]) == 2
            left, right = row["paired_views"]
            assert row["source_ids"] == [left["meta"]["source_id"], right["meta"]["source_id"]]
            assert left["questions"] == right["questions"]
            assert {left["targets"]["decision"]["label"], right["targets"]["decision"]["label"]} == {"yes", "no"}
            for view in (left, right):
                meta = view["meta"]
                assert view["questions"]["decision"]["type"] == "choice"
                assert set(view["questions"]["decision"]["criteria"]) == {"yes", "no"}
                assert execute(meta["program"], meta["clauses"]) is meta["gold"]


def test_splits_hold_out_compositions_and_instances_but_match_nuisance_pools(tmp_path):
    manifest = build(tmp_path, seed=45)
    observed = {}
    for split in SPLITS:
        rows = _rows(tmp_path / f"{split}.jsonl")
        seen = {"compositions": set(), "entities": set(), "renderers": set(), "states": set(),
                "numeric_thresholds": []}
        for row in rows:
            meta = row["meta"]
            seen["compositions"].add(tuple(meta["composition"]))
            seen["entities"].add(meta["entity_vocab"])
            seen["renderers"].add(meta["renderer"])
            for view in row["paired_views"]:
                seen["states"].add(view["state"])
                seen["numeric_thresholds"].extend(
                    Decimal(clause["threshold"]) for clause in view["meta"]["clauses"]
                    if clause["kind"] in {"decimal", "ledger"})
        assert seen["compositions"] == set(COMPOSITIONS[split])
        assert seen["entities"] == set(SPLITS[split]["entities"])
        assert seen["renderers"] == set(SPLITS[split]["renderers"])
        assert manifest["leakage_contract"]["disjoint_compositions"]
        assert manifest["leakage_contract"]["disjoint_instances"]
        assert manifest["leakage_contract"]["matched_nuisance_pools"]
        observed[split] = seen

    for left, right in (("train", "cal"), ("train", "test"), ("cal", "test")):
        assert observed[left]["compositions"].isdisjoint(observed[right]["compositions"])
        assert observed[left]["entities"] == observed[right]["entities"]
        assert observed[left]["renderers"] == observed[right]["renderers"]
        assert observed[left]["states"].isdisjoint(observed[right]["states"])
        assert min(observed[left]["numeric_thresholds"]) <= max(observed[right]["numeric_thresholds"])
        assert min(observed[right]["numeric_thresholds"]) <= max(observed[left]["numeric_thresholds"])

    other = tmp_path / "other-seed"
    build(other, seed=46)
    assert (_rows(tmp_path / "train.jsonl")[0]["paired_views"][0]["state"] !=
            _rows(other / "train.jsonl")[0]["paired_views"][0]["state"])


def test_computational_counterfactual_and_language_twins_stay_in_one_split(tmp_path):
    build(tmp_path, seed=12)
    all_rows = {split: _rows(tmp_path / f"{split}.jsonl") for split in SPLITS}
    owner = {}
    for split, rows in all_rows.items():
        for row in rows:
            source = row["meta"]["source_id"]
            owner.setdefault(source, set()).add(split)
            assert row["meta"]["language"] in ("en", "he")
            left, right = row["paired_views"]
            a, b = left["meta"]["clauses"], right["meta"]["clauses"]
            changed = [one["kind"] for one, two in zip(a, b) if one != two]
            assert changed == [left["meta"]["decisive_kind"]]
    assert all(len(splits) == 1 for splits in owner.values())
    twins = Counter((split, row["meta"]["twin_id"]) for split, rows in all_rows.items() for row in rows)
    assert set(twins.values()) == {2}
    assert (set(_evaluate(c) for rows in all_rows.values() for row in rows
                for view in row["paired_views"] for c in view["meta"]["clauses"])
            == {True, False})


def test_rendered_contract_is_explicit_and_shortcuts_are_crossed(tmp_path):
    build(tmp_path, seed=45)
    crossed = Counter()
    option_orders = {}
    scenarios = {}
    temporal_surfaces = {}
    datetime_rollover = False
    fraction_equality = False
    policy_orders = set()
    for split in SPLITS:
        for row in _rows(tmp_path / f"{split}.jsonl"):
            composition = tuple(row["meta"]["composition"])
            crossed[(split, composition, row["meta"]["renderer"],
                     row["paired_views"][0]["meta"]["decisive_kind"])] += 1
            scenario_key = (split, composition, row["meta"]["scenario_id"], row["meta"]["language"])
            scenarios.setdefault(scenario_key, []).append(row)
            for view in row["paired_views"]:
                order_key = (split, view["meta"]["language"])
                option_orders.setdefault(order_key, set()).add(
                    tuple(view["questions"]["decision"]["criteria"]))
                assert "-pass" not in view["id"] and "-fail" not in view["id"]
                assert DECISION_RULE[view["meta"]["language"]] in view["state"]
                condition = "Condition" if view["meta"]["language"] == "en" else "תנאי"
                assert view["state"].count(condition + " ") == len(composition)
                assert view["meta"]["rendered_contract"] == "explicit-all-of-v2"
                if view["meta"]["language"] == "en":
                    assert _rendered_oracle(view["state"]) is (
                        view["targets"]["decision"]["label"] == "yes")
                for clause in view["meta"]["clauses"]:
                    assert (">=" in view["state"] if clause["kind"] not in {"datetime", "policy"}
                            else "<=" in view["state"] if clause["kind"] == "datetime"
                            else "priority" in view["state"] or "עדיפות" in view["state"])
                    if clause["kind"] == "decimal":
                        net = sum(map(Decimal, clause["credits"])) - Decimal(clause["fee"])
                        assert len(clause["credits"]) == 2 and all(Decimal(v) for v in clause["credits"])
                        assert all(value in view["state"] for value in
                                   (*clause["credits"], clause["fee"], clause["threshold"], clause["unit"]))
                        assert (net >= Decimal(clause["threshold"])) is _evaluate(clause)
                    elif clause["kind"] == "fraction":
                        assert clause["denominator"] != 100
                        assert all(str(value) in view["state"] for value in
                                   (clause["numerator"], clause["denominator"], *clause["threshold"]))
                        fraction_equality |= (clause["numerator"] * clause["threshold"][1]
                                              == clause["threshold"][0] * clause["denominator"])
                    elif clause["kind"] == "datetime":
                        assert all(value in view["state"] for value in
                                   (clause["local_time"], clause["timezone"], clause["deadline"]))
                        datetime_rollover |= clause["local_time"][:10] != clause["deadline"][:10]
                        if view["meta"]["decisive_kind"] == "datetime":
                            label = view["targets"]["decision"]["label"]
                            local, deadline, zone = (clause["local_time"][11:16],
                                                     clause["deadline"][11:16], clause["timezone"])
                            for name, value in (("local", local), ("deadline", deadline), ("zone", zone),
                                                ("zone_deadline", (zone, deadline)),
                                                ("zone_local", (zone, local))):
                                temporal_surfaces.setdefault((split, name), {}).setdefault(value, set()).add(label)
                    elif clause["kind"] == "ledger":
                        deltas = [Decimal(event["delta"]) for event in clause["events"]]
                        assert len(deltas) == 3 and all(deltas) and min(deltas) < 0 < max(deltas)
                        assert all(event["event_id"] in view["state"] and event["delta"] in view["state"]
                                   for event in clause["events"])
                        assert clause["threshold"] in view["state"] and clause["currency"] in view["state"]
                    elif clause["kind"] == "conditional":
                        assert len(clause["groups"]) == 2
                        assert clause["condition_group"] in {group["name"] for group in clause["groups"]}
                        assert all(group["name"] in view["state"] and
                                   str(group["success"]) in view["state"] and
                                   str(group["failure"]) in view["state"] for group in clause["groups"])
                        assert all(str(value) in view["state"] for value in clause["threshold"])
                    elif clause["kind"] == "policy":
                        assert all(rule["active"] for rule in clause["rules"])
                        assert {rule["scope"] for rule in clause["rules"]} >= {"service", "unrelated"}
                        hebrew = view["meta"]["language"] == "he"
                        assert (HE_SCOPE[clause["request_scope"]] if hebrew else clause["request_scope"]) in view["state"]
                        assert all(rule["name"] in view["state"] and
                                   (HE_EFFECT[rule["effect"]] if hebrew else rule["effect"]) in view["state"] and
                                   str(rule["priority"]) in view["state"] and
                                   (HE_SCOPE[rule["scope"]] if hebrew else rule["scope"]) in view["state"]
                                   for rule in clause["rules"])
                        policy_orders.add(tuple(rule["name"].split("-")[0] for rule in clause["rules"]))
            for kind in composition:
                left = next(c for c in row["paired_views"][0]["meta"]["clauses"] if c["kind"] == kind)
                right = next(c for c in row["paired_views"][1]["meta"]["clauses"] if c["kind"] == kind)
                if kind == "datetime" and left != right:
                    assert left["local_time"] == right["local_time"]
                    assert left["deadline"] != right["deadline"]
                    assert left["deadline"].split(":")[1] == right["deadline"].split(":")[1]
                    assert left["timezone"] == right["timezone"]
                if kind == "decimal" and left != right:
                    assert left["fee"] == right["fee"] and left["credits"][1] == right["credits"][1]
                if kind == "policy" and left != right:
                    assert [(rule["name"], rule["effect"], rule["scope"]) for rule in left["rules"]] == [
                        (rule["name"], rule["effect"], rule["scope"]) for rule in right["rules"]]
                    assert [rule["priority"] for rule in left["rules"] if rule["effect"] == "deny"] == [
                        rule["priority"] for rule in right["rules"] if rule["effect"] == "deny"]

    assert all(orders == {("yes", "no"), ("no", "yes")} for orders in option_orders.values())
    assert all(labels == {"yes", "no"} for values in temporal_surfaces.values()
               for labels in values.values())
    assert fraction_equality and datetime_rollover and len(policy_orders) > 1
    for (_, composition, _, _), rows in scenarios.items():
        assert len(rows) == len(composition)
        assert {row["meta"]["decisive_kind"] for row in rows} == set(composition)
        yes_states = {next(view["state"] for view in row["paired_views"]
                           if view["targets"]["decision"]["label"] == "yes") for row in rows}
        assert len(yes_states) == 1
    for split, compositions in COMPOSITIONS.items():
        for composition in compositions:
            if len(composition) == 1:
                continue
            counts = [count for (seen_split, seen_composition, _, _), count in crossed.items()
                      if seen_split == split and seen_composition == composition]
            assert len(counts) == 2 * len(composition)
            assert set(counts) == {CASES_PER_COMPOSITION // len(composition)}


@pytest.mark.parametrize("path,content", [
    ("/tmp/JevBench/train.jsonl", None),
    (None, "copied WorkflowEvals prompt and answer"),
    ("/tmp/workflow_evals.jsonl", None),
])
def test_rejects_benchmark_source_references(tmp_path, path, content):
    with pytest.raises(ValueError, match="prohibited benchmark source"):
        build(tmp_path / "out", source_paths=[path] if path else (),
              source_contents=[content] if content else ())


def test_domain_offset_moves_every_decisive_pair(tmp_path):
    base, shifted = tmp_path / "base", tmp_path / "shifted"
    assert build(base, seed=46)["domain_offset"] == 0
    assert build(shifted, seed=46, domain_offset=1)["domain_offset"] == 1

    def decisive_pairs(path):
        result = set()
        for row in _rows(path / "test.jsonl"):
            kind = row["meta"]["decisive_kind"]
            clauses = [next(clause for clause in view["meta"]["clauses"]
                            if clause["kind"] == kind) for view in row["paired_views"]]
            result.add((kind, tuple(sorted(json.dumps(clause, sort_keys=True)
                                           for clause in clauses))))
        return result

    assert decisive_pairs(base).isdisjoint(decisive_pairs(shifted))
