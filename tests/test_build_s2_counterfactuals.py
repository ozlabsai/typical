import json
from collections import Counter

from scripts.build_s2_counterfactuals import (
    ARM_A_QUESTION, ARM_B_QUESTION, CLOSED_WORLD_CAVEAT, EVAL_QUESTIONS,
    _explicit_criteria, _hash_rows, _make_view, build,
)
from scripts.train_gemma_decision_lora import flatten_row
from scripts.workflow_corpus import NOUL_POLICY_SPEC, make_row, NOUL_STYLES


def donor(group, label_index, style="plain"):
    spec = dict(NOUL_POLICY_SPEC, prefix=f"Test policy {group} variant {label_index}.")
    row = make_row("noul", "policy_permit", spec, f"Facts for {group}.", ["no", "yes"],
                   label_index, (style, dict(NOUL_STYLES)[style]))
    row["id"] = f"{group}-{label_index}"
    row["meta"].update(rubric_group=group, rubric_variant=label_index)
    return row


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reconstructs_matched_arms_and_locked_crossed_eval(tmp_path):
    s1_rows = [{"id": f"s1-{i}", "state": f"state-{i}",
                "questions": {"decision": {"type": "noul", "instructions": "Truth?",
                                             "criteria": {"no": "No", "yes": "Yes"}}},
                "targets": {"decision": {"label": "yes"}},
                "soft_targets": {"decision": [0.1, 0.9]}}
               for i in range(2050)]
    train = [donor(f"tr-{i}", label) for i in range(70) for label in (0, 1)]
    val = [donor(f"va-{i}", label) for i in range(70) for label in (0, 1)]
    s1_path = tmp_path / "s1.jsonl"
    write_rows(s1_path, s1_rows)

    arm_a, arm_b, arm_c, crossed_eval, manifest = build([s1_path], train, val, seed=42)
    assert len(arm_a) == len(arm_b) == 2000
    assert arm_a[:1488] == arm_b[:1488]
    assert arm_a[:1488] == arm_c[:1488]
    assert all("soft_targets" in row for row in arm_a[:1488])
    a, b = arm_a[1488:], arm_b[1488:]
    assert len(a) == len(b) == 512
    c = arm_c[1488:]
    assert len(c) == 512
    for row_c, row_a, row_b in zip(c, a, b):
        assert row_c["state"] == row_a["state"] == row_b["state"]
        assert row_c["targets"] == row_a["targets"] == row_b["targets"]
        assert row_c["meta"]["source_id"] == row_a["meta"]["source_id"] == row_b["meta"]["source_id"]
        assert row_c["meta"]["context_view"] == row_a["meta"]["context_view"] == row_b["meta"]["context_view"]
        assert row_c["meta"]["polarity"] == row_a["meta"]["polarity"] == row_b["meta"]["polarity"]
        assert list(row_c["questions"]["decision"]["criteria"]) == list(
            row_a["questions"]["decision"]["criteria"]) == list(
                row_b["questions"]["decision"]["criteria"])
    assert Counter(row["meta"]["s2_arm"] for row in c) == {"A": 256, "B": 256}
    assert Counter(row["targets"]["decision"]["label"] for row in c) == {"no": 256, "yes": 256}
    assert Counter(row["meta"]["source_label"] for row in c) == {"no": 256, "yes": 256}
    for field in ("polarity", "context_view", "decisive_placement"):
        counts = Counter((row["meta"][field], row["meta"]["s2_arm"]) for row in c)
        for category in {row["meta"][field] for row in c}:
            assert counts[(category, "A")] == counts[(category, "B")]
    for order in (("no", "yes"), ("yes", "no")):
        counts = Counter(row["meta"]["s2_arm"] for row in c
                         if tuple(row["questions"]["decision"]["criteria"]) == order)
        assert counts == {"A": 128, "B": 128}
    joint = Counter((row["meta"]["s2_arm"], row["meta"]["context_view"],
                     row["meta"]["polarity"],
                     tuple(row["questions"]["decision"]["criteria"]),
                     row["targets"]["decision"]["label"]) for row in c)
    for arm in ("A", "B"):
        for view in ("compact", "full"):
            for polarity in ("positive", "negative"):
                for order in (("no", "yes"), ("yes", "no")):
                    # Each visible conjunction has both target labels equally often.
                    assert joint[(arm, view, polarity, order, "no")] == 16
                    assert joint[(arm, view, polarity, order, "yes")] == 16
    # Explicitly reject the former shortcut: answer index was a function of
    # format, context, and polarity alone under a single parity orientation.
    old_shortcut = {}
    for row in c:
        format_b = int(row["meta"]["s2_arm"] == "B")
        full = int(row["meta"]["context_view"] == "full")
        negative = int(row["meta"]["polarity"] == "negative")
        answer_index = list(row["questions"]["decision"]["criteria"]).index(
            row["targets"]["decision"]["label"])
        old_shortcut.setdefault((format_b, full, negative), set()).add(answer_index)
    assert len(old_shortcut) == 8 and all(indices == {0, 1}
                                          for indices in old_shortcut.values())
    assert all(joint[(arm, view, polarity, order, label)] > 0
               for arm in ("A", "B") for view in ("compact", "full")
               for polarity in ("positive", "negative")
               for order in (("no", "yes"), ("yes", "no"))
               for label in ("no", "yes"))
    assert len({row["meta"]["source_group"] for row in c}) == 64
    orientations_by_polarity = Counter()
    for group in {row["meta"]["source_group"] for row in c}:
        rows = [row for row in c if row["meta"]["source_group"] == group]
        inferred = {int(row["meta"]["source_label"] == "yes") ^
                    int(row["meta"]["context_view"] == "full") ^
                    int(list(row["questions"]["decision"]["criteria"]) == ["yes", "no"]) ^
                    int(row["meta"]["s2_arm"] == "B") for row in rows}
        assert len(inferred) == 1
        orientations_by_polarity[(rows[0]["meta"]["polarity"], next(iter(inferred)))] += 1
    assert orientations_by_polarity == Counter({
        ("positive", 0): 16, ("positive", 1): 16,
        ("negative", 0): 16, ("negative", 1): 16,
    })
    assert all(CLOSED_WORLD_CAVEAT in row["state"] for row in c)
    assert all(row["meta"]["question_style"] ==
               ("interrogative" if row["meta"]["s2_arm"] == "A" else "declarative") and
               row["meta"]["criteria_style"] ==
               ("explicit" if row["meta"]["s2_arm"] == "A" else "generic") for row in c)
    assert len({row["id"] for row in arm_a}) == len(arm_a)
    assert len({row["id"] for row in arm_b}) == len(arm_b)
    assert [(r["state"], r["targets"]) for r in a] == [(r["state"], r["targets"]) for r in b]
    assert Counter(row["targets"]["decision"]["label"] for row in a) == {"no": 256, "yes": 256}
    assert Counter(tuple(row["questions"]["decision"]["criteria"]) for row in a) == {
        ("no", "yes"): 256, ("yes", "no"): 256}
    assert Counter(row["meta"]["context_view"] for row in a) == {"compact": 256, "full": 256}
    assert Counter(row["meta"]["polarity"] for row in a[::8]) == {"positive": 32, "negative": 32}
    assert len({row["meta"]["source_group"] for row in a}) == 64
    assert all(row["meta"]["source_id"] for row in a + b)
    assert all(row["targets"]["decision"]["label"] ==
               (row["meta"]["source_label"] if row["meta"]["polarity"] == "positive"
                else ("yes" if row["meta"]["source_label"] == "no" else "no"))
               for row in a + b)
    assert all(row["questions"]["decision"]["criteria"] == {"no": "No", "yes": "Yes"}
               for row in b)
    assert {row["questions"]["decision"]["instructions"] for row in b} == {
        "The requested action is permitted under the stated policy.",
        "The requested action is not permitted under the stated policy.",
    }
    assert all(row["meta"]["fixed_option_order"] for row in a + b)
    for row_a, row_b in zip(a, b):
        assert row_a["meta"]["source_id"] == row_b["meta"]["source_id"]
        assert {k: v for k, v in row_a["meta"].items()
                if k not in {"s2_arm", "question_style", "criteria_style"}} == {
                    k: v for k, v in row_b["meta"].items()
                    if k not in {"s2_arm", "question_style", "criteria_style"}}
    assert flatten_row(b[0])[0]["fixed_order"] is True
    assert all(CLOSED_WORLD_CAVEAT in row["state"] and
               "Conditions the state does not establish" not in row["questions"]["decision"]["instructions"]
               for row in a + b)
    assert min(len(row["state"]) for row in a if row["meta"]["context_view"] == "full") >= 12_500
    assert all(row["state"].endswith(CLOSED_WORLD_CAVEAT)
               for row in a if row["meta"]["context_view"] == "compact"
               and row["meta"]["decisive_placement"] == "end")
    assert all(row["state"].startswith("State:")
               for row in a if row["meta"]["context_view"] == "full"
               and row["meta"]["decisive_placement"] == "beginning")
    assert all(row["state"].find("State:") > 0
               and row["state"].find("State:") < len(row["state"]) - 100
               for row in a if row["meta"]["context_view"] == "full"
               and row["meta"]["decisive_placement"] == "middle")
    assert len(crossed_eval) == 2048
    assert Counter(row["targets"]["decision"]["label"] for row in crossed_eval) == {
        "no": 1024, "yes": 1024}
    assert manifest["crossed_eval"]["cell_counts"]["each_of_16_cells"] == 128
    assert manifest["train_eval_group_intersection"] == 0
    assert manifest["train_eval_context_group_intersection"] == 0
    assert not set(manifest["selected_groups"]) & set(manifest["crossed_eval"]["groups"])
    assert not set(manifest["train_distractor_groups"]) & set(manifest["crossed_eval"]["distractor_groups"])
    assert manifest["output_sha256"] == {
        "arm_a": _hash_rows(arm_a), "arm_b": _hash_rows(arm_b),
        "arm_c": _hash_rows(arm_c),
        "crossed_eval": _hash_rows(crossed_eval),
    }
    assert manifest["arm_c"]["groups_each_with_4_A_4_B"] == 64
    assert manifest["arm_c"]["format_counts"] == {"A": 256, "B": 256}
    assert manifest["arm_c"]["orientation_groups_by_polarity"] == {
        "positive_0": 16, "positive_1": 16, "negative_0": 16, "negative_1": 16,
    }
    assert all(len(row["state"]) >= 40_000 for row in crossed_eval
               if row["meta"]["context_view"] == "full")
    assert max(len(row["state"]) for row in a if row["meta"]["context_view"] == "full") < 14_000
    assert max(len(row["state"]) for row in crossed_eval if row["meta"]["context_view"] == "full") < 42_000
    train_wordings = {row["questions"]["decision"]["instructions"] for row in a + b}
    assert not train_wordings & {row["questions"]["decision"]["instructions"] for row in crossed_eval}


def test_negative_propositions_rerender_style_markers_after_condition_inversion():
    for style, _ in NOUL_STYLES:
        for source_index in (0, 1):
            source = donor(f"style-{style}-{source_index}", source_index, style)
            source_label = "no" if source_index == 0 else "yes"
            for polarity in ("positive", "negative"):
                expected = _explicit_criteria(source, polarity)
                expected_label = source_label if polarity == "positive" else (
                    "yes" if source_label == "no" else "no")
                for arm in ("A", "B"):
                    row = _make_view(source, arm, "compact", ("no", "yes"), [], 0,
                                     f"style-{style}", polarity=polarity)
                    assert row["targets"]["decision"]["label"] == expected_label
                    criteria = row["questions"]["decision"]["criteria"]
                    if arm == "A":
                        assert criteria == expected
                    else:
                        assert criteria == {"no": "No", "yes": "Yes"}
                    expected_instruction = (
                        ARM_A_QUESTION if polarity == "positive" else
                        "Is the requested action not permitted under the stated policy?"
                    ) if arm == "A" else (
                        ARM_B_QUESTION if polarity == "positive" else
                        "The requested action is not permitted under the stated policy."
                    )
                    assert row["questions"]["decision"]["instructions"] == expected_instruction

                for wording in ("interrogative", "declarative"):
                    row = _make_view(source, "crossed_eval", "compact", ("yes", "no"), [], 1,
                                     f"style-{style}", question_style=wording,
                                     criteria_style="explicit", polarity=polarity)
                    assert row["targets"]["decision"]["label"] == expected_label
                    assert row["questions"]["decision"]["criteria"] == {
                        label: expected[label] for label in ("yes", "no")}
                    assert row["questions"]["decision"]["instructions"] == EVAL_QUESTIONS[(polarity, wording)]
