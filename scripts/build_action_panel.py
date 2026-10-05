"""Build a leakage-safe executable multi-decision workflow panel.

The panel is for internal renderer/calibration and action-loss experiments.  It
does not reuse public workflow cases: every field label and final action comes
from the pure `action_for` policy below.
"""

import argparse
import json
import random
from pathlib import Path


SPLITS = {
    "train": {"families": ("access", "commerce"), "templates": ("plain", "checklist"),
              "bands": ("low", "mid")},
    "cal": {"families": ("safety", "support"), "templates": ("review", "formal"),
            "bands": ("upper_low", "upper_mid")},
    "eval": {"families": ("finance", "operations"), "templates": ("terse", "explicit"),
             "bands": ("high", "off_grid")},
}
LIMITS = {
    "low": (100, 20), "mid": (250, 45), "upper_low": (175, 30),
    "upper_mid": (400, 60), "high": (700, 90), "off_grid": (325, 52),
}
DECISION_THRESHOLDS = {
    "low": 0.15, "mid": 0.30, "upper_low": 0.45,
    "upper_mid": 0.60, "high": 0.75, "off_grid": 0.52,
}
FAMILY_NAMES = {
    "access": "access request", "commerce": "purchase request", "safety": "safety review",
    "support": "support case", "finance": "payment review", "operations": "operations request",
}


def action_for(fields):
    """Map semantic field booleans to the policy's exact final action."""
    if not fields["safe"]:
        return "deny"
    if not fields["approved"] or any(not value for name, value in fields.items()
                                      if name not in {"safe", "approved", "within_limit"}):
        return "review"
    if "within_limit" in fields and not fields["within_limit"]:
        return "escalate"
    return "allow"


def _criteria(template, yes, no):
    if template in {"plain", "terse"}:
        return {"no": "No", "yes": "Yes"}
    return {"no": no, "yes": yes}


def _question(template, name, value_limit, age_limit, truth):
    yes, no = {
        "safe": ("The risk score is at or below the policy limit.",
                 "The risk score exceeds the policy limit."),
        "approved": ("A manager approval is recorded.", "No manager approval is recorded."),
        "within_limit": (f"The amount is at most ${value_limit}.", f"The amount exceeds ${value_limit}."),
        "timely": (f"The request is no older than {age_limit} days.",
                   f"The request is older than {age_limit} days."),
        "evidence": ("All required evidence is present.", "Required evidence is missing."),
        "covered": ("The request is covered by the policy.", "The request is not covered by the policy."),
    }[name]
    prompts = {
        "plain": f"Is this statement true: {yes}",
        "checklist": f"Check the case facts. Is the following true? {yes}",
        "review": f"As a reviewer, determine whether this is true: {yes}",
        "formal": f"Using only the recorded facts, decide: {yes}",
        "terse": yes,
        "explicit": f"Return true only if this condition holds: {yes}",
    }
    criteria = _criteria(template, yes, no)
    if not isinstance(truth, bool):
        raise ValueError("program facts must be Boolean")
    return {"type": "noul", "instructions": prompts[template], "criteria": criteria}


def _state(family, value_limit, age_limit, fields, rng):
    value = value_limit - rng.randint(1, 20) if fields.get("within_limit", True) else value_limit + rng.randint(1, 80)
    age = age_limit - rng.randint(0, 5) if fields.get("timely", True) else age_limit + rng.randint(1, 30)
    risk = 20 if fields["safe"] else 91
    facts = [
        f"Case type: {FAMILY_NAMES[family]}.", f"Recorded risk score: {risk}/100.",
        "Policy risk limit: 50/100.",
        f"Requested amount: ${value}.", f"Request age: {age} days.",
        "Manager approval: recorded." if fields["approved"] else "Manager approval: not recorded.",
    ]
    if "evidence" in fields:
        facts.append("Required evidence: complete." if fields["evidence"] else "Required evidence: missing.")
    if "covered" in fields:
        facts.append("Policy coverage: confirmed." if fields["covered"] else "Policy coverage: not confirmed.")
    rng.shuffle(facts)
    return "\n".join(facts)


def _truths(size, requested_action):
    names = ("safe", "approved", "within_limit", "timely", "evidence", "covered")[:size]
    action = requested_action if requested_action != "escalate" or size >= 3 else "allow"
    fields = {name: True for name in names}
    if action == "deny":
        fields["safe"] = False
    elif action == "review":
        fields["approved"] = False
    elif action == "escalate":
        fields["within_limit"] = False
    assert action_for(fields) == action
    return fields


def build(seed, *, split, cases_per_cell):
    """Return deterministic grouped rows and a minimal provenance manifest."""
    if split not in SPLITS or cases_per_cell <= 0:
        raise ValueError("unknown split or nonpositive cases_per_cell")
    rng, spec = random.Random(seed), SPLITS[split]
    rows, identifier, option_position = [], 0, 0
    for family in spec["families"]:
        for template in spec["templates"]:
            for band in spec["bands"]:
                value_limit, age_limit = LIMITS[band]
                for cell_index in range(cases_per_cell):
                    identifier += 1
                    size = 2 + cell_index % 5
                    fields = _truths(size, ("allow", "review", "escalate", "deny")[cell_index % 4])
                    questions, targets = {}, {}
                    for question_index, (name, truth) in enumerate(fields.items()):
                        question = _question(template, name, value_limit, age_limit, truth)
                        if option_position % 2:
                            question["criteria"] = dict(reversed(list(question["criteria"].items())))
                        option_position += 1
                        qid = f"q{question_index:02d}"
                        questions[qid], targets[qid] = question, {"label": "yes" if truth else "no"}
                    action = action_for(fields)
                    rows.append({
                        "id": f"ap-{split}-{identifier:06d}",
                        "state": _state(family, value_limit, age_limit, fields, rng),
                        "questions": questions, "targets": targets,
                        "meta": {"panel": "action_v1", "split": split, "family": family,
                                 "template": template, "threshold_band": band,
                                 "decision_threshold": DECISION_THRESHOLDS[band],
                                 "action_gold": action, "action_trace": fields,
                                 "group_id": f"{split}:{family}:{template}:{band}:{cell_index}"},
                    })
    validate(rows, split=split)
    return rows, {"panel": "action_v1", "seed": seed, "split": split,
                  "cases_per_cell": cases_per_cell, "rows": len(rows),
                  "families": list(spec["families"]), "templates": list(spec["templates"]),
                  "threshold_bands": list(spec["bands"])}


def validate(rows, *, split):
    spec = SPLITS[split]
    for row in rows:
        meta, questions, targets = row["meta"], row["questions"], row["targets"]
        if not 2 <= len(questions) <= 6 or set(questions) != set(targets):
            raise ValueError(f"{row['id']}: invalid question group")
        if (meta["family"] not in spec["families"] or meta["template"] not in spec["templates"]
                or meta["threshold_band"] not in spec["bands"]):
            raise ValueError(f"{row['id']}: split contamination")
        fields = {name: targets[f"q{index:02d}"]["label"] == "yes"
                  for index, name in enumerate(meta["action_trace"])}
        if fields != meta["action_trace"] or action_for(fields) != meta["action_gold"]:
            raise ValueError(f"{row['id']}: action trace mismatch")
        if any(question["type"] != "noul" or set(question["criteria"]) != {"no", "yes"}
               for question in questions.values()):
            raise ValueError(f"{row['id']}: invalid binary question")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data_action")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-cell", type=int, default=1000)
    parser.add_argument("--cal-per-cell", type=int, default=250)
    parser.add_argument("--eval-per-cell", type=int, default=250)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"panel": "action_v1", "seed": args.seed, "splits": {}}
    for split, count in (("train", args.train_per_cell), ("cal", args.cal_per_cell),
                         ("eval", args.eval_per_cell)):
        rows, report = build(args.seed, split=split, cases_per_cell=count)
        with (out / f"{split}.jsonl").open("w") as target:
            for row in rows:
                target.write(json.dumps(row, sort_keys=True) + "\n")
        manifest["splits"][split] = report
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
