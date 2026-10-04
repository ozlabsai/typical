"""Compare two saved WorkflowEvals runs without changing or refitting labels.

Usage: uv run python scripts/compare_workflow_runs.py RUN_A_DIR RUN_B_DIR
Each directory contains scores.json and results.json from one workflow run.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path):
    with open(path) as source:
        return json.load(source)


def score_rows(run_dir, label_set):
    data = read_json(Path(run_dir) / "scores.json")
    rows = [row for row in data["scores"] if row["label_set_id"] == label_set]
    if not rows:
        raise ValueError(f"{run_dir}: no score rows for {label_set}")
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError(f"{run_dir}: expected one run for {label_set}, found {sorted(run_ids)}")
    return {(row["case_id"], row["policy_id"]): row["metrics"] for row in rows}


def summarize_probabilities(run_dir):
    """Only compute margins when an artifact records an explicit threshold."""
    data = read_json(Path(run_dir) / "results.json")
    recorded = 0
    margins = []
    for run in data.get("runs", []):
        for case in run.get("cases", []):
            for obs in case.get("observations", []):
                probs = obs.get("probabilities")
                if not isinstance(probs, dict):
                    continue
                recorded += 1
                threshold = obs.get("threshold")
                if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
                    if set(probs) >= {"true", "false"}:
                        margins.append(abs(float(probs["true"]) - float(threshold)))
    return {"probability_observations": recorded, "explicit_threshold_margins": margins}


def compare(run_a, run_b, label_set="labels-consensus"):
    left, right = score_rows(run_a, label_set), score_rows(run_b, label_set)
    shared = sorted(left.keys() & right.keys())
    transitions = Counter()
    per_run = []
    for rows in (left, right):
        exact_yes = primary_yes = 0
        for key in shared:
            metrics = rows[key]
            exact = metrics.get("exact_actions")
            primary = metrics.get("primary_action")
            if isinstance(exact, bool):
                exact_yes += exact
            if isinstance(primary, bool):
                primary_yes += primary
        per_run.append({"exact_correct": exact_yes, "primary_correct": primary_yes,
                        "primary_only_correct": sum(rows[k].get("primary_action") is True and
                                                     rows[k].get("exact_actions") is False for k in shared),
                        "exact_only_correct": sum(rows[k].get("primary_action") is False and
                                                  rows[k].get("exact_actions") is True for k in shared)})
    for key in shared:
        a, b = left[key].get("exact_actions"), right[key].get("exact_actions")
        if isinstance(a, bool) and isinstance(b, bool):
            transitions[f"{'correct' if a else 'wrong'}_to_{'correct' if b else 'wrong'}"] += 1
    return {
        "label_set": label_set,
        "shared_case_policy_pairs": len(shared),
        "exact_action_transitions": dict(sorted(transitions.items())),
        "run_a_primary_exact_gap": per_run[0],
        "run_b_primary_exact_gap": per_run[1],
        "run_a_probabilities": summarize_probabilities(run_a),
        "run_b_probabilities": summarize_probabilities(run_b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--label-set", default="labels-consensus")
    args = parser.parse_args()
    print(json.dumps(compare(args.run_a, args.run_b, args.label_set), indent=2))


if __name__ == "__main__":
    main()
