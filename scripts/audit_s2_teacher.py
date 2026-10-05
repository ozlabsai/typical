"""Apply the pre-registered identity-aligned R034 teacher pass rule."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _read_panel(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 256:
        raise ValueError(f"canonical panel must contain exactly 256 rows; got {len(rows)}")
    by_source = defaultdict(list)
    by_id = {}
    for row in rows:
        meta = row.get("meta") or {}
        q = row.get("questions", {}).get("decision", {})
        source_id = meta.get("source_id")
        order = tuple(q.get("criteria", {}).keys())
        if not source_id or row.get("id") in by_id:
            raise ValueError("canonical panel has missing source IDs or duplicate row IDs")
        if (order not in (("no", "yes"), ("yes", "no")) or
                row.get("targets", {}).get("decision", {}).get("label") not in ("no", "yes")):
            raise ValueError(f"{row['id']}: malformed gold label or option order")
        if (meta.get("question_style") != "interrogative" or
                meta.get("criteria_style") != "explicit" or meta.get("polarity") != "positive" or
                meta.get("context_view") != "compact" or meta.get("distractor_groups") != [] or
                not meta.get("rubric_style") or
                meta.get("arm_d_train_polarity") not in ("positive", "negative")):
            raise ValueError(f"{row['id']}: canonical teacher metadata mismatch")
        by_id[row["id"]] = row
        by_source[source_id].append(row)
    if len(by_source) != 128 or any(len(pair) != 2 for pair in by_source.values()):
        raise ValueError("canonical panel must contain exactly 128 two-order source pairs")
    source_labels = Counter(pair[0]["targets"]["decision"]["label"]
                            for pair in by_source.values())
    if source_labels != {"no": 64, "yes": 64}:
        raise ValueError(f"canonical source labels must be balanced 64/64: {dict(source_labels)}")
    for source_id, pair in by_source.items():
        orders = [tuple(row["questions"]["decision"]["criteria"]) for row in pair]
        labels = {row["targets"]["decision"]["label"] for row in pair}
        if Counter(orders) != {("no", "yes"): 1, ("yes", "no"): 1} or len(labels) != 1:
            raise ValueError(f"canonical source pair malformed: {source_id}")
        if any(len({row["meta"][key] for row in pair}) != 1 for key in
               ("source_group", "rubric_style", "arm_d_train_polarity")):
            raise ValueError(f"canonical pair metadata differs across orders: {source_id}")
    return raw, rows, by_id, by_source


def _coverage(cases, field):
    totals, eligible = Counter(), Counter()
    for case in cases:
        value = case[field]
        totals[value] += 1
        eligible[value] += int(case["eligible"])
    return {key: {"eligible": eligible[key], "total": total,
                  "rate": eligible[key] / total}
            for key, total in sorted(totals.items())}


def audit_results(result_path, panel_path):
    panel_raw, panel_rows, panel_by_id, panel_by_source = _read_panel(panel_path)
    result_raw = Path(result_path).read_bytes()
    result = json.loads(result_raw)
    panel_hash = hashlib.sha256(panel_raw).hexdigest()
    if result.get("data_sha256") != panel_hash:
        raise ValueError("eval result data_sha256 does not match canonical panel")
    if result.get("complete") is not True or result.get("subset") != "all" or result.get("rows") != 256:
        raise ValueError("eval result must be complete over all 256 canonical rows")
    if len(result.get("results", [])) != 1:
        raise ValueError("eval result must contain exactly one adapter result")
    predictions = result["results"][0].get("predictions", [])
    if len(predictions) != 256:
        raise ValueError(f"eval result must have 256 predictions; got {len(predictions)}")
    predictions_by_id = {}
    for prediction in predictions:
        example_id = prediction.get("example_id")
        if example_id not in panel_by_id or example_id in predictions_by_id:
            raise ValueError(f"unknown or duplicate prediction identity: {example_id}")
        row = panel_by_id[example_id]
        meta = row["meta"]
        q = row["questions"]["decision"]
        order = tuple(q["criteria"].keys())
        gold = row["targets"]["decision"]["label"]
        if (prediction.get("source_id") != meta["source_id"] or
                prediction.get("source_group") != meta["source_group"] or
                tuple(prediction.get("option_order", ())) != order or
                prediction.get("gold") != gold):
            raise ValueError(f"prediction identity/gold mismatch: {example_id}")
        p_yes = prediction.get("p_yes")
        if not isinstance(p_yes, (int, float)) or not math.isfinite(p_yes) or not 0 <= p_yes <= 1:
            raise ValueError(f"invalid p_yes for {example_id}")
        if prediction.get("predicted") not in ("no", "yes") or \
                prediction.get("correct") != (prediction["predicted"] == gold):
            raise ValueError(f"prediction correctness mismatch: {example_id}")
        predictions_by_id[example_id] = prediction
    if set(predictions_by_id) != set(panel_by_id):
        raise ValueError("prediction IDs do not exactly cover canonical panel IDs")

    cases = []
    for source_id, pair in sorted(panel_by_source.items()):
        row_by_order = {tuple(row["questions"]["decision"]["criteria"]): row for row in pair}
        no_yes = row_by_order[("no", "yes")]
        yes_no = row_by_order[("yes", "no")]
        meta = no_yes["meta"]
        gold = no_yes["targets"]["decision"]["label"]
        pred_pair = [predictions_by_id[no_yes["id"]], predictions_by_id[yes_no["id"]]]
        if any(pred["gold"] != gold for pred in pred_pair):
            raise ValueError(f"gold labels differ across option orders: {source_id}")
        p_yes = [float(pred["p_yes"]) for pred in pred_pair]
        p_gold = [value if gold == "yes" else 1.0 - value for value in p_yes]
        both_correct = all(pred["predicted"] == gold for pred in pred_pair)
        delta = abs(p_yes[0] - p_yes[1])
        reasons = []
        if not both_correct:
            reasons.append("not_both_correct")
        if min(p_gold) < 0.80 - 1e-12:
            reasons.append("min_p_gold_below_0.80")
        if delta > 0.05 + 1e-12:
            reasons.append("p_yes_delta_above_0.05")
        cases.append({
            "source_id": source_id, "source_group": meta["source_group"], "gold": gold,
            "train_polarity": meta["arm_d_train_polarity"], "rubric_style": meta["rubric_style"],
            "p_yes": {"no_yes": p_yes[0], "yes_no": p_yes[1]},
            "p_gold": {"no_yes": p_gold[0], "yes_no": p_gold[1]},
            "min_p_gold": min(p_gold), "p_yes_delta": delta,
            "both_correct": both_correct, "eligible": not reasons, "reasons": reasons,
        })
    gold_coverage = _coverage(cases, "gold")
    eligible_total = sum(case["eligible"] for case in cases)
    passed = (eligible_total >= 116 and
              all(gold_coverage[label]["eligible"] >= 58 for label in ("no", "yes")))
    return {
        "panel_sha256": panel_hash,
        "results_sha256": hashlib.sha256(result_raw).hexdigest(),
        "model": result.get("model"), "adapter": result["results"][0].get("adapter"),
        "rows": 256, "source_cases": len(cases),
        "eligible": eligible_total, "pass": passed,
        "gold_coverage": gold_coverage,
        "train_polarity_coverage": _coverage(cases, "train_polarity"),
        "rubric_style_coverage": _coverage(cases, "rubric_style"),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="raw eval_s2_crossed result JSON")
    parser.add_argument("--panel", required=True, help="canonical-teacher.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit_results(args.results, args.panel)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("panel_sha256", "rows", "source_cases", "eligible", "pass")}))


if __name__ == "__main__":
    main()
