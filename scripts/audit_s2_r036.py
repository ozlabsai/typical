"""Apply the locked R036 retention and promotion gates."""

import argparse
import json
import math
from pathlib import Path

from scripts.audit_s2_r035 import (
    _accuracy,
    _final_validation,
    _interface_middle,
    _only_result,
    _overall,
    _read_json,
    _sha256,
    _slice,
)


FORMAT_FLOORS = {
    "declarative|explicit": 0.938984375,
    "declarative|generic": 0.93703125,
    "interrogative|explicit": 0.94875,
    "interrogative|generic": 0.942890625,
}


def _format_polarity(rows):
    cells = {}
    for row in rows:
        key = "|".join((row["question_style"], row["criteria_style"], row["polarity"]))
        cells.setdefault(key, []).append(row)
    return {key: _accuracy(cell) for key, cell in sorted(cells.items())}


def _formats(rows):
    cells = {}
    for row in rows:
        key = "|".join((row["question_style"], row["criteria_style"]))
        cells.setdefault(key, []).append(row)
    return {key: _accuracy(cell) for key, cell in sorted(cells.items())}


def _crossed_cells(rows):
    cells = {}
    for row in rows:
        key = "|".join((row["question_style"], row["criteria_style"], row["polarity"],
                        row["context_view"], "_".join(row["option_order"])))
        cells.setdefault(key, []).append(row)
    return {key: _accuracy(cell) for key, cell in sorted(cells.items())}


def _final_projection(path, expected_steps=250):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    stepped = [row for row in rows if isinstance(row.get("step"), int) and row["step"] > 0]
    by_step = {row["step"]: row for row in stepped}
    if len(by_step) != len(stepped) or set(by_step) != set(range(1, expected_steps + 1)):
        raise ValueError(f"candidate metrics must contain unique complete steps 1..{expected_steps}")
    if any("early_stop" in row for row in stepped):
        raise ValueError("candidate metrics contain an early-stop event")
    final = by_step[expected_steps]
    if "validation" not in final:
        raise ValueError(f"candidate metrics need one validated step {expected_steps}")
    energies = []
    for step in range(1, expected_steps + 1):
        row = by_step[step]
        values = (row.get("projection_removed_energy"), row.get("projection_cons_energy"))
        if any(value is None or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"step {step} lacks finite nonnegative projection energies")
        energies.append(values)
    def ratio(through):
        removed = sum(item[0] for item in energies[:through])
        consistency = sum(item[1] for item in energies[:through])
        return removed / consistency if consistency else 0.0
    step_50_ratio, final_ratio = ratio(50), ratio(expected_steps)
    if step_50_ratio < 0.01:
        raise ValueError("candidate should have stopped at step 50 for inactive projection")
    logged_ratio = final.get("projection_cumulative_removed_energy_fraction")
    if logged_ratio is None or not math.isclose(logged_ratio, final_ratio, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("logged final projection ratio does not match raw step energies")
    return final["validation"], final_ratio


def audit(baseline_crossed, reference_crossed, candidate_crossed, baseline_metrics,
          candidate_metrics, expected_rows=2048, expected_steps=250):
    base_payload = _read_json(baseline_crossed)
    ref_payload = _read_json(reference_crossed)
    cand_payload = _read_json(candidate_crossed)
    hashes = {payload.get("data_sha256") for payload in (base_payload, ref_payload, cand_payload)}
    if len(hashes) != 1:
        raise ValueError("D, R035, and R036 must use identical crossed data")
    base = _only_result(base_payload, expected_rows)
    ref = _only_result(ref_payload, expected_rows)
    cand = _only_result(cand_payload, expected_rows)
    base_rows, ref_rows, cand_rows = (result["predictions"] for result in (base, ref, cand))
    identities = [{row["example_id"] for row in rows} for rows in (base_rows, ref_rows, cand_rows)]
    if identities[0] != identities[1] or identities[0] != identities[2]:
        raise ValueError("D, R035, and R036 prediction identities differ")

    target = {"question_style": "declarative", "criteria_style": "generic",
              "polarity": "negative"}
    target_full_no = {**target, "context_view": "full", "gold": "no"}
    cand_target = _accuracy(_slice(cand_rows, **target))
    cand_full_no = _accuracy(_slice(cand_rows, **target_full_no))
    cand_interface = _interface_middle(cand_rows)
    cand_overall = _overall(cand_rows)
    ref_overall = _overall(ref_rows)

    base_cells, cand_cells = _crossed_cells(base_rows), _crossed_cells(cand_rows)
    if set(base_cells) != set(cand_cells):
        raise ValueError("candidate crossed-cell coverage differs from D")
    target_prefix = "declarative|generic|negative|"
    protected_deltas = {key: cand_cells[key] - base_cells[key]
                        for key in base_cells if not key.startswith(target_prefix)}
    target_full_cells = {key: value for key, value in cand_cells.items()
                         if key.startswith(target_prefix + "full|")}
    format_polarity = _format_polarity(cand_rows)
    formats = _formats(cand_rows)
    format_floor_passes = {key: formats[key] >= floor for key, floor in FORMAT_FLOORS.items()}
    recalls = {gold: _accuracy(_slice(cand_rows, gold=gold)) for gold in ("no", "yes")}
    both_orders_correct = cand["order_robustness"]["both_correct_rate"]

    base_validation = _final_validation(baseline_metrics, expected_steps)
    cand_validation, projection_ratio = _final_projection(candidate_metrics, expected_steps)
    retention_deltas = {name: cand_validation[name]["accuracy"] - base_validation[name]["accuracy"]
                        for name in ("wf", "wh", "replay")}
    cand_order = cand["order_robustness"]["mean_abs_aligned_p_yes_delta"]

    gates = {
        "projection_removed_energy_at_least_0.01": projection_ratio >= 0.01,
        "target_accuracy_at_least_0.9448": cand_target >= 0.9448,
        "full_negative_no_recall_at_least_0.9331": cand_full_no >= 0.9331,
        "interface_b_accuracy_at_least_0.905": cand_interface["b_accuracy"] >= 0.905,
        "each_target_full_order_cell_at_least_0.90": min(target_full_cells.values()) >= 0.90,
        "each_format_polarity_at_least_0.90": min(format_polarity.values()) >= 0.90,
        "declarative_generic_aggregate_at_least_0.95": formats["declarative|generic"] >= 0.95,
        "each_format_retains_stronger_endpoint_minus_0.02": all(format_floor_passes.values()),
        "each_label_recall_at_least_0.90": min(recalls.values()) >= 0.90,
        "both_orders_correct_at_least_0.90": both_orders_correct >= 0.90,
        "each_protected_cell_drop_at_most_0.02": min(protected_deltas.values()) >= -0.02,
        "wf_drop_at_most_0.02": retention_deltas["wf"] >= -0.02,
        "wh_drop_at_most_0.02": retention_deltas["wh"] >= -0.02,
        "replay_drop_at_most_0.02": retention_deltas["replay"] >= -0.02,
        "replay_accuracy_at_least_0.8628125": cand_validation["replay"]["accuracy"] >= 0.8628125,
        "order_delta_at_most_0.0396": cand_order <= 0.0396,
        "overall_nll_at_most_1.05x_r035": cand_overall["nll"] <= 1.05 * ref_overall["nll"],
        "overall_brier_at_most_1.05x_r035": cand_overall["brier"] <= 1.05 * ref_overall["brier"],
    }
    return {
        "baseline_crossed_sha256": _sha256(baseline_crossed),
        "reference_crossed_sha256": _sha256(reference_crossed),
        "candidate_crossed_sha256": _sha256(candidate_crossed),
        "baseline_metrics_sha256": _sha256(baseline_metrics),
        "candidate_metrics_sha256": _sha256(candidate_metrics),
        "crossed_data_sha256": hashes.pop(),
        "candidate": {"target_accuracy": cand_target,
                      "full_negative_no_recall": cand_full_no,
                      "interface": cand_interface, "overall": cand_overall,
                      "order_delta": cand_order, "format_polarity": format_polarity,
                      "formats": formats, "format_floor_passes": format_floor_passes,
                      "label_recalls": recalls, "both_orders_correct": both_orders_correct,
                      "target_full_cells": target_full_cells},
        "retention_deltas_vs_d": retention_deltas,
        "protected_cell_deltas_vs_d": protected_deltas,
        "projection_cumulative_removed_energy_fraction": projection_ratio,
        "gates": gates,
        "promotion_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-crossed", required=True)
    parser.add_argument("--reference-crossed", required=True)
    parser.add_argument("--candidate-crossed", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.baseline_crossed, args.reference_crossed, args.candidate_crossed,
                   args.baseline_metrics, args.candidate_metrics)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
