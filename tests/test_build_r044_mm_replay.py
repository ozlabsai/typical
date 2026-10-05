import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_r043_causal_pairs import _source
from scripts.build_r044_mm_replay import NULL_OPTION, build
from scripts.train_gemma_decision_lora import load_examples


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def _single(index, source):
    return {
        "id": f"{source}-{index}", "state": f"state {source} {index}",
        "questions": {"decision": {"type": "choice", "instructions": "Choose.",
                                     "criteria": {"no": "No", "yes": "Yes"}}},
        "targets": {"decision": {"label": "yes" if index % 2 else "no"}},
        "meta": {"source": source, "license": "fixture-license", "source_id": f"{source}-{index}",
                 **({"renderer": "prose"} if source == "r042_programmatic" else {})},
    }


def _flat(index, source, choices=12):
    return {
        "state": f"flat state {index}", "query": f"flat query {index}",
        "candidates": [f"choice {index}-{choice}" for choice in range(choices)],
        "target": [float(choice == index % choices) for choice in range(choices)],
        "label": index % choices, "p_null": 0.0,
        "meta": {"source": source, "license": "fixture-license", "source_id": f"{source}-{index}",
                 **({"lang": "he"} if source == "MASSIVE" else {})},
    }


def _fixture(tmp_path, broad_count=200, visual_safe=200):
    r043 = [_source("train", "invoice", index, 43, 0, index) for index in range(100)]
    prose = [_single(index, "r042_programmatic") for index in range(200)]
    replay = [_single(index, "broad") for index in range(broad_count)]
    hebrew = [_flat(index, "MASSIVE") for index in range(100)]

    image_dir = tmp_path / "science" / "images"
    image_dir.mkdir(parents=True)
    protected_image = image_dir / "protected.png"
    protected_image.write_bytes(b"protected image")
    protected = _flat(999, "ScienceQA", 4)
    protected["image"] = "images/protected.png"

    visual = []
    same_image = _flat(10_000, "ScienceQA", 4)
    same_image["image"] = "images/protected.png"
    visual.append(same_image)
    same_decision = dict(protected)
    same_decision["meta"] = {**protected["meta"], "source_id": "same-decision"}
    decision_image = image_dir / "same-decision.png"
    decision_image.write_bytes(b"same decision different image")
    same_decision["image"] = "images/same-decision.png"
    visual.append(same_decision)
    for index in range(visual_safe):
        row = _flat(index, "ScienceQA", 4)
        path = image_dir / f"safe-{index}.png"
        path.write_bytes(f"safe image {index}".encode())
        row["image"] = f"images/{path.name}"
        visual.append(row)

    paths = {
        "r043": _write(tmp_path / "r043.jsonl", r043),
        "prose": _write(tmp_path / "prose.jsonl", prose),
        "replay": _write(tmp_path / "replay.jsonl", replay),
        "hebrew": _write(tmp_path / "hebrew.jsonl", hebrew),
        "science": _write(tmp_path / "science" / "train.jsonl", visual),
        "protect": _write(tmp_path / "science" / "protect.jsonl", [protected]),
        "out": tmp_path / "out",
    }
    return paths


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_builds_matched_leakage_free_arms(tmp_path):
    paths = _fixture(tmp_path)
    manifest = build(paths["r043"], paths["prose"], paths["replay"], paths["hebrew"],
                     paths["science"], [paths["protect"]], paths["out"], seed=44)
    arm_a, arm_b = _rows(paths["out"] / "arm-a.jsonl"), _rows(paths["out"] / "arm-b.jsonl")

    assert len(arm_a) == len(arm_b) == 800
    assert len(load_examples([paths["out"] / "arm-a.jsonl"])) == 800
    assert len(load_examples([paths["out"] / "arm-b.jsonl"])) == 800
    assert arm_a == [{key: value for key, value in row.items() if key != "image"} for row in arm_b]
    assert Counter((row.get("meta") or {}).get("r044_category") for row in arm_b) == {
        "counterfactual": 100, "prose": 200, "replay": 200, "hebrew": 100, "visual": 200}

    hebrew = [row for row in arm_b if row["meta"]["r044_category"] == "hebrew"]
    assert {len(row["questions"]["decision"]["criteria"]) for row in hebrew} == {10}
    assert Counter(row["meta"]["r044_gold_position"] for row in hebrew) == {index: 10 for index in range(10)}

    visual = [row for row in arm_b if row["meta"]["r044_category"] == "visual"]
    assert Counter(row["meta"]["r044_variant"] for row in visual) == {
        "real": 100, "mismatch": 50, "gold_dropped": 50}
    assert all(list(row["questions"]["decision"]["criteria"].values())[-1] == NULL_OPTION for row in visual)
    assert all((paths["out"] / row["image"]).is_file() for row in visual)
    assert all(row["meta"]["r044_base_image_sha256"] != row["meta"]["r044_assigned_image_sha256"]
               for row in visual if row["meta"]["r044_variant"] == "mismatch")
    assert all(row["targets"]["decision"]["label"] == "null"
               for row in visual if row["meta"]["r044_variant"] != "real")
    assert "ScienceQA-10000" not in {row["meta"]["source_id"] for row in visual}
    assert "same-decision" not in {row["meta"]["source_id"] for row in visual}

    assert manifest["leakage_checks"]["protected_image_overlap"] == 0
    assert manifest["leakage_checks"]["protected_decision_overlap"] == 0
    assert manifest["images"]["files"] == 200
    assert manifest["outputs"]["arm-a"]["sha256"] != manifest["outputs"]["arm-b"]["sha256"]
    assert manifest["inputs"]["scienceqa_protection"][0]["sha256"]
    assert len(manifest["selection"]["hebrew"]) == 100
    assert len(manifest["selection"]["visual"]) == 200


@pytest.mark.parametrize("shortage", ["counterfactual", "prose", "replay", "hebrew", "visual"])
def test_fails_closed_on_category_shortage(tmp_path, shortage):
    paths = _fixture(tmp_path)
    path = paths[{"counterfactual": "r043", "prose": "prose", "replay": "replay",
                  "hebrew": "hebrew", "visual": "science"}[shortage]]
    rows = _rows(path)
    if shortage == "visual":
        rows = rows[:-1]
    else:
        rows = rows[:{"counterfactual": 99, "prose": 199, "replay": 199, "hebrew": 99}[shortage]]
    _write(path, rows)
    with pytest.raises(ValueError, match="needs (100 valid counterfactual|200 valid prose|200 valid replay|100 valid hebrew|200 unique leakage-free visual)"):
        build(paths["r043"], paths["prose"], paths["replay"], paths["hebrew"],
              paths["science"], [paths["protect"]], paths["out"], seed=44)


def test_requires_protection_inputs(tmp_path):
    with pytest.raises(ValueError, match="protection path"):
        build("missing", "missing", "missing", "missing", "missing", [], tmp_path / "out")
