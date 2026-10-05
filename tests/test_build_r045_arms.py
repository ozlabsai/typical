import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_r045_arms import (
    COUNTS,
    DEFECTIVE_REPLAY_IDS,
    EXPECTED_PAIR_SLOTS,
    RETENTION_ROWS,
    SHARED_REPEAT_ROWS,
    build,
)
from scripts.build_r045_compositional_transfer import build as build_compositional


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def _view(identifier, label, source_id=None):
    source_id = source_id or identifier
    return {"id": identifier, "state": f"synthetic state {identifier}",
            "questions": {"decision": {"type": "choice", "instructions": "Choose.",
                                          "criteria": {"yes": "Yes", "no": "No"}}},
            "targets": {"decision": {"label": label}},
            "meta": {"source_id": source_id}}


def _pair(identifier, source="r045_generated", split="train", language="en", twin=None):
    source_id = f"source-{identifier}"
    return {"id": identifier, "pair_mode": "counterfactual", "source_ids": [source_id, source_id],
            "paired_views": [_view(f"{identifier}-yes", "yes", source_id),
                             _view(f"{identifier}-no", "no", source_id)],
            "meta": {"source": source, "source_id": source_id, "split": split,
                     "language": language, "twin_id": twin or identifier,
                     "composition": ["decimal"]}}


def _single(identifier, category, source, extra_meta=None, image=None):
    row = {"id": identifier, "state": f"state {identifier}",
           "questions": {"decision": {"type": "choice", "instructions": "Choose.",
                                         "criteria": {"yes": "Yes", "no": "No"}}},
           "targets": {"decision": {"label": "yes"}},
           "meta": {"r044_category": category, "source": source,
                    **(extra_meta or {})}}
    if image:
        row["image"] = image
    return row


def _fixture(tmp_path, wh_count=320):
    r044_dir = tmp_path / "r044"
    image_dir = r044_dir / "images"
    image_dir.mkdir(parents=True)
    r044 = []
    r044.extend(_pair(f"r044-counterfactual-{i:04d}", source="r043_programmatic")
                for i in range(100))
    for row in r044[:100]:
        row["meta"]["r044_category"] = "counterfactual"
    r044.extend(_single(f"r044-prose-{i:04d}", "prose", "r042_programmatic",
                        {"renderer": "prose"}) for i in range(200))
    r044.extend(_single(f"r044-replay-{i:04d}", "replay", "archive") for i in range(200))
    r044.extend(_single(f"r044-hebrew-{i:04d}", "hebrew", "MASSIVE", {"lang": "he"})
                for i in range(100))
    for index in range(200):
        image_name = f"asset-{index:03d}.png"
        (image_dir / image_name).write_bytes(f"image bytes {index}".encode())
        r044.append(_single(f"r044-visual-{index:04d}", "visual", "ScienceQA",
                            {"r044_variant": ("real", "mismatch", "gold_dropped")[index % 3]},
                            f"images/{image_name}"))
    r044_path = _write(r044_dir / "arm-b.jsonl", r044)

    # Intern-adapter output has one grouped record with one or more questions.
    wh_rows = []
    for index in range(wh_count):
        wh_rows.append({"id": f"wh-{index:04d}", "state": f"WH state {index}",
                        "questions": {"q000": {"type": "choice", "instructions": "Choose.",
                                                "criteria": {"a": "A", "b": "B"}}},
                        "targets": {"q000": {"label": "a"}}})
    wh_path = _write(tmp_path / "wh.jsonl", wh_rows)

    compositional_dir = tmp_path / "compositional"
    build_compositional(compositional_dir, seed=45)
    compositional_path = compositional_dir / "train.jsonl"
    return r044_path, wh_path, compositional_path, set(row["id"] for row in r044)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_builds_matched_800_row_arms_with_clean_common_rows_and_assets(tmp_path):
    r044, wh, compositional, r044_ids = _fixture(tmp_path)
    out = tmp_path / "out"
    manifest = build(r044, wh, compositional, out, seed=45)
    arm_a, arm_b = _rows(out / "arm-a.jsonl"), _rows(out / "arm-b.jsonl")

    assert len(arm_a) == len(arm_b) == 800
    assert len({row["id"] for row in arm_a}) == len({row["id"] for row in arm_b}) == 800
    assert {row["id"] for row in arm_a if "r045_repeat_of" not in row["meta"]} == (
        r044_ids - DEFECTIVE_REPLAY_IDS)
    common = {row["id"] for row in arm_a} & {row["id"] for row in arm_b}
    assert len(common) == RETENTION_ROWS == 400
    assert {row["id"] for row in arm_a} & {row["id"] for row in arm_b} == common
    a_by_id = {row["id"]: row for row in arm_a}
    b_by_id = {row["id"]: row for row in arm_b}
    assert all(a_by_id[row_id] == b_by_id[row_id] for row_id in common)
    assert all(bool(a_by_id[row_id].get("image")) == bool(b_by_id[row_id].get("image"))
               for row_id in common)
    assert all((out / row["image"]).is_file() for row in arm_a if row.get("image"))
    assert Counter(row["meta"]["r044_category"] for row in arm_b
                   if row["id"] in common and "r045_repeat_of" not in row["meta"]) == {
        "hebrew": 100, "visual": 200, "prose": 44, "replay": 45}
    repeats = [row for row in arm_b if "r045_repeat_of" in row.get("meta", {})]
    assert len(repeats) == SHARED_REPEAT_ROWS == 11
    assert all("paired_views" not in row for row in repeats)
    assert {row["id"] for row in repeats} <= common
    assert Counter((row.get("meta") or {}).get("source") for row in arm_b
                   if (row.get("meta") or {}).get("source") == "r045_generated") == {
        "r045_generated": 100}
    wh_selected = [row for row in arm_b
                   if not row.get("meta", {}).get("r044_category") and row.get("id", "").startswith("wh-")]
    assert len(wh_selected) == COUNTS["wh"]
    assert all(len(row["questions"]) == 1 and "::" in row["id"] for row in wh_selected)
    selected_scenarios = Counter(row["meta"]["scenario_id"] for row in arm_b
                                 if row.get("meta", {}).get("source") == "r045_generated")
    assert set(selected_scenarios.values()) <= {2, 4}
    assert not ({row["id"] for row in arm_b} & DEFECTIVE_REPLAY_IDS)
    assert manifest["logical_rows"] == {"arm-a": 800, "arm-b": 800}
    assert manifest["composition"]["common_retention"] == 400
    assert manifest["composition"]["shared_repeats"] == 11
    assert manifest["compute_contract"]["matched_presentations"]
    assert not manifest["compute_contract"]["matched_state_characters"]
    assert {arm: values["presentations"] for arm, values in
            manifest["compute_contract"]["exposure"].items()} == {"arm-a": 900, "arm-b": 900}
    assert {arm: values["pair_slots"] for arm, values in
            manifest["compute_contract"]["exposure"].items()} == {
        "arm-a": EXPECTED_PAIR_SLOTS, "arm-b": EXPECTED_PAIR_SLOTS}
    assert manifest["inputs"]["r044_arm_b"]["sha256"]
    assert manifest["outputs"]["arm-a"]["sha256"]
    assert manifest["images"]["files"] == 200


def test_seed_repeats_selection_order_and_hashes(tmp_path):
    r044, wh, compositional, _ = _fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest_a = build(r044, wh, compositional, first, seed=18)
    manifest_b = build(r044, wh, compositional, second, seed=18)
    for filename in ("arm-a.jsonl", "arm-b.jsonl"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert manifest_a["selection"] == manifest_b["selection"]
    assert manifest_a["outputs"] == manifest_b["outputs"]


@pytest.mark.parametrize("source,mutate", [
    ("r044", lambda paths: paths[0].write_text(paths[0].read_text().replace("r044-replay-0010", "r044-replay-9000"))),
    ("wh", lambda paths: paths[1].write_text(paths[1].read_text().replace("wh-0000", "r044-prose-0000", 1))),
    ("compositional", lambda paths: paths[2].write_text(paths[2].read_text().replace("r045_generated", "unknown", 1))),
])
def test_fails_closed_on_defective_inputs(tmp_path, source, mutate):
    paths = list(_fixture(tmp_path)[:3])
    mutate(paths)
    with pytest.raises(ValueError):
        build(*paths, tmp_path / "out")


@pytest.mark.parametrize("filename,text", [
    ("JevBench-input.jsonl", None),
    ("ordinary.jsonl", "copied WorkflowEvals case text"),
])
def test_rejects_benchmark_source_paths_and_content(tmp_path, filename, text):
    r044, wh, compositional, _ = _fixture(tmp_path)
    if text:
        wh = _write(tmp_path / filename, [{"id": "x", "state": text, "questions": {}, "targets": {}}])
    else:
        renamed = tmp_path / filename
        renamed.write_bytes(r044.read_bytes())
        r044 = renamed
    with pytest.raises(ValueError, match="prohibited benchmark source"):
        build(r044, wh, compositional, tmp_path / "out")
