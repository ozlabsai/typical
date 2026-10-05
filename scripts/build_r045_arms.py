"""Build matched 789-row R045 control and treatment training arms."""

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


COUNTS = {"hebrew": 100, "visual": 200, "prose": 44, "replay": 45,
          "wh": 272, "compositional": 128}
DEFECTIVE_REPLAY_IDS = {f"r044-replay-{index:04d}" for index in
                        (10, 25, 72, 79, 85, 86, 108, 137, 147, 161, 180)}
EXPECTED_ARM_ROWS = 789
RETENTION_ROWS = sum(COUNTS[name] for name in ("hebrew", "visual", "prose", "replay"))
BENCHMARK_RE = re.compile(r"jev[\W_]*bench|workflow[\W_]*evals", re.IGNORECASE)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    try:
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL input: {path}") from error
    _reject_benchmark(path, rows)
    info = {"path": str(path), "rows": len(rows), "sha256": _sha(raw)}
    return rows, info


def _reject_benchmark(path, rows):
    if BENCHMARK_RE.search(str(path)):
        raise ValueError(f"prohibited benchmark source path: {path}")

    def visit(value):
        if isinstance(value, dict):
            return any(visit(key) or visit(item) for key, item in value.items())
        if isinstance(value, list):
            return any(visit(item) for item in value)
        return isinstance(value, str) and bool(BENCHMARK_RE.search(value))

    if visit(rows):
        raise ValueError(f"prohibited benchmark source content: {path}")


def _criteria_count(question):
    criteria = question.get("criteria")
    return len(criteria) if isinstance(criteria, (dict, list)) else 0


def _valid_view(row):
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        return False
    questions, targets = row.get("questions"), row.get("targets")
    if not isinstance(row.get("state"), (str, dict, list)) or not isinstance(questions, dict) or not questions:
        return False
    if not isinstance(targets, dict) or set(questions) != set(targets):
        return False
    for qid, question in questions.items():
        if not isinstance(question, dict) or _criteria_count(question) < 2:
            return False
        target = targets[qid]
        if not isinstance(target, dict) or str(target.get("label")) not in (
                [str(index) for index in range(_criteria_count(question))]
                if isinstance(question.get("criteria"), list)
                else [str(key) for key in question["criteria"]]):
            return False
    return True


def _valid_row(row):
    if not isinstance(row, dict):
        return False
    if "paired_views" not in row:
        return _valid_view(row)
    views, source_ids = row.get("paired_views"), row.get("source_ids")
    if (row.get("pair_mode") != "counterfactual" or not isinstance(views, list) or len(views) != 2
            or not isinstance(source_ids, list) or len(source_ids) != 2
            or source_ids[0] != source_ids[1] or not all(_valid_view(view) for view in views)):
        return False
    labels = [view["targets"][next(iter(view["targets"]))]["label"] for view in views]
    return set(labels) == {"yes", "no"} and all(
        view.get("meta", {}).get("source_id") == source_id
        for view, source_id in zip(views, source_ids))


def _require_ids(rows, source):
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{source} rows require non-empty IDs")
    duplicates = [row_id for row_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate {source} IDs: {duplicates[:3]}")


def _r044_categories(rows):
    categories = defaultdict(list)
    for row in rows:
        if not _valid_row(row):
            raise ValueError(f"invalid trainer row in R044 input: {row.get('id')}")
        category = (row.get("meta") or {}).get("r044_category")
        if category not in {"counterfactual", "prose", "replay", "hebrew", "visual"}:
            raise ValueError(f"unknown R044 source category: {row.get('id')}")
        has_image = bool(row.get("image") or row.get("images"))
        if has_image != (category == "visual"):
            raise ValueError(f"R044 image/category mismatch: {row.get('id')}")
        meta = row["meta"]
        if category == "hebrew" and (meta.get("source") != "MASSIVE" or meta.get("lang") != "he"):
            raise ValueError(f"R044 Hebrew provenance mismatch: {row['id']}")
        if category == "visual" and meta.get("source") != "ScienceQA":
            raise ValueError(f"R044 visual provenance mismatch: {row['id']}")
        if category == "prose" and (meta.get("source") != "r042_programmatic"
                                     or meta.get("renderer") != "prose"):
            raise ValueError(f"R044 prose provenance mismatch: {row['id']}")
        categories[category].append(row)
    expected = {"counterfactual": 100, "prose": 200, "replay": 200, "hebrew": 100, "visual": 200}
    actual = {key: len(categories[key]) for key in expected}
    if actual != expected:
        raise ValueError(f"R044 category counts mismatch: expected {expected}, found {actual}")
    replay_ids = {row["id"] for row in categories["replay"]}
    missing = DEFECTIVE_REPLAY_IDS - replay_ids
    if missing:
        raise ValueError(f"R044 replay pool is missing expected defective IDs: {sorted(missing)}")
    clean = [row for row in rows if row["id"] not in DEFECTIVE_REPLAY_IDS]
    if len(clean) != EXPECTED_ARM_ROWS:
        raise ValueError(f"R044 clean pool must contain {EXPECTED_ARM_ROWS} rows")
    categories["replay"] = [row for row in categories["replay"]
                             if row["id"] not in DEFECTIVE_REPLAY_IDS]
    return categories, clean


def _wh_valid(rows):
    if any(not _valid_view(row) or row.get("image") or row.get("images") for row in rows):
        raise ValueError("WH input must contain clean grouped text rows")
    return rows


def _compositional_valid(rows):
    for row in rows:
        if not _valid_row(row) or row.get("meta", {}).get("source") != "r045_generated":
            raise ValueError(f"invalid R045 compositional pair: {row.get('id')}")
        meta = row["meta"]
        if meta.get("split") != "train" or meta.get("language") not in {"en", "he"}:
            raise ValueError(f"R045 source/split provenance mismatch: {row['id']}")
        if meta.get("twin_id") != meta.get("source_id"):
            raise ValueError(f"R045 twin provenance mismatch: {row['id']}")
        if any((view.get("meta") or {}).get("source") != "r045_generated"
               or (view.get("meta") or {}).get("split") != "train"
               or (view.get("meta") or {}).get("language") != meta["language"]
               or (view.get("meta") or {}).get("twin_id") != meta["twin_id"]
               for view in row["paired_views"]):
            raise ValueError(f"R045 view provenance mismatch: {row['id']}")
        if row.get("image") or row.get("images"):
            raise ValueError(f"R045 text pair contains an image: {row['id']}")
    twins = defaultdict(list)
    for row in rows:
        twins[row["meta"]["twin_id"]].append(row["meta"]["language"])
    if any(Counter(languages) != {"en": 1, "he": 1} for languages in twins.values()):
        raise ValueError("R045 English/Hebrew twins must be selected together")
    return rows


def _sample(rows, count, seed, stratum):
    groups = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    keys = sorted(groups)
    for name in sorted(groups):
        values = groups[name]
        random.Random(f"r045-arm:{seed}:select:{name}").shuffle(values)
    ordered = []
    while keys:
        remaining = []
        for key in keys:
            ordered.append(groups[key].pop())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    if len(ordered) < count:
        raise ValueError(f"need {count} eligible rows; found {len(ordered)}")
    return ordered[:count]


def _shuffle_stratified(rows, seed, arm):
    groups = defaultdict(list)
    for row in rows:
        groups[_stratum(row)].append(row)
    for key, group in groups.items():
        random.Random(f"r045-arm:{seed}:{arm}:{key}").shuffle(group)
    keys = list(groups)
    random.Random(f"r045-arm:{seed}:{arm}:strata").shuffle(keys)
    result = []
    while keys:
        remaining = []
        for key in keys:
            result.append(groups[key].pop())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return result


def _stratum(row):
    meta = row.get("meta") or {}
    if meta.get("r044_category"):
        return ("r044", meta["r044_category"], meta.get("r044_variant", ""), meta.get("lang", ""))
    if meta.get("source") == "r045_generated":
        return ("r045", meta.get("language", ""), tuple(meta.get("composition", ())))
    return ("wh", len(row.get("questions", {})))


def _image_source(input_path, row):
    image = row.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"visual row has no image reference: {row.get('id')}")
    source = Path(image)
    source = source if source.is_absolute() else Path(input_path).parent / source
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"missing R044 image asset: {source}")
    return source


def _copy_images(rows, input_path, out):
    image_manifest, copied = {}, {}
    for row in rows:
        if not row.get("image"):
            continue
        source = _image_source(input_path, row)
        digest = _sha(source.read_bytes())
        suffix = source.suffix.lower() or ".img"
        target_rel = f"images/{digest}{suffix}"
        target = Path(out) / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if digest not in copied:
            shutil.copy2(source, target)
            if _sha(target.read_bytes()) != digest:
                raise IOError(f"copied image hash mismatch: {target}")
            copied[digest] = target_rel
        row["image"] = copied[digest]
        image_manifest[digest] = {"path": copied[digest], "sha256": digest}
    return image_manifest


def _tree_hash(out, image_manifest):
    entries = []
    for info in image_manifest.values():
        path = Path(out) / info["path"]
        entries.append((info["path"], _sha(path.read_bytes())))
    payload = "".join(f"{path}\0{digest}\n" for path, digest in sorted(entries)).encode()
    return _sha(payload), len(entries)


def build(r044_path, wh_path, compositional_path, out, seed=45):
    inputs = {}
    r044, inputs["r044_arm_b"] = _read(r044_path)
    wh, inputs["wh"] = _read(wh_path)
    compositional, inputs["compositional_train"] = _read(compositional_path)
    for name, rows in (("r044", r044), ("wh", wh), ("compositional", compositional)):
        _require_ids(rows, name)
    all_ids = [row["id"] for rows in (r044, wh, compositional) for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("source ID overlap across R044, WH, and R045 inputs")

    categories, clean_r044 = _r044_categories(r044)
    _wh_valid(wh)
    _compositional_valid(compositional)
    if len(wh) < COUNTS["wh"]:
        raise ValueError(f"WH pool needs at least {COUNTS['wh']} rows; found {len(wh)}")
    if len(compositional) != COUNTS["compositional"]:
        raise ValueError(f"R045 input needs exactly {COUNTS['compositional']} rows; found {len(compositional)}")

    common = []
    selections = {}
    for category in ("hebrew", "visual", "prose", "replay"):
        picked = _sample(categories[category], COUNTS[category], seed,
                         lambda row, category=category: category)
        common.extend(copy.deepcopy(picked))
        selections[category] = [row["id"] for row in picked]
    wh_rows = _sample(wh, COUNTS["wh"], seed,
                      lambda row: ("wh", len(row.get("questions", {}))))
    selections["wh"] = [row["id"] for row in wh_rows]
    r045_rows = copy.deepcopy(compositional)
    selections["compositional"] = [row["id"] for row in r045_rows]

    arm_a = copy.deepcopy(clean_r044)
    arm_b = [*common, *copy.deepcopy(wh_rows), *r045_rows]
    if len(common) != RETENTION_ROWS or len(arm_a) != EXPECTED_ARM_ROWS or len(arm_b) != EXPECTED_ARM_ROWS:
        raise AssertionError("arm row count invariant failed")
    _require_ids(arm_a, "control output")
    _require_ids(arm_b, "treatment output")
    common_ids = {row["id"] for row in common}
    if common_ids != ({row["id"] for row in arm_a} & {row["id"] for row in arm_b}):
        raise ValueError("arm overlap differs from the declared common retention rows")
    a_common = {row["id"]: row for row in arm_a if row["id"] in common_ids}
    b_common = {row["id"]: row for row in arm_b if row["id"] in common_ids}
    if a_common != b_common:
        raise ValueError("matched common rows differ between arms")
    if any(bool(a_common[row_id].get("image")) != bool(b_common[row_id].get("image"))
           for row_id in common_ids):
        raise ValueError("matched common rows have image-presence mismatch")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    image_manifest = _copy_images(arm_a + arm_b, r044_path, out)
    arm_a = _shuffle_stratified(arm_a, seed, "a")
    arm_b = _shuffle_stratified(arm_b, seed, "b")
    payloads = {"arm-a": _jsonl(arm_a), "arm-b": _jsonl(arm_b)}
    for name, payload in payloads.items():
        (out / f"{name}.jsonl").write_bytes(payload)
    image_tree_hash, image_count = _tree_hash(out, image_manifest)
    manifest = {
        "experiment": "r045_matched_arms", "seed": seed,
        "logical_rows": {name: len(rows) for name, rows in (("arm-a", arm_a), ("arm-b", arm_b))},
        "composition": {"common_retention": RETENTION_ROWS, **COUNTS,
                        "control_old_text": EXPECTED_ARM_ROWS - RETENTION_ROWS},
        "inputs": inputs,
        "selection": selections,
        "excluded_replay_ids": sorted(DEFECTIVE_REPLAY_IDS),
        "leakage_checks": {"arm_overlap_ids": len(common_ids), "common_rows_equal": True,
                           "common_image_presence_equal": True,
                           "extra_arm_ids_disjoint": True,
                           "benchmark_sources_rejected": True},
        "images": {"files": image_count, "tree_sha256": image_tree_hash,
                   "files_by_sha256": image_manifest},
        "outputs": {name: {"rows": len(rows), "sha256": _sha(payloads[name])}
                    for name, rows in (("arm-a", arm_a), ("arm-b", arm_b))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r044-arm-b", required=True)
    parser.add_argument("--wh", required=True)
    parser.add_argument("--compositional", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args(argv)
    manifest = build(args.r044_arm_b, args.wh, args.compositional, args.out, args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
