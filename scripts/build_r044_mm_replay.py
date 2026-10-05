"""Build matched text-only and pixel-replay continuation corpora for R044."""

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

try:
    from scripts.build_r043_causal_pairs import _replay_skip_reason, _validate_true_pair
except ModuleNotFoundError:  # Direct ``uv run scripts/...`` execution.
    from build_r043_causal_pairs import _replay_skip_reason, _validate_true_pair


NULL_OPTION = "None of the options is supported by the image."
COUNTS = {"counterfactual": 100, "prose": 200, "replay": 200,
          "hebrew": 100, "visual": 200}
VISUAL_COUNTS = {"real": 100, "mismatch": 50, "gold_dropped": 50}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _read(path):
    path = Path(path)
    data = path.read_bytes()
    try:
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL input: {path}") from error
    return rows, {"path": str(path), "rows": len(rows),
                  "sha256": hashlib.sha256(data).hexdigest()}


def _rng(seed, name):
    return random.Random(f"r044:{seed}:{name}")


def _sample(rows, count, seed, name, accept):
    positions = [index for index, row in enumerate(rows) if accept(row)]
    _rng(seed, name).shuffle(positions)
    if len(positions) < count:
        raise ValueError(f"R044 needs {count} valid {name} rows; found {len(positions)}")
    return [(position, copy.deepcopy(rows[position])) for position in positions[:count]]


def _source(row, fallback):
    meta = row.get("meta") or {}
    return str(meta.get("source") or fallback), str(meta.get("license") or "not-recorded")


def _tag(row, category, index, source_index):
    original_id = row.get("id")
    row["id"] = f"r044-{category}-{index:04d}"
    row["meta"] = {**(row.get("meta") or {}), "r044_category": category,
                   "r044_source_index": source_index, "r044_original_id": original_id}
    return row


def _grouped_single(row):
    return (not row.get("image") and not row.get("images")
            and _replay_skip_reason(row) is None)


def _pairs(rows, seed):
    def valid(row):
        if row.get("pair_mode") != "counterfactual" or len(row.get("paired_views", [])) != 2:
            return False
        try:
            _validate_true_pair(row)
        except (KeyError, TypeError, ValueError):
            return False
        return not any(view.get("image") or view.get("images") for view in row["paired_views"])

    selected = _sample(rows, COUNTS["counterfactual"], seed, "counterfactual", valid)
    return [_tag(row, "counterfactual", index, position)
            for index, (position, row) in enumerate(selected)]


def _singles(rows, count, seed, category):
    def valid(row):
        if not _grouped_single(row):
            return False
        meta = row.get("meta") or {}
        return category != "prose" or (meta.get("source") == "r042_programmatic"
                                       and meta.get("renderer") == "prose")

    selected = _sample(rows, count, seed, category, valid)
    return [_tag(row, category, index, position)
            for index, (position, row) in enumerate(selected)]


def _flat_valid(row):
    candidates, label, target = row.get("candidates"), row.get("label"), row.get("target")
    return (isinstance(row.get("state"), (str, dict, list))
            and isinstance(row.get("query"), str)
            and isinstance(candidates, list) and len(candidates) >= 2
            and len(set(map(str, candidates))) == len(candidates)
            and isinstance(label, int) and 0 <= label < len(candidates)
            and isinstance(target, list) and len(target) == len(candidates)
            and target[label] == 1.0 and sum(target) == 1.0
            and not row.get("p_null"))


def _source_id(row, position, prefix):
    meta = row.get("meta") or {}
    explicit = row.get("id") or meta.get("source_id") or meta.get("pair")
    return str(explicit) if explicit is not None else f"{prefix}-line-{position + 1}"


def _grouped(row, row_id, candidates, gold, meta, image=None, null_last=False):
    labels = [f"option_{index:02d}" for index in range(len(candidates))]
    if null_last:
        labels[-1] = "null"
    if len(labels) != len(candidates) or not 0 <= gold < len(labels):
        raise ValueError(f"{row_id}: invalid grouped decision")
    result = {
        "id": row_id,
        "state": row["state"],
        "questions": {"decision": {"type": "choice", "instructions": row["query"],
                                      "criteria": dict(zip(labels, candidates))}},
        "targets": {"decision": {"label": labels[gold]}},
        "meta": meta,
    }
    if image is not None:
        result["image"] = image
    if _replay_skip_reason(result) is not None:
        raise ValueError(f"{row_id}: conversion is not trainer-valid")
    return result


def _hebrew(rows, seed):
    selected = _sample(
        rows, COUNTS["hebrew"], seed, "hebrew",
        lambda row: (_flat_valid(row) and len(row["candidates"]) >= 10
                     and (row.get("meta") or {}).get("source") == "MASSIVE"
                     and (row.get("meta") or {}).get("lang") == "he"),
    )
    gold_positions = list(range(10)) * 10
    _rng(seed, "hebrew-gold-positions").shuffle(gold_positions)
    result, selections = [], []
    for index, ((position, row), gold_position) in enumerate(zip(selected, gold_positions)):
        source_id = _source_id(row, position, "massive")
        gold = row["candidates"][row["label"]]
        distractors = [candidate for choice, candidate in enumerate(row["candidates"])
                       if choice != row["label"]]
        _rng(seed, f"hebrew:{source_id}:{position}").shuffle(distractors)
        choices = distractors[:9]
        choices.insert(gold_position, gold)
        source, license_name = _source(row, "MASSIVE")
        if license_name == "not-recorded":
            license_name = "CC-BY-4.0"
        meta = {**(row.get("meta") or {}), "source": source, "license": license_name,
                "source_id": source_id, "r044_category": "hebrew",
                "r044_source_index": position, "r044_original_label": row["label"],
                "r044_selected_candidate_indices": [row["candidates"].index(value) for value in choices],
                "r044_gold_position": gold_position}
        result.append(_grouped(row, f"r044-hebrew-{index:04d}", choices, gold_position, meta))
        selections.append({"source_id": source_id, "source_index": position,
                           "gold_position": gold_position,
                           "candidate_indices": meta["r044_selected_candidate_indices"]})
    if Counter(row["meta"]["r044_gold_position"] for row in result) != {index: 10 for index in range(10)}:
        raise AssertionError("Hebrew gold positions are not balanced")
    return result, selections


def _image_path(jsonl_path, row):
    image = row.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"ScienceQA row has no image reference in {jsonl_path}")
    path = (Path(jsonl_path).parent / image).resolve()
    if not path.is_file():
        raise ValueError(f"missing ScienceQA image: {path}")
    return path


def _decision_key(row):
    return json.dumps([row.get("state"), row.get("query"), row.get("candidates")],
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _protected(paths):
    hashes, decisions, inputs = set(), set(), []
    for path in paths:
        rows, info = _read(path)
        inputs.append(info)
        for row in rows:
            decisions.add(_decision_key(row))
            if row.get("image"):
                hashes.add(_sha256(_image_path(path, row)))
    return hashes, decisions, inputs


def _visual(train_path, rows, protected_hashes, protected_decisions, out, seed):
    safe, rejected, seen_hashes, seen_sources = [], Counter(), set(), set()
    order = list(range(len(rows)))
    _rng(seed, "visual").shuffle(order)
    for position in order:
        row = rows[position]
        if not _flat_valid(row) or (row.get("meta") or {}).get("source") != "ScienceQA":
            rejected["invalid"] += 1
            continue
        try:
            image_path = _image_path(train_path, row)
        except ValueError:
            raise
        image_sha = _sha256(image_path)
        source_id = _source_id(row, position, "scienceqa")
        if image_sha in protected_hashes:
            rejected["protected_image"] += 1
        elif _decision_key(row) in protected_decisions:
            rejected["protected_decision"] += 1
        elif image_sha in seen_hashes:
            rejected["duplicate_image"] += 1
        elif source_id in seen_sources:
            rejected["duplicate_source"] += 1
        elif len(row["candidates"]) + 1 > 26:
            rejected["too_many_options"] += 1
        else:
            safe.append((position, row, image_path, image_sha, source_id))
            seen_hashes.add(image_sha)
            seen_sources.add(source_id)
            if len(safe) == COUNTS["visual"]:
                break
    if len(safe) < COUNTS["visual"]:
        raise ValueError(f"R044 needs 200 unique leakage-free visual rows; found {len(safe)}")

    image_dir = Path(out) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{image_sha}{source_path.suffix.lower() or '.img'}"
                      for _, _, source_path, image_sha, _ in safe}
    stale = [path for path in image_dir.rglob("*")
             if path.is_file() and path.relative_to(image_dir).as_posix() not in expected_names]
    if stale:
        raise ValueError(f"output image directory contains {len(stale)} stale files")
    image_refs = []
    for _, _, source_path, image_sha, _ in safe:
        suffix = source_path.suffix.lower() or ".img"
        target = image_dir / f"{image_sha}{suffix}"
        shutil.copy2(source_path, target)
        if _sha256(target) != image_sha:
            raise IOError(f"copied image hash mismatch: {target}")
        image_refs.append(target.relative_to(out).as_posix())

    mismatch_indices = list(range(VISUAL_COUNTS["real"],
                                  VISUAL_COUNTS["real"] + VISUAL_COUNTS["mismatch"]))
    donors = mismatch_indices[1:] + mismatch_indices[:1]
    result, selections = [], []
    boundaries = (VISUAL_COUNTS["real"], VISUAL_COUNTS["real"] + VISUAL_COUNTS["mismatch"])
    for index, ((position, row, _, image_sha, source_id), image_ref) in enumerate(zip(safe, image_refs)):
        if index < boundaries[0]:
            variant, assigned_ref, candidates, gold = "real", image_ref, list(row["candidates"]), row["label"]
            assigned_sha = image_sha
        elif index < boundaries[1]:
            variant, donor = "mismatch", donors[index - boundaries[0]]
            assigned_ref, assigned_sha = image_refs[donor], safe[donor][3]
            candidates, gold = list(row["candidates"]), len(row["candidates"])
            if assigned_sha == image_sha:
                raise AssertionError("mismatch donor did not change the image")
        else:
            variant, assigned_ref, assigned_sha = "gold_dropped", image_ref, image_sha
            candidates = [candidate for choice, candidate in enumerate(row["candidates"])
                          if choice != row["label"]]
            gold = len(candidates)
        candidates.append(NULL_OPTION)
        source, license_name = _source(row, "ScienceQA")
        meta = {**(row.get("meta") or {}), "source": source, "license": license_name,
                "source_id": source_id, "r044_category": "visual", "r044_variant": variant,
                "r044_source_index": position, "r044_base_image_sha256": image_sha,
                "r044_assigned_image_sha256": assigned_sha}
        result.append(_grouped(row, f"r044-visual-{index:04d}", candidates, gold, meta,
                               assigned_ref, null_last=True))
        selections.append({"source_id": source_id, "source_index": position, "variant": variant,
                           "base_image_sha256": image_sha, "assigned_image_sha256": assigned_sha})
    return result, selections, dict(rejected)


def _tree_hash(path):
    entries = [(file.relative_to(path).as_posix(), _sha256(file))
               for file in sorted(Path(path).rglob("*")) if file.is_file()]
    payload = "".join(f"{name}\0{digest}\n" for name, digest in entries).encode()
    return hashlib.sha256(payload).hexdigest(), len(entries)


def _licenses(named_rows):
    counts = Counter()
    for name, rows in named_rows.items():
        for row in rows:
            source, license_name = _source(row, name)
            counts[(name, source, license_name)] += 1
    return [{"category": category, "source": source, "license": license_name, "logical_rows": count}
            for (category, source, license_name), count in sorted(counts.items())]


def build(r043_path, prose_path, replay_path, hebrew_path, scienceqa_train,
          protection_paths, out, seed=44):
    if not protection_paths:
        raise ValueError("at least one ScienceQA protection path is required")
    loaded, inputs = {}, {}
    for name, path in (("r043", r043_path), ("prose", prose_path), ("replay", replay_path),
                       ("hebrew", hebrew_path), ("scienceqa_train", scienceqa_train)):
        loaded[name], inputs[name] = _read(path)
    protected_hashes, protected_decisions, protection_inputs = _protected(protection_paths)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = _pairs(loaded["r043"], seed)
    prose = _singles(loaded["prose"], COUNTS["prose"], seed, "prose")
    replay = _singles(loaded["replay"], COUNTS["replay"], seed, "replay")
    hebrew, hebrew_selection = _hebrew(loaded["hebrew"], seed)
    visual, visual_selection, rejected = _visual(scienceqa_train, loaded["scienceqa_train"],
                                                   protected_hashes, protected_decisions, out, seed)
    arm_b = pairs + prose + replay + hebrew + visual
    arm_a = copy.deepcopy(arm_b)
    for row in arm_a:
        if (row.get("meta") or {}).get("r044_category") == "visual":
            row.pop("image")
    if len(arm_a) != 800 or len({row["id"] for row in arm_a}) != 800:
        raise AssertionError("R044 arms need exactly 800 uniquely identified logical slots")
    stripped_b = [{key: value for key, value in row.items() if key != "image"}
                  for row in arm_b]
    if arm_a != stripped_b:
        raise AssertionError("R044 arms differ by more than visual image references")

    payloads = {"arm-a": _jsonl(arm_a), "arm-b": _jsonl(arm_b)}
    for name, payload in payloads.items():
        (out / f"{name}.jsonl").write_bytes(payload)
    image_hash, image_count = _tree_hash(out / "images")
    named = {"counterfactual": pairs, "prose": prose, "replay": replay,
             "hebrew": hebrew, "visual": visual}
    manifest = {
        "experiment": "r044_mm_replay", "seed": seed,
        "logical_rows": {"arm-a": len(arm_a), "arm-b": len(arm_b)},
        "composition": COUNTS, "visual_variants": VISUAL_COUNTS,
        "sources_and_licenses": _licenses(named),
        "inputs": {**inputs, "scienceqa_protection": protection_inputs},
        "protection": {"image_sha256_count": len(protected_hashes),
                       "decision_tuple_count": len(protected_decisions)},
        "selection": {"hebrew": hebrew_selection, "visual": visual_selection},
        "leakage_checks": {"selected_base_image_hashes_unique": len({x["base_image_sha256"] for x in visual_selection}) == 200,
                           "selected_source_ids_unique": len({x["source_id"] for x in visual_selection}) == 200,
                           "protected_image_overlap": len({x["base_image_sha256"] for x in visual_selection} & protected_hashes),
                           "protected_decision_overlap": sum(_decision_key(loaded["scienceqa_train"][x["source_index"]]) in protected_decisions for x in visual_selection),
                           "scienceqa_rejections": rejected,
                           "arms_equal_without_visual_image": True},
        "images": {"files": image_count, "tree_sha256": image_hash},
        "outputs": {name: {"rows": 800, "sha256": hashlib.sha256(payload).hexdigest()}
                    for name, payload in payloads.items()},
    }
    if (manifest["leakage_checks"]["protected_image_overlap"]
            or manifest["leakage_checks"]["protected_decision_overlap"]):
        raise AssertionError("protected ScienceQA leakage survived selection")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r043-pairs", required=True)
    parser.add_argument("--r042-prose", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--hebrew", required=True)
    parser.add_argument("--scienceqa-train", required=True)
    parser.add_argument("--scienceqa-protect", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    manifest = build(args.r043_pairs, args.r042_prose, args.replay, args.hebrew,
                     args.scienceqa_train, args.scienceqa_protect, args.out, args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
