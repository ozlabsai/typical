"""Compile Typical row JSONL into Intern-Decision's grouped hard-label schema.

Soft distributions and null rows are deliberately rejected: Intern's released
masked-token objective cannot represent either without changing the label.

uv run scripts/intern_adapter.py --input data_wf/train.jsonl data_wh/train.jsonl \
  --out .context/intern-g1/train.jsonl
"""

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path

MAX_QUESTIONS = 16


def _one_hot(target):
    return bool(target) and sum(v == 1.0 for v in target) == 1 and all(v in (0.0, 1.0) for v in target)


def _group_id(path, line_no, row):
    meta = row.get("meta", {})
    for key in ("rubric_group", "state_id", "case"):
        if meta.get(key) is not None:
            return f"{path}:{key}:{meta[key]}"
    return f"{path}:line:{line_no}"


def _question(row):
    candidates = row.get("candidates", [])
    target = row.get("target", [])
    label = row.get("label")
    if row.get("p_null", 0) > 0:
        return None, None, "null"
    if not _one_hot(target):
        return None, None, "soft_target"
    if not isinstance(label, int) or not 0 <= label < len(candidates) or target[label] != 1.0:
        return None, None, "bad_label"
    if not 2 <= len(candidates) <= 62 or len(set(candidates)) != len(candidates):
        return None, None, "bad_candidates"
    source = row.get("question")
    if not isinstance(source, dict):
        # Legacy E/K replay rows are plain classifications whose candidate text
        # is the complete criterion. Workflow rows must be regenerated instead
        # of trying to reverse their lossy rendered rubric.
        meta = row.get("meta", {})
        if meta.get("fam_bucket") == "W" or str(row.get("task", "")).startswith("wf_"):
            return None, None, "missing_question"
        source = {"type": "choice", "instructions": row.get("query", ""),
                  "criteria": {candidate: candidate for candidate in candidates}}
    kind = source.get("type")
    instructions = source.get("instructions", "")
    criteria = source.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or any(candidate not in criteria for candidate in candidates):
            return None, None, "bad_criteria"
        question = {"type": kind, "instructions": instructions,
                    "criteria": {candidate: criteria[candidate] for candidate in candidates}}
        answer = candidates[label]
    elif kind == "score":
        if isinstance(criteria, list):
            if len(criteria) != len(candidates):
                return None, None, "bad_criteria"
            descriptions = criteria
        elif isinstance(criteria, dict) and all(candidate in criteria for candidate in candidates):
            descriptions = [criteria[candidate] for candidate in candidates]
        else:
            return None, None, "bad_criteria"
        question = {"type": kind, "instructions": instructions, "criteria": descriptions}
        answer = str(label)
    elif kind == "noul":
        normalized = ["yes" if str(v).lower() in {"yes", "true"} else
                      "no" if str(v).lower() in {"no", "false"} else None for v in candidates]
        if sorted(normalized) != ["no", "yes"]:
            return None, None, "bad_noul"
        descriptions = criteria if isinstance(criteria, dict) else {}
        yes = next((descriptions[k] for k in descriptions if str(k).lower() in {"yes", "true"}), "Yes")
        no = next((descriptions[k] for k in descriptions if str(k).lower() in {"no", "false"}), "No")
        question = {"type": kind, "instructions": instructions, "criteria": {"no": no, "yes": yes}}
        answer = normalized[label]
    else:
        return None, None, "bad_type"
    return question, {"label": answer}, None


def compile_files(paths):
    groups = OrderedDict()
    skipped = Counter()
    for path in map(Path, paths):
        with path.open() as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                question, target, reason = _question(row)
                if reason:
                    skipped[reason] += 1
                    continue
                key = _group_id(path, line_no, row)
                images = row.get("images") or ([row["image"]] if row.get("image") else [])
                group = groups.setdefault(key, {"state": row.get("state"), "images": images, "items": []})
                if group["state"] != row.get("state") or group["images"] != images:
                    group["invalid"] = "inconsistent_group"
                group["items"].append((question, target))

    records = []
    for key, group in groups.items():
        reason = group.get("invalid")
        if len(group["items"]) > MAX_QUESTIONS:
            reason = "too_many_questions"
        if reason:
            skipped[reason] += len(group["items"])
            continue
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        questions, targets = {}, {}
        for index, (question, target) in enumerate(group["items"]):
            field = f"q{index:03d}"
            questions[field], targets[field] = question, target
        record = {"id": f"typical-{digest}", "state": group["state"],
                  "questions": questions, "targets": targets}
        if group["images"]:
            record["images"] = group["images"]
        records.append(record)
    return records, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    records, skipped = compile_files(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"inputs": args.input, "records": len(records),
                "questions": sum(len(r["questions"]) for r in records), "skipped": dict(skipped)}
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
