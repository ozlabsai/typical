"""Build the filtered public-data pilot for Intern-Decision.

uv run --with datasets scripts/intern_g2_data.py --out .context/intern-g2 \
  --replay .context/intern-g1/replay.jsonl
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def valid(row):
    options, target = row.get("options"), row.get("target")
    return (row.get("kind") in {"choice", "score", "noul"}
            and isinstance(options, list) and 2 <= len(options) <= 62
            and len(options) == len(set(options)) and len(target or []) == len(options)
            and all(math.isfinite(value) and value >= 0 for value in target)
            and sum(target) > 0 and max(target) - min(target) > 1e-6)


def canonical(source, row):
    options = [str(value) for value in row["options"]]
    target = [float(value) / sum(row["target"]) for value in row["target"]]
    kind = row["kind"]
    if kind == "noul":
        normalized = ["yes" if value.lower() in {"yes", "true", "1"} else
                      "no" if value.lower() in {"no", "false", "0"} else None for value in options]
        if sorted(normalized) != ["no", "yes"]:
            return None
        ordered = [target[normalized.index("no")], target[normalized.index("yes")]]
        question = {"type": "noul", "instructions": row["question"],
                    "criteria": {"no": "No", "yes": "Yes"}}
        answer, probabilities = ("no", "yes")[ordered[1] > ordered[0]], ordered
    elif kind == "score":
        question = {"type": "score", "instructions": row["question"], "criteria": options}
        answer, probabilities = str(max(range(len(target)), key=target.__getitem__)), target
    else:
        question = {"type": "choice", "instructions": row["question"],
                    "criteria": dict(zip(options, options))}
        answer, probabilities = options[max(range(len(target)), key=target.__getitem__)], target
    state = row.get("state", row.get("state_json"))
    if source == "openjev":
        state = json.loads(state)
    record = {
        "id": f"g2-{source}-{row['id']}",
        "state": state,
        "questions": {"decision": question},
        "targets": {"decision": {"label": answer}},
    }
    if any(value not in {0.0, 1.0} for value in probabilities):
        record["soft_targets"] = {"decision": probabilities}
    return record


def take(dataset, source, limit, predicate):
    rows, seen = [], Counter()
    for row in dataset.shuffle(seed=42, buffer_size=50_000):
        seen["read"] += 1
        if not valid(row) or not predicate(row):
            seen["filtered"] += 1
            continue
        record = canonical(source, row)
        if record is None:
            seen["invalid_noul"] += 1
            continue
        rows.append(record)
        if len(rows) == limit:
            break
    if len(rows) != limit:
        raise RuntimeError(f"{source}: found {len(rows)} of {limit} rows")
    return rows, seen


def write(path, rows):
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--tasksource", type=int, default=12_000)
    parser.add_argument("--openjev", type=int, default=12_000)
    parser.add_argument("--sarge", type=int, default=4_000)
    parser.add_argument("--replay-count", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    task, task_seen = take(
        load_dataset("tasksource/tasksource-jev-typed-decisions", split="train", streaming=True),
        "tasksource", args.tasksource,
        lambda row: row["license_use"] == "commercial" and row["variant"] == "direct",
    )
    blocked = ("workflow-control", "customer-control", "vizdoom")
    openjev, open_seen = take(
        load_dataset("ZefanCai/Open-Jev-v1.1", "community-hard-mix-v2-redistributable",
                     split="train", streaming=True),
        "openjev", args.openjev,
        lambda row: not any(value in row["source"].lower() for value in blocked),
    )
    sarge, sarge_seen = take(
        load_dataset("SargeDev/jev-distill-corpus-v3", split="train", streaming=True),
        "sarge", args.sarge, lambda row: row["source"] == "yuri_v3",
    )
    replay = [json.loads(line) for line in args.replay.open() if line.strip()]
    replay = random.Random(args.seed).sample(replay, args.replay_count)
    rows = task + openjev + sarge + replay
    random.Random(args.seed).shuffle(rows)
    write(args.out / "train.jsonl", rows)
    manifest = {
        "counts": {"tasksource": len(task), "openjev": len(openjev), "sarge": len(sarge),
                   "replay": len(replay), "total": len(rows)},
        "scanned": {"tasksource": dict(task_seen), "openjev": dict(open_seen), "sarge": dict(sarge_seen)},
        "soft_records": sum("soft_targets" in row for row in rows),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
