"""Fail before GPU work unless R043 files match their manifest and trainer semantics."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.train_gemma_decision_lora import load_examples


EXPECTED = {
    "arm-a": (2800, 4000),
    "arm-b": (2800, 4000),
    "arm-c": (2800, 4000),
    "eval": (256, 512),
}


def preflight(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("experiment") != "r043_causal_pairs" or manifest.get("seed") != 43:
        raise ValueError("unexpected R043 experiment or seed")
    if manifest.get("logical_rows") != {name: expected[0] for name, expected in EXPECTED.items()}:
        raise ValueError("manifest logical counts are not frozen R043 counts")
    if manifest.get("presentations") != {name: expected[1] for name, expected in EXPECTED.items()}:
        raise ValueError("manifest presentation counts are not frozen R043 counts")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(EXPECTED):
        raise ValueError("manifest must hash exactly arm-a/arm-b/arm-c/eval")
    result = {}
    for name, (logical, presentations) in EXPECTED.items():
        path = run / f"{name}.jsonl"
        data = path.read_bytes()
        rows = [line for line in data.splitlines() if line.strip()]
        declared = outputs[name]
        if declared.get("rows") != len(rows):
            raise ValueError(f"{name}: manifest row count mismatch")
        digest = hashlib.sha256(data).hexdigest()
        if declared.get("sha256") != digest:
            raise ValueError(f"{name}: manifest hash mismatch")
        examples = load_examples([path])
        loaded_presentations = sum(len(item.get("paired_views", [item])) for item in examples)
        if len(examples) != logical or loaded_presentations != presentations:
            raise ValueError(
                f"{name}: trainer sees {len(examples)} logical/{loaded_presentations} presentations; "
                f"expected {logical}/{presentations}")
        result[name] = {"json_rows": len(rows), "logical": len(examples),
                        "presentations": loaded_presentations, "sha256": digest}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
