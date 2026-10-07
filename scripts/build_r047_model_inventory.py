"""Build and verify the hash-locked base-model inventory used by R047."""

import argparse
import json
from pathlib import Path

from scripts.preflight_r043_assets import file_hash, tree_hash


SCHEMA_VERSION = "r047-model-inventory-v1"
MODEL_ID = "google/gemma-4-12B-it"
FROZEN_BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
FROZEN_R044_BASE_REFERENCE = "/workspace/g1/hf-gemma"
INVENTORY_FIELDS = {"schema_version", "model_id", "revision", "tree_sha256", "files"}


def is_frozen_base_reference(value):
    """Accept only the model ID, pinned snapshot, or R044's authenticated cache reference."""
    return (isinstance(value, str)
            and (value in {MODEL_ID, FROZEN_R044_BASE_REFERENCE}
                 or Path(value).name == FROZEN_BASE_REVISION))


def validate_model_path(model_path, revision=FROZEN_BASE_REVISION):
    """Require the exact pinned Gemma snapshot layout and its config payload."""
    path = Path(model_path).resolve()
    if revision != FROZEN_BASE_REVISION:
        raise ValueError("base revision differs from the frozen R047 revision")
    if (path.name != revision or path.parent.name != "snapshots"
            or path.parent.parent.name != "models--google--gemma-4-12B-it"
            or not (path / "config.json").is_file()):
        raise ValueError("model path is not the frozen Gemma snapshot")
    return path


def build_inventory(model_path, revision=FROZEN_BASE_REVISION):
    path = validate_model_path(model_path, revision)
    digest, count = tree_hash(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "revision": revision,
        "tree_sha256": digest,
        "files": count,
    }


def verify_inventory(inventory_path, model_path, revision=FROZEN_BASE_REVISION):
    """Verify the inventory bytes and the full staged model tree they describe."""
    path = Path(inventory_path)
    try:
        inventory = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid model inventory JSON") from error
    if (not isinstance(inventory, dict) or set(inventory) != INVENTORY_FIELDS
            or inventory.get("schema_version") != SCHEMA_VERSION
            or inventory.get("model_id") != MODEL_ID
            or inventory.get("revision") != revision
            or type(inventory.get("files")) is not int or inventory["files"] <= 0):
        raise ValueError("model inventory schema is invalid")
    actual = build_inventory(model_path, revision)
    if (inventory["tree_sha256"] != actual["tree_sha256"]
            or inventory["files"] != actual["files"]):
        raise ValueError("staged model tree differs from model inventory")
    return inventory, file_hash(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", default=FROZEN_BASE_REVISION)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    inventory = build_inventory(args.model_path, args.revision)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
