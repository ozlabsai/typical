"""Select and run the frozen R050 training-only gradient-scale probe.

The probe sets each exact-outcome coefficient so its initial gradient norm is
25% of the field-CE gradient norm.  It never constructs an optimizer or mutates
model parameters, and its immutable receipt binds every input and implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import scripts.train_gemma_decision_lora as trainer
from scripts.build_r047_model_inventory import FROZEN_BASE_REVISION, verify_inventory
from scripts.preflight_r043_assets import file_hash, tree_hash


SCHEMA_VERSION = "r050-gradient-scale-probe-v1"
DEFAULT_TARGET_RATIO = 0.25
DEFAULT_SIGNATURES = 64
STRATUM_KEYS = (
    "policy_mode",
    "value_regime",
    "operator_composition",
    "field_count",
)
PROBE_VARIANTS = tuple(product((0, 1), ("en", "he"), ("no_yes", "yes_no")))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _read_regular(path: str | Path, label: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return path.read_bytes()


def read_jsonl(path: str | Path) -> tuple[bytes, list[dict[str, Any]]]:
    payload = _read_regular(path, "training panel")
    rows = []
    try:
        for line in payload.decode().splitlines():
            if line.strip():
                row = json.loads(line, object_pairs_hook=trainer._unique_json_object)
                if not isinstance(row, dict):
                    raise ValueError("training panel rows must be JSON objects")
                rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training panel must be valid UTF-8 JSONL") from error
    if not rows:
        raise ValueError("training panel must not be empty")
    return payload, rows


def verify_panel_manifest(manifest_path: str | Path, panel_payload: bytes,
                          rows: list[dict[str, Any]], builder_path: str | Path) -> dict[str, Any]:
    payload = _read_regular(manifest_path, "panel manifest")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("panel manifest must be valid JSON") from error
    output = manifest.get("outputs", {}).get("train") if isinstance(manifest, dict) else None
    if (not isinstance(output, dict)
            or output.get("sha256") != _sha256(panel_payload)
            or output.get("rows") != len(rows)):
        raise ValueError("panel manifest does not bind the supplied training panel")
    builder_payload = _read_regular(builder_path, "panel builder")
    if manifest.get("generator_sha256") != _sha256(builder_payload):
        raise ValueError("panel builder differs from the panel manifest")
    return {
        "path": str(Path(manifest_path).resolve()),
        "sha256": _sha256(payload),
        "panel_version": manifest.get("panel_version"),
        "generator_sha256": manifest["generator_sha256"],
    }


def _checked_signature(value: Any) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())):
        raise ValueError("semantic_signature must be a 64-character hexadecimal sha256")
    return value.lower()


def _row_probe_metadata(row: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    meta = row.get("meta")
    if not isinstance(meta, dict) or meta.get("panel") != "exact_policy_v1":
        raise ValueError("gradient probe accepts only exact_policy_v1 panel rows")
    if meta.get("split") != "train":
        raise ValueError("gradient probe accepts only the training split")
    signature = _checked_signature(meta.get("semantic_signature"))
    questions = row.get("questions")
    policy = meta.get("policy")
    fields = policy.get("fields") if isinstance(policy, dict) else None
    if (not isinstance(questions, dict) or not questions
            or not isinstance(fields, list) or len(fields) != len(questions)):
        raise ValueError("probe row has inconsistent policy fields and questions")
    values = (
        meta.get("policy_mode"),
        meta.get("value_regime"),
        meta.get("operator_composition"),
        len(fields),
    )
    if (any(not isinstance(value, str) or not value for value in values[:3])
            or not 2 <= values[3] <= 26):
        raise ValueError("probe row is missing a valid stratification cell")
    if meta.get("language") not in {"en", "he"}:
        raise ValueError("probe row has an invalid language")
    if meta.get("option_order") not in {"no_yes", "yes_no"}:
        raise ValueError("probe row has an invalid option order")
    if type(meta.get("counterfactual_side")) is not int or meta["counterfactual_side"] not in {0, 1}:
        raise ValueError("probe row has an invalid counterfactual side")
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("probe row id must be a nonempty string")
    return signature, values


def _variant(row: dict[str, Any]) -> tuple[int, str, str]:
    meta = row["meta"]
    return meta["counterfactual_side"], meta["language"], meta["option_order"]


def select_probe_rows(rows: list[dict[str, Any]], requested_signatures: int,
                      seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one stable canonical view per signature, round-robin by factor cell."""
    if type(requested_signatures) is not int or requested_signatures <= 0:
        raise ValueError("requested_signatures must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("probe seed must be a nonnegative integer")
    by_signature = defaultdict(list)
    signature_strata = {}
    seen_ids = set()
    for row in rows:
        signature, stratum = _row_probe_metadata(row)
        if row["id"] in seen_ids:
            raise ValueError("training panel row ids must be unique")
        seen_ids.add(row["id"])
        previous = signature_strata.setdefault(signature, stratum)
        if previous != stratum:
            raise ValueError("one semantic_signature spans multiple factor cells")
        by_signature[signature].append(row)
    if requested_signatures > len(by_signature):
        raise ValueError(
            f"requested {requested_signatures} signatures but only {len(by_signature)} are available")

    buckets = defaultdict(list)
    for signature, orbit in by_signature.items():
        by_variant = {_variant(row): row for row in orbit}
        if len(by_variant) != len(orbit) or set(by_variant) != set(PROBE_VARIANTS):
            raise ValueError(
                "each probe semantic signature needs one complete eight-view orbit"
            )
        buckets[signature_strata[signature]].append((signature, by_variant))
    for stratum in buckets:
        buckets[stratum].sort(key=lambda item: (_sha256(
            f"{seed}:{item[0]}".encode()), item[0]))

    selected = []
    offsets = Counter()
    ordered_strata = sorted(buckets, key=lambda cell: tuple(map(str, cell)))
    while len(selected) < requested_signatures:
        progressed = False
        for stratum in ordered_strata:
            offset = offsets[stratum]
            if offset < len(buckets[stratum]):
                selected.append((*buckets[stratum][offset], stratum))
                offsets[stratum] += 1
                progressed = True
                if len(selected) == requested_signatures:
                    break
        if not progressed:
            raise RuntimeError("stratified selection exhausted unexpectedly")

    selected_rows = []
    selected_with_views = []
    rotation = seed % len(PROBE_VARIANTS)
    for index, (signature, by_variant, stratum) in enumerate(selected):
        variant = PROBE_VARIANTS[(index + rotation) % len(PROBE_VARIANTS)]
        row = by_variant[variant]
        selected_rows.append(row)
        selected_with_views.append((signature, row, stratum, variant))
    records = [
        {
            "semantic_signature": signature,
            "row_id": row["id"],
            "stratum": dict(zip(STRATUM_KEYS, stratum)),
            "counterfactual_side": variant[0],
            "language": variant[1],
            "option_order": variant[2],
        }
        for signature, row, stratum, variant in selected_with_views
    ]
    stratum_counts = Counter(
        "|".join(map(str, stratum)) for _, _, stratum, _ in selected_with_views)
    variant_counts = Counter(
        f"side{side}|{language}|{order}"
        for _, _, _, (side, language, order) in selected_with_views
    )
    selection = {
        "seed": seed,
        "requested_signatures": requested_signatures,
        "available_signatures": len(by_signature),
        "selected_signatures": len(selected),
        "representative_rule": (
            "seed-rotated counterfactual/language/option-order cycle over complete orbits"
        ),
        "stratum_keys": list(STRATUM_KEYS),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "records": records,
        "semantic_signature_digest": _canonical_sha(
            sorted(record["semantic_signature"] for record in records)),
        "row_id_digest": _canonical_sha(sorted(record["row_id"] for record in records)),
    }
    return selected_rows, selection


def trainable_parameters(model) -> tuple[list[str], list[torch.nn.Parameter]]:
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    if not named:
        raise ValueError("gradient probe needs trainable parameters")
    return [name for name, _ in named], [parameter for _, parameter in named]


def parameter_digest(names: list[str], parameters: list[torch.nn.Parameter]) -> str:
    if len(names) != len(parameters) or not parameters:
        raise ValueError("parameter digest needs aligned nonempty inputs")
    digest = hashlib.sha256()
    for name, parameter in zip(names, parameters):
        tensor = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(name.encode() + b"\0")
        digest.update(str(tuple(parameter.shape)).encode() + b"\0")
        digest.update(str(parameter.dtype).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _accumulate_gradients(loss, parameters, buffers, scale, retain_graph):
    if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
        raise ValueError("probe loss components must be finite differentiable scalars")
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=False)
    for gradient, buffer in zip(gradients, buffers):
        value = gradient.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError("gradient probe produced a nonfinite gradient")
        buffer.add_(value, alpha=scale)


def gradient_statistics(gradients: dict[str, list[torch.Tensor]],
                        target_ratio: float = DEFAULT_TARGET_RATIO) -> dict[str, Any]:
    names = ("field_ce", "outcome_nll", "outcome_brier")
    if (isinstance(target_ratio, bool) or not isinstance(target_ratio, (int, float))
            or not math.isfinite(target_ratio) or target_ratio <= 0):
        raise ValueError("target gradient ratio must be finite and positive")
    if set(gradients) != set(names):
        raise ValueError("gradient sets must contain field CE, NLL, and Brier")
    lengths = {len(gradients[name]) for name in names}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("gradient sets must be aligned and nonempty")

    def dot(left, right):
        total = torch.zeros((), dtype=torch.float32)
        for a, b in zip(gradients[left], gradients[right]):
            a, b = a.float(), b.float()
            if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("gradient tensors must be aligned and finite")
            total += torch.sum(a * b)
        value = float(total.item())
        if not math.isfinite(value):
            raise ValueError("gradient dot product is nonfinite")
        return value

    squares = {name: dot(name, name) for name in names}
    if any(value <= 0 for value in squares.values()):
        raise ValueError("gradient norms must be nonzero")
    norms = {name: math.sqrt(value) for name, value in squares.items()}
    pairs = (
        ("field_ce", "outcome_nll"),
        ("field_ce", "outcome_brier"),
        ("outcome_nll", "outcome_brier"),
    )
    dots = {f"{left}__{right}": dot(left, right) for left, right in pairs}
    cosines = {
        f"{left}__{right}": dots[f"{left}__{right}"] / (norms[left] * norms[right])
        for left, right in pairs
    }
    if any(not math.isfinite(value) or not -1.000001 <= value <= 1.000001
           for value in cosines.values()):
        raise ValueError("gradient cosine is invalid")
    return {
        "accumulator_dtype": "float32",
        "target_gradient_ratio": float(target_ratio),
        "norms": norms,
        "dots": dots,
        "cosines": cosines,
        "coefficients": {
            "lambda_nll": float(target_ratio) * norms["field_ce"] / norms["outcome_nll"],
            "lambda_brier": (
                float(target_ratio) * norms["field_ce"] / norms["outcome_brier"]),
        },
    }


def probe_gradient_norms(model, examples, tokenizer, option_groups, device, max_length,
                         microbatch=1, target_ratio=DEFAULT_TARGET_RATIO):
    """Compute matched mean gradients without populating ``.grad`` or taking a step."""
    if type(microbatch) is not int or microbatch <= 0:
        raise ValueError("probe microbatch must be a positive integer")
    if not examples:
        raise ValueError("gradient probe needs selected examples")
    names, parameters = trainable_parameters(model)
    before = parameter_digest(names, parameters)
    buffers = {
        name: [torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
               for parameter in parameters]
        for name in ("field_ce", "outcome_nll", "outcome_brier")
    }
    try:
        for start in range(0, len(examples), microbatch):
            batch = examples[start:start + microbatch]
            _, _, _, components = trainer.loss_for_examples(
                model, batch, tokenizer, option_groups, max_length, device,
                shuffle=False, action_loss_weight=0.0, action_temperature=1.0,
                exact_policy_outcome_loss="nll",
                return_exact_policy_components=True)
            scale = len(batch) / len(examples)
            _accumulate_gradients(
                components["field_ce"], parameters, buffers["field_ce"], scale, True)
            _accumulate_gradients(
                components["outcome_nll"], parameters, buffers["outcome_nll"], scale, True)
            _accumulate_gradients(
                components["outcome_brier"], parameters, buffers["outcome_brier"], scale, False)
    finally:
        after = parameter_digest(names, parameters)
        if before != after:
            raise RuntimeError("gradient probe mutated trainable parameters")
    return gradient_statistics(buffers, target_ratio), before


def build_receipt(*, selection, gradient_report, parameter_sha256, bindings,
                  configuration, environment):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "selection": selection,
        "gradient_report": gradient_report,
        "parameter_sha256_before_after": parameter_sha256,
        "parameter_digest_scope": "trainable_parameters",
        "bindings": bindings,
        "configuration": configuration,
        "environment": environment,
    }


def _write_new(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x") as destination:
            destination.write(payload)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite immutable receipt: {path}") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--parent-adapter", required=True)
    parser.add_argument("--train-panel", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--panel-builder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--signatures", type=int, default=DEFAULT_SIGNATURES)
    parser.add_argument("--seed", type=int, default=50_050)
    parser.add_argument("--target-ratio", type=float, default=DEFAULT_TARGET_RATIO)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("R050 gradient probe requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    panel_payload, rows = read_jsonl(args.train_panel)
    manifest_binding = verify_panel_manifest(
        args.panel_manifest, panel_payload, rows, args.panel_builder)
    selected_rows, selection = select_probe_rows(rows, args.signatures, args.seed)
    examples = [trainer._exact_policy_group(row, Path(args.train_panel).resolve().parent)
                for row in selected_rows]

    inventory, inventory_sha = verify_inventory(
        args.model_inventory, args.model, FROZEN_BASE_REVISION)
    parent_input = Path(args.parent_adapter)
    if parent_input.is_symlink() or not parent_input.is_dir():
        raise ValueError("parent adapter must be a regular directory")
    parent_path = parent_input.resolve()
    parent_sha, parent_files = tree_hash(parent_path)

    tokenizer, _ = trainer.load_tokenizer_and_processor(args.model, multimodal=False)
    option_groups = trainer.letter_token_groups(tokenizer)
    model = trainer.AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model = trainer.attach_lora(model, str(parent_path), rank=32, alpha=64)
    trainer.validate_trainable_path(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    device = torch.device("cuda")
    model.to(device).train()

    gradient_report, parameter_sha = probe_gradient_norms(
        model, examples, tokenizer, option_groups, device, args.max_length,
        microbatch=args.microbatch, target_ratio=args.target_ratio)
    receipt = build_receipt(
        selection=selection,
        gradient_report=gradient_report,
        parameter_sha256=parameter_sha,
        bindings={
            "training_panel": {
                "path": str(Path(args.train_panel).resolve()),
                "sha256": _sha256(panel_payload),
                "rows": len(rows),
            },
            "panel_manifest": manifest_binding,
            "panel_builder_sha256": file_hash(args.panel_builder),
            "model_inventory": {
                "path": str(Path(args.model_inventory).resolve()),
                "sha256": inventory_sha,
                **inventory,
            },
            "parent_adapter": {
                "path": str(parent_path),
                "tree_sha256": parent_sha,
                "files": parent_files,
            },
            "trainer_sha256": file_hash(Path(trainer.__file__)),
            "probe_sha256": file_hash(Path(__file__)),
        },
        configuration={
            "signatures": args.signatures,
            "seed": args.seed,
            "target_ratio": args.target_ratio,
            "microbatch": args.microbatch,
            "max_length": args.max_length,
            "action_loss_weight": 0.0,
            "action_temperature": 1.0,
            "option_shuffle": False,
            "optimizer_created": False,
        },
        environment={
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    )
    _write_new(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
