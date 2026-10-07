"""Safely orchestrate the frozen, internal-only R050 GPU screen.

Invoke this script through the repository's UV environment.  Child processes use
the same Python interpreter directly; the orchestrator never invokes a shell or
an environment/package manager.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preflight_r043_assets import file_hash, tree_hash


SCHEMA = "r050-pipeline-state-v1"
COMPLETE_SCHEMA = "r050-pipeline-completion-v1"
HASH_LEN = 64
OBJECTIVES = {
    "field_ce",
    "field_ce_plus_exact_outcome_nll",
    "field_ce_plus_exact_outcome_brier",
}
FROZEN_CODE_SHA256 = {
    "build_r043_causal_pairs.py": "fb620544bf52468dee50e6dbd91dceb3c4561c40d10fa76fc8d24c3a61e4bf3b",
    "build_r044_mm_replay.py": "d7125df1efb7608fe9ca0fd57d3880c2e0664aa88404da8c1819c233af4a097b",
    "build_r047_model_inventory.py": "3a47a7adfc212fff863fe7209557778c077d3c58e88f128bbf3aa091de2bd071",
    "build_r048_action_panel.py": "a9945518f134d5b5a62195c8ab727373743a7194e54a96f688a817120f671f5e",
    "build_r050_headroom.py": "5f9ee1cf4d1fbfae46863bad06cbc2d3bb062b1b115ee89c8941914362fa8e12",
    "build_r050_mix.py": "a943444063d6884ee1a76ca395e7c7c2e965f70ef8736e0dfc93c0514628cbcb",
    "build_r050_replay_pools.py": "371d7cfcd1c32d15d3bb811645888c784553e24847f865f7bcc8447d3213530a",
    "compare_r049_calibrated.py": "ea810ea3a7ef5d55385896701ac8e6b0626908b900431852d2dc11384a448e67",
    "exact_policy_executor.py": "34a6f967f311f5243a5eb4949a8d576c2bafcb49923d4d1d804de652bd84cc06",
    "preflight_r043_assets.py": "d37172c19ac0477d49b32e9804e5800d7060cb95edda2853f036bac7f9a0e03b",
    "prepare_r050_arms.py": "5f3c38074f9d571e81d1df6a08c3fc556da6dfa002b2f54b7f3d86c27d67dd6c",
    "probe_r050_gradient_scale.py": "39a337801b57ab547e69303e57216d4815200b934f052b979b43dbea749829df",
    "run_r050_arm.py": "30c8455bb8bcf12ea061fc1872d0cac902c7b182fd9cbfbf0d1d2f6d0fc36747",
    "run_r050_eval.py": "184b03bf54e6b0c5964ea8b2e81f06e9d5f1bf1010d05bf6d2d5d5d83a0a118a",
    "run_r050_headroom.py": "d2329954146a730db2bed6138982206159b5189acdfd39a568c35a357eee5fe8",
    "run_r050_retention.py": "b4a146be22bd9e478e08b90e674a5a2944ca0b34422129e87a4c8869a0f3a998",
    "score_exact_policy_panel.py": "64be9dd52f9e5d32cc883b4167dff6278a3b4993bd207bbd3aa81897145aeaba",
    "singleton_option_scorer.py": "2bd954ec37121d50beea3cc238c71d4c57429f5d8466898521c3e3631a4aad2c",
    "soft_action_executor.py": "3b8ea41646f2d4c45e1c5cac47298867c8ee5cd00b6f97b6b8b0b59b58d191be",
    "train_gemma_decision_lora.py": "74077c28e7f195fe61f2454d38fba6c7baa3338bb5d90c588346fefce30a570c",
}
# The orchestrator cannot contain its own digest.  Its hash remains a mandatory
# external binding; every child and transitive local dependency is literal here.
CODE_FILES = ("run_r050_pipeline.py", *FROZEN_CODE_SHA256)
OFFICIAL_EVALUATORS = ("jevbench", "workfloweval")
FROZEN_MIX_SHA256 = "499dd9c9a8341e55d1d965bbd1ffc45c893880fcdab4d35d295d06324c1c2003"
FROZEN_MIX_MANIFEST_SHA256 = "2b52da6fac16a803d5dd2fa14bcc54c68693f01754b3039561386602f287b53d"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checked_hash(value: str, label: str) -> str:
    if (not isinstance(value, str) or len(value) != HASH_LEN
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"{label} must be a 64-character hexadecimal sha256")
    return value.lower()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file: {path}")
    try:
        value = json.loads(path.read_bytes(), parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _regular_hash(path: Path, expected: str, label: str) -> str:
    expected = _checked_hash(expected, f"expected {label} sha256")
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file: {path}")
    observed = file_hash(path)
    if observed != expected:
        raise ValueError(f"{label} sha256 mismatch")
    return observed


def _atomic_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_new(path: Path, value: dict[str, Any]) -> None:
    """Atomically create a JSON file without replacing a concurrent winner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite immutable marker: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_assignments(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or not name or not item or name in result:
            raise ValueError(f"{label} values must be unique NAME=VALUE pairs")
        result[name] = item
    return result


@dataclass(frozen=True)
class PipelineConfig:
    out: Path
    headroom_data: Path
    headroom_data_sha256: str
    headroom_manifest: Path
    headroom_manifest_sha256: str
    train_panel: Path
    train_panel_sha256: str
    eval_panel: Path
    eval_panel_sha256: str
    panel_manifest: Path
    panel_manifest_sha256: str
    mix: Path
    mix_sha256: str
    mix_manifest: Path
    mix_manifest_sha256: str
    model: Path
    model_inventory: Path
    model_inventory_sha256: str
    parent_adapter: Path
    parent_tree_sha256: str
    parent_tree_files: int
    retention_asset_inventory: Path
    retention_asset_inventory_sha256: str
    retention_suites: dict[str, Path]
    retention_suite_hashes: dict[str, str]
    code_hashes: dict[str, str]
    python: Path = Path(sys.executable)

    @property
    def scripts(self) -> Path:
        return Path(__file__).resolve().parent

    def binding(self) -> dict[str, Any]:
        paths = {
            name: str(Path(value).resolve()) for name, value in {
                "headroom_data": self.headroom_data,
                "headroom_manifest": self.headroom_manifest,
                "train_panel": self.train_panel,
                "eval_panel": self.eval_panel,
                "panel_manifest": self.panel_manifest,
                "mix": self.mix,
                "mix_manifest": self.mix_manifest,
                "model": self.model,
                "model_inventory": self.model_inventory,
                "parent_adapter": self.parent_adapter,
                "retention_asset_inventory": self.retention_asset_inventory,
            }.items()
        }
        hashes = {
            "headroom_data": self.headroom_data_sha256,
            "headroom_manifest": self.headroom_manifest_sha256,
            "train_panel": self.train_panel_sha256,
            "eval_panel": self.eval_panel_sha256,
            "panel_manifest": self.panel_manifest_sha256,
            "mix": self.mix_sha256,
            "mix_manifest": self.mix_manifest_sha256,
            "model_inventory": self.model_inventory_sha256,
            "parent_tree": self.parent_tree_sha256,
            "retention_asset_inventory": self.retention_asset_inventory_sha256,
        }
        return {
            "paths": paths,
            "hashes": hashes,
            "parent_tree_files": self.parent_tree_files,
            "retention_suites": {name: str(path.resolve())
                                  for name, path in sorted(self.retention_suites.items())},
            "retention_suite_hashes": dict(sorted(self.retention_suite_hashes.items())),
            "code_hashes": dict(sorted(self.code_hashes.items())),
            "python": str(self.python.resolve()),
        }


def _runtime_contracts() -> dict[str, Any]:
    # Import the contracts, not copied values, so an audited runner update is
    # visible and then authenticated by the caller-supplied code hashes.
    from scripts import compare_r049_calibrated as comparator
    from scripts import run_r050_eval as evaluator
    from scripts import run_r050_headroom as headroom
    from scripts import run_r050_retention as retention
    return {
        "headroom_data_sha256": headroom.FROZEN_DATA_SHA256,
        "headroom_manifest_sha256": headroom.FROZEN_MANIFEST_SHA256,
        "eval_panel_sha256": evaluator.FROZEN_EVAL_SHA256,
        "train_panel_sha256": evaluator.FROZEN_OUTPUTS["train"]["sha256"],
        "panel_manifest_sha256": evaluator.FROZEN_MANIFEST_SHA256,
        "mix_sha256": FROZEN_MIX_SHA256,
        "mix_manifest_sha256": FROZEN_MIX_MANIFEST_SHA256,
        "model_inventory_sha256": evaluator.FROZEN_INVENTORY_SHA256,
        "parent_tree_sha256": headroom.FROZEN_PARENT_TREE_SHA256,
        "parent_tree_files": headroom.FROZEN_PARENT_TREE_FILES,
        "comparator_schema": comparator.SCHEMA,
        "retention_suites": tuple(retention.SUITES),
        "retention_suite_hashes": dict(retention.SOURCE_SHA256),
        "retention_asset_inventory_sha256": retention.FROZEN_ASSET_INVENTORY_SHA256,
        "retention_completion_schema": retention.COMPLETION_SCHEMA,
    }


def preflight(config: PipelineConfig, *, check_model_assets: bool) -> dict[str, Any]:
    contracts = _runtime_contracts()
    literals = {
        "headroom_data_sha256": config.headroom_data_sha256,
        "headroom_manifest_sha256": config.headroom_manifest_sha256,
        "eval_panel_sha256": config.eval_panel_sha256,
        "train_panel_sha256": config.train_panel_sha256,
        "panel_manifest_sha256": config.panel_manifest_sha256,
        "mix_sha256": config.mix_sha256,
        "mix_manifest_sha256": config.mix_manifest_sha256,
        "model_inventory_sha256": config.model_inventory_sha256,
        "parent_tree_sha256": config.parent_tree_sha256,
        "retention_asset_inventory_sha256": config.retention_asset_inventory_sha256,
    }
    for name, supplied in literals.items():
        if _checked_hash(supplied, name) != contracts[name]:
            raise ValueError(f"{name} differs from the audited literal")
    if config.parent_tree_files != contracts["parent_tree_files"]:
        raise ValueError("parent adapter file count differs from the audited literal")

    expected_code = set(CODE_FILES)
    if set(config.code_hashes) != expected_code:
        raise ValueError(f"code hash names must be exactly: {', '.join(CODE_FILES)}")
    for name in CODE_FILES:
        supplied = _checked_hash(config.code_hashes[name], f"expected current code {name} sha256")
        if name in FROZEN_CODE_SHA256 and supplied != FROZEN_CODE_SHA256[name]:
            raise ValueError(f"current code {name} differs from the audited literal")
        _regular_hash(config.scripts / name, config.code_hashes[name], f"current code {name}")

    files = (
        (config.headroom_data, config.headroom_data_sha256, "headroom data"),
        (config.headroom_manifest, config.headroom_manifest_sha256, "headroom manifest"),
        (config.train_panel, config.train_panel_sha256, "training panel"),
        (config.eval_panel, config.eval_panel_sha256, "evaluation panel"),
        (config.panel_manifest, config.panel_manifest_sha256, "panel manifest"),
        (config.mix, config.mix_sha256, "training mix"),
        (config.mix_manifest, config.mix_manifest_sha256, "training mix manifest"),
        (config.model_inventory, config.model_inventory_sha256, "model inventory"),
        (config.retention_asset_inventory, config.retention_asset_inventory_sha256,
         "retention asset inventory"),
    )
    for path, expected, label in files:
        _regular_hash(path, expected, label)

    expected_suites = set(contracts["retention_suites"])
    if set(config.retention_suites) != expected_suites or set(config.retention_suite_hashes) != expected_suites:
        raise ValueError("retention suite paths and hashes must exactly cover the runner suites")
    for name in contracts["retention_suites"]:
        expected = _checked_hash(config.retention_suite_hashes[name], f"{name} suite sha256")
        if expected != contracts["retention_suite_hashes"][name]:
            raise ValueError(f"{name} suite hash differs from the audited runner literal")
        _regular_hash(config.retention_suites[name], expected, f"retention suite {name}")

    # These small JSON checks catch accidentally crossed artifacts without loading a model.
    core_manifest = _read_json(config.panel_manifest, "panel manifest")
    if core_manifest.get("outputs", {}).get("train", {}).get("sha256") != config.train_panel_sha256:
        raise ValueError("panel manifest does not bind the training panel")
    if core_manifest.get("outputs", {}).get("eval", {}).get("sha256") != config.eval_panel_sha256:
        raise ValueError("panel manifest does not bind the evaluation panel")
    mix_manifest = _read_json(config.mix_manifest, "mix manifest")
    if mix_manifest.get("outputs", {}).get("train", {}).get("sha256") != config.mix_sha256:
        raise ValueError("mix manifest does not bind the training mix")

    if check_model_assets:
        if config.model.is_symlink() or not config.model.is_dir():
            raise ValueError("model must be a regular directory")
        if config.parent_adapter.is_symlink() or not config.parent_adapter.is_dir():
            raise ValueError("parent adapter must be a regular directory")
        digest, files_count = tree_hash(config.parent_adapter)
        if (digest, files_count) != (config.parent_tree_sha256, config.parent_tree_files):
            raise ValueError("parent adapter tree differs from the audited literal")

    return {"binding": config.binding(), "binding_sha256": _sha(_canonical(config.binding()))}


class Pipeline:
    def __init__(self, config: PipelineConfig, *, subprocess_run: Callable[..., Any] = subprocess.run):
        self.config = config
        self.run_process = subprocess_run
        self.root = config.out.resolve()
        self.state_path = self.root / "state.json"
        self.complete_path = self.root / "complete.json"
        self.logs = self.root / "logs"
        self.state: dict[str, Any] = {}

    def _save(self) -> None:
        _atomic_replace(self.state_path, self.state)

    def _set_stage(self, name: str, status: str, evidence: dict[str, Any] | None = None) -> None:
        self.state["stages"][name] = {"status": status, **({"evidence": evidence} if evidence else {})}
        self._save()

    def _stage(self, name: str, output: Path, command: list[str],
               validator: Callable[[], dict[str, Any]], *, accepted=(0,)) -> dict[str, Any]:
        existing = self.state["stages"].get(name)
        if existing:
            status = existing.get("status")
            if status == "complete":
                evidence = validator()
                if evidence != existing.get("evidence"):
                    raise ValueError(f"stage {name} completion evidence drifted")
                return evidence
            if status == "running":
                if not os.path.lexists(output):
                    raise RuntimeError(
                        f"stage {name} was interrupted before publishing output; "
                        "refusing an unsafe retry")
                try:
                    evidence = validator()
                except Exception as error:
                    raise RuntimeError(
                        f"stage {name} has partial or invalid interrupted output; "
                        "refusing an unsafe retry") from error
                self._set_stage(name, "complete", evidence)
                return evidence
            # A failed child was observed, rather than an ambiguous crash.  Its
            # logs/output remain forensic evidence and require a fresh root.
            raise RuntimeError(
                f"stage {name} previously failed; use a new output root after investigation")
        if os.path.lexists(output):
            raise FileExistsError(f"untracked or partial stage output exists: {output}")
        self._set_stage(name, "running")
        try:
            return_code = self._invoke(name, command)
            if return_code not in accepted:
                raise RuntimeError(f"stage {name} exited with status {return_code}")
            evidence = validator()
        except Exception:
            self._set_stage(name, "failed")
            raise
        self._set_stage(name, "complete", evidence)
        return evidence

    def _invoke(self, stage: str, command: list[str]) -> int:
        if (not command or Path(command[0]).resolve() != self.config.python.resolve()
                or any(token in " ".join(command[1:]).casefold() for token in OFFICIAL_EVALUATORS)):
            raise ValueError("invalid or forbidden R050 subprocess command")
        self.logs.mkdir(parents=True, exist_ok=True)
        stdout_path = self.logs / f"{stage}.stdout.log"
        stderr_path = self.logs / f"{stage}.stderr.log"
        if stdout_path.exists() or stderr_path.exists():
            raise FileExistsError(f"refusing to overwrite stage logs: {stage}")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = self.run_process(command, cwd=str(self.config.scripts.parent),
                                      stdout=stdout, stderr=stderr, check=False)
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
        return int(result.returncode)

    def _validate_code_receipt(self, receipt: dict[str, Any], names: tuple[str, ...]) -> None:
        observed = receipt.get("code_sha256")
        expected = {name: self.config.code_hashes[name] for name in names}
        if observed != expected:
            raise ValueError("child receipt code hashes differ from the pipeline binding")

    def _dir_completion(self, directory: Path, schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"completed stage directory is missing: {directory}")
        marker_path = directory / "complete.json"
        receipt_path = directory / "receipt.json"
        marker = _read_json(marker_path, "completion marker")
        receipt = _read_json(receipt_path, "receipt")
        if marker.get("schema") != schema:
            raise ValueError(f"wrong completion schema in {marker_path}")
        if marker.get("receipt_sha256") != file_hash(receipt_path):
            raise ValueError(f"completion marker does not bind receipt: {directory}")
        outputs = marker.get("outputs", receipt.get("outputs"))
        if isinstance(outputs, dict):
            for name, binding in outputs.items():
                target = directory / name
                if (not isinstance(binding, dict) or binding.get("sha256") != file_hash(target)
                        or binding.get("bytes") not in {None, target.stat().st_size}):
                    raise ValueError(f"completion output binding mismatch: {target}")
        return marker, receipt

    def _headroom_evidence(self) -> dict[str, Any]:
        directory = self.root / "headroom"
        marker, receipt = self._dir_completion(directory, "r050_headroom_completion_v1")
        report = _read_json(directory / "report.json", "headroom report")
        self._validate_code_receipt(receipt, (
            "run_r050_headroom.py", "build_r050_headroom.py", "build_r048_action_panel.py",
            "score_exact_policy_panel.py", "singleton_option_scorer.py",
            "exact_policy_executor.py", "train_gemma_decision_lora.py",
            "build_r047_model_inventory.py", "preflight_r043_assets.py",
            "build_r050_replay_pools.py"))
        sufficient = report.get("headroom_sufficient")
        if (type(sufficient) is not bool or marker.get("headroom_sufficient") is not sufficient
                or receipt.get("headroom_sufficient") is not sufficient):
            raise ValueError("headroom sufficiency receipts disagree")
        return {"complete_sha256": file_hash(directory / "complete.json"),
                "receipt_sha256": file_hash(directory / "receipt.json"),
                "report_sha256": file_hash(directory / "report.json"),
                "headroom_sufficient": sufficient}

    def _probe_evidence(self) -> dict[str, Any]:
        path = self.root / "gradient-probe.json"
        value = _read_json(path, "gradient probe receipt")
        if value.get("schema_version") != "r050-gradient-scale-probe-v1" or value.get("status") != "complete":
            raise ValueError("gradient probe receipt is not complete")
        bindings = value.get("bindings", {})
        expected = {
            "trainer_sha256": self.config.code_hashes["train_gemma_decision_lora.py"],
            "probe_sha256": self.config.code_hashes["probe_r050_gradient_scale.py"],
            "panel_builder_sha256": self.config.code_hashes["build_r048_action_panel.py"],
        }
        if any(bindings.get(name) != digest for name, digest in expected.items()):
            raise ValueError("gradient probe code hashes differ from the pipeline binding")
        return {"sha256": file_hash(path)}

    def _suite_evidence(self) -> dict[str, Any]:
        directory = self.root / "arms"
        manifest_path = directory / "manifest.json"
        manifest = _read_json(manifest_path, "arm suite manifest")
        arms = manifest.get("arms")
        if (manifest.get("schema") != "r050-matched-arm-suite-v1" or not isinstance(arms, list)
                or len(arms) != 3 or sorted(item.get("run_order") for item in arms) != [1, 2, 3]
                or len({item.get("arm_id") for item in arms}) != 3):
            raise ValueError("arm suite is not the complete preassigned three-arm suite")
        recipes = {}
        for arm in arms:
            path = directory / arm["recipe"]
            digest = file_hash(path)
            if digest != arm.get("sha256"):
                raise ValueError("arm recipe hash differs from suite manifest")
            recipe = _read_json(path, "arm recipe")
            # Do not inspect objective identity until every evaluation exists.
            if (recipe.get("arm_id"), recipe.get("run_order")) != (arm["arm_id"], arm["run_order"]):
                raise ValueError("arm recipe identity differs from suite manifest")
            bindings = recipe.get("bindings", {})
            expected_code = {
                "panel_builder": self.config.code_hashes["build_r048_action_panel.py"],
                "trainer": self.config.code_hashes["train_gemma_decision_lora.py"],
                "gradient_probe_code": self.config.code_hashes["probe_r050_gradient_scale.py"],
            }
            if any(bindings.get(name, {}).get("sha256") != digest
                   for name, digest in expected_code.items()):
                raise ValueError("arm recipe code hashes differ from the pipeline binding")
            recipes[arm["arm_id"]] = digest
        return {"manifest_sha256": file_hash(manifest_path), "recipes": recipes}

    def _arms(self) -> list[dict[str, Any]]:
        manifest = _read_json(self.root / "arms" / "manifest.json", "arm suite manifest")
        return sorted(manifest["arms"], key=lambda item: item["run_order"])

    def _training_evidence(self, arm: dict[str, Any]) -> dict[str, Any]:
        directory = self.root / "runs" / arm["arm_id"]
        completion_path = directory / "completion.json"
        completion = _read_json(completion_path, "arm completion")
        checkpoint = completion.get("artifacts", {}).get("checkpoint")
        if (completion.get("schema") != "r050-matched-arm-completion-v1" or completion.get("status") != "complete"
                or completion.get("arm_id") != arm["arm_id"]
                or completion.get("run_order") != arm["run_order"]
                or completion.get("recipe_sha256") != arm["sha256"]
                or not isinstance(checkpoint, dict)):
            raise ValueError("arm completion lineage is invalid")
        adapter = Path(checkpoint.get("path", ""))
        try:
            adapter.resolve().relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError("candidate adapter escapes its immutable arm output") from error
        if adapter.is_symlink() or not adapter.is_dir() or any(item.is_symlink() for item in adapter.rglob("*")):
            raise ValueError("candidate adapter tree is missing or contains symlinks")
        digest, files = tree_hash(adapter)
        if (digest, files) != (checkpoint.get("tree_sha256"), checkpoint.get("files")):
            raise ValueError("candidate adapter tree differs from arm completion")
        return {"completion_sha256": file_hash(completion_path),
                "adapter": str(adapter.resolve()), "adapter_tree_sha256": digest,
                "adapter_tree_files": files}

    def _eval_evidence(self, arm: dict[str, Any]) -> dict[str, Any]:
        directory = self.root / "eval" / arm["arm_id"]
        marker, receipt = self._dir_completion(directory, "r050_full_eval_completion_v1")
        self._validate_code_receipt(receipt, (
            "run_r050_eval.py", "run_r050_headroom.py", "build_r048_action_panel.py",
            "score_exact_policy_panel.py", "compare_r049_calibrated.py",
            "singleton_option_scorer.py", "exact_policy_executor.py",
            "train_gemma_decision_lora.py", "build_r047_model_inventory.py",
            "preflight_r043_assets.py", "run_r050_arm.py"))
        completion_sha = self._training_evidence(arm)["completion_sha256"]
        if receipt.get("arm_completion", {}).get("sha256") != completion_sha:
            raise ValueError("evaluation receipt does not bind the arm completion")
        prediction_sha = file_hash(directory / "predictions.jsonl")
        if marker.get("prediction_sha256") != prediction_sha:
            raise ValueError("evaluation completion does not bind predictions")
        return {"complete_sha256": file_hash(directory / "complete.json"),
                "receipt_sha256": file_hash(directory / "receipt.json"),
                "prediction_sha256": prediction_sha,
                "report_sha256": file_hash(directory / "report.json")}

    def _comparison_evidence(self, candidate_id: str, control_id: str) -> dict[str, Any]:
        path = self.root / "comparisons" / f"{candidate_id}-vs-{control_id}.json"
        report = _read_json(path, "comparison report")
        if report.get("schema") != _runtime_contracts()["comparator_schema"]:
            raise ValueError("comparison does not use comparator v5")
        expected = {
            "panel": self.config.eval_panel_sha256,
            "control": self._eval_evidence(next(a for a in self._arms() if a["arm_id"] == control_id))["prediction_sha256"],
            "candidate": self._eval_evidence(next(a for a in self._arms() if a["arm_id"] == candidate_id))["prediction_sha256"],
        }
        inputs = report.get("provenance", {}).get("inputs", {})
        if any(inputs.get(name, {}).get("sha256") != digest for name, digest in expected.items()):
            raise ValueError("comparison provenance differs from literal panel/prediction hashes")
        gate = report.get("internal_stage1_gate", {}).get("passed")
        delta = report.get("overall", {}).get("paired_delta", {}).get("equal_semantic_cluster", {})
        outcome = delta.get("complete_outcome_accuracy")
        brier = delta.get("categorical_brier_absolute_improvement")
        if (type(gate) is not bool or isinstance(outcome, bool)
                or not isinstance(outcome, (int, float)) or not math.isfinite(outcome)
                or isinstance(brier, bool) or not isinstance(brier, (int, float))
                or not math.isfinite(brier)):
            raise ValueError("comparison is missing its gate or deterministic tie-break metrics")
        return {"sha256": file_hash(path), "passed": gate,
                "complete_outcome_gain": outcome, "brier_gain": brier}

    def _retention_evidence(self, arm_id: str) -> dict[str, Any]:
        directory = self.root / "retention" / arm_id
        marker, receipt = self._dir_completion(
            directory, _runtime_contracts()["retention_completion_schema"])
        self._validate_code_receipt(receipt, (
            "run_r050_retention.py", "singleton_option_scorer.py",
            "train_gemma_decision_lora.py", "build_r047_model_inventory.py",
            "preflight_r043_assets.py", "run_r050_eval.py", "run_r050_arm.py",
            "build_r050_replay_pools.py"))
        report = _read_json(directory / "report.json", "retention report")
        passed = report.get("promotion_pass")
        if type(passed) is not bool or marker.get("promotion_pass") is not passed:
            raise ValueError("retention gate receipts disagree")
        return {"complete_sha256": file_hash(directory / "complete.json"),
                "receipt_sha256": file_hash(directory / "receipt.json"),
                "report_sha256": file_hash(directory / "report.json"), "passed": passed}

    def _command(self, script: str, *arguments: Any) -> list[str]:
        return [str(self.config.python.resolve()), str((self.config.scripts / script).resolve()),
                *map(str, arguments)]

    def execute(self, binding: dict[str, Any]) -> dict[str, Any]:
        c = self.config
        headroom_out = self.root / "headroom"
        headroom = self._stage("01-headroom", headroom_out, self._command(
            "run_r050_headroom.py", "--data", c.headroom_data, "--manifest", c.headroom_manifest,
            "--model", c.model, "--inventory", c.model_inventory, "--adapter", c.parent_adapter,
            "--out", headroom_out, "--expected-data-sha256", c.headroom_data_sha256,
            "--expected-manifest-sha256", c.headroom_manifest_sha256,
            "--expected-inventory-sha256", c.model_inventory_sha256,
            "--expected-adapter-tree-sha256", c.parent_tree_sha256,
            "--expected-adapter-tree-files", c.parent_tree_files), self._headroom_evidence,
            accepted=(0, 2))
        if not headroom["headroom_sufficient"]:
            result = {"schema": COMPLETE_SCHEMA, "status": "headroom_insufficient",
                      "binding_sha256": binding["binding_sha256"], "winner": None}
            if self.complete_path.exists():
                if _read_json(self.complete_path, "pipeline completion") != result:
                    raise ValueError("pipeline completion marker drifted")
            else:
                _atomic_new(self.complete_path, result)
            return result

        probe_path = self.root / "gradient-probe.json"
        probe = self._stage("02-gradient-probe", probe_path, self._command(
            "probe_r050_gradient_scale.py", "--model", c.model,
            "--model-inventory", c.model_inventory, "--parent-adapter", c.parent_adapter,
            "--train-panel", c.train_panel, "--panel-manifest", c.panel_manifest,
            "--panel-builder", c.scripts / "build_r048_action_panel.py", "--out", probe_path,
            "--signatures", 64, "--seed", 50050, "--target-ratio", 0.25,
            "--microbatch", 1, "--max-length", 8192), self._probe_evidence)

        suite_out = self.root / "arms"
        self._stage("03-prepare-arms", suite_out, self._command(
            "prepare_r050_arms.py", "--mix", c.mix, "--mix-manifest", c.mix_manifest,
            "--model", c.model, "--model-inventory", c.model_inventory,
            "--gradient-receipt", probe_path, "--panel", c.train_panel,
            "--panel-manifest", c.panel_manifest,
            "--panel-builder", c.scripts / "build_r048_action_panel.py",
            "--parent-adapter", c.parent_adapter,
            "--trainer", c.scripts / "train_gemma_decision_lora.py",
            "--probe", c.scripts / "probe_r050_gradient_scale.py", "--out", suite_out,
            "--expected-mix-sha256", c.mix_sha256,
            "--expected-mix-manifest-sha256", c.mix_manifest_sha256,
            "--expected-model-inventory-sha256", c.model_inventory_sha256,
            "--expected-gradient-receipt-sha256", probe["sha256"],
            "--expected-panel-sha256", c.train_panel_sha256,
            "--expected-panel-manifest-sha256", c.panel_manifest_sha256,
            "--expected-builder-sha256", c.code_hashes["build_r048_action_panel.py"],
            "--expected-trainer-sha256", c.code_hashes["train_gemma_decision_lora.py"],
            "--expected-probe-sha256", c.code_hashes["probe_r050_gradient_scale.py"],
            "--expected-parent-tree-sha256", c.parent_tree_sha256,
            "--expected-parent-tree-files", c.parent_tree_files, "--assignment-seed", 50050),
            self._suite_evidence)

        arms = self._arms()
        for arm in arms:
            recipe = suite_out / arm["recipe"]
            run_out = self.root / "runs" / arm["arm_id"]
            self._stage(f"04-train-{arm['arm_id']}", run_out, self._command(
                "run_r050_arm.py", "--recipe", recipe, "--expected-recipe-sha256", arm["sha256"],
                "--out", run_out), lambda arm=arm: self._training_evidence(arm))

        for arm in arms:
            trained = self._training_evidence(arm)
            eval_out = self.root / "eval" / arm["arm_id"]
            completion = self.root / "runs" / arm["arm_id"] / "completion.json"
            self._stage(f"05-eval-{arm['arm_id']}", eval_out, self._command(
                "run_r050_eval.py", "--data", c.eval_panel, "--manifest", c.panel_manifest,
                "--model", c.model, "--inventory", c.model_inventory,
                "--adapter", trained["adapter"], "--out", eval_out,
                "--expected-data-sha256", c.eval_panel_sha256,
                "--expected-manifest-sha256", c.panel_manifest_sha256,
                "--expected-inventory-sha256", c.model_inventory_sha256,
                "--expected-adapter-tree-sha256", trained["adapter_tree_sha256"],
                "--expected-adapter-tree-files", trained["adapter_tree_files"],
                "--arm-completion", completion,
                "--expected-arm-completion-sha256", trained["completion_sha256"]),
                lambda arm=arm: self._eval_evidence(arm))

        # Blinding ends only now: all three training and full-evaluation outputs exist.
        objectives: dict[str, str] = {}
        for arm in arms:
            recipe_path = suite_out / arm["recipe"]
            if file_hash(recipe_path) != arm["sha256"]:
                raise ValueError("sealed recipe changed before unblinding")
            objective = _read_json(recipe_path, "sealed arm recipe").get("objective", {}).get("objective")
            if objective not in OBJECTIVES or objective in objectives:
                raise ValueError("sealed recipes do not define exactly one of each R050 objective")
            objectives[objective] = arm["arm_id"]
        if set(objectives) != OBJECTIVES:
            raise ValueError("sealed recipes do not define the complete objective set")
        control_id = objectives["field_ce"]
        candidates = [objectives["field_ce_plus_exact_outcome_nll"],
                      objectives["field_ce_plus_exact_outcome_brier"]]

        comparisons: dict[str, dict[str, Any]] = {}
        for candidate_id in candidates:
            control_eval = self._eval_evidence(next(a for a in arms if a["arm_id"] == control_id))
            candidate_eval = self._eval_evidence(next(a for a in arms if a["arm_id"] == candidate_id))
            output = self.root / "comparisons" / f"{candidate_id}-vs-{control_id}.json"
            comparisons[candidate_id] = self._stage(
                f"06-compare-{candidate_id}", output, self._command(
                    "compare_r049_calibrated.py", "--panel", c.eval_panel,
                    "--control-predictions", self.root / "eval" / control_id / "predictions.jsonl",
                    "--candidate-predictions", self.root / "eval" / candidate_id / "predictions.jsonl",
                    "--manifest", c.panel_manifest, "--manifest-sha256", c.panel_manifest_sha256,
                    "--panel-sha256", c.eval_panel_sha256,
                    "--control-sha256", control_eval["prediction_sha256"],
                    "--candidate-sha256", candidate_eval["prediction_sha256"], "--out", output),
                lambda candidate_id=candidate_id: self._comparison_evidence(candidate_id, control_id))

        retention: dict[str, dict[str, Any]] = {}
        for candidate_id in candidates:
            if not comparisons[candidate_id]["passed"]:
                continue
            arm = next(item for item in arms if item["arm_id"] == candidate_id)
            trained = self._training_evidence(arm)
            output = self.root / "retention" / candidate_id
            arguments: list[Any] = []
            for suite in _runtime_contracts()["retention_suites"]:
                arguments.extend((f"--{suite}", c.retention_suites[suite],
                                  f"--expected-{suite}-sha256", c.retention_suite_hashes[suite]))
            arguments.extend((
                "--asset-inventory", c.retention_asset_inventory,
                "--expected-asset-inventory-sha256", c.retention_asset_inventory_sha256,
                "--model", c.model, "--model-inventory", c.model_inventory,
                "--expected-model-inventory-sha256", c.model_inventory_sha256,
                "--parent-adapter", c.parent_adapter, "--candidate-adapter", trained["adapter"],
                "--expected-candidate-tree-sha256", trained["adapter_tree_sha256"],
                "--expected-candidate-tree-files", trained["adapter_tree_files"],
                "--candidate-completion", self.root / "runs" / candidate_id / "completion.json",
                "--expected-candidate-completion-sha256", trained["completion_sha256"],
                "--out", output))
            retention[candidate_id] = self._stage(
                f"07-retention-{candidate_id}", output,
                self._command("run_r050_retention.py", *arguments),
                lambda candidate_id=candidate_id: self._retention_evidence(candidate_id),
                accepted=(0, 2))

        eligible = [arm_id for arm_id in candidates
                    if comparisons[arm_id]["passed"] and retention[arm_id]["passed"]]
        winner = min(eligible, key=lambda arm_id: (
            -comparisons[arm_id]["complete_outcome_gain"],
            -comparisons[arm_id]["brier_gain"], arm_id)) if eligible else None
        result = {
            "schema": COMPLETE_SCHEMA, "status": "complete", "binding_sha256": binding["binding_sha256"],
            "control_arm_id": control_id, "winner": winner,
            "eligible_candidates": sorted(eligible),
            "comparisons": comparisons, "retention": retention,
            "official_evaluations_run": [],
        }
        if self.complete_path.exists():
            if _read_json(self.complete_path, "pipeline completion") != result:
                raise ValueError("pipeline completion marker drifted")
        else:
            _atomic_new(self.complete_path, result)
        return result

    def run(self) -> dict[str, Any]:
        binding = preflight(self.config, check_model_assets=True)
        try:
            self.root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        lock = self.root / ".pipeline.lock"
        # The file is a stable rendezvous inode, not the lock itself.  flock is
        # released by the OS on process death, so a SIGKILL cannot brick resume.
        lock_stream = lock.open("a+b")
        try:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another pipeline invocation is active") from error
            try:
                if self.state_path.exists():
                    self.state = _read_json(self.state_path, "pipeline state")
                    if (self.state.get("schema") != SCHEMA
                            or self.state.get("binding_sha256") != binding["binding_sha256"]
                            or self.state.get("binding") != binding["binding"]):
                        raise ValueError("pipeline state input/code binding drifted")
                else:
                    # A crash between mkdir/lock and the first atomic ledger
                    # publish can leave an empty root or only this stale inode.
                    unexpected = [item for item in self.root.iterdir() if item != lock]
                    if unexpected:
                        raise FileExistsError("output root exists without a valid state ledger")
                    self.state = {"schema": SCHEMA, **binding, "stages": {}}
                    self._save()
                return self.execute(binding)
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            lock_stream.close()


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    suites = {name: Path(value) for name, value in
              _parse_assignments(args.retention_suite, "retention suite").items()}
    suite_hashes = _parse_assignments(args.retention_suite_sha256, "retention suite hash")
    code_hashes = {
        name: _checked_hash(value, f"expected current code {name} sha256")
        for name, value in _parse_assignments(
            args.expected_code_sha256, "code hash").items()
    }
    return PipelineConfig(
        out=args.out, headroom_data=args.headroom_data,
        headroom_data_sha256=args.headroom_data_sha256,
        headroom_manifest=args.headroom_manifest,
        headroom_manifest_sha256=args.headroom_manifest_sha256,
        train_panel=args.train_panel, train_panel_sha256=args.train_panel_sha256,
        eval_panel=args.eval_panel, eval_panel_sha256=args.eval_panel_sha256,
        panel_manifest=args.panel_manifest, panel_manifest_sha256=args.panel_manifest_sha256,
        mix=args.mix, mix_sha256=args.mix_sha256, mix_manifest=args.mix_manifest,
        mix_manifest_sha256=args.mix_manifest_sha256, model=args.model,
        model_inventory=args.model_inventory,
        model_inventory_sha256=args.model_inventory_sha256,
        parent_adapter=args.parent_adapter, parent_tree_sha256=args.parent_tree_sha256,
        parent_tree_files=args.parent_tree_files,
        retention_asset_inventory=args.retention_asset_inventory,
        retention_asset_inventory_sha256=args.retention_asset_inventory_sha256,
        retention_suites=suites, retention_suite_hashes=suite_hashes,
        code_hashes=code_hashes, python=args.python)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    for name in ("headroom-data", "headroom-manifest", "train-panel", "eval-panel",
                 "panel-manifest", "mix", "mix-manifest", "model-inventory"):
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--parent-adapter", required=True, type=Path)
    parser.add_argument("--parent-tree-sha256", required=True)
    parser.add_argument("--parent-tree-files", required=True, type=int)
    parser.add_argument("--retention-asset-inventory", required=True, type=Path)
    parser.add_argument("--retention-asset-inventory-sha256", required=True)
    parser.add_argument("--retention-suite", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--retention-suite-sha256", action="append", default=[], metavar="NAME=SHA256")
    parser.add_argument("--expected-code-sha256", action="append", default=[], metavar="FILE=SHA256")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="interpreter from the externally activated UV environment")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    if args.dry_run:
        checked = preflight(config, check_model_assets=False)
        print(json.dumps({"status": "dry_run", "gpu_calls": 0,
                          "binding_sha256": checked["binding_sha256"],
                          "stages": ["headroom", "gradient_probe", "prepare_arms",
                                     "train_three_preassigned", "eval_all_three",
                                     "unblind_and_compare", "retention_for_passing", "select_winner"],
                          "official_evaluations": []}, sort_keys=True))
        return 0
    result = Pipeline(config).run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
