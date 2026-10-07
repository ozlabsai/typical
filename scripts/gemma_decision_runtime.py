#!/usr/bin/env python3
"""Shared direct-logit runtime for Gemma typed-decision adapters and servers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import threading
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.preflight_r043_assets import file_hash, tree_hash
from scripts.singleton_option_scorer import singleton_option_probabilities

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)

JsonContent = str | dict[str, Any] | list[Any]
Option = tuple[str, Any]
QUESTION_TYPES = frozenset(("noul", "choice", "score"))
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUNTIME_DEPENDENCIES = (
    "accelerate",
    "peft",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)


def runtime_environment_identity() -> dict[str, Any]:
    """Bind inference to the repository lock and its directly used ML stack."""
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    lock = tomllib.loads(lock_path.read_text())
    versions = {
        package.get("name"): package.get("version")
        for package in lock.get("package", [])
        if isinstance(package, dict)
    }
    locked = {name: versions.get(name) for name in RUNTIME_DEPENDENCIES}
    if any(not isinstance(version, str) or not version for version in locked.values()):
        raise ValueError("uv.lock lacks a required inference dependency")
    installed = {name: importlib.metadata.version(name) for name in RUNTIME_DEPENDENCIES}
    if installed != locked:
        raise ValueError("installed inference dependencies differ from uv.lock")
    return {
        "uv_lock_sha256": file_hash(lock_path),
        "dependencies": {
            name: {"locked": locked[name], "installed": installed[name]}
            for name in RUNTIME_DEPENDENCIES
        },
    }


class Overlength(ValueError):
    """A prompt exceeded the explicitly configured model input limit."""

    def __init__(self, tokens: int, maximum: int) -> None:
        self.tokens = tokens
        self.maximum = maximum
        super().__init__(f"prompt has {tokens} tokens; configured maximum is {maximum}")


@dataclass(frozen=True)
class OptionScores:
    probabilities: dict[str, float]
    input_tokens: int


def render_decision_prompt(
    state: JsonContent,
    instructions: Any,
    items: Sequence[Option],
) -> str:
    """Render the frozen training/JevBench prompt byte-for-byte."""
    if not 1 <= len(items) <= len(LETTERS):
        raise ValueError("options must contain 1-26 items")
    rendered_state = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    )
    rendered_instructions = instructions or ""
    if not isinstance(rendered_instructions, str):
        rendered_instructions = json.dumps(rendered_instructions, ensure_ascii=False)
    lines = [
        rendered_state.rstrip(),
        "",
        rendered_instructions.rstrip(),
        "",
        "Options:",
    ]
    for index, (_, description) in enumerate(items):
        lines.append(f"{LETTERS[index]}. {description}")
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    return "\n".join(lines)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value in ({}, [])


def _structured(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_typesafe_decision_prompt(
    state: JsonContent,
    instructions: Any,
    items: Sequence[Option],
    *,
    question_type: str,
) -> str:
    """Render TypeSafe criteria with explicit semantic labels and stable JSON."""
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unsupported question type {question_type!r}")
    rendered: list[Option] = []
    for label, description in items:
        if question_type == "choice":
            text = label if _blank(description) else f"{label}: {_structured(description)}"
        elif question_type == "score":
            text = f"Level {label}" if _blank(description) else _structured(description)
        else:
            semantic = {"true": "Yes", "false": "No"}.get(label)
            if semantic is None:
                raise ValueError("noul options must use true/false semantic labels")
            text = semantic if _blank(description) else f"{semantic}: {_structured(description)}"
        rendered.append((label, text))
    # The values above are strings, so legacy option interpolation is now exact
    # while state/instruction formatting stays shared with training.
    return render_decision_prompt(state, instructions, rendered)


def artifact_identity(value: str, revision: str | None) -> dict[str, Any]:
    """Use the canonical R043/R050 hashes, or bind a remote pinned identity."""
    path = Path(value)
    if path.is_file():
        digest = file_hash(path)
        return {
            "kind": "file",
            "canonical_sha256": digest,
            "file_sha256": digest,
            "files": 1,
            "bytes": path.stat().st_size,
        }
    if path.is_dir():
        digest, count = tree_hash(path)
        total_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        return {
            "kind": "tree",
            "canonical_sha256": digest,
            "tree_sha256": digest,
            "files": count,
            "bytes": total_bytes,
        }
    payload = json.dumps(
        {"source": value, "revision": revision}, sort_keys=True, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "kind": "pinned-identity",
        "canonical_sha256": digest,
        "identity_sha256": digest,
        "files": None,
        "bytes": None,
    }


def bind_artifact(
    value: str,
    revision: str | None,
    expected_tree_sha256: str | None,
    *,
    name: str,
) -> dict[str, Any]:
    """Fail closed on local tree drift; keep remote identities revision-pinned."""
    path = Path(value)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"local {name} must be a directory tree")
        if expected_tree_sha256 is None:
            raise ValueError(f"local {name} requires --expected-{name}-tree-sha256")
        if SHA256_RE.fullmatch(expected_tree_sha256) is None:
            raise ValueError(f"expected {name} tree sha256 must be 64 lowercase hex digits")
        artifact = artifact_identity(value, revision)
        if artifact["tree_sha256"] != expected_tree_sha256:
            raise ValueError(f"{name} tree sha256 mismatch")
        return artifact
    if not revision:
        raise ValueError(f"remote {name} requires a pinned revision")
    if expected_tree_sha256 is not None:
        raise ValueError(f"remote {name} cannot use an expected tree sha256")
    return artifact_identity(value, revision)


class GemmaDecisionRuntime:
    """Load one pinned Gemma/LoRA and expose its exact singleton option probabilities."""

    def __init__(
        self,
        model: str,
        model_revision: str,
        adapter: str | None,
        adapter_revision: str | None,
        max_context: int,
        dtype: str,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
    ) -> None:
        if max_context < 1:
            raise ValueError("max_context must be positive")
        self.model = model
        self.model_revision = model_revision
        self.adapter = adapter
        self.adapter_revision = adapter_revision
        self.max_context = max_context
        self.dtype = dtype
        self.device = device
        self.attn_implementation = attn_implementation
        self._model = self._tokenizer = self._groups = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.model + (f"+{self.adapter}" if self.adapter else "")

    def load(self) -> GemmaDecisionRuntime:
        if self._model is not None:
            return self
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        from scripts.train_gemma_decision_lora import letter_token_groups

        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype not in dtypes:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            revision=self.model_revision,
            local_files_only=Path(self.model).exists(),
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self.model,
            revision=self.model_revision,
            dtype=dtypes[self.dtype],
            local_files_only=Path(self.model).exists(),
            attn_implementation=self.attn_implementation,
        )
        advertised = getattr(
            getattr(model.config, "text_config", model.config), "max_position_embeddings", None
        )
        if not isinstance(advertised, int) or self.max_context > advertised:
            raise ValueError(
                f"configured max_context={self.max_context} exceeds or cannot verify "
                f"model limit {advertised!r}"
            )
        if self.adapter:
            model = PeftModel.from_pretrained(
                model,
                self.adapter,
                revision=self.adapter_revision,
                local_files_only=Path(self.adapter).exists(),
            )
        model.config.use_cache = False
        self._model = model.to(self.device).eval()
        self._groups = letter_token_groups(self._tokenizer)
        self.load_s = time.perf_counter() - started
        self.model_context = advertised
        return self

    def _input_ids(self, prompt: str):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self._tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )["input_ids"]
        tokens = int(ids.shape[-1])
        if tokens > self.max_context:
            raise Overlength(tokens, self.max_context)
        return ids, tokens

    def score_prompt(self, prompt: str, labels: Sequence[str]) -> OptionScores:
        """Run B=1 and bind each option logit to the caller's declared semantic label."""
        import torch

        self.load()
        ids, tokens = self._input_ids(prompt)
        with self._lock, torch.inference_mode():
            probabilities = singleton_option_probabilities(
                self._model,
                ids.to(self.device),
                self._groups,
                list(labels),
            )
        return OptionScores(probabilities=probabilities, input_tokens=tokens)

    def score_options(
        self,
        state: JsonContent,
        instructions: Any,
        items: Sequence[Option],
        *,
        question_type: str,
    ) -> OptionScores:
        labels = [label for label, _ in items]
        if len(set(labels)) != len(labels):
            raise ValueError("option labels must be unique")
        return self.score_prompt(
            render_typesafe_decision_prompt(
                state, instructions, items, question_type=question_type
            ),
            labels,
        )
