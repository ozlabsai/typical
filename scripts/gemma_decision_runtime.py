#!/usr/bin/env python3
"""Shared direct-logit runtime for Gemma typed-decision adapters and servers."""

from __future__ import annotations

import hashlib
import json
import threading
import time
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
