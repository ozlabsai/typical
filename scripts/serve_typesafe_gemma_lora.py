#!/usr/bin/env python3
"""Serve a pinned Typical Gemma LoRA through a TypeSafe-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import re
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.gemma_decision_runtime import (
    GemmaDecisionRuntime,
    JsonContent,
    Option,
    OptionScores,
    Overlength,
    bind_artifact,
    runtime_environment_identity,
)
from scripts.preflight_r043_assets import file_hash
from scripts.singleton_option_scorer import validate_native_probs

MAX_OPTIONS = 26
DEFAULT_MAX_QUESTIONS = 256
MODEL_ID_PATTERN = re.compile(r"-sha256-([0-9a-f]{12,64})$")


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: JsonContent | None = None
    criteria: dict[str, JsonContent | None] | None = Field(default=None, max_length=2)

    @field_validator("criteria")
    @classmethod
    def semantic_criteria(cls, value):
        if value is not None and not set(value) <= {"true", "false"}:
            raise ValueError("noul criteria keys must be true and false")
        return value


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: JsonContent | None = None
    criteria: dict[str, JsonContent | None] = Field(min_length=1, max_length=MAX_OPTIONS)


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: JsonContent | None = None
    criteria: list[JsonContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(StrictModel):
    state: JsonContent
    model: str = Field(min_length=1)
    questions: dict[str, Question] = Field(min_length=1)


class DecisionRuntime(Protocol):
    def score_options(
        self,
        state: JsonContent,
        instructions: Any,
        items: Sequence[Option],
        *,
        question_type: str,
    ) -> OptionScores: ...


def _sha256_file(path: Path) -> str:
    return file_hash(path)


def _source(value: str) -> str:
    path = Path(value)
    return str(path.resolve()) if path.exists() else value


def build_startup_receipt(
    *,
    model_id: str,
    base_model: str,
    base_revision: str,
    expected_model_tree_sha256: str | None,
    adapter: str,
    adapter_revision: str | None,
    expected_adapter_tree_sha256: str | None,
    max_context: int,
    dtype: str,
    device: str,
    attn_implementation: str,
    bidirectional_prompt: bool = False,
) -> dict[str, Any]:
    """Bind the served name to artifacts, exact code, and inference configuration."""
    model_artifact = bind_artifact(
        base_model,
        base_revision,
        expected_model_tree_sha256,
        name="model",
    )
    adapter_artifact = bind_artifact(
        adapter,
        adapter_revision,
        expected_adapter_tree_sha256,
        name="adapter",
    )
    match = MODEL_ID_PATTERN.search(model_id)
    if match is None or not adapter_artifact["canonical_sha256"].startswith(match.group(1)):
        raise ValueError("model_id must end with an adapter sha256 prefix of at least 12 hex digits")
    receipt: dict[str, Any] = {
        "schema_version": "typical.typesafe_startup.v1",
        "model_id": model_id,
        "base_model": {
            "source": _source(base_model),
            "revision": base_revision,
            "artifact": model_artifact,
        },
        "adapter": {
            "source": _source(adapter),
            "revision": adapter_revision,
            "artifact": adapter_artifact,
        },
        "runtime": {
            "max_context": max_context,
            "dtype": dtype,
            "device": device,
            "attn_implementation": attn_implementation,
            # R057: only recorded when on, so default receipts stay byte-identical
            **({"prompt_attention": "bidirectional"} if bidirectional_prompt else {}),
        },
        "code": {
            "runtime_sha256": _sha256_file(Path(__file__).with_name("gemma_decision_runtime.py")),
            "server_sha256": _sha256_file(Path(__file__)),
            "scorer_sha256": _sha256_file(Path(__file__).with_name("singleton_option_scorer.py")),
            "training_helper_sha256": _sha256_file(
                Path(__file__).with_name("train_gemma_decision_lora.py")
            ),
        },
        "environment": runtime_environment_identity(),
    }
    encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    receipt["receipt_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return receipt


def write_startup_receipt(path: Path, receipt: dict[str, Any]) -> None:
    """Create an immutable receipt, accepting only an identical restart."""
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing startup receipt differs") from error
        if existing != receipt:
            raise ValueError("existing startup receipt differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _require_safe_bind(
    host: str,
    *,
    api_key: str | None,
    allow_unauthenticated_remote: bool,
) -> None:
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.casefold() == "localhost"
    if not loopback and api_key is None and not allow_unauthenticated_remote:
        raise ValueError("refusing unauthenticated non-loopback bind")


def _items(question: Question) -> list[Option]:
    if isinstance(question, NoulQuestion):
        criteria = question.criteria or {}
        # Preserve explicitly supplied order, then add any implicit semantic outcome.
        # P(true) is looked up by label, so neither meaning depends on a fixed slot.
        labels = [*criteria, *(key for key in ("true", "false") if key not in criteria)]
        return [(label, criteria.get(label)) for label in labels]
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria.items())
    return [(str(index), value) for index, value in enumerate(question.criteria)]


def _answer(question: Question, scores: OptionScores) -> dict[str, Any]:
    labels = [label for label, _ in _items(question)]
    probabilities = validate_native_probs(scores.probabilities, labels)
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": probabilities["true"]}
    count = len(probabilities)
    confidence = 1.0 if count == 1 else max(
        0.0,
        min(1.0, (count * max(probabilities.values()) - 1.0) / (count - 1.0)),
    )
    if isinstance(question, ChoiceQuestion):
        choice = max(labels, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": probabilities,
        }
    return {
        "type": "score",
        "score": sum(index * probabilities[str(index)] for index in range(len(labels))),
        "confidence": confidence,
        "legend": {str(index): value for index, value in enumerate(question.criteria)},
        "probabilities": probabilities,
    }


def create_app(
    runtime: DecisionRuntime,
    *,
    model_id: str,
    api_key: str | None = None,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    release_date: str = "2026-10-07",
) -> FastAPI:
    """Create an injectable, exact-model server without loading GPU state at import time."""
    if not model_id.strip():
        raise ValueError("model_id must be nonempty")
    if max_questions < 120:
        raise ValueError("max_questions must be at least 120")
    if api_key is not None and not api_key:
        raise ValueError("api_key must be nonempty when provided")

    app = FastAPI(title="Typical TypeSafe-compatible decision API")
    request_lock = threading.Lock()

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        scheme, separator, credential = (authorization or "").partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not hmac.compare_digest(credential, api_key)
        ):
            raise HTTPException(
                status_code=401,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    authenticated = Depends(authenticate)

    @app.get("/v1/models", dependencies=[authenticated])
    def models() -> dict[str, Any]:
        return {"models": [{
            "name": model_id,
            "description": "Typical Gemma decision model",
            "release_date": release_date,
        }]}

    @app.post("/v1/systemone", dependencies=[authenticated])
    def system_one(request: SystemOneRequest) -> dict[str, Any]:
        if request.model != model_id:
            raise HTTPException(status_code=404, detail="model not found")
        if len(request.questions) > max_questions:
            raise HTTPException(
                status_code=422,
                detail=f"questions must contain at most {max_questions} items",
            )

        answers: dict[str, Any] = {}
        input_tokens = 0
        try:
            # Keep all questions from one workflow node contiguous. The runtime's
            # forward lock remains narrower and is never acquired in reverse order.
            with request_lock:
                for question_id, question in request.questions.items():
                    scores = runtime.score_options(
                        request.state,
                        question.instructions,
                        _items(question),
                        question_type=question.type,
                    )
                    if (
                        isinstance(scores.input_tokens, bool)
                        or not isinstance(scores.input_tokens, int)
                        or scores.input_tokens < 0
                    ):
                        raise ValueError("runtime returned an invalid input token count")
                    answers[question_id] = _answer(question, scores)
                    input_tokens += scores.input_tokens
        except Overlength as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail="decision runtime unavailable") from error

        if list(answers) != list(request.questions):
            raise HTTPException(status_code=503, detail="decision runtime unavailable")
        return {
            "model": model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The TypeSafe SDK base URL is the host root (for example "
            "http://127.0.0.1:8090), because the SDK appends /v1/systemone."
        ),
    )
    parser.add_argument("--model", required=True, help="pinned Gemma model path or repository")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected-model-tree-sha256")
    parser.add_argument("--adapter", required=True, help="winner LoRA path or repository")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--expected-adapter-tree-sha256")
    parser.add_argument("--model-id", required=True, help="immutable identifier returned by the API")
    parser.add_argument("--receipt", type=Path, required=True, help="immutable startup receipt JSON")
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--bidirectional-prompt", action="store_true",
                        help="R057: read the prompt before the answer slot bidirectionally (adapter trained so)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--allow-unauthenticated-remote",
        action="store_true",
        help="explicitly permit a non-loopback bind when the API-key environment variable is empty",
    )
    parser.add_argument(
        "--api-key-env",
        default="TYPICAL_API_KEY",
        help="environment variable containing the optional bearer token; empty disables auth",
    )
    parser.add_argument("--release-date", default="2026-10-07")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_context < 16384:
        raise SystemExit("--max-context must be at least 16384 for WorkflowEvals")
    if not Path(args.adapter).exists() and not args.adapter_revision:
        raise SystemExit("a remote --adapter requires --adapter-revision")
    api_key = (os.getenv(args.api_key_env) or "").strip() or None
    try:
        _require_safe_bind(
            args.host,
            api_key=api_key,
            allow_unauthenticated_remote=args.allow_unauthenticated_remote,
        )
        receipt = build_startup_receipt(
            model_id=args.model_id,
            base_model=args.model,
            base_revision=args.model_revision,
            expected_model_tree_sha256=args.expected_model_tree_sha256,
            adapter=args.adapter,
            adapter_revision=args.adapter_revision,
            expected_adapter_tree_sha256=args.expected_adapter_tree_sha256,
            max_context=args.max_context,
            dtype=args.dtype,
            device=args.device,
            attn_implementation=args.attn_implementation,
            bidirectional_prompt=args.bidirectional_prompt,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    runtime = GemmaDecisionRuntime(
        args.model,
        args.model_revision,
        args.adapter,
        args.adapter_revision,
        args.max_context,
        args.dtype,
        args.device,
        args.attn_implementation,
        args.bidirectional_prompt,
    ).load()
    write_startup_receipt(args.receipt, receipt)
    app = create_app(
        runtime,
        model_id=args.model_id,
        api_key=api_key,
        max_questions=args.max_questions,
        release_date=args.release_date,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
