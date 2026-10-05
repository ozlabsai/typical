#!/usr/bin/env python3
"""Run a pinned Gemma-4 base and optional Typical LoRA in JevBench's native runner.

This writes raw per-item evidence and a run manifest.  It deliberately does not
aggregate or claim a public, sealed, or headline JevBench score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)


class InvalidTask(ValueError):
    pass


class Overlength(ValueError):
    pass


def task_items(task) -> list[tuple[str, object]]:
    """Return options in the task's declared label order, never mapping order."""
    labels = list(task.labels)
    if not labels or len(labels) > len(LETTERS) or any(not isinstance(x, str) for x in labels):
        raise InvalidTask("labels must be 1-26 strings")
    if len(set(labels)) != len(labels):
        raise InvalidTask("labels must be unique")
    qtype = task.question.get("type")
    criteria = task.question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or set(criteria) != set(labels):
            raise InvalidTask("choice criteria keys must exactly match labels")
        return [(label, criteria[label]) for label in labels]
    if qtype == "noul":
        if set(labels) != {"no", "yes"} or not isinstance(criteria, dict):
            raise InvalidTask("noul requires no/yes labels and false/true criteria")
        folded = {str(key).casefold(): value for key, value in criteria.items()}
        if len(criteria) != 2 or set(folded) != {"false", "true"}:
            raise InvalidTask("noul criteria keys must be exactly false/true")
        return [(label, folded["true" if label == "yes" else "false"]) for label in labels]
    if qtype == "score":
        if labels != [str(i) for i in range(len(labels))]:
            raise InvalidTask("score labels must be ordered string indices")
        if not isinstance(criteria, list) or len(criteria) != len(labels):
            raise InvalidTask("score criteria must align one-to-one with labels")
        return list(zip(labels, criteria))
    raise InvalidTask(f"unsupported question type {qtype!r}")


def render_prompt(task) -> tuple[str, list[str]]:
    items = task_items(task)
    state = task.state if isinstance(task.state, str) else json.dumps(
        task.state, ensure_ascii=False, indent=1
    )
    instructions = task.question.get("instructions") or ""
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    lines = [state.rstrip(), "", instructions.rstrip(), "", "Options:"]
    lines.extend(f"{LETTERS[i]}. {description}" for i, (_, description) in enumerate(items))
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    return "\n".join(lines), [label for label, _ in items]


def validate_native_probs(probs: dict, labels: list[str]) -> dict[str, float]:
    if not isinstance(probs, dict) or list(probs) != labels:
        raise ValueError("probability keys/order must exactly match task labels")
    clean = {}
    for label, value in probs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probability for {label!r} is not numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"probability for {label!r} is not finite in [0,1]")
        clean[label] = value
    if not math.isclose(sum(clean.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("probabilities do not sum to one")
    return clean


def probabilities_from_option_scores(labels: list[str], scores: list[float]) -> dict[str, float]:
    """Stable softmax with an exact final residual for JSON-safe native probabilities."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and aligned")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("option scores must be finite")
    peak = max(float(score) for score in scores)
    weights = [math.exp(float(score) - peak) for score in scores]
    total = sum(weights)
    values = [weight / total for weight in weights]
    if len(values) > 1:
        values[-1] = 1.0 - sum(values[:-1])
    return validate_native_probs(dict(zip(labels, values)), labels)


class GemmaLoraAdapter:
    name = "typical_gemma_lora"
    cost_basis = "local_gpu_no_provider_tariff"
    price_input_per_m = None
    price_output_per_m = None

    def __init__(self, model: str, model_revision: str, adapter: str | None,
                 adapter_revision: str | None, max_context: int, dtype: str,
                 device: str = "cuda", attn_implementation: str = "sdpa"):
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

    @property
    def model_name(self) -> str:
        return self.model + (f"+{self.adapter}" if self.adapter else "")

    def load(self):
        if self._model is not None:
            return self
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from scripts.train_gemma_decision_lora import letter_token_groups

        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if self.dtype not in dtypes:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, revision=self.model_revision, local_files_only=Path(self.model).exists()
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self.model, revision=self.model_revision, dtype=dtypes[self.dtype],
            local_files_only=Path(self.model).exists(), attn_implementation=self.attn_implementation,
        )
        advertised = getattr(getattr(model.config, "text_config", model.config),
                             "max_position_embeddings", None)
        if not isinstance(advertised, int) or self.max_context > advertised:
            raise ValueError(
                f"configured max_context={self.max_context} exceeds or cannot verify model limit {advertised!r}"
            )
        if self.adapter:
            model = PeftModel.from_pretrained(
                model, self.adapter, revision=self.adapter_revision,
                local_files_only=Path(self.adapter).exists(),
            )
        model.config.use_cache = False
        self._model = model.to(self.device).eval()
        self._groups = letter_token_groups(self._tokenizer)
        self.load_s = time.perf_counter() - started
        self.model_context = advertised
        return self

    def build_request(self, task) -> dict:
        return {"state": task.state, "questions": {"decision": task.question},
                "labels": list(task.labels)}

    def _input_ids(self, prompt: str):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self._tokenizer(text, add_special_tokens=False, return_tensors="pt",
                              truncation=False)["input_ids"]
        tokens = int(ids.shape[-1])
        if tokens > self.max_context:
            raise Overlength(f"prompt has {tokens} tokens; configured maximum is {self.max_context}")
        return ids, tokens

    def _predict(self, task) -> tuple[dict[str, float], int]:
        import torch

        prompt, labels = render_prompt(task)
        ids, tokens = self._input_ids(prompt)
        with torch.inference_mode():
            output = self._model(input_ids=ids.to(self.device), use_cache=False, logits_to_keep=1)
            logits = output.logits[0, -1].float()
            scores = [float(torch.logsumexp(logits[group.to(self.device)], dim=0).item())
                      for group in self._groups[:len(labels)]]
        return probabilities_from_option_scores(labels, scores), tokens

    def run(self, task):
        from jevbench.adapters.base import DecisionResult

        body = self.build_request(task)
        result = DecisionResult(adapter=self.name, ok=False, probs_source="native",
                                model=self.model_name, request_body=body)
        try:
            task_items(task)
            self.load()
            probs, tokens = self._predict(task)
        except (InvalidTask, Overlength) as error:
            result.status = 422
            result.error = f"{type(error).__name__}: {error}"
            result.raw = {"runtime": {"max_context": self.max_context, "truncated": False}}
            return result
        except Exception as error:  # a model/runtime failure must remain a failed attempt
            result.error = f"{type(error).__name__}: {str(error)[:300]}"
            return result
        result.probs = validate_native_probs(probs, list(task.labels))
        result.usage = {"input_tokens": tokens, "output_tokens": 0}
        result.raw = {"runtime": {"input_tokens": tokens, "max_context": self.max_context,
                                  "truncated": False, "probability_origin": "native-option-logits"}}
        result.status = 200
        result.ok = True
        return result

    def reserve_estimate(self, task) -> float:
        return 0.0


def _artifact_hash(value: str | None, revision: str | None) -> dict | None:
    if value is None:
        return None
    path = Path(value)
    digest = hashlib.sha256()
    if path.is_file():
        paths = [path]
        root = path.parent
        kind = "file"
    elif path.is_dir():
        paths = sorted(item for item in path.rglob("*") if item.is_file())
        root = path
        kind = "tree"
    else:
        blob = json.dumps({"source": value, "revision": revision}, sort_keys=True).encode()
        return {"sha256": hashlib.sha256(blob).hexdigest(), "kind": "pinned-identity"}
    for item in paths:
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    return {"sha256": digest.hexdigest(), "kind": kind}


def _git_commit(path: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jevbench-dir", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True,
                        help="explicit input-token cap; prompts over it fail with status 422")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.adapter and not Path(args.adapter).exists() and not args.adapter_revision:
        parser.error("a remote --adapter requires --adapter-revision")

    sys.path.insert(0, str(args.jevbench_dir.resolve()))
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.tasks import dataset_hash, load_jsonl, sha256_file

    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tasks = []
    inputs = []
    for path in args.task_file:
        loaded = load_jsonl(str(path))
        inputs.append({"path": str(path), "sha256": sha256_file(str(path)),
                       "count": len(loaded), "dataset_hash": dataset_hash(loaded)})
        tasks.extend(loaded)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("task ids must be unique across input files")

    args.out.mkdir(parents=True, exist_ok=False)
    adapter = GemmaLoraAdapter(
        args.model, args.model_revision, args.adapter, args.adapter_revision,
        args.max_context, args.dtype, args.device, args.attn_implementation,
    )
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    records = Runner(adapter, Ledger(args.out / "ledger.jsonl"), args.out / "raw",
                     default_reserve_usd=0.0).run_all(
                         tasks, results_path=args.out / "results.jsonl"
                     )
    statuses = {}
    status_codes = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        code = str(record["status_code"])
        status_codes[code] = status_codes.get(code, 0) + 1
    manifest = {
        "adapter": adapter.name,
        "model": args.model,
        "model_revision": args.model_revision,
        "model_artifact": _artifact_hash(args.model, args.model_revision),
        "lora_adapter": args.adapter,
        "lora_revision": args.adapter_revision,
        "lora_artifact": _artifact_hash(args.adapter, args.adapter_revision),
        "harness_commit": _git_commit(args.jevbench_dir),
        "task_inputs": inputs,
        "selected_dataset_hash": dataset_hash(tasks),
        "max_context": args.max_context,
        "model_context": getattr(adapter, "model_context", None),
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "counts": {"planned": len(tasks), "attempted": len(records),
                   "unattempted": len(tasks) - len(records),
                   "valid": sum(bool(record["valid"]) for record in records),
                   "strict_valid": sum(bool(record["strict_valid"]) for record in records)},
        "statuses": statuses,
        "status_codes": status_codes,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "load_s": getattr(adapter, "load_s", None),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "score_scope": "per-item harness evidence only; no sealed or headline score computed",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
