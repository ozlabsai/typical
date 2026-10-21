#!/usr/bin/env python3
"""R053 stage 2: soft-label node questions with the benchmark's frontier teachers.

Messages, schema, fence stripping, validation, the one corrective retry and probability
normalization all come from system_one_adapter itself (prompted probability mode, i.e.
`structured_outputs=False`); only the transport is ours: OpenRouter (default; the exact WorkflowEvals
label teachers, reasoning effort high, key from the repo .env) or the `claude -p` CLI (`--backend cli`).
Every raw response is cached under .context/r053-data/cache, so reruns are free and resumable.

uv run --project .context/research_sources/WorkflowEvals python scripts/r053_teacher_label.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WE = ROOT / ".context" / "research_sources" / "WorkflowEvals"
DATA = ROOT / ".context" / "r053-data"
CACHE = DATA / "cache"
WORKFLOWS = ("security_incidents", "customer_service", "agent_trace_observability", "invoice_processing")
TEACHERS = {"openrouter": ("openai/gpt-6-astra", "anthropic/claude-fable-5.1"),
            "cli": ("claude-fable-5-1", "claude-opus-5-5")}
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"


class BudgetExceeded(RuntimeError):
    pass


LEDGER = {"usd": 0.0, "max_usd": 25.0, "calls": 0, "cached": 0, "wall_s": 0.0}
STATS = defaultdict(Counter)  # model -> calls/cost/tokens, cached calls included (labels' true cost)
_LOCK = threading.Lock()


def usage_of(raw: dict) -> dict:
    """Normalize a raw CLI or OpenRouter response."""
    if "choices" in raw:
        u = raw.get("usage") or {}
        return {"text": raw["choices"][0]["message"].get("content") or "", "usd": float(u.get("cost") or 0),
                "input_tokens": u.get("prompt_tokens") or 0, "output_tokens": u.get("completion_tokens") or 0,
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}
    u = raw.get("usage") or {}
    text = json.dumps(raw["structured_output"]) if raw.get("structured_output") is not None else raw.get("result") or ""
    return {"text": text, "usd": float(raw.get("total_cost_usd") or 0),
            "input_tokens": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                        "cache_read_input_tokens")),
            "output_tokens": u.get("output_tokens") or 0,
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0}


def _cached(key_parts: list, model: str, fetch) -> dict:
    """Disk cache + budget hard-stop around one transport call; returns the raw response."""
    path = CACHE / f"{hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()}.json"
    if path.exists():
        with _LOCK:
            LEDGER["cached"] += 1
        return json.loads(path.read_text())
    with _LOCK:
        if LEDGER["usd"] >= LEDGER["max_usd"]:
            raise BudgetExceeded(f"spent ${LEDGER['usd']:.2f} >= --max-usd {LEDGER['max_usd']}")
    started = time.perf_counter()
    raw = fetch()
    raw["wall_s"] = time.perf_counter() - started
    usd = usage_of(raw)["usd"]
    with _LOCK:
        LEDGER["usd"] += usd
        LEDGER["calls"] += 1
        LEDGER["wall_s"] += raw["wall_s"]
        print(f"  [{model}] ${usd:.3f} {raw['wall_s']:.0f}s | total ${LEDGER['usd']:.2f} over "
              f"{LEDGER['calls']} calls", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(raw))
    tmp.replace(path)
    return raw


def claude(model: str, system: str, user: str, effort: str = "high", schema: dict | None = None) -> dict:
    """Cached `claude -p` call; returns the CLI JSON (text in `.result`, `.structured_output` with a schema)."""
    cmd = ["claude", "-p", "--model", model, "--effort", effort, "--system-prompt", system,
           "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--no-session-persistence", "--output-format", "json"]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema)]

    def fetch():
        error = ""
        for _ in range(2):  # ponytail: one transport retry; schema retries belong to the adapter
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=3600, cwd="/tmp")
            try:
                out = json.loads(proc.stdout)
            except json.JSONDecodeError:
                error = (proc.stderr or proc.stdout)[-500:]
                continue
            if not out.get("is_error"):
                return out
            error = str(out.get("result"))[-500:]
        raise RuntimeError(f"claude CLI failed for {model}: {error}")

    return _cached([model, effort, system, user] + ([schema] if schema is not None else []), model, fetch)


def _api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("OPENROUTER_API_KEY missing from environment and .env")


def openrouter(model: str, messages: list[dict], effort: str = "high", schema: dict | None = None) -> dict:
    """Cached OpenRouter chat completion; cost comes from the response usage. A schema = the gold run's strict
    json_schema output (WorkflowEvals core/clients.py: name "evaluation", strict)."""
    import httpx

    body = {"model": model, "messages": messages, "reasoning": {"effort": effort}, "usage": {"include": True}}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "evaluation", "strict": True, "schema": schema}}

    def fetch():
        for attempt in range(4):
            try:
                r = httpx.post(OPENROUTER, json=body, timeout=1800,
                               headers={"Authorization": f"Bearer {_api_key()}"})
            except httpx.TransportError as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                out = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                if r.status_code == 200 and out.get("choices") and not out.get("error"):
                    if out["choices"][0].get("finish_reason") == "length":
                        raise RuntimeError(f"{model}: response truncated (finish_reason=length)")
                    return out
                error = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 402, 403, 404):
                    break  # not transient
            time.sleep(5 * 2 ** attempt)
        raise RuntimeError(f"OpenRouter failed for {model}: {error}")

    return _cached(["openrouter", model, effort, messages] + ([schema] if schema is not None else []), model, fetch)


class Provider:
    """system_one_adapter SyncProvider over OpenRouter or the Claude Code CLI."""

    def __init__(self, backend: str, model: str):
        self.backend, self.model_name = backend, model

    def request(self, messages, *, schema, structured):
        from system_one_adapter.providers import ProviderResult

        if self.backend == "openrouter":
            raw = openrouter(self.model_name, [{"role": m.role, "content": m.content} for m in messages],
                             schema=schema if structured else None)
        else:
            system, first, *rest = messages
            # ponytail: the CLI takes one user turn, so a corrective retry is replayed as a transcript.
            user = first.content + "".join(f"\n\n[{m.role}]\n{m.content}" for m in rest)
            raw = claude(self.model_name, system.content, user, schema=schema if structured else None)
        u = usage_of(raw)
        with _LOCK:
            STATS[self.model_name].update({"calls": 1, "usd": u["usd"], "wall_s": raw.get("wall_s", 0),
                                           **{k: u[k] for k in ("input_tokens", "output_tokens", "reasoning_tokens")}})
        return ProviderResult(text=u["text"], input_tokens=u["input_tokens"], output_tokens=u["output_tokens"])


def wire_probs(question: dict, answer) -> dict[str, float]:
    """Adapter answer -> {label: p} in the question's own label order."""
    if question["type"] == "noul":
        return {"true": answer.noul, "false": 1 - answer.noul}
    if question["type"] == "score":
        return {str(i): float(answer.probabilities[i]) for i in range(len(question["criteria"]))}
    return {k: float(answer.probabilities[k]) for k in question["criteria"]}


def ask(client, backend: str, model: str, state, questions: dict) -> dict[str, dict[str, float]]:
    response = client.system_one(state=state, questions=questions, model=Provider(backend, model))
    return {qid: wire_probs(questions[qid], response.answers[qid]) for qid in questions}


def main(argv=None) -> int:
    sys.path.insert(0, str(WE))
    from system_one_adapter import SystemOneAdapterClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", nargs="+", default=list(WORKFLOWS))
    parser.add_argument("--backend", choices=tuple(TEACHERS), default="openrouter")
    parser.add_argument("--teachers", nargs="+", help="default: the backend's two teachers")
    parser.add_argument("--question-mode", choices=("one", "all"), default="one",
                        help="one = one request per question, as the benchmark labels were made")
    parser.add_argument("--noul-sample", type=int, help="only label N noul questions (spread over workflows)")
    parser.add_argument("--out-name", default="labels.jsonl")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-usd", type=float, default=25.0)
    args = parser.parse_args(argv)
    LEDGER["max_usd"] = args.max_usd
    args.teachers = args.teachers or list(TEACHERS[args.backend])
    client = SystemOneAdapterClient(structured_outputs=False, llm_answer_mode="probabilities",
                                    normalize_probabilities=True, n_retry_malformed_structure=1)

    # Work unit = (workflow, node row, question subset); each is sent to every teacher.
    units = []
    for wf in args.workflows:
        nodes = [json.loads(line) for line in (DATA / wf / "nodes.jsonl").read_text().splitlines()]
        for node in nodes:
            qs = node["questions"]
            if args.noul_sample:
                qs = {q: v for q, v in qs.items() if v["type"] == "noul"}
            if not qs:
                continue
            groups = [{q: v} for q, v in qs.items()] if args.question_mode == "one" else [qs]
            units += [(wf, node, g) for g in groups]
    if args.noul_sample:  # deterministic hash order, then round-robin over workflows (stable sort on rank)
        units.sort(key=lambda u: hashlib.sha256(f"{u[1]['request_id']}/{next(iter(u[2]))}".encode()).hexdigest())
        rank = Counter()
        units = sorted(units, key=lambda u: rank.update([u[0]]) or rank[u[0]])[: args.noul_sample]

    def label(job):
        (wf, node, qs), teacher = job
        try:
            return wf, node, qs, teacher, ask(client, args.backend, teacher, node["state"], qs), None
        except Exception as exc:  # keep going (budget stops are instant); report failures
            return wf, node, qs, teacher, None, f"{type(exc).__name__}: {exc}"[:300]

    jobs = [(u, t) for u in units for t in args.teachers]
    print(f"{len(units)} requests x {len(args.teachers)} teachers = {len(jobs)} calls", flush=True)
    results, failures = {}, []
    with ThreadPoolExecutor(args.workers) as pool:
        for wf, node, qs, teacher, probs, err in pool.map(label, jobs):
            if err:
                failures.append({"workflow": wf, "request_id": node["request_id"], "node": node["node"],
                                 "teacher": teacher, "error": err})
                continue
            for qid, p in probs.items():
                results.setdefault((wf, node["request_id"], qid), {})[teacher] = p

    for wf in args.workflows:
        nodes = {n["request_id"]: n for n in map(json.loads, (DATA / wf / "nodes.jsonl").read_text().splitlines())}
        rows = []
        for (w, rid, qid), teachers in sorted(results.items()):
            if w != wf or set(teachers) != set(args.teachers):
                continue  # consensus needs every teacher
            labels = list(next(iter(teachers.values())))
            consensus = {k: sum(t[k] for t in teachers.values()) / len(teachers) for k in labels}
            rows.append({"case_id": nodes[rid]["case_id"], "request_id": rid, "node": nodes[rid]["node"],
                         "qid": qid, "question": nodes[rid]["questions"][qid],
                         "question_mode": args.question_mode, "teachers": teachers, "consensus": consensus})
        (DATA / wf / args.out_name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{wf}: {len(rows)} labeled questions -> {DATA / wf / args.out_name}")
    labeled = Counter(t for teachers in results.values() for t in teachers)
    per_teacher = {t: {"calls": STATS[t]["calls"], "labeled_questions": labeled[t],
                       **{f"{k}_per_question": round(STATS[t][k] / max(labeled[t], 1), 4)
                          for k in ("usd", "input_tokens", "output_tokens", "reasoning_tokens")},
                       "wall_s_per_call": round(STATS[t]["wall_s"] / max(STATS[t]["calls"], 1), 1)}
                   for t in args.teachers}
    summary = {"backend": args.backend, "question_mode": args.question_mode, "new_calls": LEDGER["calls"],
               "cached_calls": LEDGER["cached"], "new_cost_usd": round(LEDGER["usd"], 4),
               "per_teacher": per_teacher, "failures": failures}
    (DATA / f"{Path(args.out_name).stem}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
