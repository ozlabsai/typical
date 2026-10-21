#!/usr/bin/env python3
"""R053 stage 1: generate novel raw WorkflowEvals cases and expand them into node requests.

The generator (Claude Opus 5.5 via OpenRouter; `--backend cli` for the Claude Code CLI) sees only a structural skeleton derived from the
benchmark schema (keys, types, numeric ranges, lowercase structural enums; all free text replaced
by placeholders) plus hand-written format notes. Generated cases then pass a contamination filter
against every benchmark case of the workflow (13-word-gram overlap, ignoring boilerplate 13-grams found in
>=5% of that workflow's benchmark cases; 5-shingle Jaccard > 0.3; reused ID/hostname/name strings, where
IDs/names found in >=5% of benchmark cases and generic field names (JSON keys, `key:` record labels,
snake/dotted identifiers) are vocabulary, not identity; R056 fix) and are run through WorkflowEvals' own adapters with stub answers
(several answer variants) so every reachable node request {state, questions} is recorded.

uv run --project .context/research_sources/WorkflowEvals python scripts/r053_generate_cases.py --n 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WE = ROOT / ".context" / "research_sources" / "WorkflowEvals"
DATA = ROOT / ".context" / "r053-data"
GENERATOR = {"openrouter": "anthropic/claude-opus-5.5", "cli": "claude-opus-5-5",
             "vllm": "google/gemma-4-12B-it", "codex": "gpt-5.6-terra"}  # vllm: $R056_VLLM_URL (OpenAI-compatible)
PER_CALL = {"security_incidents": 3, "customer_service": 3, "agent_trace_observability": 1,
            "invoice_processing": 1}  # several cases per call where the schema is small
NGRAM, SHINGLE, MAX_JACCARD = 13, 5, 0.3
BOILERPLATE_FRAC = 0.05  # strings in >=5% of a workflow's benchmark cases are its template, not case content
VARIANTS = 5  # stub answer variants per case: high, low, three random

# ---------------------------------------------------------------- contamination filter

def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def leaves(value) -> list[tuple[str, str]]:
    """(key, string) for every string leaf of a JSON value."""
    out = []
    stack = [("", value)]
    while stack:
        key, v = stack.pop()
        if isinstance(v, dict):
            stack += list(v.items())
        elif isinstance(v, list):
            stack += [(key, x) for x in v]
        elif isinstance(v, str):
            out.append((key, v))
    return out


def grams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


NAME_KEY = re.compile(r"name|holder|vendor|company|entit|merchant|principal|author|user|email|inbox|host|from|^to$|^cc$")
ID_TOKEN = re.compile(r"[\w.@\\/:-]+")
DATE_LIKE = re.compile(r"[\d.:/tz+-]+")
# number+unit/ordinal/word compounds and month-name dates are vocabulary, not identifiers
GENERIC = re.compile(r"\d+[a-z]{1,3}|\d+(-[a-z]+)+|[a-z]+-\d{1,2}|\d{1,2}-[a-z]{3}-\d{2,4}|[\d.]+/[a-z]+|t\d{4}(\.\d{3})?"
                     # R056: field/event names (card_last4, address_line1, user.session.start) and cloud regions
                     r"|[a-z]+(?:_[a-z]+\d{0,2})+|[a-z]{2}-[a-z]+-\d"
                     r"|[a-z]+\d{0,2}\.(?:exe|dll|sys|ps1|bat|cmd|msi|vbs|js|py|sh)")  # stock binaries (rundll32.exe)
TLDS = {"com", "net", "org", "io", "local", "corp", "internal", "lan", "ai", "co", "us", "uk", "edu", "gov", "dev", "app"}
JSON_KEY = re.compile(r'"([A-Za-z_][\w.-]*)"\s*:|^\s*([A-Za-z_][\w.-]*):\s', re.M)


def id_tokens(text: str) -> set[str]:
    """Identifier-ish tokens: letter+digit mixes, emails, dotted hostnames, initial.surname handles."""
    out = set()
    for tok in ID_TOKEN.findall(text.lower()):
        tok = tok.strip(".:/\\-")
        if len(tok) < 8 or DATE_LIKE.fullmatch(tok) or GENERIC.fullmatch(tok):
            continue  # dates, versions, units, MITRE technique ids: shared vocabulary, not case identity
        if (re.search(r"[a-z]", tok) and re.search(r"\d", tok)) or "@" in tok \
                or (tok.count(".") >= 2 and (re.search(r"\d", tok) or tok.rsplit(".", 1)[-1] in TLDS)) \
                or re.fullmatch(r"[a-z]\.[a-z]{3,}", tok):  # R056: dotted event names (user.session.start) are not hosts
            out.add(tok)
    return out


def contamination_index(cases: list) -> dict:
    """Everything the filter compares against, from benchmark inputs (+ metadata)."""
    index = {"grams": set(), "shingles": [], "ids": set(), "names": set()}
    enum_like, df, id_df, name_df, fields = Counter(), Counter(), Counter(), Counter(), set()
    for case in cases:
        case_shingles, case_grams, case_ids, case_names = set(), set(), set(), set()
        stack = [case]
        while stack:  # every dict key anywhere is a field name
            v = stack.pop()
            if isinstance(v, dict):
                fields |= {k.lower() for k in v}
                stack += list(v.values())
            elif isinstance(v, list):
                stack += v
        for key, text in leaves(case):
            toks = words(text)
            case_grams |= grams(toks, NGRAM)
            case_shingles |= grams(toks, SHINGLE)
            case_ids |= id_tokens(text)
            fields |= {(a or b).lower() for a, b in JSON_KEY.findall(text)}  # keys inside JSON strings / records
            if NAME_KEY.search(key.lower()) and 6 <= len(text) <= 60 and not DATE_LIKE.fullmatch(text.lower()):
                case_names.add(text.strip().lower())
            enum_like[text] += 1
        index["shingles"].append(case_shingles)
        df.update(case_grams)
        id_df.update(case_ids)
        name_df.update(case_names)
    # n-grams/ids/names present in >=5% of the workflow's cases are its own template/format boilerplate (e.g. the
    # charter headers WE's trace parser keys on), not case content; every rarer one rejects.
    template = max(2, BOILERPLATE_FRAC * len(cases))
    index["grams"] = {g for g, c in df.items() if c < template}
    index["template"] = set(df) - index["grams"]
    index["ids"] = {t for t, c in id_df.items() if c < template and t not in fields}
    index["names"] = {t for t, c in name_df.items() if c < template and t not in fields}
    # Frequent lowercase snake values (span names, roles) are structure, not identity.
    index["names"] -= {v.lower() for v, c in enum_like.items() if c >= 3 and re.fullmatch(r"[a-z_.]+", v)}
    return index


def contaminated(case, index: dict) -> str | None:
    """Return the rejection reason, or None if the case is clean."""
    shingles = set()
    for key, text in leaves(case):
        toks = words(text)
        if hit := grams(toks, NGRAM) & index["grams"]:
            return f"ngram{NGRAM}:" + " ".join(sorted(hit)[0])
        shingles |= grams(toks, SHINGLE)
        if text.strip().lower() in index["names"]:
            return f"name:{text.strip()[:40]}"
        if hit := id_tokens(text) & index["ids"]:
            return f"id:{sorted(hit)[0]}"
    if shingles and any(len(shingles & b) / len(shingles | b) > MAX_JACCARD for b in index["shingles"] if b):
        return "jaccard"
    return None

# ---------------------------------------------------------------- structural skeleton

ENUM = re.compile(r"[a-z][a-z_.]{0,30}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*")


def _disc(e: dict):
    sd = e.get("span_data")
    if isinstance(sd, dict):
        return (sd.get("type"), sd.get("name") if sd.get("type") == "custom" else None)
    return tuple(str(e.get(k)) for k in ("type", "role", "speaker", "channel"))


def skeleton(values: list, key: str = ""):
    """Merge every benchmark value at one path into a content-free structural description."""
    present = [v for v in values if v is not None]
    null = " or null" if len(present) < len(values) else ""
    if not present:
        return None
    if all(isinstance(v, dict) for v in present):
        seen = Counter(k for v in present for k in v)
        # Rare keys mean a free-form object (tool parameters, structured_data): their names are content.
        keys = [k for k, c in seen.items() if c >= 0.5 * len(present)]
        out = {k: skeleton([v.get(k) for v in present], k) for k in keys}
        if len(keys) < len(seen):
            out["..."] = "<plus free-form keys of your own>"
        return out
    if all(isinstance(v, list) for v in present):
        items = [x for v in present for x in v]
        lo, hi = min(map(len, present)), max(map(len, present))
        if items and all(isinstance(x, dict) for x in items):
            groups = defaultdict(list)
            for x in items:
                groups[_disc(x)].append(x)
            body = [skeleton(g, key) for g in list(groups.values())[:8]]
        else:
            body = [skeleton(items, key)] if items else []
        return body + [f"<list of {lo}..{hi} items{null}>"]
    if all(isinstance(v, bool) for v in present):
        return f"<bool{null}>"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        kind = "int" if all(isinstance(v, int) for v in present) else "number"
        return f"<{kind} {min(present)}..{max(present)}{null}>"
    if all(isinstance(v, str) for v in present):
        counts = Counter(present)
        # Enumerate only structural codes; names (workflow/agent/vendor names...) are content.
        if len(counts) <= 12 and not NAME_KEY.search(key.lower()) \
                and all(ENUM.fullmatch(v) and c >= 3 for v, c in counts.items()):
            return f"<one of: {' | '.join(sorted(counts))}{null}>"
        if all(DATE.fullmatch(v) for v in present):
            return f"<date YYYY-MM-DD{null}>"
        if all(DATETIME.fullmatch(v) for v in present):
            return f"<ISO-8601 datetime{null}>"
        n = int(statistics.median(map(len, present)))
        lines = " multi-line" if any("\n" in v for v in present) else ""
        return f"<free text ~{n} chars{lines}{null}>"
    return f"<mixed JSON value{null}>"

# ---------------------------------------------------------------- per-workflow notes and seeds

NOTES = {
    "security_incidents": """A raw security alert plus SOC context records. Fields WorkflowEvals code reads:
alert prefix (detector) and context.asset.environment/tier.
- alert: multi-line string. Line 1 is "<DETECTOR>-<4 digits> | <detection title>" with DETECTOR one of
  EDR (endpoint), IDP (identity), NET (network), MAIL (email). Then "asset: <HOST>    principal: <account>
  observed: <YYYY-MM-DD HH:MM UTC>", "events correlated: <n>", a blank line, then indicator lines such as
  "process: ...", "file: ...", "registry: ...", "source events: <event ids>" (domain-appropriate indicators,
  e.g. sign-in, URL, sender, destination lines are fine for IDP/NET/MAIL).
- context.asset: environment (prod|staging|dev), tier (0 domain controller/critical, 1 server, 2 workstation), type.
- context.records: 9-21 strings. Each record is "<system>.<object>" on the first line followed by indented
  "  key: value" lines - inventory/log exports such as an EDR host record, directory computer/user objects,
  IdP user/factor/system-log entries, CMDB CI, ITSM incidents and change requests, HR time-off, badge events,
  firewall rules, scheduled-task and managed-app inventories. Most records are routine background; the
  records that matter (tickets, change windows, authorizations, sign-in logs) decide whether the alert is
  explained, and they may cover the same actor/asset/window or only something adjacent.""",
    "customer_service": """One customer-service turn: the chat so far (ending with the customer's latest message),
the customer profile, the account records on file, and the assistant's state before this turn.
- transcript alternates customer/assistant and ENDS with a customer message; turn = number of earlier
  customer messages (0 for a single-message transcript). 1, 3 or 5 messages are typical.
- account.transaction/refund/order are null unless that record exists; outstanding_fee_usd > 0 only with a fee.
- state.pending is the proposal the assistant made in its previous message and is awaiting a reply to
  (issue_refund | cancel_subscription | freeze_card | open_dispute | hardship_referral) or null.
- Also include a top-level "theme": one of access | billing | conduct | fraud | refunds | retention.""",
    "agent_trace_observability": """One exported agent run in the OpenAI Agents SDK trace format (trace + spans).
WorkflowEvals parses it as follows, so these must be consistent:
- The agent span lists tools; each tool has type "function", function {name "Service.operation",
  description, parameters JSON schema} and annotations {readOnlyHint, destructiveHint, idempotentHint,
  openWorldHint}.
- Generation spans carry input/output chat messages. The FIRST generation's input starts with a system
  message whose content is the agent charter: a purpose paragraph, then blank-line separated sections
  starting with exactly these headers: "Your policy:" (lines "- rule"), "What your role is entitled to:"
  (lines "- Service.operation: entitled|not entitled, within <limits>. [You must have the customer's explicit
  confirmation in this conversation before using it.] [Requires supervisor approval.]"),
  "How much you may do on your own:" and "When you hand off, it goes to:". Later generations repeat the history.
- Assistant messages with tool_calls [{id, type "function", function {name, arguments (JSON string)}}] are
  followed by role "tool" messages {tool_call_id, content}. Each call has a matching function span
  (span_data name = tool name, input, output) with span "error" null or {message, data {code}}.
- Custom span "intent_classification" data {category: action|research, confidence}; custom span
  "user_feedback" data {signal, csat (1-5 or null), comment, trailing_message, reopened_within_24h}.
  Optional guardrail and handoff spans. Timestamps are ISO-8601 and spans are ordered by started_at.""",
    "invoice_processing": """An accounts-payable review packet (schema_version "1.0") validated by the pydantic model whose
JSON schema is given below: company, invoice (fields, line_items, raw_text as the vendor's printed
document), purchase_order, contract, vendor master record (payment profile, bank change log, history
summary, known contacts), prior_invoices, communications (emails/portal messages), delivery_evidence and
approvals. Numbers must be internally coherent with the scenario (line amounts = qty x unit price unless
the scenario is about a discrepancy; totals as printed). Dates are YYYY-MM-DD, timestamps ISO-8601.""",
}

DIFFICULTY = [
    "clear-cut: the evidence points one way",
    "clear-cut but with distracting routine noise",
    "genuinely ambiguous: evidence partially supports both readings",
    "genuinely ambiguous: a key fact is missing, stale or only implied",
    "mixed signals: two pieces of evidence point in different directions",
]
INDUSTRIES = [
    "regional hospital network", "logistics and freight", "university", "municipal government", "biotech lab",
    "retail chain", "law firm", "manufacturing plant", "insurance carrier", "media streaming startup",
    "credit union", "airline", "energy utility", "SaaS company", "architecture firm", "food distributor",
    "telecom operator", "nonprofit foundation", "hotel group", "construction contractor",
]
TOPICS = {
    "security_incidents": [
        "credential dumping or LSASS access", "remote admin tool (PsExec/WMI) use", "suspicious scheduled task",
        "impossible-travel or anomalous sign-in", "MFA push fatigue", "OAuth consent grant to unknown app",
        "inbox forwarding rule creation", "phishing message delivered to several mailboxes",
        "large outbound transfer to cloud storage", "DNS tunneling or beaconing", "ransomware precursor activity",
        "new local admin account", "authorized penetration test or red-team activity",
        "IT maintenance script flagged by EDR", "service account behaving unusually", "token replay from new device",
    ],
    "customer_service": [
        "unrecognized card charge", "refund for a damaged or missing item", "subscription cancellation",
        "late or overdraft fee dispute", "login / 2FA access problem", "order not delivered",
        "card frozen while travelling", "double charge", "rude or threatening customer", "hardship / cannot pay",
        "reply to a pending refund or cancellation proposal", "complaint about an earlier assistant promise",
    ],
    "agent_trace_observability": [
        "airline rebooking agent", "telecom plan-change agent", "insurance claim intake agent",
        "bank card services agent", "e-commerce returns agent", "IT helpdesk agent",
        "clinic appointment scheduling agent", "utility billing agent", "HR benefits agent", "travel booking agent",
    ],
    "invoice_processing": [
        "clean three-way match", "invoice lines only partially match the PO", "bank details change request by email",
        "possible duplicate of a prior invoice", "invoice exceeds the PO authorized amount",
        "unapproved add-on or out-of-scope line", "wrong bill-to entity or department",
        "goods billed but receipt missing or partial", "sender address on a lookalike domain",
        "credit note or statement instead of invoice", "price variance vs contract rate", "missing approval",
    ],
}
OUTCOMES = {
    "agent_trace_observability": [
        "the task fully succeeds", "partial success: some requests done, others not",
        "silent failure: a tool errors but the assistant claims success", "overt failure the assistant admits",
        "a destructive or conditioned action taken without the required confirmation",
        "a conditioned action taken correctly after explicit confirmation",
        "success, but the user is unhappy (expectation gap)", "the request itself is vague or contradictory",
    ],
}


def seed_for(wf: str, i: int, tag: str = "r053") -> dict:
    rng = random.Random(f"{tag}/{wf}/{i}")
    seed = {"industry": rng.choice(INDUSTRIES), "topic": rng.choice(TOPICS[wf]), "difficulty": rng.choice(DIFFICULTY)}
    if wf in OUTCOMES:
        seed["outcome"] = rng.choice(OUTCOMES[wf])
        seed["user_reaction"] = rng.choice(["none recorded", "positive", "negative", "neutral", "reopened within 24h"])
    if wf == "security_incidents":
        seed["detector"] = rng.choice(["EDR", "IDP", "NET", "MAIL"])
    return seed


SYSTEM = ("You write synthetic but realistic test fixtures for workflow-evaluation software. "
          "You reply with exactly one JSON object and nothing else - no prose, no Markdown fences.")


def prompt(wf: str, skel, schema, seeds: list[dict], size: int | None = None) -> str:
    k = len(seeds)
    parts = [
        f"Write ONE new raw input case for the `{wf}` workflow." if k == 1 else
        f"Write {k} new, mutually different raw input cases for the `{wf}` workflow.",
        "Structural skeleton (keys and types only; every <...> is a placeholder for you to fill; "
        "a list shows one representative element per element kind, then its observed length range):",
        json.dumps(skel, indent=1),
    ]
    if schema:
        parts += ["Pydantic JSON schema the case must validate against:", json.dumps(schema)]
    parts += [
        "Format notes:", NOTES[wf],
        "Scenario seed:" if k == 1 else "Scenario seeds (one per case, in order):",
        json.dumps(seeds[0] if k == 1 else seeds),
        "Requirements:\n- Realistic, specific and internally consistent, like a real export from production systems.\n"
        "- Invent every organization, person, product, merchant, host, domain, account and ID; do not reuse "
        "well-known sample names (Contoso, Northwind, Fabrikam, Acme ...).\n"
        "- Honour the difficulty: for ambiguous seeds, make the evidence genuinely support more than one "
        "reading (partial explanations, mixed customer signals, partial agent success, partially matching "
        "lines) - a careful reviewer should be unsure. For clear seeds, make it clear.\n"
        "- Never state the intended outcome, label or difficulty inside the case.\n"
        "- Same keys and nesting as the skeleton; keep size similar to the observed lengths.\n"
        + (f"- Size: about {size:,} characters of JSON per case (typical of real exports); records/messages "
           "carry realistic detail, not one-liners.\n" if size else "")
        + "- No annotation fields or comments that state a conclusion (e.g. 'note: suspicious', 'id: none'); "
        "let the raw facts speak.\n"
        "Return only the JSON object." if k == 1 else
        f'Return only one JSON object {{"cases": [...]}} holding the {k} case objects in seed order.',
    ]
    return "\n\n".join(parts)


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)

# ---------------------------------------------------------------- node expansion through WE code

class StubSession:
    """Stands in for core.session.Session: records each node request, answers with a fixed variant."""

    def __init__(self, variant: int, policy_ids: list[str]):
        self.variant, self.rng = variant, random.Random(variant)
        self.config = SimpleNamespace(model="stub", thinking=None)
        self.policy_ids, self.node, self.seen = policy_ids, "semantic", []

    def _value(self, wire: dict):
        if wire["type"] == "noul":
            return [0.9, 0.1][self.variant] if self.variant < 2 else self.rng.random()
        labels = list(wire["criteria"]) if wire["type"] == "choice" else [str(i) for i in range(len(wire["criteria"]))]
        if self.variant < 2:
            top = labels[0] if self.variant == 0 else labels[-1]
            return {k: 0.9 if k == top else 0.1 / (len(labels) - 1) for k in labels}
        raw = [self.rng.gammavariate(0.5, 1) + 1e-9 for _ in labels]
        return {k: r / sum(raw) for k, r in zip(labels, raw)}

    def system_one(self, _model, state, questions):
        from core.question_types import question_record
        from system_one_adapter._client import _convert_llm_value_to_typesafe_answer
        from system_one_adapter._schema import convert_question_collection_to_validated_api_question_models
        from typesafe_sdk import SystemOneResponse, Usage

        wire = {k: question_record(q) if hasattr(q, "model_dump") else q for k, q in questions.items()}
        self.seen.append({"node": self.node, "state": state, "questions": wire})
        prepared = convert_question_collection_to_validated_api_question_models(wire)
        answers = {k: _convert_llm_value_to_typesafe_answer(prepared[k], self._value(w), "probabilities",
                                                            should_normalize_probabilities=True)[0]
                   for k, w in wire.items()}
        return SystemOneResponse(model="stub", answers=answers, usage=Usage())


def strip_extra(raw: dict, model, tries: int = 50) -> None:
    """Drop keys the strict pydantic packet forbids (generators add stray fields); other errors still raise later."""
    from pydantic import ValidationError
    for _ in range(tries):
        try:
            model.model_validate(raw)
            return
        except ValidationError as exc:
            extra = [e["loc"] for e in exc.errors() if e["type"] == "extra_forbidden"]
            if not extra:
                return
            for loc in extra:
                node = raw
                for k in loc[:-1]:
                    node = node[k]
                node.pop(loc[-1], None)


def expand(wf: str, case_id: str, raw: dict, metadata: dict) -> list[dict]:
    """Run the case through WE's own adapter under every stub variant; return unique node requests."""
    from importlib import import_module

    from core.contract import fingerprint

    adapter = import_module(f"evals.{wf}.eval_adapter").Adapter(
        WE / "evals" / wf, SimpleNamespace(inputs={case_id: raw}, metadata={case_id: metadata}))
    policies = [p.policy_id for p in adapter.bundle.policies]
    nodes = {}
    for variant in range(VARIANTS):
        session = StubSession(variant, policies)
        adapter.execute(case_id, session)
        for req in session.seen:
            rid = fingerprint(req)[:16]
            nodes.setdefault(rid, {"case_id": case_id, "request_id": rid, **req})
    return list(nodes.values())


def open_model(model: str, user: str) -> dict:
    """Cached chat call to a self-hosted vLLM (R056 open-model generators), thinking off, cost 0."""
    import urllib.request

    import scripts.r053_teacher_label as r053

    def fetch():
        body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "max_tokens": 30000, "temperature": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(os.environ["R056_VLLM_URL"].rstrip("/") + "/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as resp:
            return json.loads(resp.read())

    return r053._cached([model, "vllm-gen", SYSTEM, user], model, fetch)


def codex_text(model: str, user: str, effort: str) -> str:
    """Cached `codex exec` (GPT family; subscription, $0 notional), free-form JSON answer."""
    import subprocess
    import tempfile

    import scripts.r053_teacher_label as r053

    def fetch():
        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run(["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                                   "--skip-git-repo-check", "-s", "read-only", "-C", d, "-m", model,
                                   "-c", f"model_reasoning_effort={effort}", "-o", f"{d}/out.txt", "-"],
                                  input=SYSTEM + "\n\n" + user, capture_output=True, text=True, timeout=3600)
            out = Path(d, "out.txt")
            if not out.exists() or not out.read_text().strip():
                raise RuntimeError(f"codex failed: {(proc.stderr or proc.stdout)[-300:]}")
            return {"result": out.read_text(), "total_cost_usd": 0.0}

    return r053._cached([model, "codex-gen", effort, SYSTEM, user], model, fetch)["result"]


def main(argv=None) -> int:
    sys.path.insert(0, str(WE))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from core.dataset import load_dataset
    from evals.invoice_processing.structured.state import Packet

    sys.path.insert(0, str(ROOT))
    import scripts.r053_teacher_label as r053
    from scripts.r053_teacher_label import LEDGER, claude, openrouter, usage_of

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", nargs="+", default=list(NOTES))
    parser.add_argument("--n", type=int, default=6, help="cases to generate per workflow")
    parser.add_argument("--start", type=int, default=0, help="first case index (new seeds and ids)")
    parser.add_argument("--backend", choices=tuple(GENERATOR), default="openrouter")
    parser.add_argument("--model", help="default: Claude Opus 5.5 id for the backend")
    parser.add_argument("--per-call", type=int, help="cases per generation call (default: per workflow)")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--max-usd", type=float, default=25.0)
    parser.add_argument("--out", default=str(DATA), help="output dir (cases/nodes per workflow, cache/, report)")
    parser.add_argument("--tag", default="r053", help="case-id prefix and seed namespace")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    data = Path(args.out)
    r053.CACHE = data / "cache"
    if args.out != str(DATA):  # cap counts every cached response under this output dir, across runs
        LEDGER["usd"] = sum(usage_of(json.loads(f.read_text()))["usd"] for f in r053.CACHE.glob("*.json"))
    LEDGER["max_usd"] = args.max_usd
    model = args.model or GENERATOR[args.backend]

    def llm(user: str) -> str:
        if args.backend == "vllm":
            return usage_of(open_model(model, user))["text"]
        if args.backend == "codex":
            return codex_text(model, user, args.effort)
        if args.backend == "openrouter":
            return usage_of(openrouter(model, [{"role": "system", "content": SYSTEM},
                                               {"role": "user", "content": user}], args.effort))["text"]
        return claude(model, SYSTEM, user, args.effort)["result"]

    from concurrent.futures import ThreadPoolExecutor

    report_path = data / ("generation_report.json" if args.out == str(DATA) else
                          f"generation_report.{args.workflows[0] if len(args.workflows) == 1 else 'all'}.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for wf in args.workflows:
        spent_before = LEDGER["usd"]
        bench = load_dataset(wf)
        records = [{**bench.metadata[c], **bench.inputs[c]} if wf == "customer_service" else bench.inputs[c]
                   for c in sorted(bench.inputs)]
        skel = skeleton(records)
        size = int(statistics.median(len(json.dumps(r)) for r in records)) if args.tag != "r053" else None
        if wf == "customer_service":  # metadata: keep only the structural theme; ids are ours
            skel = {k: v for k, v in skel.items() if k not in ("note", "case_id", "dialog_id")}
        schema = Packet.model_json_schema() if wf == "invoice_processing" else None
        index = contamination_index([bench.inputs[c] | {"_meta": bench.metadata[c]} for c in bench.inputs])
        ids = list(range(args.start, args.start + args.n))
        seeds = {i: seed_for(wf, i, args.tag) for i in ids}
        per_call = args.per_call or PER_CALL[wf]
        batches = [ids[j:j + per_call] for j in range(0, len(ids), per_call)]

        def generate(batch):
            try:
                out = parse_json(llm(prompt(wf, skel, schema, [seeds[i] for i in batch], size)))
                out = [out] if len(batch) == 1 else out["cases"]
                return dict(zip(batch, out)) if len(out) == len(batch) else {i: ValueError("wrong case count") for i in batch}
            except Exception as exc:  # unparseable output or transport failure: drop the cases
                return {i: exc for i in batch}

        with ThreadPoolExecutor(args.workers) as pool:
            generated = {i: c for part in pool.map(generate, batches) for i, c in part.items()}

        out = data / wf
        out.mkdir(parents=True, exist_ok=True)
        cases, nodes, stats, offending = [], [], Counter(), Counter()
        for i, raw in sorted(generated.items()):
            case_id = f"{args.tag}-{wf.split('_')[0]}-{i:04d}"
            if not isinstance(raw, dict):
                stats["unparseable"] += 1
                print(f"  {case_id}: generation failed: {str(raw)[:200]}")
                continue
            reason = contaminated(raw, index)
            if reason:
                stats[f"rejected_{reason.split(':')[0]}"] += 1
                offending[reason] += 1
                print(f"  {case_id}: rejected ({reason})")
                continue
            metadata = {}
            if wf == "customer_service":
                metadata = {"theme": raw.pop("theme", "billing"), "note": ""}
                raw |= {"case_id": case_id, "dialog_id": case_id}
            elif wf == "agent_trace_observability":
                raw["id"] = case_id
            elif wf == "invoice_processing":
                raw["case_id"] = case_id
                strip_extra(raw, Packet)
            try:
                case_nodes = expand(wf, case_id, raw, metadata)
            except Exception as exc:
                stats["invalid"] += 1
                print(f"  {case_id}: invalid under WE code: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            stats["kept"] += 1
            cases.append({"case_id": case_id, "workflow": wf, "generator": model, "seed": seeds[i],
                          "input": raw, "metadata": metadata})
            nodes += case_nodes
        # Merge by case_id so runs over other --start ranges accumulate.
        for name, rows in (("cases.jsonl", cases), ("nodes.jsonl", nodes)):
            old = [json.loads(x) for x in (out / name).read_text().splitlines()] if (out / name).exists() else []
            redo = {f"{args.tag}-{wf.split('_')[0]}-{i:04d}" for i in ids}
            rows[:0] = [r for r in old if r["case_id"] not in redo]
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        template_hits = Counter(" ".join(g) for c in generated.values() if isinstance(c, dict)
                                for _, t in leaves(c) for g in grams(words(t), NGRAM) & index["template"])
        prev = report.get(wf, {}).get("runs", [])
        run = {"generator": model, "ids": [ids[0], ids[-1]], **stats, "cost_usd": round(LEDGER["usd"] - spent_before, 4),
               "usd_per_kept_case": round((LEDGER["usd"] - spent_before) / max(stats["kept"], 1), 4),
               "rejection_reasons": dict(offending.most_common(10)),
               "top_ignored_boilerplate_ngrams": dict(template_hits.most_common(10))}
        report[wf] = {"runs": [r for r in prev if r["ids"] != run["ids"] or r["generator"] != model] + [run],
                      "kept_cases": len(cases), "nodes": len(nodes),
                      "questions": sum(len(n["questions"]) for n in nodes),
                      "node_counts": dict(Counter(re.sub(r"_\d+$", "_k", n["node"]) for n in nodes))}
        print(wf, json.dumps(run), flush=True)
    report["generation_cost_usd_this_run"] = round(LEDGER["usd"], 4)
    report_path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
