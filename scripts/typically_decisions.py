"""typically decisions (spike): a company's email archive -> the company's business DECISIONS (approve a deal, extend credit,
pick a vendor...), one row per decision: the case (the thread verbatim up to the decision), a recurring question, options, the
chosen option. Not inbox behaviour (that is typically_mail's rows); this reuses its parsing, cleaning and threading only.

    uv run --no-sync python scripts/typically_decisions.py                  # every stage, cached under OUT (resumable)
    uv run --no-sync python scripts/typically_decisions.py --triage 3000 --extract 600 --budget 40

Stages: parse the maildir corpus-wide (dedupe the copies that sit in many mailboxes) -> threads (typically_mail's
subject + people + time fallback) -> cheap candidate filter -> LLM triage on a stratified sample ("is a business decision made
here, outcome visible?") -> LLM extraction (question, options, chosen, the message that decides) -> taxonomy of recurring
decision types + mapping -> leakage spot-check -> extrapolation to the whole corpus.
Leakage control by construction: the case is the cleaned messages BEFORE the deciding message, never an LLM summary.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typically_llm as tllm
import typically_mail as tm
import typically_plan as tp

REPO = Path(__file__).resolve().parent.parent
MAILDIR = REPO / ".context" / "enron" / "maildir"
OUT = REPO / ".context" / "typically" / "decisions-spike"
TRIAGE_MODEL = "anthropic/claude-haiku-4.5"   # cheap yes/no; extraction, taxonomy and the leakage judge use the default (opus)
MAX_MSGS, MAX_CHARS = 15, 9000   # ponytail: a thread is shown from its start, cut here; a decision past the cut is missed
WORKERS = 16

REQUEST = re.compile(r"\b(approv\w*|sign[- ]?off|authori[sz]\w*|should we|shall we|do we want|can we|could we|ok to|okay to|"
                     r"go ahead|green light|decid\w*|decision|recommend\w*|propos\w*|request\w*|agree\w*|accept\w*|reject\w*|"
                     r"declin\w*|waive\w*|exception|extend\w*|offer\w*|vendor|bid|quote|budget|contract|amend\w*|hire|hiring|"
                     r"candidate|credit|limit|escalat\w*|den(y|ied)|consent|permission|let me know if|your call)\b", re.I)
AUTOMATED = re.compile(r"unsubscribe|click here|to be removed from|mailing list|newsletter|this is an automated|"
                       r"do not reply to this", re.I)


# ---------------------------------------------------------------- corpus -> threads

def _mailbox(user: str) -> list[tuple]:
    """(dedupe key, Msg with its cleaned body) for every message under maildir/<user>."""
    out = []
    for f in sorted(p for p in (MAILDIR / user).rglob("*") if p.is_file()):
        m = tm._parse(f.read_bytes(), f"{user}/{f.parent.relative_to(MAILDIR / user).as_posix()}")
        if m:
            key = (m.frm, m.date, tm.subject_key(m.subject), m.body[:200])   # typically_mail.messages' key
            m.body = tm.clean_body(m.body)
            out.append((key, m))
    return out


def load_threads() -> list[dict]:
    """Corpus-wide threads, cached: [{"id", "mailbox", "msgs": [Msg]}]; a message filed in many mailboxes is kept once."""
    cache = OUT / "threads.pkl"   # our own cache under .context, never shared
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    seen, files = {}, 0
    with Pool(max(2, (os.cpu_count() or 4) - 2)) as pool:
        for part in pool.imap_unordered(_mailbox, sorted(p.name for p in MAILDIR.iterdir() if p.is_dir())):
            for key, m in part:
                files += 1
                if key in seen:
                    seen[key].folders |= m.folders
                else:
                    seen[key] = m
    msgs = [m for m in seen.values() if not m.bulk]
    threads, _ = tm.outcomes(msgs, set())   # no owner: corpus-wide, everyone counts as a participant
    out = []
    for t in threads:
        first = t["msgs"][0]
        tid = hashlib.sha1(f"{first.frm}|{first.date}|{first.subject}".encode()).hexdigest()[:12]
        out.append({"id": tid, "mailbox": sorted(first.folders)[0].split("/")[0], "msgs": t["msgs"]})
    stats = {"files": files, "unique_messages": len(seen), "non_bulk_messages": len(msgs), "threads": len(out)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus.json").write_text(json.dumps(stats, indent=1))
    cache.write_bytes(pickle.dumps(out))
    return out


def is_candidate(t: dict) -> bool:
    ms = t["msgs"]
    if len(ms) < 2 or len({m.frm for m in ms}) < 2 or not any("enron" in a for m in ms for a in (m.frm, *m.to, *m.cc)):
        return False
    text = "\n".join(m.subject + "\n" + m.body for m in ms)
    return bool(REQUEST.search(text)) and not AUTOMATED.search(text)


def stratified(cands: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n threads, each mailbox's share proportional to its candidate count (largest remainder)."""
    rng, by = random.Random(seed), defaultdict(list)
    for t in cands:
        by[t["mailbox"]].append(t)
    quota = {b: n * len(ts) / len(cands) for b, ts in by.items()}
    take = {b: int(q) for b, q in quota.items()}
    for b in sorted(quota, key=lambda b: take[b] - quota[b])[:n - sum(take.values())]:
        take[b] += 1
    return [t for b in sorted(by) for t in rng.sample(by[b], min(take[b], len(by[b])))]


# ---------------------------------------------------------------- rendering: the LLM's view and the case are the same text

def _who(m) -> str:
    return m.name if m.name and "@" not in m.name else tm._pretty(m.frm)


def render(msgs, upto: int | None = None) -> str:
    """Numbered messages [1]..[n] (the first `upto`), cleaned bodies, emails / phones / account numbers masked."""
    names = {a: n for m in msgs for n, a in m.named if n and "@" not in n}
    parts, size = [], 0
    for i, m in enumerate(msgs[:MAX_MSGS if upto is None else upto], 1):
        to = [names.get(a) or tm._pretty(a) for a in m.to + m.cc]
        to = ", ".join(to[:4]) + (f" +{len(to) - 4}" if len(to) > 4 else "")
        s = f"[{i}] From: {_who(m)} | To: {to} | {m.date:%d %b %Y %H:%M} | Subject: {m.subject or '(none)'}\n{m.body or '(empty)'}"
        if size + len(s) > MAX_CHARS and parts:
            break
        parts.append(s)
        size += len(s)
    return tm._ACCOUNT.sub("[number]", tp.mask("\n\n".join(parts)))


# ---------------------------------------------------------------- LLM steps (each cached as jsonl, keyed by id)

class Spend:
    """Running USD per step, persisted; `ok()` is False once the budget is spent."""
    def __init__(self, budget: float):
        self.path, self.budget, self.lock = OUT / "spend.json", budget, threading.Lock()
        self.by = json.loads(self.path.read_text()) if self.path.exists() else {}

    def add(self, step: str, usage: dict, model: str | None = None):
        with self.lock:
            d = self.by.setdefault(step, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
            d["calls"] += 1
            d["in"] += usage["input_tokens"]
            d["out"] += usage["output_tokens"]
            d["usd"] += tllm.usd(usage["input_tokens"], usage["output_tokens"], model)
            self.path.write_text(json.dumps(self.by, indent=1))

    def total(self) -> float:
        return sum(d["usd"] for d in self.by.values())

    def ok(self) -> bool:
        return self.total() < self.budget


def cached_map(step: str, items: list[tuple[str, object]], fn, spend: Spend) -> dict:
    """{id: fn(item)} for every (id, item), resuming from OUT/<step>.jsonl; stops submitting once the budget is spent."""
    path, done, lock = OUT / f"{step}.jsonl", {}, threading.Lock()
    if path.exists():
        done = {r["id"]: r["out"] for r in map(json.loads, path.read_text().splitlines())}
    todo = [(k, x) for k, x in items if k not in done]

    def run(kx):
        k, x = kx
        if not spend.ok():
            return
        try:
            out = fn(x)
        except tllm.LLMError as e:
            print(f"  {step} {k}: {e}")
            return
        with lock, path.open("a") as f:
            f.write(json.dumps({"id": k, "out": out}) + "\n")
            done[k] = out
            if len(done) % 100 == 0:
                print(f"  {step}: {len(done)}/{len(items)}  ${spend.total():.2f}")

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, todo))
    return {k: done[k] for k, _ in items if k in done}


def call(step: str, spend: Spend, system: str, user: str, schema: dict, max_tokens: int, model: str | None = None) -> dict:
    out, usage = tllm.complete_json(system, user, schema, max_tokens=max_tokens, model=model, effort="low")
    spend.add(step, usage, model)
    return out


def obj(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
TRIAGE_SYS = """You read an email thread from inside a company (Enron, 1999-2002) and decide whether it records a BUSINESS DECISION:
someone with authority, acting for the company, chose between alternatives - approve/reject a deal, trade, credit line or limit,
contract term or amendment, hire/offer, vendor or bid, price, budget or expense, an exception or waiver, escalate to legal, etc.
The question/request AND the outcome must both be visible in the thread.
NOT a decision: scheduling meetings, travel, personal or social mail, news/FYI, pure information requests, IT support,
a request with no visible answer. `what`: the decision in at most 12 words, or "none"."""
TRIAGE_SCHEMA = obj(decision=B, what=S)

EXTRACT_SYS = """You turn an email thread into training rows for a model that learns a COMPANY'S DECISION POLICY.
For each business decision made in the thread (usually one, at most 3; none if there is no real business decision with a visible outcome):
- decision_question: phrased GENERICALLY so the same question recurs across many threads and companies, with no names,
  amounts or outcome in it ("Should we approve this credit limit increase for the counterparty?", "Which vendor should we pick?").
- type: noul (a yes/no decision), choice (pick one of several named alternatives), score (a level or amount, options are ordered buckets).
- options: 2 to 6 short options; for noul exactly the two answers (e.g. ["approve", "reject"] or ["yes", "no"]).
- chosen: exactly one of options, the outcome actually taken.
- decision_message_index: the number [k] of the FIRST message that states or communicates the outcome. Must be >= 2: the
  messages before it are the case the model will see, so they must hold the request but not the answer.
- evidence: a short verbatim quote from message [k] showing the outcome.
- decided_by_name, decided_by_role (e.g. "VP Credit", "trader", "in-house counsel"; "" if unknown), domain (e.g. credit, trading,
  legal/contracts, HR/hiring, procurement, regulatory, finance/budget, operations, deal origination)."""
EXTRACT_SCHEMA = obj(decisions={"type": "array", "items": obj(
    decision_question=S, type={"type": "string", "enum": ["noul", "choice", "score"]}, options={"type": "array", "items": S},
    chosen=S, decision_message_index=I, evidence=S, decided_by_name=S, decided_by_role=S, domain=S)})

TAXO_SYS = """You design a taxonomy of RECURRING business decision types from a list of decisions extracted from a company's email.
Merge synonyms and near-duplicates aggressively: a type must be a decision that recurs (e.g. "Approve a credit limit / credit
support request?"), not one instance. Give each type a fixed option set (2-6) that every instance can be mapped onto; yes/no
families use two options such as ["approve", "reject"]. Aim for 15-40 types plus nothing else; rare one-offs will be mapped to "other"."""
TAXO_SCHEMA = obj(types={"type": "array", "items": obj(
    type_id=S, question=S, kind={"type": "string", "enum": ["noul", "choice", "score"]}, options={"type": "array", "items": S},
    description=S)})
MAP_SYS = """Map each extracted decision onto one decision type of the taxonomy (or "other" if none fits well), and its chosen
answer onto exactly one of that type's options (or "unmappable")."""
MAP_SCHEMA = obj(rows={"type": "array", "items": obj(i=I, type_id=S, chosen_option=S)})
LEAK_SYS = """You audit a training row for leakage. The model will see CASE (the email messages before the decision) and must predict
the decision. Does the CASE already state or make plain WHICH option was chosen (e.g. someone already announces the outcome, or
a later reply is quoted)? A strong recommendation that the decider may still reject is NOT leakage. Answer yes / partly / no."""
LEAK_SCHEMA = obj(reveals={"type": "string", "enum": ["yes", "partly", "no"]}, why=S)


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.casefold()).strip()


def validate(t: dict, d: dict) -> tuple[dict | None, str]:
    """A row, or None and why: chosen in options, 2-6 options, index in range and > 1, a non-empty case, evidence in that message."""
    opts, k, n = d["options"], d["decision_message_index"], min(len(t["msgs"]), MAX_MSGS)
    if not 2 <= len(opts) <= 6 or len({norm(o) for o in opts}) != len(opts):
        return None, "options"
    chosen = next((o for o in opts if norm(o) == norm(d["chosen"])), None)
    if chosen is None:
        return None, "chosen not in options"
    if not 2 <= k <= n:
        return None, "index"
    case = render(t["msgs"], k - 1)
    if len(case) < 80:
        return None, "empty case"
    ev = norm(d["evidence"])
    found = bool(ev) and (ev[:60] in norm(t["msgs"][k - 1].body + " " + t["msgs"][k - 1].subject))
    return {"thread": t["id"], "mailbox": t["mailbox"], "question": d["decision_question"], "type": d["type"], "options": opts,
            "chosen": chosen, "decided_by": f"{d['decided_by_name']} ({d['decided_by_role']})".replace(" ()", ""),
            "domain": d["domain"], "index": k, "evidence": d["evidence"], "evidence_found": found, "case": case}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", type=int, default=3000, help="candidate threads to triage (stratified by mailbox)")
    ap.add_argument("--extract", type=int, default=600, help="max decision threads to extract")
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--leak", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.budget)

    threads = load_threads()
    cands = [t for t in threads if is_candidate(t)]
    by_id = {t["id"]: t for t in threads}
    print(f"threads {len(threads):,}  candidates {len(cands):,}")

    sample = stratified(cands, a.triage)
    tri = cached_map("triage", [(t["id"], t) for t in sample], lambda t: call(
        "triage", spend, TRIAGE_SYS, render(t["msgs"]), TRIAGE_SCHEMA, 200, TRIAGE_MODEL), spend)
    yes = [by_id[k] for k, v in tri.items() if v["decision"]]
    print(f"triaged {len(tri):,}  decision threads {len(yes):,}")

    ext_ids = sorted(t["id"] for t in yes)[:a.extract]   # ids are hashes: a random subset
    ext = cached_map("extract", [(k, by_id[k]) for k in ext_ids], lambda t: call(
        "extract", spend, EXTRACT_SYS, render(t["msgs"]), EXTRACT_SCHEMA, 2000), spend)
    rows, why = [], Counter()
    for k, out in ext.items():
        for d in out["decisions"][:3]:
            r, w = validate(by_id[k], d)
            why[w] += 1
            if r:
                rows.append(r)
    print(f"extracted {len(ext):,} threads -> {sum(why.values())} decisions, valid {len(rows)}  {dict(why)}")

    # taxonomy: one pass over the question list, then map every instance in batches
    tpath = OUT / "taxonomy_raw.json"
    if not tpath.exists():
        lines = "\n".join(f"{i}. [{r['domain']}] {r['question']} {r['options']}" for i, r in enumerate(rows))
        tpath.write_text(json.dumps(call("taxonomy", spend, TAXO_SYS, lines, TAXO_SCHEMA, 8000), indent=1))
    types = {t["type_id"]: t for t in json.loads(tpath.read_text())["types"]}
    tlist = "\n".join(f"{t['type_id']}: {t['question']} options={t['options']}" for t in types.values())
    batches = [(f"b{j}", list(range(j, min(j + 40, len(rows))))) for j in range(0, len(rows), 40)]
    mapped = cached_map("map", batches, lambda idx: call("map", spend, MAP_SYS, "TAXONOMY\n" + tlist + "\n\nDECISIONS\n" + "\n".join(
        f"{i}. [{rows[i]['domain']}] {rows[i]['question']} options={rows[i]['options']} chosen={rows[i]['chosen']!r}" for i in idx),
        MAP_SCHEMA, 4000), spend)
    for out in mapped.values():
        for m in out["rows"]:
            if 0 <= m["i"] < len(rows):
                t = types.get(m["type_id"])
                rows[m["i"]]["type_id"] = m["type_id"] if t else "other"
                rows[m["i"]]["type_chosen"] = next((o for o in t["options"] if norm(o) == norm(m["chosen_option"])), None) if t else None

    # leakage spot-check
    rng = random.Random(1)
    leak_rows = rng.sample(rows, min(a.leak, len(rows)))
    leak = cached_map("leak", [(f"{r['thread']}:{r['index']}:{norm(r['question'])[:40]}", r) for r in leak_rows], lambda r: call(
        "leak", spend, LEAK_SYS, f"CASE\n{r['case']}\n\nQUESTION {r['question']}\nOPTIONS {r['options']}\nCHOSEN {r['chosen']}",
        LEAK_SCHEMA, 400), spend)

    report(threads, cands, tri, yes, ext, rows, why, types, leak, leak_rows, spend)


def report(threads, cands, tri, yes, ext, rows, why, types, leak, leak_rows, spend):
    yes_rate = len(yes) / max(1, len(tri))
    scale = len(cands) * yes_rate / max(1, len(ext))   # full-corpus decision threads per extracted thread
    counts = Counter(r.get("type_id", "other") for r in rows)
    out_types = []
    for tid, n in counts.most_common():
        t = types.get(tid, {"question": "(unmapped)", "kind": "", "options": []})
        bal = Counter(r.get("type_chosen") or "unmappable" for r in rows if r.get("type_id", "other") == tid)
        out_types.append({"type_id": tid, "question": t["question"], "kind": t["kind"], "options": t["options"], "observed": n,
                          "extrapolated": round(n * scale), "option_balance": dict(bal.most_common())})
    (OUT / "types.json").write_text(json.dumps(out_types, indent=1))
    with (OUT / "instances.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # 25 examples spanning types: round-robin over types by size
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("type_id", "other")].append(r)
    ex, rng = [], random.Random(2)
    for lst in by_type.values():
        rng.shuffle(lst)
    while len(ex) < 25 and any(by_type.values()):
        for tid, _ in counts.most_common():
            if by_type[tid] and len(ex) < 25:
                r = by_type[tid].pop()
                ex.append({"type_id": tid, "question": r["question"], "options": r["options"], "chosen": r["chosen"],
                           "decided_by": r["decided_by"], "case_excerpt": r["case"][-600:]})
    (OUT / "examples.json").write_text(json.dumps(ex, indent=1))

    lk = []
    for r in leak_rows:
        v = leak.get(f"{r['thread']}:{r['index']}:{norm(r['question'])[:40]}")
        if v:
            lk.append({"question": r["question"], "chosen": r["chosen"], "reveals": v["reveals"], "why": v["why"],
                       "case_tail": r["case"][-400:]})
    (OUT / "leakage.json").write_text(json.dumps({"counts": Counter(x["reveals"] for x in lk), "rows": lk}, indent=1))

    n_ext = max(1, len(ext))
    per = lambda step: spend.by.get(step, {}).get("usd", 0) / max(1, spend.by.get(step, {}).get("calls", 1))
    full_cost = len(cands) * per("triage") + len(cands) * yes_rate * per("extract") + len(rows) * scale / 40 * per("map")
    funnel = {
        **json.loads((OUT / "corpus.json").read_text()), "candidates": len(cands), "triaged": len(tri), "decision_threads": len(yes),
        "triage_yes_rate": round(yes_rate, 3), "extracted_threads": len(ext), "extracted_decisions": sum(why.values()),
        "valid_instances": len(rows), "invalid": {k: v for k, v in why.items() if k != "ok"},
        "evidence_found_rate": round(sum(r["evidence_found"] for r in rows) / max(1, len(rows)), 3),
        "instances_per_decision_thread": round(len(rows) / n_ext, 2),
        "extrapolated": {"decision_threads": round(len(cands) * yes_rate), "instances": round(len(rows) * scale),
                         "types_ge_30": sum(t["extrapolated"] >= 30 for t in out_types if t["type_id"] != "other"),
                         "types_ge_100": sum(t["extrapolated"] >= 100 for t in out_types if t["type_id"] != "other")},
        "spend_usd": {k: round(v["usd"], 2) for k, v in spend.by.items()} | {"total": round(spend.total(), 2)},
        "projected_full_corpus_usd": round(full_cost, 2),
    }
    (OUT / "funnel.json").write_text(json.dumps(funnel, indent=1))
    print(json.dumps(funnel, indent=1))
    for t in out_types[:15]:
        print(f"{t['observed']:4} -> {t['extrapolated']:6}  {t['type_id']:32} {t['option_balance']}")
    print("leakage:", dict(Counter(x["reveals"] for x in lk)))


if __name__ == "__main__":
    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        k, _, v = line.partition("=")
        if k.strip() and not k.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    main()
