"""typically decisions: a company's email archive -> its BUSINESS DECISIONS as a table (one text column `case` + one column per
recurring decision type), built by reading EVERY email, not by keyword gating or search-based sampling.

    uv run --no-sync python scripts/typically_decisions_agent.py index [SRC]      # maildir dir | .mbox | .eml | .zip -> corpus.db
    uv run --no-sync python scripts/typically_decisions_agent.py run --budget 150  # the pass + types + audit (resumable)
    uv run --no-sync python scripts/typically_decisions_agent.py export            # -> site/data/typically_enron_decisions.jsonl.gz

Library: build_dataset(corpus_db, budget_usd=..., progress=print, workdir=...) -> (rows, types, stats).
1. index: mail parsed mechanically (typically_mail's format readers), identical messages across mailboxes kept once, SQLite + FTS5.
2. pass: every unique message in a seeded random order, one cheap-model turn each (tools: `context` = earlier messages of the
   conversation, `submit` = the decisions made in it). Code validates every row (validate()); a failed extraction is retried once
   on the default (opus) model. Per-message results persist in workdir/state.db: a rerun resumes, nothing is read twice.
3. types: an LLM canonicalises the free-form questions into recurring types with fixed options and maps each row onto one;
   one-off questions are kept apart (oneoff.jsonl), not in the table.
4. audit: an LLM judge reads each row (case, question, chosen); leaky or mislabelled rows are dropped.
Every LLM call goes through one Spend with a hard cap: the pass stops cleanly when only the post-pass reserve is left.
Tests: tests/test_typically_decisions_agent.py
"""
import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typically_llm as tllm
import typically_mail as tm

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / ".context" / "enron" / "maildir"
WORK = REPO / ".context" / "typically" / "decisions-agent"
EXPORT = REPO / "site" / "data" / "typically_enron_decisions.jsonl.gz"

# ---------------------------------------------------------------- 1. corpus index (mechanical only)

SCHEMA = """
CREATE TABLE IF NOT EXISTS msg(id INTEGER PRIMARY KEY, key TEXT UNIQUE, ts REAL, date TEXT, frm TEXT, name TEXT, rcpt TEXT,
                               cc TEXT, subject TEXT, skey TEXT, body TEXT, box TEXT, copies INTEGER DEFAULT 1);
CREATE INDEX IF NOT EXISTS msg_skey ON msg(skey, ts);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(subject, body, frm, rcpt, content='msg', content_rowid='id');
CREATE TABLE IF NOT EXISTS done(unit TEXT PRIMARY KEY);
"""


def _rows(unit: tuple[str, str]) -> list[tuple]:
    """(key, ts, date, frm, name, to, cc, subject, skey, body, box) for every message of one unit: a maildir mailbox, or one file."""
    kind, path = unit
    p = Path(path)
    if kind == "dir":
        raws = ((f.parent.relative_to(p.parent).as_posix(), f.read_bytes()) for f in sorted(p.rglob("*")) if f.is_file())
    else:
        raws = tm._raw(p.read_bytes())
    out = []
    for folder, raw in raws:
        if not (m := tm._parse(raw, folder)):
            continue
        key = hashlib.sha1(f"{m.frm}|{m.date.isoformat()}|{tm.subject_key(m.subject)}|{m.body.strip()[:200]}".encode()).hexdigest()   # ~typically_mail.messages. key
        out.append((key, m.date.timestamp(), m.date.isoformat(), m.frm, m.name, ",".join(m.to), ",".join(m.cc), m.subject,
                    tm.subject_key(m.subject), m.body, (folder or p.name).split("/")[0]))
    return out


def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db, check_same_thread=False, timeout=60)
    con.executescript(SCHEMA)
    con.row_factory = sqlite3.Row
    return con


def index(src: Path, db: Path, log=print) -> dict:
    """Streaming and resumable: one unit (mailbox dir / file) per transaction, finished units recorded in `done`."""
    db.parent.mkdir(parents=True, exist_ok=True)
    con = connect(db)
    units = ([("dir", str(d)) for d in sorted(src.iterdir()) if d.is_dir()] if src.is_dir() and not (src / "cur").exists()
             else [("dir", str(src))] if src.is_dir() else [("file", str(src))])
    done = {u for (u,) in con.execute("SELECT unit FROM done")}
    todo = [u for u in units if u[1] not in done]
    log(f"index: {len(units)} units, {len(todo)} to go")
    with Pool(max(2, (os.cpu_count() or 4) - 2)) as pool:
        for n, (unit, rows) in enumerate(pool.imap_unordered(_unit, todo), 1):
            with con:
                for r in rows:
                    cur = con.execute("INSERT OR IGNORE INTO msg(key, ts, date, frm, name, rcpt, cc, subject, skey, body, box) "
                                      "VALUES (?,?,?,?,?,?,?,?,?,?,?)", r)
                    if cur.rowcount:
                        con.execute("INSERT INTO fts(rowid, subject, body, frm, rcpt) VALUES (?,?,?,?,?)", (cur.lastrowid, r[7], r[9], r[3], r[5]))
                    else:
                        con.execute("UPDATE msg SET copies = copies + 1 WHERE key = ?", (r[0],))
                con.execute("INSERT INTO done VALUES (?)", (unit[1],))
            log(f"index: {n}/{len(todo)} {Path(unit[1]).name}: {len(rows)} messages")
    stats = overview(con)
    log(f"index: {stats['messages']:,} unique messages")
    return stats


def _unit(u):
    return u, _rows(u)


def overview(con: sqlite3.Connection) -> dict:
    """Corpus stats; company_domains = sender domains with >= 20% of the top domain's messages (the company's own mail)."""
    n, lo, hi = con.execute("SELECT count(*), min(date), max(date) FROM msg").fetchone()
    doms = Counter(dict(con.execute("SELECT substr(frm, instr(frm, '@') + 1) d, count(*) FROM msg GROUP BY d ORDER BY 2 DESC LIMIT 20")))
    top = max(doms.values(), default=0)
    return {"messages": n, "first": lo, "last": hi, "sender_domains": dict(doms.most_common(10)),
            "company_domains": [d for d, k in doms.items() if k >= 0.2 * top],
            "copies": con.execute("SELECT sum(copies) FROM msg").fetchone()[0]}



# ---------------------------------------------------------------- what the model sees: one email, rendered the same way it is validated

MAX_BODY, MAX_CTX_BODY, MAX_TO = 12_000, 4_000, 8   # ponytail: a longer body is cut (its tail is usually old quoted history)


def render(r: sqlite3.Row, cap: int = MAX_BODY) -> str:
    """[#id] + header block + raw body (quoted history included). The header block ends at the first blank line."""
    to = [a for a in (r["rcpt"] or "").split(",") if a]
    cc = [a for a in (r["cc"] or "").split(",") if a]
    more = lambda xs: ", ".join(xs[:MAX_TO]) + (f" (+{len(xs) - MAX_TO} more)" if len(xs) > MAX_TO else "")
    head = [f"[#{r['id']}]", f"From: {r['name']} <{r['frm']}>" if r["name"] and r["name"] != r["frm"] else f"From: {r['frm']}",
            f"To: {more(to)}"] + ([f"Cc: {more(cc)}"] if cc else []) + [f"Date: {r['date'][:16].replace('T', ' ')}",
                                                                       f"Subject: {r['subject'] or '(none)'}"]
    body = (r["body"] or "").replace("\r\n", "\n").strip()
    if len(body) > cap:
        body = body[:cap] + "\n[... rest of the email cut]"
    return "\n".join(head) + "\n\n" + body


def get(con: sqlite3.Connection, mid: int) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM msg WHERE id = ?", (mid,)).fetchone()


def context(con: sqlite3.Connection, mid: int, k: int = 3) -> list[sqlite3.Row]:
    """Up to k earlier messages of the same conversation: same normalised subject, sharing a participant, before this one, latest first."""
    r = get(con, mid)
    if r is None or not r["skey"]:
        return []
    people = lambda x: {x["frm"], *(x["rcpt"] or "").split(","), *(x["cc"] or "").split(",")} - {""}
    me = people(r)
    rows = con.execute("SELECT * FROM msg WHERE skey = ? AND ts < ? ORDER BY ts DESC LIMIT 40", (r["skey"], r["ts"])).fetchall()
    return [x for x in rows if people(x) & me][:k]


# ---------------------------------------------------------------- validation: code checks every row the model writes

def norm(s: str) -> str:
    """Whitespace collapsed and '>' quote markers dropped; nothing else (the case must be verbatim)."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^[ \t>]+", "", s or "")).strip()


def key(s: str) -> str:   # dedupe identity of an evidence quote
    return re.sub(r"\W+", " ", s.casefold()).strip()


TYPES = ("noul", "choice", "score")
ADDR = re.compile(r"[\w.+'-]+@[\w-]+(\.[\w-]+)+")


def validate(d: dict, email: str, ctx: list[str], company: list[str]) -> tuple[dict | None, str]:
    """One decision the model submitted -> (clean row, "ok") or (None, why). `email` is the rendered email the decision is
    made in, `ctx` the rendered earlier messages it fetched. Rules: options 2-6 and distinct, chosen one of them; evidence verbatim
    in the email; every case paragraph verbatim in the email's header block, in the email BELOW the evidence (older quoted
    history) or in an earlier message; the evidence not inside the case; the decider a company address seen in these messages."""
    q, typ, opts = (d.get("question") or "").strip(), d.get("type"), [str(o).strip() for o in d.get("options") or []]
    if not q or typ not in TYPES:
        return None, "question/type"
    if not 2 <= len(opts) <= 6 or len({o.casefold() for o in opts}) != len(opts) or not all(opts) or (typ == "noul" and len(opts) != 2):
        return None, "options"
    chosen = next((o for o in opts if o.casefold() == str(d.get("chosen", "")).strip().casefold()), None)
    if chosen is None:
        return None, "chosen not in options"
    ev, E = norm(d.get("evidence_quote", "")), norm(email)
    if len(ev) < 8 or (pos := E.find(ev)) < 0:
        return None, "evidence not verbatim in the email"
    paras = [norm(p) for p in re.split(r"\n\s*\n", d.get("case_text") or "") if norm(p)]
    case = "\n\n".join(paras)
    if len(case) < 60:
        return None, "case too short"
    if len(case) > 6000:
        return None, "case too long"
    if ev.casefold() in norm(case).casefold():
        return None, "evidence inside the case"
    head, C = norm(email.split("\n\n", 1)[0]), [norm(c) for c in ctx]
    for p in paras:
        if not (p in head or E.find(p, pos + len(ev)) >= 0 or any(p in c for c in C)):
            return None, "case not verbatim from before the decision: " + p[:80]
    who = (d.get("decided_by") or "").strip().lower()
    if not ADDR.fullmatch(who) or who not in (email + " ".join(ctx)).lower():
        return None, "decided_by is not an address in these messages"
    if not any(who.endswith("@" + c) or who.endswith("." + c) for c in company):
        return None, "decided_by is outside the company"
    return {"question": q, "type": typ, "options": opts, "chosen": chosen, "decided_by": who, "evidence": ev,
            "case": (d.get("case_text") or "").strip()}, "ok"


# ---------------------------------------------------------------- 2. the pass: one model turn per email

CHEAP, ESCALATE_TO = "anthropic/claude-haiku-4.5", None   # None: typically_llm's default (the newest opus)
MAX_TURNS, WORKERS = 3, 32   # turn 3 must submit: at most 2 context() calls per email
SYSTEM = """You read ONE email from a company's archive and report the BUSINESS DECISIONS made in it. The rows you report train a
model that learns how this company decides, so precision matters more than recall: report a decision only when it is clearly
there and every field can be filled exactly as the rules below demand. Most emails (newsletters, chatter, scheduling, FYIs,
status reports, questions without an answer) contain no decision; then submit decision_made=false with an empty list.

WHAT COUNTS AS A BUSINESS DECISION
Someone acting for the company (an address at a company domain, given to you below) chooses between real alternatives on a
matter of the business, and the choice is stated in this email:
- approve or reject a deal, trade, transaction, credit line, credit limit, collateral, prepayment, guarantee
- authorize a counterparty, a trader, a product, a system access, a signature, a payment, an expense, a budget
- accept, reject or counter a contract, an amendment, a term, a revision proposed by the other side
- hire, advance, make an offer to, or turn down a candidate; promote; approve a raise, a transfer, an exception
- extend, renew, terminate, restructure or settle an agreement, a dispute, a claim, an invoice
- file or not file with a regulator, take or not take a legal position, intervene in a proceeding
- pick a vendor, a bid, a site, a price, a volume, a level or an amount
- correct or keep a booking, a schedule, a nomination; give or refuse permission
NOT decisions: scheduling a meeting or a call; travel; social or personal mail; news, newsletters, research notes; pure
information (sending a file, a report, prices); IT help; an open question or a request that is not answered in this email; a
decision made by another company (a counterparty accepting OUR offer is theirs, not ours); a plan or opinion with no choice made.
Report decisions made in THIS email's own new text (the part its sender wrote, usually on top). A decision that appears only in
the quoted history below belongs to the earlier email, which is read separately: do not report it here.

FIELDS (each decision)
- question: the decision phrased GENERICALLY so the same question recurs across many emails: no names, companies, amounts,
  dates or outcome in it. Good: "Approve the counterparty for trading on the online platform?", "Accept the counterparty's
  proposed contract revisions?", "Advance this candidate to an offer?", "Which credit support should we require?"
  Bad: "Should John approve the $5MM Duke deal on 5/14?" (names, amount, date), "Was the GTC approved?" (past tense, outcome).
- type: "noul" = a yes/no decision; "choice" = one of several named alternatives; "score" = an ordered level or amount bucket.
- options: for noul exactly ["yes", "no"] with the question phrased so that "yes" is the action (approve / accept / authorize /
  hire / file). For choice: 2-6 short GENERIC alternatives that cover what could reasonably have been chosen (e.g. ["approve as
  drafted", "approve with changes", "reject"], ["full access", "restricted access", "no access"]). For score: 2-6 ordered
  buckets from least to most.
- chosen: exactly one of options, copied character for character: what was actually decided.
- decided_by: the email address of the person who made (or, for the company, communicated) the decision; it must be a company
  address that appears in these messages, normally the sender of this email.
- evidence_quote: a short verbatim quote (one sentence or phrase, copied exactly) from this email that states the decision,
  e.g. "Credit approves the attached deal" or "we will not be signing the amendment".
- case_text: what the decision-maker knew BEFORE deciding, copied VERBATIM: the request, the proposal, the question, the facts.
  It is what the model will read to predict the decision, so it must never contain or reveal the outcome. Rules, checked by code:
  * copy whole sentences or paragraphs exactly as written (line breaks may differ); separate excerpts with a blank line; you
    may leave text out but never reword, summarise, abbreviate or add anything of your own;
  * take it only from (a) the quoted history BELOW the decision in this email (the older messages the sender replied to or
    forwarded), (b) messages returned by the context tool, or (c) this email's Subject line copied as "Subject: ...";
  * never from the decision text itself or anything written after the decision; never include the evidence_quote;
  * leave out signatures, disclaimers, long distribution lists and anything that tells what was decided;
  * it must be substantive (at least two full sentences that explain what is being decided).
  If this email decides but neither quotes what it answers nor contains enough context, call context(message_id) for the
  earlier messages of the conversation. If there is still no verbatim context for the decision, do not report it.

TOOLS
- context(message_id): earlier messages of the same conversation (same subject, shared participants, before this one), newest
  first, each with its [#id] header. You may call it at most twice. Use it only when the email answers something it does not quote.
- submit(decision_made, decisions): your answer, exactly once. Code validates every field; a row that breaks a rule is lost.

WORKED EXAMPLES (abridged)

Example 1 - a decision with quoted context.
  [#101] From: Tanya Rohauer <tanya.rohauer@acme.com> ... Subject: RE: Credit request - Northgate Energy
  "Northgate is approved for a $5MM unsecured line, 12 month tenor. Please put the guaranty in the file.
   -----Original Message----- From: Smith, Pat ... Pat writes: Northgate Energy has asked to trade physical gas with us
   through Q3. They have offered a parent guaranty from Northgate Holdings. Can credit approve an unsecured line of $5MM?"
  submit: decision_made=true, decisions=[{question: "Approve the requested unsecured credit line for the counterparty?",
  type: "noul", options: ["yes", "no"], chosen: "yes", decided_by: "tanya.rohauer@acme.com",
  evidence_quote: "Northgate is approved for a $5MM unsecured line, 12 month tenor.",
  case_text: "Subject: RE: Credit request - Northgate Energy\n\nNorthgate Energy has asked to trade physical gas with us
  through Q3. They have offered a parent guaranty from Northgate Holdings. Can credit approve an unsecured line of $5MM?"}]

Example 2 - a decision among alternatives.
  [#102] From: Mark Taylor <mark.taylor@acme.com> Subject: RE: EOL profile - Westfield Power
  "Set them up for financial products only; no physical until we get their financials.
   > Westfield Power applied for access to both physical and financial products on the online platform. They are a new
   > counterparty and we have no financial statements on file."
  -> question "What trading access should the counterparty get on the online platform?", type "choice",
  options ["full access", "financial only", "physical only", "no access"], chosen "financial only",
  evidence_quote "Set them up for financial products only", case_text = the two quoted sentences (without the ">" marks is fine).

Example 3 - a rejection (just as valuable as an approval).
  [#103] From: Sara Shackleton <sara.shackleton@acme.com> Subject: RE: ISDA amendment - Brant Bank
  "We cannot accept their change to the termination currency. Please send back our original language.
   ---- Forwarded by ... ---- Brant Bank marked up section 5 so that the termination currency is the currency of their choice,
   and they ask us to accept the revised schedule as is."
  -> question "Accept the counterparty's proposed contract revisions?", noul, ["yes", "no"], chosen "no".

Example 4 - NOT a decision: "Can you send me the latest credit list? Thanks" (a request, no choice made).
Example 5 - NOT a decision: "Attached is the weekly gas report. Let me know if you have questions." (information).
Example 6 - NOT ours: "Northgate has signed the confirm and accepted our price." (the counterparty decided; the company did not).
Example 7 - NOT reportable: "Approved." with no quoted text, and context returns nothing about what was approved
  (no verbatim context: the model would have nothing to read).
Example 8 - two decisions in one email are two rows, each with its own question, evidence_quote and case_text.

Example 9 - a hiring decision (choice).
  [#109] From: Vince Kaminski <vince.kaminski@acme.com> Subject: RE: Interview feedback - J. Alvarez
  "Let's bring him back for a second round with the trading desk; I am not ready to make an offer yet.
   -----Original Message----- Shirley wrote: Feedback from today's interviews is attached. Two of three interviewers
   recommend an offer at the associate level. He has a PhD in operations research and two years at a utility."
  -> question "What should happen next with this candidate?", choice, options ["make an offer", "another interview round",
  "decline"], chosen "another interview round", decided_by "vince.kaminski@acme.com",
  evidence_quote "Let's bring him back for a second round with the trading desk",
  case_text = "Subject: RE: Interview feedback - J. Alvarez" + a blank line + the quoted feedback sentences.

Example 10 - a regulatory decision.
  [#110] From: Jim Steffes <james.steffes@acme.com> Subject: RE: FERC filing on the price cap order
  "We will file comments but will not request rehearing. Draft due Friday.
   > The commission's order caps prices in the western markets. Options are (1) do nothing, (2) file comments, or (3) request
   > rehearing, which preserves our right to appeal but is costly and visible."
  -> question "How should we respond to the regulatory order?", choice, ["do nothing", "file comments", "request rehearing"],
  chosen "file comments". Note how the options come from the alternatives the case itself lists.

Example 11 - an amount bucket (score).
  [#111] From: Rick Buy <rick.buy@acme.com> Subject: RE: VaR limit increase request - gas desk
  "I'll approve a temporary increase to $40MM through month end, not the $60MM requested.
   > The gas desk is requesting its VaR limit be raised from $30MM to $60MM for the remainder of the month because of the
   > storage position and the volatility since the cold snap."
  -> question "How much of the requested limit increase should be granted?", score, ["none", "part of it", "all of it"],
  chosen "part of it". Prefer generic buckets like these over raw amounts.

Example 12 - an extension of an existing deal.
  [#112] From: Daren Farmer <daren.farmer@acme.com> Subject: RE: Meter 1234 flow for April
  "Extend deal 456789 through April to cover the flow; do not create a new ticket.
   > Meter 1234 flowed 2,000 MMBtu/d in April but the purchase deal ends March 31. Should we extend the existing deal or set
   > up a new one so the volumes can be allocated?"
  -> question "How should flow outside the deal's term be handled?", choice, ["extend the existing deal", "create a new deal",
  "leave unallocated"], chosen "extend the existing deal".

Example 13 - NOT a decision: "I think we should probably look at extending the deal, what do you think?" (an opinion and a
  question; nothing is decided yet). The later reply that says "Agreed, extend it" is the decision, in that later email.
Example 14 - NOT a decision: an automated notice ("Your password will expire", "Trade confirmation #123 attached"), a news
  digest, a conference invitation, a lunch order, a fantasy-football update, an out-of-office reply.
Example 15 - NOT reportable here: the decision is visible only inside a forwarded or quoted older message and this email's
  sender only adds "FYI" or "see below": that decision belongs to the older email.
Example 16 - careful with the case: in "Approved, but cap it at 10,000/day. > Can we sell 15,000/day to Northgate in May?",
  the case is the quoted question only; the words "Approved, but cap it at 10,000/day" are the evidence and must not appear
  in the case. A good case reads like the moment just before someone decided.

Example 17 - picking among bids or vendors.
  [#117] From: Beth Perlman <beth.perlman@acme.com> Subject: RE: Data center UPS replacement - bids
  "Go with the second vendor; their price is higher but the service terms are what we need. Please send them the PO.
   > We received three bids for replacing the UPS units: the first is the cheapest but offers business-hours support only,
   > the second is 8% more and includes 24x7 on-site service, the third is the incumbent at 15% more."
  -> question "Which bid should we accept?", choice, ["lowest price", "best service terms", "incumbent", "none"],
  chosen "best service terms". Generic options describe each alternative by its role, not by the vendor's name.

Example 18 - a long thread where the email both answers and asks: report only the part that is decided ("Yes, sign the NDA;
  I'll get back to you on the exclusivity clause" is one decision about signing the NDA; the clause is still open).

COMMON MISTAKES THAT LOSE A ROW
- paraphrasing the case ("Northgate wants credit") instead of copying it; the case must be found word for word in the messages
- quoting the decision itself, or a later reply, in the case
- an evidence_quote that is not copied exactly (fixed typos, changed punctuation, joined two sentences)
- a chosen value that is not exactly one of the options; options that are specific to one email instead of generic
- a decided_by that is a name or a non-company address instead of the decider's company email address
- reporting a counterparty's decision, a question, a plan or an FYI as if the company had decided something

CHECKLIST BEFORE YOU SUBMIT A DECISION
1. Who decided? A person at the company, in this email's own new text, not a counterparty and not someone quoted below.
2. Was something actually chosen? Look for a verb of decision: approve, agree, accept, decline, reject, deny, authorize, sign,
   go ahead, proceed, hold off, extend, terminate, hire, offer, pass on, file, set, cap, keep, change, use, pick, go with.
   Conditional or tentative language ("probably", "I'd lean towards", "if legal agrees") is not yet a decision.
3. Is the question generic enough that it would recur across hundreds of emails at this company and at other companies?
4. Do the options cover the realistic alternatives, including the rejection, and is chosen copied from them exactly?
5. Is the evidence_quote a short exact copy from this email's new text?
6. Is every paragraph of case_text an exact copy from the quoted history below the decision, from the Subject line, or from a
   message the context tool returned? Would a reader of the case alone be unable to tell what was decided?
7. Is decided_by the decider's company email address as it appears in the headers?
If any answer is no, fix it or leave that decision out. Submitting decision_made=false is always acceptable.

Be literal and careful: copy text exactly, keep the case free of the outcome, and when unsure, report nothing."""

TOOLS = [
    {"type": "function", "function": {"name": "context", "description": "Earlier messages of this email's conversation.",
                                      "parameters": {"type": "object", "properties": {"message_id": {"type": "integer"}},
                                                     "required": ["message_id"]}}},
    {"type": "function", "function": {"name": "submit", "description": "The business decisions made in this email (call exactly once).",
                                      "parameters": {"type": "object", "required": ["decision_made", "decisions"], "properties": {
                                          "decision_made": {"type": "boolean"},
                                          "decisions": {"type": "array", "items": {"type": "object", "required": [
                                              "question", "type", "options", "chosen", "decided_by", "evidence_quote", "case_text"],
                                              "properties": {"question": {"type": "string"}, "type": {"type": "string", "enum": list(TYPES)},
                                                             "options": {"type": "array", "items": {"type": "string"}},
                                                             "chosen": {"type": "string"}, "decided_by": {"type": "string"},
                                                             "evidence_quote": {"type": "string"}, "case_text": {"type": "string"}}}}}}}},
]


class BudgetSpent(Exception):
    pass


class Spend:
    """USD by role, persisted; a HARD cap: a call runs only if spent + every in-flight call's worst case + its own fits under it."""
    def __init__(self, path: Path, cap: float):
        self.path, self.cap, self.lock, self.inflight = path, cap, threading.Lock(), 0.0
        self.by = json.loads(path.read_text()) if path.exists() else {}

    def total(self) -> float:
        return sum(d["usd"] for d in self.by.values())

    def call(self, role: str, model: str | None, prompt_chars: int, max_tokens: int, fn, limit: float | None = None):
        pin, pout = tllm._model(model)[1]
        est = (prompt_chars / 3 * pin * 1.25 + max_tokens * pout) / 1e6   # 1.25: a cache write
        with self.lock:
            if self.total() + self.inflight + est > min(self.cap, limit or self.cap):
                raise BudgetSpent(role)
            self.inflight += est
        try:
            out, u = fn()
        finally:
            with self.lock:
                self.inflight -= est
        with self.lock:
            d = self.by.setdefault(role, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
            cost = u.get("usd", tllm.usd(u["input_tokens"], u["output_tokens"], model))
            d["calls"], d["in"], d["out"], d["usd"] = d["calls"] + 1, d["in"] + u["input_tokens"], d["out"] + u["output_tokens"], d["usd"] + cost
            self.path.write_text(json.dumps(self.by, indent=1))
        return out, cost


STATE = """
CREATE TABLE IF NOT EXISTS seen(id INTEGER PRIMARY KEY, made INTEGER, ok INTEGER, bad INTEGER, escalated INTEGER, usd REAL, why TEXT);
CREATE TABLE IF NOT EXISTS row(rid INTEGER PRIMARY KEY, msg INTEGER, model TEXT, question TEXT, type TEXT, options TEXT, chosen TEXT,
                               decided_by TEXT, evidence TEXT, ekey TEXT UNIQUE, case_text TEXT);
"""


def extract(con, mid: int, model: str | None, spend: Spend, limit: float) -> tuple[list[dict], Counter, float, bool]:
    """One email through one model: -> (valid rows, invalid reasons, usd, decision_made)."""
    r = get(con, mid)
    email, ctx, usd = render(r), {}, 0.0
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Company domains: {', '.join(COMPANY)}\n\nEMAIL\n{email}\n\nRead it, then call submit "
                                        "(call context first only if it decides something it does not quote)."}]
    for turn in range(MAX_TURNS):
        force = turn == MAX_TURNS - 1
        msg, cost = spend.call("pass" if model == CHEAP else "escalate", model, sum(len(str(m["content"])) for m in msgs) + len(SYSTEM),
                               2500, lambda: tllm.chat_tools(msgs, TOOLS, model=model, max_tokens=2500, tool_choice=(   # opus 5.5 refuses forced tool_choice
                                   "auto" if model != CHEAP else {"type": "function", "function": {"name": "submit"}} if force else "required")), limit)
        usd += cost
        calls = msg.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                args = {}
            if c["function"]["name"] == "submit":
                rows, why, ds = [], Counter(), args.get("decisions") or []
                if isinstance(ds, str):   # a model sometimes sends the array JSON-encoded
                    try:
                        ds = json.loads(ds)
                    except ValueError:
                        ds = []
                for d in ds if isinstance(ds, list) else []:
                    row, w = validate(d if isinstance(d, dict) else {}, email, list(ctx.values()), COMPANY)
                    why[w] += 1
                    if row:
                        rows.append(row)
                return rows, Counter({k: v for k, v in why.items() if k != "ok"}), usd, bool(args.get("decision_made")) or bool(why)
            prev = context(con, int(args.get("message_id") or mid))
            ctx.update({x["id"]: render(x, MAX_CTX_BODY) for x in prev})
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": "\n\n".join(render(x, MAX_CTX_BODY) for x in prev) or "No earlier messages found."})
        if not calls:
            msgs.append({"role": "user", "content": "Call submit now (decision_made=false with an empty list if there is none)."})
    return [], Counter({"no submit": 1}), usd, False


COMPANY: list[str] = []   # set by run_pass from the corpus overview


def run_pass(corpus_db: Path, workdir: Path, spend: Spend, limit: float, progress, seed: int = 0, max_emails: int | None = None) -> dict:
    """Every unique message in a seeded random order until done, `max_emails` or the spend `limit`; resumable through state.db."""
    con = connect(corpus_db)
    COMPANY[:] = overview(con)["company_domains"]
    st = sqlite3.connect(workdir / "state.db", check_same_thread=False, timeout=60)
    st.executescript(STATE)
    ids = [r[0] for r in con.execute("SELECT id FROM msg ORDER BY id")]
    random.Random(seed).shuffle(ids)
    seen = {i for (i,) in st.execute("SELECT id FROM seen")}
    todo = iter([i for i in ids if i not in seen][:max_emails])
    lock, stop = threading.Lock(), threading.Event()
    n0 = len(seen)

    def one(con, mid: int):
        rows, why, usd, made = extract(con, mid, CHEAP, spend, limit)
        rows, esc, lost = [x | {"by": "cheap"} for x in rows], bool(why), why
        if esc:   # the cheap model's extraction broke a rule: the default model reads this email once more
            progress(f"Escalating #{mid} to the strong model ({', '.join(why)})")
            rows2, lost, usd2, made2 = extract(con, mid, ESCALATE_TO, spend, limit)
            rows, usd, made = rows + [x | {"by": "strong"} for x in rows2], usd + usd2, made or made2
        why = Counter({f"cheap: {k}": v for k, v in why.items()}) + Counter({f"lost: {k}": v for k, v in lost.items()})
        with lock, st:
            ok = 0
            for x in rows:
                try:
                    st.execute("INSERT INTO row(msg, model, question, type, options, chosen, decided_by, evidence, ekey, case_text) "
                               "VALUES (?,?,?,?,?,?,?,?,?,?)", (mid, x["by"], x["question"], x["type"],
                                                               json.dumps(x["options"]), x["chosen"], x["decided_by"], x["evidence"],
                                                               key(x["evidence"]), x["case"]))
                    ok += 1
                    progress(f"Decision in #{mid}: {x['question']} -> {x['chosen']}")
                except sqlite3.IntegrityError:   # the same evidence already recorded (a copy of this decision elsewhere)
                    why["lost: duplicate"] += 1
            st.execute("INSERT INTO seen VALUES (?,?,?,?,?,?,?)", (mid, made, ok, sum(v for k, v in why.items() if not k.startswith("cheap")),
                                                                 esc, usd, json.dumps(why)))

    def worker():
        con = connect(corpus_db)   # one connection per thread
        while not stop.is_set():
            with lock:
                mid = next(todo, None)
            if mid is None:
                return
            try:
                one(con, mid)
            except BudgetSpent:
                stop.set()
            except tllm.LLMError as e:   # not marked seen: a rerun retries it
                progress(f"LLM error on #{mid}: {e}")
                time.sleep(5)

    def ticker():
        while not stop.wait(30):
            with lock:
                n = st.execute("SELECT count(*), sum(ok) FROM seen").fetchone()
            progress(f"Read {n[0]:,} of {len(ids):,} emails, {n[1] or 0:,} decisions, ${spend.total():.2f}")
    threading.Thread(target=ticker, daemon=True).start()
    with ThreadPoolExecutor(WORKERS) as ex:
        for f in [ex.submit(worker) for _ in range(WORKERS)]:
            f.result()
    stop.set()
    n, made, ok, bad, esc, usd = st.execute("SELECT count(*), sum(made), sum(ok), sum(bad), sum(escalated), sum(usd) FROM seen").fetchone()
    out = {"emails": len(ids), "read": n, "read_now": n - n0, "coverage": round(n / len(ids), 4), "decision_emails": made or 0,
           "rows": ok or 0, "invalid": bad or 0, "escalated": esc or 0, "usd": round(usd or 0, 2),
           "usd_per_email": round((usd or 0) / max(n, 1), 5), "decision_rate": round((made or 0) / max(n, 1), 4),
           "rows_per_email": round((ok or 0) / max(n, 1), 4), "projected_full_usd": round((usd or 0) / max(n, 1) * len(ids), 0),
           "projected_full_rows": round((ok or 0) / max(n, 1) * len(ids))}
    reasons = Counter()
    for (w,) in st.execute("SELECT why FROM seen WHERE why != '{}'"):
        reasons.update({k.split(": ")[0] + ": " + k.split(": ")[1].split(":")[0]: v for k, v in json.loads(w).items()})
    out["invalid_reasons"] = dict(reasons.most_common())
    progress(f"Pass {'finished' if out['read'] == len(ids) else 'stopped'}: {n:,} emails read, {out['rows']:,} decisions, ${out['usd']:.2f}")
    return out


# ---------------------------------------------------------------- 3. recurring decision types (the old spike's taxonomy step)

MIN_TYPE = 25   # rows a type needs to become a column; fewer = one-off rows (oneoff.jsonl), reported apart
SLUG = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


def obj(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


S, I = {"type": "string"}, {"type": "integer"}
TAXO_SYS = """You design the RECURRING business decision types of one company from the decisions found in its email archive. Each
line is one decision: its free-form question, options and the chosen option. Merge synonyms and near-duplicates aggressively:
a type is a decision the company makes again and again (e.g. "Approve the counterparty for online trading?", "Accept the
counterparty's proposed contract revisions?"), not one instance. Each type gets:
- type_id: a short snake_case slug (it becomes a column name), e.g. approve_counterparty_online, accept_contract_revisions
- question: generic business English; kind: noul (yes/no) | choice | score (ordered)
- options: a FIXED set every instance can be mapped onto: noul types use exactly ["yes", "no"] with "yes" = the action;
  choice / score 2-6 short generic options (score ordered least to most)
- description: what belongs in it and what does not
Prefer fewer, broader types: each should cover at least ~3% of the lines (8-30 types in all); leave out questions that do not
recur (they stay as one-off rows)."""
TAXO_SCHEMA = obj(types={"type": "array", "items": obj(type_id=S, question=S, kind={"type": "string", "enum": list(TYPES)},
                                                         options={"type": "array", "items": S}, description=S)})
MAP_SYS = """Map each decision onto ONE decision type of the taxonomy, or "other" when none fits well (do not force a fit), and its
chosen answer onto exactly one of that type's options, or "unmappable" if the answer does not correspond to any of them.
Use the evidence (what the decider wrote) to pick the option."""
MAP_SCHEMA = obj(rows={"type": "array", "items": obj(i=I, type_id=S, option=S)})


def _json(spend: Spend, role: str, system: str, user: str, schema: dict, max_tokens: int, model: str | None = None) -> dict:
    out, _ = spend.call(role, model, len(system) + len(user), max_tokens,
                        lambda: tllm.complete_json(system, user, schema, max_tokens=max_tokens, model=model, effort="low"))
    return out


def _decisions(st: sqlite3.Connection) -> list[dict]:
    st.row_factory = sqlite3.Row
    return [dict(r) | {"options": json.loads(r["options"])} for r in st.execute("SELECT * FROM row ORDER BY rid")]


def canonicalize(st: sqlite3.Connection, workdir: Path, spend: Spend, progress) -> dict:
    """types.json (rebuilt when the pass added rows since) + the mapped table: rid -> (type_id, option)."""
    rows, f = _decisions(st), workdir / "types.json"
    st.executescript("CREATE TABLE IF NOT EXISTS mapped(rid INTEGER PRIMARY KEY, type_id TEXT, option TEXT);"
                     "CREATE TABLE IF NOT EXISTS audit(rid INTEGER, judge TEXT, reveals TEXT, label TEXT, why TEXT, PRIMARY KEY(rid, judge));")
    saved = json.loads(f.read_text()) if f.exists() else {}
    if saved.get("rows") != len(rows):
        lines = "\n".join(f"{i}. {r['question']} options={r['options']} chosen={r['chosen']!r}" for i, r in enumerate(rows))
        progress(f"Designing recurring decision types from {len(rows):,} decisions")
        raw = _json(spend, "types", TAXO_SYS, lines, TAXO_SCHEMA, 16000)["types"]
        types = {}
        for t in raw:
            opts = [o.strip() for o in t["options"] if o.strip()]
            if (SLUG.fullmatch(t["type_id"]) and t["type_id"] not in types and 2 <= len(opts) <= 6
                    and len({o.casefold() for o in opts}) == len(opts) and not (t["kind"] == "noul" and len(opts) != 2)):
                types[t["type_id"]] = {**t, "options": opts}
                progress(f"Defined: {t['question']} ({' / '.join(opts)})")
        saved = {"rows": len(rows), "types": types}
        f.write_text(json.dumps(saved, indent=1))
        with st:
            st.execute("DELETE FROM mapped")
            st.execute("DELETE FROM audit")
    types = saved["types"]
    tlist = "\n".join(f"{k}: {t['question']} options={t['options']} - {t['description']}" for k, t in types.items())
    done = {r[0] for r in st.execute("SELECT rid FROM mapped")}
    todo = [r for r in rows if r["rid"] not in done]
    lock = threading.Lock()

    def batch(part):
        text = "\n".join(f"{r['rid']}. {r['question']} options={r['options']} chosen={r['chosen']!r} evidence={r['evidence'][:200]!r}"
                         for r in part)
        try:
            out = _json(spend, "map", MAP_SYS, f"TAXONOMY\n{tlist}\n\nDECISIONS\n{text}", MAP_SCHEMA, 6000)["rows"]
        except (BudgetSpent, tllm.LLMError) as e:
            return progress(f"Mapping stopped for a batch: {type(e).__name__}")
        ids = {r["rid"] for r in part}
        with lock, st:
            for m in out:
                t = types.get(m["type_id"])
                opt = next((o for o in t["options"] if o.casefold() == m["option"].strip().casefold()), None) if t else None
                if m["i"] in ids:
                    st.execute("INSERT OR REPLACE INTO mapped VALUES (?,?,?)", (m["i"], m["type_id"] if t and opt else "other", opt))
        progress(f"Mapped {len(out)} decisions onto the types")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(batch, [todo[j:j + 50] for j in range(0, len(todo), 50)]))
    return types


# ---------------------------------------------------------------- 4. audit: leakage + label, LLM judge

JUDGE_SYS = """You audit one training row for a model that predicts a company's decisions. The model will read CASE and must answer
QUESTION with one of OPTIONS; the row says the company chose LABEL. EVIDENCE is what the decider actually wrote (the model never sees it).
1. reveals: does CASE already state or make plain which option was chosen (an announced outcome, a quoted reply, "as agreed",
   "approved" in a subject line)? A request, a recommendation or a strong argument that the decider could still reject is NOT
   leakage. Answer yes / partly / no.
2. label: given EVIDENCE, is LABEL the right option for QUESTION, and is this really a decision of that kind? correct / wrong / unsure."""
JUDGE_SCHEMA = obj(reveals={"type": "string", "enum": ["yes", "partly", "no"]}, label={"type": "string", "enum": ["correct", "wrong", "unsure"]},
                   why=S)


def judge(spend: Spend, r: dict, t: dict, model: str | None) -> dict:
    user = (f"QUESTION {t['question']}\nOPTIONS {t['options']}\nLABEL {r['option']}\nEVIDENCE {r['evidence']}\n\nCASE\n{r['case_text']}")
    return _json(spend, "audit", JUDGE_SYS, user, JUDGE_SCHEMA, 600, model)


def audit(st: sqlite3.Connection, types: dict, spend: Spend, progress, ab: int = 30) -> dict:
    """A/B the cheap judge against the strong one on `ab` rows; the cheap one audits every row if it agrees on >= 85% of the
    drop decisions, else the strong one does (as far as the budget goes). Returns the rates."""
    rows = [r | {"type_id": m[0], "option": m[1]} for r in _decisions(st) if (m := st.execute(
        "SELECT type_id, option FROM mapped WHERE rid = ?", (r["rid"],)).fetchone()) and m[0] in types]
    have = lambda judge_: {x[0]: {"reveals": x[1], "label": x[2]} for x in st.execute("SELECT rid, reveals, label FROM audit WHERE judge = ?", (judge_,))}
    lock = threading.Lock()

    def run(model, name, todo):
        def one(r):
            try:
                v = judge(spend, r, types[r["type_id"]], model)
            except (BudgetSpent, tllm.LLMError):
                return
            with lock, st:
                st.execute("INSERT OR REPLACE INTO audit VALUES (?,?,?,?,?)", (r["rid"], name, v["reveals"], v["label"], v["why"]))
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(one, [r for r in todo if r["rid"] not in have(name)]))
    sample = random.Random(1).sample(rows, min(ab, len(rows)))
    run(CHEAP, "cheap", sample)
    run(ESCALATE_TO, "strong", sample)
    c, s = have("cheap"), have("strong")
    both = [k for k in c if k in s]
    drop = lambda v: v["reveals"] == "yes" or v["label"] == "wrong"
    agree = sum(drop(c[k]) == drop(s[k]) for k in both) / max(len(both), 1)
    name = "cheap" if agree >= 0.85 else "strong"
    progress(f"Audit A/B on {len(both)} rows: the cheap judge agrees with the strong one on {agree:.0%} of drop decisions; "
             f"auditing with the {name} judge")
    run(CHEAP if name == "cheap" else ESCALATE_TO, name, rows)
    v = have(name)
    seen = [v[r["rid"]] for r in rows if r["rid"] in v]
    with st:
        st.execute("CREATE TABLE IF NOT EXISTS dropped(rid INTEGER PRIMARY KEY)")
        st.execute("DELETE FROM dropped")
        st.executemany("INSERT INTO dropped VALUES (?)", [(r["rid"],) for r in rows if r["rid"] in v and drop(v[r["rid"]])])
    n = max(len(seen), 1)
    out = {"judge": name, "ab_rows": len(both), "ab_agreement": round(agree, 3),
           "ab_strong_drop_rate": round(sum(map(drop, s.values())) / max(len(s), 1), 3), "audited": len(seen), "unaudited": len(rows) - len(seen),
           "leaky": round(sum(x["reveals"] == "yes" for x in seen) / n, 3), "partly_leaky": round(sum(x["reveals"] == "partly" for x in seen) / n, 3),
           "label_wrong": round(sum(x["label"] == "wrong" for x in seen) / n, 3), "label_unsure": round(sum(x["label"] == "unsure" for x in seen) / n, 3),
           "dropped": sum(map(drop, seen))}
    progress(f"Audit: {out['leaky']:.0%} leaky, {out['label_wrong']:.0%} mislabelled -> dropped {out['dropped']} of {len(seen)} rows")
    return out


# ---------------------------------------------------------------- 5. the table + the library entry point

def table(st: sqlite3.Connection, types: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """-> (table rows {case, <type_id>: option}, kept types with counts + balance, one-off rows). Rows sharing a case merge."""
    import typically_plan as tp
    st.executescript("CREATE TABLE IF NOT EXISTS dropped(rid INTEGER PRIMARY KEY)")
    gone = {r[0] for r in st.execute("SELECT rid FROM dropped")}
    mapped = {r[0]: (r[1], r[2]) for r in st.execute("SELECT rid, type_id, option FROM mapped")}
    rows = [r for r in _decisions(st) if r["rid"] not in gone]
    n = Counter(mapped[r["rid"]][0] for r in rows if r["rid"] in mapped)
    keep = {k for k in types if n[k] >= MIN_TYPE}
    by_case, oneoff, bal = {}, [], defaultdict(Counter)
    for r in rows:
        tid, opt = mapped.get(r["rid"], ("other", None))
        if tid in keep:
            case = tm._ACCOUNT.sub("[number]", tp.mask(r["case_text"]))
            by_case.setdefault(case, {"case": case})[tid] = opt
            bal[tid][opt] += 1
        else:
            oneoff.append({"case": r["case_text"], "question": r["question"], "type": r["type"], "options": r["options"], "chosen": r["chosen"],
                           "near_type": tid})
    kept = [{"id": k, **{f: types[k][f] for f in ("question", "kind", "options", "description")}, "count": sum(bal[k].values()),
             "balance": dict(bal[k].most_common())} for k in sorted(keep, key=lambda k: -n[k])]
    return [{"case": r["case"], **{t["id"]: r.get(t["id"], "") for t in kept}} for r in by_case.values()], kept, oneoff


def build_dataset(corpus_db: Path, *, budget_usd: float, progress=lambda event: None, workdir: Path = WORK,
                  max_emails: int | None = None, reserve: float = 0.12) -> tuple[list[dict], list[dict], dict]:
    """The email archive indexed in corpus_db -> (table rows, decision types, stats). Reads emails until done or until only
    `reserve` of the budget is left for the types + audit; a rerun with the same workdir resumes. budget_usd is a hard cap on spend."""
    workdir.mkdir(parents=True, exist_ok=True)
    spend = Spend(workdir / "spend.json", budget_usd)
    p = run_pass(Path(corpus_db), workdir, spend, budget_usd * (1 - reserve), progress, max_emails=max_emails)
    st = sqlite3.connect(workdir / "state.db", check_same_thread=False, timeout=60)
    types = canonicalize(st, workdir, spend, progress)
    a = audit(st, types, spend, progress)
    rows, kept, oneoff = table(st, types)
    stats = {"pass": p, "types_defined": len(types), "types_kept": len(kept), "audit": a, "table_rows": len(rows), "oneoff_rows": len(oneoff),
             "spend": {k: round(v["usd"], 2) for k, v in spend.by.items()} | {"total": round(spend.total(), 2)}}
    for name, data in (("table.jsonl", rows), ("oneoff.jsonl", oneoff)):
        (workdir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
    (workdir / "kept_types.json").write_text(json.dumps(kept, indent=1))
    (workdir / "stats.json").write_text(json.dumps(stats, indent=1))
    progress(f"Done: {len(rows):,} rows over {len(kept)} decision types, {len(oneoff):,} one-off decisions, ${spend.total():.2f} spent")
    return rows, kept, stats


def export(workdir: Path = WORK, out: Path = EXPORT) -> int:
    import gzip
    data = (workdir / "table.jsonl").read_bytes()
    with gzip.open(out, "wb") as f:
        f.write(data)
    return data.count(b"\n")


def _env():
    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        k, _, v = line.partition("=")
        if k.strip() and not k.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "run", "export"])
    ap.add_argument("src", nargs="?", default=str(SRC), help="index: a maildir directory, .mbox, .eml or .zip")
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--max-emails", type=int)
    a = ap.parse_args()
    if a.cmd == "index":
        index(Path(a.src), WORK / "corpus.db")
    elif a.cmd == "run":
        _env()
        stamp = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
        print(json.dumps(build_dataset(WORK / "corpus.db", budget_usd=a.budget, progress=stamp, max_emails=a.max_emails)[2], indent=1))
    else:
        print(f"{export():,} rows -> {EXPORT}")
