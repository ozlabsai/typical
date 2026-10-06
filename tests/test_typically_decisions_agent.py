"""Email archive -> decisions table (scripts/typically_decisions_agent.py): record validation, the per-email pass with a scripted
fake LLM (context tool, escalation, resume, hard budget cap), types + audit + export shape. No network.
PYTHONPATH=scripts:site uv run --no-sync --with pytest --with httpx pytest tests/test_typically_decisions_agent.py -q"""
import json
import sys
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts")]
import typically_decisions_agent as da  # noqa: E402
import typically_llm as tllm  # noqa: E402

T0 = datetime(2001, 5, 14, 9, 0, tzinfo=timezone.utc)
ASK = "Northgate Energy has asked to trade physical gas with us through Q3. They offered a parent guaranty. Can credit approve a $5MM line?"
OK = "Northgate is approved for a $5MM unsecured line."


def mail(frm, to, subj, at, body):
    return f"From {frm} Mon May 14 09:00:00 2001\nFrom: {frm}\nTo: {to}\nSubject: {subj}\nDate: {format_datetime(at)}\n\n{body}\n"


def archive(n_filler=6) -> str:
    """An mbox: a request, the approval quoting it, a reply that answers without quoting (needs context), filler, and one copy."""
    m = [mail("pat.smith@acme.com", "tanya@acme.com", "Credit request - Northgate", T0, ASK),
         mail("tanya@acme.com", "pat.smith@acme.com", "RE: Credit request - Northgate", T0 + timedelta(hours=2), f"{OK}\n\n> {ASK}"),
         mail("pat.smith@acme.com", "tanya@acme.com", "Credit request - Westfield", T0 + timedelta(days=1),
              "Westfield Power asked for a $2MM unsecured line for financial swaps. They have no audited financials on file yet."),
         mail("tanya@acme.com", "pat.smith@acme.com", "RE: Credit request - Westfield", T0 + timedelta(days=1, hours=1), "Declined for now."),
         mail("news@vendor.com", "tanya@acme.com", "Weekly digest", T0, "Prices were up this week.")]
    m += [mail(f"joe{i}@acme.com", "tanya@acme.com", f"Lunch {i}", T0 + timedelta(minutes=i), "Pizza at noon?") for i in range(n_filler)]
    m.append(m[1])   # the same message filed twice: indexed once
    return "".join(m)


@pytest.fixture
def corpus(tmp_path):
    f = tmp_path / "box.mbox"
    f.write_text(archive())
    db = tmp_path / "corpus.db"
    da.index(f, db, log=lambda *_: None)
    return db


def test_index_dedupes_and_overview(corpus):
    con = da.connect(corpus)
    ov = da.overview(con)
    assert ov["messages"] == 11 and ov["copies"] == 12 and ov["company_domains"] == ["acme.com"]
    rid = con.execute("SELECT id FROM msg WHERE subject = 'RE: Credit request - Westfield'").fetchone()[0]
    assert [r["subject"] for r in da.context(con, rid)] == ["Credit request - Westfield"]   # earlier, same subject, shared people
    assert da.index(corpus.parent / "box.mbox", corpus, log=lambda *_: None)["messages"] == 11   # resumable: nothing re-added


# ---------------------------------------------------------------- record validation

EMAIL = f"[#2]\nFrom: tanya@acme.com\nTo: pat.smith@acme.com\nDate: 2001-05-14 11:00\nSubject: RE: Credit request\n\n{OK}\n\n> {ASK}"
GOOD = {"question": "Approve the requested credit line?", "type": "noul", "options": ["yes", "no"], "chosen": "Yes",
        "decided_by": "tanya@acme.com", "evidence_quote": OK, "case_text": "Subject: RE: Credit request\n\n" + ASK.replace(". ", ".\n")}


def test_validate_accepts_verbatim_row():
    row, why = da.validate(GOOD, EMAIL, [], ["acme.com"])
    assert why == "ok" and row["chosen"] == "yes" and row["evidence"] == OK   # chosen canonicalised to the option


@pytest.mark.parametrize("change, reason", [
    ({"chosen": "maybe"}, "chosen not in options"),
    ({"options": ["yes", "no", "later"]}, "options"),
    ({"case_text": "Northgate wants credit for gas trading through the third quarter, backed by a parent guaranty."}, "case not verbatim"),
    ({"case_text": f"{OK}\n\n{ASK}"}, "evidence inside the case"),
    ({"evidence_quote": "Northgate is approved for a big line"}, "evidence not verbatim"),
    ({"decided_by": "someone@other.com"}, "decided_by is not an address"),
    ({"case_text": "Subject: RE: Credit request"}, "case too short"),
])
def test_validate_rejects(change, reason):
    row, why = da.validate({**GOOD, **change}, EMAIL, [], ["acme.com"])
    assert row is None and why.startswith(reason), why


def test_validate_company_decider_and_case_before_decision():
    outside = EMAIL.replace("To: pat.smith@acme.com", "To: pat.smith@acme.com, boss@partner.com")
    assert da.validate({**GOOD, "decided_by": "boss@partner.com"}, outside, [], ["acme.com"])[1] == "decided_by is outside the company"
    bottom = EMAIL.replace(f"{OK}\n\n> {ASK}", f"> {ASK}\n\n{OK}")   # the case sits ABOVE the decision: not 'before' it
    assert da.validate(GOOD, bottom, [], ["acme.com"])[1].startswith("case not verbatim")
    assert da.validate(GOOD, bottom, [f"[#1]\nFrom: pat.smith@acme.com\n\n{ASK}"], ["acme.com"])[1] == "ok"   # unless an earlier message holds it


# ---------------------------------------------------------------- the pass + types + audit with a scripted fake LLM

def call(name, **args):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{name}", "type": "function",
                                                                 "function": {"name": name, "arguments": json.dumps(args)}}]}


def decision(chosen, evidence, case, by="tanya@acme.com"):
    return {"decision_made": True, "decisions": [{"question": "Approve the credit line?", "type": "noul", "options": ["yes", "no"],
                                                  "chosen": chosen, "decided_by": by, "evidence_quote": evidence, "case_text": case}]}


@pytest.fixture
def fake(monkeypatch):
    """chat_tools: the Northgate approval is found; the Westfield decline needs context(); the cheap model botches Westfield's
    case (paraphrase) so it is escalated. complete_json: taxonomy, mapping and both judges."""
    calls = {"chat": [], "json": []}
    monkeypatch.setattr(tllm, "_model", lambda model=None: (model or "opus", (1.0, 5.0)))

    def chat(messages, tools, *, model=None, max_tokens=0, tool_choice="auto"):
        email = messages[1]["content"]
        calls["chat"].append((model, email.split("Subject: ")[1].split("\n")[0]))
        u = {"input_tokens": 1000, "output_tokens": 50, "usd": 0.01}
        if "Declined for now." in email:
            if messages[-1]["role"] != "tool":
                return call("context", message_id=int(email.split("[#")[1].split("]")[0])), u
            ctx = messages[-1]["content"]
            case = ("Westfield Power asked for a credit line." if model == da.CHEAP else
                    ctx.split("\n\n", 1)[1].strip())
            return call("submit", **decision("no", "Declined for now.", "Subject: RE: Credit request - Westfield\n\n" + case)), u
        if OK in email and "> " in email:
            return call("submit", **decision("yes", OK, ASK)), u
        return call("submit", decision_made=False, decisions=[]), u

    def cj(system, user, schema, *, max_tokens, model=None, effort=None, **_):
        calls["json"].append(system[:20])
        u = {"input_tokens": 500, "output_tokens": 100}
        if system is da.TAXO_SYS:
            return {"types": [{"type_id": "approve_credit_line", "question": "Approve the credit line?", "kind": "noul",
                               "options": ["yes", "no"], "description": "credit"},
                              {"type_id": "Bad Slug", "question": "x", "kind": "noul", "options": ["yes", "no"], "description": ""}]}, u
        if system is da.MAP_SYS:
            ids = [int(line.split(".")[0]) for line in user.split("DECISIONS\n")[1].splitlines()]
            chosen = [line.split("chosen=")[1].split(" evidence=")[0].strip("'").upper() for line in user.split("DECISIONS\n")[1].splitlines()]
            return {"rows": [{"i": i, "type_id": "approve_credit_line", "option": c} for i, c in zip(ids, chosen)]}, u   # upper: canonicalised
        return {"reveals": "no", "label": "correct", "why": ""}, u
    monkeypatch.setattr(tllm, "chat_tools", chat)
    monkeypatch.setattr(tllm, "complete_json", cj)
    monkeypatch.setattr(da, "MIN_TYPE", 2)
    return calls


def test_build_dataset_end_to_end_and_resume(corpus, tmp_path, fake):
    events = []
    rows, types, stats = da.build_dataset(corpus, budget_usd=5, progress=events.append, workdir=tmp_path / "w")
    assert stats["pass"]["read"] == 11 and stats["pass"]["rows"] == 2 and stats["pass"]["escalated"] == 1
    assert {m for m, _ in fake["chat"]} == {da.CHEAP, da.ESCALATE_TO}
    assert [t["id"] for t in types] == ["approve_credit_line"] and types[0]["balance"] == {"yes": 1, "no": 1}
    assert all(set(r) == {"case", "approve_credit_line"} and r["approve_credit_line"] in ("yes", "no") for r in rows) and len(rows) == 2
    assert "Westfield Power asked for a $2MM unsecured line" in next(r["case"] for r in rows if r["approve_credit_line"] == "no")
    assert any(e.startswith("Defined: Approve the credit line?") for e in events) and any(e.startswith("Escalating #") for e in events)
    assert stats["audit"]["ab_agreement"] == 1.0 and stats["audit"]["judge"] == "cheap" and stats["audit"]["dropped"] == 0
    assert (tmp_path / "w" / "table.jsonl").read_text().count("\n") == 2
    n_chat, n_json = len(fake["chat"]), len(fake["json"])
    da.build_dataset(corpus, budget_usd=5, workdir=tmp_path / "w")   # resume: nothing read, typed, mapped or judged twice
    assert len(fake["chat"]) == n_chat and len(fake["json"]) == n_json


def test_audit_drops_leaky_rows(corpus, tmp_path, fake, monkeypatch):
    monkeypatch.setattr(da, "MIN_TYPE", 1)
    real = tllm.complete_json
    monkeypatch.setattr(tllm, "complete_json", lambda system, user, *a, **k: (
        ({"reveals": "yes", "label": "correct", "why": "announced"}, {"input_tokens": 1, "output_tokens": 1})
        if system is da.JUDGE_SYS and "Declined" in user else real(system, user, *a, **k)))
    rows, _, stats = da.build_dataset(corpus, budget_usd=5, workdir=tmp_path / "w")
    assert stats["audit"]["dropped"] == 1 and stats["audit"]["leaky"] == 0.5 and len(rows) == 1


def test_hard_budget_cap_stops_cleanly(corpus, tmp_path, fake):
    sp = da.Spend(tmp_path / "spend.json", 0.035)   # each call is reserved at its worst case before it runs
    out = da.run_pass(corpus, tmp_path, sp, 0.035, lambda *_: None)
    assert out["read"] < 11 and sp.total() <= 0.035
    again = da.run_pass(corpus, tmp_path, da.Spend(tmp_path / "spend.json", 1.0), 1.0, lambda *_: None)
    assert again["read"] == 11 and again["read_now"] == 11 - out["read"]   # resumes where it stopped


def test_export_writes_gzip_table(tmp_path):
    import gzip
    (tmp_path / "table.jsonl").write_text('{"case": "x", "a": "yes"}\n{"case": "y", "a": ""}\n')
    assert da.export(tmp_path, tmp_path / "t.jsonl.gz") == 2
    assert [json.loads(l)["case"] for l in gzip.open(tmp_path / "t.jsonl.gz", "rt")] == ["x", "y"]


def test_out_of_credit_waits_then_stops(corpus, tmp_path, fake, monkeypatch):
    monkeypatch.setattr(da, "PAY_WAIT_S", 0)
    monkeypatch.setattr(da, "WORKERS", 1)
    n = {"calls": 0}
    real = tllm.chat_tools

    def flaky(*a, **k):   # two 402s (a top-up landing), then fine
        n["calls"] += 1
        if n["calls"] <= 2:
            raise tllm.LLMError("HTTPStatusError 402")
        return real(*a, **k)
    monkeypatch.setattr(tllm, "chat_tools", flaky)
    events = []
    out = da.run_pass(corpus, tmp_path, da.Spend(tmp_path / "s.json", 5), 5, events.append)
    assert out["read"] == 11 and sum("Out of LLM credit" in e for e in events) == 2
    monkeypatch.setattr(tllm, "chat_tools", lambda *a, **k: (_ for _ in ()).throw(tllm.LLMError("HTTPStatusError 402")))
    (tmp_path / "state.db").unlink()
    events.clear()
    out = da.run_pass(corpus, tmp_path, da.Spend(tmp_path / "s2.json", 5), 5, events.append)
    assert out["read"] == 0 and "Stopping: the LLM account stayed out of credit" in events
