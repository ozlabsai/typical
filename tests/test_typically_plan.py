"""scripts/typically_plan.py: profile, heuristic plan, validation, rendering, LLM path with a fake client. No network."""
import copy
import json
import re
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from scripts import typically_plan as tp

RECORDS = tp.load_records({"kind": "sample"})
PROF = tp.profile(RECORDS)
PLAN = tp.heuristic_plan(PROF, RECORDS)
DEC = {d["column"]: d for d in PLAN["decisions"]}


def test_profile_sample():
    assert PROF["n_rows"] == len(RECORDS) and PROF["columns"]["ticket"]["kind"] == "text"
    assert PROF["columns"]["team"]["distinct"] == 4 and PROF["languages"][0]["code"] == "en"
    assert {v["value"] for v in PROF["columns"]["escalate"]["values"]} == {"yes", "no"}


def test_heuristic_plan_sample():
    assert [(q["column"], q["role"]) for q in PLAN["case"]["parts"]] == [("ticket", "body")]
    assert list(DEC) == ["team", "escalate", "urgency", "refund"]
    assert DEC["escalate"]["type"] == DEC["refund"]["type"] == "noul" and DEC["escalate"]["labels"] == ["no", "yes"]
    assert DEC["urgency"]["type"] == "score" and DEC["urgency"]["labels"] == ["0", "1", "2", "3"]
    assert DEC["team"]["type"] == "choice" and DEC["team"]["question"] == "Which team?"
    assert tp.validate_plan(copy.deepcopy(PLAN), PROF) == []


def test_normalise_ordinal_yesno_rare_conflicts():
    recs = [{"t": f"this is a long enough message body for case number {i}", "sev": ["Sev-1", "sev 2", "SEV3"][i % 3], "ok": ["Y", "n", "TRUE", "No"][i % 4],
             "lvl": ["low", "medium", "high", "low"][i % 4]} for i in range(60)]
    recs += [{"t": "this exact message body shows up on two rows, twice over", "sev": "Sev-1", "ok": "y", "lvl": "low"}, {"t": "this exact message body shows up on two rows, twice over", "sev": "Sev-1", "ok": "n", "lvl": "low"}] * 5
    d = tp.profile(recs)["decisions"]
    assert d["sev"]["ordinal"]["labels"] == ["sev3", "sev 2", "sev 1"]   # 1 is the most severe: last
    assert d["ok"]["yes_no"] and d["lvl"]["ordinal"]["labels"] == ["low", "medium", "high"]
    assert d["ok"]["conflicts"]["rows"] == 10 and d["ok"]["rare"] == []
    assert tp.norm("Sev-1") == tp.norm(" sev  1 ") == "sev 1" and tp.norm("-1") == "-1"


def test_pii_and_masking():
    text = "mail a.b@x.com or call +44 20 7946 0958, card 4111 1111 1111 1111, DE89 3704 0044 0532 0130 00"
    assert tp.mask(text) == "mail [email] or call [phone], card [card], [iban]"


def test_leakage_flagged():
    recs = [{"t": f"c{i}", "a": "x" if i % 2 else "y"} for i in range(100)]
    recs = [{**r, "copy": r["a"].upper()} for r in recs]
    assert tp.profile(recs)["decisions"]["a"]["leaks"] == [{"column": "copy", "purity": 1.0}]


def test_profile_beyond_the_3000_row_text_scan():
    words = ["alpha", "bravo", "charlie", "delta"]
    recs = [{"t": f"ticket {i} says {words[i % 4]} loudly", "label": words[i % 4]} for i in range(5000)]
    d = tp.profile(recs)["decisions"]["label"]   # used to raise TypeError (`text in None`) past 3,000 rows
    assert d["text_leaks"] == [{"column": "t", "share": 1.0}]


def test_validate_plan_rejections():
    def broken(fn):
        p = copy.deepcopy(PLAN)
        fn(p)
        return tp.validate_plan(p, PROF)
    assert any("'tomorrow'" in e for e in broken(lambda p: DEC_OF(p, "team")["mapping"].update(tomorrow="Sales")))
    assert any("not one of its labels" in e for e in broken(lambda p: DEC_OF(p, "team")["mapping"].update(sales="Nope")))
    assert any("exactly" in e for e in broken(lambda p: DEC_OF(p, "refund").update(labels=["no", "yes", "maybe"])))
    assert any("not in the data" in e for e in broken(lambda p: DEC_OF(p, "team").update(column="nope")))
    assert any("both a decision" in e for e in broken(lambda p: p["case"]["parts"].append({"column": "team", "role": "fact", "sentence": None})))
    assert any("{value}" in e for e in broken(lambda p: p["case"]["parts"][0].update(sentence="no placeholder")))


def DEC_OF(plan, col):
    return next(d for d in plan["decisions"] if d["column"] == col)


def test_validate_plan_unmapped_value_is_soft():
    p = copy.deepcopy(PLAN)
    del DEC_OF(p, "team")["mapping"]["billing"]
    assert tp.validate_plan(p, PROF) == []
    d = DEC_OF(p, "team")
    assert d["mapping"]["billing"] is None and d["needs_review"]


def test_sheets_url():
    u = "https://docs.google.com/spreadsheets/d/1AbC_-x/edit?usp=sharing#gid=42"
    assert tp.sheets_export_url(u) == "https://docs.google.com/spreadsheets/d/1AbC_-x/export?format=csv&gid=42"
    assert tp.sheets_export_url("https://docs.google.com/spreadsheets/d/1AbC/edit").endswith("gid=0")
    with pytest.raises(ValueError):
        tp.sheets_export_url("https://example.com/x")


def test_render_case_and_answers():
    plan = {"case": {"parts": [
        {"column": "ticket", "role": "body", "sentence": None},
        {"column": "tier", "role": "fact", "sentence": "The customer is on the {value} plan."},
        {"column": "region", "role": "fact", "sentence": None}]}, "decisions": PLAN["decisions"]}
    r = {"ticket": "It broke.", "tier": "gold", "region": "EU", "team": " Claims ", "escalate": "YES", "urgency": "2", "refund": "No"}
    assert tp.render_case(r, plan) == "The customer is on the gold plan.\nregion: EU.\n\nIt broke."
    assert tp.answers(r, plan) == {"team": "Claims", "escalate": "yes", "urgency": "2", "refund": "no"}


def test_csv_source():
    assert tp.load_records({"kind": "csv", "text": "a,b\n1, x \n2,y\n"}) == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
    with pytest.raises(ValueError):
        tp.load_records({"kind": "csv", "text": "a,b\n"})


# ---- LLM path, fake client

def llm_json(**over):
    """A valid LLM answer for the sample: same decisions, mapping as pairs."""
    out = {"case": {"parts": [{"column": "ticket", "role": "body", "sentence": None, "confidence": 0.9, "reason": "the message"}]},
           "decisions": [{**{k: d[k] for k in ("column", "include", "question", "type", "labels", "confidence", "reasons", "needs_review")},
                          "mapping": [{"value": k, "label": v} for k, v in d["mapping"].items()]} for d in PLAN["decisions"]],
           "excluded": []}
    out.update(over)
    return out


class FakeClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(r))], usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def run_llm(*replies):
    c = FakeClient(*replies)
    return c, tp.llm_plan(PROF, tp.sample_rows(PROF, RECORDS), RECORDS, client=c)


def test_llm_plan_ok():
    c, (plan, source, warnings) = run_llm(llm_json())
    assert source == "llm" and warnings == [] and len(c.calls) == 1
    kw = c.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["effort"] == "medium"
    assert kw["output_config"]["format"]["schema"] is tp.PLAN_SCHEMA
    assert plan["decisions"][0]["mapping"]["claims"] == "Claims" and {"issues", "languages"} <= set(plan)


def test_llm_plan_retries_then_falls_back():
    bad = llm_json()
    bad["decisions"][0]["mapping"].append({"value": "invented", "label": "Sales"})
    c, (plan, source, _) = run_llm(bad, llm_json())
    assert source == "llm" and len(c.calls) == 2 and "invented" in c.calls[1]["messages"][-1]["content"]
    c, (plan, source, warnings) = run_llm(bad, bad)
    assert source == "heuristic" and "validation" in warnings[0] and len(c.calls) == 2 and plan["decisions"]
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    c, (plan, source, warnings) = run_llm(err)
    assert source == "heuristic" and "APIConnectionError" in warnings[0]


def test_llm_prompt_cap_and_sample_rows():
    big = {**PROF, "junk": "x" * 200_000}
    c = FakeClient()
    _, source, warnings = tp.llm_plan(big, [], RECORDS, client=c)
    assert source == "heuristic" and not c.calls and "too large" in warnings[0]
    rows = tp.sample_rows(PROF, RECORDS)
    assert len(rows) == 25 and len({r["team"] for r in rows}) == 4


def test_hebrew_heuristic_plan_and_analyze_lang(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "site"))
    import typically_analyze
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(typically_analyze, "UPLOADS", Path(__import__("tempfile").mkdtemp()))
    app = FastAPI()
    app.include_router(typically_analyze.router)
    c = TestClient(app)
    en = c.post("/api/typically/analyze", json={"source": {"kind": "sample"}}).json()
    he = c.post("/api/typically/analyze", json={"source": {"kind": "sample"}, "lang": "he"}).json()
    assert en["lang"] == "en" and he["lang"] == "he" and he["name_hint"] == en["name_hint"] == "Northwind"
    q = {d["column"]: d["question"] for d in he["plan"]["decisions"]}
    assert q == {"team": "איזה team מתאים?", "escalate": "האם escalate?", "urgency": "מה רמת הurgency?", "refund": "האם refund?"}
    assert all(re.search("[א-ת]", d["reasons"][0]) for d in he["plan"]["decisions"])
    assert [d["labels"] for d in he["plan"]["decisions"]] == [d["labels"] for d in en["plan"]["decisions"]]   # data labels are not translated
    assert all(re.search("[א-ת]", i["detail"] + i["action"]) for i in he["plan"]["issues"])
    assert all(re.search("[א-ת]", e["reason"]) for e in he["plan"]["excluded"])
