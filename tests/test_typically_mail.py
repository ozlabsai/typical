"""Email history -> decision table (scripts/typically_mail.py) and the "mail" / Enron sample sources of /analyze. No network.
PYTHONPATH=scripts:site uv run --no-sync --with pytest --with httpx pytest tests/test_typically_mail.py -q"""
import base64
import io
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "site"), str(ROOT / "scripts")]
import server  # noqa: E402
import typically_analyze as ta  # noqa: E402
import typically_job as tj  # noqa: E402
import typically_mail as tm  # noqa: E402
import typically_plan as tp  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ME = "me@acme.com"
T0 = datetime(2001, 5, 14, 9, 0, tzinfo=timezone(timedelta(hours=-7)))   # a Monday


def mk(frm, to, subj, at, body, cc=(), mid=None, irt=None, extra=""):
    h = [f"From: {frm}", f"To: {', '.join(to)}", f"Subject: {subj}", f"Date: {format_datetime(at)}"]
    h += [f"Cc: {', '.join(cc)}"] * bool(cc) + [f"Message-ID: {mid}"] * bool(mid) + [f"In-Reply-To: {irt}"] * bool(irt)
    return ("\n".join(h) + extra + "\n\n" + body + "\n").encode()


@pytest.fixture(autouse=True)
def few(monkeypatch):
    monkeypatch.setattr(tm, "MIN_ANSWERS", 5)   # a small box still keeps its decisions


def mailbox(ids: bool):
    """(folder, raw) list: 40 filler threads (half answered at varied speeds, some forwarded / filed) + the named cases."""
    out = []
    for i in range(40):
        at = T0 + timedelta(days=i % 10, hours=i)
        mid = f"<f{i}@x>" if ids else None
        out.append(("projects" if i % 3 == 0 else "legal" if i % 3 == 1 else "inbox",
                    mk(f"Person {i} <p{i}@ext{i % 4}.com>", [ME], f"Topic {i}", at, f"Please look at item {i} and let me know.", mid=mid)))
        if i % 2 == 0:   # replies after 10 min .. 3 days
            out.append(("sent", mk(ME, [f"p{i}@ext{i % 4}.com"], f"RE: Topic {i}", at + timedelta(minutes=10 * 4 ** (i % 6)), "sure", irt=mid)))
        if i % 5 == 0:
            out.append(("sent", mk(ME, [["Dave Doe <dave@acme.com>", "hal@acme.com"][i % 2]], f"FW: Topic {i}", at + timedelta(hours=2), "fyi")))
    out += [
        ("inbox", mk('"Alice Ames" <alice@x.com>', [ME], "Budget Q3", T0, "Can you approve the budget? Call me at 555-123-4567.\n"
                     "Card 4111 1111 1111 1111, account 12345678901.\n\n-- \nAlice\nVP Finance")),
        ("inbox", mk("Frank <frank@y.com>", [ME], "Budget Q3", T0 + timedelta(minutes=10), "Unrelated budget question from Frank.")),
        ("sent", mk(ME, ["alice@x.com"], "RE: Budget Q3", T0 + timedelta(minutes=30), "SECRETREPLYTOKEN approved, go ahead.")),
        ("inbox", mk("Erin <erin@x.com>", [ME, "alice@x.com"], "Re: Budget Q3", T0 + timedelta(days=30), "Following up a month later.")),
        ("inbox", mk("Carol <carol@z.com>", [ME], "Contract draft", T0, "Here is the draft we discussed.\n\n"
                     "-----Original Message-----\nFrom: me\nSent: Monday\nTo: Carol\nSubject: Contract\n\nOLDQUOTE please send it")),
        ("sent", mk(ME, ["Dave Doe <dave@acme.com>"], "FW: Contract draft", T0 + timedelta(days=1), "Dave, please review. FWDTOKEN")),
        ("inbox", mk("Service <no-reply@svc.com>", [ME], "Your receipt", T0, "Thanks for your order.")),
        ("inbox", mk("Gus <gus@z.com>", [ME], "Quick note", T0, "Thanks!\n> old quoted line QUOTEDTOKEN\nOn Mon, Bob wrote:\nmore QUOTEDTOKEN")),
    ]
    return out


def as_mbox(msgs):
    return b"".join(b"From x@y Mon May 14 09:00:00 2001\n" + raw.replace(b"\nFrom ", b"\n>From ") + b"\n"
                    for f, raw in msgs)


def as_maildir_zip(msgs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, (folder, raw) in enumerate(msgs):
            z.writestr(f"maildir/me-m/{folder}/{i + 1}.", raw)
    return buf.getvalue()


def by_subject(rows, subj, sender=None):
    return [r for r in rows if f"Subject: {subj}\n" in r["email"] + "\n" and (sender is None or f"From: {sender}" in r["email"])]


@pytest.mark.parametrize("ids", [False, True])
def test_maildir_outcomes(ids):
    rows, stats, warnings = tm.read_mail(as_maildir_zip(mailbox(ids)))
    assert stats["owner"] == ME
    assert stats["inbound"] == len(rows) == 40 + 5   # the no-reply receipt is bulk; sent mail is not a row
    alice, = by_subject(rows, "Budget Q3", "Alice Ames")
    assert alice["reply"] == "yes" and alice["reply_within"] == "within an hour"
    frank, = by_subject(rows, "Budget Q3", "Frank")   # same subject, no shared people: the reply to Alice is not his
    assert frank["reply"] == "no" and frank["reply_within"] == ""   # how fast: only asked when they replied
    erin, = by_subject(rows, "Re: Budget Q3")   # a month later: a new thread
    assert erin["reply"] == "no"
    carol, = by_subject(rows, "Contract draft")
    assert carol["forward"] == "yes" and carol["forward_to"] == "Dave Doe" and carol["reply"] == "no"
    assert alice["forward"] == "no" and alice["forward_to"] == ""
    assert {r["folder"] for r in rows} == {"projects", "legal", ""}   # inbox is a system folder: no answer
    assert set(rows[0]) == {"email", "reply", "reply_within", "forward", "forward_to", "folder"}
    assert {r["reply_within"] for r in rows if r["reply"] == "yes"} == {"within an hour", "within a day", "after a day"}


def test_case_text_clean_redacted_no_leak():
    rows, _, _ = tm.read_mail(as_maildir_zip(mailbox(False)))
    alice, = by_subject(rows, "Budget Q3", "Alice Ames")
    assert alice["email"].startswith("From: Alice Ames (x.com)\nTo: you\nReceived: Monday 09:00\nSubject: Budget Q3\n\nCan you approve")
    assert "555-123-4567" not in alice["email"] and "4111" not in alice["email"] and "12345678901" not in alice["email"]
    assert "VP Finance" not in alice["email"], "signature cut"
    carol, = by_subject(rows, "Contract draft")
    assert "OLDQUOTE" not in carol["email"] and "draft we discussed" in carol["email"]
    gus, = by_subject(rows, "Quick note")
    assert "QUOTEDTOKEN" not in gus["email"]
    blob = "\n".join(v for r in rows for v in r.values())
    assert "SECRETREPLYTOKEN" not in blob and "FWDTOKEN" not in blob, "what the owner wrote later never reaches a row"
    assert "@" not in blob, "no addresses in the case"


def test_mbox_owner_and_degenerate_drop():
    rows, stats, warnings = tm.read_mail(as_mbox(mailbox(True)))   # no folders: the owner is the top recipient
    assert stats["owner"] == ME and stats["inbound"] == len(rows) == 45
    assert "folder" not in rows[0] and any("'folder'" in w for w in warnings)
    _, stats2, _ = tm.read_mail(as_mbox(mailbox(True)), owner="P1@ext1.com")
    assert stats2["owner"] == "p1@ext1.com" and stats2["inbound"] == stats2["messages"] - 2   # all but p1's one and the bulk receipt
    one, _, _ = tm.read_mail(mailbox(False)[0][1])   # a single .eml: too few rows for any decision
    assert set(one[0]) == {"email"}


def test_clean_body_forward_only_keeps_original():
    s = "FYI\n\n---------------------- Forwarded by Bob/HOU/ECT on 05/14/2001 ----------\nFrom: X\nTo: Y\nSubject: Z\n\nThe pipeline is down."
    assert tm.clean_body(s) == "FYI\n\n[forwarded]\nThe pipeline is down."
    lotus = "Sounds good.\n\n\n\nJeff Dasovich@ENRON\n05/14/2001 04:39 PM\nTo: Vince\ncc: \nSubject: x\n\nold text"
    assert tm.clean_body(lotus) == "Sounds good."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(ta, "_records", ta.OrderedDict())
    monkeypatch.setattr(server, "TYPICALLY", tmp_path)
    monkeypatch.setattr(tj, "TYPICALLY", tmp_path)
    for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return TestClient(server.app)


def test_analyze_mail_plans_outcomes_as_decisions(client):
    data = base64.b64encode(as_maildir_zip(mailbox(False))).decode()
    r = client.post("/api/typically/analyze", json={"source": {"kind": "mail", "data_base64": data, "name": "me.zip"}})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["mail"]["owner"] == ME and a["mail"]["inbound"] == a["n_rows"] and a["name_hint"] == "me.zip"
    assert [q["column"] for q in a["plan"]["case"]["parts"]] == ["email"]
    assert {d["column"] for d in a["plan"]["decisions"]} <= {"reply", "reply_within", "forward", "forward_to", "folder"}
    assert {"reply", "folder"} <= {d["column"] for d in a["plan"]["decisions"]}
    big = client.post("/api/typically/analyze", json={"source": {"kind": "mail", "data_base64": "A" * (ta.MAX_MAIL_B64 + 4)}})
    assert big.status_code == 413 and "MB" in big.json()["detail"]


def test_enron_sample(client):
    r = client.post("/api/typically/analyze", json={"source": {"kind": "sample", "name": "enron"}})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["n_rows"] > 1000 and "email" in a["columns"] and "Enron" in a["name_hint"]
    assert any(d["column"] == "reply" for d in a["plan"]["decisions"])
    ds = client.get("/api/typically/datasets").json()["datasets"]
    assert {d["token"] for d in ds if d["sample"]} == {"sample", "sample-enron"}
    assert tp.load_records({"kind": "sample"})[0].keys() != a["columns"]   # Northwind is still the default sample
