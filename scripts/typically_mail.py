"""typically: an email history -> a table the planner (typically_plan) reads unchanged, one row per message the owner received.

    records, stats, warnings = read_mail(data, owner=None)   # .mbox | .eml | .zip of .eml files or a maildir tree

Row = {"email": the case, then decisions}. The case is the message as the owner saw it in one text column (from, to, when,
subject, the sender's own words; quoted replies, forwarded originals, signatures cut; phone / card / account numbers masked):
separate short columns would be profiled as candidate decisions, ids or personal data. Decisions come only from what the
owner did AFTER the message: reply (yes/no), reply_within (replied rows only: a nested yes/no would read as leakage of 'reply'),
forward (yes/no), forward_to, folder (where they filed it; system folders like inbox / deleted are no answer).
A decision with one answer on >= 95% of rows is dropped with a warning.
Tests: tests/test_typically_mail.py
"""
import email
import email.utils
import html
import io
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from pathlib import PurePosixPath

import typically_plan as tp

MAX_MESSAGES = 50_000   # ponytail: everything is parsed in memory; a bigger export needs a streaming reader per zip member
MAX_UNZIPPED = 1_000_000_000   # a zip that inflates past this is refused (zip bomb)
MAX_BODY = 1500
THREAD_GAP = timedelta(days=14)   # subject fallback: a quiet thread this long is closed; the same subject later is a new thread
DEGENERATE, MIN_ANSWERS = 0.95, 30   # a decision needs answers on >= 30 rows, none of them on >= 95%
SYSTEM_FOLDERS = re.compile(r"(\w+[ _])?inbox|.*sent.*|deleted([ _]items)?|trash|bin|drafts?|outbox|junk|spam|all[ _]?(mail|documents)|"
                            r"discussion_threads|notes|archived?|important|starred|unread|opened|category .*|chats?|calendar|contacts|tasks", re.I)
_SENT = re.compile(r"(^|[/_ ])sent", re.I)
_PREFIX = re.compile(r"^\s*((re|fw|fwd|aw|sv|tr|wg)\s*(\[\d+\])?\s*:\s*)+", re.I)
_FWD = re.compile(r"^\s*(fw|fwd|wg|tr)\s*:", re.I)
_BULK_FROM = re.compile(r"no-?reply|do-?not-?reply|mailer-daemon|postmaster|notifications?@|newsletter|bounce", re.I)
MAX_RECIPIENTS = 20   # more = a distribution list, not a message to decide on
# where the sender's own words end: quoted / forwarded originals, signature, legal footer
_QUOTE = re.compile(r"-{2,}\s*original message|-{2,}\s*forwarded|forwarded by |begin forwarded message|^on .{0,300}wrote:$", re.I)
_FWD_MARK = re.compile(r"forward", re.I)
_HDR = re.compile(r"^(from|to|cc|sent|date|subject)\s*:", re.I)
_FOOT = re.compile(r"^(--\s*$|_{10,}|\*{10,}|this (e-?mail|message) (and any|is|contains|may)|confidentiality notice|"
                   r"the information contained in this|sent from my )", re.I)
_ACCOUNT = re.compile(r"\b\d(?:-?\d){7,}\b")   # ponytail: 8+ digit runs only; spaced or shorter account numbers stay visible
TEXT = {   # warnings, in the language of the analysis
    "en": {"cap": "Only the first {n:,} messages were read.", "newest": "Kept the newest {n:,} of {m:,} received messages.",
           "drop": "Left out '{c}': {why}.", "few": "only {n} rows have an answer", "needs": "it depends on '{c}', which was left out",
           "lopsided": "'{v}' is the answer {pct}% of the time, too lopsided to learn from"},
    "he": {"cap": "נקראו רק {n:,} ההודעות הראשונות.", "newest": "נשמרו {n:,} ההודעות האחרונות מתוך {m:,} שהתקבלו.",
           "drop": "העמודה '{c}' הושמטה: {why}.", "few": "רק ל-{n} שורות יש תשובה", "needs": "היא תלויה ב-'{c}', שהושמטה",
           "lopsided": "'{v}' היא התשובה ב-{pct}% מהמקרים, חד-צדדי מדי כדי ללמוד ממנו"},
}
_DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()


@dataclass
class Msg:
    mid: str
    refs: list[str]
    date: datetime
    frm: str
    name: str
    to: list[str]
    cc: list[str]
    subject: str
    body: str
    named: list[tuple[str, str]] = field(default_factory=list)   # (display name, address) from To / Cc
    folders: set[str] = field(default_factory=set)
    bulk: bool = False
    out: bool = False


def _hdr(m, k: str) -> str:
    v = m.get(k)
    if v is None:
        return ""
    try:
        return re.sub(r"\s+", " ", str(make_header(decode_header(str(v))))).strip()
    except Exception:   # malformed encoded words: the raw header is better than nothing
        return re.sub(r"\s+", " ", str(v)).strip()


def _addrs(m, *keys: str) -> list[tuple[str, str]]:
    vals = [_hdr(m, k) for k in keys]
    return [(n.strip(" '\""), a.lower()) for n, a in email.utils.getaddresses([v for v in vals if v]) if "@" in a]


def _text(m) -> str:
    """The first text/plain part (or text/html with the tags taken out)."""
    parts = [p for p in m.walk() if p.get_content_maintype() == "text" and not p.get_filename()]
    for want in ("plain", "html"):
        for p in parts:
            if p.get_content_subtype() == want:
                raw = p.get_payload(decode=True) or b""
                try:
                    s = raw.decode(p.get_content_charset() or "utf-8", "replace")
                except LookupError:
                    s = raw.decode("latin-1")
                if want == "html":
                    s = re.sub(r"(?is)<(script|style).*?</\1>", "", s)
                    s = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", s)))
                return s.replace("\r\n", "\n")
    return ""


def clean_body(s: str, _again: bool = True) -> str:
    """The sender's own words: cut at the first quoted / forwarded original or signature, '>' lines dropped. A bare
    forward ("FYI" + the original) keeps the forwarded text, cleaned the same way, so the case is not empty."""
    lines = [l for l in s.split("\n") if not l.lstrip().startswith(">")]
    cut = mark = len(lines)
    fwd = False
    for i, l in enumerate(lines):
        t = l.strip()
        if _QUOTE.search(t) or (_HDR.match(t) and sum(bool(_HDR.match(x.strip())) for x in lines[i + 1:i + 5]) >= 1):
            cut = mark = i
            fwd = bool(_FWD_MARK.search(t))
            while cut and (not lines[cut - 1].strip() or re.search(r"@|\d{1,2}/\d{1,2}/\d{2,4}", lines[cut - 1])) and i - cut < 3:
                cut -= 1   # Lotus Notes puts "Name@ORG" / "05/14/2001 04:39 PM" just above the quoted headers
            break
        if _FOOT.match(t):
            cut = i
            break
    own = "\n".join(lines[:cut]).strip()
    if fwd and _again and len(own) < 40:
        rest = [l for l in lines[mark + 1:] if not _HDR.match(l.strip())]
        own = (own + "\n\n[forwarded]\n" + clean_body("\n".join(rest), False)).strip()
    own = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", own))
    return own[:MAX_BODY].rstrip() + (" …" if len(own) > MAX_BODY else "")


def subject_key(s: str) -> str:
    k = re.sub(r"\s+", " ", _PREFIX.sub("", s)).strip().casefold()
    return "" if k in ("(no subject)", "no subject") else k


def _parse(raw: bytes, folder: str) -> Msg | None:
    m = email.message_from_bytes(raw)
    frm = _addrs(m, "from")
    try:
        date = email.utils.parsedate_to_datetime(_hdr(m, "date"))
    except (TypeError, ValueError, IndexError):
        return None
    if not frm or date is None:
        return None
    date = date if date.tzinfo else date.replace(tzinfo=timezone.utc)
    xfrom = re.sub(r"\s*<.*?>|@\w+$|\"", "", _hdr(m, "x-from")).strip()   # Enron: "Jeff Dasovich@ENRON", "Kean, Steven <...>"
    labels = [l.strip() for l in _hdr(m, "x-gmail-labels").split(",") if l.strip()]
    folders = {folder} if folder else set(labels) or {PurePosixPath(_hdr(m, "x-folder").replace("\\", "/")).name} - {""}
    bulk = (bool(m.get("list-id") or m.get("list-unsubscribe")) or _hdr(m, "precedence").lower() in ("bulk", "list", "junk")
            or _hdr(m, "auto-submitted").lower() not in ("", "no") or bool(_BULK_FROM.search(frm[0][1])))
    to, cc = _addrs(m, "to"), _addrs(m, "cc")
    return Msg(mid=_hdr(m, "message-id"), refs=re.findall(r"<[^>]+>", _hdr(m, "in-reply-to") + " " + _hdr(m, "references")),
               date=date, frm=frm[0][1], name=frm[0][0] or xfrom, to=[a for _, a in to], cc=[a for _, a in cc], named=to + cc,
               subject=_hdr(m, "subject"), body=_text(m), folders=folders, bulk=bulk or len(to) + len(cc) > MAX_RECIPIENTS)


def _mbox(data: bytes):
    for chunk in re.split(rb"(?:\A|\r?\n)From [^\r\n]*\r?\n", data):
        if chunk.strip():
            yield chunk


def _raw(data: bytes):
    """(folder, raw message) for every message in an mbox, an .eml or a zip of either / of a maildir tree."""
    if data[:2] == b"PK":
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise ValueError("could not read that zip file")
        files = [i for i in z.infolist() if not i.is_dir() and "__MACOSX" not in i.filename and not PurePosixPath(i.filename).name.startswith(".")]
        if sum(i.file_size for i in files) > MAX_UNZIPPED:
            raise ValueError("that zip unpacks to more than 1 GB; export one folder or a shorter date range")
        dirs = [PurePosixPath(i.filename).parent.parts for i in files]
        root = len(os.path.commonprefix(dirs)) if dirs else 0   # folders = the paths below the directory every file shares
        for i, d in zip(files, dirs):
            body = z.read(i)
            parts = [p for p in d[root:] if p not in ("cur", "new", "tmp")]
            if body.startswith(b"From ") or i.filename.lower().endswith(".mbox"):
                for chunk in _mbox(body):
                    yield PurePosixPath(i.filename).stem, chunk
            else:
                yield "/".join(parts), body
    elif data.startswith(b"From "):
        yield from (("", c) for c in _mbox(data))
    else:
        yield "", data


def _pretty(a: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"[._]+", a.split("@")[0]) if w)


def messages(data: bytes) -> tuple[list[Msg], bool]:
    """Every message, once (the same message filed in several folders keeps all its folders); True when MAX_MESSAGES cut it."""
    msgs, seen = [], {}
    for n, (folder, raw) in enumerate(_raw(data)):
        if n == MAX_MESSAGES:
            return msgs, True
        if not (m := _parse(raw, folder)):
            continue
        key = (m.frm, m.date, subject_key(m.subject), m.body[:200])
        if key in seen:
            seen[key].folders |= m.folders
            continue
        seen[key] = m
        msgs.append(m)
    return msgs, False


def owners_of(msgs: list[Msg], owner: str | None = None) -> set[str]:
    """`owner` (comma-separated) or the sender of the sent folders (with aliases >= 1% as often), else the top recipient; plus
    addresses that look like the same person (first initial + surname: vince.kaminski -> vkamins@, vkaminski@aol).
    Sets Msg.out. ponytail: a name-shaped guess; a wrong owner is fixed in the UI by typing the addresses."""
    sent = lambda m: any(_SENT.search(f) for f in m.folders)
    froms = Counter(m.frm for m in msgs if sent(m))
    if owner:
        owners = {a.strip().lower() for a in owner.split(",") if "@" in a}
    elif froms:
        owners = {a for a, k in froms.items() if k >= 0.01 * froms.most_common(1)[0][1]}
    else:
        owners = {Counter(a for m in msgs for a in m.to + m.cc).most_common(1)[0][0]}
    if not owner:
        top = max(owners, key=lambda a: sum(m.frm == a for m in msgs))
        words = [re.sub(r"[^a-z]", "", w) for w in re.split(r"[._-]+", top.split("@")[0])]
        if len(words) > 1 and len(words[-1]) >= 4:
            letters = lambda a: re.sub(r"[^a-z]", "", a.split("@")[0])
            owners |= {a for m in msgs for a in (m.frm, *m.to, *m.cc) if words[-1][:6] in letters(a)
                       and (letters(a)[:1] == words[0][:1] or a in froms)}   # froms: a rare sent-folder alias (j..kaminski@)
    for m in msgs:
        m.out = m.frm in owners or sent(m)
    return owners


def outcomes(msgs: list[Msg], owners: set[str]) -> tuple[list[dict], list[tuple]]:
    """-> (threads, [(received msg, the owner's first reply or None, the owner's forward of it or None)]).
    Threads: In-Reply-To / References when they resolve, else the same subject among overlapping people (not counting the
    owner) within THREAD_GAP; the owner's own forward joins the latest live thread of its subject.
    ponytail: subject + people + time is a heuristic; two parallel threads with one subject and shared people merge."""
    msgs = sorted(msgs, key=lambda m: (m.date, m.out))
    by_id, open_, threads = {}, defaultdict(list), []
    for m in msgs:
        people = {m.frm, *m.to, *m.cc} - owners
        t = next((by_id[r] for r in reversed(m.refs) if r in by_id), None)
        if t is None and (k := subject_key(m.subject)):
            live = [t for t in open_[k] if m.date - t["last"] <= THREAD_GAP]
            t = next((t for t in reversed(live) if (m.out and _FWD.match(m.subject)) or t["people"] & people), None)
        if t is None:
            t = {"people": set(), "msgs": []}
            threads.append(t)
            open_[subject_key(m.subject)].append(t)
        t["people"] |= people
        t["last"] = m.date
        t["msgs"].append(m)
        if m.mid:
            by_id[m.mid] = t
    found = []
    for t in threads:
        for i, m in enumerate(t["msgs"]):
            if m.out or m.bulk:
                continue
            later = t["msgs"][i + 1:]
            reply = next((x for x in later if x.out and not _FWD.match(x.subject)), None)
            nxt = next((j for j, x in enumerate(later) if not x.out), len(later))   # a forward forwards the latest message before it
            fwd = next((x for x in later[:nxt] if x.out and _FWD.match(x.subject) and set(x.to + x.cc) - owners), None)   # not to self
            found.append((m, reply, fwd))
    return threads, sorted(found, key=lambda f: f[0].date)


def read_mail(data: bytes, owner: str | None = None, lang: str = "en") -> tuple[list[dict], dict, list[str]]:
    """-> (rows, {owner, messages, threads, inbound}, warnings in `lang`). ValueError carries a message fit for the user."""
    say = lambda key, **kw: TEXT.get(lang, TEXT["en"])[key].format(**kw)
    msgs, capped = messages(data)
    warnings = [say("cap", n=MAX_MESSAGES)] if capped else []
    if not msgs:
        raise ValueError("no email messages found; upload an .mbox, an .eml, or a .zip of .eml files or a mail folder")
    owners = owners_of(msgs, owner)
    names: dict[str, Counter] = defaultdict(Counter)
    for m in msgs:
        for n, a in [(m.name, m.frm), *m.named]:
            if n and "@" not in n:
                names[a][n] += 1
    who = lambda a: names[a].most_common(1)[0][0] if names[a] else _pretty(a)
    threads, found = outcomes(msgs, owners)
    if not found:
        raise ValueError("no messages to the mailbox owner found (only sent or bulk mail); is the owner right?")
    if len(found) > tp.MAX_ROWS:
        warnings.append(say("newest", n=tp.MAX_ROWS, m=len(found)))
        found = found[-tp.MAX_ROWS:]

    speed = _speed_labels([(r.date - m.date).total_seconds() for m, r, _ in found if r])
    target = lambda f: next((who(a) for a in f.to + f.cc if a not in owners), "") if f else ""
    top_fwd = {n for n, _ in Counter(target(f) for _, _, f in found if target(f)).most_common(6)}
    filed = lambda m: next((n for n in sorted(PurePosixPath(f).name for f in m.folders) if not SYSTEM_FOLDERS.fullmatch(n)), "")
    top_folders = {f for f, _ in Counter(filed(m) for m, _, _ in found if filed(m)).most_common(8)}
    rows = []
    for m, r, f in found:
        rows.append({
            "email": _case(m, owners, who),
            "reply": "yes" if r else "no",
            "reply_within": speed((r.date - m.date).total_seconds()) if r else "",   # when they replied: how fast
            "forward": "yes" if f else "no",
            "forward_to": "" if not (x := target(f)) else x if x in top_fwd else "someone else",
            "folder": "" if not (x := filed(m)) else x if x in top_folders else "other",
        })
    for col in list(rows[0])[1:]:
        vals = Counter(r[col] for r in rows if r[col])
        tot = sum(vals.values())
        top, k = vals.most_common(1)[0] if vals else ("", 0)
        why = (say("few", n=tot) if tot < MIN_ANSWERS
               else say("lopsided", v=top, pct=round(100 * k / tot)) if k >= DEGENERATE * tot
               else say("needs", c=dep) if (dep := {"reply_within": "reply", "forward_to": "forward"}.get(col)) and dep not in rows[0] else None)
        if why:
            warnings.append(say("drop", c=col, why=why))
            for r in rows:
                del r[col]
    stats = {"owner": sorted(owners, key=lambda a: -sum(m.frm == a for m in msgs))[0], "messages": len(msgs),
             "threads": len(threads), "inbound": len(rows)}
    return rows, stats, warnings


_SPEED = [(3600, "an hour"), (86400, "a day"), (7 * 86400, "a week"), (math.inf, "")]


def _speed_labels(delays: list[float], min_share: float = 0.25):
    """delay -> "within an hour" / "within a day" / ... / "after a week": buckets merged from the fastest until each holds
    >= 25% of the replies (a near-empty class teaches nothing)."""
    n = len(delays)
    groups, count, lo = [], 0, 0
    for hi, _ in _SPEED:
        count += sum(lo <= d < hi for d in delays)
        lo = hi
        if count >= min_share * n or hi == math.inf:
            groups.append([hi, count])
            count = 0
    if len(groups) > 1 and groups[-1][1] < min_share * n:
        groups.pop()
        groups[-1][0] = math.inf
    word = dict(_SPEED)
    label = lambda i: ("within " + word[groups[i][0]] if groups[i][0] != math.inf
                       else "after " + word[groups[i - 1][0]] if i else "replied")
    return lambda d: label(next(i for i, (hi, _) in enumerate(groups) if d < hi))


def _case(m: Msg, owners: set[str], who) -> str:
    to, cc = [a for a in m.to if a not in owners], [a for a in m.cc if a not in owners]
    others = len(to) + len(cc)
    if set(m.to) & owners:
        line = "To: you" + (f" and {others} other{'s' * (others > 1)}" if others else "")
    elif set(m.cc) & owners:
        line = f"Cc: you (sent to {len(to)} other{'s' * (len(to) != 1)})"
    else:
        line = f"To: {others} {'person' if others == 1 else 'people'} (not you by name)" if others else "To: you (bcc)"
    dom = m.frm.split("@")[1]
    head = [f"From: {m.name if m.name and '@' not in m.name else who(m.frm)} ({dom})", line,
            f"Received: {_DAYS[m.date.weekday()]} {m.date:%H:%M}", f"Subject: {m.subject or '(no subject)'}"]
    body = clean_body(m.body)
    text = "\n".join(head) + ("\n\n" + body if body else "")
    return _ACCOUNT.sub("[number]", tp.mask(text))

