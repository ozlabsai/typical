"""typically: read a table, profile it, and plan how to teach a model from it (the DatasetPlan of
.context/typically/FLOW.md).

    records = load_records({"kind": "sample"})            # csv | xlsx | hf | sheets | sample -> [dict[str, str]]
    prof = profile(records)                               # facts about the data, no opinions
    plan, source, warnings = llm_plan(prof, sample_rows(prof, records), key, records)   # or heuristic_plan(prof, records)
    render_case(record, plan), answers(record, plan)      # what the model will see / be taught

Every plan, LLM or heuristic, goes through validate_plan: mapping keys must be values that are really in the data.
Self-check: uv run python scripts/typically_plan.py
"""
import base64
import csv
import datetime as dt
import io
import json
import os
import random
import re
import statistics
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import typically_llm as tllm

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO / "site" / "data" / "typically_sample.csv"
MAX_ROWS, MAX_VALUES, MAX_CANDIDATES = 20_000, 200, 12
MAX_PROMPT_TOKENS = 30_000
MAX_CASE_TOKENS = 1024

# ---------------------------------------------------------------- loading

_SHEET = re.compile(r"/spreadsheets/d/([\w-]+)")
_GID = re.compile(r"[#&?]gid=(\d+)")


def sheets_export_url(url: str) -> str:
    m = _SHEET.search(url)
    if not m:
        raise ValueError("that does not look like a Google Sheets link (docs.google.com/spreadsheets/d/...)")
    g = _GID.search(url)
    return f"https://docs.google.com/spreadsheets/d/{m[1]}/export?format=csv&gid={g[1] if g else 0}"


def _parse_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k and k.strip()} for r in islice(reader, MAX_ROWS)]
    if not rows:
        raise ValueError("the file has no rows")
    return rows


def _cell(v) -> str:
    return "" if v is None else v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)


def _xcell(v) -> str:
    if isinstance(v, dt.datetime) and v.time() == dt.time():
        v = v.date()   # Excel stores dates as midnight datetimes
    if isinstance(v, (dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return _cell(v).strip()


def read_xlsx(data_base64: str, sheet: str | None = None) -> tuple[list[str], list[dict]]:
    """(sheet names, records of `sheet` or the first sheet). Header = the first non-empty row; columns without a header are dropped."""
    import openpyxl
    from openpyxl.utils.exceptions import InvalidFileException
    from zipfile import BadZipFile
    try:
        wb = openpyxl.load_workbook(io.BytesIO(base64.b64decode(data_base64 or "", validate=True)), read_only=True, data_only=True)
    except (InvalidFileException, BadZipFile, KeyError, ValueError):
        raise ValueError("could not read that Excel file; save it as .xlsx (or CSV) and try again")
    try:
        if sheet and sheet not in wb.sheetnames:
            raise ValueError(f"the workbook has no sheet {sheet!r}")
        rows = (r for r in wb[sheet or wb.sheetnames[0]].iter_rows(values_only=True) if any(c not in (None, "") for c in r))
        header = [_xcell(c) for c in next(rows, ())]
        records = [{h: _xcell(v) for h, v in zip(header, r) if h} for r in islice(rows, MAX_ROWS)]
    finally:
        wb.close()
    if not records:
        raise ValueError("the sheet has no rows")
    return wb.sheetnames, records


def load_records(source: dict) -> list[dict]:
    """source: {kind: csv, text} | {kind: xlsx, data_base64, sheet?} | {kind: hf, dataset, config?, split?, limit?} | {kind: sheets, url} | {kind: sample}.
    All values become stripped strings, at most 20k rows; ValueError carries a message fit for the user."""
    kind = source.get("kind")
    if kind == "csv":
        return _parse_csv(source.get("text") or "")
    if kind == "xlsx":
        return read_xlsx(source.get("data_base64") or "", source.get("sheet"))[1]
    if kind == "sample":
        return _parse_csv(SAMPLE_CSV.read_text())
    if kind == "sheets":
        try:
            with urllib.request.urlopen(sheets_export_url(source.get("url") or ""), timeout=30) as r:
                body, ctype = r.read(50_000_000).decode("utf-8", "replace"), r.headers.get("content-type", "")
        except urllib.error.URLError as e:
            raise ValueError(f"could not open the sheet ({e}); is it shared as 'Anyone with the link can view'?")
        if "csv" not in ctype:   # a private sheet answers with the Google sign-in page
            raise ValueError("the sheet is not public: share it as 'Anyone with the link can view' and try again")
        return _parse_csv(body)
    if kind == "hf":
        from datasets import load_dataset   # heavy import, only when needed
        limit = min(int(source.get("limit") or 5000), MAX_ROWS)
        try:   # public datasets only, unless the server has an HF_TOKEN
            ds = load_dataset(source["dataset"], source.get("config") or None, split=source.get("split") or "train",
                              streaming=True, token=os.environ.get("HF_TOKEN") or None)
            # ponytail: datasets are often sorted by label; a 10k-row shuffle buffer keeps the first `limit` rows representative
            rows = [{k: _cell(v).strip() for k, v in r.items()} for r in islice(ds.shuffle(seed=0, buffer_size=10_000), limit)]
        except Exception as e:   # datasets raises a zoo of types; the message is what the user needs
            raise ValueError(f"could not load Hugging Face dataset {source.get('dataset')!r}: {str(e)[:300]}")
        if not rows:
            raise ValueError("the dataset split has no rows")
        return rows
    raise ValueError(f"unknown source kind {kind!r}")


def name_hint(source: dict) -> str:
    kind = source.get("kind")
    if kind == "hf":
        return source["dataset"].split("/")[-1]
    if kind in ("csv", "xlsx"):
        return source.get("name") or "upload"
    return {"sheets": "sheet", "sample": "Northwind"}.get(kind, "data")


# ---------------------------------------------------------------- profiling

_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}.*)?|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}( .*)?")
YES = {"yes", "y", "true", "t", "1", "si", "oui", "ja"}
NO = {"no", "n", "false", "f", "0", "non", "nein"}
RANK = ["none", "trivial", "minor", "low", "normal", "medium", "med", "moderate", "high", "major", "urgent", "severe", "critical", "blocker"]
SEVERITY = {"p", "sev", "severity", "priority", "prio", "pri", "s"}   # P1 / Sev1: 1 is the MOST severe
PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"),
    "card": re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}\b"),
    "phone": re.compile(r"(?<![\w.-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}(?![\w-])|\+\d{10,15}\b"),
}
STOPWORDS = {
    "en": "the and to of is in it you that for on with this are was be have not".split(),
    "es": "el la de que y en los del las por con para una un no se".split(),
    "fr": "le la les de des et en un une est pour que dans pas ce qui sur".split(),
    "de": "der die das und ist nicht ein eine zu den mit für auf ich sie".split(),
}
SCRIPTS = {"he": re.compile(r"[֐-׿]"), "ar": re.compile(r"[؀-ۿ]")}


def norm(v) -> str:
    """The identity of a value: stripped, casefolded, punctuation collapsed ("Sev-1" -> "sev 1"); numbers stay as written."""
    v = str(v).strip()
    if _NUMBER.fullmatch(v):
        return v.lstrip("+")
    return re.sub(r"[\W_]+", " ", v.casefold()).strip() or v.casefold()


def mask(text: str) -> str:
    for name, rx in PII.items():
        text = rx.sub(f"[{name}]", text)
    return text


def guess_lang(text: str) -> str:
    words = re.findall(r"[^\W\d_]+", text.casefold())
    if not words:
        return "und"
    letters = sum(map(len, words))
    for code, rx in SCRIPTS.items():
        if len(rx.findall(text)) > 0.3 * letters:
            return code
    hits = {code: sum(w in sw for w in words) for code, sw in ((c, set(s)) for c, s in STOPWORDS.items())}
    best = max(hits, key=hits.get)
    return best if hits[best] >= 2 else "und"


def _ordinal(values: list[str]) -> dict | None:
    """Observed (normalised) values -> {labels: low..high, why} when they form a scale. Order is LEAST to MOST."""
    if len(values) >= 3 and all(v in RANK for v in values):
        return {"labels": sorted(values, key=RANK.index), "why": msg("en", "why_words"), "how": ("why_words", {})}
    m = [re.fullmatch(r"([a-z]+) ?(\d+)", v) for v in values]
    if len(values) >= 2 and all(m) and len({x[1] for x in m}) == 1:
        sev = m[0][1] in SEVERITY
        labels = [x[0] for x in sorted(zip(values, m), key=lambda t: int(t[1][2]), reverse=sev)]
        how = ("why_sev" if sev else "why_more", {"p": m[0][1].upper()})
        return {"labels": labels, "why": msg("en", how[0], **how[1]), "how": how}
    if 3 <= len(values) <= 10 and all(re.fullmatch(r"-?\d+", v) for v in values):
        return {"labels": sorted(values, key=int), "why": msg("en", "why_whole"), "how": ("why_whole", {})}
    return None


def _column(vals: list[str]) -> dict:
    ne = [v for v in (x.strip() for x in vals) if v]
    n, m = len(vals), len(ne)
    norms = [norm(v) for v in ne]
    counts = Counter(norms)
    raws: dict[str, Counter] = defaultdict(Counter)
    for v, k in zip(ne, norms):
        raws[k][v] += 1
    distinct, ratio = len(counts), len(counts) / max(m, 1)
    lens = sorted(len(v) for v in ne) or [0]
    med_len = statistics.median(lens)
    med_words = statistics.median(len(v.split()) for v in ne) if ne else 0
    numeric = m > 0 and all(_NUMBER.fullmatch(v.replace(",", "")) for v in ne)
    if distinct <= 1:
        kind = "constant"
    elif sum(bool(_DATE.fullmatch(v)) for v in ne) >= 0.95 * m:
        kind = "date"
    elif med_len > 40 or med_words >= 6:
        kind = "text"
    elif ratio >= 0.9 and m >= 20 and med_words < 3 and not (numeric and any("." in v for v in ne)):
        kind = "id-like"
    elif ratio >= 0.9 and m >= 20 or (med_words >= 3 and distinct > 50):
        kind = "text"
    elif numeric and distinct > 20:
        kind = "numeric"
    else:
        kind = "categorical"
    show = kind in ("categorical", "numeric", "constant")   # value counts of free text / ids are all 1s: noise
    return {
        "kind": kind, "non_empty": m, "empty_rate": round(1 - m / max(n, 1), 4),
        "distinct": distinct, "distinct_ratio": round(ratio, 4),
        "median_len": med_len, "p95_len": lens[int(0.95 * (len(lens) - 1))],
        "values": [{"value": v, "count": c, "raws": [r for r, _ in raws[v].most_common(3)]}
                   for v, c in counts.most_common(MAX_VALUES)] if show else [],
        "values_complete": distinct <= MAX_VALUES,
        "pii": {k: round(sum(bool(rx.search(v)) for v in ne) / m, 4) for k, rx in PII.items()
                if m and any(rx.search(v) for v in ne)},
    }


def _purity(xs: list, ys: list) -> float:
    """How well the majority label per x value predicts y (1.0 = x gives the answer away)."""
    g: dict = defaultdict(Counter)
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    for x, y in pairs:
        g[x][y] += 1
    return sum(max(c.values()) for c in g.values()) / max(len(pairs), 1)


def _decision_stats(c: str, cols: dict, N: dict, keys: list, texts: dict) -> dict:
    ans = N[c]
    counts = Counter(a for a in ans if a is not None)
    m = sum(counts.values())
    base = max(counts.values()) / m
    groups: dict = defaultdict(Counter)
    for k, a in zip(keys, ans):
        if k and a is not None:
            groups[k][a] += 1
    conflict_rows = sum(sum(g.values()) for g in groups.values() if len(g) > 1)
    leaks = []
    for x, p in cols.items():
        if x != c and x in N and p["distinct"] >= 2 and p["distinct_ratio"] <= 0.2:
            pur = _purity(N[x], ans)
            if pur >= 0.95 and pur - base >= 0.5 * (1 - base):   # removes at least half the error left by guessing the majority
                leaks.append({"column": x, "purity": round(pur, 4)})
    text_leaks = []
    for x, ts in texts.items():   # ts covers only the first rows: zip stops there
        rows = [(a, t) for a, t in zip(ans, ts) if a and len(a) >= 4 and a not in YES | NO and not _NUMBER.fullmatch(a)]
        if len(rows) >= 20:
            share = sum(f" {a} " in t for a, t in rows) / len(rows)
            if share >= 0.3:
                text_leaks.append({"column": x, "share": round(share, 4)})
    values = list(counts)
    return {
        "yes_no": set(values) <= YES | NO and bool(set(values) & YES) and bool(set(values) & NO),
        "ordinal": _ordinal(values),
        "majority_share": round(base, 4),
        "rare": [v for v, k in counts.items() if k < max(10, 0.01 * m)],
        "empty": len(ans) - m,
        "conflicts": {"rows": conflict_rows, "share": round(conflict_rows / m, 4),
                      "best_accuracy": round(sum(max(g.values()) for g in groups.values()) / max(sum(sum(g.values()) for g in groups.values()), 1), 4)},
        "leaks": leaks, "text_leaks": text_leaks,
    }


def profile(records: list[dict]) -> dict:
    n = len(records)
    names = list(dict.fromkeys(k for r in records for k in r))
    cols = {c: _column([r.get(c, "") for r in records]) for c in names}
    text_cols = [c for c, p in cols.items() if p["kind"] == "text"]
    keys = [" ".join(norm(r.get(c, "")) for c in text_cols or names) for r in records]   # the case, normalised
    seen = Counter(keys)
    dup_rows = sum(v - 1 for v in seen.values() if v > 1)
    sample = records[:2000]
    langs = Counter(l for l in (guess_lang(" ".join(r.get(c, "") for c in text_cols)[:1000]) for r in sample) if l != "und")
    cand = [c for c, p in cols.items() if p["kind"] == "categorical" and 2 <= p["distinct"] <= 50][:MAX_CANDIDATES]
    # ponytail: leakage/text-leak scans look at the first 3000 rows of text; O(cands x columns x rows) otherwise
    N = {c: [norm(r[c]) if r.get(c, "").strip() else None for r in records] for c, p in cols.items() if p["kind"] in ("categorical", "numeric")}
    texts = {c: [f" {norm(r.get(c, ''))} " for r in records[:3000]] for c in text_cols}
    return {
        "n_rows": n, "text_columns": text_cols, "columns": cols,
        "duplicates": {"rows": dup_rows, "share": round(dup_rows / max(n, 1), 4)},
        "languages": [{"code": c, "share": round(k / sum(langs.values()), 4)} for c, k in langs.most_common(5)],
        "decisions": {c: _decision_stats(c, cols, N, keys, texts) for c in cand},
    }


# ---------------------------------------------------------------- plans

LANGS = {"en": "English", "he": "Hebrew"}
# Every user-facing string of a heuristic plan / warning, per language. Column names, labels and values are interpolated untouched.
TEXT = {
    "en": {
        "q_noul": "Should we {h}?", "q_score": "What {h} level?", "q_choice": "Which {h}?",
        "r_yesno": "Answers look like yes/no ({vals}).",
        "r_scale": "Answers form a scale ({why}); ordered {lo} to {hi}.",
        "r_choice": "{n} different answers, none that look like yes/no or a scale.",
        "why_words": "low-to-high words", "why_whole": "small whole-number scale",
        "why_sev": "{p} numbers: 1 is the most severe", "why_more": "{p} numbers: higher is more",
        "x_constant": "Every row has the same value.", "x_timestamp": "A date or time; the decision is not made from it.",
        "x_near_unique": "Almost every row has a different value.", "x_pii": "Looks like personal data ({kinds}).",
        "x_after_decision": "Reads like something written after the decision was made.",
        "x_leakage": "Predicts a decision almost perfectly on its own.", "x_other": "The plan did not use this column.",
        "p_body": "Free text: the case itself.", "p_fact": "Short column kept as context.", "p_longest": "The longest column, used as the case.",
        "unmapped": "Answer {v!r} was not mapped; it is ignored until you map it.",
        "i_dup": "{n} rows repeat another case.", "a_dup": "merge into soft answers",
        "i_conflict": "The same case has different answers; expect about {pct}% at best.", "a_conflict": "review or merge into soft answers",
        "i_rare": "Rare answers: {vals}.", "a_rare": "balance rare answers",
        "i_empty": "Rows with no answer are not used.", "a_empty": "skip those rows",
        "i_leak": "Predicts '{c}' {pct}% of the time by itself.", "a_leak": "remove from the case",
        "i_textleak": "The text often contains the answer to '{c}' word for word.", "a_textleak": "review",
        "i_pii": "Contains {kinds}.", "a_pii": "review or remove",
        "i_long": "{n} cases are longer than {t} tokens; the end is cut.", "a_long": "truncate",
        "w_nodecision": "No column looks like a decision (2-20 different answers); add one in the next step.",
        "w_big": "The data profile is too large for AI analysis; used the built-in rules instead.",
        "w_fail": "AI analysis failed ({err}); used the built-in rules instead.",
        "w_invalid": "The AI plan did not pass validation ({err}); used the built-in rules instead.",
    },
    "he": {
        "q_noul": "האם {h}?", "q_score": "מה רמת ה{h}?", "q_choice": "איזה {h} מתאים?",
        "r_yesno": "התשובות נראות כמו כן/לא ({vals}).",
        "r_scale": "התשובות יוצרות סולם ({why}); מסודרות מ-{lo} עד {hi}.",
        "r_choice": "{n} תשובות שונות, אף אחת מהן לא נראית כמו כן/לא או כסולם.",
        "why_words": "מילים מנמוך לגבוה", "why_whole": "סולם קטן של מספרים שלמים",
        "why_sev": "מספרי {p}: 1 הוא החמור ביותר", "why_more": "מספרי {p}: גבוה יותר פירושו יותר",
        "x_constant": "לכל השורות אותו ערך.", "x_timestamp": "תאריך או שעה; ההחלטה לא מתקבלת על סמך זה.",
        "x_near_unique": "כמעט לכל שורה ערך שונה.", "x_pii": "נראה כמו מידע אישי ({kinds}).",
        "x_after_decision": "נראה כמו משהו שנכתב אחרי שההחלטה התקבלה.",
        "x_leakage": "מנבא החלטה כמעט בדיוק גם לבדו.", "x_other": "התוכנית לא השתמשה בעמודה הזו.",
        "p_body": "טקסט חופשי: המקרה עצמו.", "p_fact": "עמודה קצרה שנשמרה כהקשר.", "p_longest": "העמודה הארוכה ביותר, משמשת כמקרה.",
        "unmapped": "התשובה {v!r} לא מופתה; היא תתעלם עד שתמפו אותה.",
        "i_dup": "{n} שורות חוזרות על מקרה אחר.", "a_dup": "לאחד לתשובות רכות",
        "i_conflict": "לאותו מקרה יש תשובות שונות; צפו לדיוק של כ-{pct}% לכל היותר.", "a_conflict": "לבדוק או לאחד לתשובות רכות",
        "i_rare": "תשובות נדירות: {vals}.", "a_rare": "לאזן תשובות נדירות",
        "i_empty": "שורות ללא תשובה אינן בשימוש.", "a_empty": "לדלג על השורות האלה",
        "i_leak": "מנבא את '{c}' ב-{pct}% מהמקרים גם לבדו.", "a_leak": "להסיר מהמקרה",
        "i_textleak": "הטקסט מכיל לעיתים קרובות את התשובה ל-'{c}' מילה במילה.", "a_textleak": "לבדוק",
        "i_pii": "מכיל {kinds}.", "a_pii": "לבדוק או להסיר",
        "i_long": "{n} מקרים ארוכים מ-{t} טוקנים; הסוף נחתך.", "a_long": "לחתוך",
        "w_nodecision": "אף עמודה לא נראית כהחלטה (2-20 תשובות שונות); הוסיפו אחת בשלב הבא.",
        "w_big": "פרופיל הנתונים גדול מדי לניתוח בבינה מלאכותית; נעשה שימוש בכללים המובנים.",
        "w_fail": "הניתוח בבינה מלאכותית נכשל ({err}); נעשה שימוש בכללים המובנים.",
        "w_invalid": "התוכנית של הבינה המלאכותית לא עברה בדיקה ({err}); נעשה שימוש בכללים המובנים.",
    },
}


def msg(lang: str, key: str, **kw) -> str:
    return TEXT.get(lang, TEXT["en"])[key].format(**kw)


def _human(col: str) -> str:
    return re.sub(r"[\W_]+", " ", col).strip().lower()


_ID_NAME = re.compile(r"(^|[\W_])(id|uuid|guid|key|ref|number|num|sku)$|^id($|[\W_])", re.I)
_AFTER_NAME = re.compile(r"response|reply|answer|resolution|outcome|solution|completion|output", re.I)   # written after the decision


def _labels(col_profile: dict) -> dict[str, str]:
    return {v["value"]: v["raws"][0] for v in col_profile["values"]}   # normalised -> most common spelling


def _heuristic_decision(c: str, p: dict, d: dict, lang: str = "en") -> dict:
    disp, h = _labels(p), _human(c)
    if d["yes_no"]:
        typ, labels = "noul", ["no", "yes"]
        mapping, conf = {v: "yes" if v in YES else "no" for v in disp}, 0.85
        reasons = [msg(lang, "r_yesno", vals=", ".join(disp.values()))]
    elif d["ordinal"]:
        typ, labels = "score", [disp[v] for v in d["ordinal"]["labels"]]
        mapping, conf = {v: disp[v] for v in disp}, 0.75
        key, kw = d["ordinal"]["how"]
        reasons = [msg(lang, "r_scale", why=msg(lang, key, **kw), lo=labels[0], hi=labels[-1])]
    else:
        typ, labels = "choice", sorted(disp.values(), key=str.casefold)
        mapping, conf = dict(disp), 0.6
        reasons = [msg(lang, "r_choice", n=len(labels))]
    question = msg(lang, "q_" + typ, h=h)
    return {"column": c, "include": True, "question": question, "type": typ, "labels": labels, "mapping": mapping,
            "confidence": conf, "reasons": reasons,
            "needs_review": bool(d["leaks"] or d["text_leaks"] or d["conflicts"]["share"] > 0.1)}


def heuristic_plan(prof: dict, records: list[dict] | None = None, lang: str = "en") -> dict:
    """The always-available plan: free text is the case body, short categorical columns (2-20 answers) are decisions,
    ids/dates/constants/pii are left out, other short columns are facts."""
    cols, cand = prof["columns"], prof["decisions"]
    parts, decisions, excluded = [], [], []
    ex = lambda c, why, key=None, **kw: excluded.append({"column": c, "why": why, "confidence": 0.7, "reason": msg(lang, "x_" + (key or why), **kw)})
    for c, p in cols.items():
        kind = p["kind"]
        if c in cand and p["distinct"] <= 20:
            decisions.append(_heuristic_decision(c, p, cand[c], lang))
        elif kind == "constant":
            ex(c, "constant")
        elif kind == "date":
            ex(c, "timestamp")
        elif kind == "id-like":
            ex(c, "id" if _ID_NAME.search(c) else "near_unique", "near_unique")
        elif p["pii"] and max(p["pii"].values()) >= 0.5 and kind != "text":
            ex(c, "pii", kinds=", ".join(p["pii"]))
        elif kind == "text" and _AFTER_NAME.search(c):
            ex(c, "after_decision")
        elif kind == "text":
            parts.append({"column": c, "role": "body", "sentence": None, "confidence": 0.7, "reason": msg(lang, "p_body")})
        else:
            parts.append({"column": c, "role": "fact", "sentence": None, "confidence": 0.5, "reason": msg(lang, "p_fact")})
    for part in [q for q in parts if q["role"] == "fact"]:   # a fact that gives a decision away is not a fact
        if any(l["column"] == part["column"] for d in decisions for l in cand[d["column"]]["leaks"]):
            parts.remove(part)
            ex(part["column"], "leakage")
    if not any(q["role"] == "body" for q in parts):
        facts = sorted((q for q in parts if q["role"] == "fact"), key=lambda q: -cols[q["column"]]["median_len"])
        if not facts:
            raise ValueError("no free-text column found to describe each case")
        facts[0].update(role="body", reason=msg(lang, "p_longest"))
    return _finish({"case": {"parts": parts, "max_tokens": MAX_CASE_TOKENS, "overflow": "truncate"},
                    "decisions": decisions, "excluded": excluded}, prof, records, lang)


def _finish(plan: dict, prof: dict, records: list[dict] | None, lang: str = "en") -> dict:
    """Fill the parts of the plan the data decides (issues, languages), never the model."""
    used = {q["column"] for q in plan["case"]["parts"]}
    issues, cand = [], prof["decisions"]
    dup = prof["duplicates"]
    if dup["rows"]:
        issues.append({"kind": "duplicates", "count": dup["rows"], "detail": msg(lang, "i_dup", n=dup["rows"]), "action": msg(lang, "a_dup")})
    for d in (d for d in plan["decisions"] if d["include"] and (records or d["column"] in cand)):
        c = d["column"]
        s = _label_stats(records, plan, d) if records else cand[c]   # labels after the plan's merges, not raw spellings
        if s["conflicts"]["rows"]:
            issues.append({"kind": "conflicts", "column": c, "count": s["conflicts"]["rows"],
                           "detail": msg(lang, "i_conflict", pct=round(100 * s["conflicts"]["best_accuracy"])), "action": msg(lang, "a_conflict")})
        if s["rare"]:
            issues.append({"kind": "rare_labels", "column": c, "count": len(s["rare"]), "detail": msg(lang, "i_rare", vals=", ".join(s["rare"][:5])), "action": msg(lang, "a_rare")})
        if s["empty"]:
            issues.append({"kind": "empty", "column": c, "count": s["empty"], "detail": msg(lang, "i_empty"), "action": msg(lang, "a_empty")})
        for l in cand[c]["leaks"] if c in cand else []:
            if l["column"] in used:
                issues.append({"kind": "leakage", "column": l["column"], "count": prof["n_rows"], "detail": msg(lang, "i_leak", c=c, pct=round(100 * l["purity"])), "action": msg(lang, "a_leak")})
        for l in cand[c]["text_leaks"] if c in cand else []:
            if l["column"] in used:
                issues.append({"kind": "leakage", "column": l["column"], "count": round(l["share"] * prof["n_rows"]), "detail": msg(lang, "i_textleak", c=c), "action": msg(lang, "a_textleak")})
    for c in used:
        if c in prof["columns"] and prof["columns"][c]["pii"]:
            p = prof["columns"][c]
            issues.append({"kind": "pii", "column": c, "count": round(max(p["pii"].values()) * p["non_empty"]), "detail": msg(lang, "i_pii", kinds=", ".join(p["pii"])), "action": msg(lang, "a_pii")})
    if records:
        long = sum(len(render_case(r, plan)) > 4 * plan["case"]["max_tokens"] for r in records)   # ~4 chars per token
        if long:
            issues.append({"kind": "long_cases", "count": long, "detail": msg(lang, "i_long", n=long, t=plan["case"]["max_tokens"]), "action": msg(lang, "a_long")})
    return {**plan, "issues": issues, "languages": prof["languages"]}


def _label_stats(records: list[dict], plan: dict, d: dict) -> dict:
    """rare / empty / conflicts of a decision as the model will be taught it: through the plan's mapping (so "y" and "yes" are one
    answer, a value mapped to null is no answer) and over the rendered case (two rows conflict when the model sees the same case)."""
    groups: dict = defaultdict(Counter)
    for r in records:
        if case := norm(render_case(r, plan)):   # an empty case is never trained on
            groups[case][answers(r, {"decisions": [d]})[d["column"]]] += 1
    empty = sum(g.pop(None, 0) for g in groups.values())
    counts = sum(groups.values(), Counter())
    m = sum(counts.values()) or 1
    return {"rare": [l for l, k in counts.items() if k < max(10, 0.01 * m)], "empty": empty,
            "conflicts": {"rows": sum(sum(g.values()) for g in groups.values() if len(g) > 1),
                          "best_accuracy": round(sum(max(g.values(), default=0) for g in groups.values()) / m, 4)}}


def validate_plan(plan: dict, prof: dict, lang: str = "en") -> list[str]:
    """Hard problems -> list of messages (empty = fine). Soft ones are fixed in place: an observed value with >= 1% of
    the rows that the plan forgot maps to null with needs_review; columns the plan never mentions are excluded."""
    cols, errs = prof["columns"], []
    parts, seen = plan["case"]["parts"], set()
    for q in parts:
        if q["column"] not in cols:
            errs.append(f"case part column {q['column']!r} is not in the data")
        if q["role"] not in ("fact", "body"):
            errs.append(f"case part {q['column']!r} has role {q['role']!r}; use fact or body")
        if q.get("sentence") and "{value}" not in q["sentence"]:
            errs.append(f"sentence for {q['column']!r} must contain {{value}}")
    if not any(q["role"] == "body" for q in parts):
        errs.append("the case needs at least one body column")
    part_cols = {q["column"] for q in parts}
    for d in plan["decisions"]:
        c = d["column"]
        if c not in cols:
            errs.append(f"decision column {c!r} is not in the data")
            continue
        if c in part_cols:
            errs.append(f"column {c!r} is both a decision and part of the case")
        if c in seen:
            errs.append(f"decision column {c!r} appears twice")
        seen.add(c)
        labels = d["labels"]
        if d["type"] not in ("choice", "noul", "score"):
            errs.append(f"decision {c!r} has type {d['type']!r}")
        if len(labels) < 2 or len(set(labels)) != len(labels):
            errs.append(f"decision {c!r} needs at least 2 distinct labels")
        if d["type"] == "noul" and labels != ["no", "yes"]:
            errs.append(f"decision {c!r} is yes/no: labels must be exactly [\"no\", \"yes\"]")
        obs = {v["value"]: v["count"] for v in cols[c]["values"]}
        if cols[c]["values_complete"] and obs:
            for k in d["mapping"]:
                if k not in obs:
                    errs.append(f"decision {c!r} maps {k!r}, which does not occur in the data (observed: {sorted(obs)[:20]})")
        for k, t in d["mapping"].items():
            if t is not None and t not in labels:
                errs.append(f"decision {c!r} maps {k!r} to {t!r}, which is not one of its labels {labels}")
        if d["include"]:
            for v, k in obs.items():
                if v not in d["mapping"] and k >= 0.01 * cols[c]["non_empty"]:
                    d["mapping"][v] = None
                    d["needs_review"] = True
                    d["reasons"] = [*d["reasons"], msg(lang, "unmapped", v=v)]
    for e in plan["excluded"]:
        if e["column"] not in cols:
            errs.append(f"excluded column {e['column']!r} is not in the data")
        elif e["column"] in part_cols or e["column"] in seen:
            errs.append(f"column {e['column']!r} is excluded and also used")
    if not any(d["include"] for d in plan["decisions"]):
        errs.append("the plan has no included decision")
    named = part_cols | seen | {e["column"] for e in plan["excluded"]}
    plan["excluded"] += [{"column": c, "why": "other", "confidence": 0.5, "reason": msg(lang, "x_other")}
                         for c in cols if c not in named]
    return errs


# ---------------------------------------------------------------- LLM plan

_STR = {"type": "string"}
_NULL_STR = {"anyOf": [_STR, {"type": "null"}]}
_obj = lambda **p: {"type": "object", "properties": p, "required": list(p), "additionalProperties": False}
_arr = lambda i: {"type": "array", "items": i}
_enum = lambda *v: {"type": "string", "enum": list(v)}
# The DatasetPlan of FLOW.md as the model fills it. Differences, all filled in by code: `mapping` is a list of
# pairs (structured outputs need fixed keys), max_tokens/overflow/issues/languages come from the data.
PLAN_SCHEMA = _obj(
    case=_obj(parts=_arr(_obj(column=_STR, role=_enum("fact", "body"), sentence=_NULL_STR, confidence={"type": "number"}, reason=_STR))),
    decisions=_arr(_obj(column=_STR, include={"type": "boolean"}, question=_STR, type=_enum("choice", "noul", "score"),
                        labels=_arr(_STR), mapping=_arr(_obj(value=_STR, label=_NULL_STR)), confidence={"type": "number"},
                        reasons=_arr(_STR), needs_review={"type": "boolean"})),
    excluded=_arr(_obj(column=_STR, why=_enum("id", "timestamp", "pii", "leakage", "after_decision", "near_unique", "constant", "other"),
                       confidence={"type": "number"}, reason=_STR)),
)

SYSTEM = """You plan how to teach a small decision model from a company's table of past cases. Each row is one case; some columns describe the case, some are decisions people made. You get a statistical profile of every column (with the complete list of normalised values and counts) and 25 example rows.

Return the plan as JSON matching the schema. Rules:
- Every column gets exactly one role: a case part (role "fact" = short context with a sentence template containing {value}, e.g. "The customer is on the {value} plan."; role "body" = the free text of the case, no sentence), a decision, or excluded (with why: id, timestamp, pii, leakage, after_decision, near_unique, constant, other).
- Decide whether each column is KNOWN AT DECISION TIME. A column written or filled in after the decision (resolution, agent reply, outcome, close reason) must be excluded as after_decision. A column that gives a decision away by itself (see the leakage numbers) must be excluded as leakage.
- Decisions are the columns the business wants predicted. Question: plain business English, e.g. "Should we escalate this ticket?", "Which team should handle it?", "How urgent is it?". type: "noul" for yes/no (labels exactly ["no","yes"]), "score" when the answers have an order (labels ordered from LEAST to MOST, so for P1..P4 where P1 is most severe the last label is P1), otherwise "choice".
- mapping lists observed normalised values (copy the "value" strings from the profile EXACTLY; never invent or retype one) with the label each one means. Merge synonyms and typos ("yes", "y", "true" -> "yes"; "biling" -> "billing") ONLY between values that are really observed. Map values that mean "no answer / unknown" to null. EVERY observed value of an included decision must appear, rare ones too: map a long tail you do not want to a label or to null explicitly, never leave it out.
- Labels for choice/score are short, clean, in the words of the data: the readable original spelling from "raws" ("Sales and Pre-Sales", not "sales and pre sales"; "very_low" becomes "very low").
- Set needs_review true and lower confidence when you are unsure; give one plain-English reason per decision ("we think X because Y").
- Never put a column in two roles. Do not leave a column out."""


def sample_rows(prof: dict, records: list[dict], k: int = 25) -> list[dict]:
    """k rows, stratified by the first candidate decision, rarest labels first so each gets >= 2; masked, cells cut to 300 chars."""
    rng = random.Random(0)
    dec = next(iter(prof["decisions"]), None)
    by: dict = defaultdict(list)
    for r in records:
        by[norm(r.get(dec, "")) if dec else ""].append(r)
    for rows in by.values():
        rng.shuffle(rows)
    out = []
    for i in range(k):   # round-robin over labels, rarest first
        for lab in sorted(by, key=lambda x: len(by[x])):
            if i < len(by[lab]) and len(out) < k:
                out.append(by[lab][i])
    return [{c: mask(v[:300]) for c, v in r.items()} for r in out]


def _from_llm(raw: dict) -> dict:
    return {
        "case": {"parts": raw["case"]["parts"], "max_tokens": MAX_CASE_TOKENS, "overflow": "truncate"},
        "decisions": [{**d, "mapping": {m["value"]: m["label"] for m in d["mapping"]}} for d in raw["decisions"]],
        "excluded": raw["excluded"],
    }


def _coverage_errors(plan: dict, prof: dict, max_share: float = 0.02) -> list[str]:
    """An included decision whose mapping leaves out observed values covering > 2% of its rows: those rows would silently drop.
    LLM path only (a user's confirmed plan may drop values on purpose)."""
    errs = []
    for d in (d for d in plan["decisions"] if d["include"] and d["column"] in prof["columns"]):
        p = prof["columns"][d["column"]]
        left = [v for v in p["values"] if v["value"] not in d["mapping"]]
        if p["values_complete"] and sum(v["count"] for v in left) > max_share * p["non_empty"]:
            errs.append(f"decision {d['column']!r} leaves {len(left)} observed values ({sum(v['count'] for v in left)} rows) out of "
                        f"its mapping: {[v['value'] for v in left][:30]}; map each one to a label or to null")
    return errs


def llm_plan(prof: dict, sample: list[dict], records: list[dict] | None = None, client=None, lang: str = "en"):
    """One structured-output call (plus one retry with the validation errors) -> (plan, "llm"|"heuristic", warnings).
    Any API failure, an oversized prompt or two invalid plans fall back to heuristic_plan with a warning."""
    fallback = lambda why: (heuristic_plan(prof, records, lang), "heuristic", [why])
    user = json.dumps({"profile": prof, "sample_rows": sample}, ensure_ascii=False)
    if (len(SYSTEM) + len(user)) / 4 > MAX_PROMPT_TOKENS:   # chars/4 ~ tokens
        return fallback(msg(lang, "w_big"))
    system = SYSTEM if lang == "en" else SYSTEM + f"\n- Write every question, reason and sentence template in {LANGS[lang]} (natural, neutral phrasing). Do not translate column names, mapping values or labels (labels stay in the data's own words and language), and keep the {{value}} placeholder."
    messages, errs = [{"role": "user", "content": user}], []
    for _ in range(2):
        try:
            raw, _usage = tllm.complete_json(system, messages, PLAN_SCHEMA, max_tokens=16000, client=client)
        except tllm.LLMError as e:   # type + status only, never a key
            return fallback(msg(lang, "w_fail", err=str(e)))
        try:
            plan = _from_llm(raw)
            errs = _coverage_errors(plan, prof) + validate_plan(plan, prof, lang)
        except (ValueError, KeyError, TypeError) as e:
            errs = [f"the JSON does not match the schema: {e!r}"]
        if not errs:
            return _finish(plan, prof, records, lang), "llm", []
        messages += [{"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
                     {"role": "user", "content": "The plan has these problems, return a corrected full plan:\n- " + "\n- ".join(errs[:20])}]
    return fallback(msg(lang, "w_invalid", err=errs[0][:120]))


# ---------------------------------------------------------------- rendering

def render_case(record: dict, plan: dict) -> str:
    """Facts first (one per line: sentence with {value}, or "<column>: <value>."), then each body part after a blank line."""
    facts, bodies = [], []
    for q in plan["case"]["parts"]:
        v = (record.get(q["column"]) or "").strip()
        if v and q["role"] == "fact":
            facts.append(q["sentence"].replace("{value}", v) if q.get("sentence") else f"{q['column']}: {v}.")
        elif v:
            bodies.append(v)
    return "\n\n".join(filter(None, ["\n".join(facts), *bodies]))


def answers(record: dict, plan: dict) -> dict:
    """{decision column: label or None} for the included decisions."""
    return {d["column"]: d["mapping"].get(norm(record.get(d["column"], ""))) for d in plan["decisions"] if d["include"]}


if __name__ == "__main__":
    rng = random.Random(0)
    recs = [{"note": f"customer {i} wrote something about {rng.choice('abcdefgh')} " * 3, "esc": rng.choice(["Y", "n", "N", "y"]),
             "sev": rng.choice(["P1", "P2", "P3", "P4"]), "copy": ""} for i in range(300)]
    for r in recs:
        r["copy"] = r["sev"].lower()   # a column that is the answer in disguise
    prof = profile(recs)
    plan = heuristic_plan(prof, recs)
    dec = {d["column"]: d for d in plan["decisions"]}
    assert dec["esc"]["type"] == "noul" and dec["esc"]["mapping"] == {"y": "yes", "n": "no"}, dec["esc"]
    assert dec["sev"]["type"] == "score" and dec["sev"]["labels"] == ["P4", "P3", "P2", "P1"], dec["sev"]["labels"]   # P1 = top
    assert any(l["column"] == "copy" and l["purity"] == 1.0 for l in prof["decisions"]["sev"]["leaks"])
    bad = json.loads(json.dumps(plan))
    next(d for d in bad["decisions"] if d["column"] == "sev")["mapping"]["p9"] = "P1"
    assert any("p9" in e for e in validate_plan(bad, prof)), "invented mapping key must be rejected"
    fact = {"column": "sev", "role": "fact", "sentence": None, "confidence": 1, "reason": ""}
    p2 = {"case": {"parts": [fact, {"column": "note", "role": "body", "sentence": None, "confidence": 1, "reason": ""}]}}
    assert render_case({"sev": "P2", "note": "hi"}, p2) == "sev: P2.\n\nhi"
    p2["case"]["parts"].reverse()
    assert render_case({"sev": "P2", "note": "hi"}, p2) == "sev: P2.\n\nhi", "facts first even when listed last"
    print("ok")
