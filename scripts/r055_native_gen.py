#!/usr/bin/env python3
"""R055 D2-C: native Hebrew decision cases grounded in real Hebrew text -> .context/he-train/d2/native_unlabeled.jsonl

Grounding snippets (deterministic, seeded):
  law     Israeli statutes/regulations from gkour/israeli_law (Hebrew Wikisource, CC BY-SA 4.0; statutes are
          copyright-free under Copyright Act 2007 s.6), chunked by section, chosen per business domain.
  mfaq    company FAQ pages from clips/mfaq `he` (CC0 packaging).
  tenders data.gov.il government tenders + exemption contracts (CKAN datastore, "Other (Open)", attribute).
For each snippet claude-haiku-5-5 writes one realistic Israeli business case (state) plus 1-3 typed questions
(choice / noul / score) in Hebrew, some deliberately ambiguous, with its intended answer in meta.intended.
Rows are in the trainer schema (questions; noul = no/yes, score = list criteria, label = level index) WITHOUT
targets: they are labeled later on GPU by Qwen3.8-27B thinking and kept only where label == meta.intended.

Every CLI response is cached by sha256 under .context/he-train/d2/cache (reruns are free; the $ cap counts cache).
  uv run python scripts/r055_native_gen.py --pilot 4      # print a few, no files
  uv run python scripts/r055_native_gen.py                # full run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r053_teacher_label as r053  # noqa: E402  (cached `claude -p` transport + $ ledger)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".context" / "he-train" / "d2"
RAW = OUT / "raw"
r053.CACHE = OUT / "cache"
MODEL = "claude-haiku-5-5"
SEED = 55

DOMAINS = {
    "invoices_ap": "חשבוניות וספקים (AP): חשבוניות מס, מע\"מ, ניהול פנקסים, מוסר תשלומים לספקים",
    "procurement": "רכש ומכרזים: מכרזים ציבוריים, פטור ממכרז, ועדת מכרזים, תנאי סף, התקשרויות",
    "customer_service": "שירות לקוחות: ביטול עסקה, החזרים, אחריות ושירות, חוזים אחידים, ניתוב פניות",
    "insurance_banking": "ביטוח ובנקאות: תביעות ביטוח, כרטיסי חיוב, שירות ללקוח בבנק, נתוני אשראי, הלבנת הון",
    "hr": "משאבי אנוש: חופשה שנתית, דמי מחלה, הודעה מוקדמת, פיצויי פיטורים, שעות עבודה, עבודת נשים, שכר שווה",
    "privacy_security": "פרטיות ואבטחת מידע: מאגרי מידע, אירועי אבטחה, הרשאות גישה, דיווח לרשות, עיון ותיקון מידע",
    "gov_services": "שירותי ממשלה ורשויות: חופש המידע, רישוי עסקים, ארנונה, ביטוח לאומי, זכאויות",
}
LAWS = {
    "invoices_ap": ["חוק_מס_ערך_מוסף", "הוראות_מס_הכנסה_(ניהול_פנקסי_חשבונות)", "חוק_מוסר_תשלומים_לספקים",
                    "חוק_עסקאות_גופים_ציבוריים"],
    "procurement": ["חוק_חובת_המכרזים", "תקנות_חובת_המכרזים"],
    "customer_service": ["חוק_הגנת_הצרכן", "תקנות_הגנת_הצרכן_(ביטול_עסקה)", "חוק_החוזים_האחידים",
                         "תקנות_הגנת_הצרכן_(אחריות_ושירות_לאחר_מכירה)"],
    "insurance_banking": ["חוק_חוזה_הביטוח", "חוק_הבנקאות_(שירות_ללקוח)", "חוק_כרטיסי_חיוב",
                          "חוק_איסור_הלבנת_הון", "חוק_נתוני_אשראי"],
    "hr": ["חוק_חופשה_שנתית", "חוק_דמי_מחלה", "חוק_הודעה_מוקדמת_לפיטורים_ולהתפטרות", "חוק_פיצויי_פיטורים",
           "חוק_שעות_עבודה_ומנוחה", "חוק_עבודת_נשים", "חוק_שכר_שווה_לעובדת_ולעובד"],
    "privacy_security": ["חוק_הגנת_הפרטיות", "תקנות_הגנת_הפרטיות_(אבטחת_מידע)", "כללי_נתוני_אשראי_(אבטחת_מידע)"],
    "gov_services": ["חוק_חופש_המידע", "חוק_רישוי_עסקים", "חוק_הרשויות_המקומיות_(ערר_על_קביעת_ארנונה_כללית)",
                     "חוק_הביטוח_הלאומי"],
}
LICENSES = {"law": "CC BY-SA 4.0 (Hebrew Wikisource via gkour/israeli_law; statute text is s.6 copyright-free)",
            "mfaq": "CC0-1.0 (clips/mfaq packaging; FAQ text belongs to the source sites)",
            "tenders": "data.gov.il Other (Open), attribution: Government Procurement Administration"}
DATASTORE = {"tender": "7038b3e6-a74d-442e-b16b-466c8196124a", "exemption": "65c8fced-c50c-400e-94fb-ef1a208c43e5"}
FAMILIES = ("routing", "adequacy", "policy", "intent", "ordinal", "extraction", "long_policy", "tradeoff",
            "ambiguous", "trap", "multi_hop", "temporal_numeric")

# --------------------------------------------------------------------------- snippets


def clean_wikitext(text: str) -> str:
    """Hebrew Wikisource law markup -> plain text; sections start with a '§§' marker line."""
    def template(m):
        parts = m.group(1).split("|")
        name, args = parts[0].removeprefix("ח:").strip(), [a for a in parts[1:] if "=" not in a]
        if name.startswith("סעיף"):
            return "\n§§ סעיף " + ". ".join(a.strip() for a in args[:2]) + "\n"
        if name in ("פנימי", "חיצוני"):
            return args[-1] if args else ""
        if name in ("ת", "תת", "תתת", "תתתת"):
            return (args[0] + " ") if args else ""
        return ""
    text = text.split("---", 2)[-1]
    for _ in range(4):  # nested templates
        text = re.sub(r"\{\{([^{}]*)\}\}", template, text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text).replace("'''", "")
    return re.sub(r"\n{2,}", "\n", text)


def law_chunks(text: str, lo=350, hi=1800) -> list[str]:
    chunks, cur = [], ""
    for sec in clean_wikitext(text).split("§§")[1:]:
        sec = sec.strip()
        if len(cur) + len(sec) > hi and cur:
            chunks.append(cur)
            cur = ""
        cur = (cur + "\n" + sec).strip()[:hi]
        if len(cur) >= lo * 2:
            chunks.append(cur)
            cur = ""
    return [c for c in chunks + [cur] if len(c) >= lo]


def law_snippets(per_domain: int, rng: random.Random) -> list[dict]:
    from huggingface_hub import hf_hub_download

    out = []
    for domain, files in LAWS.items():
        pool = []
        for f in files:
            text = Path(hf_hub_download("gkour/israeli_law", f"markdown/{f}.md", repo_type="dataset")).read_text()
            chunks = law_chunks(text)
            pool.append([(f, i, c) for i, c in enumerate(chunks)])
        take = per_domain // len(pool)
        for chunks in pool:
            for f, i, c in rng.sample(chunks, min(take, len(chunks))):
                out.append({"sid": f"law-{zlib.crc32(f.encode()) % 10**6:06d}-{i:03d}", "kind": "law", "domain": domain,
                            "ref": f"gkour/israeli_law:markdown/{f}.md#chunk{i}",
                            "text": f"{f.replace('_', ' ')}\n{c}"})
    return out


def mfaq_snippets(n: int, rng: random.Random) -> list[dict]:
    from huggingface_hub import hf_hub_download

    pages = [json.loads(line) for line in
             Path(hf_hub_download("clips/mfaq", "data/he/train.jsonl", repo_type="dataset")).open()]
    by_site = {}
    for p in pages:  # ponytail: at most 3 pages per site so big FAQ sites don't dominate
        if p["qa_pairs"] and len(by_site.setdefault(p["domain"], [])) < 3:
            by_site[p["domain"]].append(p)
    pool = sorted((p for ps in by_site.values() for p in ps), key=lambda p: p["id"])
    out = []
    for p in rng.sample(pool, min(n, len(pool))):
        qa = rng.sample(p["qa_pairs"], min(3, len(p["qa_pairs"])))
        text = "\n\n".join("ש: " + q["question"].strip() + "\nת: " + re.sub(r"\s+", " ", q["answer"]).strip()[:700]
                           for q in qa)
        out.append({"sid": f"mfaq-{p['id']}", "kind": "mfaq", "domain": None, "ref": f"clips/mfaq:he:{p['id']}",
                    "text": f"שאלות נפוצות מהאתר {p['domain']}\n{text}"[:2000]})
    return out


def datastore_rows(kind: str, n: int, rng: random.Random) -> list[dict]:
    path = RAW / "tenders" / f"{kind}.json"
    if not path.exists():
        base = "https://data.gov.il/api/3/action/datastore_search"
        q = lambda **kw: json.load(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
            f"{base}?{urllib.parse.urlencode({'resource_id': DATASTORE[kind], **kw})}",
            headers={"User-Agent": "Mozilla/5.0"}), timeout=120))["result"]
        total = q(limit=1)["total"]
        rows = []
        for off in sorted(rng.sample(range(0, total - 100), 6)):
            rows += q(limit=100, offset=off)["records"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False))
    rows = [r for r in json.loads(path.read_text()) if r.get("שם הליך")]
    return rng.sample(rows, min(n, len(rows)))


def tender_snippets(n: int, rng: random.Random) -> list[dict]:
    out = []
    for kind, label in (("tender", "מכרז ממשלתי"), ("exemption", "התקשרות בפטור ממכרז")):
        for r in datastore_rows(kind, n // 2, rng):
            fields = "\n".join(f"{k}: {v}" for k, v in r.items() if k != "_id" and v not in (None, ""))
            out.append({"sid": f"{kind}-{r['_id']}", "kind": "tenders", "domain": "procurement",
                        "ref": f"data.gov.il:{DATASTORE[kind]}#_id={r['_id']}",
                        "text": f"רשומה מתוך מאגר data.gov.il ({label}):\n{fields}"[:2000]})
    return out


# --------------------------------------------------------------------------- generation

SYSTEM = """You write training cases for a typed-decision model used by Israeli organizations. Write everything
in native, natural Israeli Hebrew (not translated-sounding).

You get one real Hebrew source snippet (a statute/regulation excerpt, a company FAQ, or a government procurement
record). Use it as grounding: write ONE realistic Israeli business case whose decision turns on the rule, policy
or facts in the snippet, then 1-3 decision questions about that case.

The case (state):
- A realistic document a business system would see: an email, WhatsApp/chat transcript, call summary, ticket,
  invoice or contract excerpt, internal memo, form, approval request or system log. Israeli names, companies,
  ₪ amounts, DD/MM/YYYY dates, real Israeli institutions and procedures.
- Self-contained: include the governing rule (quote or faithfully paraphrase the relevant part of the snippet,
  e.g. as a policy clause or a "per section X" note) plus all facts needed. No outside knowledge needed.
- 300-1500 characters. Never state or hint the answer; no label fields.

Questions (each independent, all about the same case):
- type "choice": 3-6 options; option ids are short English snake_case (e.g. approve_payment); descriptions Hebrew.
- type "noul" (yes/no): exactly two options with ids "yes" then "no"; descriptions say in Hebrew what makes
  the answer yes / no. intended is "yes" or "no".
- type "score": 3-5 ordered levels, ids "0","1",... lowest to highest; descriptions are Hebrew rubric levels.
  intended is the level id.
- instructions: the Hebrew question, precise enough that a careful expert reaches the intended answer.
- clarity "clear": an expert answers with high confidence. clarity "ambiguous": deliberately borderline,
  evidence genuinely points two ways (conflicting signals, a missing detail, a judgment call) yet a careful
  expert still leans to the intended option. Ambiguous is not unanswerable.
- rationale: 1-2 Hebrew sentences why the intended answer is right.
- domain: the business area of the case.
Vary which option is correct (do not always make the first option or "yes" correct)."""

TYPE_MIX = ("one choice question", "a choice and a noul question", "a noul and a score question",
            "a choice and a score question", "a choice, a noul and a score question",
            "two noul questions and one choice question", "two choice questions", "one noul question")  # mean 2.0


def gen_schema() -> dict:
    option = {"type": "object", "properties": {"id": {"type": "string"}, "description": {"type": "string"}},
              "required": ["id", "description"], "additionalProperties": False}
    question = {"type": "object", "properties": {
        "type": {"type": "string", "enum": ["choice", "noul", "score"]},
        "family": {"type": "string", "enum": list(FAMILIES)},
        "clarity": {"type": "string", "enum": ["clear", "ambiguous"]},
        "instructions": {"type": "string"},
        "options": {"type": "array", "items": option},
        "intended": {"type": "string"},
        "rationale": {"type": "string"}},
        "required": ["type", "family", "clarity", "instructions", "options", "intended", "rationale"],
        "additionalProperties": False}
    return {"type": "object", "properties": {
        "domain": {"type": "string", "enum": list(DOMAINS)},
        "state": {"type": "string"},
        "questions": {"type": "array", "items": question}},
        "required": ["domain", "state", "questions"], "additionalProperties": False}


def to_question(q: dict) -> tuple[dict, str] | None:
    """Generated question -> (trainer-schema question, intended label); None if malformed."""
    ids = [str(o["id"]).strip() for o in q["options"]]
    intended = str(q["intended"]).strip()
    if intended not in ids or len(set(ids)) != len(ids) or not q["instructions"].strip():
        return None
    if q["type"] == "noul":
        if sorted(ids) != ["no", "yes"]:
            return None
        desc = {i: o["description"] for i, o in zip(ids, q["options"])}
        criteria = {"no": desc["no"], "yes": desc["yes"]}
    elif q["type"] == "score":
        if ids != [str(i) for i in range(len(ids))] or not 3 <= len(ids) <= 5:
            return None
        criteria = [o["description"] for o in q["options"]]
    else:
        if not 3 <= len(ids) <= 6 or not all(re.fullmatch(r"[a-z][a-z0-9_]*", i) for i in ids):
            return None
        criteria = {i: o["description"] for i, o in zip(ids, q["options"])}
    return {"type": q["type"], "instructions": q["instructions"].strip(), "criteria": criteria}, intended


def to_row(snip: dict, g: dict) -> dict | None:
    if not 150 <= len(g["state"]) <= 3000 or not re.search(r"[א-ת]", g["state"]):
        return None
    qs, meta_q = {}, {}
    for q in g["questions"][:3]:
        parsed = to_question(q)
        if parsed:
            qid = f"q{len(qs)}"
            qs[qid] = parsed[0]
            meta_q[qid] = {"intended": parsed[1], "clarity": q["clarity"], "family": q["family"],
                           "rationale": q["rationale"]}
    if not qs:
        return None
    return {"id": f"r055-native-{snip['sid']}", "state": g["state"].strip(), "questions": qs,
            "meta": {"lang": "he", "source": f"r055-native/{snip['kind']}", "domain": g["domain"],
                     "intended": {k: v["intended"] for k, v in meta_q.items()},
                     "clarity": {k: v["clarity"] for k, v in meta_q.items()},
                     "family": {k: v["family"] for k, v in meta_q.items()},
                     "rationale": {k: v["rationale"] for k, v in meta_q.items()},
                     "grounding": {"kind": snip["kind"], "ref": snip["ref"], "license": LICENSES[snip["kind"]],
                                   "text": snip["text"]},
                     "generator": MODEL, "needs_label": "Qwen3.8-27B thinking; keep where argmax == intended"}}


def generate(snip: dict, mix: str) -> dict:
    hint = f"Suggested domain: {DOMAINS[snip['domain']]}\n" if snip["domain"] else ""
    user = (f"{hint}Write {mix}. Make about a third of all questions you write clarity ambiguous.\n\n"
            f"SOURCE SNIPPET ({snip['kind']}):\n{snip['text']}")
    raw = r053.claude(MODEL, SYSTEM, user, effort="medium", schema=gen_schema())
    if raw.get("structured_output") is None:
        raise RuntimeError("no structured_output")
    return raw["structured_output"]


def spent() -> float:
    return sum(json.loads(p.read_text()).get("total_cost_usd") or 0 for p in r053.CACHE.glob("*.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--law-per-domain", type=int, default=190)
    ap.add_argument("--mfaq", type=int, default=330)
    ap.add_argument("--tenders", type=int, default=300)
    ap.add_argument("--max-usd", type=float, default=80.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pilot", type=int, help="generate for N snippets (one per kind first), print, write nothing")
    args = ap.parse_args(argv)
    rng = random.Random(SEED)
    snippets = law_snippets(args.law_per_domain, rng) + mfaq_snippets(args.mfaq, rng) + tender_snippets(args.tenders, rng)
    assert len({s["sid"] for s in snippets}) == len(snippets), "duplicate snippet ids"
    print(f"{len(snippets)} snippets: {dict(Counter(s['kind'] for s in snippets))}", flush=True)
    jobs = [(s, TYPE_MIX[i % len(TYPE_MIX)]) for i, s in enumerate(rng.sample(snippets, len(snippets)))]
    if args.pilot:
        jobs = sorted(jobs, key=lambda j: ["law", "mfaq", "tenders"].index(j[0]["kind"]))
        jobs = [j for k in ("law", "mfaq", "tenders") for j in jobs if j[0]["kind"] == k][:: max(1, len(jobs) // args.pilot)][: args.pilot]
    r053.LEDGER.update(usd=spent(), max_usd=args.max_usd)

    def run(job):
        try:
            return job[0], generate(*job), None
        except Exception as exc:  # budget / CLI failure: record and keep going
            return job[0], None, f"{type(exc).__name__}: {exc}"[:300]

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, jobs))
    rows, errors, malformed = [], Counter(), 0
    for snip, g, err in results:
        if err:
            errors[err.split(":")[0]] += 1
            continue
        row = to_row(snip, g)
        malformed += row is None
        rows += [row] if row else []
    if args.pilot:
        for r in rows:
            print(json.dumps({k: r[k] for k in ("id", "state", "questions")} | {"meta": {
                k: r["meta"][k] for k in ("domain", "intended", "clarity")}}, ensure_ascii=False, indent=1))
    else:
        path = OUT / "native_unlabeled.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} rows -> {path}")
    nq = sum(len(r["questions"]) for r in rows)
    stats = {"snippets": len(jobs), "rows": len(rows), "questions": nq, "malformed": malformed,
             "errors": dict(errors), "by_kind": dict(Counter(r["meta"]["grounding"]["kind"] for r in rows)),
             "by_domain": dict(Counter(r["meta"]["domain"] for r in rows)),
             "by_type": dict(Counter(q["type"] for r in rows for q in r["questions"].values())),
             "ambiguous": sum(c == "ambiguous" for r in rows for c in r["meta"]["clarity"].values()),
             "notional_usd": round(spent(), 2)}
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if not args.pilot:
        (OUT / "native_gen_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
