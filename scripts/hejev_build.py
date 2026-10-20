#!/usr/bin/env python3
"""Hebrew JevBench v0 builder -> .context/hebrew-bench/hejev-v0/

EVAL-ONLY. These are Hebrew versions of public JevBench items plus new teacher-labeled Hebrew items;
they must never be used for training (or for selecting training data).

  translate  translated twins (claude-fable-5-1) -> deterministic digit/ID check -> back-translation
             (codex, else claude-opus-5-5) -> judge (claude-opus-5-5) -> one fix retry or drop.
             Writes translated.jsonl and mixed.jsonl (Hebrew state + English question/options).
  native     Hebrew decision items authored by claude-fable-5-1, soft-labeled by two teachers
             (claude-fable-5-1, claude-opus-5-5) through WorkflowEvals' system_one probability prompt,
             one question per call; kept only when both teachers agree on argmax. Writes native.jsonl.
  sample     30-item stratified native spot-check list (native_review_sample.md).

Every CLI response is cached on disk by sha256 (cache/), so reruns are free; the $ cap counts every
cached response, so it holds across runs. Run from the repo root:
  uv run --project .context/research_sources/WorkflowEvals python scripts/hejev_build.py translate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r053_teacher_label as r053  # noqa: E402  (cached `claude -p` transport + teacher labeling)
from hejev_score import jsd  # noqa: E402  (also puts jevbench on sys.path)
from jevbench.scoring import argmax_label as argmax  # noqa: E402
from jevbench.tasks import Task  # noqa: E402  (every written row must load in the stock runner)

ROOT = Path(__file__).resolve().parents[1]
JB = ROOT / ".context" / "jevbench-v154"
OUT = ROOT / ".context" / "hebrew-bench" / "hejev-v0"
TIERS = ("original", "easy", "hard")
TRANSLATOR, JUDGE, BACKUP_BT = "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5-5"
CODEX_MODEL = "gpt-5.6-terra"
EVAL_ONLY = ("EVAL-ONLY: derived from public JevBench items (or teacher-labeled for this benchmark). "
             "Never use these items, their translations, or their labels for training, distillation, "
             "or training-data selection.")
r053.CACHE = OUT / "cache"
os.environ["DISABLE_PROMPT_CACHING"] = "1"  # one-shot calls: a 1 h cache write would double the input price

# --------------------------------------------------------------------------- translatable fields


def walk(x, fn, path):
    if isinstance(x, str):
        return fn(path, x)
    if isinstance(x, dict):
        return {k: walk(v, fn, f"{path}.{k}") for k, v in x.items()}
    if isinstance(x, list):
        return [walk(v, fn, f"{path}[{i}]") for i, v in enumerate(x)]
    return x


def rewrite(item: dict, fn):
    """Visit translatable leaves (state, instructions, criteria values; never labels or dict keys) in a fixed
    order as fn(key, text) -> replacement. State leaves that are a single token (enum values, IDs) are kept."""
    n = 0

    def leaf(path, s):
        nonlocal n
        if not re.search(r"[A-Za-z]{2,}", s) or (path.startswith("state") and not re.search(r"\s", s.strip())):
            return s
        key = f"{n:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", path)[:40].strip("_")
        n += 1
        return fn(key, s)

    q = item["question"]
    state = walk(item["state"], leaf, "state")
    question = {**q, "instructions": walk(q["instructions"], leaf, "instructions"),
                "criteria": walk(q.get("criteria"), leaf, "criteria")}
    return state, question


def segments(item: dict) -> dict:
    segs = {}
    rewrite(item, lambda k, s: segs.setdefault(k, s))
    return segs


def det_issues(src: dict, tr: dict, labels: list) -> list[str]:
    """Deterministic QA: every digit run and every snake_case option ID in the source survives verbatim."""
    issues = []
    for k, en in src.items():
        he = tr.get(k, "")
        lost = Counter(re.findall(r"\d+", en)) - Counter(re.findall(r"\d+", he))
        if lost:
            issues.append(f"{k}: digits missing or changed: {' '.join(sorted(lost))}")
        issues += [f"{k}: option ID {lab!r} not kept verbatim" for lab in labels
                   if "_" in lab and re.search(rf"(?<!\w){re.escape(lab)}(?!\w)", en) and lab not in he]
        if k.split("_", 1)[1].startswith("instructions") and not re.search(r"[א-ת]", he):
            issues.append(f"{k}: instructions not translated")
    return issues


def mirror_schema(keys) -> dict:
    keys = list(keys)
    return {"type": "object", "properties": {k: {"type": "string"} for k in keys}, "required": keys,
            "additionalProperties": False}


# --------------------------------------------------------------------------- transports

_CODEX_DOWN = threading.Event()


def claude(model, system, user, schema):
    raw = r053.claude(model, system, user, schema=schema)
    if raw.get("structured_output") is None:
        raise RuntimeError(f"{model}: no structured_output")
    return raw["structured_output"]


def codex(prompt: str, schema: dict) -> dict:
    """Cached `codex exec` call (GPT family; subscription, so $0 notional)."""
    path = r053.CACHE / f"{hashlib.sha256(json.dumps(['codex', CODEX_MODEL, prompt, schema]).encode()).hexdigest()}.json"
    if path.exists():
        return json.loads(path.read_text())["structured_output"]
    with tempfile.TemporaryDirectory() as d:
        Path(d, "schema.json").write_text(json.dumps(schema))
        proc = subprocess.run(["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                               "--skip-git-repo-check", "-s", "read-only", "-C", d, "-m", CODEX_MODEL,
                               "-c", "model_reasoning_effort=low", "--output-schema", f"{d}/schema.json",
                               "-o", f"{d}/out.json", "-"], input=prompt, capture_output=True, text=True,
                              timeout=1800)
        out = Path(d, "out.json")
        if not out.exists() or not out.read_text().strip():
            raise RuntimeError(f"codex failed: {(proc.stderr or proc.stdout)[-300:]}")
        result = json.loads(out.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"backend": "codex", "model": CODEX_MODEL, "structured_output": result,
                                "total_cost_usd": 0.0}))
    return result


def spent() -> float:
    return sum(json.loads(p.read_text()).get("total_cost_usd") or 0 for p in r053.CACHE.glob("*.json"))


# --------------------------------------------------------------------------- part 1+2: translated twins

TRANSLATE_SYSTEM = """You translate items of an AI decision benchmark from English into Hebrew.
You receive a JSON object of English text fields from one benchmark item (the case text, the question, and the
option descriptions). Return an object with exactly the same keys whose values are the Hebrew translations.

Write natural, fluent, idiomatic Hebrew in the register used in Israeli workplaces and services (not a literal
calque). Use the same Hebrew term every time for the same English term, across all fields: the question and the
option descriptions refer to the case text.

Keep EXACTLY as written, character for character:
- every number and digit: amounts, prices, percentages, dates, times, ages, durations, counts, section and
  clause numbers (e.g. 4.3.1), versions, phone numbers. Do not convert units, currencies, time formats or
  date formats, do not spell numbers out in words, do not round;
- currency symbols and codes, IDs, ticket/order/claim/policy numbers, codes, SKUs, emails, URLs, file paths,
  hostnames, usernames, commands, code, log fields, JSON keys and enum values;
- option IDs (the machine labels listed in the request) wherever they appear in the text;
- product, company and system names; standard/legal references.
Inside code, JSON, logs or tables, translate only the natural-language parts and keep the structure.
People's names: keep them unless they are purely decorative and appear nowhere else (contacts, emails,
signatures, matching rules); when in doubt keep them. Common English tech terms that Israelis use as-is
(API, CSV, PDF, SLA, VPN, CRM) may stay in English.

Preserve meaning exactly. The items test judgment: keep every fact, condition, negation, quantifier, hedge,
ambiguity, distractor, misleading note and error of the source. Never add, remove, resolve, explain or hint
at an answer, and never make the item easier or harder. Keep line breaks, lists, headings and formatting."""

BT_PROMPT = """Translate each value of the JSON object below from Hebrew into English, faithfully and literally,
keeping every number, ID and code unchanged. Do not interpret, correct or improve anything.
Return an object with exactly the same keys.

{payload}"""

JUDGE_SYSTEM = """You are a bilingual Hebrew-English QA reviewer for a translated AI decision benchmark.
You get one item: the English source fields, their Hebrew translation, an independent back-translation of the
Hebrew into English, the option IDs and the gold answer. Read the Hebrew itself; use the back-translation only
as supporting evidence.

Flag an issue only if it is material:
1. meaning drift that could change the correct answer or the difficulty: an added, dropped or altered fact,
   condition, negation, quantifier, hedge, time relation, ambiguity or distractor;
2. any changed, missing or reformatted number, date, amount, ID, code or option ID;
3. answer leakage: the Hebrew makes the gold answer more obvious than the English does (added hints,
   explanations, resolved ambiguity, a term that echoes the gold option but not the others), or less obvious;
4. natural-language text left untranslated, or Hebrew so unnatural that a native speaker would misread it.
Do not flag style, kept English tech terms, kept names, or reasonable term choices.
Return ok=true when there is no material issue; otherwise ok=false and one short, specific issue per entry
(quote the field key)."""
JUDGE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"},
                                                 "issues": {"type": "array", "items": {"type": "string"}}},
                "required": ["ok", "issues"], "additionalProperties": False}


def translate(item: dict, src: dict, prev: tuple | None) -> dict:
    user = (f"Item {item['id']} (family {item['family']}, question type {item['question']['type']}).\n"
            f"Option IDs (machine labels, keep verbatim wherever they appear): {json.dumps(item['labels'])}\n"
            "Translate every value of this JSON object into Hebrew; return exactly the same keys.\n\n"
            + json.dumps(src, ensure_ascii=False, indent=1))
    if prev:
        he, issues = prev
        user += ("\n\nA QA reviewer rejected a previous Hebrew translation of this item:\n"
                 + "\n".join(f"- {i}" for i in issues)
                 + "\n\nPrevious translation:\n" + json.dumps(he, ensure_ascii=False, indent=1)
                 + "\n\nReturn a corrected full translation that fixes every issue.")
    return claude(TRANSLATOR, TRANSLATE_SYSTEM, user, mirror_schema(src))


def back_translate(he: dict) -> tuple[dict, str]:
    prompt = BT_PROMPT.format(payload=json.dumps(he, ensure_ascii=False, indent=1))
    if not _CODEX_DOWN.is_set():
        try:
            return codex(prompt, mirror_schema(he)), f"codex:{CODEX_MODEL}"
        except Exception as exc:  # usage limit / outage: switch the rest of the run to the Claude fallback
            print(f"codex unavailable, falling back to {BACKUP_BT}: {str(exc)[:200]}", flush=True)
            _CODEX_DOWN.set()
    return claude(BACKUP_BT, "You are a careful Hebrew-to-English translator.", prompt, mirror_schema(he)), BACKUP_BT


def judge(item, src, he, bt) -> dict:
    payload = {"item_id": item["id"], "question_type": item["question"]["type"], "option_ids": item["labels"],
               "gold_answer": item["expected"], "english_source": src, "hebrew_translation": he,
               "back_translation_to_english": bt}
    return claude(JUDGE, JUDGE_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=1), JUDGE_SCHEMA)


def twin(item: dict) -> dict:
    """Translate one item with QA and at most one fix retry. Returns the QA record (with `he` when accepted)."""
    src, attempts, prev = segments(item), [], None
    try:
        for attempt in range(2):
            he = translate(item, src, prev)
            det = det_issues(src, he, item["labels"])
            rec = {"attempt": attempt, "det_issues": det}
            if not det:  # deterministic failures go straight to the retry; no need to pay for a judge
                rec["back_translation"], rec["bt_model"] = back_translate(he)
                rec["judge"] = judge(item, src, he, rec["back_translation"])
            attempts.append(rec)
            issues = det or ([] if rec["judge"]["ok"] else rec["judge"]["issues"] or ["judge: not ok"])
            if not issues:
                return {"id": item["id"], "status": "accepted", "attempts": attempts, "he": he}
            prev = (he, issues)
        return {"id": item["id"], "status": "dropped", "attempts": attempts}
    except r053.BudgetExceeded as exc:
        return {"id": item["id"], "status": "budget", "attempts": attempts, "error": str(exc)}
    except Exception as exc:
        return {"id": item["id"], "status": "error", "attempts": attempts, "error": f"{type(exc).__name__}: {exc}"[:300]}


def variant(item: dict, prefix: str, state, question: dict, note: str, qa: dict) -> dict:
    return {**item, "id": prefix + item["id"], "group": item["group"] and prefix + item["group"],
            "state": state, "question": question,
            "provenance": {**item["provenance"], "source": f"Hebrew JevBench v0 {note} of JevBench public "
                           f"{item['id']} ({item['provenance'].get('source')})", "twin_of": item["id"],
                           "lang": "he" if prefix == "he-" else "he-state/en-question", "eval_only": True,
                           "translator": TRANSLATOR, "qa": qa}}


def load_source() -> list[dict]:
    return [json.loads(line) for t in TIERS for line in (JB / "datasets" / "public" / f"{t}.jsonl").open()]


def cmd_translate(args):
    items = load_source()
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]
    items = items[: args.limit or None]
    print(f"{len(items)} items; spent so far ${spent():.2f} of ${args.max_usd}", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(twin, items))
    if args.ids or args.limit:  # pilot: print, don't publish partial files
        print(json.dumps([{k: r[k] for k in ("id", "status")} | {"attempts": [
            {k: a.get(k) for k in ("det_issues", "judge", "bt_model")} for a in r["attempts"]]} for r in records],
            ensure_ascii=False, indent=1))
        print(f"spent ${spent():.2f}")
        return
    translated, mixed = [], []
    for item, rec in zip(items, records):
        if rec["status"] != "accepted":
            continue
        qa = {"attempts": len(rec["attempts"]), "back_translator": rec["attempts"][-1]["bt_model"],
              "judge": JUDGE, "det_check": "pass"}
        state, question = rewrite(item, lambda k, s: rec["he"][k])
        translated.append(variant(item, "he-", state, question, "translated twin", qa))
        mixed.append(variant(item, "mx-", state, item["question"], "mixed-language variant", qa))
    write_tasks(OUT / "translated.jsonl", translated)
    write_tasks(OUT / "mixed.jsonl", mixed)
    (OUT / "qa.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    first = [r["attempts"][0] for r in records if r["attempts"]]
    stats = {"source_items": len(items), "status": dict(Counter(r["status"] for r in records)),
             "first_pass_det_flagged": sum(bool(a["det_issues"]) for a in first),
             "first_pass_judge_flagged": sum(not a.get("judge", {"ok": True})["ok"] for a in first),
             "first_pass_flagged": sum(bool(a["det_issues"]) or not a.get("judge", {"ok": True})["ok"] for a in first),
             "retried": sum(len(r["attempts"]) > 1 for r in records),
             "fixed_by_retry": sum(len(r["attempts"]) > 1 and r["status"] == "accepted" for r in records),
             "back_translator": dict(Counter(a.get("bt_model") for r in records for a in r["attempts"] if a.get("bt_model"))),
             "errors": [{k: r[k] for k in ("id", "error")} for r in records if r["status"] in ("error", "budget")],
             "dropped_ids": [r["id"] for r in records if r["status"] == "dropped"]}
    update_manifest("translation", stats)
    print(json.dumps(stats, indent=1))


# --------------------------------------------------------------------------- part 3: native Hebrew items

DOMAINS = {
    "invoices": "חשבוניות מס, חשבונית מס/קבלה, מספרי הקצאה של רשות המסים, מע\"מ, ספקים, תנאי תשלום (שוטף+30/60), הנהלת חשבונות",
    "tenders": "מכרזים ציבוריים ופרטיים: תנאי סף, ערבות מכרז וערבות ביצוע, שאלות הבהרה, פסילת הצעות, ועדת מכרזים",
    "customer_service": "שירות לקוחות: סלולר, אינטרנט, קמעונאות ומסחר מקוון, ביטול עסקה לפי חוק הגנת הצרכן, החזרים, ניתוב פניות",
    "infosec": "אבטחת מידע: התראות SOC, פישינג, הרשאות וגישה, תקנות הגנת הפרטיות (אבטחת מידע), אירועי סייבר ודיווח",
    "insurance_banking": "ביטוח ובנקאות: תביעות ביטוח רכב/דירה/בריאות, משכנתאות והלוואות, הוראות קבע, כרטיסי אשראי, איסור הלבנת הון",
    "hr": "משאבי אנוש: חופשה, ימי מחלה, מילואים, הודעה מוקדמת, שימוע, פיצויים, דמי הבראה, החזר נסיעות, גיוס",
    "gov_services": "שירותי ממשלה ורשויות: ביטוח לאומי, רשות האוכלוסין וההגירה, ארנונה ועירייה, משרד הרישוי, זכאויות והנחות",
}
FAMILIES = ("routing", "adequacy", "policy", "intent", "ordinal", "extraction", "long_policy", "tradeoff",
            "ambiguous", "trap", "multi_hop", "temporal_numeric", "adversarial", "judge_hard", "probability")
BATCH_FOCUS = ("everyday, routine cases", "exceptions, escalations, edge cases and conflicting documents")
PER_BATCH = 12

GEN_SYSTEM = """You author evaluation items for a typed-decision benchmark (JevBench format), written directly in
Hebrew for Israeli organizations. Each item is one decision a business system must make: a case text (state),
one question, and a fixed set of answer options.

Requirements for every item:
- Native, natural Israeli Hebrew (not translated-sounding). Realistic Israeli context: ₪ amounts, Israeli dates
  (DD/MM/YYYY), Israeli names, companies, institutions, laws and procedures as they actually work. Case texts
  are realistic documents: emails, WhatsApp/chat transcripts, call summaries, invoice or policy excerpts,
  procedure clauses, system logs, forms. 300-1500 characters; English tech terms may appear as Israelis use them.
- The state must never state the answer or contain a label field. All facts needed to decide are in the state
  (and in the question for rules); no outside knowledge beyond common sense and the stated rules.
- question type "choice": 3-6 options; each option id is a short English snake_case identifier
  (e.g. route_to_billing), its description is Hebrew.
  question type "noul" (yes/no): exactly two options with ids "true" then "false"; descriptions say in Hebrew
  what makes the answer yes / no. "intended" is "true" or "false".
  question type "score": 3-5 ordered levels with ids "0","1",... from lowest to highest; descriptions are Hebrew
  rubric levels. "intended" is the level id.
- instructions: the Hebrew question, precise enough that a careful expert reaches the intended answer.
- clarity "clear": an expert would answer with high confidence. clarity "ambiguous": deliberately borderline,
  the evidence genuinely points two ways (conflicting signals, missing detail, a judgment call) but a careful
  expert would still lean to the intended option with moderate confidence. Ambiguous does not mean unanswerable.
- rationale: one or two Hebrew sentences explaining the intended answer (kept out of the model input).
- Vary scenarios, families, document styles and which option is correct; avoid near-duplicates."""


def gen_schema() -> dict:
    option = {"type": "object", "properties": {"id": {"type": "string"}, "description": {"type": "string"}},
              "required": ["id", "description"], "additionalProperties": False}
    item = {"type": "object", "properties": {
        "family": {"type": "string", "enum": list(FAMILIES)},
        "clarity": {"type": "string", "enum": ["clear", "ambiguous"]},
        "state": {"type": "string"},
        "type": {"type": "string", "enum": ["choice", "noul", "score"]},
        "instructions": {"type": "string"},
        "options": {"type": "array", "items": option},
        "intended": {"type": "string"},
        "rationale": {"type": "string"}},
        "required": ["family", "clarity", "state", "type", "instructions", "options", "intended", "rationale"],
        "additionalProperties": False}
    return {"type": "object", "properties": {"items": {"type": "array", "items": item}}, "required": ["items"],
            "additionalProperties": False}


def to_task(g: dict, tid: str, domain: str) -> dict | None:
    """Generated item -> JevBench record (expected/provenance filled in after teacher labeling); None if malformed."""
    ids = [o["id"] for o in g["options"]]
    if g["type"] == "noul":
        if sorted(ids) != ["false", "true"] or g["intended"] not in ids:
            return None
        criteria, labels = {o["id"]: o["description"] for o in g["options"]}, ["no", "yes"]
    elif g["type"] == "score":
        if ids != [str(i) for i in range(len(ids))] or not 3 <= len(ids) <= 5 or g["intended"] not in ids:
            return None
        criteria, labels = [o["description"] for o in g["options"]], ids
    else:
        if not 3 <= len(ids) <= 6 or len(set(ids)) != len(ids) or g["intended"] not in ids \
                or not all(re.fullmatch(r"[a-z][a-z0-9_]*", i) for i in ids):
            return None
        criteria, labels = {o["id"]: o["description"] for o in g["options"]}, ids
    return {"id": tid, "family": g["family"], "state": g["state"], "labels": labels, "expected": None,
            "question": {"type": g["type"], "instructions": g["instructions"], "criteria": criteria},
            "split": "public", "group": None,
            "provenance": {"source": "Hebrew JevBench v0 native item", "domain": domain, "clarity": g["clarity"],
                           "author_model": TRANSLATOR, "intended": g["intended"], "rationale": g["rationale"]}}


def to_jev_probs(task: dict, p: dict) -> dict:
    return {"yes": p["true"], "no": p["false"]} if task["question"]["type"] == "noul" else p


def cmd_native(args):
    sys.path.insert(0, str(r053.WE))
    from system_one_adapter import SystemOneAdapterClient

    r053.LEDGER.update(usd=spent(), max_usd=args.max_usd)
    batches = [(d, b) for b in range(len(BATCH_FOCUS)) for d in DOMAINS][: args.batches or None]
    spec = (f"Write {PER_BATCH} items: 5 choice, 4 noul, 3 score; 4 of them clarity ambiguous, 8 clear "
            "(spread the ambiguous ones over the three types).")

    def generate(job):
        domain, b = job
        user = (f"Domain: {DOMAINS[domain]}\nFocus of this batch: {BATCH_FOCUS[b]}.\n{spec}\n"
                "Use at least 5 different families.")
        try:
            return domain, b, claude(TRANSLATOR, GEN_SYSTEM, user, gen_schema())["items"], None
        except Exception as exc:
            return domain, b, [], f"{type(exc).__name__}: {exc}"[:300]

    with ThreadPoolExecutor(args.workers) as pool:
        generated = list(pool.map(generate, batches))
    tasks, malformed = [], 0
    for domain, b, gens, _ in generated:
        for i, g in enumerate(gens):
            t = to_task(g, f"native-{domain}-{b}-{i:02d}", domain)
            malformed += t is None
            tasks += [t] if t else []
    print(f"{len(tasks)} well-formed generated items ({malformed} malformed); spent ${spent():.2f}", flush=True)

    client = SystemOneAdapterClient(structured_outputs=False, llm_answer_mode="probabilities",
                                    normalize_probabilities=True, n_retry_malformed_structure=1)
    teachers = r053.TEACHERS["cli"]

    def label(job):
        t, teacher = job
        try:
            q = {"decision": {k: t["question"][k] for k in ("type", "instructions", "criteria")}}
            return t["id"], teacher, to_jev_probs(t, r053.ask(client, "cli", teacher, t["state"], q)["decision"]), None
        except Exception as exc:
            return t["id"], teacher, None, f"{type(exc).__name__}: {exc}"[:300]

    soft, failures = defaultdict(dict), []
    with ThreadPoolExecutor(args.workers) as pool:
        for tid, teacher, p, err in pool.map(label, [(t, m) for t in tasks for m in teachers]):
            if err:
                failures.append({"id": tid, "teacher": teacher, "error": err})
            else:
                soft[tid][teacher] = p
    labeled = [t for t in tasks if len(soft[t["id"]]) == len(teachers)]
    agreed, outcomes = [], []  # outcomes: (clarity, type, teachers agree)
    for t in labeled:
        ps = [soft[t["id"]][m] for m in teachers]
        mean = {k: sum(p[k] for p in ps) / len(ps) for k in t["labels"]}
        heads = {argmax(p) for p in ps}
        pv = t["provenance"]
        intended = {"true": "yes", "false": "no"}.get(pv["intended"], pv["intended"])
        outcomes.append((pv["clarity"], t["question"]["type"], len(heads) == 1))
        pv["teacher_jsd"] = round(jsd(*ps), 4)
        if len(heads) > 1:
            continue
        exp = argmax(mean)
        t["expected"] = int(exp) if t["question"]["type"] == "score" else exp
        pv.update(soft_gold={k: round(v, 4) for k, v in mean.items()},
                  teachers={m: {k: round(v, 4) for k, v in soft[t["id"]][m].items()} for m in teachers},
                  label_basis="teacher-labeled: mean of claude-fable-5-1 and claude-opus-5-5 (effort high, "
                              "WorkflowEvals system_one probability prompt, one question per call); both "
                              "teachers agree on argmax; NOT human-verified",
                  matches_author_intent=exp == intended, eval_only=True, license="MIT")
        agreed.append(t)
    # cap at the target, round-robin over domains so every domain keeps its share
    by_domain = defaultdict(list)
    for t in agreed:
        by_domain[t["provenance"]["domain"]].append(t)
    kept = [t for rnd in range(PER_BATCH * len(BATCH_FOCUS)) for d in DOMAINS
            for t in by_domain[d][rnd:rnd + 1]][: args.target]
    write_tasks(OUT / "native.jsonl", kept)

    def agree(*cell):
        sel = [ok for *k, ok in outcomes if all(c in k for c in cell)]
        return f"{sum(sel)}/{len(sel)}"

    stats = {"batches": len(batches), "generation_failures": [e for *_, e in generated if e],
             "generated_well_formed": len(tasks), "malformed": malformed, "labeled_by_both": len(labeled),
             "teacher_argmax_agreement": {"all": agree(), "clear": agree("clear"), "ambiguous": agree("ambiguous"),
                                          **{q: agree(q) for q in ("choice", "noul", "score")}},
             "mean_teacher_jsd": round(sum(t["provenance"]["teacher_jsd"] for t in labeled) / max(len(labeled), 1), 4),
             "agreed": len(agreed), "kept": len(kept),
             "kept_matches_author_intent": sum(t["provenance"]["matches_author_intent"] for t in kept),
             "kept_by_type": dict(Counter(t["question"]["type"] for t in kept)),
             "kept_by_clarity": dict(Counter(t["provenance"]["clarity"] for t in kept)),
             "kept_by_domain": dict(Counter(t["provenance"]["domain"] for t in kept)),
             "label_failures": failures}
    update_manifest("native", stats)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- sample, outputs


def cmd_sample(args):
    items = [json.loads(line) for line in (OUT / "native.jsonl").open()]
    cells = [("choice", "clear"), ("noul", "ambiguous"), ("score", "clear"),
             ("choice", "ambiguous"), ("noul", "clear"), ("score", "ambiguous")]
    groups = defaultdict(list)
    for t in sorted(items, key=lambda t: hashlib.sha256(t["id"].encode()).hexdigest()):
        groups[(t["provenance"]["domain"], t["question"]["type"], t["provenance"]["clarity"])].append(t)
    queues = {}  # per domain: its items interleaved over the 6 cells, each domain starting at a different cell
    for off, d in enumerate(DOMAINS):
        rot = [groups[(d, *c)] for c in cells[off % 6:] + cells[:off % 6]]
        queues[d] = [g[i] for i in range(len(items)) for g in rot if i < len(g)]
    picks = [q[i] for i in range(len(items)) for q in queues.values() if i < len(q)][: args.n]
    out = ["# Hebrew JevBench v0 — native items: native-speaker spot check (30 items)", "",
           "EVAL-ONLY items. Labels are teacher-labeled (mean of claude-fable-5-1 and claude-opus-5-5), "
           "not human-verified. Stratified by domain x question type x clarity.", "",
           "For each item mark: **Hebrew natural?** (y/n), **gold correct?** (y/n/unsure), notes.", ""]
    for n, t in enumerate(picks, 1):
        pv, q = t["provenance"], t["question"]
        crit = q["criteria"] if isinstance(q["criteria"], dict) else dict(enumerate(q["criteria"]))
        out += [f"## {n}. `{t['id']}` — {pv['domain']} / {q['type']} / {pv['clarity']} / {t['family']}", "",
                "<div dir=\"rtl\">", "", "**מצב:**", "", t["state"], "", f"**שאלה:** {q['instructions']}", "",
                *[f"- `{k}`: {v}" for k, v in crit.items()], "", "</div>", "",
                f"Gold: `{t['expected']}` — soft gold {json.dumps(pv['soft_gold'])}; author intent "
                f"`{pv['intended']}`", "",
                "- [ ] Hebrew natural   - [ ] gold correct   Notes:", ""]
    (OUT / "native_review_sample.md").write_text("\n".join(out))
    print(f"{len(picks)} items -> {OUT / 'native_review_sample.md'}")


def write_tasks(path: Path, rows: list[dict]) -> None:
    for r in rows:
        Task.from_dict(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} -> {path}")


def update_manifest(key: str, stats: dict) -> None:
    path = OUT / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {}
    m.update({"name": "Hebrew JevBench v0", "eval_only": True, "eval_only_rule": EVAL_ONLY,
              "source": "JevBench public original/easy/hard (231 items, MIT) at .context/jevbench-v154",
              "models": {"translator_and_native_author": TRANSLATOR, "judge": JUDGE,
                         "back_translator": f"codex:{CODEX_MODEL} (fallback {BACKUP_BT})",
                         "native_teachers": list(r053.TEACHERS["cli"])},
              key: stats, "notional_cli_cost_usd": round(spent(), 2),
              "files": {f.name: {"rows": sum(1 for _ in f.open()),
                                 "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                        for f in sorted(OUT.glob("*.jsonl")) if f.name != "qa.jsonl"}})
    path.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("translate", "native", "sample"))
    ap.add_argument("--max-usd", type=float, default=60.0, help="hard cap on notional CLI cost, all runs")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="translate: pilot on the first N items (prints, no files)")
    ap.add_argument("--ids", nargs="+", help="translate: pilot on these item ids (prints, no files)")
    ap.add_argument("--batches", type=int, help="native: only the first N generation batches")
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)
    args.workers = min(args.workers, 6)
    r053.LEDGER.update(usd=spent(), max_usd=args.max_usd)
    {"translate": cmd_translate, "native": cmd_native, "sample": cmd_sample}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
