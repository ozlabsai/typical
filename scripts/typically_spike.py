"""typically spike: does a small per-company fine-tune of typical-small learn a company's *unwritten* policy?

Two companies, both in the training row schema (pcdm/data.py):
  data_co_a  "Northwind Freight" -- described-only company. Ticket = metadata header (the facts the policy reads)
             + an agent-written body (.context/typically/bodies.jsonl). Labels come from POLICY below, never
             from an LLM, so the gold is exact.
  data_co_b  imported history -- Tobi-Bueck/customer-support-tickets (en, CC-BY-NC, spike-only): the dataset's
             own queue/priority labels are the company's past decisions.

Eval files (data_co_*/eval/), all on held-out tickets:
  *_oneliner   the question alone ("Should we escalate this?") -- the headline before/after
  a_rubric     the question + the full written policy (what a user must type without a fine-tune)
  a_flip       the question + a DIFFERENT explicit rule -> gold follows the stated rule (still not a classifier?)
  a_flip_heldout  an explicit rule on a field no training rubric uses (region) -- the real not-a-classifier check
  a_novel      a question never trained on these states (general ability kept?)
  data_co_c    spike 2: company A's exact tickets, header numbers put into words with the user's thresholds
  data_co_e    spike 2: company A + random explicit rules on every trained question (anti-classifier augmentation)
  data_co_d    spike 2: imported with clean labels -- Bitext utterances, company-specific intent -> desk map
  data_co_f/g  spike 3: e with 10% paired flips (f) / refund twins across the $300 or starter boundary (g)

uv run scripts/typically_spike.py attrs     # -> .context/typically/body_specs.jsonl (what the agent must write)
uv run scripts/typically_spike.py build     # -> data_co_a/, data_co_b/
"""
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import typically_llm as tllm
import typically_plan as tp
CTX = ROOT / ".context" / "typically"
ISSUES = ["lost_package", "damaged_goods", "invoice_dispute", "delivery_delay", "address_change", "quote_request"]
MOODS = ["calm", "annoyed", "furious"]
TIERS = ["enterprise", "business", "starter"]
BODY_VARIANTS, TEST_VARIANTS = 8, {6, 7}   # bodies 6-7 only ever appear in held-out tickets
N_TRAIN, N_TEST = 800, 200
LEGAL_WORDS = ("lawyer", "legal", "attorney", "sue ", "suing", "lawsuit", "court", "litigation")


# ============================== Northwind's unwritten policy ==============================
def route(a):
    if a["issue"] == "quote_request":
        return "Sales"
    if a["issue"] == "invoice_dispute":
        return "Sales" if a["tier"] == "enterprise" else "Billing"   # account managers own enterprise billing
    if a["issue"] == "delivery_delay":
        return "Claims" if a["days"] > 10 else "Dispatch"            # >10 days late is treated as lost
    return "Dispatch" if a["issue"] == "address_change" else "Claims"


def escalate(a):
    return a["legal"] or a["prior"] >= 3 or a["value"] >= 5000 or (a["tier"] == "enterprise" and a["mood"] == "furious")


def urgency(a):
    u = {"lost_package": 2, "damaged_goods": 2, "invoice_dispute": 1, "delivery_delay": 1,
         "address_change": 2, "quote_request": 0}[a["issue"]]            # address changes race the truck
    u += (a["tier"] == "enterprise") + (a["value"] >= 2000) - (a["tier"] == "starter" and a["issue"] != "address_change")
    return max(0, min(3, u))


def refund(a):
    return a["issue"] in ("lost_package", "damaged_goods", "delivery_delay") and a["value"] < 300 and a["tier"] != "starter"


# (key, qtype, one-liner, candidates, policy fn -> gold candidate, written policy criteria)
DECISIONS = [
    ("route", "choice", "Which team should handle this ticket?", ["Claims", "Billing", "Dispatch", "Sales"], route,
     {"Claims": "lost or damaged shipments, and any delivery more than 10 days late",
      "Billing": "invoice disputes from business and starter customers",
      "Dispatch": "address changes and deliveries late by 10 days or less",
      "Sales": "quote requests, and invoice disputes from enterprise customers"}),
    ("escalate", "noul", "Should we escalate this to a manager?", ["no", "yes"], lambda a: "yes" if escalate(a) else "no",
     {"true": "the customer mentions legal action, has 3+ tickets in 30 days, the shipment is worth $5,000+, "
              "or an enterprise customer is furious", "false": "none of those hold"}),
    ("urgency", "score", "How urgent is this ticket?", ["0", "1", "2", "3"], lambda a: str(urgency(a)),
     "Start from the issue: quote request 0; invoice dispute or delivery delay 1; lost goods, damaged goods or address "
     "change 2. Add 1 for an enterprise customer and 1 for a shipment worth $2,000 or more. Subtract 1 for a starter-plan "
     "customer unless it is an address change. Keep the result between 0 and 3."),
    ("refund", "noul", "Can the agent refund this without approval?", ["no", "yes"], lambda a: "yes" if refund(a) else "no",
     {"true": "lost, damaged or late shipment worth under $300 for a business or enterprise customer",
      "false": "anything else, including every starter-plan customer"}),
]
FLIP = ("Should we escalate this to a manager? Escalate only when the customer is on the starter plan.",
        {"true": "the customer is on the starter plan", "false": "the customer is on any other plan"},
        lambda a: "yes" if a["tier"] == "starter" else "no")
# held-out flips: rules on a field (region) no training rubric ever uses -> does an explicit new rule still win?
FLIP_HELDOUT = [
    ("escalate", "noul", "Should we escalate this to a manager? Escalate only when the customer is in the EU.", ["no", "yes"],
     {"true": "the customer is in the EU region", "false": "the customer is in the US or APAC"},
     lambda a: "yes" if a["region"] == "EU" else "no"),
    ("route", "choice", "Which team should handle this ticket? This week every APAC customer goes to Sales and everyone "
     "else goes to Dispatch.", ["Claims", "Billing", "Dispatch", "Sales"],
     {"Claims": "not used this week", "Billing": "not used this week", "Dispatch": "every customer outside APAC",
      "Sales": "every APAC customer"}, lambda a: "Sales" if a["region"] == "APAC" else "Dispatch"),
]
# spike 2 arm e: random explicit rules on every trained question (never on region -- FLIP_HELDOUT's field)
CONDS = [(f"the customer is on the {t} plan", lambda a, t=t: a["tier"] == t) for t in TIERS] + \
        [(f"the customer sounds {m}", lambda a, m=m: a["mood"] == m) for m in MOODS] + \
        [(f"the ticket is a {i.replace('_', ' ')}", lambda a, i=i: a["issue"] == i) for i in ISSUES] + \
        [("the customer mentions legal action", lambda a: a["legal"])]


def random_flip(a, key, qtype, q, cands, rng):
    text, cond = rng.choice(CONDS)
    hit = cond(a)
    if qtype == "noul":
        return query(q, {"true": text, "false": "otherwise"}, qtype) if rng.random() < 0.5 else \
            f"{q} For this question, answer yes only when {text}.", ("yes" if hit else "no")
    x, y = rng.sample(cands, 2)
    if qtype == "score":
        return f"{q} For this question, rate {x} when {text} and {y} otherwise.", (x if hit else y)
    return (query(q, {c: (text if c == x else "everything else" if c == y else "not used") for c in cands}, qtype),
            x if hit else y)


NOVEL = ("Does the customer mention legal action or a lawyer?", lambda a: "yes" if a["legal"] else "no")


def query(q, crit, qtype):
    """Mirror of pcdm_jev/decider.query_text for the three criteria shapes."""
    if crit is None:
        return q
    if isinstance(crit, str):
        return f"{q} {crit}"
    if qtype == "noul":
        return f"{q}\nyes: {crit['true']}  no: {crit['false']}"
    if isinstance(crit, list):
        return q + "\n" + "  ".join(f"{i}: {c}" for i, c in enumerate(crit))
    return q + "\n" + "  ".join(f"{k}: {v}" for k, v in crit.items())


def row(state, q, cands, gold, qtype, task, **meta):
    i = cands.index(gold)
    return {"state": state, "query": q, "candidates": cands, "target": [float(j == i) for j in range(len(cands))],
            "p_null": 0.0, "task": task, "label": i, "meta": {"qtype": qtype, "family": task, **meta}}


# ============================== company A ==============================
def sample_attrs(rng):
    issue = rng.choice(ISSUES)
    return {"issue": issue, "mood": rng.choice(MOODS), "legal": rng.random() < 0.12,
            "tier": rng.choices(TIERS, [0.25, 0.45, 0.30])[0], "region": rng.choice(["US", "EU", "APAC"]),
            "value": int(math.exp(rng.uniform(math.log(20), math.log(12000)))),
            "days": 0 if issue == "quote_request" else rng.randint(1, 25), "prior": rng.choices(range(5), [40, 25, 15, 12, 8])[0],
            "customer": f"{rng.choice(['Acme', 'Borealis', 'Cedar', 'Delta', 'Everly', 'Foxglove', 'Granite', 'Harbor'])} "
                        f"{rng.choice(['Retail', 'Labs', 'Supply', 'Foods', 'Medical', 'Outdoors'])}"}


def header(a):
    shipped = "no shipment yet" if a["issue"] == "quote_request" else f"shipped {a['days']} days ago"
    return (f"Customer: {a['customer']} ({a['tier']} plan, {a['region']}). Tickets from this customer in the last 30 days: "
            f"{a['prior']}. Shipment value: ${a['value']:,}, {shipped}.")


def header_words(a):
    """Spike 2: the agent puts the user's own thresholds ($300 / $2,000 / $5,000, 10 days, 3 tickets) into words."""
    v, band = a["value"], ("small, under $300" if a["value"] < 300 else "medium, $300 to $1,999" if a["value"] < 2000
                          else "large, $2,000 to $4,999" if a["value"] < 5000 else "very large, $5,000 or more")
    shipped = "no shipment yet" if a["issue"] == "quote_request" else \
        f"shipped {a['days']} days ago ({'more than 10 days, overdue' if a['days'] > 10 else '10 days or less'})"
    return (f"Customer: {a['customer']} ({a['tier']} plan, {a['region']}). Tickets from this customer in the last 30 days: "
            f"{a['prior']} ({'repeat contact, 3 or more' if a['prior'] >= 3 else 'fewer than 3'}). "
            f"Shipment value: ${v:,} ({band}), {shipped}.")


def cmd_attrs():
    """One spec per (issue, mood, legal, variant): the agent writes a ~60-120 word ticket body for each."""
    CTX.mkdir(parents=True, exist_ok=True)
    with open(CTX / "body_specs.jsonl", "w") as f:
        for issue in ISSUES:
            for mood in MOODS:
                for legal in (False, True):
                    for v in range(BODY_VARIANTS):
                        f.write(json.dumps({"issue": issue, "mood": mood, "legal": legal, "variant": v}) + "\n")


REFUNDABLE = ("lost_package", "damaged_goods", "delivery_delay")


def twin_attrs(a, trng):
    """Spike 3: the same ticket with exactly one field moved so the refund label FLIPS ($300 value line, or starter
    tier). None when no single move flips it (value >= $300 AND starter needs two)."""
    low, paid = a["value"] < 300, a["tier"] != "starter"
    if not (low or paid):
        return None
    t = dict(a)
    if paid and (not low or trng.random() < 0.7):
        t["value"] = trng.randint(300, 900) if low else trng.randint(60, 299)
    else:
        t["tier"] = "starter" if paid else "business"
    assert refund(t) != refund(a)
    return t


def ticket_rows(a, state, p, written, tag, frng=None, flip_rate=0.0, pair=False, group=()):
    """Train rows for one ticket, one base row per DECISION (+ a random-flip row when frng fires).
    pair: a fired flip forces its base row to the plain one-liner and puts both in one micro-batch (meta.ms_group,
    pcdm/train.py group_units); group: decision keys whose base row joins ms_group `tag_key` regardless (twins)."""
    out = []
    for w, (key, qtype, q, cands, fn, crit) in zip(written, DECISIONS):
        flip = frng is not None and frng.random() < flip_rate
        meta = {"ms_group": f"{tag}_{key}"} if (flip and pair) or key in group else {}
        out.append(row(state, query(q, crit if w and not (flip and pair) else None, qtype), cands, fn(a), qtype, f"co_{p}_{key}", **meta))
        if flip:
            fq, fgold = random_flip(a, key, qtype, q, cands, frng)
            out.append(row(state, fq, cands, fgold, qtype, f"co_{p}_rflip", **(meta if pair else {})))
    return out


def build_a(rng, hdr=header, p="a", diverse_flips=False, flip_rate=0.25, pair=False, twins=False):
    bodies, dropped = {}, 0
    for line in open(CTX / "bodies.jsonl"):
        b = json.loads(line)
        if b["legal"] != any(w in b["text"].lower() for w in LEGAL_WORDS):   # body contradicts its spec -> label noise
            dropped += 1
            continue
        bodies.setdefault((b["issue"], b["mood"], b["legal"], b["variant"] in TEST_VARIANTS), []).append(b["text"])
    print(f"bodies: {sum(map(len, bodies.values()))} kept, {dropped} dropped")
    frng, trng = random.Random(1), random.Random(2)   # own streams: flips/twins must not shift the ticket sampling
    split = {"train": [], "val": [], **{f"{p}_{k}": [] for k in ("oneliner", "rubric", "flip", "flip_heldout", "novel")}}
    for n, test in ((N_TRAIN, False), (N_TEST, True)):
        for t in range(n):
            a = sample_attrs(rng)
            body = rng.choice(bodies[(a["issue"], a["mood"], a["legal"], test)])
            state = hdr(a) + "\n\n" + body
            if test:
                for key, qtype, q, cands, fn, crit in DECISIONS:
                    gold, task = fn(a), f"co_{p}_{key}"
                    split[f"{p}_oneliner"].append(row(state, q, cands, gold, qtype, task))
                    split[f"{p}_rubric"].append(row(state, query(q, crit, qtype), cands, gold, qtype, task))
            else:
                # train mix: one-liner mostly, the written policy 30% of the time so the rubric stays load-bearing
                written = [rng.random() < 0.3 for _ in DECISIONS]
                dest = "val" if t % 10 == 0 else "train"
                ta = twin_attrs(a, trng) if twins and dest == "train" and a["issue"] in REFUNDABLE else None
                twin = ta is not None
                split[dest] += ticket_rows(a, state, p, written, f"{p}{t}", frng if diverse_flips else None, flip_rate, pair,
                                           ("refund",) if twin else ())
                if twin:   # same body, header rebuilt from the moved field, plain one-liners, refund row grouped with the original's
                    split[dest] += ticket_rows(ta, hdr(ta) + "\n\n" + body, p, [False] * len(DECISIONS), f"{p}{t}", group=("refund",))
            if not test and rng.random() < 0.25:   # rubric-flip augmentation: an explicit different rule wins
                split["val" if t % 10 == 0 else "train"].append(
                    row(state, query(FLIP[0], FLIP[1], "noul"), ["no", "yes"], FLIP[2](a), "noul", f"co_{p}_flip"))
            if test:   # no rng draws below: keeps spike-1 tickets byte-identical
                split[f"{p}_flip"].append(row(state, query(FLIP[0], FLIP[1], "noul"), ["no", "yes"], FLIP[2](a), "noul", f"co_{p}_flip"))
                for key, qtype, q, cands, crit, fn in FLIP_HELDOUT:
                    split[f"{p}_flip_heldout"].append(row(state, query(q, crit, qtype), cands, fn(a), qtype, f"co_{p}_flipho_{key}"))
                split[f"{p}_novel"].append(row(state, NOVEL[0], ["no", "yes"], NOVEL[1](a), "noul", f"co_{p}_novel"))
    return split


# ============================== company B (imported) ==============================
def build_b(rng):
    from datasets import load_dataset
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train").filter(lambda r: r["language"] == "en")
    seen, tickets = set(), []
    for r in ds:
        body = (r["body"] or "").replace("\\n", "\n").strip()
        if body and body not in seen and len(body) < 2500:
            seen.add(body)
            tickets.append((f"Subject: {r['subject'] or '(none)'}\n\n{body}", r["queue"], r["priority"]))
    rng.shuffle(tickets)
    queues = sorted({q for _, q, _ in tickets})
    split = {"train": [], "val": [], "b_oneliner": []}
    for i, (state, queue, prio) in enumerate(tickets[:3000]):
        dest = "b_oneliner" if i < 500 else "val" if i < 700 else "train"
        split[dest].append(row(state, "Which department should handle this ticket?", queues, queue, "choice", "co_b_route"))
        split[dest].append(row(state, "How urgent is this ticket?", ["low", "medium", "high"], prio, "score", "co_b_urgency"))
    return split


# ============================== company D (imported, clean labels) ==============================
# Bitext customer-support utterances (CDLA-Sharing-1.0, spike-only); the company's history = its own intent -> desk map
DESKS = {"Money desk": ["check_invoice", "get_invoice", "payment_issue", "check_payment_methods", "get_refund", "track_refund",
                        "check_refund_policy"],
         "Retention desk": ["cancel_order", "delete_account", "complaint", "check_cancellation_fee", "newsletter_subscription"],
         "Logistics desk": ["delivery_options", "delivery_period", "track_order", "change_shipping_address",
                            "set_up_shipping_address", "change_order", "place_order"],
         "Account desk": ["create_account", "edit_account", "recover_password", "registration_problems", "switch_account"],
         "Front desk": ["contact_customer_service", "contact_human_agent", "review"]}
AT_RISK = {"cancel_order", "delete_account", "complaint", "check_cancellation_fee"}


def build_d(rng):
    from datasets import load_dataset
    desk_of = {i: d for d, intents in DESKS.items() for i in intents}
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    seen, msgs = set(), []
    for r in ds:
        text = re.sub(r"\{\{([^}]+)\}\}", lambda m: f"[{m.group(1).lower()}]", r["instruction"]).strip()
        if text.lower() not in seen:
            seen.add(text.lower())
            msgs.append((f"Customer message: {text}", r["intent"]))
    rng.shuffle(msgs)
    split = {"train": [], "val": [], "d_oneliner": []}
    for i, (state, intent) in enumerate(msgs[:3700]):
        dest = "d_oneliner" if i < 500 else "val" if i < 700 else "train"
        split[dest].append(row(state, "Which desk should handle this message?", list(DESKS), desk_of[intent], "choice", "co_d_route"))
        split[dest].append(row(state, "Is this customer at risk of leaving?", ["no", "yes"],
                               "yes" if intent in AT_RISK else "no", "noul", "co_d_churn"))
    return split


# ============================== typically UI import ==============================
YESNO = {"yes": "yes", "true": "yes", "1": "yes", "no": "no", "false": "no", "0": "no"}


def import_flips(rows, typ, cands, conds, rate, frng):
    """Spike 3 arm f, generic: each TRAIN row fires with prob `rate` -> a rule-flip row on the same case whose condition is
    one of conds [(text, pred(state) -> bool)]; both share meta.ms_group (one micro-batch, as build_a pair=True). Returns the flip rows."""
    out = []
    for n, r in enumerate({id(r): r for r in rows}.values()):   # balancing repeats row objects: one draw per distinct row
        if frng.random() >= rate:
            continue
        (text, pred), q, g = frng.choice(conds), r["query"], f"{r['task']}_{n}"
        hit = pred(r["state"])
        if typ == "noul":
            fq, gold = f"{q} For this question, answer yes only when {text}.", "yes" if hit else "no"
        else:
            x, y = frng.sample(cands, 2)
            fq, gold = f"{q} For this question, pick {x} when {text}, otherwise {y}.", x if hit else y
        r["meta"] = {**r["meta"], "ms_group": g}
        out.append(row(r["state"], fq, cands, gold, typ, "import_rflip", ms_group=g))
    return out


def keyword_conds(states):
    """Rule conditions on the case text: alphabetic tokens (4+ letters) that occur in 5-40% of the states."""
    toks = {s: set(re.findall(r"[a-z]{4,}", s.lower())) for s in states}
    df = Counter(w for t in toks.values() for w in t)
    return [(f'the case mentions "{w}"', lambda s, w=w: w in toks[s]) for w in sorted(df) if 0.05 <= df[w] / len(toks) <= 0.40]


def balance_rows(rows):
    """Oversample minority labels up to 1/3 of the majority count, no further (more = memorising).
    ponytail: a 1/2 floor for yes/no decisions lost on Enron "reply" (2026-10-04, 88/12): catch@20 .51 vs .53, agreement .81 vs .87,
    mean p(yes) .28 vs actual .12 -- ranking did not improve, calibration got worse. Kept 1/3 for every decision."""
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    floor = max(map(len, by.values())) // 3
    return rows + [x for g in by.values() for x in (g * (floor // len(g) + 1))[:max(0, floor - len(g))]]


def build_import(records, text_col, decisions, rng, balance=True, flips=0.10):
    """CSV rows (dicts) + the user's decisions [{"column", "question", "type"}] -> split dict for write().
    Split by record 70/10/20 (eval = import_oneliner); balance oversamples minority labels in TRAIN only;
    flips = per (case, decision) chance of a paired rule-flip row in TRAIN only (own rng stream: 0 leaves output unchanged)."""
    order = list(range(len(records)))
    rng.shuffle(order)
    n_tr, n_va = int(0.7 * len(order)), int(0.1 * len(order))
    part = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "import_oneliner": order[n_tr + n_va:]}
    split = {k: [] for k in part}
    frng = random.Random(1)
    conds = keyword_conds([r[text_col] for r in records])   # none -> no flips
    for d in decisions:
        col, typ = d["column"], d["type"]
        vals = [r[col].strip() for r in records]
        if "" in vals:
            raise ValueError(f"column {col!r} has empty cells")
        if typ == "noul":
            bad = {v for v in vals if v.lower() not in YESNO}
            if bad:
                raise ValueError(f"column {col!r} is not yes/no: {sorted(bad)[:3]}")
            vals, cands = [YESNO[v.lower()] for v in vals], ["no", "yes"]
        else:
            ints = typ == "score" and all(v.lstrip("-").isdigit() for v in vals)   # "10" must sort after "2"
            cands = sorted(set(vals), key=int if ints else str)
        if len(cands) < 2:
            raise ValueError(f"column {col!r} needs at least 2 different values")
        for name, idx in part.items():
            rows = [row(records[i][text_col], d["question"], cands, vals[i], typ, f"import_{col}") for i in idx]
            if balance and name == "train" and rows:
                rows = balance_rows(rows)
            if flips and name == "train" and conds:
                rows += import_flips(rows, typ, cands, conds, flips, frng)
            split[name] += rows
    for rows in split.values():
        rng.shuffle(rows)
    return split


# ============================== typically UI import v2: from a confirmed DatasetPlan ==============================
LLM_WORKERS = 8   # parallel enrichment calls (OpenRouter rate limits are far above this)
FLIP_RATE, POLICY_RATE, SYNTH_SHARE, SYNTH_MIN, TRANSLATE_RATE, EVAL_TRANSLATE_RATE = 0.10, 0.30, 0.15, 10, 0.20, 0.10
TEXTS = {"type": "object", "properties": {"texts": {"type": "array", "items": {"type": "string"}}}, "required": ["texts"],
         "additionalProperties": False}


def case_split(state, seed, holdout):
    """Split by the hash of the normalised state, so duplicate cases never straddle splits: holdout% -> eval, 10% of the rest -> val."""
    h = hashlib.sha1(f"{seed}|{tp.norm(state)}".encode()).hexdigest()
    return "import_oneliner" if int(h[:8], 16) / 2**32 < holdout / 100 else "val" if int(h[8:16], 16) / 2**32 < 0.10 else "train"


def plan_rows(pairs, d, soft):
    """[(state, label)] -> one row per pair; soft: cases with the same normalised state (the split's identity) merge into one row
    (the first state is kept) whose target is the answer mix (label = argmax)."""
    groups = {}
    for i, (s, g) in enumerate(pairs):
        groups.setdefault(tp.norm(s) if soft else i, (s, Counter()))[1][g] += 1
    out = []
    for s, c in groups.values():
        r = row(s, d["question"], d["labels"], max(d["labels"], key=lambda l: c[l]), d["type"], f"import_{d['column']}")   # first label on ties
        # ponytail: score decisions merge to the majority hard label, never a soft target: train.py --ordinal_smooth (always on for
        # imports) overwrites a score row's target with smoothing around its argmax, so a soft mix would be silently discarded.
        if len(c) > 1 and d["type"] != "score":
            r["target"], r["meta"]["soft"] = [c[l] / sum(c.values()) for l in d["labels"]], True
        out.append(r)
    return out


def fact_conds(facts):
    """Rule conditions on the case's fact columns ("the case says <column> is <value>") for values in 3-70% of the states."""
    cnt = Counter((c, v) for f in facts.values() for c, v in f.items() if v)
    return [(f"the case says {c} is {v}", lambda s, c=c, v=v: facts[s][c] == v) for (c, v), k in sorted(cnt.items())
            if 0.03 <= k / len(facts) <= 0.70]


def label_dist(rows):
    d = {}
    for r in rows:
        if r["task"] != "import_rflip":
            d.setdefault(r["task"].removeprefix("import_"), Counter())[r["candidates"][r["label"]]] += 1
    return {k: dict(v) for k, v in d.items()}


def _copy(rows, text, lang):   # a translated case keeps its labels; ms_group must not leave its pair
    return [{**r, "state": text, "meta": {**{k: v for k, v in r["meta"].items() if k != "ms_group"}, "lang": lang}} for r in rows]


def _synthetic_tasks(d, pairs, budget, rng, emit):
    """LLM tasks (prompt, estimated output chars, apply(texts)) that write new cases for the decision's rare labels
    (< max(10, 1% of its train rows)): 5 real cases of the label + 5 of a confusable one; then as many for the majority label
    (style balance). Capped at 1x the real count per label; budget = [rows left of the 15%-of-train cap]."""
    labels, by = d["labels"], {}
    for s, g in pairs:
        by.setdefault(g, []).append(s)
    n_of = lambda l: len(by.get(l, ()))
    major, tasks = max(labels, key=n_of), []
    for lab in labels:
        if lab == major or not 0 < n_of(lab) < max(SYNTH_MIN, 0.01 * len(pairs)):
            continue
        n = min(n_of(lab), budget[0] // 2)
        if n < 1:
            break
        budget[0] -= 2 * n
        others = [l for l in labels if l != lab and n_of(l)]   # never empty: the majority label is in it
        conf = min(others, key=lambda l: abs(labels.index(l) - labels.index(lab))) if d["type"] == "score" else max(others, key=n_of)
        for target, shown in ((lab, [lab, conf]), (major, [major])):
            ex = [(l, rng.sample(by[l], min(5, n_of(l)))) for l in shown]
            prompt = (f'A company labelled its past cases for the question "{d["question"]}". Possible labels: {", ".join(labels)}.\n\n'
                      + "\n\n".join(f'Real cases labelled "{l}":\n' + "\n".join(f"---\n{s}" for s in ss) + "\n---" for l, ss in ex)
                      + f'\n\nWrite {n} NEW, different cases that this company would label "{target}". Match the format, length, language '
                        f'and tone of the real cases; do not copy them and do not mention the label. Return {{"texts": [...]}} with exactly {n} strings.')
            avg = sum(len(s) for _, ss in ex for s in ss) / sum(len(ss) for _, ss in ex)
            tasks.append((prompt, n * avg, lambda texts, t=target, n=n: emit(d, t, texts[:n])))
    return tasks


def _translate_tasks(states, lang, emit):
    """LLM tasks that translate cases (in chunks of <= 20 cases / 20k chars); emit(lang, state, translation) per case."""
    chunks, cur, size = [], [], 0
    for s in states:
        if cur and (len(cur) >= 20 or size + len(s) > 20_000):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(s)
    chunks += [cur] if cur else []

    def apply(texts, chunk):
        if len(texts) != len(chunk):
            raise ValueError("wrong number of translations")
        for s, t in zip(chunk, texts):
            if isinstance(t, str) and t.strip():
                emit(lang, s, t)
    return [(f"Translate each case into the language with ISO 639-1 code '{lang}'. Keep names, numbers, ids, product codes and line "
             "breaks unchanged. Translate recurring field labels and category words (plan names, tiers, regions) the same way "
             f'every time, in every batch, or keep them in the original language. Return {{"texts": [...]}} with exactly {len(c)} strings in the same order.\n\n'
             + json.dumps(c, ensure_ascii=False), 1.5 * sum(map(len, c)), lambda texts, c=c: apply(texts, c)) for c in chunks]


def build_from_plan(records, plan, enrich, settings, rng, llm=None):
    """records + a confirmed DatasetPlan -> (split dict for write(), stats). enrich = {balance, dedupe_soft, policy: {column: text},
    synthetic, languages: [code]}, settings = {holdout %, seed}; llm = tllm.complete_json (or a stand-in), needed only for synthetic / languages.
    Eval and val are never enriched (except the separate translated eval files import_<lang>). Rule flips are always on (10%, paired).
    ponytail: split is by case hash only; a time split (newest held out) for a timestamp column is not implemented."""
    decisions = [d for d in plan["decisions"] if d["include"]]
    if not decisions:
        raise ValueError("the plan has no included decision")
    seed, holdout = settings.get("seed", 0), settings.get("holdout", 20)
    fact_cols = [q["column"] for q in plan["case"]["parts"] if q["role"] == "fact"]
    part, facts, warnings = {"train": [], "val": [], "import_oneliner": []}, {}, []
    for r in records:
        state = tp.render_case(r, plan)
        if state:
            name = case_split(state, seed, holdout)
            part[name].append((state, tp.answers(r, plan)))
            if name == "train":
                facts.setdefault(state, {c: tp.norm(r.get(c, "")) for c in fact_cols})
    conds = (fact_conds(facts) if fact_cols else []) or keyword_conds(list(facts))   # fact conditions first, keywords when there are none
    frng = random.Random(seed + 1)   # own stream: flips must not shift the other draws
    split, base, pairs, before, dropped, soft = {k: [] for k in part}, {"train": {}, "import_oneliner": {}}, {}, {}, {}, 0
    for d in decisions:
        col, policy = d["column"], ((enrich.get("policy") or {}).get(d["column"]) or "").strip()
        dropped[col] = sum(a.get(col) is None for v in part.values() for _, a in v)
        for name, items in part.items():
            ps = [(s, a[col]) for s, a in items if a.get(col) is not None]
            rows = plan_rows(ps, d, name == "train" and enrich.get("dedupe_soft", True))
            if name == "val":
                split[name] += rows
                continue
            for r in rows:
                base[name].setdefault(r["state"], []).append(r)   # unbalanced, unflipped: what translation copies
            if name == "import_oneliner":
                split[name] += rows
                continue
            pairs[col], before[col] = ps, dict(Counter(g for _, g in ps))
            soft += sum("soft" in r["meta"] for r in rows)
            for r in rows if policy else []:
                if rng.random() < POLICY_RATE:
                    r["query"], r["meta"] = f"{r['query']}\n{policy}", {**r["meta"], "policy": True}
            if enrich.get("balance", True) and rows:
                rows = balance_rows(rows)
            if conds:
                rows += import_flips(rows, d["type"], d["labels"], conds, FLIP_RATE, frng)
            split[name] += rows
    if any(not v for v in split.values()):
        raise ValueError("not enough usable rows: a split came out empty; add more rows or lower the held-out share")
    syn, tr, ev, cost, collisions = [], [], {}, 0.0, 0
    langs = enrich.get("languages") or []
    if (enrich.get("synthetic") or langs) and llm is None:
        warnings.append("Synthetic cases and added languages are not available right now; they were skipped.")
    elif enrich.get("synthetic") or langs:
        if bad := [c for c in langs if not re.fullmatch(r"[a-z]{2,3}", str(c))]:
            raise ValueError(f"unknown language code {bad[0]!r}; use a two-letter code such as es or de")
        seen, tasks = {r["state"] for v in split.values() for r in v}, []

        def emit_syn(d, lab, texts):
            for t in texts:
                if isinstance(t, str) and t.strip() and t not in seen:
                    seen.add(t)
                    syn.append(row(t, d["question"], d["labels"], lab, d["type"], f"import_{d['column']}", synthetic=True))
        if enrich.get("synthetic"):
            budget = [int(SYNTH_SHARE * len(split["train"]))]
            for d in decisions:
                tasks += _synthetic_tasks(d, pairs[d["column"]], budget, rng, emit_syn)
        ev_states = rng.sample(sorted(base["import_oneliner"]), max(1, round(EVAL_TRANSLATE_RATE * len(base["import_oneliner"]))))
        for lang in langs:
            tr_states = rng.sample(sorted(base["train"]), max(1, round(TRANSLATE_RATE * len(base["train"]))))
            tasks += _translate_tasks(tr_states, lang, lambda l, s, t: tr.extend(_copy(base["train"][s], t, l)))
            tasks += _translate_tasks(ev_states, lang, lambda l, s, t: ev.setdefault(l, []).extend(_copy(base["import_oneliner"][s], t, l)))
        def call(task):   # no spend cap: the operator pays (llm_cost_usd is still reported)
            prompt, out_chars, _ = task
            try:
                return llm("Return only the JSON requested.", prompt, TEXTS, max_tokens=min(16000, int(out_chars / 2) + 500))
            except Exception as e:   # experimental: one failed call skips its cases; never echo the message (may hold the key)
                return e
        # calls run in parallel; results are applied in task order on this thread (apply mutates shared lists; reproducible)
        with ThreadPoolExecutor(LLM_WORKERS) as pool:
            for (_, _, apply), res in zip(tasks, pool.map(call, tasks)):
                if isinstance(res, Exception):
                    warnings.append(f"An AI enrichment call failed ({type(res).__name__}); its cases were skipped.")
                    continue
                out, usage = res
                cost += tllm.usd(usage["input_tokens"], usage["output_tokens"])
                apply(out["texts"])
        # a synthetic / translated case must not share a normalised state with the other side: train side vs val + real eval + eval
        # translations, then eval translations vs the final train
        norms = lambda rows: {tp.norm(r["state"]) for r in rows}
        held = norms(split["val"]) | norms(split["import_oneliner"]) | norms(r for v in ev.values() for r in v)
        new = [r for r in syn + tr if tp.norm(r["state"]) not in held]
        train_n = norms(split["train"]) | norms(new)
        kept = {l: [r for r in v if tp.norm(r["state"]) not in train_n] for l, v in ev.items()}
        collisions = len(syn) + len(tr) - len(new) + sum(len(v) for v in ev.values()) - sum(len(v) for v in kept.values())
        syn, tr, ev = [r for r in new if r["meta"].get("synthetic")], [r for r in new if not r["meta"].get("synthetic")], kept
        split["train"] += new
        split.update({f"import_{l}": rows for l, rows in ev.items()})
    for rows in split.values():
        rng.shuffle(rows)
    return split, {"rows": {k: len(v) for k, v in split.items()}, "labels": {"before": before, "after": label_dist(split["train"])},
                   "flips": sum(r["task"] == "import_rflip" for r in split["train"]), "soft_rows": soft, "dropped": dropped,
                   "synthetic": len(syn), "collisions": collisions, "translated": {"train": len(tr), "eval": {l: len(v) for l, v in ev.items()}},
                   "llm_cost_usd": round(cost, 2), "warnings": warnings}


def write(out, split):
    (out / "eval").mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        path = out / (f"{name}.jsonl" if name in ("train", "val") else f"eval/{name}.jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {len(rows)}  {Counter(r['task'] for r in rows).most_common(3)}")


def selftest():
    a = {"issue": "delivery_delay", "days": 12, "tier": "enterprise", "value": 250, "legal": False, "prior": 0, "mood": "calm"}
    assert route(a) == "Claims" and urgency(a) == 2 and refund(a) and not escalate(a)
    assert route({**a, "issue": "invoice_dispute"}) == "Sales" and route({**a, "issue": "invoice_dispute", "tier": "business"}) == "Billing"
    assert urgency({**a, "issue": "quote_request", "tier": "starter"}) == 0 and not refund({**a, "tier": "starter"})
    assert "overdue" in header_words({**a, "issue": "lost_package", "customer": "X", "region": "EU"})
    assert escalate({**a, "mood": "furious"}) and escalate({**a, "prior": 3}) and not escalate({**a, "tier": "business", "mood": "furious"})


if __name__ == "__main__":
    selftest()
    if sys.argv[1:] == ["attrs"]:
        cmd_attrs()
    elif sys.argv[1:] == ["build"]:
        write(ROOT / "data_co_a", build_a(random.Random(0)))
        write(ROOT / "data_co_b", build_b(random.Random(0)))
        write(ROOT / "data_co_c", build_a(random.Random(0), header_words, "c"))   # spike 2: same tickets, numbers in words
        write(ROOT / "data_co_d", build_d(random.Random(0)))
        write(ROOT / "data_co_e", build_a(random.Random(0), header, "e", diverse_flips=True))   # spike 2: flips on every question
        # spike 3: f = fewer flips, each paired with a plain one-liner in one micro-batch; g = refund twins (policy vs rule-following)
        for slug, kw in (("f", dict(flip_rate=0.10, pair=True)), ("g", dict(twins=True))):
            write(ROOT / f"data_co_{slug}", build_a(random.Random(0), header, slug, diverse_flips=True, **kw))
            for f in (ROOT / f"data_co_{slug}" / "eval").glob(f"{slug}_*.jsonl"):   # same tickets as a: states + labels row for row
                ref = [json.loads(l) for l in open(ROOT / "data_co_a" / "eval" / f.name.replace(f"{slug}_", "a_", 1))]
                got = [json.loads(l) for l in open(f)]
                assert [(r["state"], r["label"]) for r in got] == [(r["state"], r["label"]) for r in ref], f
