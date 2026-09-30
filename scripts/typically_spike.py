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
  data_co_d    spike 2: imported with clean labels -- Bitext utterances, company-specific intent -> desk map

uv run scripts/typically_spike.py attrs     # -> .context/typically/body_specs.jsonl (what the agent must write)
uv run scripts/typically_spike.py build     # -> data_co_a/, data_co_b/
"""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def build_a(rng, hdr=header, p="a"):
    bodies, dropped = {}, 0
    for line in open(CTX / "bodies.jsonl"):
        b = json.loads(line)
        if b["legal"] != any(w in b["text"].lower() for w in LEGAL_WORDS):   # body contradicts its spec -> label noise
            dropped += 1
            continue
        bodies.setdefault((b["issue"], b["mood"], b["legal"], b["variant"] in TEST_VARIANTS), []).append(b["text"])
    print(f"bodies: {sum(map(len, bodies.values()))} kept, {dropped} dropped")
    split = {"train": [], "val": [], **{f"{p}_{k}": [] for k in ("oneliner", "rubric", "flip", "flip_heldout", "novel")}}
    for n, test in ((N_TRAIN, False), (N_TEST, True)):
        for t in range(n):
            a = sample_attrs(rng)
            state = hdr(a) + "\n\n" + rng.choice(bodies[(a["issue"], a["mood"], a["legal"], test)])
            for key, qtype, q, cands, fn, crit in DECISIONS:
                gold, task = fn(a), f"co_{p}_{key}"
                if test:
                    split[f"{p}_oneliner"].append(row(state, q, cands, gold, qtype, task))
                    split[f"{p}_rubric"].append(row(state, query(q, crit, qtype), cands, gold, qtype, task))
                    continue
                # train mix: one-liner mostly, the written policy 30% of the time so the rubric stays load-bearing
                written = rng.random() < 0.3
                dest = "val" if t % 10 == 0 else "train"
                split[dest].append(row(state, query(q, crit if written else None, qtype), cands, gold, qtype, task))
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
    import re
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


def write(out, split):
    (out / "eval").mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        path = out / (f"{name}.jsonl" if name in ("train", "val") else f"eval/{name}.jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{path.relative_to(ROOT)}: {len(rows)}  {Counter(r['task'] for r in rows).most_common(3)}")


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
