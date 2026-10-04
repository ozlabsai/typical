"""PLAN6 item 4: rubric-conditioned typed-workflow corpus (Noul / Choice / Score) in JevBench's *shape*.

Builds data_wf/{train,val}.jsonl + data_wf/eval/*.jsonl + manifest.json (row schema state/query/candidates/
target/p_null/task/label/meta, see pcdm/data.py). Rows render exactly like pcdm_jev/decider.py at benchmark
time: query = query_text({"type","instructions","criteria"}), candidates = the label strings verbatim.
Nothing is copied from JevBench: every rubric, policy, scenario and label set here is our own; a leak check
against the public JevBench states (exact + normalised stem) asserts 0 hits.

Families (task = wf_<family>; meta.family / rubric_style / qtype / fam_bucket="W"):
  noul   policy_permit (rule engine), adequacy (SQuAD), support (SNLI/MNLI), fact (BoolQ), eligibility [HELD OUT]
  choice routing (intent sets), action_select (rule engine), enum_extract (templates), categorical (topic/
         sentiment with per-label criteria), tool_select [HELD OUT]
  score  quality (yelp/sst5), severity (templated incidents), relevance (SQuAD), completeness (templated),
         urgency [HELD OUT]
Rubric groups (PLAN6 "Queue review"): ~half the rows of the programmatic families come in groups of 2-3 rows
with the SAME state and SAME labels but different rubrics -> different gold (meta.rubric_group /
rubric_variant): policy A vs B in the instructions, permuted per-option criteria over opaque labels,
current-vs-original value, three severity/urgency rule sets, TBD-counts-missing vs -present, thresholds.
One rubric style per type (the last in each *_STYLES list) never appears in train/val.
Eval files: wf_heldout_{noul,choice,score} (held-out families), wf_heldout_style (trained families, held-out
style), wf_rubric_flip (held-out-family pairs (x,r1,A,y1)/(x,r2,A,y2), y1!=y2, meta.flip_pair; scored by
metrics.rubric_flip), wf_rubric_shuffled (held-out rows with another same-label row's rubric -> Δ_r).

--long_share TARGET (PLAN7 track B, long_e45 run): policy_permit/action_select rows are long (1-2.6k-token,
multi-domain-policy states, assemble_state's `long` flag) at LONG_FRAC (8%) by default -- a small share of
train.jsonl. --long_share raises the *combined* train.jsonl + train_long.jsonl long share to TARGET (e.g.
0.25) by GENERATING extra always-long rows (gen_long_extra) into a separate <out>/train_long.jsonl, so
train.jsonl itself is byte-identical to a run without the flag; point a training run's --extra_data at both
files (e.g. --bucket_map data_wf_long=W) to train on the raised mix.

uv run scripts/workflow_corpus.py [--out data_wf] [--limit N] [--seed 0] [--jevbench /tmp/jevbench/datasets/public]
                                   [--long_share TARGET]
"""
import argparse
import datetime as dt
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from data import norm_text, group_split, write_jsonl, words  # noqa: E402
from pcdm_jev.decider import query_text  # noqa: E402  -- the benchmark-time renderer

VAL_TOTAL = 3000
HELDOUT_N = 1500
NULL_FRAC = 0.15   # single choice rows with the gold removed (p_null=1); groups never -> ~5% of all choice rows
LONG_FRAC = 0.08   # single policy rows whose state is a 1-3k-token concatenation of policy docs
GROUP_FRAC = 0.35  # share of a groupable family's rows generated as rubric groups
MAX_STATE_CHARS = 14000
CAPS = dict(policy_permit=9000, adequacy=5000, support=6000, fact=5000, eligibility=HELDOUT_N,
            routing=9000, action_select=7000, enum_extract=7000, categorical=7000, tool_select=HELDOUT_N,
            quality=6000, severity=6000, relevance=4000, completeness=4000, urgency=HELDOUT_N)
HELD_OUT = {"noul": "eligibility", "choice": "tool_select", "score": "urgency"}
_gid = itertools.count(1)


def lc(s):
    return s[0].lower() + s[1:] if s else s


def cap(s):
    return s[0].upper() + s[1:] if s else s


# ============================== rubric styles ==============================
# Each style maps a family "spec" (what yes/no / each option / each level means) to
# (instructions, criteria). The last style of each list is the held-out style.
NOUL_STYLES = [
    ("plain", lambda f: (f"{cap(f['q'])}? {f['caveat']}", {"true": f["yes"], "false": f["no"]})),
    ("checklist", lambda f: (f"Decide yes or no: {f['q']}. Base the decision only on the state; {lc(f['caveat'])}",
                             {"true": f"Answer yes when {lc(f['yes'])}", "false": f"Answer no when {lc(f['no'])}"})),
    ("formal", lambda f: (f"Determine the answer to: {f['q']}? Use only the information given.",
                          {"true": f"YES - {f['yes']}", "false": f"NO - {f['no']}"})),
    ("terse", lambda f: (f"{cap(f['q'])}?", {"true": f["yes"], "false": f["no"]})),
    ("reviewer", lambda f: (f"You are the reviewer. {cap(f['q'])}? {f['caveat']} Do not assume facts that are not stated.",
                            {"true": f"Condition for yes: {lc(f['yes'])}", "false": f"Condition for no: {lc(f['no'])}"})),
    ("iff", lambda f: (f"Return yes if and only if {lc(f['yes'])} Otherwise return no. Question: {f['q']}.",
                       {"true": f["yes"], "false": f["no"]})),
    ("flag", lambda f: (f"Evaluate the boolean flag `{f['flag']}` for the state below. {f['caveat']}",
                        {"true": f"The flag is true when {lc(f['yes'])}", "false": f"The flag is false when {lc(f['no'])}"})),
]
CHOICE_STYLES = [
    ("plain", lambda f: (f"{cap(f['task'])}. {f['caveat']}", dict(f["criteria"]))),
    ("select", lambda f: (f"Select exactly one {f['noun']} for the state. {f['caveat']}",
                          {k: f"Choose when {lc(v)}" for k, v in f["criteria"].items()})),
    ("which", lambda f: (f"Which {f['noun']} applies? Each option's criterion is listed; pick the single best match.",
                         dict(f["criteria"]))),
    ("classify", lambda f: (f"Classify the state into one of the listed options ({f['noun']}). The option descriptions are the "
                            f"definitions; {lc(f['caveat'])}", {k: f"{v}." for k, v in f["criteria"].items()})),
    ("strict", lambda f: (f"{cap(f['task'])}. Choose the option whose criterion is fully met by the state; when two match, "
                          f"prefer the more specific one.", dict(f["criteria"]))),
    ("short", lambda f: (f"{cap(f['task'])}.", dict(f["criteria"]))),
    ("spec", lambda f: (f"Decision: {f['noun']}. Rule: read every option's criterion, then output the one the state "
                        f"satisfies. {f['caveat']}", {k: f"applies if {lc(v)}" for k, v in f["criteria"].items()})),
]


def _crit(f, fmt=lambda i, n, d: d):
    """score criteria: list when labels are "0".."n" (rendered `i: desc`), dict when they are level names."""
    items = [fmt(i, n, d) for i, (n, d) in enumerate(zip(f["names"], f["levels"]))]
    return items if f["numeric"] else dict(zip(f["names"], items))


SCORE_STYLES = [
    ("plain", lambda f: (f"Rate the {f['what']} on the scale below, from {f['names'][0]} (lowest) to {f['names'][-1]} "
                         f"(highest). {f['caveat']}", _crit(f))),
    ("highest", lambda f: (f"Assign the highest level of {f['what']} whose description is fully supported by the state. "
                           f"Levels are ordered {' < '.join(f['names'])}.", _crit(f))),
    ("ordinal", lambda f: (f"Score the {f['what']}. The options are ordinal: {' < '.join(f['names'])}. {f['caveat']}",
                           _crit(f, lambda i, n, d: f"{d}."))),
    ("rubric", lambda f: (f"Grade the {f['what']} using the rubric. Pick exactly one level; levels increase from "
                          f"{f['names'][0]} to {f['names'][-1]}.", _crit(f, lambda i, n, d: f"Level {n}: {lc(d)}"))),
    ("judge", lambda f: (f"As a grader, place the state at the matching level of {f['what']}. Do not round up; "
                         f"{lc(f['caveat'])}", _crit(f))),
    ("terse", lambda f: (f"{cap(f['what'])} level?", _crit(f))),
    ("band", lambda f: (f"Which band best describes the {f['what']}? Bands run {f['names'][0]} (lowest) through "
                        f"{f['names'][-1]} (highest); {lc(f['caveat'])}", _crit(f, lambda i, n, d: f"band {n} - {lc(d)}"))),
]
STYLES = {"noul": NOUL_STYLES, "choice": CHOICE_STYLES, "score": SCORE_STYLES}


def make_row(qtype, family, spec, state, labels, gold, style, p_null=0.0, extra_meta=None):
    name, fn = style
    instructions, criteria = fn(spec)
    if spec.get("prefix"):  # rubric groups: the deciding rule (policy / thresholds) lives in the instructions
        instructions = f"{spec['prefix']} {instructions}"
    question = {"type": qtype, "instructions": instructions, "criteria": criteria}
    query = query_text(question)
    if p_null and len(labels) < 3:
        p_null = 0.0  # never leave a single candidate
    if p_null:
        labels, gold = [l for i, l in enumerate(labels) if i != gold], -1
        target = [1.0 / len(labels)] * len(labels)
    else:
        target = [1.0 if i == gold else 0.0 for i in range(len(labels))]
    return {"state": state[:MAX_STATE_CHARS], "question": question, "query": query,
            "candidates": list(labels), "target": target,
            "p_null": float(p_null), "task": f"wf_{family}", "label": gold,
            "meta": {"family": family, "rubric_style": name, "qtype": qtype, "fam_bucket": "W", **(extra_meta or {})}}


def pick_style(qtype, rng, heldout=False):
    styles = STYLES[qtype]
    return styles[-1] if heldout else rng.choice(styles[:-1])


def emit_group(qtype, family, state, labels, variants, rng, heldout_style, extra_meta=None):
    """variants: [(spec, gold)] over one state + one label list -> one row each, tagged as a rubric group.
    Callers guarantee len({gold}) > 1."""
    gid = f"{family}-{next(_gid)}"
    return [make_row(qtype, family, spec, state, labels, gold, pick_style(qtype, rng, heldout_style),
                     extra_meta={"rubric_group": gid, "rubric_variant": vi, **(extra_meta or {})})
            for vi, (spec, gold) in enumerate(variants)]


def differing(variants):
    return len({g for _, g in variants}) > 1


def fill_to(n, single, group, rng, group_frac=GROUP_FRAC, tries=12):
    """Generate ~n rows: with prob group_frac a rubric group (2-3 rows, retried until golds differ), else one row."""
    rows = []
    while len(rows) < n:
        if group is not None and rng.random() < group_frac:
            for _ in range(tries):
                g = group()
                if g:
                    rows += g
                    break
            else:
                rows += single()
        else:
            rows += single()
    return rows  # ponytail: may overshoot n by <= 2 rows rather than cut a group


# ============================== policy engine ==============================
# cond = (requirement, satisfied-fact, violated-fact, unproven-fact-or-None(omit)); prohib = (rule, triggered, clear)
DOMAINS = {
    "refund": dict(
        action="issue a refund", noun="refunds",
        conds=[("a receipt or order number is on file", "The customer provided the order number.",
                "No receipt or order number could be located.", "Whether a receipt exists is not recorded."),
               ("the purchase is no more than 30 days old", "The purchase was made 12 days ago.",
                "The purchase was made 47 days ago.", "The purchase date is not recorded."),
               ("the item is unused", "The item is unopened.", "The item shows clear signs of use.", None),
               ("the refund amount is under $500", "The refund would be $120.", "The refund would be $780.",
                "The refund amount has not been calculated."),
               ("the order was shipped to the billing address", "Shipping and billing addresses match.",
                "The order was shipped to a different address than the billing address.", None)],
        prohibs=[("no refund is issued for clearance items", "The item was bought on clearance.",
                  "The item was sold at regular price."),
                 ("no refund is issued when a chargeback is already open", "A chargeback is pending on this order.",
                  "No chargeback exists for the order.")]),
    "access": dict(
        action="grant write access to the production repository", noun="access grants",
        conds=[("the requester has completed security training", "Security training was completed last month.",
                "Security training has lapsed.", "Training status is unknown."),
               ("a manager has approved the request", "The manager approved the request in the ticket.",
                "The manager declined to approve.", "No manager response is recorded yet."),
               ("the requester has been employed for at least 90 days", "The requester joined 8 months ago.",
                "The requester joined 3 weeks ago.", None),
               ("two-factor authentication is enabled", "2FA is enabled on the account.",
                "The account has no 2FA.", "2FA status was not checked.")],
        prohibs=[("contractors may never receive write access", "The requester is a contractor.",
                  "The requester is a full-time employee."),
                 ("access is never granted during a change freeze", "A change freeze is in effect this week.",
                  "No change freeze is active.")]),
    "expense": dict(
        action="approve the expense claim", noun="expense approvals",
        conds=[("an itemised receipt is attached", "An itemised receipt is attached.",
                "Only a card statement line is attached, no receipt.", "The attachment has not been reviewed."),
               ("the claim is filed within 60 days of the expense", "The expense was 20 days ago.",
                "The expense was 95 days ago.", "The expense date is missing from the claim."),
               ("the amount is within the $200 per-day meal limit", "Meals total $85 for the day.",
                "Meals total $310 for the day.", None),
               ("the trip was pre-approved", "The trip appears on the approved travel list.",
                "The trip was never submitted for approval.", "Pre-approval could not be verified.")],
        prohibs=[("alcohol is never reimbursed", "The receipt includes a bottle of wine.",
                  "The receipt contains no alcohol."),
                 ("claims from suspended cost centres are not approved", "The cost centre is suspended.",
                  "The cost centre is active.")]),
    "shipping": dict(
        action="ship the order internationally", noun="international shipments",
        conds=[("the destination country is on the supported list", "The destination is Germany, which is supported.",
                "The destination country is not on the supported list.", "The destination is not stated."),
               ("the parcel weighs at most 20 kg", "The parcel weighs 6 kg.", "The parcel weighs 32 kg.",
                "The parcel has not been weighed."),
               ("customs paperwork is complete", "The customs form is filled in and signed.",
                "The customs form is missing.", None),
               ("payment has cleared", "Payment cleared yesterday.", "Payment is still pending.", "Payment status is unknown.")],
        prohibs=[("lithium batteries are never shipped by air", "The parcel contains loose lithium batteries and goes by air.",
                  "The parcel contains no batteries."),
                 ("orders flagged for fraud review are not shipped", "The order is flagged for fraud review.",
                  "The order has no fraud flag.")]),
    "leave": dict(
        action="approve the leave request", noun="leave approvals",
        conds=[("the employee has enough accrued leave", "The employee has 9 days accrued and asks for 4.",
                "The employee has 1 day accrued and asks for 5.", "The accrued balance was not provided."),
               ("the request is made at least 14 days in advance", "The request was submitted 3 weeks ahead.",
                "The request was submitted 2 days ahead.", None),
               ("a cover person is named", "A colleague has agreed to cover.", "No cover has been arranged.",
                "The cover field is blank."),
               ("the team keeps at least two people on duty", "Three other team members remain on duty.",
                "Everyone else on the team is already off that week.", None)],
        prohibs=[("leave is not approved during the annual audit week", "The dates fall in audit week.",
                  "The dates are outside audit week."),
                 ("employees on a performance plan cannot take discretionary leave", "The employee is on a performance plan.",
                  "The employee is in good standing.")]),
    "discount": dict(
        action="apply the promotional discount", noun="promotional discounts",
        conds=[("the promo code is still valid", "The code expires next month.", "The code expired last week.",
                "The code's expiry is not shown."),
               ("the cart total is at least $50", "The cart total is $72.", "The cart total is $31.", None),
               ("the customer has not used the code before", "This is the customer's first use of the code.",
                "The customer already used this code twice.", "Usage history is unavailable."),
               ("the account is in good standing", "The account has no outstanding balance.",
                "The account is past due.", None)],
        prohibs=[("discounts never apply to gift cards", "The cart contains only gift cards.",
                  "The cart contains no gift cards."),
                 ("promo codes cannot be combined with a sale price", "The items are already on sale.",
                  "The items are at full price.")]),
    "publish": dict(
        action="publish the article", noun="publications",
        conds=[("an editor has signed off", "The editor signed off this morning.", "The editor requested changes.",
                "Editor sign-off is not recorded."),
               ("all images are licensed", "Every image has a license on file.",
                "One image has no license.", "Image licensing was not checked."),
               ("legal review is complete for named individuals", "Legal review is complete.",
                "Legal review found an unresolved issue.", None),
               ("the piece is under 3,000 words", "The piece is 1,800 words.", "The piece is 4,200 words.", None)],
        prohibs=[("articles are never published under embargo", "The story is under embargo until Friday.",
                  "There is no embargo."),
                 ("pieces by suspended contributors are not published", "The author's contributor account is suspended.",
                  "The author is an active contributor.")]),
    "export": dict(
        action="export the customer's personal data", noun="data exports",
        conds=[("the requester's identity is verified", "Identity was verified by two-factor confirmation.",
                "The identity check failed.", "Identity has not been verified yet."),
               ("the request comes from the account owner", "The request comes from the account owner.",
                "The request comes from a third party.", "The requester's relationship to the account is unclear."),
               ("the account is not under legal hold", "No legal hold applies.", "A legal hold is active on the account.", None),
               ("the export format is CSV or JSON", "CSV was requested.", "A PDF dump was requested.", None)],
        prohibs=[("exports are never sent to unverified email addresses", "The destination email is unverified.",
                  "The destination email is verified."),
                 ("minors' data is never exported without guardian consent", "The account belongs to a minor and no guardian consent is on file.",
                  "The account holder is an adult.")]),
}
DISTRACTORS = ["The customer is based in Ohio.", "The ticket was opened on a Tuesday.", "The agent's name is Priya.",
               "The conversation took place over chat.", "The account was created in 2021.", "The customer prefers email.",
               "Weather at the warehouse was clear.", "The request reference is #{n}.", "The customer speaks Spanish.",
               "The team lead is on holiday.", "The last login was two days ago."]
ESCALATIONS = ["The amount involved exceeds the escalation threshold.", "The customer is a VIP account.",
               "This is the third identical request this month.", "The requester mentioned legal action."]


def render_policy(dom, conds, prohibs, rng):
    d = DOMAINS[dom]
    reqs = [c[0] for c in conds]
    v = rng.randint(0, 2)
    if v == 0:
        pol = f"Policy: to {d['action']}, {', '.join(reqs[:-1])}{' and ' if len(reqs) > 1 else ''}{reqs[-1]} must hold."
        pol += "".join(f" {cap(p[0])}." for p in prohibs)
    elif v == 1:
        lines = [f"{i + 1}. {cap(r)}." for i, r in enumerate(reqs)] + [f"{len(reqs) + i + 1}. {cap(p[0])}." for i, p in enumerate(prohibs)]
        pol = f"Rules for {d['noun']}:\n" + "\n".join(lines)
    else:
        pol = f"{cap(d['noun'])} require that " + "; ".join(reqs) + "."
        if prohibs:
            pol += " Exceptions: " + "; ".join(p[0] for p in prohibs) + "."
    return pol


def policy_case(dom, rng, want_yes=None, esc=False, n_conds=(2, 4), n_proh=(0, 2)):
    """A scenario over a superset of conditions. -> dict(conds=[(cond, status)], prohibs=[(proh, triggered)],
    facts, request, esc). want_yes None: statuses random (groups); True/False: balanced single rows."""
    d = DOMAINS[dom]
    conds = rng.sample(d["conds"], rng.randint(n_conds[0], min(n_conds[1], len(d["conds"]))))
    prohibs = rng.sample(d["prohibs"], rng.randint(n_proh[0], min(n_proh[1], len(d["prohibs"]))))
    if want_yes is None:
        statuses = [rng.choices(["sat", "viol", "unk"], [0.6, 0.15, 0.25])[0] for _ in conds]
        trig = [rng.random() < 0.3 for _ in prohibs]
    elif want_yes:
        statuses, trig = ["sat"] * len(conds), [False] * len(prohibs)
    else:
        statuses, trig = ["sat"] * len(conds), [False] * len(prohibs)
        options = [("cond", i) for i in range(len(conds))] + [("proh", i) for i in range(len(prohibs))]
        for kind, i in rng.sample(options, rng.randint(1, min(2, len(options)))):
            if kind == "cond":
                statuses[i] = rng.choice(["viol", "unk"])
            else:
                trig[i] = True
        for i in range(len(conds)):
            if statuses[i] == "sat" and rng.random() < 0.15:
                statuses[i] = rng.choice(["viol", "unk"])
    facts = []
    for c, s in zip(conds, statuses):
        if s == "sat":
            facts.append(c[1])
        elif s == "viol":
            facts.append(c[2])
        elif c[3] and rng.random() < 0.6:
            facts.append(c[3])  # explicit "unknown"; else silently omitted
    for p, t in zip(prohibs, trig):
        if t:
            facts.append(p[1])
        elif rng.random() < 0.6:
            facts.append(p[2])
    if esc:
        facts.append(rng.choice(ESCALATIONS))
    facts += [x.replace("{n}", str(rng.randint(1000, 9999))) for x in rng.sample(DISTRACTORS, rng.randint(0, 2))]
    rng.shuffle(facts)
    request = rng.choice([f"Request: {d['action']}.", f"The agent wants to {d['action']}.",
                          f"Proposed action: {d['action']}.", f"May we {d['action']}?"])
    return dict(dom=dom, conds=list(zip(conds, statuses)), prohibs=list(zip(prohibs, trig)),
                facts=" ".join(facts), request=request, esc=esc)


def verdict(case, ci=None, pi=None):
    """Evaluate the case under the policy made of condition indices ci / prohibition indices pi (None = all)."""
    conds = [case["conds"][i] for i in ci] if ci is not None else case["conds"]
    prohs = [case["prohibs"][i] for i in pi] if pi is not None else case["prohibs"]
    return dict(prohibited=any(t for _, t in prohs), failed=any(s == "viol" for _, s in conds),
                unproven=any(s == "unk" for _, s in conds), escalate=case["esc"])


def permitted(v):
    return not (v["prohibited"] or v["failed"] or v["unproven"])


def action_of(v):  # precedence: prohibition > escalation > unmet > unproven > allow
    return 1 if v["prohibited"] else 3 if v["escalate"] else 1 if v["failed"] else 2 if v["unproven"] else 0


def policy_text(case, ci=None, pi=None, rng=None):
    conds = [case["conds"][i][0] for i in ci] if ci is not None else [c for c, _ in case["conds"]]
    prohs = [case["prohibs"][i][0] for i in pi] if pi is not None else [p for p, _ in case["prohibs"]]
    return render_policy(case["dom"], conds, prohs, rng)


def long_prefix(dom, rng):
    """Filler policy docs from OTHER domains (1-3k tokens) to bury the relevant policy in."""
    secs = []
    for _ in range(rng.randint(20, 55)):
        od = rng.choice([k for k in DOMAINS if k != dom])
        d = DOMAINS[od]
        secs.append(f"## {cap(d['noun'])}\n" + render_policy(od, rng.sample(d["conds"], rng.randint(2, 4)),
                                                          rng.sample(d["prohibs"], rng.randint(0, 2)), rng))
    return secs


def assemble_state(policy, facts, request, dom, rng, long):
    if long:
        # tl1b (REPORT S3ab/S3af): Case FIRST, policy filler after -- states right-truncate at
        # --max_state at train/eval time, so a Case rendered last (past the 20-55 filler
        # sections + real policy) was silently dropped 98.8% of the time at max_state=1,024,
        # training the model to answer confidently from facts it never saw.
        secs = long_prefix(dom, rng)
        secs.insert(rng.randint(0, len(secs)), f"## {cap(DOMAINS[dom]['noun'])}\n{policy}")
        return f"Case: {facts} {request}\n\n" + "\n\n".join(secs)
    order = rng.random()
    if order < 0.6:
        return f"{policy}\n{facts} {request}"
    if order < 0.8:
        return f"{facts} {request}\n{policy}"
    return f"{policy} {facts} {request}"


def policy_subsets(case, rng, k=None):
    """2-3 distinct (cond idx, proh idx) policies over the case's superset, for rubric groups."""
    k = k or rng.randint(2, 3)
    out, seen = [], set()
    for _ in range(20):
        ci = tuple(sorted(rng.sample(range(len(case["conds"])), rng.randint(2, min(3, len(case["conds"]))))))
        pi = tuple(sorted(rng.sample(range(len(case["prohibs"])), rng.randint(0, len(case["prohibs"])))))
        if (ci, pi) not in seen:
            seen.add((ci, pi)); out.append((ci, pi))
        if len(out) == k:
            break
    return out


NOUL_POLICY_SPEC = dict(
    q="under the stated policy, is the requested action allowed",
    yes="all required conditions are shown to hold and nothing in the policy forbids the action.",
    no="at least one requirement is unmet or unproven, or a prohibition is triggered.",
    caveat="Conditions the state does not establish count as unmet.", flag="permitted")


def gen_policy_permit(n, rng, heldout_style=False, force_long=None):
    """force_long (--long_share, gen_long_extra): None = today's LONG_FRAC-chance single rows,
    interleaved with rubric groups as usual; True = every row is a long single row (used only to
    fill data_wf/train_long.jsonl, so groups -- which never carry a `long` state -- are skipped)."""
    def single():
        dom = rng.choice(list(DOMAINS))
        case = policy_case(dom, rng, want_yes=rng.random() < 0.5)
        long = (rng.random() < LONG_FRAC) if force_long is None else force_long
        state = assemble_state(policy_text(case, rng=rng), case["facts"], case["request"], dom, rng, long)
        return [make_row("noul", "policy_permit", NOUL_POLICY_SPEC, state, ["no", "yes"], int(permitted(verdict(case))),
                         pick_style("noul", rng, heldout_style), extra_meta={"long": long})]

    def group():  # same facts, policy A / B / C in the instructions -> different verdicts
        dom = rng.choice(list(DOMAINS))
        case = policy_case(dom, rng, n_conds=(3, 5), n_proh=(1, 2))
        variants = [(dict(NOUL_POLICY_SPEC, prefix=policy_text(case, ci, pi, rng)), int(permitted(verdict(case, ci, pi))))
                    for ci, pi in policy_subsets(case, rng)]
        if not differing(variants):
            return None
        return emit_group("noul", "policy_permit", f"{case['facts']} {case['request']}", ["no", "yes"], variants, rng, heldout_style)
    return fill_to(n, single, None if force_long else group, rng)


ACTION_VOCABS = [("allow", "deny", "review", "escalate"), ("approve", "reject", "request_info", "refer_to_manager"),
                 ("proceed", "refuse", "hold", "hand_off"), ("green", "red", "amber", "escalate"),
                 ("GO", "NO_GO", "PENDING", "ESCALATE")]
ACTION_DESC = ["every required condition is established and no prohibition or escalation trigger applies",
               "a prohibition applies or a required condition is shown to be unmet",
               "no prohibition applies and nothing is shown unmet, but at least one required condition is unproven",
               "an escalation trigger is present (large amount, VIP, repeated request or legal threat) and no prohibition applies"]


def action_spec(labels, prefix=None):
    return dict(noun="action", task="decide the handling action for the request under the stated policy",
                caveat="Precedence: prohibition > escalation > unmet condition > unproven condition > allow. "
                       "Unstated conditions are unproven.", criteria=dict(zip(labels, ACTION_DESC)), prefix=prefix)


def gen_action_select(n, rng, heldout_style=False, force_long=None):
    """force_long: see gen_policy_permit."""
    def setup():
        dom = rng.choice(list(DOMAINS))
        k = 4 if rng.random() < 0.7 else 3
        esc = k == 4 and rng.random() < 0.35
        return dom, list(rng.choice(ACTION_VOCABS)[:k]), esc

    def single():
        dom, labels, esc = setup()
        case = policy_case(dom, rng, want_yes=rng.random() < 0.4, esc=esc)
        long = (rng.random() < LONG_FRAC) if force_long is None else force_long
        state = assemble_state(policy_text(case, rng=rng), case["facts"], case["request"], dom, rng, long)
        p_null = 1.0 if rng.random() < NULL_FRAC else 0.0
        return [make_row("choice", "action_select", action_spec(labels), state, labels, action_of(verdict(case)),
                         pick_style("choice", rng, heldout_style), p_null, {"long": long})]

    def group():
        dom, labels, esc = setup()
        case = policy_case(dom, rng, esc=esc, n_conds=(3, 5), n_proh=(1, 2))
        variants = [(action_spec(labels, policy_text(case, ci, pi, rng)), action_of(verdict(case, ci, pi)))
                    for ci, pi in policy_subsets(case, rng)]
        if not differing(variants) or any(g >= len(labels) for _, g in variants):
            return None
        return emit_group("choice", "action_select", f"{case['facts']} {case['request']}", labels, variants, rng, heldout_style)
    return fill_to(n, single, None if force_long else group, rng)


LONG_EXTRA_SPLIT = {"policy_permit": CAPS["policy_permit"], "action_select": CAPS["action_select"]}


def gen_long_extra(n, rng):
    """--long_share (PLAN7 track B, long_e45 run): n extra always-long rows for data_wf/train_long.jsonl,
    generated (not carved out of the base families) so train.jsonl is untouched. Split between the two
    long-capable families (policy_permit, action_select) in their normal CAPS ratio.
    ponytail: a 2-entry ratio table, not a generic per-family registry -- these are the only families
    assemble_state's `long` flag applies to."""
    if n <= 0:
        return []
    n_p = round(n * LONG_EXTRA_SPLIT["policy_permit"] / sum(LONG_EXTRA_SPLIT.values()))
    return gen_policy_permit(n_p, rng, force_long=True) + gen_action_select(n - n_p, rng, force_long=True)


# ============================== eligibility (held-out noul) ==============================
ELIG_FIELDS = [  # (name, unit prefix, unit suffix, op, threshold range)
    ("age", "", " years", ">=", (18, 65)), ("tenure", "", " months", ">=", (3, 36)),
    ("account balance", "$", "", "<=", (500, 20000)), ("credit score", "", "", ">=", (550, 750)),
    ("order count", "", "", ">=", (1, 20)), ("parcel weight", "", " kg", "<=", (2, 40)),
    ("delivery distance", "", " km", "<=", (5, 200)), ("team size", "", " people", "<=", (3, 50)),
    ("response time", "", " hours", "<=", (1, 72)), ("uptime", "", "%", ">=", (90, 99)),
]
ELIG_SUBJECTS = ["the applicant", "the account", "the shipment", "the vendor", "the candidate", "the site"]
ELIG_SPEC = dict(q="does the case meet every stated threshold",
                 yes="each threshold is met by a value given in the state ('at least'/'at most' include equality).",
                 no="some value misses its threshold, or a required value is not given at all.",
                 caveat="A value that is not stated cannot satisfy a threshold.", flag="eligible")


def elig_rule_text(rules, subj, rng):
    return rng.choice([f"Eligibility requires: {'; '.join(rules)}.", f"To qualify, {subj} must have {', and '.join(rules)}.",
                       "Criteria: " + "; ".join(rules) + "."])


def elig_rule(f, thr):
    name, pre, suf, op, _ = f
    return f"{name} of {'at least' if op == '>=' else 'at most'} {pre}{thr}{suf}"


def elig_ok(f, val, thr):
    return val is not None and (val >= thr if f[3] == ">=" else val <= thr)


def gen_eligibility(n, rng, heldout_style=False):
    def case():
        subj = rng.choice(ELIG_SUBJECTS)
        fields = rng.sample(ELIG_FIELDS, rng.randint(2, 3))
        vals = [None if rng.random() < 0.15 else rng.randint(lo, hi) for (_, _, _, _, (lo, hi)) in fields]
        facts = [f"{cap(subj)}'s {f[0]} is {f[1]}{v}{f[2]}." for f, v in zip(fields, vals) if v is not None]
        date = None
        if rng.random() < 0.4:
            d0 = dt.date(2024, 1, 1) + dt.timedelta(days=rng.randint(0, 600))
            gap = rng.randint(0, 120)
            date = (d0, d0 + dt.timedelta(days=gap), gap)
            facts.append(f"The event took place on {d0.isoformat()} and the claim was filed on {date[1].isoformat()}.")
        rng.shuffle(facts)
        return subj, fields, vals, date, " ".join(facts)

    def rules_for(fields, vals, date, want):
        """Thresholds around the values so the case passes (want) or fails; None-valued fields always fail when asked."""
        rules, ok = [], True
        fail_at = None if want else rng.randrange(len(fields) + (1 if date else 0))
        for j, (f, v) in enumerate(zip(fields, vals)):
            lo, hi = f[4]
            base = v if v is not None else rng.randint(lo, hi)
            m = max(1, base // 5)
            if fail_at == j or v is None:
                thr = base + rng.randint(1, m) if f[3] == ">=" else max(0, base - rng.randint(1, m))
            else:
                thr = max(0, base - rng.randint(0, m)) if f[3] == ">=" else base + rng.randint(0, m)
            rules.append(elig_rule(f, thr)); ok &= elig_ok(f, v, thr)
        if date:
            win = date[2] - rng.randint(1, 30) if fail_at == len(fields) else date[2] + rng.randint(0, 30)
            win = max(1, win)
            rules.append(f"the claim filed within {win} days of the event"); ok &= date[2] <= win
        return rules, ok

    def single():
        subj, fields, vals, date, facts = case()
        rules, ok = rules_for(fields, vals, date, rng.random() < 0.5)
        rt = elig_rule_text(rules, subj, rng)
        state = f"{rt}\n{facts}" if rng.random() < 0.7 else f"{facts}\n{rt}"
        return [make_row("noul", "eligibility", ELIG_SPEC, state, ["no", "yes"], int(ok), pick_style("noul", rng, heldout_style))]

    def group():  # same facts, thresholds in the rubric differ
        subj, fields, vals, date, facts = case()
        variants = []
        for want in rng.sample([True, False, rng.random() < 0.5], rng.randint(2, 3)):
            rules, ok = rules_for(fields, vals, date, want)
            variants.append((dict(ELIG_SPEC, prefix=elig_rule_text(rules, subj, rng)), int(ok)))
        return emit_group("noul", "eligibility", facts, ["no", "yes"], variants, rng, heldout_style) if differing(variants) else None
    return fill_to(n, single, group, rng)


# ============================== tool_select (held-out choice) ==============================
TOOLS = [  # (name, description, request templates)
    ("get_weather", "Fetches the current or forecast weather for a location", ["Will it rain in {city} tomorrow?", "What's the temperature in {city} right now?"]),
    ("send_email", "Sends an email to a given address", ["Email {name} the quarterly numbers.", "Send a note to {name} saying I'm running late."]),
    ("create_calendar_event", "Adds an event to the user's calendar", ["Put lunch with {name} on my calendar for Friday.", "Schedule a dentist visit next Tuesday at 3."]),
    ("set_timer", "Starts a countdown timer", ["Set a timer for {n} minutes.", "Start a {n}-minute countdown."]),
    ("play_music", "Plays a song, album or playlist", ["Play some jazz.", "Put on the album by {name}."]),
    ("web_search", "Searches the web for a query", ["Look up who won the {n}th Super Bowl.", "Search for reviews of the new {city} museum."]),
    ("translate_text", "Translates text between languages", ["How do you say 'thank you' in Japanese?", "Translate this paragraph into French."]),
    ("convert_units", "Converts between measurement units", ["How many miles is {n} kilometres?", "Convert {n} ounces to grams."]),
    ("get_directions", "Returns a route between two places", ["How do I get from {city} to the airport?", "Directions to the nearest pharmacy."]),
    ("order_food", "Places a food delivery order", ["Order a large pepperoni pizza to my place.", "Get me pad thai from the usual spot."]),
    ("check_stock_price", "Returns the latest price for a ticker", ["What is {name} Corp trading at?", "How did the market close today for ACME?"]),
    ("create_reminder", "Creates a reminder at a time or place", ["Remind me to call {name} at 5.", "Remind me to buy milk when I'm at the store."]),
    ("book_ride", "Requests a car ride", ["Get me a cab to {city} station.", "Book a ride home in 20 minutes."]),
    ("read_news", "Reads headlines on a topic", ["What's the latest on the {city} elections?", "Any news about {name} today?"]),
    ("calculate", "Evaluates an arithmetic expression", ["What's {n} times 37?", "Compute 15% of {n}."]),
    ("define_word", "Gives the definition of a word", ["What does 'ephemeral' mean?", "Define the word 'sonder'."]),
    ("control_lights", "Turns smart lights on/off or dims them", ["Dim the living room lights to 40%.", "Turn off the bedroom lights."]),
    ("track_package", "Reports the delivery status of a parcel", ["Where is my order #{n}?", "Has the package from {name} shipped yet?"]),
    ("add_to_shopping_list", "Adds an item to the shopping list", ["Add eggs to the shopping list.", "Put {n} lemons on my grocery list."]),
    ("find_recipe", "Finds a recipe by dish or ingredients", ["How do I make shakshuka?", "Find a recipe using chickpeas and spinach."]),
]
CHITCHAT = ["Thanks, that's all for now.", "You're pretty funny for a bot.", "Good morning!", "Never mind, I figured it out.",
            "How's your day going?", "Tell me a joke about {city}.", "I'm bored."]
FILL = dict(city=["Lisbon", "Denver", "Osaka", "Nairobi", "Leeds", "Austin"], name=["Sam", "Dr. Okafor", "Marta", "the Patels", "Lee"],
            n=[3, 12, 25, 40, 7, 90])
TOOL_NAMER = [lambda s: s, lambda s: "".join(w.capitalize() for w in s.split("_")),
              lambda s: "tools." + s, lambda s: s.replace("_", "-"), lambda s: s.replace("_", " ")]
OPAQUE = [["team_a", "team_b", "team_c", "team_d", "team_e", "team_f", "team_g", "team_h"],
          ["queue_1", "queue_2", "queue_3", "queue_4", "queue_5", "queue_6", "queue_7", "queue_8"],
          ["desk-north", "desk-south", "desk-east", "desk-west", "desk-central", "desk-remote", "desk-night", "desk-day"],
          ["option_1", "option_2", "option_3", "option_4", "option_5", "option_6", "option_7", "option_8"],
          ["slot A", "slot B", "slot C", "slot D", "slot E", "slot F", "slot G", "slot H"]]


def fill(t, rng):
    return t.format(**{k: rng.choice(v) for k, v in FILL.items()})


def permuted_groups(qtype, family, state, labels, descs, gold_desc, spec_fn, rng, heldout_style, k=None):
    """Opaque labels + 2-3 distinct permutations of the descriptions -> gold index moves with the criteria."""
    k = k or rng.randint(2, 3)
    perms, variants = set(), []
    for _ in range(20):
        p = list(range(len(descs))); rng.shuffle(p)
        if tuple(p) in perms:
            continue
        perms.add(tuple(p))
        crit = {lb: descs[j] for lb, j in zip(labels, p)}
        variants.append((spec_fn(crit), p.index(gold_desc)))
        if len(variants) == k:
            break
    return emit_group(qtype, family, state, labels, variants, rng, heldout_style) if differing(variants) else None


def tool_spec(crit):
    return dict(noun="tool", task="pick the tool the assistant should call for the user's message",
                caveat="Pick the tool whose description covers the request; small talk needs no tool.", criteria=crit)


def gen_tool_select(n, rng, heldout_style=False):
    def single():
        tools = rng.sample(TOOLS, rng.randint(3, 6))
        namer = rng.choice(TOOL_NAMER)
        with_none = rng.random() < 0.4
        chit = with_none and rng.random() < 0.35
        gold_tool = rng.choice(tools)
        req = fill(rng.choice(CHITCHAT if chit else gold_tool[2]), rng)
        labels = [namer(t[0]) for t in tools]
        crit = {namer(t[0]): t[1] for t in tools}
        if with_none:
            labels.append("none"); crit["none"] = "No tool call is needed for this message"
        gold = len(labels) - 1 if chit else labels.index(namer(gold_tool[0]))
        state = rng.choice([f"User: {req}", f"Message: {req}", req])
        p_null = 1.0 if (not chit and rng.random() < NULL_FRAC) else 0.0
        return [make_row("choice", "tool_select", tool_spec(crit), state, labels, gold, pick_style("choice", rng, heldout_style), p_null)]

    def group():
        tools = rng.sample(TOOLS, rng.randint(3, 6))
        gi = rng.randrange(len(tools))
        labels = list(rng.choice(OPAQUE)[:len(tools)])
        state = f"User: {fill(rng.choice(tools[gi][2]), rng)}"
        return permuted_groups("choice", "tool_select", state, labels, [t[1] for t in tools], gi, tool_spec, rng, heldout_style)
    return fill_to(n, single, group, rng)


# ============================== enum_extract (choice) ==============================
ENUM_FIELDS = {
    "priority": (["P0", "P1", "P2", "P3"], ["critical", "high", "medium", "low"]),
    "status": (["open", "in progress", "resolved", "closed", "on hold"],),
    "plan": (["free", "starter", "pro", "enterprise"],),
    "payment method": (["credit card", "bank transfer", "PayPal", "invoice"],),
    "region": (["EU", "US", "APAC", "LATAM"],),
    "shipping option": (["standard", "express", "overnight", "pickup"],),
    "channel": (["email", "chat", "phone", "web form"],),
}
FIELD_SENT = {
    "priority": ["Priority: {v}.", "The ticket is marked {v} priority.", "Severity/priority set to {v}."],
    "status": ["Status: {v}.", "The case is currently {v}.", "Current state: {v}."],
    "plan": ["Plan: {v}.", "The customer is on the {v} plan.", "Subscription tier {v}."],
    "payment method": ["Paid via {v}.", "Payment method: {v}.", "The order was paid by {v}."],
    "region": ["Region: {v}.", "The account is in the {v} region.", "Billing region {v}."],
    "shipping option": ["Shipping: {v}.", "The customer chose {v} shipping.", "Delivery method: {v}."],
    "channel": ["Channel: {v}.", "Received via {v}.", "The request came in by {v}."],
}
CHANGE_SENT = ["{F} was {old} when opened and was changed to {new} after review.",
               "Originally {old}; updated to {new} this morning.", "{F} moved from {old} to {new}."]


def enum_spec(target, crit, original=False):
    if original:
        return dict(noun=target, task=f"extract the {target} the record had when it was first opened",
                    caveat=f"Report the original {target}; when it was changed later, the earlier value counts.", criteria=crit)
    return dict(noun=target, task=f"extract the {target} from the record",
                caveat=f"Report the current {target}; when it was changed, the latest value counts.", criteria=crit)


def gen_enum_extract(n, rng, heldout_style=False):
    fields = list(ENUM_FIELDS)

    def record(absent, changed):
        target = rng.choice(fields)
        vocab = list(rng.choice(ENUM_FIELDS[target]))
        others = rng.sample([f for f in fields if f != target], rng.randint(2, 4))
        sents = [rng.choice(FIELD_SENT[f]).format(v=rng.choice(ENUM_FIELDS[f][0])) for f in others]
        new = rng.choice(vocab)
        old = rng.choice([v for v in vocab if v != new])
        if changed:
            sents.append(rng.choice(CHANGE_SENT).format(F=cap(target), old=old, new=new))
        elif not absent:
            sents.append(rng.choice(FIELD_SENT[target]).format(v=new))
        rng.shuffle(sents)
        head = rng.choice([f"Ticket #{rng.randint(1000, 99999)}.", f"Order {rng.randint(100000, 999999)}.",
                           "Case notes:", f"Customer {rng.choice(FILL['name'])}."])
        return target, vocab, f"{head} {' '.join(sents)}", new, old

    def single():
        absent = rng.random() < 0.2
        changed = not absent and rng.random() < 0.3
        target, vocab, state, new, old = record(absent, changed)
        labels, crit = list(vocab), {v: f"the {target} is {v}" for v in vocab}
        if absent or rng.random() < 0.3:
            ns = rng.choice(["not_stated", "unknown", "absent"])
            labels.append(ns); crit[ns] = f"the state does not give a {target}"
        gold = len(labels) - 1 if absent else labels.index(new)
        p_null = 1.0 if (not absent and rng.random() < NULL_FRAC) else 0.0
        return [make_row("choice", "enum_extract", enum_spec(target, crit), state, labels, gold,
                         pick_style("choice", rng, heldout_style), p_null)]

    def group():  # changed record: "current value" vs "value at opening"
        target, vocab, state, new, old = record(False, True)
        crit = {v: f"the {target} is {v}" for v in vocab}
        variants = [(enum_spec(target, crit), vocab.index(new)), (enum_spec(target, crit, original=True), vocab.index(old))]
        return emit_group("choice", "enum_extract", state, vocab, variants, rng, heldout_style)
    return fill_to(n, single, group, rng)


# ============================== severity / urgency / completeness (score) ==============================
def level_names(k, rng, numeric_p=0.5):
    """-> (names, numeric). numeric labels "0".."k-1" or a named ordered vocabulary."""
    if rng.random() < numeric_p:
        return [str(i) for i in range(k)], True
    pools = {3: [["low", "medium", "high"], ["minor", "moderate", "major"], ["L1", "L2", "L3"], ["green", "amber", "red"]],
             4: [["minor", "moderate", "major", "critical"], ["S4", "S3", "S2", "S1"], ["low", "medium", "high", "urgent"],
                 ["negligible", "limited", "serious", "severe"]],
             5: [["very poor", "poor", "fair", "good", "excellent"], ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
                 ["terrible", "bad", "okay", "good", "great"]]}
    return list(rng.choice(pools[k])), False


def score_family(family, what, rules, state_fn, n, rng, heldout_style, k=4):
    """rules: {name: (levels_desc, fn(facts)->level)}. Groups = same facts under 2-3 rule sets with differing levels."""
    def spec(rule, names, numeric):
        return dict(what=what, names=names, levels=rules[rule][0], numeric=numeric, caveat=rules[rule][2])

    def single():
        facts, state = state_fn()
        rule = rng.choice(list(rules))
        names, numeric = level_names(k, rng)
        return [make_row("score", family, spec(rule, names, numeric), state, names, rules[rule][1](facts),
                         pick_style("score", rng, heldout_style), extra_meta={"rule": rule})]

    def group():
        facts, state = state_fn()
        names, numeric = level_names(k, rng)
        variants = [(spec(r, names, numeric), rules[r][1](facts)) for r in rng.sample(list(rules), min(len(rules), rng.randint(2, 3)))]
        return emit_group("score", family, state, names, variants, rng, heldout_style) if differing(variants) else None
    return fill_to(n, single, group if len(rules) > 1 else None, rng)


SEV_RULES = {
    "impact": (["No functional impact; cosmetic or informational only",
                "A non-core function or a single user is impaired and a workaround exists",
                "A core function is impaired for a team or many users, or data was lost but is recoverable",
                "Irreversible data loss, or a core function is down for all users with no workaround"],
               lambda f: max(f["imp"], {"none": 0, "recoverable": 2, "irreversible": 3}[f["data"]]),
               "Use only facts stated in the report."),
    "factors": (["None of the aggravating factors is present",
                 "Exactly one aggravating factor (core function, many users, any data loss, no workaround)",
                 "Exactly two aggravating factors", "Three or more aggravating factors"],
                lambda f: min(3, (f["func"] == "core") + (f["affected"] in ("a team of eight", "all users")) + (f["data"] != "none") + (not f["wa"])),
                "Count the factors: core function affected, many users affected, any data loss, no workaround."),
    "data_first": (["No data loss and no core function affected", "No data loss but a core function is affected",
                    "Recoverable data loss, regardless of anything else", "Irreversible data loss, regardless of anything else"],
                   lambda f: 3 if f["data"] == "irreversible" else 2 if f["data"] == "recoverable" else int(f["func"] == "core"),
                   "Data loss outranks every other factor."),
}


def severity_state(rng):
    affected = rng.choice(["nobody", "one user", "a team of eight", "all users"])
    func = rng.choice(["cosmetic", "secondary", "core"])
    data = rng.choice(["none", "none", "recoverable", "irreversible"])
    wa = rng.random() < 0.5
    imp = 0 if func == "cosmetic" or affected == "nobody" else 1 if func == "secondary" or affected == "one user" \
        else 2 if affected == "a team of eight" else (2 if wa else 3)
    parts = [rng.choice([f"Affected: {affected}.", f"Reported by {affected}.", f"Scope: {affected} affected."]),
             {"cosmetic": rng.choice(["A label is misaligned on the settings page.", "The logo renders in the wrong colour."]),
              "secondary": rng.choice(["The export-to-PDF feature errors out.", "Notifications arrive late."]),
              "core": rng.choice(["Checkout fails at payment.", "Users cannot log in.", "The main dashboard does not load."])}[func],
             {"none": "", "recoverable": "Some records were deleted but are restorable from last night's backup.",
              "irreversible": "Records were overwritten with no backup available."}[data],
             "A workaround is available." if wa else "There is no workaround."]
    parts = [p for p in parts if p]
    rng.shuffle(parts)
    return dict(affected=affected, func=func, data=data, wa=wa, imp=imp), f"Incident report: {' '.join(parts)}"


def gen_severity(n, rng, heldout_style=False):
    return score_family("severity", "incident severity", SEV_RULES, lambda: severity_state(rng), n, rng, heldout_style)


URG_RULES = {
    "combined": (["No deadline within two weeks and nothing is blocked", "A deadline within two weeks, nothing blocked",
                  "Work is blocked, or a deadline within three days",
                  "Work is blocked with a deadline inside 24 hours, or production is down"],
                 lambda f: 3 if f["prod"] or (f["blocked"] and f["h"] is not None and f["h"] <= 24)
                 else 2 if f["blocked"] or (f["h"] is not None and f["h"] <= 72) else 1 if f["h"] is not None and f["h"] <= 336 else 0,
                 "Judge from the stated deadline and blocking facts only."),
    "deadline_only": (["No deadline, or more than two weeks away", "Deadline within two weeks", "Deadline within three days",
                       "Deadline within 24 hours"],
                      lambda f: 0 if f["h"] is None or f["h"] > 336 else 1 if f["h"] > 72 else 2 if f["h"] > 24 else 3,
                      "Only the deadline matters; ignore blocking and outages."),
    "blocking_first": (["Nothing blocked and no deadline within three days", "Nothing blocked but a deadline within three days",
                        "Work is blocked (production up)", "Production is down"],
                       lambda f: 3 if f["prod"] else 2 if f["blocked"] else int(f["h"] is not None and f["h"] <= 72),
                       "Outages outrank blocking, which outranks deadlines."),
}


def urgency_state(rng):
    h = rng.choice([None, None, 6, 12, 20, 30, 48, 70, 120, 200, 300, 400, 800, 1000])
    blocked, prod = rng.random() < 0.4, rng.random() < 0.12
    dl = ("No deadline was mentioned." if h is None else
          rng.choice([f"The deadline is in {h} hours.", f"Due in {h // 24} days and {h % 24} hours.", f"Must be done within {h} hours."]))
    parts = [dl, rng.choice(["The requester cannot continue until this is fixed." if blocked else "The requester has other work in the meantime.",
                             "This blocks the whole team." if blocked else "Nothing is blocked."]),
             "Production is currently down." if prod else rng.choice(["", "Production is unaffected."]),
             rng.choice(["Requester: finance team.", "Requester: a customer on the pro plan.", "Requester: an intern.", ""])]
    parts = [p for p in parts if p]
    rng.shuffle(parts)
    return dict(h=h, blocked=blocked, prod=prod), f"Ticket: {' '.join(parts)}"


def gen_urgency(n, rng, heldout_style=False):
    return score_family("urgency", "ticket urgency", URG_RULES, lambda: urgency_state(rng), n, rng, heldout_style)


COMP_FIELDS = ["a cover letter", "two references", "a signed consent form", "proof of address", "a photo ID",
               "the completed questionnaire", "a budget table", "the project timeline", "a risk statement", "an abstract"]
COMP_RULES = {
    "tbd_missing": (["Three or more required items are missing or marked pending", "Exactly two required items are missing or pending",
                     "Exactly one required item is missing or pending", "Every required item is present and final"],
                    lambda f: 3 - min(3, f["missing"] + f["pending"]), "An item marked TBD or pending counts as missing."),
    "tbd_present": (["Three or more required items are absent", "Exactly two required items are absent",
                     "Exactly one required item is absent", "Every required item is at least listed (TBD counts as listed)"],
                    lambda f: 3 - min(3, f["missing"]), "An item marked TBD or pending counts as present."),
}


def completeness_state(rng):
    req = rng.sample(COMP_FIELDS, rng.randint(3, 6))
    gone = rng.sample(req, rng.randint(0, min(4, len(req))))
    pend = [f for f in gone if rng.random() < 0.5]
    sub = [f"{cap(f)}: attached." for f in req if f not in gone] + [f"{cap(f)}: marked TBD." for f in pend]
    rng.shuffle(sub)
    state = f"Required: {', '.join(req)}.\nSubmission: {' '.join(sub) if sub else 'nothing attached yet.'}"
    return dict(missing=len(gone) - len(pend), pending=len(pend)), state


def gen_completeness(n, rng, heldout_style=False):
    return score_family("completeness", "submission completeness", COMP_RULES, lambda: completeness_state(rng), n, rng, heldout_style)


# ============================== dataset-backed families ==============================
def _load(hf_id, split, **kw):
    from datasets import load_dataset
    return load_dataset(hf_id, split=split, **kw)


def take(rows, n, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


ADEQ_SPEC = dict(q="does the given answer actually address the question asked",
                 yes="the answer supplies the specific information the question asks for.",
                 no="the answer is about something else, evades the question, or gives no usable information.",
                 caveat="Judge relevance to the question, not writing quality.", flag="answered")
EVASIONS = ["I'm not sure, please check the documentation.", "That depends on many factors.", "Great question!",
            "Let me get back to you on that.", "It is mentioned somewhere in the text."]


def gen_adequacy(squad, n, rng, heldout_style=False):
    ans = [r for r in squad if r["answers"]["text"]]
    by_title = defaultdict(list)
    for r in ans:
        by_title[r["title"]].append(r["answers"]["text"][0])
    rows = []
    for r in take(ans, n, rng):
        gold = r["answers"]["text"][0]
        yes = rng.random() < 0.5
        if yes:
            a = rng.choice([gold, f"The answer is {gold}.", f"{gold}.", f"It was {gold}."])
        else:
            mode = rng.random()
            pool = [x for x in by_title[r["title"]] if x != gold]
            if mode < 0.55 and pool:
                a = rng.choice(pool)  # answer to a different question on the same topic
            elif mode < 0.8:
                a = rng.choice(EVASIONS)
            else:
                a = rng.choice(rng.choice(list(by_title.values())))
                if a == gold:
                    continue
        state = rng.choice([f"Question: {r['question']}\nAnswer: {a}", f"Q: {r['question']}\nA: {a}",
                            f"Asked: {r['question']}\nReply: {a}"])
        if rng.random() < 0.3:
            state = f"Context: {r['context']}\n{state}"
        rows.append(make_row("noul", "adequacy", ADEQ_SPEC, state, ["no", "yes"], int(yes), pick_style("noul", rng, heldout_style)))
    return rows


SUPPORT_SPEC = dict(q="is the claim supported by the passage",
                    yes="the passage, read literally, makes the claim true.",
                    no="the passage contradicts the claim or does not settle it.",
                    caveat="A claim that merely could be true is not supported.", flag="supported")


def gen_support(nli, n, rng, heldout_style=False):
    rows = []
    ent, non = [r for r in nli if r["label"] == 0], [r for r in nli if r["label"] != 0]
    for r in take(ent, n // 2, rng) + take(non, n - n // 2, rng):  # balanced yes/no
        state = rng.choice([f"Passage: {r['premise']}\nClaim: {r['hypothesis']}", f"Text: {r['premise']}\nStatement: {r['hypothesis']}",
                            f"{r['premise']}\nClaim under review: {r['hypothesis']}"])
        rows.append(make_row("noul", "support", SUPPORT_SPEC, state, ["no", "yes"], int(r["label"] == 0),
                             pick_style("noul", rng, heldout_style)))
    return rows


FACT_SPEC = dict(q="according to the passage, is the answer to the question yes",
                 yes="the passage establishes that the question's answer is yes.",
                 no="the passage establishes the answer is no, or does not answer it.",
                 caveat="Do not use outside knowledge.", flag="affirmed")


def gen_fact(boolq, n, rng, heldout_style=False):
    rows = []
    for r in take(boolq, n, rng):
        state = rng.choice([f"Passage: {r['passage']}\nQuestion: {r['question']}?", f"{r['passage']}\nQ: {r['question']}?"])
        rows.append(make_row("noul", "fact", FACT_SPEC, state, ["no", "yes"], int(bool(r["answer"])),
                             pick_style("noul", rng, heldout_style)))
    return rows


INTENT_SOURCES = [
    ("hwu64", lambda: _load("FastFit/hwu_64", "train"), "text", "label"),
    ("snips", lambda: _load("benayas/snips", "train"), "text", "category"),
    ("massive", lambda: _load("json", "train", data_files="hf://datasets/mteb/amazon_massive_intent/train/en.json.gz"), "text", "label"),
    ("mtop", lambda: _load("parquet", "train", data_files="hf://datasets/mteb/mtop_intent/en/train-00000-of-00001.parquet"), "text", "label_text"),
    ("bitext", lambda: _load("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "train"), "instruction", "intent"),
]
DESC_T = ["Requests concerning {w}", "Messages whose main goal is: {w}", "Handles anything about {w}", "{W}"]


def routing_spec(crit):
    return dict(noun="team", task="route the message to the team that should own it",
                caveat="A topic mentioned in passing is not a request.", criteria=crit)


def gen_routing(n, rng, heldout_style=False):
    per = max(1, n // len(INTENT_SOURCES))
    rows = []
    for name, loader, tcol, lcol in INTENT_SOURCES:
        try:
            ds = loader()
        except Exception as e:
            print(f"WARNING: routing source {name} skipped: {e}")
            continue
        uniq = sorted({r[lcol] for r in ds})
        items = iter(take(ds, per * 2, rng))

        def draw():
            r = next(items)
            k = rng.randint(2, min(8, len(uniq)))
            cands = [r[lcol]] + rng.sample([u for u in uniq if u != r[lcol]], k - 1)
            rng.shuffle(cands)
            return r, cands, cands.index(r[lcol])

        def single():
            r, cands, gold = draw()
            style = rng.random()
            if style < 0.2:
                names = list(rng.choice(OPAQUE)[:len(cands)]); rng.shuffle(names)
            elif style < 0.55:
                names = [words(c) for c in cands]
            elif style < 0.8:
                names = list(cands)
            else:
                names = [words(c).replace(" ", "_") for c in cands]
            dt_ = rng.choice(DESC_T)
            crit = {nm: dt_.format(w=words(c), W=cap(words(c))) for nm, c in zip(names, cands)}
            labels = list(names)
            other = rng.random() < 0.3
            drop_gold = other and rng.random() < 0.35
            if other:
                o = rng.choice(["other", "none_of_these", "unrouted"])
                labels.append(o); crit[o] = "None of the listed teams is the right owner"
            if drop_gold:  # gold team absent but an "other" option exists -> route to other
                labels.pop(gold); crit.pop(names[gold]); gold = len(labels) - 1
            p_null = 1.0 if (not other and rng.random() < NULL_FRAC) else 0.0
            return [make_row("choice", "routing", routing_spec(crit), r[tcol], labels, gold, pick_style("choice", rng, heldout_style),
                             p_null, {"source": name})]

        def group():  # opaque team names; the criteria assignment is permuted across variants
            r, cands, gold = draw()
            dt_ = rng.choice(DESC_T)
            return permuted_groups("choice", "routing", r[tcol], list(rng.choice(OPAQUE)[:len(cands)]),
                                   [dt_.format(w=words(c), W=cap(words(c))) for c in cands], gold, routing_spec, rng, heldout_style)
        try:
            rows += [dict(x, meta=dict(x["meta"], source=name)) for x in fill_to(per, single, group, rng)]
        except StopIteration:
            pass
    return rows


CAT_SOURCES = {  # (loader, text col, label col, classlabel?, per-label criteria)
    "ag_news": (lambda: _load("fancyzhx/ag_news", "train"), "text", "label", True,
                {"World": "International news, politics or conflict outside business/sport/science",
                 "Sports": "Athletes, matches, leagues or sporting results", "Business": "Companies, markets, earnings or the economy",
                 "Sci/Tech": "Science, technology, software or research"}),
    "bbc_news": (lambda: _load("SetFit/bbc-news", "train"), "text", "label_text", False,
                 {"business": "Corporate, market or economic reporting", "entertainment": "Film, music, television or celebrities",
                  "politics": "Government, parties, elections or policy", "sport": "Sporting events, teams or athletes",
                  "tech": "Technology products, the internet or gadgets"}),
    "emotion": (lambda: _load("dair-ai/emotion", "train"), "text", "label", True,
                {"sadness": "The writer expresses grief, disappointment or loss", "joy": "The writer expresses happiness or delight",
                 "love": "The writer expresses affection or tenderness", "anger": "The writer expresses irritation or rage",
                 "fear": "The writer expresses anxiety or dread", "surprise": "The writer expresses astonishment"}),
    "tweet_sentiment": (lambda: _load("cardiffnlp/tweet_eval", "train", name="sentiment"), "text", "label", True,
                        {"negative": "Overall attitude is unfavourable", "neutral": "No clear attitude either way",
                         "positive": "Overall attitude is favourable"}),
    "dbpedia": (lambda: _load("fancyzhx/dbpedia_14", "train"), "content", "label", True,
                {"Company": "A commercial organisation", "EducationalInstitution": "A school, college or university",
                 "Artist": "A musician, painter, writer or other creative person", "Athlete": "A sportsperson",
                 "OfficeHolder": "A politician or public official", "MeanOfTransportation": "A vehicle, ship, aircraft or vehicle model",
                 "Building": "A building or structure", "NaturalPlace": "A river, mountain, lake or other natural feature",
                 "Village": "A village or small settlement", "Animal": "An animal species", "Plant": "A plant species",
                 "Album": "A music album", "Film": "A movie", "WrittenWork": "A book, novel, journal or other written work"}),
}
CAT_NOUN = {"ag_news": "news section", "bbc_news": "news desk", "emotion": "emotion", "tweet_sentiment": "sentiment", "dbpedia": "entity type"}


def gen_categorical(n, rng, heldout_style=False):
    per = max(1, n // len(CAT_SOURCES))
    rows = []
    for name, (loader, tcol, lcol, classlabel, desc) in CAT_SOURCES.items():
        try:
            ds = loader()
        except Exception as e:
            print(f"WARNING: categorical source {name} skipped: {e}")
            continue
        names = ds.features[lcol].names if classlabel else sorted(desc)
        items = iter(take(ds, per * 2, rng))
        noun = CAT_NOUN[name]

        def spec_fn(crit):
            return dict(noun=noun, task=f"assign the {noun} that best fits the text",
                        caveat="Use the option descriptions as the definitions.", criteria=crit)

        def draw():
            r = next(items)
            gold_name = names[r[lcol]] if classlabel else r[lcol]
            cands = [gold_name] + rng.sample([u for u in names if u != gold_name], rng.randint(2, min(8, len(names))) - 1)
            rng.shuffle(cands)
            return r[tcol][:1200], cands, cands.index(gold_name)

        def single():
            text, cands, gold = draw()
            labels = [words(c) if rng.random() < 0.5 else c for c in cands]
            p_null = 1.0 if rng.random() < NULL_FRAC else 0.0
            return [make_row("choice", "categorical", spec_fn({lb: desc[c] for lb, c in zip(labels, cands)}), text, labels, gold,
                             pick_style("choice", rng, heldout_style), p_null)]

        def group():
            text, cands, gold = draw()
            return permuted_groups("choice", "categorical", text, list(rng.choice(OPAQUE)[:len(cands)]),
                                   [desc[c] for c in cands], gold, spec_fn, rng, heldout_style)
        try:
            rows += [dict(x, meta=dict(x["meta"], source=name)) for x in fill_to(per, single, group, rng)]
        except StopIteration:
            pass
    return rows


QUAL5 = ["Strongly negative; the reviewer would not return", "Mostly negative with minor positives",
         "Mixed or lukewarm; no strong lean", "Mostly positive with minor complaints", "Strongly positive; enthusiastic"]
QUAL3 = ["Negative overall", "Mixed or neutral", "Positive overall"]


def gen_quality(n, rng, heldout_style=False):
    rows = []
    srcs = [("yelp", _load("SetFit/yelp_review_full", "train"), lambda r: int(r["label_text"][0]) - 1),
            ("sst5", _load("SetFit/sst5", "train"), lambda r: ["very negative", "negative", "neutral", "positive", "very positive"].index(r["label_text"]))]
    for name, ds, gl in srcs:
        for r in take(ds, n // 2, rng):
            g5 = gl(r)
            if rng.random() < 0.3:
                k, lvl, levels = 3, {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}[g5], QUAL3
            else:
                k, lvl, levels = 5, g5, QUAL5
            names, numeric = level_names(k, rng)
            spec = dict(what="reviewer satisfaction", names=names, levels=levels, numeric=numeric,
                        caveat="Rate the reviewer's stance, not the product itself.")
            rows.append(make_row("score", "quality", spec, r["text"][:1200], names, lvl, pick_style("score", rng, heldout_style),
                                 extra_meta={"source": name}))
    return rows


REL_LEVELS = ["The document is about an unrelated subject", "The document is on the query's topic but does not contain the answer",
              "The document directly answers the query"]


def gen_relevance(squad, n, rng, heldout_style=False):
    ans = [r for r in squad if r["answers"]["text"]]
    by_title = defaultdict(list)
    for r in ans:
        by_title[r["title"]].append(r)
    titles = list(by_title)
    rows = []
    for r in take(ans, n, rng):
        lvl = rng.randint(0, 2)
        gold = r["answers"]["text"][0]
        if lvl == 2:
            ctx = r["context"]
        elif lvl == 1:
            pool = [x["context"] for x in by_title[r["title"]] if x["context"] != r["context"] and gold.lower() not in x["context"].lower()]
            if not pool:
                continue
            ctx = rng.choice(pool)
        else:
            ctx = rng.choice(by_title[rng.choice([x for x in titles if x != r["title"]])])["context"]
        state = rng.choice([f"Query: {r['question']}\nDocument: {ctx}", f"Search: {r['question']}\nResult: {ctx}"])
        names, numeric = level_names(3, rng)
        spec = dict(what="search-result relevance", names=names, levels=REL_LEVELS, numeric=numeric,
                    caveat="Being on-topic is not the same as answering.")
        rows.append(make_row("score", "relevance", spec, state, names, lvl, pick_style("score", rng, heldout_style)))
    return rows


# ============================== probes over the held-out families ==============================
def flip_pairs(rows):
    """(x, r1, A, y1) / (x, r2, A, y2), y1 != y2, from rubric groups: the first two differing variants per group."""
    by = defaultdict(list)
    for r in rows:
        if r["meta"].get("rubric_group"):
            by[r["meta"]["rubric_group"]].append(r)
    out = []
    for gid, g in by.items():
        for a, b in itertools.combinations(g, 2):
            if a["label"] != b["label"]:
                out += [dict(a, task=a["task"] + "_flip", meta=dict(a["meta"], flip_pair=gid)),
                        dict(b, task=b["task"] + "_flip", meta=dict(b["meta"], flip_pair=gid))]
                break
    return out


def pair_shuffle(rows, rng):
    """Shuffle wf_rubric_flip by PAIR, not by row: rows come in consecutive (a, b) units sharing
    meta.flip_pair (see flip_pairs above); a plain rng.shuffle(rows) would split a pair across the
    file, and skipping the shuffle entirely (the old behaviour) left every pair in family-block
    order -- rubric_group ids are assigned per family in a single pass, so the file's head was 250
    eligibility pairs, 1 of 3 held-out families (REPORT.md §3w correction). Shuffling whole pairs
    keeps each pair adjacent while interleaving families across the file, so any head-N slice
    (e.g. eval_wf.py's old --limit) is representative. A pair with only one surviving row (its
    partner deduped away against train/val) is kept as its own singleton unit."""
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["meta"]["flip_pair"]].append(r)
    units = list(by_pair.values())
    rng.shuffle(units)
    return [r for u in units for r in u]


def shuffled_rubric(rows, rng):
    """Each row gets another row's query from the same family with the identical candidate list (a coherent but
    wrong rubric); label kept. Δ_r = acc(normal) - acc(shuffled), the analogue of scripts/mmlu_probes.py's Δ_q."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["meta"]["family"], tuple(r["candidates"]))].append(i)
    out = []
    for idx in by.values():
        if len(idx) < 2:
            continue
        perm = idx[1:] + idx[:1]  # rotation: never the row's own query
        for i, j in zip(idx, perm):
            if rows[i]["query"] != rows[j]["query"]:
                r = rows[i]
                out.append(dict(r, query=rows[j]["query"], task=r["task"] + "_shufr", meta=dict(r["meta"], probe="shuffled_rubric")))
    return out


def leak_check(rows, jev_dir):
    """Exact + normalised-stem match of our states against the public JevBench states must be 0."""
    p = Path(jev_dir)
    files = list(p.glob("*.jsonl")) if p.exists() else []
    if not files:
        print(f"WARNING: no JevBench files at {jev_dir}; leak check skipped")
        return None
    jev = [json.loads(l)["state"] for f in files for l in open(f) if l.strip()]
    jev = [s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in jev]
    exact, stems = set(jev), {norm_text(s)[:60] for s in jev}
    hits = [r for r in rows if r["state"] in exact or norm_text(r["state"])[:60] in stems]
    assert not hits, f"{len(hits)} JevBench state overlaps, e.g. {hits[0]['state'][:80]!r}"
    print(f"leak check vs {len(jev)} JevBench states: 0 exact / 0 stem hits")
    return len(jev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_wf")
    ap.add_argument("--limit", type=int, default=0, help="cap every family at N rows (fast check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jevbench", default="/tmp/jevbench/datasets/public")
    ap.add_argument("--long_share", type=float, default=None,
                    help="target combined long-policy share of train+train_long (e.g. 0.25); unset (default) "
                         "= today's LONG_FRAC-only share and no train_long.jsonl is written. When set, extra "
                         "always-long rows (gen_long_extra) are generated -- never subtracted from train.jsonl "
                         "-- to close the gap to the target, and written to <out>/train_long.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n = lambda fam: args.limit or CAPS[fam]  # noqa: E731

    print("loading HF sources (cached)...")
    squad = [r for r in _load("rajpurkar/squad_v2", "train") if len(r["context"]) <= 1000]
    nli = [r for r in take(_load("stanfordnlp/snli", "train"), 60000, rng) if r["label"] != -1] + \
          [r for r in take(_load("nyu-mll/multi_nli", "train"), 60000, rng) if r["label"] != -1]
    boolq = list(_load("google/boolq", "train"))

    gens = {  # family -> (qtype, callable(n, rng, heldout_style))
        "policy_permit": ("noul", gen_policy_permit), "adequacy": ("noul", lambda k, r, h=False: gen_adequacy(squad, k, r, h)),
        "support": ("noul", lambda k, r, h=False: gen_support(nli, k, r, h)), "fact": ("noul", lambda k, r, h=False: gen_fact(boolq, k, r, h)),
        "eligibility": ("noul", gen_eligibility),
        "routing": ("choice", gen_routing), "action_select": ("choice", gen_action_select),
        "enum_extract": ("choice", gen_enum_extract), "categorical": ("choice", gen_categorical), "tool_select": ("choice", gen_tool_select),
        "quality": ("score", gen_quality), "severity": ("score", gen_severity),
        "relevance": ("score", lambda k, r, h=False: gen_relevance(squad, k, r, h)), "completeness": ("score", gen_completeness),
        "urgency": ("score", gen_urgency),
    }
    manifest = {"families": {}, "held_out_families": HELD_OUT, "held_out_style": {t: s[-1][0] for t, s in STYLES.items()},
                "caps": CAPS, "seed": args.seed, "limit": args.limit}
    train_pool, heldout = [], []
    evals = {f"wf_heldout_{t}": [] for t in HELD_OUT}
    evals["wf_heldout_style"] = []
    for fam, (qtype, gen) in gens.items():
        rows = gen(n(fam), rng)  # held-out families: trained styles only, so the family (not the wording) is what is new
        if fam in HELD_OUT.values():
            evals[f"wf_heldout_{qtype}"] += rows
            heldout += rows
        else:
            train_pool += rows
            evals["wf_heldout_style"] += gen(max(10, min(HELDOUT_N // 10, n(fam) // 4)), rng, True)
        grp = sum(1 for r in rows if r["meta"].get("rubric_group"))
        manifest["families"][fam] = {"qtype": qtype, "n": len(rows), "held_out": fam in HELD_OUT.values(), "in_rubric_groups": grp}
        print(f"  {fam:<14}{qtype:<7}{len(rows):>7} rows  ({grp} in rubric groups)")
    evals["wf_rubric_flip"] = flip_pairs(heldout)
    evals["wf_rubric_shuffled"] = shuffled_rubric(heldout, rng)

    print("\n=== split / dedupe / leak check ===")
    train_rows, val_rows = group_split(train_pool, min(VAL_TOTAL, len(train_pool) // 4), rng)
    seen = {norm_text(r["state"]) for r in train_rows}
    all_rows = list(train_rows) + list(val_rows)
    for name, rows in evals.items():
        rows = [r for r in rows if norm_text(r["state"]) not in seen]
        if name == "wf_rubric_flip":  # shuffle by pair so families interleave but pairs stay adjacent
            rows = pair_shuffle(rows, rng)
        else:
            rng.shuffle(rows)
        write_jsonl(out / "eval" / f"{name}.jsonl", rows)
        manifest[name] = len(rows)
        all_rows += rows
        print(f"  {name}: {len(rows)} rows")
    manifest["jevbench_states_checked"] = leak_check(all_rows, args.jevbench)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    manifest["total_train"], manifest["total_val"] = len(train_rows), len(val_rows)
    by = defaultdict(int)
    for r in train_rows:
        by[(r["meta"]["qtype"], r["meta"]["family"])] += 1
    manifest["train_by_family"] = {f"{q}/{f}": c for (q, f), c in sorted(by.items())}
    manifest["train_p_null_rows"] = sum(1 for r in train_rows if r["p_null"] >= 1)
    manifest["train_long_rows"] = sum(1 for r in train_rows if r["meta"].get("long"))
    if args.long_share is not None:
        assert 0 <= args.long_share < 1, "--long_share must be in [0, 1)"
        L, N = manifest["train_long_rows"], len(train_rows)
        need = max(0, round((args.long_share * N - L) / (1 - args.long_share)))
        long_rows = [r for r in gen_long_extra(need, rng) if norm_text(r["state"]) not in seen]
        leak_check(long_rows, args.jevbench)
        write_jsonl(out / "train_long.jsonl", long_rows)
        combined = N + len(long_rows)
        manifest["long_share_target"] = args.long_share
        manifest["train_long_extra_rows"] = len(long_rows)
        manifest["long_share_actual"] = (L + len(long_rows)) / combined if combined else 0.0
        print(f"  train_long.jsonl: {len(long_rows)} extra long rows (base long share {L}/{N} -> "
              f"combined {manifest['long_share_actual']:.3f}, target {args.long_share})")
    manifest["train_rubric_group_rows"] = sum(1 for r in train_rows if r["meta"].get("rubric_group"))
    manifest["flip_pairs"] = manifest["wf_rubric_flip"] // 2
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTOTAL: {len(train_rows)} train / {len(val_rows)} val -> {out}/   "
          f"rubric-group share {manifest['train_rubric_group_rows'] / max(1, len(train_rows)):.2f}, "
          f"flip pairs {manifest['flip_pairs']}")
    if len(train_rows) < 60000 and not args.limit:
        print(f"WARNING: train rows {len(train_rows)} < 60k target")


if __name__ == "__main__":
    main()
