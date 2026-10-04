"""PLAN7 Track D: DecisionMix v2 — hard workflow curriculum (data_wh) + uncertainty corpus (data_u).

New module (does not edit workflow_corpus.py, which is being edited concurrently for data_wf). Reuses its
generic, family-agnostic helpers (differing, flip_pairs, shuffled_rubric, leak_check, level_names, cap/lc)
and pcdm/data.py's row-file helpers (norm_text, group_split, write_jsonl) and the benchmark-time renderer
pcdm_jev.decider.query_text, rather than reimplementing them.

data_wh (PLAN7 Phase 9 "hard decision curriculum"): rule_depth levels 1-7, programmatic gold from a small
rule engine written as data (DOMAINS: field-level (requirement, true-fact, false-fact) triples evaluated in
Python). Every unit is a counterfactual rubric group (2-3 rows, same state + candidate set, different rubric
-> different gold; meta.rubric_group/rubric_variant). Levels 1-6 -> train/val; level 7 (temporal/numeric/
probability/trade-off composition) is entirely eval-only. Holdouts: one rule "grammar" (domain) per level,
3 whole workflow-family domains, 2 whole rubric styles ("terse"/"audit").

data_u (PLAN7 Phase 7A "U — uncertainty"): genuine probability targets. Real annotator-grounded sources —
UNLI's *validation* split (scalar entailment prob; UNLI *train* is already ~fully consumed by pcdm/data.py's
"unli" E-task and UNLI *test* is already pcdm/data.py's "unli_test" eval, so both are left untouched here to
avoid training on / duplicating an existing eval set) and metaeval/ambient (ambiguity sets, untouched
elsewhere in this repo). metaeval/chaos-mnli-ambiguity is fully consumed by pcdm/data.py's "chaos_mnli" eval
already, so it is used here **only** to build our own eval file (read-only reuse of an eval set, never
trained on). Plus four synthetic generators with exactly-computed distributions (partial evidence, noisy
sensor, ordinal confusion, conflicting sources), each with a held-out parameter range reserved for eval.

uv run --no-sync python scripts/decisionmix_v2.py [--out_wh data_wh] [--out_u data_u] [--limit N] [--seed 0]
                                                    [--corpus wh|u|both] [--jevbench /tmp/jevbench/datasets/public]
"""
import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pcdm"))
from data import norm_text, group_split, write_jsonl  # noqa: E402
from pcdm_jev.decider import query_text  # noqa: E402
from scripts.workflow_corpus import cap, differing, flip_pairs, shuffled_rubric, leak_check, level_names  # noqa: E402

MAX_STATE_CHARS = 14000
STYLE_NAMES = ["plain", "checklist", "formal", "terse", "audit"]
HELD_OUT_STYLES = {"terse", "audit"}  # 2 whole rubric styles, never trained
_gid = itertools.count(1)

# ============================================================================
# DecisionMix v2 required row metadata (memo "Required metadata"), shared by data_wh and data_u.
# ============================================================================


def make_row(decision_type, task, candidates, gold_idx, query, state, *, question=None,
             rule_depth=0, exception_depth=0,
             workflow_family="", rubric_family="", rubric_style="", gold_source="programmatic",
             soft_target=False, source_dataset="synthetic:decisionmix_v2", fam_bucket="W", target=None,
             catch_all_present=False, extra_meta=None):
    """Finish a row in the training schema (state/query/candidates/target/p_null/task/label/meta).
    Soft-target rows (data_u) pass `target` explicitly and gold_idx = argmax (label kept for convenience,
    p_null stays 0). Null rows (data_wh) pass gold_idx=-1 and a uniform `target` over the remaining
    candidates; p_null is then 1."""
    gold_present = gold_idx is not None and gold_idx >= 0
    if target is None:
        target = [1.0 if i == gold_idx else 0.0 for i in range(len(candidates))]
    meta = {
        "decision_type": decision_type, "task_family": "decisionmix_v2", "workflow_family": workflow_family,
        "family": workflow_family,  # alias: lets workflow_corpus.shuffled_rubric (keys on meta.family) work unmodified
        "rubric_family": rubric_family, "rubric_style": rubric_style, "rule_depth": rule_depth,
        "exception_depth": exception_depth, "candidate_count": len(candidates), "gold_present": gold_present,
        "catch_all_present": catch_all_present, "requires_temporal": False, "requires_numeric": False,
        "requires_probability": False, "requires_tradeoff": False, "gold_source": gold_source,
        "soft_target": soft_target, "source_dataset": source_dataset, "fam_bucket": fam_bucket,
    }
    if extra_meta:
        meta.update(extra_meta)
    row = {"state": state[:MAX_STATE_CHARS], "query": query, "candidates": list(candidates), "target": target,
            "p_null": 1.0 if not gold_present and not soft_target else 0.0,
            "task": task, "label": gold_idx if gold_present else (target.index(max(target)) if soft_target else -1),
            "meta": meta}
    if question is not None:
        row["question"] = question
    return row


def render_state(sentences, rng):
    parts = list(sentences)
    rng.shuffle(parts)
    return " ".join(parts)


def pick_style(rng, heldout=False):
    return rng.choice(list(HELD_OUT_STYLES)) if heldout else rng.choice([s for s in STYLE_NAMES if s not in HELD_OUT_STYLES])


STYLE_NOUL = {
    "plain": lambda s: (f"Should you {s['verb']}? Answer true if {s['rule']}; otherwise answer false.",
                        {"true": f"{cap(s['rule'])}.", "false": f"It is not the case that {s['rule']}."}),
    "checklist": lambda s: (f"Decision: {s['verb']}. Work only from facts stated in the state. Rule: {s['rule']}.",
                            {"true": f"Answer true when {s['rule']}.", "false": "Answer false otherwise."}),
    "formal": lambda s: (f"Determine whether the following holds: {s['rule']}. This determines whether to {s['verb']}.",
                         {"true": f"Yes — {s['rule']}.", "false": "No — the stated rule is not satisfied."}),
    "terse": lambda s: (f"{cap(s['verb'])}? Rule: {s['rule']}.", {"true": "condition met", "false": "condition not met"}),
    "audit": lambda s: (f"As reviewer, confirm: {s['rule']}. This governs whether to {s['verb']}. Use only stated facts.",
                        {"true": f"Confirmed: {s['rule']}.", "false": "Not confirmed."}),
}
STYLE_CHOICE = {
    "plain": lambda s: (f"Which outcome applies for {s['verb']}? {s.get('caveat', '')}", dict(s["criteria"])),
    "checklist": lambda s: (f"Select exactly one outcome for {s['verb']}. {s.get('caveat', '')}",
                            {k: f"Choose when {v}" for k, v in s["criteria"].items()}),
    "formal": lambda s: (f"Determine the applicable outcome for: {s['verb']}. {s.get('caveat', '')}",
                         {k: f"Applies when {v}" for k, v in s["criteria"].items()}),
    "terse": lambda s: (f"Outcome for {s['verb']}?", dict(s["criteria"])),
    "audit": lambda s: (f"As reviewer, select the outcome for {s['verb']} per the rules below. {s.get('caveat', '')}",
                        {k: f"Select when {v}" for k, v in s["criteria"].items()}),
}
STYLE_SCORE = {
    "plain": lambda s: (f"Rate the {s['what']}. {s.get('caveat', '')}", list(s["levels"])),
    "checklist": lambda s: (f"Assign a {s['what']} level using only the facts given. {s.get('caveat', '')}", list(s["levels"])),
    "formal": lambda s: (f"Determine the {s['what']} level. {s.get('caveat', '')}", list(s["levels"])),
    "terse": lambda s: (f"{cap(s['what'])}?", list(s["levels"])),
    "audit": lambda s: (f"As reviewer, assign the {s['what']} level per the rubric below. {s.get('caveat', '')}", list(s["levels"])),
}
RENDER = {"noul": STYLE_NOUL, "choice": STYLE_CHOICE, "score": STYLE_SCORE}


def finish(decision_type, spec, candidates, gold_idx, rng, *, task, rule_depth, workflow_family, rubric_family,
           state_sentences, exception_depth=0, style_name=None, heldout_style=False, catch_all=False,
           requires=None, extra_meta=None):
    style_name = style_name or pick_style(rng, heldout_style)
    instructions, criteria = RENDER[decision_type][style_name](spec)
    question = {"type": decision_type, "instructions": instructions, "criteria": criteria}
    query = query_text(question)
    # state_sentences is rendered ONCE per group by the caller and passed as a string across all
    # variants (a group must share the exact same state string); a raw list is still accepted and
    # rendered here for single-row callers.
    state = state_sentences if isinstance(state_sentences, str) else render_state(state_sentences, rng)
    row = make_row(decision_type, task, candidates, gold_idx, query, state, question=question,
                   rule_depth=rule_depth,
                   exception_depth=exception_depth, workflow_family=workflow_family, rubric_family=rubric_family,
                   rubric_style=style_name, catch_all_present=catch_all, extra_meta=extra_meta)
    for k in (requires or []):
        row["meta"][k] = True
    return row


def null_group(rows, rng, frac):
    """~frac of same-candidate-set groups: remove one variant's gold from every row's candidate list (keeps
    the group's candidate set identical across variants), so that variant becomes a true null row (gold_present
    False, p_null=1, uniform target) while the others keep their own gold at the shifted index."""
    if len(rows) < 2 or len(rows[0]["candidates"]) < 3 or rng.random() >= frac:
        return rows
    cands = rows[0]["candidates"]
    golds = [r["label"] for r in rows if r["label"] >= 0]
    if not golds:
        return rows
    victim = rng.choice(golds)
    new_cands = [c for i, c in enumerate(cands) if i != victim]
    out = []
    for r in rows:
        r = dict(r, candidates=list(new_cands))
        question = dict(r["question"])
        criteria = question["criteria"]
        question["criteria"] = ([v for i, v in enumerate(criteria) if i != victim]
                                if isinstance(criteria, list)
                                else {k: v for k, v in criteria.items() if k != cands[victim]})
        r["question"] = question
        r["meta"] = dict(r["meta"], candidate_count=len(new_cands))
        if r["label"] == victim:
            r["label"], r["p_null"] = -1, 1.0
            r["target"] = [1.0 / len(new_cands)] * len(new_cands)
            r["meta"]["gold_present"] = False
        else:
            old = r["label"]
            r["label"] = old - 1 if old > victim else old
            r["target"] = [1.0 if i == r["label"] else 0.0 for i in range(len(new_cands))]
        out.append(r)
    return out


def emit_group(builder, tries=8):
    """builder() -> list[row] sharing state+candidates, or []/None if this attempt's golds don't differ.
    Retries until >=2 distinct golds (mirrors workflow_corpus.differing / emit_group)."""
    for _ in range(tries):
        rows = builder()
        if rows and differing([(None, r["label"]) for r in rows]):
            gid = f"g{next(_gid)}"
            for vi, r in enumerate(rows):
                r["meta"] = dict(r["meta"], rubric_group=gid, rubric_variant=vi)
            return rows
    return None


# ============================================================================
# data_wh — rule engine "written as data": each domain is a verb + 6 independent boolean
# conditions (requirement phrase, true-fact sentence, false-fact sentence) + 4 named outcomes
# used by the precedence (L4) generator. Nothing here is copied from workflow_corpus.py's DOMAINS.
# ============================================================================
DOMAINS = {
    "loan_approval": dict(verb="approve the loan", routes=["approve", "approve_with_conditions", "deny", "refer_to_underwriter"], fields=[
        ("the applicant's income was independently verified", "The applicant's income was verified with pay stubs.", "The applicant's income could not be verified."),
        ("the applicant's credit score is at least 650", "The applicant's credit score is 710.", "The applicant's credit score is 590."),
        ("the applicant's debt-to-income ratio is under 40%", "The applicant's debt-to-income ratio is 28%.", "The applicant's debt-to-income ratio is 55%."),
        ("collateral was pledged", "The applicant pledged a vehicle as collateral.", "No collateral was pledged."),
        ("a qualified cosigner is on the application", "A cosigner with sufficient income is listed.", "There is no cosigner."),
        ("the applicant has no prior default on file", "The applicant has no prior defaults.", "The applicant defaulted on a loan two years ago."),
    ]),
    "shift_swap": dict(verb="approve the shift swap", routes=["approve", "approve_with_manager_review", "deny", "waitlist"], fields=[
        ("both employees are qualified for the shift", "Both employees hold the required certification.", "One employee lacks the required certification."),
        ("the swap does not create overtime", "Neither employee exceeds 40 hours after the swap.", "One employee would exceed 40 hours after the swap."),
        ("the request was submitted at least 48 hours ahead", "The request was submitted 5 days ahead.", "The request was submitted 6 hours ahead."),
        ("the team keeps minimum staffing that day", "Staffing stays at or above the minimum.", "Staffing would fall below the minimum."),
        ("neither employee is on a written warning", "Neither employee has a current written warning.", "One employee has a current written warning."),
        ("the swap is within the same department", "Both shifts are in the same department.", "The shifts are in different departments."),
    ]),
    "vendor_onboarding": dict(verb="approve the vendor for onboarding", routes=["approve", "approve_probationary", "reject", "escalate_to_legal"], fields=[
        ("the vendor passed the security questionnaire", "The vendor's security questionnaire score was 92%.", "The vendor's security questionnaire score was 41%."),
        ("the vendor carries adequate insurance", "The vendor's insurance certificate meets the minimum.", "The vendor's insurance is below the required minimum."),
        ("references were checked and positive", "Two references were checked and both were positive.", "References were not checked."),
        ("the vendor is not on a sanctions list", "The vendor does not appear on any sanctions list.", "The vendor appears on a sanctions watchlist."),
        ("a signed data processing agreement is on file", "A signed DPA is on file.", "No DPA has been signed."),
        ("the contract value is under the delegated authority limit", "The contract value is $40,000.", "The contract value is $600,000."),
    ]),
    "content_moderation": dict(verb="approve the post for publication", routes=["approve", "approve_with_label", "reject", "escalate_to_trust_and_safety"], fields=[
        ("the post contains no prohibited content", "An automated scan found no prohibited content.", "The post contains content flagged as prohibited."),
        ("the claims are sourced", "Every factual claim links to a cited source.", "The post makes unsourced factual claims."),
        ("the account is in good standing", "The posting account has no active strikes.", "The posting account has two active strikes."),
        ("the post was not mass-reported", "The post has zero user reports.", "The post has 40 user reports in the last hour."),
        ("images in the post are licensed", "All images have a usage license on file.", "One image has no usage license."),
        ("the post is not paid political content", "The post is not tagged as paid political content.", "The post is tagged as paid political content."),
    ]),
    "warranty_claim": dict(verb="approve the warranty claim", routes=["approve", "approve_partial", "deny", "send_for_inspection"], fields=[
        ("the product is within its warranty period", "The product was purchased 8 months ago; the warranty is 24 months.", "The product was purchased 40 months ago; the warranty is 24 months."),
        ("the defect is not from misuse", "The technician's note rules out misuse.", "The technician's note attributes the defect to misuse."),
        ("proof of purchase was provided", "A dated receipt was provided.", "No proof of purchase was provided."),
        ("the serial number matches the manufacturer record", "The serial number matches the manufacturer's record.", "The serial number does not match any record."),
        ("the product was not previously repaired by a third party", "No third-party repair history exists.", "The product shows signs of a prior third-party repair."),
        ("the claim was filed within 30 days of the defect", "The claim was filed 6 days after the defect appeared.", "The claim was filed 95 days after the defect appeared."),
    ]),
    "visa_extension": dict(verb="approve the visa extension", routes=["approve", "approve_short_term", "deny", "refer_to_consulate"], fields=[
        ("the applicant's current visa has not expired", "The current visa expires in 6 weeks.", "The current visa expired 2 weeks ago."),
        ("the applicant has sufficient funds documented", "Bank statements show sufficient funds.", "No financial documentation was provided."),
        ("the applicant has no unresolved immigration violations", "No violations are on record.", "An unresolved overstay is on record."),
        ("a sponsoring employer or institution confirmed status", "The employer confirmed continued employment.", "No sponsor confirmation was received."),
        ("the applicant has valid travel document coverage", "The passport is valid for 18 more months.", "The passport expires in 2 months."),
        ("the extension request was filed before expiry", "The request was filed 3 weeks before expiry.", "The request was filed 10 days after expiry."),
    ]),
    "insurance_claim": dict(verb="approve the insurance claim", routes=["approve", "approve_partial", "deny", "refer_to_siu"], fields=[
        ("the policy was active on the date of loss", "The policy was active on the date of loss.", "The policy had lapsed before the date of loss."),
        ("the loss matches a covered peril", "The loss matches a peril named in the policy.", "The loss does not match any covered peril."),
        ("the claimed amount is supported by documentation", "Itemised receipts support the claimed amount.", "No documentation supports the claimed amount."),
        ("there is no indication of fraud", "The adjuster found no indication of fraud.", "The adjuster flagged inconsistencies suggesting fraud."),
        ("the deductible has been accounted for", "The deductible was correctly subtracted.", "The deductible was not applied."),
        ("the claim was reported within the policy's notice window", "The claim was reported 4 days after the loss.", "The claim was reported 120 days after the loss."),
    ]),
    "feature_flag_rollout": dict(verb="roll out the feature flag", routes=["rollout_100", "rollout_canary", "hold", "rollback"], fields=[
        ("error rates stayed within the acceptable band", "Error rate held at 0.2%, within the 0.5% band.", "Error rate spiked to 4.1%, above the 0.5% band."),
        ("the on-call engineer signed off", "The on-call engineer signed off.", "The on-call engineer has not signed off."),
        ("a rollback plan is documented", "A tested rollback plan is documented.", "No rollback plan is documented."),
        ("latency stayed within budget", "P99 latency stayed under the 200ms budget.", "P99 latency rose to 900ms, over budget."),
        ("the change has passed staging for 48 hours", "The change has run cleanly in staging for 5 days.", "The change has been in staging for 3 hours."),
        ("no related incident is currently open", "No related incident is open.", "A related incident is currently open."),
    ]),
    "subscription_cancellation": dict(verb="waive the cancellation fee", routes=["waive_fully", "waive_partial", "deny", "escalate_to_retention"], fields=[
        ("the cancellation is within the cooling-off period", "The cancellation request was made 4 days after signup.", "The cancellation request was made 200 days after signup."),
        ("the customer cites a service outage", "The customer cites a documented multi-day outage.", "No outage is documented for this period."),
        ("no prior fee waiver was granted", "No prior waiver was granted on this account.", "A waiver was already granted on this account this year."),
        ("the account has no outstanding balance", "The account has no outstanding balance.", "The account has an outstanding balance of $340."),
        ("the customer is on a month-to-month plan", "The customer is on a month-to-month plan.", "The customer is on a 12-month contract with 8 months remaining."),
        ("the request came through an official channel", "The request was submitted through the official portal.", "The request was made verbally with no ticket on file."),
    ]),
    "procurement_request": dict(verb="approve the purchase order", routes=["approve", "approve_with_budget_note", "deny", "escalate_to_finance"], fields=[
        ("the request is within the department's remaining budget", "The department has $50,000 remaining and the PO is $12,000.", "The department has $2,000 remaining and the PO is $40,000."),
        ("at least two competing quotes were obtained", "Three competing quotes were obtained.", "Only one quote was obtained."),
        ("the vendor is on the approved vendor list", "The vendor is on the approved list.", "The vendor is not on the approved list."),
        ("the request has manager sign-off", "The requesting manager signed off.", "No manager sign-off is recorded."),
        ("the item is not on the restricted-purchase list", "The item is not restricted.", "The item is on the restricted-purchase list."),
        ("delivery falls within the current fiscal year", "Delivery is scheduled for this fiscal year.", "Delivery falls in the next fiscal year."),
    ]),
    "research_ethics_review": dict(verb="approve the study protocol", routes=["approve", "approve_with_conditions", "reject", "refer_to_full_board"], fields=[
        ("informed consent procedures are adequate", "The consent form covers all required disclosures.", "The consent form omits required risk disclosures."),
        ("the study poses no more than minimal risk", "The protocol is assessed as minimal risk.", "The protocol involves more than minimal risk."),
        ("a data safety monitoring plan is included", "A monitoring plan is included.", "No monitoring plan is included."),
        ("vulnerable populations are adequately protected", "Extra safeguards for minors are specified.", "The protocol involves minors with no extra safeguards."),
        ("the sample size is statistically justified", "A power analysis justifies the sample size.", "No justification is given for the sample size."),
        ("conflicts of interest are disclosed", "All investigator conflicts of interest are disclosed.", "An investigator's funding conflict is undisclosed."),
    ]),
}
HELD_OUT_FAMILIES = {"subscription_cancellation", "procurement_request", "research_ethics_review"}
TRAIN_DOMAINS = [d for d in DOMAINS if d not in HELD_OUT_FAMILIES]
DISTRACTOR_BANK = [
    "The ticket was opened on a Wednesday.", "The case was assigned to queue 14.", "The customer's timezone is UTC-5.",
    "This is the second contact on this case.", "The account was created three years ago.", "The reviewer's shift ends at 6pm.",
    "The request came in through the mobile app.", "A similar request was handled last quarter.",
    "The record was last touched by a different reviewer.", "The customer's preferred language is Spanish.",
    "The office handling this case is in the EMEA region.", "The internal reference number is on file.",
    "The case has no linked duplicate tickets.", "The last status update was routine.",
    "The reviewer noted the weather was unrelated to the request.", "A macro was used to draft the initial reply.",
    "The customer has an active loyalty membership.", "The form was submitted from a shared workstation.",
    "No screenshots were attached to the ticket.", "The case queue depth was normal that day.",
]


# ============================================================================
# Level generators. Each returns a builder() usable with emit_group: samples fresh facts, builds
# 2-3 rubric variants over the SAME state + candidate set, returns their rows (golds may agree; the
# caller retries via emit_group until >=2 golds differ).
# ============================================================================

def sample_facts(dom, rng):
    return {i: rng.random() < 0.5 for i in range(len(dom["fields"]))}


def fact_sentences(dom, facts, idxs=None):
    idxs = idxs if idxs is not None else range(len(dom["fields"]))
    return [dom["fields"][i][1] if facts[i] else dom["fields"][i][2] for i in idxs]


CHOICE_RENDER_FRAC = 0.42  # boolean-rule levels (1,2,3,5): render as noul(true/false) vs choice(proceed/hold)
SCORE_FRAC = 0.42          # levels 2,3,4: delegate to score_case (ordinal risk tier) instead of the level's own rule


def bool_render(is_choice, verb, rule_text, gold_bool):
    """One boolean rule -> noul(true/false) or choice(proceed/hold) spec, decided ONCE per group (is_choice
    fixed for the whole group) so every variant shares the same decision_type + candidate set."""
    if is_choice:
        crit = {"proceed": f"applies when {rule_text}", "hold": "applies when the condition above does not hold"}
        return "choice", ["proceed", "hold"], (0 if gold_bool else 1), {"verb": verb, "criteria": crit, "caveat": ""}
    return "noul", ["true", "false"], (0 if gold_bool else 1), {"verb": verb, "rule": rule_text}


def score_case(domain, rng, heldout_style, rule_depth):
    """Ordinal risk tier from counting unmet conditions among 4 sampled fields; 2 rubric variants = 2
    different counting rules -> different tier at the same facts. Reuses level_names for label naming."""
    dom = DOMAINS[domain]
    facts = sample_facts(dom, rng)
    idxs = rng.sample(range(len(dom["fields"])), 4)
    state = render_state(fact_sentences(dom, facts), rng)
    unmet = sum(1 for i in idxs if not facts[i])
    first_unmet = not facts[idxs[0]]
    names, _ = level_names(4, rng)
    # two genuinely different rules (not just re-labelled staircases): plain count vs. "first field is
    # decisive" -- these disagree whenever idxs[0] is unmet but the overall count is still low.
    rules = {"count": lambda u, fu: min(3, u),
             "worst_first": lambda u, fu: 3 if fu else min(2, u)}
    rows = []
    for rname, fn in rules.items():
        tier = fn(unmet, first_unmet)
        spec = {"what": f"{domain.replace('_', ' ')} risk", "levels": names, "caveat": f"Counting rule: {rname}."}
        rows.append(finish("score", spec, names, tier, rng, task=f"wh_L{rule_depth}_{domain}", rule_depth=rule_depth,
                           workflow_family=domain, rubric_family=f"{domain}_score_{rname}", state_sentences=state,
                           heldout_style=heldout_style, extra_meta={"unmet_count": unmet}))
    return rows


def l1_builder(domain, rng, heldout_style=False):
    dom = DOMAINS[domain]

    def build():
        facts = sample_facts(dom, rng)
        idxs = rng.sample(range(len(dom["fields"])), rng.randint(2, 3))
        state = render_state(fact_sentences(dom, facts), rng)
        is_choice = rng.random() < CHOICE_RENDER_FRAC
        rows = []
        for i in idxs:
            req = dom["fields"][i][0]
            dtype, cands, gold, spec = bool_render(is_choice, dom["verb"], req, facts[i])
            rows.append(finish(dtype, spec, cands, gold, rng, task=f"wh_L1_{domain}", rule_depth=1,
                               workflow_family=domain, rubric_family=f"{domain}_single:{i}", state_sentences=state,
                               heldout_style=heldout_style))
        return rows
    return build


def l2_builder(domain, rng, heldout_style=False):
    dom = DOMAINS[domain]

    def build():
        if rng.random() < SCORE_FRAC:
            return score_case(domain, rng, heldout_style, 2)
        facts = sample_facts(dom, rng)
        base_idxs = rng.sample(range(len(dom["fields"])), 4)
        state = render_state(fact_sentences(dom, facts), rng)
        is_choice = rng.random() < CHOICE_RENDER_FRAC
        rows = []
        for _ in range(rng.randint(2, 3)):
            i, j = rng.sample(base_idxs, 2)
            op = rng.choice(["and", "or"])
            req = f"{dom['fields'][i][0]} {op} {dom['fields'][j][0]}"
            gold_bool = (facts[i] and facts[j]) if op == "and" else (facts[i] or facts[j])
            dtype, cands, gold, spec = bool_render(is_choice, dom["verb"], req, gold_bool)
            rows.append(finish(dtype, spec, cands, gold, rng, task=f"wh_L2_{domain}",
                               rule_depth=2, workflow_family=domain, rubric_family=f"{domain}_{op}:{i}.{j}",
                               state_sentences=state, heldout_style=heldout_style))
        return rows
    return build


def l3_builder(domain, rng, heldout_style=False):
    """Base rule + exception ("unless <field>"): gold = base AND NOT exception."""
    dom = DOMAINS[domain]

    def build():
        if rng.random() < SCORE_FRAC:
            return score_case(domain, rng, heldout_style, 3)
        facts = sample_facts(dom, rng)
        idxs = rng.sample(range(len(dom["fields"])), 4)
        base_i = idxs[0]
        state = render_state(fact_sentences(dom, facts), rng)
        is_choice = rng.random() < CHOICE_RENDER_FRAC
        rows = []
        for exc_j in idxs[1:]:
            req = (f"{dom['fields'][base_i][0]}, unless {dom['fields'][exc_j][0]} "
                   f"(in which case the exception overrides and the answer is false)")
            gold_bool = facts[base_i] and not facts[exc_j]
            dtype, cands, gold, spec = bool_render(is_choice, dom["verb"], req, gold_bool)
            rows.append(finish(dtype, spec, cands, gold, rng, task=f"wh_L3_{domain}",
                               rule_depth=3, exception_depth=1, workflow_family=domain,
                               rubric_family=f"{domain}_exc:{base_i}/{exc_j}", state_sentences=state,
                               heldout_style=heldout_style))
        return rows
    return build


def l4_builder(domain, rng, heldout_style=False):
    """Precedence: 3 rules each mapped to a distinct named outcome, first-true-wins, else the 4th (default)
    outcome. Variants = the SAME 3 rules under a different precedence order -> the rubric IS the ordering."""
    dom = DOMAINS[domain]

    def build():
        if rng.random() < SCORE_FRAC:
            return score_case(domain, rng, heldout_style, 4)
        facts = sample_facts(dom, rng)
        idxs = rng.sample(range(len(dom["fields"])), 3)
        outcomes = dom["routes"][:3]
        default = dom["routes"][3]
        for i in rng.sample(idxs, 2):  # bias so >=2 conditions hold -> precedence order can matter
            facts[i] = True
        state = render_state(fact_sentences(dom, facts), rng)
        rules = list(zip(idxs, outcomes))
        rows = []
        for _ in range(rng.randint(2, 3)):
            order = rules[:]
            rng.shuffle(order)
            gold_label = next((o for i, o in order if facts[i]), default)
            crit = {o: f"applies when {dom['fields'][i][0]}" for i, o in order}
            crit[default] = "applies when none of the above rules fire"
            cands = [o for _, o in order] + [default]
            spec = {"verb": dom["verb"], "criteria": crit,
                    "caveat": "Apply the rules in the stated precedence order; the first matching rule wins."}
            rows.append(finish("choice", spec, cands, cands.index(gold_label), rng, task=f"wh_L4_{domain}",
                               rule_depth=4, exception_depth=len(order) - 1, workflow_family=domain,
                               rubric_family=f"{domain}_prec:{sorted(idxs)}", state_sentences=state,
                               heldout_style=heldout_style, extra_meta={"precedence_order": [i for i, _ in order]}))
        return rows
    return build


def l5_builder(domain, rng, heldout_style=False):
    """Multi-hop chain (2-4 hops): stage_1 = field[0]; stage_k = stage_{k-1} <op_k> field[k]. Facts are
    stated as independent sentences; the reader must combine them. Variants: same fields, different op
    sequence (AND/OR at a hop) -> different final outcome."""
    dom = DOMAINS[domain]

    def build():
        facts = sample_facts(dom, rng)
        depth = rng.randint(2, 4)
        idxs = rng.sample(range(len(dom["fields"])), depth)
        state = render_state(fact_sentences(dom, facts), rng)
        is_choice = rng.random() < CHOICE_RENDER_FRAC
        rows = []
        for _ in range(rng.randint(2, 3)):
            ops = [rng.choice(["and", "or"]) for _ in range(depth - 1)]
            stage = facts[idxs[0]]
            chain_desc = dom["fields"][idxs[0]][0]
            for k in range(1, depth):
                stage = (stage and facts[idxs[k]]) if ops[k - 1] == "and" else (stage or facts[idxs[k]])
                chain_desc = f"({chain_desc}) {ops[k - 1]} {dom['fields'][idxs[k]][0]}"
            rule_text = f"working through each step in order, {chain_desc}"
            dtype, cands, gold, spec = bool_render(is_choice, dom["verb"], rule_text, stage)
            rows.append(finish(dtype, spec, cands, gold, rng, task=f"wh_L5_{domain}",
                               rule_depth=5, workflow_family=domain, rubric_family=f"{domain}_chain{depth}:{idxs}",
                               state_sentences=state, heldout_style=heldout_style, extra_meta={"hops": depth}))
        return rows
    return build


L1_L4 = [l1_builder, l2_builder, l3_builder, l4_builder]


def l6_builder(domain, rng, heldout_style=False):
    """Long policy: wrap a random L1-L4 rule kind, but pad the state to ~1-2.5k tokens with irrelevant
    clauses (rule_depth 6 = length/distraction, not new logic)."""
    base_fn = rng.choice(L1_L4)

    def build():
        rows = base_fn(domain, rng, heldout_style)()
        # ponytail: pad the already-rendered state directly rather than threading a long= flag through
        # every L1-L4 builder -- same effect (irrelevant-clause padding), smaller diff. Padded ONCE (not
        # per row) since every variant in the group must share the exact same state string.
        if rows:
            target_chars = rng.randint(4200, 10000)  # ~1-2.5k tokens at ~4 chars/token
            parts = [rows[0]["state"]]
            while sum(len(p) for p in parts) < target_chars:
                parts.append(rng.choice(DISTRACTOR_BANK))
            rng.shuffle(parts)
            padded = " ".join(parts)[:MAX_STATE_CHARS]
        for r in rows:
            r["state"] = padded
            for lv in (1, 2, 3, 4):
                r["task"] = r["task"].replace(f"_L{lv}_", "_L6_")
            r["meta"]["rule_depth"] = 6
        return rows
    return build


LEVEL_BUILDERS = {1: l1_builder, 2: l2_builder, 3: l3_builder, 4: l4_builder, 5: l5_builder, 6: l6_builder}


# ============================================================================
# Level 7 — temporal / numeric / probability / trade-off composition. Eval-only (never trained);
# see build_wh(). Each subtype still emits a 2-variant counterfactual group (boundary rubric change).
# ============================================================================

def l7_temporal_builder(rng, heldout_style=False):
    def build():
        submit_off, resp_off = rng.randint(-8, 8), rng.randint(-8, 8)
        elapsed = rng.choice([2, 6, 11, 12, 13, 22, 23, 24, 25, 26, 47, 48, 49, 70])
        deadline_h = rng.choice([12, 24, 48])
        resp_hour, resp_days = (9 + elapsed) % 24, (9 + elapsed) // 24
        state = render_state([f"Request submitted at 09:00 (UTC{submit_off:+d}) on day 1.",
                              f"Response sent at {resp_hour:02d}:00 (UTC{resp_off:+d}) on day {1 + resp_days}."], rng)
        rows = []
        for grace, tag in [(0, "strict"), (2, "grace")]:
            ok = elapsed <= deadline_h + grace
            rule = f"the elapsed time from submission to response, converted to a common timezone, is at most {deadline_h} hours"
            if grace:
                rule += f" plus a {grace}-hour grace period"
            spec = {"verb": "mark the response as on time", "rule": rule}
            rows.append(finish("noul", spec, ["true", "false"], 0 if ok else 1, rng, task="wh_L7_temporal",
                               rule_depth=7, workflow_family="temporal", rubric_family=f"temporal_{tag}",
                               state_sentences=state, heldout_style=heldout_style, requires=["requires_temporal"],
                               extra_meta={"deadline_h": deadline_h, "elapsed_h": elapsed}))
        return rows
    return build


UNIT_PAIRS = [("lbs", "kg", lambda x: x * 0.453592), ("miles", "km", lambda x: x * 1.609344),
              ("minutes", "hours", lambda x: x / 60), ("fahrenheit", "celsius", lambda f: (f - 32) * 5 / 9)]


def l7_numeric_builder(rng, heldout_style=False):
    def build():
        unit_a, unit_b, conv = rng.choice(UNIT_PAIRS)
        threshold_b = rng.choice([5, 10, 20, 50, 100])
        near = rng.random() < 0.4
        val_b = threshold_b if near else threshold_b * rng.uniform(0.5, 1.8)
        # invert conv (all convs here are affine) to get a measurement in unit_a that maps near val_b
        val_a = val_b * 9 / 5 + 32 if unit_a == "fahrenheit" else val_b / conv(1.0)
        val_a = round(val_a, 1)
        measured_b = conv(val_a)
        state = render_state([f"Measurement: {val_a} {unit_a}.", f"Policy threshold is stated in {unit_b}."], rng)
        rows = []
        for op, tag in [(">=", "meets_or_exceeds"), (">", "strictly_exceeds")]:
            ok = measured_b >= threshold_b if op == ">=" else measured_b > threshold_b
            spec = {"verb": "flag the measurement as over threshold",
                    "rule": f"the measurement, converted to {unit_b}, is {op} {threshold_b} {unit_b}"}
            rows.append(finish("noul", spec, ["true", "false"], 0 if ok else 1, rng, task="wh_L7_numeric",
                               rule_depth=7, workflow_family="numeric", rubric_family=f"numeric_{tag}:{unit_a}-{unit_b}",
                               state_sentences=state, heldout_style=heldout_style, requires=["requires_numeric"],
                               extra_meta={"threshold_b": threshold_b, "measured_b": round(measured_b, 2)}))
        return rows
    return build


def l7_probability_builder(rng, heldout_style=False):
    def build():
        opts = [f"option_{c}" for c in "abc"[:rng.choice([2, 3])]]
        params = {o: (round(rng.uniform(0.1, 0.9), 2), rng.choice([50, 100, 200, 500]), rng.choice([10, 20, 50, 100]))
                  for o in opts}
        state = render_state([f"{o}: probability of success {p}, payoff on success {pay}, cost on failure {c}."
                              for o, (p, pay, c) in params.items()], rng)
        ev = {o: p * pay - (1 - p) * c for o, (p, pay, c) in params.items()}
        worst_case = {o: -c for o, (p, pay, c) in params.items()}  # worst possible outcome (failure)
        rows = []
        for key, tag, rule in [(ev, "max_ev", "choose the option with the highest expected value (probability-weighted payoff minus expected cost)"),
                                (worst_case, "max_worst_case", "choose the option that is safest in the worst case (smallest possible loss)")]:
            best = max(key, key=key.get)
            crit = {o: f"the expected outcome under this rule is highest for {o}" for o in opts}
            spec = {"verb": "select an option", "criteria": crit, "caveat": f"Rule: {rule}."}
            rows.append(finish("choice", spec, opts, opts.index(best), rng, task="wh_L7_probability", rule_depth=7,
                               workflow_family="probability", rubric_family=f"probability_{tag}", state_sentences=state,
                               heldout_style=heldout_style, requires=["requires_probability"]))
        return rows
    return build


def l7_tradeoff_builder(rng, heldout_style=False):
    def build():
        opts = [f"option_{c}" for c in "abc"[:rng.choice([2, 3])]]
        scores = {o: (rng.randint(1, 10), rng.randint(1, 10)) for o in opts}  # (quality, cost_saved), deliberately conflicting
        state = render_state([f"{o}: quality score {q}/10, cost-saved score {c}/10." for o, (q, c) in scores.items()], rng)
        rows = []
        for wq, wc, tag in [(0.7, 0.3, "quality_weighted"), (0.3, 0.7, "cost_weighted")]:
            weighted = {o: wq * q + wc * c for o, (q, c) in scores.items()}
            best = max(weighted, key=weighted.get)
            crit = {o: "highest weighted score under this rubric's weighting" for o in opts}
            spec = {"verb": "select an option", "criteria": crit,
                    "caveat": f"Weight quality {int(wq * 100)}% and cost-saved {int(wc * 100)}% when combining the two scores."}
            rows.append(finish("choice", spec, opts, opts.index(best), rng, task="wh_L7_tradeoff", rule_depth=7,
                               workflow_family="tradeoff", rubric_family=f"tradeoff_{tag}", state_sentences=state,
                               heldout_style=heldout_style, requires=["requires_tradeoff", "requires_numeric"]))
        return rows
    return build


L7_BUILDERS = [l7_temporal_builder, l7_numeric_builder, l7_probability_builder, l7_tradeoff_builder]

# one domain per level (1-6) whose rows for that level are held out entirely -> wh_heldout_grammar
HELD_OUT_GRAMMAR = dict(zip(range(1, 7), TRAIN_DOMAINS[:6]))
NULL_GROUP_FRAC = 0.50  # only K>=3 groups (score + L4-precedence) are null-eligible; tuned empirically for ~10% overall
CATCH_ALL_FRAC = 0.20


def _apply_catch_all(rows, rng):
    if rows[0]["meta"]["decision_type"] not in ("choice", "score") or rng.random() >= CATCH_ALL_FRAC:
        return rows
    out = []
    for r in rows:
        cands, target = list(r["candidates"]), list(r["target"])
        extra_cand = "not applicable" if r["meta"]["decision_type"] == "score" else "none_apply"
        r = dict(r, candidates=cands + [extra_cand], target=target + [0.0])
        question = dict(r["question"])
        criteria = question["criteria"]
        if isinstance(criteria, list):
            question["criteria"] = criteria + ["The rubric does not apply to this case"]
        else:
            question["criteria"] = {**criteria, extra_cand: "None of the listed outcomes apply"}
        r["question"] = question
        r["meta"] = dict(r["meta"], catch_all_present=True, candidate_count=len(r["candidates"]))
        out.append(r)
    return out


def gen_level(level, n_target, rng, domains):
    rows = []
    while len(rows) < n_target:
        domain = rng.choice(domains)
        g = emit_group(LEVEL_BUILDERS[level](domain, rng, heldout_style=False))
        if not g:
            continue
        g = _apply_catch_all(g, rng)
        g = null_group(g, rng, NULL_GROUP_FRAC)
        rows += g
    return rows  # ponytail: may overshoot n_target by <= group size rather than cut a group in half


def gen_style_holdout(level, n_target, rng, domains):
    rows = []
    while len(rows) < n_target:
        domain = rng.choice(domains)
        g = emit_group(LEVEL_BUILDERS[level](domain, rng, heldout_style=True))
        if g:
            rows += g
    return rows  # ponytail: may overshoot n_target by <= group size rather than cut a group in half


def gen_level7(n_per_subtype, rng):
    rows = []
    for fn in L7_BUILDERS:
        got = 0
        while got < n_per_subtype:
            g = emit_group(fn(rng))
            if not g:
                continue
            if g[0]["meta"]["decision_type"] == "choice":
                g = _apply_catch_all(g, rng)
            rows += g
            got += len(g)
    return rows


def build_wh(args):
    out = Path(args.out_wh)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n = lambda k: args.limit or k  # noqa: E731
    manifest = {"levels": {}, "held_out_families": sorted(HELD_OUT_FAMILIES), "held_out_grammar": HELD_OUT_GRAMMAR,
                "held_out_styles": sorted(HELD_OUT_STYLES), "seed": args.seed, "limit": args.limit}

    train_pool, heldout_pool = [], []
    for level in range(1, 7):
        grammar_domain = HELD_OUT_GRAMMAR[level]
        train_domains = [d for d in TRAIN_DOMAINS if d != grammar_domain]
        rows = gen_level(level, n(10600), rng, train_domains)
        train_pool += rows
        grammar_rows = gen_level(level, n(140), rng, [grammar_domain])
        heldout_pool += grammar_rows
        manifest["levels"].setdefault(str(level), {})["train"] = len(rows)
        manifest["levels"][str(level)]["heldout_grammar"] = len(grammar_rows)
        print(f"  L{level}: {len(rows)} train rows, {len(grammar_rows)} held-out-grammar rows ({grammar_domain})")

    family_rows = []
    for level in range(1, 7):
        family_rows += gen_level(level, n(70), rng, list(HELD_OUT_FAMILIES))
    heldout_pool += family_rows

    style_rows = []
    for level in range(1, 7):
        style_rows += gen_style_holdout(level, n(70), rng, TRAIN_DOMAINS)
    heldout_pool += style_rows

    level7_rows = gen_level7(n(220), rng)
    heldout_pool += level7_rows

    print("\n=== split / null / catch-all / leak check ===")
    train_rows, val_rows = group_split(train_pool, min(3000, len(train_pool) // 4), rng)
    seen = {norm_text(r["state"]) for r in train_rows}
    # identity-based (not value-based) filtering: two unrelated rows can be structurally-equal dicts
    exclude_ids = {id(r) for r in family_rows} | {id(r) for r in style_rows} | {id(r) for r in level7_rows}
    evals = {
        "wh_heldout_family": family_rows,
        "wh_heldout_style": style_rows,
        "wh_level7": level7_rows,
        "wh_heldout_grammar": [r for r in heldout_pool if id(r) not in exclude_ids],
    }
    evals["wh_rubric_flip"] = flip_pairs(heldout_pool)
    evals["wh_rubric_shuffled"] = shuffled_rubric(heldout_pool, rng)

    all_rows = list(train_rows) + list(val_rows)
    for name, rows in evals.items():
        rows = [r for r in rows if norm_text(r["state"]) not in seen]
        if name != "wh_rubric_flip":
            rng.shuffle(rows)
        write_jsonl(out / "eval" / f"{name}.jsonl", rows)
        manifest[name] = len(rows)
        all_rows += rows
        print(f"  {name}: {len(rows)} rows")
    manifest["jevbench_states_checked"] = leak_check(all_rows, args.jevbench)

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    manifest["total_train"], manifest["total_val"] = len(train_rows), len(val_rows)
    by_level, by_type = defaultdict(int), defaultdict(int)
    for r in train_rows:
        by_level[r["meta"]["rule_depth"]] += 1
        by_type[r["meta"]["decision_type"]] += 1
    manifest["train_by_level"] = {str(k): v for k, v in sorted(by_level.items())}
    manifest["train_by_decision_type"] = dict(by_type)
    manifest["train_null_rows"] = sum(1 for r in train_rows if not r["meta"]["gold_present"])
    manifest["train_catch_all_rows"] = sum(1 for r in train_rows if r["meta"]["catch_all_present"])
    manifest["train_rubric_group_rows"] = sum(1 for r in train_rows if r["meta"].get("rubric_group"))
    manifest["flip_pairs"] = manifest["wh_rubric_flip"] // 2
    manifest["sources_and_licenses"] = {
        "decisionmix_v2_wh": "synthetic, generated by scripts/decisionmix_v2.py; no external data; repo-owned/MIT-equivalent"}
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWH TOTAL: {len(train_rows)} train / {len(val_rows)} val -> {out}/  "
          f"null share {manifest['train_null_rows'] / max(1, len(train_rows)):.3f}, "
          f"rubric-group share {manifest['train_rubric_group_rows'] / max(1, len(train_rows)):.3f}")
    if len(train_rows) < 60000 and not args.limit:
        print(f"WARNING: WH train rows {len(train_rows)} < 60k target")


# ============================================================================
# data_u — genuine probability targets. See module docstring for why UNLI-train/UNLI-test/
# chaos-mnli-ambiguity (fully claimed by pcdm/data.py's existing "unli"/"unli_test"/"chaos_mnli") are left
# untouched for training here.
# ============================================================================
NLI3 = ["entailment", "neutral", "contradiction"]


def u_row(decision_type, task, candidates, target, query, state, *, gold_source, source_dataset,
          requires_probability=True, extra_meta=None):
    return make_row(decision_type, task, candidates, target.index(max(target)), query, state,
                     gold_source=gold_source, soft_target=True, source_dataset=source_dataset, fam_bucket="U",
                     target=target, extra_meta={"requires_probability": requires_probability, **(extra_meta or {})})


def jsonl_rows(repo, path):
    with open(hf_hub_download(repo, path, repo_type="dataset")) as f:
        return [json.loads(l) for l in f if l.strip()]


def unli_rows(records, rng):
    out = []
    for r in records:
        p = float(r["label"])
        query = query_text({"type": "noul", "instructions": f"Is this true given the state? {r['hypothesis']}",
                            "criteria": {"true": "The hypothesis is true given the state.", "false": "The hypothesis is false given the state."}})
        out.append(u_row("noul", "u_unli", ["true", "false"], [p, 1 - p], query, r["premise"],
                         gold_source="human", source_dataset="Zhengping/UNLI", extra_meta={"task_family": "u_nli"}))
    return out


def ambient_rows(records, rng):
    out = []
    for r in records:
        labels = [x.strip() for x in r["labels"].split(",") if x.strip() in NLI3]
        if len(labels) < 2 or not (r.get("premise_ambiguous") or r.get("hypothesis_ambiguous")):
            continue
        target = [1.0 / len(labels) if c in labels else 0.0 for c in NLI3]
        query = query_text({"type": "choice", "instructions": f"What is the relation of the text to: {r['hypothesis']}",
                            "criteria": {"entailment": "the hypothesis follows from the text",
                                        "neutral": "the hypothesis is undetermined by the text",
                                        "contradiction": "the hypothesis contradicts the text"}})
        out.append(u_row("choice", "u_ambient", list(NLI3), target, query, r["premise"],
                         gold_source="human", source_dataset="metaeval/ambient", requires_probability=False,
                         extra_meta={"task_family": "u_nli", "ambiguous_label_count": len(labels)}))
    return out


def chaosnli_eval_rows(records):
    out = []
    for r in records:
        c = {"e": r["label_counter"].get("e", 0), "n": r["label_counter"].get("n", 0), "c": r["label_counter"].get("c", 0)}
        total = sum(c.values()) or 1
        target = [c["e"] / total, c["n"] / total, c["c"] / total]
        query = query_text({"type": "choice", "instructions": f"What is the relation of the text to: {r['hypothesis']}",
                            "criteria": {"entailment": "the hypothesis follows from the text",
                                        "neutral": "the hypothesis is undetermined by the text",
                                        "contradiction": "the hypothesis contradicts the text"}})
        out.append(u_row("choice", "u_chaosnli", list(NLI3), target, query, r["premise"],
                         gold_source="human", source_dataset="metaeval/chaos-mnli-ambiguity", requires_probability=False,
                         extra_meta={"task_family": "u_nli", "n_annotations": total}))
    return out


# Each synthetic generator takes a (lo, hi) parameter range; TRAIN_RANGES vs EVAL_RANGES are disjoint so the
# eval file probes generalisation to parameter values never seen in training (memo: "held-out parameter range").
def partial_evidence_row(rng, prior_range):
    n_required = rng.randint(2, 4)
    k_revealed = rng.randint(0, n_required - 1)
    revealed_all_true = rng.random() < 0.7
    state = []
    for i in range(k_revealed):
        val = revealed_all_true or rng.random() < 0.85  # mostly true when "revealed_all_true", else mostly a mix
        state.append(f"Condition {i + 1} of {n_required}: {'met' if val else 'not met'} (confirmed).")
        if not val:
            revealed_all_true = False
    hidden = n_required - k_revealed
    priors = [round(rng.uniform(*prior_range), 2) for _ in range(hidden)]
    for i, p in enumerate(priors):
        state.append(f"Condition {k_revealed + i + 1} of {n_required}: not yet checked; historically met in "
                     f"{int(p * 100)}% of comparable cases.")
    if not revealed_all_true:
        p_all = 0.0
    else:
        p_all = 1.0
        for p in priors:
            p_all *= p
    query = query_text({"type": "noul", "instructions": f"Will all {n_required} required conditions be met?",
                        "criteria": {"true": "All conditions are met.", "false": "At least one condition is not met."}})
    return u_row("noul", "u_partial_evidence", ["true", "false"], [p_all, 1 - p_all], query, render_state(state, rng),
                gold_source="programmatic", source_dataset="synthetic:decisionmix_v2_u:partial_evidence",
                extra_meta={"n_required": n_required, "k_revealed": k_revealed})


def noisy_sensor_row(rng, tpr_range):
    base_rate = round(rng.uniform(0.05, 0.5), 2)
    tpr = round(rng.uniform(*tpr_range), 2)
    fpr = round(rng.uniform(0.02, 0.2), 2)
    reading_positive = rng.random() < 0.5
    if reading_positive:
        num = tpr * base_rate
        denom = num + fpr * (1 - base_rate)
    else:
        num = (1 - tpr) * base_rate
        denom = num + (1 - fpr) * (1 - base_rate)
    p_true = num / denom if denom > 0 else base_rate
    state = [f"Base rate of a true incident in comparable cases is {int(base_rate * 100)}%.",
             f"The sensor has a {int(tpr * 100)}% true-positive rate and a {int(fpr * 100)}% false-positive rate.",
             f"The sensor reading is {'positive' if reading_positive else 'negative'}."]
    query = query_text({"type": "noul", "instructions": "Given the sensor reading and its known error rates, is the incident real?",
                        "criteria": {"true": "The incident is real.", "false": "The incident is not real (a sensor error)."}})
    return u_row("noul", "u_noisy_sensor", ["true", "false"], [p_true, 1 - p_true], query, render_state(state, rng),
                gold_source="programmatic", source_dataset="synthetic:decisionmix_v2_u:noisy_sensor",
                extra_meta={"base_rate": base_rate, "tpr": tpr, "fpr": fpr})


def ordinal_confusion_row(rng, eps_range):
    k = rng.choice([3, 4])
    eps = round(rng.uniform(*eps_range), 2)
    true_level = rng.randrange(k)
    # confusion row: P(obs=j | true=i) -- diagonal 1-eps, remainder split over the other k-1 levels
    off = eps / (k - 1)
    obs = rng.choices(range(k), weights=[1 - eps if j == true_level else off for j in range(k)])[0]
    # posterior P(true=i | obs) with a uniform prior over true levels (Bayes, since we don't reveal true_level)
    likelihoods = [(1 - eps if i == obs else off) for i in range(k)]
    s = sum(likelihoods)
    posterior = [x / s for x in likelihoods]
    names, _ = level_names(k, rng)
    state = [f"Observed level: {names[obs]}.", f"The observation channel is correct {int((1 - eps) * 100)}% of the time "
             f"and otherwise reports an adjacent-or-other level uniformly at random."]
    query = query_text({"type": "score", "instructions": "What is the true level, given only the (possibly noisy) observed level?",
                        "criteria": list(names)})
    return u_row("score", "u_ordinal_confusion", list(names), posterior, query, render_state(state, rng),
                gold_source="programmatic", source_dataset="synthetic:decisionmix_v2_u:ordinal_confusion",
                extra_meta={"eps": eps, "k": k, "observed": obs})


def conflicting_sources_row(rng, reliability_range):
    n = rng.choice([2, 3])
    sources = [(rng.random() < 0.5, round(rng.uniform(*reliability_range), 2)) for _ in range(n)]
    odds = 1.0
    for verdict, r in sources:
        odds *= (r / (1 - r)) if verdict else ((1 - r) / r)
    p_true = odds / (1 + odds)
    state = [f"Source {i + 1} reports {'true' if v else 'false'} and is correct {int(r * 100)}% of the time."
             for i, (v, r) in enumerate(sources)]
    query = query_text({"type": "noul", "instructions": "Weighing the sources by their stated reliability, is the claim true?",
                        "criteria": {"true": "The claim is true.", "false": "The claim is false."}})
    return u_row("noul", "u_conflicting_sources", ["true", "false"], [p_true, 1 - p_true], query, render_state(state, rng),
                gold_source="programmatic", source_dataset="synthetic:decisionmix_v2_u:conflicting_sources",
                extra_meta={"n_sources": n})


SYNTH_GENERATORS = {
    "partial_evidence": (partial_evidence_row, (0.55, 0.95), (0.30, 0.55)),
    "noisy_sensor": (noisy_sensor_row, (0.75, 0.97), (0.5, 0.75)),
    "ordinal_confusion": (ordinal_confusion_row, (0.05, 0.25), (0.25, 0.45)),
    "conflicting_sources": (conflicting_sources_row, (0.65, 0.95), (0.5, 0.65)),
}


def build_u(args):
    out = Path(args.out_u)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n = lambda k: args.limit or k  # noqa: E731
    manifest = {"seed": args.seed, "limit": args.limit, "sources_and_licenses": {
        "Zhengping/UNLI": "validation split only (train/test are already consumed by pcdm/data.py's unli/unli_test) -- academic use, see repo",
        "metaeval/ambient": "AmbiEnt ambiguity sets (arXiv:2304.14399); no license field on the HF mirror -- check jjessyli/ambient upstream before redistribution",
        "metaeval/chaos-mnli-ambiguity": "eval-only (already pcdm/data.py's chaos_mnli eval); no license field on the HF mirror -- check the original ChaosNLI release upstream",
        "synthetic:decisionmix_v2_u:*": "programmatic, generated by scripts/decisionmix_v2.py; exact closed-form posteriors",
    }}

    print("fetching UNLI validation, AmbiEnt, chaos-mnli-ambiguity (HF)...")
    unli_val = jsonl_rows("Zhengping/UNLI", "validation.jsonl")[:n(3040)]
    ambient_raw = jsonl_rows("metaeval/ambient", "dev.jsonl") + jsonl_rows("metaeval/ambient", "test.jsonl")
    chaos_raw = jsonl_rows("metaeval/chaos-mnli-ambiguity", "chaos_mnli.jsonl")[:n(1599)]
    rng.shuffle(unli_val)
    rng.shuffle(ambient_raw)

    unli_all = unli_rows(unli_val, rng)
    ambient_all = ambient_rows(ambient_raw, rng)[:n(1200)]
    manifest["unli_raw"], manifest["ambient_ambiguous_raw"] = len(unli_all), len(ambient_all)

    unli_train, unli_eval = unli_all[: n(2200)], unli_all[n(2200):n(2200) + n(500)]
    ambient_train, ambient_eval = ambient_all[: len(ambient_all) - n(150)], ambient_all[len(ambient_all) - n(150):]

    real_train = unli_train + ambient_train
    real_val_pool = unli_eval + ambient_eval  # held-out real-source slice: goes to eval, not val

    synth_train, synth_eval = [], []
    per_gen_train = n(7600)
    for name, (fn, train_range, eval_range) in SYNTH_GENERATORS.items():
        tr = [fn(rng, train_range) for _ in range(per_gen_train)]
        ev = [fn(rng, eval_range) for _ in range(n(120))]
        for r in tr + ev:
            r["meta"]["source_dataset"] = f"synthetic:decisionmix_v2_u:{name}"
        synth_train += tr
        synth_eval += ev
        manifest[f"synth_{name}_train"] = len(tr)
        print(f"  synthetic {name}: {len(tr)} train, {len(ev)} held-out-range eval")

    train_pool = real_train + synth_train
    train_rows, val_rows = group_split(train_pool, min(2000, len(train_pool) // 4), rng)
    seen = {norm_text(r["state"]) for r in train_rows}

    evals = {
        "u_chaosnli": chaosnli_eval_rows(chaos_raw),
        "u_real_heldout": real_val_pool,
        "u_synthetic_heldout": synth_eval,
    }
    all_rows = list(train_rows) + list(val_rows)
    for name, rows in evals.items():
        rows = [r for r in rows if norm_text(r["state"]) not in seen]
        rng.shuffle(rows)
        write_jsonl(out / "eval" / f"{name}.jsonl", rows)
        manifest[name] = len(rows)
        all_rows += rows
        print(f"  {name}: {len(rows)} rows")
    manifest["jevbench_states_checked"] = leak_check(all_rows, args.jevbench)

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    manifest["total_train"], manifest["total_val"] = len(train_rows), len(val_rows)
    by_type, by_source = defaultdict(int), defaultdict(int)
    for r in train_rows:
        by_type[r["meta"]["decision_type"]] += 1
        by_source[r["meta"]["source_dataset"].split(":")[0]] += 1
    manifest["train_by_decision_type"] = dict(by_type)
    manifest["train_by_source"] = dict(by_source)
    manifest["train_soft_target_rows"] = sum(1 for r in train_rows if r["meta"]["soft_target"])
    for r in train_rows:
        assert abs(sum(r["target"]) - 1.0) < 1e-6, r
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nU TOTAL: {len(train_rows)} train / {len(val_rows)} val -> {out}/  "
          f"soft-target share {manifest['train_soft_target_rows'] / max(1, len(train_rows)):.3f}")
    if len(train_rows) < 30000 and not args.limit:
        print(f"WARNING: U train rows {len(train_rows)} < 30k target")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_wh", default="data_wh")
    ap.add_argument("--out_u", default="data_u")
    ap.add_argument("--corpus", choices=["wh", "u", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="cap every generator at N rows (fast smoke check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jevbench", default="/tmp/jevbench/datasets/public")
    args = ap.parse_args()
    if args.corpus in ("wh", "both"):
        print("=== data_wh ===")
        build_wh(args)
    if args.corpus in ("u", "both"):
        print("\n=== data_u ===")
        build_u(args)


if __name__ == "__main__":
    main()
