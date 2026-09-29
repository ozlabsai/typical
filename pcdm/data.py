"""HF datasets -> DecisionExample JSONL (train/val/eval). Run: uv run pcdm/data.py [--small] [--selftest]

Schema: one JSON object per line with state, query, candidates, target (sums to 1
among candidates), p_null, task, label, plus optional meta (hyp/question/family)
used by baselines' prompt builders. See PLAN2.md "Data" for the full spec.
"""
import argparse
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

SEED = 0
DATA = Path("data")
MAX_STATE = 1200
CAND3 = ("entailment", "neutral", "contradiction")
# Held-out label wordings, never used in training; snli_test_paraphrase samples one set per item.
PARAPHRASES = [
    ("the hypothesis follows from the text", "the hypothesis is undetermined by the text", "the hypothesis contradicts the text"),
    ("implied", "not enough information", "ruled out"),
    ("definitely true", "could be either", "definitely false"),
    ("consistent and implied", "neither implied nor denied", "denied"),
    ("supported", "unrelated", "incompatible"),
]
# Training-time label paraphrases so the scorer must read the candidate text
# instead of memorising three vectors. PARAPHRASES above is never used in training.
TRAIN_NAMES = [CAND3] * 4 + [
    ("entails", "undetermined", "contradicts"),
    ("true", "unknown", "false"),
    ("yes", "maybe", "no"),
    ("the text supports the claim", "the text does not say", "the text refutes the claim"),
]
# Exact-string guard: SQuAD answers are free text pulled straight from a passage and can
# coincidentally equal one of the held-out PARAPHRASES words (e.g. a real answer "supported").
# That's not an actual leak of the paraphrase wording scheme, but the audit can't tell the
# difference, so any candidate that collides verbatim is dropped from training.
PARAPHRASE_WORDS = {w.strip().lower() for tup in PARAPHRASES for w in tup}
SNLI_ZIP_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
BOOLQ_PAIRS = [("yes", "no"), ("true", "false"), ("correct", "incorrect"),
               ("the passage supports this", "the passage does not support this")]
NG20_NAMES = {
    "alt.atheism": "atheism", "comp.graphics": "computer graphics", "comp.os.ms-windows.misc": "windows os",
    "comp.sys.ibm.pc.hardware": "pc hardware", "comp.sys.mac.hardware": "mac hardware", "comp.windows.x": "x windows",
    "misc.forsale": "for sale", "rec.autos": "cars", "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball", "rec.sport.hockey": "hockey", "sci.crypt": "cryptography",
    "sci.electronics": "electronics", "sci.med": "medicine", "sci.space": "space",
    "soc.religion.christian": "christianity", "talk.politics.guns": "gun politics",
    "talk.politics.mideast": "middle east politics", "talk.politics.misc": "politics",
    "talk.religion.misc": "religion",
}

# ---- query templates: [0] = v0 wording (eval), [1-7] train, [8-9] *_qpara eval only ----
TEMPLATES = {
    "nli": [
        "Relation of the text to: {hyp}", "How does the text relate to: {hyp}",
        "Given the text, what about: {hyp}", "Does the text support this: {hyp}",
        "Text vs. claim: {hyp}", "Considering the text, is this so: {hyp}",
        "What's the connection between the text and: {hyp}", "The text and this statement: {hyp}",
        "With respect to the passage, this idea: {hyp}", "Reading the text above, this notion: {hyp}",
    ],
    "prop": [
        "Is this true given the text? {hyp}", "Given the text, how likely is this? {hyp}",
        "Based on the text, is this so? {hyp}", "Does the text make this true? {hyp}",
        "How true is this given the text? {hyp}", "From the text, is the following the case? {hyp}",
        "Judging by the text, is this accurate? {hyp}", "Does the passage confirm this? {hyp}",
        "Weighing the text, would you say this holds? {hyp}", "In light of the text, is this claim correct? {hyp}",
    ],
    "intent": [
        "What is the user's intent?", "What does the user want?", "What is the user trying to do?",
        "Which intent matches this utterance?", "What action is being requested?",
        "What is this request about?", "Identify the user's goal.", "What task does the user want done?",
        "What is the speaker asking for?", "Classify the intent of this message.",
    ],
    "topic": [
        "What is the topic of this text?", "Which category best fits this text?",
        "What subject does this text cover?", "How would you classify this text?",
        "What is this passage about?", "Pick the topic that matches this text.",
        "What section would this article belong to?", "What is the main subject here?",
        "Under which heading does this text fall?", "What genre or category is this?",
    ],
    "emotion": [
        "What is the emotion expressed?", "What sentiment is being expressed?",
        "How does the speaker feel?", "What is the emotional tone of this text?",
        "Which feeling best matches this text?", "What mood does this text convey?",
        "Identify the emotion in this text.", "What is the writer feeling?",
        "Classify the sentiment of this text.", "What emotion does this express?",
    ],
    "qa": [
        "Answer the question: {q}", "Question: {q}", "Given the passage, answer: {q}",
        "What is the answer to: {q}", "Based on the text, answer: {q}",
        "Find the answer to this question: {q}", "According to the passage: {q}",
        "Using the text above, answer: {q}", "Reading the passage, respond to: {q}",
        "From the given text, answer: {q}",
    ],
    "boolq": [
        "Question: {q}", "Answer yes or no: {q}", "Based on the passage: {q}",
        "Is this correct? {q}", "According to the text: {q}", "Given the passage, answer: {q}",
        "True or false: {q}", "From the text: {q}", "Reading above, answer: {q}",
        "Does the passage say: {q}",
    ],
}

FAMILY = {
    "snli": "nli", "mnli": "nli", "anli": "nli", "snli_soft": "nli",
    "snli_test": "nli", "snli_test_paraphrase": "nli", "snli_null": "nli",
    "snli_test_soft": "nli", "snli_test_qpara": "nli", "snli_test_hyponly": "nli", "mnli_val": "nli",
    "chaos_mnli": "nli", "anli_test": "nli", "snli_noev": "nli", "mnli_noev": "nli", "cse_snli": "nli",
    "unli": "prop", "unli_test": "prop", "boolq_prop": "prop",
    "boolq_pair": "boolq", "boolq_val": "boolq",
    "squad": "qa", "squad_null": "qa", "null_irrq_squad": "qa", "mmlu_pro": "qa", "mmlu_choicesonly": "qa", "mmlu_shuffledq": "qa",  # eval-only set (scripts/mmlu_pro_eval.py)
    "clinc": "intent", "clinc_test": "intent", "clinc_oos": "intent",
    "clinc_heldout": "intent", "clinc_k": "intent",
    "hwu64": "intent", "hwu64_test": "intent", "snips": "intent",
    "massive": "intent", "massive_he": "intent", "mtop": "intent",
    "banking77_test": "intent", "banking77_k": "intent",
    "cse_clinc": "intent", "cse_hwu64": "intent", "cse_banking77": "intent",
    "ksweep_clinc": "intent", "ksweep_banking77": "intent",
    "null_irrq_intent": "intent",
    "null_nearmiss_clinc": "intent", "null_nearmiss_hwu64": "intent", "null_nearmiss_banking77": "intent",
    "trec_coarse": "topic", "trec_fine": "topic", "ng20_test": "topic",
    "ag_news": "topic", "dbpedia": "topic", "yahoo": "topic",
    "go_emotions": "emotion", "tweet_emotion": "emotion", "tweet_sentiment": "emotion",
    "tweet_hate": "emotion", "tweet_irony": "emotion", "tweet_offensive": "emotion",
    "dair_emotion": "emotion", "sst5": "emotion",
}
# Training-label strings from other families, used as distractors in cse_* "add_irr" rows
# and the cross-family listwise augmentation (never drawn from a held-out eval space).
IRRELEVANT = ["world", "sports", "business", "sadness", "joy", "anger", "film", "album", "village", "animal"]
# ~20 function words; siblings() ignores these so label pairs like "cancel reservation" /
# "make reservation" count as siblings on "reservation", not on nothing.
STOP = {"a", "an", "the", "of", "to", "for", "and", "or", "is", "are", "in", "on", "at",
        "with", "this", "that", "do", "does", "not", "my"}

# ~130-entry hand-written synonym table for label-paraphrase (word -> alternatives).
SYNONYMS = {
    "yes": ["yeah", "correct", "affirmative"], "no": ["nope", "negative", "incorrect"],
    "true": ["correct", "accurate", "right"], "false": ["incorrect", "wrong", "untrue"],
    "happy": ["glad", "joyful", "pleased"], "sad": ["unhappy", "sorrowful", "down"],
    "angry": ["mad", "furious", "irritated"], "afraid": ["scared", "fearful", "frightened"],
    "surprised": ["astonished", "amazed", "shocked"], "love": ["adore", "affection", "fondness"],
    "joy": ["happiness", "delight", "elation"], "fear": ["dread", "fright", "terror"],
    "anger": ["rage", "fury", "irritation"], "disgust": ["revulsion", "distaste", "aversion"],
    "sadness": ["sorrow", "grief", "unhappiness"], "excitement": ["thrill", "enthusiasm", "exhilaration"],
    "confusion": ["bewilderment", "puzzlement", "perplexity"], "curiosity": ["interest", "inquisitiveness"],
    "gratitude": ["thankfulness", "appreciation"], "pride": ["satisfaction", "self-esteem"],
    "relief": ["ease", "comfort"], "remorse": ["regret", "guilt"], "optimism": ["hopefulness", "positivity"],
    "admiration": ["respect", "esteem"], "amusement": ["entertainment", "fun"], "approval": ["endorsement", "agreement"],
    "disapproval": ["disagreement", "rejection"], "embarrassment": ["awkwardness", "shame"],
    "nervousness": ["anxiety", "unease"], "realization": ["awareness", "recognition"],
    "desire": ["want", "wish", "craving"], "caring": ["concern", "compassion"],
    "annoyance": ["irritation", "frustration"], "grief": ["sorrow", "mourning"],
    "positive": ["good", "favorable", "upbeat"], "negative": ["bad", "unfavorable", "downbeat"],
    "neutral": ["indifferent", "impartial"], "hate": ["hatred", "loathing", "contempt"],
    "irony": ["sarcasm", "mockery"], "offensive": ["insulting", "rude", "abusive"],
    "world": ["global", "international", "worldwide"], "sports": ["athletics", "games"],
    "business": ["commerce", "finance", "trade"], "science": ["research", "sci"],
    "technology": ["tech", "engineering"], "health": ["wellness", "medical"],
    "education": ["schooling", "learning"], "politics": ["government", "policy"],
    "entertainment": ["media", "showbiz"], "family": ["household", "relatives"],
    "relationships": ["relations", "bonds"], "computers": ["computing", "pcs"],
    "internet": ["web", "online"], "music": ["songs", "audio"],
    "movies": ["films", "cinema"], "books": ["literature", "novels"],
    "weather": ["climate", "forecast"], "alarm": ["reminder", "alert"],
    "reminder": ["alert", "notification"], "calendar": ["schedule", "agenda"],
    "cancel": ["stop", "abort", "revoke"], "confirm": ["verify", "approve"],
    "request": ["ask", "query"], "order": ["purchase", "buy"],
    "payment": ["transaction", "pay"], "transfer": ["move", "send"],
    "balance": ["amount", "funds"], "account": ["profile", "membership"],
    "card": ["credit card", "debit card"], "restaurant": ["diner", "eatery"],
    "reservation": ["booking", "appointment"], "flight": ["plane trip", "air travel"],
    "hotel": ["lodging", "inn"], "taxi": ["cab", "ride"],
    "directions": ["navigation", "route"], "translate": ["interpret", "convert"],
    "define": ["explain", "describe"], "spell": ["letter", "spell out"],
    "joke": ["gag", "quip"], "greeting": ["hello", "salutation"],
    "goodbye": ["farewell", "bye"], "thanks": ["gratitude", "appreciation"],
    "complaint": ["grievance", "objection"], "feedback": ["review", "comment"],
    "support": ["help", "assistance"], "insurance": ["coverage", "policy"],
    "traffic": ["congestion", "road conditions"], "recipe": ["dish instructions", "cooking guide"],
    "meaning": ["definition", "sense"], "conversion": ["change", "transformation"],
    "distance": ["length", "range"], "measurement": ["measure", "metric"],
    "government": ["administration", "state"], "athletics": ["sports", "competitions"],
    "finance": ["money matters", "economics"], "mathematics": ["math", "arithmetic"],
    "reference": ["resource", "lookup"], "culture": ["society", "customs"],
    "society": ["community", "public"], "animal": ["creature", "beast"],
    "plant": ["vegetation", "flora"], "building": ["structure", "edifice"],
    "village": ["hamlet", "settlement"], "album": ["record", "lp"],
    "film": ["movie", "picture"], "company": ["firm", "corporation"],
    "artist": ["performer", "creator"], "athlete": ["sportsperson", "player"],
    "vehicle": ["transport", "automobile"], "transportation": ["transport", "travel"],
    "institution": ["organization", "establishment"], "correct": ["right", "accurate"],
    "incorrect": ["wrong", "inaccurate"], "supports": ["confirms", "backs"],
    "refutes": ["contradicts", "denies"], "unknown": ["undetermined", "uncertain"],
    "claim": ["assertion", "statement"], "text": ["passage", "excerpt"],
    "passage": ["text", "excerpt"], "question": ["query", "inquiry"],
    "answer": ["reply", "response"], "topic": ["subject", "theme"],
    "category": ["class", "group"], "intent": ["goal", "purpose"],
    "sentiment": ["feeling", "mood"], "emotion": ["feeling", "mood"],
    "message": ["note", "text"], "play": ["start", "run"], "stop": ["halt", "end"],
    "add": ["insert", "include"], "remove": ["delete", "take out"],
    "search": ["look up", "find"], "check": ["verify", "inspect"],
    "set": ["schedule", "configure"], "volume": ["loudness", "level"],
}


def trunc(s, n=MAX_STATE):
    return s.strip()[:n]


def norm_text(s):
    return " ".join(s.lower().split())


NO_STATE_TASKS = {"snli_test_hyponly", "mmlu_choicesonly"}  # deliberately blanked-out state (1-char), never filtered out below


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # empty states (e.g. bodiless 20NG posts) -> NaN logits; hyponly/no-evidence probes
    # deliberately use a 1-char state ("."), so exempt those tasks by name/suffix.
    rows = [r for r in rows if len(r["state"].strip()) >= 3
            or r["task"] in NO_STATE_TASKS or r["task"].endswith("_noev")]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pair_key(premise, hyp):
    return norm_text(premise), norm_text(hyp)


def render(family, idx, **kw):
    return TEMPLATES[family][idx].format(**kw)


def words(name):
    """snake_case / CamelCase / ALLCAPS -> lowercase words."""
    s = str(name).replace("_", " ").replace(".", " ").replace("/", " ")
    if " " in s or s.isupper() or s.islower():
        return " ".join(s.split()).lower()
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s).lower()


def style_names(names, rng):
    """Per-example label paraphrase: one style for ALL candidates (0.5 raw / 0.2 title / 0.3 synonym)."""
    r = rng.random()
    if r < 0.5:
        return list(names)
    if r < 0.7:
        return [n.title() for n in names]

    def swap(n):
        return " ".join(rng.choice(SYNONYMS[w]) if w in SYNONYMS and rng.random() < 0.5 else w for w in n.split())

    return [swap(n) for n in names]


def null_synth(dist3, gold_idx, rng, p_keep3, p_drop_non):
    """K-reduction + null synthesis over a 3-way distribution."""
    r = rng.random()
    if r < p_keep3:
        idx, p_null = [0, 1, 2], 0.0
    elif r < p_keep3 + p_drop_non:
        others = [i for i in range(3) if i != gold_idx]
        drop = rng.choice(others)
        idx, p_null = [i for i in range(3) if i != drop], 0.0
    else:
        idx, p_null = [i for i in range(3) if i != gold_idx], 1.0
    sub = [dist3[i] for i in idx]
    s = sum(sub)
    target = [t / s for t in sub] if s > 0 else [1.0 / len(idx)] * len(idx)
    return idx, target, p_null


def nli_synth(dist3, gold_idx, rng):
    """Train-time K/null scheme for NLI (data v4): K in {2,3} each w.p. 0.5. K=3 keeps all three
    labels and is never null (there's nothing left to drop the gold to). K=2 drops one of the two
    non-gold labels 0.85 of the time (gold present) and drops the gold itself 0.15 of the time
    (gold-absent, p_null=1) -- so all of NLI's null mass sits at K=2, none at K=3 (documented in
    REVIEW.md item 6: NLI only has 3 labels, so K=3 gold-absent is not constructible)."""
    if rng.random() < 0.5:
        return [0, 1, 2], list(dist3), 0.0
    others = [i for i in range(3) if i != gold_idx]
    if rng.random() < 0.15:
        idx, p_null = others, 1.0
    else:
        drop = rng.choice(others)
        idx, p_null = [i for i in range(3) if i != drop], 0.0
    sub = [dist3[i] for i in idx]
    s = sum(sub)
    target = [t / s for t in sub] if s > 0 else [1.0 / len(idx)] * len(idx)
    return idx, target, p_null


def nli_examples(rows, task, rng, cand_names=CAND3, mode="synth", label_none=False, qidx=(1, 7)):
    """rows: iterable of (premise, hyp, dist3, gold_idx).
    mode: "synth" (data v4 K/null scheme, see nli_synth), "k2" (0/0.50/0.50, K=2), "keep" (K=3).
    qidx: fixed template index (int, e.g. 0 for eval) or an inclusive (lo, hi) range to sample per row.
    """
    examples, pairs = [], set()
    for premise, hyp, dist, gold_idx in rows:
        names = rng.choice(TRAIN_NAMES) if mode == "synth" else rng.choice(cand_names) if isinstance(cand_names, list) else cand_names
        if mode == "keep":
            idx, target, p_null = [0, 1, 2], dist, 0.0
        elif mode == "synth":
            idx, target, p_null = nli_synth(dist, gold_idx, rng)
        else:
            idx, target, p_null = null_synth(dist, gold_idx, rng, 0.0, 0.5)
        if label_none:
            label = None
        elif p_null >= 1.0 or gold_idx not in idx:
            label = -1
        else:
            label = idx.index(gold_idx)
        qi = qidx if isinstance(qidx, int) else rng.randint(*qidx)
        examples.append({
            "state": trunc(premise),
            "query": render("nli", qi, hyp=hyp.strip()),
            "candidates": [names[i] for i in idx],
            "target": [float(t) for t in target],
            "p_null": float(p_null),
            "task": task,
            "label": label,
            "meta": {"hyp": hyp.strip(), "family": "nli"},
        })
        pairs.add(pair_key(premise, hyp))
    return examples, pairs


def unli_row(premise, hyp, p, task, rng, qidx=(1, 7)):
    qi = qidx if isinstance(qidx, int) else rng.randint(*qidx)
    return {
        "state": trunc(premise),
        "query": render("prop", qi, hyp=hyp.strip()),
        "candidates": ["yes"],
        "target": [1.0],
        "p_null": 1.0 - float(p),
        "task": task,
        "label": None,
        "meta": {"hyp": hyp.strip(), "family": "prop"},
    }


def siblings(names):
    """Map label index -> indices of labels sharing >=1 non-stopword token, e.g. "cancel
    reservation" / "make reservation" are siblings on "reservation". Used for near-miss
    nulls (item 2): a sibling in a gold-absent candidate set is a much harder negative
    than a random label."""
    toksets = [set(words(n).split()) - STOP for n in names]
    sib = defaultdict(list)
    for i, ti in enumerate(toksets):
        if not ti:
            continue
        for j, tj in enumerate(toksets):
            if i != j and tj and (ti & tj):
                sib[i].append(j)
    return sib


def build_sibling_map(names, tag):
    sib = siblings(names)
    covered = sum(1 for i in range(len(names)) if sib.get(i))
    print(f"  siblings[{tag}]: {covered}/{len(names)} labels have >=1 sibling")
    return sib


def pick_labels(gold_id, pool, k, include_gold, rng, sibling_map=None):
    """sibling_map (train only, see build_sibling_map): in the gold-absent branch, w.p. 0.5
    force one sibling of the gold label into the candidate set (a near-miss null instead of
    K random unrelated labels)."""
    others = [i for i in pool if i != gold_id]
    if include_gold:
        k = min(k, len(pool))
        idx = rng.sample(others, min(k - 1, len(others))) + [gold_id]
        rng.shuffle(idx)
        target = [1.0 if i == gold_id else 0.0 for i in idx]
        return idx, target, 0.0, idx.index(gold_id)
    k = min(k, len(others))
    idx = rng.sample(others, k)
    if sibling_map and idx and rng.random() < 0.5:
        sibs = [s for s in sibling_map.get(gold_id, []) if s in others and s not in idx]
        if sibs:
            idx[rng.randrange(len(idx))] = rng.choice(sibs)
    return idx, [1.0 / len(idx)] * len(idx), 1.0, -1


def k_null_choice(rng, N):
    """Data v4 K/null scheme (decouples K from p_null, REVIEW.md item 6): K from
    {3, 10, min(N,50), N} w.p. {0.15, 0.30, 0.20, 0.35}; gold-absent chosen INDEPENDENTLY
    w.p. 0.15 at every K, including K=N (gold-absent at K=N is "all labels but the gold",
    a near-miss full set -- balanced by the K=N gold-present rows at the same rate)."""
    k = rng.choices([3, 10, min(N, 50), N], weights=[0.15, 0.30, 0.20, 0.35])[0]
    return k, rng.random() >= 0.15


cls_k_choice = k_null_choice  # classification default k_choice; CLINC's in_scope_k reuses the same scheme


def leaks_paraphrase(cands):
    return any(c.strip().lower() in PARAPHRASE_WORDS for c in cands)


def cls_examples(rows, task, family, names, rng, k_choice=None, qidx=(1, 7), paraphrase=True, sibling_null=False):
    """rows: iterable of (text, gold_id) into a full 0..len(names)-1 label space.
    sibling_null: train only (see pick_labels) -- eval-set callers leave this False so their
    RNG draws (and hence byte-identical eval files) are unaffected."""
    pool = list(range(len(names)))
    k_choice = k_choice or cls_k_choice
    sib = build_sibling_map(names, task) if sibling_null else None
    out = []
    for text, gold in rows:
        k, include_gold = k_choice(rng, len(names))
        idx, target, p_null, label = pick_labels(gold, pool, k, include_gold, rng, sib)
        cnames = [names[i] for i in idx]
        cnames = style_names(cnames, rng) if paraphrase else cnames
        if leaks_paraphrase(cnames):
            continue
        qi = qidx if isinstance(qidx, int) else rng.randint(*qidx)
        out.append({
            "state": trunc(text), "query": render(family, qi), "candidates": cnames,
            "target": target, "p_null": p_null, "task": task, "label": label,
            "meta": {"family": family},
        })
    return out


def clinc_examples(rows, task, pool, k_choice, id2name, rng, qidx=(1, 7), paraphrase=True, sibling_null=False):
    """CLINC-shaped: pool is a subset of a larger id2name space (held-out intents excluded).
    sibling_null: see cls_examples."""
    sib = build_sibling_map(id2name, task) if sibling_null else None
    out = []
    for text, gold_id in rows:
        k, include_gold = k_choice(rng)
        idx, target, p_null, label = pick_labels(gold_id, pool, k, include_gold, rng, sib)
        names = [id2name[i] for i in idx]
        cnames = style_names(names, rng) if paraphrase else names
        if leaks_paraphrase(cnames):
            continue
        qi = qidx if isinstance(qidx, int) else rng.randint(*qidx)
        out.append({
            "state": trunc(text), "query": render("intent", qi), "candidates": cnames,
            "target": target, "p_null": p_null, "task": task, "label": label,
            "meta": {"family": "intent"},
        })
    return out


def boolq_examples(raw, task, rng, mode="train"):
    """mode "train": 50% K=1 prop ("yes", p_null=1-answer) / 50% K=2 binary pair.
    mode "eval": always K=2 with ("yes","no"), template 0."""
    out = []
    for r in raw:
        passage, question, answer = r["passage"], r["question"], bool(r["answer"])
        if mode == "train" and rng.random() < 0.5:
            qi = rng.randint(1, 7)
            out.append({
                "state": trunc(passage), "query": render("prop", qi, hyp=question),
                "candidates": ["yes"], "target": [1.0], "p_null": 1.0 - float(answer),
                "task": f"{task}_prop", "label": None, "meta": {"question": question, "family": "prop"},
            })
            continue
        names = rng.choice(BOOLQ_PAIRS) if mode == "train" else ("yes", "no")
        gold = 0 if answer else 1
        idx = [0, 1]
        rng.shuffle(idx)
        cnames = [names[i] for i in idx]
        label = idx.index(gold)
        target = [1.0 if i == label else 0.0 for i in range(2)]
        qi = rng.randint(1, 7) if mode == "train" else 0
        out.append({
            "state": trunc(passage), "query": render("boolq", qi, q=question),
            "candidates": cnames, "target": target, "p_null": 0.0,
            "task": f"{task}_pair" if mode == "train" else task, "label": label,
            "meta": {"question": question, "family": "boolq"},
        })
    return out


def qa_examples(target_rows, context_rows, task, rng, qidx=(1, 7)):
    """SQuAD v2 shaped: candidates = gold + 3 distractors (answerable) or 4 (unanswerable),
    drawn from other questions on the same title, falling back to a global answer pool."""
    by_title = defaultdict(list)
    for r in context_rows:
        if r["answers"]["text"]:
            by_title[r["title"]].append(r["answers"]["text"][0])
    global_pool = [a for v in by_title.values() for a in v] or ["yes"]

    def distractors(title, gold, n):
        pool = [a for a in by_title.get(title, []) if a != gold]
        rng.shuffle(pool)
        picked = pool[:n]
        if len(picked) < n:
            extra = [a for a in global_pool if a != gold and a not in picked]
            picked += rng.sample(extra, min(n - len(picked), len(extra)))
        return picked[:n]

    out = []
    for r in target_rows:
        ans, starts = r["answers"]["text"], r["answers"]["answer_start"]
        answerable = bool(ans) and starts[0] < 900
        qi = qidx if isinstance(qidx, int) else rng.randint(*qidx)
        query = render("qa", qi, q=r["question"])
        if answerable:
            gold = ans[0]
            cands = distractors(r["title"], gold, 3) + [gold]
            order = list(range(len(cands)))
            rng.shuffle(order)
            cands = [cands[i] for i in order]
            label = cands.index(gold)
            target = [1.0 if i == label else 0.0 for i in range(len(cands))]
            p_null = 0.0
        else:
            cands = distractors(r["title"], None, 4)
            label, p_null = -1, 1.0
            target = [1.0 / len(cands)] * len(cands) if cands else []
        if not cands or leaks_paraphrase(cands):
            continue
        out.append({
            "state": trunc(r["context"], 1000), "query": query, "candidates": cands,
            "target": target, "p_null": p_null, "task": task, "label": label,
            "meta": {"question": r["question"], "family": "qa"},
        })
    return out


def load_snli_votes(split):
    """5-vote annotator label distributions from the official SNLI zip; {} if unavailable."""
    zpath = DATA / "snli_1.0.zip"
    if not zpath.exists():
        try:
            DATA.mkdir(exist_ok=True)
            urllib.request.urlretrieve(SNLI_ZIP_URL, zpath)
        except Exception as e:
            print(f"WARNING: SNLI zip download failed ({e}); skipping soft-label sets")
            return {}
    try:
        votes = {}
        with zipfile.ZipFile(zpath) as z, z.open(f"snli_1.0/snli_1.0_{split}.jsonl") as f:
            for line in f:
                d = json.loads(line)
                labels = [l for l in d["annotator_labels"] if l in CAND3]
                if not labels:
                    continue
                n = len(labels)
                dist = [labels.count(c) / n for c in CAND3]
                votes[pair_key(d["sentence1"], d["sentence2"])] = dist
        return votes
    except Exception as e:
        print(f"WARNING: reading SNLI zip failed ({e}); skipping soft-label sets")
        return {}


def summarize(paths):
    print(f"\n{'file':<32}{'count':>8}{'mean K':>10}{'frac p_null=1':>16}")
    for path in paths:
        rows = [json.loads(l) for l in open(path)]
        n = len(rows)
        mk = sum(len(r["candidates"]) for r in rows) / n if n else 0.0
        fn = sum(1 for r in rows if r["p_null"] >= 1.0) / n if n else 0.0
        print(f"{str(path):<32}{n:>8}{mk:>10.2f}{fn:>16.3f}")


def group_split(examples, val_size, rng):
    """Split so no normalised state string is shared between train and val."""
    groups = defaultdict(list)
    for e in examples:
        groups[norm_text(e["state"])].append(e)
    keys = list(groups.keys())
    rng.shuffle(keys)
    val, train = [], []
    for k in keys:
        (val if len(val) < val_size else train).extend(groups[k])
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


# HF ids/configs for the plain single-label classification sources; loaded uniformly
# and fed through cls_examples. massive/mtop use load_dataset("json"/"parquet", ...).
SIMPLE_SOURCES = [
    dict(task="ag_news", hf_id="fancyzhx/ag_news", split="train", text=lambda r: r["text"],
         label_col="label", classlabel=True, family="topic", full_n=30000),
    dict(task="dbpedia", hf_id="fancyzhx/dbpedia_14", split="train", text=lambda r: f'{r["title"]}. {r["content"]}',
         label_col="label", classlabel=True, family="topic", full_n=30000),
    dict(task="yahoo", hf_id="community-datasets/yahoo_answers_topics", split="train",
         text=lambda r: f'{r["question_title"]} {r["question_content"]}',
         label_col="topic", classlabel=True, family="topic", full_n=30000),
    dict(task="tweet_emotion", hf_id="cardiffnlp/tweet_eval", config="emotion", split="train",
         text=lambda r: r["text"], label_col="label", classlabel=True, family="emotion", full_n=None),
    dict(task="tweet_sentiment", hf_id="cardiffnlp/tweet_eval", config="sentiment", split="train",
         text=lambda r: r["text"], label_col="label", classlabel=True, family="emotion", full_n=20000),
    dict(task="tweet_hate", hf_id="cardiffnlp/tweet_eval", config="hate", split="train",
         text=lambda r: r["text"], label_col="label", classlabel=True, family="emotion", full_n=None),
    dict(task="tweet_irony", hf_id="cardiffnlp/tweet_eval", config="irony", split="train",
         text=lambda r: r["text"], label_col="label", classlabel=True, family="emotion", full_n=None),
    dict(task="tweet_offensive", hf_id="cardiffnlp/tweet_eval", config="offensive", split="train",
         text=lambda r: r["text"], label_col="label", classlabel=True, family="emotion", full_n=None),
    dict(task="dair_emotion", hf_id="dair-ai/emotion", split="train", text=lambda r: r["text"],
         label_col="label", classlabel=True, family="emotion", full_n=None),
    dict(task="sst5", hf_id="SetFit/sst5", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="emotion", full_n=None),
    dict(task="hwu64", hf_id="FastFit/hwu_64", split="train", text=lambda r: r["text"],
         label_col="label", classlabel=False, family="intent", full_n=None),
    dict(task="snips", hf_id="benayas/snips", split="train", text=lambda r: r["text"],
         label_col="category", classlabel=False, family="intent", full_n=None),
    dict(task="massive", hf_id="json", data_files="hf://datasets/mteb/amazon_massive_intent/train/en.json.gz",
         split="train", text=lambda r: r["text"], label_col="label", classlabel=False, family="intent", full_n=None),
    dict(task="mtop", hf_id="parquet", data_files="hf://datasets/mteb/mtop_intent/en/train-00000-of-00001.parquet",
         split="train", text=lambda r: r["text"], label_col="label_text", classlabel=False, family="intent", full_n=None),
    # v5 (REPORT.md 3d): many more label vocabularies at lower rows/source each (cls_cap), so
    # the model can't just memorise ~16 fixed label sets. All verified to load as of 2026-09.
    dict(task="bbc_news", hf_id="SetFit/bbc-news", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="topic", full_n=None),
    dict(task="subj", hf_id="SetFit/subj", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="emotion", full_n=None),
    dict(task="cr_sentiment", hf_id="SetFit/CR", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="emotion", full_n=None),
    dict(task="enron_spam", hf_id="SetFit/enron_spam", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="topic", full_n=None),
    dict(task="amazon_cf", hf_id="SetFit/amazon_counterfactual_en", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="emotion", full_n=None),
    dict(task="sst2", hf_id="SetFit/sst2", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="emotion", full_n=None),
    dict(task="tweet_stance", hf_id="cardiffnlp/tweet_eval", config="stance_abortion", split="train",
         text=lambda r: r["text"], label_col="label", classlabel=True, family="emotion", full_n=None),
    dict(task="arxiv_cls", hf_id="ccdv/arxiv-classification", config="no_ref", split="train",
         text=lambda r: r["text"][:1000], label_col="label", classlabel=True, family="topic", full_n=None),
    dict(task="patent_cls", hf_id="ccdv/patent-classification", split="train",
         text=lambda r: r["text"][:1000], label_col="label", classlabel=True, family="topic", full_n=None),
    dict(task="bitext_support", hf_id="bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train",
         text=lambda r: r["instruction"], label_col="intent", classlabel=False, family="intent", full_n=None),
    dict(task="student_subj", hf_id="SetFit/student-question-categories", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="topic", full_n=None),
    dict(task="hate_offensive", hf_id="SetFit/hate_speech_offensive", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="emotion", full_n=None),
    dict(task="imdb", hf_id="stanfordnlp/imdb", split="train", text=lambda r: r["text"][:1000],
         label_col="label", classlabel=True, family="emotion", full_n=None),
    dict(task="yelp_full", hf_id="SetFit/yelp_review_full", split="train", text=lambda r: r["text"],
         label_col="label_text", classlabel=False, family="emotion", full_n=None),
]

HELD_OUT_ENTIRELY = {"banking77_test", "banking77_k", "trec_coarse", "trec_fine", "ng20_test"}
# Eval sets whose own dataset reuses state text across its official train/test splits by
# design (ANLI premises are annotated with many hypotheses across rounds; BoolQ/HWU64 questions
# repeat verbatim). State-based dedupe would gut these to near-nothing (e.g. anli_test 3200 -> 62
# rows observed), so -- like the held-out sets -- they stay intact and the few hundred/thousand
# colliding TRAIN rows are dropped instead (a rounding error against ~800k train rows).
PROTECTED_EVAL = HELD_OUT_ENTIRELY | {"anli_test", "boolq_val", "hwu64_test"}

CLS_FAMILIES = ("topic", "emotion", "intent")


def listwise_augment(train_examples, rng):
    """Item 3: post-pass over classification rows (p_null==0 only). 3% duplicate a random
    candidate (target mass split 50/50 between the two copies, label unchanged -- it still
    points at the first copy since ties break to the lowest index, see selftest). 5% append
    one label string from a DIFFERENT family's training label space (target 0 for it; the
    pool is built only from rows already in train_examples, so held-out eval-only spaces like
    banking77/TREC/20NG can never be drawn from)."""
    family_names = defaultdict(set)
    for r in train_examples:
        fam = r.get("meta", {}).get("family")
        if fam in CLS_FAMILIES:
            family_names[fam].update(r["candidates"])

    extra, dup_n, xfam_n = [], 0, 0
    for r in train_examples:
        fam = r.get("meta", {}).get("family")
        if fam not in CLS_FAMILIES or r["p_null"] != 0.0:
            continue
        u = rng.random()
        if u < 0.03:
            j = rng.randrange(len(r["candidates"]))
            target = list(r["target"])
            mass = target[j]
            target[j] = mass / 2
            target.append(mass / 2)
            extra.append({**r, "candidates": r["candidates"] + [r["candidates"][j]], "target": target})
            dup_n += 1
        elif u < 0.08:
            others = [n for f, pool in family_names.items() if f != fam for n in pool]
            if others:
                extra.append({**r, "candidates": r["candidates"] + [rng.choice(others)],
                              "target": r["target"] + [0.0]})
                xfam_n += 1
    print(f"  listwise augmentation: {dup_n} dup rows, {xfam_n} cross-family distractor rows")
    return train_examples + extra


def add_noev_rows(train_examples, rng):
    """Item 4 (Stage-B-lite): 3% of SNLI/MNLI rows get a copy with the premise blanked out and
    a flat target over the same candidates (p_null 0) -- no evidence, so the right answer is
    "don't know which of these", not a confident guess. Marked snli_noev/mnli_noev."""
    extra = []
    for r in train_examples:
        if r["task"] in ("snli", "mnli") and rng.random() < 0.03:
            k = len(r["candidates"])
            extra.append({**r, "state": ".", "target": [1.0 / k] * k, "p_null": 0.0,
                          "task": f"{r['task']}_noev", "label": None})
    print(f"  stage-B-lite no-evidence rows: {len(extra)}")
    return train_examples + extra


def print_k_null_table(rows, label="train"):
    """Item 1 reporting: null rate should now be roughly flat across K buckets instead of the
    old K>=50 -> never-null / K=N -> never-null coupling (REVIEW.md item 6)."""
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        b = buckets[len(r["candidates"])]
        b[0] += 1
        b[1] += r["p_null"] >= 1.0
    print(f"\n=== Per-K null rate ({label}) ===")
    print(f"{'K':>6}{'n':>10}{'null_rate':>12}")
    for k in sorted(buckets):
        n, nulls = buckets[k]
        print(f"{k:>6}{n:>10}{nulls / n:>12.3f}")


def near_dup_name(name, rng):
    """One word dropped or one word synonym-swapped (SYNONYMS); None if neither is possible."""
    toks = name.split()
    options = []
    if len(toks) > 1:
        options += [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
    for i, w in enumerate(toks):
        for syn in SYNONYMS.get(w, []):
            options.append(" ".join(toks[:i] + [syn] + toks[i + 1:]))
    return rng.choice(options) if options else None


def cse_variants(uid, state, query, task, gold_name, distractor_pool, rng, k_base=10):
    """Item 5: base K=k_base set (gold + k_base-1 distractors, canonical names) plus the
    choice-set-effects battery: add_irr / remove_gold / dup / reorder / near_dup, all
    meta.pair-linked to the base row so metrics.choice_set_effects can align them."""
    pool = [n for n in distractor_pool if n != gold_name]
    rng.shuffle(pool)
    base_cands = pool[:k_base - 1] + [gold_name]
    rng.shuffle(base_cands)
    gold_i = base_cands.index(gold_name)
    base_target = [1.0 if i == gold_i else 0.0 for i in range(len(base_cands))]
    fam = FAMILY.get(task, "intent")

    def mk(variant, cands, target, p_null, label, **extra_meta):
        return {"state": state, "query": query, "candidates": cands, "target": target,
                "p_null": p_null, "task": task, "label": label,
                "meta": {"family": fam, "pair": uid, "variant": variant, **extra_meta}}

    rows = [mk("base", base_cands, base_target, 0.0, gold_i)]

    irr = rng.choice(IRRELEVANT)
    rows.append(mk("add_irr", base_cands + [irr], base_target + [0.0], 0.0, gold_i))

    rg_cands = [c for c in base_cands if c != gold_name]
    rows.append(mk("remove_gold", rg_cands, [1.0 / len(rg_cands)] * len(rg_cands), 1.0, -1))

    dup_i = gold_i if rng.random() < 0.5 else rng.randrange(len(base_cands))
    dup_target = list(base_target)
    mass = dup_target[dup_i]
    dup_target[dup_i] = mass / 2
    dup_target.append(mass / 2)
    rows.append(mk("dup", base_cands + [base_cands[dup_i]], dup_target, 0.0, gold_i, src=base_cands[dup_i]))

    order = list(range(len(base_cands)))
    rng.shuffle(order)
    rows.append(mk("reorder", [base_cands[i] for i in order], [base_target[i] for i in order],
                   0.0, order.index(gold_i)))

    nd_i = gold_i if rng.random() < 0.5 else rng.randrange(len(base_cands))
    nd_name = near_dup_name(base_cands[nd_i], rng)
    if nd_name is not None:
        rows.append(mk("near_dup", base_cands + [nd_name], base_target + [0.0], 0.0, gold_i,
                       src=base_cands[nd_i]))
    return rows


def build_cse_set(task, utterances, rng, k_base=10):
    """utterances: list of (state, query, gold_name, distractor_pool)."""
    rows = []
    for i, (state, query, gold_name, dpool) in enumerate(utterances):
        rows += cse_variants(f"{task}_{i}", state, query, task, gold_name, dpool, rng, k_base=k_base)
    return rows


def build_ksweep(task, utterances, id2name, all_ids, rng, k_list=(2, 3, 5, 10, 20, 50, 100), sibling_map=None):
    """Item 5: 400 utterances x K x {present, absent}, nested sets from one permutation of
    non-gold labels per utterance so P(null | absent) can be plotted as a function of K.
    sibling_map: when given, the gold's siblings are excluded from the distractors (REPORT §3h:
    with siblings in, "absent" at large K is a near-miss judgment and the sweep conflates K with
    difficulty; this variant measures pure K-dependence). K = N then means all non-siblings."""
    N = len(all_ids)
    ks = sorted({k for k in k_list if k < N} | {N})
    fam = FAMILY.get(task, "intent")
    rows = []
    for u, (state, query, gold_id) in enumerate(utterances):
        excl = {gold_id} | set(sibling_map.get(gold_id, [])) if sibling_map else {gold_id}
        others = [i for i in all_ids if i not in excl]
        rng.shuffle(others)
        for k in ks:
            pres_ids = others[:max(k - 1, 0)] + [gold_id]
            pres_names = [id2name[i] for i in pres_ids]
            rows.append({"state": state, "query": query, "candidates": pres_names,
                         "target": [1.0 if i == gold_id else 0.0 for i in pres_ids],
                         "p_null": 0.0, "task": task, "label": pres_ids.index(gold_id),
                         "meta": {"family": fam, "K": k, "u": u, "present": True}})
            abs_ids = others[:min(k, len(others))]
            abs_names = [id2name[i] for i in abs_ids]
            rows.append({"state": state, "query": query, "candidates": abs_names,
                         "target": [1.0 / len(abs_names)] * len(abs_names) if abs_names else [],
                         "p_null": 1.0, "task": task, "label": -1,
                         "meta": {"family": fam, "K": k, "u": u, "present": False}})
    return rows


def pick_nearmiss(gold_id, pool, k, include_gold, rng, sibling_map):
    """Like pick_labels, but a sibling of the gold is GUARANTEED in the set (not p=0.5) --
    returns None if the gold has no sibling in this label space."""
    others = [i for i in pool if i != gold_id]
    sibs = [s for s in sibling_map.get(gold_id, []) if s in others]
    if not sibs:
        return None
    sib = rng.choice(sibs)
    rest = [o for o in others if o != sib]
    if include_gold:
        k = min(k, len(pool))
        extra = rng.sample(rest, max(min(k - 2, len(rest)), 0))
        idx = extra + [sib, gold_id]
        rng.shuffle(idx)
        target = [1.0 if i == gold_id else 0.0 for i in idx]
        return idx, target, 0.0, idx.index(gold_id)
    k = min(k, len(others))
    extra = rng.sample(rest, max(min(k - 1, len(rest)), 0))
    idx = extra + [sib]
    rng.shuffle(idx)
    return idx, [1.0 / len(idx)] * len(idx), 1.0, -1


def build_null_nearmiss(task, rows, pool, id2name, rng, n=1000):
    """Item 5: <=1000 rows, half K=10 gold-absent with a forced sibling (p_null=1), half
    K=10 gold-present with a sibling also present. Skips utterances whose gold has no
    sibling in this label space, so the file can be smaller than n."""
    sib = siblings(id2name)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    half = n // 2
    absent, present = [], []
    for text, gold in shuffled:
        if len(absent) >= half and len(present) >= half:
            break
        if len(absent) < half:
            res = pick_nearmiss(gold, pool, 10, False, rng, sib)
            if res is not None:
                absent.append((text, res))
                continue
        if len(present) < half:
            res = pick_nearmiss(gold, pool, 10, True, rng, sib)
            if res is not None:
                present.append((text, res))
    fam = FAMILY.get(task, "intent")
    out = [{"state": trunc(text), "query": render("intent", 0), "candidates": [id2name[i] for i in idx],
            "target": target, "p_null": p_null, "task": task, "label": label, "meta": {"family": fam}}
           for text, (idx, target, p_null, label) in absent + present]
    print(f"  {task}: {len(absent)} absent / {len(present)} present (target {half}/{half} each)")
    return out


def build_null_irrq_squad(squad_val_all, rng, half=1000):
    """Item 5: 1000 normal answerable rows + 1000 where the (question, candidates) come from
    a different-title row than the passage -- the right answer literally isn't in this
    passage, so p_null=1 regardless of what the candidates say."""
    ans_val = [r for r in squad_val_all if r["answers"]["text"] and r["answers"]["answer_start"][0] < 900]
    rng.shuffle(ans_val)
    normal_src, irr_src = ans_val[:half], ans_val[half:half * 2]
    normal = qa_examples(normal_src, squad_val_all, "null_irrq_squad", rng, qidx=0)
    irr_rows = qa_examples(irr_src, squad_val_all, "null_irrq_squad", rng, qidx=0)

    # qa_examples can drop a row (leaks_paraphrase); match back to its title by question text
    # rather than position, so a drop can't silently pair the wrong context with a question.
    q2title = {r["question"]: r["title"] for r in irr_src}
    by_title = defaultdict(list)
    for r in irr_src:
        by_title[r["title"]].append(r["context"])
    titles = list(by_title)
    mismatched = []
    for row in irr_rows:
        src_title = q2title.get(row["meta"]["question"])
        other_titles = [t for t in titles if t != src_title]
        if not other_titles:
            continue
        ctx = rng.choice(by_title[rng.choice(other_titles)])
        k = len(row["candidates"])
        mismatched.append({**row, "state": trunc(ctx, 1000), "p_null": 1.0,
                            "target": [1.0 / k] * k, "label": -1})
    return normal + mismatched


def build_null_irrq_intent(snli_pool, in_scope_test, names, in_scope_ids, rng, n=1000):
    """Item 5: 1000 SNLI premises scored against the intent template + K=10 CLINC labels
    (nothing in a premise is an intent, so p_null=1) + 1000 normal CLINC K=10 gold-present
    rows as the contrast."""
    pool = list(snli_pool)
    rng.shuffle(pool)
    rows = []
    for r in pool[:n]:
        k_ids = rng.sample(in_scope_ids, min(10, len(in_scope_ids)))
        cnames = [names[i] for i in k_ids]
        rows.append({"state": trunc(r["premise"]), "query": render("intent", 0), "candidates": cnames,
                     "target": [1.0 / len(cnames)] * len(cnames), "p_null": 1.0,
                     "task": "null_irrq_intent", "label": -1, "meta": {"family": "intent"}})
    cchosen = list(in_scope_test)
    rng.shuffle(cchosen)
    rows += clinc_examples(cchosen[:n], "null_irrq_intent", in_scope_ids,
                            lambda rng: (10, True), names, rng, qidx=0, paraphrase=False)
    return rows


def _shingles(s, k=5):
    return [s[i:i + k].encode() for i in range(len(s) - k + 1)] if len(s) >= k else [s.encode()]


def _minhash(s):
    m = MinHash(num_perm=64)
    m.update_batch(_shingles(s))
    return m


def prune_near_dup_train(train_examples, eval_states, tasks, threshold=0.8):
    """Coordinator addition: scripts/leak_audit.py's exact-match dedupe misses near-duplicate
    leakage (boolq_val 4.9% / hwu64_test 5.4% near-identical twins in train, runs/leak_audit.json).
    Drops TRAIN rows in `tasks` whose normalised state is within `threshold` Jaccard (char
    5-gram MinHash, num_perm=64 -- same recipe as leak_audit.py, reimplemented locally to
    avoid a pcdm/data.py <-> scripts/leak_audit.py import cycle) of any state in `eval_states`."""
    if not eval_states:
        return train_examples
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    mh = {}
    for i, s in enumerate(eval_states):
        key = f"e{i}"
        mh[key] = _minhash(s)
        lsh.insert(key, mh[key])
    kept, dropped = [], 0
    for r in train_examples:
        if r["task"] not in tasks:
            kept.append(r)
            continue
        qm = _minhash(norm_text(r["state"]))
        if any(qm.jaccard(mh[c]) >= threshold for c in lsh.query(qm)):
            dropped += 1
        else:
            kept.append(r)
    print(f"  near-dup prune ({sorted(tasks)}): dropped {dropped} train rows")
    return kept


def main(small=False, out_name=None, cls_cap=None):
    out = Path(out_name) if out_name else (Path("data_small") if small else DATA)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)  # SNLI zip cache lives here regardless of `out`

    def capT(rows, full_n):
        return rows[: (200 if small else full_n)]

    def capE(rows, full_n):
        return rows[: (100 if small else full_n)]

    rng = random.Random(SEED)
    train_examples, train_pairs = [], set()
    label_space = {}

    print("=== NLI (hard) ===")
    snli_train_raw = [r for r in load_dataset("stanfordnlp/snli", split="train") if r["label"] != -1]
    by_label = defaultdict(list)
    for r in snli_train_raw:
        by_label[r["label"]].append(r)
    for v in by_label.values():
        rng.shuffle(v)
    per_label = (200 if small else 150000) // 3
    snli_sample = [r for v in by_label.values() for r in v[:per_label]]
    ex, pairs = nli_examples(
        ((r["premise"], r["hypothesis"], [float(i == r["label"]) for i in range(3)], r["label"]) for r in snli_sample),
        "snli", rng)
    train_examples += ex; train_pairs |= pairs

    mnli_train_raw = [r for r in load_dataset("nyu-mll/multi_nli", split="train") if r["label"] != -1]
    rng.shuffle(mnli_train_raw)
    ex, pairs = nli_examples(
        ((r["premise"], r["hypothesis"], [float(i == r["label"]) for i in range(3)], r["label"])
         for r in capT(mnli_train_raw, 120000)),
        "mnli", rng)
    train_examples += ex; train_pairs |= pairs

    anli_train_raw = (list(load_dataset("facebook/anli", "plain_text", split="train_r1"))
                       + list(load_dataset("facebook/anli", "plain_text", split="train_r2"))
                       + list(load_dataset("facebook/anli", "plain_text", split="train_r3"))[:40000])
    rng.shuffle(anli_train_raw)
    ex, pairs = nli_examples(
        ((r["premise"], r["hypothesis"], [float(i == r["label"]) for i in range(3)], r["label"])
         for r in capT(anli_train_raw, len(anli_train_raw))),
        "anli", rng)
    train_examples += ex; train_pairs |= pairs

    dev_votes = list(load_snli_votes("dev").items())
    rng.shuffle(dev_votes)
    ex, pairs = nli_examples(
        ((p, h, dist, dist.index(max(dist))) for (p, h), dist in capT(dev_votes, 10000)),
        "snli_soft", rng)
    train_examples += ex; train_pairs |= pairs

    train_examples = add_noev_rows(train_examples, rng)

    print("=== Soft proposition (UNLI) + BoolQ ===")
    try:
        unli_test_raw = list(load_dataset("Zhengping/UNLI", split="test"))
        unli_train_pool = list(load_dataset("Zhengping/UNLI", split="train"))
    except Exception:
        print("WARNING: UNLI test split missing; holding out 3k from train instead")
        unli_all = list(load_dataset("Zhengping/UNLI", split="train"))
        rng.shuffle(unli_all)
        unli_test_raw, unli_train_pool = unli_all[:3000], unli_all[3000:]
    rng.shuffle(unli_train_pool)
    for r in capT(unli_train_pool, 55000):
        train_examples.append(unli_row(r["premise"], r["hypothesis"], r["label"], "unli", rng))
        train_pairs.add(pair_key(r["premise"], r["hypothesis"]))

    boolq_train_raw = list(load_dataset("google/boolq", split="train"))
    rng.shuffle(boolq_train_raw)
    train_examples += boolq_examples(capT(boolq_train_raw, len(boolq_train_raw)), "boolq", rng, mode="train")

    print("=== QA (SQuAD v2, real null) ===")
    squad_train_all = [r for r in load_dataset("rajpurkar/squad_v2", split="train") if len(r["context"]) <= 1000]
    ans_rows = [r for r in squad_train_all if r["answers"]["text"] and r["answers"]["answer_start"][0] < 900]
    unans_rows = [r for r in squad_train_all if not r["answers"]["text"]]
    rng.shuffle(ans_rows); rng.shuffle(unans_rows)
    squad_target = capT(ans_rows, 60000) + capT(unans_rows, 40000)
    rng.shuffle(squad_target)
    train_examples += qa_examples(squad_target, squad_train_all, "squad", rng)

    print("=== CLINC (8k in-scope + all oos; 30 intents held out) ===")
    clinc_train_raw = list(load_dataset("clinc/clinc_oos", "plus", split="train"))
    clinc_test_raw = list(load_dataset("clinc/clinc_oos", "plus", split="test"))
    names = [words(n) for n in load_dataset("clinc/clinc_oos", "plus", split="train").features["intent"].names]
    oos_id = names.index("oos")
    in_scope_ids = sorted(i for i in range(len(names)) if i != oos_id)
    held_out = set(rng.sample(in_scope_ids, 30))
    visible = [i for i in in_scope_ids if i not in held_out]
    json.dump(sorted(names[i] for i in held_out), open(out / "held_out_intents.json", "w"), indent=2)

    def in_scope_k(rng):
        return k_null_choice(rng, len(visible))

    def oos_k(rng):
        return (10 if rng.random() < 0.5 else len(visible)), False

    visible_rows = [(r["text"], r["intent"]) for r in clinc_train_raw if r["intent"] in visible]
    rng.shuffle(visible_rows)
    oos_rows = [(r["text"], -1) for r in clinc_train_raw if r["intent"] == oos_id]
    train_examples += clinc_examples(capT(visible_rows, 8000), "clinc", visible, in_scope_k, names, rng, sibling_null=True)
    train_examples += clinc_examples(capT(oos_rows, len(oos_rows)), "clinc", visible, oos_k, names, rng, sibling_null=True)

    print("=== Intent / topic / emotion (label-diverse) ===")
    for spec in SIMPLE_SOURCES:
        try:
            kwargs = {}
            if spec.get("config"):
                kwargs["name"] = spec["config"]
            if spec.get("data_files"):
                kwargs["data_files"] = spec["data_files"]
            ds = load_dataset(spec["hf_id"], split=spec["split"], **kwargs)
            if spec["classlabel"]:
                names_ = [words(n) for n in ds.features[spec["label_col"]].names]
                rows = [(spec["text"](r), r[spec["label_col"]]) for r in ds]
            else:
                uniq = sorted({r[spec["label_col"]] for r in ds})
                id2 = {u: i for i, u in enumerate(uniq)}
                names_ = [words(u) for u in uniq]
                rows = [(spec["text"](r), id2[r[spec["label_col"]]]) for r in ds]
                label_space[spec["task"]] = (uniq, names_)
            rng.shuffle(rows)
            full_n = spec["full_n"] or len(rows)
            # ponytail: cls_cap (v5) caps every classification source so more label
            # vocabularies fit the same train budget; CLINC is untouched (per spec).
            if cls_cap is not None:
                full_n = min(full_n, cls_cap)
            rows = capT(rows, full_n)
            train_examples += cls_examples(rows, spec["task"], spec["family"], names_, rng, sibling_null=True)
            print(f"  {spec['task']}: {len(rows)} rows / {len(names_)} labels")
        except Exception as e:
            print(f"WARNING: skipping {spec['task']} ({spec['hf_id']}): {e}")

    try:
        ge = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
        ge_names = [words(n) for n in ge.features["labels"].feature.names]
        ge_rows = [(r["text"], r["labels"][0]) for r in ge if len(r["labels"]) == 1]
        rng.shuffle(ge_rows)
        ge_rows = capT(ge_rows, min(30000, cls_cap) if cls_cap is not None else 30000)
        train_examples += cls_examples(ge_rows, "go_emotions", "emotion", ge_names, rng, sibling_null=True)
        print(f"  go_emotions: {len(ge_rows)} rows / {len(ge_names)} labels")
    except Exception as e:
        print(f"WARNING: skipping go_emotions: {e}")

    print("=== Listwise augmentation (train, classification, p_null==0 only) ===")
    train_examples = listwise_augment(train_examples, rng)

    print("=== Held-out eval (banking77 / TREC / 20newsgroups / hwu64_test) ===")
    eval_sets = {}
    try:
        bank_test = load_dataset("mteb/banking77", split="test")
        buniq = sorted({r["label_text"] for r in bank_test})
        bid2 = {u: i for i, u in enumerate(buniq)}
        bnames = [words(u) for u in buniq]
        brows = capE([(r["text"], bid2[r["label_text"]]) for r in bank_test], 10 ** 9)
        eval_sets["banking77_test"] = cls_examples(brows, "banking77_test", "intent", bnames, rng,
                                                     k_choice=lambda rng, N: (N, True), qidx=0, paraphrase=False)
        eval_sets["banking77_k"] = cls_examples(brows, "banking77_k", "intent", bnames, rng,
                                                  k_choice=lambda rng, N: (10, rng.random() < 0.5), qidx=0, paraphrase=False)
    except Exception as e:
        print(f"WARNING: skipping banking77 ({e})")

    try:
        trec_test = load_dataset("SetFit/TREC-QC", split="test")
        cuniq = sorted({r["label_coarse_text"] for r in trec_test})
        cid2 = {u: i for i, u in enumerate(cuniq)}
        cnames = [words(u) for u in cuniq]
        crows = capE([(r["text"], cid2[r["label_coarse_text"]]) for r in trec_test], 10 ** 9)
        eval_sets["trec_coarse"] = cls_examples(crows, "trec_coarse", "topic", cnames, rng,
                                                  k_choice=lambda rng, N: (N, True), qidx=0, paraphrase=False)
        funiq = sorted({r["label_text"] for r in trec_test})
        fid2 = {u: i for i, u in enumerate(funiq)}
        fnames = [words(u) for u in funiq]
        frows = capE([(r["text"], fid2[r["label_text"]]) for r in trec_test], 10 ** 9)
        eval_sets["trec_fine"] = cls_examples(frows, "trec_fine", "topic", fnames, rng,
                                                k_choice=lambda rng, N: (N, True), qidx=0, paraphrase=False)
    except Exception as e:
        print(f"WARNING: skipping TREC-QC ({e})")

    try:
        ng_test = load_dataset("SetFit/20_newsgroups", split="test")
        nuniq = sorted({r["label_text"] for r in ng_test})
        nid2 = {u: i for i, u in enumerate(nuniq)}
        nnames = [NG20_NAMES.get(u, words(u)) for u in nuniq]
        nrows = capE([(r["text"], nid2[r["label_text"]]) for r in ng_test], 10 ** 9)
        eval_sets["ng20_test"] = cls_examples(nrows, "ng20_test", "topic", nnames, rng,
                                                k_choice=lambda rng, N: (N, True), qidx=0, paraphrase=False)
    except Exception as e:
        print(f"WARNING: skipping 20_newsgroups ({e})")

    try:
        hwu_test = load_dataset("FastFit/hwu_64", split="test")
        huniq, hnames = label_space.get("hwu64", (None, None))
        if huniq:
            hid2 = {u: i for i, u in enumerate(huniq)}
            hrows = capE([(r["text"], hid2[r["label"]]) for r in hwu_test if r["label"] in hid2], 10 ** 9)
            eval_sets["hwu64_test"] = cls_examples(hrows, "hwu64_test", "intent", hnames, rng,
                                                     k_choice=lambda rng, N: (N, True), qidx=0)
    except Exception as e:
        print(f"WARNING: skipping hwu64_test ({e})")

    print("=== NLI / CLINC eval (v0-compatible eval RNG) ===")
    rng = random.Random(SEED + 1)

    snli_test_all = [r for r in load_dataset("stanfordnlp/snli", split="test") if r["label"] != -1]
    rng.shuffle(snli_test_all)
    snli_test_base = capE(snli_test_all, 5000)

    def snli_rows(base):
        return ((r["premise"], r["hypothesis"], [float(i == r["label"]) for i in range(3)], r["label"]) for r in base)

    ex, _ = nli_examples(snli_rows(snli_test_base), "snli_test", rng, mode="keep", qidx=0)
    eval_sets["snli_test"] = ex
    ex, _ = nli_examples(snli_rows(snli_test_base), "snli_test_paraphrase", rng, cand_names=PARAPHRASES, mode="keep", qidx=0)
    eval_sets["snli_test_paraphrase"] = ex
    ex, _ = nli_examples(snli_rows(snli_test_base), "snli_null", rng, mode="k2", qidx=0)
    eval_sets["snli_null"] = ex

    test_votes = load_snli_votes("test")
    if test_votes:
        soft_rows = [(r["premise"], r["hypothesis"], test_votes[pair_key(r["premise"], r["hypothesis"])], None)
                     for r in snli_test_base if pair_key(r["premise"], r["hypothesis"]) in test_votes]
        soft_rows = [(p, h, d, d.index(max(d))) for p, h, d, _ in soft_rows]
        ex, _ = nli_examples(soft_rows, "snli_test_soft", rng, mode="keep", qidx=0)
        eval_sets["snli_test_soft"] = ex
    else:
        print("  skipping snli_test_soft (no SNLI zip)")

    ex, _ = nli_examples(snli_rows(capE(snli_test_base, 2000)), "snli_test_qpara", rng, mode="keep", qidx=(8, 9))
    eval_sets["snli_test_qpara"] = ex

    mnli_val_all = list(load_dataset("nyu-mll/multi_nli", split="validation_matched"))
    rng.shuffle(mnli_val_all)
    ex, _ = nli_examples(
        ((r["premise"], r["hypothesis"], [float(i == r["label"]) for i in range(3)], r["label"])
         for r in capE(mnli_val_all, 3000)),
        "mnli_val", rng, mode="keep", qidx=0)
    eval_sets["mnli_val"] = ex

    chaos_all = list(load_dataset("metaeval/chaos-mnli-ambiguity", split="train"))
    ex, _ = nli_examples(
        ((r["premise"], r["hypothesis"], r["label_dist"], r["label_dist"].index(max(r["label_dist"]))) for r in chaos_all),
        "chaos_mnli", rng, mode="keep", label_none=True, qidx=0)
    eval_sets["chaos_mnli"] = ex

    eval_sets["unli_test"] = [unli_row(r["premise"], r["hypothesis"], r["label"], "unli_test", rng, qidx=0)
                               for r in unli_test_raw]

    anli_test_raw = []
    for i in (1, 2, 3):
        anli_test_raw += list(load_dataset("facebook/anli", "plain_text", split=f"test_r{i}"))
    ex, _ = nli_examples(
        ((r["premise"], r["hypothesis"], [float(i == r["label"]) for i in range(3)], r["label"]) for r in anli_test_raw),
        "anli_test", rng, mode="keep", qidx=0)
    eval_sets["anli_test"] = ex

    in_scope_test = [(r["text"], r["intent"]) for r in clinc_test_raw if r["intent"] != oos_id]
    oos_test = [(r["text"], -1) for r in clinc_test_raw if r["intent"] == oos_id]
    heldout_test = [(t, g) for t, g in in_scope_test if g in held_out]

    eval_sets["clinc_test"] = clinc_examples(in_scope_test, "clinc_test", in_scope_ids,
                                              lambda rng: (len(in_scope_ids), True), names, rng, qidx=0, paraphrase=False)
    eval_sets["clinc_oos"] = clinc_examples(oos_test, "clinc_oos", in_scope_ids,
                                             lambda rng: (len(in_scope_ids), False), names, rng, qidx=0, paraphrase=False)
    eval_sets["clinc_heldout"] = clinc_examples(heldout_test, "clinc_heldout", list(held_out),
                                                 lambda rng: (10, True), names, rng, qidx=0, paraphrase=False)
    eval_sets["clinc_k"] = clinc_examples(
        in_scope_test, "clinc_k", in_scope_ids,
        lambda rng: (rng.choice([2, 5, 10, 50]), rng.random() < 0.5), names, rng, qidx=0, paraphrase=False)

    print("=== BoolQ / SQuAD eval ===")
    boolq_val_raw = list(load_dataset("google/boolq", split="validation"))
    eval_sets["boolq_val"] = boolq_examples(capE(boolq_val_raw, len(boolq_val_raw)), "boolq_val", rng, mode="eval")

    squad_val_all = [r for r in load_dataset("rajpurkar/squad_v2", split="validation") if len(r["context"]) <= 1000]
    ans_val = [r for r in squad_val_all if r["answers"]["text"] and r["answers"]["answer_start"][0] < 900]
    unans_val = [r for r in squad_val_all if not r["answers"]["text"]]
    rng.shuffle(ans_val); rng.shuffle(unans_val)
    half_n = 100 if small else 2000
    squad_null_target = ans_val[:half_n] + unans_val[:half_n]
    rng.shuffle(squad_null_target)
    eval_sets["squad_null"] = qa_examples(squad_null_target, squad_val_all, "squad_null", rng, qidx=0)

    # ============ New eval sets (item 5); all new RNG draws come after the ones above,
    # so every eval set built up to this line is unaffected byte-for-byte. ============
    print("=== cse_* (choice-set effects) ===")
    try:
        pool = list(in_scope_test); rng.shuffle(pool)
        clinc_pool_names = [names[i] for i in in_scope_ids]
        utter = [(t, render("intent", 0), names[g], clinc_pool_names) for t, g in pool[:500]]
        eval_sets["cse_clinc"] = build_cse_set("cse_clinc", utter, rng, k_base=10)
    except Exception as e:
        print(f"WARNING: skipping cse_clinc ({e})")
    try:
        pool = list(hrows); rng.shuffle(pool)
        utter = [(t, render("intent", 0), hnames[g], hnames) for t, g in pool[:500]]
        eval_sets["cse_hwu64"] = build_cse_set("cse_hwu64", utter, rng, k_base=10)
    except Exception as e:
        print(f"WARNING: skipping cse_hwu64 ({e})")
    try:
        pool = list(brows); rng.shuffle(pool)
        utter = [(t, render("intent", 0), bnames[g], bnames) for t, g in pool[:500]]
        eval_sets["cse_banking77"] = build_cse_set("cse_banking77", utter, rng, k_base=10)
    except Exception as e:
        print(f"WARNING: skipping cse_banking77 ({e})")
    try:
        pool = list(snli_test_base); rng.shuffle(pool)
        utter = [(r["premise"], render("nli", 0, hyp=r["hypothesis"].strip()), CAND3[r["label"]], list(CAND3))
                 for r in pool[:500]]
        eval_sets["cse_snli"] = build_cse_set("cse_snli", utter, rng, k_base=3)
    except Exception as e:
        print(f"WARNING: skipping cse_snli ({e})")

    print("=== ksweep_* ===")
    try:
        pool = list(in_scope_test); rng.shuffle(pool)
        utter = [(t, render("intent", 0), g) for t, g in pool[:400]]
        eval_sets["ksweep_clinc"] = build_ksweep("ksweep_clinc", utter, names, in_scope_ids, rng)
        eval_sets["ksweep_clinc_nosib"] = build_ksweep("ksweep_clinc_nosib", utter, names, in_scope_ids, rng,
                                                       sibling_map=build_sibling_map(names, "ksweep_clinc"))
    except Exception as e:
        print(f"WARNING: skipping ksweep_clinc ({e})")
    try:
        pool = list(brows); rng.shuffle(pool)
        utter = [(t, render("intent", 0), g) for t, g in pool[:400]]
        eval_sets["ksweep_banking77"] = build_ksweep("ksweep_banking77", utter, bnames, list(range(len(bnames))), rng)
    except Exception as e:
        print(f"WARNING: skipping ksweep_banking77 ({e})")

    print("=== null_irrq_* ===")
    try:
        eval_sets["null_irrq_squad"] = build_null_irrq_squad(squad_val_all, rng)
    except Exception as e:
        print(f"WARNING: skipping null_irrq_squad ({e})")
    try:
        eval_sets["null_irrq_intent"] = build_null_irrq_intent(snli_test_base, in_scope_test, names, in_scope_ids, rng)
    except Exception as e:
        print(f"WARNING: skipping null_irrq_intent ({e})")

    print("=== null_nearmiss_* ===")
    try:
        eval_sets["null_nearmiss_clinc"] = build_null_nearmiss("null_nearmiss_clinc", in_scope_test, in_scope_ids, names, rng)
    except Exception as e:
        print(f"WARNING: skipping null_nearmiss_clinc ({e})")
    try:
        eval_sets["null_nearmiss_hwu64"] = build_null_nearmiss("null_nearmiss_hwu64", hrows, list(range(len(hnames))), hnames, rng)
    except Exception as e:
        print(f"WARNING: skipping null_nearmiss_hwu64 ({e})")
    try:
        eval_sets["null_nearmiss_banking77"] = build_null_nearmiss("null_nearmiss_banking77", brows, list(range(len(bnames))), bnames, rng)
    except Exception as e:
        print(f"WARNING: skipping null_nearmiss_banking77 ({e})")

    # ================= Leak resolution =================
    print("=== Leak resolution ===")
    # Protected eval sets (held-out-entirely + ones whose HF splits reuse state text by
    # design) must stay intact -> drop the overlapping TRAIN rows instead.
    protected_states = set()
    for name in PROTECTED_EVAL:
        protected_states |= {norm_text(r["state"]) for r in eval_sets.get(name, [])}
    before = len(train_examples)
    train_examples = [r for r in train_examples if norm_text(r["state"]) not in protected_states]
    print(f"  dropped {before - len(train_examples)} train rows overlapping protected eval states")

    # Near-duplicate leakage that exact state matching above can't see (runs/leak_audit.json:
    # boolq_val 4.9%, hwu64_test 5.4% near-identical twins in train). Drop the TRAIN side.
    train_examples = prune_near_dup_train(
        train_examples, {norm_text(r["state"]) for r in eval_sets.get("boolq_val", [])}, {"boolq_prop", "boolq_pair"})
    train_examples = prune_near_dup_train(
        train_examples, {norm_text(r["state"]) for r in eval_sets.get("hwu64_test", [])}, {"hwu64"})

    # Every other eval set: drop rows whose normalised state also occurs in train
    # (e.g. CLINC's own train/test splits reuse identical utterances across intents).
    train_states = {norm_text(r["state"]) for r in train_examples}
    for name, rows in eval_sets.items():
        if name in PROTECTED_EVAL:
            continue
        kept = [r for r in rows if norm_text(r["state"]) not in train_states]
        if len(kept) != len(rows):
            print(f"  dedupe {name}: dropped {len(rows) - len(kept)}/{len(rows)}")
        eval_sets[name] = kept

    # Hypothesis-only probe: same (deduped) rows as snli_test but with the premise blanked
    # out, so a model scoring above chance here is reading the hypothesis alone rather than
    # the premise. Built from the post-dedup snli_test so row counts match exactly; appended
    # last and uses no RNG, so it can't shift any RNG draw made by an earlier eval set.
    eval_sets["snli_test_hyponly"] = [
        {**r, "state": ".", "task": "snli_test_hyponly"} for r in eval_sets["snli_test"]
    ]

    print_k_null_table(train_examples, "train")

    # ================= Write =================
    val_size = 100 if small else 4000
    train, val = group_split(train_examples, val_size, rng)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    eval_paths = []
    for name, rows in eval_sets.items():
        p = out / "eval" / f"{name}.jsonl"
        write_jsonl(p, rows)
        eval_paths.append(p)

    summarize([out / "train.jsonl", out / "val.jsonl"] + sorted(eval_paths))
    if not selftest(out):
        sys.exit(1)


def selftest(out_dir):
    """Schema + leak audit (PLAN2 "Leak audit"). Prints a pass/fail per check; fails loudly."""
    print(f"\n=== Self-test: {out_dir} ===")
    ok = True
    train_path = out_dir / "train.jsonl"
    if not train_path.exists():
        print(f"FAIL: {train_path} missing")
        return False

    def load(p):
        return [json.loads(l) for l in open(p)]

    eval_paths = sorted((out_dir / "eval").glob("*.jsonl"))
    train_rows = load(train_path)

    for p in [train_path, out_dir / "val.jsonl"] + eval_paths:
        if not p.exists():
            continue
        for r in load(p):
            if len(r["target"]) != len(r["candidates"]):
                print(f"FAIL schema {p.name}: target/candidates length mismatch"); ok = False
            if r["target"] and abs(sum(r["target"]) - 1.0) > 1e-6:
                print(f"FAIL schema {p.name}: target sums to {sum(r['target'])}"); ok = False
            if not (0.0 <= r["p_null"] <= 1.0):
                print(f"FAIL schema {p.name}: p_null={r['p_null']} out of range"); ok = False
            lbl = r.get("label")
            if isinstance(lbl, int) and lbl >= 0 and r["target"]:
                amax = max(range(len(r["target"])), key=lambda i: r["target"][i])
                if amax != lbl:
                    print(f"FAIL schema {p.name}: label {lbl} != argmax {amax}"); ok = False

    new_eval_sets = {"cse_clinc", "cse_hwu64", "cse_banking77", "cse_snli", "ksweep_clinc",
                      "ksweep_banking77", "null_irrq_squad", "null_irrq_intent",
                      "null_nearmiss_clinc", "null_nearmiss_hwu64", "null_nearmiss_banking77"}
    present = {p.stem for p in eval_paths}
    for name in new_eval_sets:
        if name in present:
            n = sum(1 for _ in open(out_dir / "eval" / f"{name}.jsonl"))
            if n == 0:
                print(f"FAIL: {name}.jsonl is empty"); ok = False
    missing = new_eval_sets - present
    if missing:
        print(f"  (data v4 eval sets not built this run, presumably a source failed to load: {sorted(missing)})")

    # noev rows deliberately reuse the same "." placeholder as the hyponly probe (item 4); that's
    # not a real leak (there's no information in "."), so exclude both from the overlap check.
    train_states = {norm_text(r["state"]) for r in train_rows
                     if r["task"] not in NO_STATE_TASKS and not r["task"].endswith("_noev")}
    for p in eval_paths:
        states = {norm_text(r["state"]) for r in load(p)}
        overlap = train_states & states
        if overlap:
            print(f"FAIL leak {p.name}: {len(overlap)} state(s) shared with train"); ok = False

    train_nli_pairs = {(norm_text(r["state"]), norm_text(r["meta"]["hyp"]))
                        for r in train_rows if r.get("meta", {}).get("hyp")}
    for p in eval_paths:
        rows = load(p)
        pairs = {(norm_text(r["state"]), norm_text(r["meta"]["hyp"]))
                 for r in rows if r.get("meta", {}).get("hyp")}
        overlap = train_nli_pairs & pairs
        if overlap:
            print(f"FAIL leak {p.name}: {len(overlap)} (premise,hyp) pair(s) shared with train"); ok = False

    train_queries = {r["query"] for r in train_rows}
    for fam, tpls in TEMPLATES.items():
        for idx in (8, 9):
            prefix = tpls[idx].split("{")[0].strip()
            if prefix and any(q.startswith(prefix) for q in train_queries):
                print(f"FAIL leak: qpara template ({fam},{idx}) found in train queries"); ok = False

    para_words = {w for tup in PARAPHRASES for w in tup}
    train_nli_cands = {c for r in train_rows if r.get("meta", {}).get("family") == "nli" for c in r["candidates"]}
    if para_words & train_nli_cands:
        print("FAIL leak: PARAPHRASES wordings leaked into train candidates"); ok = False

    print("Self-test PASSED" if ok else "Self-test FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="cap each source at 200 rows (100 for eval) -> data_small/")
    ap.add_argument("--selftest", action="store_true", help="only run the leak audit on an existing data dir")
    ap.add_argument("--out", default=None, help="output dir (default: data_small/ with --small, else data/)")
    ap.add_argument("--cls_cap", type=int, default=None,
                     help="cap each SIMPLE_SOURCES/go_emotions classification source at N rows (v5: more "
                          "label vocabularies, fewer rows each); CLINC is untouched. Default: uncapped.")
    args = ap.parse_args()
    if args.selftest:
        ok = selftest(Path(args.out) if args.out else (Path("data_small") if args.small else DATA))
        sys.exit(0 if ok else 1)
    main(small=args.small, out_name=args.out, cls_cap=args.cls_cap)
