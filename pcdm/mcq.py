"""MCQ-LoRA readout: identical backbone/LoRA/data as the energy-head pipeline, but the
answer is read from next-token logits over option letters instead of a learned decision
tower. See REVIEW.md sec 3.

Prompt: [eos] + state + "\n" + query + "\n" + "A. opt0\nB. opt1\n...\n<L>. none of the
above\nAnswer:" -> next-token logits restricted to the option letters, null letter last,
so decision_loss/summarize (which expect [B, K+1] with null last) work unchanged.

K > MAXK_DIRECT (more single-token letters than exist): score in chunks of CHUNK options
+ "none of the above" each, keep each chunk's winner, then re-score the winners against
each other. ponytail: two-stage heuristic, not a true top-1-over-K argmax; upgrade to a
listwise/tournament scheme if this is shown to lose accuracy at large K.
"""
import string

import torch
import torch.nn as nn

from encode import Backbone
from model import collate_teacher

LETTERS = string.ascii_uppercase + string.ascii_lowercase  # 52 single-token letters (leading space) on the Qwen3 vocab
MAXK_DIRECT = len(LETTERS) - 1  # one letter reserved for "none of the above"
CHUNK = 50  # + 1 null letter = 51 = MAXK_DIRECT, per first-stage chunk

# PLAN3 Task C: 5 hand-written generic MCQs for --shots, letting a *base* model (no
# instruction tuning) pick up the "Answer: <letter>" format zero-shot can't teach it.
FEWSHOT_EXAMPLES = [
    {"question": "What is the capital of France?", "options": ["Paris", "London", "Berlin", "Madrid"], "answer": 0},
    {"question": "Which planet is known as the Red Planet?", "options": ["Venus", "Mars", "Jupiter", "Saturn"], "answer": 1},
    {"question": "What is 7 + 5?", "options": ["10", "11", "12", "13"], "answer": 2},
    {"question": "Which gas do plants absorb from the air to make food?", "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], "answer": 2},
    {"question": "Who wrote 'Romeo and Juliet'?", "options": ["Charles Dickens", "William Shakespeare", "Mark Twain", "Jane Austen"], "answer": 1},
]


def build_shots(n: int) -> str:
    """First n of FEWSHOT_EXAMPLES, formatted identically to the eval prompt (question,
    lettered options, "Answer: X"), blank-line separated. n<=0 -> "" so the zero-shot
    path (--shots 0, the default) is byte-for-byte unchanged."""
    if n <= 0:
        return ""
    blocks = []
    for ex in FEWSHOT_EXAMPLES[:n]:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(ex["options"]))
        blocks.append(f"{ex['question']}\n{opts}\nAnswer: {LETTERS[ex['answer']]}")
    return "\n\n".join(blocks) + "\n\n"


class MCQHead(nn.Module):
    """Backbone (frozen + top-layer LoRA, identical to the energy head) + a frozen LM
    head, restricted at read time to the single-token option letters."""

    def __init__(self, name: str, lora_layers: int = 8, lora_r: int = 16, device: str = "auto", tap_layer: int = 0,
                 dtype: torch.dtype = torch.bfloat16, multimodal: bool = False,
                 max_image_pixels: int = 262144):
        super().__init__()
        self.backbone = Backbone(name, lora_layers=lora_layers, lora_r=lora_r, device=device, tap_layer=tap_layer,
                                 dtype=dtype, multimodal=multimodal, max_image_pixels=max_image_pixels)
        self.device = self.backbone.device
        if self.backbone.model.config.tie_word_embeddings:
            weight = self.backbone.model.embed_tokens.weight  # tied -> no second model load
        else:
            # untied (e.g. Qwen3-8B): the base AutoModel has no lm_head, so pull it from
            # a CausalLM load and drop the rest immediately (transient 2x memory).
            from transformers import AutoModelForCausalLM
            causal = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
            weight = causal.lm_head.weight.detach().clone().to(self.device)
            del causal
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
        self.lm_head_weight = weight
        tok = self.backbone.tokenizer
        # the token that follows "Answer:" naturally carries a leading space; verified
        # single-token for A-Z/a-z (not 0-9) on the Qwen3 vocab.
        ids = [tok.encode(" " + c, add_special_tokens=False) for c in LETTERS]
        assert all(len(i) == 1 for i in ids), "LETTERS assumption broken for this tokenizer"
        self.letter_ids = torch.tensor([i[0] for i in ids], dtype=torch.long)

    def trainable_parameters(self):
        return self.backbone.trainable_parameters()

    def lora_state_dict(self):
        return self.backbone.lora_state_dict()

    def load_lora_state_dict(self, sd):
        self.backbone.load_lora_state_dict(sd)


def _label(i: int) -> str:
    # ponytail: past the 52 single-token letters (only reachable via --readout native, which never
    # reads letters out) fall back to a 1-based number rather than wrapping the alphabet.
    return LETTERS[i] if i < len(LETTERS) else str(i + 1)


def _render(query: str, cand_texts: list[str]):
    """-> (suffix text, char spans): query, one "<label>. text" line per option, the
    "none of the above" line, "Answer:". spans[k] = (start, end) of option k's text
    (letters excluded); spans[-1] is the null line's text -- pcdm/native.py pools these."""
    text, spans = query + "\n", []
    for i, c in enumerate(cand_texts + ["none of the above"]):
        text += f"{_label(i)}. "
        spans.append((len(text), len(text) + len(c)))
        text += c + "\n"
    return text + "Answer:", spans


def _build_suffix(query: str, cand_texts: list[str]) -> str:
    return _render(query, cand_texts)[0]


def _ids(tok, texts: list[str], max_len: int, offsets: bool = False):
    """Token ids (no specials, truncated), optionally with char offsets; empty text -> "."."""
    texts = [t if t.strip() else "." for t in texts]
    enc = tok(texts, truncation=True, max_length=max_len, add_special_tokens=False, return_offsets_mapping=offsets)
    return (enc["input_ids"], enc["offset_mapping"]) if offsets else enc["input_ids"]


def _pack(tok, s_ids, x_ids, tail=(), sink=True):
    """rows = [eos] + state + suffix + tail, right-padded -> (input_ids, attention_mask,
    lengths); lengths includes eos/tail, so the last real position of row i is lengths[i] - 1.
    sink=False drops the leading eos (native_kv_decide: the sink lives in the cached prefix)."""
    eos, pad = tok.eos_token_id, tok.pad_token_id
    rows = [([eos] if sink else []) + s + x + list(tail) for s, x in zip(s_ids, x_ids)]
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    T = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), T), pad, dtype=torch.long)
    # bool, not long: transformers' SDPA mask prep keeps a boolean 2D padding mask on the fast
    # (flash/mem-efficient) path; a long or float mask gets converted to a 4D additive float
    # bias, which forces the O(B*H*L^2) "math" backend -- material at L=3072 (tl1b's --max_state,
    # long-state OOM investigation, 2026-09-21).
    attention_mask = torch.zeros(len(rows), T, dtype=torch.bool)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        attention_mask[i, :len(r)] = True
    return input_ids, attention_mask, lengths


def _tokenize_mcq(backbone, states: list[str], suffixes: list[str], max_state: int, max_suffix: int):
    """ids = [eos] + state_tokens(<=max_state) + suffix_tokens(<=max_suffix), right-padded."""
    tok = backbone.tokenizer
    return _pack(tok, _ids(tok, states, max_state), _ids(tok, suffixes, max_suffix))


def _score_batch_direct(head: MCQHead, states, queries, cand_lists, max_state, max_suffix, images=None):
    """cand_lists: list of candidate-text lists, each len <= MAXK_DIRECT.
    -> (logits [B, Kmax+1] on head.device, n_tokens int); null last, pad = finfo.min."""
    dev = head.device
    B = len(states)
    Kmax = max(len(c) for c in cand_lists)
    suffixes = [_build_suffix(q, c) for q, c in zip(queries, cand_lists)]
    if images is None:
        input_ids, attention_mask, lengths = _tokenize_mcq(head.backbone, states, suffixes, max_state, max_suffix)
        input_ids, attention_mask = input_ids.to(dev), attention_mask.to(dev)
        out = head.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        if len(images) != B or any(image is None for image in images):
            raise ValueError("visual batches require exactly one image per row")
        prefix = head.backbone.visual_prefix(states, images, max_state)
        rows = [ids[mask.bool()].tolist() for ids, mask in zip(prefix["input_ids"], prefix["attention_mask"])]
        types = [ids[mask.bool()].tolist() for ids, mask in zip(prefix["mm_token_type_ids"],
                                                                 prefix["attention_mask"])]
        rows = [[head.backbone.tokenizer.eos_token_id] + row for row in rows]
        types = [[0] + row for row in types]
        suffix_ids = _ids(head.backbone.tokenizer, suffixes, max_suffix)
        input_ids, attention_mask, lengths = _pack(head.backbone.tokenizer, rows, suffix_ids, sink=False)
        mm_types = torch.zeros_like(input_ids)
        for i, (prefix_types, suffix) in enumerate(zip(types, suffix_ids)):
            mm_types[i, :len(prefix_types) + len(suffix)] = torch.tensor(prefix_types + [0] * len(suffix))
        out = head.backbone.multimodal_model(
            input_ids=input_ids.to(dev), attention_mask=attention_mask.to(dev),
            mm_token_type_ids=mm_types.to(dev), pixel_values=prefix["pixel_values"].to(dev),
            image_grid_thw=prefix["image_grid_thw"].to(dev))
    last = out.last_hidden_state[torch.arange(B, device=dev), (lengths - 1).to(dev)]  # [B, d]

    letter_ids = head.letter_ids.to(dev)
    w = head.lm_head_weight.to(dev)[letter_ids].float()  # [52, d] -- indexed before matmul, never the full vocab
    letter_logits = last.float() @ w.T  # [B, 52]

    logits = torch.full((B, Kmax + 1), torch.finfo(torch.float32).min, device=dev)
    for i, c in enumerate(cand_lists):
        k = len(c)
        logits[i, :k] = letter_logits[i, :k]
        logits[i, -1] = letter_logits[i, k]  # this row's null letter is LETTERS[k]
    return logits, int(attention_mask.sum().item())


def _score_chunked(score_fn, cand_texts, idx_chunks):
    """Two-stage: score_fn(list of option-text lists) -> (logits [n, Kc+1] with null last,
    n_tokens). Stage 1 scores every chunk (+null) in one call, keeps each chunk's winner;
    stage 2 re-scores the winners against each other. -> (logits [K+1], n_tokens).
    Hierarchical distribution (finite, differentiable for every candidate):
      log P(j) = log P2(chunk(j) wins) + log P1(j | its chunk);  log P(null) = log P2(none).
    Stage-1 per chunk: log-softmax over the chunk's K_c options (null column excluded);
    stage-2: log-softmax over the chunk winners + none.  Using finfo.min for losers made
    soft-CE infinite whenever the gold lost its chunk (seen at step 500 of mcq_lora).
    ponytail: two stages only -- stage 2 assumes the winners fit one call."""
    stage1, n_tok1 = score_fn([[cand_texts[i] for i in idxs] for idxs in idx_chunks])
    n = len(idx_chunks)
    winner_idx = [idxs[int(stage1[row, :len(idxs)].argmax())] for row, idxs in enumerate(idx_chunks)]
    stage2, n_tok2 = score_fn([[cand_texts[i] for i in winner_idx]])

    out = torch.empty(len(cand_texts) + 1, device=stage1.device)
    lp2 = torch.log_softmax(stage2[0, :n + 1], dim=-1)          # [n chunks + none]
    for row, idxs in enumerate(idx_chunks):
        lp1 = torch.log_softmax(stage1[row, :len(idxs)], dim=-1)
        for j, orig in enumerate(idxs):
            out[orig] = lp2[row] + lp1[j]
    out[-1] = lp2[n]
    return out, n_tok1 + n_tok2


def _score_one_chunked(head: MCQHead, state, query, cand_texts, max_state, max_suffix, image=None):
    """CHUNK-sized chunks through _score_batch_direct, composed by _score_chunked."""
    idx_chunks = [list(range(i, min(i + CHUNK, len(cand_texts)))) for i in range(0, len(cand_texts), CHUNK)]
    score_fn = lambda cl: _score_batch_direct(head, [state] * len(cl), [query] * len(cl), cl, max_state,
                                               max_suffix, [image] * len(cl) if image is not None else None)
    return _score_chunked(score_fn, cand_texts, idx_chunks)


def collate_mcq(examples):
    """cmask/target/p_null only -- no cached candidate features (MCQ reads candidate
    text straight into the prompt). Mirrors model.collate's cmask/target/p_null build;
    duplicated rather than shared since pcdm/model.py is frozen for this change. teacher/
    has_teacher (PLAN3 E3-T soft labels, scripts/teacher_label.py) ARE shared, via
    model.collate_teacher -- see decision_loss."""
    Kmax = max(len(ex["candidates"]) for ex in examples)
    cmask = torch.zeros(len(examples), Kmax, dtype=torch.bool)
    target = torch.zeros(len(examples), Kmax, dtype=torch.float32)
    p_null = torch.zeros(len(examples), dtype=torch.float32)
    for i, ex in enumerate(examples):
        K = len(ex["candidates"])
        cmask[i, :K] = True
        target[i, :K] = torch.tensor(ex["target"], dtype=torch.float32)
        p_null[i] = ex["p_null"]
    teacher, has_teacher = collate_teacher(examples, Kmax)
    return {"cmask": cmask, "target": target, "p_null": p_null, "teacher": teacher, "has_teacher": has_teacher}


def run_batch_mcq(head: MCQHead, batch, examples, max_state: int = 256, max_suffix: int = 512, shots: int = 0):
    """-> logits [B, Kmax+1] (Kmax = batch["cmask"].shape[1]); sets batch["n_tokens"].
    shots>0: prepend build_shots(shots) to every state; max_state grows by the prefix's
    own token count so the real question keeps its full max_state budget (the prefix is
    fixed text, so truncation would otherwise eat into it first)."""
    prefix = build_shots(shots)
    states = [prefix + ex["state"] for ex in examples]
    queries = [ex["query"] for ex in examples]
    cand_lists = [ex["candidates"] for ex in examples]
    images = [ex.get("image") for ex in examples]
    if prefix:
        max_state += len(head.backbone.tokenizer.encode(prefix, add_special_tokens=False))
    Kmax = batch["cmask"].shape[1]
    dev = head.device
    logits = torch.full((len(examples), Kmax + 1), torch.finfo(torch.float32).min, device=dev)
    n_tokens = 0

    direct_i = [i for i, c in enumerate(cand_lists) if len(c) <= MAXK_DIRECT]
    chunk_i = [i for i, c in enumerate(cand_lists) if len(c) > MAXK_DIRECT]

    for visual in (False, True):
        idx = [i for i in direct_i if bool(images[i]) == visual]
        if not idx:
            continue
        sub, n_tok = _score_batch_direct(head, [states[i] for i in idx], [queries[i] for i in idx],
                                          [cand_lists[i] for i in idx], max_state, max_suffix,
                                          [images[i] for i in idx] if visual else None)
        n_tokens += n_tok
        for row, i in enumerate(idx):
            k = len(cand_lists[i])
            logits[i, :k] = sub[row, :k]
            logits[i, -1] = sub[row, -1]

    for i in chunk_i:
        row, n_tok = _score_one_chunked(head, states[i], queries[i], cand_lists[i], max_state, max_suffix, images[i])
        n_tokens += n_tok
        k = len(cand_lists[i])
        logits[i, :k] = row[:k]
        logits[i, -1] = row[-1]

    batch["n_tokens"] = n_tokens
    return logits
