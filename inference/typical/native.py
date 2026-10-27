"""native_choice_v1/v2/v3 decision head + the KV-cached serving path -- trimmed port of the
training repo's pcdm/native.py + the two pcdm/mcq.py helpers it needs (_label/_ids/_pack), for
inference only. Kept: NativeHead (factored null, per-row Bernoulli "noul" routing via
_is_bern_row, the letters/tags/letters_nonull/query_only renderers) and native_kv_decide,
because pcdm_jev.decider.PCDMDecider(mode="native") calls exactly those two things and
nothing else. Dropped: run_batch_native/native_features/_score_chunked/_fit_chunks (the
training/eval batch path -- decider.py never calls them) and the n2 (Qwen3-Embedding
candidate-vector) branch, since neither shipped checkpoint (typical-small,
typical-small-preview) trains with nc_head n2/n2n3.
"""
import bisect
import copy
import functools
import json
import string

import torch
from torch.profiler import record_function
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 1024  # Qwen3-Embedding-0.6B hidden size; only referenced by the (unsupported) n2 branch
MAX_SUFFIX = 1024
CHOICE_OPEN, CHOICE_CLOSE = "<choice>\n", "\n</choice>\n"
LETTERS = string.ascii_uppercase + string.ascii_lowercase


def _label(i: int) -> str:
    return LETTERS[i] if i < len(LETTERS) else str(i + 1)


def _render(query: str, cand_texts: list[str]):
    """native_v1 (--nc_render letters): query, lettered options, a rendered "none of the
    above" line, "Answer:". spans[k] = (start, end) of option k's text; spans[-1] is the
    null line's text."""
    text, spans = query + "\n", []
    for i, c in enumerate(cand_texts + ["none of the above"]):
        text += f"{_label(i)}. "
        spans.append((len(text), len(text) + len(c)))
        text += c + "\n"
    return text + "Answer:", spans


def _render_tags(query: str, cand_texts: list[str]):
    """native_v2 (--nc_render tags): letter-free, one <choice>...</choice> block per option,
    no null line, no "Answer:". len(spans) == K (the null head reads a learned constant,
    not a rendered line)."""
    text, spans = query + "\n", []
    for c in cand_texts:
        text += CHOICE_OPEN
        spans.append((len(text), len(text) + len(c)))
        text += c + CHOICE_CLOSE
    return text, spans


def _render_letters_nonull(query: str, cand_texts: list[str]):
    """native_v3 (--nc_render letters_nonull): lettered options (keeps slot identity) but no
    rendered null line and no "Answer:" -- both shipped checkpoints train with this render."""
    text, spans = query + "\n", []
    for i, c in enumerate(cand_texts):
        text += f"{_label(i)}. "
        spans.append((len(text), len(text) + len(c)))
        text += c + "\n"
    return text, spans


def _render_query_only(query: str, cand_texts: list[str]):
    """--noul_head bern: suffix is the query alone (the Bernoulli head reads h_D only, never
    the candidate text -- that's what makes it label-order invariant). Empty spans."""
    return query + "\n", []


# Verbatim from pcdm_jev.decider.SEMIF_SYSTEM / native.SEMIF_SYSTEM (SemIf's core.py DIRECT_SYSTEM).
SEMIF_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                "Respond with only its uppercase letter, with no explanation or reasoning.")


@functools.lru_cache(maxsize=4)
def _semif_head_tail(tok):
    """(pre_head, tail): chat-template text around a sentinel "evidence" value -- pre_head =
    system turn + user-turn opening + `{"evidence": "` (prepended to the state text to form the
    KV-cached prefix), tail = end of user turn + assistant-turn opening (appended after the rest
    of the JSON, see _render_semif). Verbatim port of native._semif_head_tail."""
    sentinel = "\x00SEMIF_STATE\x00"
    messages = [{"role": "system", "content": SEMIF_SYSTEM}, {"role": "user", "content": sentinel}]
    try:
        full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    before, after = full.split(sentinel)
    return before + '{"evidence": "', after


def _render_semif(query: str, cand_texts: list[str], tail_text: str):
    """native_v4 (--nc_render semif): closes the "evidence" string opened in the cached prefix,
    then criterion + lettered options (json.dumps-escaped) + tail_text. spans[k] = (start, end)
    of option k's description text -- K spans, no null line. Verbatim port of native._render_semif."""
    text = '", "criterion": ' + json.dumps(query) + ', "options": ['
    spans = []
    for i, c in enumerate(cand_texts):
        if i:
            text += ", "
        text += '{"letter": "' + _label(i) + '", "description": "'
        esc = json.dumps(c)[1:-1]
        spans.append((len(text), len(text) + len(esc)))
        text += esc + '"}'
    text += ']}' + tail_text
    return text, spans


RENDERS = {"letters": _render, "tags": _render_tags, "letters_nonull": _render_letters_nonull,
           "query_only": _render_query_only, "semif": _render_semif}


def _render_row(tok, render: str, query: str, cand_texts: list[str]):
    """Dispatch RENDERS[render](query, cand_texts) for every render except "semif", which also
    needs the tokenizer's chat-template tail text (_semif_head_tail). Verbatim port of
    native._render_row."""
    if render == "semif":
        return _render_semif(query, cand_texts, _semif_head_tail(tok)[1])
    return RENDERS[render](query, cand_texts)


def _is_bern_row(cand_texts, meta=None):
    """A row is bern-eligible iff its candidate set is exactly {"yes","no"} (case-insensitive),
    or meta.qtype == "noul" and K == 2."""
    if {c.strip().lower() for c in cand_texts} == {"yes", "no"}:
        return True
    return bool(meta) and meta.get("qtype") == "noul" and len(cand_texts) == 2


def _yes_idx(cand_lists, dev):
    """[B] index of the literal "yes" candidate per row (case-insensitive) -- places P(yes)
    there regardless of rendered order, so ["yes","no"] and ["no","yes"] give identical
    P(yes). No literal "yes" -> defaults to index 0 (out-of-domain K-way row)."""
    idx = []
    for cl in cand_lists:
        matches = [i for i, c in enumerate(cl) if c.strip().lower() == "yes"]
        idx.append(matches[0] if matches else 0)
    return torch.tensor(idx, dtype=torch.long, device=dev)


def _ids(tok, texts: list[str], max_len: int, offsets: bool = False):
    """Token ids (no specials, truncated), optionally with char offsets; empty text -> "."."""
    texts = [t if t.strip() else "." for t in texts]
    enc = tok(texts, truncation=True, max_length=max_len, add_special_tokens=False, return_offsets_mapping=offsets)
    return (enc["input_ids"], enc["offset_mapping"]) if offsets else enc["input_ids"]


def _pack(tok, s_ids, x_ids, tail=(), sink=True):
    """rows = [eos] + state + suffix + tail, right-padded -> (input_ids, attention_mask,
    lengths); sink=False drops the leading eos (native_kv_decide: the sink lives in the
    cached prefix)."""
    eos, pad = tok.eos_token_id, tok.pad_token_id
    rows = [([eos] if sink else []) + s + x + list(tail) for s, x in zip(s_ids, x_ids)]
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    T = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), T), pad, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), T, dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        attention_mask[i, :len(r)] = 1
    return input_ids, attention_mask, lengths


def factored_null_logits(gate, s, c, h, cmask, temperature: float = 1.0):
    """r = sigmoid(gate(z)): z is O(K) permutation-invariant set statistics over the valid
    candidate scores s + rep h + masked candidate-set mean/var of c. Composed as logits so
    softmax(logits) gives P(a_j) = (1-r)*p_j and P(null) = r exactly. Verbatim port of
    model.factored_null_logits (self-contained: only torch/F, no other pcdm/model.py state)."""
    K = cmask.sum(-1)
    Kf = K.float().clamp_min(1)
    top2 = torch.topk(s, k=min(2, s.shape[1]), dim=-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[1] == 2 else torch.zeros_like(top2[:, 0])
    margin = torch.where(K > 1, margin, torch.zeros_like(margin))
    mean_s = s.masked_fill(~cmask, 0.0).sum(-1) / Kf
    lse = torch.logsumexp(s, dim=-1) - torch.log(Kf)
    c0 = c.masked_fill(~cmask.unsqueeze(-1), 0.0)
    mean_c = c0.sum(1) / Kf.unsqueeze(-1)
    var_c = (c0 - mean_c.unsqueeze(1)).pow(2).masked_fill(~cmask.unsqueeze(-1), 0.0).sum(1) / Kf.unsqueeze(-1)
    z = torch.cat([top2[:, :1], margin.unsqueeze(-1), mean_s.unsqueeze(-1), lse.unsqueeze(-1),
                   h, mean_c, var_c], dim=-1)
    r_logit = gate(z).squeeze(-1)
    log_1mr, log_r = -F.softplus(r_logit), -F.softplus(-r_logit)
    s_t = (s.masked_fill(~cmask, 0.0) / temperature).masked_fill(~cmask, torch.finfo(s.dtype).min)
    logp = F.log_softmax(s_t, dim=-1)
    cand_logits = (logp + log_1mr.unsqueeze(-1)).masked_fill(~cmask, torch.finfo(s.dtype).min)
    return torch.cat([cand_logits, log_r.unsqueeze(-1)], dim=-1)


class NativeHead(nn.Module):
    """Verbatim port of native.NativeHead. n2 (nc_head in {"n2","n2n3"}) is accepted for
    state_dict shape compatibility but native_kv_decide below raises if a loaded checkpoint
    actually needs it -- see module docstring."""

    def __init__(self, d: int, nc_head: str = "n2n3", null: str = "factored", d_proj: int = 256,
                 d_emb: int = EMBED_DIM, render: str = "letters", score_head: str = "choice",
                 noul_head: str = "choice"):
        super().__init__()
        assert nc_head in ("n2", "n3", "n2n3"), nc_head
        assert null in ("softmax", "factored"), null
        assert render in RENDERS, render
        assert score_head in ("choice", "cumlink"), score_head
        assert noul_head in ("choice", "bern"), noul_head
        self.nc_head, self.null, self.render = nc_head, null, render
        self.score_head, self.noul_head = score_head, noul_head
        self.use2, self.use3 = "n2" in nc_head, "n3" in nc_head
        dv = d_proj * (self.use2 + self.use3)
        self.dv = dv
        self.register_buffer("mu_h", torch.zeros(d)); self.register_buffer("sd_h", torch.ones(d))
        self.register_buffer("mu_c", torch.zeros(d)); self.register_buffer("sd_c", torch.ones(d))
        self.proj_h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dv))
        if self.use2:
            self.proj_2 = nn.Sequential(nn.LayerNorm(d_emb), nn.Linear(d_emb, d_proj))
            self.null_emb = nn.Parameter(torch.randn(d_emb) * 0.02)
        if self.use3:
            self.proj_3 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_proj))
        self.w = nn.Linear(dv, 1)
        self.res = nn.Sequential(nn.Linear(dv, dv), nn.GELU(), nn.Linear(dv, 1))
        if null == "factored":
            self.null_gate = nn.Sequential(nn.Linear(4 + 4 * dv, 256), nn.GELU(), nn.Linear(256, 1))
        else:
            self.score_null = nn.Sequential(nn.Linear(2 * dv, dv // 2), nn.GELU(), nn.Linear(dv // 2, 1))
        if score_head == "cumlink" or noul_head == "bern":
            self.null_gate_typed = nn.Sequential(nn.Linear(4 + 4 * d, 256), nn.GELU(), nn.Linear(256, 1))
        if score_head == "cumlink":
            self.w_o = nn.Linear(d, 1)
            self.w_t = nn.Linear(d, 1)
            self.theta0 = nn.Parameter(torch.zeros(1))
        if noul_head == "bern":
            self.w_n = nn.Linear(d, 1)
        self.temperature = 1.0

    def forward(self, h, cmask, c3=None, c2=None, yes_idx=None, bern_mask=None):
        """h [B, d]; cmask [B, Kmax]; c3 [B, Kmax+1, d] (null line last); c2 [B, Kmax, d_emb].
        -> logits [B, Kmax+1], null last, pads finfo.min. Verbatim port of native.NativeHead.forward."""
        hz = (h - self.mu_h) / self.sd_h
        if self.score_head == "cumlink":
            return self._typed_logits(self._cumlink_probs(hz, c3, cmask), hz, c3, cmask)
        bm = None
        if self.noul_head == "bern":
            bm = bern_mask if bern_mask is not None else torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
            if bm.all():
                return self._typed_logits(self._bern_probs(hz, cmask, yes_idx), hz, c3, cmask)

        B = h.shape[0]
        u = self.proj_h(hz)
        vs = []
        if self.use2:
            vs.append(self.proj_2(torch.cat([c2, self.null_emb.expand(B, 1, -1)], 1)))
        if self.use3:
            vs.append(self.proj_3((c3 - self.mu_c) / self.sd_c))
        v = torch.cat(vs, -1)
        vc, vn = v[:, :-1], v[:, -1]
        uv = u.unsqueeze(1) * vc
        s = uv.sum(-1) * self.dv ** -0.5 + self.w(uv).squeeze(-1) + self.res(uv).squeeze(-1)
        s = s.masked_fill(~cmask, torch.finfo(s.dtype).min)
        hn = torch.cat([u, vn], -1)
        if self.null == "factored":
            choice_logits = factored_null_logits(self.null_gate, s, vc, hn, cmask, self.temperature)
        else:
            choice_logits = torch.cat([s, self.score_null(hn)], dim=-1)

        if bm is not None and bm.any():
            bern_logits = self._typed_logits(self._bern_probs(hz, cmask, yes_idx), hz, c3, cmask)
            return torch.where(bm.unsqueeze(-1), bern_logits, choice_logits)
        return choice_logits

    def _bern_probs(self, hz, cmask, yes_idx):
        B, Kmax = cmask.shape
        p_yes = torch.sigmoid(self.w_n(hz).squeeze(-1))
        idx = yes_idx if yes_idx is not None else torch.ones(B, dtype=torch.long, device=hz.device)
        K = cmask.sum(-1)
        other = (1 - p_yes) / (K - 1).clamp_min(1).float()
        p = other.unsqueeze(-1).expand(B, Kmax) * cmask.float()
        ar = torch.arange(B, device=hz.device)
        p[ar, idx] = p_yes
        return p.masked_fill(~cmask, 0.0)

    def _cumlink_probs(self, hz, c3, cmask):
        B, Kmax = cmask.shape
        c3z = (c3 - self.mu_c) / self.sd_c
        u = self.w_o(hz).squeeze(-1)
        width = F.softplus(self.w_t(c3z[:, :-1]).squeeze(-1)).masked_fill(~cmask, 0.0)
        theta = self.theta0 + torch.cumsum(width, dim=-1)
        cdf = torch.sigmoid(theta - u.unsqueeze(-1))
        p_prev = F.pad(cdf, (1, 0))[:, :-1]
        p = (cdf - p_prev).clamp_min(0.0)
        K = cmask.sum(-1)
        last = (K - 1).clamp_min(0)
        remainder = torch.where(K > 1, 1.0 - cdf.gather(1, (last - 1).clamp_min(0).unsqueeze(1)).squeeze(1),
                                 torch.ones_like(u))
        p = p.scatter(1, last.unsqueeze(1), remainder.unsqueeze(1)).masked_fill(~cmask, 0.0)
        return p

    def _typed_logits(self, p, hz, c3, cmask):
        c3z = (c3 - self.mu_c) / self.sd_c
        s = torch.log(p.clamp_min(1e-12)).masked_fill(~cmask, torch.finfo(p.dtype).min)
        hn = torch.cat([hz, c3z[:, -1]], dim=-1)
        return factored_null_logits(self.null_gate_typed, s, c3z[:, :-1], hn, cmask, self.temperature)


def _pool_matrix(spans, offsets, base, Kmax, T, K=None):
    """[Kmax+1, T] mean-pooling rows over each option's rendered char span (bisected against
    the tokenizer's char offsets)."""
    K = len(spans) - 1 if K is None else K
    starts, ends = [o[0] for o in offsets], [o[1] for o in offsets]
    pool = torch.zeros(Kmax + 1, T)
    for k, (cs, ce) in enumerate(spans):
        j0, j1 = bisect.bisect_right(ends, cs), bisect.bisect_left(starts, ce)
        pool[Kmax if k == K else k, base + j0:base + j1] = 1.0
    return pool / pool.sum(-1, keepdim=True).clamp_min(1.0)


def _expand_state_cache(cache, repeats: int):
    """Replaces the old `copy.deepcopy(state_cache)` + `_cache_batch_repeat_interleave`: a
    new Cache whose K/V tensors are `repeats`-wide views of `cache`'s (batch=1) K/V via
    `torch.Tensor.expand` -- stride-0 broadcast, zero bytes copied -- instead of a full deep
    copy of the (up to 1024-token) prefix followed by a real repeat_interleave copy on top.
    Safe because every attention layer's `.update()` (transformers' DynamicLayer et al.)
    does `self.keys = torch.cat([self.keys, key_states], dim=-2)`: a rebind to a brand-new
    tensor, never an in-place write to the old one, so the expanded view of the original
    prefix is read (by torch.cat, which handles stride-0 inputs like any other tensor -- no
    approximation, bit-identical to a materialised repeat) but never mutated. Only the outer
    Cache object and each layer object are shallow-copied (cheap: Python attribute dicts,
    not tensor storage).
    ponytail: Gated-DeltaNet (Qwen3.5 hybrid cache) linear-attention layers have no K/V
    tensors, only small conv_states/recurrent_states dicts (O(1) in sequence length) that
    genuinely get mutated by the recurrent update -- those still get a real
    repeat_interleave, just onto a copied dict so the write can't alias back into the
    original state_cache. Only plain K/V (DynamicLayer-family) and this dict-state shape are
    handled -- a future model with e.g. DynamicIndexedLayer's extra indexer_keys would need
    its own expand branch here."""
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        if hasattr(nl, "batch_repeat_interleave"):
            if nl.keys is not None and nl.keys.numel() > 0:
                nl.keys = nl.keys.expand(repeats, *nl.keys.shape[1:])
                nl.values = nl.values.expand(repeats, *nl.values.shape[1:])
        else:
            nl.conv_states = dict(layer.conv_states)
            nl.recurrent_states = dict(layer.recurrent_states)
            for i in range(getattr(nl, "number_of_states", 1)):
                if nl.is_conv_states_initialized[i]:
                    nl.conv_states[i] = nl.conv_states[i].repeat_interleave(repeats, dim=0)
                if nl.is_recurrent_states_initialized[i]:
                    nl.recurrent_states[i] = nl.recurrent_states[i].repeat_interleave(repeats, dim=0)
        new.layers.append(nl)
    return new


@functools.lru_cache(maxsize=64)
def _causal_template(q_len: int, kv_len: int, device_str: str):
    """[q_len, kv_len] boolean causal template (True = attend), cached per (q_len, kv_len,
    device) -- a pure function of shape, so a repeat call at the same (prefix_len,
    suffix_len) bucket reuses it instead of rebuilding the two arange()s + comparison."""
    device = torch.device(device_str)
    past_len = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= (past_len + q_idx)


@functools.lru_cache(maxsize=64)
def _position_ids_row(T: int, Ls: int, device_str: str):
    """[T] position ids (Ls, Ls+1, ..., Ls+T-1), cached per (T, Ls, device)."""
    return torch.arange(T, device=torch.device(device_str)) + Ls


@functools.lru_cache(maxsize=64)
def _sink_ones(m: int, Ls: int, device_str: str):
    """[m, Ls] all-ones long tensor (the cached prefix is always fully attended), cached per
    (m, Ls, device) -- avoids a fresh CPU allocation on every decision."""
    return torch.ones(m, Ls, dtype=torch.long, device=torch.device(device_str))


@functools.lru_cache(maxsize=64)
def _arange(n: int, device_str: str):
    return torch.arange(n, device=torch.device(device_str))


def _causal_pad_mask(attn2d, q_len, dtype):
    """Explicit additive (batch,1,q_len,kv_len) mask -- transformers' automatic 2D-mask +
    past_key_values path mis-handles a right-padded batch sharing one KV cache; this
    reproduces the un-batched per-row computation exactly."""
    batch, kv_len = attn2d.shape
    causal = _causal_template(q_len, kv_len, str(attn2d.device))
    allowed = causal.unsqueeze(0) & attn2d.bool().unsqueeze(1)
    mask = torch.zeros(batch, 1, q_len, kv_len, dtype=dtype, device=attn2d.device)
    return mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


DEFAULT_MAX_OPTION_TOKENS = 24


def _truncate_options(tok, cand_texts: list[str], max_tokens: int | None) -> list[str]:
    """Cap each rendered option at max_tokens tokens (append a visible "..." if cut) --
    "suffix diet": the K rendered options dominate suffix length, so a long option balloons
    every subsequent forward. None/<=0 disables (no-op, returns cand_texts unchanged)."""
    if not max_tokens or max_tokens <= 0:
        return cand_texts
    out = []
    for c in cand_texts:
        ids = tok(c, add_special_tokens=False)["input_ids"]
        out.append(tok.decode(ids[:max_tokens]) + "..." if len(ids) > max_tokens else c)
    return out


@torch.inference_mode()
def encode_state(head, model, state: str, max_state: int = 256, semif: bool | None = None):
    """[eos] + state (or, --nc_render semif, the SemIf chat-template wrapper + state -- see
    _semif_head_tail) -> (past_key_values, Ls), the once-per-state prefix encode that used to
    run inside native_kv_decide on every call. Split out so a caller (Typical's state LRU)
    can compute it once and reuse the KV cache across many decisions on the same state --
    see native_kv_decide's `state_cache` param. Pure function of (backbone weights, state
    text, semif): deterministic in eval/inference_mode, so a cached entry is bit-identical to a
    fresh encode, not just numerically close. @inference_mode here (not just relying on the
    caller) because the resulting KV tensors get `.expand()`'d and read (never in-place
    mutated) by every subsequent decision on this state (see `_expand_state_cache`) -- an
    inference_mode tensor is a stable, non-grad-tracked view source, so this must never run
    under regular autograd tracking, regardless of what context the caller happens to be in.
    semif=None defaults to `model.render == "semif"`; native_kv_decide passes semif=False
    explicitly for a bern (yes/no) row's own prefix, since those always use the plain eos+state
    sink regardless of nc_render (see pcdm/native.py's module docstring)."""
    if semif is None:
        semif = getattr(model, "render", "letters") == "semif"
    tok, dev, lm = head.backbone.tokenizer, head.device, head.backbone.model
    with record_function("typical/tokenize_state"):
        if semif:
            prefix_ids = _ids(tok, [_semif_head_tail(tok)[0] + state], max_state)[0]
        else:
            prefix_ids = [tok.eos_token_id] + _ids(tok, [state], max_state)[0]
        prefix = torch.tensor([prefix_ids], device=dev)
    with record_function("typical/state_prefix_forward"):
        cache = lm(input_ids=prefix, use_cache=True).past_key_values
    return cache, prefix.shape[1]


@torch.inference_mode()
def native_kv_decide(head, model, state, queries, chunk: int = 32, max_state: int = 256,
                     max_suffix: int = MAX_SUFFIX, vec_cache=None, state_cache=None,
                     max_option_tokens: int | None = DEFAULT_MAX_OPTION_TOKENS):
    """Serving path: [eos] + state (or, --nc_render semif, the SemIf chat-template wrapper +
    state) encoded ONCE into a KV cache on head.backbone.model; every query's suffix runs
    against that cached prefix, `chunk` queries per forward. queries = [(query, candidates)] ->
    list of [K_i + 1] probability vectors (null last). Verbatim port of native.native_kv_decide
    (head/model contract unchanged -- head needs only .backbone (tokenizer + model) and .device;
    model is a NativeHead), plus two additions: `state_cache`, an optional (past_key_values, Ls)
    pair from encode_state -- pass it to skip re-encoding the state prefix (Typical.choice/
    score/noul/decide do this automatically via an LRU; see core.py) -- and `max_option_tokens`
    ("suffix diet": caps each rendered option's length, see _truncate_options; None/0 disables,
    matching the training repo's untruncated behaviour exactly). No hierarchical fallback: a
    suffix longer than max_suffix raises (raise max_suffix instead).

    --nc_render semif: a bern (yes/no) row always renders query_only against the PLAIN eos+state
    prefix, never the semif JSON wrapper (see pcdm/native.py's module docstring), so it can't share
    `state_cache` when one is supplied (Typical's per-state LRU key/encode_state default to
    building the semif-wrapped cache whenever model.render=="semif" -- see encode_state). A
    call that mixes bern and non-bern queries under a semif checkpoint therefore builds a
    second, plain prefix on demand for the bern rows only; every other render (incl. no bern
    rows present) is the original single-cache loop, unchanged."""
    if model.use2:
        raise NotImplementedError(
            "nc_head n2/n2n3 (Qwen3-Embedding candidate vectors) isn't ported to the "
            "inference package -- neither typical-small nor typical-small-preview needs it. "
            "Port encode.EmbedEncoder + native._n2_vectors from the training repo if a future "
            "checkpoint requires it."
        )
    tok, dev, lm = head.backbone.tokenizer, head.device, head.backbone.model
    base_render = getattr(model, "render", "letters")
    # is_bern/yes-idx/cmask all key off candidate COUNT and the literal "yes"/"no" text (never
    # truncated -- 1 token), so they read the original `c`; only the rendered text is diet'd.
    is_bern = [model.noul_head == "bern" and _is_bern_row(c) for _, c in queries]
    renders = ["query_only" if b else base_render for b in is_bern]
    out = [None] * len(queries)

    def run_group(idxs, cache, Ls, tail_ids):
        if not idxs:
            return
        for start in range(0, len(idxs), chunk):
            grp = idxs[start:start + chunk]
            qs = [queries[i] for i in grp]
            rnds = [renders[i] for i in grp]
            m = len(qs)
            with record_function("typical/tokenize_suffix"):
                rendered = [_render_row(tok, r, q, _truncate_options(tok, c, max_option_tokens))
                            for r, (q, c) in zip(rnds, qs)]
                enc = tok([r[0] for r in rendered], add_special_tokens=False, return_offsets_mapping=True)
                x_ids, offs = enc["input_ids"], enc["offset_mapping"]
                assert max(len(x) for x in x_ids) <= max_suffix, f"suffix > max_suffix={max_suffix}; raise it"
                input_ids, am, lengths = _pack(tok, [[]] * m, x_ids, tail=tail_ids, sink=False)
            with record_function("typical/mask_build"):
                T = input_ids.shape[1]
                Kmax = max(len(c) for _, c in qs)
                pool = torch.stack([_pool_matrix(r[1], o, 0, Kmax, T, K=len(c)) for r, o, (_, c) in zip(rendered, offs, qs)])
                cmask = torch.zeros(m, Kmax, dtype=torch.bool)
                for i, (_, c) in enumerate(qs):
                    cmask[i, :len(c)] = True

            with record_function("typical/state_cache_expand"):
                exp_cache = _expand_state_cache(cache, m)
            with record_function("typical/suffix_forward"):
                attn = torch.cat([_sink_ones(m, Ls, "cpu"), am], dim=1).to(dev)
                position_ids = _position_ids_row(T, Ls, str(dev)).expand(m, -1)
                H = lm(input_ids=input_ids.to(dev), attention_mask=_causal_pad_mask(attn, T, lm.dtype),
                       position_ids=position_ids, past_key_values=exp_cache, use_cache=False).last_hidden_state.float()
            with record_function("typical/head_forward"):
                h = H[_arange(m, str(dev)), (lengths - 1).to(dev)]
                C3 = torch.bmm(pool.to(dev), H)
                C2 = None  # n2 unsupported -- guarded above
                yi = _yes_idx([c for _, c in qs], dev) if model.noul_head == "bern" else None
                bm = (torch.tensor([r == "query_only" for r in rnds], dtype=torch.bool, device=dev)
                      if model.noul_head == "bern" else None)
                probs = torch.softmax(model(h, cmask.to(dev), C3, C2, yes_idx=yi, bern_mask=bm), dim=-1)
            with record_function("typical/cpu_sync"):
                for row, i in enumerate(grp):
                    _, c = qs[row]
                    out[i] = torch.cat([probs[row, :len(c)], probs[row, -1:]])

    non_bern, bern = [i for i, b in enumerate(is_bern) if not b], [i for i, b in enumerate(is_bern) if b]
    if base_render != "semif" or not bern:
        cache, Ls = state_cache if state_cache is not None else encode_state(head, model, state, max_state)
        # semif suffixes already end in the chat template's assistant-turn opening: no eos tail (training's _kv_score_group)
        run_group(list(range(len(queries))), cache, Ls, [] if base_render == "semif" else [tok.eos_token_id])
    else:
        semif_cache, semif_Ls = state_cache if state_cache is not None else encode_state(head, model, state, max_state)
        run_group(non_bern, semif_cache, semif_Ls, [])
        plain_cache, plain_Ls = encode_state(head, model, state, max_state, semif=False)
        run_group(bern, plain_cache, plain_Ls, [tok.eos_token_id])
    return out
