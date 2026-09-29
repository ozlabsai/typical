"""Frozen Qwen3 backbone + LoRA top layers + candidate feature cache. See PLAN2.md."""
import argparse
import glob
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer

_printed_device = False


def pick_device(device: str = "auto") -> str:
    global _printed_device
    if device != "auto":
        resolved = device
    elif torch.cuda.is_available():
        resolved = "cuda"
    elif torch.backends.mps.is_available():
        resolved = "mps"
    else:
        resolved = "cpu"
    if not _printed_device:
        if resolved == "cuda":
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"device: cuda ({name}, {mem:.1f} GB)")
        else:
            print(f"device: {resolved}")
        _printed_device = True
    return resolved


class LoRALinear(nn.Module):
    """Frozen base nn.Linear (bf16) + a low-rank fp32 adapter on top."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        self.drop = nn.Dropout(dropout)
        self.scale = alpha / r

    def forward(self, x):
        # fp32 master weights, bf16 compute: avoids an fp32 copy of every activation
        out = self.base(x)
        lora = self.drop(x) @ self.A.to(x.dtype).T @ self.B.to(x.dtype).T * self.scale
        return out + lora


# Qwen3: q/k/v/o_proj (self_attn) + gate/up/down_proj (mlp). Qwen3.5 hybrid stack adds
# in_proj_{qkv,z,b,a}/out_proj for its Gated-DeltaNet linear-attention layers (linear_attn,
# no self_attn) -- listed here too since hasattr() gates each name per parent module, so this
# tuple is a superset that's a no-op for any module lacking a given name (Qwen3 untouched).
_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")


class Backbone(nn.Module):
    def __init__(self, name: str = "Qwen/Qwen3-1.7B-Base", lora_layers: int = 8, lora_r: int = 16,
                 lora_alpha: float = 32, lora_dropout: float = 0.05, device: str = "auto", tap_layer: int = 0,
                 extra_tap: int = 0, dtype: torch.dtype = torch.bfloat16, multimodal: bool = False,
                 max_image_pixels: int = 262144):
        super().__init__()
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # dtype: bf16 everywhere in training/serving; a test can pass float32 to isolate a
        # correctness question (e.g. KV-cache parity) from bf16 rounding noise.
        model = AutoModel.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
        wrapper = model if hasattr(model, "language_model") and hasattr(model, "visual") else None
        if multimodal and wrapper is None:
            raise ValueError(f"--multimodal requires a vision-language backbone, got {name}")
        # Keep `model` as the text trunk: every existing text path and checkpoint key stays
        # unchanged. The optional wrapper shares this exact trunk and only handles image rows.
        self.model = wrapper.language_model if wrapper is not None else model
        self.multimodal_model = wrapper if multimodal else None
        self.processor = AutoProcessor.from_pretrained(name, padding_side="right") if multimodal else None
        self.max_image_pixels = max_image_pixels
        self.model.requires_grad_(False)
        if self.multimodal_model is not None:
            self.multimodal_model.requires_grad_(False)
        self.d = self.model.config.hidden_size
        n_layers = self.model.config.num_hidden_layers
        # tap_layer: take the tower's memory from layer T (< n_layers) instead of the last layer —
        # last-layer states of a base LM are next-token-shaped; mid-depth is better for semantics.
        # Layers above T are dropped (never needed); LoRA goes on the lora_layers blocks below T.
        # On Qwen3.5's hybrid stack this keeps whole layers (linear-attention or full-attention,
        # whichever layer_types[i] says) -- see self.info["layer_types"] for what survives.
        if 0 < tap_layer < n_layers:
            self.model.layers = self.model.layers[:tap_layer]
            self.model.config.num_hidden_layers = tap_layer
            n_layers = tap_layer
        self.tap_layer = n_layers
        # layer_types: Qwen3 is uniform full-attention (no such config field); Qwen3.5 alternates
        # linear_attention (Gated DeltaNet) / full_attention blocks -- record what's left post-tap.
        self.info = {"n_layers": n_layers,
                     "layer_types": list(getattr(self.model.config, "layer_types", None)
                                          or ["full_attention"] * n_layers)[:n_layers]}
        # extra_tap: also read layer E (< tap) and concatenate it onto the top state, [h_E; h_top] --
        # PLAN3 E1: h_20 for evidence alignment + h_28 for answer formation, same head, width 2d.
        self.extra_tap = extra_tap if 0 < extra_tap < n_layers else 0
        if self.extra_tap:
            self.d *= 2
        self.split_layer = n_layers - lora_layers
        self.lora_r = lora_r
        if lora_r > 0:
            lora_targets = set()
            for layer in self.model.layers[-lora_layers:]:
                # Qwen3: self_attn + mlp. Qwen3.5: linear_attention layers have linear_attn (Gated
                # DeltaNet) instead of self_attn; mlp is present on every layer either way.
                parents = [p for p in (getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None),
                                        getattr(layer, "mlp", None)) if p is not None]
                for parent in parents:
                    for mod_name in _LORA_MODULES:
                        if hasattr(parent, mod_name):
                            setattr(parent, mod_name,
                                    LoRALinear(getattr(parent, mod_name), lora_r, lora_alpha, lora_dropout))
                            lora_targets.add(mod_name)
            self.info["lora_modules"] = sorted(lora_targets)
            print(f"Backbone: LoRA (r={lora_r}) on {sorted(lora_targets)}, last {lora_layers} layers")
        self.to(self.device)

    def visual_prefix(self, texts: list[str], images, max_len: int):
        """Processor output for one image + one truncated text state per row."""
        if self.processor is None:
            raise ValueError("image rows require Backbone(..., multimodal=True)")
        from PIL import Image

        opened = []
        for image in images:
            if isinstance(image, (str, os.PathLike)):
                with Image.open(image) as src:
                    opened.append(src.convert("RGB"))
            elif isinstance(image, Image.Image):
                opened.append(image.convert("RGB"))
            else:
                raise TypeError(f"image must be a path or PIL image, got {type(image).__name__}")
        state_ids = self.tokenizer(texts, truncation=True, max_length=max_len,
                                   add_special_tokens=False)["input_ids"]
        placeholder = "<|vision_start|><|image_pad|><|vision_end|>\n"
        prompt = [placeholder + self.tokenizer.decode(ids, skip_special_tokens=False) for ids in state_ids]
        return self.processor(text=prompt, images=opened, padding=True, return_tensors="pt",
                              max_pixels=self.max_image_pixels)

    def tokenize(self, texts: list[str], max_len: int):
        # eos prepended as an explicit attention sink: position 0 of a causal LM is
        # near-constant regardless of token, so without this every single-token
        # candidate would share one vector. Dropped again in forward()/frozen_features().
        # empty text -> zero real tokens -> all-masked attention -> NaN; give it one token
        batch = [self.tokenizer.eos_token + (t if t.strip() else ".") for t in texts]
        enc = self.tokenizer(batch, truncation=True, max_length=max_len + 1, padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def tokenize_plain(self, texts: list[str], max_len: int):
        """Batch-tokenise, no special tokens, no sink, right-padded. Used for the
        query chunk in decide()'s joint/KV-cache path (state sink lives only in the
        cached prefix, not here)."""
        texts = [t if t.strip() else "." for t in texts]
        enc = self.tokenizer(texts, truncation=True, max_length=max_len, padding=True,
                              add_special_tokens=False, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def tokenize_joint(self, states: list[str], queries: list[str], max_state: int, max_query: int):
        """Per row: ids = [eos] + state_tokens + query_tokens, right-padded to the batch
        max. State and query are tokenised separately (add_special_tokens=False) so each
        keeps its own truncation length; concatenating pre-padding (rather than padding
        each side independently first) keeps query tokens immediately after the state's
        real tokens -- no mid-sequence pad gap that would shift RoPE positions.
        Returns (input_ids [B,T], attention_mask [B,T], ls [B], lq [B]); ls/lq exclude
        the eos sink, so after Backbone.forward drops position 0, row i's state tokens
        are at [0, ls_i) and query tokens at [ls_i, ls_i+lq_i)."""
        def ids_only(texts, max_len):
            texts = [t if t.strip() else "." for t in texts]
            return self.tokenizer(texts, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]

        s_ids, q_ids = ids_only(states, max_state), ids_only(queries, max_query)
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        rows = [[eos] + s + q for s, q in zip(s_ids, q_ids)]
        ls = torch.tensor([len(s) for s in s_ids], dtype=torch.long)
        lq = torch.tensor([len(q) for q in q_ids], dtype=torch.long)
        T = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), T), pad, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), T, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
            attention_mask[i, :len(r)] = 1
        return input_ids, attention_mask, ls, lq

    def top(self, out):
        """Top representation of a model output: last_hidden_state, or [norm(h_extra); last] with extra_tap."""
        if not self.extra_tap:
            return out.last_hidden_state
        return torch.cat([self.model.norm(out.hidden_states[self.extra_tap]), out.last_hidden_state], dim=-1)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h_frozen = self.model.norm(out.hidden_states[self.split_layer])
        h_top = self.top(out)
        mask = attention_mask[:, 1:].bool()
        return h_top[:, 1:].float(), h_frozen[:, 1:].float(), mask

    @torch.inference_mode()
    def frozen_features(self, texts: list[str], max_len: int, batch_size: int = 64) -> list[torch.Tensor]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[torch.Tensor] = [None] * len(texts)
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            input_ids, attention_mask = self.tokenize([texts[i] for i in idx], max_len)
            # lengths on CPU: boolean-mask indexing on-device forces a per-item sync
            # that can deadlock the Metal stream under GPU contention (see MPS notes).
            lengths = attention_mask.sum(1).tolist()
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            hs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True).hidden_states[self.split_layer]
            h_frozen = self.model.norm(hs)
            for j, i in enumerate(idx):
                out[i] = h_frozen[j, 1:lengths[j]].to(torch.bfloat16).cpu()
            if self.device == "mps" and start % (batch_size * 4) == 0:
                torch.mps.empty_cache()
        return out

    def lora_state_dict(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if n.endswith((".A", ".B"))}

    def load_lora_state_dict(self, sd: dict):
        own = dict(self.named_parameters())
        for k, v in sd.items():
            own[k].data.copy_(v.to(own[k].device, own[k].dtype))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


class FeatureCache:
    """Flat bf16 feature tensor + dict[str, (offset, length)] index. Never holds a Backbone."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.feats = torch.empty(0, 0, dtype=torch.bfloat16)
        self.index: dict[str, tuple[int, int]] = {}
        self._pool: dict[str, torch.Tensor] = {}

    def add(self, backbone, texts: list[str], max_len: int):
        seen: set[str] = set()
        new_texts = [t for t in texts if t not in self.index and not (t in seen or seen.add(t))]
        if not new_texts:
            return
        encoded = backbone.frozen_features(new_texts, max_len)  # CPU bf16 [len_i, d] each
        if self.feats.shape[1] == 0:
            self.feats = torch.empty(0, encoded[0].shape[1], dtype=torch.bfloat16)
        offset = self.feats.shape[0]
        chunks = [self.feats]  # accumulate once, cat once (avoids O(n) re-concat per text)
        for t, feat in zip(new_texts, encoded):
            self.index[t] = (offset, feat.shape[0])
            offset += feat.shape[0]
            chunks.append(feat)
        self.feats = torch.cat(chunks, dim=0).to(self.device)

    def get(self, text: str) -> torch.Tensor:
        offset, length = self.index[text]
        return self.feats[offset:offset + length]

    def pooled(self, text: str) -> torch.Tensor:
        if text not in self._pool:
            self._pool[text] = self.get(text).float().mean(0).to(self.device)
        return self._pool[text]

    def save(self, path: str):
        torch.save({"feats": self.feats.cpu(), "index": self.index}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FeatureCache":
        obj = cls(device=device)
        data = torch.load(path, map_location="cpu", weights_only=True)
        obj.feats = data["feats"].to(device)
        obj.index = data["index"]
        return obj


def build_cache(data_dir: str = "data", out: str = "data/cache.pt", backbone=None, device: str = "auto"):
    device = pick_device(device)
    if backbone is None:
        backbone = Backbone(lora_r=0, device=device)
    cache = FeatureCache.load(out, device="cpu") if os.path.exists(out) else FeatureCache(device="cpu")

    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    files += sorted(glob.glob(os.path.join(data_dir, "eval", "*.jsonl")))

    candidates: set[str] = set()
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                candidates.update(json.loads(line)["candidates"])

    cache.add(backbone, sorted(candidates), max_len=16)
    cache.save(out)
    print(f"files={len(files)} candidates={len(candidates)} unique_strings={len(cache.index)} "
          f"total_tokens={sum(length for _, length in cache.index.values())}")
    return cache


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024  # Qwen3-Embedding-0.6B hidden size
CAND_RENDER = "label: {}"  # generic candidate rendering (cosine_probe.py's "intent: {}" is task-specific)


class EmbedEncoder:
    """Qwen/Qwen3-Embedding-0.6B, last-token pooling, L2-normalised -- the exact recipe
    scripts/cosine_probe.py validated (embed_last_token). States/queries embed raw; pass
    render=CAND_RENDER for candidates."""

    def __init__(self, name: str = EMBED_MODEL, device: str = "auto"):
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="right")
        self.model = AutoModel.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model.requires_grad_(False)
        self.model.eval().to(self.device)
        self.d = self.model.config.hidden_size

    @torch.inference_mode()
    def embed(self, texts: list[str], max_len: int = 32, batch_size: int = 128,
              render: str | None = None) -> torch.Tensor:
        texts = [render.format(t) if render else t for t in texts]
        texts = [t + self.tokenizer.eos_token for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=max_len,
                                  padding=True, add_special_tokens=False, return_tensors="pt")
            ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
            h = self.model(input_ids=ids, attention_mask=am).last_hidden_state
            seq_lens = am.sum(1) - 1
            pooled = h[torch.arange(h.shape[0], device=self.device), seq_lens]
            pooled = F.normalize(pooled.float(), p=2, dim=-1).cpu()
            for j, i in enumerate(idx):
                out[i] = pooled[j]
        return torch.stack(out)


class VecCache:
    """dict str -> [d] fp16 vector (whole-text embeddings, already pooled -- unlike
    FeatureCache's per-token feats). Duck-types FeatureCache's .device/.index/.pooled()
    so collate()/run_batch() take either cache type unmodified."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.index: dict[str, torch.Tensor] = {}

    def add(self, encoder: "EmbedEncoder", texts: list[str], render: str | None = None,
            max_len: int = 32, batch_size: int = 128):
        new_texts = sorted({t for t in texts if t not in self.index})
        if not new_texts:
            return
        vecs = encoder.embed(new_texts, max_len=max_len, batch_size=batch_size, render=render)
        for t, v in zip(new_texts, vecs):
            self.index[t] = v.to(torch.float16)

    def get(self, text: str) -> torch.Tensor:
        return self.index[text]

    def pooled(self, text: str) -> torch.Tensor:
        return self.get(text).float().to(self.device)

    def save(self, path: str):
        torch.save({"index": self.index}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "VecCache":
        obj = cls(device=device)
        data = torch.load(path, map_location="cpu", weights_only=True)
        obj.index = data["index"]
        return obj


class TokenCandCache:
    """--cand_encoder tiny: str -> token ids (no sink/special tokens, <= max_len). The
    candidate encoder itself is TRAINED and lives in DecisionModel (proj_ct + tiny layers
    over the backbone's frozen input-embedding table), so only ids are cached; strings not
    in the cache are tokenised on the fly inside batch() (the cold path). Duck-types
    FeatureCache's .device/.index/.pooled() (pooled = mean input embedding, for
    train.calibrate_zscore's candidate sample)."""

    def __init__(self, backbone, max_len: int = 32, device: str = "cpu"):
        self.tokenizer, self.embed = backbone.tokenizer, backbone.model.embed_tokens
        self.max_len, self.device = max_len, device
        self.index: dict[str, list[int]] = {}

    def add(self, texts: list[str]):
        new = sorted({t for t in texts if t not in self.index})
        if not new:
            return
        ids = self.tokenizer([t if t.strip() else "." for t in new], truncation=True, max_length=self.max_len,
                             add_special_tokens=False)["input_ids"]
        dot = self.tokenizer(".", add_special_tokens=False)["input_ids"]
        self.index.update((t, r or dot) for t, r in zip(new, ids))

    def batch(self, cand_lists: list[list[str]]):
        """One pass over the batch's candidates -> (ids [N, L] long, tmask [N, L] bool,
        cidx [B, Kmax] long, cmask [B, Kmax] bool): N = unique candidate strings, cidx maps
        each (row, slot) to its unique row (0 at pad slots)."""
        uniq = list(dict.fromkeys(c for cands in cand_lists for c in cands))
        self.add(uniq)
        pos = {c: n for n, c in enumerate(uniq)}
        rows = [self.index[c] for c in uniq]
        L = max(len(r) for r in rows)
        ids = torch.full((len(uniq), L), self.tokenizer.pad_token_id, dtype=torch.long)
        tmask = torch.zeros(len(uniq), L, dtype=torch.bool)
        for n, r in enumerate(rows):
            ids[n, :len(r)] = torch.tensor(r, dtype=torch.long)
            tmask[n, :len(r)] = True
        Kmax = max(len(c) for c in cand_lists)
        cidx = torch.zeros(len(cand_lists), Kmax, dtype=torch.long)
        cmask = torch.zeros(len(cand_lists), Kmax, dtype=torch.bool)
        for b, cands in enumerate(cand_lists):
            cidx[b, :len(cands)] = torch.tensor([pos[c] for c in cands], dtype=torch.long)
            cmask[b, :len(cands)] = True
        return ids, tmask, cidx, cmask

    @torch.no_grad()
    def pooled(self, text: str) -> torch.Tensor:
        self.add([text])
        ids = torch.tensor(self.index[text], dtype=torch.long, device=self.embed.weight.device)
        return self.embed(ids).float().mean(0).to(self.device)


def build_vec_cache(data_dir: str = "data", out: str = "data/veccache_qwen3emb.pt",
                     encoder: "EmbedEncoder | None" = None, device: str = "auto") -> "VecCache":
    """Incremental, like build_cache: embeds candidates (rendered) + all state/query
    strings (raw) across every jsonl under data_dir, topping up an existing cache file."""
    device = pick_device(device)
    if encoder is None:
        encoder = EmbedEncoder(device=device)
    cache = VecCache.load(out) if os.path.exists(out) else VecCache()

    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    files += sorted(glob.glob(os.path.join(data_dir, "eval", "*.jsonl")))

    candidates, states, queries = set(), set(), set()
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                candidates.update(row["candidates"])
                states.add(row["state"])
                queries.add(row["query"])

    cache.add(encoder, sorted(candidates), render=CAND_RENDER, max_len=32)
    cache.add(encoder, sorted(states), max_len=128)
    cache.add(encoder, sorted(queries), max_len=32)
    cache.save(out)
    print(f"files={len(files)} candidates={len(candidates)} states={len(states)} queries={len(queries)} "
          f"unique_strings={len(cache.index)}")
    return cache


def selftest():
    print("running pcdm/encode.py selftest on Qwen/Qwen3-0.6B-Base ...")
    name = "Qwen/Qwen3-0.6B-Base"
    device = pick_device("auto")
    bb = Backbone(name=name, lora_layers=4, lora_r=8, device=device)
    bb0 = Backbone(name=name, lora_layers=4, lora_r=0, device=device)

    texts = ["The cat sat on the mat.", "neutral", "yes",
              "A much longer sentence to test padding and ragged batches end to end."]
    input_ids, attention_mask = bb.tokenize(texts, max_len=32)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    with torch.no_grad():
        h_top, h_frozen, mask = bb(input_ids, attention_mask)
        h_top0, h_frozen0, mask0 = bb0(input_ids, attention_mask)
    assert torch.allclose(h_top, h_top0, atol=1e-2), "zero-init LoRA changed h_top"
    print("PASS: zero-init LoRA => h_top matches lora_r=0 model")

    h_top, h_frozen, mask = bb(input_ids, attention_mask)
    h_top.sum().backward()
    n_lora_grad = n_lora_nograd = n_base_grad = 0
    for pname, p in bb.named_parameters():
        if pname.endswith((".A", ".B")):
            n_lora_grad += p.grad is not None
            n_lora_nograd += p.grad is None
        else:
            n_base_grad += p.grad is not None
    assert n_lora_nograd == 0 and n_lora_grad > 0, "some LoRA params missing grads"
    assert n_base_grad == 0, "base params received grads"
    print(f"PASS: grads on {n_lora_grad} LoRA tensors only, base params grad-free")

    trunc = AutoModel.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    trunc.layers = trunc.layers[:bb0.split_layer]
    with torch.no_grad():
        h_v0 = trunc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 1:].float()
    m = mask0
    ok = torch.allclose(h_v0[m], h_frozen0[m], atol=1e-2)
    assert ok, "h_frozen diverges from v0 truncated-encoder semantics"
    print("PASS: h_frozen matches v0 Encoder(layers=n_layers-lora_layers) semantics")

    feats = bb0.frozen_features(["neutral", "yes"], max_len=16)
    a, b = feats[0].float().mean(0), feats[1].float().mean(0)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert cos < 0.95, f"cosine too high ({cos:.3f}): sink prepend not working"
    print(f"PASS: frozen_features cosine(neutral,yes)={cos:.3f} < 0.95")

    ragged = ["a", "a much longer sentence here with several tokens to pad against"]
    ids, am = bb0.tokenize(ragged, max_len=32)
    ids, am = ids.to(device), am.to(device)
    h_top_r, h_frozen_r, mask_r = bb0(ids, am)
    B, T = ids.shape
    assert h_top_r.shape == (B, T - 1, bb0.d) and h_frozen_r.shape == (B, T - 1, bb0.d)
    assert mask_r.shape == (B, T - 1) and mask_r.dtype == torch.bool
    assert torch.equal(mask_r.sum(1), am.sum(1) - 1)
    print("PASS: tokenize -> forward shape/mask contract on ragged input")

    print("pcdm/encode.py selftest passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--lora_layers", type=int, default=8)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="data/cache.pt")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
    else:
        dev = pick_device("auto")
        bb = Backbone(name=args.backbone, lora_layers=args.lora_layers, lora_r=0, device=dev)
        build_cache(data_dir=args.data, out=args.out, backbone=bb, device=dev)
