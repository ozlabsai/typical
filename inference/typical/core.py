"""Typical: minimal from_pretrained + decide API around a native-readout PCDM checkpoint.
Mirrors pcdm_jev.decider.PCDMDecider(mode="native") exactly (same query_text/to_labels,
same native_kv_decide call) -- see that module's docstring for the mapping this
reproduces. Everything the checkpoint needs to rebuild backbone+head lives in
ckpt["args"]: backbone id, tap_layer, lora_r/lora_layers, nc_head, nc_render, null,
noul_head, score_head. z-score calibration (--zscore) needs no special handling here --
its buffers (mu_h/sd_h/mu_c/sd_c) are just part of ckpt["tower"]'s state_dict.
"""
import json
import os
import time
from collections import OrderedDict

import torch
from huggingface_hub import hf_hub_download

from .backbone import Backbone, pick_device
from .native import DEFAULT_MAX_OPTION_TOKENS, NativeHead, encode_state, native_kv_decide

MAX_QUERY = 64   # trained query length (see pcdm_jev.decider) -- informational only below
MAX_STATE = 256  # trained state length -- native_kv_decide's own max_state default is separate


class _Head:
    """Duck-types the training repo's mcq.MCQHead just enough for native_kv_decide: a
    .backbone (with .tokenizer/.model) and .device. The real MCQHead also builds a
    restricted LM head over single-token letters for the mcq_zero_shot/energy readout
    paths -- native_kv_decide never touches it, so it's not ported here."""

    def __init__(self, backbone: Backbone):
        self.backbone = backbone
        self.device = backbone.device


def query_text(question: dict) -> str:
    """instructions + criteria: `yes: .. no: ..` (noul), `0: .. 1: ..` (score), `label: desc`
    (choice). Verbatim port of pcdm_jev.decider.query_text."""
    q, crit = question["instructions"], question.get("criteria")
    if not crit:
        return q
    if question["type"] == "noul":
        rubric = f"yes: {crit.get('true', 'Yes')}  no: {crit.get('false', 'No')}"
    elif isinstance(crit, list):
        rubric = "  ".join(f"{i}: {c}" for i, c in enumerate(crit))
    else:
        rubric = "  ".join(f"{k}: {v or k}" for k, v in crit.items())
    return f"{q}\n{rubric}"


def to_labels(p: torch.Tensor, labels: list[str]) -> tuple[dict, float]:
    """[K+1] (null last) -> (probs renormalised over labels, p_null). Verbatim port of
    pcdm_jev.decider.to_labels."""
    p = p.detach().float().cpu()
    if not torch.isfinite(p).all():
        raise RuntimeError("non-finite model output")
    s = p[:-1].sum()
    lab = p[:-1] / s if s > 0 else torch.zeros_like(p[:-1])
    return dict(zip(labels, lab.tolist())), float(p[-1])


class Typical:
    """from_pretrained("OzLabs/typical-small") -> a ready-to-decide native-readout model."""

    def __init__(self, head: _Head, model: NativeHead, device: str, max_state: int = 4096,
                max_states: int = 16, max_option_tokens: int | None = DEFAULT_MAX_OPTION_TOKENS):
        self.head, self.model, self.device, self.max_state = head, model, device, max_state
        self.max_option_tokens = max_option_tokens
        self.tok = head.backbone.tokenizer
        # Persistent state -> prefix-KV cache (LRU, keyed on the exact state text): choice/
        # score/noul/decide all route through _raw -> _state_kv_for, so a warm call (same
        # state, new question) never re-runs the state_prefix_forward -- see pcdm/native.py's
        # encode_state/native_kv_decide docstrings. max_states caps how many distinct states
        # are held at once (each entry is one KV cache, ~O(state_tokens) memory).
        self.max_states = max_states
        self._state_kv: OrderedDict[str, tuple] = OrderedDict()

    @classmethod
    def from_pretrained(cls, repo_id: str, device: str = "auto", filename: str = "best.pt",
                        revision: str | None = None, cache_dir: str | None = None,
                        max_state: int = 4096, max_states: int = 16,
                        max_option_tokens: int | None = DEFAULT_MAX_OPTION_TOKENS) -> "Typical":
        path = repo_id if os.path.isfile(repo_id) else hf_hub_download(repo_id, filename, revision=revision, cache_dir=cache_dir)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        args = ckpt.get("args", {})
        if args.get("readout") != "native":
            raise ValueError(f"{repo_id}/{filename} is not a native-readout checkpoint (readout={args.get('readout')!r})")
        dev = pick_device(device)
        backbone = Backbone(args["backbone"], lora_layers=args.get("lora_layers", 8),
                            lora_r=args.get("lora_r", 16), device=dev, tap_layer=args.get("tap_layer", 0))
        backbone.load_lora_state_dict(ckpt["lora"])
        model = NativeHead(backbone.d, nc_head=args.get("nc_head", "n2n3"), null=args.get("null", "factored"),
                           render=args.get("nc_render", "letters"), score_head=args.get("score_head", "choice"),
                           noul_head=args.get("noul_head", "choice")).to(dev)
        model.load_state_dict(ckpt["tower"])
        model.eval()
        return cls(_Head(backbone), model, dev, max_state=max_state, max_states=max_states,
                   max_option_tokens=max_option_tokens)

    def _state_kv_for(self, state: str):
        """LRU lookup/insert of (past_key_values, Ls) for this exact state text. A hit skips
        encode_state entirely -- bit-identical result to a fresh encode (deterministic
        eval-mode forward), just without paying for it again."""
        entry = self._state_kv.get(state)
        if entry is not None:
            self._state_kv.move_to_end(state)
            return entry
        entry = encode_state(self.head, self.model, state, self.max_state)
        self._state_kv[state] = entry
        if len(self._state_kv) > self.max_states:
            self._state_kv.popitem(last=False)
        return entry

    def _raw(self, state, query: str, labels: list[str]) -> torch.Tensor:
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        with torch.inference_mode():
            cache = self._state_kv_for(state)
            return native_kv_decide(self.head, self.model, state, [(query, labels)],
                                    max_state=self.max_state, state_cache=cache,
                                    max_option_tokens=self.max_option_tokens)[0]

    def choice(self, state, question: str, labels: list[str]) -> dict:
        """-> {label: p, ...} + "p_null"."""
        probs, p_null = to_labels(self._raw(state, question, labels), labels)
        return {**probs, "p_null": p_null}

    def noul(self, state, question: str) -> float:
        """P(yes), via the Bernoulli path if the checkpoint has one (noul_head="bern"), else
        the plain K-way choice over ["no", "yes"] -- both paths are label-order invariant for
        a genuine yes/no row (see native._is_bern_row / _yes_idx)."""
        probs, _ = to_labels(self._raw(state, question, ["no", "yes"]), ["no", "yes"])
        return probs["yes"]

    def score(self, state, question: str, levels: list[str]) -> dict:
        """-> {level: p, ...} + "p_null" + "expected" (E[index] under the candidate-
        conditional distribution, matching the training repo's metrics.ordinal_metrics --
        candidate index IS the ordinal level, by construction of how levels are rendered)."""
        probs, p_null = to_labels(self._raw(state, question, levels), levels)
        expected = sum(i * probs[lv] for i, lv in enumerate(levels))
        return {**probs, "p_null": p_null, "expected": expected}

    def decide(self, state, question: dict, labels: list[str]) -> tuple[dict, dict]:
        """-> (probs over exactly `labels`, runtime block). Mirrors
        pcdm_jev.decider.PCDMDecider.decide for a JevBench-style question dict
        ({"type", "instructions", "criteria"})."""
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        query = query_text(question)
        n_state = len(self.tok(state, add_special_tokens=False)["input_ids"])
        n_query = len(self.tok(query, add_special_tokens=False)["input_ids"])
        t0 = time.perf_counter()
        p = self._raw(state, query, labels)
        probs, p_null = to_labels(p, labels)
        latency = time.perf_counter() - t0
        return probs, {"mode": "native", "device": self.device, "latency_s": latency, "p_null": p_null,
                       "state_tokens": n_state, "max_state_tokens": self.max_state,
                       "state_truncated": n_state > self.max_state,
                       "trained_max_state": MAX_STATE, "state_beyond_train_len": n_state > MAX_STATE,
                       "query_tokens": n_query, "query_truncated": False,
                       "trained_max_query": MAX_QUERY, "query_beyond_train_len": False,
                       "probability_origin": "native-softmax"}
