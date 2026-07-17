"""Decoder — reconstruct the tokenizer's typed arrays from the latent ALONE (design §4.3).

Decoding from the pooled latent (not from any per-token encoder output) is the measurement that matters:
*what does the latent retain?* Architecture:

1. The latent becomes a small set of **memory tokens**. In ``flat`` mode ``z`` (``z_dim``) projects/expands
   to ``n_mem`` memory tokens; in ``tokens`` mode the encoder's ``latent_k`` latent tokens ARE the memory
   directly (no expansion projection — the A/B variant that removes the flatten/expand bottleneck).
2. Per token type, a bank of **learned slot queries** (one per padded slot, + a type embedding) cross-
   attends into the memory and self-attends among the whole query set (``TransformerDecoderLayer``), so
   slots can coordinate on set sizes / presence.
3. **Per-type heads** map each slot's decoder output to the tokenizer's array space:
   * categorical logits per ``*_idx`` column (card-catalog index, zone, power index, relic/potion index,
     creature kind/identity bucket, intent type, enums) — cross-entropy;
   * a numeric vector per ``*_num`` block — regression on the symlog values (the same array
     :func:`lts2_agent.tokens.detokenize` inverts); MSE (see trainer note on why not two-hot);
   * a per-slot **presence** logit for every variable-length type (BCE against the tokenizer mask);
   * card **keyword** multi-hot logits (BCE).

Canonical-dict reconstruction (:func:`reconstruct_arrays`) assembles argmax'd categoricals + regressed
numerics + thresholded presence back into the exact array dict :func:`tokens.detokenize` consumes — the
tokenizer owns detokenization; this module never reimplements it.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .. import tokens
from . import spec as S


# ==================================================================================================
# Two-hot numeric encoding (roadmap 3.1 probe: --num-head twohot).
#
# DreamerV3-style two-hot classification over a fixed bin grid in *symlog space* (the tokenizer already
# emits symlog values, so the grid lives in the same space the MSE head regresses). NUM_BINS bins span
# the symlog clamp range [-symlog(NUM_CLIP), +symlog(NUM_CLIP)] symmetrically (NUM_CLIP is the
# tokenizer's exact-round-trip magnitude bound). A scalar value is encoded as the linear-interpolation
# weights on its two adjacent bins; the decode is the expectation over bins (softmax(logits)·centers),
# which yields a single symlog value that the existing symexp+round downstream (reconstruct_arrays /
# report) consumes unchanged — so the head swap is invisible to everything but the loss.
# ==================================================================================================

NUM_BINS = 64
_SYMLOG_LIMIT = float(np.log1p(tokens.NUM_CLIP))   # symlog(NUM_CLIP) ~= 11.51


def symlog_bins(num_bins: int = NUM_BINS, device=None, dtype=torch.float32) -> torch.Tensor:
    """The fixed bin-center grid (ascending) spanning the symlog clamp range."""
    return torch.linspace(-_SYMLOG_LIMIT, _SYMLOG_LIMIT, num_bins, device=device, dtype=dtype)


def twohot_targets(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot target distribution for ``values`` (``[...]`` symlog scalars) over ``bins`` (``[K]``):
    ``[..., K]`` with the two adjacent bins carrying the linear-interpolation weights. By construction
    ``twohot_expectation(twohot_targets(v)) == v`` for any ``v`` inside the grid (round-trip exact)."""
    K = bins.shape[0]
    v = values.clamp(bins[0], bins[-1])
    above = torch.searchsorted(bins, v, right=True).clamp(1, K - 1)   # [...], in [1, K-1]
    below = above - 1
    b_below = bins[below]
    b_above = bins[above]
    w_above = (v - b_below) / (b_above - b_below)
    w_below = 1.0 - w_above
    target = torch.zeros(*v.shape, K, device=v.device, dtype=v.dtype)
    target.scatter_(-1, below.unsqueeze(-1), w_below.unsqueeze(-1))
    target.scatter_add_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return target


def twohot_expectation(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected symlog value under ``probs`` (``[..., K]``) over ``bins`` (``[K]``) -> ``[...]``."""
    return (probs * bins).sum(dim=-1)


class _TypeHeads(nn.Module):
    """Per-slot heads for one token type: categorical columns, numeric block, presence, keywords.

    ``num_head`` selects the numeric decode: ``"mse"`` (default) regresses the symlog block directly
    (a ``Linear -> num_width``); ``"twohot"`` classifies each numeric column over ``num_bins`` symlog
    bins (a ``Linear -> num_width*num_bins``) and exposes both the classification ``num_logits`` (for
    the CE loss) and the expectation-decoded ``num`` (identical shape to the MSE head, so downstream is
    unchanged).

    ``relic_head`` (relic type only) selects the relic decode: ``"slots"`` (default) is the per-slot
    categorical-over-catalog head shared by every type; ``"set"`` replaces the relic branch with ONE
    multi-hot head over the whole relic catalog (mean-pool the relic slot outputs -> ``Linear ->
    relic_vocab``), emitting ``set_logits`` ``[B, relic_vocab]`` and no per-slot cat/presence. The set
    head structurally forbids duplicate relics (each catalog id is one output unit), which the 24
    independent slot-categoricals could not — the CP4 residual-error target."""

    def __init__(self, tspec: S.TypeSpec, d_model: int, num_head: str = "mse",
                 num_bins: int = NUM_BINS, relic_head: str = "slots"):
        super().__init__()
        self.spec = tspec
        self.num_mode = num_head
        self.num_bins = num_bins
        self.relic_set = (tspec.name == "relic" and relic_head == "set")
        if self.relic_set:
            # One multi-hot head over the relic catalog; no per-slot cat/num/presence/kw.
            self.set_head = nn.Linear(d_model, tspec.cat_cols[0][1])
            self.cat_heads = nn.ModuleList()
            self.num_head = None
            self.presence_head = None
            self.kw_head = None
            return
        self.set_head = None
        self.cat_heads = nn.ModuleList(nn.Linear(d_model, v) for _, v in tspec.cat_cols)
        if tspec.num_width:
            out_dim = tspec.num_width * num_bins if num_head == "twohot" else tspec.num_width
            self.num_head = nn.Linear(d_model, out_dim)
        else:
            self.num_head = None
        self.presence_head = nn.Linear(d_model, 1) if tspec.mask_key else None
        self.kw_head = nn.Linear(d_model, tokens.KW_BUCKETS) if tspec.has_kw else None

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.relic_set:
            # Mean-pool the relic slot outputs into one representation -> multi-hot catalog logits.
            return {"set_logits": self.set_head(h.mean(dim=1))}     # [B, relic_vocab]
        out: Dict[str, torch.Tensor] = {}
        out["cat"] = [head(h) for head in self.cat_heads]           # list of [B, slots, vocab]
        if self.num_head is not None:
            raw = self.num_head(h)
            if self.num_mode == "twohot":
                w = self.spec.num_width
                logits = raw.reshape(*raw.shape[:-1], w, self.num_bins)   # [B, slots, w, bins]
                bins = symlog_bins(self.num_bins, device=logits.device, dtype=logits.dtype)
                out["num_logits"] = logits
                out["num"] = twohot_expectation(logits.softmax(dim=-1), bins)  # [B, slots, w]
            else:
                out["num"] = raw                                     # [B, slots, num_width]
        if self.presence_head is not None:
            out["presence"] = self.presence_head(h).squeeze(-1)      # [B, slots]
        if self.kw_head is not None:
            out["kw"] = self.kw_head(h)                              # [B, slots, KW]
        return out


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 512, d_model: int = 256, n_heads: int = 4, n_layers: int = 3,
                 n_mem: int = 16, ff_mult: int = 2, latent_mode: str = "flat", latent_k: int = 16,
                 num_head: str = "mse", relic_head: str = "slots"):
        super().__init__()
        if latent_mode not in ("flat", "tokens"):
            raise ValueError(f"latent_mode must be 'flat' or 'tokens', got {latent_mode!r}")
        if num_head not in ("mse", "twohot"):
            raise ValueError(f"num_head must be 'mse' or 'twohot', got {num_head!r}")
        if relic_head not in ("slots", "set"):
            raise ValueError(f"relic_head must be 'slots' or 'set', got {relic_head!r}")
        self.d_model = d_model
        self.n_mem = n_mem
        self.latent_mode = latent_mode
        self.latent_k = latent_k
        # flat: expand the pooled z into n_mem memory tokens; tokens: the latent_k latent tokens ARE the
        # memory, so no expansion projection.
        self.to_mem = nn.Linear(z_dim, n_mem * d_model) if latent_mode == "flat" else None
        self.mem_norm = nn.LayerNorm(d_model)

        # Learned slot queries: one contiguous bank per type, + a per-type embedding.
        self.slot_queries = nn.ParameterDict(
            {t.name: nn.Parameter(torch.randn(t.max_slots, d_model) * 0.02) for t in S.TYPES})
        self.type_emb = nn.Embedding(len(S.TYPES), d_model)
        self._offsets, off = {}, 0
        for t in S.TYPES:
            self._offsets[t.name] = (off, off + t.max_slots)
            off += t.max_slots
        self.total_slots = off

        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model * ff_mult,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.trunk = nn.TransformerDecoder(layer, n_layers)
        self.heads = nn.ModuleDict({t.name: _TypeHeads(t, d_model, num_head=num_head,
                                                       relic_head=relic_head)
                                    for t in S.TYPES})

    def forward(self, z: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        B = z.shape[0]
        if self.latent_mode == "tokens":
            mem = self.mem_norm(z)                                       # [B, latent_k, d] used directly
        else:
            mem = self.mem_norm(self.to_mem(z).reshape(B, self.n_mem, self.d_model))
        q = []
        for ti, t in enumerate(S.TYPES):
            qt = self.slot_queries[t.name].unsqueeze(0).expand(B, -1, -1) + self.type_emb.weight[ti]
            q.append(qt)
        queries = torch.cat(q, dim=1)                                # [B, total_slots, d]
        h = self.trunk(queries, mem)
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for t in S.TYPES:
            a, b = self._offsets[t.name]
            out[t.name] = self.heads[t.name](h[:, a:b, :])
        return out


# ==================================================================================================
# Array reconstruction — decoder outputs -> the tokenizer's array dict, for detokenize + exact-state.
# ==================================================================================================

def _dedup_slot_ids(logits: torch.Tensor, present: np.ndarray) -> np.ndarray:
    """Greedy-by-confidence deduplicated slot ids for a single categorical column (CP4 relic fix 1).

    ``logits`` ``[B, slots, vocab]``, ``present`` ``[B, slots]`` bool. Absent slots keep their plain
    argmax (masked out downstream). Present slots are assigned in DESCENDING max-softmax-probability
    order: each takes its highest-probability id among those not yet claimed by a more-confident slot,
    so the most-confident assignment always wins and no id repeats within a state. This is a pure
    decode-time reassignment of the EXISTING slot head — no training effect, no new parameters."""
    probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()   # [B, slots, vocab]
    out = probs.argmax(axis=-1).astype(np.int32)                   # default: plain argmax
    B = probs.shape[0]
    for b in range(B):
        pres = np.nonzero(present[b])[0]
        # Most-confident present slot first (its argmax is safe to keep).
        order = sorted(pres.tolist(), key=lambda s: float(probs[b, s].max()), reverse=True)
        taken: List[int] = []
        for s in order:
            row = probs[b, s]
            if taken:
                row = row.copy()
                row[taken] = -1.0
            cid = int(row.argmax())
            out[b, s] = cid
            taken.append(cid)
    return out


def _decode_set_head(o: Dict[str, torch.Tensor], max_slots: int) -> "tuple[np.ndarray, np.ndarray]":
    """Decode a multi-hot relic set head (``relic_head=set``) into the standard slot arrays.

    ``k = clamp(round(sum sigmoid(logits)), 0, max_slots)`` sets the cardinality; the ``k`` highest-
    probability catalog ids (excluding index 0 = none) are emitted, sorted by catalog index, into
    ``[max_slots]`` idx + presence-mask arrays so detokenize/report consume them unchanged. Top-k over
    distinct catalog units cannot produce a duplicate. Returns ``(idx [B, max_slots], mask [B, max_slots])``."""
    probs = torch.sigmoid(o["set_logits"].detach()).cpu().numpy()  # [B, vocab]
    B, vocab = probs.shape
    idx = np.zeros((B, max_slots), dtype=np.int32)
    mask = np.zeros((B, max_slots), dtype=bool)
    if "count_logits" in o:
        # Dedicated cardinality head: k by argmax, immune to membership-probability calibration
        # (pos_weight inflates sigmoids and would explode round(sum p)).
        card = o["count_logits"].detach().argmax(dim=-1).cpu().numpy().astype(np.float64)
    else:
        card = probs.sum(axis=1)
    for b in range(B):
        k = int(np.clip(round(float(card[b])), 0, max_slots))
        if k <= 0:
            continue
        p = probs[b].copy()
        p[0] = -1.0                                                # never emit index 0 (none)
        top = np.argpartition(p, -k)[-k:]                          # k highest-prob ids (unordered)
        ids = np.sort(top)                                         # emit sorted by catalog index
        idx[b, :k] = ids
        mask[b, :k] = True
    return idx, mask


def reconstruct_arrays(outputs: Dict[str, Dict[str, torch.Tensor]],
                       dedup: bool = False) -> List[Dict[str, np.ndarray]]:
    """Turn a batched decoder output into a list (one per batch element) of numpy array dicts in
    :data:`tokens.TOKEN_KEYS` layout — argmax categoricals, regressed numerics, thresholded presence /
    keywords — ready for :func:`tokens.detokenize`.

    ``dedup`` (relic slot head only) applies :func:`_dedup_slot_ids` to the relic-identity column so the
    decoded relic set carries no duplicate ids — a pure decode-time option, no effect on training or on
    the ``relic_head=set`` path (which is duplicate-free by construction)."""
    # Materialize predictions on CPU once.
    pred: Dict[str, Dict[str, np.ndarray]] = {}
    set_decoded: Dict[str, "tuple[np.ndarray, np.ndarray]"] = {}
    for name, o in outputs.items():
        if "set_logits" in o:                                     # relic set head
            set_decoded[name] = _decode_set_head(o, S.TYPE_BY_NAME[name].max_slots)
            continue
        entry: Dict[str, np.ndarray] = {}
        entry["cat"] = [c.detach().argmax(dim=-1).cpu().numpy() for c in o["cat"]]  # each [B, slots]
        if "presence" in o:
            entry["presence"] = (torch.sigmoid(o["presence"].detach()) >= 0.5).cpu().numpy()
        if dedup and name == "relic" and o["cat"]:
            entry["cat"][0] = _dedup_slot_ids(o["cat"][0], entry["presence"])
        if "num" in o:
            entry["num"] = o["num"].detach().cpu().numpy()                    # [B, slots, w]
        if "kw" in o:
            entry["kw"] = (torch.sigmoid(o["kw"].detach()) >= 0.5).cpu().numpy().astype(np.float32)
        pred[name] = entry

    B = pred["global"]["num"].shape[0]
    results: List[Dict[str, np.ndarray]] = []
    for b in range(B):
        arr: Dict[str, np.ndarray] = {}
        for t in S.TYPES:
            if t.name in set_decoded:                             # relic set head -> slot arrays
                s_idx, s_mask = set_decoded[t.name]
                arr[t.idx_key] = s_idx[b].reshape(t.max_slots, len(t.cat_cols)).astype(np.int32)
                arr[t.mask_key] = s_mask[b].astype(bool)
                continue
            p = pred[t.name]
            n_cols = len(t.cat_cols)
            if t.idx_key:
                if n_cols:
                    arr[t.idx_key] = np.stack([p["cat"][c][b] for c in range(n_cols)],
                                              axis=-1).astype(np.int32)
                if t.name == "global":
                    arr[t.idx_key] = arr[t.idx_key]  # [1, cols]
            if t.num_key:
                arr[t.num_key] = p["num"][b].astype(np.float32)
            if t.mask_key:
                arr[t.mask_key] = p["presence"][b].astype(bool)
            if t.has_kw:
                arr["card_kw"] = p["kw"][b].astype(np.float32)
            if t.name == "global":
                arr["token_type_global"] = np.int32(tokens.TOKEN_TYPE_ID["global"])
        results.append(arr)
    return results
