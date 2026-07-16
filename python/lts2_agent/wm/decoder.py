"""Decoder — reconstruct the tokenizer's typed arrays from the latent ``z`` ALONE (design §4.3).

Decoding from the pooled latent (not from any per-token encoder output) is the measurement that matters:
*what does ``z`` retain?* Architecture:

1. ``z`` projects to a small set of **memory tokens**.
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


class _TypeHeads(nn.Module):
    """Per-slot heads for one token type: categorical columns, numeric block, presence, keywords."""

    def __init__(self, tspec: S.TypeSpec, d_model: int):
        super().__init__()
        self.spec = tspec
        self.cat_heads = nn.ModuleList(nn.Linear(d_model, v) for _, v in tspec.cat_cols)
        self.num_head = nn.Linear(d_model, tspec.num_width) if tspec.num_width else None
        self.presence_head = nn.Linear(d_model, 1) if tspec.mask_key else None
        self.kw_head = nn.Linear(d_model, tokens.KW_BUCKETS) if tspec.has_kw else None

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        out["cat"] = [head(h) for head in self.cat_heads]           # list of [B, slots, vocab]
        if self.num_head is not None:
            out["num"] = self.num_head(h)                            # [B, slots, num_width]
        if self.presence_head is not None:
            out["presence"] = self.presence_head(h).squeeze(-1)      # [B, slots]
        if self.kw_head is not None:
            out["kw"] = self.kw_head(h)                              # [B, slots, KW]
        return out


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 512, d_model: int = 256, n_heads: int = 4, n_layers: int = 3,
                 n_mem: int = 16, ff_mult: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_mem = n_mem
        self.to_mem = nn.Linear(z_dim, n_mem * d_model)
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
        self.heads = nn.ModuleDict({t.name: _TypeHeads(t, d_model) for t in S.TYPES})

    def forward(self, z: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        B = z.shape[0]
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

def reconstruct_arrays(outputs: Dict[str, Dict[str, torch.Tensor]]) -> List[Dict[str, np.ndarray]]:
    """Turn a batched decoder output into a list (one per batch element) of numpy array dicts in
    :data:`tokens.TOKEN_KEYS` layout — argmax categoricals, regressed numerics, thresholded presence /
    keywords — ready for :func:`tokens.detokenize`."""
    # Materialize predictions on CPU once.
    pred: Dict[str, Dict[str, np.ndarray]] = {}
    for name, o in outputs.items():
        entry: Dict[str, np.ndarray] = {}
        entry["cat"] = [c.detach().argmax(dim=-1).cpu().numpy() for c in o["cat"]]  # each [B, slots]
        if "num" in o:
            entry["num"] = o["num"].detach().cpu().numpy()                    # [B, slots, w]
        if "presence" in o:
            entry["presence"] = (torch.sigmoid(o["presence"].detach()) >= 0.5).cpu().numpy()
        if "kw" in o:
            entry["kw"] = (torch.sigmoid(o["kw"].detach()) >= 0.5).cpu().numpy().astype(np.float32)
        pred[name] = entry

    B = pred["global"]["num"].shape[0]
    results: List[Dict[str, np.ndarray]] = []
    for b in range(B):
        arr: Dict[str, np.ndarray] = {}
        for t in S.TYPES:
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
