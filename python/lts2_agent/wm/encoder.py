"""Encoder — set transformer over the tokenizer's typed arrays -> a normalized latent ``z`` (design §4.2).

Pipeline (leading batch dim ``B`` throughout):

1. **Per-type input projections.** Each token type's ``*_idx`` (catalog/enum indices) + ``*_num`` (symlog
   numerics) + gathered static-catalog row (cards/powers/relics/potions) + card-keyword multi-hot are
   summed/concatenated and linearly projected to ``d_model``. A learned token-type embedding is added so
   attention can tell types apart. Powers/intents carry a parent-slot embedding so the set keeps the
   child->creature association without folding.
2. **Self-attention over the packed token set.** All tokens (global, pending, cards, creatures, powers,
   intents, orbs, relics, potions) form one sequence with a key-padding mask over the pad slots; a stack
   of pre-norm ``TransformerEncoderLayer`` blocks contextualizes them (efficient SDPA masking — no
   sort/pack needed at batch 256-512 on a 3090).
3. **Attention pooling.** A few learned latent queries cross-attend over the token set (Perceiver/Set-
   Transformer inducing points), then the flattened latents project to the latent vector.
4. **SimNorm.** ``z`` is reshaped into groups and softmax'd within each group (TD-MPC2's SimNorm), giving
   a latent that is a concatenation of probability simplices — bounded (each group sums to 1) so it can
   neither explode nor collapse to a constant scale, the normalization design §4.4/§11 adopts. Chosen
   over plain L2 because grouped-simplex latents empirically preserve more categorical structure at equal
   width and are the published default for latent world models.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .. import catalog, tokens
from . import spec as S


def simnorm(z: torch.Tensor, group: int) -> torch.Tensor:
    """SimNorm (TD-MPC2): split ``z`` (``[..., D]``, ``D % group == 0``) into ``D/group`` groups and
    softmax within each, so the output is a concatenation of probability simplices."""
    shape = z.shape
    z = z.reshape(*shape[:-1], shape[-1] // group, group)
    z = torch.softmax(z, dim=-1)
    return z.reshape(*shape)


class _MultiEmbed(nn.Module):
    """Sum of per-column embeddings for an ``[..., C]`` integer index tensor."""

    def __init__(self, sizes: List[int], dim: int):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(n, dim) for n in sizes)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # idx [..., C] long
        out = self.embs[0](idx[..., 0])
        for c in range(1, len(self.embs)):
            out = out + self.embs[c](idx[..., c])
        return out


class _TypeEmbedder(nn.Module):
    """Projects one token type's arrays to ``d_model`` (cat embeddings ++ static row ++ numeric ++ kw)."""

    def __init__(self, tspec: S.TypeSpec, d_model: int, cat_dim: int,
                 static_tables: Dict[str, np.ndarray]):
        super().__init__()
        self.spec = tspec
        self.cat = _MultiEmbed([v for _, v in tspec.cat_cols], cat_dim) if tspec.cat_cols else None
        static_dim = 0
        if tspec.has_static:
            tbl = np.asarray(static_tables[tspec.has_static], dtype=np.float32)
            self.register_buffer("static", torch.from_numpy(tbl))
            static_dim = tbl.shape[1]
        else:
            self.static = None
        kw_dim = tokens.KW_BUCKETS if tspec.has_kw else 0
        in_dim = (cat_dim if tspec.cat_cols else 0) + static_dim + tspec.num_width + kw_dim
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, idx: torch.Tensor, num: torch.Tensor, kw: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.cat is not None:
            parts.append(self.cat(idx))
        if self.static is not None:
            parts.append(self.static[idx[..., 0]])  # gather static row by the catalog index column
        if num is not None and self.spec.num_width:
            parts.append(num)
        if kw is not None:
            parts.append(kw)
        return self.proj(torch.cat(parts, dim=-1))


class _PoolLayer(nn.Module):
    """Perceiver-style pooling: latents cross-attend to the token set, then self-attend; pre-norm."""

    def __init__(self, d: int, heads: int, ff_mult: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.GELU(), nn.Linear(d * ff_mult, d))

    def forward(self, lat: torch.Tensor, toks: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        q = self.n1(lat)
        a, _ = self.cross(q, toks, toks, key_padding_mask=key_pad)
        lat = lat + a
        s, _ = self.self_attn(self.n2(lat), self.n2(lat), self.n2(lat))
        lat = lat + s
        lat = lat + self.ff(self.n3(lat))
        return lat


class Encoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 4,
                 n_pool_layers: int = 2, n_latents: int = 8, z_dim: int = 512,
                 simnorm_group: int = 8, cat_dim: int = 24, ff_mult: int = 2,
                 static_tables=None):
        super().__init__()
        if z_dim % simnorm_group != 0:
            raise ValueError(f"z_dim {z_dim} must be divisible by simnorm_group {simnorm_group}")
        self.d_model = d_model
        self.z_dim = z_dim
        self.simnorm_group = simnorm_group
        if static_tables is None:
            static_tables = {k: catalog.load(k).static_table for k in ("cards", "powers", "relics",
                                                                        "potions")}
        self.embedders = nn.ModuleDict(
            {t.name: _TypeEmbedder(t, d_model, cat_dim, static_tables) for t in S.TYPES})
        # Token-type embedding (one row per type in S.TYPES order).
        self.type_emb = nn.Embedding(len(S.TYPES), d_model)
        # Parent-slot embedding shared by power/intent child tokens (which creature they belong to).
        self.parent_emb = nn.Embedding(tokens.MAX_CREATURES, d_model)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * ff_mult,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(layer, n_layers)

        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.pool = nn.ModuleList(_PoolLayer(d_model, n_heads, ff_mult) for _ in range(n_pool_layers))
        self.to_z = nn.Linear(n_latents * d_model, z_dim)

    def _tokens_and_mask(self, batch: Dict[str, torch.Tensor]):
        """Embed every type into one ``[B, T, d]`` sequence + a ``[B, T]`` bool key-padding mask
        (True = pad/ignore)."""
        B = batch["global_idx"].shape[0]
        device = batch["global_idx"].device
        seq = []
        masks = []
        for ti, t in enumerate(S.TYPES):
            idx = batch.get(t.idx_key) if t.idx_key else None
            num = batch.get(t.num_key) if t.num_key else None
            kw = batch.get("card_kw") if t.has_kw else None
            emb = self.embedders[t.name](idx, num, kw)
            if emb.dim() == 2:                       # single-token types come in as [B, F]
                emb = emb.unsqueeze(1)
            emb = emb + self.type_emb.weight[ti]
            if t.name in ("power", "intent"):        # add which creature this child belongs to
                emb = emb + self.parent_emb(idx[..., 1])
            seq.append(emb)
            if t.mask_key:
                masks.append(batch[t.mask_key])
            else:
                masks.append(torch.ones(B, emb.shape[1], dtype=torch.bool, device=device))
        toks = torch.cat(seq, dim=1)
        valid = torch.cat(masks, dim=1)
        return toks, ~valid

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        toks, key_pad = self._tokens_and_mask(batch)
        toks = self.trunk(toks, src_key_padding_mask=key_pad)
        B = toks.shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.pool:
            lat = layer(lat, toks, key_pad)
        z = self.to_z(lat.reshape(B, -1))
        return simnorm(z, self.simnorm_group)
