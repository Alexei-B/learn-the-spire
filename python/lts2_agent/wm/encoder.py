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
   Transformer inducing points). In ``flat`` mode the flattened latents project to a single latent vector
   ``z`` (``z_dim``); in ``tokens`` mode the ``latent_k`` pooled latents ARE the latent (no flatten, no
   ``z_dim`` projection) — the A/B variant that removes the flatten-to-``z_dim`` bottleneck between the
   pool and the decoder (design §10, first bullet — the P2/CP4 latent-shape decision).
4. **SimNorm.** the latent is reshaped into groups and softmax'd within each group (TD-MPC2's SimNorm),
   giving a latent that is a concatenation of probability simplices — bounded (each group sums to 1) so it
   can neither explode nor collapse to a constant scale, the normalization design §4.4/§11 adopts. Chosen
   over plain L2 because grouped-simplex latents empirically preserve more categorical structure at equal
   width and are the published default for latent world models. In ``tokens`` mode SimNorm is applied
   *per latent token* over its ``d_model`` channels (same ``simnorm_group``), so each token is its own
   concatenation of simplices.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .. import catalog, tokens
from . import spec as S

# --------------------------------------------------------------------------------------------------
# Numeric INPUT featurization (roadmap M3.5 cross-expert numeric fix, INPUT half).
#
# WHY: numerics enter the encoder as ONE symlog float per column. symlog(500) vs symlog(501) differ by
# ~0.002 in a single dimension, so large-value precision is lost at the INPUT — the latent never receives
# the resolution the digit OUTPUT heads try (and fail) to emit. `num_input` optionally ADDS a
# high-resolution featurization of each numeric column ALONGSIDE the (kept) symlog float:
#   * "symlog" (default): unchanged — one symlog float per column, byte-identical to the pre-featurization
#     model (no extra params, in_dim unchanged).
#   * "digits": each ranged (non-flag) column additionally enters as base-`_DIGIT_BASE` per-digit learned
#     embeddings — SAME digit decomposition the e03b289 RangeBinHeads output heads use (bin index =
#     round((symexp(stored)-lo)/res), offset by the column's value-lo so negatives work; n_digits the
#     smallest nd with base**nd >= n_bins). The per-digit embeddings are concatenated into the projection
#     input next to the symlog float.
#   * "fourier": each ranged column additionally enters as sin/cos at `_FOURIER_MAX_K` geometrically-spaced
#     frequencies chosen from its range (finest wavelength ~2 resolves resolution 1, coarsest ~4*span),
#     projected by a per-type linear that is ADDED to the token embedding.
#   * "both": digits + fourier.
# Everything is derived from the STORED symlog float via the exact symexp->round the RangeBinHeads target
# already uses, so no tokenizer / cache / generator change is needed — a real-cache batch and a synth batch
# featurize identically. Flag columns (n_bins <= 2) and columns with no measured range are left as-is.
NUM_INPUT_MODES = ("symlog", "digits", "fourier", "both")
_DIGIT_BASE = 10        # base-10 digits — matches experts.DIGIT_BASE (the output-head convention)
_DIGIT_EMB_DIM = 8      # learned embedding dim per digit position (small)
_FOURIER_MAX_K = 8      # cap on the number of geometrically-spaced sin/cos frequencies per column

# The tokenizer's numeric column names per type (mirrors experts._num_cols; kept here to avoid importing
# experts, which imports this module). Only these types carry a numeric block that has measured ranges.
_NUM_COLS: Dict[str, List[str]] = {
    "card": tokens.CARD_NUM, "creature": tokens.CREATURE_NUM, "power": tokens.POWER_NUM,
    "intent": tokens.INTENT_NUM, "orb": tokens.ORB_NUM,
}


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
    """Projects one token type's arrays to ``d_model`` (cat embeddings ++ static row ++ numeric ++ kw).

    ``num_input`` (default ``"symlog"`` == byte-identical prior behavior) optionally ADDS a
    high-resolution featurization of each ranged numeric column derived from the stored symlog float — see
    :data:`NUM_INPUT_MODES`. Digit embeddings are concatenated into the projection input; a fourier block
    is projected by a per-type linear ADDED to the token embedding. Flag / no-range columns are untouched.
    """

    def __init__(self, tspec: S.TypeSpec, d_model: int, cat_dim: int,
                 static_tables: Dict[str, np.ndarray], num_input: str = "symlog"):
        super().__init__()
        if num_input not in NUM_INPUT_MODES:
            raise ValueError(f"num_input {num_input!r} not in {NUM_INPUT_MODES}")
        self.spec = tspec
        self.num_input = num_input
        self.cat = _MultiEmbed([v for _, v in tspec.cat_cols], cat_dim) if tspec.cat_cols else None
        static_dim = 0
        if tspec.has_static:
            tbl = np.asarray(static_tables[tspec.has_static], dtype=np.float32)
            self.register_buffer("static", torch.from_numpy(tbl))
            static_dim = tbl.shape[1]
        else:
            self.static = None
        kw_dim = tokens.KW_BUCKETS if tspec.has_kw else 0

        # Numeric-input featurization: which columns get it, and its extra input width.
        self._use_digits = num_input in ("digits", "both")
        self._use_fourier = num_input in ("fourier", "both")
        self._feat_meta: List[Dict] = self._build_feat_meta(tspec) if (self._use_digits or
                                                                        self._use_fourier) else []
        digit_dim = 0
        if self._use_digits and self._feat_meta:
            # One ModuleList of ``nd`` base-embeddings per featurized column (nested so a column's digit
            # positions stay grouped). Concatenated into the projection input alongside the symlog float.
            self.digit_embs = nn.ModuleList(
                nn.ModuleList(nn.Embedding(_DIGIT_BASE, _DIGIT_EMB_DIM) for _ in range(m["nd"]))
                for m in self._feat_meta)
            digit_dim = sum(m["nd"] * _DIGIT_EMB_DIM for m in self._feat_meta)
        if self._use_fourier and self._feat_meta:
            fourier_dim = 0
            for j, m in enumerate(self._feat_meta):
                # Geometric wavelengths from ~4*span down to ~2 (Nyquist for integer resolution 1).
                self.register_buffer(f"_fourier_freq_{j}", self._fourier_freqs(m["nbins"]),
                                     persistent=False)
                fourier_dim += 2 * int(getattr(self, f"_fourier_freq_{j}").numel())
            self.fourier_proj = nn.Linear(fourier_dim, d_model)

        in_dim = (cat_dim if tspec.cat_cols else 0) + static_dim + tspec.num_width + kw_dim + digit_dim
        self.proj = nn.Linear(in_dim, d_model)

    # -- featurization spec ------------------------------------------------------------------------
    @staticmethod
    def _build_feat_meta(tspec: S.TypeSpec) -> List[Dict]:
        """Per featurized numeric column: its index into the num block + range params + digit count. A
        column is featurized iff it has a measured range with > 2 bins (flags / no-range columns are not).
        The bin mapping (``round((symexp(stored)-lo)/res)``) is IDENTICAL to experts.RangeBinHeads."""
        meta: List[Dict] = []
        for f, col in enumerate(_NUM_COLS.get(tspec.name, [])):
            rng = S.NUMERIC_RANGES.get(tspec.name, {}).get(col)
            if rng is None or rng.n_bins <= 2:
                continue
            nd = 1
            while _DIGIT_BASE ** nd < rng.n_bins:
                nd += 1
            meta.append({"f": f, "col": col, "lo": rng.lo, "res": rng.resolution,
                         "nbins": rng.n_bins, "nd": nd})
        return meta

    @staticmethod
    def _fourier_freqs(nbins: int) -> torch.Tensor:
        """Angular frequencies for the value offset ``0..nbins-1``: geometrically spaced wavelengths from
        ~4*span (coarsest) down to ~2 (finest, resolves resolution 1), capped at :data:`_FOURIER_MAX_K`."""
        span = max(1, nbins - 1)
        lam_max, lam_min = 4.0 * span, 2.0
        k = min(_FOURIER_MAX_K, max(1, int(math.ceil(math.log2(lam_max / lam_min))) + 1))
        if k == 1:
            lambdas = [lam_max]
        else:
            lambdas = [lam_max * (lam_min / lam_max) ** (i / (k - 1)) for i in range(k)]
        return torch.tensor([2.0 * math.pi / lam for lam in lambdas], dtype=torch.float32)

    # -- exact recovery of the integer bin from the stored symlog float ----------------------------
    def _bins(self, num: torch.Tensor) -> List[torch.Tensor]:
        """``num`` [..., W] stored block -> list over featurized cols of [...] long bin indices. Uses the
        exact ``round(symexp(.))`` inverse of the tokenizer's symlog (ranged columns are never raw), so the
        recovered integer is exact for any in-range value including negatives — the same mapping the output
        heads bin against."""
        out: List[torch.Tensor] = []
        for m in self._feat_meta:
            col = num[..., m["f"]]
            v = torch.round(torch.sign(col) * torch.expm1(torch.abs(col)))
            b = torch.round((v - m["lo"]) / m["res"]).clamp(0, m["nbins"] - 1).long()
            out.append(b)
        return out

    def digit_features(self, num: torch.Tensor) -> torch.Tensor:
        """Concatenated per-digit embeddings for every featurized column, shape [..., digit_dim]."""
        bins = self._bins(num)
        parts: List[torch.Tensor] = []
        for j, m in enumerate(self._feat_meta):
            b = bins[j]
            for d in range(m["nd"]):
                digit = torch.div(b, _DIGIT_BASE ** d, rounding_mode="floor") % _DIGIT_BASE
                parts.append(self.digit_embs[j][d](digit))
        return torch.cat(parts, dim=-1)

    def fourier_features(self, num: torch.Tensor) -> torch.Tensor:
        """Concatenated sin/cos features for every featurized column, shape [..., fourier_dim]."""
        bins = self._bins(num)
        parts: List[torch.Tensor] = []
        for j in range(len(self._feat_meta)):
            o = bins[j].float().unsqueeze(-1)                      # [..., 1] value offset (linear domain)
            ang = o * getattr(self, f"_fourier_freq_{j}")         # [..., K]
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

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
        has_num = num is not None and self.spec.num_width and self._feat_meta
        if self._use_digits and has_num:
            parts.append(self.digit_features(num))
        out = self.proj(torch.cat(parts, dim=-1))
        if self._use_fourier and has_num:
            out = out + self.fourier_proj(self.fourier_features(num))
        return out


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
                 static_tables=None, latent_mode: str = "flat", latent_k: int = 16):
        super().__init__()
        if latent_mode not in ("flat", "tokens"):
            raise ValueError(f"latent_mode must be 'flat' or 'tokens', got {latent_mode!r}")
        if latent_mode == "flat" and z_dim % simnorm_group != 0:
            raise ValueError(f"z_dim {z_dim} must be divisible by simnorm_group {simnorm_group}")
        if latent_mode == "tokens" and d_model % simnorm_group != 0:
            raise ValueError(f"d_model {d_model} must be divisible by simnorm_group {simnorm_group} "
                             f"(tokens mode applies SimNorm per latent token)")
        self.d_model = d_model
        self.z_dim = z_dim
        self.simnorm_group = simnorm_group
        self.latent_mode = latent_mode
        # In flat mode the pool keeps n_latents inducing points then flattens+projects to z_dim; in tokens
        # mode it keeps latent_k inducing points that ARE the latent (each d_model wide).
        self.latent_k = latent_k if latent_mode == "tokens" else n_latents
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

        self.latents = nn.Parameter(torch.randn(self.latent_k, d_model) * 0.02)
        self.pool = nn.ModuleList(_PoolLayer(d_model, n_heads, ff_mult) for _ in range(n_pool_layers))
        # flat mode flattens the pooled latents and projects to z_dim; tokens mode has no projection.
        self.to_z = nn.Linear(self.latent_k * d_model, z_dim) if latent_mode == "flat" else None

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
        if self.latent_mode == "tokens":
            return simnorm(lat, self.simnorm_group)          # [B, latent_k, d_model], per-token simplices
        z = self.to_z(lat.reshape(B, -1))
        return simnorm(z, self.simnorm_group)                # [B, z_dim]
