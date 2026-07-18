"""Per-category EXPERTS for the T3 factored autoencoder (roadmap M3.5, design "expert-per-category").

The monolith (:mod:`lts2_agent.wm.encoder`/:mod:`decoder`/:mod:`model`) attends over ONE packed token
set and pools it to a single latent. The factored design instead gives each entity CATEGORY its own
independent expert — a small set autoencoder with its own latent slice — and concatenates the slices into
the state latent. There is deliberately **no cross-category attention inside the AE**: independence keeps
each expert's attention scope tiny (a card expert never attends to creatures), which is both the speed win
and the clean seam the *future* predictor uses to read/write slices by name.

Three tiers:

* **Tier 1 — scalar codec (`ScalarCodec`).** The global token (its 3 enum categoricals + 14 numeric
  fields) and the pending token (4 numerics) are encoded by a **deterministic, parameter-free** codec:
  each field's integer maps to its :data:`spec.NUMERIC_RANGES` bin index, written into the latent slice as
  a one-hot (small enums) or a fixed binary code (numerics). Decode reads the code straight back — exact
  for any in-range integer *by construction*, with no learned weights, so ``eval.scalar_exact`` pins to
  1.0 from the very first val pass (the wiring canary). Out-of-range integers clamp loudly
  (:func:`spec.clamp_to_range`).
* **Tier 2 — small experts.** creatures (folding their powers + intents into one expert), relics
  (positional per-slot categorical — one row per relic instance + a `slot` acquisition-order column,
  duplicates kept), potions (per-slot categorical — potions can duplicate), orbs.
* **Tier 3 — the card-population expert.** the biggest slice; a set encoder/decoder over the v3
  population rows (content categoricals + keyword multi-hot + dynamic numerics + the per-zone count
  vector), every numeric decoded through exact per-field range bins.

All learned numeric decoding uses **per-field range-bin classification** (`RangeBinHeads`) instead of the
monolith's shared symlog MSE regression: creature HP gets resolution-1 bins over ``[0,1000]``, so an
in-distribution HP decodes to the exact integer (no ±1 rounding tail). Each head still emits a ``num``
symlog block *identical in shape/role to the MSE head* (argmax bin → integer → symlog), so
:func:`decoder.reconstruct_arrays` and :mod:`report` consume factored outputs UNCHANGED.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import catalog, tokens
from . import spec as S
from .encoder import _PoolLayer, _TypeEmbedder, simnorm

# Two-hot / distance-aware numeric target (roadmap M3.5 solo-dynamics fix).
#
# The monolith restored metric structure to numeric decoding with DreamerV3 two-hot over a COARSE 64-bin
# symlog grid: real values fall BETWEEN bin centers, so the two-adjacent-bin split genuinely interpolates
# and nearby-bin predictions get partial credit. The factored `RangeBinHeads` instead bin each field at
# integer resolution (`spec.NUMERIC_RANGES`, resolution 1), so every integer target lands EXACTLY on a
# bin center — a literal two-adjacent-bin two-hot degenerates to one-hot and gives no partial credit
# (hard CE's failure mode: predicting bin k±1 costs the same as bin k±100, so the head never learns the
# ordinal geometry — the measured slow, spiky, near-linear ramps).
#
# The faithful generalization for a fine integer grid is a small SYMMETRIC triangular kernel centered on
# the true bin: weights fall off linearly with |bin - center| and renormalize to sum 1. It (a) still puts
# most mass on the exact bin (so argmax decode stays exact — the exact-bin contract), (b) spreads the
# remaining mass to the immediate neighbours so a near-miss is rewarded (restores metric structure), and
# (c) keeps `expectation == center` exactly for a symmetric kernel away from a boundary (the round-trip
# the decoder.py twohot guarantees). half_width 1 == hard one-hot; the default 2 gives the {0.25,0.50,
# 0.25} three-bin kernel. Flag/boolean fields (n_bins<=2) are ALWAYS kept one-hot — smearing a 0/1 flag
# is never desirable.
TWOHOT_HALF_WIDTH = 2

# Numeric columns the tokenizer stores RAW (a plain 0/1 flag) rather than symlog-compressed. Every other
# numeric column is symlog (`tokens.symlog`), inverted by `round(symexp(.))`. These must be known
# per-type so the range-bin target/decode uses the right integer<->stored mapping (mirrors `tokenize`).
RAW_NUM_COLS: Dict[str, set] = {
    "card": {"costsX", "upgraded", "canPlay", "hasDamage", "hasBlock", "hasSummon"},
    "creature": {"active"},
    "intent": {"hasDamage", "hasHits"},
}

# Which expert owns which token types (partition of tokens.TOKEN_TYPES, used by report/expert_dist).
EXPERT_TYPES: Dict[str, List[str]] = {
    "scalars": ["global", "pending"],
    "creatures": ["creature", "power", "intent"],
    "cards": ["card"],
    "relics": ["relic"],
    "potions": ["potion"],
    "orbs": ["orb"],
}
EXPERT_ORDER = ["scalars", "creatures", "cards", "relics", "potions", "orbs"]


def t_symexp(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * torch.expm1(torch.abs(y))


def t_symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _effective_range(type_name: str, col: str) -> S.RangeSpec:
    """Measured range for a numeric column, or a 0/1 flag range for the boolean/presence columns that
    carry no measured range (they are 0/1 by construction)."""
    rng = S.NUMERIC_RANGES.get(type_name, {}).get(col)
    return rng if rng is not None else S.RangeSpec(0, 1, 1)


# ==================================================================================================
# Tier-1: deterministic scalar codec (parameter-free, exact by construction).
# ==================================================================================================

def _n_bits(n_bins: int) -> int:
    return max(1, int(math.ceil(math.log2(max(2, n_bins)))))


class ScalarCodec(nn.Module):
    """Deterministic, parameter-free autoencoder for the global + pending tokens (tier 1).

    Latent slice layout (all values in ``[0,1]``): the global enum categoricals as concatenated
    one-hots, then the global numerics as concatenated fixed-width binary bin codes, then the pending
    numerics likewise. Because both the encode (integer → bin index → code) and the decode (code → bin
    index → integer) are fixed functions with no learned parameters, the round-trip is EXACT for any
    in-range integer regardless of training — this is what makes ``eval.scalar_exact`` a real wiring
    canary (it is 1.0 at step 0). Out-of-range integers clamp loudly via the range's ``[lo,hi]``."""

    def __init__(self) -> None:
        super().__init__()
        gspec = S.TYPE_BY_NAME["global"]
        # Global categoricals (phase, side, turnPhase): one-hot each.
        self.g_cat_vocab: List[int] = [v for _, v in gspec.cat_cols]
        # Global numerics: (col, RangeSpec, is_raw=False — all global nums are symlog-stored).
        self.g_num: List[Tuple[str, S.RangeSpec]] = [(c, _effective_range("global", c))
                                                     for c in tokens.GLOBAL_NUM]
        # Pending numerics: [present(raw flag), minSelect(symlog), maxSelect(symlog),
        # isUpgradeSelection(raw flag)] — matches tokenize()'s pending block.
        self.p_fields: List[Tuple[str, S.RangeSpec, bool]] = [
            ("present", S.RangeSpec(0, 1, 1), True),
            ("minSelect", _effective_range("pending", "minSelect"), False),
            ("maxSelect", _effective_range("pending", "maxSelect"), False),
            ("isUpgradeSelection", S.RangeSpec(0, 1, 1), True),
        ]
        self.g_num_bits = [_n_bits(r.n_bins) for _, r in self.g_num]
        self.p_bits = [_n_bits(r.n_bins) for _, r, _ in self.p_fields]
        self.width = (sum(self.g_cat_vocab) + sum(self.g_num_bits) + sum(self.p_bits))
        # Buffers for vectorized bit (de)coding.
        self.register_buffer("_pow2_g", torch.tensor([1 << i for i in range(max(self.g_num_bits))],
                                                     dtype=torch.long), persistent=False)

    # -- helpers ----------------------------------------------------------------------------------
    @staticmethod
    def _to_bits(idx: torch.Tensor, nbits: int) -> torch.Tensor:
        """``idx`` [B] long -> [B, nbits] float bits (LSB first)."""
        shifts = torch.arange(nbits, device=idx.device)
        return ((idx.unsqueeze(-1) >> shifts) & 1).to(torch.float32)

    @staticmethod
    def _from_bits(bits: torch.Tensor) -> torch.Tensor:
        """``bits`` [B, nbits] (any real; thresholded at 0.5) -> [B] long integer (LSB first)."""
        nbits = bits.shape[-1]
        w = (2 ** torch.arange(nbits, device=bits.device)).to(torch.long)
        return ((bits >= 0.5).to(torch.long) * w).sum(dim=-1)

    def _num_to_bins(self, col: torch.Tensor, r: S.RangeSpec, is_raw: bool) -> torch.Tensor:
        v = col.round() if is_raw else t_symexp(col).round()
        idx = ((v - r.lo) / r.resolution).round().long()
        return idx.clamp(0, r.n_bins - 1)

    # -- encode -----------------------------------------------------------------------------------
    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        gi = batch["global_idx"][:, 0, :].long()          # [B, 3]
        gn = batch["global_num"][:, 0, :]                 # [B, 14]
        pv = batch["pending"][:, 0, :]                    # [B, 4]
        parts: List[torch.Tensor] = []
        for c, vocab in enumerate(self.g_cat_vocab):
            parts.append(torch.nn.functional.one_hot(gi[:, c].clamp(0, vocab - 1), vocab).float())
        for j, (_, r) in enumerate(self.g_num):
            idx = self._num_to_bins(gn[:, j], r, is_raw=False)
            parts.append(self._to_bits(idx, self.g_num_bits[j]))
        for j, (_, r, is_raw) in enumerate(self.p_fields):
            idx = self._num_to_bins(pv[:, j], r, is_raw=is_raw)
            parts.append(self._to_bits(idx, self.p_bits[j]))
        return torch.cat(parts, dim=-1)                   # [B, width], in {0,1}

    # -- decode -----------------------------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        B = z.shape[0]
        off = 0
        # Global categoricals -> one-hot logits [B,1,vocab] (argmax recovers the exact index).
        cats: List[torch.Tensor] = []
        for vocab in self.g_cat_vocab:
            seg = z[:, off:off + vocab]
            off += vocab
            cats.append(seg.unsqueeze(1))                 # already a (near-)one-hot; argmax == exact
        # Global numerics -> symlog block [B,1,14].
        g_num = torch.zeros(B, len(self.g_num), device=z.device)
        for j, (_, r) in enumerate(self.g_num):
            nb = self.g_num_bits[j]
            idx = self._from_bits(z[:, off:off + nb]).clamp(0, r.n_bins - 1)
            off += nb
            g_num[:, j] = t_symlog((r.lo + idx * r.resolution).float())
        # Pending -> [present, symlog(min), symlog(max), isUpgrade] block [B,1,4].
        p_num = torch.zeros(B, len(self.p_fields), device=z.device)
        for j, (_, r, is_raw) in enumerate(self.p_fields):
            nb = self.p_bits[j]
            idx = self._from_bits(z[:, off:off + nb]).clamp(0, r.n_bins - 1)
            off += nb
            val = (r.lo + idx * r.resolution).float()
            p_num[:, j] = val if is_raw else t_symlog(val)
        return {
            "global": {"cat": cats, "num": g_num.unsqueeze(1)},
            "pending": {"cat": [], "num": p_num.unsqueeze(1)},
        }


# ==================================================================================================
# Range-bin per-type heads (learned experts). Categoricals + per-field range-bin numerics + presence + kw.
# ==================================================================================================

class RangeBinHeads(nn.Module):
    """Per-slot heads for one token type. Categorical columns are per-column linear→vocab (CE). Numerics
    are decoded as **per-field classification over the measured integer range** (`spec.NUMERIC_RANGES`):
    one linear head emits ``sum(n_bins)`` logits, split per field; argmax → bin index → integer → the
    same symlog ``num`` block the MSE head produced (so downstream reconstruct/report are unchanged). A
    flag column (no measured range) is a 2-bin head over {0,1}."""

    def __init__(self, tspec: S.TypeSpec, d_model: int):
        super().__init__()
        self.spec = tspec
        self.cat_heads = nn.ModuleList(nn.Linear(d_model, v) for _, v in tspec.cat_cols)
        self.presence_head = nn.Linear(d_model, 1) if tspec.mask_key else None
        self.kw_head = nn.Linear(d_model, tokens.KW_BUCKETS) if tspec.has_kw else None
        self.num_cols: List[str] = list(_num_cols(tspec))
        if self.num_cols:
            raw = RAW_NUM_COLS.get(tspec.name, set())
            ranges = [_effective_range(tspec.name, c) for c in self.num_cols]
            self.register_buffer("_lo", torch.tensor([r.lo for r in ranges], dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("_res", torch.tensor([r.resolution for r in ranges],
                                                      dtype=torch.float32), persistent=False)
            self.register_buffer("_nbins", torch.tensor([r.n_bins for r in ranges], dtype=torch.long),
                                 persistent=False)
            self.register_buffer("_is_raw", torch.tensor([c in raw for c in self.num_cols],
                                                        dtype=torch.bool), persistent=False)
            self._bin_sizes = [r.n_bins for r in ranges]
            self.num_head = nn.Linear(d_model, int(sum(self._bin_sizes)))
        else:
            self.num_head = None
            self._bin_sizes = []

    def bin_targets(self, num: torch.Tensor) -> torch.Tensor:
        """``num`` [B, slots, W] stored block -> [B, slots, W] long bin-index targets (for the CE loss)."""
        v = torch.where(self._is_raw, num.round(), t_symexp(num).round())
        idx = ((v - self._lo) / self._res).round().long()
        return idx.clamp(min=0).minimum(self._nbins - 1)

    def soft_bin_targets(self, num: torch.Tensor,
                         half_width: int = TWOHOT_HALF_WIDTH) -> List[torch.Tensor]:
        """Distance-aware (two-hot-style) soft targets: a list of ``W`` tensors, entry ``f`` shaped
        ``[B, slots, n_bins_f]`` — a symmetric triangular distribution centred on the true bin (weights
        ``max(0, half_width - |bin - center|)`` renormalized to sum 1). ``half_width==1`` reproduces the
        hard one-hot; flag fields (``n_bins<=2``) are always one-hot (a boolean must not smear). See
        :data:`TWOHOT_HALF_WIDTH` for why the fine integer grid needs a kernel rather than a literal
        two-adjacent-bin split."""
        centers = self.bin_targets(num)                              # [B, slots, W] long
        out: List[torch.Tensor] = []
        for f in range(len(self.num_cols)):
            nb = int(self._nbins[f].item())
            c = centers[..., f]                                       # [B, slots] long
            if nb <= 2 or half_width <= 1:
                out.append(F.one_hot(c, nb).to(torch.float32))
                continue
            bins = torch.arange(nb, device=c.device).view(*([1] * c.dim()), nb)   # [1,..,nb]
            d = (bins - c.unsqueeze(-1)).abs().to(torch.float32)      # [B, slots, nb]
            w = (half_width - d).clamp(min=0.0)
            out.append(w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return out

    def expectation_decode(self, num_bin_logits: List[torch.Tensor]) -> torch.Tensor:
        """Alternate numeric decode: per-field softmax expectation over bin centers, rounded to the
        nearest bin, mapped back to the stored (symlog / raw) block — the shape/role of the argmax decode
        in :meth:`forward`. Kept out of the default path: argmax preserves the exact-bin contract, whereas
        a straddling expectation can round to a neighbour and miss the exact integer (measured on the
        probe). Used by the diagnostic to compare the two decodes."""
        idxs = []
        for f, logits in enumerate(num_bin_logits):
            nb = logits.shape[-1]
            centers = torch.arange(nb, device=logits.device, dtype=torch.float32)
            e = (logits.float().softmax(dim=-1) * centers).sum(dim=-1)   # [B, slots] expected bin
            idxs.append(e.round().clamp(0, nb - 1).long())
        idx = torch.stack(idxs, dim=-1)                              # [B, slots, W]
        val = self._lo + idx.float() * self._res
        return torch.where(self._is_raw, val, t_symlog(val))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        out["cat"] = [head(h) for head in self.cat_heads]
        if self.num_head is not None:
            logits = self.num_head(h)                                  # [B, slots, sum_bins]
            per_field = list(torch.split(logits, self._bin_sizes, dim=-1))
            out["num_bin_logits"] = per_field
            # Decode: argmax bin -> integer -> the stored (symlog / raw) block, identical to the MSE head.
            idxs = torch.stack([f.argmax(dim=-1) for f in per_field], dim=-1)  # [B, slots, W]
            val = (self._lo + idxs.float() * self._res)
            out["num"] = torch.where(self._is_raw, val, t_symlog(val))
        if self.presence_head is not None:
            out["presence"] = self.presence_head(h).squeeze(-1)
        if self.kw_head is not None:
            out["kw"] = self.kw_head(h)
        return out


def _num_cols(tspec: S.TypeSpec) -> List[str]:
    """The tokenizer's numeric column names for a type (empty when the type has no numeric block)."""
    return {
        "card": tokens.CARD_NUM, "creature": tokens.CREATURE_NUM, "power": tokens.POWER_NUM,
        "intent": tokens.INTENT_NUM, "orb": tokens.ORB_NUM,
    }.get(tspec.name, [])


# ==================================================================================================
# Generic set expert (creatures / cards / orbs / potions).
# ==================================================================================================

class SetExpert(nn.Module):
    """A self-contained set autoencoder over one or more token types (its category). Encoder: per-type
    embed → self-attention over ONLY this category's tokens → Perceiver pool → SimNorm'd latent slice.
    Decoder: slice → memory tokens → per-type learned slot queries cross-attend → :class:`RangeBinHeads`.

    ``child_parent`` types (power/intent) add a parent-slot embedding so the folded children keep their
    creature association inside the creatures expert."""

    def __init__(self, name: str, type_names: List[str], latent_width: int, d_model: int,
                 static_tables: Dict[str, np.ndarray], cat_dim: int = 24, n_heads: int = 4,
                 enc_layers: int = 2, dec_layers: int = 2, pool_layers: int = 1, pool_latents: int = 4,
                 n_mem: int = 6, ff_mult: int = 2, simnorm_group: int = 8):
        super().__init__()
        if latent_width % simnorm_group != 0:
            raise ValueError(f"{name} latent_width {latent_width} not divisible by simnorm_group "
                             f"{simnorm_group}")
        self.name = name
        self.types = [S.TYPE_BY_NAME[n] for n in type_names]
        self.latent_width = latent_width
        self.simnorm_group = simnorm_group
        self.d_model = d_model
        self.n_mem = n_mem
        self.embedders = nn.ModuleDict(
            {t.name: _TypeEmbedder(t, d_model, cat_dim, static_tables) for t in self.types})
        self.enc_type_emb = nn.Embedding(len(self.types), d_model)
        self.parent_emb = nn.Embedding(tokens.MAX_CREATURES, d_model)
        # Always-valid sentinel token: a category can be empty in a sample (no orbs / no potions), which
        # would leave the encoder set fully padded -> NaN softmax. This learned token is never padded, so
        # every sample has >=1 valid key; it is not decoded (pure attention anchor).
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * ff_mult,
                                               batch_first=True, norm_first=True, activation="gelu")
        self.enc_trunk = nn.TransformerEncoder(enc_layer, enc_layers)
        self.latents = nn.Parameter(torch.randn(pool_latents, d_model) * 0.02)
        self.pool = nn.ModuleList(_PoolLayer(d_model, n_heads, ff_mult) for _ in range(pool_layers))
        self.to_slice = nn.Linear(pool_latents * d_model, latent_width)
        # Normalize the pre-SimNorm logits. Without this the unbounded `to_slice` output runs away in
        # magnitude (measured: pre-SimNorm std 0.1 -> 217 across states) and the grouped softmax SATURATES
        # to a state-INDEPENDENT one-hot whose gradient vanishes — a representation-collapse runaway that
        # floors solo learning (the fixed batch could not be overfit; the curves were slow, spiky, non-
        # log). LayerNorm (no affine, so weight_decay can't shrink a scale back toward the uniform-softmax
        # collapse) bounds the input so SimNorm stays sensitive to the state. The output is still a
        # concatenation of probability simplices (SimNorm's contract for the predictor).
        self.slice_norm = nn.LayerNorm(latent_width, elementwise_affine=False)
        # Decoder.
        self.from_slice = nn.Linear(latent_width, n_mem * d_model)
        self.mem_norm = nn.LayerNorm(d_model)
        self.slot_queries = nn.ParameterDict(
            {t.name: nn.Parameter(torch.randn(t.max_slots, d_model) * 0.02) for t in self.types})
        self.dec_type_emb = nn.Embedding(len(self.types), d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model * ff_mult,
                                               batch_first=True, norm_first=True, activation="gelu")
        self.dec_trunk = nn.TransformerDecoder(dec_layer, dec_layers)
        self.heads = nn.ModuleDict({t.name: RangeBinHeads(t, d_model) for t in self.types})
        self._offsets: Dict[str, Tuple[int, int]] = {}
        off = 0
        for t in self.types:
            self._offsets[t.name] = (off, off + t.max_slots)
            off += t.max_slots

    def _embed(self, batch: Dict[str, torch.Tensor]):
        B = batch["global_idx"].shape[0]
        device = batch["global_idx"].device
        seq = [self.cls.view(1, 1, -1).expand(B, 1, -1)]
        masks = [torch.ones(B, 1, dtype=torch.bool, device=device)]
        for ti, t in enumerate(self.types):
            idx = batch.get(t.idx_key) if t.idx_key else None
            num = batch.get(t.num_key) if t.num_key else None
            kw = batch.get("card_kw") if t.has_kw else None
            emb = self.embedders[t.name](idx, num, kw) + self.enc_type_emb.weight[ti]
            if t.name in ("power", "intent"):
                emb = emb + self.parent_emb(idx[..., 1])
            seq.append(emb)
            masks.append(batch[t.mask_key])
        toks = torch.cat(seq, dim=1)
        valid = torch.cat(masks, dim=1)
        return toks, ~valid

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        toks, key_pad = self._embed(batch)
        toks = self.enc_trunk(toks, src_key_padding_mask=key_pad)
        B = toks.shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.pool:
            lat = layer(lat, toks, key_pad)
        z = self.to_slice(lat.reshape(B, -1))
        return simnorm(self.slice_norm(z), self.simnorm_group)

    def decode(self, z: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        B = z.shape[0]
        mem = self.mem_norm(self.from_slice(z).reshape(B, self.n_mem, self.d_model))
        q = []
        for ti, t in enumerate(self.types):
            q.append(self.slot_queries[t.name].unsqueeze(0).expand(B, -1, -1)
                     + self.dec_type_emb.weight[ti])
        h = self.dec_trunk(torch.cat(q, dim=1), mem)
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for t in self.types:
            a, b = self._offsets[t.name]
            out[t.name] = self.heads[t.name](h[:, a:b, :])
        return out


# ==================================================================================================
# Relics (v5) use the generic SetExpert directly: they are now a POSITIONAL single-type set (one row per
# relic instance, an explicit `slot` categorical carrying acquisition order) — structurally identical to
# potions/orbs. The old bespoke RelicExpert (duplicate-free set-membership head + cardinality head, and
# the slots-variant's confidence dedup) is deleted: a positional slot decode neither needs nor wants a
# set head, so relics ride the same per-slot categorical + presence path as every other set type.
# ==================================================================================================
