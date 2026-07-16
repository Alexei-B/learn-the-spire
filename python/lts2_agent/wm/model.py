"""World-model encoder/decoder (roadmap 3.1) — ties the encoder + decoder, the reconstruction loss, and
checkpointing (tokenizer/catalog-signature stamped) together.

Loss = ``loss_categorical`` (cross-entropy on every ``*_idx`` column + card-keyword BCE, over *present*
slots) + ``loss_numeric`` (MSE on the symlog ``*_num`` blocks over present slots) + ``loss_presence``
(BCE on per-slot presence for every variable-length type). The three stream to the dashboard separately
so a stalled sub-loss is visible (the product-owner requirement). Single-token types (global/pending) are
always present, so their categorical/numeric terms are unmasked.

Why MSE (not two-hot): the tokenizer already symlog-compresses every numeric, and integer game quantities
round-trip exactly through ``round(symexp(·))``; a scalar-regression target on symlog values is DreamerV3's
symlog-regression and keeps the decoder output *literally* the array ``tokens.detokenize`` inverts, so the
canonical-dict reconstruction is a straight array hand-off with no extra decoding layer. Two-hot would add
a bin vocabulary per field for no accuracy the exact-round-trip already gives us here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import catalog, tokens
from . import spec as S
from .decoder import Decoder
from .encoder import Encoder

# Array keys (subset of tokens.TOKEN_KEYS) the model consumes — the scalar token_type_* are dropped
# (the encoder supplies type identity from its own embedding).
BATCH_KEYS = tuple(k for k in tokens.TOKEN_KEYS if not k.startswith("token_type_"))

_INT_KEYS = {"global_idx", "card_idx", "creature_idx", "power_idx", "intent_idx", "orb_idx",
             "relic_idx", "potion_idx"}
_BOOL_KEYS = {"card_mask", "creature_mask", "power_mask", "intent_mask", "orb_mask", "relic_mask",
              "potion_mask"}


def featurize(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """State -> the model's array dict (unbatched). Non-strict tokenize: a pathological over-cap state
    truncates rather than raising, so streaming never dies on one bad record."""
    tok = tokens.tokenize(state, strict=False)
    return {k: tok[k] for k in BATCH_KEYS}


def collate(feats: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack a list of single-obs array dicts into ``[B, ...]`` arrays."""
    return {k: np.stack([f[k] for f in feats]) for k in BATCH_KEYS}


def to_tensors(stacked: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in BATCH_KEYS:
        v = np.asarray(stacked[k])
        if k in _INT_KEYS:
            out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        elif k in _BOOL_KEYS:
            out[k] = torch.as_tensor(v, dtype=torch.bool, device=device)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


class WorldModelAE(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, enc_layers: int = 4,
                 dec_layers: int = 3, n_pool_layers: int = 2, n_latents: int = 8,
                 z_dim: int = 512, simnorm_group: int = 8, cat_dim: int = 24, n_mem: int = 16):
        super().__init__()
        self.cfg = dict(d_model=d_model, n_heads=n_heads, enc_layers=enc_layers,
                        dec_layers=dec_layers, n_pool_layers=n_pool_layers, n_latents=n_latents,
                        z_dim=z_dim, simnorm_group=simnorm_group, cat_dim=cat_dim, n_mem=n_mem)
        self.encoder = Encoder(d_model=d_model, n_heads=n_heads, n_layers=enc_layers,
                               n_pool_layers=n_pool_layers, n_latents=n_latents, z_dim=z_dim,
                               simnorm_group=simnorm_group, cat_dim=cat_dim)
        self.decoder = Decoder(z_dim=z_dim, d_model=d_model, n_heads=n_heads, n_layers=dec_layers,
                               n_mem=n_mem)

    def forward(self, batch: Dict[str, torch.Tensor]):
        z = self.encoder(batch)
        return z, self.decoder(z)


# ==================================================================================================
# Loss.
# ==================================================================================================

def _masked_mean(per_slot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``per_slot`` over ``mask==True`` entries; 0 (with grad) when nothing is present."""
    m = mask.to(per_slot.dtype)
    denom = m.sum().clamp_min(1.0)
    return (per_slot * m).sum() / denom


def compute_losses(batch: Dict[str, torch.Tensor],
                   outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """The three reconstruction losses + their weighted total."""
    cat_terms: List[torch.Tensor] = []
    num_terms: List[torch.Tensor] = []
    pres_terms: List[torch.Tensor] = []

    for t in S.TYPES:
        o = outputs[t.name]
        if t.mask_key:
            mask = batch[t.mask_key]                                  # [B, slots] bool
        else:
            mask = None
        # Presence (variable types only).
        if t.mask_key:
            pres_logit = o["presence"]
            pres_terms.append(F.binary_cross_entropy_with_logits(pres_logit, mask.to(pres_logit.dtype)))
        # Categoricals over present slots.
        if t.cat_cols:
            tgt_idx = batch[t.idx_key]                                # [B, slots, C]
            for c in range(len(t.cat_cols)):
                logits = o["cat"][c]                                  # [B, slots, V]
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     tgt_idx[..., c].reshape(-1), reduction="none")
                ce = ce.reshape(tgt_idx.shape[:-1])                   # [B, slots]
                cat_terms.append(_masked_mean(ce, mask) if mask is not None else ce.mean())
        # Numerics over present slots.
        if t.num_width:
            pred = o["num"]
            tgt = batch[t.num_key]
            se = F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)  # [B, slots] or [B, 1]
            num_terms.append(_masked_mean(se, mask) if mask is not None else se.mean())
        # Card keyword multi-hot (BCE over present cards) -> categorical bucket.
        if t.has_kw:
            kw_logit = o["kw"]
            kw_tgt = batch["card_kw"]
            bce = F.binary_cross_entropy_with_logits(kw_logit, kw_tgt, reduction="none").mean(dim=-1)
            cat_terms.append(_masked_mean(bce, mask))

    loss_cat = torch.stack(cat_terms).mean()
    loss_num = torch.stack(num_terms).mean()
    loss_pres = torch.stack(pres_terms).mean()
    total = loss_cat + loss_num + loss_pres
    return {"loss": total, "loss_categorical": loss_cat, "loss_numeric": loss_num,
            "loss_presence": loss_pres}


# ==================================================================================================
# Checkpointing (state_dict + a meta sidecar stamped with the tokenizer parity contract).
# ==================================================================================================

def save_checkpoint(path: str, m: WorldModelAE, *, step: int = 0,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(m.state_dict(), path)
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, path + ".train")
    meta = {
        "backend": "wm-encdec", "config": m.cfg, "step": step,
        "tokenizer_version": tokens.TOKENIZER_VERSION,
        "tokenizer_signature": tokens.tokenizer_signature(),
        "catalog_signatures": tokens.CATALOG_SIGNATURES,
    }
    if extra:
        meta.update(extra)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: str, device="cpu") -> Tuple[WorldModelAE, dict]:
    """Load ``(model, meta)``, rejecting a checkpoint whose tokenizer/catalog signature no longer
    matches (a stale corpus/catalog), exactly like the PPO checkpoints do."""
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    want = tokens.tokenizer_signature()
    if meta.get("tokenizer_signature") != want:
        raise ValueError(
            f"Checkpoint {path} was trained with a different tokenizer/catalog "
            f"(meta={meta.get('tokenizer_signature')} vs current {want}); retrain.")
    m = WorldModelAE(**meta["config"])
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    return m, meta


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
