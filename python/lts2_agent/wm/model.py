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

Latent contract (M4/predictor note)
------------------------------------
``WorldModelAE`` has two ``latent_mode`` values, selected at construction and stamped into the checkpoint
meta (``latent_mode`` + ``latent_k``): ``flat`` — the encoder returns a single SimNorm vector ``z`` of
shape ``[B, z_dim]`` (today's default, byte-identical); ``tokens`` — the encoder returns a SimNorm token
set of shape ``[B, latent_k, d_model]`` (per-token simplices) which the decoder consumes directly as
memory. The predictor (roadmap M4) that learns dynamics over this latent MUST read ``latent_mode`` (and
``latent_k``) from the checkpoint meta and shape its state accordingly — the latent is now *either*
``z[z_dim]`` *or* a ``latent_k × d_model`` token set, not a fixed flat vector.
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
from .decoder import Decoder, symlog_bins, twohot_targets
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
                 z_dim: int = 512, simnorm_group: int = 8, cat_dim: int = 24, n_mem: int = 16,
                 latent_mode: str = "flat", latent_k: int = 16, num_head: str = "mse"):
        super().__init__()
        self.cfg = dict(d_model=d_model, n_heads=n_heads, enc_layers=enc_layers,
                        dec_layers=dec_layers, n_pool_layers=n_pool_layers, n_latents=n_latents,
                        z_dim=z_dim, simnorm_group=simnorm_group, cat_dim=cat_dim, n_mem=n_mem,
                        latent_mode=latent_mode, latent_k=latent_k, num_head=num_head)
        self.latent_mode = latent_mode
        self.latent_k = latent_k
        self.num_head = num_head
        self.encoder = Encoder(d_model=d_model, n_heads=n_heads, n_layers=enc_layers,
                               n_pool_layers=n_pool_layers, n_latents=n_latents, z_dim=z_dim,
                               simnorm_group=simnorm_group, cat_dim=cat_dim,
                               latent_mode=latent_mode, latent_k=latent_k)
        self.decoder = Decoder(z_dim=z_dim, d_model=d_model, n_heads=n_heads, n_layers=dec_layers,
                               n_mem=n_mem, latent_mode=latent_mode, latent_k=latent_k,
                               num_head=num_head)

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
                   outputs: Dict[str, Dict[str, torch.Tensor]],
                   card_ce_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """The three reconstruction losses + their weighted total.

    Two flag-gated variants (default OFF -> byte-identical to the base loss):
    * numeric two-hot: when a type's decoder output carries ``num_logits`` (``--num-head twohot``), the
      numeric term is a cross-entropy against the two-hot symlog-bin target instead of MSE on ``num``.
    * balanced card CE: when ``card_ce_weights`` is given (``--card-ce balanced``), the card-identity
      column (card type, categorical column 0) is weighted per-class; all other columns are unweighted.
    """
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
                # Class-balanced weights apply ONLY to the card-identity column (card, column 0).
                w = card_ce_weights if (t.name == "card" and c == 0
                                        and card_ce_weights is not None) else None
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     tgt_idx[..., c].reshape(-1), weight=w, reduction="none")
                ce = ce.reshape(tgt_idx.shape[:-1])                   # [B, slots]
                cat_terms.append(_masked_mean(ce, mask) if mask is not None else ce.mean())
        # Numerics over present slots — two-hot CE if the head emitted bin logits, else MSE.
        if t.num_width:
            tgt = batch[t.num_key]
            if "num_logits" in o:
                logits = o["num_logits"]                              # [B, slots, w, bins]
                bins = symlog_bins(logits.shape[-1], device=logits.device, dtype=logits.dtype)
                probs = twohot_targets(tgt, bins)                     # [B, slots, w, bins]
                ce = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean(dim=-1)  # [B, slots]
                num_terms.append(_masked_mean(ce, mask) if mask is not None else ce.mean())
            else:
                pred = o["num"]
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
# Class-balanced card CE (roadmap 3.1 probe: --card-ce balanced).
# ==================================================================================================

def card_ce_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    """Per-class cross-entropy weights for the card-identity vocabulary from an occurrence-count vector.

    Weight ``w_c = 1/sqrt(max(count_c, 1))`` (add-one floor so a never-seen class gets a finite, bounded
    up-weight rather than +inf), then rescaled so the **frequency-weighted mean weight equals 1**
    (``sum_c count_c*w_c / sum_c count_c == 1``). That normalization keeps the *expected* per-sample card
    CE at the same scale as the plain (all-ones) weighting, so the total-loss magnitude is comparable
    across the plain/balanced probe pair. Returns a ``float32`` vector of length ``len(counts)``."""
    counts = np.asarray(counts, dtype=np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    total = float(counts.sum())
    if total > 0:
        avg = float((counts * w).sum() / total)
        if avg > 0:
            w = w / avg
    return w.astype(np.float32)


# ==================================================================================================
# Weight EMA (roadmap 3.1 probe: --ema DECAY). Evaluated on the val pass; saved/restored alongside raw.
# ==================================================================================================

class EMA:
    """Exponential moving average of a model's parameters+buffers, updated once per training step.

    ``update`` blends each floating tensor ``shadow = decay*shadow + (1-decay)*live`` (non-float buffers
    are copied verbatim). For a val pass the trainer calls ``store``/``copy_to`` to swap the EMA weights
    into the live model, runs eval, then ``restore``. ``state_dict``/``load_state_dict`` persist the
    shadow to/from the checkpoint (a ``.ema`` sidecar) so ``--resume`` continues the average."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {k: v.detach().clone()
                                                 for k, v in model.state_dict().items()}
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if torch.is_floating_point(v):
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Snapshot the live weights so ``restore`` can put them back after an EMA-evaluated val pass."""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Load the EMA (shadow) weights into ``model`` (for the val pass)."""
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore the live weights snapshotted by ``store``."""
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None


# ==================================================================================================
# Checkpointing (state_dict + a meta sidecar stamped with the tokenizer parity contract).
# ==================================================================================================

def save_checkpoint(path: str, m: WorldModelAE, *, step: int = 0,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    extra: Optional[Dict[str, Any]] = None,
                    ema_state: Optional[Dict[str, torch.Tensor]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(m.state_dict(), path)
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, path + ".train")
    # EMA weights live in a sidecar so the raw checkpoint stays byte-identical when EMA is off.
    if ema_state is not None:
        torch.save(ema_state, path + ".ema")
    meta = {
        "backend": "wm-encdec", "config": m.cfg, "step": step,
        # Latent contract, surfaced top-level so the M4 predictor can read it without unpacking config.
        "latent_mode": m.cfg.get("latent_mode", "flat"),
        "latent_k": m.cfg.get("latent_k", 16),
        # Numeric-head recipe, surfaced top-level (mse | twohot).
        "num_head": m.cfg.get("num_head", "mse"),
        "tokenizer_version": tokens.TOKENIZER_VERSION,
        "tokenizer_signature": tokens.tokenizer_signature(),
        "catalog_signatures": tokens.CATALOG_SIGNATURES,
    }
    if extra:
        meta.update(extra)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: str, device="cpu",
                    expect_latent_mode: Optional[str] = None,
                    expect_num_head: Optional[str] = None) -> Tuple[WorldModelAE, dict]:
    """Load ``(model, meta)``, rejecting a checkpoint whose tokenizer/catalog signature no longer
    matches (a stale corpus/catalog), exactly like the PPO checkpoints do. When ``expect_latent_mode``
    is given, also reject loudly if the checkpoint's ``latent_mode`` differs (a flat/tokens A/B mixup —
    e.g. resuming a flat run under ``--latent-mode tokens``); ``expect_num_head`` does the same for the
    mse/twohot numeric-head recipe (resuming under a different ``--num-head``)."""
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    want = tokens.tokenizer_signature()
    if meta.get("tokenizer_signature") != want:
        raise ValueError(
            f"Checkpoint {path} was trained with a different tokenizer/catalog "
            f"(meta={meta.get('tokenizer_signature')} vs current {want}); retrain.")
    ckpt_mode = meta["config"].get("latent_mode", "flat")
    if expect_latent_mode is not None and ckpt_mode != expect_latent_mode:
        raise ValueError(
            f"Checkpoint {path} latent_mode={ckpt_mode!r} does not match requested "
            f"latent_mode={expect_latent_mode!r}; the flat and tokens variants are not interchangeable.")
    ckpt_num_head = meta["config"].get("num_head", "mse")
    if expect_num_head is not None and ckpt_num_head != expect_num_head:
        raise ValueError(
            f"Checkpoint {path} num_head={ckpt_num_head!r} does not match requested "
            f"num_head={expect_num_head!r}; the mse and twohot numeric heads are not interchangeable.")
    m = WorldModelAE(**meta["config"])
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    return m, meta


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
