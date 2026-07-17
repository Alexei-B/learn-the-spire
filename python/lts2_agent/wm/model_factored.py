"""Factored world-model autoencoder (roadmap M3.5) — ties the per-category experts, the reconstruction
loss, and checkpointing (arch + slice-layout stamped) together.

The state latent is the **concatenation of per-expert slices** (a named, addressable layout the future
predictor reads/writes by slice — never a single opaque vector). Each learned expert applies SimNorm to
its own slice; the tier-1 scalar slice is a deterministic exact code (no SimNorm — that would break its
exactness). ``forward`` returns ``(z, outputs)`` where ``z`` is the concatenated latent and ``outputs`` is
the same per-token-type dict the monolith produces, so :func:`decoder.reconstruct_arrays` and :mod:`report`
consume it verbatim.

Loss = the same three dashboard streams as the monolith (categorical / numeric / presence), summed over
the learned experts. The scalar expert is parameter-free and contributes nothing to the loss (its
reconstruction is exact by construction — see :class:`experts.ScalarCodec`).
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
from . import experts as E

# Default per-expert latent-slice widths (divisible by simnorm_group). Cards is the largest (most
# complex category); scalars is not listed — its deterministic code width is fixed by the field layout.
DEFAULT_SLICE_WIDTHS: Dict[str, int] = {
    "creatures": 768,    # creatures + their powers + intents (folded)
    "cards": 1536,       # the largest — v3 population rows (content + keywords + zone-count vector)
    "relics": 512,       # ~298-way set-membership head needs capacity
    "potions": 128,      # <=8 slots, a 66-way catalog id
    "orbs": 128,         # <=16 slots, a small id + 2 numerics
}


def _static_tables() -> Dict[str, np.ndarray]:
    return {k: catalog.load(k).static_table for k in ("cards", "powers", "relics", "potions")}


class FactoredWorldModelAE(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, enc_layers: int = 2, dec_layers: int = 2,
                 pool_layers: int = 1, pool_latents: int = 4, n_mem: int = 6, cat_dim: int = 24,
                 simnorm_group: int = 8, slice_widths: Optional[Dict[str, int]] = None,
                 static_tables: Optional[Dict[str, np.ndarray]] = None):
        super().__init__()
        widths = dict(DEFAULT_SLICE_WIDTHS)
        if slice_widths:
            widths.update(slice_widths)
        self.slice_widths = widths
        self.cfg = dict(d_model=d_model, n_heads=n_heads, enc_layers=enc_layers, dec_layers=dec_layers,
                        pool_layers=pool_layers, pool_latents=pool_latents, n_mem=n_mem, cat_dim=cat_dim,
                        simnorm_group=simnorm_group, slice_widths=widths)
        if static_tables is None:
            static_tables = _static_tables()
        self.scalars = E.ScalarCodec()
        common = dict(d_model=d_model, static_tables=static_tables, cat_dim=cat_dim, n_heads=n_heads,
                      pool_layers=pool_layers, pool_latents=pool_latents, n_mem=n_mem,
                      simnorm_group=simnorm_group)
        # The two structurally-rich categories (creatures fold powers/intents; cards are the largest,
        # keyword-bearing population) get the full encoder/decoder depth; the small single-type experts
        # (relics/potions/orbs — a flat catalog id + at most two numerics) get 1 enc / 1 dec layer, so
        # their parameter budget matches their content instead of replicating a deep transformer.
        deep = dict(common, enc_layers=enc_layers, dec_layers=dec_layers)
        shallow = dict(common, enc_layers=1, dec_layers=1)
        self.experts = nn.ModuleDict({
            "creatures": E.SetExpert("creatures", ["creature", "power", "intent"],
                                     widths["creatures"], **deep),
            "cards": E.SetExpert("cards", ["card"], widths["cards"], **deep),
            "relics": E.RelicExpert("relics", widths["relics"], **shallow),
            "potions": E.SetExpert("potions", ["potion"], widths["potions"], **shallow),
            "orbs": E.SetExpert("orbs", ["orb"], widths["orbs"], **shallow),
        })
        # Named slice layout (offsets into the concatenated latent) — the predictor addresses by name.
        self.slice_layout: Dict[str, Tuple[int, int]] = {}
        off = 0
        for name in E.EXPERT_ORDER:
            w = self.scalars.width if name == "scalars" else widths[name]
            self.slice_layout[name] = (off, off + w)
            off += w
        self.latent_dim = off

    # -- forward ----------------------------------------------------------------------------------
    def encode_slices(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        slices = {"scalars": self.scalars.encode(batch)}
        for name, ex in self.experts.items():
            slices[name] = ex.encode(batch)
        return slices

    def forward(self, batch: Dict[str, torch.Tensor]):
        slices = self.encode_slices(batch)
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        outputs.update(self.scalars.decode(slices["scalars"]))
        for name, ex in self.experts.items():
            outputs.update(ex.decode(slices[name]))
        z = torch.cat([slices[name] for name in E.EXPERT_ORDER], dim=-1)   # [B, latent_dim]
        return z, outputs

    def slice_layout_list(self) -> List[Dict[str, Any]]:
        """Serializable slice contract (name, start, end, width) in latent order."""
        return [{"name": n, "start": a, "end": b, "width": b - a}
                for n, (a, b) in ((k, self.slice_layout[k]) for k in E.EXPERT_ORDER)]


# ==================================================================================================
# Loss.
# ==================================================================================================

def _masked_mean(per_slot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(per_slot.dtype)
    return (per_slot * m).sum() / m.sum().clamp_min(1.0)


def compute_losses(batch: Dict[str, torch.Tensor],
                   outputs: Dict[str, Dict[str, torch.Tensor]],
                   model: FactoredWorldModelAE,
                   balance: str = "term") -> Dict[str, torch.Tensor]:
    """The three reconstruction losses (categorical / numeric / presence) + their sum. Numerics are
    range-bin cross-entropy (per field, over present slots); the scalar expert contributes nothing
    (exact by construction, no parameters).

    ``balance`` controls gradient allocation across experts:
    - ``"term"`` (legacy): every loss term weighs equally, so an expert's gradient share scales with
      how many terms it owns — cards (~9 terms) drowned relics (1 term) in the first T3 run
      (relics/orbs never learned).
    - ``"expert"``: terms are meaned within each expert first, then experts are meaned — every
      expert gets an equal gradient share regardless of term count.
    """
    cat_terms: List[torch.Tensor] = []
    num_terms: List[torch.Tensor] = []
    pres_terms: List[torch.Tensor] = []
    owner: Dict[int, str] = {}   # id(tensor) -> expert name, for expert-balanced grouping

    def _tag(ename, lst, t):
        owner[id(t)] = ename
        lst.append(t)

    for ename, ex in model.experts.items():
        if ename == "relics":
            o = outputs["relic"]
            t = S.TYPE_BY_NAME["relic"]
            logits = o["set_logits"]
            idx = batch[t.idx_key][..., 0]
            m = batch[t.mask_key]
            tgt = torch.zeros_like(logits)
            b_sel, s_sel = torch.where(m)
            tgt[b_sel, idx[b_sel, s_sel].clamp(0, logits.shape[-1] - 1)] = 1.0
            _tag(ename, cat_terms, F.binary_cross_entropy_with_logits(logits, tgt))
            continue
        for t in ex.types:
            o = outputs[t.name]
            mask = batch[t.mask_key]
            _tag(ename, pres_terms, F.binary_cross_entropy_with_logits(
                o["presence"], mask.to(o["presence"].dtype)))
            if t.cat_cols:
                tgt_idx = batch[t.idx_key]
                for c in range(len(t.cat_cols)):
                    logits = o["cat"][c]
                    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         tgt_idx[..., c].reshape(-1), reduction="none")
                    ce = ce.reshape(tgt_idx.shape[:-1])
                    _tag(ename, cat_terms, _masked_mean(ce, mask))
            if "num_bin_logits" in o:
                head: E.RangeBinHeads = ex.heads[t.name]
                tgt_bins = head.bin_targets(batch[t.num_key])            # [B, slots, W]
                field_ce = []
                for f, logits in enumerate(o["num_bin_logits"]):
                    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         tgt_bins[..., f].reshape(-1), reduction="none")
                    field_ce.append(ce.reshape(tgt_bins.shape[:-1]))
                ce = torch.stack(field_ce, dim=0).mean(dim=0)           # [B, slots]
                _tag(ename, num_terms, _masked_mean(ce, mask))
            if t.has_kw:
                bce = F.binary_cross_entropy_with_logits(o["kw"], batch["card_kw"],
                                                         reduction="none").mean(dim=-1)
                _tag(ename, cat_terms, _masked_mean(bce, mask))

    def _reduce(terms: List[torch.Tensor]) -> torch.Tensor:
        if not terms:
            return torch.zeros((), device=next(model.parameters()).device)
        if balance == "expert":
            by_expert: Dict[str, List[torch.Tensor]] = {}
            for t in terms:
                by_expert.setdefault(owner[id(t)], []).append(t)
            return torch.stack([torch.stack(ts).mean() for ts in by_expert.values()]).mean()
        return torch.stack(terms).mean()

    loss_cat = _reduce(cat_terms)
    loss_num = _reduce(num_terms)
    loss_pres = _reduce(pres_terms)
    total = loss_cat + loss_num + loss_pres
    return {"loss": total, "loss_categorical": loss_cat, "loss_numeric": loss_num,
            "loss_presence": loss_pres}


# ==================================================================================================
# Checkpointing (arch + slice-layout stamped; loads reject a mismatch).
# ==================================================================================================

def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def save_checkpoint(path: str, m: FactoredWorldModelAE, *, step: int = 0,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    extra: Optional[Dict[str, Any]] = None,
                    ema_state: Optional[Dict[str, torch.Tensor]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(m.state_dict(), path)
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, path + ".train")
    if ema_state is not None:
        torch.save(ema_state, path + ".ema")
    meta = {
        "backend": "wm-encdec", "arch": "factored", "config": m.cfg, "step": step,
        "latent_dim": m.latent_dim,
        "slice_layout": m.slice_layout_list(),
        "tokenizer_version": tokens.TOKENIZER_VERSION,
        "tokenizer_signature": tokens.tokenizer_signature(),
        "catalog_signatures": tokens.CATALOG_SIGNATURES,
    }
    if extra:
        meta.update(extra)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: str, device="cpu") -> Tuple[FactoredWorldModelAE, dict]:
    """Load ``(model, meta)``, rejecting a stale tokenizer/catalog signature, a non-factored checkpoint,
    or a slice-layout mismatch (the layout is the predictor's addressing contract)."""
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    want = tokens.tokenizer_signature()
    if meta.get("tokenizer_signature") != want:
        raise ValueError(
            f"Checkpoint {path} was trained with a different tokenizer/catalog "
            f"(meta={meta.get('tokenizer_signature')} vs current {want}); retrain.")
    if meta.get("arch") != "factored":
        raise ValueError(f"Checkpoint {path} arch={meta.get('arch')!r} is not 'factored'; "
                         f"load it with the monolithic loader (wm.model.load_checkpoint).")
    m = FactoredWorldModelAE(**meta["config"])
    got_layout = m.slice_layout_list()
    if meta.get("slice_layout") != got_layout:
        raise ValueError(
            f"Checkpoint {path} slice layout {meta.get('slice_layout')} does not match the current "
            f"model layout {got_layout}; the predictor addresses slices by these offsets.")
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    return m, meta
