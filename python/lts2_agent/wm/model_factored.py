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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import catalog, tokens
from . import spec as S
from . import experts as E

# Default per-expert latent-slice widths (divisible by simnorm_group). Cards is the largest (most
# complex category); scalars is not listed — its deterministic code width is fixed by the field layout.
#
# Creature-family split (owner ruling, 2026-07-18): the old single "creatures" slice (768) floored at dist
# ~0.29 and was judged too big. It is replaced by three parameter-disjoint experts. Each new width is sized
# from capacity arithmetic — a SimNorm slice of width W (group 8) carries (W/8)*log2(8) = (W/8)*3 robust
# bits, and the sizing statistic is the p99 of the per-STATE information cost (convergence.state_bits_sample,
# split into per-type components), targeting (W/8)*3 >= ~1.3x p99 at the smallest multiple of 64:
#   creature-stats   p99 138.4 bits -> 1.3x = 180 -> W=512  (cap (512/8)*3 = 192 bits)
#   creature-powers  p99 140.7 bits -> 1.3x = 183 -> W=512  (cap 192 bits)
#   creature-intents p99  62.6 bits -> 1.3x =  81 -> W=256  (cap  96 bits)
DEFAULT_SLICE_WIDTHS: Dict[str, int] = {
    "creature-stats": 512,     # the `creature` token type (identity + kind + HP/block/... numerics)
    "creature-powers": 512,    # the `power` token type (powerIndex + amount + parent-creature slot)
    "creature-intents": 256,   # the `intent` token type (type + damage/hits numerics + parent slot)
    "cards": 1536,       # the largest — v3 population rows (content + keywords + zone-count vector)
    "relics": 512,       # positional relic rows (~298-way id + slot); ample capacity for the set
    "potions": 128,      # <=8 slots, a 66-way catalog id
    "orbs": 128,         # <=16 slots, a small id + 2 numerics
}


def _static_tables() -> Dict[str, np.ndarray]:
    return {k: catalog.load(k).static_table for k in ("cards", "powers", "relics", "potions")}


class FactoredWorldModelAE(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, enc_layers: int = 2, dec_layers: int = 2,
                 pool_layers: int = 1, pool_latents: int = 4, n_mem: int = 6, cat_dim: int = 24,
                 simnorm_group: int = 8, slice_widths: Optional[Dict[str, int]] = None,
                 expert_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 static_tables: Optional[Dict[str, np.ndarray]] = None, qk_norm: bool = True,
                 num_head: str = "bins", num_decode: str = "expected"):
        super().__init__()
        widths = dict(DEFAULT_SLICE_WIDTHS)
        if slice_widths:
            widths.update(slice_widths)
        self.slice_widths = widths
        # QK-norm in the expert trunks (Wortsman et al. 2023 fix (a)) — model-wide flag, default ON for new
        # runs. It bounds the attention logits that ratchet into the flat-LR collapse (grad-norm 17 -> 445
        # -> 1171 into an all-absent presence solution). Stamped in cfg (below) so save/load/compose
        # reconstruct the right trunk architecture; an OLD checkpoint's meta has NO qk_norm key, which
        # load_checkpoint reads as qk_norm=False (stock trunks) so it loads byte-identically.
        self.qk_norm = qk_norm
        # Numeric HEAD layout ("bins" | "digits") + eval DECODE mode ("argmax" | "expected"), model-wide
        # and stamped in cfg (below) so save/load/compose reconstruct the right head shape and decode. Old
        # checkpoints have neither key: load_checkpoint defaults them to "bins"/"argmax" so they rebuild the
        # flat head and reproduce their historically-reported (argmax) metrics byte-identically.
        self.num_head = num_head
        self.num_decode = num_decode
        # Per-expert construction-kwarg overrides (e.g. a deeper relic decoder, a per-expert n_mem A/B) —
        # stamped in cfg so save/load/compose reconstruct the exact per-expert architecture.
        self.expert_overrides = {k: dict(v) for k, v in (expert_overrides or {}).items()}
        self.cfg = dict(d_model=d_model, n_heads=n_heads, enc_layers=enc_layers, dec_layers=dec_layers,
                        pool_layers=pool_layers, pool_latents=pool_latents, n_mem=n_mem, cat_dim=cat_dim,
                        simnorm_group=simnorm_group, slice_widths=widths, qk_norm=qk_norm,
                        num_head=num_head, num_decode=num_decode,
                        expert_overrides=self.expert_overrides)
        if static_tables is None:
            static_tables = _static_tables()
        self.scalars = E.ScalarCodec()
        common = dict(d_model=d_model, static_tables=static_tables, cat_dim=cat_dim, n_heads=n_heads,
                      pool_layers=pool_layers, pool_latents=pool_latents, n_mem=n_mem,
                      simnorm_group=simnorm_group, qk_norm=qk_norm, num_head=num_head,
                      num_decode=num_decode)
        # The structurally-rich categories get the full encoder/decoder depth; the small single-type
        # experts (relics/potions/orbs — a flat catalog id + at most two numerics) get 1 enc / 1 dec layer,
        # so their parameter budget matches their content instead of replicating a deep transformer. The
        # creature family (owner ruling, 2026-07-18) is split into three parameter-disjoint single-type
        # experts — creature-stats / creature-powers / creature-intents — each kept at the SAME deep depth
        # the old folded "creatures" expert used, so a sub-task loses no modelling power from the split
        # (powers/intents retain real numerics + a parent-slot embedding + up to MAX_POWERS/MAX_INTENTS
        # slots, richer than any flat catalog).
        deep = dict(common, enc_layers=enc_layers, dec_layers=dec_layers)
        shallow = dict(common, enc_layers=1, dec_layers=1)
        ov = self.expert_overrides

        def _kw(name: str, base: Dict[str, Any]) -> Dict[str, Any]:
            return dict(base, **ov.get(name, {}))

        self.experts = nn.ModuleDict({
            "creature-stats": E.SetExpert("creature-stats", ["creature"], widths["creature-stats"],
                                          **_kw("creature-stats", deep)),
            "creature-powers": E.SetExpert("creature-powers", ["power"], widths["creature-powers"],
                                           **_kw("creature-powers", deep)),
            "creature-intents": E.SetExpert("creature-intents", ["intent"], widths["creature-intents"],
                                            **_kw("creature-intents", deep)),
            "cards": E.SetExpert("cards", ["card"], widths["cards"], **_kw("cards", deep)),
            "relics": E.SetExpert("relics", ["relic"], widths["relics"], **_kw("relics", shallow)),
            "potions": E.SetExpert("potions", ["potion"], widths["potions"], **_kw("potions", shallow)),
            "orbs": E.SetExpert("orbs", ["orb"], widths["orbs"], **_kw("orbs", shallow)),
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

    def forward(self, batch: Dict[str, torch.Tensor],
                active_experts: Optional[Iterable[str]] = None):
        """Autoencode the batch. ``active_experts`` (per-expert training / trained-only val) restricts
        both the encode AND the decode to the named experts — a frozen expert's forward is SKIPPED
        entirely (no wasted compute, no output), and the concatenated latent ``z`` is returned only for a
        full pass (``None``). Default ``None`` == the full model (byte-identical to the joint path)."""
        active: Optional[Set[str]] = None if active_experts is None else set(active_experts)
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        slices: Dict[str, torch.Tensor] = {}
        if active is None or "scalars" in active:
            slices["scalars"] = self.scalars.encode(batch)
            outputs.update(self.scalars.decode(slices["scalars"]))
        for name, ex in self.experts.items():
            if active is not None and name not in active:
                continue
            slices[name] = ex.encode(batch)
            outputs.update(ex.decode(slices[name]))
        z = (torch.cat([slices[name] for name in E.EXPERT_ORDER], dim=-1)
             if active is None else None)                                  # [B, latent_dim] full only
        return z, outputs

    def slice_layout_list(self) -> List[Dict[str, Any]]:
        """Serializable slice contract (name, start, end, width) in latent order."""
        return [{"name": n, "start": a, "end": b, "width": b - a}
                for n, (a, b) in ((k, self.slice_layout[k]) for k in E.EXPERT_ORDER)]

    # -- per-expert addressing (freeze/skip, warm-start, compose) ----------------------------------
    def expert_params(self, name: str) -> List[nn.Parameter]:
        """The parameter list of one learned expert (``self.experts[name]``) — the freeze/skip and
        per-expert-optimizer seam. ``scalars`` is parameter-free (returns [])."""
        if name == "scalars":
            return []
        return list(self.experts[name].parameters())

    def expert_stamp(self, name: str) -> Dict[str, Any]:
        """Per-expert provenance stamp: its slice layout + the tokenizer signature + the exact kwargs its
        module was built with (any per-expert override). The compose/warm-start contract validates
        against this, so a slice can only be assembled into a checkpoint whose layout it fits."""
        a, b = self.slice_layout[name]
        cfg: Dict[str, Any] = {"d_model": self.cfg["d_model"], "n_heads": self.cfg["n_heads"],
                               "cat_dim": self.cfg["cat_dim"], "simnorm_group": self.cfg["simnorm_group"],
                               "pool_layers": self.cfg["pool_layers"],
                               "pool_latents": self.cfg["pool_latents"], "n_mem": self.cfg["n_mem"],
                               "qk_norm": self.cfg["qk_norm"], "num_head": self.cfg["num_head"],
                               "num_decode": self.cfg["num_decode"]}
        cfg.update(self.expert_overrides.get(name, {}))
        return {"name": name, "slice": [a, b], "width": b - a,
                "tokenizer_signature": tokens.tokenizer_signature(), "config": cfg,
                "param_count": param_count(self.experts[name]) if name != "scalars" else 0}

    def expert_stamps(self) -> Dict[str, Dict[str, Any]]:
        return {n: self.expert_stamp(n) for n in E.EXPERT_ORDER}

    def load_expert_from_state_dict(self, name: str, src_state: Dict[str, torch.Tensor]) -> None:
        """Copy one expert's weights out of a full factored ``state_dict`` (keys ``experts.<name>.*``)
        into this model — the warm-start / compose primitive.

        Non-trunk weights (embedders, to_slice, slice_norm, from_slice, heads, slot_queries, cls,
        latents, ...) MUST match by name and shape (a mismatch there is a genuine incompatibility and
        raises). The attention TRUNK (``enc_trunk``/``pool``/``dec_trunk``) is exempt: turning qk_norm on
        vs off changes the trunk's param NAMES, so warm-starting a qk_norm=True expert from an OLD
        (qk_norm=False) checkpoint would otherwise fail. Such missing/mismatched trunk keys are SKIPPED
        with a printed notice (the fresh trunk keeps its init), so the expensive non-trunk weights still
        warm-start across the fix."""
        if name == "scalars":
            return   # parameter-free deterministic codec; nothing to copy
        prefix = f"experts.{name}."
        own = self.experts[name].state_dict()
        sub = {k[len(prefix):]: v for k, v in src_state.items() if k.startswith(prefix)}
        missing = set(own) - set(sub)
        mismatched = {k for k in own if k in sub and tuple(sub[k].shape) != tuple(own[k].shape)}
        skip = missing | mismatched
        non_trunk = {k for k in skip if not k.startswith(E.TRUNK_PREFIXES)}
        if non_trunk:
            raise ValueError(f"source checkpoint is missing/mismatched expert {name!r} non-trunk params: "
                             f"{sorted(non_trunk)[:4]}{'...' if len(non_trunk) > 4 else ''}")
        if skip:
            print(f"[model_factored] warm-start {name!r}: skipping {len(skip)} trunk param(s) "
                  f"(qk-norm trunk architecture differs from source); non-trunk weights copied.",
                  flush=True)
        self.experts[name].load_state_dict({k: sub[k] for k in own if k not in skip}, strict=False)


# ==================================================================================================
# Loss.
# ==================================================================================================

def _masked_mean(per_slot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(per_slot.dtype)
    return (per_slot * m).sum() / m.sum().clamp_min(1.0)


def compute_losses(batch: Dict[str, torch.Tensor],
                   outputs: Dict[str, Dict[str, torch.Tensor]],
                   model: FactoredWorldModelAE,
                   balance: str = "term",
                   active: Optional[Iterable[str]] = None,
                   num_targets: str = "hard",
                   num_loss_norm: str = "none",
                   z_weight: float = 0.0) -> Dict[str, torch.Tensor]:
    """The three reconstruction losses (categorical / numeric / presence) + their sum. Numerics are
    range-bin cross-entropy (per field, over present slots); the scalar expert contributes nothing
    (exact by construction, no parameters).

    ``z_weight`` (default 0.0 = OFF, so existing behavior is byte-identical) adds Wortsman et al. 2023
    fix (b) — an output z-loss ``z_weight * mean(logsumexp(logits)^2)`` over the categorical + presence
    classification heads. It keeps each head's log-partition (logsumexp) near 0, i.e. it penalizes the
    output logits from growing in magnitude — the OTHER half of the flat-LR collapse (the grad-norm
    ratchet into a degenerate all-absent presence solution) that QK-norm alone does not fully bound.
    Emitted as ``loss_zloss`` (0.0 when off).

    ``num_targets`` picks the range-bin target geometry: ``"hard"`` (legacy) is one-hot CE on the exact
    bin — no partial credit for a near-miss, which loses the numeric metric structure; ``"twohot"`` is a
    distance-aware symmetric triangular target (:meth:`experts.RangeBinHeads.soft_bin_targets`) that
    rewards nearby-bin predictions, restoring the ordinal geometry (the M3.5 solo-dynamics fix). A DIGIT
    column (num_head="digits") ignores this and always uses a per-digit one-hot CE.

    ``num_loss_norm`` scales the per-field numeric CE before the fields are averaged: ``"none"`` (default,
    byte-identical to prior behavior) sums raw CEs, so a 1001-bin field (~ln 1001 = 6.9 at chance)
    dominates the summed numeric gradient over a converged narrow field (~0); ``"logbins"`` divides each
    field's CE by ``ln(max(n_bins, 2))`` so every numeric column contributes comparably.

    ``balance`` controls gradient allocation across experts:
    - ``"term"`` (legacy): every loss term weighs equally, so an expert's gradient share scales with
      how many terms it owns — cards (~9 terms) drowned relics (1 term) in the first T3 run
      (relics/orbs never learned).
    - ``"expert"``: terms are meaned within each expert first, then experts are meaned — every
      expert gets an equal gradient share regardless of term count.

    ``active`` (per-expert training) restricts the loss to the named experts — a frozen/skipped expert
    contributes no term (its outputs are absent). ``None`` == every expert (the joint default).
    """
    active_set: Optional[Set[str]] = None if active is None else set(active)
    cat_terms: List[torch.Tensor] = []
    num_terms: List[torch.Tensor] = []
    pres_terms: List[torch.Tensor] = []
    zloss_terms: List[torch.Tensor] = []
    owner: Dict[int, str] = {}   # id(tensor) -> expert name, for expert-balanced grouping

    def _tag(ename, lst, t):
        owner[id(t)] = ename
        lst.append(t)

    for ename, ex in model.experts.items():
        if active_set is not None and ename not in active_set:
            continue
        for t in ex.types:
            o = outputs[t.name]
            mask = batch[t.mask_key]
            _tag(ename, pres_terms, F.binary_cross_entropy_with_logits(
                o["presence"], mask.to(o["presence"].dtype)))
            if z_weight > 0.0:
                # Presence is a binary head: logsumexp of the equivalent 2-class logits [z, 0] is
                # softplus(z). Squaring + meaning penalizes over-large presence logits.
                _tag(ename, zloss_terms, F.softplus(o["presence"]).pow(2).mean())
            if t.cat_cols:
                tgt_idx = batch[t.idx_key]
                for c in range(len(t.cat_cols)):
                    logits = o["cat"][c]
                    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         tgt_idx[..., c].reshape(-1), reduction="none")
                    ce = ce.reshape(tgt_idx.shape[:-1])
                    _tag(ename, cat_terms, _masked_mean(ce, mask))
                    if z_weight > 0.0:
                        lse = torch.logsumexp(logits, dim=-1)        # [B, slots] log-partition
                        _tag(ename, zloss_terms, _masked_mean(lse.pow(2), mask))
            if "num_logits" in o:
                head: E.RangeBinHeads = ex.heads[t.name]
                # Per-field CE (bins twohot/hard, or per-digit one-hot for a digit column), optionally
                # log-bins-normalized so a wide column doesn't dominate — all centralized in the head.
                field_ce = head.numeric_field_ce(o["num_logits"], batch[t.num_key],
                                                 num_targets=num_targets, num_loss_norm=num_loss_norm)
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
    # z-loss (0.0 when off — zloss_terms is empty, _reduce returns a zero scalar). Always keyed in the
    # returned dict so the train-window accumulator can wire it consistently.
    loss_zloss = z_weight * _reduce(zloss_terms)
    total = loss_cat + loss_num + loss_pres + loss_zloss
    return {"loss": total, "loss_categorical": loss_cat, "loss_numeric": loss_num,
            "loss_presence": loss_pres, "loss_zloss": loss_zloss}


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
        # Per-expert stamp (slice + tokenizer signature + build kwargs): the compose/warm-start contract.
        # Weights live in the full state_dict under `experts.<name>.*` — addressable per expert by prefix.
        "experts": m.expert_stamps(),
        "tokenizer_version": tokens.TOKENIZER_VERSION,
        "tokenizer_signature": tokens.tokenizer_signature(),
        "catalog_signatures": tokens.CATALOG_SIGNATURES,
    }
    if extra:
        meta.update(extra)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# Experts that no longer exist in the roster — an OLD checkpoint naming one is rejected with a CLEAR error
# rather than silently remapped. The single "creatures" expert was split into three parameter-disjoint
# experts (owner ruling, 2026-07-18); it floored at dist ~0.29, so there is nothing worth preserving.
_LEGACY_EXPERTS: Dict[str, str] = {
    "creatures": "creature-stats/creature-powers/creature-intents",
}


def _meta_expert_names(meta: dict) -> Set[str]:
    """Every expert name a checkpoint meta references — across the per-expert stamps, the slice layout, and
    the config's slice_widths — so a legacy name is caught wherever it was stamped."""
    names: Set[str] = set((meta.get("experts") or {}).keys())
    names |= {s.get("name") for s in (meta.get("slice_layout") or []) if isinstance(s, dict)}
    names |= set(((meta.get("config") or {}).get("slice_widths") or {}).keys())
    return {n for n in names if n}


def _reject_legacy_experts(meta: dict, path: str) -> None:
    """Raise a clear error if ``meta`` names an expert that was removed from the roster (e.g. the split of
    'creatures'). Do NOT silently map it — the old expert's weights are not compatible with the new layout
    and (for creatures) were not worth keeping."""
    legacy = _LEGACY_EXPERTS.keys() & _meta_expert_names(meta)
    if legacy:
        detail = "; ".join(f"'{n}' (now {_LEGACY_EXPERTS[n]})" for n in sorted(legacy))
        raise ValueError(
            f"Checkpoint {path} has legacy expert {detail}; the roster changed and there is no compatible "
            f"mapping — retrain against the current expert roster.")


def load_checkpoint(path: str, device="cpu") -> Tuple[FactoredWorldModelAE, dict]:
    """Load ``(model, meta)``, rejecting a stale tokenizer/catalog signature, a non-factored checkpoint, a
    legacy (removed) expert name, or a slice-layout mismatch (the layout is the predictor's addressing
    contract)."""
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
    _reject_legacy_experts(meta, path)
    # Backward compat: a checkpoint trained before the QK-norm fix has NO qk_norm key in its config.
    # Default it to False so the stock trunks are rebuilt and the old state_dict loads byte-identically.
    config = dict(meta["config"])
    config.setdefault("qk_norm", False)
    # Numeric head layout / decode mode (roadmap M3.5): a checkpoint predating the cross-expert numeric fix
    # has neither key. Default to the historical behavior — flat "bins" head, "argmax" eval decode — so the
    # old state_dict loads byte-identically and its reported metrics reproduce exactly.
    config.setdefault("num_head", "bins")
    config.setdefault("num_decode", "argmax")
    m = FactoredWorldModelAE(**config)
    got_layout = m.slice_layout_list()
    if meta.get("slice_layout") != got_layout:
        raise ValueError(
            f"Checkpoint {path} slice layout {meta.get('slice_layout')} does not match the current "
            f"model layout {got_layout}; the predictor addresses slices by these offsets.")
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    return m, meta


def read_meta(path: str) -> dict:
    """Load a factored checkpoint's ``.meta.json`` (validating arch + tokenizer signature) WITHOUT
    building the model — used by compose/warm-start to inspect per-expert stamps cheaply."""
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    want = tokens.tokenizer_signature()
    if meta.get("tokenizer_signature") != want:
        raise ValueError(f"Checkpoint {path} tokenizer signature {meta.get('tokenizer_signature')!r} "
                         f"!= current {want!r}; retrain.")
    if meta.get("arch") != "factored":
        raise ValueError(f"Checkpoint {path} arch={meta.get('arch')!r} is not 'factored'.")
    _reject_legacy_experts(meta, path)
    return meta


def init_expert_from(m: FactoredWorldModelAE, name: str, src_path: str, device="cpu") -> Dict[str, Any]:
    """Warm-start expert ``name`` in ``m`` from the matching slice of the full factored checkpoint at
    ``src_path`` (``--init-expert-from name=ckpt`` — seed a solo run from the stopped joint run's partial
    progress). Validates the source is factored, tokenizer-compatible, and that the source expert's slice
    width + build config match ``m``'s (so its weights actually fit). Returns the source expert stamp."""
    meta = read_meta(src_path)
    src_stamp = (meta.get("experts") or {}).get(name)
    my_stamp = m.expert_stamp(name)
    if src_stamp is not None:
        if src_stamp.get("width") != my_stamp["width"]:
            raise ValueError(f"--init-expert-from {name}: source slice width {src_stamp.get('width')} "
                             f"!= target {my_stamp['width']}.")
        # A qk_norm difference is TOLERATED here: it only changes the attention trunk, which the graceful
        # loader skips (non-trunk weights still warm-start across the fix). num_decode is a pure EVAL knob
        # (no weights), so it is tolerated too. num_head DOES change the numeric-head Linear shape, so it is
        # checked separately below (missing key == the historical flat "bins" head). Every OTHER config key
        # must match — those govern shapes the non-trunk copy depends on.
        _skip = ("qk_norm", "num_decode", "num_head")
        src_cfg = {k: v for k, v in (src_stamp.get("config") or {}).items() if k not in _skip}
        my_cfg = {k: v for k, v in my_stamp["config"].items() if k not in _skip}
        if src_cfg != my_cfg:
            raise ValueError(f"--init-expert-from {name}: source expert config {src_stamp.get('config')} "
                             f"!= target {my_stamp['config']} (architecture differs).")
        src_nh = (src_stamp.get("config") or {}).get("num_head", "bins")
        my_nh = my_stamp["config"].get("num_head", "bins")
        if src_nh != my_nh:
            raise ValueError(f"--init-expert-from {name}: source numeric head layout {src_nh!r} != target "
                             f"{my_nh!r} (the num_head Linear shape differs — cannot warm-start).")
    src_state = torch.load(src_path, map_location=device)
    m.load_expert_from_state_dict(name, src_state)
    return src_stamp or my_stamp
