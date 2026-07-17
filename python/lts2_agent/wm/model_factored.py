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
                 relic_head: str = "set", expert_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 static_tables: Optional[Dict[str, np.ndarray]] = None):
        super().__init__()
        widths = dict(DEFAULT_SLICE_WIDTHS)
        if slice_widths:
            widths.update(slice_widths)
        self.slice_widths = widths
        self.relic_head = relic_head
        # Per-expert construction-kwarg overrides (e.g. a deeper relic decoder for the M3.5 bake-off) —
        # stamped in cfg so save/load/compose reconstruct the exact per-expert architecture.
        self.expert_overrides = {k: dict(v) for k, v in (expert_overrides or {}).items()}
        self.cfg = dict(d_model=d_model, n_heads=n_heads, enc_layers=enc_layers, dec_layers=dec_layers,
                        pool_layers=pool_layers, pool_latents=pool_latents, n_mem=n_mem, cat_dim=cat_dim,
                        simnorm_group=simnorm_group, slice_widths=widths, relic_head=relic_head,
                        expert_overrides=self.expert_overrides)
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
        ov = self.expert_overrides

        def _kw(name: str, base: Dict[str, Any]) -> Dict[str, Any]:
            return dict(base, **ov.get(name, {}))

        self.experts = nn.ModuleDict({
            "creatures": E.SetExpert("creatures", ["creature", "power", "intent"],
                                     widths["creatures"], **_kw("creatures", deep)),
            "cards": E.SetExpert("cards", ["card"], widths["cards"], **_kw("cards", deep)),
            "relics": E.RelicExpert("relics", widths["relics"], relic_head=relic_head,
                                    **_kw("relics", shallow)),
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
        module was built with (relic_head / any override). The compose/warm-start contract validates
        against this, so a slice can only be assembled into a checkpoint whose layout it fits."""
        a, b = self.slice_layout[name]
        cfg: Dict[str, Any] = {"d_model": self.cfg["d_model"], "n_heads": self.cfg["n_heads"],
                               "cat_dim": self.cfg["cat_dim"], "simnorm_group": self.cfg["simnorm_group"],
                               "pool_layers": self.cfg["pool_layers"],
                               "pool_latents": self.cfg["pool_latents"], "n_mem": self.cfg["n_mem"]}
        cfg.update(self.expert_overrides.get(name, {}))
        if name == "relics":
            cfg["relic_head"] = self.relic_head
        return {"name": name, "slice": [a, b], "width": b - a,
                "tokenizer_signature": tokens.tokenizer_signature(), "config": cfg,
                "param_count": param_count(self.experts[name]) if name != "scalars" else 0}

    def expert_stamps(self) -> Dict[str, Dict[str, Any]]:
        return {n: self.expert_stamp(n) for n in E.EXPERT_ORDER}

    def load_expert_from_state_dict(self, name: str, src_state: Dict[str, torch.Tensor]) -> None:
        """Copy one expert's weights out of a full factored ``state_dict`` (keys ``experts.<name>.*``)
        into this model — the warm-start / compose primitive. Shapes must match (raises otherwise)."""
        if name == "scalars":
            return   # parameter-free deterministic codec; nothing to copy
        prefix = f"experts.{name}."
        own = self.experts[name].state_dict()
        sub = {k[len(prefix):]: v for k, v in src_state.items() if k.startswith(prefix)}
        missing = set(own) - set(sub)
        if missing:
            raise ValueError(f"source checkpoint is missing expert {name!r} params: "
                             f"{sorted(missing)[:4]}{'...' if len(missing) > 4 else ''}")
        self.experts[name].load_state_dict({k: sub[k] for k in own})


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
                   relic_pos_weight: float = 5.0,
                   active: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """The three reconstruction losses (categorical / numeric / presence) + their sum. Numerics are
    range-bin cross-entropy (per field, over present slots); the scalar expert contributes nothing
    (exact by construction, no parameters).

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
    owner: Dict[int, str] = {}   # id(tensor) -> expert name, for expert-balanced grouping

    def _tag(ename, lst, t):
        owner[id(t)] = ename
        lst.append(t)

    for ename, ex in model.experts.items():
        if active_set is not None and ename not in active_set:
            continue
        if ename == "relics":
            o = outputs["relic"]
            t = S.TYPE_BY_NAME["relic"]
            m = batch[t.mask_key]
            if "set_logits" in o:
                logits = o["set_logits"]
                idx = batch[t.idx_key][..., 0]
                tgt = torch.zeros_like(logits)
                b_sel, s_sel = torch.where(m)
                tgt[b_sel, idx[b_sel, s_sel].clamp(0, logits.shape[-1] - 1)] = 1.0
                # Rare-positive multi-label (about 5 of 298 ids present): unweighted BCE collapses
                # toward predict-nothing (measured: relic F1 stuck ~0.58 while every other expert
                # learned). pos_weight rebalances the positive-class gradient DIRECTION, which per-
                # parameter Adam scale-invariance cannot recover on its own.
                pw = torch.full((), float(relic_pos_weight), device=logits.device)
                _tag(ename, cat_terms,
                     F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw))
                if "count_logits" in o:
                    true_k = m.sum(dim=1).long().clamp(0, o["count_logits"].shape[-1] - 1)
                    _tag(ename, cat_terms, F.cross_entropy(o["count_logits"], true_k))
            else:
                # Slots relic head: per-slot categorical CE (over present slots) + presence BCE, the
                # generic per-type loss (the bake-off's monolith-style variant; decode dedups).
                _tag(ename, pres_terms, F.binary_cross_entropy_with_logits(
                    o["presence"], m.to(o["presence"].dtype)))
                tgt_idx = batch[t.idx_key]
                for c in range(len(t.cat_cols)):
                    logits = o["cat"][c]
                    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         tgt_idx[..., c].reshape(-1), reduction="none")
                    _tag(ename, cat_terms, _masked_mean(ce.reshape(tgt_idx.shape[:-1]), m))
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
        if src_stamp.get("config") != my_stamp["config"]:
            raise ValueError(f"--init-expert-from {name}: source expert config {src_stamp.get('config')} "
                             f"!= target {my_stamp['config']} (architecture differs).")
    src_state = torch.load(src_path, map_location=device)
    m.load_expert_from_state_dict(name, src_state)
    return src_stamp or my_stamp
