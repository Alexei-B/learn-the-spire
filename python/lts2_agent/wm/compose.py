"""Compose a full factored checkpoint from per-expert sources (roadmap M3.5, sequential per-expert
training). The owner trains each expert to its exactness bar in its own solo run, keeps the good one, and
at the end **composes** the kept experts into one standard :class:`~lts2_agent.wm.model_factored.
FactoredWorldModelAE` checkpoint — the artifact the M4 predictor consumes.

Each ``name=ckpt`` pulls expert ``name``'s weights out of that (full factored) checkpoint's slice; a
``--base`` fills every unassigned learned expert. The result is a normal factored checkpoint
(``save_checkpoint`` format, with per-expert provenance in the meta), so :mod:`lts2_agent.eval_encdec`
loads it transparently.

Validation is strict: every source must be factored + tokenizer-compatible, all sources must agree on the
shared global config (``d_model``/heads/layers/pool/…), and each pulled expert's slice width + build
config must match the composite it is assembled into. A slice can only land in a checkpoint it fits.

Run::

    python -m lts2_agent.wm.compose --out checkpoints/composite.pt \\
        --base checkpoints/wm_t3_v3.pt.best \\
        --experts relics=checkpoints/relic_solo_d.pt.best cards=checkpoints/cards_solo.pt.best
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import torch

from . import model_factored as MF
from .experts import EXPERT_ORDER

# Learned experts (scalars is the parameter-free tier-1 codec — nothing to compose).
LEARNED = [n for n in EXPERT_ORDER if n != "scalars"]
# Config keys that must be IDENTICAL across every source (they apply model-wide, not per expert).
_GLOBAL_KEYS = ["d_model", "n_heads", "enc_layers", "dec_layers", "pool_layers", "pool_latents",
                "n_mem", "cat_dim", "simnorm_group"]


def _parse_assignments(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--experts entry {it!r} must be name=checkpoint")
        name, path = it.split("=", 1)
        if name not in LEARNED:
            raise SystemExit(f"--experts: unknown expert {name!r} (choose from {LEARNED})")
        out[name] = path
    return out


def compose(out_path: str, assignments: Dict[str, str], base: str = "",
            device: str = "cpu") -> Dict[str, Any]:
    """Assemble ``out_path`` from the per-expert ``assignments`` (+ ``base`` for the rest). Returns the
    provenance map {expert: source_path}."""
    sources: Dict[str, str] = dict(assignments)
    if base:
        for name in LEARNED:
            sources.setdefault(name, base)
    missing = [n for n in LEARNED if n not in sources]
    if missing:
        raise SystemExit(f"compose: no source for expert(s) {missing}; assign them via --experts or "
                         f"provide --base.")

    # Read every unique source's meta (validates factored + tokenizer signature) and cache it.
    metas: Dict[str, dict] = {p: MF.read_meta(p) for p in set(sources.values())}

    # All sources must agree on the shared global config.
    ref_path = sources[LEARNED[0]]
    ref_cfg = metas[ref_path]["config"]
    for name, path in sources.items():
        cfg = metas[path]["config"]
        diffs = {k: (ref_cfg.get(k), cfg.get(k)) for k in _GLOBAL_KEYS if ref_cfg.get(k) != cfg.get(k)}
        if diffs:
            raise SystemExit(f"compose: source for {name!r} ({path}) disagrees on global config {diffs} "
                             f"vs reference {ref_path}; cannot assemble one coherent model.")

    # Composite config: shared globals, then per-expert slice width / override from the source each
    # expert is pulled from.
    comp_cfg: Dict[str, Any] = {k: ref_cfg[k] for k in _GLOBAL_KEYS}
    slice_widths: Dict[str, int] = {}
    overrides: Dict[str, Dict[str, Any]] = {}
    for name in LEARNED:
        cfg = metas[sources[name]]["config"]
        slice_widths[name] = cfg["slice_widths"][name]
        ov = (cfg.get("expert_overrides") or {}).get(name)
        if ov:
            overrides[name] = ov
    comp_cfg["slice_widths"] = slice_widths
    comp_cfg["expert_overrides"] = overrides

    model = MF.FactoredWorldModelAE(**comp_cfg).to(device)

    # Copy each expert's weights from its source, validating slice width + build config match.
    for name in LEARNED:
        path = sources[name]
        stamp = (metas[path].get("experts") or {}).get(name)
        my = model.expert_stamp(name)
        if stamp is not None:
            if stamp.get("width") != my["width"]:
                raise SystemExit(f"compose: expert {name!r} slice width {stamp.get('width')} from "
                                 f"{path} != composite {my['width']}.")
            if stamp.get("config") != my["config"]:
                raise SystemExit(f"compose: expert {name!r} config {stamp.get('config')} from {path} "
                                 f"!= composite {my['config']}.")
        src_state = torch.load(path, map_location=device)
        model.load_expert_from_state_dict(name, src_state)

    provenance = {n: sources[n] for n in LEARNED}
    steps = {n: int(metas[sources[n]].get("step", 0)) for n in LEARNED}
    MF.save_checkpoint(out_path, model, step=max(steps.values(), default=0),
                       extra={"composed_from": provenance, "composed_steps": steps})
    return provenance


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose a full factored checkpoint from per-expert sources")
    ap.add_argument("--out", required=True, help="output composite checkpoint path")
    ap.add_argument("--experts", nargs="*", default=[],
                    help="expert=checkpoint assignments, e.g. relics=ckptC.pt.best cards=ckptA.pt.best")
    ap.add_argument("--base", default="",
                    help="full factored checkpoint sourcing every expert not named in --experts")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    assignments = _parse_assignments(args.experts)
    prov = compose(args.out, assignments, base=args.base, device=args.device)
    print(f"[compose] wrote {args.out}")
    for name in LEARNED:
        print(f"[compose]   {name:10s} <- {prov[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
