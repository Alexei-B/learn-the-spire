"""Overfit-one-batch diagnostic for the factored experts (roadmap M3.5 solo-dynamics gate).

Trains ONE expert alone on a SINGLE fixed batch and reports the step at which that expert's
``expert_dist`` drops below a threshold (default 0.01). An expert that CANNOT drive a fixed batch to
near-zero reconstruction distance has a WIRING bug (a head that can't reach its target, a target/decode
mismatch, a masking error) — memorizing one batch is the easiest possible task, so this is the primary
correctness gate for the per-expert solo path. It also compares the two numeric decodes (argmax vs
expectation) so the two-hot decode choice is measured, not assumed.

Run (real cache, the GPU gate)::

    python -m lts2_agent.wm.overfit --expert orbs
    python -m lts2_agent.wm.overfit --expert all --batch 512 --steps 800

``--synthetic`` builds a random in-range batch with no corpus/GPU (the CLI smoke path)."""

from __future__ import annotations

import argparse
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .. import tokens
from . import data as D
from . import experts as E
from . import model as M
from . import model_factored as MF
from . import report
from . import spec as S
from .experts import EXPERT_ORDER, EXPERT_TYPES

# Experts that overfit is meaningful for (scalars is parameter-free — exact by construction, nothing to
# train, so it is reported n/a rather than run).
OVERFIT_EXPERTS = [n for n in EXPERT_ORDER if n != "scalars"]


# ==================================================================================================
# Synthetic batch (no corpus / no GPU) — random IN-RANGE integers stored exactly as the tokenizer/head
# expect (raw cols raw, everything else symlog), so a healthy head can reconstruct them to dist 0.
# ==================================================================================================

def _rand_num(tname: str, cols: List[str], shape: Tuple[int, ...], rng) -> np.ndarray:
    raw = E.RAW_NUM_COLS.get(tname, set())
    out = np.zeros(tuple(shape) + (len(cols),), np.float32)
    for j, col in enumerate(cols):
        r = S.NUMERIC_RANGES.get(tname, {}).get(col)
        lo, hi = (r.lo, r.hi) if r is not None else (0, 1)
        v = rng.integers(lo, hi + 1, size=shape).astype(np.float64)
        out[..., j] = v if col in raw else np.sign(v) * np.log1p(np.abs(v))
    return out


def synthetic_batch(B: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """A random but self-consistent stacked batch (all ``M.BATCH_KEYS``) for the CLI smoke path."""
    rng = np.random.default_rng(seed)
    stacked: Dict[str, np.ndarray] = {}
    g = S.TYPE_BY_NAME["global"]
    gi = np.zeros((B, 1, len(g.cat_cols)), np.int64)
    for c, (_, v) in enumerate(g.cat_cols):
        gi[:, 0, c] = rng.integers(0, v, B)
    stacked["global_idx"] = gi
    stacked["global_num"] = _rand_num("global", list(tokens.GLOBAL_NUM), (B, 1), rng)
    pv = np.zeros((B, 1, 4), np.float32)
    pv[:, 0, 0] = rng.integers(0, 2, B)
    pv[:, 0, 1] = np.sign(rng.integers(0, 11, B)) * np.log1p(rng.integers(0, 11, B))
    pv[:, 0, 2] = np.log1p(rng.integers(0, 101, B))
    pv[:, 0, 3] = rng.integers(0, 2, B)
    stacked["pending"] = pv
    for t in S.TYPES:
        if not t.mask_key:
            continue
        ms = t.max_slots
        mask = np.zeros((B, ms), bool)
        for b in range(B):
            mask[b, : int(rng.integers(1, ms + 1))] = True
        stacked[t.mask_key] = mask
        if t.idx_key:
            idx = np.zeros((B, ms, len(t.cat_cols)), np.int64)
            for c, (_, v) in enumerate(t.cat_cols):
                idx[:, :, c] = rng.integers(0, v, (B, ms))
            stacked[t.idx_key] = idx
        if t.num_key:
            stacked[t.num_key] = _rand_num(t.name, list(E._num_cols(t)), (B, ms), rng)
        if t.has_kw:
            stacked["card_kw"] = rng.integers(0, 2, (B, ms, tokens.KW_BUCKETS)).astype(np.float32)
    return stacked


# ==================================================================================================
# Batch loading + metrics.
# ==================================================================================================

def load_cache_batch(cache_dir: str, split: str, batch_size: int, experts: List[str],
                     frac_present: float, seed: int) -> Dict[str, np.ndarray]:
    """One present-heavy fixed batch from the cache (via the focus-present sampler), so the id/numeric
    heads are actually exercised (an all-empty sparse-expert batch overfits trivially without testing
    them)."""
    rng = random.Random(seed)
    gen = D.focus_present_batches_cpu(cache_dir, split, batch_size, rng, experts, frac_present)
    stacked, _acts = next(gen)
    return stacked


@torch.no_grad()
def _expert_dist(model: MF.FactoredWorldModelAE, batch: Dict[str, torch.Tensor], expert: str) -> float:
    model.eval()
    _z, out = model(batch, active_experts=[expert])
    pairs = report.report_pairs_experts_only(batch, out, [expert])
    val = report.aggregate(pairs)[f"expert_dist::{expert}"]
    model.train()
    return float(val)


@torch.no_grad()
def numeric_decode_compare(model: MF.FactoredWorldModelAE, batch: Dict[str, torch.Tensor],
                           expert: str) -> Dict[str, Tuple[float, float]]:
    """Per-numeric-type exact-bin match fraction over present slots under the two decodes
    ``(argmax, expectation)`` — the two-hot decode-choice measurement."""
    model.eval()
    _z, out = model(batch, active_experts=[expert])
    res: Dict[str, Tuple[float, float]] = {}
    for tn in EXPERT_TYPES[expert]:
        o = out.get(tn)
        if not o or "num_bin_logits" not in o:
            continue
        t = S.TYPE_BY_NAME[tn]
        head = model.experts[expert].heads[tn]
        tgt = head.bin_targets(batch[t.num_key])                        # [B, slots, W] true bins
        amax = torch.stack([lg.argmax(-1) for lg in o["num_bin_logits"]], dim=-1)
        ebins = []
        for lg in o["num_bin_logits"]:
            nb = lg.shape[-1]
            centers = torch.arange(nb, device=lg.device, dtype=torch.float32)
            ebins.append((lg.float().softmax(-1) * centers).sum(-1).round().clamp(0, nb - 1).long())
        eb = torch.stack(ebins, dim=-1)
        m = batch[t.mask_key].bool().unsqueeze(-1).expand_as(tgt)
        denom = m.sum().clamp_min(1)
        res[tn] = (float(((amax == tgt) & m).sum() / denom),
                   float(((eb == tgt) & m).sum() / denom))
    model.train()
    return res


def overfit_batch(model: MF.FactoredWorldModelAE, batch: Dict[str, torch.Tensor], expert: str,
                  steps: int = 800, lr: float = 2e-3, thresh: float = 0.01,
                  num_targets: str = "twohot", report_every: int = 25, verbose: bool = True
                  ) -> Dict[str, Any]:
    """Train ONLY ``expert`` on the fixed ``batch``; return ``{hit_step, final_dist, history}``. Stops
    at the first step whose ``expert_dist < thresh`` (``hit_step``), else runs the full ``steps``."""
    active = [expert]
    params: List[torch.nn.Parameter] = []
    for name, ex in model.experts.items():
        on = name == expert
        for p in ex.parameters():
            p.requires_grad_(on)
        if on:
            params += list(ex.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    hist: List[Tuple[int, float, float]] = []
    hit: Optional[int] = None
    for step in range(1, steps + 1):
        model.train()
        _z, out = model(batch, active_experts=active)
        losses = MF.compute_losses(batch, out, model, active=active, num_targets=num_targets)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % report_every == 0 or step == steps:
            dist = _expert_dist(model, batch, expert)
            hist.append((step, dist, float(losses["loss"])))
            if verbose:
                print(f"    [{expert:9s} step {step:4d}] loss={float(losses['loss']):.4f} "
                      f"expert_dist={dist:.4f}", flush=True)
            if dist < thresh:
                hit = step
                break
    return {"expert": expert, "hit_step": hit, "final_dist": hist[-1][1] if hist else float("nan"),
            "history": hist}


# ==================================================================================================
# CLI.
# ==================================================================================================

def _run_one(expert: str, args, device) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    model = MF.FactoredWorldModelAE(d_model=args.d_model).to(device)
    if args.synthetic:
        stacked = synthetic_batch(args.batch, seed=args.seed)
    else:
        stacked = load_cache_batch(args.cache, args.split, args.batch, [expert],
                                   args.focus, args.seed)
    nat = float(D.expert_present_mask(stacked, [expert]).mean())
    batch = M.to_tensors(stacked, device)
    print(f"[overfit] expert={expert} batch={args.batch} present-fraction={nat:.3f} "
          f"num_targets={args.num_targets}", flush=True)
    res = overfit_batch(model, batch, expert, steps=args.steps, lr=args.lr, thresh=args.thresh,
                        num_targets=args.num_targets, report_every=args.report_every,
                        verbose=not args.quiet)
    res["present_fraction"] = nat
    res["decode"] = numeric_decode_compare(model, batch, expert)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Overfit-one-batch wiring gate for the factored experts.")
    ap.add_argument("--expert", default="all",
                    help="expert to overfit, or 'all' (every learned expert). "
                         f"Choices: all, {', '.join(OVERFIT_EXPERTS)}")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--thresh", type=float, default=0.01, help="expert_dist target (steps-to-hit)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--num-targets", default="twohot", choices=["twohot", "hard"])
    ap.add_argument("--focus", type=float, default=0.9,
                    help="present-state fraction of the fixed batch (cache path only)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--cache", default="data/corpus_tok_v3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--synthetic", action="store_true", help="random in-range batch (no corpus/GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=25)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[overfit] CUDA unavailable; using CPU.", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)

    which = OVERFIT_EXPERTS if args.expert == "all" else [args.expert]
    for e in which:
        if e not in OVERFIT_EXPERTS:
            raise SystemExit(f"--expert {e!r} invalid; choose 'all' or one of {OVERFIT_EXPERTS}")

    results = [_run_one(e, args, device) for e in which]

    print("\n" + "=" * 74, flush=True)
    print(f"OVERFIT-ONE-BATCH GATE   batch={args.batch} steps<={args.steps} thresh={args.thresh} "
          f"num_targets={args.num_targets}", flush=True)
    print("=" * 74, flush=True)
    print(f"  {'expert':10s} {'present':>8s} {'steps->thr':>11s} {'final_dist':>11s}  "
          f"{'numeric decode (argmax/expect exact)':s}", flush=True)
    for r in results:
        hit = str(r["hit_step"]) if r["hit_step"] is not None else "FAIL"
        dec = " ".join(f"{tn}={a:.3f}/{e:.3f}" for tn, (a, e) in r["decode"].items()) or "-"
        print(f"  {r['expert']:10s} {r['present_fraction']:8.3f} {hit:>11s} "
              f"{r['final_dist']:11.4f}  {dec}", flush=True)
    print("  scalars    (parameter-free — exact by construction, not trained)", flush=True)
    print("=" * 74, flush=True)
    failed = [r["expert"] for r in results if r["hit_step"] is None]
    if failed:
        print(f"[overfit] WIRING SUSPECT — did not reach {args.thresh} on a fixed batch: {failed}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
