"""GPU trainer for the world-model encoder/decoder (roadmap 3.1, design §4.2-4.3).

Streams the transition corpus (both ``state`` and ``nextState`` of every record ~= 2M states, **train
split only**, through a shuffle buffer), autoencodes each state (encoder -> SimNorm ``z`` -> symbolic
decoder), and optimizes the three reconstruction losses. Per-field reconstruction accuracy streams to the
dashboard DURING training (``kind="wm-encdec"``) so the corpus-sufficiency curves are watchable live — the
product-owner requirement.

Run::

    python -m lts2_agent.train_encdec --steps 50000 --batch 384 --val-every 500 \
        --ckpt checkpoints/wm_encdec.pt --run-label wm-encdec

Metrics
-------
Per train step-window (phase="train"): ``train.loss``, ``train.loss_categorical``,
``train.loss_numeric``, ``train.loss_presence``, ``train.lr``, ``train.states_per_s``.
Per val pass (phase="eval"): the per-field report card (``eval.card_id_top1``, ``eval.card_zone_acc``,
``eval.power_id_top1``, ``eval.power_amount_mae``, ``eval.creature_hp_mae``, ``eval.creature_block_mae``,
``eval.intent_damage_mae``, ``eval.energy_acc``, ``eval.relic_set_f1``, ``eval.potion_set_f1``,
``eval.hand_size_acc``, ``eval.pile_size_acc``, ``eval.pending_choice_acc``, ``eval.exact_state_rate``),
each emitted a second time tagged ``{"act": ...}`` for the dashboard's group-by. MAEs are RAW game units.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import tokens
from .metrics import MetricsWriter
from .wm import cache as C
from .wm import data as D
from .wm import model as M
from .wm import model_factored as MF
from .wm import report
from .wm import spec as S
from .wm import synth as SY
from .wm.experts import EXPERT_ORDER


def _parse_data(spec: str) -> tuple:
    """Parse the --data spec into ``(mode, frac_synth)``: 'real'->('real',0.0), 'synth'->('synth',1.0),
    'mixed:R'->('mixed',R) with 0<R<1."""
    spec = (spec or "real").strip().lower()
    if spec == "real":
        return "real", 0.0
    if spec == "synth":
        return "synth", 1.0
    if spec.startswith("mixed:"):
        try:
            r = float(spec.split(":", 1)[1])
        except ValueError:
            raise SystemExit(f"--data {spec!r}: mixed ratio must be a float (e.g. mixed:0.5)")
        if not 0.0 < r < 1.0:
            raise SystemExit(f"--data {spec!r}: mixed ratio must be in (0,1)")
        return "mixed", r
    raise SystemExit(f"--data {spec!r}: expected 'real', 'synth', or 'mixed:R'")


def _lr_lambda(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))  # cosine decay to 0
    return fn


@torch.no_grad()
def run_val(model, sample_stacked, sample_acts, batch_size, device, loss_fn, experts=False,
            active=None, trained_only=False):
    """Full pass over the fixed val sample -> (overall metrics, by-act metrics, mean losses).

    ``trained_only`` (per-expert solo runs): forward + loss + report are restricted to the ``active``
    experts, so the val pass decodes ONLY that expert's token types (focused metrics, not the full card)."""
    model.eval()
    accum: Dict[str, Any] = {}
    loss_sums = {"loss": 0.0, "loss_categorical": 0.0, "loss_numeric": 0.0, "loss_presence": 0.0}
    nb = 0
    fwd_active = active if trained_only else None
    for batch, acts in D.iter_fixed_batches(sample_stacked, sample_acts, batch_size, device):
        z, out = model(batch, active_experts=fwd_active) if experts else model(batch)
        losses = loss_fn(batch, out, active=fwd_active) if experts else loss_fn(batch, out)
        for k in loss_sums:
            loss_sums[k] += float(losses[k])
        nb += 1
        if trained_only:
            pairs = report.report_pairs_experts_only(batch, out, active)
        else:
            pairs = report.report_pairs(batch, out, experts=experts)
        report.merge_pairs(accum, pairs, acts)
    overall, by_act = report.finalize(accum)
    mean_losses = {k: v / max(1, nb) for k, v in loss_sums.items()}
    model.train()
    return overall, by_act, mean_losses


def _card_index_counts(args, n_states: int) -> np.ndarray:
    """Occurrence counts of the card-identity index (card categorical column 0) over PRESENT card slots,
    scanned from the train split until ``n_states`` states are seen. Uses the pre-tokenized cache when
    present (fast), else featurizes the corpus stream (skipping the same failures the trainer does)."""
    n_cards = S.TYPE_BY_NAME["card"].cat_cols[0][1]
    counts = np.zeros(n_cards, dtype=np.int64)
    seen = 0
    cache_dir = args.cache or None
    manifest = C.resolve_manifest(cache_dir)
    if manifest is not None:
        for path in C.shard_files(cache_dir, "train"):
            stacked, acts = C.load_shard(path)
            idx = stacked["card_idx"][..., 0]                       # [n, slots]
            mask = stacked["card_mask"].astype(bool)                # [n, slots]
            counts += np.bincount(idx[mask].astype(np.int64), minlength=n_cards)[:n_cards]
            seen += len(acts)
            if seen >= n_states:
                break
    else:
        for state, _act in D.iter_states(args.corpus, "train"):
            f = D._featurize_safe(state)
            if f is None:
                continue
            idx = f["card_idx"][:, 0]
            mask = f["card_mask"].astype(bool)
            counts += np.bincount(idx[mask].astype(np.int64), minlength=n_cards)[:n_cards]
            seen += 1
            if seen >= n_states:
                break
    print(f"[train_encdec] card-CE: scanned {seen:,} states, {int(counts.sum()):,} card slots, "
          f"{int((counts > 0).sum())}/{n_cards} card ids seen", flush=True)
    return counts


def _card_ce_weights(args, device) -> torch.Tensor:
    """The class-balanced card-CE weight vector (1/sqrt(freq), freq-weighted-mean-normalized to 1),
    cached to disk keyed by the tokenizer signature + scan size so restarts are cheap."""
    sig = tokens.tokenizer_signature()
    n = args.card_ce_states
    key = hashlib.sha1(f"{sig}|N={n}".encode()).hexdigest()[:16]
    cache_dir = args.card_ce_cache or (os.path.dirname(args.ckpt) or ".")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"card_ce_w_{key}.npy")
    if os.path.exists(path):
        w = np.load(path)
        print(f"[train_encdec] card-CE: loaded cached weights {path} (shape {w.shape})", flush=True)
    else:
        counts = _card_index_counts(args, n)
        w = M.card_ce_weights_from_counts(counts)
        np.save(path, w)
        print(f"[train_encdec] card-CE: computed + cached weights -> {path} "
              f"(min {w.min():.3f}, max {w.max():.3f})", flush=True)
    return torch.tensor(w, device=device, dtype=torch.float32)


def _slice_width_overrides(args):
    """Parse --slice-width NAME=W entries into the constructor's slice_widths override dict."""
    if not getattr(args, "slice_width", None):
        return None
    out = dict(MF.DEFAULT_SLICE_WIDTHS)
    for item in args.slice_width:
        name, w = item.split("=")
        out[name.strip()] = int(w)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="World-model encoder/decoder trainer (roadmap 3.1)")
    ap.add_argument("--corpus", default="data/corpus", help="corpus root (train split streamed)")
    ap.add_argument("--cache", default="",
                    help="OPT-IN pre-tokenized cache dir (default empty = none). Synthetic-first doctrine: "
                         "training is synthetic and real-val is tokenized on the fly (2k states = seconds), "
                         "so no full corpus cache is built or required. Pass a dir to opt into cache-backed "
                         "real/mixed training; used only when it exists and its signature matches.")
    ap.add_argument("--data", default=None,
                    help="factored SOLO runs only (--train-experts set): training data source. Default is "
                         "'synth' for a factored SOLO run (synthetic-space training is the doctrine — real "
                         "corpus data is EVAL-ONLY): mechanically-generated uniform configurations in "
                         "tokenizer-array space (no cache/corpus needed — kills the rare-tail coverage "
                         "floors; roadmap M3.5). 'real' opts back into corpus-cache training (deployment "
                         "distribution); 'mixed:R' = a fraction R synthetic per batch, 1-R real (e.g. "
                         "'mixed:0.5'). A joint run always uses real. Val ALWAYS runs the real fixed val "
                         "(deployment yardstick, tokenized on the fly — no full cache required) AND a "
                         "seeded synthetic coverage val regardless.")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--halt-step", type=int, default=0,
                    help="stop training at this step while the LR schedule still spans --steps — for "
                         "short fair-comparison probes that must share the long-run schedule. 0 = off.")
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--beta2", type=float, default=0.999,
                    help="AdamW beta2. The 0.999 default is the textbook late-collapse ingredient at "
                         "sustained LR (stale second moments); 0.95-0.98 is transformer practice and "
                         "raises the stable-LR ceiling.")
    ap.add_argument("--loss-balance", default="term", choices=["term", "expert"],
                    help="factored arch only: 'expert' gives every expert an equal gradient share "
                         "(fixes relic/orb starvation under the per-term default).")
    ap.add_argument("--num-targets", default="twohot", choices=["twohot", "hard"],
                    help="factored arch only: range-bin numeric target geometry. 'twohot' (default) is a "
                         "distance-aware symmetric triangular target that rewards near-miss bins "
                         "(restores the numeric metric structure — the M3.5 solo-dynamics fix); 'hard' is "
                         "the legacy one-hot CE (no partial credit). Decode stays argmax either way.")
    ap.add_argument("--focus-present", type=float, default=0.0,
                    help="factored SOLO runs only (--train-experts set): oversample states where a "
                         "trained expert has >=1 present token. Fraction R of each batch is drawn from "
                         "present states, 1-R from empty states (kept for presence calibration). 0 = off "
                         "(default). Needs the pre-tokenized cache; e.g. 0.9 for a sparse expert (orbs).")
    # Per-expert training (roadmap M3.5 sequential strategy). Train one/few experts at a time; the rest
    # are FROZEN (excluded from the optimizer) and their encode/decode is SKIPPED — a solo small-expert
    # run pays only that expert's compute (very high states/s).
    ap.add_argument("--train-experts", default="",
                    help="factored only: comma-separated experts to TRAIN (e.g. 'relics'); the rest are "
                         "frozen + skipped. Empty (default) = all experts (the joint run).")
    ap.add_argument("--val-experts", default="all", choices=["all", "trained-only"],
                    help="factored only: 'all' (default) runs the full report card; 'trained-only' "
                         "evaluates ONLY the trained experts (expert_dist/expert_exact/relic_set_f1) so a "
                         "solo run doesn't pay the full-model val decode.")
    ap.add_argument("--init-expert-from", nargs="*", default=[], metavar="name=ckpt",
                    help="factored only: warm-start an expert's weights from the matching slice of a full "
                         "factored checkpoint (e.g. cards=wm_t3_v3.pt.best) — seed a solo run from the "
                         "joint run's partial progress. Repeatable; validates slice width + config match.")
    ap.add_argument("--slice-width", action="append", default=None, metavar="NAME=W",
                    help="factored: override an expert's latent slice width, e.g. --slice-width "
                         "potions=256 (repeatable). Widths must divide by simnorm-group.")
    ap.add_argument("--relic-dec-layers", type=int, default=1,
                    help="factored only: relic-expert decoder depth (default 1 = the shallow expert "
                         "default; a deeper-decoder variant uses e.g. 3).")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--buffer", type=int, default=16384, help="shuffle-buffer size (states)")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-states", type=int, default=2000, help="fixed val-split sample size")
    ap.add_argument("--val-batch", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=50, help="train step-window for metric emit")
    # Architecture.
    ap.add_argument("--arch", default="mono", choices=["mono", "factored"],
                    help="'mono' (default) = the monolithic single-latent AE (byte-identical to prior "
                         "runs); 'factored' = the T3 expert-per-category AE (roadmap M3.5): independent "
                         "per-category experts, each with its own named latent slice (scalars/creatures/"
                         "cards/relics/potions/orbs), tier-1 scalars exact by construction. Stamped in "
                         "checkpoint meta with the slice layout; loads reject a mismatch.")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=3)
    ap.add_argument("--pool-layers", type=int, default=2)
    ap.add_argument("--latents", type=int, default=8)
    ap.add_argument("--z-dim", type=int, default=512)
    ap.add_argument("--simnorm-group", type=int, default=8)
    ap.add_argument("--cat-dim", type=int, default=24)
    ap.add_argument("--n-mem", type=int, default=16)
    ap.add_argument("--latent-mode", default="flat", choices=["flat", "tokens"],
                    help="latent structure A/B (design §10 / CP4): 'flat' = pooled z_dim vector (default, "
                         "byte-identical to prior runs); 'tokens' = latent_k x d_model token set consumed "
                         "directly as decoder memory (no flatten/expand bottleneck).")
    ap.add_argument("--latent-k", type=int, default=16,
                    help="tokens-mode only: number of latent tokens kept by the Perceiver pool.")
    # Loss/recipe probe flags (roadmap 3.1 experiment series — each default OFF, byte-identical when off,
    # toggled independently for one-change-at-a-time 5k-step probes vs the tokens control curve).
    ap.add_argument("--num-head", default="mse", choices=["mse", "twohot"],
                    help="numeric decode: 'mse' (default) regresses the symlog block; 'twohot' is "
                         "DreamerV3 two-hot classification over a 64-bin symlog grid per numeric column "
                         "(CE loss; expectation-decoded, so reconstruction/report are unchanged). "
                         "Stamped in checkpoint meta; --resume rejects a mismatch.")
    ap.add_argument("--card-ce", default="plain", choices=["plain", "balanced"],
                    help="card-identity (card column 0) cross-entropy weighting: 'plain' (default) or "
                         "'balanced' = per-class 1/sqrt(freq), computed once from the corpus card-index "
                         "distribution and cached to disk by tokenizer signature. Other columns unchanged.")
    ap.add_argument("--card-ce-states", type=int, default=200000,
                    help="states scanned from the train split to estimate the card-index distribution "
                         "for --card-ce balanced.")
    ap.add_argument("--card-ce-cache", default="",
                    help="dir for the cached card-CE weight vector (default: the checkpoint's dir).")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay for a weight moving average (e.g. 0.999); 0 = off (default). Val "
                         "passes evaluate the EMA weights; checkpoints save raw + EMA; --resume restores "
                         "both.")
    # Plumbing.
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/wm_encdec.pt")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true", help="resume model+optimizer+step from --ckpt")
    ap.add_argument("--run-dir", default="checkpoints/runs")
    ap.add_argument("--run-label", default=None)
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--val-cache", default=None, help="path to cache the tokenized val sample (.npz)")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"],
                    help="autocast the training forward+loss to bfloat16 (Ampere+; backward/optimizer "
                         "stay fp32, no GradScaler needed). Val passes always run fp32.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train_encdec] CUDA not available; falling back to CPU (slow).")
        args.device = "cpu"
    device = torch.device(args.device)
    # TF32 matmuls/convs: ~free Ampere speedup, standard for training; exactness lives in the
    # detokenize-side rounding, not fp32 matmul precision.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and args.amp == "bf16" and torch.cuda.is_bf16_supported()
    print(f"[train_encdec] device={device} "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} "
          f"tf32={device.type == 'cuda'} amp={'bf16' if use_amp else 'off'}")

    factored = args.arch == "factored"
    # Per-expert training set (factored). None = all experts (joint run); else the trained subset.
    train_active: Optional[List[str]] = None
    if factored and args.train_experts.strip():
        train_active = [n.strip() for n in args.train_experts.split(",") if n.strip()]
    # Synthetic-space training source (roadmap M3.5). Only a factored SOLO run may draw synthetic batches
    # (the generators are per-expert; a joint run has no single designed category). DOCTRINE: synthetic is
    # the DEFAULT for a factored solo run (real corpus data is eval-only); an explicit --data opts back in.
    if args.data is None:
        args.data = "synth" if (factored and train_active is not None) else "real"
    data_mode, frac_synth = _parse_data(args.data)
    if data_mode != "real":
        if not factored or train_active is None:
            raise SystemExit(f"--data {args.data!r} requires --arch factored with --train-experts set "
                             f"(the synthetic generators are per-expert; a joint run has no designed "
                             f"category).")
        unknown = [n for n in train_active if n not in SY._FILLERS]
        if unknown:
            raise SystemExit(f"--data {args.data!r}: no synthetic generator for {unknown}; "
                             f"choose from {sorted(SY._FILLERS)}")
    if factored:
        relic_overrides = ({"relics": {"dec_layers": args.relic_dec_layers}}
                           if args.relic_dec_layers != 1 else None)
        model = MF.FactoredWorldModelAE(slice_widths=_slice_width_overrides(args), d_model=args.d_model, n_heads=args.heads, cat_dim=args.cat_dim,
                                        simnorm_group=args.simnorm_group,
                                        expert_overrides=relic_overrides).to(device)
        if train_active is not None:
            unknown = [n for n in train_active if n not in model.experts]
            if unknown:
                raise SystemExit(f"--train-experts: unknown expert(s) {unknown}; "
                                 f"choose from {list(model.experts.keys())}")
    else:
        model = M.WorldModelAE(d_model=args.d_model, n_heads=args.heads, enc_layers=args.enc_layers,
                               dec_layers=args.dec_layers, n_pool_layers=args.pool_layers,
                               n_latents=args.latents, z_dim=args.z_dim,
                               simnorm_group=args.simnorm_group, cat_dim=args.cat_dim, n_mem=args.n_mem,
                               latent_mode=args.latent_mode, latent_k=args.latent_k,
                               num_head=args.num_head).to(device)

    def _freeze_and_params(m) -> list:
        """Factored per-expert freeze: set requires_grad per the trained set and return ONLY the trained
        params for the optimizer (frozen experts are excluded, so they stay byte-identical)."""
        if not factored or train_active is None:
            return list(m.parameters())
        params: list = []
        for name, ex in m.experts.items():
            on = name in train_active
            for p in ex.parameters():
                p.requires_grad_(on)
            if on:
                params += list(ex.parameters())
        return params

    # Warm-start experts from a full checkpoint's slices (fresh run only; --resume carries its own weights).
    if factored and args.init_expert_from and not args.resume:
        for item in args.init_expert_from:
            if "=" not in item:
                raise SystemExit(f"--init-expert-from entry {item!r} must be name=checkpoint")
            name, src = item.split("=", 1)
            if name not in model.experts:
                raise SystemExit(f"--init-expert-from: unknown expert {name!r}")
            stamp = MF.init_expert_from(model, name, src, device)
            print(f"[train_encdec] warm-started expert {name} from {src} "
                  f"(source step {(MF.read_meta(src)).get('step')})", flush=True)

    opt = torch.optim.AdamW(_freeze_and_params(model), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, args.beta2))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(args.warmup, args.steps))

    def save_fn(path, **kw):
        (MF.save_checkpoint if factored else M.save_checkpoint)(path, model, **kw)

    start_step = 0
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        if factored:
            model, meta = MF.load_checkpoint(args.ckpt, device)
        else:
            model, meta = M.load_checkpoint(args.ckpt, device, expect_latent_mode=args.latent_mode,
                                            expect_num_head=args.num_head)
        model = model.to(device)
        opt = torch.optim.AdamW(_freeze_and_params(model), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, args.beta2))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(args.warmup, args.steps))
        if os.path.exists(args.ckpt + ".train"):
            ts = torch.load(args.ckpt + ".train", map_location=device)
            try:
                opt.load_state_dict(ts["optimizer"])
            except Exception as e:
                print(f"[train_encdec] optimizer not restored ({e}); fresh optimizer.")
            start_step = int(ts.get("step", 0))
            for _ in range(start_step):
                sched.step()
        print(f"[train_encdec] resumed from {args.ckpt} at step {start_step}")

    # Class-balanced card CE (default OFF). Computed/cached once, then passed to every loss call.
    card_ce_weights: Optional[torch.Tensor] = None
    if args.card_ce == "balanced":
        if factored:
            print("[train_encdec] --card-ce balanced ignored under --arch factored", flush=True)
        else:
            card_ce_weights = _card_ce_weights(args, device)

    # Loss closure — factored sums the per-expert losses (over the trained subset); mono uses the shared
    # reconstruction loss.
    def loss_fn(batch_, out_, active=None):
        if factored:
            return MF.compute_losses(batch_, out_, model, balance=args.loss_balance, active=active,
                                      num_targets=args.num_targets)
        return M.compute_losses(batch_, out_, card_ce_weights=card_ce_weights)

    # Weight EMA (default OFF). Built from the (possibly resumed) live model; restores its shadow too.
    ema: Optional[M.EMA] = None
    if args.ema and args.ema > 0.0:
        ema = M.EMA(model, args.ema)
        if args.resume and args.ckpt and os.path.exists(args.ckpt + ".ema"):
            ema.load_state_dict(torch.load(args.ckpt + ".ema", map_location=device))
            print(f"[train_encdec] restored EMA shadow (decay={args.ema})", flush=True)
        else:
            print(f"[train_encdec] EMA enabled (decay={args.ema})", flush=True)

    def _ckpt_extra() -> Optional[Dict[str, Any]]:
        extra: Dict[str, Any] = {}
        if args.card_ce != "plain":
            extra["card_ce"] = args.card_ce
        if ema is not None:
            extra["ema_decay"] = args.ema
        return extra or None

    n_params = M.param_count(model)
    if factored:
        latent_desc = f"factored[{model.latent_dim}]"
        layout = " ".join(f"{n}={b - a}" for n, (a, b) in model.slice_layout.items())
        trained_desc = ("ALL" if train_active is None else ",".join(train_active))
        trained_params = sum(MF.param_count(model.experts[n]) for n in
                             (train_active if train_active is not None else list(model.experts)))
        print(f"[train_encdec] arch=factored params={n_params:,} "
              f"d_model={args.d_model} latent_dim={model.latent_dim} slices[{layout}] "
              f"batch={args.batch}", flush=True)
        print(f"[train_encdec] TRAIN experts=[{trained_desc}] trainable_params={trained_params:,} "
              f"val_experts={args.val_experts}", flush=True)
        for ename, ex in model.experts.items():
            tag = "" if (train_active is None or ename in train_active) else " (frozen)"
            print(f"[train_encdec]   expert {ename:10s} params={MF.param_count(ex):>10,d}{tag}",
                  flush=True)
    else:
        latent_desc = (f"tokens[{args.latent_k}x{args.d_model}]" if args.latent_mode == "tokens"
                       else f"flat[{args.z_dim}]")
        print(f"[train_encdec] params={n_params:,} d_model={args.d_model} latent={latent_desc} "
              f"simnorm_group={args.simnorm_group} enc={args.enc_layers} dec={args.dec_layers} "
              f"batch={args.batch}", flush=True)

    print(f"[train_encdec] loading fixed val sample ({args.val_states} states)...", flush=True)
    cache_dir = args.cache or None
    val_stacked, val_acts = D.load_fixed_sample(args.corpus, "val", args.val_states, args.val_cache,
                                                cache_dir=cache_dir)
    print(f"[train_encdec] val sample: {len(val_acts)} states", flush=True)

    label = args.run_label or (os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt else "wm")
    mw = MetricsWriter(run_dir=args.run_dir, label=label, argv=sys.argv, config=vars(args),
                       kind="wm-encdec", feature_version=tokens.TOKENIZER_VERSION,
                       catalog_signature=tokens.tokenizer_signature(), enabled=not args.no_metrics)
    if mw.enabled:
        print(f"[train_encdec] metrics -> {mw.run_dir}", flush=True)

    # Training stream. Synthetic / mixed sources (factored solo) generate batches in tokenizer-array
    # space; the real source streams the corpus cache (optionally focus-present oversampled).
    if data_mode != "real":
        npr = np.random.default_rng(args.seed + 0x5117)
        if data_mode == "synth":
            print(f"[train_encdec] DATA=synth: generating uniform-with-design batches for "
                  f"{train_active} (no cache/corpus needed)", flush=True)
            cpu = SY.synth_batches(train_active, args.batch, npr)
        else:
            if C.resolve_manifest(cache_dir) is None:
                raise SystemExit(f"--data {args.data!r} needs the real cache at {cache_dir!r} for the "
                                 f"real fraction; build one first.")
            print(f"[train_encdec] DATA=mixed:{frac_synth:.2f}: {frac_synth:.0%} synthetic + "
                  f"{1 - frac_synth:.0%} real per batch for {train_active}", flush=True)
            cpu = SY.mixed_batches(cache_dir, "train", train_active, args.batch, frac_synth, rng)
        stream = ((M.to_tensors(s, device), a) for s, a in D.prefetch(cpu, depth=4))
    else:
        # Focus-present sampling (solo runs only): oversample states with a present trained-expert token.
        focus_experts = (train_active if (factored and args.focus_present > 0.0
                                          and train_active is not None) else None)
        if args.focus_present > 0.0 and focus_experts is None:
            print("[train_encdec] --focus-present ignored: it applies only to factored SOLO runs "
                  "(--train-experts set).", flush=True)
        if focus_experts is not None:
            nat = float(D.expert_present_mask(val_stacked, focus_experts).mean())
            print(f"[train_encdec] focus-present R={args.focus_present:.2f} experts={focus_experts}: "
                  f"natural present-state fraction (val) {nat:.3f} -> per-batch target "
                  f"{args.focus_present:.3f}", flush=True)
        stream = D.train_batches(args.corpus, "train", args.batch, args.buffer, device, rng,
                                 cache_dir=cache_dir, focus_experts=focus_experts,
                                 focus_present=args.focus_present)

    # Synthetic coverage-val (factored solo): a FIXED seeded 2000-config synthetic sample per trained
    # expert, evaluated every val pass alongside the real fixed val — the coverage yardstick (the real
    # val stays the deployment yardstick). Always built for a factored solo run, whatever --data is.
    cov_stacked = cov_acts = None
    if factored and train_active is not None:
        cov_stacked, cov_acts = SY.coverage_val_sample(train_active, args.val_states, SY.COVERAGE_VAL_SEED)
        print(f"[train_encdec] coverage-val: {len(cov_acts)} fixed synthetic configs for {train_active} "
              f"(seed {SY.COVERAGE_VAL_SEED:#x})", flush=True)

    win = {"loss": 0.0, "loss_categorical": 0.0, "loss_numeric": 0.0, "loss_presence": 0.0}
    win_states = 0
    win_t0 = time.perf_counter()
    model.train()
    step = start_step
    best_dist = float("inf")   # lowest val state_dist so far (drives the .best checkpoint)
    try:
        for step in range(start_step + 1, args.steps + 1):
            if args.halt_step and step > args.halt_step:
                print(f"[train_encdec] halt-step {args.halt_step} reached; stopping (schedule spans "
                      f"{args.steps}).", flush=True)
                break
            batch, _acts = next(stream)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                if factored:
                    z, out = model(batch, active_experts=train_active)
                    losses = loss_fn(batch, out, active=train_active)
                else:
                    z, out = model(batch)
                    losses = loss_fn(batch, out)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)

            for k in win:
                win[k] += float(losses[k])
            win_states += args.batch

            if step % args.log_every == 0:
                dt = time.perf_counter() - win_t0
                sps = win_states / max(1e-6, dt)
                lr = sched.get_last_lr()[0]
                avg = {k: win[k] / args.log_every for k in win}
                print(f"[step {step:6d}] loss={avg['loss']:.4f} cat={avg['loss_categorical']:.4f} "
                      f"num={avg['loss_numeric']:.4f} pres={avg['loss_presence']:.4f} lr={lr:.2e} "
                      f"sps={sps:.0f}", flush=True)
                if mw.enabled:
                    mw.emit("train", step, "train.loss", avg["loss"])
                    mw.emit("train", step, "train.loss_categorical", avg["loss_categorical"])
                    mw.emit("train", step, "train.loss_numeric", avg["loss_numeric"])
                    mw.emit("train", step, "train.loss_presence", avg["loss_presence"])
                    mw.emit("train", step, "train.lr", lr)
                    mw.emit("train", step, "train.states_per_s", sps)
                win = {k: 0.0 for k in win}
                win_states = 0
                win_t0 = time.perf_counter()

            if args.val_every > 0 and (step % args.val_every == 0 or step == start_step + 1):
                # Val evaluates the EMA weights when EMA is on (swap in/out around the pass).
                if ema is not None:
                    ema.store(model)
                    ema.copy_to(model)
                trained_only = factored and args.val_experts == "trained-only" and train_active is not None
                overall, by_act, vloss = run_val(model, val_stacked, val_acts, args.val_batch, device,
                                                 loss_fn, experts=factored, active=train_active,
                                                 trained_only=trained_only)
                # Synthetic coverage-val: the same trained experts, focused, on the fixed synthetic
                # coverage set — evaluated under the same (EMA) weights as the real val.
                cov_overall = None
                if cov_stacked is not None:
                    cov_overall, _cov_by_act, _cov_loss = run_val(
                        model, cov_stacked, cov_acts, args.val_batch, device, loss_fn, experts=True,
                        active=train_active, trained_only=True)
                if ema is not None:
                    ema.restore(model)
                if trained_only:
                    # Solo run: only the trained experts' focused metrics exist.
                    ee = " ".join(f"{n}: dist={overall['expert_dist::' + n]:.4f} "
                                  f"exact={overall['expert_exact::' + n]:.4f}" for n in train_active)
                    extra_f1 = (f" relic_f1={overall['relic_set_f1']:.4f}"
                                if "relic_set_f1" in overall else "")
                    print(f"         VAL[{len(val_acts)}] step={step} loss={vloss['loss']:.3f} "
                          f"{ee}{extra_f1}", flush=True)
                    # .best driven by the trained experts' mean reconstruction distance.
                    sel = float(np.mean([overall[f"expert_dist::{n}"] for n in train_active]))
                else:
                    print(f"         VAL[{len(val_acts)}] loss={vloss['loss']:.3f} "
                          f"card_id={overall['card_id_top1']:.3f} zone={overall['card_zone_acc']:.3f} "
                          f"pw_id={overall['power_id_top1']:.3f} hp_mae={overall['creature_hp_mae']:.2f} "
                          f"energy={overall['energy_acc']:.3f} exact={overall['exact_state_rate']:.3f}",
                          flush=True)
                    if factored:
                        ed = " ".join(f"{n}={overall['expert_dist::' + n]:.3f}" for n in EXPERT_ORDER)
                        ex_str = " ".join(f"{n}={overall['expert_exact::' + n]:.3f}"
                                          for n in EXPERT_ORDER)
                        print(f"         VAL scalar_exact={overall['scalar_exact']:.4f} "
                              f"expert_dist[{ed}]", flush=True)
                        print(f"         VAL expert_exact[{ex_str}]", flush=True)
                    sel = overall["state_dist"]
                if cov_overall is not None:
                    cc = " ".join(f"{n}: dist={cov_overall['expert_dist::' + n]:.4f} "
                                  f"exact={cov_overall['expert_exact::' + n]:.4f}" for n in train_active)
                    cf1 = (f" relic_f1={cov_overall['relic_set_f1']:.4f}"
                           if "relic_set_f1" in cov_overall else "")
                    print(f"         COVERAGE-VAL[{len(cov_acts)}] {cc}{cf1}", flush=True)
                if mw.enabled:
                    if trained_only:
                        # Just the active experts' focused metrics (no full report card was computed).
                        for k in vloss:
                            mw.emit("eval", step, f"eval.{k}", vloss[k])
                        for ename in train_active:
                            mw.emit("eval", step, "eval.expert_dist",
                                    overall[f"expert_dist::{ename}"], tags={"expert": ename})
                            mw.emit("eval", step, "eval.expert_exact",
                                    overall[f"expert_exact::{ename}"], tags={"expert": ename})
                        if "relic_set_f1" in overall:
                            mw.emit("eval", step, "eval.relic_set_f1", overall["relic_set_f1"])
                    else:
                        for name in report.METRIC_NAMES:
                            mw.emit("eval", step, f"eval.{name}", overall[name])
                        for act, met in by_act.items():
                            for name in report.METRIC_NAMES:
                                mw.emit("eval", step, f"eval.{name}", met[name], tags={"act": act})
                        for k in vloss:
                            mw.emit("eval", step, f"eval.{k}", vloss[k])
                        if factored:
                            # Per-expert reconstruction error + exactness, one line per expert tagged by
                            # name (the dashboard groups by "expert" to overlay per-decoder curves).
                            for ename in EXPERT_ORDER:
                                mw.emit("eval", step, "eval.expert_dist",
                                        overall[f"expert_dist::{ename}"], tags={"expert": ename})
                                mw.emit("eval", step, "eval.expert_exact",
                                        overall[f"expert_exact::{ename}"], tags={"expert": ename})
                            mw.emit("eval", step, "eval.scalar_exact", overall["scalar_exact"])
                    # Synthetic coverage-val: DISTINCT metric names (eval.*_cov). Sharing the real
                    # metrics' names with only a val=coverage tag polluted the dashboard: group-by
                    # "expert" merged and AVERAGED the real and coverage series into one line, making
                    # every probe look much worse than its deployment metric (owner-reported).
                    if cov_overall is not None:
                        for ename in train_active:
                            mw.emit("eval", step, "eval.expert_dist_cov",
                                    cov_overall[f"expert_dist::{ename}"], tags={"expert": ename})
                            mw.emit("eval", step, "eval.expert_exact_cov",
                                    cov_overall[f"expert_exact::{ename}"], tags={"expert": ename})
                        if "relic_set_f1" in cov_overall:
                            mw.emit("eval", step, "eval.relic_set_f1_cov",
                                    cov_overall["relic_set_f1"])
                # Best-val checkpoint: the periodic checkpoint overwrites in place, so a late-run
                # divergence used to destroy the best weights of the run (learned the hard way: the
                # first gate run collapsed at step ~63k and took its step-51k best with it). Keep the
                # lowest-distance model in a separate .best sidecar, always retrievable. In a solo run the
                # distance is the trained experts' mean expert_dist (state_dist isn't computed).
                if args.ckpt and sel < best_dist:
                    best_dist = sel
                    save_fn(args.ckpt + ".best", step=step, optimizer=None,
                            extra=dict(_ckpt_extra() or {}, best_state_dist=best_dist),
                            ema_state=ema.state_dict() if ema is not None else None)
                win_t0 = time.perf_counter()  # don't count val time against sps

            if args.ckpt and step % args.ckpt_every == 0:
                save_fn(args.ckpt, step=step, optimizer=opt, extra=_ckpt_extra(),
                        ema_state=ema.state_dict() if ema is not None else None)
    finally:
        if args.ckpt:
            save_fn(args.ckpt, step=step, optimizer=opt, extra=_ckpt_extra(),
                    ema_state=ema.state_dict() if ema is not None else None)
        mw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
