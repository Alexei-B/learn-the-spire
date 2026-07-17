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
from .wm import report
from .wm import spec as S


def _lr_lambda(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))  # cosine decay to 0
    return fn


@torch.no_grad()
def run_val(model, sample_stacked, sample_acts, batch_size, device, card_ce_weights=None):
    """Full pass over the fixed val sample -> (overall metrics, by-act metrics, mean losses)."""
    model.eval()
    accum: Dict[str, Any] = {}
    loss_sums = {"loss": 0.0, "loss_categorical": 0.0, "loss_numeric": 0.0, "loss_presence": 0.0}
    nb = 0
    for batch, acts in D.iter_fixed_batches(sample_stacked, sample_acts, batch_size, device):
        z, out = model(batch)
        losses = M.compute_losses(batch, out, card_ce_weights=card_ce_weights)
        for k in loss_sums:
            loss_sums[k] += float(losses[k])
        nb += 1
        pairs = report.report_pairs(batch, out)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="World-model encoder/decoder trainer (roadmap 3.1)")
    ap.add_argument("--corpus", default="data/corpus", help="corpus root (train split streamed)")
    ap.add_argument("--cache", default="data/corpus_tok",
                    help="pre-tokenized cache dir; used automatically when it exists and its signature "
                         "matches (GPU-bound). Empty string disables. Build: python -m "
                         "lts2_agent.wm.cache build --corpus <corpus> --out <cache>")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--halt-step", type=int, default=0,
                    help="stop training at this step while the LR schedule still spans --steps — for "
                         "short fair-comparison probes that must share the long-run schedule. 0 = off.")
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--buffer", type=int, default=16384, help="shuffle-buffer size (states)")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-states", type=int, default=2000, help="fixed val-split sample size")
    ap.add_argument("--val-batch", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=50, help="train step-window for metric emit")
    # Architecture.
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
    ap.add_argument("--relic-head", default="slots", choices=["slots", "set"],
                    help="relic decode: 'slots' (default) is 24 independent per-slot categoricals over "
                         "the relic catalog (can decode duplicate relics under uncertainty); 'set' is ONE "
                         "multi-hot head over the catalog (BCE loss; top-k-by-cardinality decode, "
                         "duplicate-free by construction — the CP4 relic-error fix). Stamped in "
                         "checkpoint meta; --resume rejects a mismatch.")
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

    model = M.WorldModelAE(d_model=args.d_model, n_heads=args.heads, enc_layers=args.enc_layers,
                           dec_layers=args.dec_layers, n_pool_layers=args.pool_layers,
                           n_latents=args.latents, z_dim=args.z_dim, simnorm_group=args.simnorm_group,
                           cat_dim=args.cat_dim, n_mem=args.n_mem, latent_mode=args.latent_mode,
                           latent_k=args.latent_k, num_head=args.num_head,
                           relic_head=args.relic_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(args.warmup, args.steps))

    start_step = 0
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        model, meta = M.load_checkpoint(args.ckpt, device, expect_latent_mode=args.latent_mode,
                                        expect_num_head=args.num_head,
                                        expect_relic_head=args.relic_head)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        card_ce_weights = _card_ce_weights(args, device)

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

    stream = D.train_batches(args.corpus, "train", args.batch, args.buffer, device, rng,
                             cache_dir=cache_dir)

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
                z, out = model(batch)
                losses = M.compute_losses(batch, out, card_ce_weights=card_ce_weights)
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
                overall, by_act, vloss = run_val(model, val_stacked, val_acts, args.val_batch, device,
                                                 card_ce_weights=card_ce_weights)
                if ema is not None:
                    ema.restore(model)
                print(f"         VAL[{len(val_acts)}] loss={vloss['loss']:.3f} "
                      f"card_id={overall['card_id_top1']:.3f} zone={overall['card_zone_acc']:.3f} "
                      f"pw_id={overall['power_id_top1']:.3f} hp_mae={overall['creature_hp_mae']:.2f} "
                      f"energy={overall['energy_acc']:.3f} exact={overall['exact_state_rate']:.3f}",
                      flush=True)
                if mw.enabled:
                    for name in report.METRIC_NAMES:
                        mw.emit("eval", step, f"eval.{name}", overall[name])
                    for act, met in by_act.items():
                        for name in report.METRIC_NAMES:
                            mw.emit("eval", step, f"eval.{name}", met[name], tags={"act": act})
                    for k in vloss:
                        mw.emit("eval", step, f"eval.{k}", vloss[k])
                # Best-val checkpoint: the periodic checkpoint overwrites in place, so a late-run
                # divergence used to destroy the best weights of the run (learned the hard way: the
                # first gate run collapsed at step ~63k and took its step-51k best with it). Keep the
                # lowest-val-state_dist model in a separate .best sidecar, always retrievable.
                if args.ckpt and overall["state_dist"] < best_dist:
                    best_dist = overall["state_dist"]
                    M.save_checkpoint(args.ckpt + ".best", model, step=step, optimizer=None,
                                      extra=dict(_ckpt_extra() or {}, best_state_dist=best_dist),
                                      ema_state=ema.state_dict() if ema is not None else None)
                win_t0 = time.perf_counter()  # don't count val time against sps

            if args.ckpt and step % args.ckpt_every == 0:
                M.save_checkpoint(args.ckpt, model, step=step, optimizer=opt, extra=_ckpt_extra(),
                                  ema_state=ema.state_dict() if ema is not None else None)
    finally:
        if args.ckpt:
            M.save_checkpoint(args.ckpt, model, step=step, optimizer=opt, extra=_ckpt_extra(),
                              ema_state=ema.state_dict() if ema is not None else None)
        mw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
