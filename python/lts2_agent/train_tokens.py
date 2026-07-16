"""GPU PPO trainer over the entity tokenizer (roadmap 2.2 — the PPO-on-tokens sanity pass).

Same PPO update (:mod:`lts2_agent.ppo_torch`), same vectorized rollout (:mod:`lts2_agent.rollout_torch`),
same :class:`lts2_agent.scenario.ScenarioConfig` knobs and reward, same metrics event stream — only the
model + featurization change, injected via :class:`lts2_agent.adapters.TokensAdapter`. This is a separate
CLI from ``train_torch`` on purpose: ``train_torch``'s eval/bodyguard helpers are hand-feature-coupled and
it is the recorded PPO baseline we must not perturb; a ``--model`` flag would fork most of ``main`` anyway.

Run (default scenario settings match the M0.5 baseline: random character, acts any, elite 0.2 boss 0.05):

    python -m lts2_agent.train_tokens --envs 16 --iterations 300 --eval-every 10 \
        --ckpt checkpoints/tokens_m2.pt

Metrics land as a ``kind="ppo-tokens"`` run under ``checkpoints/runs/`` so the dashboard can overlay it on
the baseline. Requires a CUDA PyTorch (falls back to CPU with a warning).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

from . import model_tokens, navigator, ppo_torch, reward
from .adapters import TokensAdapter
from .env import Lts2Env
from .metrics import MetricsWriter
from .rollout_torch import TorchScenarioRollout
from .scenario import ScenarioConfig

_EVAL_COMBAT = model_tokens.COMBAT_OPTION_KINDS


@torch.no_grad()
def greedy_winrate(model, adapter, envs, scfg, n_seeds, device, tag="EVAL", greedy=True):
    """Play a FIXED set of seeded fights (same seeds each call) with the GREEDY (argmax, deployed) or
    SAMPLED policy — the token-model counterpart of :func:`lts2_agent.train_torch.greedy_winrate`. Returns
    win rate, EndTurn fraction, avg HP lost, and a per-fight list for tagged metrics."""
    model.eval()
    wins = fights = endturns = steps = 0
    hp_lost_sum = 0.0
    per_fight: list[dict] = []
    for env in envs:
        for s in range(n_seeds):
            try:
                obs = env.reset_combat(seed=f"{tag}-{s}", character=scfg.character, elite_pct=scfg.elite_pct,
                                       boss_pct=scfg.boss_pct, starter_deck=scfg.starter_deck, act=scfg.act)
            except Exception:
                continue
            players = obs["state"].get("players") or []
            character = players[0].get("character") if players else None
            for _ in range(scfg.max_fight_len + 6):
                if obs["done"] or not obs["options"]:
                    break
                st, opts = obs["state"], obs["options"]
                ph = st.get("phase")
                if ph == "Choice":
                    try:
                        obs = env.step(navigator.noncombat_action(st, opts))
                    except Exception:
                        break
                    continue
                if ph != "Combat":
                    break
                f = adapter.featurize(st, opts)
                args = adapter.to_device(adapter.stack([f]), device)
                logits, _ = adapter.forward(model, args)
                a = int(logits[0].argmax()) if greedy else int(adapter.sample_action(logits)[0][0])
                if opts[a].get("kind") == "EndTurn":
                    endturns += 1
                steps += 1
                try:
                    obs = env.step(a)
                except Exception:
                    break
            fights += 1
            info = obs.get("info", {})
            won = bool(info.get("won"))
            hp_lost = info.get("hpLost") or 0
            if won:
                wins += 1
            hp_lost_sum += hp_lost
            per_fight.append({"won": won, "hp_lost": hp_lost, "act": info.get("act"),
                              "room": info.get("roomType"), "character": character})
    return {"win": wins / max(1, fights), "endturn": endturns / max(1, steps),
            "hp_lost": hp_lost_sum / max(1, fights), "n": fights, "fights": per_fight}


def scenario_summary(outcomes: list[dict]) -> dict:
    if not outcomes:
        return {"n": 0, "win_rate": 0.0, "mean_hp_lost": 0.0, "trunc_rate": 0.0}
    n = len(outcomes)
    wins = sum(1 for o in outcomes if o["won"])
    trunc = sum(1 for o in outcomes if o.get("truncated"))
    hp = float(np.mean([o["hp_lost"] for o in outcomes]))
    return {"n": n, "win_rate": wins / n, "mean_hp_lost": hp, "trunc_rate": trunc / n}


def main() -> int:
    ap = argparse.ArgumentParser(description="PyTorch/GPU PPO trainer over the entity tokenizer (roadmap 2.2)")
    ap.add_argument("--character", default=None, help="substring; omit for random character per env")
    ap.add_argument("--starter-deck", action="store_true")
    ap.add_argument("--act", type=int, default=-1, help="0/1/2; -1 = any")
    ap.add_argument("--elite-pct", type=float, default=0.2)
    ap.add_argument("--boss-pct", type=float, default=0.05)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=96, help="combat decisions per env per iteration")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01, help="entropy bonus (lower = sharper greedy)")
    ap.add_argument("--logit-reg", type=float, default=0.01, help="L2 penalty on logits (lower = more confident)")
    ap.add_argument("--ent-floor", type=float, default=0.6,
                    help="hard-penalize mean entropy below this so the served (sampled) policy can't "
                         "collapse onto EndTurn; 0 = off")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--step-penalty", type=float, default=0.0,
                    help="per-decision reward penalty (0 is safest; truncation already prevents stalling)")
    ap.add_argument("--hp-weight", type=float, default=1.0, help="penalty per fraction of start HP lost")
    ap.add_argument("--damage-weight", type=float, default=0.5, help="dense reward per fraction dmg dealt")
    ap.add_argument("--kill-weight", type=float, default=0.2, help="dense reward per enemy killed")
    # Model architecture (kept in the low-millions of params — see model_tokens).
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--pool-layers", type=int, default=2)
    ap.add_argument("--latents", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="checkpoints/tokens_m2.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shaping-ramp-frac", type=float, default=0.75,
                    help="fraction of training over which the dense damage/kill shaping ramps 1->0 "
                         "(0 = shaping off; the HP + anti-stall terms are always on)")
    ap.add_argument("--no-dense", action="store_true",
                    help="disable the dense damage/kill shaping entirely (win/loss + per-step HP + the "
                         "fight-length cap remain, so fights still can't stall)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start training from scratch")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="every N iters, measure fixed-seed GREEDY+SAMPLED win rate. 0 = off.")
    ap.add_argument("--eval-envs", type=int, default=3)
    ap.add_argument("--eval-fights", type=int, default=8, help="fights per eval env")
    ap.add_argument("--run-dir", default="checkpoints/runs",
                    help="root for per-run metrics dirs (<run-dir>/<run_id>/{manifest.json,events.jsonl})")
    ap.add_argument("--run-label", default=None, help="human label in the run id; default = ckpt stem")
    ap.add_argument("--no-metrics", action="store_true", help="disable the JSONL metrics event stream")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train_tokens] CUDA not available; falling back to CPU (this will be slow).")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[train_tokens] device={device} "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    adapter = TokensAdapter()
    pcfg = ppo_torch.PPOConfig(lr=args.lr, epochs=args.epochs, minibatch_size=args.minibatch,
                               ent_coef=args.ent_coef, logit_reg=args.logit_reg, ent_floor=args.ent_floor)

    train_state_path = (args.ckpt + ".train") if args.ckpt else None
    start_it = 1
    if args.ckpt and os.path.exists(args.ckpt) and not args.fresh:
        model, _meta = model_tokens.load_checkpoint(args.ckpt, device)
        model = model.to(device)
        opt = ppo_torch.make_optimizer(model, pcfg)
        if train_state_path and os.path.exists(train_state_path):
            ts = torch.load(train_state_path, map_location=device)
            try:
                opt.load_state_dict(ts["optimizer"])
            except Exception as e:
                print(f"[train_tokens] could not restore optimizer ({e}); continuing with a fresh one.")
            start_it = int(ts.get("iteration", 0)) + 1
        print(f"[train_tokens] resumed from {args.ckpt} at iter {start_it}")
    else:
        model = model_tokens.TokenActorCritic(d_model=args.d_model, n_heads=args.heads,
                                              n_pool_layers=args.pool_layers, n_latents=args.latents).to(device)
        opt = ppo_torch.make_optimizer(model, pcfg)
    n_params = sum(p.numel() for p in model.parameters())

    scfg = ScenarioConfig(
        n_envs=args.envs, steps_per_env=args.steps, gamma=args.gamma,
        character=args.character, elite_pct=args.elite_pct, boss_pct=args.boss_pct,
        starter_deck=args.starter_deck, act=(args.act if args.act >= 0 else None),
        weights=reward.ScenarioWeights(hp=args.hp_weight, damage=args.damage_weight,
                                       kill=args.kill_weight, step_penalty=args.step_penalty))

    print(f"[train_tokens] envs={args.envs} steps/env={args.steps} "
          f"batch={args.envs * args.steps} minibatch={args.minibatch} params={n_params} "
          f"d_model={args.d_model} lr={args.lr}")

    eval_envs = [Lts2Env() for _ in range(args.eval_envs)] if args.eval_every > 0 else []

    label = args.run_label or (os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt else "tokens")
    mw = MetricsWriter(run_dir=args.run_dir, label=label, argv=sys.argv, config=vars(args),
                       kind="ppo-tokens", feature_version=model_tokens.tokens.TOKENIZER_VERSION,
                       catalog_signature=adapter.signature(), enabled=not args.no_metrics)
    if mw.enabled:
        print(f"[train_tokens] metrics -> {mw.run_dir}", flush=True)

    rollout = TorchScenarioRollout(scfg, adapter=adapter)
    try:
        ramp_end = max(1.0, args.shaping_ramp_frac * args.iterations)
        for it in range(start_it, args.iterations + 1):
            if args.no_dense or args.shaping_ramp_frac <= 0.0:
                shaping_coef = 0.0
            else:
                shaping_coef = max(0.0, 1.0 - (it - 1) / ramp_end)
            t0 = time.perf_counter()
            batch, outcomes = rollout.collect(model, device, shaping_coef=shaping_coef)
            t_collect = time.perf_counter() - t0
            metrics = ppo_torch.update(model, opt, batch, pcfg, device, adapter=adapter)
            t_update = time.perf_counter() - t0 - t_collect

            n = int(batch["action"].shape[0])
            sps = n / (time.perf_counter() - t0)
            s = scenario_summary(outcomes)
            flen = n / max(1, s["n"])
            print(f"[it {it:04d}] steps={n} fights={s['n']} flen={flen:.0f} win={s['win_rate']:.2f} "
                  f"trunc={s['trunc_rate']:.2f} "
                  f"hpLost~{s['mean_hp_lost']:.1f} shape={shaping_coef:.2f} loss={metrics['loss']:.3f} "
                  f"kl={metrics['approx_kl']:.4f} ent={metrics['entropy']:.3f} ev={metrics['explained_var']:+.2f} "
                  f"rstd={metrics['ret_std']:.2f} vstd={metrics['val_std']:.2f} rmean={metrics['ret_mean']:+.2f} "
                  f"maxRatio={metrics['max_ratio']:.1f} maxLogit={metrics['max_logit']:.1f} | "
                  f"collect={t_collect:.2f}s update={t_update:.2f}s sps={sps:.0f}", flush=True)

            if mw.enabled:
                train_scalars = {
                    "train.win_rate": s["win_rate"], "train.hp_lost": s["mean_hp_lost"],
                    "train.trunc_rate": s["trunc_rate"], "train.fights": s["n"], "train.steps": n,
                    "train.flen": flen, "train.loss": metrics["loss"], "train.approx_kl": metrics["approx_kl"],
                    "train.entropy": metrics["entropy"], "train.explained_var": metrics["explained_var"],
                    "train.ret_std": metrics["ret_std"], "train.val_std": metrics["val_std"],
                    "train.ret_mean": metrics["ret_mean"], "train.max_ratio": metrics["max_ratio"],
                    "train.max_logit": metrics["max_logit"], "train.shaping": shaping_coef,
                    "train.sps": sps, "train.collect_s": t_collect, "train.update_s": t_update,
                }
                for name, value in train_scalars.items():
                    mw.emit("train", it, name, value)
                for o in outcomes:
                    tags = {"act": str(o.get("act")), "room": str(o.get("room")),
                            "character": str(o.get("character")),
                            "truncated": "true" if o.get("truncated") else "false"}
                    mw.emit("train", it, "fight.won", 1.0 if o["won"] else 0.0, tags=tags)
                    mw.emit("train", it, "fight.hp_lost", float(o.get("hp_lost") or 0), tags=tags)

            if eval_envs and (it % args.eval_every == 0 or it == start_it):
                g = greedy_winrate(model, adapter, eval_envs, scfg, args.eval_fights, device, greedy=True)
                sm = greedy_winrate(model, adapter, eval_envs, scfg, args.eval_fights, device, greedy=False)
                print(f"         EVAL[fixed {g['n']}] greedy: win={g['win']:.2f} hp~{g['hp_lost']:.0f} "
                      f"endT={g['endturn']:.2f}  |  sampled: win={sm['win']:.2f} hp~{sm['hp_lost']:.0f} "
                      f"endT={sm['endturn']:.2f}", flush=True)
                if mw.enabled:
                    for mode, res in (("greedy", g), ("sampled", sm)):
                        mw.emit("eval", it, f"eval.{mode}.win", res["win"])
                        mw.emit("eval", it, f"eval.{mode}.hp_lost", res["hp_lost"])
                        mw.emit("eval", it, f"eval.{mode}.endturn", res["endturn"])
                        for fo in res["fights"]:
                            tags = {"act": str(fo.get("act")), "room": str(fo.get("room")),
                                    "character": str(fo.get("character")), "mode": mode}
                            mw.emit("eval", it, "eval_fight.won", 1.0 if fo["won"] else 0.0, tags=tags)
                            mw.emit("eval", it, "eval_fight.hp_lost", float(fo.get("hp_lost") or 0), tags=tags)

            if args.ckpt:
                model_tokens.save_checkpoint(args.ckpt, model)
                if train_state_path:
                    torch.save({"optimizer": opt.state_dict(), "iteration": it}, train_state_path)
    finally:
        mw.close()
        rollout.close()
        for e in eval_envs:
            try:
                e.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
