"""PPO training loop for the combat decision engine.

Spawns N environment processes, collects on-policy combat transitions, runs a PPO update, checkpoints,
and logs per-iteration metrics (mean floor / win rate / score / reward) to stderr and a CSV. The
checkpoint it writes is loaded unchanged by :mod:`lts2_agent.policies.jax_policy` to serve the policy
in the TUI — same features, same model.

Run (after building Lts2.AgentHost)::

    python -m lts2_agent.train --iterations 200 --envs 8 --ckpt checkpoints/ppo

Everything on stdout would corrupt nothing here (this is the driver, not a protocol peer), but metrics
go to stderr so a redirected stdout stays clean if you pipe it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Optional

import jax
import numpy as np

from . import model, ppo
from .reward import RewardWeights, ScenarioWeights
from .rollout import Rollout, RolloutConfig
from .scenario import ScenarioConfig, ScenarioRollout


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _episode_summary(episodes: list[dict]) -> dict:
    if not episodes:
        return {"n": 0, "mean_floor": float("nan"), "win_rate": float("nan"),
                "mean_score": float("nan"), "max_floor": 0}
    floors = [e["floor"] for e in episodes]
    return {
        "n": len(episodes),
        "mean_floor": float(np.mean(floors)),
        "max_floor": int(np.max(floors)),
        "win_rate": float(np.mean([e["victory"] for e in episodes])),
        "mean_score": float(np.mean([e["score"] for e in episodes])),
    }


def _scenario_summary(outcomes: list[dict]) -> dict:
    if not outcomes:
        return {"n": 0, "win_rate": float("nan"), "mean_hp_lost": float("nan")}
    return {
        "n": len(outcomes),
        "win_rate": float(np.mean([o["won"] for o in outcomes])),
        "mean_hp_lost": float(np.mean([o["hp_lost"] for o in outcomes])),
    }


def train(args) -> int:
    scenario_mode = args.mode == "scenario"
    pcfg = ppo.PPOConfig(lr=args.lr, clip_eps=args.clip, vf_coef=args.vf_coef,
                         ent_coef=args.ent_coef, epochs=args.epochs, minibatch_size=args.minibatch)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    resumed = bool(args.resume and args.ckpt and os.path.exists(args.ckpt + ".meta.json"))
    if resumed:
        # Continue from an existing checkpoint (params carry over; the optimizer state restarts —
        # fine for on-policy PPO). The model config comes from the checkpoint, not the CLI flags.
        m, params, meta = model.load_checkpoint(args.ckpt)
        _log(f"[train] resumed params from {args.ckpt} (hidden={meta['hidden']}).")
    else:
        m = model.ActorCritic(hidden=args.hidden, embed_dim=args.embed_dim)
        params = model.init_params(init_key, m)
    state = ppo.create_train_state(m, params, pcfg)

    header = (["iter", "steps", "episodes", "win_rate", "mean_hp_lost", "mean_return",
               "loss", "pg_loss", "v_loss", "entropy", "approx_kl", "sps"] if scenario_mode else
              ["iter", "steps", "episodes", "mean_floor", "max_floor", "win_rate", "mean_score",
               "mean_return", "loss", "pg_loss", "v_loss", "entropy", "approx_kl", "sps"])
    csv_writer = csv_file = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        # When resuming, append to the existing metric history instead of clobbering it.
        append = resumed and os.path.exists(args.csv)
        csv_file = open(args.csv, "a" if append else "w", newline="")
        csv_writer = csv.writer(csv_file)
        if not append:
            csv_writer.writerow(header)

    _log(f"[train] mode={args.mode} envs={args.envs} steps/env={args.steps} "
         f"batch={args.envs * args.steps} lr={args.lr}")

    if scenario_mode:
        rollout = ScenarioRollout(ScenarioConfig(
            n_envs=args.envs, steps_per_env=args.steps, gamma=args.gamma, lam=args.lam,
            character=args.character, elite_pct=args.elite_pct, boss_pct=args.boss_pct,
            weights=ScenarioWeights(win=args.sw_win, loss=args.sw_loss, hp=args.sw_hp)), m=m)
    else:
        rollout = Rollout(RolloutConfig(
            n_envs=args.envs, steps_per_env=args.steps, gamma=args.gamma, lam=args.lam,
            character=args.character, ascension=args.ascension,
            weights=RewardWeights(hp=args.w_hp, damage=args.w_dmg, kill=args.w_kill,
                                  floor=args.w_floor, win=args.w_win, death=args.w_death)), m=m)

    try:
        for it in range(1, args.iterations + 1):
            t0 = time.perf_counter()
            key, sub = jax.random.split(key)
            batch, info_list = rollout.collect(state.params, sub)

            key, sub = jax.random.split(key)
            state, metrics = ppo.update(state, batch, pcfg, sub)

            n_steps = batch["action"].shape[0]
            sps = n_steps / (time.perf_counter() - t0)
            mean_return = float(np.mean(batch["ret"]))

            if scenario_mode:
                s = _scenario_summary(info_list)
                _log(f"[it {it:04d}] steps={n_steps} fights={s['n']} "
                     f"win={s['win_rate']:.2f} hpLost~{s['mean_hp_lost']:.1f} ret~{mean_return:.2f} "
                     f"loss={metrics['loss']:.3f} kl={metrics['approx_kl']:.4f} "
                     f"ent={metrics['entropy']:.3f} sps={sps:.0f}")
                row = [it, n_steps, s["n"], s["win_rate"], s["mean_hp_lost"], mean_return]
            else:
                s = _episode_summary(info_list)
                _log(f"[it {it:04d}] steps={n_steps} eps={s['n']} "
                     f"floor~{s['mean_floor']:.1f} (max {s['max_floor']}) win={s['win_rate']:.2f} "
                     f"score~{s['mean_score']:.0f} ret~{mean_return:.2f} "
                     f"loss={metrics['loss']:.3f} kl={metrics['approx_kl']:.4f} "
                     f"ent={metrics['entropy']:.3f} sps={sps:.0f}")
                row = [it, n_steps, s["n"], s["mean_floor"], s["max_floor"], s["win_rate"],
                       s["mean_score"], mean_return]

            if csv_writer:
                csv_writer.writerow(row + [metrics["loss"], metrics["pg_loss"], metrics["v_loss"],
                                           metrics["entropy"], metrics["approx_kl"], sps])
                csv_file.flush()

            if args.ckpt and (it % args.save_every == 0 or it == args.iterations):
                model.save_checkpoint(args.ckpt, state.params, m)
                _log(f"[train] checkpoint saved to {args.ckpt}")
    finally:
        rollout.close()
        if csv_file:
            csv_file.close()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PPO trainer for the STS2 combat decision engine.")
    p.add_argument("--mode", choices=["run", "scenario"], default="run",
                   help="'run' = full playthroughs; 'scenario' = isolated random combats")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=128, help="combat decisions per env per iteration")
    p.add_argument("--ckpt", default="checkpoints/ppo", help="checkpoint path prefix ('' to disable)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the existing --ckpt (params carry over; optimizer restarts)")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--csv", default=None, help="optional metrics CSV path")
    p.add_argument("--character", default="Ironclad")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    # model
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=16)
    # ppo
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    # run-mode reward weights
    p.add_argument("--w-hp", type=float, default=1.0)
    p.add_argument("--w-dmg", type=float, default=0.5)
    p.add_argument("--w-kill", type=float, default=0.5)
    p.add_argument("--w-floor", type=float, default=0.5)
    p.add_argument("--w-win", type=float, default=10.0)
    p.add_argument("--w-death", type=float, default=5.0)
    # scenario-mode knobs
    p.add_argument("--elite-pct", type=float, default=0.2, help="fraction of scenarios that are elites")
    p.add_argument("--boss-pct", type=float, default=0.05, help="fraction of scenarios that are bosses")
    p.add_argument("--sw-win", type=float, default=1.0, help="scenario reward: win bonus")
    p.add_argument("--sw-loss", type=float, default=1.0, help="scenario reward: loss penalty")
    p.add_argument("--sw-hp", type=float, default=1.0, help="scenario reward: penalty per fraction HP lost")
    args = p.parse_args(argv)
    if args.ckpt == "":
        args.ckpt = None
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
