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
from . import synthetic_eval
from .scenario import ScenarioConfig, ScenarioEvalSet, ScenarioRollout


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
    header += ["eval_win", "eval_hp_lost", "eval_hp_frac"]   # deterministic held-out eval set
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

    evalset = None
    bodyguard_fixture = None
    if scenario_mode:
        rollout = ScenarioRollout(ScenarioConfig(
            n_envs=args.envs, steps_per_env=args.steps, gamma=args.gamma, lam=args.lam,
            character=args.character, elite_pct=args.elite_pct, boss_pct=args.boss_pct,
            starter_deck=args.starter_deck, act=args.act,
            weights=ScenarioWeights(win=args.sw_win, loss=args.sw_loss, hp=args.sw_hp)), m=m)
        if args.eval_every > 0 and args.eval_seeds > 0:
            evalset = ScenarioEvalSet(
                seeds=[f"EVALSET{i}" for i in range(args.eval_seeds)], m=m, n_envs=args.eval_envs,
                elite_pct=args.elite_pct, boss_pct=args.boss_pct, character=args.character,
                starter_deck=args.starter_deck, act=args.act)
        if args.stop_on_bodyguard:
            bodyguard_fixture = synthetic_eval.load_fixture(args.bodyguard_fixture)
            _apply_bg = jax.jit(m.apply)
    else:
        rollout = Rollout(RolloutConfig(
            n_envs=args.envs, steps_per_env=args.steps, gamma=args.gamma, lam=args.lam,
            character=args.character, ascension=args.ascension,
            weights=RewardWeights(hp=args.w_hp, damage=args.w_dmg, kill=args.w_kill,
                                  floor=args.w_floor, win=args.w_win, death=args.w_death)), m=m)
    bg_streak = 0

    try:
        for it in range(1, args.iterations + 1):
            t0 = time.perf_counter()
            key, sub = jax.random.split(key)
            batch, info_list = rollout.collect(state.params, sub)
            t_collect = time.perf_counter() - t0

            key, sub = jax.random.split(key)
            state, metrics = ppo.update(state, batch, pcfg, sub)
            t_update = time.perf_counter() - t0 - t_collect

            n_steps = batch["action"].shape[0]
            sps = n_steps / (time.perf_counter() - t0)
            if os.environ.get("LTS2_TIMING"):
                _log(f"[timing it {it}] collect={t_collect:.2f}s ({100*t_collect/(t_collect+t_update):.0f}%) "
                     f"update={t_update:.2f}s | {n_steps} steps -> {sps:.0f} sps")
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

            # Deterministic held-out eval on the fixed seed set — the low-noise progress signal.
            ev = None
            is_eval_it = args.eval_every > 0 and (
                it == 1 or it % args.eval_every == 0 or it == args.iterations)
            if evalset is not None and is_eval_it:
                ev = evalset.evaluate(state.params)
                _log(f"[it {it:04d}] EVAL(n={ev['n']}) win={ev['win_rate']:.3f} "
                     f"hpLost~{ev['mean_hp_lost']:.1f} hpFrac={ev['mean_hp_frac']:.3f}")

            # Synthetic (harness-free) decision check on the fixed Bodyguard fixture: print the ranking
            # and early-stop once the model orders the starter hand correctly for several evals in a row
            # (a single pass on a barely-trained model is just noise).
            bg_stop = False
            if bodyguard_fixture is not None and is_eval_it:
                scores = synthetic_eval.per_card_scores(_apply_bg, state.params, bodyguard_fixture)
                bg_pass, _why = synthetic_eval.bodyguard_pass(scores)
                bg_streak = bg_streak + 1 if bg_pass else 0
                bg_stop = bg_streak >= args.bodyguard_patience
                order = " > ".join(f"{k}({v:.2f})" for k, v in synthetic_eval.ranking(scores))
                _log(f"[it {it:04d}] BODYGUARD pass={bg_pass} streak={bg_streak}/{args.bodyguard_patience} | {order}")

            if csv_writer:
                eval_cols = [ev["win_rate"], ev["mean_hp_lost"], ev["mean_hp_frac"]] if ev else ["", "", ""]
                csv_writer.writerow(row + [metrics["loss"], metrics["pg_loss"], metrics["v_loss"],
                                           metrics["entropy"], metrics["approx_kl"], sps] + eval_cols)
                csv_file.flush()

            finite = np.isfinite(mean_return) and np.isfinite(metrics["loss"])
            if args.ckpt and finite and (it % args.save_every == 0 or it == args.iterations or bg_stop):
                model.save_checkpoint(args.ckpt, state.params, m)
                _log(f"[train] checkpoint saved to {args.ckpt}")
            elif not finite:
                _log(f"[train] WARNING it {it}: non-finite metrics, skipping checkpoint save.")

            if bg_stop:
                _log(f"[train] Bodyguard decision correct for {bg_streak} evals (it {it}) — stopping early.")
                break
    finally:
        rollout.close()
        if evalset is not None:
            evalset.close()
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
    p.add_argument("--character", default=None,
                   help="fixed character id; omit for RANDOM per fight (scenario) / first char (run)")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    # focused scenario knobs
    p.add_argument("--starter-deck", action="store_true",
                   help="use the character's fixed starting deck + starter relic (low-noise)")
    p.add_argument("--act", type=int, default=None, help="restrict encounters to this act (0/1/2)")
    # deterministic held-out eval set (scenario mode)
    p.add_argument("--eval-every", type=int, default=5, help="run the fixed eval set every N iters (0=off)")
    p.add_argument("--eval-seeds", type=int, default=48, help="number of fixed fights in the eval set")
    p.add_argument("--eval-envs", type=int, default=2, help="env processes for the eval set")
    # synthetic Bodyguard decision check (harness-free) — print ranking + early-stop when correct
    p.add_argument("--stop-on-bodyguard", action="store_true",
                   help="on eval steps, check the Bodyguard fixture; stop when the ordering is correct")
    p.add_argument("--bodyguard-fixture", default=synthetic_eval.DEFAULT_FIXTURE)
    p.add_argument("--bodyguard-patience", type=int, default=5,
                   help="require the correct Bodyguard ordering this many consecutive evals before stopping")
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
