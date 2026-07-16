"""GPU training loop for the PyTorch policy (scenario / combat mode).

Vectorized rollout (one batched GPU forward per step across all envs) + GPU PPO update. Run:

    python -m lts2_agent.train_torch --character Necrobinder --starter-deck --act 0 \
        --envs 16 --iterations 200

Requires a CUDA PyTorch (falls back to CPU with a warning). The env/feature/reward/scenario code is
shared with the JAX path; only the model + PPO + rollout are PyTorch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

from . import features, model_torch, navigator, ppo_torch, reward, synthetic_eval
from .env import Lts2Env
from .metrics import MetricsWriter
from .rollout_torch import TorchScenarioRollout
from .scenario import ScenarioConfig

_EVAL_COMBAT = {"PlayCard", "EndTurn", "UsePotion", "DiscardPotion"}


@torch.no_grad()
def greedy_winrate(model, envs, scfg, n_seeds, device, tag="EV", greedy=True):
    """Play a FIXED set of fights (same seeds each call, for comparable measurement) with either the
    GREEDY (argmax) policy — the deployed behavior — or the SAMPLED policy the trainer optimizes.
    Returns win rate, EndTurn fraction, avg HP lost, plus a per-fight ``fights`` list (won/hp_lost/
    act/room/character) so callers can emit tagged per-fight metrics."""
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
            character = players[0].get("character") if players else None   # captured at fight start
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
                f = features.encode(st, opts)
                for j, o in enumerate(opts[:features.MAX_OPTIONS]):
                    if o.get("kind") not in _EVAL_COMBAT:
                        f["mask"][j] = False
                args = model_torch.to_tensors({k: f[k][None] for k in features.MODEL_KEYS}, device)
                logits, _ = model(*args)
                a = int(logits[0].argmax()) if greedy else int(model_torch.sample_action(logits)[0][0])
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


@torch.no_grad()
def bodyguard_scores(model: model_torch.ActorCritic, obs: dict, device) -> dict:
    """Per-move model score on the fixed Bodyguard fixture (torch port of synthetic_eval.per_card_scores)."""
    feats = features.encode(obs["state"], obs["options"])
    args = model_torch.to_tensors({k: feats[k][None] for k in features.MODEL_KEYS}, device)
    logits, _ = model(*args)
    logits = logits[0].cpu().numpy()
    best: dict[str, float] = {}
    for i, o in enumerate(obs["options"][:features.MAX_OPTIONS]):
        key = (o.get("card") or {}).get("cardId") if o.get("kind") == "PlayCard" else o.get("kind")
        best[key] = max(best.get(key, -1e18), float(logits[i]))
    return best


def scenario_summary(outcomes: list[dict]) -> dict:
    if not outcomes:
        return {"n": 0, "win_rate": 0.0, "mean_hp_lost": 0.0, "trunc_rate": 0.0}
    n = len(outcomes)
    wins = sum(1 for o in outcomes if o["won"])
    trunc = sum(1 for o in outcomes if o.get("truncated"))
    hp = float(np.mean([o["hp_lost"] for o in outcomes]))
    return {"n": n, "win_rate": wins / n, "mean_hp_lost": hp, "trunc_rate": trunc / n}


def main() -> int:
    ap = argparse.ArgumentParser(description="PyTorch/GPU PPO trainer (scenario mode)")
    ap.add_argument("--character", default=None, help="substring; omit for random character per env")
    ap.add_argument("--starter-deck", action="store_true")
    ap.add_argument("--act", type=int, default=-1, help="0/1/2; -1 = any")
    ap.add_argument("--elite-pct", type=float, default=0.2)
    ap.add_argument("--boss-pct", type=float, default=0.05)
    ap.add_argument("--iterations", type=int, default=200)
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
    ap.add_argument("--step-penalty", type=float, default=0.02,
                    help="per-decision reward penalty. WARNING: >0 rewards ending the turn early "
                         "(fewer decisions = less penalty); truncation already prevents stalling, so 0 is safer.")
    ap.add_argument("--hp-weight", type=float, default=1.0, help="penalty per fraction of start HP lost")
    ap.add_argument("--damage-weight", type=float, default=0.5, help="dense reward per fraction dmg dealt")
    ap.add_argument("--kill-weight", type=float, default=0.2, help="dense reward per enemy killed")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no-static", action="store_true",
                    help="drop the 147-dim per-card static multi-hots (test if they hinder the policy)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="checkpoints/necro_torch.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shaping-ramp-frac", type=float, default=0.75,
                    help="fraction of training over which the dense damage/kill shaping ramps 1->0 "
                         "(0 = shaping off; the HP + anti-stall terms are always on)")
    ap.add_argument("--bodyguard-eval", action="store_true",
                    help="each iter, run the harness-free Bodyguard-before-Unleash fixture check and "
                         "track the consecutive-pass streak (the Necrobinder synergy benchmark)")
    ap.add_argument("--bodyguard-patience", type=int, default=8,
                    help="report the iteration at which the Bodyguard decision is correct this many "
                         "consecutive evals (the benchmark: prior pipeline needed ~100 iters)")
    ap.add_argument("--no-dense", action="store_true",
                    help="disable the dense damage/kill shaping entirely (win/loss + per-step HP + the "
                         "fight-length cap remain, so fights still can't stall)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start training from scratch")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="every N iters, measure the GREEDY (argmax) win rate on fresh fights (the real "
                         "deployed behavior). 0 = off.")
    ap.add_argument("--eval-envs", type=int, default=3)
    ap.add_argument("--eval-fights", type=int, default=8, help="greedy fights per eval env")
    ap.add_argument("--run-dir", default="checkpoints/runs",
                    help="root for per-run metrics dirs (<run-dir>/<run_id>/{manifest.json,events.jsonl})")
    ap.add_argument("--run-label", default=None,
                    help="human label in the run id; default = the checkpoint filename stem")
    ap.add_argument("--no-metrics", action="store_true", help="disable the JSONL metrics event stream")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train_torch] CUDA not available; falling back to CPU (this will be slow).")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[train_torch] device={device} "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    pcfg = ppo_torch.PPOConfig(lr=args.lr, epochs=args.epochs, minibatch_size=args.minibatch,
                               ent_coef=args.ent_coef, logit_reg=args.logit_reg, ent_floor=args.ent_floor)

    # Resume: if the checkpoint exists (and --fresh wasn't passed), continue from it — the model weights,
    # the optimizer state, and the iteration counter — so re-running the same command picks up where it
    # stopped. The model checkpoint (.pt) is what serve/eval load; the optimizer + iteration live in a
    # companion .train file.
    train_state_path = (args.ckpt + ".train") if args.ckpt else None
    start_it = 1
    if args.ckpt and os.path.exists(args.ckpt) and not args.fresh:
        model, _meta = model_torch.load_checkpoint(args.ckpt, device)
        model = model.to(device)
        opt = ppo_torch.make_optimizer(model, pcfg)
        if train_state_path and os.path.exists(train_state_path):
            ts = torch.load(train_state_path, map_location=device)
            try:
                opt.load_state_dict(ts["optimizer"])
            except Exception as e:
                print(f"[train_torch] could not restore optimizer ({e}); continuing with a fresh one.")
            start_it = int(ts.get("iteration", 0)) + 1
        print(f"[train_torch] resumed from {args.ckpt} at iter {start_it}")
    else:
        import numpy as _np
        static = _np.zeros((features.CARD_VOCAB, 0), _np.float32) if args.no_static else None
        model = model_torch.ActorCritic(hidden=args.hidden, static_table=static).to(device)
        opt = ppo_torch.make_optimizer(model, pcfg)
    n_params = sum(p.numel() for p in model.parameters())

    scfg = ScenarioConfig(
        n_envs=args.envs, steps_per_env=args.steps, gamma=args.gamma,
        character=args.character, elite_pct=args.elite_pct, boss_pct=args.boss_pct,
        starter_deck=args.starter_deck, act=(args.act if args.act >= 0 else None),
        weights=reward.ScenarioWeights(hp=args.hp_weight, damage=args.damage_weight,
                                       kill=args.kill_weight, step_penalty=args.step_penalty))

    print(f"[train_torch] envs={args.envs} steps/env={args.steps} "
          f"batch={args.envs * args.steps} minibatch={args.minibatch} params={n_params} lr={args.lr}")

    bg_fixture = synthetic_eval.load_fixture(synthetic_eval.DEFAULT_FIXTURE) if args.bodyguard_eval else None
    bg_streak = 0
    bg_reached: int | None = None

    eval_envs = [Lts2Env() for _ in range(args.eval_envs)] if args.eval_every > 0 else []

    label = args.run_label or (os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt else "run")
    mw = MetricsWriter(run_dir=args.run_dir, label=label, argv=sys.argv, config=vars(args),
                       feature_version=features.FEATURE_VERSION,
                       catalog_signature=features.CATALOG_SIGNATURE, enabled=not args.no_metrics)
    if mw.enabled:
        print(f"[train_torch] metrics -> {mw.run_dir}", flush=True)

    rollout = TorchScenarioRollout(scfg)
    try:
        ramp_end = max(1.0, args.shaping_ramp_frac * args.iterations)
        for it in range(start_it, args.iterations + 1):
            if args.no_dense or args.shaping_ramp_frac <= 0.0:
                shaping_coef = 0.0
            else:
                shaping_coef = max(0.0, 1.0 - (it - 1) / ramp_end)   # 1.0 -> 0.0 over ramp_end iterations
            t0 = time.perf_counter()
            batch, outcomes = rollout.collect(model, device, shaping_coef=shaping_coef)
            t_collect = time.perf_counter() - t0
            metrics = ppo_torch.update(model, opt, batch, pcfg, device)
            t_update = time.perf_counter() - t0 - t_collect

            n = int(batch["action"].shape[0])
            sps = n / (time.perf_counter() - t0)
            s = scenario_summary(outcomes)
            flen = n / max(1, s["n"])   # avg decisions per fight — should stay low with the anti-stall term
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
                g = greedy_winrate(model, eval_envs, scfg, args.eval_fights, device, tag="EVAL", greedy=True)
                sm = greedy_winrate(model, eval_envs, scfg, args.eval_fights, device, tag="EVAL", greedy=False)
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

            if bg_fixture is not None:
                scores = bodyguard_scores(model, bg_fixture, device)
                bg_pass, why = synthetic_eval.bodyguard_pass(scores)
                bg_streak = bg_streak + 1 if bg_pass else 0
                if bg_reached is None and bg_streak >= args.bodyguard_patience:
                    bg_reached = it
                print(f"         BODYGUARD pass={bg_pass} streak={bg_streak}/{args.bodyguard_patience} "
                      f"({why})", flush=True)
                if mw.enabled:
                    mw.emit("eval", it, "bodyguard.pass", 1.0 if bg_pass else 0.0)

            if args.ckpt:
                model_torch.save_checkpoint(args.ckpt, model)
                if train_state_path:
                    torch.save({"optimizer": opt.state_dict(), "iteration": it}, train_state_path)

        if bg_fixture is not None:
            print(f"[train_torch] Bodyguard benchmark: reached {args.bodyguard_patience}-streak at "
                  f"{('iter ' + str(bg_reached - args.bodyguard_patience + 1)) if bg_reached else 'NOT REACHED'} "
                  f"(prior pipeline baseline ~100 iters)", flush=True)
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
