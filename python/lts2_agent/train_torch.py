"""GPU training loop for the PyTorch policy (scenario / combat mode).

Vectorized rollout (one batched GPU forward per step across all envs) + GPU PPO update. Run:

    python -m lts2_agent.train_torch --character Necrobinder --starter-deck --act 0 \
        --envs 16 --iterations 200

Requires a CUDA PyTorch (falls back to CPU with a warning). The env/feature/reward/scenario code is
shared with the JAX path; only the model + PPO + rollout are PyTorch.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import features, model_torch, ppo_torch, reward
from .rollout_torch import TorchScenarioRollout
from .scenario import ScenarioConfig


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
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="checkpoints/necro_torch.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shaping-ramp-frac", type=float, default=0.75,
                    help="fraction of training over which the dense damage/kill shaping ramps 1->0 "
                         "(0 = shaping off; the HP + anti-stall terms are always on)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train_torch] CUDA not available; falling back to CPU (this will be slow).")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[train_torch] device={device} "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    model = model_torch.ActorCritic(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    pcfg = ppo_torch.PPOConfig(lr=args.lr, epochs=args.epochs, minibatch_size=args.minibatch)
    opt = ppo_torch.make_optimizer(model, pcfg)

    scfg = ScenarioConfig(
        n_envs=args.envs, steps_per_env=args.steps,
        character=args.character, elite_pct=args.elite_pct, boss_pct=args.boss_pct,
        starter_deck=args.starter_deck, act=(args.act if args.act >= 0 else None),
        weights=reward.ScenarioWeights())

    print(f"[train_torch] envs={args.envs} steps/env={args.steps} "
          f"batch={args.envs * args.steps} minibatch={args.minibatch} params={n_params} lr={args.lr}")

    rollout = TorchScenarioRollout(scfg)
    try:
        ramp_end = max(1.0, args.shaping_ramp_frac * args.iterations)
        for it in range(1, args.iterations + 1):
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
                  f"kl={metrics['approx_kl']:.4f} ent={metrics['entropy']:.3f} "
                  f"maxRatio={metrics['max_ratio']:.1f} | "
                  f"collect={t_collect:.2f}s update={t_update:.2f}s sps={sps:.0f}", flush=True)

            if args.ckpt:
                model_torch.save_checkpoint(args.ckpt, model)
    finally:
        rollout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
