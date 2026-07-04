"""Evaluate combat policies over a fixed set of seeds and compare them.

Each policy decides combat; the scripted :mod:`lts2_agent.navigator` handles every non-combat phase, so
the comparison isolates *combat* quality. Reports win rate, mean/median floor reached, mean final score,
and mean combats survived — for the trained PPO checkpoint against the reference heuristic and a random
baseline (a learned policy should beat random and approach/exceed the heuristic).

Run::

    python -m lts2_agent.eval --policies ppo,heuristic,random --ckpt checkpoints/ppo --seeds 20
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from typing import Any, Callable, Optional

from . import features, navigator
from .env import Lts2Env
from .policies import heuristic

CombatPolicy = Callable[[dict[str, Any], list[dict[str, Any]]], list]


def _argmax_index(ranking: list) -> Optional[int]:
    if not ranking:
        return None
    best_i, best_s = None, None
    for entry in ranking:
        i, s = (entry["index"], entry.get("score", 0.0)) if isinstance(entry, dict) else entry
        if best_s is None or s > best_s:
            best_i, best_s = i, s
    return best_i


def random_policy(seed: int = 0) -> CombatPolicy:
    rng = random.Random(seed)
    def policy(state, options):
        return [(i, rng.random()) for i in range(len(options))]
    return policy


def build_policy(name: str, ckpt: Optional[str]) -> CombatPolicy:
    if name == "heuristic":
        return heuristic.policy
    if name == "random":
        return random_policy()
    if name == "ppo":
        if not ckpt:
            raise SystemExit("--ckpt is required for the 'ppo' policy")
        from .policies import jax_policy
        return jax_policy.make_policy(ckpt)
    raise SystemExit(f"unknown policy '{name}'")


def play_run(env: Lts2Env, seed: str, combat_policy: CombatPolicy, max_steps: int = 6000) -> dict:
    obs = env.reset(seed=seed)
    combats = 0
    was_combat = False
    steps = 0
    while not obs["done"] and obs["options"] and steps < max_steps:
        state, options = obs["state"], obs["options"]
        in_combat = features.is_combat(state)
        if in_combat and not was_combat:
            combats += 1
        was_combat = in_combat

        if in_combat:
            action = _argmax_index(combat_policy(state, options))
            if action is None:
                action = navigator.combat_action(state, options)
        else:
            action = navigator.noncombat_action(state, options)
        obs = env.step(action)
        steps += 1

    info = obs["info"]
    return {"floor": info["floor"], "act": info["act"], "victory": bool(info["victory"]),
            "score": info["score"], "combats": combats, "hp": info["players"][0]["currentHp"]}


def evaluate(name: str, policy: CombatPolicy, seeds: list[str], character: Optional[str]) -> dict:
    results = []
    with Lts2Env(character=character) as env:
        for s in seeds:
            results.append(play_run(env, s, policy))
    floors = [r["floor"] for r in results]
    return {
        "policy": name,
        "runs": len(results),
        "win_rate": statistics.mean(r["victory"] for r in results),
        "mean_floor": statistics.mean(floors),
        "median_floor": statistics.median(floors),
        "max_floor": max(floors),
        "mean_score": statistics.mean(r["score"] for r in results),
        "mean_combats": statistics.mean(r["combats"] for r in results),
    }


def play_fight(env: Lts2Env, seed: str, combat_policy: CombatPolicy,
               elite_pct: float, boss_pct: float, max_steps: int = 2000) -> Optional[dict]:
    """Play one isolated combat scenario; None if it hit a harness edge (abandoned)."""
    try:
        obs = env.reset_combat(seed=seed, elite_pct=elite_pct, boss_pct=boss_pct)
        while not obs["done"] and obs["options"]:
            phase = obs["state"].get("phase")
            if phase == "Choice":
                obs = env.step(navigator.noncombat_action(obs["state"], obs["options"]))
                continue
            if phase != "Combat":
                break
            action = _argmax_index(combat_policy(obs["state"], obs["options"]))
            if action is None:
                action = navigator.combat_action(obs["state"], obs["options"])
            obs = env.step(action)
    except RuntimeError:
        return None
    info = obs["info"]
    return {"won": bool(info.get("won")), "hp_lost": info.get("hpLost") or 0, "room": info.get("roomType")}


def evaluate_scenarios(name: str, policy: CombatPolicy, seeds: list[str],
                       elite_pct: float, boss_pct: float) -> dict:
    results = []
    with Lts2Env() as env:
        for s in seeds:
            r = play_fight(env, s, policy, elite_pct, boss_pct)
            if r is not None:
                results.append(r)
    n = max(1, len(results))
    return {
        "policy": name,
        "fights": len(results),
        "win_rate": statistics.mean(r["won"] for r in results) if results else float("nan"),
        "mean_hp_lost": statistics.mean(r["hp_lost"] for r in results) if results else float("nan"),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate/compare STS2 combat policies.")
    p.add_argument("--mode", choices=["run", "scenario"], default="run",
                   help="'run' = full playthroughs; 'scenario' = isolated random combats")
    p.add_argument("--policies", default="ppo,heuristic,random",
                   help="comma list from: ppo, heuristic, random")
    p.add_argument("--ckpt", default="checkpoints/ppo")
    p.add_argument("--seeds", type=int, default=20, help="number of seeded runs/fights per policy")
    p.add_argument("--seed-prefix", default="EVAL")
    p.add_argument("--character", default="Ironclad")
    p.add_argument("--elite-pct", type=float, default=0.2)
    p.add_argument("--boss-pct", type=float, default=0.05)
    args = p.parse_args(argv)

    seeds = [f"{args.seed_prefix}{i}" for i in range(args.seeds)]
    for name in [n.strip() for n in args.policies.split(",") if n.strip()]:
        policy = build_policy(name, args.ckpt)
        if args.mode == "scenario":
            row = evaluate_scenarios(name, policy, seeds, args.elite_pct, args.boss_pct)
            print(f"[{name:9s}] fights={row['fights']} win={row['win_rate']:.2f} "
                  f"hpLost~{row['mean_hp_lost']:.1f}", file=sys.stderr, flush=True)
        else:
            row = evaluate(name, policy, seeds, args.character)
            print(f"[{name:9s}] runs={row['runs']} win={row['win_rate']:.2f} "
                  f"floor mean={row['mean_floor']:.1f} median={row['median_floor']:.0f} "
                  f"max={row['max_floor']} score={row['mean_score']:.0f} "
                  f"combats={row['mean_combats']:.1f}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
