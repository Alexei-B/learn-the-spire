"""A minimal training-loop skeleton against :class:`lts2_agent.env.Lts2Env`.

This is deliberately *not* a real RL algorithm — it shows the shape of an episode loop, where a
policy chooses actions and where you would compute a reward and update a learner. Here the "learner"
is the fixed heuristic policy and the reward is the per-step change in the game's score (swap in
whatever signal your algorithm needs — the raw scalars are in ``obs['info']``).

Run it (after building Lts2.AgentHost) with::

    python -m lts2_agent.examples.train_stub --episodes 2 --seed DEMO
"""

from __future__ import annotations

import argparse
from typing import Any

from ..env import Lts2Env
from ..policies import heuristic


def choose_action(state: dict[str, Any], options: list[dict[str, Any]]) -> int:
    """Pick an action index from the policy's ranking, falling back to option 0."""
    ranking = heuristic.policy(state, options)
    if ranking:
        return max(ranking, key=lambda pair: pair[1])[0]
    # Out of combat the heuristic declines; take the first legal option to keep the run moving.
    return 0


def run_episode(env: Lts2Env, seed: str, max_steps: int = 2000) -> tuple[float, int]:
    """Play one run to termination (or ``max_steps``). Returns (total_reward, final_score)."""
    obs = env.reset(seed=seed)
    prev_score = obs["info"]["score"]
    total_reward = 0.0

    for _ in range(max_steps):
        if obs["done"] or not obs["options"]:
            break
        action = choose_action(obs["state"], obs["options"])
        obs = env.step(action)

        # Reward shaping lives here — this example uses the score delta; substitute your own.
        score = obs["info"]["score"]
        reward = score - prev_score
        prev_score = score
        total_reward += reward

        # <-- a real trainer would store (obs, action, reward) and update its model here.

    return total_reward, obs["info"]["score"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lts2 training-loop skeleton.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", default="DEMO")
    parser.add_argument("--character", default=None)
    args = parser.parse_args(argv)

    with Lts2Env(character=args.character) as env:
        for ep in range(args.episodes):
            total_reward, final_score = run_episode(env, seed=f"{args.seed}{ep}")
            print(f"episode {ep}: total_reward={total_reward:.0f} final_score={final_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
