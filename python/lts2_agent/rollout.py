"""On-policy rollout collection over N parallel environment processes.

The learned net decides only combat, so each env is advanced through non-combat with the scripted
:mod:`lts2_agent.navigator`; a transition is recorded only at a combat decision. Because one run lives
per process (game singletons), we run **N ``Lts2Env`` subprocesses**, one per worker thread — while a
worker blocks on its child's ``step`` (the C# does the compute), the others run, so threads give real
parallelism here. Each worker collects its own trajectory and computes GAE locally (respecting episode
boundaries); the batches are concatenated for the PPO update.

``Rollout`` owns the persistent envs and each env's current observation, so successive ``collect``
calls continue the same runs (resetting only when an episode ends) — standard on-policy behavior.
"""

from __future__ import annotations

import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import numpy as np

from . import features, model, navigator, reward
from .env import Lts2Env


@dataclass
class RolloutConfig:
    n_envs: int = 4
    steps_per_env: int = 128           # combat decisions collected per env per iteration
    gamma: float = 0.99
    lam: float = 0.95
    character: Optional[str] = "Ironclad"
    ascension: int = 0
    weights: reward.RewardWeights = field(default_factory=reward.RewardWeights)
    max_navigate: int = 10000          # safety cap while advancing through non-combat


def featurize(state: dict[str, Any], options: list[dict[str, Any]]):
    """Observation dict -> the model-input feature dict (keyed by ``features.MODEL_KEYS``, unbatched)."""
    return features.encode(state, options)


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized advantage estimation over one trajectory (dones mark episode resets)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


class Rollout:
    """Persistent pool of envs + a thread pool that collects on-policy batches."""

    def __init__(self, config: RolloutConfig, m: Optional[model.ActorCritic] = None,
                 host_command: Optional[list[str]] = None):
        self.config = config
        self._host_command = host_command
        self._envs = [
            Lts2Env(host_command=host_command, character=config.character, ascension=config.ascension)
            for _ in range(config.n_envs)
        ]
        self._seed_counters = [itertools.count() for _ in range(config.n_envs)]
        self._cur: list[Optional[dict[str, Any]]] = [None] * config.n_envs
        self._pool = ThreadPoolExecutor(max_workers=config.n_envs)
        self._model = m or model.ActorCritic()
        # A compiled, thread-safe forward (fixed batch=1 shape → compiled once, reused across threads).
        self._apply = jax.jit(self._model.apply)

    # --- env stepping helpers ---

    def _next_seed(self, i: int) -> str:
        return f"E{i}-{next(self._seed_counters[i])}"

    def _recreate_env(self, i: int) -> None:
        """Replace a broken env process (a game-logic error can leave a run unusable)."""
        try:
            self._envs[i].close()
        except Exception:
            pass
        self._envs[i] = Lts2Env(host_command=self._host_command,
                                character=self.config.character, ascension=self.config.ascension)

    def _reset_to_combat(self, i: int) -> dict[str, Any]:
        # A fresh run occasionally hits a game-logic error; recreate the env and retry a few times.
        for attempt in range(3):
            try:
                obs = self._envs[i].reset(seed=self._next_seed(i))
                return self._advance_to_combat(i, obs)
            except Exception as e:
                print(f"[rollout] env {i} reset failed ({e}); recreating.", file=sys.stderr)
                self._recreate_env(i)
        raise RuntimeError(f"env {i} could not reset after retries")

    def _advance_to_combat(self, i: int, obs: dict[str, Any]) -> dict[str, Any]:
        """Navigate non-combat phases until a combat decision or terminal."""
        env = self._envs[i]
        for _ in range(self.config.max_navigate):
            if obs["done"] or not obs["options"] or features.is_combat(obs["state"]):
                return obs
            obs = env.step(navigator.noncombat_action(obs["state"], obs["options"]))
        return obs

    # --- one worker's trajectory ---

    def _run_worker(self, i: int, params, key, steps: int):
        cfg = self.config
        if self._cur[i] is None:
            self._cur[i] = self._reset_to_combat(i)
        obs = self._cur[i]

        feats_buf: dict[str, list] = {k: [] for k in features.MODEL_KEYS}
        actions, logps, values, rewards, dones = [], [], [], [], []
        episodes: list[dict[str, Any]] = []

        while len(actions) < steps:
            if obs["done"] or not obs["options"]:
                obs = self._reset_to_combat(i)
                continue

            feats = featurize(obs["state"], obs["options"])
            logits, value = model.forward1(self._apply, params, feats)
            key, sub = jax.random.split(key)
            action, logp = model.sample_action(logits, sub)
            a = int(action[0])

            decision = obs
            try:
                nxt = self._envs[i].step(a)
                nxt = self._advance_to_combat(i, nxt)
            except Exception as e:
                # A game-logic error mid-run (e.g. a rare NRE on some option): abandon this run and
                # drop the in-flight transition, then continue collecting from a fresh run.
                print(f"[rollout] env {i} step failed ({e}); resetting.", file=sys.stderr)
                self._recreate_env(i)
                obs = self._reset_to_combat(i)
                continue
            r = reward.compute(decision, nxt, cfg.weights)
            done = bool(nxt["done"] or not nxt["options"])

            for k in features.MODEL_KEYS:
                feats_buf[k].append(feats[k])
            actions.append(a); logps.append(float(logp[0])); values.append(float(value[0]))
            rewards.append(r); dones.append(1.0 if done else 0.0)

            if done:
                info = nxt["info"]
                episodes.append({"floor": info["floor"], "act": info["act"],
                                 "victory": bool(info["victory"]), "score": info["score"]})
                obs = self._reset_to_combat(i)
            else:
                obs = nxt

        self._cur[i] = obs
        # Bootstrap value for the (non-terminal) tail.
        if dones[-1] > 0.5:
            last_value = 0.0
        else:
            _, v = model.forward1(self._apply, params, featurize(obs["state"], obs["options"]))
            last_value = float(v[0])

        adv, returns = gae(rewards, values, dones, last_value, cfg.gamma, cfg.lam)
        batch = {k: np.stack(feats_buf[k]) for k in features.MODEL_KEYS}
        batch.update({
            "action": np.asarray(actions, np.int32), "logp": np.asarray(logps, np.float32),
            "value": np.asarray(values, np.float32), "adv": adv, "ret": returns,
            "episodes": episodes,
        })
        return batch

    def collect(self, params, key: jax.Array):
        """Collect one on-policy batch across all envs. Returns (batch_dict, episode_summaries)."""
        keys = jax.random.split(key, self.config.n_envs)
        futures = [
            self._pool.submit(self._run_worker, i, params, keys[i], self.config.steps_per_env)
            for i in range(self.config.n_envs)
        ]
        results = [f.result() for f in futures]

        batch = {
            k: np.concatenate([r[k] for r in results], axis=0)
            for k in (*features.MODEL_KEYS, "action", "logp", "value", "adv", "ret")
        }
        episodes = [e for r in results for e in r["episodes"]]
        return batch, episodes

    def close(self):
        self._pool.shutdown(wait=False)
        for env in self._envs:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
