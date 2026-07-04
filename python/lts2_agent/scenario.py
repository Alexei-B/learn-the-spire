"""On-policy rollout collection over isolated **combat scenarios** (the ``reset_combat`` env mode).

Unlike the full-run rollout (:mod:`lts2_agent.rollout`), each episode here is a single fight: reset to
a random character/deck/relics/encounter, let the net decide every combat action, and end when the
combat is over. Reward is terminal (win/loss minus HP lost — :func:`lts2_agent.reward.scenario_reward`).
Mid-combat card choices (discover/scry) are resolved by the scripted navigator and not recorded, so the
net only owns ``PlayCard``/``EndTurn`` — the same combat-only scope as the run trainer. N env processes
run in parallel (one worker thread each); GAE is computed per fight.
"""

from __future__ import annotations

import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import numpy as np

from . import model, navigator, reward
from .env import Lts2Env
from .rollout import featurize, gae


@dataclass
class ScenarioConfig:
    n_envs: int = 4
    steps_per_env: int = 128       # combat decisions collected per env per iteration
    gamma: float = 0.99
    lam: float = 0.95
    character: Optional[str] = None    # None = random character each fight
    elite_pct: float = 0.2
    boss_pct: float = 0.05
    weights: reward.ScenarioWeights = field(default_factory=reward.ScenarioWeights)
    max_navigate: int = 500


class ScenarioRollout:
    """Persistent pool of scenario envs + a thread pool that collects on-policy batches."""

    def __init__(self, config: ScenarioConfig, m: Optional[model.ActorCritic] = None,
                 host_command: Optional[list[str]] = None):
        self.config = config
        self._host_command = host_command
        self._envs = [Lts2Env(host_command=host_command) for _ in range(config.n_envs)]
        self._seed_counters = [itertools.count() for _ in range(config.n_envs)]
        self._cur: list[Optional[dict[str, Any]]] = [None] * config.n_envs
        self._start_hp: list[int] = [1] * config.n_envs
        self._pool = ThreadPoolExecutor(max_workers=config.n_envs)
        self._model = m or model.ActorCritic()
        self._apply = jax.jit(self._model.apply)

    def _new_combat(self, i: int) -> tuple[dict[str, Any], int]:
        cfg = self.config
        obs = self._envs[i].reset_combat(
            seed=f"C{i}-{next(self._seed_counters[i])}",
            character=cfg.character, elite_pct=cfg.elite_pct, boss_pct=cfg.boss_pct)
        start_hp = obs["state"]["players"][0].get("maxHp", 1) if obs["state"].get("players") else 1
        return obs, max(1, start_hp)

    def _recreate_env(self, i: int) -> None:
        try:
            self._envs[i].close()
        except Exception:
            pass
        self._envs[i] = Lts2Env(host_command=self._host_command)

    def _run_worker(self, i: int, params, key, steps: int):
        cfg = self.config
        if self._cur[i] is None:
            self._cur[i], self._start_hp[i] = self._new_combat(i)
        obs, start_hp = self._cur[i], self._start_hp[i]

        gs, ds, cis, masks = [], [], [], []
        actions, logps, values, rewards, dones = [], [], [], [], []
        outcomes: list[dict[str, Any]] = []

        while len(actions) < steps:
            if obs["done"] or not obs["options"]:
                obs, start_hp = self._new_combat(i)
                continue

            phase = obs["state"].get("phase")
            if phase == "Choice":
                # Mid-combat card selection (discover/scry): the scripted navigator resolves it (no
                # transition recorded) so the net only owns the actual combat decisions.
                try:
                    obs = self._envs[i].step(navigator.noncombat_action(obs["state"], obs["options"]))
                except Exception as e:
                    print(f"[scenario] env {i} choice-step failed ({e}); new fight.", file=sys.stderr)
                    self._recreate_env(i)
                    obs, start_hp = self._new_combat(i)
                continue
            if phase != "Combat":
                # Any other non-done, non-combat state: the fight is effectively over (a post-combat
                # screen that lingered) — abandon it and start a fresh one rather than acting on it.
                obs, start_hp = self._new_combat(i)
                continue

            g, dense, card_idx, mask = featurize(obs["state"], obs["options"])
            logits, value = self._apply(params, g[None], dense[None], card_idx[None], mask[None])
            key, sub = jax.random.split(key)
            action, logp = model.sample_action(logits, sub)
            a = int(action[0])

            try:
                nxt = self._envs[i].step(a)
            except Exception as e:
                print(f"[scenario] env {i} step failed ({e}); new fight.", file=sys.stderr)
                self._recreate_env(i)
                obs, start_hp = self._new_combat(i)
                continue

            done = bool(nxt["done"])
            r = reward.scenario_reward(nxt, start_hp, cfg.weights) if done else 0.0

            gs.append(np.asarray(g)); ds.append(np.asarray(dense))
            cis.append(np.asarray(card_idx)); masks.append(np.asarray(mask))
            actions.append(a); logps.append(float(logp[0])); values.append(float(value[0]))
            rewards.append(r); dones.append(1.0 if done else 0.0)

            if done:
                info = nxt["info"]
                outcomes.append({"won": bool(info.get("won")), "hp_lost": info.get("hpLost") or 0,
                                 "room": info.get("roomType"), "act": info.get("act")})
                obs, start_hp = self._new_combat(i)
            else:
                obs = nxt

        self._cur[i], self._start_hp[i] = obs, start_hp
        if dones[-1] > 0.5:
            last_value = 0.0
        else:
            g, dense, card_idx, mask = featurize(obs["state"], obs["options"])
            _, v = self._apply(params, g[None], dense[None], card_idx[None], mask[None])
            last_value = float(v[0])

        adv, returns = gae(rewards, values, dones, last_value, cfg.gamma, cfg.lam)
        return {
            "g": np.stack(gs), "dense": np.stack(ds), "card_idx": np.stack(cis), "mask": np.stack(masks),
            "action": np.asarray(actions, np.int32), "logp": np.asarray(logps, np.float32),
            "value": np.asarray(values, np.float32), "adv": adv, "ret": returns, "outcomes": outcomes,
        }

    def collect(self, params, key: jax.Array):
        keys = jax.random.split(key, self.config.n_envs)
        futures = [self._pool.submit(self._run_worker, i, params, keys[i], self.config.steps_per_env)
                   for i in range(self.config.n_envs)]
        results = [f.result() for f in futures]
        batch = {k: np.concatenate([r[k] for r in results], axis=0)
                 for k in ("g", "dense", "card_idx", "mask", "action", "logp", "value", "adv", "ret")}
        outcomes = [o for r in results for o in r["outcomes"]]
        return batch, outcomes

    def close(self):
        self._pool.shutdown(wait=False)
        for env in self._envs:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
