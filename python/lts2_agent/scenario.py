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

from . import features, model, navigator, reward
from .env import Lts2Env
from .rollout import featurize, gae

# The options the net is allowed to pick during a fight. The harness can rarely offer a reward option
# mixed in with combat moves at the exact end of a fight; masking to these keeps the exploring policy
# from stepping a post-combat option (which errors), without depending on that quirk not happening.
_COMBAT_KINDS = {"PlayCard", "EndTurn", "UsePotion", "DiscardPotion"}


@dataclass
class ScenarioConfig:
    n_envs: int = 4
    steps_per_env: int = 128       # combat decisions collected per env per iteration
    gamma: float = 0.99
    lam: float = 0.95
    character: Optional[str] = None    # None = random character each fight
    elite_pct: float = 0.2
    boss_pct: float = 0.05
    starter_deck: bool = False         # use the character's fixed starting deck (focused, low-noise)
    act: Optional[int] = None          # restrict encounters to this act (0/1/2)
    weights: reward.ScenarioWeights = field(default_factory=reward.ScenarioWeights)
    max_navigate: int = 500
    # Hard cap on decisions per fight: if the policy hasn't won by here the fight is truncated and scored
    # as a loss, so stalling (chip damage + defend forever) can never beat winning. A real fight is well
    # under this; it only bites a degenerate non-terminating policy.
    max_fight_len: int = 60


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
            character=cfg.character, elite_pct=cfg.elite_pct, boss_pct=cfg.boss_pct,
            starter_deck=cfg.starter_deck, act=cfg.act)
        start_hp = obs["state"]["players"][0].get("maxHp", 1) if obs["state"].get("players") else 1
        return obs, max(1, start_hp)

    def _recreate_env(self, i: int) -> None:
        try:
            self._envs[i].close()
        except Exception:
            pass
        self._envs[i] = Lts2Env(host_command=self._host_command)

    def _run_worker(self, i: int, params, key, steps: int):
        """One env's on-policy trajectory (runs in its own thread). Thread-per-env overlaps the expensive
        env I/O — a fight's ~160&#160;ms reset on one env proceeds while the others keep stepping — which
        beats a step-synchronous barrier when resets are costly. Throughput then scales with env
        processes (we have far more cores than the default env count)."""
        cfg = self.config
        if self._cur[i] is None:
            self._cur[i], self._start_hp[i] = self._new_combat(i)
        obs, start_hp = self._cur[i], self._start_hp[i]

        feats_buf: dict[str, list] = {k: [] for k in features.MODEL_KEYS}
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
                obs, start_hp = self._new_combat(i)
                continue

            feats = featurize(obs["state"], obs["options"])
            for j, o in enumerate(obs["options"][:features.MAX_OPTIONS]):
                if o.get("kind") not in _COMBAT_KINDS:
                    feats["mask"][j] = False
            if not feats["mask"].any():
                obs, start_hp = self._new_combat(i)
                continue
            logits, value = model.forward1(self._apply, params, feats)
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

            for k in features.MODEL_KEYS:
                feats_buf[k].append(feats[k])
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
            _, v = model.forward1(self._apply, params, featurize(obs["state"], obs["options"]))
            last_value = float(v[0])

        adv, returns = gae(rewards, values, dones, last_value, cfg.gamma, cfg.lam)
        batch = {k: np.stack(feats_buf[k]) for k in features.MODEL_KEYS}
        batch.update({
            "action": np.asarray(actions, np.int32), "logp": np.asarray(logps, np.float32),
            "value": np.asarray(values, np.float32), "adv": adv, "ret": returns, "outcomes": outcomes,
        })
        return batch

    def collect(self, params, key: jax.Array):
        keys = jax.random.split(key, self.config.n_envs)
        futures = [self._pool.submit(self._run_worker, i, params, keys[i], self.config.steps_per_env)
                   for i in range(self.config.n_envs)]
        results = [f.result() for f in futures]
        batch = {k: np.concatenate([r[k] for r in results], axis=0)
                 for k in (*features.MODEL_KEYS, "action", "logp", "value", "adv", "ret")}
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


class ScenarioEvalSet:
    """A **fixed set of seeded fights, played greedily** — a deterministic held-out metric.

    Random-fight training win-rate is too noisy to see progress (different fights each iteration, plus
    exploration, plus some fights are unwinnable). Here the same seeds produce the same fights every
    time, and greedy (argmax) actions make each fight's outcome a deterministic function of the params —
    so the win-rate and (more sensitively) the mean HP-lost fraction move only when the policy actually
    changes. Call :meth:`evaluate` periodically during training to tell real improvement from noise.
    """

    def __init__(self, seeds, m: Optional[model.ActorCritic] = None, n_envs: int = 2,
                 elite_pct: float = 0.2, boss_pct: float = 0.05, character: Optional[str] = None,
                 starter_deck: bool = False, act: Optional[int] = None,
                 host_command: Optional[list[str]] = None):
        self.seeds = list(seeds)
        self.elite_pct, self.boss_pct, self.character = elite_pct, boss_pct, character
        self.starter_deck, self.act = starter_deck, act
        self._model = m or model.ActorCritic()
        self._apply = jax.jit(self._model.apply)
        self._envs = [Lts2Env(host_command=host_command) for _ in range(n_envs)]
        self._pool = ThreadPoolExecutor(max_workers=n_envs)

    def _play(self, env: Lts2Env, params, seed: str) -> Optional[dict[str, Any]]:
        try:
            obs = env.reset_combat(seed=seed, character=self.character,
                                   elite_pct=self.elite_pct, boss_pct=self.boss_pct,
                                   starter_deck=self.starter_deck, act=self.act)
            start_hp = max(1, obs["state"]["players"][0].get("maxHp", 1)) if obs["state"].get("players") else 1
            for _ in range(2000):
                if obs["done"] or not obs["options"]:
                    break
                phase = obs["state"].get("phase")
                if phase == "Choice":
                    obs = env.step(navigator.noncombat_action(obs["state"], obs["options"]))
                    continue
                if phase != "Combat":
                    break
                feats = featurize(obs["state"], obs["options"])
                for j, o in enumerate(obs["options"][:features.MAX_OPTIONS]):
                    if o.get("kind") not in _COMBAT_KINDS:
                        feats["mask"][j] = False
                if not feats["mask"].any():
                    break
                logits, _ = model.forward1(self._apply, params, feats)
                obs = env.step(int(np.argmax(np.asarray(logits[0]))))   # greedy
        except RuntimeError:
            return None   # abandoned (e.g. an enemy-turn timeout) — excluded from the metric
        info = obs["info"]
        hp_lost = info.get("hpLost") or 0
        return {"won": bool(info.get("won")), "hp_lost": hp_lost, "hp_frac": hp_lost / start_hp}

    def _play_many(self, env, params, seeds):
        return [self._play(env, params, s) for s in seeds]

    def evaluate(self, params) -> dict[str, float]:
        chunks = [self.seeds[i::len(self._envs)] for i in range(len(self._envs))]
        futs = [self._pool.submit(self._play_many, self._envs[i], params, chunks[i])
                for i in range(len(self._envs))]
        results = [r for f in futs for r in f.result() if r is not None]
        n = max(1, len(results))
        return {
            "n": len(results),
            "win_rate": sum(r["won"] for r in results) / n,
            "mean_hp_lost": sum(r["hp_lost"] for r in results) / n,
            "mean_hp_frac": sum(r["hp_frac"] for r in results) / n,
        }

    def close(self):
        self._pool.shutdown(wait=False)
        for env in self._envs:
            env.close()
