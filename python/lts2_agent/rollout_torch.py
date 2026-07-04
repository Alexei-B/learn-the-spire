"""Vectorized on-policy rollout for the PyTorch policy.

Every outer step advances *all* envs to a combat decision (in parallel threads — the C# env I/O
releases the GIL), runs ONE batched GPU forward across all envs, then steps all envs (in parallel).
This replaces the old thread-per-env batch-1 forward (which serialized on the GIL): a single GPU
dispatch per N transitions instead of N CPU dispatches per step. It only pays off with cheap resets —
the full reset is now ~16ms, so the step-synchronous barrier no longer stalls on a resetting env.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import torch

from . import features, model_torch, navigator, reward
from .env import Lts2Env
from .rollout import featurize, gae
from .scenario import ScenarioConfig, _COMBAT_KINDS


class TorchScenarioRollout:
    """Persistent pool of scenario envs; collects on-policy batches with a batched GPU forward."""

    def __init__(self, config: ScenarioConfig, host_command: Optional[list[str]] = None):
        self.config = config
        self._host_command = host_command
        # LTS2_ENV_STDERR surfaces the C# host's stderr (incl. the LTS2_PROFILE per-step breakdown).
        log_stderr = bool(os.environ.get("LTS2_ENV_STDERR"))
        self._envs = [Lts2Env(host_command=host_command, log_stderr=log_stderr)
                      for _ in range(config.n_envs)]
        self._seed_counters = [itertools.count() for _ in range(config.n_envs)]
        self._cur: list[Optional[dict[str, Any]]] = [None] * config.n_envs
        self._start_hp: list[int] = [1] * config.n_envs
        self._fight_steps: list[int] = [0] * config.n_envs   # decisions taken in the current fight
        self._pool = ThreadPoolExecutor(max_workers=config.n_envs)

    # --- env management (mirrors scenario.ScenarioRollout) -----------------------------------------

    def _new_combat(self, i: int) -> tuple[dict[str, Any], int]:
        cfg = self.config
        # The host stays alive after a *caught* combat error (timeout / stale-action residue), so a fresh
        # reset_combat recovers in ~16ms. Try that first; only recreate the whole process (seconds) as a
        # last resort if the reset itself keeps failing (the host actually died).
        last_err: Exception | None = None
        for attempt in range(3):
            if attempt == 2:
                self._recreate_env(i)
            try:
                obs = self._envs[i].reset_combat(
                    seed=f"C{i}-{next(self._seed_counters[i])}",
                    character=cfg.character, elite_pct=cfg.elite_pct, boss_pct=cfg.boss_pct,
                    starter_deck=cfg.starter_deck, act=cfg.act)
                start_hp = obs["state"]["players"][0].get("maxHp", 1) if obs["state"].get("players") else 1
                self._fight_steps[i] = 0
                return obs, max(1, start_hp)
            except Exception as e:
                last_err = e
                print(f"[torch-rollout] env {i} reset attempt {attempt} failed ({e}).", file=sys.stderr)
        raise RuntimeError(f"env {i} could not reset after retries: {last_err}")

    def _recreate_env(self, i: int) -> None:
        try:
            self._envs[i].close()
        except Exception:
            pass
        self._envs[i] = Lts2Env(host_command=self._host_command)

    def _advance_env(self, i: int) -> None:
        """Advance env ``i`` until it sits on a live combat decision (reset finished/lingering fights,
        let the navigator resolve a mid-combat card choice). I/O only — safe to run in parallel."""
        if self._cur[i] is None:
            self._cur[i], self._start_hp[i] = self._new_combat(i)
        obs = self._cur[i]
        for _ in range(self.config.max_navigate):
            if obs["done"] or not obs["options"]:
                obs, self._start_hp[i] = self._new_combat(i)
                continue
            phase = obs["state"].get("phase")
            if phase == "Choice":
                try:
                    obs = self._envs[i].step(navigator.noncombat_action(obs["state"], obs["options"]))
                except Exception as e:
                    print(f"[torch-rollout] env {i} choice-step failed ({e}); new fight.", file=sys.stderr)
                    obs, self._start_hp[i] = self._new_combat(i)   # cheap reset (host survives caught errors)
                continue
            if phase != "Combat":
                obs, self._start_hp[i] = self._new_combat(i)
                continue
            break
        self._cur[i] = obs

    def _step_env(self, i: int, action: int):
        try:
            return self._envs[i].step(action)
        except Exception as e:
            # The host caught this and is still alive; drop the fight and let the next _advance_env do a
            # cheap reset on the same process rather than restarting it.
            print(f"[torch-rollout] env {i} step failed ({e}); new fight.", file=sys.stderr)
            self._cur[i] = None
            return None

    # --- collection --------------------------------------------------------------------------------

    @torch.no_grad()
    def collect(self, model: model_torch.ActorCritic, device, shaping_coef: float = 1.0):
        cfg = self.config
        N = cfg.n_envs
        model.eval()

        feats_buf = [{k: [] for k in features.MODEL_KEYS} for _ in range(N)]
        actions_b = [[] for _ in range(N)]
        logps_b = [[] for _ in range(N)]
        values_b = [[] for _ in range(N)]
        rewards_b = [[] for _ in range(N)]
        dones_b = [[] for _ in range(N)]
        outcomes: list[dict[str, Any]] = []

        # Opt-in per-phase timing (LTS2_TIMING) to see where collect goes: env I/O (advance/step),
        # featurize (Python, GIL), host->device transfer, GPU forward, and the record loop.
        prof = bool(os.environ.get("LTS2_TIMING"))
        tt = {"advance": 0.0, "featurize": 0.0, "transfer": 0.0, "forward": 0.0, "step": 0.0, "record": 0.0}
        obs_bytes = 0

        for _ in range(cfg.steps_per_env):
            t = time.perf_counter()
            list(self._pool.map(self._advance_env, range(N)))   # parallel I/O: all at a decision
            if prof: tt["advance"] += time.perf_counter() - t

            t = time.perf_counter()
            feats_list = []
            for i in range(N):
                f = featurize(self._cur[i]["state"], self._cur[i]["options"])
                for j, o in enumerate(self._cur[i]["options"][:features.MAX_OPTIONS]):
                    if o.get("kind") not in _COMBAT_KINDS:
                        f["mask"][j] = False
                feats_list.append(f)
            stacked = model_torch.stack_feats(feats_list)
            if prof: tt["featurize"] += time.perf_counter() - t

            t = time.perf_counter()
            batch_feats = model_torch.to_tensors(stacked, device)
            if prof: tt["transfer"] += time.perf_counter() - t

            t = time.perf_counter()
            logits, values = model(*batch_feats)                 # ONE batched GPU forward [N, M], [N]
            actions, logps = model_torch.sample_action(logits)
            a = actions.cpu().numpy(); lp = logps.cpu().numpy(); v = values.cpu().numpy()
            if prof: tt["forward"] += time.perf_counter() - t

            t = time.perf_counter()
            starts = list(self._start_hp)
            prevs = list(self._cur)   # pre-step observations, for the dense per-step reward
            nxts = list(self._pool.map(lambda i: self._step_env(i, int(a[i])), range(N)))
            if prof: tt["step"] += time.perf_counter() - t

            t = time.perf_counter()
            for i in range(N):
                nxt = nxts[i]
                if nxt is None:
                    continue
                if prof:
                    obs_bytes += nxt.get("_bytes", 0)
                self._fight_steps[i] += 1
                prev = prevs[i] if prevs[i] is not None else nxt
                r = reward.scenario_dense_reward(prev, nxt, starts[i], cfg.weights, shaping_coef)

                # Truncate a fight that runs past the cap and score it as a loss, so stalling can't beat
                # winning. Terminates the episode here (with a done flag for GAE) and starts a fresh fight.
                truncated = not nxt["done"] and self._fight_steps[i] >= cfg.max_fight_len
                if truncated:
                    r -= cfg.weights.truncate_penalty
                done = bool(nxt["done"]) or truncated

                for k in features.MODEL_KEYS:
                    feats_buf[i][k].append(feats_list[i][k])
                actions_b[i].append(int(a[i])); logps_b[i].append(float(lp[i]))
                values_b[i].append(float(v[i])); rewards_b[i].append(r)
                dones_b[i].append(1.0 if done else 0.0)
                if done:
                    info = nxt["info"]
                    won = bool(info.get("won")) and not truncated
                    outcomes.append({"won": won, "hp_lost": info.get("hpLost") or 0,
                                     "room": info.get("roomType"), "act": info.get("act"),
                                     "truncated": truncated, "flen": self._fight_steps[i]})
                    self._cur[i] = None
                else:
                    self._cur[i] = nxt
            if prof: tt["record"] += time.perf_counter() - t

        if prof:
            total = sum(tt.values()) or 1.0
            steps = cfg.steps_per_env * N
            print("[collect-timing] "
                  + "  ".join(f"{k}={v:.2f}s({100 * v / total:.0f}%)" for k, v in tt.items())
                  + f"  total={total:.2f}s  obs~{obs_bytes // max(1, steps)}B/step", file=sys.stderr, flush=True)

        # Bootstrap non-terminal tails with one more batched forward, then GAE per env.
        list(self._pool.map(self._advance_env, range(N)))
        tail_feats = [featurize(self._cur[i]["state"], self._cur[i]["options"]) for i in range(N)]
        _, tail_v = model(*model_torch.to_tensors(model_torch.stack_feats(tail_feats), device))
        tail_v = tail_v.cpu().numpy()

        per_env = []
        for i in range(N):
            if not actions_b[i]:
                continue
            last_value = 0.0 if dones_b[i][-1] > 0.5 else float(tail_v[i])
            adv, returns = gae(rewards_b[i], values_b[i], dones_b[i], last_value, cfg.gamma, cfg.lam)
            b = {k: np.stack(feats_buf[i][k]) for k in features.MODEL_KEYS}
            b.update({"action": np.asarray(actions_b[i], np.int32),
                      "logp": np.asarray(logps_b[i], np.float32),
                      "value": np.asarray(values_b[i], np.float32), "adv": adv, "ret": returns})
            per_env.append(b)

        batch = {k: np.concatenate([b[k] for b in per_env], axis=0)
                 for k in (*features.MODEL_KEYS, "action", "logp", "value", "adv", "ret")}
        return batch, outcomes

    def close(self):
        self._pool.shutdown(wait=False)
        for env in self._envs:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
