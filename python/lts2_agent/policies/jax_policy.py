"""Serve a trained PPO checkpoint as a decision-server policy (combat only).

Loaded by :mod:`lts2_agent.decision_server` behind the TUI's ``ProcessDecisionEngine``. In combat it
returns a per-option score (the policy logit) for every legal option; **out of combat it returns an
empty ranking** so the game/TUI default handles non-combat — the same decline contract as the built-in
``RulesDecisionEngine``. It uses the *same* :mod:`lts2_agent.features`/:mod:`lts2_agent.model` as the
trainer, so what was trained is exactly what serves.

The checkpoint path comes from the ``LTS2_PPO_CKPT`` environment variable (default ``checkpoints/ppo``)
because the decision server loads a policy by ``module:attr`` with no extra args. Point the TUI at it::

    $env:LTS2_PPO_CKPT = "checkpoints/ppo"
    $env:LTS2_AGENT_CMD = "python"
    $env:LTS2_AGENT_ARGS = "-m lts2_agent.decision_server lts2_agent.policies.jax_policy:policy"
    $env:LTS2_AGENT_NAME = "PPO (jax)"

The model is JIT-compiled and **warmed up at load** so the first real ``evaluate`` doesn't blow the C#
side's response timeout while JAX compiles.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

import jax
import numpy as np

from .. import features, model

CombatPolicy = Callable[[dict[str, Any], list[dict[str, Any]]], list]


def make_policy(ckpt_path: str) -> CombatPolicy:
    """Build a ``policy(state, options)`` from a checkpoint (JIT-compiled and warmed up)."""
    m, params, meta = model.load_checkpoint(ckpt_path)
    apply = jax.jit(m.apply)

    # Warm up the compile with a dummy legal observation so the first request is fast.
    dummy = features.encode(features._SAMPLE_STATE, [{"kind": "EndTurn"}])
    jax.block_until_ready(model.forward1(apply, params, dummy))
    print(f"[jax_policy] loaded {ckpt_path} (hidden={meta['hidden']}) and warmed up.",
          file=sys.stderr, flush=True)

    def policy(state: dict[str, Any], options: list[dict[str, Any]]):
        if not features.is_combat(state) or not options:
            return []   # decline: the game/TUI default drives non-combat
        feats = features.encode(state, options)
        logits, _ = model.forward1(apply, params, feats)
        logits = np.asarray(logits[0])
        n = min(len(options), features.MAX_OPTIONS)
        return [(i, float(logits[i])) for i in range(n)]

    return policy


_cached: CombatPolicy = None  # type: ignore[assignment]


def policy(state: dict[str, Any], options: list[dict[str, Any]]):
    """Module-level entry point for the decision server (lazily loads the checkpoint once)."""
    global _cached
    if _cached is None:
        _cached = make_policy(os.environ.get("LTS2_PPO_CKPT", "checkpoints/ppo"))
    return _cached(state, options)
