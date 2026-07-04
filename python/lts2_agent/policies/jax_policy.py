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

    def _forward(g, dense, card_idx, mask):
        return apply(params, g, dense, card_idx, mask)

    # Warm up the compile with a dummy legal batch so the first request is fast.
    B, M = 1, features.MAX_OPTIONS
    dummy_mask = np.zeros((B, M), bool); dummy_mask[0, 0] = True
    jax.block_until_ready(_forward(
        np.zeros((B, features.STATE_DIM), np.float32),
        np.zeros((B, M, features.OPTION_DIM), np.float32),
        np.zeros((B, M), np.int32), dummy_mask))
    print(f"[jax_policy] loaded {ckpt_path} (hidden={meta['hidden']}) and warmed up.",
          file=sys.stderr, flush=True)

    def policy(state: dict[str, Any], options: list[dict[str, Any]]):
        if not features.is_combat(state) or not options:
            return []   # decline: the game/TUI default drives non-combat
        g = features.encode_state(state)
        dense, card_idx, mask = features.encode_options(state, options)
        logits, _ = _forward(g[None], dense[None], card_idx[None], mask[None])
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
