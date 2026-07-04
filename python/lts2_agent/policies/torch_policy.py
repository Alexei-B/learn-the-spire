"""Serve a trained PyTorch PPO checkpoint as a decision-server policy (combat only).

Torch counterpart of :mod:`lts2_agent.policies.jax_policy`, loaded by
:mod:`lts2_agent.decision_server` behind the TUI's ``ProcessDecisionEngine``. In combat it returns a
per-option score (the policy logit) for every legal option; **out of combat it returns an empty
ranking** so the game/TUI default handles non-combat (the same decline contract as ``RulesDecisionEngine``).
It uses the *same* :mod:`lts2_agent.features` / :mod:`lts2_agent.model_torch` as the trainer, so what was
trained is exactly what serves.

The checkpoint path comes from the ``LTS2_PPO_CKPT`` environment variable (default
``checkpoints/necro_random.pt``). Served on the CPU — one decision at a time is fast there, and it keeps
the GPU free for a training run in progress. Point the TUI at it via ``lts2.agent.json``::

    "arguments": "-m lts2_agent.decision_server lts2_agent.policies.torch_policy:policy",
    "environment": { "LTS2_PPO_CKPT": "checkpoints/necro_random.pt" }
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

import torch

from .. import features, model_torch

CombatPolicy = Callable[[dict[str, Any], list[dict[str, Any]]], list]


def make_policy(ckpt_path: str, device: str = "cpu") -> CombatPolicy:
    """Build a ``policy(state, options)`` from a torch checkpoint (loaded + warmed up)."""
    model, meta = model_torch.load_checkpoint(ckpt_path, device=device)
    model.eval()

    def _forward(feats: dict) -> "torch.Tensor":
        args = model_torch.to_tensors({k: feats[k][None] for k in features.MODEL_KEYS}, device)
        with torch.no_grad():
            logits, _ = model(*args)
        return logits[0]

    _forward(features.encode(features._SAMPLE_STATE, [{"kind": "EndTurn"}]))  # warm up
    print(f"[torch_policy] loaded {ckpt_path} (hidden={meta['hidden']}, static_dim={meta.get('static_dim')}) "
          f"on {device}.", file=sys.stderr, flush=True)

    def policy(state: dict[str, Any], options: list[dict[str, Any]]):
        if not features.is_combat(state) or not options:
            return []   # decline: the game/TUI default drives non-combat
        logits = _forward(features.encode(state, options)).cpu().numpy()
        n = min(len(options), features.MAX_OPTIONS)
        return [(i, float(logits[i])) for i in range(n)]

    return policy


_cached: CombatPolicy = None  # type: ignore[assignment]


def policy(state: dict[str, Any], options: list[dict[str, Any]]):
    """Module-level entry point for the decision server (lazily loads the checkpoint once)."""
    global _cached
    if _cached is None:
        _cached = make_policy(os.environ.get("LTS2_PPO_CKPT", "checkpoints/necro_random.pt"))
    return _cached(state, options)
