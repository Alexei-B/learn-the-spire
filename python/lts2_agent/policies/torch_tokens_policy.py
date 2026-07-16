"""Serve a trained token-model PPO checkpoint (roadmap 2.2) as a decision-server policy (combat only).

Token-model counterpart of :mod:`lts2_agent.policies.torch_policy`: loaded by
:mod:`lts2_agent.decision_server` behind the TUI's ``ProcessDecisionEngine`` so a ``train_tokens``
checkpoint is pickable in the TUI. In combat it returns a recommended legal action; **out of combat it
returns an empty ranking** (the same decline contract as the features policy). It featurizes through the
*same* :mod:`lts2_agent.model_tokens` as the trainer, so what was trained is exactly what serves.

The checkpoint path comes from ``LTS2_PPO_TOKENS_CKPT`` (default ``checkpoints/tokens_m2.pt``). Served on
the CPU (one decision at a time is fast; keeps the GPU free for a training run). Point the TUI at it::

    "arguments": "-m lts2_agent.decision_server lts2_agent.policies.torch_tokens_policy:policy",
    "environment": { "LTS2_PPO_TOKENS_CKPT": "checkpoints/tokens_m2.pt" }
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

import torch

from .. import model_tokens, tokens

CombatPolicy = Callable[[dict[str, Any], list[dict[str, Any]]], list]


def _is_combat(state: dict) -> bool:
    return state.get("phase") == "Combat" and bool(state.get("combat"))


def make_policy(ckpt_path: str, device: str = "cpu") -> CombatPolicy:
    """Build a ``policy(state, options)`` from a token-model checkpoint (loaded + warmed up)."""
    import time as _time
    last: Exception | None = None
    model = meta = None
    for _ in range(6):   # a running trainer rewrites the checkpoint every ~2s; retry a partial read
        try:
            model, meta = model_tokens.load_checkpoint(ckpt_path, device=device)
            break
        except Exception as e:
            last = e
            _time.sleep(0.3)
    if model is None:
        raise RuntimeError(f"could not load {ckpt_path} after retries: {last}")
    model.eval()

    def _forward(state: dict, options: list) -> "torch.Tensor":
        feats = model_tokens.featurize(state, options)
        args = model_tokens.to_tensors(model_tokens.stack([feats]), device)
        with torch.no_grad():
            logits, _ = model(**args)
        return logits[0]

    # Warm up (a trivial one-enemy combat).
    _warm = {"phase": "Combat", "players": [{"currentHp": 1, "maxHp": 1, "block": 0,
             "combatState": {"energy": 1, "maxEnergy": 1, "hand": []}}],
             "combat": {"enemies": [{"combatId": 1, "currentHp": 1, "maxHp": 1, "isHittable": True,
                                     "intents": [], "powers": []}]}}
    _forward(_warm, [{"kind": "EndTurn"}])
    print(f"[torch_tokens_policy] loaded {ckpt_path} (d_model={meta['d_model']}, "
          f"{tokens.tokenizer_signature()}) on {device}.", file=sys.stderr, flush=True)

    # Like the features policy: mass splits across many similar card options while EndTurn is one option,
    # so raw ARGMAX collapses onto EndTurn and plays terribly. Default to SAMPLING from the trained
    # (temperature-scaled) softmax over legal options — how the policy was trained. LTS2_POLICY_GREEDY=1
    # forces argmax.
    greedy = os.environ.get("LTS2_POLICY_GREEDY") in ("1", "true")
    temp = float(os.environ.get("LTS2_POLICY_TEMP", "1.0"))

    def policy(state: dict[str, Any], options: list[dict[str, Any]]):
        if not _is_combat(state) or not options:
            return []   # decline: the game/TUI default drives non-combat
        logits = _forward(state, options)
        n = min(len(options), model_tokens.MAX_OPTIONS)
        legal = logits[:n]
        if greedy:
            return [(i, float(legal[i])) for i in range(n)]
        finite = torch.isfinite(legal)
        if not bool(finite.any()):
            return []
        probs = torch.softmax(legal / max(temp, 1e-3), dim=-1)
        choice = int(torch.multinomial(probs, 1).item())
        return [{"index": choice, "score": 1.0, "rationale": "sampled from the trained token policy"}]

    return policy


_cached: CombatPolicy = None  # type: ignore[assignment]


def policy(state: dict[str, Any], options: list[dict[str, Any]]):
    """Module-level entry point for the decision server (lazily loads the checkpoint once)."""
    global _cached
    if _cached is None:
        _cached = make_policy(os.environ.get("LTS2_PPO_TOKENS_CKPT", "checkpoints/tokens_m2.pt"))
    res = _cached(state, options)
    if res:
        def ix_sc(e):
            return (e["index"], e["score"]) if isinstance(e, dict) else (e[0], e[1])
        ti, ts = max((ix_sc(e) for e in res), key=lambda t: t[1])
        o = options[ti]
        label = (o.get("card") or {}).get("cardId") if o.get("kind") == "PlayCard" else o.get("kind")
        print(f"[torch_tokens_policy] phase={state.get('phase')} opts={len(options)} -> pick {label} ({ts:.2f})",
              file=sys.stderr, flush=True)
    else:
        print(f"[torch_tokens_policy] phase={state.get('phase')} opts={len(options)} -> decline",
              file=sys.stderr, flush=True)
    return res
