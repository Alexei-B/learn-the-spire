"""Model adapters: the small seam that lets the shared PPO rollout + update drive either the
hand-crafted-features actor-critic (:mod:`lts2_agent.model_torch`) or the entity-token set-transformer
(:mod:`lts2_agent.model_tokens`) without duplicating the rollout/PPO code.

An adapter hides the two differences between the stacks: how an observation is featurized, and how the
featurized batch is fed to the model (features uses a positional tuple; tokens uses keyword args + a
dict). Everything else — GAE, the PPO surrogate, minibatching — is identical, so it stays in
:mod:`lts2_agent.rollout_torch` / :mod:`lts2_agent.ppo_torch`, which take an ``adapter`` parameter that
**defaults to the features adapter** (so the baseline ``train_torch`` path is byte-for-byte unchanged).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from . import features, model_tokens, model_torch


class FeaturesAdapter:
    """Drives :mod:`lts2_agent.model_torch` over :mod:`lts2_agent.features` — the PPO baseline path.

    ``tensors`` is a positional tuple in ``features.MODEL_KEYS`` order (what the model's ``forward``
    consumes). Reproduces today's rollout/PPO calls exactly."""

    kind = "ppo-scenario"
    MODEL_KEYS = features.MODEL_KEYS

    def featurize(self, state: dict, options: list) -> Dict[str, np.ndarray]:
        f = features.encode(state, options)
        for j, o in enumerate(options[:features.MAX_OPTIONS]):
            if o.get("kind") not in _COMBAT_KINDS:
                f["mask"][j] = False
        return f

    def stack(self, feats_list: List[dict]) -> Dict[str, np.ndarray]:
        return model_torch.stack_feats(feats_list)

    def to_device(self, batch: dict, device):
        return model_torch.to_tensors(batch, device)

    def forward(self, model, tensors):
        return model(*tensors)

    def index(self, tensors, idx):
        return tuple(t[idx] for t in tensors)

    def mask(self, tensors) -> "torch.Tensor":
        return tensors[features.MODEL_KEYS.index("mask")]

    def sample_action(self, logits):
        return model_torch.sample_action(logits)

    def log_prob(self, logits, action):
        return model_torch.log_prob(logits, action)

    def entropy(self, logits):
        return model_torch.entropy(logits)

    def signature(self) -> str:
        return features.CATALOG_SIGNATURE


class TokensAdapter:
    """Drives :mod:`lts2_agent.model_tokens` over the entity tokenizer.

    ``tensors`` is a dict keyed by ``model_tokens.MODEL_KEYS`` (the model takes keyword args)."""

    kind = "ppo-tokens"
    MODEL_KEYS = model_tokens.MODEL_KEYS

    def featurize(self, state: dict, options: list) -> Dict[str, np.ndarray]:
        return model_tokens.featurize(state, options)  # masks non-combat kinds internally

    def stack(self, feats_list: List[dict]) -> Dict[str, np.ndarray]:
        return model_tokens.stack(feats_list)

    def to_device(self, batch: dict, device):
        return model_tokens.to_tensors(batch, device)

    def forward(self, model, tensors):
        return model(**tensors)

    def index(self, tensors, idx):
        return {k: v[idx] for k, v in tensors.items()}

    def mask(self, tensors) -> "torch.Tensor":
        return tensors["opt_mask"]

    def sample_action(self, logits):
        return model_tokens.sample_action(logits)

    def log_prob(self, logits, action):
        return model_tokens.log_prob(logits, action)

    def entropy(self, logits):
        return model_tokens.entropy(logits)

    def signature(self) -> str:
        return model_tokens.tokens.tokenizer_signature()


# Kept module-local to avoid importing scenario (circular): same set as scenario._COMBAT_KINDS.
_COMBAT_KINDS = {"PlayCard", "EndTurn", "UsePotion", "DiscardPotion"}
