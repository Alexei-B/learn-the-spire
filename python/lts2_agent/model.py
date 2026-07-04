"""The actor-critic network: a per-option action-scoring policy + a state value head (Flax).

The policy scores each legal option independently from the global state context and the option's own
features (dense features from :mod:`lts2_agent.features` plus a learned embedding of its hashed
card-id bucket), then a masked softmax over the legal set gives the action distribution. This handles
a *variable* number of options and target selection natively — each ``(card, target)`` is its own
option — so there is no separate target head. The value head reads the state context for the GAE
critic.

Batch convention (fixed shapes so everything jits):
  * ``g``        — ``float32[B, STATE_DIM]``
  * ``dense``    — ``float32[B, MAX_OPTIONS, OPTION_DIM]``
  * ``card_idx`` — ``int32[B, MAX_OPTIONS]``
  * ``mask``     — ``bool[B, MAX_OPTIONS]``
Returns ``(masked_logits[B, MAX_OPTIONS], value[B])`` with illegal options at ``-1e9``.
"""

from __future__ import annotations

import json
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import serialization

from . import features

NEG_INF = -1e9


class ActorCritic(nn.Module):
    hidden: int = 128
    embed_dim: int = 16
    card_vocab: int = features.CARD_VOCAB

    @nn.compact
    def __call__(self, g, hand_dense, hand_idx, hand_mask, dense, card_idx, mask):
        card_embed = nn.Embed(self.card_vocab, self.embed_dim)  # shared across hand + options

        # State context from the global scalars.
        h = nn.relu(nn.Dense(self.hidden)(g))
        h = nn.relu(nn.Dense(self.hidden)(h))                       # [B, H]

        # Hand summary: encode each held card (features ⊕ embedding), then masked mean-pool. This is
        # what lets the policy see what else is in hand and learn card synergies.
        hand_emb = card_embed(hand_idx)                             # [B, MAX_HAND, E]
        hand_in = jnp.concatenate([hand_dense, hand_emb], axis=-1)
        hz = nn.relu(nn.Dense(self.hidden)(hand_in))                # [B, MAX_HAND, H]
        hm = hand_mask.astype(jnp.float32)[..., None]
        hand_vec = jnp.sum(hz * hm, axis=1) / jnp.clip(jnp.sum(hm, axis=1), 1.0)  # [B, H]

        # Combined context: battlefield + hand contents.
        ctx = nn.relu(nn.Dense(self.hidden)(jnp.concatenate([h, hand_vec], axis=-1)))  # [B, H]

        # Per-option input: option features ⊕ card embedding ⊕ broadcast combined context.
        opt_emb = card_embed(card_idx)                             # [B, M, E]
        n_opt = dense.shape[1]
        ctx_broadcast = jnp.broadcast_to(ctx[:, None, :], (ctx.shape[0], n_opt, ctx.shape[1]))
        opt_in = jnp.concatenate([dense, opt_emb, ctx_broadcast], axis=-1)  # [B, M, *]

        z = nn.relu(nn.Dense(self.hidden)(opt_in))
        z = nn.relu(nn.Dense(self.hidden)(z))
        logits = nn.Dense(1)(z)[..., 0]                             # [B, M]
        masked_logits = jnp.where(mask, logits, NEG_INF)

        value = nn.Dense(1)(nn.relu(nn.Dense(self.hidden)(ctx)))[..., 0]  # [B]
        return masked_logits, value


# --- Masked categorical distribution over options --------------------------------------------------

def log_prob(logits, action):
    """Log-prob of ``action`` (int index) under the masked-softmax over ``logits`` ([B, M])."""
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp_all, action[:, None], axis=-1)[:, 0]


def entropy(logits):
    """Entropy of the masked-softmax policy ([B, M] -> [B]); ignores -inf (illegal) options."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    p = jnp.exp(logp)
    return -jnp.sum(jnp.where(jnp.isfinite(logp), p * logp, 0.0), axis=-1)


def sample_action(logits, key):
    """Sample a legal action index from the masked policy; returns ``(action[B], logp[B])``."""
    action = jax.random.categorical(key, logits, axis=-1)
    return action, log_prob(logits, action)


def forward1(apply_fn, params, feats):
    """Forward one observation's (unbatched) feature dict → ``(logits[1, M], value[1])``."""
    args = tuple(jnp.asarray(feats[k])[None] for k in features.MODEL_KEYS)
    return apply_fn(params, *args)


def init_params(rng, model: ActorCritic):
    """Initialize params with a single dummy batch of the correct fixed shapes."""
    B, M, Hd = 1, features.MAX_OPTIONS, features.MAX_HAND
    g = jnp.zeros((B, features.STATE_DIM), jnp.float32)
    hand_dense = jnp.zeros((B, Hd, features.CARD_FEAT_DIM), jnp.float32)
    hand_idx = jnp.zeros((B, Hd), jnp.int32)
    hand_mask = jnp.zeros((B, Hd), bool).at[:, 0].set(True)
    dense = jnp.zeros((B, M, features.OPTION_DIM), jnp.float32)
    card_idx = jnp.zeros((B, M), jnp.int32)
    mask = jnp.zeros((B, M), bool).at[:, 0].set(True)
    return model.init(rng, g, hand_dense, hand_idx, hand_mask, dense, card_idx, mask)


# --- Checkpointing (params + a meta sidecar so the server can rebuild the exact model) --------------

def save_checkpoint(path: str, params, m: "ActorCritic") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    meta = {
        "hidden": m.hidden, "embed_dim": m.embed_dim, "card_vocab": m.card_vocab,
        "state_dim": features.STATE_DIM, "option_dim": features.OPTION_DIM,
        "max_options": features.MAX_OPTIONS, "feature_version": features.FEATURE_VERSION,
    }
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)


def load_checkpoint(path: str):
    """Load ``(model, params, meta)``, rejecting a checkpoint whose feature layout no longer matches."""
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    if (meta["state_dim"], meta["option_dim"], meta["feature_version"]) != (
            features.STATE_DIM, features.OPTION_DIM, features.FEATURE_VERSION):
        raise ValueError(
            f"Checkpoint {path} was trained with a different feature encoding "
            f"(meta={meta['state_dim']}/{meta['option_dim']}/v{meta['feature_version']} vs "
            f"current {features.STATE_DIM}/{features.OPTION_DIM}/v{features.FEATURE_VERSION}); retrain.")
    m = ActorCritic(hidden=meta["hidden"], embed_dim=meta["embed_dim"], card_vocab=meta["card_vocab"])
    params = init_params(jax.random.PRNGKey(0), m)
    with open(path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    return m, params, meta
