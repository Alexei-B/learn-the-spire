"""PyTorch port of the per-option action-scoring actor-critic (see model.py for the design).

Same architecture and same framework-agnostic feature dict from :mod:`lts2_agent.features`, but on
PyTorch so it runs on the GPU. The policy scores each legal option from the global state context, the
hand summary, the option's dense features, and a learned card embedding; a masked softmax over the
legal set gives the action distribution. A value head reads the state context for the GAE critic.

Batch convention (fixed shapes): g[B,STATE_DIM]; hand_dense[B,MAX_HAND,CARD_FEAT_DIM];
hand_idx/hand_mask[B,MAX_HAND]; dense[B,MAX_OPTIONS,OPTION_DIM]; card_idx/mask[B,MAX_OPTIONS].
Returns (masked_logits[B,MAX_OPTIONS], value[B]) with illegal options at NEG_INF.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128, embed_dim: int = 16, card_vocab: int = features.CARD_VOCAB,
                 static_table=None):
        super().__init__()
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.card_vocab = card_vocab

        self.card_embed = nn.Embedding(card_vocab, embed_dim)  # learned, shared across hand + options

        # Fixed per-card static metadata (tags/keywords/var-keys multi-hots), gathered by the card index
        # exactly like the embedding — so the rich card semantics live on the GPU, not on the wire.
        if static_table is None:
            static_table = features.card_static_table()
        static_table = np.asarray(static_table, dtype=np.float32)
        self.static_dim = int(static_table.shape[1])
        self.register_buffer("card_static", torch.from_numpy(static_table))  # [card_vocab, static_dim]

        # State context from the global scalars.
        self.g1 = nn.Linear(features.STATE_DIM, hidden)
        self.g2 = nn.Linear(hidden, hidden)

        # Hand: per-card (features ⊕ embedding ⊕ static) -> hidden, then masked mean-pool.
        self.hand = nn.Linear(features.CARD_FEAT_DIM + embed_dim + self.static_dim, hidden)

        # Combined context: battlefield ⊕ hand summary.
        self.ctx = nn.Linear(hidden + hidden, hidden)

        # Per-option scorer: option features ⊕ card embedding ⊕ static ⊕ broadcast context -> logit.
        self.logit_scale = 10.0
        self.o1 = nn.Linear(features.OPTION_DIM + embed_dim + self.static_dim + hidden, hidden)
        self.o2 = nn.Linear(hidden, hidden)
        self.o_logit = nn.Linear(hidden, 1)

        # Value head. Its output is tanh-bounded to ±value_scale: returns are clipped to ±return_clip
        # (10), so the critic never needs to exceed this, and bounding it means the value can't run away
        # to ~1e26 and drag the shared trunk (and thus the policy logits) into divergence.
        self.value_scale = 20.0
        self.v1 = nn.Linear(hidden, hidden)
        self.v_out = nn.Linear(hidden, 1)

    def forward(self, g, hand_dense, hand_idx, hand_mask, dense, card_idx, mask):
        h = F.relu(self.g1(g))
        h = F.relu(self.g2(h))                                    # [B, H]

        hand_emb = self.card_embed(hand_idx)                     # [B, MAX_HAND, E]
        hand_static = self.card_static[hand_idx]                 # [B, MAX_HAND, static_dim]
        hand_in = torch.cat([hand_dense, hand_emb, hand_static], dim=-1)
        hz = F.relu(self.hand(hand_in))                          # [B, MAX_HAND, H]
        hm = hand_mask.unsqueeze(-1).to(hz.dtype)
        hand_vec = (hz * hm).sum(dim=1) / hm.sum(dim=1).clamp_min(1.0)  # [B, H]

        ctx = F.relu(self.ctx(torch.cat([h, hand_vec], dim=-1)))  # [B, H]

        opt_emb = self.card_embed(card_idx)                      # [B, M, E]
        opt_static = self.card_static[card_idx]                  # [B, M, static_dim]
        n_opt = dense.shape[1]
        ctx_b = ctx.unsqueeze(1).expand(-1, n_opt, -1)
        opt_in = torch.cat([dense, opt_emb, opt_static, ctx_b], dim=-1)  # [B, M, *]

        z = F.relu(self.o1(opt_in))
        z = F.relu(self.o2(z))
        # Bound the logits (tanh to ±logit_scale) so no single option's score can run away. This caps the
        # PPO importance ratio exp(new_logp - old_logp): unbounded logits let it explode to ~1e27 and
        # diverge the policy once training sharpens. ±logit_scale still allows very peaked softmaxes.
        raw_logits = self.o_logit(z).squeeze(-1)                 # [B, M]
        logits = self.logit_scale * torch.tanh(raw_logits / self.logit_scale)
        masked_logits = torch.where(mask, logits, logits.new_full((), NEG_INF))

        raw_value = self.v_out(F.relu(self.v1(ctx))).squeeze(-1)          # [B]
        value = self.value_scale * torch.tanh(raw_value / self.value_scale)
        return masked_logits, value


# --- Masked categorical over options ---------------------------------------------------------------

def log_prob(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Log-prob of ``action`` (int index) under the masked-softmax over ``logits`` ([B, M])."""
    logp_all = F.log_softmax(logits, dim=-1)
    return logp_all.gather(-1, action.unsqueeze(-1)).squeeze(-1)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the masked policy ([B, M] -> [B]); illegal options (-inf) contribute 0."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    term = torch.where(torch.isfinite(logp), p * logp, torch.zeros_like(logp))
    return -term.sum(dim=-1)


@torch.no_grad()
def sample_action(logits: torch.Tensor):
    """Sample a legal action index from the masked policy; returns ``(action[B], logp[B])``."""
    # Gumbel-max = categorical sampling, done on-device.
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)).clamp_min(1e-20))
    action = (logits + g).argmax(dim=-1)
    return action, log_prob(logits, action)


# --- Feature-dict <-> tensor helpers ---------------------------------------------------------------

_INT_KEYS = {"hand_idx", "card_idx"}
_BOOL_KEYS = {"hand_mask", "mask"}


def to_tensors(feats_stacked: dict, device) -> tuple:
    """Move a dict of stacked numpy feature arrays ([B, ...]) to model-argument tensors on ``device``."""
    out = []
    for k in features.MODEL_KEYS:
        v = feats_stacked[k]
        if k in _INT_KEYS:
            t = torch.as_tensor(np.asarray(v), dtype=torch.long, device=device)
        elif k in _BOOL_KEYS:
            t = torch.as_tensor(np.asarray(v), dtype=torch.bool, device=device)
        else:
            t = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
        out.append(t)
    return tuple(out)


def stack_feats(feats_list: list[dict]) -> dict:
    """Stack a list of single-obs feature dicts into [B, ...] numpy arrays."""
    return {k: np.stack([f[k] for f in feats_list]) for k in features.MODEL_KEYS}


# --- Checkpointing (state_dict + a meta sidecar) ---------------------------------------------------

def save_checkpoint(path: str, m: ActorCritic) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(m.state_dict(), path)
    meta = {
        "backend": "torch", "hidden": m.hidden, "embed_dim": m.embed_dim, "card_vocab": m.card_vocab,
        "static_dim": m.static_dim, "catalog": features.CATALOG_SIGNATURE,
        "state_dim": features.STATE_DIM, "option_dim": features.OPTION_DIM,
        "max_options": features.MAX_OPTIONS, "feature_version": features.FEATURE_VERSION,
    }
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)


def load_checkpoint(path: str, device="cpu"):
    """Load ``(model, meta)``, rejecting a checkpoint whose feature layout no longer matches."""
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    if (meta["state_dim"], meta["option_dim"], meta["feature_version"], meta.get("catalog")) != (
            features.STATE_DIM, features.OPTION_DIM, features.FEATURE_VERSION, features.CATALOG_SIGNATURE):
        raise ValueError(
            f"Checkpoint {path} was trained with a different feature encoding "
            f"(meta={meta['state_dim']}/{meta['option_dim']}/v{meta['feature_version']}/{meta.get('catalog')} vs "
            f"current {features.STATE_DIM}/{features.OPTION_DIM}/v{features.FEATURE_VERSION}/"
            f"{features.CATALOG_SIGNATURE}); retrain.")
    m = ActorCritic(hidden=meta["hidden"], embed_dim=meta["embed_dim"], card_vocab=meta["card_vocab"])
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    return m, meta
