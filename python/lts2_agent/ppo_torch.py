"""PyTorch PPO update (port of ppo.py): clipped surrogate + Huber value loss + entropy bonus.

Advantages come pre-computed (GAE) from the rollout; this normalizes them and runs a few epochs of
minibatch SGD on the clipped PPO loss, on the GPU. Masked-softmax log-probs/entropy (model_torch)
keep illegal (padded) options out of the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import features, model_torch


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 4
    minibatch_size: int = 512
    max_grad_norm: float = 0.5
    return_clip: float = 10.0


def make_optimizer(model: model_torch.ActorCritic, config: PPOConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=config.lr)


def update(model: model_torch.ActorCritic, optimizer: torch.optim.Optimizer,
           batch: dict, config: PPOConfig, device) -> dict:
    """Run PPO epochs over ``batch`` (dict of stacked numpy arrays); returns mean metrics."""
    n = int(batch["action"].shape[0])

    adv = batch["adv"].astype(np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = np.clip(batch["ret"], -config.return_clip, config.return_clip).astype(np.float32)

    # Whole batch to device once; minibatches are indexed on-device.
    feats = model_torch.to_tensors({k: batch[k] for k in features.MODEL_KEYS}, device)
    action = torch.as_tensor(batch["action"], dtype=torch.long, device=device)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)

    mb_size = min(config.minibatch_size, n)
    acc: dict[str, float] = {}
    count = 0

    model.train()
    for _ in range(config.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, mb_size):
            idx = perm[start:start + mb_size]
            logits, values = model(*(t[idx] for t in feats))
            new_logp = model_torch.log_prob(logits, action[idx])
            ratio = (new_logp - old_logp[idx]).exp()

            adv_mb = adv_t[idx]
            pg_loss = -torch.min(ratio * adv_mb,
                                 ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_mb).mean()
            v_loss = F.smooth_l1_loss(values, ret_t[idx], beta=1.0)   # Huber, delta=1
            ent = model_torch.entropy(logits).mean()
            loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if torch.isfinite(gnorm):   # skip a NaN/inf grad rather than poisoning params
                optimizer.step()

            with torch.no_grad():
                approx_kl = (old_logp[idx] - new_logp).mean()
                clip_frac = ((ratio - 1.0).abs() > config.clip_eps).float().mean()
            m = {"loss": loss, "pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent,
                 "approx_kl": approx_kl, "clip_frac": clip_frac}
            for k, val in m.items():
                acc[k] = acc.get(k, 0.0) + float(val)
            count += 1

    return {k: v / count for k, v in acc.items()}
