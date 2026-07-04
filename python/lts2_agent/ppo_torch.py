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
    # Stability guards (a run that trained cleanly for ~27 iters then diverged: KL spiked and the critic
    # ran away to ~1e26). target_kl early-stops the update epochs once the policy has moved too far;
    # value clipping bounds how much the critic can move per update (PPO's clipped value loss).
    target_kl: float = 0.03
    clip_vloss: bool = True
    adv_clip: float = 10.0
    # L2 penalty on the (legal) option logits: holds them in a sane range so the PPO ratio can't explode,
    # replacing the old tanh clamp whose vanishing gradient froze the policy. Small enough to still allow a
    # confident, discriminating softmax.
    logit_reg: float = 0.01


def make_optimizer(model: model_torch.ActorCritic, config: PPOConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=config.lr)


def update(model: model_torch.ActorCritic, optimizer: torch.optim.Optimizer,
           batch: dict, config: PPOConfig, device) -> dict:
    """Run PPO epochs over ``batch`` (dict of stacked numpy arrays); returns mean metrics."""
    n = int(batch["action"].shape[0])

    adv = batch["adv"].astype(np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    # Bound normalized advantages: as the policy improves, terminal-only rewards make advantages nearly
    # uniform, adv.std() collapses, and normalization amplifies noise into oversized policy updates that
    # blow up the logits. Clipping caps the policy-gradient magnitude.
    adv = np.clip(adv, -config.adv_clip, config.adv_clip)
    ret = np.clip(batch["ret"], -config.return_clip, config.return_clip).astype(np.float32)

    # Whole batch to device once; minibatches are indexed on-device.
    feats = model_torch.to_tensors({k: batch[k] for k in features.MODEL_KEYS}, device)
    action = torch.as_tensor(batch["action"], dtype=torch.long, device=device)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=device)
    old_value = torch.as_tensor(batch["value"], dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)

    mb_size = min(config.minibatch_size, n)
    acc: dict[str, float] = {}
    count = 0
    max_ratio = 0.0
    max_val = 0.0

    model.train()
    stop = False
    max_logit = 0.0
    for _ in range(config.epochs):
        if stop:
            break
        perm = torch.randperm(n, device=device)
        for start in range(0, n, mb_size):
            idx = perm[start:start + mb_size]
            mb_mask = feats[features.MODEL_KEYS.index("mask")][idx]   # [mb, M] bool: legal options
            logits, values = model(*(t[idx] for t in feats))
            new_logp = model_torch.log_prob(logits, action[idx])
            logratio = new_logp - old_logp[idx]
            ratio = logratio.exp()

            adv_mb = adv_t[idx]
            pg_loss = -torch.min(ratio * adv_mb,
                                 ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_mb).mean()

            # Clipped value loss: also penalize the value moving more than clip_eps from its value at
            # collection, so the critic can't run away in a single update.
            if config.clip_vloss:
                v_unclipped = F.smooth_l1_loss(values, ret_t[idx], beta=1.0, reduction="none")
                v_clipped_pred = old_value[idx] + (values - old_value[idx]).clamp(-config.clip_eps, config.clip_eps)
                v_clipped = F.smooth_l1_loss(v_clipped_pred, ret_t[idx], beta=1.0, reduction="none")
                v_loss = torch.max(v_unclipped, v_clipped).mean()
            else:
                v_loss = F.smooth_l1_loss(values, ret_t[idx], beta=1.0)
            ent = model_torch.entropy(logits).mean()
            # L2-penalize the legal logits so they stay in a sane range (bounds the ratio) without a
            # gradient-killing clamp.
            logit_reg = (logits[mb_mask] ** 2).mean()
            loss = (pg_loss + config.vf_coef * v_loss - config.ent_coef * ent
                    + config.logit_reg * logit_reg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if torch.isfinite(gnorm):   # skip a NaN/inf grad rather than poisoning params
                optimizer.step()

            with torch.no_grad():
                # Schulman's non-negative KL estimator (k3): stable, always >= 0, unlike (old-new).mean().
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clip_frac = ((ratio - 1.0).abs() > config.clip_eps).float().mean()
                if torch.isfinite(ratio).all():
                    max_ratio = max(max_ratio, float(ratio.max()))
                max_val = max(max_val, float(values.abs().max()))
                if mb_mask.any():
                    max_logit = max(max_logit, float(logits[mb_mask].abs().max()))
            m = {"loss": loss, "pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent,
                 "approx_kl": approx_kl, "clip_frac": clip_frac}
            for k, val in m.items():
                acc[k] = acc.get(k, 0.0) + float(val)
            count += 1

            # Per-minibatch KL early-stop: bail out of the whole update as soon as the policy has moved
            # too far from the behavior policy, so one bad batch can't run the ratio away.
            if config.target_kl is not None and float(approx_kl) > 1.5 * config.target_kl:
                stop = True
                break

    out = {k: v / count for k, v in acc.items()}
    out["max_ratio"] = max_ratio
    out["max_val"] = max_val
    out["max_logit"] = max_logit
    return out
