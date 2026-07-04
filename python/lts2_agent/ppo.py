"""PPO update: the clipped-surrogate objective + value loss + entropy bonus over collected batches.

Advantages come pre-computed (GAE) from :mod:`lts2_agent.rollout`; this module normalizes them, then
runs a few epochs of minibatch SGD on the clipped PPO loss. The policy log-probs and entropy use the
masked-softmax helpers in :mod:`lts2_agent.model`, so illegal (padded) options never receive gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from . import features, model


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 4
    minibatch_size: int = 512
    max_grad_norm: float = 0.5
    # Rewards are bounded, so true returns are small; clip the value target well above their real range
    # to stop the critic from ever running away (a rare PPO divergence that otherwise corrupts params).
    return_clip: float = 10.0


def create_train_state(m: model.ActorCritic, params, config: PPOConfig) -> TrainState:
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr),
    )
    return TrainState.create(apply_fn=m.apply, params=params, tx=tx)


def _loss(params, apply_fn, mb, clip_eps, vf_coef, ent_coef):
    logits, values = apply_fn(params, *(mb[k] for k in features.MODEL_KEYS))
    new_logp = model.log_prob(logits, mb["action"])
    ratio = jnp.exp(new_logp - mb["logp"])

    adv = mb["adv"]
    pg1 = ratio * adv
    pg2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_loss = -jnp.mean(jnp.minimum(pg1, pg2))

    # Huber (smooth-L1) value loss: its gradient is bounded for large errors, so a mispredicted value
    # can't produce an enormous gradient that overshoots (MSE did — the critic diverged to ~1e4 and,
    # via the shared global grad-norm clip, starved the actor's gradient and stalled learning).
    v_loss = jnp.mean(optax.huber_loss(values, mb["ret"], delta=1.0))
    ent = jnp.mean(model.entropy(logits))
    loss = pg_loss + vf_coef * v_loss - ent_coef * ent

    approx_kl = jnp.mean(mb["logp"] - new_logp)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
    return loss, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent,
                  "approx_kl": approx_kl, "clip_frac": clip_frac}


def _make_train_step(config: PPOConfig):
    grad_fn = jax.value_and_grad(_loss, has_aux=True)

    @jax.jit
    def step(state: TrainState, mb):
        (loss, aux), grads = grad_fn(
            state.params, state.apply_fn, mb, config.clip_eps, config.vf_coef, config.ent_coef)
        # Skip a non-finite gradient (NaN/inf) rather than poisoning the params with it.
        finite = jnp.all(jnp.asarray(
            [jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
        grads = jax.tree_util.tree_map(lambda g: jnp.where(finite, g, 0.0), grads)
        state = state.apply_gradients(grads=grads)
        aux = {**aux, "loss": loss}
        return state, aux

    return step


def update(state: TrainState, batch: dict, config: PPOConfig, key):
    """Run PPO epochs over ``batch``; returns ``(new_state, mean_metrics)``."""
    n = batch["action"].shape[0]
    adv = batch["adv"]
    ret = np.clip(batch["ret"], -config.return_clip, config.return_clip)  # bound the value target
    batch = {**batch, "adv": (adv - adv.mean()) / (adv.std() + 1e-8), "ret": ret}
    # To device once.
    data = {k: jnp.asarray(v) for k, v in batch.items()}

    step = _make_train_step(config)
    metrics_acc: list[dict] = []
    mb_size = min(config.minibatch_size, n)

    for _ in range(config.epochs):
        key, sub = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sub, n))
        for start in range(0, n, mb_size):
            idx = perm[start:start + mb_size]
            mb = {k: v[idx] for k, v in data.items()}
            state, m = step(state, mb)
            metrics_acc.append(m)

    mean = {k: float(np.mean([float(m[k]) for m in metrics_acc])) for k in metrics_acc[0]}
    return state, mean
