"""Model contract tests: forward-pass shapes, masking, and always-legal sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lts2_agent import features, model


def _batch(n_legal_per_row):
    B = len(n_legal_per_row)
    M = features.MAX_OPTIONS
    g = jnp.zeros((B, features.STATE_DIM), jnp.float32)
    dense = jnp.zeros((B, M, features.OPTION_DIM), jnp.float32)
    card_idx = jnp.zeros((B, M), jnp.int32)
    mask = np.zeros((B, M), bool)
    for i, k in enumerate(n_legal_per_row):
        mask[i, :k] = True
    return g, dense, card_idx, jnp.asarray(mask)


def test_forward_shapes_and_masking():
    m = model.ActorCritic()
    params = model.init_params(jax.random.PRNGKey(0), m)
    g, dense, card_idx, mask = _batch([3, 1, 10])
    logits, value = m.apply(params, g, dense, card_idx, mask)
    assert logits.shape == (3, features.MAX_OPTIONS)
    assert value.shape == (3,)
    # Illegal options are driven to -inf so the softmax never puts mass on them.
    assert bool(jnp.all(logits[0, 3:] == model.NEG_INF))
    assert bool(jnp.all(logits[1, 1:] == model.NEG_INF))


def test_sampled_action_is_always_legal():
    m = model.ActorCritic()
    params = model.init_params(jax.random.PRNGKey(1), m)
    n_legal = [3, 1, 7, 2]
    g, dense, card_idx, mask = _batch(n_legal)
    logits, _ = m.apply(params, g, dense, card_idx, mask)
    key = jax.random.PRNGKey(2)
    for _ in range(50):
        key, sub = jax.random.split(key)
        action, logp = model.sample_action(logits, sub)
        a = np.asarray(action)
        for i, k in enumerate(n_legal):
            assert a[i] < k, f"sampled illegal action {a[i]} with only {k} legal"
        assert np.all(np.isfinite(np.asarray(logp)))


def test_entropy_is_finite_and_nonnegative():
    m = model.ActorCritic()
    params = model.init_params(jax.random.PRNGKey(3), m)
    g, dense, card_idx, mask = _batch([5, 1, 3])
    logits, _ = m.apply(params, g, dense, card_idx, mask)
    ent = np.asarray(model.entropy(logits))
    assert np.all(np.isfinite(ent))
    assert np.all(ent >= -1e-6)
    # A single legal option has ~zero entropy.
    assert ent[1] < 1e-4


def test_encode_to_model_roundtrip():
    # A real (state, options) pair featurizes into shapes the model consumes.
    state = {"phase": "Combat", "players": [{"currentHp": 50, "maxHp": 80, "block": 0,
             "combatState": {"energy": 3, "maxEnergy": 3, "hand": []}}],
             "combat": {"enemies": [{"combatId": 1, "currentHp": 10, "maxHp": 10, "isHittable": True,
                                     "intents": [], "powers": []}]}}
    options = [{"kind": "PlayCard", "card": {"cardId": "StrikeIronclad", "type": "Attack",
               "energyCost": 1, "damage": 6}, "targetCombatId": 1}, {"kind": "EndTurn"}]
    dense, card_idx, mask = features.encode_options(state, options)
    g = features.encode_state(state)
    m = model.ActorCritic()
    params = model.init_params(jax.random.PRNGKey(0), m)
    logits, value = m.apply(params, g[None], dense[None], card_idx[None], mask[None])
    assert logits.shape == (1, features.MAX_OPTIONS)
    assert bool(mask[0]) and bool(mask[1]) and not bool(mask[2])
