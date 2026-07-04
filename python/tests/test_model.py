"""Model contract tests: forward-pass shapes, masking, and always-legal sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lts2_agent import features, model


def _batch(n_legal_per_row):
    B = len(n_legal_per_row)
    M, Hd = features.MAX_OPTIONS, features.MAX_HAND
    args = dict(
        g=jnp.zeros((B, features.STATE_DIM), jnp.float32),
        hand_dense=jnp.zeros((B, Hd, features.CARD_FEAT_DIM), jnp.float32),
        hand_idx=jnp.zeros((B, Hd), jnp.int32),
        hand_mask=jnp.zeros((B, Hd), bool).at[:, 0].set(True),
        dense=jnp.zeros((B, M, features.OPTION_DIM), jnp.float32),
        card_idx=jnp.zeros((B, M), jnp.int32),
    )
    mask = np.zeros((B, M), bool)
    for i, k in enumerate(n_legal_per_row):
        mask[i, :k] = True
    args["mask"] = jnp.asarray(mask)
    return tuple(args[k] for k in features.MODEL_KEYS)


def test_forward_shapes_and_masking():
    m = model.ActorCritic()
    params = model.init_params(jax.random.PRNGKey(0), m)
    logits, value = m.apply(params, *_batch([3, 1, 10]))
    assert logits.shape == (3, features.MAX_OPTIONS)
    assert value.shape == (3,)
    # Illegal options are driven to -inf so the softmax never puts mass on them.
    assert bool(jnp.all(logits[0, 3:] == model.NEG_INF))
    assert bool(jnp.all(logits[1, 1:] == model.NEG_INF))


def test_sampled_action_is_always_legal():
    m = model.ActorCritic()
    params = model.init_params(jax.random.PRNGKey(1), m)
    n_legal = [3, 1, 7, 2]
    logits, _ = m.apply(params, *_batch(n_legal))
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
    logits, _ = m.apply(params, *_batch([5, 1, 3]))
    ent = np.asarray(model.entropy(logits))
    assert np.all(np.isfinite(ent))
    assert np.all(ent >= -1e-6)
    # A single legal option has ~zero entropy.
    assert ent[1] < 1e-4


def test_encode_to_model_roundtrip():
    # A real (state, options) pair featurizes into shapes the model consumes (incl. the hand).
    state = {"phase": "Combat", "players": [{"currentHp": 50, "maxHp": 80, "block": 0,
             "combatState": {"energy": 3, "maxEnergy": 3,
                             "hand": [{"cardId": "Bodyguard", "type": "Skill", "summon": 5}]}}],
             "combat": {"enemies": [{"combatId": 1, "currentHp": 10, "maxHp": 10, "isHittable": True,
                                     "intents": [], "powers": []}]}}
    options = [{"kind": "PlayCard", "card": {"cardId": "StrikeIronclad", "type": "Attack",
               "energyCost": 1, "damage": 6}, "targetCombatId": 1}, {"kind": "EndTurn"}]
    feats = features.encode(state, options)
    assert feats["hand_dense"].shape == (features.MAX_HAND, features.CARD_FEAT_DIM)
    assert bool(feats["hand_mask"][0]) and not bool(feats["hand_mask"][1])  # one card in hand
    m = model.ActorCritic()
    params = model.init_params(jax.random.PRNGKey(0), m)
    logits, value = model.forward1(m.apply, params, feats)
    assert logits.shape == (1, features.MAX_OPTIONS)
    assert bool(feats["mask"][0]) and bool(feats["mask"][1]) and not bool(feats["mask"][2])
