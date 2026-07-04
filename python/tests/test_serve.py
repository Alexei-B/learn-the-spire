"""Train/serve parity: a checkpoint round-trips and jax_policy agrees with the model directly.

No C# environment needed — this validates that features + model + checkpoint + the served policy all
agree, which is the portability guarantee the whole design rests on.
"""

from __future__ import annotations

import jax
import numpy as np

from lts2_agent import features, model
from lts2_agent.policies import jax_policy


def _combat_pair():
    state = {
        "phase": "Combat",
        "players": [{"currentHp": 50, "maxHp": 80, "block": 0,
                     "combatState": {"energy": 3, "maxEnergy": 3, "hand": [], "powers": []}}],
        "combat": {"enemies": [
            {"combatId": 1, "currentHp": 8, "maxHp": 40, "block": 0, "isHittable": True,
             "intents": [{"type": "Attack", "damage": 12, "hits": 1}], "powers": []},
            {"combatId": 2, "currentHp": 40, "maxHp": 40, "block": 0, "isHittable": True,
             "intents": [{"type": "Attack", "damage": 5, "hits": 1}], "powers": []},
        ]},
    }
    strike = {"cardId": "StrikeIronclad", "type": "Attack", "energyCost": 1, "damage": 9}
    defend = {"cardId": "DefendIronclad", "type": "Skill", "energyCost": 1, "block": 5}
    options = [
        {"kind": "PlayCard", "card": strike, "targetCombatId": 1},
        {"kind": "PlayCard", "card": strike, "targetCombatId": 2},
        {"kind": "PlayCard", "card": defend},
        {"kind": "EndTurn"},
    ]
    return state, options


def test_checkpoint_roundtrip_and_serve_parity(tmp_path):
    m = model.ActorCritic(hidden=32, embed_dim=8)
    params = model.init_params(jax.random.PRNGKey(7), m)
    ckpt = str(tmp_path / "ck")
    model.save_checkpoint(ckpt, params, m)

    state, options = _combat_pair()

    # Direct model argmax over the legal options.
    g = features.encode_state(state)
    dense, card_idx, mask = features.encode_options(state, options)
    logits, _ = m.apply(params, g[None], dense[None], card_idx[None], mask[None])
    direct_argmax = int(np.argmax(np.asarray(logits[0])[:len(options)]))

    # Served policy argmax.
    policy = jax_policy.make_policy(ckpt)
    ranking = policy(state, options)
    assert len(ranking) == len(options)
    served_argmax = max(ranking, key=lambda p: p[1])[0]

    assert served_argmax == direct_argmax


def test_serve_declines_out_of_combat(tmp_path):
    m = model.ActorCritic(hidden=32, embed_dim=8)
    params = model.init_params(jax.random.PRNGKey(1), m)
    ckpt = str(tmp_path / "ck")
    model.save_checkpoint(ckpt, params, m)
    policy = jax_policy.make_policy(ckpt)
    assert policy({"phase": "Map"}, [{"kind": "MoveTo"}]) == []
