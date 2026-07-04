"""Encoder contract tests: fixed shapes, stable buckets, correct masking/lethality."""

from __future__ import annotations

import numpy as np

from lts2_agent import features as F


def _combat_state():
    return {
        "phase": "Combat",
        "floor": 3,
        "actIndex": 0,
        "players": [{
            "currentHp": 60, "maxHp": 80, "block": 5, "gold": 99, "maxEnergy": 3,
            "combatState": {
                "energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 1,
                "hand": [{"cardId": "BONE_SHARDS"}], "drawPile": [], "discardPile": [],
                "exhaustPile": [], "powers": [{"powerId": "Strength", "amount": 2}],
                "orbs": [], "orbSlots": 0,
            },
        }],
        "combat": {"enemies": [
            {"combatId": 1, "monsterId": "Cultist", "currentHp": 6, "maxHp": 48, "block": 0,
             "isHittable": True, "powers": [], "intents": [{"type": "Attack", "damage": 6, "hits": 1}]},
            {"combatId": 2, "monsterId": "JawWorm", "currentHp": 40, "maxHp": 40, "block": 3,
             "isHittable": True, "powers": [], "intents": [{"type": "Attack", "damage": 11, "hits": 1}]},
        ]},
    }


def _combat_options():
    strike = {"cardId": "BONE_SHARDS", "type": "Attack", "energyCost": 1, "damage": 9, "upgraded": False}
    return [
        {"kind": "PlayCard", "card": strike, "targetCombatId": 1, "handIndex": 0},   # lethal on 6hp Cultist
        {"kind": "PlayCard", "card": strike, "targetCombatId": 2, "handIndex": 0},   # not lethal (40hp+3blk)
        {"kind": "EndTurn"},
    ]


def test_dims_are_fixed_and_positive():
    assert F.STATE_DIM > 0 and F.OPTION_DIM > 0
    g = F.encode_state(_combat_state())
    assert g.shape == (F.STATE_DIM,)
    assert g.dtype == np.float32


def test_encode_options_shapes_and_mask():
    state, options = _combat_state(), _combat_options()
    dense, card_idx, mask = F.encode_options(state, options)
    assert dense.shape == (F.MAX_OPTIONS, F.OPTION_DIM)
    assert card_idx.shape == (F.MAX_OPTIONS,)
    assert mask.shape == (F.MAX_OPTIONS,)
    assert mask[:3].all() and not mask[3:].any()      # exactly the 3 real options are legal
    assert card_idx[0] > 0 and card_idx[2] == 0        # PlayCard has a card bucket; EndTurn does not


def test_card_index_is_stable():
    base = {"cardId": "BONE_SHARDS", "upgraded": False}
    up = {"cardId": "BONE_SHARDS", "upgraded": True}
    assert F.card_index(base) == F.card_index(base)     # deterministic
    assert F.card_index(base) > 0                        # a real card maps to a positive index
    assert F.card_index(base) == F.card_index(up)        # keyed by base id (upgrade is a separate feature)
    assert F.card_index(None) == 0
    if F._CATALOG is not None:                           # with the catalog: distinct cards, unknown -> 0
        assert F.card_index({"cardId": "FETCH"}) != F.card_index(base)
        assert F.card_index({"cardId": "not_a_real_card_xyz"}) == 0


# Option-vector tail layout: ..., is_targeted, lethal, is_weakest, target_hp, target_block,
#                            is_aoe, num_targets, total_damage, kills
_LETHAL = -8
_WEAKEST = -7
_IS_AOE = -4
_TOTAL_DMG = -2
_KILLS = -1


def test_lethal_and_weakest_flags():
    state, options = _combat_state(), _combat_options()
    dense, _, _ = F.encode_options(state, options)
    assert dense[0, _LETHAL] == 1.0    # 9 dmg >= 6 hp + 0 block on the Cultist
    assert dense[1, _LETHAL] == 0.0    # 9 dmg < 40 hp + 3 block on the JawWorm
    assert dense[0, _WEAKEST] == 1.0   # Cultist (6hp) is the weakest
    assert dense[1, _WEAKEST] == 0.0


def test_aoe_option_captures_total_damage_and_kills():
    state = _combat_state()   # two enemies: Cultist 6hp, JawWorm 40hp(+3 block)
    sow = {"cardId": "Sow", "type": "Attack", "targetType": "AllEnemies", "energyCost": 1, "damage": 8}
    dense, _, _ = F.encode_options(state, [{"kind": "PlayCard", "card": sow}])  # untargeted AoE
    assert dense[0, _IS_AOE] == 1.0
    assert dense[0, _TOTAL_DMG] == np.float32((8 * 2) / 30.0)   # hits both live enemies (16 total)
    assert dense[0, _KILLS] == np.float32(1 / 5.0)              # kills the 6-hp Cultist (8>=6), not JawWorm


def test_incoming_damage():
    assert F.incoming_damage(_combat_state()) == 17   # 6 + 11


def test_non_combat_state_encodes_without_error():
    g = F.encode_state({"phase": "Map", "players": [], "floor": 1})
    assert g.shape == (F.STATE_DIM,)
    assert not F.is_combat({"phase": "Map"})
