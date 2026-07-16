"""Unit tests for the entity tokenizer (:mod:`lts2_agent.tokens`) — synthetic states only, no C# host.

Covers: token array shapes/masks, draw-pile order invariance (shuffle can't leak), exact round-trip
incl. a pendingChoice, version/signature stability, and the catalog hashing fallback path.
"""

from __future__ import annotations

import copy
import random

import numpy as np

from lts2_agent import catalog, tokens


def _card(card_id, **kw):
    c = {"cardId": card_id, "energyCost": kw.get("energyCost", 1), "costsX": kw.get("costsX", False),
         "type": kw.get("type", "Attack"), "rarity": kw.get("rarity", "Basic"),
         "targetType": kw.get("targetType", "AnyEnemy"), "upgraded": kw.get("upgraded", False),
         "poolId": "IRONCLAD_CARD_POOL", "canPlay": kw.get("canPlay", True),
         "starCost": kw.get("starCost", 0), "replayCount": kw.get("replayCount", 0),
         "addedKeywords": kw.get("addedKeywords", [])}
    for k in ("damage", "baseDamage", "block", "baseBlock", "summon", "enchantmentId",
              "afflictionId"):
        if k in kw:
            c[k] = kw[k]
    return c


def _state(hand=None, draw=None, discard=None, exhaust=None, enemies=None, pending=None,
           powers=None, relics=None, potions=None, orbs=None, osty=None):
    cs = {"energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 2, "phase": "Play",
          "hand": hand or [], "drawPile": draw or [], "discardPile": discard or [],
          "exhaustPile": exhaust or [], "powers": powers or [], "orbs": orbs or [],
          "orbSlots": 3 if orbs else 0, "osty": osty}
    player = {"netId": 1, "character": "IRONCLAD", "currentHp": 55, "maxHp": 72, "block": 4,
              "gold": 99, "maxEnergy": 3, "deck": [], "relics": relics or [],
              "potions": potions if potions is not None else [], "combatState": cs}
    st = {"phase": "Combat", "seed": "T-1", "actIndex": 1, "floor": 5, "ascensionLevel": 0,
          "isGameOver": False, "isVictory": False, "score": 123, "players": [player],
          "combat": {"roundNumber": 2, "currentSide": "Player", "enemies": enemies or []}}
    if pending is not None:
        st["pendingChoice"] = pending
    return st


def _enemy(monster_id, hp, combat_id=100, intents=None, powers=None):
    return {"combatId": combat_id, "monsterId": monster_id, "currentHp": hp, "maxHp": hp + 10,
            "block": 0, "isHittable": True, "powers": powers or [],
            "intents": intents or [{"type": "Attack", "damage": 6, "baseDamage": 6, "hits": 2}]}


def test_shapes_and_masks():
    st = _state(hand=[_card("StrikeIronclad", damage=6, baseDamage=6),
                      _card("DefendIronclad", type="Skill", targetType="Self", block=5, baseBlock=5)],
                draw=[_card("Bash", damage=8, baseDamage=8)],
                enemies=[_enemy("JawWorm", 40)],
                powers=[{"powerId": "StrengthPower", "amount": 2}])
    tok = tokens.tokenize(st)
    # Every declared key present.
    for k in tokens.TOKEN_KEYS:
        assert k in tok, k
    assert tok["card_idx"].shape == (tokens.MAX_CARDS, len(tokens.CARD_IDX))
    assert tok["card_num"].shape == (tokens.MAX_CARDS, len(tokens.CARD_NUM))
    assert tok["card_kw"].shape == (tokens.MAX_CARDS, tokens.KW_BUCKETS)
    assert tok["card_mask"].shape == (tokens.MAX_CARDS,)
    assert tok["creature_idx"].shape == (tokens.MAX_CREATURES, len(tokens.CREATURE_IDX))
    assert tok["global_idx"].shape == (1, len(tokens.GLOBAL_IDX))
    assert tok["global_num"].shape == (1, len(tokens.GLOBAL_NUM))
    # 3 cards, 2 creatures (player+enemy), 2 powers (player strength + none-listed here).
    assert int(tok["card_mask"].sum()) == 3
    assert int(tok["creature_mask"].sum()) == 2
    assert int(tok["power_mask"].sum()) == 1
    assert int(tok["intent_mask"].sum()) == 1
    # token-type ids are the fixed enum values.
    assert int(tok["token_type_card"]) == tokens.TOKEN_TYPE_ID["card"]
    assert int(tok["token_type_creature"]) == tokens.TOKEN_TYPE_ID["creature"]


def test_draw_pile_is_unordered_multiset():
    draw = [_card("StrikeIronclad", damage=6, baseDamage=6),
            _card("DefendIronclad", type="Skill", targetType="Self", block=5, baseBlock=5),
            _card("Bash", damage=8, baseDamage=8),
            _card("StrikeIronclad", damage=6, baseDamage=6)]
    base = _state(draw=list(draw), enemies=[_enemy("JawWorm", 40)])
    t0 = tokens.tokenize(base)
    for seed in range(6):
        shuffled = list(draw)
        random.Random(seed).shuffle(shuffled)
        t = tokens.tokenize(_state(draw=shuffled, enemies=[_enemy("JawWorm", 40)]))
        for k in ("card_idx", "card_num", "card_kw", "card_mask"):
            assert np.array_equal(t0[k], t[k]), (k, seed)


def test_round_trip_exact_with_pending_choice():
    pending = {"minSelect": 0, "maxSelect": 999999999, "isUpgradeSelection": False,
               "options": [_card("Inflame", type="Power", targetType="Self"),
                           _card("Cleave", type="Attack", targetType="AllEnemies", damage=8,
                                 baseDamage=8)]}
    st = _state(hand=[_card("StrikeIronclad", damage=9, baseDamage=6,
                            addedKeywords=["Retain", "Ethereal"])],
                draw=[_card("DefendIronclad", type="Skill", targetType="Self", block=5, baseBlock=5)],
                discard=[_card("Bash", damage=8, baseDamage=8)],
                exhaust=[_card("Anger")],
                enemies=[_enemy("JawWorm", 40, powers=[{"powerId": "StrengthPower", "amount": 3}],
                                intents=[{"type": "Attack", "damage": 12, "baseDamage": 10, "hits": 1},
                                         {"type": "Buff"}])],
                powers=[{"powerId": "StrengthPower", "amount": 2},
                        {"powerId": "VulnerablePower", "amount": -1}],
                relics=["BURNING_BLOOD", "AKABEKO"], potions=["ATTACK_POTION", None],
                orbs=[{"orbId": "Lightning", "passiveValue": 3, "evokeValue": 8}],
                osty={"currentHp": 20, "maxHp": 30, "block": 2, "isAlive": True,
                      "powers": [{"powerId": "StrengthPower", "amount": 1}]},
                pending=pending)
    ok, diff = tokens.round_trip(st)
    assert ok, "round-trip mismatch at " + str(diff)
    # The 999999999 'no-limit' sentinel clamps to NUM_CLIP and round-trips exactly to the clamp.
    got = tokens.detokenize(tokens.tokenize(st))
    assert got["pending"]["maxSelect"] == tokens.NUM_CLIP


def test_coverage_no_lost_fields():
    st = _state(hand=[_card("StrikeIronclad", damage=6, baseDamage=6)],
                enemies=[_enemy("JawWorm", 40)])
    covered, waived, lost = tokens.coverage_check(st)
    assert not lost, "unexpected lost fields: " + str(sorted(lost))
    assert "state/players[]/combatState/hand[]/cardId" in covered
    assert "state/seed" in waived


def test_null_potion_slot_round_trips():
    st = _state(potions=[None, "ATTACK_POTION", None])
    got = tokens.detokenize(tokens.tokenize(st))
    # slot order preserved, empty slots -> index 0, the real potion -> its catalog index.
    assert got["potions"][0] == 0 and got["potions"][2] == 0
    assert got["potions"][1] == catalog.load("potions").index_of("ATTACK_POTION")


def test_version_and_signature_stable():
    assert isinstance(tokens.TOKENIZER_VERSION, int)
    sig = tokens.tokenizer_signature()
    assert sig == tokens.tokenizer_signature()
    assert sig.startswith("tok-v" + str(tokens.TOKENIZER_VERSION))
    for kind in ("cards", "powers", "relics", "potions"):
        assert kind in tokens.CATALOG_SIGNATURES


def test_catalog_hash_fallback():
    fb = catalog.HashFallback("cards")
    assert fb.size == catalog.FALLBACK_VOCAB
    assert fb.static_dim == 0
    assert fb.index_of("") == 0
    assert fb.index_of("SomeUnknownCard") == fb.index_of("SomeUnknownCard")  # stable
    assert 1 <= fb.index_of("SomeUnknownCard") < fb.size
    assert fb.id_of(5) == ""  # not invertible
    assert fb.signature == "cards-hash"


def test_entity_catalog_round_trips_ids():
    cat = catalog.load("relics")
    if isinstance(cat, catalog.HashFallback):
        return  # dump absent on this clone; nothing to assert
    # index_of / id_of are exact inverses for real ids.
    some_id = cat.id_of(1)
    assert some_id and cat.index_of(some_id) == 1
    assert cat.index_of("NOT_A_REAL_RELIC") == 0


def test_tokenize_does_not_mutate_state():
    st = _state(hand=[_card("StrikeIronclad", damage=6, baseDamage=6)],
                enemies=[_enemy("JawWorm", 40)])
    before = copy.deepcopy(st)
    tokens.tokenize(st)
    assert st == before
