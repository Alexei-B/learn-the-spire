"""Unit tests for legal-action derivation (:mod:`lts2_agent.legal_actions`).

Synthetic states for the rule logic, plus checked-in real corpus fixtures that must reproduce the
recorded option set exactly.
"""

from __future__ import annotations

import json
import os

from lts2_agent import legal_actions, statefmt, tokens

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _card(card_id, **kw):
    c = {"cardId": card_id, "energyCost": kw.get("energyCost", 1), "costsX": kw.get("costsX", False),
         "type": kw.get("type", "Attack"), "rarity": kw.get("rarity", "Basic"),
         "targetType": kw.get("targetType", "AnyEnemy"), "upgraded": kw.get("upgraded", False),
         "poolId": "IRONCLAD_CARD_POOL", "canPlay": kw.get("canPlay", True),
         "starCost": kw.get("starCost", -1), "replayCount": kw.get("replayCount", 0),
         "addedKeywords": kw.get("addedKeywords", [])}
    for k in ("damage", "baseDamage", "block", "baseBlock", "summon"):
        if k in kw:
            c[k] = kw[k]
    return c


def _enemy(monster_id, hp, combat_id, hittable=True):
    return {"combatId": combat_id, "monsterId": monster_id, "currentHp": hp, "maxHp": hp + 10,
            "block": 0, "isHittable": hittable, "powers": [],
            "intents": [{"type": "Attack", "damage": 6, "baseDamage": 6, "hits": 1}]}


def _state(hand=None, enemies=None, pending=None, potions=None):
    cs = {"energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 2, "phase": "Play",
          "hand": hand or [], "drawPile": [], "discardPile": [], "exhaustPile": [],
          "powers": [], "orbs": [], "orbSlots": 0, "osty": None}
    player = {"netId": 1, "character": "IRONCLAD", "currentHp": 55, "maxHp": 72, "block": 4,
              "gold": 99, "maxEnergy": 3, "deck": [], "relics": [],
              "potions": potions if potions is not None else [], "combatState": cs}
    st = {"phase": "Combat" if pending is None else "Choice", "seed": "T-1", "actIndex": 1,
          "floor": 5, "ascensionLevel": 0, "isGameOver": False, "isVictory": False, "score": 1,
          "players": [player],
          "combat": {"roundNumber": 2, "currentSide": "Player", "enemies": enemies or []}}
    if pending is not None:
        st["pendingChoice"] = pending
    return st


def test_playcard_target_expansion_and_endturn():
    st = _state(hand=[_card("STRIKE_IRONCLAD", targetType="AnyEnemy", damage=6, baseDamage=6),
                      _card("DEFEND_IRONCLAD", type="Skill", targetType="Self", block=5, baseBlock=5),
                      _card("TWIN_STRIKE", targetType="AllEnemies", damage=8, baseDamage=8)],
                enemies=[_enemy("JawWorm", 40, combat_id=1), _enemy("Cultist", 30, combat_id=2)])
    keys = legal_actions.derive_option_keys(st)
    # Strike targets each of the two enemies; Defend/Cleave are single untargeted; EndTurn present.
    assert ("PlayCard", "STRIKE_IRONCLAD", 1) in keys
    assert ("PlayCard", "STRIKE_IRONCLAD", 2) in keys
    assert ("PlayCard", "DEFEND_IRONCLAD", None) in keys
    assert ("PlayCard", "TWIN_STRIKE", None) in keys
    assert ("EndTurn",) in keys


def test_unplayable_card_excluded():
    st = _state(hand=[_card("STRIKE_IRONCLAD", canPlay=False, damage=6, baseDamage=6)],
                enemies=[_enemy("JawWorm", 40, combat_id=1)])
    keys = legal_actions.derive_option_keys(st)
    assert not any(k[0] == "PlayCard" for k in keys)
    assert ("EndTurn",) in keys


def test_dead_enemy_not_targeted():
    st = _state(hand=[_card("STRIKE_IRONCLAD", targetType="AnyEnemy", damage=6, baseDamage=6)],
                enemies=[_enemy("JawWorm", 40, combat_id=1),
                         _enemy("Cultist", 0, combat_id=2, hittable=False)])
    keys = legal_actions.derive_option_keys(st)
    assert ("PlayCard", "STRIKE_IRONCLAD", 1) in keys
    assert ("PlayCard", "STRIKE_IRONCLAD", 2) not in keys


def test_potion_use_and_discard_and_empty_slot():
    # ATTACK_POTION exists in the catalog; a None slot is empty and must yield no option.
    pid = "ATTACK_POTION"
    st = _state(hand=[], enemies=[_enemy("JawWorm", 40, combat_id=1)], potions=[pid, None])
    keys = legal_actions.derive_option_keys(st)
    assert ("DiscardPotion", pid) in keys
    # No option for the empty slot.
    assert all(not (k[0] == "DiscardPotion" and k[1] == "") for k in keys)


def test_choice_single_select_options():
    pending = {"minSelect": 0, "maxSelect": 1, "isUpgradeSelection": False,
               "options": [_card("INFLAME", type="Power", targetType="Self"),
                           _card("TWIN_STRIKE", targetType="AllEnemies", damage=8, baseDamage=8)]}
    st = _state(pending=pending)
    keys = legal_actions.derive_option_keys(st)
    assert ("SelectCards", ("INFLAME",)) in keys
    assert ("SelectCards", ("TWIN_STRIKE",)) in keys
    assert ("SelectCards", ()) in keys   # skip, since minSelect == 0
    assert ("EndTurn",) not in keys       # a pending choice suppresses combat options


def test_option_key_from_wire_matches_derivation():
    play = {"kind": "PlayCard", "card": {"cardId": "STRIKE_IRONCLAD"}, "targetCombatId": 3}
    assert legal_actions.option_key(play) == ("PlayCard", "STRIKE_IRONCLAD", 3)
    assert legal_actions.option_key({"kind": "EndTurn"}) == ("EndTurn",)
    pot = {"kind": "UsePotion", "potionId": "FIRE_POTION", "targetCombatId": 2}
    assert legal_actions.option_key(pot) == ("UsePotion", "FIRE_POTION", 2)
    sel = {"kind": "SelectCards", "selectedCards": [{"cardId": "B"}, {"cardId": "A"}]}
    assert legal_actions.option_key(sel) == ("SelectCards", ("A", "B"))


def _load(name):
    return json.load(open(os.path.join(FIXTURES, name + ".json"), encoding="utf-8"))


def test_real_fixtures_derive_exactly():
    for name in ("combat_playcard", "combat_potion", "choice_select", "potion_target"):
        rec = _load(name)
        cv = statefmt.as_canonical(rec["state"])
        derived = legal_actions.derive_option_keys(cv)
        recorded = legal_actions.recorded_keys(rec["options"])
        assert derived == recorded, (
            name + " mismatch:\n  extra=" + str(derived - recorded)
            + "\n  missing=" + str(recorded - derived))


def test_real_fixture_via_token_roundtrip():
    # Derivation must also hold on the tokenized-then-detokenized state (what the model consumes).
    rec = _load("combat_playcard")
    cv = statefmt.from_tokens(rec["state"])
    derived = legal_actions.derive_option_keys(cv)
    recorded = legal_actions.recorded_keys(rec["options"])
    assert derived == recorded


def test_tally_prf():
    t = legal_actions.Tally()
    t.add({("EndTurn",), ("PlayCard", "A", 1)}, {("EndTurn",), ("PlayCard", "A", 2)}, "Combat")
    # 1 TP (EndTurn), 1 FP (A->1), 1 FN (A->2)
    assert t.tp == 1 and t.fp == 1 and t.fn == 1
    prec, rec, f1 = legal_actions._prf(t.tp, t.fp, t.fn)
    assert abs(prec - 0.5) < 1e-9 and abs(rec - 0.5) < 1e-9
