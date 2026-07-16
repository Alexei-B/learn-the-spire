"""Unit tests for the canonical-state pretty-printer + diff (:mod:`lts2_agent.statefmt`).

Synthetic states only, no C# host; plus a couple of checked-in real corpus fixtures.
"""

from __future__ import annotations

import json
import os

from lts2_agent import statefmt, tokens

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _card(card_id, **kw):
    c = {"cardId": card_id, "energyCost": kw.get("energyCost", 1), "costsX": kw.get("costsX", False),
         "type": kw.get("type", "Attack"), "rarity": kw.get("rarity", "Basic"),
         "targetType": kw.get("targetType", "AnyEnemy"), "upgraded": kw.get("upgraded", False),
         "poolId": "IRONCLAD_CARD_POOL", "canPlay": kw.get("canPlay", True),
         "starCost": kw.get("starCost", -1), "replayCount": kw.get("replayCount", 0),
         "addedKeywords": kw.get("addedKeywords", [])}
    for k in ("damage", "baseDamage", "block", "baseBlock", "summon", "enchantmentId",
              "afflictionId"):
        if k in kw:
            c[k] = kw[k]
    return c


def _enemy(monster_id, hp, combat_id=1, intents=None, powers=None, hittable=True):
    return {"combatId": combat_id, "monsterId": monster_id, "currentHp": hp, "maxHp": hp + 10,
            "block": 0, "isHittable": hittable, "powers": powers or [],
            "intents": intents or [{"type": "Attack", "damage": 6, "baseDamage": 6, "hits": 1}]}


def _state(hand=None, draw=None, discard=None, exhaust=None, enemies=None, pending=None,
           powers=None, relics=None, potions=None, currentHp=55):
    cs = {"energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 2, "phase": "Play",
          "hand": hand or [], "drawPile": draw or [], "discardPile": discard or [],
          "exhaustPile": exhaust or [], "powers": powers or [], "orbs": [], "orbSlots": 0,
          "osty": None}
    player = {"netId": 1, "character": "IRONCLAD", "currentHp": currentHp, "maxHp": 72, "block": 4,
              "gold": 99, "maxEnergy": 3, "deck": [], "relics": relics or [],
              "potions": potions if potions is not None else [], "combatState": cs}
    st = {"phase": "Combat" if pending is None else "Choice", "seed": "T-1", "actIndex": 1,
          "floor": 5, "ascensionLevel": 0, "isGameOver": False, "isVictory": False, "score": 1,
          "players": [player],
          "combat": {"roundNumber": 2, "currentSide": "Player", "enemies": enemies or []}}
    if pending is not None:
        st["pendingChoice"] = pending
    return st


def test_format_state_renders_key_fields():
    st = _state(hand=[_card("STRIKE_IRONCLAD", damage=6, baseDamage=6),
                      _card("DEFEND_IRONCLAD", type="Skill", targetType="Self", block=5, baseBlock=5)],
                draw=[_card("BASH", damage=8, baseDamage=8), _card("BASH", damage=8, baseDamage=8)],
                enemies=[_enemy("JawWorm", 40)],
                powers=[{"powerId": "DEXTERITY_POWER", "amount": 2}],
                relics=["BURNING_BLOOD"])
    out = statefmt.format_state(st)
    assert "Combat" in out
    assert "energy 3/3" in out
    assert "STRIKE_IRONCLAD" in out
    assert "DEFEND_IRONCLAD" in out
    assert "Hand (2)" in out
    assert "BASH x2" in out           # draw multiset counted
    assert "HP 55/72" in out          # player hp
    assert "DEXTERITY_POWER+2" in out   # power rendered with amount
    assert "BURNING_BLOOD" in out


def test_format_state_accepts_canonical_and_wire():
    st = _state(hand=[_card("STRIKE_IRONCLAD", damage=6, baseDamage=6)],
                enemies=[_enemy("JawWorm", 40)])
    wire_out = statefmt.format_state(st)
    canon_out = statefmt.format_state(statefmt.as_canonical(st))
    tok_out = statefmt.format_state(statefmt.from_tokens(st))
    # Same rendering whichever canonical source it comes from.
    assert wire_out == canon_out == tok_out


def test_hash_names_resolve_monster():
    st = _state(enemies=[_enemy("JawWorm", 40)])
    # Without a map, the monster shows as an opaque bucket.
    bare = statefmt.format_state(st)
    assert "JawWorm" not in bare and "#" in bare
    # With a map built for the bucket "JawWorm" hashes to, the name shows.
    bucket = statefmt._monster_bucket("JawWorm")
    hn = {"monster": {str(bucket): ["JawWorm"]}}
    named = statefmt.format_state(st, hn)
    assert "JawWorm" in named


def test_diff_hp_block_and_moved_card_and_power():
    a = _state(hand=[_card("STRIKE_IRONCLAD", damage=6, baseDamage=6),
                     _card("DEFEND_IRONCLAD", type="Skill", targetType="Self")],
               discard=[],
               enemies=[_enemy("JawWorm", 40)],
               currentHp=55)
    # b: player took damage + gained block, Strike moved hand->discard, enemy lost HP + gained power.
    b = _state(hand=[_card("DEFEND_IRONCLAD", type="Skill", targetType="Self")],
               discard=[_card("STRIKE_IRONCLAD", damage=6, baseDamage=6)],
               enemies=[_enemy("JawWorm", 32, powers=[{"powerId": "VULNERABLE_POWER", "amount": 2}])],
               currentHp=48)
    b["players"][0]["block"] = 9
    out = statefmt.diff_states(a, b)
    assert "STRIKE_IRONCLAD" in out                 # the moved card appears
    assert "hand:" in out and "discard:" in out     # both zones report the move
    assert "HP 40->32" in out or "HP 40->32 (-8)" in out
    assert "VULNERABLE_POWER" in out                 # new power detected
    # player HP change captured
    assert "48" in out


def test_diff_enemy_died():
    a = _state(enemies=[_enemy("JawWorm", 5, combat_id=1)])
    b = _state(enemies=[_enemy("JawWorm", 0, combat_id=1, hittable=False)])
    out = statefmt.diff_states(a, b)
    assert "DIED" in out


def test_diff_no_changes():
    st = _state(hand=[_card("STRIKE_IRONCLAD", damage=6, baseDamage=6)],
                enemies=[_enemy("JawWorm", 40)])
    out = statefmt.diff_states(st, st)
    assert "no changes" in out


def test_build_hash_names_helper_collects_and_buckets():
    st = _state(enemies=[_enemy("JawWorm", 40)],
                hand=[_card("STRIKE_IRONCLAD", addedKeywords=["Retain"])])
    acc = {v: set() for v in statefmt._HASH_VOCABS}
    statefmt._collect_strings(st, acc)
    assert "JawWorm" in acc["monster"]
    assert "IRONCLAD" in acc["character"]
    assert "Retain" in acc["keyword"]


def test_real_fixtures_render():
    for name in ("combat_playcard", "choice_select", "potion_target"):
        rec = json.load(open(os.path.join(FIXTURES, name + ".json"), encoding="utf-8"))
        out = statefmt.format_state(rec["state"])
        assert out.startswith("===")
        assert "Hand" in out or "Offered" in out
        if rec.get("nextState"):
            d = statefmt.diff_states(rec["state"], rec["nextState"])
            assert d.startswith("=== DIFF")
