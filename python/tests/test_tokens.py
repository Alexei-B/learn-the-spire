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
    assert tokens.TOKENIZER_VERSION == 3  # v3 = factored population rows (zone-count vector)
    sig = tokens.tokenizer_signature()
    assert sig == tokens.tokenizer_signature()
    assert sig.startswith("tok-v" + str(tokens.TOKENIZER_VERSION))
    for kind in ("cards", "powers", "relics", "potions"):
        assert kind in tokens.CATALOG_SIGNATURES


# --------------------------------------------------------------------------------------------------
# v3: factored population rows — one row per card CONTENT, with a per-zone count vector (zone removed
# from the grouping key).
# --------------------------------------------------------------------------------------------------

def _count_cols():
    return [tokens.CARD_NUM.index(f) for f in tokens.ZONE_COUNT_FIELDS]


def _row_zone_counts(tok):
    """For each present card row: a dict {zone: count} decoded from the count-vector columns."""
    cols = {z: tokens.CARD_NUM.index("count_" + z) for z in tokens.ZONES}
    rows = []
    for i in range(tokens.MAX_CARDS):
        if not tok["card_mask"][i]:
            continue
        rows.append({z: max(0, tokens._int(tok["card_num"][i, c])) for z, c in cols.items()})
    return rows


def test_count_vector_columns_exist():
    # v3: zone left CARD_IDX; the five count_<zone> columns are the CARD_NUM tail.
    assert "zone" not in tokens.CARD_IDX
    assert tokens.ZONE_COUNT_FIELDS == ["count_" + z for z in tokens.ZONES]
    assert tokens.CARD_NUM[-len(tokens.ZONES):] == tokens.ZONE_COUNT_FIELDS


def test_identical_cards_group_into_one_row_with_zone_count():
    draw = [_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(5)]
    st = _state(draw=draw, enemies=[_enemy("JawWorm", 40)])
    tok = tokens.tokenize(st)
    # Five identical strikes collapse to ONE population row carrying count_draw = 5.
    assert int(tok["card_mask"].sum()) == 1
    counts = _row_zone_counts(tok)[0]
    assert counts["draw"] == 5
    assert sum(counts.values()) == 5


def test_same_card_in_three_zones_is_one_row_with_zone_counts():
    # THE core v3 change: identical content in hand + draw + discard -> ONE row, counts spread by zone.
    st = _state(hand=[_card("StrikeIronclad", damage=6, baseDamage=6)],
                draw=[_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(2)],
                discard=[_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(2)],
                enemies=[_enemy("JawWorm", 40)])
    tok = tokens.tokenize(st)
    assert int(tok["card_mask"].sum()) == 1
    counts = _row_zone_counts(tok)[0]
    assert counts == {"hand": 1, "draw": 2, "discard": 2, "exhaust": 0, "offered": 0}


def test_cross_zone_live_field_divergence_stays_separate_rows():
    # A cost-reduced copy in hand vs its full-cost twin in draw differ in a live field (energyCost),
    # so they remain TWO distinct population rows — divergence is correct, not merged.
    st = _state(hand=[_card("StrikeIronclad", energyCost=0, damage=6, baseDamage=6)],
                draw=[_card("StrikeIronclad", energyCost=1, damage=6, baseDamage=6)],
                enemies=[_enemy("JawWorm", 40)])
    tok = tokens.tokenize(st)
    assert int(tok["card_mask"].sum()) == 2
    rows = _row_zone_counts(tok)
    # One row lives entirely in hand, the other entirely in draw (no merge).
    zones_used = sorted(tuple(sorted(z for z, n in r.items() if n)) for r in rows)
    assert zones_used == [("draw",), ("hand",)]


def test_mixed_upgrades_stay_separate_rows():
    # 5 plain strikes + 2 upgraded strikes (different content) -> two rows, count_draw 5 and 2.
    draw = [_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(5)]
    draw += [_card("StrikeIronclad", damage=9, baseDamage=6, upgraded=True) for _ in range(2)]
    st = _state(draw=draw, enemies=[_enemy("JawWorm", 40)])
    tok = tokens.tokenize(st)
    assert int(tok["card_mask"].sum()) == 2
    assert sorted(r["draw"] for r in _row_zone_counts(tok)) == [2, 5]


def test_detokenize_expands_zone_counts_exactly():
    # Duplicates spread across zones round-trip exactly through zone-count expansion.
    strikes_hand = [_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(3)]
    strikes_draw = [_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(7)]
    defends = [_card("DefendIronclad", type="Skill", targetType="Self", block=5, baseBlock=5)
               for _ in range(4)]
    discard = [_card("Bash", damage=8, baseDamage=8) for _ in range(3)]
    st = _state(hand=strikes_hand, draw=strikes_draw + defends, discard=discard,
                enemies=[_enemy("JawWorm", 40)])
    ok, diff = tokens.round_trip(st)
    assert ok, "round-trip mismatch at " + str(diff)
    got = tokens.detokenize(tokens.tokenize(st))
    # Instances restored per zone exactly (strikes merge across hand+draw into one row).
    assert len(got["cards"]["hand"]) == 3
    assert len(got["cards"]["draw"]) == 11
    assert len(got["cards"]["discard"]) == 3
    # Strikes are ONE row (hand+draw) + Defend row + Bash row = 3 rows for 21 instances.
    assert int(tokens.tokenize(st)["card_mask"].sum()) == 3


def test_grouping_preserves_shuffle_invariance():
    # Grouping does not reintroduce order sensitivity: any shuffle of a duplicate-heavy pile is identical.
    draw = ([_card("StrikeIronclad", damage=6, baseDamage=6)] * 4
            + [_card("DefendIronclad", type="Skill", targetType="Self", block=5, baseBlock=5)] * 3
            + [_card("Bash", damage=8, baseDamage=8)] * 2)
    base = tokens.tokenize(_state(draw=list(draw), enemies=[_enemy("JawWorm", 40)]))
    for seed in range(6):
        shuffled = list(draw)
        random.Random(seed).shuffle(shuffled)
        t = tokens.tokenize(_state(draw=shuffled, enemies=[_enemy("JawWorm", 40)]))
        for k in ("card_idx", "card_num", "card_kw", "card_mask"):
            assert np.array_equal(base[k], t[k]), (k, seed)


def test_numeric_ranges_present_and_clamp():
    # v3 exactness contract: the spec carries a measured integer range per numeric column, and
    # out-of-range values clamp loudly.
    from lts2_agent.wm import spec as S
    assert S.NUMERIC_RANGES, "spec must carry measured per-field ranges"
    # Core fields every corpus has.
    energy = S.NUMERIC_RANGES["global"]["energy"]
    assert energy.lo <= 0 <= energy.hi and energy.resolution >= 1
    assert energy.n_bins == (energy.hi - energy.lo) // energy.resolution + 1
    hp = S.NUMERIC_RANGES["creature"]["currentHp"]
    # An absurd HP clamps to the measured hi and reports the clamp; an in-range value passes through.
    clamped, was = S.clamp_to_range("creature", "currentHp", hp.hi + 10_000)
    assert clamped == hp.hi and was is True
    ok_val, was2 = S.clamp_to_range("creature", "currentHp", hp.hi)
    assert ok_val == hp.hi and was2 is False
    # Card per-zone count columns carry ranges too (the v3 population vector).
    assert "count_draw" in S.NUMERIC_RANGES["card"]


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
