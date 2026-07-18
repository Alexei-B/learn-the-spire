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
    # v4: the belt LEFT-PACKS — a rare non-left-packed raw belt [empty, ATTACK, empty] canonicalizes to
    # [ATTACK, empty, empty], preserving belt SIZE (3 slots). Slot identity is decision-irrelevant.
    st = _state(potions=[None, "ATTACK_POTION", None])
    got = tokens.detokenize(tokens.tokenize(st))
    assert len(got["potions"]) == 3            # belt size preserved
    assert got["potions"][0] == catalog.load("potions").index_of("ATTACK_POTION")
    assert got["potions"][1] == 0 and got["potions"][2] == 0   # empties trail


def test_potion_belt_is_left_pack_canonical_over_permutations():
    # THE potions well-posedness property: every wire permutation of the same belt multiset tokenizes to
    # ONE canonical (left-packed, id-sorted) potion array — byte-identical tokens for equal multisets.
    belt = ["ATTACK_POTION", None, "BLOCK_POTION", None]
    base = tokens.tokenize(_state(potions=list(belt)))
    for seed in range(6):
        perm = list(belt)
        random.Random(seed).shuffle(perm)
        t = tokens.tokenize(_state(potions=perm))
        assert np.array_equal(base["potion_idx"], t["potion_idx"]), seed
        assert np.array_equal(base["potion_mask"], t["potion_mask"]), seed
    # Non-empty potions come first, sorted by catalog index; empties (id 0) trail; size preserved.
    got = tokens.detokenize(base)
    ids = got["potions"]
    assert len(ids) == 4
    nonempty = [x for x in ids if x != 0]
    assert nonempty == sorted(nonempty) and ids[len(nonempty):] == [0] * (4 - len(nonempty))


def test_orb_tokens_carry_explicit_slot_position():
    # v4: orbs gain a `slot` categorical (belt position) so the permutation-invariant orb expert can
    # represent evoke order. Present orb i has slot == i, and the round-trip preserves it exactly.
    st = _state(orbs=[{"orbId": "Lightning", "passiveValue": 3, "evokeValue": 8},
                      {"orbId": "Frost", "passiveValue": 2, "evokeValue": 5},
                      {"orbId": "Dark", "passiveValue": 6, "evokeValue": 12}])
    tok = tokens.tokenize(st)
    slot_col = tokens.ORB_IDX.index("slot")
    assert "slot" in tokens.ORB_IDX
    present = int(tok["orb_mask"].sum())
    assert present == 3
    assert list(tok["orb_idx"][:present, slot_col]) == [0, 1, 2]
    ok, diff = tokens.round_trip(st)
    assert ok, "orb round-trip mismatch at " + str(diff)
    got = tokens.detokenize(tok)
    assert [o["slot"] for o in got["orbs"]] == [0, 1, 2]


def _relic_id(i):
    return catalog.load("relics").id_of(i)


def test_relics_positional_keep_order_and_duplicates():
    # v5: relics are positional — one row per instance, wire order preserved, duplicates kept apart by
    # their slot; the `slot` categorical equals the acquisition index.
    a, b, c = _relic_id(5), _relic_id(9), _relic_id(5)   # a and c are the SAME relic (a duplicate)
    st = _state(relics=[a, b, c])
    tok = tokens.tokenize(st)
    m = tok["relic_mask"]
    rows = tok["relic_idx"][m].tolist()                  # [[id, slot], ...] in wire order
    assert rows == [[5, 0], [9, 1], [5, 2]]              # duplicate id 5 at slots 0 and 2, order kept
    # Round-trips exactly to the flat wire-order id list (with the duplicate).
    got = tokens.detokenize(tok)["relics"]
    assert got == [5, 9, 5]
    ok, diff = tokens.round_trip(st)
    assert ok, diff


def test_relic_order_is_semantic():
    # Reversing acquisition order changes the tokens (order carries the wax-relic-expiry state).
    a, b = _relic_id(5), _relic_id(9)
    t1 = tokens.tokenize(_state(relics=[a, b]))
    t2 = tokens.tokenize(_state(relics=[b, a]))
    assert not np.array_equal(t1["relic_idx"], t2["relic_idx"])


def test_relic_overflow_is_loud():
    a = _relic_id(5)
    big = _state(relics=[a] * (tokens.MAX_RELICS + 1))
    try:
        tokens.tokenize(big, strict=True)
        assert False, "expected a TokenOverflow past MAX_RELICS"
    except tokens.TokenOverflow as e:
        assert "relics" in str(e)
    # Non-strict truncates to the cap rather than raising.
    tok = tokens.tokenize(big, strict=False)
    assert int(tok["relic_mask"].sum()) == tokens.MAX_RELICS


def test_version_and_signature_stable():
    assert isinstance(tokens.TOKENIZER_VERSION, int)
    assert tokens.TOKENIZER_VERSION == 6  # v6 = cards positional instance rows (zone + slot; no counts)
    sig = tokens.tokenizer_signature()
    assert sig == tokens.tokenizer_signature()
    assert sig.startswith("tok-v" + str(tokens.TOKENIZER_VERSION))
    for kind in ("cards", "powers", "relics", "potions"):
        assert kind in tokens.CATALOG_SIGNATURES


# --------------------------------------------------------------------------------------------------
# v6: card INSTANCE rows — one token row per physical copy, laid out zone-major with within-zone content
# sort + a `slot` == layout-index column and a `zone` categorical. No per-zone count vector.
# --------------------------------------------------------------------------------------------------

_ZONE_COL = tokens.CARD_IDX.index("zone")
_SLOT_COL = tokens.CARD_IDX.index("slot")


def _present_rows(tok):
    """List of present card rows as (zone_name, slot, cardIndex) tuples, in array order."""
    rows = []
    for i in range(tokens.MAX_CARDS):
        if not tok["card_mask"][i]:
            continue
        zi = int(tok["card_idx"][i, _ZONE_COL])
        rows.append((tokens.ZONES[zi], int(tok["card_idx"][i, _SLOT_COL]),
                     int(tok["card_idx"][i, 0])))
    return rows


def test_card_layout_columns_v6():
    # v6: zone + slot are categorical columns; the numeric block is exactly the 14 content numerics.
    assert tokens.CARD_IDX == ["cardIndex", "type", "rarity", "targetType", "enchant", "afflict",
                               "zone", "slot"]
    assert len(tokens.CARD_NUM) == 14
    assert not hasattr(tokens, "ZONE_COUNT_FIELDS")
    assert not hasattr(tokens, "_group_cards")


def test_identical_cards_are_separate_instance_rows():
    draw = [_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(5)]
    st = _state(draw=draw, enemies=[_enemy("JawWorm", 40)])
    tok = tokens.tokenize(st)
    # Five identical strikes -> FIVE instance rows (no grouping), all in the draw zone.
    assert int(tok["card_mask"].sum()) == 5
    rows = _present_rows(tok)
    assert all(z == "draw" for z, _slot, _ci in rows)


def test_zone_major_layout_and_slot_index():
    # Rows are laid out ZONE-MAJOR (fixed ZONES order) and slot == the row's index in that layout.
    st = _state(hand=[_card("StrikeIronclad", damage=6, baseDamage=6)],
                draw=[_card("Bash", damage=8, baseDamage=8) for _ in range(2)],
                discard=[_card("DefendIronclad", type="Skill", targetType="Self", block=5, baseBlock=5)],
                enemies=[_enemy("JawWorm", 40)])
    tok = tokens.tokenize(st)
    rows = _present_rows(tok)
    # slot == array index for every present row.
    assert [slot for _z, slot, _ci in rows] == list(range(len(rows)))
    # Zone order is the fixed ZONES order (hand, then draw, then discard here).
    zone_seq = [z for z, _slot, _ci in rows]
    assert zone_seq == ["hand", "draw", "draw", "discard"]


def test_within_zone_content_sorted():
    # Within a zone, instances are content-sorted by _card_content_key (shuffle-invariant).
    draw = [_card("StrikeIronclad", damage=6, baseDamage=6),
            _card("Bash", damage=8, baseDamage=8),
            _card("Anger", damage=6, baseDamage=6)]
    base = tokens.tokenize(_state(draw=list(draw), enemies=[_enemy("JawWorm", 40)]))
    keys = []
    for i in range(tokens.MAX_CARDS):
        if not base["card_mask"][i]:
            continue
        c = {tokens.CARD_IDX[j]: int(base["card_idx"][i, j]) for j in range(len(tokens.CARD_IDX))
             if tokens.CARD_IDX[j] != "slot"}
        for j, k in enumerate(tokens.CARD_NUM):
            c[k] = int(round(tokens.symexp(base["card_num"][i, j])))
        c["keywords"] = sorted(int(b) for b in np.nonzero(base["card_kw"][i])[0])
        keys.append(tokens._card_content_key(c))
    assert keys == sorted(keys)


def test_detokenize_rebuilds_per_zone_lists():
    # v6: instance rows rebuild the per-zone lists exactly (one instance per row).
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
    assert len(got["cards"]["hand"]) == 3
    assert len(got["cards"]["draw"]) == 11
    assert len(got["cards"]["discard"]) == 3
    # 3 + 11 + 3 = 17 instances -> 17 instance rows.
    assert int(tokens.tokenize(st)["card_mask"].sum()) == 17


def test_duplicate_copies_across_zones_round_trip():
    # The same content living in several zones round-trips exactly — each copy is its own row filed by its
    # `zone` categorical (no merging), so the per-zone lists reconstruct byte-for-byte.
    st = _state(hand=[_card("StrikeIronclad", damage=6, baseDamage=6)],
                draw=[_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(2)],
                discard=[_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(2)],
                enemies=[_enemy("JawWorm", 40)])
    ok, diff = tokens.round_trip(st)
    assert ok, "round-trip mismatch at " + str(diff)
    tok = tokens.tokenize(st)
    assert int(tok["card_mask"].sum()) == 5
    zones = sorted(z for z, _s, _c in _present_rows(tok))
    assert zones == ["discard", "discard", "draw", "draw", "hand"]


def test_shuffle_invariance_instance_rows():
    # Instance rows do not leak pile order: any shuffle of a duplicate-heavy pile is byte-identical.
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


def test_overflow_truncation_and_strict():
    # A state with more than MAX_CARDS instances raises TokenOverflow under strict, and truncates to the
    # cap (one row per copy) under strict=False.
    import pytest
    draw = [_card("StrikeIronclad", damage=6, baseDamage=6) for _ in range(tokens.MAX_CARDS + 10)]
    st = _state(draw=draw, enemies=[_enemy("JawWorm", 40)])
    with pytest.raises(tokens.TokenOverflow):
        tokens.tokenize(st, strict=True)
    tok = tokens.tokenize(st, strict=False)
    assert int(tok["card_mask"].sum()) == tokens.MAX_CARDS


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
    # v6: cards are instance rows — the numeric block is content only (no per-zone count columns).
    assert "count_draw" not in S.NUMERIC_RANGES["card"]
    assert "damage" in S.NUMERIC_RANGES["card"]


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
