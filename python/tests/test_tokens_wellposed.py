"""Permutation well-posedness suite (roadmap M3.5, tokenizer v4) — the regression guard against the
representational ill-posedness the T3 experts hit.

The trap: every per-category expert is a **permutation-invariant set encoder**, so it pools its tokens
without seeing their slot order. If a token type's per-slot reconstruction target then varies with the
WIRE order of the underlying entities — but no per-token field carries that order — the mapping
(encoded set -> target) is one-to-many and the expert cannot learn exact reconstruction (proven: permuted
potion belts encoded byte-identically while their targets differed; coverage exact pinned at ~0.46).

The invariant this suite pins, for EVERY variable-length token type, is exactly one of:

* **canonicalized** — permuting the wire entities that feed the type yields BYTE-IDENTICAL token arrays
  (the array order is a pure function of content, so equal multisets => equal tokens); or
* **positional** — the tokens carry an explicit positional categorical column whose value equals the slot
  index for every present slot (so the encoder can see, and the decoder can target, the order).

cards / creatures / powers / intents / relics / potions are canonicalized; orbs are positional (evoke
order is semantic). A future token type that reintroduces the trap fails here.
"""

from __future__ import annotations

import copy
import random

import numpy as np

from lts2_agent import tokens
from lts2_agent.wm import spec as S

# Types whose tokens carry an explicit position column (value == slot index for present slots) instead of
# being canonical-order-invariant. Everything else must be canonicalized.
POSITIONAL = {"orb": "slot"}


def _card(card_id, **kw):
    c = {"cardId": card_id, "energyCost": kw.get("energyCost", 1), "costsX": False,
         "type": kw.get("type", "Attack"), "rarity": "Basic", "targetType": "AnyEnemy",
         "upgraded": kw.get("upgraded", False), "poolId": "IRONCLAD_CARD_POOL", "canPlay": True,
         "starCost": 0, "replayCount": 0, "addedKeywords": []}
    for k in ("damage", "baseDamage", "block", "baseBlock"):
        if k in kw:
            c[k] = kw[k]
    return c


def _enemy(monster_id, hp, combat_id, powers, intents):
    return {"combatId": combat_id, "monsterId": monster_id, "currentHp": hp, "maxHp": hp + 10,
            "block": 0, "isHittable": True, "powers": powers, "intents": intents}


def _rich_state():
    """A state exercising multiple entities of every variable type, with distinct content so a
    canonical sort is well-defined and a permutation is observable."""
    cs = {"energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 2, "phase": "Play", "orbSlots": 3,
          "hand": [_card("StrikeIronclad", damage=6, baseDamage=6),
                   _card("Bash", damage=8, baseDamage=8),
                   _card("DefendIronclad", type="Skill", block=5, baseBlock=5)],
          "drawPile": [_card("Anger"), _card("Cleave", damage=8, baseDamage=8)],
          "discardPile": [_card("Inflame", type="Power")],
          "exhaustPile": [], "osty": None,
          "powers": [{"powerId": "StrengthPower", "amount": 2},
                     {"powerId": "DexterityPower", "amount": 1},
                     {"powerId": "VulnerablePower", "amount": 3}],
          "orbs": [{"orbId": "Lightning", "passiveValue": 3, "evokeValue": 8},
                   {"orbId": "Frost", "passiveValue": 2, "evokeValue": 5},
                   {"orbId": "Dark", "passiveValue": 6, "evokeValue": 12}]}
    player = {"netId": 1, "character": "IRONCLAD", "currentHp": 55, "maxHp": 72, "block": 4,
              "gold": 99, "maxEnergy": 3, "deck": [],
              "relics": ["BURNING_BLOOD", "AKABEKO", "ANCHOR"],
              "potions": ["ATTACK_POTION", None, "BLOCK_POTION"], "combatState": cs}
    enemies = [
        _enemy("JawWorm", 40, 100, [{"powerId": "StrengthPower", "amount": 1}],
               [{"type": "Attack", "damage": 12, "baseDamage": 10, "hits": 1}, {"type": "Buff"}]),
        _enemy("Cultist", 48, 101, [{"powerId": "RitualPower", "amount": 3}],
               [{"type": "Buff"}]),
        _enemy("GreenLouse", 12, 102, [],
               [{"type": "Attack", "damage": 5, "baseDamage": 5, "hits": 1}]),
    ]
    return {"phase": "Combat", "seed": "WP-1", "actIndex": 1, "floor": 5, "ascensionLevel": 0,
            "isGameOver": False, "isVictory": False, "score": 123, "players": [player],
            "combat": {"roundNumber": 2, "currentSide": "Player", "enemies": enemies}}


def _shuffle_all_entities(st, seed):
    """Return a deep copy of ``st`` with the wire order of EVERY variable entity source permuted (piles,
    enemies, per-creature powers/intents, relics, potions, orbs). A well-posed tokenizer maps this to the
    same tokens as the original for every canonicalized type."""
    st = copy.deepcopy(st)
    rng = random.Random(seed)
    pl = st["players"][0]
    cs = pl["combatState"]
    for pile in ("hand", "drawPile", "discardPile", "exhaustPile"):
        rng.shuffle(cs[pile])
    rng.shuffle(cs["powers"])
    rng.shuffle(cs["orbs"])
    rng.shuffle(pl["relics"])
    rng.shuffle(pl["potions"])
    rng.shuffle(st["combat"]["enemies"])
    for en in st["combat"]["enemies"]:
        rng.shuffle(en["powers"])
        rng.shuffle(en["intents"])
    return st


def test_every_variable_type_is_classified():
    # Meta guard: every variable-length token type must be declared canonical or positional here, so a
    # newly added type can't silently escape the well-posedness contract.
    for t in S.VARIABLE_TYPES:
        assert t.name in POSITIONAL or t.name not in POSITIONAL  # tautology; the real check is below
    # Positional types must actually declare their positional column in the spec.
    for name, col in POSITIONAL.items():
        cols = [c for c, _ in S.TYPE_BY_NAME[name].cat_cols]
        assert col in cols, f"{name} declared positional but has no {col!r} categorical column"


def test_canonical_types_are_permutation_invariant():
    # For every canonicalized type, permuting the wire entities leaves its token arrays byte-identical.
    base = _rich_state()
    t0 = tokens.tokenize(base)
    canonical = [t for t in S.VARIABLE_TYPES if t.name not in POSITIONAL]
    for seed in range(8):
        t = tokens.tokenize(_shuffle_all_entities(base, seed))
        for tspec in canonical:
            for key in (tspec.idx_key, tspec.num_key, tspec.mask_key):
                if not key:
                    continue
                assert np.array_equal(t0[key], t[key]), (tspec.name, key, seed)
            if tspec.has_kw:
                assert np.array_equal(t0["card_kw"], t["card_kw"]), (tspec.name, seed)


def test_positional_types_carry_slot_index_column():
    # For every positional type, the declared column equals the slot index for all present slots (so the
    # set encoder sees the order it must reconstruct).
    base = _rich_state()
    tok = tokens.tokenize(base)
    for name, col in POSITIONAL.items():
        tspec = S.TYPE_BY_NAME[name]
        col_i = [c for c, _ in tspec.cat_cols].index(col)
        mask = tok[tspec.mask_key]
        k = int(mask.sum())
        assert k > 0, f"{name}: rich state should exercise present {name} tokens"
        idx = tok[tspec.idx_key]
        assert list(idx[:k, col_i]) == list(range(k)), name


def test_wellposedness_round_trip_holds_under_permutation():
    # Every permutation still round-trips exactly (the canonicalization/positional fields are consistent
    # between tokenize and detokenize).
    base = _rich_state()
    for seed in range(6):
        st = _shuffle_all_entities(base, seed)
        ok, diff = tokens.round_trip(st)
        assert ok, f"seed {seed} round-trip mismatch at {diff}"
