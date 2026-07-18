"""Entity tokenizer — the world-model train/serve parity contract (design §4.1, roadmap 2.1).

Successor to :mod:`lts2_agent.features` for the world-model stack. Where ``features.py`` hand-crafts a
fixed scalar vector (the "feature treadmill" the design §3 calls out), this module encodes a game state
as a **set of typed entity tokens** — one per card, creature, power, intent, orb, relic, potion, plus a
global token and an optional pending-choice token. Rule: *if the wire exposes it, tokenize it*. New
mechanics arrive as new catalog ids + generic numeric fields, never as bespoke features.

Products
--------
* :func:`tokenize` — a state dict -> fixed-shape, padded numpy arrays + masks (batchable), each token
  carrying a token-type id, categorical **catalog/enum indices**, and **symlog-encoded** numeric fields.
* :func:`detokenize` — the exact inverse into a canonical dict (round-trip validator).
* :func:`coverage_check` — walks the raw wire dict and classifies **every** field as covered / waived /
  lost; ``lost`` must stay empty over the corpus (the CP3 contract).
* :data:`TOKENIZER_VERSION` + the four catalog signatures — stamped into checkpoints/corpora/protocol.

Two invariants worth shouting about
-----------------------------------
* **The draw pile (and discard/exhaust/hand) is an unordered MULTISET.** Card tokens within a zone are
  sorted by their full content tuple, so the wire's shuffle order can *never* leak into the tokens. Two
  different shuffles of the same pile produce byte-identical tokens (unit-tested).
* **Numerics use symlog** (``sign(x)*log1p(|x|)``, DreamerV3-style): bounded, so a runaway block/scaling
  value can't blow up the encoder, and **exactly invertible** for the integer game quantities (round of
  ``symexp``), which is what makes the round-trip validator exact.

Categoricals are indices into a catalog (:mod:`lts2_agent.catalog` — cards/powers/relics/potions) or a
small fixed enum (zones, token types, intent/target/card types, phases — enumerated from
``GameState.cs``). Open string ids with no catalog dump (monster/character/orb/enchant/affliction ids
and granted keywords) are hashed into fixed vocabs; these are **covered-lossy** (documented below), not
exactly invertible — see :data:`LOSSY_FIELDS`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import catalog

# ==================================================================================================
# Version + catalog signatures (roadmap contract 6 — stamps corpora/checkpoints/protocol).
# ==================================================================================================

# Bump whenever the token layout / vocab semantics change so stale artifacts reject loudly.
# v6 (2026-07): **cards become INSTANCE rows** (roadmap wm-t3-factored) — the encoding style that solved
# relics/orbs, applied to cards (ONE change this iteration; creatures/relics/potions/orbs/scalars keep
# their v5 semantics EXACTLY). The v3 "population rows" (one content-sorted row per distinct content with a
# 5-column per-zone count vector) are replaced by one token row per physical card COPY: two identical
# Strikes are two rows. The five ZONE_COUNT_FIELDS are DELETED from CARD_NUM (and spec.NUMERIC_RANGES);
# instead each row carries a `zone` categorical (the 5 ZONES + the reserved UNKNOWN top slot, same enum
# convention as every other enum column) and a `slot` positional categorical (0..MAX_CARDS-1, like
# relics/orbs). CARD_NUM shrinks to the 14 content numerics (energyCost..summon). Keywords multi-hot per row
# unchanged. CANONICAL LAYOUT: rows are laid out ZONE-MAJOR in the fixed ZONES order; WITHIN a zone,
# instances are sorted by the content key (`_card_content_key`, unchanged — it never carried zone/counts);
# slot == the row's index in that layout. RATIONALE (well-posedness): slot is a per-state anchor for the
# permutation-invariant card expert with NO global joint constraint; identical copies are interchangeable so
# within-zone ties are harmless (no information leaks through which duplicate lands in which tied slot); and
# draw-pile ORDER is deliberately NOT exposed — zone-major position within the sorted draw group carries no
# pile-order information, so the information-set rule stands. Cards remain permutation-invariant
# (canonicalized): a shuffle of any pile is a re-sort to the same multiset -> byte-identical tokens including
# the deterministic slot column. Cross-state-stable instance ids (deck order) are a later C# iteration, not
# now. MAX_CARDS stays 64 this iteration (decks can exceed 100 via doubling effects — a later iteration will
# raise the cap to ~256; the strict-overflow _check is the loud clamp meanwhile). detokenize rebuilds the
# per-zone card lists from the instance rows (the canonical dict is BYTE-IDENTICAL to v5's — same per-zone
# lists — so statefmt/legal_actions/corpus consumers are unaffected).
# v5 / "v3.2" (2026-07): **relics become a POSITIONAL type** (roadmap M3.5). Product facts corrected the
# v3.1 relic model on two counts: (a) relic ORDER IS SEMANTIC — wax relics (e.g. Tezcatara via Toy Box)
# expire in ACQUISITION order and the wire's ordered id list is the only carrier of that state; and (b)
# duplicate relics ARE possible (rare) and the total can exceed the old 24 cap in long runs. So relics get
# the ORB treatment: WIRE ORDER is preserved (v3.1's `relics.sort()` is reverted), one token row per relic
# INSTANCE (duplicates are separate rows, distinguished by order), and each token carries an explicit
# `slot` categorical (0..MAX_RELICS-1) == its list index, so the permutation-invariant relic expert can
# see and target the order. detokenize emits the ordered id list VERBATIM (the canonical `relics` list is
# back to a flat wire-order id list — id-based consumers unaffected). MAX_RELICS rises to 40 (bounding
# TOTAL relics again, not distinct) with the strict-overflow loud clamp. The old duplicate-free set head +
# slot-dedup machinery is deleted (git history preserves it) — a positional slot decode has no use for it.
# v4 / "v3.1" (2026-07): **representational well-posedness fix** (roadmap M3.5). A set encoder pools its
# category permutation-invariantly, so a POSITION-SPECIFIC target that varies with input order but is not
# carried in any per-token field is ill-posed (proven: permuted potion belts encode byte-identically while
# their per-slot targets differ). The fix is per-expert by SEMANTICS:
#   * potions — slot identity is decision-irrelevant (options key on potion id), so position is
#     CANONICALIZED AWAY: the belt is LEFT-PACKED (non-empty potions first, sorted by catalog index, then
#     index-0 empties), preserving belt SIZE. detokenize emits the same left-packed layout, so a rare
#     non-left-packed raw belt (e.g. [empty, empty, X]) round-trips to its canonical [X, empty, empty].
#   * orbs — position IS semantic (evoke order), so it is made VISIBLE: each orb token gains an explicit
#     `slot` categorical (0..MAX_ORBS-1) the set encoder can represent.
#   * creatures — CANONICALIZED (sorted by content) so their per-slot targets are a function of the
#     multiset, not the wire order (combatId already carries a creature's identity). (v3.1 also sorted
#     relics; v5 reverts that — see above — because relic order turned out to be semantic.)
# The card population rows were already content-sorted (well-posed) and are unchanged.
# v3 (2026-07): **factored population rows** — the T3 "expert-per-category" redesign (roadmap M3.5).
# `zone` leaves the card grouping key: one row per distinct card CONTENT (id + every dynamic field +
# keywords), carrying a **count-per-zone vector** (`count_hand/draw/discard/exhaust/offered`) instead of
# a single `count`. (Superseded by v6's instance rows.)
# v2 (2026-07): count-grouped card tokens WITH zone in the grouping key (one `count` per zone-scoped row).
# v1: raw per-instance card tokens.
TOKENIZER_VERSION = 6

_CARDS = catalog.load("cards")
_POWERS = catalog.load("powers")
_RELICS = catalog.load("relics")
_POTIONS = catalog.load("potions")

CATALOG_SIGNATURES = {
    "cards": _CARDS.signature, "powers": _POWERS.signature,
    "relics": _RELICS.signature, "potions": _POTIONS.signature,
}


def tokenizer_signature() -> str:
    """A single stamp string: version + all four catalog signatures, for artifact parity checks."""
    sigs = "|".join(f"{k}={CATALOG_SIGNATURES[k]}" for k in sorted(CATALOG_SIGNATURES))
    return f"tok-v{TOKENIZER_VERSION}|{sigs}"


# ==================================================================================================
# Fixed enums (serialized as strings on the wire — enumerated from GameState.cs / the game enums).
# Index = position; an unseen value maps to a reserved trailing UNKNOWN slot (kept consistent both
# ways, so it never causes a round-trip mismatch, only a CLI "unknown enum" note).
# ==================================================================================================

TOKEN_TYPES = ["global", "card", "creature", "power", "intent", "orb", "relic", "potion", "pending"]
TOKEN_TYPE_ID = {name: i for i, name in enumerate(TOKEN_TYPES)}

GAME_PHASES = ["NotStarted", "Map", "Combat", "Choice", "Reward", "Event", "BundleChoice",
               "Treasure", "RestSite", "Shop", "CrystalSphere", "GameOver", "Other"]
SIDES = ["None", "Player", "Enemy"]
TURN_PHASES = ["None", "Start", "AutoPrePlay", "Play", "AutoPostPlay", "End"]
CARD_TYPES = ["None", "Attack", "Skill", "Power", "Status", "Curse", "Quest"]
CARD_RARITIES = ["None", "Basic", "Common", "Uncommon", "Rare", "Ancient", "Event", "Token",
                 "Status", "Curse", "Quest"]
TARGET_TYPES = ["None", "Self", "AnyEnemy", "AllEnemies", "RandomEnemy", "AnyPlayer", "AnyAlly",
                "AllAllies", "TargetedNoCreature", "Osty"]
INTENT_TYPES = ["Attack", "Buff", "Debuff", "DebuffStrong", "Defend", "Escape", "Heal", "Hidden",
                "Summon", "Sleep", "Stun", "StatusCard", "CardDebuff", "DeathBlow", "Unknown"]
ZONES = ["hand", "draw", "discard", "exhaust", "offered"]
CREATURE_KINDS = ["player", "osty", "enemy"]

# Zone <-> the combatState pile key it comes from (offered comes from pendingChoice.options).
_ZONE_PILE = {"hand": "hand", "draw": "drawPile", "discard": "discardPile", "exhaust": "exhaustPile"}

# Hash vocabs for open string ids with no catalog dump (covered-lossy).
MONSTER_VOCAB = 512
CHAR_VOCAB = 64
ORB_VOCAB = 32
ENCHANT_VOCAB = 128
AFFLICT_VOCAB = 128
KW_BUCKETS = 32  # addedKeywords hashed multi-hot on each card token


def _enum_idx(name: Optional[str], table: List[str]) -> int:
    """Index of ``name`` in ``table``; a value not present maps to the reserved UNKNOWN slot (len)."""
    if name is None:
        return 0 if "None" in table else len(table)
    try:
        return table.index(name)
    except ValueError:
        return len(table)


def _enum_name(idx: int, table: List[str]) -> str:
    return table[idx] if 0 <= idx < len(table) else "UNKNOWN"


# ==================================================================================================
# Measured corpus maxima (200k-record scan, July 2026) + generous slack -> fixed padded dims.
# Scan: hand<=10, draw<=44, discard<=46, exhaust<=40, offered<=29 (sum<=169); enemies<=6; powers
# player<=13 / enemy<=7 each / osty<=1; intents<=3/enemy; orbs<=8; relics<=8; potions(slots)<=5.
#
# v6: cards are INSTANCE ROWS again (one token per physical copy). MAX_CARDS stays 64 this iteration:
# most act-0..2 states hold far fewer instances than 64, and the strict-overflow _check is the loud clamp.
# NOTE (owner): decks CAN exceed 100 instances via doubling effects — a LATER iteration will raise the cap
# to ~256 (and add cross-state-stable C# instance ids); not this iteration (ONE change at a time). The v6
# corpus re-scan reports the instances/state distribution + the count of states over this cap.
# ==================================================================================================

MAX_CARDS = 64         # v6 instance rows; owner: raise to ~256 in a later iteration (doubling-effect decks)
MAX_CREATURES = 12     # player + osty + <=6 enemies (measured <=8)
MAX_POWERS = 96        # across all creatures (measured <=56)
MAX_INTENTS = 32       # across all enemies (measured <=18)
MAX_ORBS = 16          # (measured <=8)
# v5: MAX_RELICS bounds TOTAL relic INSTANCES again (relics are positional, one token/instance, duplicates
# kept). data/corpus2 scan (4.0M states, 2026-07): max total relics/state 8, max copies of one relic 2
# (3238 states carry a duplicate — index 198 x2 — so duplicates are real but rare). This corpus is
# act-0..2 homogeneous; long runs accumulate more, so the cap is raised well past 24 to 40 (generous slack)
# so a deep run never truncates; the strict-overflow check is the loud clamp.
MAX_RELICS = 40        # total relic instances over a run (positional; measured max 8 in data/corpus2)
MAX_POTIONS = 8        # potion belt slots (measured <=5)

# --- Per-token-type field layouts (names double as the detokenize decode order) -------------------

GLOBAL_IDX = ["phase", "side", "turnPhase"]
GLOBAL_NUM = ["act", "floor", "ascension", "score", "isGameOver", "isVictory", "roundNumber",
              "energy", "maxEnergy", "stars", "turnNumber", "orbSlots", "gold", "maxEnergyRun"]
# pending: [present, minSelect, maxSelect, isUpgradeSelection]
PENDING_NUM = ["present", "minSelect", "maxSelect", "isUpgradeSelection"]

# v6: cards are INSTANCE rows. `cardIndex` stays first (the static-catalog gather column). `zone` returns
# as a categorical (the 5 ZONES + reserved UNKNOWN top slot) and `slot` (0..MAX_CARDS-1) is the row's index
# in the zone-major, within-zone-content-sorted layout — the positional anchor the card expert reads, the
# same treatment relics/orbs get. Keep cardIndex..afflict as the first six columns (their positions are a
# wire contract several consumers index by; zone/slot are appended).
CARD_IDX = ["cardIndex", "type", "rarity", "targetType", "enchant", "afflict", "zone", "slot"]
# v6: the 14 content numerics ONLY (the five per-zone count columns are deleted — instance membership is now
# the `zone` categorical, not a count vector). symlog-stored (flags raw), inverted by round(symexp(.)).
CARD_NUM = ["energyCost", "costsX", "starCost", "upgraded", "canPlay", "replayCount",
            "hasDamage", "damage", "baseDamage", "hasBlock", "block", "baseBlock",
            "hasSummon", "summon"]

CREATURE_IDX = ["kind", "identity"]
CREATURE_NUM = ["currentHp", "maxHp", "block", "active", "combatId"]

POWER_IDX = ["powerIndex", "parent"]
POWER_NUM = ["amount"]

INTENT_IDX = ["type", "parent"]
INTENT_NUM = ["hasDamage", "damage", "baseDamage", "hasHits", "hits"]

# v4: `slot` is the orb's belt POSITION (0..MAX_ORBS-1). Orb order is semantic (evoke order) but the orb
# expert is a permutation-invariant set encoder, so position must be an explicit per-token field for the
# encoder to represent it (see the well-posedness note in the version header).
ORB_IDX = ["orb", "slot"]
ORB_NUM = ["passiveValue", "evokeValue"]

# v5: `slot` is the relic's acquisition POSITION (0..MAX_RELICS-1) == its wire list index. Relic order is
# semantic (wax relics expire in acquisition order) but the relic expert is a permutation-invariant set
# encoder, so — exactly like orbs — position must be an explicit per-token field for the encoder to
# represent it and the decoder to target it. Duplicate relics are separate rows kept apart by their slot.
RELIC_IDX = ["relicIndex", "slot"]
POTION_IDX = ["potionIndex"]


# ==================================================================================================
# symlog numeric encoding (invertible for integers).
# ==================================================================================================

# Numeric clamp before symlog. float32 symlog is exactly invertible (round of symexp) only while the
# stored magnitude stays within ~7 significant digits; every real game quantity (HP, block, gold,
# score, power amounts) sits far inside 1e5. Saturating here bounds the round-trip error to <0.1 and,
# à la features.py's FEATURE_CLIP, only touches the pathological tail — notably the game's 999999999
# "no maximum" select sentinel, which clamps to NUM_CLIP (an "effectively unbounded" marker).
NUM_CLIP = 100000


def _q(x: Any) -> int:
    """Clamp an integer game quantity into the exactly-round-trippable range [-NUM_CLIP, NUM_CLIP]."""
    v = int(round(float(x or 0)))
    return NUM_CLIP if v > NUM_CLIP else (-NUM_CLIP if v < -NUM_CLIP else v)


def symlog(x: float) -> float:
    return float(np.sign(x) * np.log1p(np.abs(x)))


def symexp(y: float) -> float:
    return float(np.sign(y) * np.expm1(np.abs(y)))


def _int(y: float) -> int:
    """Recover an integer game quantity from its symlog value (exact for the observed range)."""
    return int(round(symexp(y)))


def _sl(x: Optional[float]) -> float:
    return symlog(float(x or 0))


# ==================================================================================================
# Hash helpers for the covered-lossy open categoricals.
# ==================================================================================================

def _mon(mid: Optional[str]) -> int:
    return catalog.stable_hash(mid or "", MONSTER_VOCAB)


def _char(cid: Optional[str]) -> int:
    return catalog.stable_hash(cid or "", CHAR_VOCAB)


def _orb(oid: Optional[str]) -> int:
    return catalog.stable_hash(oid or "", ORB_VOCAB)


def _ench(e: Optional[str]) -> int:
    return catalog.stable_hash(e or "", ENCHANT_VOCAB)


def _affl(a: Optional[str]) -> int:
    return catalog.stable_hash(a or "", AFFLICT_VOCAB)


def _kw_multi(keywords: Optional[List[str]]) -> List[int]:
    """Sorted list of set hashed keyword buckets (multi-hot; order/collision lossy)."""
    if not keywords:
        return []
    return sorted({catalog.stable_hash(k, KW_BUCKETS + 1) - 1 for k in keywords})


# ==================================================================================================
# Canonical view — the shared target of tokenize/detokenize round-trip. Built by BOTH
# :func:`_canonical_from_state` (straight from the wire) and :func:`detokenize` (from the arrays);
# equality between the two is the round-trip contract.
# ==================================================================================================

def _card_canonical(card: Dict[str, Any], zone: str) -> Dict[str, Any]:
    has_dmg = card.get("damage") is not None
    has_blk = card.get("block") is not None
    has_sum = card.get("summon") is not None
    return {
        "cardIndex": _CARDS.index_of(card.get("cardId")),
        "zone": _enum_idx(zone, ZONES),
        "type": _enum_idx(card.get("type"), CARD_TYPES),
        "rarity": _enum_idx(card.get("rarity"), CARD_RARITIES),
        "targetType": _enum_idx(card.get("targetType"), TARGET_TYPES),
        "enchant": _ench(card.get("enchantmentId")),
        "afflict": _affl(card.get("afflictionId")),
        "energyCost": _q(card.get("energyCost")),
        "costsX": 1 if card.get("costsX") else 0,
        "starCost": _q(card.get("starCost")),
        "upgraded": 1 if card.get("upgraded") else 0,
        "canPlay": 1 if card.get("canPlay") else 0,
        "replayCount": _q(card.get("replayCount")),
        "hasDamage": 1 if has_dmg else 0,
        "damage": _q(card.get("damage")),
        "baseDamage": _q(card.get("baseDamage")),
        "hasBlock": 1 if has_blk else 0,
        "block": _q(card.get("block")),
        "baseBlock": _q(card.get("baseBlock")),
        "hasSummon": 1 if has_sum else 0,
        "summon": _q(card.get("summon")),
        "keywords": _kw_multi(card.get("addedKeywords")),
    }


def _card_sort_key(c: Dict[str, Any]) -> Tuple:
    """Full per-instance ordering key (includes ``zone``) — orders the canonical per-zone lists."""
    return (c["cardIndex"], c["zone"], c["type"], c["rarity"], c["targetType"], c["enchant"],
            c["afflict"], c["energyCost"], c["costsX"], c["starCost"], c["upgraded"], c["canPlay"],
            c["replayCount"], c["hasDamage"], c["damage"], c["baseDamage"], c["hasBlock"], c["block"],
            c["baseBlock"], c["hasSummon"], c["summon"], tuple(c["keywords"]))


def _card_content_key(c: Dict[str, Any]) -> Tuple:
    """v6 within-zone order key: the full content tuple WITHOUT ``zone`` (and without ``slot`` — that is
    the layout index, not content). Instance rows in one zone are laid out in this order; identical copies
    tie here (harmless — they are interchangeable). Unchanged from v3: it never carried zone or the deleted
    count columns, so the ``slot`` layout matches the generator's content sort bit-for-bit."""
    return (c["cardIndex"], c["type"], c["rarity"], c["targetType"], c["enchant"],
            c["afflict"], c["energyCost"], c["costsX"], c["starCost"], c["upgraded"], c["canPlay"],
            c["replayCount"], c["hasDamage"], c["damage"], c["baseDamage"], c["hasBlock"], c["block"],
            c["baseBlock"], c["hasSummon"], c["summon"], tuple(c["keywords"]))


def _power_canonical(p: Dict[str, Any]) -> Dict[str, Any]:
    return {"idx": _POWERS.index_of(p.get("powerId")), "amount": _q(p.get("amount"))}


def _intent_canonical(i: Dict[str, Any]) -> Dict[str, Any]:
    has_dmg = i.get("damage") is not None
    has_hits = i.get("hits") is not None
    return {
        "type": _enum_idx(i.get("type"), INTENT_TYPES),
        "hasDamage": 1 if has_dmg else 0,
        "damage": _q(i.get("damage")),
        "baseDamage": _q(i.get("baseDamage")),
        "hasHits": 1 if has_hits else 0,
        "hits": _q(i.get("hits")),
    }


def _creature_canonical(kind: str, identity: int, cur: int, mx: int, block: int, active: int,
                        combat_id: int, powers: List[Dict], intents: List[Dict]) -> Dict[str, Any]:
    return {
        "kind": _enum_idx(kind, CREATURE_KINDS), "identity": identity,
        "currentHp": _q(cur), "maxHp": _q(mx), "block": _q(block), "active": active,
        "combatId": _q(combat_id),
        "powers": sorted((_power_canonical(p) for p in powers),
                         key=lambda p: (p["idx"], p["amount"])),
        "intents": sorted((_intent_canonical(i) for i in intents),
                          key=lambda i: (i["type"], i["damage"], i["baseDamage"], i["hits"])),
    }


def _creature_sort_key(cr: Dict[str, Any]) -> Tuple:
    """Canonical creature order key (v4): kind first (keeps player < osty < enemy), then combatId (a
    creature's stable identity) and the scalar fields as a deterministic tiebreak. Powers/intents are
    excluded — they are sorted within a creature and follow it via the parent-slot flatten in
    :func:`tokenize`."""
    return (cr["kind"], cr["combatId"], cr["identity"], cr["currentHp"], cr["maxHp"],
            cr["block"], cr["active"])


def _players(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return state.get("players") or []


def _canonical_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    players = _players(state)
    p0 = players[0] if players else {}
    cs0 = p0.get("combatState") or {}
    combat = state.get("combat") or {}

    g = {
        "phase": _enum_idx(state.get("phase"), GAME_PHASES),
        "side": _enum_idx(combat.get("currentSide"), SIDES),
        "turnPhase": _enum_idx(cs0.get("phase"), TURN_PHASES),
        "act": _q(state.get("actIndex")),
        "floor": _q(state.get("floor")),
        "ascension": _q(state.get("ascensionLevel")),
        "score": _q(state.get("score")),
        "isGameOver": 1 if state.get("isGameOver") else 0,
        "isVictory": 1 if state.get("isVictory") else 0,
        "roundNumber": _q(combat.get("roundNumber")),
        "energy": _q(cs0.get("energy")),
        "maxEnergy": _q(cs0.get("maxEnergy")),
        "stars": _q(cs0.get("stars")),
        "turnNumber": _q(cs0.get("turnNumber")),
        "orbSlots": _q(cs0.get("orbSlots")),
        "gold": _q(p0.get("gold")),
        "maxEnergyRun": _q(p0.get("maxEnergy")),
    }

    pc = state.get("pendingChoice")
    pending = None
    if pc is not None:
        pending = {"minSelect": _q(pc.get("minSelect")),
                   "maxSelect": _q(pc.get("maxSelect")),
                   "isUpgradeSelection": 1 if pc.get("isUpgradeSelection") else 0}

    # Cards, pooled by zone and canonicalized as multisets (kills shuffle order).
    cards: Dict[str, List[Dict]] = {z: [] for z in ZONES}
    for pl in players:
        cs = pl.get("combatState") or {}
        for zone, pile in _ZONE_PILE.items():
            for card in (cs.get(pile) or []):
                cards[zone].append(_card_canonical(card, zone))
    for card in ((pc or {}).get("options") or []):
        cards["offered"].append(_card_canonical(card, "offered"))
    for z in ZONES:
        cards[z].sort(key=_card_sort_key)

    # Creatures (players, ostys, enemies) with nested powers/intents; orbs/relics/potions.
    creatures: List[Dict] = []
    orbs: List[Dict] = []
    relics: List[int] = []
    potions: List[int] = []
    for pl in players:
        cs = pl.get("combatState") or {}
        creatures.append(_creature_canonical(
            "player", _char(pl.get("character")), int(pl.get("currentHp") or 0),
            int(pl.get("maxHp") or 0), int(pl.get("block") or 0), 1, 0,
            cs.get("powers") or [], []))
        osty = cs.get("osty")
        if osty is not None:
            creatures.append(_creature_canonical(
                "osty", 0, int(osty.get("currentHp") or 0), int(osty.get("maxHp") or 0),
                int(osty.get("block") or 0), 1 if osty.get("isAlive") else 0, 0,
                osty.get("powers") or [], []))
        # Orbs carry an explicit `slot` (belt position) so the permutation-invariant orb expert can
        # represent the semantic evoke order (v4). Slot is the running index across the belt.
        for orb in (cs.get("orbs") or []):
            orbs.append({"orb": _orb(orb.get("orbId")), "slot": len(orbs),
                         "passiveValue": _q(orb.get("passiveValue")),
                         "evokeValue": _q(orb.get("evokeValue"))})
        for rid in (pl.get("relics") or []):
            relics.append(_RELICS.index_of(rid))
        # Potions: LEFT-PACK the belt (v4) — non-empty potions first (sorted by catalog index for a
        # deterministic canonical order), then the index-0 empty slots, preserving the belt SIZE. Slot
        # identity is decision-irrelevant (options key on potion id), so canonicalizing position away
        # makes the per-slot target a function of the belt MULTISET, not its wire order.
        belt = [_POTIONS.index_of(slot) for slot in (pl.get("potions") or [])]
        n_empty = sum(1 for pid in belt if pid == 0)
        potions.extend(sorted(pid for pid in belt if pid != 0))
        potions.extend([0] * n_empty)
    for enemy in (combat.get("enemies") or []):
        creatures.append(_creature_canonical(
            "enemy", _mon(enemy.get("monsterId")), int(enemy.get("currentHp") or 0),
            int(enemy.get("maxHp") or 0), int(enemy.get("block") or 0),
            1 if enemy.get("isHittable") else 0, int(enemy.get("combatId") or 0),
            enemy.get("powers") or [], enemy.get("intents") or []))

    # Canonicalize the creature order (v4): sort by a content key (kind keeps player<osty<enemy grouping;
    # combatId + the scalar fields make it deterministic). The set encoder pools creatures permutation-
    # invariantly and the decoder reconstructs per fixed slot, so a wire-order-dependent slot target is
    # ill-posed; sorting makes slot a function of content. combatId already carries a creature's identity,
    # so its order is not independently semantic. Powers/intents flatten to parent = list index in
    # `tokenize`, so their creature association follows this order automatically.
    creatures.sort(key=_creature_sort_key)
    # v5: relics keep WIRE ORDER (no sort) — order is semantic (wax relics expire in acquisition order).
    # The flat id list's index IS each relic's slot (tokenize stamps it as the positional `slot` column),
    # so the positional relic expert can represent the order the way orbs represent evoke order.

    return {"global": g, "pending": pending, "cards": cards, "creatures": creatures,
            "orbs": orbs, "relics": relics, "potions": potions}


# ==================================================================================================
# tokenize — canonical view -> fixed-shape padded arrays + masks.
# ==================================================================================================

# Keys of the returned token dict (stable order).
TOKEN_KEYS = (
    "global_idx", "global_num", "pending",
    "card_idx", "card_num", "card_kw", "card_mask",
    "creature_idx", "creature_num", "creature_mask",
    "power_idx", "power_num", "power_mask",
    "intent_idx", "intent_num", "intent_mask",
    "orb_idx", "orb_num", "orb_mask",
    "relic_idx", "relic_mask",
    "potion_idx", "potion_mask",
    "token_type_global", "token_type_card", "token_type_creature", "token_type_power",
    "token_type_intent", "token_type_orb", "token_type_relic", "token_type_potion",
)


class TokenOverflow(ValueError):
    """Raised when a state exceeds a fixed padded dimension (would drop tokens -> lossy)."""


def tokenize(state: Dict[str, Any], *, strict: bool = True) -> Dict[str, np.ndarray]:
    """Encode ``state`` into padded token arrays + masks. With ``strict`` (default), overflowing a
    fixed dim raises :class:`TokenOverflow` (so the coverage CLI flags it) rather than silently
    dropping tokens; with ``strict=False`` extras are truncated."""
    cv = _canonical_from_state(state)

    def _check(name: str, n: int, cap: int) -> None:
        if strict and n > cap:
            raise TokenOverflow(f"{name}: {n} > cap {cap}")

    out: Dict[str, np.ndarray] = {}

    g = cv["global"]
    out["global_idx"] = np.array([[g[k] for k in GLOBAL_IDX]], dtype=np.int32)
    out["global_num"] = np.array([[symlog(g[k]) for k in GLOBAL_NUM]], dtype=np.float32)
    p = cv["pending"]
    if p is None:
        out["pending"] = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    else:
        out["pending"] = np.array([[1.0, symlog(p["minSelect"]), symlog(p["maxSelect"]),
                                    float(p["isUpgradeSelection"])]], dtype=np.float32)
    out["token_type_global"] = np.int32(TOKEN_TYPE_ID["global"])

    # Cards: v6 INSTANCE rows — one row per physical copy, laid out ZONE-MAJOR (fixed ZONES order); within
    # a zone sorted by the content key. `zone` and `slot` (== the row's layout index) are categorical
    # columns; identical copies tie on the content key (harmless — interchangeable) and draw-pile order is
    # never exposed (the sorted zone-major position carries no pile order).
    all_cards: List[Dict] = []
    for z in ZONES:
        for c in sorted(cv["cards"][z], key=_card_content_key):
            all_cards.append(c)
    _check("cards", len(all_cards), MAX_CARDS)
    card_idx = np.zeros((MAX_CARDS, len(CARD_IDX)), dtype=np.int32)
    card_num = np.zeros((MAX_CARDS, len(CARD_NUM)), dtype=np.float32)
    card_kw = np.zeros((MAX_CARDS, KW_BUCKETS), dtype=np.float32)
    card_mask = np.zeros(MAX_CARDS, dtype=bool)
    for i, c in enumerate(all_cards[:MAX_CARDS]):
        row = {**c, "slot": i}                       # slot == the row's index in the zone-major layout
        card_idx[i] = [row[k] for k in CARD_IDX]
        card_num[i] = [_num_field(c, k) for k in CARD_NUM]
        for b in c["keywords"]:
            card_kw[i, b] = 1.0
        card_mask[i] = True
    out["card_idx"], out["card_num"], out["card_kw"], out["card_mask"] = \
        card_idx, card_num, card_kw, card_mask
    out["token_type_card"] = np.int32(TOKEN_TYPE_ID["card"])

    # Creatures + their powers/intents (parent = creature slot index).
    creatures = cv["creatures"]
    _check("creatures", len(creatures), MAX_CREATURES)
    creature_idx = np.zeros((MAX_CREATURES, len(CREATURE_IDX)), dtype=np.int32)
    creature_num = np.zeros((MAX_CREATURES, len(CREATURE_NUM)), dtype=np.float32)
    creature_mask = np.zeros(MAX_CREATURES, dtype=bool)
    powers: List[Tuple[int, Dict]] = []
    intents: List[Tuple[int, Dict]] = []
    for slot, cr in enumerate(creatures[:MAX_CREATURES]):
        creature_idx[slot] = [cr["kind"], cr["identity"]]
        creature_num[slot] = [symlog(cr["currentHp"]), symlog(cr["maxHp"]), symlog(cr["block"]),
                              float(cr["active"]), symlog(cr["combatId"])]
        creature_mask[slot] = True
        for pw in cr["powers"]:
            powers.append((slot, pw))
        for it in cr["intents"]:
            intents.append((slot, it))
    out["creature_idx"], out["creature_num"], out["creature_mask"] = \
        creature_idx, creature_num, creature_mask
    out["token_type_creature"] = np.int32(TOKEN_TYPE_ID["creature"])

    _check("powers", len(powers), MAX_POWERS)
    power_idx = np.zeros((MAX_POWERS, len(POWER_IDX)), dtype=np.int32)
    power_num = np.zeros((MAX_POWERS, len(POWER_NUM)), dtype=np.float32)
    power_mask = np.zeros(MAX_POWERS, dtype=bool)
    for i, (slot, pw) in enumerate(powers[:MAX_POWERS]):
        power_idx[i] = [pw["idx"], slot]
        power_num[i] = [symlog(pw["amount"])]
        power_mask[i] = True
    out["power_idx"], out["power_num"], out["power_mask"] = power_idx, power_num, power_mask
    out["token_type_power"] = np.int32(TOKEN_TYPE_ID["power"])

    _check("intents", len(intents), MAX_INTENTS)
    intent_idx = np.zeros((MAX_INTENTS, len(INTENT_IDX)), dtype=np.int32)
    intent_num = np.zeros((MAX_INTENTS, len(INTENT_NUM)), dtype=np.float32)
    intent_mask = np.zeros(MAX_INTENTS, dtype=bool)
    for i, (slot, it) in enumerate(intents[:MAX_INTENTS]):
        intent_idx[i] = [it["type"], slot]
        intent_num[i] = [float(it["hasDamage"]), symlog(it["damage"]), symlog(it["baseDamage"]),
                         float(it["hasHits"]), symlog(it["hits"])]
        intent_mask[i] = True
    out["intent_idx"], out["intent_num"], out["intent_mask"] = intent_idx, intent_num, intent_mask
    out["token_type_intent"] = np.int32(TOKEN_TYPE_ID["intent"])

    orbs = cv["orbs"]
    _check("orbs", len(orbs), MAX_ORBS)
    orb_idx = np.zeros((MAX_ORBS, len(ORB_IDX)), dtype=np.int32)
    orb_num = np.zeros((MAX_ORBS, len(ORB_NUM)), dtype=np.float32)
    orb_mask = np.zeros(MAX_ORBS, dtype=bool)
    for i, orb in enumerate(orbs[:MAX_ORBS]):
        orb_idx[i] = [orb["orb"], min(orb["slot"], MAX_ORBS - 1)]
        orb_num[i] = [symlog(orb["passiveValue"]), symlog(orb["evokeValue"])]
        orb_mask[i] = True
    out["orb_idx"], out["orb_num"], out["orb_mask"] = orb_idx, orb_num, orb_mask
    out["token_type_orb"] = np.int32(TOKEN_TYPE_ID["orb"])

    # Relics: v5 positional rows — one row per relic INSTANCE in wire order, carrying its acquisition
    # `slot` (== list index) so the permutation-invariant relic expert can represent the semantic order.
    relics = cv["relics"]
    _check("relics", len(relics), MAX_RELICS)
    relic_idx = np.zeros((MAX_RELICS, len(RELIC_IDX)), dtype=np.int32)
    relic_mask = np.zeros(MAX_RELICS, dtype=bool)
    for i, rid in enumerate(relics[:MAX_RELICS]):
        relic_idx[i] = [rid, min(i, MAX_RELICS - 1)]
        relic_mask[i] = True
    out["relic_idx"], out["relic_mask"] = relic_idx, relic_mask
    out["token_type_relic"] = np.int32(TOKEN_TYPE_ID["relic"])

    potions = cv["potions"]
    _check("potions", len(potions), MAX_POTIONS)
    potion_idx = np.zeros((MAX_POTIONS, len(POTION_IDX)), dtype=np.int32)
    potion_mask = np.zeros(MAX_POTIONS, dtype=bool)
    for i, pid in enumerate(potions[:MAX_POTIONS]):
        potion_idx[i] = [pid]
        potion_mask[i] = True
    out["potion_idx"], out["potion_mask"] = potion_idx, potion_mask
    out["token_type_potion"] = np.int32(TOKEN_TYPE_ID["potion"])

    return out


def _num_field(c: Dict[str, Any], key: str) -> float:
    """symlog for the content card numerics, pass-through 0/1 for the boolean/presence flags (v6: the
    per-zone count columns are gone — a card row's zone is now the ``zone`` categorical)."""
    if key in ("costsX", "upgraded", "canPlay", "hasDamage", "hasBlock", "hasSummon"):
        return float(c[key])
    return symlog(c[key])


# ==================================================================================================
# detokenize — arrays -> canonical dict (exact inverse of _canonical_from_state, up to the
# documented lossy fields, which round-trip as their stored hash/quantized value).
# ==================================================================================================

def detokenize(tok: Dict[str, np.ndarray]) -> Dict[str, Any]:
    gi = tok["global_idx"][0]
    gn = tok["global_num"][0]
    g = {GLOBAL_IDX[j]: int(gi[j]) for j in range(len(GLOBAL_IDX))}
    for j, k in enumerate(GLOBAL_NUM):
        g[k] = _int(gn[j])

    pv = tok["pending"][0]
    pending = None
    if pv[0] >= 0.5:
        pending = {"minSelect": _int(pv[1]), "maxSelect": _int(pv[2]),
                   "isUpgradeSelection": int(round(pv[3]))}

    # Cards: v6 instance rows. Each present row rebuilds one canonical per-instance card dict and is filed
    # into the pile named by its `zone` categorical; `slot` is positional (the layout index) so it is NOT
    # part of the canonical dict. Piles are re-sorted by the full per-instance key, matching
    # _canonical_from_state, so the canonical dict is byte-identical to v5's.
    cards: Dict[str, List[Dict]] = {z: [] for z in ZONES}
    ci, cn, ckw, cm = tok["card_idx"], tok["card_num"], tok["card_kw"], tok["card_mask"]
    zone_col = CARD_IDX.index("zone")
    for i in range(len(cm)):
        if not cm[i]:
            continue
        c: Dict[str, Any] = {CARD_IDX[j]: int(ci[i, j]) for j in range(len(CARD_IDX))
                             if CARD_IDX[j] != "slot"}
        for j, k in enumerate(CARD_NUM):
            c[k] = int(round(cn[i, j])) if k in ("costsX", "upgraded", "canPlay",
                                                 "hasDamage", "hasBlock", "hasSummon") else _int(cn[i, j])
        c["keywords"] = sorted(int(b) for b in np.nonzero(ckw[i])[0])
        zi = int(ci[i, zone_col])
        zone = ZONES[zi] if 0 <= zi < len(ZONES) else ZONES[0]
        cards[zone].append(c)
    for z in ZONES:
        cards[z].sort(key=_card_sort_key)

    # Creatures, then attach powers/intents by parent slot.
    cri, crn, crm = tok["creature_idx"], tok["creature_num"], tok["creature_mask"]
    creatures: List[Dict] = []
    slot_of: Dict[int, Dict] = {}
    for i in range(len(crm)):
        if not crm[i]:
            continue
        cr = {"kind": int(cri[i, 0]), "identity": int(cri[i, 1]),
              "currentHp": _int(crn[i, 0]), "maxHp": _int(crn[i, 1]), "block": _int(crn[i, 2]),
              "active": int(round(crn[i, 3])), "combatId": _int(crn[i, 4]),
              "powers": [], "intents": []}
        slot_of[i] = cr
        creatures.append(cr)

    pi, pn, pm = tok["power_idx"], tok["power_num"], tok["power_mask"]
    for i in range(len(pm)):
        if not pm[i]:
            continue
        slot = int(pi[i, 1])
        if slot in slot_of:
            slot_of[slot]["powers"].append({"idx": int(pi[i, 0]), "amount": _int(pn[i, 0])})

    ii, inm, im = tok["intent_idx"], tok["intent_num"], tok["intent_mask"]
    for i in range(len(im)):
        if not im[i]:
            continue
        slot = int(ii[i, 1])
        if slot in slot_of:
            slot_of[slot]["intents"].append({
                "type": int(ii[i, 0]), "hasDamage": int(round(inm[i, 0])), "damage": _int(inm[i, 1]),
                "baseDamage": _int(inm[i, 2]), "hasHits": int(round(inm[i, 3])), "hits": _int(inm[i, 4])})
    for cr in creatures:
        cr["powers"].sort(key=lambda p: (p["idx"], p["amount"]))
        cr["intents"].sort(key=lambda i: (i["type"], i["damage"], i["baseDamage"], i["hits"]))

    orbs = []
    oi, on, om = tok["orb_idx"], tok["orb_num"], tok["orb_mask"]
    for i in range(len(om)):
        if om[i]:
            orbs.append({"orb": int(oi[i, 0]), "slot": int(oi[i, 1]),
                         "passiveValue": _int(on[i, 0]), "evokeValue": _int(on[i, 1])})

    # v5: relics are positional rows read in array order (== wire/acquisition order); the flat id list's
    # index is each relic's slot, so we emit the relicIndex column verbatim (order preserved).
    relics = [int(tok["relic_idx"][i, 0]) for i in range(len(tok["relic_mask"]))
              if tok["relic_mask"][i]]
    potions = [int(tok["potion_idx"][i, 0]) for i in range(len(tok["potion_mask"]))
               if tok["potion_mask"][i]]

    return {"global": g, "pending": pending, "cards": cards, "creatures": creatures,
            "orbs": orbs, "relics": relics, "potions": potions}


def round_trip(state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Tokenize then detokenize ``state`` and compare to its canonical view. Returns
    ``(ok, first_diff_path)``; ``ok`` is False with a path describing the first mismatch."""
    try:
        tok = tokenize(state, strict=True)
    except TokenOverflow as ex:
        return False, f"overflow:{ex}"
    got = detokenize(tok)
    want = _canonical_from_state(state)
    return _deep_diff("", want, got)


def _deep_diff(path: str, a: Any, b: Any) -> Tuple[bool, Optional[str]]:
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return False, path or "/"
        for k in a:
            ok, p = _deep_diff(f"{path}/{k}", a[k], b[k])
            if not ok:
                return False, p
        return True, None
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False, f"{path}[len {len(a) if isinstance(a, list) else '?'}!={len(b) if isinstance(b, list) else '?'}]"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, p = _deep_diff(f"{path}[{i}]", x, y)
            if not ok:
                return False, p
        return True, None
    if a != b:
        return False, f"{path}({a!r}!={b!r})"
    return True, None


# ==================================================================================================
# Coverage — walk the raw wire dict, classify every field covered / waived / lost.
# ==================================================================================================

# Fields tokenized but not exactly invertible (hashed / multi-hot). Covered, with the quantization
# documented. Listed for the report and so reviewers know what does NOT round-trip to a string.
LOSSY_FIELDS = {
    "state/combat/enemies[]/monsterId": "CRC32-hashed monster-id identity (no monster catalog dump)",
    "state/players[]/character": "CRC32-hashed character identity (no character catalog dump)",
    "state/players[]/combatState/orbs[]/orbId": "CRC32-hashed orb identity (no orb catalog dump)",
    "state/players[]/combatState/hand[]/enchantmentId": "CRC32-hashed enchantment id",
    "state/players[]/combatState/drawPile[]/enchantmentId": "CRC32-hashed enchantment id",
    "state/players[]/combatState/discardPile[]/enchantmentId": "CRC32-hashed enchantment id",
    "state/players[]/combatState/exhaustPile[]/enchantmentId": "CRC32-hashed enchantment id",
    "state/pendingChoice/options[]/enchantmentId": "CRC32-hashed enchantment id",
    "state/players[]/combatState/hand[]/afflictionId": "CRC32-hashed affliction id",
    "state/players[]/combatState/drawPile[]/afflictionId": "CRC32-hashed affliction id",
    "state/players[]/combatState/discardPile[]/afflictionId": "CRC32-hashed affliction id",
    "state/players[]/combatState/exhaustPile[]/afflictionId": "CRC32-hashed affliction id",
    "state/pendingChoice/options[]/afflictionId": "CRC32-hashed affliction id",
    "state/players[]/combatState/hand[]/addedKeywords[]": "hashed multi-hot keyword buckets (order/collision lossy)",
    "state/players[]/combatState/drawPile[]/addedKeywords[]": "hashed multi-hot keyword buckets",
    "state/players[]/combatState/discardPile[]/addedKeywords[]": "hashed multi-hot keyword buckets",
    "state/players[]/combatState/exhaustPile[]/addedKeywords[]": "hashed multi-hot keyword buckets",
    "state/pendingChoice/options[]/addedKeywords[]": "hashed multi-hot keyword buckets",
}

# Waivers: wire fields deliberately NOT tokenized, each with a reason. A field waived here (by exact
# path, or because an ancestor prefix is waived, or by a trailing-suffix rule) is never "lost".
WAIVERS: List[Tuple[str, str, str]] = [
    # Non-combat room views. The tokenizer is a COMBAT world-model (design §4.1); these views appear
    # only at fight boundaries / choice states in the corpus and are modelled by the harness directly.
    ("prefix", "state/map", "non-combat map view (out of world-model scope)"),
    ("prefix", "state/rewards", "post-combat rewards view (out of scope)"),
    ("prefix", "state/bundleChoice", "card-bundle selection view (out of scope)"),
    ("prefix", "state/event", "event view (out of scope)"),
    ("prefix", "state/shop", "shop view (out of scope)"),
    ("prefix", "state/restSite", "rest-site view (out of scope)"),
    ("prefix", "state/treasure", "treasure view (out of scope)"),
    ("prefix", "state/crystalSphere", "crystal-sphere minigame view (out of scope)"),
    # Run-level / identity fields not part of the mechanical combat state.
    ("exact", "state/seed", "episode identifier, not mechanical state"),
    ("suffix", "netId", "network identity, not mechanical state"),
    ("prefix", "state/players[]/deck", "persistent run deck; in combat the live cards are the four "
                                        "piles (tokenized); out of combat it is run-level, out of scope"),
    # Static per-card attribute already carried by the card catalog's static row (derivable from id).
    ("suffix", "poolId", "static card home-pool, already in the card-catalog static row (derivable "
                         "from cardId)"),
]


def _matches_waiver(path: str) -> Optional[str]:
    for kind, pat, reason in WAIVERS:
        if kind == "exact" and path == pat:
            return reason
        if kind == "prefix" and (path == pat or path.startswith(pat + "/") or path.startswith(pat + "[")):
            return reason
        if kind == "suffix" and (path == pat or path.endswith("/" + pat) or path.endswith("/" + pat + "[]")):
            return reason
    return None


def _covered_paths() -> set:
    """The set of normalized leaf paths the tokenizer covers (exact + lossy)."""
    paths = {
        "state/phase", "state/actIndex", "state/floor", "state/ascensionLevel", "state/score",
        "state/isGameOver", "state/isVictory",
        "state/combat/roundNumber", "state/combat/currentSide",
        "state/combat/enemies[]/combatId", "state/combat/enemies[]/monsterId",
        "state/combat/enemies[]/currentHp", "state/combat/enemies[]/maxHp",
        "state/combat/enemies[]/block", "state/combat/enemies[]/isHittable",
        "state/combat/enemies[]/powers[]/powerId", "state/combat/enemies[]/powers[]/amount",
        "state/combat/enemies[]/intents[]/type", "state/combat/enemies[]/intents[]/damage",
        "state/combat/enemies[]/intents[]/baseDamage", "state/combat/enemies[]/intents[]/hits",
        "state/players[]/character", "state/players[]/currentHp", "state/players[]/maxHp",
        "state/players[]/block", "state/players[]/gold", "state/players[]/maxEnergy",
        "state/players[]/relics[]", "state/players[]/potions[]",
        "state/players[]/combatState/energy", "state/players[]/combatState/maxEnergy",
        "state/players[]/combatState/stars", "state/players[]/combatState/turnNumber",
        "state/players[]/combatState/phase", "state/players[]/combatState/orbSlots",
        "state/players[]/combatState/orbs[]/orbId",
        "state/players[]/combatState/orbs[]/passiveValue",
        "state/players[]/combatState/orbs[]/evokeValue",
        "state/players[]/combatState/osty/currentHp", "state/players[]/combatState/osty/maxHp",
        "state/players[]/combatState/osty/block", "state/players[]/combatState/osty/isAlive",
        "state/players[]/combatState/osty/powers[]/powerId",
        "state/players[]/combatState/osty/powers[]/amount",
        "state/players[]/combatState/powers[]/powerId",
        "state/players[]/combatState/powers[]/amount",
        "state/pendingChoice/minSelect", "state/pendingChoice/maxSelect",
        "state/pendingChoice/isUpgradeSelection",
    }
    card_fields = ["cardId", "energyCost", "costsX", "starCost", "upgraded", "canPlay",
                   "replayCount", "type", "rarity", "targetType", "damage", "baseDamage", "block",
                   "baseBlock", "summon", "enchantmentId", "afflictionId", "addedKeywords[]"]
    bases = ["state/players[]/combatState/hand[]", "state/players[]/combatState/drawPile[]",
             "state/players[]/combatState/discardPile[]", "state/players[]/combatState/exhaustPile[]",
             "state/pendingChoice/options[]"]
    for base in bases:
        for f in card_fields:
            paths.add(base + "/" + f)
    return paths


COVERED = _covered_paths()


def _leaf_paths(state: Dict[str, Any]) -> set:
    out: set = set()

    def walk(prefix: str, val: Any) -> None:
        if isinstance(val, dict):
            if not val:
                out.add(prefix)
                return
            for k, v in val.items():
                walk(f"{prefix}/{k}" if prefix else k, v)
        elif isinstance(val, list):
            if not val:
                out.add(prefix + "[]")
                return
            for v in val:
                walk(prefix + "[]", v)
        else:
            out.add(prefix)

    walk("state", state)
    return out


def coverage_check(state: Dict[str, Any]) -> Tuple[set, set, set]:
    """Classify every leaf field path in ``state`` into ``(covered, waived, lost)`` path sets."""
    covered, waived, lost = set(), set(), set()
    for path in _leaf_paths(state):
        if _matches_waiver(path) is not None:
            waived.add(path)
        elif path in COVERED:
            covered.add(path)
        elif _is_empty_covered_container(path):
            # An empty collection (e.g. `enemies[]`, `powers[]`, `hand[]` == []) whose element fields
            # are covered — 0 tokens is a faithful encoding of an empty pile.
            covered.add(path)
        else:
            lost.add(path)
    return covered, waived, lost


def _is_empty_covered_container(path: str) -> bool:
    """True if ``path`` is a bare list-container whose element leaf paths are covered."""
    prefix = path + "/"
    return any(cp.startswith(prefix) for cp in COVERED)


# ==================================================================================================
# CLI: coverage + round-trip report over the corpus (the CP3 review artifact).
# ==================================================================================================

def _iter_states(root: str, limit: Optional[int]):
    """Yield (record-index, which, state) for every state and nextState in the corpus."""
    from . import corpus
    n = 0
    for rec in corpus.iter_records(root):
        for which in ("state", "nextState"):
            st = rec.get(which)
            if st:
                yield n, which, st
        n += 1
        if limit and n >= limit:
            return


def _check(root: str, limit: Optional[int]) -> int:
    covered_seen: Dict[str, int] = {}
    waived_seen: Dict[str, int] = {}
    lost_seen: Dict[str, int] = {}
    maxima = {"cards": 0, "creatures": 0, "powers": 0, "intents": 0, "orbs": 0, "relics": 0,
              "potions": 0}
    n_states = 0
    n_records = 0
    rt_fail = 0
    rt_examples: List[str] = []
    # v6: cards are INSTANCE rows (one token per physical copy) — track total instances/state and how many
    # states exceed the MAX_CARDS padded cap (the strict-overflow clamp).
    sum_instances = 0
    max_instances = 0
    n_overflow = 0

    for ridx, which, st in _iter_states(root, limit):
        n_records = ridx + 1
        n_states += 1
        cov, wai, lost = coverage_check(st)
        for p in cov:
            covered_seen[p] = covered_seen.get(p, 0) + 1
        for p in wai:
            waived_seen[p] = waived_seen.get(p, 0) + 1
        for p in lost:
            lost_seen[p] = lost_seen.get(p, 0) + 1

        cv = _canonical_from_state(st)
        instances = sum(len(cv["cards"][z]) for z in ZONES)
        sum_instances += instances
        max_instances = max(max_instances, instances)
        if instances > MAX_CARDS:
            n_overflow += 1
        maxima["cards"] = max(maxima["cards"], instances)
        maxima["creatures"] = max(maxima["creatures"], len(cv["creatures"]))
        maxima["powers"] = max(maxima["powers"], sum(len(c["powers"]) for c in cv["creatures"]))
        maxima["intents"] = max(maxima["intents"], sum(len(c["intents"]) for c in cv["creatures"]))
        maxima["orbs"] = max(maxima["orbs"], len(cv["orbs"]))
        maxima["relics"] = max(maxima["relics"], len(cv["relics"]))
        maxima["potions"] = max(maxima["potions"], len(cv["potions"]))

        ok, diff = round_trip(st)
        if not ok:
            rt_fail += 1
            if len(rt_examples) < 10:
                rt_examples.append(f"rec#{ridx} {which}: {diff}")

    caps = {"cards": MAX_CARDS, "creatures": MAX_CREATURES, "powers": MAX_POWERS,
            "intents": MAX_INTENTS, "orbs": MAX_ORBS, "relics": MAX_RELICS, "potions": MAX_POTIONS}

    print("=" * 78)
    print(f"TOKENIZER COVERAGE REPORT   ({tokenizer_signature()})")
    print("=" * 78)
    print(f"records scanned: {n_records}   states scanned: {n_states}   root: {root}")
    print()
    print("Measured token maxima (this scan) vs padded caps:")
    for k in ("cards", "creatures", "powers", "intents", "orbs", "relics", "potions"):
        flag = "  !! OVER CAP" if maxima[k] > caps[k] else ""
        note = "  (v6 instance rows)" if k == "cards" else ""
        print(f"  {k:10s} max {maxima[k]:4d}   cap {caps[k]:4d}{flag}{note}")
    if n_states:
        print(f"  card tokens/state — mean instances {sum_instances / n_states:6.2f}  "
              f"max {max_instances}  states over cap({MAX_CARDS}): {n_overflow} "
              f"({100.0 * n_overflow / n_states:.3f}%)")
    print()
    print(f"COVERED fields ({len(covered_seen)}):")
    for p in sorted(covered_seen):
        tag = "  [lossy: " + LOSSY_FIELDS[p] + "]" if p in LOSSY_FIELDS else ""
        print(f"  + {p}  (x{covered_seen[p]}){tag}")
    print()
    print(f"WAIVED fields ({len(waived_seen)}):")
    for p in sorted(waived_seen):
        reason = _matches_waiver(p) or "?"
        print(f"  ~ {p}  (x{waived_seen[p]})  — {reason}")
    print()
    print(f"LOST fields ({len(lost_seen)}):")
    for p in sorted(lost_seen):
        print(f"  X {p}  (x{lost_seen[p]})")
    if not lost_seen:
        print("  (none)")
    print()
    print(f"Round-trip: {n_states - rt_fail}/{n_states} states exact; {rt_fail} mismatches.")
    for ex in rt_examples:
        print("  mismatch: " + ex)
    print("=" * 78)

    failed = bool(lost_seen) or rt_fail > 0
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tokenizer coverage + round-trip report over a corpus.")
    ap.add_argument("--check", metavar="CORPUS_ROOT",
                    help="run coverage + round-trip over every state/nextState under CORPUS_ROOT")
    ap.add_argument("--limit", type=int, default=None, help="max records to scan")
    args = ap.parse_args(argv)
    if args.check:
        return _check(args.check, args.limit)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
