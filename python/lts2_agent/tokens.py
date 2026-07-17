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
# v3 (2026-07): **factored population rows** — the T3 "expert-per-category" redesign (roadmap M3.5).
# `zone` leaves the card grouping key: one row per distinct card CONTENT (id + every dynamic field +
# keywords), carrying a **count-per-zone vector** (`count_hand/draw/discard/exhaust/offered`) instead of
# a single `count`. Population membership is now structural (a card that moves hand->discard is the SAME
# row with the count shifting between two columns — a future predictor expresses zone transitions as
# count arithmetic, and creation/transform as rows appearing/disappearing). Cards whose live fields
# differ across zones (e.g. a cost-reduced copy in hand vs its twin in draw) fall into separate rows,
# which is correct. detokenize expands the zone-count vector back to per-instance-per-zone canonical
# dicts, so the canonical dict stays BYTE-IDENTICAL to v2/v1 (statefmt/legal_actions/corpus untouched).
# Every numeric column also gains a measured per-field integer range in wm.spec (the exactness contract
# a per-field decoder bins against); the tokenizer keeps symlog storage for cache/decoder compat.
# v2 (2026-07): count-grouped card tokens WITH zone in the grouping key (one `count` per zone-scoped row).
# v1: raw per-instance card tokens.
TOKENIZER_VERSION = 3

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

# v3: per-zone count columns on a card population row (one integer per zone; symlog-stored like every
# other numeric). The vector is the row's membership across all piles at once.
ZONE_COUNT_FIELDS = ["count_" + z for z in ZONES]

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
# v3: cards are POPULATION ROWS (one per distinct content, zone excluded from the key), not raw
# instances. Re-measured over a shard-strided 336k-state scan of data/corpus
# (`python -m lts2_agent.wm.ranges --shard-stride 12`): v3 rows max 32 (v2 zone-scoped grouped max 42,
# v1 instance max 82); mean 14.21 instances/state -> 10.21 rows (1.39x shorter). Dropping zone from the
# key merges a content's hand/draw/… copies into one row, so v3 rows <= v2. Cap 64 keeps generous slack
# (~2x) over the worst case while holding the padded card dim at <1/3 of v1's.
# ==================================================================================================

MAX_CARDS = 64         # all zones pooled, v3 population rows (strided max 32; v2 grouped max 42; v1 cap 200)
MAX_CREATURES = 12     # player + osty + <=6 enemies (measured <=8)
MAX_POWERS = 96        # across all creatures (measured <=56)
MAX_INTENTS = 32       # across all enemies (measured <=18)
MAX_ORBS = 16          # (measured <=8)
MAX_RELICS = 24        # relics accumulate over a run (measured <=8 in the act0-2 corpus)
MAX_POTIONS = 8        # potion belt slots (measured <=5)

# --- Per-token-type field layouts (names double as the detokenize decode order) -------------------

GLOBAL_IDX = ["phase", "side", "turnPhase"]
GLOBAL_NUM = ["act", "floor", "ascension", "score", "isGameOver", "isVictory", "roundNumber",
              "energy", "maxEnergy", "stars", "turnNumber", "orbSlots", "gold", "maxEnergyRun"]
# pending: [present, minSelect, maxSelect, isUpgradeSelection]
PENDING_NUM = ["present", "minSelect", "maxSelect", "isUpgradeSelection"]

# v3: `zone` is gone from the card categorical block — population membership lives in the trailing
# count-per-zone vector instead, so one row spans every pile a given content occupies.
CARD_IDX = ["cardIndex", "type", "rarity", "targetType", "enchant", "afflict"]
# v3: the trailing five `count_<zone>` columns are how many identical-content instances this row holds
# in each zone (symlog; each >= 0, sum >= 1). detokenize expands them back into per-zone per-instance
# copies so the canonical dict is unchanged from v1/v2.
CARD_NUM = ["energyCost", "costsX", "starCost", "upgraded", "canPlay", "replayCount",
            "hasDamage", "damage", "baseDamage", "hasBlock", "block", "baseBlock",
            "hasSummon", "summon"] + ZONE_COUNT_FIELDS

CREATURE_IDX = ["kind", "identity"]
CREATURE_NUM = ["currentHp", "maxHp", "block", "active", "combatId"]

POWER_IDX = ["powerIndex", "parent"]
POWER_NUM = ["amount"]

INTENT_IDX = ["type", "parent"]
INTENT_NUM = ["hasDamage", "damage", "baseDamage", "hasHits", "hits"]

ORB_IDX = ["orb"]
ORB_NUM = ["passiveValue", "evokeValue"]

RELIC_IDX = ["relicIndex"]
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
    """v3 population key: the full content tuple WITHOUT ``zone`` — the grouping/order key for the
    zone-spanning population rows. Two instances with identical live fields in different zones share
    this key (and merge into one row); any live-field divergence keeps them apart."""
    return (c["cardIndex"], c["type"], c["rarity"], c["targetType"], c["enchant"],
            c["afflict"], c["energyCost"], c["costsX"], c["starCost"], c["upgraded"], c["canPlay"],
            c["replayCount"], c["hasDamage"], c["damage"], c["baseDamage"], c["hasBlock"], c["block"],
            c["baseBlock"], c["hasSummon"], c["summon"], tuple(c["keywords"]))


def _group_cards(cv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v3: pool ALL zones and collapse identical-CONTENT cards (zone excluded from the key) into ONE
    population row carrying a per-zone ``counts`` vector. A content that occupies several piles is a
    single row whose counts spread across those zones; content that differs in any live field (an
    upgraded twin, a cost-reduced copy) stays a separate row. Rows are order-canonicalized by their
    content key so the token order is deterministic and shuffle-invariant. Returns representative
    canonical card dicts each with an added ``counts`` (a ``{zone: n}`` dict, each n >= 0, sum >= 1);
    the underlying per-instance dicts are left untouched."""
    groups: Dict[Tuple, Dict[str, Any]] = {}
    for z in ZONES:
        for c in cv["cards"][z]:
            key = _card_content_key(c)
            g = groups.get(key)
            if g is None:
                groups[key] = {"card": c, "counts": {zz: 0 for zz in ZONES}}
                g = groups[key]
            g["counts"][z] += 1
    rows: List[Dict[str, Any]] = []
    for key in sorted(groups):
        g = groups[key]
        rows.append({**g["card"], "counts": g["counts"]})
    return rows


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
        for orb in (cs.get("orbs") or []):
            orbs.append({"orb": _orb(orb.get("orbId")),
                         "passiveValue": _q(orb.get("passiveValue")),
                         "evokeValue": _q(orb.get("evokeValue"))})
        for rid in (pl.get("relics") or []):
            relics.append(_RELICS.index_of(rid))
        for slot in (pl.get("potions") or []):
            potions.append(_POTIONS.index_of(slot))
    for enemy in (combat.get("enemies") or []):
        creatures.append(_creature_canonical(
            "enemy", _mon(enemy.get("monsterId")), int(enemy.get("currentHp") or 0),
            int(enemy.get("maxHp") or 0), int(enemy.get("block") or 0),
            1 if enemy.get("isHittable") else 0, int(enemy.get("combatId") or 0),
            enemy.get("powers") or [], enemy.get("intents") or []))

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

    # Cards: v3 population rows — identical content pooled across ALL zones into one row carrying a
    # per-zone count vector (zone is no longer a categorical column). Rows are content-ordered.
    all_cards: List[Dict] = _group_cards(cv)
    _check("cards", len(all_cards), MAX_CARDS)
    card_idx = np.zeros((MAX_CARDS, len(CARD_IDX)), dtype=np.int32)
    card_num = np.zeros((MAX_CARDS, len(CARD_NUM)), dtype=np.float32)
    card_kw = np.zeros((MAX_CARDS, KW_BUCKETS), dtype=np.float32)
    card_mask = np.zeros(MAX_CARDS, dtype=bool)
    for i, c in enumerate(all_cards[:MAX_CARDS]):
        card_idx[i] = [c[k] for k in CARD_IDX]
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
        orb_idx[i] = [orb["orb"]]
        orb_num[i] = [symlog(orb["passiveValue"]), symlog(orb["evokeValue"])]
        orb_mask[i] = True
    out["orb_idx"], out["orb_num"], out["orb_mask"] = orb_idx, orb_num, orb_mask
    out["token_type_orb"] = np.int32(TOKEN_TYPE_ID["orb"])

    relics = cv["relics"]
    _check("relics", len(relics), MAX_RELICS)
    relic_idx = np.zeros((MAX_RELICS, len(RELIC_IDX)), dtype=np.int32)
    relic_mask = np.zeros(MAX_RELICS, dtype=bool)
    for i, rid in enumerate(relics[:MAX_RELICS]):
        relic_idx[i] = [rid]
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
    """symlog for numeric card fields, pass-through 0/1 for the boolean/presence flags. The v3
    ``count_<zone>`` columns read the row's ``counts`` vector; for an ungrouped single card (e.g. an
    option-card featurization, which has no ``counts``) they default to 1 in the card's own zone and 0
    elsewhere, so a lone option card reads as a one-instance population row."""
    if key in ("costsX", "upgraded", "canPlay", "hasDamage", "hasBlock", "hasSummon"):
        return float(c[key])
    if key in ZONE_COUNT_FIELDS:
        zone = key[len("count_"):]
        counts = c.get("counts")
        if counts is not None:
            return symlog(float(counts.get(zone, 0)))
        own = ZONES[c["zone"]] if 0 <= c.get("zone", -1) < len(ZONES) else None
        return symlog(1.0 if zone == own else 0.0)
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

    cards: Dict[str, List[Dict]] = {z: [] for z in ZONES}
    ci, cn, ckw, cm = tok["card_idx"], tok["card_num"], tok["card_kw"], tok["card_mask"]
    for i in range(len(cm)):
        if not cm[i]:
            continue
        c: Dict[str, Any] = {CARD_IDX[j]: int(ci[i, j]) for j in range(len(CARD_IDX))}
        counts: Dict[str, int] = {}
        for j, k in enumerate(CARD_NUM):
            if k in ZONE_COUNT_FIELDS:
                # v3: this population row holds `n` identical instances in this zone (clamp >= 0).
                counts[k[len("count_"):]] = max(0, _int(cn[i, j]))
                continue
            c[k] = int(round(cn[i, j])) if k in ("costsX", "upgraded", "canPlay",
                                                 "hasDamage", "hasBlock", "hasSummon") else _int(cn[i, j])
        c["keywords"] = sorted(int(b) for b in np.nonzero(ckw[i])[0])
        # Expand the per-zone counts back into per-instance canonical dicts, each stamped with its own
        # `zone` index (byte-identical to v1/v2's per-instance-per-zone canonical list).
        for zone in ZONES:
            zone_idx = _enum_idx(zone, ZONES)
            for _ in range(counts.get(zone, 0)):
                cards[zone].append({**c, "zone": zone_idx, "keywords": list(c["keywords"])})
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
            orbs.append({"orb": int(oi[i, 0]), "passiveValue": _int(on[i, 0]),
                         "evokeValue": _int(on[i, 1])})

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
    # v3 sequence-length saving: card INSTANCES (v1 token count) vs POPULATION ROWS (v3 token count).
    sum_instances = 0
    sum_grouped = 0
    max_instances = 0

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
        grouped = len(_group_cards(cv))
        sum_instances += instances
        sum_grouped += grouped
        max_instances = max(max_instances, instances)
        maxima["cards"] = max(maxima["cards"], grouped)
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
        note = "  (v3 population rows; v1 instance max %d)" % max_instances if k == "cards" else ""
        print(f"  {k:10s} max {maxima[k]:4d}   cap {caps[k]:4d}{flag}{note}")
    if n_states:
        print(f"  card tokens/state — mean instances (v1) {sum_instances / n_states:6.2f}  ->  "
              f"mean rows (v3) {sum_grouped / n_states:6.2f}  "
              f"({sum_instances / max(1, sum_grouped):.2f}x shorter)")
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
