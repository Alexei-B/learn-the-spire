"""Field spec — the single description of the tokenizer's typed-array layout the encoder and decoder
both build against (roadmap 3.1, design §4.2/§4.3).

The tokenizer (:mod:`lts2_agent.tokens`) emits, per state, a set of fixed-shape padded arrays. This
module names, for each token type:

* its **categorical columns** (``*_idx`` columns) with the embedding/head vocab size of each,
* the width of its **numeric block** (``*_num``, symlog values — the decoder regresses these directly,
  the same array :func:`lts2_agent.tokens.detokenize` inverts),
* whether it is a **variable-length** type (padded with a presence mask; the decoder predicts presence)
  or a single fixed token (``global`` / ``pending``),
* its **max slot count** (the tokenizer's padded dim).

Both the encoder's per-type input projections and the decoder's per-type heads iterate this spec, so the
model's output space *is* the tokenizer's array space — reconstruction reuses :func:`tokens.detokenize`
verbatim (never reimplemented). Vocab sizes come straight from the live catalogs + tokenizer enums, so a
catalog growth is picked up automatically (and stamped into the checkpoint via the tokenizer signature).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

from .. import catalog, tokens


def _enum_size(table: List[str]) -> int:
    """Rows for a fixed enum embedding/head: one per value + the reserved trailing UNKNOWN slot."""
    return len(table) + 1


# Catalog sizes (dense id->index maps; 0 = none/unknown).
_CARDS_N = catalog.load("cards").size
_POWERS_N = catalog.load("powers").size
_RELICS_N = catalog.load("relics").size
_POTIONS_N = catalog.load("potions").size


class TypeSpec(NamedTuple):
    name: str                       # token-type name (matches tokens.TOKEN_TYPES where applicable)
    idx_key: str                    # tokenizer array key for the categorical block ("" if none)
    num_key: str                    # tokenizer array key for the numeric block ("" if none)
    mask_key: str                   # tokenizer array key for the presence mask ("" if single-token)
    cat_cols: List[Tuple[str, int]]  # (column name, vocab size) for each categorical column
    num_width: int                  # width of the numeric block
    max_slots: int                  # padded slot count
    has_static: str                 # catalog kind whose static row is gathered by cat_cols[0] ("" none)
    has_kw: bool                    # card-style hashed-keyword multi-hot block (card_kw)


# Order here defines the token order in the encoder set and the decoder query set.
TYPES: List[TypeSpec] = [
    TypeSpec("global", "global_idx", "global_num", "",
             [("phase", _enum_size(tokens.GAME_PHASES)),
              ("side", _enum_size(tokens.SIDES)),
              ("turnPhase", _enum_size(tokens.TURN_PHASES))],
             len(tokens.GLOBAL_NUM), 1, "", False),
    # pending is a single token: its 4-wide numeric block carries [present, minSelect, maxSelect,
    # isUpgradeSelection]; no categorical columns.
    TypeSpec("pending", "", "pending", "", [], 4, 1, "", False),
    # v3: `zone` left the categorical block — a card population row spans all zones, its membership
    # carried by the five `count_<zone>` numeric columns (CARD_NUM tail). cat_cols tracks CARD_IDX.
    TypeSpec("card", "card_idx", "card_num", "card_mask",
             [("cardIndex", _CARDS_N),
              ("type", _enum_size(tokens.CARD_TYPES)),
              ("rarity", _enum_size(tokens.CARD_RARITIES)),
              ("targetType", _enum_size(tokens.TARGET_TYPES)),
              ("enchant", tokens.ENCHANT_VOCAB),
              ("afflict", tokens.AFFLICT_VOCAB)],
             len(tokens.CARD_NUM), tokens.MAX_CARDS, "cards", True),
    TypeSpec("creature", "creature_idx", "creature_num", "creature_mask",
             [("kind", _enum_size(tokens.CREATURE_KINDS)),
              ("identity", tokens.MONSTER_VOCAB)],
             len(tokens.CREATURE_NUM), tokens.MAX_CREATURES, "", False),
    TypeSpec("power", "power_idx", "power_num", "power_mask",
             [("powerIndex", _POWERS_N),
              ("parent", tokens.MAX_CREATURES)],
             len(tokens.POWER_NUM), tokens.MAX_POWERS, "powers", False),
    TypeSpec("intent", "intent_idx", "intent_num", "intent_mask",
             [("type", _enum_size(tokens.INTENT_TYPES)),
              ("parent", tokens.MAX_CREATURES)],
             len(tokens.INTENT_NUM), tokens.MAX_INTENTS, "", False),
    TypeSpec("orb", "orb_idx", "orb_num", "orb_mask",
             [("orb", tokens.ORB_VOCAB)],
             len(tokens.ORB_NUM), tokens.MAX_ORBS, "", False),
    TypeSpec("relic", "relic_idx", "", "relic_mask",
             [("relicIndex", _RELICS_N)], 0, tokens.MAX_RELICS, "relics", False),
    TypeSpec("potion", "potion_idx", "", "potion_mask",
             [("potionIndex", _POTIONS_N)], 0, tokens.MAX_POTIONS, "potions", False),
]

TYPE_BY_NAME: Dict[str, TypeSpec] = {t.name: t for t in TYPES}

VARIABLE_TYPES = [t for t in TYPES if t.mask_key]        # have a presence mask
SINGLE_TYPES = [t for t in TYPES if not t.mask_key]      # global, pending

# Column indices into the numeric blocks that the report card / RAW-unit MAEs reference.
ENERGY_NUM_IDX = tokens.GLOBAL_NUM.index("energy")
CREATURE_HP_IDX = tokens.CREATURE_NUM.index("currentHp")
CREATURE_BLOCK_IDX = tokens.CREATURE_NUM.index("block")
INTENT_DAMAGE_IDX = tokens.INTENT_NUM.index("damage")
POWER_AMOUNT_IDX = 0  # POWER_NUM == ["amount"]
# v3: zone is a per-zone count VECTOR in the card numeric block, not a categorical column. These are the
# CARD_NUM column indices of the five count_<zone> fields (ZONES order); the report's card_zone_acc now
# scores the whole count vector rather than a single categorical zone id.
CARD_COUNT_COLS = [tokens.CARD_NUM.index(f) for f in tokens.ZONE_COUNT_FIELDS]
CARD_COUNT_COL_BY_ZONE = {z: tokens.CARD_NUM.index("count_" + z) for z in tokens.ZONES}
HAND_COUNT_COL = CARD_COUNT_COL_BY_ZONE["hand"]
PILE_COUNT_COLS = {z: CARD_COUNT_COL_BY_ZONE[z] for z in ("draw", "discard", "exhaust")}


# ==================================================================================================
# Per-field integer ranges (v3 exactness contract, roadmap M3.5).
#
# Every numeric column carries a measured ``(lo, hi, resolution)`` integer range, scanned from the
# corpus by ``python -m lts2_agent.wm.ranges`` (footprint.py's streaming pattern). These describe the
# EXACT observed integer domain of each field so a future per-field decoder can bin it precisely
# (``bins = (hi - lo) // resolution + 1``) instead of regressing one shared symlog float for every
# quantity.
#
# Exactness contract
# ------------------
# * The tokenizer still STORES symlog floats (``tokens.symlog``) for cache/decoder compatibility; the
#   integer round-trip stays exact via ``round(symexp(·))`` for every quantity inside the global clamp
#   ``[-tokens.NUM_CLIP, tokens.NUM_CLIP]`` (``tokens._q``).
# * These ranges are the *decode* contract: a value is guaranteed exactly representable iff it lies in
#   ``[lo, hi]``. Values outside clamp — LOUDLY and documented — to the nearest bound via
#   :func:`clamp_to_range` (the game's 999999999 "no-limit" sentinel is the canonical example; it also
#   hits ``tokens.NUM_CLIP`` in storage). ``hi`` is set with generous slack over the observed maximum so
#   real play never clamps; clamping signals genuinely out-of-distribution input.
# * ``resolution`` is 1 for every field today (all game quantities are integers); it is carried so a
#   future coarse-binned field (e.g. score in steps of 5) can widen its bin without a schema change.
# ==================================================================================================

class RangeSpec(NamedTuple):
    lo: int
    hi: int
    resolution: int

    @property
    def n_bins(self) -> int:
        return (self.hi - self.lo) // self.resolution + 1


# Measured over data/corpus by `python -m lts2_agent.wm.ranges --shard-stride 12` (336k states across
# 75 shards spread over the corpus) on 2026-07-17. {token type: {numeric column: (lo, hi, resolution)}};
# observed [min,max] in the trailing comment, ``hi`` carries slack. Booleans/presence flags (costsX,
# hasDamage, active, isGameOver-as-flag, …) are excluded (0..1 by construction).
#
# Three deliberate hand-widenings past this particular corpus (documented, so the future decoder bins a
# sane domain rather than a collection artifact; the tokenizer's symlog round-trip is unaffected either
# way, and clamp_to_range flags anything outside):
#   * act / floor / ascension / score: this corpus is act-0 / floor-2 / ascension-0 homogeneous, so
#     these observed [0,0]/[2,2] ranges reflect the COLLECTION, not the game. Widened to the game domain;
#     they will re-measure properly on the fresh multi-act data/corpus2.
#   * creature currentHp / maxHp: the observed max is the game's 999999999 "no-maximum" sentinel
#     (clamped to tokens.NUM_CLIP). Real creature HP is <=~800 (design), so the range caps at 1000 and
#     the sentinel clamps LOUDLY — a 100001-bin HP head would be absurd.
#   * pending maxSelect: same 999999999 "no-limit" sentinel; capped at 100 (a whole-deck select) with
#     the sentinel clamping loud.
_RANGES_RAW: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "global": {
        "act": (0, 10, 1),            # observed [0, 0] (corpus act-0 homogeneous; widened to game domain)
        "floor": (0, 60, 1),          # observed [2, 2] (homogeneous; widened)
        "ascension": (0, 20, 1),      # observed [0, 0] (homogeneous; widened)
        "score": (0, 5000, 1),        # observed [20, 33] (homogeneous; widened to run scale)
        "isGameOver": (0, 1, 1),      # 0/1 flag
        "isVictory": (0, 1, 1),       # 0/1 flag
        "energy": (0, 40, 1),         # observed [0, 29]
        "maxEnergy": (2, 20, 1),      # observed [2, 7]
        "maxEnergyRun": (3, 10, 1),   # observed [3, 3]
        "stars": (0, 40, 1),          # observed [0, 29]
        "roundNumber": (1, 30, 1),    # observed [1, 22]
        "turnNumber": (1, 30, 1),     # observed [1, 22]
        "orbSlots": (0, 20, 1),       # observed [0, 10]
        "gold": (0, 5000, 1),         # observed [0, 1431]
    },
    "pending": {
        "minSelect": (0, 10, 1),      # observed [0, 3]
        "maxSelect": (0, 100, 1),     # observed sentinel 100000 (999999999 "no-limit"); capped, clamps loud
    },
    "card": {
        "energyCost": (-1, 20, 1),    # observed [-1, 10] (-1 = X/unplayable marker)
        "starCost": (-1, 20, 1),      # observed [-1, 13]
        "replayCount": (0, 10, 1),    # observed [0, 2]
        "damage": (0, 200, 1),        # observed [0, 130]
        "baseDamage": (0, 150, 1),    # observed [0, 105]
        "block": (0, 150, 1),         # observed [0, 100]
        "baseBlock": (0, 70, 1),      # observed [0, 50]
        "summon": (0, 40, 1),         # observed [0, 25]
        "count_hand": (0, 20, 1),     # observed [0, 10]
        "count_draw": (0, 40, 1),     # observed [0, 28]
        "count_discard": (0, 40, 1),  # observed [0, 29]
        "count_exhaust": (0, 40, 1),  # observed [0, 29]
        "count_offered": (0, 30, 1),  # observed [0, 18]
    },
    "creature": {
        "currentHp": (0, 1000, 1),    # observed max = NUM_CLIP sentinel; capped to design ~800+slack, clamps loud
        "maxHp": (1, 1000, 1),        # observed max = NUM_CLIP sentinel; capped, clamps loud
        "block": (0, 150, 1),         # observed [0, 107]
        "combatId": (0, 30, 1),       # observed [0, 17]
    },
    "power": {
        "amount": (-30, 250, 1),      # observed [-30, 173]
    },
    "intent": {
        "damage": (0, 150, 1),        # observed [0, 104]
        "baseDamage": (0, 50, 1),     # observed [0, 35]
        "hits": (0, 20, 1),           # observed [0, 8]
    },
    "orb": {
        "passiveValue": (0, 20, 1),   # observed [0, 12]
        "evokeValue": (0, 250, 1),    # observed [0, 174]
    },
}

NUMERIC_RANGES: Dict[str, Dict[str, RangeSpec]] = {
    tname: {col: RangeSpec(*triple) for col, triple in cols.items()}
    for tname, cols in _RANGES_RAW.items()
}


def clamp_to_range(type_name: str, col_name: str, value: int) -> Tuple[int, bool]:
    """Clamp an integer ``value`` to the measured ``[lo, hi]`` range of one numeric field.

    Returns ``(clamped_value, was_clamped)``. A field with no measured range passes through unchanged
    (``was_clamped=False``). ``was_clamped=True`` marks a genuinely out-of-distribution input — the
    documented loud signal of the exactness contract."""
    rng = NUMERIC_RANGES.get(type_name, {}).get(col_name)
    if rng is None:
        return value, False
    if value < rng.lo:
        return rng.lo, True
    if value > rng.hi:
        return rng.hi, True
    return value, False
