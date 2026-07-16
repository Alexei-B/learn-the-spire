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
    TypeSpec("card", "card_idx", "card_num", "card_mask",
             [("cardIndex", _CARDS_N),
              ("zone", _enum_size(tokens.ZONES)),
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
CARD_ZONE_COL = 1     # CARD_IDX == ["cardIndex", "zone", ...]
HAND_ZONE_IDX = tokens.ZONES.index("hand")
PILE_ZONE_IDX = {z: tokens.ZONES.index(z) for z in ("draw", "discard", "exhaust")}
